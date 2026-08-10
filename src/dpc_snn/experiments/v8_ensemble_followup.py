"""Shared, frozen-architecture helpers for V8 ensemble follow-up experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from dpc_snn.baselines.neural import build_v62_neural_baseline
from dpc_snn.experiments.v62_baselines import (
    BASELINE_OPTIMIZERS,
    FixedGain,
    apply_fixed_gain,
    fit_baseline,
    fit_fixed_gain,
    predict_baseline,
    prepare_model_input,
    task_carrier,
)
from dpc_snn.experiments.v62_scaffold import (
    project_registered_max_norm_constraints_,
)
from dpc_snn.experiments.v8_ensemble import (
    entropy_residual_prediction,
    equal_probability_anchor,
    extract_atc_sequence,
    softmax_probability,
    state_digest,
)
from dpc_snn.experiments.v8_sequence_decoder_training import (
    V8SequenceDecoderFit,
    fit_v8_sequence_decoder,
    predict_v8_sequence_decoder,
)
from dpc_snn.models.v8_sequence_decoder import (
    V8ATCSequenceDecoder,
    build_v8_sequence_decoder,
)
from dpc_snn.models.v62_spatial import BCI2A_CHANNEL_NAMES
from dpc_snn.utils.metrics import classification_metrics


@dataclass
class EnsembleComponents:
    atcnet: nn.Module
    fbcnet: nn.Module
    sew_clif: V8ATCSequenceDecoder
    ann_sew: V8ATCSequenceDecoder

    def as_dict(self) -> dict[str, nn.Module]:
        return {
            "atcnet": self.atcnet,
            "fbcnet": self.fbcnet,
            "sew_clif": self.sew_clif,
            "ann_sew": self.ann_sew,
        }

    def to(self, device: str) -> "EnsembleComponents":
        for model in self.as_dict().values():
            model.to(device)
        return self

    def eval(self) -> "EnsembleComponents":
        for model in self.as_dict().values():
            model.eval()
        return self


@dataclass
class EnsembleFitResult:
    components: EnsembleComponents
    gain: FixedGain
    histories: dict[str, list[dict[str, Any]]]
    optimizer_steps: dict[str, int]
    train_seconds: dict[str, float]
    constraint_projection_counts: dict[str, int]


def resolve_frozen_channel_basis(freeze: dict[str, Any]) -> list[str]:
    """Resolve the exact BCI2a sensor order from current or legacy freezes."""
    model_config = dict(freeze["architecture"]["model_config"])
    declared = model_config.get("channel_names")
    channels = (
        [str(value) for value in declared]
        if declared is not None
        else list(BCI2A_CHANNEL_NAMES)
    )
    if channels != list(BCI2A_CHANNEL_NAMES):
        raise RuntimeError("the frozen ensemble sensor order differs from BCI2a")
    if len(channels) != 22 or len(set(channels)) != len(channels):
        raise RuntimeError("the external confirmation requires 22 unique sensors")
    return channels


def _decoder_kwargs(model_config: dict[str, Any], n_classes: int) -> dict[str, Any]:
    decoder = dict(model_config["decoder"])
    return {
        "n_classes": int(n_classes),
        "hidden_channels": int(decoder["hidden_channels"]),
        "decoder_layers": int(decoder["layers"]),
        "readout_features": int(decoder["readout_features"]),
        "dropout": float(decoder["dropout"]),
    }


def build_ensemble_components(
    *,
    source_root: str | Path,
    model_config: dict[str, Any],
    n_classes: int,
) -> EnsembleComponents:
    if int(n_classes) < 2:
        raise ValueError("the ensemble requires at least two classes")
    atcnet = build_v62_neural_baseline(
        "atcnet",
        source_root=source_root,
        n_channels=22,
        n_classes=int(n_classes),
        samples=1000,
    )
    fbcnet = build_v62_neural_baseline(
        "fbcnet",
        source_root=source_root,
        n_channels=22,
        n_classes=int(n_classes),
        samples=1000,
    )
    decoder_kwargs = _decoder_kwargs(model_config, int(n_classes))
    return EnsembleComponents(
        atcnet=atcnet,
        fbcnet=fbcnet,
        sew_clif=build_v8_sequence_decoder("sew_clif", **decoder_kwargs),
        ann_sew=build_v8_sequence_decoder("ann_sew", **decoder_kwargs),
    )


def load_ensemble_components(
    run_dir: str | Path,
    *,
    source_root: str | Path,
    model_config: dict[str, Any],
    n_classes: int,
) -> EnsembleComponents:
    root = Path(run_dir)
    components = build_ensemble_components(
        source_root=source_root,
        model_config=model_config,
        n_classes=int(n_classes),
    )
    for name, model in components.as_dict().items():
        state = torch.load(root / f"{name}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    return components


def load_fixed_gain(path: str | Path) -> FixedGain:
    with np.load(Path(path), allow_pickle=False) as archive:
        values = np.asarray(archive["values"], dtype=np.float32)
        clip = float(np.asarray(archive["clip"]).item())
    if values.ndim != 3 or values.shape[1] != 22 or not np.isfinite(values).all():
        raise ValueError("invalid frozen 22-channel gain")
    return FixedGain(values=values, clip=clip)


def component_state_digests(components: EnsembleComponents) -> dict[str, str]:
    return {name: state_digest(model) for name, model in components.as_dict().items()}


def fit_ensemble_components(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    sfreq: float,
    epoch_tmin: float,
    source_root: str | Path,
    model_config: dict[str, Any],
    component_rule: dict[str, Any],
    preprocessing: dict[str, Any],
    augmentation: dict[str, Any],
    n_classes: int,
    run_seed: int,
    decoder_seed: int,
    device: str,
    run_label: str,
) -> EnsembleFitResult:
    labels = np.asarray(y_train, dtype=np.int64)
    if sorted(np.unique(labels).tolist()) != list(range(int(n_classes))):
        raise ValueError("training labels do not match the requested class count")
    carrier = task_carrier(
        np.asarray(x_train, dtype=np.float32),
        sfreq=float(sfreq),
        epoch_tmin=float(epoch_tmin),
    )
    gain = fit_fixed_gain(carrier, clip=float(preprocessing["clip_after_gain"]))
    normalized = apply_fixed_gain(carrier, gain)
    atc_input = prepare_model_input("atcnet", normalized, sfreq=float(sfreq))
    fbc_input = prepare_model_input("fbcnet", normalized, sfreq=float(sfreq))

    atc_fit = fit_baseline(
        "atcnet",
        source_root=source_root,
        x_train=atc_input,
        y_train=labels,
        x_validation=None,
        y_validation=None,
        device=device,
        seed=int(run_seed),
        epochs=int(component_rule["atcnet_final_epoch"]),
        patience=int(component_rule["atcnet_final_epoch"]),
        augmentation=augmentation,
        fixed_epoch=int(component_rule["atcnet_final_epoch"]),
        scheduler_epochs=int(component_rule["atcnet_scheduler_horizon"]),
        run_label=f"{run_label}:atcnet",
        n_channels=22,
        n_classes=int(n_classes),
    )
    fbc_fit = fit_baseline(
        "fbcnet",
        source_root=source_root,
        x_train=fbc_input,
        y_train=labels,
        x_validation=None,
        y_validation=None,
        device=device,
        seed=int(run_seed),
        epochs=int(component_rule["fbcnet_final_epoch"]),
        patience=int(component_rule["fbcnet_final_epoch"]),
        augmentation=augmentation,
        fixed_epoch=int(component_rule["fbcnet_final_epoch"]),
        scheduler_epochs=int(component_rule["fbcnet_scheduler_horizon"]),
        run_label=f"{run_label}:fbcnet",
        n_channels=22,
        n_classes=int(n_classes),
    )
    sequence, atc_teacher = extract_atc_sequence(
        atc_fit.model,
        atc_input,
        device=device,
        n_classes=int(n_classes),
    )
    replay = predict_baseline(
        atc_fit.model,
        atc_input,
        labels,
        device=device,
        batch_size=int(BASELINE_OPTIMIZERS["atcnet"]["batch_size"]),
    )
    replay_error = float(np.max(np.abs(replay["logits"] - atc_teacher)))
    if replay_error > 1e-5:
        raise RuntimeError(f"ATC sequence wrapper changed logits by {replay_error:.3e}")

    decoder = dict(model_config["decoder"])
    common = {
        "x_train": sequence,
        "y_train": labels,
        "teacher_train": atc_teacher,
        "x_validation": None,
        "y_validation": None,
        "teacher_validation": None,
        "device": device,
        "seed": int(decoder_seed),
        "epochs": int(component_rule["decoder_final_epoch"]),
        "fixed_epoch": int(component_rule["decoder_final_epoch"]),
        "scheduler_epochs": int(component_rule["decoder_scheduler_horizon"]),
        "distillation_weight": float(decoder["distillation_weight"]),
        "distillation_temperature": float(decoder["distillation_temperature"]),
        "firing_rate_weight": float(decoder["firing_rate_weight"]),
        "model_kwargs": _decoder_kwargs(model_config, int(n_classes)),
    }
    snn_fit: V8SequenceDecoderFit = fit_v8_sequence_decoder(
        "sew_clif", **common, run_label=f"{run_label}:sew_clif"
    )
    ann_fit: V8SequenceDecoderFit = fit_v8_sequence_decoder(
        "ann_sew", **common, run_label=f"{run_label}:ann_sew"
    )
    components = EnsembleComponents(
        atcnet=atc_fit.model,
        fbcnet=fbc_fit.model,
        sew_clif=snn_fit.model,
        ann_sew=ann_fit.model,
    )
    projection_counts = {
        name: project_registered_max_norm_constraints_(model)
        for name, model in components.as_dict().items()
    }
    return EnsembleFitResult(
        components=components,
        gain=gain,
        histories={
            "atcnet": atc_fit.history,
            "fbcnet": fbc_fit.history,
            "sew_clif": snn_fit.history,
            "ann_sew": ann_fit.history,
        },
        optimizer_steps={
            "atcnet": atc_fit.optimizer_steps,
            "fbcnet": fbc_fit.optimizer_steps,
            "sew_clif": snn_fit.optimizer_steps,
            "ann_sew": ann_fit.optimizer_steps,
        },
        train_seconds={
            "atcnet": atc_fit.elapsed_seconds,
            "fbcnet": fbc_fit.elapsed_seconds,
            "sew_clif": snn_fit.elapsed_seconds,
            "ann_sew": ann_fit.elapsed_seconds,
        },
        constraint_projection_counts=projection_counts,
    )


def _log_probability(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("probability matrix is invalid")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("probability rows are not normalized")
    return np.log(np.clip(values, 1e-12, 1.0)).astype(np.float32)


@torch.no_grad()
def predict_ensemble_from_carrier(
    components: EnsembleComponents,
    carrier: np.ndarray,
    labels: np.ndarray,
    gain: FixedGain,
    *,
    sfreq: float,
    model_config: dict[str, Any],
    n_classes: int,
    device: str,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    normalized = apply_fixed_gain(np.asarray(carrier, dtype=np.float32), gain)
    atc_input = prepare_model_input("atcnet", normalized, sfreq=float(sfreq))
    fbc_input = prepare_model_input("fbcnet", normalized, sfreq=float(sfreq))
    sequence, atc_logits = extract_atc_sequence(
        components.atcnet,
        atc_input,
        device=device,
        n_classes=int(n_classes),
    )
    fbc = predict_baseline(
        components.fbcnet,
        fbc_input,
        labels,
        device=device,
        batch_size=int(BASELINE_OPTIMIZERS["fbcnet"]["batch_size"]),
    )
    snn = predict_v8_sequence_decoder(
        components.sew_clif,
        sequence,
        labels,
        atc_logits,
        device=device,
    )
    ann = predict_v8_sequence_decoder(
        components.ann_sew,
        sequence,
        labels,
        atc_logits,
        device=device,
    )
    atc_probability = softmax_probability(atc_logits)
    fbc_probability = softmax_probability(fbc["logits"])
    anchor = equal_probability_anchor(atc_logits, fbc["logits"])
    snn_probability = softmax_probability(snn["logits"])
    ann_probability = softmax_probability(ann["logits"])
    maximum_weight = float(model_config["residual_fusion"]["maximum_decoder_weight"])
    primary = entropy_residual_prediction(
        anchor, snn["logits"], maximum_weight=maximum_weight
    )
    matched_ann = entropy_residual_prediction(
        anchor, ann["logits"], maximum_weight=maximum_weight
    )
    probabilities = {
        "primary": primary["probabilities"],
        "matched_ann": matched_ann["probabilities"],
        "anchor": anchor,
        "atcnet": atc_probability,
        "fbcnet": fbc_probability,
        "decoder_snn": snn_probability,
        "decoder_ann": ann_probability,
    }
    arms: dict[str, dict[str, Any]] = {}
    for name, probability in probabilities.items():
        pred = np.asarray(probability).argmax(axis=1).astype(np.int64)
        arms[name] = {
            "logits": _log_probability(probability),
            "probabilities": np.asarray(probability, dtype=np.float32),
            "pred": pred,
            **classification_metrics(labels, pred, n_classes=int(n_classes)),
        }
    return {
        "arms": arms,
        "sequence": sequence,
        "snn_mean_firing_rate": float(snn["mean_firing_rate"]),
        "ann_mean_firing_rate": float(ann["mean_firing_rate"]),
        "primary_gate": np.asarray(primary["gate"], dtype=np.float32),
        "matched_ann_gate": np.asarray(matched_ann["gate"], dtype=np.float32),
    }


def mask_carrier_after_endpoint(
    carrier: np.ndarray, *, endpoint_seconds: float, sfreq: float
) -> np.ndarray:
    values = np.asarray(carrier, dtype=np.float32)
    stop = int(round(float(endpoint_seconds) * float(sfreq)))
    if values.ndim != 3 or stop < 1 or stop > values.shape[-1]:
        raise ValueError("early-decision endpoint is outside the task carrier")
    output = values.copy()
    output[..., stop:] = 0.0
    return np.ascontiguousarray(output)


def decoder_operation_proxy(
    decoder: V8ATCSequenceDecoder,
    *,
    mean_binary_firing_rate: float | None,
) -> dict[str, Any]:
    core = decoder.decoder
    steps = int(decoder.expected_steps)
    encoder_uses = int(core.current_encoder.weight.numel() * steps)
    temporal_uses = int(
        sum(
            (block.depthwise.weight.numel() + block.pointwise.weight.numel()) * steps
            for block in core.blocks
        )
    )
    readout_uses = int(
        sum(
            parameter.numel()
            for module in (core.readout, core.classifier)
            for parameter in module.parameters()
            if parameter.ndim >= 2
        )
    )
    dense = encoder_uses + temporal_uses + readout_uses
    if decoder.decoder_kind == "ann":
        event_weighted = float(dense)
        firing_rate = None
    else:
        if mean_binary_firing_rate is None or not 0.0 <= float(mean_binary_firing_rate) <= 1.0:
            raise ValueError("a finite binary firing rate is required for the SNN proxy")
        firing_rate = float(mean_binary_firing_rate)
        event_weighted = float(encoder_uses + readout_uses + temporal_uses * firing_rate)
    return {
        "decoder_kind": decoder.decoder_kind,
        "time_steps": steps,
        "analog_encoder_weight_uses": encoder_uses,
        "temporal_synaptic_weight_uses": temporal_uses,
        "dense_readout_weight_uses": readout_uses,
        "dense_decoder_weight_uses": dense,
        "activity_weighted_decoder_events": event_weighted,
        "binary_spike_rate": firing_rate,
        "scope": "decoder-only synaptic-operation proxy; shared ATC/FBC frontends excluded",
        "assumptions": "analog encoder and dense readout retained; temporal synapses scaled by observed binary firing rate",
        "hardware_energy_claim_allowed": False,
    }
