#!/usr/bin/env python3
"""Probe zero-safe phase fusion before the frozen ATC attention/TCN blocks."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v62_protocol import session_t_run_grouped_folds  # noqa: E402
from dpc_snn.experiments.v62_scaffold import (  # noqa: E402
    CachedRates,
    cache_analytic_fixed_channel_gain_rate_features,
    fit_official_fbc_channel_gain,
    project_registered_max_norm_constraints_,
)
from dpc_snn.experiments.v7_delay import (  # noqa: E402
    build_delay_stage_model,
    state_sha256,
)
from dpc_snn.models.v62_filterbank import causal_linear_upsample_2x  # noqa: E402
from dpc_snn.utils.io import ensure_dir, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from probe_v7_phase_pair_contrast import (  # noqa: E402
    _metadata_rows,
    _subject_file,
)
from run_v7_e3_delay import (  # noqa: E402
    _fit_phase_amplitude_scale,
    _load_audited_fold_prior,
)


def _seed(value: int) -> None:
    random.seed(int(value))
    np.random.seed(int(value))
    torch.manual_seed(int(value))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(value))


class PhaseATCFeatureFusion(nn.Module):
    """Frozen ATC with a zero-baseline-corrected phase feature residual."""

    def __init__(
        self,
        implementation: nn.Module,
        *,
        n_bands: int,
        n_nodes: int,
        node_reconstruction: torch.Tensor,
        feature_bound: float = 0.25,
        feature_initial: float = 0.05,
    ) -> None:
        super().__init__()
        self.implementation = implementation
        for parameter in self.implementation.parameters():
            parameter.requires_grad_(False)
        self.n_bands = int(n_bands)
        self.n_nodes = int(n_nodes)
        reconstruction = torch.as_tensor(node_reconstruction, dtype=torch.float32)
        if reconstruction.ndim != 2 or reconstruction.shape[1] != self.n_nodes:
            raise ValueError("phase feature reconstruction must have shape [C, K]")
        self.register_buffer("node_reconstruction", reconstruction.clone())
        self.feature_bound = float(feature_bound)
        if not 0.0 < abs(float(feature_initial)) < self.feature_bound:
            raise ValueError("initial feature gain must lie strictly inside its bound")
        self.register_buffer(
            "phase_band_gain",
            torch.ones(self.n_bands, self.n_nodes),
        )
        feature_count = int(self.implementation.atc_blocks[0].linear.in_features)
        self.register_buffer("phase_feature_scale", torch.ones(feature_count))
        self.register_buffer("phase_feature_scale_ready", torch.tensor(False))
        self.feature_gain_raw = nn.Parameter(
            torch.full(
                (feature_count,),
                math.atanh(float(feature_initial) / self.feature_bound),
            )
        )

    @property
    def phase_gain(self) -> torch.Tensor:
        return self.phase_band_gain

    @property
    def feature_gain(self) -> torch.Tensor:
        return self.feature_bound * torch.tanh(self.feature_gain_raw)

    def _classify_features(self, features: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = features.shape
        windows = int(self.implementation.n_windows)
        logits = features.new_zeros(batch, int(self.implementation.n_classes))
        for index, block in enumerate(self.implementation.atc_blocks):
            logits = logits + block(
                features[:, index : steps - windows + index + 1]
            )
        return logits / float(windows)

    def reconstruct(self, current: torch.Tensor) -> torch.Tensor:
        if current.ndim != 3 or current.shape[1] != self.n_nodes:
            raise ValueError("reconstruction input must have shape [N, K, T]")
        return torch.einsum(
            "ck,nkt->nct",
            self.node_reconstruction.to(current),
            current,
        )

    def set_phase_feature_scale(self, scale: torch.Tensor) -> None:
        value = torch.as_tensor(
            scale,
            dtype=self.phase_feature_scale.dtype,
            device=self.phase_feature_scale.device,
        )
        if value.shape != self.phase_feature_scale.shape:
            raise ValueError("phase feature scale has an incompatible shape")
        if not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
            raise ValueError("phase feature scale must be finite and positive")
        self.phase_feature_scale.copy_(value)
        self.phase_feature_scale_ready.fill_(True)

    def encoded_features(
        self,
        broadband: torch.Tensor,
        phase_current: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phase_nodes = (
            phase_current
            * self.phase_gain.to(phase_current)[None, :, :, None]
        ).sum(dim=1) / math.sqrt(float(self.n_bands))
        base = self.reconstruct(broadband)
        phase = self.reconstruct(phase_nodes)
        base_features = self.implementation.rearrange(
            self.implementation.conv_block(base)
        )
        phase_features = self.implementation.rearrange(
            self.implementation.conv_block(phase)
            - self.implementation.conv_block(torch.zeros_like(phase))
        )
        return base_features, phase_features

    def forward(
        self,
        broadband: torch.Tensor,
        phase_current: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if broadband.ndim != 3 or broadband.shape[1] != self.n_nodes:
            raise ValueError("ATC broadband input must have shape [N, K, T]")
        if phase_current.ndim != 4 or phase_current.shape[1:3] != (
            self.n_bands,
            self.n_nodes,
        ):
            raise ValueError("phase current must have shape [N, B, K, T]")
        if not bool(self.phase_feature_scale_ready):
            raise RuntimeError("phase feature RMS calibration has not been fitted")
        base_features, phase_features = self.encoded_features(
            broadband,
            phase_current,
        )
        calibrated_phase = phase_features * self.phase_feature_scale.to(
            phase_features
        )[None, None]
        residual = calibrated_phase * self.feature_gain.to(
            phase_features
        )[None, None]
        return self._classify_features(base_features + residual), residual


@torch.no_grad()
def _cache_phase_current(
    model: nn.Module,
    rates: CachedRates,
    *,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    model.eval().to(device)
    for start in range(0, len(rates.fast), int(batch_size)):
        stop = min(start + int(batch_size), len(rates.fast))
        transport = model.delay(
            rates.fast[start:stop].to(device),
            rates.slow[start:stop].to(device),
            slow_delay_override="learned",
            fast_delay_override="zero",
        )
        slow = transport.slow_phase_pair_delay_contrast_current
        if slow is None:
            raise RuntimeError("delay transport did not produce a phase-pair current")
        parts.append(causal_linear_upsample_2x(slow).float().cpu())
    return torch.cat(parts)


@torch.no_grad()
def _fit_phase_feature_scale(
    model: PhaseATCFeatureFusion,
    broadband: torch.Tensor,
    phase: torch.Tensor,
    *,
    device: str,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    base_square = None
    phase_square = None
    observations = 0
    for start in range(0, len(broadband), int(batch_size)):
        stop = min(start + int(batch_size), len(broadband))
        base_features, phase_features = model.encoded_features(
            broadband[start:stop].to(device),
            phase[start:stop].to(device),
        )
        batch_base = base_features.square().sum(dim=(0, 1)).double().cpu()
        batch_phase = phase_features.square().sum(dim=(0, 1)).double().cpu()
        base_square = batch_base if base_square is None else base_square + batch_base
        phase_square = (
            batch_phase if phase_square is None else phase_square + batch_phase
        )
        observations += base_features.shape[0] * base_features.shape[1]
    if base_square is None or phase_square is None or observations < 1:
        raise RuntimeError("phase feature calibration received no observations")
    base_rms = (base_square / observations).sqrt().float()
    phase_rms = (phase_square / observations).sqrt().float()
    positive = phase_rms[phase_rms > 0]
    if positive.numel() == 0:
        raise RuntimeError("phase convolution features are identically zero")
    floor = max(float(positive.median()) * 1e-3, 1e-8)
    scale = (base_rms / phase_rms.clamp_min(floor)).clamp(0.1, 100.0)
    model.set_phase_feature_scale(scale.to(device))
    return {
        "base_feature_rms_mean": float(base_rms.mean()),
        "phase_feature_rms_mean_before_calibration": float(phase_rms.mean()),
        "phase_feature_scale_min": float(scale.min()),
        "phase_feature_scale_mean": float(scale.mean()),
        "phase_feature_scale_max": float(scale.max()),
    }


@torch.no_grad()
def _predict(
    model: PhaseATCFeatureFusion,
    broadband: torch.Tensor,
    phase: torch.Tensor,
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    model.eval()
    logits: list[np.ndarray] = []
    residual_square = 0.0
    residual_elements = 0
    for start in range(0, len(broadband), int(batch_size)):
        stop = min(start + int(batch_size), len(broadband))
        batch_logits, residual = model(
            broadband[start:stop].to(device),
            phase[start:stop].to(device),
        )
        logits.append(batch_logits.float().cpu().numpy())
        residual_square += float(residual.float().square().sum().cpu())
        residual_elements += int(residual.numel())
    return np.concatenate(logits), math.sqrt(
        residual_square / max(1, residual_elements)
    )


@torch.no_grad()
def _predict_official_implementation(
    implementation: nn.Module,
    broadband: torch.Tensor,
    *,
    node_reconstruction: torch.Tensor,
    device: str,
    batch_size: int,
) -> np.ndarray:
    implementation.eval()
    parts: list[np.ndarray] = []
    for start in range(0, len(broadband), int(batch_size)):
        stop = min(start + int(batch_size), len(broadband))
        batch = broadband[start:stop].to(device)
        reconstructed = torch.einsum(
            "ck,nkt->nct",
            node_reconstruction.to(batch),
            batch,
        )
        parts.append(
            implementation(reconstructed)
            .float()
            .cpu()
            .numpy()
        )
    return np.concatenate(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-output", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--evidence-prior-root", required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0012)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--config",
        default="configs/experiments/v7_e3_r9_phase_pair_current.yaml",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.accumulation_steps < 1:
        parser.error("epochs, batch size, and accumulation must be positive")
    configure_cache_env()
    _seed(int(args.seed) + int(args.fold) * 1009)
    started = time.time()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    parent_config_path = (ROOT / config["parent_config"]).resolve()
    parent_config = yaml.safe_load(parent_config_path.read_text(encoding="utf-8"))
    model_config = dict(parent_config["model"])
    model_config.update(dict(config["model_overrides"]))

    subject_path = _subject_file(Path(args.data).resolve(), int(args.subject))
    data = load_processed_npz(subject_path)
    session = np.asarray(data["session"]).astype(str)
    t_indices = np.flatnonzero(session == "T")
    metadata = _metadata_rows(data, t_indices)
    folds = session_t_run_grouped_folds(metadata, n_splits=6, seed=0, shuffle=True)
    train_indices, validation_indices = folds[int(args.fold)]
    x_t = np.asarray(data["X"])[t_indices]
    y_t = np.asarray(data["y"], dtype=np.int64)[t_indices]

    checkpoint = (
        Path(args.parent_output).resolve()
        / f"subject_{int(args.subject):02d}"
        / "selection"
        / f"fold_{int(args.fold)}"
        / "best.pt"
    )
    model = build_delay_stage_model(
        seed=int(args.seed),
        model_config=model_config,
        parent_checkpoint=checkpoint,
        stage=str(config["stage"]),
        train_readout=False,
        readout_training_scope="fixed_phase_adapter",
        initial_atc_residual_scale=0.0,
        slow_carrier_residual_scale=0.0,
        delayed_statistics_interaction_gain=0.0,
        official_source_root=str(Path(args.source_root).resolve()),
    )
    gain = fit_official_fbc_channel_gain(
        x_t[train_indices],
        sfreq=float(data["sfreq"]),
        epoch_tmin=float(data["epoch_tmin"]),
        clip=float(parent_config["preprocessing"]["clip_after_gain"]),
    )
    train_rates = cache_analytic_fixed_channel_gain_rate_features(
        model,
        x_t[train_indices],
        gain,
        device=args.device,
        batch_size=max(4, int(args.batch_size)),
    )
    validation_rates = cache_analytic_fixed_channel_gain_rate_features(
        model,
        x_t[validation_indices],
        gain,
        device=args.device,
        batch_size=max(4, int(args.batch_size)),
    )
    prior, provenance = _load_audited_fold_prior(
        Path(args.evidence_prior_root).resolve(),
        subject=int(args.subject),
        fold=int(args.fold),
        scope="pooled",
    )
    if provenance["frontend_fingerprint"] != train_rates.frontend_fingerprint:
        raise RuntimeError("prior and online feature frontends do not match")
    model.load_fold_local_slow_prior(
        torch.from_numpy(prior["route_probability"]),
        torch.from_numpy(prior["positive_delay_probability"]),
        torch.from_numpy(prior["fractional_delay_target"]),
    )
    model.load_fold_local_phase_amplitude_scale(
        _fit_phase_amplitude_scale(train_rates.fast)
    )
    project_registered_max_norm_constraints_(model)
    delay_hash = state_sha256(model.delay)
    train_phase = _cache_phase_current(
        model,
        train_rates,
        device=args.device,
        batch_size=max(4, int(args.batch_size)),
    )
    validation_phase = _cache_phase_current(
        model,
        validation_rates,
        device=args.device,
        batch_size=max(4, int(args.batch_size)),
    )
    if state_sha256(model.delay) != delay_hash:
        raise RuntimeError("phase-current caching mutated the audited delay state")

    core = getattr(model.atc_readout, "core", None)
    implementation = getattr(core, "module", None)
    if implementation is None or not hasattr(implementation, "atc_blocks"):
        raise RuntimeError("feature probe requires the official ATC implementation")
    fusion = PhaseATCFeatureFusion(
        copy.deepcopy(implementation),
        n_bands=model.n_bands,
        n_nodes=model.n_nodes,
        node_reconstruction=model.atc_readout.node_reconstruction,
    ).to(args.device)
    fusion.implementation.eval()
    calibration = _fit_phase_feature_scale(
        fusion,
        train_rates.broadband,
        train_phase,
        device=args.device,
        batch_size=max(4, int(args.batch_size)),
    )
    frozen_hash = state_sha256(fusion.implementation)

    parent_prediction_path = checkpoint.parent / "validation_predictions.npz"
    with np.load(parent_prediction_path, allow_pickle=False) as archive:
        parent_indices = archive["indices"].astype(np.int64)
        parent_logits = archive["logits"].astype(np.float32)
        parent_labels = archive["labels"].astype(np.int64)
    if not np.array_equal(parent_indices, np.asarray(validation_indices)):
        raise RuntimeError("parent predictions do not match the registered fold")
    if not np.array_equal(parent_labels, y_t[validation_indices]):
        raise RuntimeError("parent labels do not match the registered fold")

    zero_logits, zero_residual_rms = _predict(
        fusion,
        validation_rates.broadband,
        torch.zeros_like(validation_phase),
        device=args.device,
        batch_size=max(4, int(args.batch_size)),
    )
    direct_logits = _predict_official_implementation(
        fusion.implementation,
        validation_rates.broadband,
        node_reconstruction=fusion.node_reconstruction,
        device=args.device,
        batch_size=max(4, int(args.batch_size)),
    )
    zero_logit_max_abs_difference = float(np.max(np.abs(zero_logits - parent_logits)))
    zero_vs_direct_max_abs_difference = float(
        np.max(np.abs(zero_logits - direct_logits))
    )
    direct_vs_parent_max_abs_difference = float(
        np.max(np.abs(direct_logits - parent_logits))
    )
    stored_prediction_matches = bool(
        np.array_equal(direct_logits.argmax(1), parent_logits.argmax(1))
    )
    if (
        zero_vs_direct_max_abs_difference != 0.0
        or zero_residual_rms != 0.0
        or direct_vs_parent_max_abs_difference > 1e-4
        or not stored_prediction_matches
    ):
        print(
            json.dumps(
                {
                    "zero_residual_rms": zero_residual_rms,
                    "zero_vs_direct_max_abs_difference": (
                        zero_vs_direct_max_abs_difference
                    ),
                    "direct_vs_parent_max_abs_difference": (
                        direct_vs_parent_max_abs_difference
                    ),
                    "zero_vs_parent_max_abs_difference": (
                        zero_logit_max_abs_difference
                    ),
                    "zero_accuracy": float(
                        (zero_logits.argmax(1) == parent_labels).mean()
                    ),
                    "direct_accuracy": float(
                        (direct_logits.argmax(1) == parent_labels).mean()
                    ),
                    "stored_parent_accuracy": float(
                        (parent_logits.argmax(1) == parent_labels).mean()
                    ),
                },
                indent=2,
            ),
            flush=True,
        )
        raise RuntimeError("zero phase feature does not reproduce the parent exactly")

    trainable = [parameter for parameter in fusion.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    dataset = TensorDataset(
        train_rates.broadband.float(),
        train_phase.float(),
        torch.from_numpy(y_t[train_indices]).long(),
    )
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        generator = torch.Generator().manual_seed(
            int(args.seed) + int(args.fold) * 1009 + epoch
        )
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        fusion.train()
        fusion.implementation.eval()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        trials = 0
        for batch_index, (broadband, phase, labels) in enumerate(loader, start=1):
            logits, _ = fusion(
                broadband.to(args.device),
                phase.to(args.device),
            )
            loss = F.cross_entropy(logits, labels.to(args.device))
            (loss / float(args.accumulation_steps)).backward()
            if (
                batch_index % int(args.accumulation_steps) == 0
                or batch_index == len(loader)
            ):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach()) * len(labels)
            trials += len(labels)
        history.append({"epoch": epoch, "train_loss": loss_sum / trials})
        print(
            f"__V7_PHASE_FEATURE_PROGRESS__ epoch={epoch}/{args.epochs} "
            f"loss={loss_sum / trials:.6f}",
            flush=True,
        )

    full_logits, full_residual_rms = _predict(
        fusion,
        validation_rates.broadband,
        validation_phase,
        device=args.device,
        batch_size=max(4, int(args.batch_size)),
    )
    final_zero_logits, final_zero_residual_rms = _predict(
        fusion,
        validation_rates.broadband,
        torch.zeros_like(validation_phase),
        device=args.device,
        batch_size=max(4, int(args.batch_size)),
    )
    final_zero_max_abs_difference = float(
        np.max(np.abs(final_zero_logits - direct_logits))
    )
    if final_zero_max_abs_difference != 0.0 or final_zero_residual_rms != 0.0:
        raise RuntimeError("trained feature probe changed its matched-zero parent")
    if state_sha256(fusion.implementation) != frozen_hash:
        raise RuntimeError("feature fusion mutated the frozen ATC implementation")
    if state_sha256(model.delay) != delay_hash:
        raise RuntimeError("feature fusion mutated the audited delay state")

    full_prediction = full_logits.argmax(axis=1)
    zero_prediction = final_zero_logits.argmax(axis=1)
    labels = y_t[validation_indices]
    full_correct = full_prediction == labels
    zero_correct = zero_prediction == labels
    corrections = np.flatnonzero(full_correct & ~zero_correct)
    regressions = np.flatnonzero(~full_correct & zero_correct)
    result = {
        "status": "completed",
        "experiment": "v7_r11b_calibrated_phase_atc_feature_fusion_probe",
        "subject": int(args.subject),
        "seed": int(args.seed),
        "fold": int(args.fold),
        "epochs": int(args.epochs),
        "selection_session": "T",
        "session_e_accessed": False,
        "trainable_parameters": int(sum(value.numel() for value in trainable)),
        "full_accuracy": float(full_correct.mean()),
        "matched_zero_accuracy": float(zero_correct.mean()),
        "delta_accuracy": float(full_correct.mean() - zero_correct.mean()),
        "corrections": int(len(corrections)),
        "regressions": int(len(regressions)),
        "correction_indices": np.asarray(validation_indices)[corrections].tolist(),
        "regression_indices": np.asarray(validation_indices)[regressions].tolist(),
        "initial_zero_logit_max_abs_difference": zero_logit_max_abs_difference,
        "initial_zero_vs_checkpoint_max_abs_difference": (
            zero_vs_direct_max_abs_difference
        ),
        "stored_parent_logit_max_abs_difference": (
            direct_vs_parent_max_abs_difference
        ),
        "stored_parent_predictions_match": stored_prediction_matches,
        "final_zero_vs_checkpoint_max_abs_difference": (
            final_zero_max_abs_difference
        ),
        "full_feature_residual_rms": full_residual_rms,
        "matched_zero_feature_residual_rms": final_zero_residual_rms,
        "phase_gain_abs_mean": float(fusion.phase_gain.detach().abs().mean()),
        "phase_gain_abs_max": float(fusion.phase_gain.detach().abs().max()),
        "feature_gain_abs_mean": float(fusion.feature_gain.detach().abs().mean()),
        "feature_gain_abs_max": float(fusion.feature_gain.detach().abs().max()),
        "feature_rms_calibration": calibration,
        "frozen_atc_state_unchanged": True,
        "audited_delay_state_unchanged": True,
        "elapsed_seconds": time.time() - started,
        "prior": provenance,
        "config": str(config_path),
    }
    output = ensure_dir(Path(args.output).resolve())
    write_csv(output / "history.csv", history)
    write_json(output / "result.json", result)
    np.savez_compressed(
        output / "paired_validation_predictions.npz",
        indices=np.asarray(validation_indices, dtype=np.int64),
        labels=labels.astype(np.int64),
        full_logits=full_logits.astype(np.float32),
        matched_zero_logits=final_zero_logits.astype(np.float32),
    )
    torch.save(
        {
            "phase_band_gain": fusion.phase_band_gain.detach().cpu(),
            "phase_feature_scale": fusion.phase_feature_scale.detach().cpu(),
            "feature_gain_raw": fusion.feature_gain_raw.detach().cpu(),
        },
        output / "adapter.pt",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
