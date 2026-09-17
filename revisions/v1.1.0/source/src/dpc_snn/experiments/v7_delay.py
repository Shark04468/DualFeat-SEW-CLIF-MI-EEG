"""Scientifically matched delay-stage controls for the V7 campaign."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dpc_snn.experiments.v62_scaffold import build_zero_scaffold
from dpc_snn.models.dasp_snn_v62 import DASPSNNV62
from dpc_snn.utils.metrics import exact_mcnemar_p


@dataclass(frozen=True)
class DelayStageSpec:
    name: str
    slow_delay: bool
    cross_band: bool
    fast_delay: bool
    dynamic_residual: bool
    phase_residual: bool
    slow_route_scale: float
    fast_route_scale: float
    slow_carrier_residual_scale: float
    initial_slow_delay_fraction: float | None
    initial_slow_delay_std_samples: float | None
    target_interaction_bound: float
    initial_target_interaction: float | None
    phase_bound: float
    initial_atc_residual_scale: float
    delayed_statistics_interaction_gain: float


def paired_delay_gate(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the pre-registered paired delay gate across subject-seed rows."""

    required_pairs = int(config["gate"]["total_subject_seed_pairs"])
    if len(rows) != required_pairs:
        return {
            "passed": False,
            "reason": f"expected {required_pairs} subject-seed pairs, found {len(rows)}",
            "next_stage_allowed": False,
        }
    deltas = np.asarray([float(row["delta_accuracy"]) for row in rows])
    amplitude = np.asarray(
        [float(row["amplitude_relative_difference"]) for row in rows]
    )
    median_threshold = float(config["gate"]["median_delta_pp"]) / 100.0
    positive = int(np.count_nonzero(deltas > 0.0))
    amplitude_tolerance = float(
        config["control"]["amplitude_rms_tolerance_fraction"]
    )
    passed = bool(
        np.median(deltas) >= median_threshold
        and positive >= int(config["gate"]["minimum_positive_pairs"])
        and np.max(amplitude) <= amplitude_tolerance
    )
    report = {
        "passed": passed,
        "median_delta_accuracy": float(np.median(deltas)),
        "mean_delta_accuracy": float(np.mean(deltas)),
        "median_delta_pp": float(100.0 * np.median(deltas)),
        "positive_pairs": positive,
        "total_pairs": len(rows),
        "maximum_amplitude_relative_difference": float(np.max(amplitude)),
        "amplitude_tolerance_fraction": amplitude_tolerance,
        "next_stage_allowed": passed,
    }
    if all(
        "first_only_correct" in row and "second_only_correct" in row
        for row in rows
    ):
        first_only = sum(int(row["first_only_correct"]) for row in rows)
        second_only = sum(int(row["second_only_correct"]) for row in rows)
        report.update(
            {
                "pooled_full_only_correct": first_only,
                "pooled_zero_only_correct": second_only,
                "pooled_discordant_predictions": first_only + second_only,
                "pooled_exact_mcnemar_p": exact_mcnemar_p(
                    first_only,
                    second_only,
                ),
            }
        )
    return report


DELAY_STAGES: dict[str, DelayStageSpec] = {
    "static_slow_within": DelayStageSpec(
        name="static_slow_within",
        slow_delay=True,
        cross_band=False,
        fast_delay=False,
        dynamic_residual=False,
        phase_residual=False,
        slow_route_scale=1.0,
        fast_route_scale=0.0,
        slow_carrier_residual_scale=0.0,
        initial_slow_delay_fraction=0.25,
        initial_slow_delay_std_samples=0.75,
        # D1 is explicitly target-conditioned even though it is still
        # within-band. Cross-band routing is introduced only in D2.
        target_interaction_bound=0.50,
        initial_target_interaction=0.10,
        phase_bound=0.0,
        # D1 is a route-only delay test. Any non-zero multiband residual alters
        # the matched-zero parent even when every audited lag is clamped to 0.
        initial_atc_residual_scale=0.0,
        delayed_statistics_interaction_gain=1.0,
    ),
    "static_slow_crossband": DelayStageSpec(
        name="static_slow_crossband",
        slow_delay=True,
        cross_band=True,
        fast_delay=False,
        dynamic_residual=False,
        phase_residual=False,
        slow_route_scale=0.10,
        fast_route_scale=0.0,
        slow_carrier_residual_scale=0.05,
        initial_slow_delay_fraction=None,
        initial_slow_delay_std_samples=None,
        target_interaction_bound=0.50,
        initial_target_interaction=None,
        phase_bound=0.0,
        initial_atc_residual_scale=0.05,
        delayed_statistics_interaction_gain=0.0,
    ),
    "static_dual_delay": DelayStageSpec(
        name="static_dual_delay",
        slow_delay=True,
        cross_band=True,
        fast_delay=True,
        dynamic_residual=False,
        phase_residual=False,
        slow_route_scale=0.10,
        fast_route_scale=0.05,
        slow_carrier_residual_scale=0.05,
        initial_slow_delay_fraction=None,
        initial_slow_delay_std_samples=None,
        target_interaction_bound=0.50,
        initial_target_interaction=None,
        phase_bound=0.0,
        initial_atc_residual_scale=0.05,
        delayed_statistics_interaction_gain=0.0,
    ),
    "dynamic_residual": DelayStageSpec(
        name="dynamic_residual",
        slow_delay=True,
        cross_band=True,
        fast_delay=True,
        dynamic_residual=True,
        phase_residual=False,
        slow_route_scale=0.10,
        fast_route_scale=0.05,
        slow_carrier_residual_scale=0.05,
        initial_slow_delay_fraction=None,
        initial_slow_delay_std_samples=None,
        target_interaction_bound=0.50,
        initial_target_interaction=None,
        phase_bound=0.0,
        initial_atc_residual_scale=0.05,
        delayed_statistics_interaction_gain=0.0,
    ),
    "phase_residual": DelayStageSpec(
        name="phase_residual",
        slow_delay=True,
        cross_band=True,
        fast_delay=True,
        dynamic_residual=False,
        phase_residual=True,
        slow_route_scale=0.10,
        fast_route_scale=0.05,
        slow_carrier_residual_scale=0.05,
        initial_slow_delay_fraction=None,
        initial_slow_delay_std_samples=None,
        target_interaction_bound=0.50,
        initial_target_interaction=None,
        phase_bound=0.15,
        initial_atc_residual_scale=0.05,
        delayed_statistics_interaction_gain=0.0,
    ),
}


def _set_module_trainable(module: torch.nn.Module | None, enabled: bool) -> None:
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _configure_atc_training_scope(
    core: torch.nn.Module,
    scope: str,
) -> tuple[torch.nn.Module, ...]:
    """Configure ATC adaptation without unfreezing its feature extractor by accident."""

    normalized = str(scope).strip().lower()
    if normalized not in {
        "frozen",
        "delay_head",
        "fixed_delay_head",
        "fixed_head",
        "fixed_full",
        "fixed_phase_adapter",
        "head",
        "full",
    }:
        raise ValueError(
            "ATC training scope must be 'frozen', 'delay_head', "
            "'fixed_delay_head', 'fixed_head', 'fixed_full', "
            "'fixed_phase_adapter', 'head', or 'full'"
        )
    _set_module_trainable(core, normalized in {"full", "fixed_full"})
    if normalized in {"full", "fixed_full"}:
        return ()
    if normalized in {"head", "fixed_head"}:
        implementation = getattr(core, "module", core)
        blocks = getattr(implementation, "atc_blocks", None)
        if blocks is None:
            raise ValueError("head-only ATC adaptation requires registered window blocks")
        heads = [getattr(block, "linear", None) for block in blocks]
        if not heads or any(head is None for head in heads):
            raise ValueError("head-only ATC adaptation could not locate every window head")
        for head in heads:
            _set_module_trainable(head, True)
    # Keep frozen feature extractors and their BatchNorm statistics in eval
    # mode. Eval mode does not prevent gradients through trainable heads.
    return (core,)


def _set_named_delay_parameters(
    model: DASPSNNV62,
    prefixes: tuple[str, ...],
    names: tuple[str, ...] = (),
) -> None:
    for name, parameter in model.delay.named_parameters():
        if name.startswith(prefixes) or name in names:
            parameter.requires_grad_(True)


def _initialize_constant_coupled_field(
    field: torch.nn.Module,
    value: float,
) -> None:
    """Initialize a coupled field to a constant using one exact rank component."""

    band = getattr(field, "band")
    node = getattr(field, "node")
    rank_scale_raw = getattr(field, "rank_scale_raw")
    if band.shape[-1] < 1 or node.shape[-1] != band.shape[-1]:
        raise ValueError("coupled target-interaction field has incompatible factors")
    scale = torch.nn.functional.softplus(rank_scale_raw[0]).clamp_min(1e-8)
    with torch.no_grad():
        band.zero_()
        # A constant node matrix has Frobenius norm K. Gauge normalization
        # removes K and immediately restores it in the band factor, leaving
        # band * softplus(rank_scale) as the exact field value.
        node[..., 0].fill_(1.0)
        # Retain finite, non-zero gauges for inactive ranks. Their zero band
        # coefficients make their forward contribution exactly zero without
        # creating a 1e-8 normalization singularity during backpropagation.
        band[..., 0].fill_(float(value) / float(scale))


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return value + torch.log(-torch.expm1(-value))


def _calibrate_within_band_delay_spread(
    model: DASPSNNV62,
    target_std_samples: float,
) -> None:
    """Scale the existing low-rank gauge to a data-independent delay spread."""

    target = float(target_std_samples)
    if target <= 0.0 or target >= float(model.delay.slow_max_delay) / 2.0:
        raise ValueError("initial slow delay std is outside the calibrated range")
    field = model.delay.slow_delay_field
    index = torch.arange(model.n_bands, device=field.band.device)
    base = field().detach()[index, index]
    bias = model.delay.slow_delay_bias.detach()

    def delay_std(multiplier: float) -> float:
        delay = model.delay.slow_max_delay * torch.sigmoid(
            float(multiplier) * base + bias
        )
        return float(delay.std(unbiased=False))

    low, high = 0.0, 1.0
    while delay_std(high) < target and high < 1e6:
        high *= 2.0
    if high >= 1e6 and delay_std(high) < target:
        raise RuntimeError("could not calibrate a non-degenerate delay field")
    for _ in range(48):
        middle = 0.5 * (low + high)
        if delay_std(middle) < target:
            low = middle
        else:
            high = middle
    multiplier = 0.5 * (low + high)
    with torch.no_grad():
        positive_scale = torch.nn.functional.softplus(field.rank_scale_raw)
        field.rank_scale_raw.copy_(_inverse_softplus(positive_scale * multiplier))


def configure_delay_stage(
    model: DASPSNNV62,
    stage: str | DelayStageSpec,
    *,
    train_readout: bool = True,
    readout_training_scope: str | None = None,
    initial_slow_delay_fraction: float | None = None,
    initial_atc_residual_scale: float | None = None,
    slow_route_scale: float | None = None,
    slow_carrier_residual_scale: float | None = None,
    delayed_statistics_interaction_gain: float | None = None,
) -> DelayStageSpec:
    """Configure one registered single-variable delay stage.

    The classifier may adapt, but every input remains downstream of mandatory
    transport. Base delay and phase preference are never trainable together.
    """

    spec = DELAY_STAGES[stage] if isinstance(stage, str) else stage
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.force_zero_delay = False
    model.slow_delay_enabled = spec.slow_delay
    model.fast_delay_enabled = spec.fast_delay
    model.cross_band_enabled = spec.cross_band
    model.phase_residual_enabled = spec.phase_residual
    model.delay.cross_band_enabled = spec.cross_band
    model.delay.slow_residual_route_scale = float(
        spec.slow_route_scale if slow_route_scale is None else slow_route_scale
    )
    if not 0.0 <= model.delay.slow_residual_route_scale <= 1.0:
        raise ValueError("slow route scale must lie in [0, 1]")
    model.delay.fast_residual_route_scale = float(spec.fast_route_scale)
    model.delay.slow_carrier_residual_scale = float(
        spec.slow_carrier_residual_scale
        if slow_carrier_residual_scale is None
        else slow_carrier_residual_scale
    )
    if not 0.0 <= model.delay.slow_carrier_residual_scale <= 1.0:
        raise ValueError("slow carrier residual scale must lie in [0, 1]")
    model.delay.target_interaction_bound = float(spec.target_interaction_bound)
    model.delay.phase_bound = float(spec.phase_bound)
    selected_delay_fraction = (
        spec.initial_slow_delay_fraction
        if initial_slow_delay_fraction is None
        else initial_slow_delay_fraction
    )
    if selected_delay_fraction is not None:
        fraction = float(selected_delay_fraction)
        if not 0.0 < fraction < 1.0:
            raise ValueError("initial slow delay fraction must lie strictly in (0, 1)")
        with torch.no_grad():
            model.delay.slow_delay_bias.fill_(
                float(torch.logit(torch.tensor(fraction)))
            )
    selected_delay_std = spec.initial_slow_delay_std_samples
    if selected_delay_std is not None:
        _calibrate_within_band_delay_spread(model, selected_delay_std)

    # The official ATC core is fine-tuned from the exact E2 checkpoint. Its
    # multiband residual is bounded and receives only mandatory delayed current.
    if model.atc_readout is not None:
        selected_interaction_gain = (
            spec.delayed_statistics_interaction_gain
            if delayed_statistics_interaction_gain is None
            else float(delayed_statistics_interaction_gain)
        )
        if selected_interaction_gain < 0.0:
            raise ValueError("delayed statistics interaction gain must be non-negative")
        if hasattr(model.atc_readout, "delayed_statistics_interaction_gain"):
            model.atc_readout.delayed_statistics_interaction_gain = float(
                selected_interaction_gain
            )
        core = getattr(model.atc_readout, "core", model.atc_readout)
        scope = (
            str(readout_training_scope)
            if readout_training_scope is not None
            else ("full" if train_readout else "frozen")
        )
        model._frozen_eval_modules = _configure_atc_training_scope(core, scope)
        delayed_heads = tuple(
            head
            for head in (
                getattr(
                    model.atc_readout,
                    "delayed_statistics_classifier",
                    None,
                ),
                getattr(
                    model.atc_readout,
                    "delayed_route_statistics_classifier",
                    None,
                ),
            )
            if head is not None
        )
        phase_pair_adapter = getattr(model, "phase_pair_adapter", None)
        _set_module_trainable(
            phase_pair_adapter,
            scope == "fixed_phase_adapter",
        )
        if scope == "fixed_phase_adapter":
            if phase_pair_adapter is None or not model.phase_pair_current_enabled:
                raise ValueError(
                    "fixed phase-adapter training requires phase-pair current"
                )
            if delayed_heads:
                raise ValueError(
                    "fixed phase-adapter training forbids delayed logit heads"
                )
        for delayed_head in delayed_heads:
            _set_module_trainable(
                delayed_head,
                scope
                in {
                    "delay_head",
                    "fixed_delay_head",
                    "fixed_head",
                    "fixed_full",
                    "head",
                    "full",
                },
            )
        if delayed_heads and scope in {
            "delay_head",
            "fixed_delay_head",
            "fixed_head",
            "fixed_full",
            "head",
        }:
            # The delay-contrast head must be exactly zero when D_tau == D_0.
            # A trainable class bias would survive the locked-zero intervention.
            for delayed_head in delayed_heads:
                with torch.no_grad():
                    delayed_head.bias.zero_()
                delayed_head.bias.requires_grad_(False)
        for name in (
            "delayed_statistics_gate_bias",
            "delayed_statistics_gate_slope_raw",
        ):
            parameter = getattr(model.atc_readout, name, None)
            if parameter is not None:
                parameter.requires_grad_(
                    scope
                    in {
                        "delay_head",
                        "fixed_delay_head",
                        "fixed_head",
                        "fixed_full",
                        "head",
                        "full",
                    }
                )
        if scope in {"delay_head", "fixed_delay_head"} and not delayed_heads:
            raise ValueError("delay-head adaptation requires delayed statistics")
        residual = getattr(model.atc_readout, "delayed_residual_mix_raw", None)
        bound = float(getattr(model.atc_readout, "delayed_residual_bound", 0.0))
        if residual is not None:
            residual.requires_grad_(
                scope not in {"fixed_delay_head", "fixed_phase_adapter"}
            )
            if bound <= 0.0:
                raise ValueError("delay stage requires a positive delayed ATC residual bound")
            initial_scale = (
                spec.initial_atc_residual_scale
                if initial_atc_residual_scale is None
                else float(initial_atc_residual_scale)
            )
            if abs(initial_scale) >= bound:
                raise ValueError("initial delayed ATC residual scale must lie inside its bound")
            target = min(abs(initial_scale) / bound, 1.0 - 1e-6)
            if float(residual.detach().abs()) < 1e-8:
                with torch.no_grad():
                    residual.fill_(math.atanh(target))
    else:
        _set_module_trainable(model.temporal_pyramid, bool(train_readout))
        _set_module_trainable(model.geometry, bool(train_readout))
        _set_module_trainable(model.decoder, bool(train_readout))
        model._frozen_eval_modules = (
            ()
            if train_readout
            else tuple(
                module
                for module in (model.temporal_pyramid, model.geometry, model.decoder)
                if module is not None
            )
        )

    if spec.slow_delay:
        _set_named_delay_parameters(
            model,
            ("slow_amplitude.", "slow_gate_field."),
            ("slow_gate_bias",),
        )
    if spec.fast_delay:
        _set_named_delay_parameters(
            model,
            ("fast_amplitude.", "fast_gate_field."),
            ("fast_gate_bias", "fast_contribution_raw"),
        )
    if spec.target_interaction_bound > 0.0:
        if spec.initial_target_interaction is not None:
            target = float(spec.initial_target_interaction)
            if abs(target) >= float(spec.target_interaction_bound):
                raise ValueError("initial target interaction must lie inside its bound")
            raw_target = math.atanh(target / float(spec.target_interaction_bound))
            _initialize_constant_coupled_field(model.delay.target_interaction, raw_target)
        _set_named_delay_parameters(model, ("target_interaction.",))

    if spec.phase_residual:
        # Tau is fixed before theta is allowed to move.
        _set_named_delay_parameters(
            model,
            (),
            ("phase_preference", "phase_strength_raw"),
        )
    elif spec.dynamic_residual:
        if model.context_encoder is None:
            raise ValueError("dynamic residual stage requires a registered EEG context encoder")
        _set_module_trainable(model.context_encoder, True)
        _set_module_trainable(model.delay.slow_context, True)
        _set_module_trainable(model.delay.fast_context, True)
    else:
        if spec.slow_delay:
            _set_named_delay_parameters(
                model,
                ("slow_delay_field.",),
                ("slow_delay_bias",),
            )
        if spec.fast_delay:
            _set_named_delay_parameters(
                model,
                ("fast_delay_field.",),
                ("fast_delay_bias",),
            )

    if model.atc_readout is not None and scope in {
        "fixed_delay_head",
        "fixed_head",
        "fixed_full",
        "fixed_phase_adapter",
    }:
        for parameter in model.delay.parameters():
            parameter.requires_grad_(False)

    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("delay stage has no trainable parameters")
    return spec


def build_delay_stage_model(
    *,
    seed: int,
    model_config: dict[str, Any],
    parent_checkpoint: str | Path,
    stage: str,
    train_readout: bool = True,
    readout_training_scope: str | None = None,
    initial_slow_delay_fraction: float | None = None,
    initial_atc_residual_scale: float | None = None,
    slow_route_scale: float | None = None,
    slow_carrier_residual_scale: float | None = None,
    delayed_statistics_interaction_gain: float | None = None,
    official_source_root: str | None = None,
) -> DASPSNNV62:
    model = build_zero_scaffold(
        seed=seed,
        model_config=model_config,
        official_source_root=official_source_root,
    )
    payload = torch.load(parent_checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(payload, strict=False)
    allowed_missing_prefixes = (
        "atc_readout.delayed_statistics_classifier.",
        "atc_readout.delayed_route_statistics_classifier.",
        "atc_readout.delayed_statistics_gate_",
        "delay.fold_slow_",
        "delay.fold_phase_",
        "phase_pair_adapter.",
    )
    invalid_missing = [
        name
        for name in incompatible.missing_keys
        if not name.startswith(allowed_missing_prefixes)
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "parent checkpoint is structurally incompatible: "
            f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
        )
    model._parent_state_migration = {
        "missing_zero_initialized_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
    configure_delay_stage(
        model,
        stage,
        train_readout=train_readout,
        readout_training_scope=readout_training_scope,
        initial_slow_delay_fraction=initial_slow_delay_fraction,
        initial_atc_residual_scale=initial_atc_residual_scale,
        slow_route_scale=slow_route_scale,
        slow_carrier_residual_scale=slow_carrier_residual_scale,
        delayed_statistics_interaction_gain=delayed_statistics_interaction_gain,
    )
    return model


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def delay_diagnostics(model: DASPSNNV62) -> dict[str, float | int | str | bool]:
    parameters = model.delay._route_parameters()
    slow_delay = parameters["slow_delay"].float()
    fast_delay = parameters["fast_delay"].float()
    slow_weight = parameters["slow_weight"].float()
    fast_weight = parameters["fast_weight"].float()
    phase_trainable = bool(model.delay.phase_preference.requires_grad)
    tau_trainable = any(
        parameter.requires_grad
        for name, parameter in model.delay.named_parameters()
        if name.startswith(("slow_delay_field.", "fast_delay_field."))
        or name in {"slow_delay_bias", "fast_delay_bias"}
    )
    residual_scale = 0.0
    if model.atc_readout is not None and hasattr(
        model.atc_readout, "delayed_residual_scale"
    ):
        residual_scale = float(model.atc_readout.delayed_residual_scale.detach())
    target_interaction = parameters["eta"].float()
    phase_pair_scale = None
    if model.phase_pair_adapter is not None:
        phase_pair_scale = model.phase_pair_adapter.scale.detach().float()
    index = torch.arange(model.n_bands, device=slow_delay.device)
    active_slow_delay = (
        slow_delay
        if model.delay.cross_band_enabled
        else slow_delay[index, index]
    )
    fold_prior_ready = bool(model.delay.fold_slow_prior_ready)
    if fold_prior_ready:
        posterior = model.delay.fold_slow_positive_delay_prior.float().clamp_min(1e-8)
        posterior_entropy = -(posterior * posterior.log()).sum(dim=-1)
        posterior_entropy /= math.log(float(model.delay.slow_max_delay + 1))
    else:
        posterior_entropy = torch.zeros((), device=slow_delay.device)
    return {
        "slow_delay_mean_samples": float(slow_delay.mean()),
        "slow_delay_nonzero_mass": float((slow_delay > 0.25).float().mean()),
        "slow_active_delay_mean_samples": float(active_slow_delay.mean()),
        "slow_active_delay_std_samples": float(
            active_slow_delay.std(unbiased=False)
        ),
        "fast_delay_mean_samples": float(fast_delay.mean()),
        "fast_delay_nonzero_mass": float((fast_delay > 0.25).float().mean()),
        "slow_weight_abs_mean": float(slow_weight.abs().mean()),
        "fast_weight_abs_mean": float(fast_weight.abs().mean()),
        "slow_gate_mean": float(parameters["slow_gate"].float().mean()),
        "fast_gate_mean": float(parameters["fast_gate"].float().mean()),
        "atc_delayed_residual_scale": residual_scale,
        "target_interaction_bound": float(model.delay.target_interaction_bound),
        "target_interaction_abs_mean": float(target_interaction.abs().mean()),
        "cross_band_enabled": bool(model.delay.cross_band_enabled),
        "phase_trainable": phase_trainable,
        "tau_trainable": tau_trainable,
        "tau_theta_simultaneously_trainable": bool(phase_trainable and tau_trainable),
        "fold_slow_prior_ready": fold_prior_ready,
        "fold_slow_route_prior_min": float(model.delay.fold_slow_route_prior.min()),
        "fold_slow_route_prior_max": float(model.delay.fold_slow_route_prior.max()),
        "fold_slow_posterior_entropy_mean": float(posterior_entropy.mean()),
        "fold_slow_route_residual_enabled": bool(
            model.delay.fold_slow_route_residual_enabled
        ),
        "fold_slow_delay_residual_enabled": bool(
            model.delay.fold_slow_delay_residual_enabled
        ),
        "phase_pair_current_enabled": bool(model.phase_pair_current_enabled),
        "phase_pair_amplitude_scale_ready": bool(
            model.delay.fold_phase_amplitude_scale_ready
        ),
        "phase_pair_adapter_scale_abs_mean": (
            0.0 if phase_pair_scale is None else float(phase_pair_scale.abs().mean())
        ),
        "phase_pair_adapter_scale_abs_max": (
            0.0 if phase_pair_scale is None else float(phase_pair_scale.abs().max())
        ),
        "parameters": model.parameter_count,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "state_sha256": state_sha256(model),
    }
