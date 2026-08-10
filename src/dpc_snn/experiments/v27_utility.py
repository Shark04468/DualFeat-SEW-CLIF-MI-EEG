"""Registered perturbations, operation accounting, and gates for V27."""

from __future__ import annotations

from typing import Any

import numpy as np


def mask_random_time_windows(
    carrier: np.ndarray,
    *,
    duration_seconds: float,
    sfreq: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero one deterministic contiguous window independently in each trial."""
    values = np.asarray(carrier, dtype=np.float32)
    width = int(round(float(duration_seconds) * float(sfreq)))
    if values.ndim != 3 or width < 1 or width >= values.shape[-1]:
        raise ValueError("time-mask duration is outside the task carrier")
    rng = np.random.default_rng(int(seed))
    starts = rng.integers(0, values.shape[-1] - width + 1, size=values.shape[0])
    output = values.copy()
    for trial, start in enumerate(starts.tolist()):
        output[trial, :, start : start + width] = 0.0
    return np.ascontiguousarray(output), starts.astype(np.int64)


def dual_feature_operation_proxy(
    model: Any,
    *,
    mean_binary_firing_rate: float | None,
) -> dict[str, Any]:
    """Count student weight uses while separating shared analog and temporal work."""
    core = model.decoder
    steps = int(model.atc_steps)
    atc_projection = int(model.atc_projection.weight.numel() * model.atc_steps)
    fbc_projection = int(model.fbc_projection.weight.numel() * model.fbc_steps)
    interaction = int(model.interaction[1].weight.numel() * model.atc_steps)
    current_encoder = int(core.current_encoder.weight.numel() * steps)
    temporal = int(
        sum(
            (block.depthwise.weight.numel() + block.pointwise.weight.numel()) * steps
            for block in core.blocks
        )
    )
    readout = int(
        sum(
            parameter.numel()
            for module in (core.readout, core.classifier)
            for parameter in module.parameters()
            if parameter.ndim >= 2
        )
    )
    shared_analog = atc_projection + fbc_projection + interaction
    decoder_dense = current_encoder + temporal + readout
    full_dense = shared_analog + decoder_dense
    if model.decoder_kind == "ann":
        rate = None
        temporal_events = float(temporal)
    else:
        rate = float(mean_binary_firing_rate) if mean_binary_firing_rate is not None else -1.0
        if not 0.0 <= rate <= 1.0:
            raise ValueError("SNN operation accounting requires a valid binary firing rate")
        temporal_events = float(temporal) * rate
    decoder_events = float(current_encoder + readout) + temporal_events
    full_events = float(shared_analog) + decoder_events
    return {
        "decoder_kind": str(model.decoder_kind),
        "time_steps": steps,
        "atc_projection_weight_uses": atc_projection,
        "fbc_projection_weight_uses": fbc_projection,
        "interaction_weight_uses": interaction,
        "shared_analog_fusion_weight_uses": shared_analog,
        "current_encoder_weight_uses": current_encoder,
        "temporal_synaptic_weight_uses": temporal,
        "dense_readout_weight_uses": readout,
        "dense_decoder_weight_uses": decoder_dense,
        "dense_full_student_weight_uses": full_dense,
        "activity_weighted_decoder_events": decoder_events,
        "activity_weighted_full_student_events": full_events,
        "binary_spike_rate": rate,
        "hardware_energy_claim_allowed": False,
    }


def utility_gate(
    *,
    clean_replay_valid: bool,
    early_auc_gain_pp: float,
    robustness_gain_pp: float,
    decoder_reduction: float,
    full_student_reduction: float,
) -> dict[str, Any]:
    wins = {
        "early_auc_gain_at_least_0_5pp": float(early_auc_gain_pp) >= 0.5,
        "robustness_gain_at_least_0_5pp": float(robustness_gain_pp) >= 0.5,
        "decoder_operation_reduction_at_least_50pct": float(decoder_reduction) >= 0.50,
        "full_student_operation_reduction_at_least_20pct": (
            float(full_student_reduction) >= 0.20
        ),
    }
    passed = bool(clean_replay_valid and any(wins.values()))
    return {
        "status": "pass" if passed else "fail",
        "clean_replay_valid": bool(clean_replay_valid),
        "utility_wins": wins,
        "at_least_one_utility_win": any(wins.values()),
        "hardware_energy_claim_allowed": False,
    }
