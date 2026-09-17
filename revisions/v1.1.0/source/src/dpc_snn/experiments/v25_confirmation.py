"""Frozen selection helpers for the V25 cross-session campaign."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
import math
from typing import Any

import numpy as np


RESIDUAL_SCALES = (0.0, 0.25, 0.5, 1.0)


def upper_median_epoch(values: Iterable[int]) -> int:
    """Return a deterministic conservative median for fixed-epoch refits."""
    epochs = np.asarray([int(value) for value in values], dtype=np.int64)
    if epochs.ndim != 1 or epochs.size == 0 or np.any(epochs < 1):
        raise ValueError("fixed-epoch evidence must contain positive integers")
    return int(math.ceil(float(np.median(epochs))))


def select_aggregate_residual_scale(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_folds: int = 6,
) -> tuple[float, list[dict[str, float | int]]]:
    """Select one T-only scale by mean fold validation metrics.

    Kappa is primary, accuracy secondary, and the smaller scale wins exact ties.
    Every candidate must have exactly one observation per fold.
    """
    grouped: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        scale = float(row["scale"])
        if scale not in RESIDUAL_SCALES:
            raise ValueError(f"unexpected residual scale: {scale}")
        kappa = float(row["validation_kappa"])
        accuracy = float(row["validation_accuracy"])
        if not np.isfinite([kappa, accuracy]).all():
            raise ValueError("residual-scale evidence contains non-finite metrics")
        grouped[scale].append((kappa, accuracy))
    if set(grouped) != set(RESIDUAL_SCALES):
        raise ValueError("residual-scale evidence does not cover every candidate")
    if any(len(values) != int(expected_folds) for values in grouped.values()):
        raise ValueError("residual-scale evidence does not have exact fold coverage")

    summary: list[dict[str, float | int]] = []
    for scale in RESIDUAL_SCALES:
        values = np.asarray(grouped[scale], dtype=np.float64)
        summary.append(
            {
                "scale": scale,
                "folds": int(values.shape[0]),
                "mean_validation_kappa": float(values[:, 0].mean()),
                "mean_validation_accuracy": float(values[:, 1].mean()),
            }
        )
    selected = max(
        summary,
        key=lambda row: (
            float(row["mean_validation_kappa"]),
            float(row["mean_validation_accuracy"]),
            -float(row["scale"]),
        ),
    )
    return float(selected["scale"]), summary


def validate_v25_freeze(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the immutable V25 freeze contract."""
    value = dict(payload)
    if value.get("schema_version") != 1 or value.get("architecture_id") != (
        "v25_equal_probability_dual_feature_sew_clif_residual"
    ):
        raise ValueError("invalid V25 freeze identity")
    if value.get("subjects") != list(range(1, 10)) or value.get("seeds") != list(range(5)):
        raise ValueError("V25 freeze must cover 9 subjects and 5 seeds")
    if value.get("variants") != ["ann_plain_ce", "sew_clif_ce"]:
        raise ValueError("V25 freeze variants differ from the matched development comparison")
    if value.get("historical_data_exposure", {}).get("bci2a_session_e") is not True:
        raise ValueError("V25 must disclose historical Session-E exposure")
    entries = value.get("entries")
    if not isinstance(entries, dict) or len(entries) != 45:
        raise ValueError("V25 freeze must contain 45 subject-seed entries")
    for seed in range(5):
        for subject in range(1, 10):
            key = f"subject_{subject:02d}_seed_{seed}"
            entry = entries.get(key)
            if not isinstance(entry, dict):
                raise ValueError(f"missing V25 freeze entry: {key}")
            epochs = entry.get("fixed_epochs", {})
            if set(epochs) != {"atcnet", "fbcnet", "ann_plain_ce", "sew_clif_ce"}:
                raise ValueError(f"invalid fixed epochs for {key}")
            if any(int(epoch) < 1 for epoch in epochs.values()):
                raise ValueError(f"non-positive fixed epoch for {key}")
            scales = entry.get("residual_scales", {})
            if set(scales) != {"ann_plain_ce", "sew_clif_ce"}:
                raise ValueError(f"invalid residual scales for {key}")
            if any(float(scale) not in RESIDUAL_SCALES for scale in scales.values()):
                raise ValueError(f"unknown residual scale for {key}")
    return value
