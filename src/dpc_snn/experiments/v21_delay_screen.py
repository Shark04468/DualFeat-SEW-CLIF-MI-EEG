"""Causal latent-lag screening before training a sparse E21 delay residual."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from dpc_snn.experiments.v17_information_replay import residual_fusion_logits
from dpc_snn.utils.metrics import classification_metrics


def causal_shift(sequence: np.ndarray, lag: int) -> np.ndarray:
    value = np.asarray(sequence, dtype=np.float32)
    if value.ndim != 3 or int(lag) < 0 or int(lag) >= value.shape[1]:
        raise ValueError("causal latent lag is outside the sequence support")
    if int(lag) == 0:
        return value.copy()
    output = np.zeros_like(value)
    output[:, int(lag) :] = value[:, : -int(lag)]
    return output


def select_lag_scale(
    base_logits: np.ndarray,
    corrections_by_lag: Mapping[int, np.ndarray],
    labels: np.ndarray,
    *,
    lags: Sequence[int] = (0, 1, 2, 4),
    scales: Sequence[float] = (0.0, 0.5, 1.0),
    n_classes: int = 4,
) -> tuple[int, float, list[dict[str, float | int]]]:
    lag_values = tuple(int(lag) for lag in lags)
    scale_values = tuple(float(scale) for scale in scales)
    if len(lag_values) * len(scale_values) > 12:
        raise ValueError("lag-scale screen exceeds 12 candidates")
    if 0 not in lag_values or 0.0 not in scale_values:
        raise ValueError("lag-scale screen must contain the matched zero control")
    y = np.asarray(labels, dtype=np.int64)
    rows: list[dict[str, float | int]] = []
    best_key: tuple[float, float, float, int] | None = None
    selected_lag = 0
    selected_scale = 0.0
    for lag in lag_values:
        if lag not in corrections_by_lag:
            raise ValueError(f"missing correction logits for lag {lag}")
        for scale in scale_values:
            logits = residual_fusion_logits(base_logits, corrections_by_lag[lag], scale)
            metrics = classification_metrics(y, logits.argmax(axis=1), n_classes=n_classes)
            row: dict[str, float | int] = {
                "lag": lag,
                "scale": scale,
                "validation_accuracy": float(metrics["accuracy"]),
                "validation_kappa": float(metrics["kappa"]),
            }
            rows.append(row)
            key = (
                float(row["validation_kappa"]),
                float(row["validation_accuracy"]),
                -scale,
                -lag,
            )
            if best_key is None or key > best_key:
                best_key = key
                selected_lag = lag
                selected_scale = scale
    return selected_lag, selected_scale, rows
