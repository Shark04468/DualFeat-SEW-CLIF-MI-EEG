"""Locked probability-fusion rules shared by V8 training and evaluation."""

from __future__ import annotations

import math

import numpy as np


def entropy_residual_probability(
    anchor_probability: np.ndarray,
    decoder_probability: np.ndarray,
    *,
    maximum_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the fixed anchor-entropy residual rule without learned calibration."""

    anchor = np.asarray(anchor_probability, dtype=np.float64)
    decoder = np.asarray(decoder_probability, dtype=np.float64)
    if anchor.shape != decoder.shape or anchor.ndim != 2 or anchor.shape[1] < 2:
        raise ValueError("anchor and decoder probabilities must have the same [N, C] shape")
    if not 0.0 <= float(maximum_weight) <= 1.0:
        raise ValueError("maximum residual weight must lie in [0, 1]")
    if not np.isfinite(anchor).all() or not np.isfinite(decoder).all():
        raise FloatingPointError("residual fusion received non-finite probabilities")
    if not np.allclose(anchor.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("anchor rows are not normalized probabilities")
    if not np.allclose(decoder.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("decoder rows are not normalized probabilities")
    clipped = np.clip(anchor, 1e-12, 1.0)
    normalized_entropy = -(anchor * np.log(clipped)).sum(axis=1) / math.log(
        anchor.shape[1]
    )
    gate = float(maximum_weight) * normalized_entropy
    fused = (1.0 - gate[:, None]) * anchor + gate[:, None] * decoder
    return fused.astype(np.float32), gate.astype(np.float32)
