"""NumPy spike encoding helpers for QC and source-data export."""

from __future__ import annotations

import numpy as np


def normalize01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    lo = x.min(axis=-1, keepdims=True)
    hi = x.max(axis=-1, keepdims=True)
    return (x - lo) / np.maximum(hi - lo, eps)


def rate_code(amplitude: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (normalize01(amplitude) >= threshold).astype(np.float32)


def phase_aware_code(
    amplitude: np.ndarray,
    phase: np.ndarray,
    phase_preference: np.ndarray | None = None,
    alpha: float = 4.0,
    beta: float = 2.0,
    threshold: float = 0.5,
) -> np.ndarray:
    if phase_preference is None:
        phase_preference = np.zeros(amplitude.shape[:-1], dtype=np.float32)[..., None]
    logits = alpha * (normalize01(amplitude) - 0.5) + beta * np.cos(phase - phase_preference)
    prob = 1.0 / (1.0 + np.exp(-logits))
    return (prob >= threshold).astype(np.float32)

