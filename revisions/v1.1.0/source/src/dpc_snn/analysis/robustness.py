"""Robustness perturbations for E22."""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(x: np.ndarray, std: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (x + rng.normal(0.0, std, size=x.shape)).astype(np.float32)


def channel_dropout(x: np.ndarray, p: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(x).copy()
    mask = rng.random(out.shape[:2]) < p
    out[mask, :] = 0.0
    return out.astype(np.float32)


def temporal_jitter(x: np.ndarray, max_shift: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        shift = int(rng.integers(-max_shift, max_shift + 1))
        out[i] = np.roll(x[i], shift=shift, axis=-1)
    return out.astype(np.float32)


def flatline_channels(x: np.ndarray, p: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(x).copy()
    for n in range(out.shape[0]):
        for c in range(out.shape[1]):
            if rng.random() < p:
                out[n, c] = out[n, c, 0]
    return out.astype(np.float32)

