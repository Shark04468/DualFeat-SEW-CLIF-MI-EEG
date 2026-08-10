"""Explainability sanity checks."""

from __future__ import annotations

import numpy as np


def topk_edge_mask(edge_weight: np.ndarray, k: int) -> np.ndarray:
    w = np.abs(np.asarray(edge_weight))
    flat = w.reshape(-1)
    eligible = np.flatnonzero(np.isfinite(flat) & (flat > 0.0))
    k = min(max(0, int(k)), int(eligible.size))
    mask = np.zeros_like(flat, dtype=bool)
    if k == 0:
        return mask.reshape(w.shape)
    idx = eligible[np.argpartition(flat[eligible], -k)[-k:]]
    mask[idx] = True
    return mask.reshape(w.shape)


def edge_jaccard(edge_a: np.ndarray, edge_b: np.ndarray, k: int) -> float:
    a = topk_edge_mask(edge_a, k)
    b = topk_edge_mask(edge_b, k)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else float("nan")


def occlude_channels(x: np.ndarray, channels: list[int]) -> np.ndarray:
    out = np.asarray(x).copy()
    out[:, channels, :] = 0.0
    return out.astype(np.float32)
