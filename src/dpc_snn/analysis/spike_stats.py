"""Spike and operation proxy metrics."""

from __future__ import annotations

import numpy as np


def spike_rate(spikes: np.ndarray) -> float:
    arr = np.asarray(spikes, dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def synops_proxy(spikes_pre: np.ndarray, edge_weight: np.ndarray) -> float:
    pre_rate = np.asarray(spikes_pre, dtype=float).mean()
    active_edges = (np.abs(edge_weight) > 1e-6).sum()
    timesteps = spikes_pre.shape[-1]
    return float(pre_rate * active_edges * timesteps)


def decision_latency_curve(logits_by_time: np.ndarray, y_true: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for t in range(logits_by_time.shape[1]):
        pred = logits_by_time[:, t].argmax(axis=-1)
        acc = float((pred == y_true).mean())
        rows.append({"time_index": int(t), "accuracy": acc})
    return rows

