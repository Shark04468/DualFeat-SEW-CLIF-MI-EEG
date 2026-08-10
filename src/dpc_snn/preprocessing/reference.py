"""EEG re-referencing utilities for E20."""

from __future__ import annotations

import numpy as np


def common_average_reference(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=1, keepdims=True)).astype(np.float32)


def mastoid_reference(x: np.ndarray, mastoid_channels: list[int] | None = None) -> np.ndarray:
    if mastoid_channels is None:
        raise ValueError("Mastoid reference requires explicit mastoid channel indices")
    ref = x[:, mastoid_channels, :].mean(axis=1, keepdims=True)
    return (x - ref).astype(np.float32)


def local_laplacian_reference(x: np.ndarray, neighbors: dict[int, list[int]] | None = None) -> np.ndarray:
    """Apply a simple channel-index local Laplacian.

    Explicit geometry-derived neighbours are mandatory. Channel index adjacency
    is not a physical scalp neighbourhood.
    """

    n_channels = x.shape[1]
    if not neighbors:
        raise ValueError("Local Laplacian requires an explicit geometry-derived neighbour map")
    out = np.empty_like(x, dtype=np.float32)
    for ch in range(n_channels):
        if ch in neighbors:
            nb = [n for n in neighbors[ch] if 0 <= n < n_channels and n != ch]
        else:
            nb = []
        if nb:
            out[:, ch, :] = x[:, ch, :] - x[:, nb, :].mean(axis=1)
        else:
            out[:, ch, :] = x[:, ch, :]
    return out.astype(np.float32)


def apply_reference(x: np.ndarray, mode: str, config: dict | None = None) -> np.ndarray:
    config = config or {}
    mode = mode.lower()
    if mode in {"none", "original"}:
        return np.asarray(x, dtype=np.float32)
    if mode in {"car", "common_average"}:
        return common_average_reference(x)
    if mode in {"mastoid", "left_mastoid"}:
        return mastoid_reference(x, config.get("mastoid_channels"))
    if mode in {"laplacian", "local_laplacian"}:
        return local_laplacian_reference(x, config.get("neighbors"))
    raise ValueError(f"Unknown reference mode: {mode}")
