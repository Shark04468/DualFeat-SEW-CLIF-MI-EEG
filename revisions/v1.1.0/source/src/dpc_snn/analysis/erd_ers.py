"""ERD/ERS summary utilities."""

from __future__ import annotations

import numpy as np

from dpc_snn.preprocessing.filters import fft_bandpass


def band_power(x: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    filtered = fft_bandpass(x, sfreq, band[0], band[1])
    return (filtered**2).mean(axis=-1)


def erd_ers(
    x: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    baseline_slice: slice | None = None,
    imagery_slice: slice | None = None,
) -> list[dict[str, float]]:
    if baseline_slice is None or imagery_slice is None:
        raise ValueError("ERD/ERS requires explicit pre-cue baseline and imagery slices.")
    # Filter the whole epoch before slicing so separate FFT calls cannot create
    # different boundary artefacts in the baseline and imagery intervals.
    filtered = fft_bandpass(x, sfreq, band[0], band[1])
    base = (filtered[..., baseline_slice] ** 2).mean(axis=-1)
    img = (filtered[..., imagery_slice] ** 2).mean(axis=-1)
    change = 100.0 * (img - base) / np.maximum(base, 1e-8)
    rows = []
    for cls in np.unique(y):
        cls_change = change[y == cls]
        for ch in range(x.shape[1]):
            rows.append({"class": int(cls), "channel": int(ch), "erd_ers_percent": float(cls_change[:, ch].mean())})
    return rows
