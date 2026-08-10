"""Dependency-light EEG filtering utilities.

The FFT bandpass is intended for reproducible preprocessing and smoke tests.
For final paper runs, prefer validated MNE/SciPy filtering and record the exact
filter implementation in the run manifest.
"""

from __future__ import annotations

import numpy as np


def _validated_sfreq(sfreq: float) -> float:
    try:
        value = float(sfreq)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Sampling rate must be a positive finite number, got {sfreq!r}.") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"Sampling rate must be a positive finite number, got {sfreq!r}.")
    return value


def fft_bandpass(x: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    """Bandpass-filter the last dimension of ``x`` using an FFT mask."""

    x = np.asarray(x, dtype=np.float32)
    sfreq = _validated_sfreq(sfreq)
    if x.ndim < 1 or x.shape[-1] < 2:
        raise ValueError(f"Expected at least two time samples, got shape={x.shape}.")
    if not (0.0 < float(low) < float(high) <= sfreq / 2.0):
        raise ValueError(
            f"Invalid band [{low}, {high}] Hz for sampling rate {sfreq} Hz; "
            "require 0 < low < high <= Nyquist."
        )
    n = x.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    spectrum = np.fft.rfft(x, axis=-1)
    mask = (freqs >= low) & (freqs <= high)
    spectrum *= mask.astype(spectrum.dtype)
    return np.fft.irfft(spectrum, n=n, axis=-1).astype(np.float32)


def filter_bank(x: np.ndarray, sfreq: float, bands: dict[str, list[float] | tuple[float, float]]) -> dict[str, np.ndarray]:
    return {name: fft_bandpass(x, sfreq, float(lo), float(hi)) for name, (lo, hi) in bands.items()}


def resample_linear(x: np.ndarray, old_sfreq: float, new_sfreq: float) -> np.ndarray:
    old_sfreq = _validated_sfreq(old_sfreq)
    new_sfreq = _validated_sfreq(new_sfreq)
    if np.isclose(old_sfreq, new_sfreq):
        return np.asarray(x)
    x = np.asarray(x)
    old_n = x.shape[-1]
    duration = old_n / old_sfreq
    new_n = int(round(duration * new_sfreq))
    old_t = np.linspace(0.0, duration, old_n, endpoint=False)
    new_t = np.linspace(0.0, duration, new_n, endpoint=False)
    flat = x.reshape(-1, old_n)
    out = np.vstack([np.interp(new_t, old_t, row) for row in flat])
    return out.reshape(*x.shape[:-1], new_n).astype(x.dtype)
