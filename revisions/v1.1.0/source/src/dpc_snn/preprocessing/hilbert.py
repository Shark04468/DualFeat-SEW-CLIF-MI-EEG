"""Hilbert amplitude and phase features using NumPy FFT."""

from __future__ import annotations

import numpy as np


def analytic_signal(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    n = x.shape[-1]
    spectrum = np.fft.fft(x, axis=-1)
    h = np.zeros(n, dtype=spectrum.dtype)
    if n % 2 == 0:
        h[0] = 1
        h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        h[0] = 1
        h[1 : (n + 1) // 2] = 2
    return np.fft.ifft(spectrum * h, axis=-1)


def amplitude_phase(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = analytic_signal(x)
    return np.abs(z).astype(np.float32), np.angle(z).astype(np.float32)


def band_amplitude_phase(
    x: np.ndarray,
    sfreq: float,
    bands: dict[str, list[float] | tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    from .filters import filter_bank

    banded = filter_bank(x, sfreq, bands)
    amps = []
    phases = []
    names = []
    for name, arr in banded.items():
        amp, phase = amplitude_phase(arr)
        amps.append(amp)
        phases.append(phase)
        names.append(name)
    return np.stack(amps, axis=1), np.stack(phases, axis=1), names

