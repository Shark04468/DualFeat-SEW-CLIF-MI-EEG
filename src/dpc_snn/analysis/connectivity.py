"""Connectivity metrics for interpretation checks."""

from __future__ import annotations

import numpy as np

from dpc_snn.preprocessing.hilbert import analytic_signal


def phase_locking_value(x: np.ndarray) -> np.ndarray:
    phase = np.angle(analytic_signal(x))
    c = phase.shape[-2]
    out = np.zeros((c, c), dtype=np.float32)
    for i in range(c):
        for j in range(c):
            out[i, j] = np.abs(np.exp(1j * (phase[..., i, :] - phase[..., j, :])).mean())
    return out


def debiased_weighted_phase_lag_index(x: np.ndarray) -> np.ndarray:
    """Estimate the debiased squared weighted phase-lag index (dwPLI).

    The prior implementation was the ordinary unsigned PLI but was labelled
    "debiased phase lag index".  This estimator uses the standard finite-sample
    debiasing numerator and is explicitly reported as dwPLI by runners.
    """

    phase = np.angle(analytic_signal(x))
    c = phase.shape[-2]
    out = np.zeros((c, c), dtype=np.float32)
    for i in range(c):
        for j in range(c):
            im = np.sin(phase[..., i, :] - phase[..., j, :]).reshape(-1).astype(float)
            im_sum = float(im.sum())
            im_sq_sum = float(np.square(im).sum())
            abs_sum = float(np.abs(im).sum())
            denominator = abs_sum**2 - im_sq_sum
            if denominator > 1e-12:
                out[i, j] = float(np.clip((im_sum**2 - im_sq_sum) / denominator, 0.0, 1.0))
    return out


def debiased_phase_lag_index(x: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for dwPLI; prefer the explicit function name."""

    return debiased_weighted_phase_lag_index(x)


def imaginary_coherence(x: np.ndarray) -> np.ndarray:
    z = analytic_signal(x)
    c = z.shape[-2]
    out = np.zeros((c, c), dtype=np.float32)
    for i in range(c):
        for j in range(c):
            sxy = (z[..., i, :] * np.conj(z[..., j, :])).mean()
            sxx = np.abs((z[..., i, :] * np.conj(z[..., i, :])).mean())
            syy = np.abs((z[..., j, :] * np.conj(z[..., j, :])).mean())
            out[i, j] = float(np.imag(sxy) / np.sqrt(max(sxx * syy, 1e-12)))
    return out


def matrix_correlation(
    a: np.ndarray,
    b: np.ndarray,
    mask_diagonal: bool = True,
    mask: np.ndarray | None = None,
) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != a.shape or a.shape != b.shape:
            raise ValueError("matrix_correlation mask and inputs must share the same shape")
        a = a[mask]
        b = b[mask]
    elif mask_diagonal and a.ndim == 2:
        diagonal_mask = ~np.eye(a.shape[0], dtype=bool)
        a = a[diagonal_mask]
        b = b[diagonal_mask]
    else:
        a = a.reshape(-1)
        b = b.reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return float("nan")
    return float(np.corrcoef(a[valid], b[valid])[0, 1])
