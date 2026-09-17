"""Standardization utilities."""

from __future__ import annotations

import numpy as np


def channelwise_train_stats(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, -1), keepdims=True)
    std = x_train.std(axis=(0, -1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_channelwise_zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    mean, std = channelwise_train_stats(x_train)
    return apply_channelwise_zscore(x_train, mean, std), apply_channelwise_zscore(x_test, mean, std), {
        "mean": mean,
        "std": std,
    }


def euclidean_alignment_matrix(x_train: np.ndarray, regularization: float = 1e-5) -> np.ndarray:
    """Fit an inverse square-root reference covariance using training trials only."""
    x = np.asarray(x_train, dtype=np.float64)
    centered = x - x.mean(axis=-1, keepdims=True)
    covariance = np.einsum("nct,ndt->ncd", centered, centered) / max(1, x.shape[-1] - 1)
    reference = covariance.mean(axis=0)
    scale = np.trace(reference) / max(1, reference.shape[0])
    reference = reference + max(float(regularization * scale), 1e-8) * np.eye(reference.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(reference)
    inverse_sqrt = (eigenvectors * np.maximum(eigenvalues, 1e-8) ** -0.5) @ eigenvectors.T
    return inverse_sqrt.astype(np.float32)
