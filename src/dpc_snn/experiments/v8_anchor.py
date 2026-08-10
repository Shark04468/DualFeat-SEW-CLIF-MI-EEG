"""Leakage-safe loading of fold-local V8 baseline anchor predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def e1_selection_prediction_path(
    root: Path,
    *,
    model: str,
    subject: int,
    seed: int,
    fold: int,
) -> Path:
    """Return the inner-train -> inner-validation prediction artifact for one fold."""

    return (
        Path(root)
        / str(model)
        / f"subject_{int(subject):02d}"
        / f"seed_{int(seed)}"
        / f"fold_{int(fold)}"
        / "selection_predictions.npz"
    )


def probability_from_logits(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise RuntimeError("anchor logits are non-finite or have an invalid shape")
    shifted = values - values.max(axis=1, keepdims=True)
    numerator = np.exp(shifted)
    probability = numerator / numerator.sum(axis=1, keepdims=True)
    return probability.astype(np.float32)


def load_e1_selection_probability(
    path: Path,
    *,
    expected_indices: Sequence[int] | np.ndarray,
    expected_labels: Sequence[int] | np.ndarray,
    n_classes: int = 4,
) -> np.ndarray:
    """Load and reorder one E1 fold-local validation prediction without outer data."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    if set(values) != {"indices", "logits", "labels"}:
        raise RuntimeError(f"invalid E1 selection prediction schema: {path}")
    indices = np.asarray(values["indices"], dtype=np.int64)
    logits = np.asarray(values["logits"], dtype=np.float32)
    labels = np.asarray(values["labels"], dtype=np.int64)
    expected_indices_array = np.asarray(expected_indices, dtype=np.int64)
    expected_labels_array = np.asarray(expected_labels, dtype=np.int64)
    if (
        indices.ndim != 1
        or logits.shape != (indices.size, int(n_classes))
        or labels.shape != indices.shape
        or expected_indices_array.ndim != 1
        or expected_labels_array.shape != expected_indices_array.shape
        or indices.size != expected_indices_array.size
        or np.unique(indices).size != indices.size
    ):
        raise RuntimeError(f"invalid E1 selection prediction dimensions: {path}")
    if set(indices.tolist()) != set(expected_indices_array.tolist()):
        raise RuntimeError(f"E1 selection indices differ from the active inner fold: {path}")
    position = {int(index): offset for offset, index in enumerate(indices.tolist())}
    order = np.asarray([position[int(index)] for index in expected_indices_array], dtype=np.int64)
    if not np.array_equal(labels[order], expected_labels_array):
        raise RuntimeError(f"E1 selection labels differ from the active inner fold: {path}")
    return probability_from_logits(logits[order])


def load_fused_e1_selection_anchor(
    atc_path: Path,
    fbc_path: Path,
    *,
    expected_indices: Sequence[int] | np.ndarray,
    expected_labels: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Return the fixed equal-weight fold-local ATCNet+FBCNet validation anchor."""

    atc = load_e1_selection_probability(
        atc_path,
        expected_indices=expected_indices,
        expected_labels=expected_labels,
    )
    fbc = load_e1_selection_probability(
        fbc_path,
        expected_indices=expected_indices,
        expected_labels=expected_labels,
    )
    fused = 0.5 * atc + 0.5 * fbc
    if not np.allclose(fused.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise RuntimeError("fold-local fused anchor is not row-normalised")
    return fused.astype(np.float32)
