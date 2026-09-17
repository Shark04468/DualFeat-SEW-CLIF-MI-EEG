"""Metric utilities with no sklearn dependency."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if n_classes is None:
        n_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=False):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return math.nan
    return float((y_true == y_pred).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None) -> float:
    cm = confusion_matrix(y_true, y_pred, n_classes=n_classes)
    f1s = []
    for k in range(cm.shape[0]):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else math.nan


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None) -> float:
    cm = confusion_matrix(y_true, y_pred, n_classes=n_classes).astype(float)
    total = cm.sum()
    if total == 0:
        return math.nan
    po = np.trace(cm) / total
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    pe = (row * col).sum() / (total * total)
    if np.isclose(1.0 - pe, 0.0):
        return 0.0
    return float((po - pe) / (1.0 - pe))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None) -> float:
    cm = confusion_matrix(y_true, y_pred, n_classes=n_classes)
    recalls = []
    for k in range(cm.shape[0]):
        denom = cm[k, :].sum()
        recalls.append(cm[k, k] / denom if denom > 0 else 0.0)
    return float(np.mean(recalls)) if recalls else math.nan


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int | None = None,
) -> dict[str, float]:
    return {
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, n_classes=n_classes),
        "kappa": cohen_kappa(y_true, y_pred, n_classes=n_classes),
        "macro_f1": macro_f1(y_true, y_pred, n_classes=n_classes),
    }


def paired_prediction_comparison(
    y_true: np.ndarray,
    first_pred: np.ndarray,
    second_pred: np.ndarray,
) -> dict[str, float | int]:
    """Compare paired classifiers with an exact two-sided McNemar test."""

    y_true = np.asarray(y_true)
    first_pred = np.asarray(first_pred)
    second_pred = np.asarray(second_pred)
    if y_true.shape != first_pred.shape or y_true.shape != second_pred.shape:
        raise ValueError("paired predictions and labels must have identical shapes")
    first_correct = first_pred == y_true
    second_correct = second_pred == y_true
    first_only = int(np.count_nonzero(first_correct & ~second_correct))
    second_only = int(np.count_nonzero(~first_correct & second_correct))
    discordant = first_only + second_only
    p_value = exact_mcnemar_p(first_only, second_only)
    return {
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "discordant_predictions": discordant,
        "exact_mcnemar_p": p_value,
    }


def exact_mcnemar_p(first_only: int, second_only: int) -> float:
    """Return the exact two-sided McNemar p value from discordant counts."""

    first_only = int(first_only)
    second_only = int(second_only)
    if first_only < 0 or second_only < 0:
        raise ValueError("McNemar discordant counts must be non-negative")
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(first_only, second_only) + 1)
    ) / (2.0**discordant)
    return float(min(1.0, 2.0 * tail))


def bootstrap_ci(values: np.ndarray, seed: int = 0, n_boot: int = 2000, alpha: float = 0.05) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": math.nan, "low": math.nan, "high": math.nan, "n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        means[i] = sample.mean()
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(means, alpha / 2.0)),
        "high": float(np.quantile(means, 1.0 - alpha / 2.0)),
        "n": int(values.size),
    }
