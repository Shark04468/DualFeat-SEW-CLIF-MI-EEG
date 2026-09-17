"""Closed-form diagnostic probes for the E17 information replay audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from dpc_snn.utils.metrics import classification_metrics


@dataclass(frozen=True)
class RidgeProbe:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: np.ndarray
    alpha: float

    def predict_logits(self, features: np.ndarray) -> np.ndarray:
        value = np.asarray(features, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] != self.mean.shape[1]:
            raise ValueError("ridge-probe feature shape mismatch")
        logits = ((value - self.mean) / self.scale) @ self.weight + self.bias
        if not np.isfinite(logits).all():
            raise FloatingPointError("ridge probe produced non-finite logits")
        return logits.astype(np.float32)


def fit_ridge_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float,
    n_classes: int = 4,
    minimum_scale: float = 1e-6,
) -> RidgeProbe:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or y.shape != (x.shape[0],) or x.shape[0] < 2:
        raise ValueError("ridge-probe features and labels are not aligned")
    if alpha <= 0.0 or minimum_scale <= 0.0:
        raise ValueError("ridge alpha and minimum scale must be positive")
    if np.any(y < 0) or np.any(y >= int(n_classes)) or not np.isfinite(x).all():
        raise ValueError("ridge-probe inputs are invalid")

    mean = x.mean(axis=0, keepdims=True)
    scale = np.maximum(x.std(axis=0, keepdims=True), float(minimum_scale))
    standardized = (x - mean) / scale
    target = np.eye(int(n_classes), dtype=np.float64)[y]
    bias = target.mean(axis=0, keepdims=True)
    centered_target = target - bias
    gram = standardized @ standardized.T
    dual = np.linalg.solve(
        gram + float(alpha) * np.eye(gram.shape[0], dtype=np.float64),
        centered_target,
    )
    weight = standardized.T @ dual
    if not all(np.isfinite(value).all() for value in (mean, scale, weight, bias)):
        raise FloatingPointError("ridge-probe fit produced non-finite parameters")
    return RidgeProbe(mean, scale, weight, bias, float(alpha))


def select_ridge_alpha(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    *,
    alphas: Sequence[float],
    n_classes: int = 4,
) -> tuple[float, list[dict[str, float]]]:
    candidates = tuple(float(alpha) for alpha in alphas)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("ridge alpha candidates must be non-empty and unique")
    rows: list[dict[str, float]] = []
    best_key: tuple[float, float, float] | None = None
    selected = candidates[0]
    for alpha in candidates:
        probe = fit_ridge_probe(
            train_features, train_labels, alpha=alpha, n_classes=n_classes
        )
        logits = probe.predict_logits(validation_features)
        metrics = classification_metrics(
            np.asarray(validation_labels, dtype=np.int64),
            logits.argmax(axis=1),
            n_classes=n_classes,
        )
        row = {
            "alpha": alpha,
            "validation_accuracy": float(metrics["accuracy"]),
            "validation_kappa": float(metrics["kappa"]),
        }
        rows.append(row)
        key = (row["validation_kappa"], row["validation_accuracy"], -alpha)
        if best_key is None or key > best_key:
            best_key = key
            selected = alpha
    return selected, rows


def centered_class_logits(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] < 2 or not np.isfinite(value).all():
        raise ValueError("class logits must be a finite rank-two array")
    return value - value.mean(axis=1, keepdims=True)


def residual_fusion_logits(
    base_logits: np.ndarray, correction_logits: np.ndarray, scale: float
) -> np.ndarray:
    base = np.asarray(base_logits, dtype=np.float32)
    correction = np.asarray(correction_logits, dtype=np.float32)
    if base.shape != correction.shape or base.ndim != 2:
        raise ValueError("base and correction logits must be aligned")
    if not 0.0 <= float(scale) <= 1.0:
        raise ValueError("residual scale must lie in [0, 1]")
    output = base + float(scale) * centered_class_logits(correction)
    if not np.isfinite(output).all():
        raise FloatingPointError("residual fusion produced non-finite logits")
    return output.astype(np.float32)


def select_residual_scale(
    base_logits: np.ndarray,
    correction_logits: np.ndarray,
    labels: np.ndarray,
    *,
    scales: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    n_classes: int = 4,
) -> tuple[float, list[dict[str, float]]]:
    candidates = tuple(float(scale) for scale in scales)
    if not candidates or 0.0 not in candidates or len(set(candidates)) != len(candidates):
        raise ValueError("residual scales must be unique and include zero")
    y = np.asarray(labels, dtype=np.int64)
    rows: list[dict[str, float]] = []
    best_key: tuple[float, float, float] | None = None
    selected = 0.0
    for scale in candidates:
        logits = residual_fusion_logits(base_logits, correction_logits, scale)
        metrics = classification_metrics(y, logits.argmax(axis=1), n_classes=n_classes)
        row = {
            "scale": scale,
            "validation_accuracy": float(metrics["accuracy"]),
            "validation_kappa": float(metrics["kappa"]),
        }
        rows.append(row)
        key = (row["validation_kappa"], row["validation_accuracy"], -scale)
        if best_key is None or key > best_key:
            best_key = key
            selected = scale
    return selected, rows


def select_residual_fusion(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    validation_base_logits: np.ndarray,
    *,
    alphas: Sequence[float] = (1.0, 10.0, 100.0),
    scales: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    n_classes: int = 4,
) -> tuple[float, float, list[dict[str, float]]]:
    ridge_alphas = tuple(float(value) for value in alphas)
    residual_scales = tuple(float(value) for value in scales)
    if not ridge_alphas or not residual_scales or 0.0 not in residual_scales:
        raise ValueError("residual search requires ridge alphas and a zero-scale candidate")
    if len(ridge_alphas) * (len(residual_scales) - 1) + 1 > 12:
        raise ValueError("residual search exceeds the 12-configuration budget")
    validation_y = np.asarray(validation_labels, dtype=np.int64)
    rows: list[dict[str, float]] = []
    baseline_metrics = classification_metrics(
        validation_y,
        np.asarray(validation_base_logits).argmax(axis=1),
        n_classes=n_classes,
    )
    rows.append(
        {
            "alpha": ridge_alphas[0],
            "scale": 0.0,
            "validation_accuracy": float(baseline_metrics["accuracy"]),
            "validation_kappa": float(baseline_metrics["kappa"]),
        }
    )
    best_key = (
        float(baseline_metrics["kappa"]),
        float(baseline_metrics["accuracy"]),
        0.0,
        -ridge_alphas[0],
    )
    selected_alpha = ridge_alphas[0]
    selected_scale = 0.0
    for alpha in ridge_alphas:
        probe = fit_ridge_probe(
            train_features, train_labels, alpha=alpha, n_classes=n_classes
        )
        correction = probe.predict_logits(validation_features)
        for scale in residual_scales:
            if scale == 0.0:
                continue
            logits = residual_fusion_logits(validation_base_logits, correction, scale)
            metrics = classification_metrics(
                validation_y, logits.argmax(axis=1), n_classes=n_classes
            )
            row = {
                "alpha": alpha,
                "scale": scale,
                "validation_accuracy": float(metrics["accuracy"]),
                "validation_kappa": float(metrics["kappa"]),
            }
            rows.append(row)
            key = (
                row["validation_kappa"],
                row["validation_accuracy"],
                -scale,
                -alpha,
            )
            if key > best_key:
                best_key = key
                selected_alpha = alpha
                selected_scale = scale
    return selected_alpha, selected_scale, rows
