"""Fold-local calibration helpers for frozen V14 residual experts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dpc_snn.utils.metrics import accuracy, cohen_kappa


E15_METHODS = (
    "r0_shared_replay",
    "c1_atc_raw",
    "c2_atc_rms",
    "c3_generic_rms",
)


@dataclass(frozen=True)
class AlphaSelection:
    alpha: float
    accuracy: float
    kappa: float
    curve: tuple[dict[str, float], ...]


def _validated_logits(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"{name} must have shape [samples, 4]")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return array


def center_residual(residual: np.ndarray) -> np.ndarray:
    """Remove the class-invariant logit component from each trial."""

    array = _validated_logits("residual", residual)
    return array - array.mean(axis=1, keepdims=True)


def rms_normalize_residual(
    residual: np.ndarray, *, epsilon: float = 1e-6
) -> np.ndarray:
    """Give each centered trial residual unit class-wise RMS when non-zero."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    centered = center_residual(residual)
    rms = np.sqrt(np.mean(np.square(centered), axis=1, keepdims=True))
    return np.divide(
        centered,
        rms,
        out=np.zeros_like(centered),
        where=rms > float(epsilon),
    )


def calibrated_logits(
    shared_logits: np.ndarray,
    residual_logits: np.ndarray,
    alpha: float,
    *,
    rms_normalize: bool,
) -> np.ndarray:
    """Apply a non-negative residual scale without changing the shared logits."""

    shared = _validated_logits("shared_logits", shared_logits)
    residual = _validated_logits("residual_logits", residual_logits)
    if shared.shape != residual.shape:
        raise ValueError("shared and residual logits must be aligned")
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    transformed = (
        rms_normalize_residual(residual) if rms_normalize else center_residual(residual)
    )
    if alpha == 0.0:
        return shared.copy()
    return shared + float(alpha) * transformed


def select_alpha(
    shared_logits: np.ndarray,
    residual_logits: np.ndarray,
    labels: np.ndarray,
    alphas: tuple[float, ...],
    *,
    rms_normalize: bool,
) -> AlphaSelection:
    """Select alpha by kappa, accuracy, then the smallest prespecified alpha."""

    shared = _validated_logits("shared_logits", shared_logits)
    residual = _validated_logits("residual_logits", residual_logits)
    targets = np.asarray(labels, dtype=np.int64)
    if targets.shape != (shared.shape[0],) or residual.shape != shared.shape:
        raise ValueError("calibration arrays are misaligned")
    if not alphas or len(alphas) != len(set(alphas)):
        raise ValueError("alpha grid must be non-empty and unique")
    if any(not np.isfinite(value) or value < 0.0 for value in alphas):
        raise ValueError("alpha grid values must be finite and non-negative")

    rows: list[dict[str, float]] = []
    candidates: list[tuple[tuple[float, float, int], float]] = []
    for index, alpha in enumerate(alphas):
        logits = calibrated_logits(
            shared,
            residual,
            alpha,
            rms_normalize=rms_normalize,
        )
        prediction = logits.argmax(axis=1)
        accuracy_value = accuracy(targets, prediction)
        kappa = cohen_kappa(targets, prediction, n_classes=4)
        if not np.isfinite(kappa):
            kappa = -1.0
        rows.append(
            {
                "alpha_index": float(index),
                "alpha": float(alpha),
                "accuracy": accuracy_value,
                "kappa": kappa,
            }
        )
        candidates.append(((kappa, accuracy_value, -index), float(alpha)))
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_key, best_alpha = candidates[0]
    return AlphaSelection(
        alpha=best_alpha,
        accuracy=float(best_key[1]),
        kappa=float(best_key[0]),
        curve=tuple(rows),
    )


def rescue_damage(
    labels: np.ndarray, shared_logits: np.ndarray, calibrated: np.ndarray
) -> dict[str, float]:
    targets = np.asarray(labels, dtype=np.int64)
    shared = _validated_logits("shared_logits", shared_logits)
    final = _validated_logits("calibrated_logits", calibrated)
    if targets.shape != (shared.shape[0],) or final.shape != shared.shape:
        raise ValueError("diagnostic arrays are misaligned")
    shared_correct = shared.argmax(axis=1) == targets
    final_correct = final.argmax(axis=1) == targets
    rescue = float(np.mean(final_correct & ~shared_correct))
    damage = float(np.mean(~final_correct & shared_correct))
    return {
        "rescue_rate": rescue,
        "damage_rate": damage,
        "net_rescue_pp": 100.0 * (rescue - damage),
    }
