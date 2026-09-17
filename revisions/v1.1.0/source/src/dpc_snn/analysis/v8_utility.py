"""Prespecified calibration and perturbation utilities for frozen V8 E7."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LogitCalibrator:
    log_temperature: float
    bias: np.ndarray

    def apply(self, logits: np.ndarray) -> np.ndarray:
        values = np.asarray(logits, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.bias.size:
            raise ValueError("calibrator logits have an incompatible shape")
        temperature = math.exp(float(self.log_temperature))
        return np.asarray(values / temperature + self.bias[None, :], dtype=np.float32)


def softmax_probabilities(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("logits must be a non-empty finite matrix")
    shifted = values - values.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def multiclass_calibration_metrics(
    logits: np.ndarray, labels: np.ndarray, *, n_bins: int = 15
) -> dict[str, float]:
    probabilities = softmax_probabilities(logits)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (probabilities.shape[0],) or np.any(labels < 0) or np.any(
        labels >= probabilities.shape[1]
    ):
        raise ValueError("calibration labels are incompatible with logits")
    if int(n_bins) < 2:
        raise ValueError("calibration requires at least two confidence bins")
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = (prediction == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    maximum_gap = 0.0
    for index in range(int(n_bins)):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (
            confidence <= upper if index == int(n_bins) - 1 else confidence < upper
        )
        if not np.any(mask):
            continue
        gap = abs(float(correct[mask].mean() - confidence[mask].mean()))
        ece += float(mask.mean()) * gap
        maximum_gap = max(maximum_gap, gap)
    selected = np.clip(probabilities[np.arange(labels.size), labels], 1e-12, 1.0)
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    return {
        "accuracy": float(correct.mean()),
        "negative_log_likelihood": float(-np.log(selected).mean()),
        "brier_score": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece": float(ece),
        "maximum_calibration_error": float(maximum_gap),
    }


def stratified_kshot_indices(
    labels: np.ndarray, *, k_per_class: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or labels.size == 0 or int(k_per_class) < 1:
        raise ValueError("K-shot split requires labels and a positive K")
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    for class_value in sorted(np.unique(labels).tolist()):
        candidates = np.flatnonzero(labels == class_value)
        if candidates.size <= int(k_per_class):
            raise ValueError("K-shot calibration must leave evaluation trials per class")
        selected.extend(rng.permutation(candidates)[: int(k_per_class)].tolist())
    calibration = np.asarray(sorted(selected), dtype=np.int64)
    mask = np.ones(labels.size, dtype=bool)
    mask[calibration] = False
    evaluation = np.flatnonzero(mask)
    return calibration, evaluation


def fit_logit_calibrator(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    l2_bias: float = 0.05,
    max_iter: int = 100,
) -> LogitCalibrator:
    """Fit positive scalar temperature plus centered class bias.

    This low-capacity calibration head is deliberately smaller than a new
    classifier and is fit only on the prespecified E7 K-shot subset.
    """

    values = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    targets = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    if values.ndim != 2 or targets.shape != (values.shape[0],) or values.shape[0] < 2:
        raise ValueError("calibrator training arrays have invalid shapes")
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(values.shape[1], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature, bias],
        lr=0.5,
        max_iter=int(max_iter),
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        centered_bias = bias - bias.mean()
        calibrated = values / log_temperature.exp().clamp(0.05, 20.0) + centered_bias
        loss = F.cross_entropy(calibrated, targets) + float(l2_bias) * centered_bias.square().mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    if not torch.isfinite(log_temperature) or not torch.isfinite(bias).all():
        raise FloatingPointError("logit calibrator produced non-finite parameters")
    centered = (bias - bias.mean()).detach().cpu().numpy()
    bounded_log_temperature = float(log_temperature.detach().clamp(math.log(0.05), math.log(20.0)))
    return LogitCalibrator(
        log_temperature=bounded_log_temperature,
        bias=np.asarray(centered, dtype=np.float64),
    )


def add_gaussian_noise_at_snr(
    x: np.ndarray, *, snr_db: float, seed: int
) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("noise perturbation expects finite [trial, channel, time] EEG")
    rng = np.random.default_rng(int(seed))
    noise = rng.standard_normal(values.shape).astype(np.float32)
    signal_rms = np.sqrt(np.mean(np.square(values, dtype=np.float64), axis=-1, keepdims=True))
    noise_rms = np.sqrt(np.mean(np.square(noise, dtype=np.float64), axis=-1, keepdims=True))
    target = signal_rms / (10.0 ** (float(snr_db) / 20.0))
    scaled = noise * (target / np.maximum(noise_rms, np.finfo(np.float32).tiny))
    return np.ascontiguousarray(values + scaled, dtype=np.float32)


def drop_eeg_channels(
    x: np.ndarray, *, count: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 3 or not 1 <= int(count) < values.shape[1]:
        raise ValueError("channel-drop count is outside the EEG channel axis")
    rng = np.random.default_rng(int(seed))
    dropped = np.sort(rng.choice(values.shape[1], size=int(count), replace=False))
    output = values.copy()
    output[:, dropped, :] = 0.0
    return np.ascontiguousarray(output), dropped.astype(np.int64)


def utility_win_summary(
    *,
    final_gain_pp: float,
    early_gain_pp: float,
    robustness_gain_pp: float,
    kshot_gain_pp: float,
    operation_reduction: float,
    maximum_final_gap_pp: float = 0.3,
    minimum_accuracy_utility_pp: float = 0.5,
    minimum_operation_reduction: float = 0.5,
) -> dict[str, Any]:
    values = (
        final_gain_pp,
        early_gain_pp,
        robustness_gain_pp,
        kshot_gain_pp,
        operation_reduction,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("E7 utility values must be finite")
    accuracy_preserved = float(final_gain_pp) >= -float(maximum_final_gap_pp)
    wins = {
        "final_accuracy": float(final_gain_pp) > 0.0,
        "early_decision": float(early_gain_pp) >= float(minimum_accuracy_utility_pp),
        "robustness": float(robustness_gain_pp) >= float(minimum_accuracy_utility_pp),
        "kshot_calibration": float(kshot_gain_pp) >= float(minimum_accuracy_utility_pp),
        "operation_proxy": float(operation_reduction) >= float(minimum_operation_reduction),
    }
    return {
        "passed": bool(accuracy_preserved and any(wins.values())),
        "accuracy_preserved": accuracy_preserved,
        "wins": wins,
        "thresholds": {
            "maximum_final_gap_pp": float(maximum_final_gap_pp),
            "minimum_accuracy_utility_pp": float(minimum_accuracy_utility_pp),
            "minimum_operation_reduction": float(minimum_operation_reduction),
        },
    }
