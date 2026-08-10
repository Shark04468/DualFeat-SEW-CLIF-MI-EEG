"""Post-hoc baseline and fusion analyses for the V8 publication campaign."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from typing import Any

import numpy as np
import torch
from torch import nn

from dpc_snn.utils.metrics import classification_metrics


PUBLICATION_BASELINES = (
    "eegnet",
    "fbcnet",
    "atcnet",
    "tcformer",
    "eeg_conformer",
    "bfatcnet",
)

FREQUENCY_OCCLUSIONS_HZ: dict[str, tuple[float, float]] = {
    "theta_4_8": (4.0, 8.0),
    "mu_8_13": (8.0, 13.0),
    "low_beta_13_20": (13.0, 20.0),
    "high_beta_20_30": (20.0, 30.0),
    "low_gamma_30_40": (30.0, 40.0),
}

REGION_CHANNELS: dict[str, tuple[str, ...]] = {
    "left_motor": ("FC3", "FC1", "C5", "C3", "C1", "CP3", "CP1"),
    "midline_motor": ("FCz", "Cz", "CPz"),
    "right_motor": ("FC2", "FC4", "C2", "C4", "C6", "CP2", "CP4"),
    "frontal": ("Fz",),
    "posterior": ("P1", "Pz", "P2", "POz"),
}


def array_sha256(array: np.ndarray) -> str:
    """Hash dtype, shape, and C-order bytes without materialising JSON."""

    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(repr(value.shape).encode("ascii"))
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def softmax_probabilities(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] < 2 or not np.isfinite(value).all():
        raise ValueError("logits must be a finite [trials, classes] array")
    shifted = value - value.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    probability = exponential / exponential.sum(axis=1, keepdims=True)
    return np.ascontiguousarray(probability, dtype=np.float32)


def validate_region_partition(channel_names: Sequence[str]) -> dict[str, tuple[int, ...]]:
    """Resolve the fixed five-region partition and require exact channel coverage."""

    names = [str(name) for name in channel_names]
    if len(names) != len(set(names)):
        raise ValueError("channel names must be unique")
    index = {name: position for position, name in enumerate(names)}
    expected = [channel for channels in REGION_CHANNELS.values() for channel in channels]
    missing = sorted(set(expected) - set(names))
    extra = sorted(set(names) - set(expected))
    duplicates = sorted({name for name in expected if expected.count(name) > 1})
    if missing or extra or duplicates or len(expected) != len(names):
        raise ValueError(
            "regional partition must cover the channel basis exactly once; "
            f"missing={missing}, extra={extra}, duplicate={duplicates}"
        )
    return {
        region: tuple(index[channel] for channel in channels)
        for region, channels in REGION_CHANNELS.items()
    }


def fft_band_occlusion(
    carrier: np.ndarray,
    *,
    sfreq: float,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    """Remove one half-open frequency band from a held-out carrier."""

    value = np.asarray(carrier, dtype=np.float32)
    if value.ndim != 3:
        raise ValueError("frequency occlusion expects [trials, channels, time]")
    if not 0.0 <= low_hz < high_hz <= float(sfreq) / 2.0:
        raise ValueError("frequency occlusion band lies outside the Nyquist interval")
    spectrum = np.fft.rfft(value.astype(np.float64), axis=-1)
    frequencies = np.fft.rfftfreq(value.shape[-1], d=1.0 / float(sfreq))
    mask = (frequencies >= float(low_hz)) & (frequencies < float(high_hz))
    if not np.any(mask):
        raise ValueError("frequency occlusion band contains no Fourier bins")
    spectrum[..., mask] = 0.0
    restored = np.fft.irfft(spectrum, n=value.shape[-1], axis=-1)
    return np.ascontiguousarray(restored, dtype=np.float32)


def zero_reference_region(
    normalized_carrier: np.ndarray,
    *,
    channel_names: Sequence[str],
    region: str,
) -> np.ndarray:
    """Replace a sensor group by the zero of the training-reference space."""

    partition = validate_region_partition(channel_names)
    if region not in partition:
        raise ValueError(f"unknown region {region!r}")
    value = np.asarray(normalized_carrier, dtype=np.float32)
    if value.ndim != 3 or value.shape[1] != len(channel_names):
        raise ValueError("region occlusion expects [trials, named channels, time]")
    output = value.copy()
    output[:, partition[region], :] = 0.0
    return np.ascontiguousarray(output)


def _model_logits(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    output = model(inputs)
    logits = output["logits"] if isinstance(output, Mapping) else output
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise RuntimeError("baseline model must return [batch, classes] logits")
    return logits


def input_gradient_channel_saliency(
    model: nn.Module,
    model_input: np.ndarray,
    labels: np.ndarray,
    *,
    device: str,
    batch_size: int,
    channel_axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-trial raw and L1-normalised absolute gradient-times-input."""

    value = np.asarray(model_input, dtype=np.float32)
    target = np.asarray(labels, dtype=np.int64)
    if value.shape[0] != target.size or value.ndim < 3:
        raise ValueError("saliency inputs and labels are misaligned")
    normalized_axis = channel_axis if channel_axis >= 0 else value.ndim + channel_axis
    if normalized_axis <= 0 or normalized_axis >= value.ndim:
        raise ValueError("channel axis must identify a non-batch input dimension")
    model.eval()
    parts: list[np.ndarray] = []
    for start in range(0, target.size, int(batch_size)):
        stop = min(start + int(batch_size), target.size)
        batch = torch.from_numpy(value[start:stop]).to(device).requires_grad_(True)
        batch_target = torch.from_numpy(target[start:stop]).to(device)
        model.zero_grad(set_to_none=True)
        logits = _model_logits(model, batch)
        selected = logits.gather(1, batch_target[:, None]).sum()
        gradient = torch.autograd.grad(selected, batch, retain_graph=False)[0]
        contribution = (gradient * batch).abs()
        reduce_axes = tuple(
            axis for axis in range(1, contribution.ndim) if axis != normalized_axis
        )
        channel_value = contribution.mean(dim=reduce_axes)
        if not torch.isfinite(channel_value).all():
            raise FloatingPointError("channel saliency contains NaN or Inf")
        parts.append(channel_value.detach().cpu().numpy())
    raw = np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)
    denominator = np.maximum(raw.sum(axis=1, keepdims=True), np.finfo(np.float32).tiny)
    normalized = np.ascontiguousarray(raw / denominator, dtype=np.float32)
    return raw, normalized


def predictive_entropy(probabilities: np.ndarray) -> np.ndarray:
    value = np.asarray(probabilities, dtype=np.float64)
    if value.ndim != 2 or np.any(value < 0.0):
        raise ValueError("probabilities must have shape [trials, classes]")
    row_sum = value.sum(axis=1)
    if not np.allclose(row_sum, 1.0, atol=1e-5):
        raise ValueError("probability rows must sum to one")
    clipped = np.clip(value, np.finfo(np.float64).tiny, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def class_recalls(labels: np.ndarray, prediction: np.ndarray, n_classes: int) -> list[float]:
    truth = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(prediction, dtype=np.int64)
    recalls: list[float] = []
    for class_index in range(int(n_classes)):
        selected = truth == class_index
        recalls.append(float(np.mean(pred[selected] == class_index)) if np.any(selected) else math.nan)
    return recalls


def equal_probability_fusion(
    first_probabilities: np.ndarray,
    second_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(first_probabilities, dtype=np.float32)
    second = np.asarray(second_probabilities, dtype=np.float32)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("fusion components must have the same [trials, classes] shape")
    probability = np.ascontiguousarray(0.5 * (first + second), dtype=np.float32)
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("fused probabilities are not normalised")
    return probability, probability.argmax(axis=1).astype(np.int64)


def component_error_profile(
    labels: np.ndarray,
    first_probabilities: np.ndarray,
    second_probabilities: np.ndarray,
) -> dict[str, Any]:
    """Describe trial-level complementarity and the resulting equal fusion."""

    truth = np.asarray(labels, dtype=np.int64)
    first = np.asarray(first_probabilities, dtype=np.float32)
    second = np.asarray(second_probabilities, dtype=np.float32)
    if first.shape != second.shape or first.shape[0] != truth.size:
        raise ValueError("component predictions and labels are misaligned")
    n_classes = int(first.shape[1])
    first_pred = first.argmax(axis=1)
    second_pred = second.argmax(axis=1)
    fused, fused_pred = equal_probability_fusion(first, second)
    first_correct = first_pred == truth
    second_correct = second_pred == truth
    fused_correct = fused_pred == truth
    oracle_correct = first_correct | second_correct
    return {
        "n_trials": int(truth.size),
        "first": classification_metrics(truth, first_pred, n_classes=n_classes),
        "second": classification_metrics(truth, second_pred, n_classes=n_classes),
        "fusion": classification_metrics(truth, fused_pred, n_classes=n_classes),
        "oracle_accuracy": float(oracle_correct.mean()),
        "disagreement_rate": float(np.mean(first_pred != second_pred)),
        "double_fault_rate": float(np.mean(~first_correct & ~second_correct)),
        "first_only_correct": int(np.count_nonzero(first_correct & ~second_correct)),
        "second_only_correct": int(np.count_nonzero(~first_correct & second_correct)),
        "fusion_only_correct": int(np.count_nonzero(fused_correct & ~oracle_correct)),
        "fusion_lost_component_correct": int(np.count_nonzero(~fused_correct & oracle_correct)),
        "first_entropy": float(predictive_entropy(first).mean()),
        "second_entropy": float(predictive_entropy(second).mean()),
        "fusion_entropy": float(predictive_entropy(fused).mean()),
        "first_class_recall": class_recalls(truth, first_pred, n_classes),
        "second_class_recall": class_recalls(truth, second_pred, n_classes),
        "fusion_class_recall": class_recalls(truth, fused_pred, n_classes),
        "fusion_gain_over_best_component": float(
            np.mean(fused_correct) - max(np.mean(first_correct), np.mean(second_correct))
        ),
    }


def saliency_profile_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_value = np.asarray(first, dtype=np.float64).reshape(-1)
    second_value = np.asarray(second, dtype=np.float64).reshape(-1)
    if first_value.shape != second_value.shape or first_value.size < 2:
        raise ValueError("saliency profiles must have equal non-trivial shape")
    if np.isclose(first_value.std(), 0.0) or np.isclose(second_value.std(), 0.0):
        return math.nan
    return float(np.corrcoef(first_value, second_value)[0, 1])

