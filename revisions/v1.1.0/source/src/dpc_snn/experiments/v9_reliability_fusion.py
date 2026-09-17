"""Leakage-safe reliability fusion primitives for the V9 MI-EEG campaign."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from dpc_snn.utils.metrics import classification_metrics


EPSILON = 1e-9
DEFAULT_TEMPERATURE_GRID = (0.25, 0.35, 0.50, 0.70, 1.0, 1.4, 2.0, 2.8, 4.0)
FEATURE_NAMES = (
    "entropy_advantage_atc",
    "margin_advantage_atc",
    "confidence_advantage_atc",
    "jensen_shannon_divergence",
    "prediction_disagreement",
)
GATE_VARIANTS = (
    "global_static",
    "class_static",
    "dynamic_global",
    "class_dynamic",
)
OUTPUT_ARMS = (
    "atcnet_raw",
    "fbcnet_raw",
    "equal_raw",
    "atcnet_calibrated",
    "fbcnet_calibrated",
    "equal_calibrated",
    *GATE_VARIANTS,
)


def softmax_probabilities(logits: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or not np.isfinite(values).all()
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise ValueError("logits must be finite and temperature must be positive")
    scaled = values / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    exponential = np.exp(scaled)
    return np.ascontiguousarray(exponential / exponential.sum(axis=1, keepdims=True))


def validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    value = np.asarray(probabilities, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] < 2:
        raise ValueError("probabilities must have shape [trials, classes]")
    if not np.isfinite(value).all() or np.any(value < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(value.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probability rows must sum to one")
    return value


def probability_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
) -> dict[str, float]:
    probability = validate_probabilities(probabilities)
    target = np.asarray(labels, dtype=np.int64)
    if target.shape != (probability.shape[0],) or np.any(target < 0) or np.any(
        target >= probability.shape[1]
    ):
        raise ValueError("probabilities and labels are not aligned")
    prediction = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    correct = prediction == target
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    maximum_gap = 0.0
    for index in range(int(n_bins)):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidence >= lower) & (
            confidence <= upper if index == int(n_bins) - 1 else confidence < upper
        )
        if not np.any(selected):
            continue
        gap = abs(float(correct[selected].mean() - confidence[selected].mean()))
        ece += float(selected.mean()) * gap
        maximum_gap = max(maximum_gap, gap)
    selected_probability = np.clip(
        probability[np.arange(target.size), target], EPSILON, 1.0
    )
    one_hot = np.eye(probability.shape[1], dtype=np.float64)[target]
    return {
        **classification_metrics(target, prediction, n_classes=probability.shape[1]),
        "negative_log_likelihood": float(-np.log(selected_probability).mean()),
        "brier_score": float(np.square(probability - one_hot).sum(axis=1).mean()),
        "ece": float(ece),
        "maximum_calibration_error": float(maximum_gap),
    }


def select_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    grid: Sequence[float] = DEFAULT_TEMPERATURE_GRID,
) -> tuple[float, list[dict[str, float]]]:
    values = np.asarray(logits, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if target.shape != (values.shape[0],):
        raise ValueError("temperature selection logits and labels are not aligned")
    rows: list[dict[str, float]] = []
    for temperature in grid:
        probability = softmax_probabilities(values, temperature=float(temperature))
        selected = np.clip(probability[np.arange(target.size), target], EPSILON, 1.0)
        rows.append(
            {
                "temperature": float(temperature),
                "negative_log_likelihood": float(-np.log(selected).mean()),
            }
        )
    if not rows:
        raise ValueError("temperature grid must not be empty")
    best = min(
        rows,
        key=lambda row: (
            row["negative_log_likelihood"],
            abs(math.log(row["temperature"])),
            row["temperature"],
        ),
    )
    return float(best["temperature"]), rows


def equal_probability_fusion(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_value = validate_probabilities(first)
    second_value = validate_probabilities(second)
    if first_value.shape != second_value.shape:
        raise ValueError("fusion inputs must have identical shapes")
    return np.ascontiguousarray(0.5 * (first_value + second_value))


def reliability_features(atc: np.ndarray, fbc: np.ndarray) -> np.ndarray:
    first = validate_probabilities(atc)
    second = validate_probabilities(fbc)
    if first.shape != second.shape:
        raise ValueError("reliability inputs must have identical shapes")
    n_classes = first.shape[1]
    normalization = math.log(n_classes)
    first_entropy = -(first * np.log(np.clip(first, EPSILON, 1.0))).sum(axis=1)
    second_entropy = -(second * np.log(np.clip(second, EPSILON, 1.0))).sum(axis=1)
    first_sorted = np.sort(first, axis=1)
    second_sorted = np.sort(second, axis=1)
    first_margin = first_sorted[:, -1] - first_sorted[:, -2]
    second_margin = second_sorted[:, -1] - second_sorted[:, -2]
    midpoint = 0.5 * (first + second)
    first_kl = (first * (np.log(np.clip(first, EPSILON, 1.0)) - np.log(midpoint))).sum(1)
    second_kl = (second * (np.log(np.clip(second, EPSILON, 1.0)) - np.log(midpoint))).sum(1)
    features = np.column_stack(
        (
            (second_entropy - first_entropy) / normalization,
            first_margin - second_margin,
            first.max(axis=1) - second.max(axis=1),
            0.5 * (first_kl + second_kl) / normalization,
            (first.argmax(axis=1) != second.argmax(axis=1)).astype(np.float64),
        )
    )
    if features.shape[1] != len(FEATURE_NAMES) or not np.isfinite(features).all():
        raise FloatingPointError("reliability features are invalid")
    return np.ascontiguousarray(features)


@dataclass(frozen=True)
class FeatureScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "FeatureScaler":
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or not np.isfinite(matrix).all():
            raise ValueError("feature scaler requires a finite matrix")
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.mean.size:
            raise ValueError("feature matrix is incompatible with fitted scaler")
        transformed = (matrix - self.mean[None, :]) / self.scale[None, :]
        if not np.isfinite(transformed).all():
            raise FloatingPointError("standardized reliability features are invalid")
        return transformed

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FeatureScaler":
        return cls(
            mean=np.asarray(value["mean"], dtype=np.float64),
            scale=np.asarray(value["scale"], dtype=np.float64),
        )


def gate_parameter_count(variant: str, n_classes: int, n_features: int) -> int:
    if variant == "global_static":
        return 1
    if variant == "class_static":
        return int(n_classes)
    if variant == "dynamic_global":
        return 1 + int(n_features)
    if variant == "class_dynamic":
        return int(n_classes) + int(n_features)
    raise ValueError(f"unknown gate variant: {variant}")


def _gate_eta(
    variant: str,
    parameters: np.ndarray,
    features: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    n_trials = features.shape[0]
    if variant == "global_static":
        return np.full((n_trials, n_classes), parameters[0], dtype=np.float64)
    if variant == "class_static":
        return np.broadcast_to(parameters[None, :], (n_trials, n_classes)).copy()
    if variant == "dynamic_global":
        value = parameters[0] + features @ parameters[1:]
        return np.broadcast_to(value[:, None], (n_trials, n_classes)).copy()
    if variant == "class_dynamic":
        return parameters[:n_classes][None, :] + (
            features @ parameters[n_classes:]
        )[:, None]
    raise ValueError(f"unknown gate variant: {variant}")


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _normalised_class_mixture(
    atc: np.ndarray,
    fbc: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    first = validate_probabilities(atc)
    second = validate_probabilities(fbc)
    gate = np.asarray(weight, dtype=np.float64)
    if first.shape != second.shape or gate.shape != first.shape:
        raise ValueError("class mixture inputs are not aligned")
    score = gate * first + (1.0 - gate) * second
    score = np.clip(score, EPSILON, None)
    return np.ascontiguousarray(score / score.sum(axis=1, keepdims=True))


def _loss_and_gradient(
    parameters: np.ndarray,
    *,
    variant: str,
    features: np.ndarray,
    atc: np.ndarray,
    fbc: np.ndarray,
    labels: np.ndarray,
    l2: float,
) -> tuple[float, np.ndarray]:
    n_trials, n_classes = atc.shape
    eta = _gate_eta(variant, parameters, features, n_classes)
    weight = _sigmoid(eta)
    difference = atc - fbc
    score = np.clip(fbc + weight * difference, EPSILON, None)
    normalizer = score.sum(axis=1)
    selected = score[np.arange(n_trials), labels]
    loss = float(np.mean(-np.log(selected) + np.log(normalizer)))
    penalty = 0.5 * float(l2) * float(np.mean(parameters**2))

    score_gradient = difference * weight * (1.0 - weight)
    eta_gradient = score_gradient / normalizer[:, None]
    eta_gradient[np.arange(n_trials), labels] -= (
        score_gradient[np.arange(n_trials), labels] / selected
    )
    eta_gradient /= float(n_trials)

    if variant == "global_static":
        gradient = np.asarray([eta_gradient.sum()])
    elif variant == "class_static":
        gradient = eta_gradient.sum(axis=0)
    elif variant == "dynamic_global":
        per_trial = eta_gradient.sum(axis=1)
        gradient = np.concatenate(([per_trial.sum()], features.T @ per_trial))
    elif variant == "class_dynamic":
        per_trial = eta_gradient.sum(axis=1)
        gradient = np.concatenate((eta_gradient.sum(axis=0), features.T @ per_trial))
    else:
        raise ValueError(f"unknown gate variant: {variant}")
    gradient += float(l2) * parameters / float(parameters.size)
    return loss + penalty, np.asarray(gradient, dtype=np.float64)


@dataclass(frozen=True)
class GateModel:
    variant: str
    parameters: np.ndarray
    scaler: FeatureScaler
    l2: float
    parameter_bound: float
    objective: float
    initial_objective: float
    converged: bool
    iterations: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "parameters": self.parameters.tolist(),
            "feature_names": list(FEATURE_NAMES),
            "feature_scaler": self.scaler.to_dict(),
            "l2": float(self.l2),
            "parameter_bound": float(self.parameter_bound),
            "objective": float(self.objective),
            "initial_objective": float(self.initial_objective),
            "converged": bool(self.converged),
            "iterations": int(self.iterations),
            "message": self.message,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GateModel":
        return cls(
            variant=str(value["variant"]),
            parameters=np.asarray(value["parameters"], dtype=np.float64),
            scaler=FeatureScaler.from_mapping(value["feature_scaler"]),
            l2=float(value["l2"]),
            parameter_bound=float(value["parameter_bound"]),
            objective=float(value["objective"]),
            initial_objective=float(value["initial_objective"]),
            converged=bool(value["converged"]),
            iterations=int(value["iterations"]),
            message=str(value["message"]),
        )


def fit_gate(
    variant: str,
    atc: np.ndarray,
    fbc: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float = 0.1,
    parameter_bound: float = 3.0,
    max_iterations: int = 250,
) -> GateModel:
    first = validate_probabilities(atc)
    second = validate_probabilities(fbc)
    target = np.asarray(labels, dtype=np.int64)
    if first.shape != second.shape or target.shape != (first.shape[0],):
        raise ValueError("gate fitting inputs are not aligned")
    if variant not in GATE_VARIANTS or float(l2) < 0.0 or float(parameter_bound) <= 0.0:
        raise ValueError("invalid gate configuration")
    scaler = FeatureScaler.fit(reliability_features(first, second))
    features = scaler.transform(reliability_features(first, second))
    count = gate_parameter_count(variant, first.shape[1], features.shape[1])
    initial = np.zeros(count, dtype=np.float64)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        return _loss_and_gradient(
            parameters,
            variant=variant,
            features=features,
            atc=first,
            fbc=second,
            labels=target,
            l2=float(l2),
        )

    initial_objective, _ = objective(initial)
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[(-float(parameter_bound), float(parameter_bound))] * count,
        options={"maxiter": int(max_iterations), "ftol": 1e-12, "gtol": 1e-8},
    )
    if not np.isfinite(result.x).all() or not math.isfinite(float(result.fun)):
        raise FloatingPointError(f"{variant} gate optimisation produced non-finite state")
    if float(result.fun) > initial_objective + 1e-8:
        raise RuntimeError(f"{variant} gate optimisation increased its objective")
    return GateModel(
        variant=variant,
        parameters=np.asarray(result.x, dtype=np.float64),
        scaler=scaler,
        l2=float(l2),
        parameter_bound=float(parameter_bound),
        objective=float(result.fun),
        initial_objective=float(initial_objective),
        converged=bool(result.success),
        iterations=int(result.nit),
        message=str(result.message),
    )


def apply_gate(
    model: GateModel,
    atc: np.ndarray,
    fbc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    first = validate_probabilities(atc)
    second = validate_probabilities(fbc)
    if first.shape != second.shape:
        raise ValueError("gate application inputs are not aligned")
    features = model.scaler.transform(reliability_features(first, second))
    expected = gate_parameter_count(model.variant, first.shape[1], features.shape[1])
    if model.parameters.shape != (expected,):
        raise ValueError("gate parameter vector has the wrong size")
    eta = _gate_eta(model.variant, model.parameters, features, first.shape[1])
    weight = _sigmoid(eta)
    return _normalised_class_mixture(first, second, weight), weight


def run_fusion_fold(
    selection_atc_logits: np.ndarray,
    selection_fbc_logits: np.ndarray,
    selection_labels: np.ndarray,
    outer_atc_logits: np.ndarray,
    outer_fbc_logits: np.ndarray,
    *,
    temperature_grid: Sequence[float] = DEFAULT_TEMPERATURE_GRID,
    gate_variants: Sequence[str] = GATE_VARIANTS,
    l2: float = 0.1,
    parameter_bound: float = 3.0,
    max_iterations: int = 250,
) -> dict[str, Any]:
    selection_target = np.asarray(selection_labels, dtype=np.int64)
    atc_temperature, atc_temperature_rows = select_temperature(
        selection_atc_logits, selection_target, grid=temperature_grid
    )
    fbc_temperature, fbc_temperature_rows = select_temperature(
        selection_fbc_logits, selection_target, grid=temperature_grid
    )
    selection_atc = softmax_probabilities(
        selection_atc_logits, temperature=atc_temperature
    )
    selection_fbc = softmax_probabilities(
        selection_fbc_logits, temperature=fbc_temperature
    )
    outer_atc_raw = softmax_probabilities(outer_atc_logits)
    outer_fbc_raw = softmax_probabilities(outer_fbc_logits)
    outer_atc = softmax_probabilities(outer_atc_logits, temperature=atc_temperature)
    outer_fbc = softmax_probabilities(outer_fbc_logits, temperature=fbc_temperature)
    probabilities: dict[str, np.ndarray] = {
        "atcnet_raw": outer_atc_raw,
        "fbcnet_raw": outer_fbc_raw,
        "equal_raw": equal_probability_fusion(outer_atc_raw, outer_fbc_raw),
        "atcnet_calibrated": outer_atc,
        "fbcnet_calibrated": outer_fbc,
        "equal_calibrated": equal_probability_fusion(outer_atc, outer_fbc),
    }
    weights: dict[str, np.ndarray] = {}
    fits: dict[str, Any] = {}
    for variant in gate_variants:
        model = fit_gate(
            variant,
            selection_atc,
            selection_fbc,
            selection_target,
            l2=float(l2),
            parameter_bound=float(parameter_bound),
            max_iterations=int(max_iterations),
        )
        probability, weight = apply_gate(model, outer_atc, outer_fbc)
        probabilities[variant] = probability
        weights[variant] = weight
        fits[variant] = model.to_dict()
    return {
        "probabilities": probabilities,
        "weights": weights,
        "fits": fits,
        "calibration": {
            "atcnet": {
                "temperature": atc_temperature,
                "grid": atc_temperature_rows,
            },
            "fbcnet": {
                "temperature": fbc_temperature,
                "grid": fbc_temperature_rows,
            },
        },
    }
