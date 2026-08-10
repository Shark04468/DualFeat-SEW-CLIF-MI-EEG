from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v9_reliability_fusion import (
    GATE_VARIANTS,
    _loss_and_gradient,
    apply_gate,
    equal_probability_fusion,
    fit_gate,
    probability_metrics,
    reliability_features,
    run_fusion_fold,
    select_temperature,
    softmax_probabilities,
)


def test_temperature_selection_uses_inner_nll() -> None:
    logits = np.asarray([[6.0, 0.0], [5.0, 0.0], [4.0, 0.0], [3.0, 0.0]])
    labels = np.asarray([0, 0, 1, 1])
    selected, rows = select_temperature(logits, labels, grid=(0.5, 1.0, 2.0, 4.0))
    losses = {row["temperature"]: row["negative_log_likelihood"] for row in rows}
    assert selected == min(losses, key=losses.get)
    assert selected > 1.0


def test_analytic_gate_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(4)
    atc = rng.dirichlet(np.ones(4), size=19)
    fbc = rng.dirichlet(np.ones(4), size=19)
    labels = rng.integers(0, 4, size=19)
    features = reliability_features(atc, fbc)
    parameters = rng.normal(0.0, 0.2, size=9)
    value, gradient = _loss_and_gradient(
        parameters,
        variant="class_dynamic",
        features=features,
        atc=atc,
        fbc=fbc,
        labels=labels,
        l2=0.1,
    )
    numerical = np.empty_like(parameters)
    epsilon = 1e-6
    for index in range(parameters.size):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_value, _ = _loss_and_gradient(
            plus,
            variant="class_dynamic",
            features=features,
            atc=atc,
            fbc=fbc,
            labels=labels,
            l2=0.1,
        )
        minus_value, _ = _loss_and_gradient(
            minus,
            variant="class_dynamic",
            features=features,
            atc=atc,
            fbc=fbc,
            labels=labels,
            l2=0.1,
        )
        numerical[index] = (plus_value - minus_value) / (2.0 * epsilon)
    assert np.isfinite(value)
    assert np.allclose(gradient, numerical, atol=2e-6)


def test_global_gate_learns_to_prefer_better_teacher() -> None:
    labels = np.tile(np.arange(4), 20)
    atc = np.full((labels.size, 4), 0.05)
    fbc = np.full((labels.size, 4), 0.20)
    atc[np.arange(labels.size), labels] = 0.85
    fbc[np.arange(labels.size), labels] = 0.40
    model = fit_gate("global_static", atc, fbc, labels, l2=0.01)
    fused, weights = apply_gate(model, atc, fbc)
    assert model.converged
    assert float(weights.mean()) > 0.75
    equal = equal_probability_fusion(atc, fbc)
    assert probability_metrics(fused, labels)["negative_log_likelihood"] < probability_metrics(
        equal, labels
    )["negative_log_likelihood"]


def test_class_gate_recovers_complementary_class_reliability() -> None:
    labels = np.tile(np.arange(4), 30)
    atc = np.full((labels.size, 4), 0.10)
    fbc = np.full((labels.size, 4), 0.10)
    for index, label in enumerate(labels):
        if label < 2:
            atc[index] = 0.05
            atc[index, label] = 0.85
            fbc[index] = 0.20
            fbc[index, label] = 0.40
        else:
            atc[index] = 0.20
            atc[index, label] = 0.40
            fbc[index] = 0.05
            fbc[index, label] = 0.85
    model = fit_gate("class_static", atc, fbc, labels, l2=0.01)
    _, weights = apply_gate(model, atc, fbc)
    mean = weights.mean(axis=0)
    assert np.all(mean[:2] > 0.5)
    assert np.all(mean[2:] < 0.5)


def test_fold_runner_returns_all_prespecified_arms_without_outer_labels() -> None:
    rng = np.random.default_rng(12)
    selection_labels = np.tile(np.arange(4), 6)
    selection_atc = rng.normal(size=(24, 4))
    selection_fbc = rng.normal(size=(24, 4))
    outer_atc = rng.normal(size=(17, 4))
    outer_fbc = rng.normal(size=(17, 4))
    result = run_fusion_fold(
        selection_atc,
        selection_fbc,
        selection_labels,
        outer_atc,
        outer_fbc,
        temperature_grid=(0.7, 1.0, 1.4),
        max_iterations=100,
    )
    expected = {
        "atcnet_raw",
        "fbcnet_raw",
        "equal_raw",
        "atcnet_calibrated",
        "fbcnet_calibrated",
        "equal_calibrated",
        *GATE_VARIANTS,
    }
    assert set(result["probabilities"]) == expected
    assert set(result["weights"]) == set(GATE_VARIANTS)
    for probability in result["probabilities"].values():
        assert probability.shape == (17, 4)
        assert np.allclose(probability.sum(axis=1), 1.0)
    assert np.allclose(
        result["probabilities"]["atcnet_raw"], softmax_probabilities(outer_atc)
    )
