from __future__ import annotations

import numpy as np

from dpc_snn.experiments.v17_information_replay import (
    residual_fusion_logits,
    fit_ridge_probe,
    select_ridge_alpha,
    select_residual_fusion,
    select_residual_scale,
)


def test_ridge_probe_recovers_linearly_separable_classes() -> None:
    rng = np.random.default_rng(4)
    labels = np.repeat(np.arange(4), 12)
    centers = np.eye(4, dtype=np.float64) * 4.0
    features = centers[labels] + 0.05 * rng.normal(size=(labels.size, 4))
    probe = fit_ridge_probe(features, labels, alpha=0.1)
    logits = probe.predict_logits(features)
    assert np.mean(logits.argmax(axis=1) == labels) == 1.0


def test_alpha_selection_is_deterministic_and_reports_all_candidates() -> None:
    rng = np.random.default_rng(8)
    train = rng.normal(size=(24, 7))
    validation = rng.normal(size=(12, 7))
    train_labels = np.arange(24) % 4
    validation_labels = np.arange(12) % 4
    selected, rows = select_ridge_alpha(
        train,
        train_labels,
        validation,
        validation_labels,
        alphas=(0.1, 1.0, 10.0),
    )
    assert selected in {0.1, 1.0, 10.0}
    assert [row["alpha"] for row in rows] == [0.1, 1.0, 10.0]


def test_zero_residual_exactly_preserves_base_logits() -> None:
    base = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    correction = np.asarray([[4.0, 1.0, -2.0, 0.0]], dtype=np.float32)
    np.testing.assert_array_equal(residual_fusion_logits(base, correction, 0.0), base)


def test_residual_search_contains_one_exact_zero_candidate() -> None:
    rng = np.random.default_rng(9)
    train = rng.normal(size=(24, 6))
    validation = rng.normal(size=(12, 6))
    train_y = np.arange(24) % 4
    validation_y = np.arange(12) % 4
    base = rng.normal(size=(12, 4))
    alpha, scale, rows = select_residual_fusion(
        train,
        train_y,
        validation,
        validation_y,
        base,
        alphas=(1.0, 10.0),
        scales=(0.0, 0.5, 1.0),
    )
    assert alpha in {1.0, 10.0}
    assert scale in {0.0, 0.5, 1.0}
    assert sum(row["scale"] == 0.0 for row in rows) == 1
    assert len(rows) == 5


def test_scale_selection_prefers_zero_when_all_predictions_tie() -> None:
    base = np.zeros((8, 4), dtype=np.float32)
    correction = np.zeros_like(base)
    labels = np.arange(8) % 4
    scale, rows = select_residual_scale(base, correction, labels)
    assert scale == 0.0
    assert len(rows) == 4
