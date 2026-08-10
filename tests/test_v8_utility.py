from __future__ import annotations

import numpy as np
import pytest

from dpc_snn.analysis.v8_utility import (
    add_gaussian_noise_at_snr,
    drop_eeg_channels,
    fit_logit_calibrator,
    multiclass_calibration_metrics,
    stratified_kshot_indices,
    utility_win_summary,
)


def test_calibration_metrics_are_exact_for_confident_correct_logits() -> None:
    logits = np.asarray([[10.0, -10.0], [-10.0, 10.0]])
    metrics = multiclass_calibration_metrics(logits, np.asarray([0, 1]), n_bins=10)

    assert metrics["accuracy"] == 1.0
    assert metrics["negative_log_likelihood"] < 1e-6
    assert metrics["brier_score"] < 1e-12
    assert metrics["ece"] < 1e-6


def test_stratified_kshot_is_deterministic_disjoint_and_balanced() -> None:
    labels = np.repeat(np.arange(4), 10)
    first = stratified_kshot_indices(labels, k_per_class=3, seed=9)
    second = stratified_kshot_indices(labels, k_per_class=3, seed=9)

    assert np.array_equal(first[0], second[0])
    assert set(first[0]).isdisjoint(first[1])
    assert sorted(np.bincount(labels[first[0]]).tolist()) == [3, 3, 3, 3]
    assert sorted(np.concatenate(first).tolist()) == list(range(40))


def test_logit_calibrator_is_finite_and_improves_biased_example() -> None:
    labels = np.tile(np.arange(4), 10)
    logits = np.eye(4)[labels] * 2.0
    logits[:, 0] += 3.0
    before = multiclass_calibration_metrics(logits, labels)

    calibrator = fit_logit_calibrator(logits, labels, max_iter=50)
    after = multiclass_calibration_metrics(calibrator.apply(logits), labels)

    assert np.isfinite(calibrator.bias).all()
    assert after["negative_log_likelihood"] < before["negative_log_likelihood"]


def test_noise_has_requested_average_snr() -> None:
    x = np.ones((4, 3, 1000), dtype=np.float32)
    perturbed = add_gaussian_noise_at_snr(x, snr_db=10.0, seed=5)
    noise = perturbed - x
    ratio = np.sqrt(np.mean(x**2, axis=-1)) / np.sqrt(np.mean(noise**2, axis=-1))

    np.testing.assert_allclose(20.0 * np.log10(ratio), 10.0, atol=1e-4)


def test_channel_drop_is_shared_across_trials_and_exact() -> None:
    x = np.ones((2, 6, 8), dtype=np.float32)
    dropped_x, dropped = drop_eeg_channels(x, count=2, seed=4)

    assert dropped.shape == (2,)
    assert np.all(dropped_x[:, dropped, :] == 0.0)
    assert np.all(np.delete(dropped_x, dropped, axis=1) == 1.0)


def test_utility_gate_requires_accuracy_preservation_and_one_win() -> None:
    passed = utility_win_summary(
        final_gain_pp=-0.2,
        early_gain_pp=0.6,
        robustness_gain_pp=0.0,
        kshot_gain_pp=0.0,
        operation_reduction=0.0,
    )
    failed = utility_win_summary(
        final_gain_pp=-0.4,
        early_gain_pp=1.0,
        robustness_gain_pp=1.0,
        kshot_gain_pp=1.0,
        operation_reduction=0.8,
    )

    assert passed["passed"]
    assert not failed["passed"]


def test_kshot_rejects_consuming_an_entire_class() -> None:
    with pytest.raises(ValueError, match="leave evaluation"):
        stratified_kshot_indices(np.repeat(np.arange(2), 2), k_per_class=2, seed=0)
