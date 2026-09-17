from __future__ import annotations

import pytest

from dpc_snn.experiments.v8_statistics import (
    V8PairingError,
    matched_snn_gate_decision,
    pair_subject_seed_rows,
    paired_delta_summary,
    static_delay_gate_decision,
)


def test_paired_rows_require_exact_coverage() -> None:
    first = [
        {"subject": 1, "seed": 0, "accuracy": 0.6},
        {"subject": 3, "seed": 0, "accuracy": 0.7},
    ]
    second = [
        {"subject": 1, "seed": 0, "accuracy": 0.62},
        {"subject": 3, "seed": 0, "accuracy": 0.69},
    ]
    paired = pair_subject_seed_rows(first, second)
    assert [round(row["delta_second_minus_first"], 3) for row in paired] == [0.02, -0.01]
    summary = paired_delta_summary(paired, bootstrap_samples=1000)
    assert summary["pairs"] == 2
    assert summary["subjects"] == 2
    assert summary["positive_pairs"] == 1

    with pytest.raises(V8PairingError, match="coverage differs"):
        pair_subject_seed_rows(first, second[:1])


def test_summary_uses_subjects_not_seeds_as_independent_units() -> None:
    paired = [
        {"subject": 1, "seed": 0, "delta_second_minus_first": 0.30},
        {"subject": 1, "seed": 1, "delta_second_minus_first": 0.30},
        {"subject": 3, "seed": 0, "delta_second_minus_first": -0.10},
        {"subject": 3, "seed": 1, "delta_second_minus_first": -0.10},
    ]
    summary = paired_delta_summary(paired, bootstrap_samples=1000)
    assert summary["subjects"] == 2
    assert summary["seeds_per_subject"] == 2
    assert summary["subject_macro_mean_delta"] == pytest.approx(0.10)
    assert summary["positive_subjects"] == 1

    with pytest.raises(V8PairingError, match="unequal seed coverage"):
        paired_delta_summary(paired[:-1], bootstrap_samples=100)


def test_static_delay_gate_uses_registered_pair_median_and_count() -> None:
    paired = [
        {
            "subject": subject,
            "seed": seed,
            "delta_second_minus_first": delta,
        }
        for (subject, seed), delta in zip(
            ((subject, seed) for subject in (1, 3, 8) for seed in (0, 1, 2)),
            (0.006, 0.007, 0.008, 0.005, 0.006, 0.007, -0.001, 0.0, -0.002),
            strict=True,
        )
    ]
    decision = static_delay_gate_decision(paired, bootstrap_samples=100)
    assert decision["passed"]
    assert decision["observed_positive_pairs"] == 6

    paired[0] = {**paired[0], "delta_second_minus_first": 0.0}
    assert not static_delay_gate_decision(paired, bootstrap_samples=100)["passed"]


def test_matched_snn_gate_selects_by_accuracy_then_requires_utility() -> None:
    decision = matched_snn_gate_decision(
        [
            {
                "variant": "plif_plain",
                "final_gain_pp": -0.2,
                "early_gain_pp": -0.1,
                "ann_final_activity_nonzero_rate": 0.95,
                "snn_final_activity_nonzero_rate": 0.10,
                "snn_binary_spike_rate": 0.08,
            },
            {
                "variant": "clif_plain",
                "final_gain_pp": -0.1,
                "early_gain_pp": 0.6,
                "ann_final_activity_nonzero_rate": 0.95,
                "snn_final_activity_nonzero_rate": 0.12,
                "snn_binary_spike_rate": 0.09,
            },
        ]
    )
    assert decision["passed"]
    assert decision["selected_variant"] == "clif_plain"
    assert decision["selected_candidate"]["early_accuracy_win"]


def test_matched_snn_gate_rejects_dead_activity_and_excess_accuracy_loss() -> None:
    candidate = {
        "variant": "sew_clif",
        "final_gain_pp": -0.31,
        "early_gain_pp": 1.0,
        "ann_final_activity_nonzero_rate": 0.95,
        "snn_final_activity_nonzero_rate": 0.001,
        "snn_binary_spike_rate": 0.001,
    }
    decision = matched_snn_gate_decision([candidate])
    assert not decision["passed"]
    assert not decision["accuracy_passed"]
    assert not decision["selected_candidate"]["activity_sparsity_proxy_win"]
