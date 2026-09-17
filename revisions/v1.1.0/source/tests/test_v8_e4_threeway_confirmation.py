from __future__ import annotations

import pytest

from scripts.evaluate_v8_e4_threeway_confirmation import evaluate_confirmation


def _lock() -> dict:
    return {
        "schema": "dpc-snn-v8-e4-three-way-confirmation-lock/v1",
        "snapshot_digest": "abc123",
        "selected_snn_variant": "sew_clif",
        "matched_ann_control": "ann_sew",
        "fusion_rule": (
            "unweighted arithmetic mean of ATCNet, FBCNet, and decoder probabilities"
        ),
        "confirmation_subjects": [3, 8],
        "confirmation_seeds": [0],
        "thresholds": {
            "macro_snn_vs_matched_ann_minimum_delta_pp": -0.3,
            "macro_snn_vs_atc_fbc_anchor_minimum_delta_pp": -0.5,
            "per_subject_snn_vs_anchor_floor_pp": -2.0,
            "per_subject_mean_firing_rate_interval": [0.005, 0.3],
        },
        "no_temperature_calibration": True,
        "no_weight_search": True,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }


def _row(subject: int, variant: str, accuracy: float, *, firing: float) -> dict:
    return {
        "subject": subject,
        "seed": 0,
        "variant": variant,
        "folds": 6,
        "three_way_equal_accuracy": accuracy,
        "anchor_atc_plus_fbc_accuracy": 0.89,
        "mean_firing_rate": firing,
        "session_e_accessed": False,
    }


def test_confirmation_gate_applies_locked_macro_and_pair_thresholds() -> None:
    rows = [
        _row(3, "ann_sew", 0.880, firing=0.0),
        _row(3, "sew_clif", 0.888, firing=0.01),
        _row(8, "ann_sew", 0.886, firing=0.0),
        _row(8, "sew_clif", 0.889, firing=0.02),
    ]
    pairs, gate = evaluate_confirmation(_lock(), rows)
    assert gate["passed"] is True
    assert gate["macro_snn_delta_vs_matched_ann_pp"] == pytest.approx(0.55)
    assert gate["macro_snn_delta_vs_atc_fbc_anchor_pp"] == pytest.approx(-0.15)
    assert len(pairs) == 2

    rows[3]["mean_firing_rate"] = 0.0
    _, failed = evaluate_confirmation(_lock(), rows)
    assert failed["passed"] is False
    assert failed["criteria"]["all_pairs_have_nondegenerate_firing"] is False


def test_confirmation_rejects_protocol_drift_and_incomplete_pairs() -> None:
    lock = _lock()
    lock["no_weight_search"] = False
    with pytest.raises(RuntimeError, match="no_weight_search"):
        evaluate_confirmation(lock, [])

    with pytest.raises(RuntimeError, match="missing locked confirmation pair"):
        evaluate_confirmation(
            _lock(),
            [
                _row(3, "ann_sew", 0.88, firing=0.0),
                _row(3, "sew_clif", 0.88, firing=0.01),
            ],
        )
