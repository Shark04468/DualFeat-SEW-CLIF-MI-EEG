from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_v8_e4_entropy_residual import (
    confirmation_gate,
    entropy_residual_probability,
)


def test_entropy_residual_is_normalized_and_zero_on_certain_anchor() -> None:
    anchor = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]])
    decoder = np.asarray([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    fused, gate = entropy_residual_probability(anchor, decoder, maximum_weight=0.05)
    assert gate.tolist() == pytest.approx([0.0, 0.05])
    assert fused[0].tolist() == pytest.approx(anchor[0].tolist())
    assert fused.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def _lock() -> dict:
    return {
        "selection": {
            "selected_snn_variant": "sew_clif",
            "matched_ann_control": "ann_sew",
        },
        "confirmation": {
            "subjects": [1, 3],
            "seeds": [1],
            "macro_snn_vs_matched_ann_minimum_delta_pp": -0.3,
            "macro_snn_vs_anchor_minimum_delta_pp": 0.0,
            "minimum_positive_subject_seed_pairs": 1,
            "per_pair_snn_vs_anchor_floor_pp": -1.0,
            "firing_rate_interval": [0.005, 0.3],
        },
    }


def _row(subject: int, variant: str, final: float, anchor: float, firing: float) -> dict:
    return {
        "subject": subject,
        "seed": 1,
        "variant": variant,
        "final_accuracy": final,
        "anchor_accuracy": anchor,
        "mean_firing_rate": firing,
    }


def test_confirmation_gate_compares_same_rule_and_requires_positive_pairs() -> None:
    rows = [
        _row(1, "ann_sew", 0.88, 0.88, 0.0),
        _row(1, "sew_clif", 0.89, 0.88, 0.01),
        _row(3, "ann_sew", 0.90, 0.90, 0.0),
        _row(3, "sew_clif", 0.90, 0.90, 0.02),
    ]
    gate = confirmation_gate(_lock(), rows)
    assert gate["passed"] is True
    assert gate["positive_subject_seed_pairs"] == 1
    assert gate["macro_snn_delta_vs_anchor_pp"] == pytest.approx(0.5)

    rows[1]["mean_firing_rate"] = 0.0
    assert confirmation_gate(_lock(), rows)["passed"] is False
