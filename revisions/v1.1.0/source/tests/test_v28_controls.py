from __future__ import annotations

from dpc_snn.experiments.v28_controls import v28_control_gate


def test_v28_gate_requires_every_registered_condition() -> None:
    row = {
        "pairs": 27,
        "mean_delta_pp": 1.1,
        "positive_pairs": 21,
        "wilcoxon_p_greater": 0.01,
    }
    assert v28_control_gate(row, worst_subject_mean_delta_pp=-0.5)["status"] == "pass"
    failed = dict(row, positive_pairs=20)
    assert v28_control_gate(failed, worst_subject_mean_delta_pp=-0.5)["status"] == "fail"
