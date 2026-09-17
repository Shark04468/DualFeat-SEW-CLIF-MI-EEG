from __future__ import annotations

import numpy as np

from dpc_snn.experiments.v29_openbmi import v29_replication_gate


def test_v29_replication_gate_is_subject_level_and_strict() -> None:
    values = np.r_[np.full(34, 1.0), np.full(20, -0.2)]
    assert v29_replication_gate(values, one_sided_wilcoxon_p=0.01)["status"] == "pass"
    assert v29_replication_gate(values, one_sided_wilcoxon_p=0.05)["status"] == "fail"
