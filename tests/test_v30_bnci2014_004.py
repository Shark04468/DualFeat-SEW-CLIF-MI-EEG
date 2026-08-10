from __future__ import annotations

import numpy as np
import pytest

from dpc_snn.data.bnci2014_004 import (
    bnci2014_004_physical_files,
    bnci2014_004_session_keys,
)
from dpc_snn.experiments.v30_bnci2014_004 import v30_blind_gate


def test_bnci2014_004_session_mapping_is_exact() -> None:
    keys, labels = bnci2014_004_session_keys(["01T", "02T", "03T"])
    assert keys == ["0train", "1train", "2train"]
    assert labels == ["01T", "02T", "03T"]
    keys, labels = bnci2014_004_session_keys(["04E", "05E"])
    assert keys == ["3test", "4test"]
    assert labels == ["04E", "05E"]


def test_bnci2014_004_session_mapping_rejects_invalid_protocol() -> None:
    with pytest.raises(ValueError):
        bnci2014_004_session_keys([])
    with pytest.raises(ValueError):
        bnci2014_004_session_keys(["01T", "01T"])
    with pytest.raises(ValueError):
        bnci2014_004_session_keys(["06E"])
    with pytest.raises(ValueError):
        bnci2014_004_physical_files(["03T", "04E"])


def test_bnci2014_004_physical_file_partition() -> None:
    assert bnci2014_004_physical_files(["01T", "02T", "03T"]) == ["T"]
    assert bnci2014_004_physical_files(["04E", "05E"]) == ["E"]


def test_v30_gate_pass_and_fail() -> None:
    passed = v30_blind_gate(
        np.asarray([1.0, 2.0, 1.5, 0.5, 0.7, 1.1, 0.4, -0.1, -0.2]),
        one_sided_wilcoxon_p=0.01,
    )
    assert passed["status"] == "pass"
    assert passed["authorized_next_stage"] == "E31_publication_baselines"

    failed = v30_blind_gate(
        np.asarray([1.0, 0.5, 0.2, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6]),
        one_sided_wilcoxon_p=0.2,
    )
    assert failed["status"] == "fail"
    assert failed["authorized_next_stage"] is None
