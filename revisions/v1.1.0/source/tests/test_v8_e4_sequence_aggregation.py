from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_v8_e4_atc_sequence_canaries import (
    MATCHED_ANN_CONTROL,
    assemble_oof,
)


def test_assemble_oof_requires_exact_nonoverlapping_coverage() -> None:
    first = {
        "indices": np.asarray([0, 2]),
        "logits": np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
        "labels": np.asarray([0, 1]),
    }
    second = {
        "indices": np.asarray([1, 3]),
        "logits": np.asarray([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32),
        "labels": np.asarray([2, 3]),
    }
    logits, labels = assemble_oof([first, second], n_trials=4)
    assert logits.shape == (4, 4)
    assert labels.tolist() == [0, 2, 1, 3]

    with pytest.raises(RuntimeError, match="overlap"):
        assemble_oof([first, first], n_trials=4)
    with pytest.raises(RuntimeError, match="complete"):
        assemble_oof([first], n_trials=4)


def test_sew_clif_uses_the_same_residual_topology_ann_control() -> None:
    assert MATCHED_ANN_CONTROL["plif_plain"] == "ann_plain"
    assert MATCHED_ANN_CONTROL["clif_plain"] == "ann_plain"
    assert MATCHED_ANN_CONTROL["sew_clif"] == "ann_sew"
