from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_v8_e4_sequence_calibration import (
    TEMPERATURE_GRID,
    negative_log_likelihood,
    select_temperature,
)


def test_temperature_selection_uses_only_fixed_grid_nll() -> None:
    labels = np.asarray([0, 1, 2, 3], dtype=np.int64)
    logits = np.eye(4, dtype=np.float32)[labels] * 8.0
    selected, rows = select_temperature(logits, labels)
    losses = {row["temperature"]: row["negative_log_likelihood"] for row in rows}
    assert selected in TEMPERATURE_GRID
    assert losses[selected] == min(losses.values())
    assert len(rows) == len(TEMPERATURE_GRID)


def test_temperature_nll_rejects_nonpositive_temperature() -> None:
    with pytest.raises(ValueError, match="positive"):
        negative_log_likelihood(np.zeros((2, 4)), np.asarray([0, 1]), 0.0)
