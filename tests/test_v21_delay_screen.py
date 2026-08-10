from __future__ import annotations

import numpy as np

from dpc_snn.experiments.v21_delay_screen import causal_shift, select_lag_scale


def test_causal_shift_never_wraps_future_values() -> None:
    sequence = np.arange(5, dtype=np.float32).reshape(1, 5, 1)
    shifted = causal_shift(sequence, 2)
    np.testing.assert_array_equal(
        shifted.reshape(-1), np.asarray([0, 0, 0, 1, 2], dtype=np.float32)
    )


def test_lag_scale_ties_prefer_exact_zero_control() -> None:
    base = np.zeros((8, 4), dtype=np.float32)
    corrections = {lag: np.zeros_like(base) for lag in (0, 1, 2, 4)}
    labels = np.arange(8) % 4
    lag, scale, rows = select_lag_scale(base, corrections, labels)
    assert (lag, scale) == (0, 0.0)
    assert len(rows) == 12
