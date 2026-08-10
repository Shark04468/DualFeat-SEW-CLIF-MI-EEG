from __future__ import annotations

import numpy as np

from scripts.analyze_v8_e1_fusion import cross_fitted_probability_weight


def test_cross_fitted_weight_never_uses_the_held_run() -> None:
    label = np.tile(np.arange(4), 6)
    runs = np.repeat(np.arange(6).astype(str), 4)
    anchor = np.eye(4)[label] * 0.8 + 0.05
    branch = anchor.copy()
    branch[:4] = np.roll(branch[:4], 1, axis=1)
    prediction, selections = cross_fitted_probability_weight(
        anchor, branch, label, runs, weights=(0.0, 0.5, 1.0)
    )
    assert np.array_equal(prediction, label)
    assert len(selections) == 6
    assert all(item["selected_anchor_weight"] == 1.0 for item in selections)
