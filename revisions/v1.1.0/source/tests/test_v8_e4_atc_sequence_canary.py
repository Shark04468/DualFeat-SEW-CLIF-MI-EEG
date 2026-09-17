from __future__ import annotations

import numpy as np
import pytest

from scripts.run_v8_e4_atc_sequence_canary import (
    fixed_gain_from_result,
    probability_fusion,
)


def test_fixed_gain_from_result_restores_saved_channel_shape() -> None:
    gain = fixed_gain_from_result({"gain": list(np.linspace(1.0, 2.0, 22))}, "gain", clip=12.0)
    assert gain.values.shape == (1, 22, 1)
    assert gain.clip == 12.0


def test_probability_fusion_is_fixed_and_shape_checked() -> None:
    first = np.asarray([[4.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    second = np.asarray([[0.0, 4.0], [0.0, 4.0]], dtype=np.float32)
    probability, prediction = probability_fusion(first, second, first_weight=0.5)
    assert probability.shape == first.shape
    assert prediction.tolist() == [0, 1]
    with pytest.raises(ValueError, match="aligned"):
        probability_fusion(first, second[:1])
