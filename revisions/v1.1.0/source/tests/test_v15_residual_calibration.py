from __future__ import annotations

import numpy as np
import pytest

from dpc_snn.experiments.v15_residual_calibration import (
    calibrated_logits,
    center_residual,
    rms_normalize_residual,
    select_alpha,
)


def test_center_and_rms_normalization() -> None:
    residual = np.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]])
    centered = center_residual(residual)
    normalized = rms_normalize_residual(residual)
    np.testing.assert_allclose(centered.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.sqrt(np.mean(normalized[0] ** 2)), 1.0)
    np.testing.assert_array_equal(normalized[1], np.zeros(4))


def test_zero_alpha_exactly_replays_shared_logits() -> None:
    shared = np.asarray([[1.0, 0.0, -1.0, 2.0]], dtype=np.float32)
    residual = np.asarray([[4.0, 1.0, 2.0, 3.0]], dtype=np.float32)
    output = calibrated_logits(shared, residual, 0.0, rms_normalize=True)
    np.testing.assert_array_equal(output, shared.astype(np.float64))


def test_alpha_selection_prefers_smallest_tied_alpha() -> None:
    shared = np.asarray(
        [[2.0, 1.0, 0.0, 0.0], [0.0, 2.0, 1.0, 0.0]], dtype=np.float64
    )
    residual = np.zeros_like(shared)
    labels = np.asarray([0, 1])
    selected = select_alpha(
        shared, residual, labels, (0.0, 0.5, 1.0), rms_normalize=False
    )
    assert selected.alpha == 0.0
    assert selected.accuracy == 1.0


def test_alpha_selection_uses_residual_when_it_corrects_errors() -> None:
    shared = np.asarray(
        [[2.0, 1.0, 0.0, 0.0], [0.0, 2.0, 1.0, 0.0]], dtype=np.float64
    )
    residual = np.asarray(
        [[-2.0, 3.0, 0.0, 0.0], [0.0, -2.0, 3.0, 0.0]], dtype=np.float64
    )
    labels = np.asarray([1, 2])
    selected = select_alpha(
        shared, residual, labels, (0.0, 0.25, 1.0), rms_normalize=False
    )
    assert selected.alpha == 0.25
    assert selected.accuracy == 1.0


def test_invalid_alpha_is_rejected() -> None:
    values = np.zeros((2, 4))
    with pytest.raises(ValueError, match="non-negative"):
        calibrated_logits(values, values, -0.1, rms_normalize=False)
