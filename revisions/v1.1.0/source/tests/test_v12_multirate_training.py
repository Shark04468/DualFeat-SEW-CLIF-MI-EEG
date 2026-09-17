from __future__ import annotations

import numpy as np
import pytest

from dpc_snn.experiments.v12_multirate_training import (
    V12_E12_OBJECTIVES,
    V12Objective,
    fit_v12,
    predict_v12,
)
from dpc_snn.models.v13_dual_rate_student import build_v13_student


def test_v12_e12_objectives_have_expected_hard_weights() -> None:
    expected = {
        "o0_current": 0.50,
        "o1_fused_070": 0.30,
        "o2_fused_080": 0.20,
        "o3_light_branch": 0.30,
    }
    assert set(V12_E12_OBJECTIVES) == set(expected)
    for name, hard_weight in expected.items():
        assert V12_E12_OBJECTIVES[name].hard_weight == pytest.approx(hard_weight)


@pytest.mark.parametrize(
    "weights",
    [(-0.1, 0.0, 0.0), (0.8, 0.3, 0.0), (float("nan"), 0.0, 0.0)],
)
def test_v12_objective_rejects_invalid_weights(
    weights: tuple[float, float, float],
) -> None:
    with pytest.raises(ValueError):
        V12Objective(*weights)


def test_v12_fixed_epoch_training_smoke() -> None:
    rng = np.random.default_rng(12)
    samples = 8
    atc = rng.normal(size=(samples, 18, 32)).astype(np.float32)
    fbc = rng.normal(size=(samples, 4, 288)).astype(np.float32)
    labels = np.arange(samples, dtype=np.int64) % 4
    atc_teacher = rng.normal(size=(samples, 4)).astype(np.float32)
    fbc_teacher = rng.normal(size=(samples, 4)).astype(np.float32)
    equal_teacher = 0.5 * (atc_teacher + fbc_teacher)
    fit = fit_v12(
        "dual_rate_branch_temporal",
        atc_train=atc,
        fbc_train=fbc,
        y_train=labels,
        equal_teacher_train=equal_teacher,
        atc_teacher_train=atc_teacher,
        fbc_teacher_train=fbc_teacher,
        atc_validation=None,
        fbc_validation=None,
        y_validation=None,
        equal_teacher_validation=None,
        atc_teacher_validation=None,
        fbc_teacher_validation=None,
        device="cpu",
        seed=7,
        fixed_epoch=1,
        epochs=1,
        scheduler_epochs=1,
        batch_size=4,
    )
    result = predict_v12(
        fit.model,
        atc,
        fbc,
        labels,
        equal_teacher,
        atc_teacher,
        fbc_teacher,
        device="cpu",
        batch_size=4,
    )
    assert result["logits"].shape == (samples, 4)
    assert result["endpoint_logits"].shape == (samples, 4, 4)
    assert result["atc_logits"].shape == (samples, 4)
    assert result["fbc_logits"].shape == (samples, 4)
    assert np.isfinite(result["logits"]).all()


def test_v12_objective_override_training_smoke() -> None:
    rng = np.random.default_rng(120)
    samples = 8
    atc = rng.normal(size=(samples, 18, 32)).astype(np.float32)
    fbc = rng.normal(size=(samples, 4, 288)).astype(np.float32)
    labels = np.arange(samples, dtype=np.int64) % 4
    atc_teacher = rng.normal(size=(samples, 4)).astype(np.float32)
    fbc_teacher = rng.normal(size=(samples, 4)).astype(np.float32)
    equal_teacher = 0.5 * (atc_teacher + fbc_teacher)
    fit = fit_v12(
        "interpolated_branch_kd",
        atc_train=atc,
        fbc_train=fbc,
        y_train=labels,
        equal_teacher_train=equal_teacher,
        atc_teacher_train=atc_teacher,
        fbc_teacher_train=fbc_teacher,
        atc_validation=None,
        fbc_validation=None,
        y_validation=None,
        equal_teacher_validation=None,
        atc_teacher_validation=None,
        fbc_teacher_validation=None,
        device="cpu",
        seed=12,
        fixed_epoch=1,
        epochs=1,
        scheduler_epochs=1,
        batch_size=4,
        objective_override=V12_E12_OBJECTIVES["o2_fused_080"],
        model_builder=lambda kind: build_v13_student(
            "dual_snn_logit_mean", decoder_kind=kind
        ),
    )
    assert fit.optimizer_steps == 2
    assert fit.history[0]["branch_kd_loss"] > 0.0
