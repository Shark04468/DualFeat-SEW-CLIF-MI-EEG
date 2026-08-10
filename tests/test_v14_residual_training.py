from __future__ import annotations

import numpy as np

from dpc_snn.experiments.v14_residual_training import fit_v14, predict_v14
from dpc_snn.models.v9_dual_feature_student import build_v9_dual_feature_student


def test_v14_two_stage_training_smoke() -> None:
    rng = np.random.default_rng(14)
    samples = 8
    atc = rng.normal(size=(samples, 18, 32)).astype(np.float32)
    fbc = rng.normal(size=(samples, 4, 288)).astype(np.float32)
    labels = np.arange(samples, dtype=np.int64) % 4
    equal = rng.normal(size=(samples, 4)).astype(np.float32)
    atc_teacher = rng.normal(size=(samples, 4)).astype(np.float32)
    fbc_teacher = rng.normal(size=(samples, 4)).astype(np.float32)
    shared = build_v9_dual_feature_student("sew_clif").state_dict()
    fit = fit_v14(
        "r3_dual_residual",
        shared,
        atc_train=atc,
        fbc_train=fbc,
        y_train=labels,
        equal_teacher_train=equal,
        atc_teacher_train=atc_teacher,
        fbc_teacher_train=fbc_teacher,
        atc_validation=None,
        fbc_validation=None,
        y_validation=None,
        equal_teacher_validation=None,
        atc_teacher_validation=None,
        fbc_teacher_validation=None,
        device="cpu",
        seed=14,
        epochs=1,
        fixed_epoch=1,
        pretrain_epochs=1,
        batch_size=4,
    )
    assert fit.optimizer_steps == 4
    assert [row["phase"] for row in fit.history] == [
        "residual_pretrain",
        "residual_classification",
    ]
    result = predict_v14(
        fit.model,
        atc,
        fbc,
        labels,
        equal,
        atc_teacher,
        fbc_teacher,
        device="cpu",
        batch_size=4,
    )
    assert result["logits"].shape == (samples, 4)
    assert result["shared_logits"].shape == (samples, 4)
    assert np.isfinite(result["logits"]).all()
    assert set(result["gates"]) == {"atc", "fbc", "generic"}
