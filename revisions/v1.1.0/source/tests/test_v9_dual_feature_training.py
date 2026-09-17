from __future__ import annotations

import numpy as np
import torch

from dpc_snn.experiments.v9_dual_feature_training import (
    V9_DUAL_FEATURE_EXPERIMENT_VARIANTS,
    equal_probability_teacher,
    fit_feature_standardizer,
    fit_v9_dual_feature,
    predict_v9_dual_feature,
)
from dpc_snn.utils.metrics import classification_metrics


def _features(trials: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    atc = generator.normal(size=(trials, 18, 32)).astype(np.float32)
    fbc = generator.normal(size=(trials, 4, 288)).astype(np.float32)
    return atc, fbc


def test_equal_probability_teacher_is_normalized() -> None:
    atc = np.asarray([[4.0, 0.0, -2.0, 1.0], [0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
    fbc = np.asarray([[0.0, 4.0, 1.0, -2.0], [3.0, 2.0, 1.0, 0.0]], dtype=np.float32)
    teacher = equal_probability_teacher(atc, fbc)
    probability = np.exp(teacher)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-6)
    expected = 0.5 * torch.softmax(torch.from_numpy(atc), 1).numpy()
    expected += 0.5 * torch.softmax(torch.from_numpy(fbc), 1).numpy()
    np.testing.assert_allclose(probability, expected, atol=1e-6)


def test_feature_standardizer_uses_supplied_training_features_only() -> None:
    atc, fbc = _features(10)
    standardizer = fit_feature_standardizer(atc, fbc)
    transformed_atc, transformed_fbc = standardizer.transform(atc, fbc)
    np.testing.assert_allclose(transformed_atc.mean(axis=(0, 1)), 0.0, atol=2e-6)
    np.testing.assert_allclose(transformed_fbc.mean(axis=(0, 1)), 0.0, atol=2e-6)
    validation_atc = atc[:2] + 1_000.0
    validation_fbc = fbc[:2] - 1_000.0
    transformed_validation = standardizer.transform(validation_atc, validation_fbc)
    assert float(transformed_validation[0].mean()) > 100.0
    assert float(transformed_validation[1].mean()) < -100.0


def test_ce_and_kd_variants_share_one_model_variant() -> None:
    assert V9_DUAL_FEATURE_EXPERIMENT_VARIANTS["ann_sew_ce"].model_variant == "ann_sew"
    assert V9_DUAL_FEATURE_EXPERIMENT_VARIANTS["ann_sew_kd"].model_variant == "ann_sew"
    assert V9_DUAL_FEATURE_EXPERIMENT_VARIANTS["sew_clif_ce"].model_variant == "sew_clif"
    assert V9_DUAL_FEATURE_EXPERIMENT_VARIANTS["sew_clif_kd"].model_variant == "sew_clif"


def test_dual_feature_training_smoke_cpu() -> None:
    atc, fbc = _features(16, seed=4)
    labels = np.arange(16, dtype=np.int64) % 4
    teacher_logits = np.eye(4, dtype=np.float32)[labels] * 3.0
    teacher = equal_probability_teacher(teacher_logits, teacher_logits)
    standardizer = fit_feature_standardizer(atc[:12], fbc[:12])
    train_atc, train_fbc = standardizer.transform(atc[:12], fbc[:12])
    validation_atc, validation_fbc = standardizer.transform(atc[12:], fbc[12:])
    fit = fit_v9_dual_feature(
        "sew_clif_kd",
        atc_train=train_atc,
        fbc_train=train_fbc,
        y_train=labels[:12],
        teacher_train=teacher[:12],
        atc_validation=validation_atc,
        fbc_validation=validation_fbc,
        y_validation=labels[12:],
        teacher_validation=teacher[12:],
        device="cpu",
        seed=11,
        epochs=2,
        patience=2,
        batch_size=4,
        model_kwargs={
            "hidden_channels": 16,
            "decoder_layers": 1,
            "readout_features": 12,
            "dropout": 0.0,
        },
    )
    assert fit.best_epoch in (1, 2)
    assert fit.optimizer_steps == 6
    prediction = predict_v9_dual_feature(
        fit.model,
        validation_atc,
        validation_fbc,
        labels[12:],
        teacher[12:],
        device="cpu",
        batch_size=4,
    )
    assert prediction["logits"].shape == (4, 4)
    assert np.isfinite(prediction["logits"]).all()


def test_dual_feature_prediction_supports_binary_external_task() -> None:
    atc, fbc = _features(6, seed=17)
    labels = np.arange(6, dtype=np.int64) % 2
    teacher_logits = np.eye(2, dtype=np.float32)[labels] * 2.0
    teacher = equal_probability_teacher(teacher_logits, teacher_logits)
    fit = fit_v9_dual_feature(
        "ann_sew_ce",
        atc_train=atc,
        fbc_train=fbc,
        y_train=labels,
        teacher_train=teacher,
        atc_validation=None,
        fbc_validation=None,
        y_validation=None,
        teacher_validation=None,
        device="cpu",
        seed=19,
        fixed_epoch=1,
        batch_size=3,
        model_kwargs={
            "n_classes": 2,
            "hidden_channels": 16,
            "decoder_layers": 1,
            "readout_features": 12,
            "dropout": 0.0,
        },
    )
    prediction = predict_v9_dual_feature(
        fit.model,
        atc,
        fbc,
        labels,
        teacher,
        device="cpu",
        batch_size=3,
    )
    assert prediction["logits"].shape == (6, 2)
    expected = classification_metrics(
        labels,
        prediction["logits"].argmax(axis=1),
        n_classes=2,
    )
    assert prediction["macro_f1"] == expected["macro_f1"]
