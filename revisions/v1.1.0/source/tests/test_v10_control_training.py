from __future__ import annotations

import numpy as np

from dpc_snn.experiments.v10_control_training import (
    fit_v10_control,
    predict_v10_control,
)


def _arrays(trials: int, seed: int) -> tuple[np.ndarray, ...]:
    generator = np.random.default_rng(seed)
    atc = generator.normal(size=(trials, 18, 32)).astype(np.float32)
    fbc = generator.normal(size=(trials, 4, 288)).astype(np.float32)
    labels = np.arange(trials, dtype=np.int64) % 4
    probability = np.full((trials, 4), 0.1 / 3.0, dtype=np.float32)
    probability[np.arange(trials), labels] = 0.9
    teacher = np.log(probability)
    return atc, fbc, labels, teacher


def test_v10_control_training_smoke_cpu() -> None:
    atc, fbc, labels, teacher = _arrays(16, 17)
    fit = fit_v10_control(
        "ann_gru",
        atc_train=atc[:12],
        fbc_train=fbc[:12],
        y_train=labels[:12],
        teacher_train=teacher[:12],
        atc_validation=atc[12:],
        fbc_validation=fbc[12:],
        y_validation=labels[12:],
        teacher_validation=teacher[12:],
        device="cpu",
        seed=23,
        epochs=2,
        patience=2,
        batch_size=4,
        model_kwargs={"fusion_features": 16, "readout_features": 12, "dropout": 0.0},
    )
    assert fit.best_epoch in (1, 2)
    assert fit.optimizer_steps == 6
    prediction = predict_v10_control(
        fit.model,
        atc[12:],
        fbc[12:],
        labels[12:],
        teacher[12:],
        device="cpu",
        batch_size=4,
    )
    assert prediction["logits"].shape == (4, 4)
    assert np.isfinite(prediction["logits"]).all()
