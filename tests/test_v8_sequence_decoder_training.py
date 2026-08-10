from __future__ import annotations

import numpy as np

from dpc_snn.experiments.v8_sequence_decoder_training import (
    fit_v8_sequence_decoder,
    predict_v8_sequence_decoder,
)


def _toy(seed: int, trials: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.arange(trials, dtype=np.int64) % 4
    sequence = rng.normal(0.0, 0.2, size=(trials, 18, 32)).astype(np.float32)
    sequence[np.arange(trials), :, labels] += 1.0
    teacher = np.eye(4, dtype=np.float32)[labels] * 3.0
    return sequence, labels, teacher


def test_v8_sequence_decoder_training_and_fixed_epoch_paths() -> None:
    x_train, y_train, teacher_train = _toy(0, 24)
    x_validation, y_validation, teacher_validation = _toy(1, 8)
    selected = fit_v8_sequence_decoder(
        "ann_plain",
        x_train=x_train,
        y_train=y_train,
        teacher_train=teacher_train,
        x_validation=x_validation,
        y_validation=y_validation,
        teacher_validation=teacher_validation,
        device="cpu",
        seed=3,
        epochs=2,
        patience=2,
        batch_size=8,
        model_kwargs={"hidden_channels": 16, "readout_features": 16, "dropout": 0.0},
    )
    assert selected.best_epoch in {1, 2}
    assert len(selected.history) == 2
    evaluated = predict_v8_sequence_decoder(
        selected.model,
        x_validation,
        y_validation,
        teacher_validation,
        device="cpu",
        batch_size=8,
    )
    assert evaluated["logits"].shape == (8, 4)

    fixed = fit_v8_sequence_decoder(
        "clif_plain",
        x_train=x_train,
        y_train=y_train,
        teacher_train=teacher_train,
        x_validation=None,
        y_validation=None,
        teacher_validation=None,
        device="cpu",
        seed=4,
        epochs=2,
        fixed_epoch=1,
        batch_size=8,
        model_kwargs={"hidden_channels": 16, "readout_features": 16, "dropout": 0.0},
    )
    assert fixed.best_epoch == 1
    assert fixed.optimizer_steps == 3
