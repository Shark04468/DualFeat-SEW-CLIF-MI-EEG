from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from dpc_snn.experiments import v62_baselines
from dpc_snn.experiments.v8_baselines import (
    V8BaselineProtocolError,
    merge_oof_predictions,
    nested_run_grouped_indices,
    rank_screening_models,
    session_t_development_view,
)


def test_fixed_epoch_baseline_reuses_full_scheduler_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    class TinyBaseline(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = nn.Linear(6, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier(x.flatten(1))

    monkeypatch.setitem(
        v62_baselines.BASELINE_OPTIMIZERS,
        "tiny_scheduled",
        {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "beta1": 0.9,
            "batch_size": 4,
            "schedule": True,
        },
    )
    monkeypatch.setattr(
        v62_baselines,
        "build_v62_neural_baseline",
        lambda *args, **kwargs: TinyBaseline(),
    )
    rng = np.random.default_rng(17)
    x = rng.normal(size=(8, 2, 3)).astype(np.float32)
    y = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)

    def train(fixed_epoch: int | None):
        return v62_baselines.fit_baseline(
            "tiny_scheduled",
            source_root="unused",
            x_train=x,
            y_train=y,
            x_validation=None,
            y_validation=None,
            device="cpu",
            seed=23,
            epochs=6,
            patience=6,
            augmentation={"enabled": False},
            fixed_epoch=fixed_epoch,
            scheduler_epochs=6,
        )

    cutoff = train(3)
    complete = train(None)
    np.testing.assert_allclose(
        [row["loss"] for row in cutoff.history],
        [row["loss"] for row in complete.history[:3]],
        rtol=0,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        [row["lr"] for row in cutoff.history],
        [row["lr"] for row in complete.history[:3]],
        rtol=0,
        atol=0,
    )


def test_gradient_accumulation_matches_full_effective_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TinyBaseline(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = nn.Linear(6, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier(x.flatten(1))

    monkeypatch.setitem(
        v62_baselines.BASELINE_OPTIMIZERS,
        "tiny_accumulation",
        {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "beta1": 0.9,
            "batch_size": 8,
            "schedule": False,
        },
    )
    monkeypatch.setattr(
        v62_baselines,
        "build_v62_neural_baseline",
        lambda *args, **kwargs: TinyBaseline(),
    )
    rng = np.random.default_rng(71)
    x = rng.normal(size=(10, 2, 3)).astype(np.float32)
    y = np.asarray([0, 1, 2, 3, 0, 1, 2, 3, 0, 1], dtype=np.int64)

    def train(physical_batch_size: int):
        return v62_baselines.fit_baseline(
            "tiny_accumulation",
            source_root="unused",
            x_train=x,
            y_train=y,
            x_validation=None,
            y_validation=None,
            device="cpu",
            seed=29,
            epochs=2,
            patience=2,
            augmentation={"enabled": False},
            fixed_epoch=2,
            physical_batch_size=physical_batch_size,
            effective_batch_size=8,
        )

    full = train(8)
    accumulated = train(4)
    assert full.optimizer_steps == accumulated.optimizer_steps == 4
    for key in full.last_state:
        torch.testing.assert_close(full.last_state[key], accumulated.last_state[key])


def test_gradient_accumulation_rejects_incommensurate_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        v62_baselines.BASELINE_OPTIMIZERS,
        "invalid_accumulation",
        {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "beta1": 0.9,
            "batch_size": 4,
            "schedule": False,
        },
    )
    with pytest.raises(ValueError, match="integer multiple"):
        v62_baselines.fit_baseline(
            "invalid_accumulation",
            source_root="unused",
            x_train=np.zeros((4, 2, 3), dtype=np.float32),
            y_train=np.asarray([0, 1, 2, 3]),
            x_validation=None,
            y_validation=None,
            device="cpu",
            seed=1,
            epochs=1,
            patience=1,
            physical_batch_size=3,
            effective_batch_size=8,
        )


def test_fbcnet_checkpoint_is_sealed_after_forward_constraint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ForwardConstrainedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.full((4, 6), 0.2))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                self.weight.copy_(torch.renorm(self.weight, p=2, dim=0, maxnorm=0.5))
            return x.flatten(1) @ self.weight.T

    monkeypatch.setitem(
        v62_baselines.BASELINE_OPTIMIZERS,
        "fbcnet",
        {
            "lr": 0.2,
            "weight_decay": 0.0,
            "beta1": 0.9,
            "batch_size": 4,
            "schedule": False,
        },
    )
    monkeypatch.setattr(
        v62_baselines,
        "build_v62_neural_baseline",
        lambda *args, **kwargs: ForwardConstrainedModel(),
    )
    x = np.arange(48, dtype=np.float32).reshape(8, 2, 3) / 10.0
    y = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    fitted = v62_baselines.fit_baseline(
        "fbcnet",
        source_root="unused",
        x_train=x,
        y_train=y,
        x_validation=None,
        y_validation=None,
        device="cpu",
        seed=3,
        epochs=2,
        patience=2,
        augmentation={"enabled": False},
        fixed_epoch=2,
        physical_batch_size=4,
        effective_batch_size=4,
    )
    before = {key: value.clone() for key, value in fitted.model.state_dict().items()}
    v62_baselines.predict_baseline(fitted.model, x, y, device="cpu", batch_size=4)
    after = fitted.model.state_dict()
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0.0, atol=0.0)


def _data() -> dict[str, object]:
    count = 16
    return {
        "X": np.zeros((count, 2, 10), dtype=np.float32),
        "y": np.tile(np.arange(4), 4),
        "subject": np.asarray(["1"] * count),
        "session": np.asarray(["T"] * 8 + ["E"] * 8),
        "run": np.asarray([f"run_{index // 4}" for index in range(count)]),
        "trial_id": np.asarray([f"trial_{index}" for index in range(count)]),
        "dataset_name": "bci2a",
        "sfreq": 250.0,
        "epoch_tmin": -1.0,
        "epoch_tmax": 4.0,
        "ch_names": ["C3", "C4"],
    }


def test_development_view_exposes_only_session_t() -> None:
    x, y, rows, manifest = session_t_development_view(_data(), expected_trials=8)
    assert x.shape[0] == y.shape[0] == len(rows) == 8
    assert {row["session"] for row in rows} == {"T"}
    assert manifest["excluded_session_trial_counts"] == {"E": 8}
    assert manifest["heldout_labels_used_by_development"] is False


def test_oof_merge_requires_exact_coverage() -> None:
    labels = np.asarray([0, 1, 2, 3])
    parts = [
        {"indices": [2, 0], "logits": np.zeros((2, 4)), "labels": [2, 0]},
        {"indices": [3, 1], "logits": np.ones((2, 4)), "labels": [3, 1]},
    ]
    indices, logits, merged_labels = merge_oof_predictions(parts, labels=labels)
    assert indices.tolist() == [0, 1, 2, 3]
    assert logits.shape == (4, 4)
    assert np.array_equal(merged_labels, labels)

    parts[1]["indices"] = [3, 0]
    with pytest.raises(V8BaselineProtocolError, match="exactly once"):
        merge_oof_predictions(parts, labels=labels)


def test_nested_run_split_keeps_outer_test_untouched() -> None:
    metadata = [{"run": f"run_{index // 2}"} for index in range(12)]
    outer_test = np.asarray([0, 1])
    outer_train = np.asarray(list(range(2, 12)))
    inner_train, inner_validation, inner_run = nested_run_grouped_indices(
        metadata, outer_train, outer_test
    )
    assert inner_run == "run_1"
    assert inner_validation.tolist() == [2, 3]
    assert set(inner_train) | set(inner_validation) == set(outer_train)
    assert not (set(inner_train) | set(inner_validation)) & set(outer_test)


def test_screening_rank_is_subject_macro_average() -> None:
    rows = [
        {"model": "a", "subject": 1, "seed": 0, "accuracy": 0.8, "kappa": 0.7, "macro_f1": 0.8},
        {"model": "a", "subject": 3, "seed": 0, "accuracy": 0.6, "kappa": 0.5, "macro_f1": 0.6},
        {"model": "b", "subject": 1, "seed": 0, "accuracy": 0.75, "kappa": 0.6, "macro_f1": 0.7},
        {"model": "b", "subject": 3, "seed": 0, "accuracy": 0.75, "kappa": 0.6, "macro_f1": 0.7},
    ]
    ranking = rank_screening_models(rows, subjects=[1, 3], screening_seed=0)
    assert [row["model"] for row in ranking] == ["b", "a"]
