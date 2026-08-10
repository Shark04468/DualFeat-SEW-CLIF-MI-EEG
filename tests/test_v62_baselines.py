from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dpc_snn.baselines.neural import (
    SOURCE_LOCKS,
    BFATCNetReconstruction,
    TorchPLIFMI,
    verify_official_source_locks,
)
from dpc_snn.experiments.v62_baselines import (
    apply_fixed_gain,
    fbcnet_filterbank,
    fit_fixed_gain,
    prepare_model_input,
    segment_reconstruction,
    task_carrier,
)


def test_official_source_locks_require_exact_commit_and_archive(tmp_path: Path) -> None:
    for name, (commit, archive) in SOURCE_LOCKS.items():
        directory = tmp_path / name
        directory.mkdir()
        (directory / "SOURCE_LOCK.txt").write_text(
            f"{name} {commit} {archive}\n", encoding="utf-8"
        )
    verified = verify_official_source_locks(tmp_path)
    assert set(verified) == set(SOURCE_LOCKS)
    (tmp_path / "TCFormer" / "SOURCE_LOCK.txt").write_text("stale\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="mismatch"):
        verify_official_source_locks(tmp_path)


def test_task_carrier_uses_car_baseline_and_exact_four_seconds() -> None:
    x = np.zeros((2, 22, 1251), dtype=np.float32)
    x[:, 0, :250] = 3.0
    x[:, 0, 250:1250] = 5.0
    carrier = task_carrier(x)
    assert carrier.shape == (2, 22, 1000)
    assert np.isfinite(carrier).all()
    assert np.allclose(carrier.mean(axis=1), 0.0, atol=1e-6)


def test_fixed_gain_is_training_fitted_and_bounded() -> None:
    train = np.ones((3, 2, 10), dtype=np.float32)
    gain = fit_fixed_gain(train, clip=2.0)
    evaluation = apply_fixed_gain(np.full((1, 2, 10), 100.0, dtype=np.float32), gain)
    assert gain.values.shape == (1, 2, 1)
    assert float(evaluation.max()) == 2.0


def test_official_fbcnet_filterbank_shape_and_finiteness() -> None:
    rng = np.random.default_rng(0)
    output = fbcnet_filterbank(rng.normal(size=(2, 22, 1000)).astype(np.float32))
    assert output.shape == (2, 1, 22, 1000, 9)
    assert np.isfinite(output).all()


def test_model_specific_official_views_have_expected_shapes() -> None:
    x = np.random.default_rng(0).normal(size=(2, 22, 1000)).astype(np.float32)
    assert prepare_model_input("eegnet", x).shape == (2, 22, 256)
    conformer = prepare_model_input("eeg_conformer", x)
    assert conformer.shape == x.shape
    assert np.isfinite(conformer).all()


def test_paper_faithful_custom_baselines_obey_logits_contract() -> None:
    x = torch.randn(2, 22, 128)
    plif = TorchPLIFMI(samples=128)
    bfatcnet = BFATCNetReconstruction()
    for model in (plif, bfatcnet):
        output = model(x)
        assert output["logits"].shape == (2, 4)
        output["logits"].square().mean().backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )


def test_segment_reconstruction_preserves_shape_and_finiteness() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 22, 64)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    augmented = segment_reconstruction(x, labels, probability=1.0)
    assert augmented.shape == x.shape
    assert torch.isfinite(augmented).all()
