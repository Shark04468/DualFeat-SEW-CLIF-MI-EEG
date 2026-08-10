from __future__ import annotations

import numpy as np
import torch

from dpc_snn.experiments.v16_training import fit_v16_continuous, predict_v16_continuous
from dpc_snn.models.v16_continuous_fusion import V16ContinuousFusion


def test_v16_forward_preserves_fbc_band_structure_and_prefixes() -> None:
    model = V16ContinuousFusion(hidden_channels=16, band_channels=4, temporal_layers=1)
    output = model(torch.randn(5, 18, 32), torch.randn(5, 4, 288))
    assert output.logits.shape == (5, 4)
    assert output.prefix_logits.shape == (5, 4, 4)
    assert output.fused_sequence.shape == (5, 18, 16)
    assert model.fbc_band_weight.shape == (9, 32, 4)
    assert torch.equal(output.logits, output.prefix_logits[:, -1])


def test_v16_short_fit_and_prediction_are_finite() -> None:
    rng = np.random.default_rng(7)
    atc = rng.normal(size=(16, 18, 32)).astype(np.float32)
    fbc = rng.normal(size=(16, 4, 288)).astype(np.float32)
    labels = np.arange(16, dtype=np.int64) % 4
    fit = fit_v16_continuous(
        atc_train=atc[:12],
        fbc_train=fbc[:12],
        y_train=labels[:12],
        atc_validation=atc[12:],
        fbc_validation=fbc[12:],
        y_validation=labels[12:],
        device="cpu",
        seed=3,
        epochs=2,
        patience=2,
        batch_size=4,
        model_kwargs={"hidden_channels": 16, "band_channels": 4, "temporal_layers": 1},
    )
    prediction = predict_v16_continuous(
        fit.model, atc[12:], fbc[12:], labels[12:], device="cpu", batch_size=4
    )
    assert fit.best_epoch in {1, 2}
    assert prediction["logits"].shape == (4, 4)
    assert prediction["prefix_logits"].shape == (4, 4, 4)
    assert np.isfinite(prediction["logits"]).all()
