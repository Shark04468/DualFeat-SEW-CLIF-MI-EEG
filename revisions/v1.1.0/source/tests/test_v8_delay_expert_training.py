from __future__ import annotations

import numpy as np
import pytest
import torch

from dpc_snn.experiments.v8_delay_expert_training import (
    fit_delay_expert_input_gain,
    fit_v8_delay_expert,
    predict_v8_delay_expert,
)
from dpc_snn.experiments.v8_training import V8CachedRates
from dpc_snn.models.v8_delay_residual_expert import build_v8_delay_residual_expert


def test_fold_fixed_delay_gain_and_prediction_are_finite() -> None:
    model = build_v8_delay_residual_expert(
        "ann_sew",
        n_bands=1,
        n_nodes=2,
        maximum_routes=2,
        maximum_delay=1,
        signal_mode="slow_envelope",
        decoder_channels=4,
        decoder_layers=0,
        sfreq=2.0,
        temporal_decimation=1,
        endpoint_seconds=(2.0,),
        readout_features=8,
        dropout=0.0,
    )
    model.load_fold_prior(
        source_band=torch.tensor([0]),
        source_node=torch.tensor([0]),
        target_band=torch.tensor([0]),
        target_node=torch.tensor([1]),
        delay_probability=torch.tensor([[0.0, 1.0]]),
        fractional_target=torch.tensor([0.0]),
        route_weight=torch.tensor([1.0]),
        route_confidence=torch.tensor([1.0]),
        phase_preference=torch.tensor([0.0]),
        amplitude_scale=torch.tensor([1.0]),
    )
    real = torch.randn(4, 1, 2, 4)
    rates = V8CachedRates(
        fast=torch.complex(real, torch.randn_like(real)),
        slow=torch.randn(4, 1, 2, 2),
        gain=torch.ones(1, 2),
        physical_frontend_fingerprint="test",
    )
    gain = fit_delay_expert_input_gain(model, rates, device="cpu", batch_size=2)
    result = predict_v8_delay_expert(
        model,
        rates,
        np.asarray([0, 1, 2, 3]),
        delay_override="full",
        device="cpu",
        batch_size=2,
    )
    assert torch.isfinite(gain).all()
    assert np.isfinite(result["logits"]).all()
    assert result["current_rms"] > 0.0


def test_sew_clif_delay_expert_has_an_end_to_end_training_gradient() -> None:
    model = build_v8_delay_residual_expert(
        "sew_clif",
        n_bands=1,
        n_nodes=2,
        maximum_routes=2,
        maximum_delay=1,
        signal_mode="slow_envelope",
        decoder_channels=4,
        decoder_layers=0,
        sfreq=2.0,
        temporal_decimation=1,
        endpoint_seconds=(2.0,),
        readout_features=8,
        dropout=0.0,
    )
    model.load_fold_prior(
        source_band=torch.tensor([0]),
        source_node=torch.tensor([0]),
        target_band=torch.tensor([0]),
        target_node=torch.tensor([1]),
        delay_probability=torch.tensor([[0.0, 1.0]]),
        fractional_target=torch.tensor([0.0]),
        route_weight=torch.tensor([1.0]),
        route_confidence=torch.tensor([1.0]),
        phase_preference=torch.tensor([0.0]),
        amplitude_scale=torch.tensor([1.0]),
    )
    real = torch.randn(4, 1, 2, 4)
    rates = V8CachedRates(
        fast=torch.complex(real, torch.randn_like(real)),
        slow=torch.randn(4, 1, 2, 2),
        gain=torch.ones(1, 2),
        physical_frontend_fingerprint="test",
    )
    fit_delay_expert_input_gain(model, rates, device="cpu", batch_size=2)
    before = model.decoder.classifier.weight.detach().clone()
    fit = fit_v8_delay_expert(
        model,
        rates,
        np.asarray([0, 1, 2, 3]),
        validation_rates=None,
        validation_labels=None,
        delay_override="full",
        device="cpu",
        seed=7,
        epochs=1,
        fixed_epoch=1,
        batch_size=2,
        gradient_accumulation_steps=1,
        firing_rate_weight=0.0,
    )
    assert fit.optimizer_steps == 2
    assert not torch.equal(before, fit.model.decoder.classifier.weight.detach())


def test_checkpoint_metric_can_use_the_locked_fused_probability() -> None:
    model = build_v8_delay_residual_expert(
        "ann_sew",
        n_bands=1,
        n_nodes=2,
        maximum_routes=2,
        maximum_delay=1,
        signal_mode="slow_envelope",
        decoder_channels=4,
        decoder_layers=0,
        sfreq=2.0,
        temporal_decimation=1,
        endpoint_seconds=(2.0,),
        readout_features=8,
        dropout=0.0,
    )
    model.load_fold_prior(
        source_band=torch.tensor([0]),
        source_node=torch.tensor([0]),
        target_band=torch.tensor([0]),
        target_node=torch.tensor([1]),
        delay_probability=torch.tensor([[0.0, 1.0]]),
        fractional_target=torch.tensor([0.0]),
        route_weight=torch.tensor([1.0]),
        route_confidence=torch.tensor([1.0]),
        phase_preference=torch.tensor([0.0]),
        amplitude_scale=torch.tensor([1.0]),
    )
    real = torch.randn(4, 1, 2, 4)
    rates = V8CachedRates(
        fast=torch.complex(real, torch.randn_like(real)),
        slow=torch.randn(4, 1, 2, 2),
        gain=torch.ones(1, 2),
        physical_frontend_fingerprint="test",
    )
    fit_delay_expert_input_gain(model, rates, device="cpu", batch_size=2)
    labels = np.asarray([0, 1, 2, 3])
    anchor = np.full((4, 4), 0.01, dtype=np.float32)
    anchor[np.arange(4), labels] = 0.97
    fit = fit_v8_delay_expert(
        model,
        rates,
        labels,
        validation_rates=rates,
        validation_labels=labels,
        validation_anchor_probability=anchor,
        maximum_residual_weight=0.05,
        delay_override="zero",
        device="cpu",
        seed=11,
        epochs=1,
        patience=1,
        minimum_epochs=1,
        batch_size=2,
        gradient_accumulation_steps=1,
        firing_rate_weight=0.0,
    )
    assert fit.history[0]["validation_accuracy"] == pytest.approx(1.0)
    assert "validation_expert_accuracy" in fit.history[0]

    with pytest.raises(ValueError, match="must be supplied together"):
        fit_v8_delay_expert(
            model,
            rates,
            labels,
            validation_rates=rates,
            validation_labels=labels,
            validation_anchor_probability=anchor,
            delay_override="zero",
            device="cpu",
            seed=11,
            epochs=1,
        )
