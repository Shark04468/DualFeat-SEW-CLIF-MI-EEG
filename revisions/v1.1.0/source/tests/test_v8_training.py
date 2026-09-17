from __future__ import annotations

import numpy as np
import torch

from dpc_snn.experiments.v8_training import (
    V8CachedRates,
    apply_v8_fold_gain,
    cache_v8_physical_rates,
    fit_v8,
    fit_v8_physical_gain,
    fit_v8_gain_from_cached_rates,
    load_v8_rates,
    paired_v8_rate_reconstruction,
    predict_v8,
    save_v8_rates,
)
from dpc_snn.models.v8_accuracy_first import V8AccuracyFirstModel


def _model(kind: str = "ann") -> V8AccuracyFirstModel:
    model = V8AccuracyFirstModel(
        n_classes=4,
        n_bands=2,
        n_latent_nodes=4,
        band_edges_hz=((4, 8), (8, 12)),
        sfreq=32,
        epoch_tmin=-0.5,
        task_tmin=0,
        task_tmax=1,
        analytic_taps=17,
        envelope_taps=9,
        fast_decimation=2,
        spatial_rank=1,
        temporal_dilations=(1,),
        temporal_depth=1,
        decoder_kind=kind,
        decoder_layers=1,
        decoder_channels=8,
        endpoint_seconds=(0.5, 1.0),
        statistical_spatial_filters=2,
        statistical_features=8,
        statistical_segments=2,
        covariance_features_per_band=2,
        fusion_features=16,
        dropout=0.0,
        parameter_ceiling=100_000,
    )
    model.set_training_gain(torch.ones(2, 22))
    return model


def _rates(count: int = 8) -> V8CachedRates:
    model = _model()
    return V8CachedRates(
        fast=torch.randn(count, 2, 22, 16, dtype=torch.complex64),
        slow=torch.randn(count, 2, 22, 8),
        gain=torch.ones(2, 22),
        physical_frontend_fingerprint=model.physical_frontend_fingerprint(),
    )


def test_rate_cache_roundtrip_and_gain_fit(tmp_path) -> None:
    model = _model()
    raw = np.random.default_rng(0).normal(size=(4, 22, 48)).astype(np.float32)
    gain = fit_v8_physical_gain(model, raw, device="cpu", batch_size=2)
    assert gain.shape == (2, 22)
    assert torch.isfinite(gain).all() and (gain > 0).all()

    rates = _rates()
    path = tmp_path / "rates.pt"
    save_v8_rates(path, rates)
    loaded = load_v8_rates(path, model)
    assert torch.equal(loaded.fast, rates.fast)
    assert torch.equal(loaded.slow, rates.slow)


def test_cached_fold_gain_matches_carrier_scaling_and_keeps_relative_envelope() -> None:
    base = _rates()
    gain = fit_v8_gain_from_cached_rates(base, np.arange(6))
    scaled = apply_v8_fold_gain(base, gain)
    assert torch.allclose(scaled.fast, base.fast * gain[None, :, :, None])
    assert torch.equal(scaled.slow, base.slow)
    assert torch.equal(scaled.gain, gain)


def test_direct_and_cached_gain_paths_match_relatively_and_share_envelope() -> None:
    torch.manual_seed(13)
    model = _model()
    raw = (
        np.random.default_rng(13).normal(size=(6, 22, 48)).astype(np.float32)
        * 1.0e-5
    )
    model.set_training_gain(torch.ones(2, 22))
    base = cache_v8_physical_rates(model, raw, device="cpu", batch_size=3)
    indices = np.asarray([0, 2, 3, 5], dtype=np.int64)
    cached_gain = fit_v8_gain_from_cached_rates(base, indices)
    direct_gain = fit_v8_physical_gain(
        model, raw[indices], device="cpu", batch_size=2
    )
    relative = (cached_gain - direct_gain).abs() / direct_gain.abs()
    assert float(relative.max()) <= 1.0e-6

    direct = cache_v8_physical_rates(
        model, raw[indices[:2]], device="cpu", batch_size=2
    )
    transformed = apply_v8_fold_gain(base.subset(indices[:2]), cached_gain)
    torch.testing.assert_close(direct.fast, transformed.fast, rtol=1.0e-5, atol=1.0e-7)
    torch.testing.assert_close(direct.slow, transformed.slow, rtol=0.0, atol=0.0)


def test_aligned_augmentation_preserves_shapes() -> None:
    rates = _rates()
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    fast, slow = paired_v8_rate_reconstruction(
        rates.fast, rates.slow, labels, probability=1.0, segments=4
    )
    assert fast.shape == rates.fast.shape
    assert slow.shape == rates.slow.shape
    assert torch.isfinite(fast.real).all() and torch.isfinite(slow).all()


def test_tiny_v8_fit_and_prediction_are_finite() -> None:
    model = _model("ann")
    rates = _rates()
    labels = np.tile(np.arange(4), 2)
    fit = fit_v8(
        model,
        rates,
        labels,
        validation_rates=rates,
        validation_labels=labels,
        device="cpu",
        seed=0,
        epochs=2,
        patience=2,
        minimum_epochs=1,
        batch_size=4,
        accumulation_steps=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        max_gradient_norm=5.0,
        warmup_fraction=0.1,
        endpoint_weights=(0.0, 0.0),
        endpoint_loss_weight=0.0,
        firing_rate_weight=0.0,
        spatial_orthogonality_weight=1e-4,
        statistical_orthogonality_weight=1e-4,
        label_smoothing=0.0,
        augmentation={"enabled": False},
        run_label="test",
    )
    result = predict_v8(fit.model, rates, labels, device="cpu", batch_size=4)
    assert fit.best_epoch >= 1
    assert np.isfinite(result["logits"]).all()
    assert result["prefix_logits"].shape == (8, 2, 4)
    assert result["binary_spike_rate"] is None
    assert 0.0 <= result["final_activity_nonzero_rate"] <= 1.0
    assert result["final_activity_absolute_mean"] >= 0.0


def test_fixed_epoch_v8_reuses_full_scheduler_prefix() -> None:
    torch.manual_seed(31)
    rates = _rates()
    labels = np.tile(np.arange(4), 2)

    def train(fixed_epoch: int | None):
        torch.manual_seed(43)
        model = _model("ann")
        return fit_v8(
            model,
            rates,
            labels,
            validation_rates=None,
            validation_labels=None,
            device="cpu",
            seed=47,
            epochs=4,
            patience=4,
            minimum_epochs=1,
            batch_size=4,
            accumulation_steps=1,
            learning_rate=1e-3,
            weight_decay=1e-4,
            max_gradient_norm=5.0,
            warmup_fraction=0.25,
            endpoint_weights=(0.0, 0.0),
            endpoint_loss_weight=0.0,
            firing_rate_weight=0.0,
            spatial_orthogonality_weight=1e-4,
            statistical_orthogonality_weight=1e-4,
            label_smoothing=0.0,
            augmentation={"enabled": False},
            fixed_epoch=fixed_epoch,
            scheduler_epochs=4,
            run_label="scheduler-prefix-test",
        )

    cutoff = train(2)
    complete = train(None)
    np.testing.assert_allclose(
        [row["train_loss"] for row in cutoff.history],
        [row["train_loss"] for row in complete.history[:2]],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        [row["learning_rate"] for row in cutoff.history],
        [row["learning_rate"] for row in complete.history[:2]],
        rtol=0,
        atol=0,
    )


def test_delay_training_can_retain_epoch_zero_and_preserve_matched_transport() -> None:
    model = V8AccuracyFirstModel(
        n_classes=4,
        n_bands=2,
        n_latent_nodes=4,
        band_edges_hz=((4, 8), (8, 12)),
        sfreq=32,
        epoch_tmin=-0.5,
        task_tmin=0,
        task_tmax=1,
        analytic_taps=17,
        envelope_taps=9,
        fast_decimation=2,
        spatial_rank=1,
        temporal_dilations=(1,),
        temporal_depth=1,
        decoder_kind="ann",
        decoder_layers=1,
        decoder_channels=8,
        endpoint_seconds=(0.5, 1.0),
        statistical_spatial_filters=2,
        statistical_features=8,
        statistical_segments=2,
        covariance_features_per_band=2,
        delay_auxiliary_enabled=True,
        delay_maximum_routes=2,
        delay_maximum_samples=2,
        fusion_features=16,
        dropout=0.0,
        parameter_ceiling=100_000,
    )
    probability = torch.zeros(1, 3)
    probability[0, 1] = 1.0
    model.load_fold_delay_prior(
        source_band=torch.tensor([0]),
        source_node=torch.tensor([0]),
        target_band=torch.tensor([0]),
        target_node=torch.tensor([1]),
        delay_probability=probability,
        fractional_target=torch.tensor([0.25]),
        route_weight=torch.tensor([1.0]),
        route_confidence=torch.tensor([0.8]),
        amplitude_scale=torch.tensor([0.5]),
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name == "delay_fusion_raw")
    rates = V8CachedRates(
        fast=torch.randn(8, 2, 22, 16, dtype=torch.complex64),
        slow=torch.randn(8, 2, 22, 8),
        gain=torch.ones(2, 22),
        physical_frontend_fingerprint=model.physical_frontend_fingerprint(),
    )
    labels = np.tile(np.arange(4), 2)
    fit = fit_v8(
        model,
        rates,
        labels,
        validation_rates=rates,
        validation_labels=labels,
        device="cpu",
        seed=61,
        epochs=1,
        patience=1,
        minimum_epochs=1,
        batch_size=4,
        accumulation_steps=1,
        learning_rate=0.0,
        weight_decay=0.0,
        max_gradient_norm=5.0,
        warmup_fraction=0.1,
        endpoint_weights=(0.0, 0.0),
        endpoint_loss_weight=0.0,
        firing_rate_weight=0.0,
        spatial_orthogonality_weight=0.0,
        statistical_orthogonality_weight=0.0,
        label_smoothing=0.0,
        augmentation={"enabled": False},
        delay_override="full",
        include_initial_validation=True,
        run_label="delay-epoch-zero-test",
    )
    assert fit.best_epoch == 0
    off = predict_v8(fit.model, rates, labels, device="cpu", batch_size=4)
    zero = predict_v8(
        fit.model,
        rates,
        labels,
        device="cpu",
        batch_size=4,
        delay_override="zero",
    )
    full = predict_v8(
        fit.model,
        rates,
        labels,
        device="cpu",
        batch_size=4,
        delay_override="full",
    )
    assert not np.array_equal(off["logits"], zero["logits"])
    assert np.isfinite(full["logits"]).all()
