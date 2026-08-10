from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_scaffold import (
    aligned_slow_rate_analytic_evidence,
    build_frontend_only_scaffold,
    CachedRates,
    build_zero_scaffold,
    cache_analytic_fixed_channel_gain_rate_features,
    cache_official_fbc_rate_features,
    cache_rate_features,
    fit_official_fbc_channel_gain,
    fit_projected_training_gain,
    fit_zero_scaffold,
    load_cached_rates,
    paired_rate_reconstruction,
    predict_scaffold,
    project_registered_max_norm_constraints_,
    save_cached_rates,
    scaffold_optimizer_groups,
)
from dpc_snn.experiments.v62_baselines import apply_fixed_gain, task_carrier  # noqa: E402
from dpc_snn.experiments.v7_delay import configure_delay_stage  # noqa: E402
from dpc_snn.models.delayed_full_window_atc_readout import (  # noqa: E402
    DelayedFullWindowATCReadout,
)


def _model_config() -> dict:
    return {
        "n_bands": 2,
        "n_nodes": 3,
        "band_edges_hz": ((8.0, 12.0), (18.0, 24.0)),
        "analytic_taps": 33,
        "envelope_taps": 9,
        "spatial_rank": 2,
        "route_rank": 2,
        "delay_rank": 2,
        "temporal_dilations": (1, 2),
        "temporal_depth": 1,
        "temporal_scale_attention": False,
        "use_geometry": False,
        "snn_channels": 8,
        "decoder_layers": 0,
        "statistical_spatial_filters": 3,
        "dropout": 0.0,
    }


def test_explicit_max_norm_projection_prevents_inference_state_mutation() -> None:
    class MutatingConstrainedLinear(torch.nn.Linear):
        def __init__(self) -> None:
            super().__init__(5, 4)
            self.max_norm = 0.25

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.weight.data = torch.renorm(
                self.weight.data,
                p=2,
                dim=0,
                maxnorm=self.max_norm,
            )
            return torch.nn.functional.linear(x, self.weight, self.bias)

    layer = MutatingConstrainedLinear()
    with torch.no_grad():
        layer.weight.fill_(10.0)
    assert project_registered_max_norm_constraints_(layer) == 1
    before = {name: value.detach().clone() for name, value in layer.state_dict().items()}

    layer(torch.randn(3, 5))

    for name, value in layer.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_fixed_full_optimizer_uses_discriminative_atc_core_lr() -> None:
    class Readout(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.BatchNorm1d(4),
            )
            self.delayed_statistics_classifier = torch.nn.Linear(5, 4)
            self.delayed_residual_mix_raw = torch.nn.Parameter(torch.zeros(()))
            self.delayed_residual_bound = 0.25

    model = build_zero_scaffold(seed=41, model_config=_model_config())
    model.atc_readout = Readout()
    configure_delay_stage(
        model,
        "static_slow_within",
        readout_training_scope="fixed_full",
    )
    groups = scaffold_optimizer_groups(
        model,
        learning_rate=3e-4,
        delay_learning_rate_multiplier=2.0,
        atc_core_learning_rate_multiplier=0.1,
    )

    by_name = {str(group["group_name"]): group for group in groups}
    assert set(by_name) == {"base", "atc_core"}
    assert float(by_name["base"]["lr"]) == pytest.approx(3e-4)
    assert float(by_name["atc_core"]["lr"]) == pytest.approx(3e-5)
    core_ids = {id(parameter) for parameter in model.atc_readout.core.parameters()}
    assert {id(parameter) for parameter in by_name["atc_core"]["params"]} == core_ids
    assert not any(parameter.requires_grad for parameter in model.delay.parameters())


def test_fold_local_gain_cache_roundtrip_and_zero_delay_freeze(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    raw = (rng.normal(size=(4, 22, 1250)) * 1e-5).astype(np.float32)
    model = build_zero_scaffold(seed=5, model_config=_model_config())

    gain = fit_projected_training_gain(model, raw[:3], device="cpu", batch_size=2)
    cached = cache_rate_features(model, raw, device="cpu", batch_size=2)

    assert gain.shape == (2, 3)
    assert cached.fast.shape == (4, 2, 3, 500)
    assert cached.slow.shape == (4, 2, 3, 250)
    assert cached.broadband.shape == (4, 3, 500)
    assert torch.isfinite(cached.fast.real).all()
    assert torch.isfinite(cached.slow).all()
    assert not any(
        parameter.requires_grad
        for name, parameter in model.delay.named_parameters()
        if name.startswith(("slow_delay_field.", "fast_delay_field."))
    )

    path = tmp_path / "rates.pt"
    save_cached_rates(path, cached)
    loaded = load_cached_rates(path, model)
    torch.testing.assert_close(loaded.fast, cached.fast)
    torch.testing.assert_close(loaded.slow, cached.slow)
    torch.testing.assert_close(loaded.broadband, cached.broadband)
    assert loaded.frontend_fingerprint == cached.frontend_fingerprint

    other_seed = build_zero_scaffold(seed=999, model_config=_model_config())
    assert other_seed.frontend_fingerprint() == model.frontend_fingerprint()


def test_paired_reconstruction_and_one_epoch_training_are_finite() -> None:
    torch.manual_seed(7)
    model = build_zero_scaffold(seed=7, model_config=_model_config())
    model.set_training_gain(torch.ones(2, 3))
    fast = torch.complex(torch.randn(8, 2, 3, 500), torch.randn(8, 2, 3, 500))
    slow = torch.randn(8, 2, 3, 250)
    broadband = torch.randn(8, 3, 500)
    labels = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    rates = CachedRates(
        fast=fast,
        slow=slow,
        broadband=broadband,
        context=None,
        gain=torch.ones(2, 3),
        frontend_fingerprint=model.frontend_fingerprint(),
    )

    mixed_fast, mixed_slow, mixed_broadband = paired_rate_reconstruction(
        fast,
        slow,
        broadband,
        torch.from_numpy(labels),
        probability=1.0,
    )
    assert mixed_fast.shape == fast.shape
    assert mixed_slow.shape == slow.shape
    assert mixed_broadband.shape == broadband.shape
    assert torch.isfinite(mixed_fast.real).all() and torch.isfinite(mixed_slow).all()

    fit = fit_zero_scaffold(
        model,
        train_rates=rates,
        y_train=labels,
        validation_rates=rates,
        y_validation=labels,
        device="cpu",
        seed=7,
        epochs=1,
        patience=1,
        minimum_epochs=1,
        batch_size=4,
        accumulation_steps=2,
        learning_rate=1e-3,
        delay_learning_rate_multiplier=2.0,
        weight_decay=1e-4,
        auxiliary_weights=(0.05, 0.10, 0.20),
        augmentation={"enabled": False},
        run_label="unit",
    )
    prediction = predict_scaffold(
        fit.model,
        rates,
        labels,
        device="cpu",
        batch_size=4,
    )
    assert fit.optimizer_steps == 1
    assert fit.history[0]["epoch"] == 0
    assert fit.best_epoch in {0, 1}
    assert fit.history[0]["delay_lr"] == pytest.approx(
        2.0 * fit.history[0]["lr"]
    )
    assert prediction["logits"].shape == (8, 4)
    assert prediction["endpoint_logits"].shape == (8, 4, 4)
    assert prediction["delay_contrast_rms"] >= 0.0
    assert prediction["route_delay_contrast_rms"] >= 0.0
    assert prediction["delayed_statistics_logits_rms"] >= 0.0
    assert prediction["delayed_statistics_gate_mean"] is None
    assert prediction["diagnostic_synthesized_carrier"] is None
    assert prediction["diagnostic_broadband_current"] is None
    assert np.isfinite(prediction["logits"]).all()


def test_fixed_cutoff_preserves_full_horizon_scheduler_prefix() -> None:
    torch.manual_seed(29)
    fast = torch.complex(torch.randn(8, 2, 3, 500), torch.randn(8, 2, 3, 500))
    slow = torch.randn(8, 2, 3, 250)
    rates = CachedRates(
        fast=fast,
        slow=slow,
        broadband=torch.randn(8, 3, 500),
        context=None,
        gain=torch.ones(2, 3),
        frontend_fingerprint=build_zero_scaffold(
            seed=29,
            model_config=_model_config(),
        ).frontend_fingerprint(),
    )
    labels = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)

    def train(fixed_epoch: int | None):
        model = build_zero_scaffold(seed=29, model_config=_model_config())
        return fit_zero_scaffold(
            model,
            train_rates=rates,
            y_train=labels,
            validation_rates=None,
            y_validation=None,
            device="cpu",
            seed=29,
            epochs=20,
            fixed_epoch=fixed_epoch,
            patience=20,
            minimum_epochs=1,
            batch_size=4,
            accumulation_steps=1,
            learning_rate=1e-3,
            weight_decay=1e-4,
            auxiliary_weights=(0.0, 0.0, 0.0),
            scheduler_warmup_epochs=5,
            augmentation={"enabled": False},
        )

    cutoff = train(5)
    complete = train(None)

    assert len(cutoff.history) == 5
    np.testing.assert_allclose(
        [row["loss"] for row in cutoff.history],
        [row["loss"] for row in complete.history[:5]],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        [row["lr"] for row in cutoff.history],
        [row["lr"] for row in complete.history[:5]],
        rtol=0,
        atol=0,
    )


def test_teacher_conditional_matched_zero_kl_trust_region_is_finite() -> None:
    model = build_zero_scaffold(seed=31, model_config=_model_config())
    configure_delay_stage(model, "static_slow_within", train_readout=False)
    rates = CachedRates(
        fast=torch.complex(
            torch.randn(8, 2, 3, 500),
            torch.randn(8, 2, 3, 500),
        ),
        slow=torch.randn(8, 2, 3, 250),
        broadband=torch.randn(8, 3, 500),
        context=None,
        gain=torch.ones(2, 3),
        frontend_fingerprint=model.frontend_fingerprint(),
    )
    fit = fit_zero_scaffold(
        model,
        train_rates=rates,
        y_train=np.asarray([0, 0, 1, 1, 2, 2, 3, 3]),
        validation_rates=None,
        y_validation=None,
        device="cpu",
        seed=31,
        epochs=1,
        patience=1,
        minimum_epochs=1,
        batch_size=4,
        accumulation_steps=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        auxiliary_weights=(0.0, 0.0, 0.0),
        matched_zero_kl_teacher_correct_weight=4.0,
        matched_zero_kl_teacher_incorrect_weight=1.0,
        augmentation={"enabled": False},
    )

    assert fit.history[0]["matched_zero_kl_teacher_correct_weight"] == 4.0
    assert fit.history[0]["matched_zero_kl_teacher_incorrect_weight"] == 1.0
    assert np.isfinite(fit.history[0]["matched_zero_kl"])
    assert fit.history[0]["matched_zero_kl"] >= 0.0
    assert np.isfinite(fit.history[0]["matched_zero_kl_penalty"])
    assert fit.history[0]["matched_zero_kl_penalty"] >= 0.0


def test_official_fbc_cache_preserves_full_rate_before_mandatory_delay() -> None:
    rng = np.random.default_rng(13)
    raw = (rng.normal(size=(2, 22, 1250)) * 1e-5).astype(np.float32)
    model = build_zero_scaffold(
        seed=13,
        model_config={
            "n_bands": 9,
            "n_nodes": 3,
            "band_edges_hz": tuple((low, low + 4) for low in range(4, 40, 4)),
            "analytic_taps": 33,
            "envelope_taps": 9,
            "fast_decimation": 1,
            "route_rank": 2,
            "delay_rank": 2,
            "temporal_dilations": (1, 2),
            "temporal_depth": 1,
            "use_geometry": False,
            "snn_channels": 8,
            "decoder_layers": 0,
            "statistical_spatial_filters": 3,
            "dropout": 0.0,
        },
    )
    gain = fit_official_fbc_channel_gain(raw, clip=12.0)
    cached = cache_official_fbc_rate_features(
        model,
        raw,
        gain,
        device="cpu",
        batch_size=1,
    )
    assert cached.fast.shape == (2, 9, 3, 1000)
    assert cached.slow.shape == (2, 9, 3, 500)
    assert cached.broadband.shape == (2, 3, 1000)
    assert torch.count_nonzero(cached.slow) == 0
    assert torch.isfinite(cached.fast.real).all()
    normalized = apply_fixed_gain(task_carrier(raw), gain)
    expected_broadband = model.spatial.project_real(torch.from_numpy(normalized))
    torch.testing.assert_close(cached.broadband, expected_broadband)


def test_frontend_only_scaffold_preserves_registered_frontend_fingerprint() -> None:
    config = _model_config()
    reference = build_zero_scaffold(seed=19, model_config=config)
    frontend = build_frontend_only_scaffold(seed=19, model_config=config)

    assert frontend.use_atc_readout is False
    assert frontend.use_statistical_readout is False
    assert frontend.use_geometry is False
    assert frontend.frontend_fingerprint() == reference.frontend_fingerprint()


def test_full_window_atc_adapter_preserves_transport_and_gradients() -> None:
    class Core(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"logits": x.mean(dim=-1)[:, :4]}

    readout = DelayedFullWindowATCReadout(
        Core(),
        n_bands=2,
        n_nodes=3,
        n_classes=4,
        endpoint_samples=(2, 4, 6, 8),
        node_reconstruction=torch.eye(4, 3),
    )
    delayed = torch.randn(2, 3, 8, requires_grad=True)
    output = readout(delayed)
    assert output.logits.shape == (2, 4, 4)
    torch.testing.assert_close(output.synthesized_carrier[:, :3], delayed)
    for endpoint in range(1, 4):
        torch.testing.assert_close(output.logits[:, 0], output.logits[:, endpoint])
    output.logits.square().mean().backward()
    assert delayed.grad is not None and torch.isfinite(delayed.grad).all()


def test_analytic_fixed_gain_cache_keeps_broadband_parity_and_delay_bands() -> None:
    rng = np.random.default_rng(17)
    raw = (rng.normal(size=(2, 22, 1250)) * 1e-5).astype(np.float32)
    model = build_zero_scaffold(seed=17, model_config=_model_config())
    gain = fit_official_fbc_channel_gain(raw, clip=12.0)
    cached = cache_analytic_fixed_channel_gain_rate_features(
        model, raw, gain, device="cpu", batch_size=1
    )
    assert cached.fast.shape == (2, 2, 3, 500)
    assert cached.slow.shape == (2, 2, 3, 250)
    assert torch.count_nonzero(cached.slow) > 0
    normalized = apply_fixed_gain(task_carrier(raw), gain)
    expected = model.spatial.project_real(torch.from_numpy(normalized))
    torch.testing.assert_close(cached.broadband, expected, rtol=1e-5, atol=1e-5)
    aligned = aligned_slow_rate_analytic_evidence(cached)
    assert aligned.shape == cached.slow.shape
    torch.testing.assert_close(aligned, cached.fast[..., ::2])


def test_full_window_atc_delayed_residual_is_zero_initialized_and_mandatory() -> None:
    class Core(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"logits": x.mean(dim=-1)[:, :4]}

    readout = DelayedFullWindowATCReadout(
        Core(),
        n_bands=2,
        n_nodes=4,
        n_classes=4,
        endpoint_samples=(2, 4, 6, 8),
        delayed_residual_bound=0.25,
    )
    broadband = torch.randn(2, 4, 8)
    delayed = torch.randn(2, 2, 4, 8)
    zero = readout(delayed, broadband_carrier=broadband)
    direct = readout(broadband)
    torch.testing.assert_close(zero.logits, direct.logits)
    with torch.no_grad():
        readout.delayed_residual_mix_raw.fill_(0.5)
    active = readout(delayed, broadband_carrier=broadband)
    assert not torch.allclose(active.logits, direct.logits)


def test_full_window_atc_delayed_statistics_are_zero_initialized_and_trainable() -> None:
    class Core(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"logits": x.mean(dim=-1)[:, :4]}

    readout = DelayedFullWindowATCReadout(
        Core(),
        n_bands=2,
        n_nodes=4,
        n_classes=4,
        endpoint_samples=(8,),
        delayed_statistics_bins=2,
    )
    delayed = torch.randn(3, 2, 4, 8, requires_grad=True)
    baseline = readout(delayed)
    direct = readout._synthesize(delayed).mean(dim=-1)[:, :4]
    torch.testing.assert_close(baseline.logits[:, 0], direct, rtol=0, atol=0)
    assert baseline.delayed_statistics_logits is not None
    assert torch.count_nonzero(baseline.delayed_statistics_logits) == 0

    target = torch.tensor([0, 1, 2])
    torch.nn.functional.cross_entropy(baseline.logits[:, 0], target).backward()
    gradient = readout.delayed_statistics_classifier.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_tensor_cp_delayed_statistics_preserve_axes_with_low_capacity() -> None:
    class Core(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"logits": x.mean(dim=-1)[:, :4]}

    readout = DelayedFullWindowATCReadout(
        Core(),
        n_bands=2,
        n_nodes=4,
        n_classes=4,
        endpoint_samples=(8,),
        delayed_statistics_bins=2,
        delayed_statistics_interaction_gain=1.0,
        delayed_statistics_cp_rank=3,
    )
    classifier = readout.delayed_statistics_classifier
    assert classifier is not None
    assert sum(parameter.numel() for parameter in classifier.parameters()) < 132
    assert float(classifier.output_gain) == pytest.approx(np.sqrt(8.0))
    delayed = torch.randn(3, 2, 4, 8)
    contrast = torch.randn_like(delayed)
    initial = readout(delayed, statistics_carrier=contrast)
    assert initial.delayed_statistics_logits is not None
    assert torch.count_nonzero(initial.delayed_statistics_logits) == 0

    with torch.no_grad():
        classifier.class_factor.fill_(0.1)
    active = readout(delayed, statistics_carrier=contrast)
    assert active.delayed_statistics_logits is not None
    assert torch.count_nonzero(active.delayed_statistics_logits) > 0
    active.delayed_statistics_logits.square().mean().backward()
    assert classifier.class_factor.grad is not None
    assert torch.isfinite(classifier.class_factor.grad).all()


def test_full_window_atc_statistics_can_read_interaction_only_current() -> None:
    class Core(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"logits": x.mean(dim=-1)[:, :4]}

    readout = DelayedFullWindowATCReadout(
        Core(),
        n_bands=2,
        n_nodes=4,
        n_classes=4,
        endpoint_samples=(8,),
        delayed_statistics_bins=2,
        delayed_statistics_interaction_gain=1.0,
    )
    delayed = torch.randn(2, 2, 4, 8)
    interaction = torch.randn_like(delayed)
    with torch.no_grad():
        readout.delayed_statistics_classifier.weight.fill_(0.1)
    first = readout(delayed, statistics_carrier=interaction)
    second = readout(delayed, statistics_carrier=torch.zeros_like(interaction))
    torch.testing.assert_close(first.synthesized_carrier, second.synthesized_carrier)
    assert not torch.allclose(first.logits, second.logits)


def test_delayed_statistics_are_exactly_zero_for_zero_contrast() -> None:
    class Core(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"logits": x.mean(dim=-1)[:, :4]}

    readout = DelayedFullWindowATCReadout(
        Core(),
        n_bands=2,
        n_nodes=4,
        n_classes=4,
        endpoint_samples=(8,),
        delayed_statistics_bins=2,
        delayed_statistics_interaction_gain=1.0,
    )
    delayed = torch.randn(2, 2, 4, 8)
    with torch.no_grad():
        readout.delayed_statistics_classifier.weight.fill_(0.1)
        readout.delayed_statistics_classifier.bias.zero_()
    zero = readout(delayed, statistics_carrier=torch.zeros_like(delayed))
    direct = readout._synthesize(delayed).mean(dim=-1)[:, :4]
    torch.testing.assert_close(zero.logits[:, 0], direct, rtol=0, atol=0)
    assert zero.delayed_statistics_logits is not None
    assert torch.count_nonzero(zero.delayed_statistics_logits) == 0


def test_directional_delay_moments_are_shift_sensitive_and_zero_for_zero_contrast() -> None:
    class Core(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"logits": x.mean(dim=-1)[:, :4]}

    readout = DelayedFullWindowATCReadout(
        Core(),
        n_bands=2,
        n_nodes=4,
        n_classes=4,
        endpoint_samples=(16,),
        delayed_statistics_bins=4,
        delayed_statistics_interaction_gain=1.0,
        delayed_statistics_cp_rank=3,
        delayed_statistics_directional_moments=True,
    )
    classifier = readout.delayed_statistics_classifier
    assert classifier is not None
    assert classifier.shape == (4, 2, 4, 4)
    delayed = torch.randn(3, 2, 4, 16)
    contrast = delayed - torch.roll(delayed, shifts=1, dims=-1)
    with torch.no_grad():
        classifier.class_factor.fill_(0.1)
        classifier.bias.zero_()

    active = readout(delayed, statistics_carrier=contrast)
    zero = readout(delayed, statistics_carrier=torch.zeros_like(contrast))

    assert active.delayed_statistics_logits is not None
    assert torch.count_nonzero(active.delayed_statistics_logits) > 0
    assert zero.delayed_statistics_logits is not None
    assert torch.count_nonzero(zero.delayed_statistics_logits) == 0


@pytest.mark.parametrize("cross_band", [False, True])
def test_route_cp_statistics_preserve_edge_identity_and_zero_control(
    cross_band: bool,
) -> None:
    class Core(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"logits": x.mean(dim=-1)[:, :4]}

    readout = DelayedFullWindowATCReadout(
        Core(),
        n_bands=2,
        n_nodes=4,
        n_classes=4,
        endpoint_samples=(16,),
        delayed_route_statistics_bins=4,
        delayed_route_statistics_cp_rank=3,
    )
    classifier = readout.delayed_route_statistics_classifier
    assert classifier is not None
    delayed = torch.randn(3, 2, 4, 16)
    route_shape = (3, 2, 2, 4, 4, 8) if cross_band else (3, 2, 4, 4, 8)
    route_contrast = torch.randn(route_shape)
    with torch.no_grad():
        classifier.class_factor.fill_(0.1)
        classifier.bias.zero_()

    active = readout(delayed, route_statistics_carrier=route_contrast)
    zero = readout(delayed, route_statistics_carrier=None)

    assert active.delayed_route_statistics_logits is not None
    assert torch.count_nonzero(active.delayed_route_statistics_logits) > 0
    assert zero.delayed_route_statistics_logits is not None
    assert torch.count_nonzero(zero.delayed_route_statistics_logits) == 0
    active.delayed_route_statistics_logits.square().mean().backward()
    assert classifier.class_factor.grad is not None
    assert torch.isfinite(classifier.class_factor.grad).all()


def test_uncertainty_gate_suppresses_residual_for_high_margin_base_logits() -> None:
    class Core(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"logits": x[:, :4, 0]}

    readout = DelayedFullWindowATCReadout(
        Core(),
        n_bands=2,
        n_nodes=4,
        n_classes=4,
        endpoint_samples=(8,),
        delayed_statistics_bins=2,
        delayed_statistics_interaction_gain=1.0,
        delayed_statistics_uncertainty_gate=True,
    )
    delayed = torch.zeros(2, 2, 4, 8)
    delayed[0, 0, :, 0] = torch.tensor([5.0, 0.0, 0.0, 0.0])
    delayed[1, 0, :, 0] = torch.tensor([0.3, 0.2, 0.1, 0.0])
    contrast = torch.ones_like(delayed)
    with torch.no_grad():
        readout.delayed_statistics_classifier.weight.fill_(0.1)
    output = readout(delayed, statistics_carrier=contrast)
    assert output.delayed_statistics_gate is not None
    assert output.delayed_statistics_gate[0] < output.delayed_statistics_gate[1]


def test_paired_reconstruction_noise_changes_only_broadband_noise_path() -> None:
    torch.manual_seed(19)
    fast = torch.ones(4, 2, 3, 8, dtype=torch.complex64)
    slow = torch.ones(4, 2, 3, 4)
    broadband = torch.ones(4, 3, 8)
    labels = torch.tensor([0, 0, 1, 1])
    clean = paired_rate_reconstruction(
        fast,
        slow,
        broadband,
        labels,
        probability=1.0,
        carrier_scale_range=(1.0, 1.0),
        noise_std=0.0,
    )
    torch.manual_seed(19)
    noisy = paired_rate_reconstruction(
        fast,
        slow,
        broadband,
        labels,
        probability=1.0,
        carrier_scale_range=(1.0, 1.0),
        noise_std=0.01,
    )
    torch.testing.assert_close(clean[0], noisy[0])
    torch.testing.assert_close(clean[1], noisy[1])
    assert not torch.allclose(clean[2], noisy[2])


def test_full_window_atc_amplitude_modulation_preserves_zero_scale() -> None:
    class Core(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"logits": x.mean(dim=-1)[:, :4]}

    readout = DelayedFullWindowATCReadout(
        Core(),
        n_bands=2,
        n_nodes=4,
        n_classes=4,
        endpoint_samples=(8,),
        delayed_residual_bound=0.25,
        delayed_fusion_mode="amplitude_modulation",
    )
    broadband = torch.randn(2, 4, 8)
    delayed = torch.randn(2, 2, 4, 8)
    zero = readout(delayed, broadband_carrier=broadband)
    torch.testing.assert_close(zero.synthesized_carrier, broadband)
    with torch.no_grad():
        readout.delayed_residual_mix_raw.fill_(0.5)
    active = readout(delayed, broadband_carrier=broadband)
    assert torch.isfinite(active.synthesized_carrier).all()
    assert not torch.allclose(active.synthesized_carrier, broadband)
    assert bool(
        (
            (active.synthesized_carrier - broadband).abs()
            <= 0.25 * broadband.abs() + 1e-6
        ).all()
    )
