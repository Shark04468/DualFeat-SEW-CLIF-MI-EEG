from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.models.dasp_snn_v62 import DASPSNNV62  # noqa: E402
from dpc_snn.models.delayed_causal_atc_readout import (  # noqa: E402
    DelayedCausalATCReadout,
)
from dpc_snn.models.delayed_geometry_sketch import (  # noqa: E402
    CausalDelayedGeometryFiLM,
)
from dpc_snn.models.delayed_temporal_pyramid import (  # noqa: E402
    DelayedTemporalPyramid,
)
from dpc_snn.models.delayed_statistical_readout import (  # noqa: E402
    DelayedStatisticalReadout,
)
from dpc_snn.models.v62_filterbank import (  # noqa: E402
    CausalAnalyticFilterBank,
    DualRateCausalResampler,
    causal_linear_upsample_2x,
)
from dpc_snn.models.v62_snn_decoder import HeterogeneousANN, V62SNNDecoder  # noqa: E402
from dpc_snn.models.v62_spatial import (  # noqa: E402
    BCI2A_CHANNEL_NAMES,
    CoupledSpatialBasis,
)


def test_causal_filterbank_rejects_future_influence_and_preserves_positive_band() -> None:
    torch.manual_seed(11)
    bank = CausalAnalyticFilterBank(((6.0, 10.0), (34.0, 40.0)), taps=65)
    first = torch.randn(1, 2, 400)
    second = first.clone()
    second[..., 240:] += 20.0 * torch.randn_like(second[..., 240:])

    out_first = bank(first)
    out_second = bank(second)

    torch.testing.assert_close(out_first[..., :240], out_second[..., :240], rtol=0, atol=0)
    response = bank.frequency_response(torch.tensor([8.0, 37.0, -8.0, -37.0]))
    assert float(response[0, 0].abs()) > 1.5
    assert float(response[1, 1].abs()) > 1.5
    assert float(response[0, 2].abs()) < 0.2
    assert float(response[1, 3].abs()) < 0.2
    assert bank.support_seconds <= 1.0


def test_dual_rate_timestamps_and_causal_interpolation() -> None:
    projected = torch.ones(1, 2, 3, 1250, dtype=torch.complex64)
    resampler = DualRateCausalResampler(
        2, 3, analytic_group_delay_samples=32, envelope_taps=17
    )
    features = resampler(projected)

    assert features.fast.shape == (1, 2, 3, 500)
    assert features.slow.shape == (1, 2, 3, 250)
    torch.testing.assert_close(features.fast_timestamps[::2], features.slow_timestamps)
    torch.testing.assert_close(features.fast_availability[::2], features.slow_availability)

    constant = torch.ones(1, 1, 5)
    torch.testing.assert_close(causal_linear_upsample_2x(constant), torch.ones(1, 1, 10))
    ramp = torch.arange(5.0).view(1, 1, 5)
    expected = torch.tensor([0.0, 0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    torch.testing.assert_close(causal_linear_upsample_2x(ramp).flatten(), expected)

    full_rate = DualRateCausalResampler(
        2,
        3,
        analytic_group_delay_samples=32,
        envelope_taps=17,
        fast_decimation=1,
    )(projected)
    assert full_rate.fast.shape[-1] == 1000
    assert full_rate.slow.shape[-1] == 500
    torch.testing.assert_close(full_rate.fast_timestamps[::2], full_rate.slow_timestamps)


def test_v8_gain_invariant_envelope_is_exact_near_epsilon() -> None:
    torch.manual_seed(17)
    projected = torch.complex(
        torch.randn(3, 2, 4, 80) * 1.0e-7,
        torch.randn(3, 2, 4, 80) * 1.0e-7,
    )
    resampler = DualRateCausalResampler(
        2,
        4,
        sfreq=32,
        analytic_group_delay_samples=0,
        envelope_cutoff_hz=6,
        envelope_taps=9,
        fast_decimation=2,
        gain_invariant_envelope=True,
    )
    resampler.set_training_gain(torch.ones(2, 4))
    unit = resampler(projected, epoch_tmin=-0.5, task_tmin=0.0, task_tmax=1.0)
    gain = torch.tensor(
        [[0.25, 2.0, 8.0, 32.0], [64.0, 4.0, 0.5, 16.0]],
        dtype=torch.float32,
    )
    resampler.set_training_gain(gain)
    scaled = resampler(projected, epoch_tmin=-0.5, task_tmin=0.0, task_tmax=1.0)

    torch.testing.assert_close(
        scaled.fast,
        unit.fast * gain[None, :, :, None],
        rtol=1.0e-6,
        atol=1.0e-12,
    )
    torch.testing.assert_close(scaled.slow, unit.slow, rtol=0.0, atol=0.0)

def test_exact_sensor_basis_is_an_orthogonal_physical_channel_permutation() -> None:
    basis = CoupledSpatialBasis(
        BCI2A_CHANNEL_NAMES,
        n_bands=2,
        n_nodes=22,
        exact_sensor_basis=True,
        trainable=False,
    )
    anchors = basis.anchors
    torch.testing.assert_close(anchors @ anchors.T, torch.eye(22))
    assert torch.equal(anchors.sum(dim=0), torch.ones(22))
    assert torch.equal(anchors.sum(dim=1), torch.ones(22))


def test_frozen_full_node_basis_has_a_stable_sensor_reconstruction() -> None:
    basis = CoupledSpatialBasis(
        BCI2A_CHANNEL_NAMES,
        n_bands=3,
        n_nodes=22,
        anchor_sigma=0.12,
        trainable=False,
    )
    weight = basis.weight()
    singular_values = torch.linalg.svdvals(weight)
    assert float((singular_values[..., 0] / singular_values[..., -1]).amax()) < 2.0
    reconstruction = torch.linalg.pinv(weight)
    identity = reconstruction @ weight
    torch.testing.assert_close(
        identity,
        torch.eye(22)[None].expand_as(identity),
        rtol=1e-5,
        atol=1e-5,
    )


def test_temporal_pyramid_and_geometry_are_causal_and_zero_neutral() -> None:
    torch.manual_seed(13)
    delayed = torch.randn(2, 2, 3, 50, requires_grad=True)
    changed = delayed.detach().clone()
    changed[..., 30:] += 10.0
    pyramid = DelayedTemporalPyramid(
        n_bands=2, n_nodes=3, dilations=(1, 2), depth=2
    )
    first, _ = pyramid(delayed)
    second, _ = pyramid(changed)
    torch.testing.assert_close(first[..., :30], second[..., :30])

    geometry = CausalDelayedGeometryFiLM(
        n_bands=2,
        n_nodes=3,
        output_channels=4,
        sfreq=20.0,
        windows_seconds=(0.10, 0.20),
        covariance_components=2,
    )
    scale_first, geometry_first = geometry(delayed)
    scale_second, _ = geometry(changed)
    torch.testing.assert_close(scale_first[..., :30], scale_second[..., :30])
    zero_scale, zero_features = geometry(torch.zeros_like(delayed))
    torch.testing.assert_close(zero_scale, torch.ones_like(zero_scale))
    assert torch.count_nonzero(zero_features) == 0
    (first.mean() + scale_first.mean()).backward()
    assert delayed.grad is not None and torch.isfinite(delayed.grad).all()


def test_post_delay_statistical_readout_is_causal_and_zero_neutral() -> None:
    torch.manual_seed(15)
    readout = DelayedStatisticalReadout(
        2,
        3,
        spatial_filters=2,
        endpoint_samples=(5, 10),
        output_features=8,
        dropout=0.0,
    ).eval()
    delayed = torch.randn(2, 2, 3, 10)
    changed = delayed.clone()
    changed[..., 5:] += 100.0
    first = readout(delayed)
    second = readout(changed)
    torch.testing.assert_close(first.encoded[:, 0], second.encoded[:, 0], rtol=0, atol=0)

    zero = readout(torch.zeros_like(delayed))
    assert torch.count_nonzero(zero.raw_statistics) == 0
    assert torch.count_nonzero(zero.encoded) == 0
    assert torch.isfinite(readout.orthogonality_loss())


def test_fbc_style_delayed_log_variance_preserves_four_temporal_segments() -> None:
    torch.manual_seed(16)
    readout = DelayedStatisticalReadout(
        2,
        3,
        spatial_filters=4,
        endpoint_samples=(8,),
        components=("carrier_log_variance",),
        encoder_kind="identity",
        temporal_segments=4,
        variance_transform="log",
        dropout=0.0,
    ).eval()
    delayed = torch.randn(2, 2, 3, 8)
    output = readout(delayed, fast_current=delayed, slow_current=delayed[..., ::2])
    assert output.raw_statistics.shape == (2, 1, 2 * 4 * 4)
    assert output.encoded.shape == output.raw_statistics.shape
    zero = readout(
        torch.zeros_like(delayed),
        fast_current=torch.zeros_like(delayed),
        slow_current=torch.zeros_like(delayed[..., ::2]),
    )
    assert torch.count_nonzero(zero.encoded) == 0


def test_delayed_causal_atc_readout_is_prefix_causal_and_differentiable() -> None:
    torch.manual_seed(18)
    readout = DelayedCausalATCReadout(
        2,
        3,
        4,
        endpoint_samples=(20, 40),
        temporal_filters=4,
        depth_multiplier=2,
        temporal_kernel=5,
        first_pool=2,
        second_pool=2,
        convolution_dropout=0.0,
        key_features=2,
        attention_heads=2,
        attention_dropout=0.0,
        tcn_depth=1,
        tcn_kernel=3,
        tcn_dropout=0.0,
        windows=3,
    ).eval()
    delayed = torch.randn(2, 2, 3, 40, requires_grad=True)
    changed = delayed.detach().clone()
    changed[..., 20:] += 100.0
    first = readout(delayed)
    second = readout(changed)

    assert first.logits.shape == (2, 2, 4)
    torch.testing.assert_close(first.logits[:, 0], second.logits[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(
        first.synthesized_carrier,
        delayed.sum(dim=1) / (2.0**0.5),
    )
    first.logits.square().mean().backward()
    assert delayed.grad is not None and torch.isfinite(delayed.grad).all()

    zero = readout(torch.zeros_like(delayed.detach()))
    assert torch.count_nonzero(zero.synthesized_carrier) == 0
    assert torch.count_nonzero(zero.logits) == 0


def test_frontend_fingerprint_ignores_post_delay_architecture_revision() -> None:
    model = DASPSNNV62(
        n_bands=2,
        n_nodes=3,
        band_edges_hz=((8.0, 12.0), (18.0, 24.0)),
        analytic_taps=33,
        envelope_taps=9,
        temporal_dilations=(1,),
        temporal_depth=1,
        use_geometry=False,
        snn_channels=8,
        decoder_layers=0,
        use_statistical_readout=False,
        dropout=0.0,
    )
    frontend = model.frontend_fingerprint()
    model.architecture_version = "post_delay_only_revision"

    assert model.frontend_fingerprint() == frontend
    model.frontend_architecture_version = "changed_frontend"
    assert model.frontend_fingerprint() != frontend


def test_model_atc_branch_reads_only_the_delayed_fast_transport() -> None:
    torch.manual_seed(20)
    model = DASPSNNV62(
        n_bands=2,
        n_nodes=3,
        band_edges_hz=((1.0, 2.0), (3.0, 5.0)),
        sfreq=16.0,
        fast_decimation=1,
        route_rank=2,
        delay_rank=2,
        temporal_dilations=(1,),
        temporal_depth=1,
        use_geometry=False,
        snn_channels=8,
        decoder_kind="ann",
        decoder_layers=0,
        dropout=0.0,
        use_statistical_readout=False,
        use_atc_readout=True,
        atc_reconstruct_sensors=True,
        atc_temporal_filters=4,
        atc_depth_multiplier=2,
        atc_temporal_kernel=5,
        atc_first_pool=2,
        atc_second_pool=2,
        atc_convolution_dropout=0.0,
        atc_key_features=2,
        atc_attention_heads=2,
        atc_attention_dropout=0.0,
        atc_tcn_depth=1,
        atc_tcn_kernel=3,
        atc_tcn_dropout=0.0,
        atc_windows=3,
    ).eval()
    fused = torch.randn(2, 2, 3, 64)
    fast = torch.randn(2, 2, 3, 64)
    slow = torch.randn(2, 2, 3, 32)
    broadband = torch.randn(2, 3, 64)
    _, auxiliary = model.decode_delayed_current(
        fused,
        fast_current=fast,
        slow_current=slow,
        broadband_current=broadband,
    )
    expected = torch.einsum(
        "ck,nkt->nct",
        model.atc_readout.node_reconstruction,
        broadband,
    )
    torch.testing.assert_close(auxiliary["delayed_atc_synthesized_carrier"], expected)


def test_decoder_counts_only_binary_threshold_outputs() -> None:
    torch.manual_seed(17)
    decoder = V62SNNDecoder(
        n_bands=2,
        n_nodes=3,
        n_classes=4,
        snn_channels=12,
        decoder_layers=2,
        sfreq=10.0,
        endpoint_seconds=(1.0, 2.0, 3.0, 4.0),
        dropout=0.0,
    )
    output = decoder(torch.randn(2, 2, 3, 40))

    assert output.logits.shape == (2, 4, 4)
    assert len(output.binary_spikes) == 4
    assert len(output.residual_activities) == 2
    for spikes in output.binary_spikes:
        assert bool(((spikes == 0) | (spikes == 1)).all())
    assert torch.isfinite(output.firing_rate_loss)


def test_vectorized_ann_state_matches_causal_recurrence() -> None:
    torch.manual_seed(23)
    layer = HeterogeneousANN(channels=5, decays=(0.65, 0.9, 0.975))
    current = torch.randn(3, 5, 17)
    activity, state = layer(current)
    expected = []
    running = torch.zeros_like(current[..., 0])
    for step in range(current.shape[-1]):
        running = layer.decay[None] * running + current[..., step]
        expected.append(running)
    expected_state = torch.stack(expected, dim=-1)
    torch.testing.assert_close(state, expected_state, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(activity, torch.tanh(expected_state), rtol=1e-5, atol=1e-6)


def test_cached_rate_forward_matches_full_frontend_and_cannot_bypass_delay() -> None:
    torch.manual_seed(29)
    model = DASPSNNV62(
        decoder_kind="ann",
        dropout=0.0,
        force_zero_delay=True,
    ).eval()
    eeg = torch.randn(1, 22, 1250) * 1e-5

    with torch.no_grad():
        full = model(eeg)
        prepared, context = model._prepare_raw(eeg)
        analytic = model.filterbank(prepared)
        projected = model.spatial(analytic)
        rates = model.resampler(
            projected,
            epoch_tmin=model.epoch_tmin,
            task_tmin=model.task_tmin,
            task_tmax=model.task_tmax,
        )
        cached = model.forward_rate_features(
            rates.fast,
            rates.slow,
            context=context,
        )

    torch.testing.assert_close(cached["logits"], full["logits"], rtol=0, atol=0)
    torch.testing.assert_close(
        cached["prefix_logits"], full["prefix_logits"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        cached["aux"]["transport"].fused_current,
        full["aux"]["transport"].fused_current,
        rtol=0,
        atol=0,
    )
    assert cached["aux"]["transport"].slow_delay_override == "zero"
    assert cached["aux"]["transport"].fast_delay_override == "zero"

    zero_cached = model.forward_rate_features(
        torch.zeros_like(rates.fast),
        torch.zeros_like(rates.slow),
        context=context,
    )
    expected_bias = model.decoder.classifier.bias[None].expand_as(zero_cached["logits"])
    torch.testing.assert_close(zero_cached["logits"], expected_bias, rtol=0, atol=0)


def test_phase_residual_is_frozen_until_the_registered_stage() -> None:
    fixed = DASPSNNV62(dropout=0.0)
    enabled = DASPSNNV62(dropout=0.0, phase_residual_enabled=True)
    assert not fixed.delay.phase_preference.requires_grad
    assert enabled.delay.phase_preference.requires_grad


def test_dynamic_context_uses_trainable_baseline_features_not_metadata() -> None:
    torch.manual_seed(31)
    model = DASPSNNV62(
        dropout=0.0,
        dynamic_context_features=4,
        decoder_kind="ann",
        force_zero_delay=False,
        n_bands=2,
        n_nodes=3,
        band_edges_hz=((8.0, 12.0), (18.0, 24.0)),
        analytic_taps=33,
        envelope_taps=9,
        route_rank=2,
        delay_rank=2,
        temporal_dilations=(1, 2),
        temporal_depth=1,
        use_geometry=False,
        snn_channels=8,
        decoder_layers=0,
        statistical_spatial_filters=3,
    )
    fast = torch.complex(
        torch.randn(2, 2, 3, 500), torch.randn(2, 2, 3, 500)
    )
    slow = torch.randn(2, 2, 3, 250)
    baseline_statistics = torch.randn(2, 44, requires_grad=True)
    output = model.forward_rate_features(fast, slow, context=baseline_statistics)
    output["logits"].square().mean().backward()

    assert baseline_statistics.grad is not None
    assert model.context_encoder[0].weight.grad is not None


def test_end_to_end_capacity_bottleneck_and_causal_prefix() -> None:
    torch.manual_seed(19)
    model = DASPSNNV62(dropout=0.0).eval()
    assert 145_000 <= model.parameter_count <= 165_000
    assert model.parameter_count < model.parameter_ceiling

    zero = torch.zeros(2, 12, 16, 500)
    decoded, _ = model.decode_delayed_current(zero)
    expected_bias = model.decoder.classifier.bias[None, None].expand_as(decoded.logits)
    torch.testing.assert_close(decoded.logits, expected_bias)

    eeg = torch.randn(1, 22, 1250) * 1e-5
    future_changed = eeg.clone()
    future_changed[..., 500:] += torch.randn_like(future_changed[..., 500:])
    with torch.no_grad():
        first = model(eeg)
        second = model(future_changed)
    torch.testing.assert_close(
        first["prefix_logits"][:, 0], second["prefix_logits"][:, 0], rtol=0, atol=1e-6
    )
    assert first["spikes"].shape == (1, 64, 500)
    assert bool(torch.isfinite(first["logits"]).all())
    assert model.delay.last_max_intermediate_elements <= 1 * 12 * 16 * 500
