from __future__ import annotations

from pathlib import Path

import pytest
import numpy as np

torch = pytest.importorskip("torch")

from dpc_snn.models.event_encoder import (
    CausalSpikeAccumulator,
    PhaseDeltaEventEncoder,
    fractional_delay_events,
)
from dpc_snn.models.dpc_snn import DPCSNN
from dpc_snn.models.delay_phase_synapse import DelayPhaseGraphSynapse
from dpc_snn.models.lif import MultiTimescaleSEWBlock, MultiTimescaleStateLayer
from dpc_snn.config import load_yaml
from dpc_snn.training.train import _fit_training_event_thresholds, make_loader


def _v5_model(**overrides) -> DPCSNN:
    kwargs = {
        "n_classes": 2,
        "n_channels": 3,
        "n_bands": 3,
        "hidden_channels": 4,
        "timesteps": 32,
        "graph_timesteps": 32,
        "sfreq": 64.0,
        "task_tmin": 0.0,
        "task_tmax": 1.0,
        "latent_nodes": 3,
        "snn_channels": 6,
        "temporal_pool_channels": 3,
        "d_max": 2,
        "slow_d_max": 2,
        "band_edges_hz": [[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]],
        "delay_evidence_band_edges_hz": [
            [2.0, 4.0],
            [4.0, 6.0],
            [6.0, 8.0],
        ],
        "classification_band_edges_hz": [
            [2.0, 4.0],
            [4.0, 6.0],
            [6.0, 8.0],
        ],
        "decoder_layers": 2,
        "n_delay_experts": 1,
        "delay_expert_topk": 1,
        "freeze_shared_physical_basis": True,
        "freeze_shared_filterbank": True,
        "freeze_delay_after_pretrain": True,
        "preserve_transport_band_pairs": True,
        "delayed_stat_channels": 0,
        "delayed_directional_moments": False,
        "event_native_transport": True,
        "multiscale_snn": True,
        "snn_native_readout": True,
        "use_membrane_readout": False,
        "architecture_version": "dpc_snn_v5_event_native",
    }
    kwargs.update(overrides)
    return DPCSNN(**kwargs)


def test_phase_and_envelope_events_are_signed_and_sparse() -> None:
    encoder = PhaseDeltaEventEncoder(
        n_bands=1,
        n_nodes=1,
        timesteps=5,
        envelope_threshold=0.1,
    )
    carrier = torch.tensor([[[[-1.0, -0.5, 0.5, 1.0, -1.0]]]])
    envelope = torch.tensor([[[[0.0, 0.0, 0.2, 0.2, -0.1]]]])
    confidence = torch.ones_like(carrier)

    encoded = encoder(carrier, envelope, confidence)

    torch.testing.assert_close(
        encoded["phase_events"],
        torch.tensor([[[[0.0, 0.0, 1.0, 0.0, -1.0]]]]),
    )
    torch.testing.assert_close(
        encoded["envelope_events"],
        torch.tensor([[[[0.0, 0.0, 1.0, 0.0, -1.0]]]]),
    )
    assert float(encoded["phase_event_density"]) == pytest.approx(0.4)
    assert float(encoded["envelope_event_density"]) == pytest.approx(0.4)


def test_fractional_event_delay_is_a_causal_two_tap_synapse() -> None:
    events = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])

    delayed = fractional_delay_events(events, 1.5)

    torch.testing.assert_close(
        delayed, torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.0]])
    )
    assert float(delayed.sum()) == pytest.approx(float(events.sum()))


def test_timing_shuffle_preserves_signed_event_counts() -> None:
    plain = PhaseDeltaEventEncoder(1, 1, 8, timing_intervention="none")
    shuffled = PhaseDeltaEventEncoder(
        1, 1, 8, timing_intervention="shuffle", timing_seed=11
    )
    carrier = torch.tensor(
        [[[[1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0]]]]
    )
    envelope = torch.zeros_like(carrier)
    confidence = torch.ones_like(carrier)

    original = plain(carrier, envelope, confidence)["phase_events"]
    intervention = shuffled(carrier, envelope, confidence)["phase_events"]

    torch.testing.assert_close(original.sum(-1), intervention.sum(-1))
    torch.testing.assert_close(original.abs().sum(-1), intervention.abs().sum(-1))
    assert not torch.equal(original, intervention)


def test_spike_accumulator_prefix_is_strictly_causal() -> None:
    accumulator = CausalSpikeAccumulator(2, 3, decays=(0.5, 0.9))
    first = torch.zeros(1, 2, 8)
    second = first.clone()
    first[..., 1] = 1.0
    second[..., 1] = 1.0
    second[..., 6] = 1.0

    _, first_prefix = accumulator(first, capture_steps=(4,))
    _, second_prefix = accumulator(second, capture_steps=(4,))

    torch.testing.assert_close(first_prefix, second_prefix)


def test_multiscale_clif_uses_distinct_timescales_and_binary_output() -> None:
    layer = MultiTimescaleStateLayer(
        channels=8,
        decays=(0.6, 0.9, 0.98),
        threshold=0.5,
        neuron_type="clif",
    )

    spikes, membrane = layer(torch.rand(2, 8, 12))

    assert spikes.shape == membrane.shape == (2, 8, 12)
    assert torch.all((spikes == 0) | (spikes == 1))
    recovered = torch.cat([group.decay.detach() for group in layer.groups])
    assert recovered.min() < 0.7
    assert recovered.max() > 0.95


def test_ann_control_matches_multiscale_snn_trainable_parameter_count() -> None:
    snn = MultiTimescaleSEWBlock(8, neuron_type="clif")
    ann = MultiTimescaleSEWBlock(8, neuron_type="ann")

    assert sum(parameter.numel() for parameter in snn.parameters()) == sum(
        parameter.numel() for parameter in ann.parameters()
    )


def test_event_synapse_has_no_undelayed_target_envelope_bypass() -> None:
    synapse = DelayPhaseGraphSynapse(
        n_bands=2,
        n_channels=2,
        d_max=1,
        slow_d_max=1,
        n_delay_experts=1,
        expert_topk=1,
        fixed_delay_steps=1.0,
        event_native_transport=True,
    ).eval()
    route = torch.zeros_like(synapse.route_mask)
    route[1, 0, 0, 1] = 1.0
    synapse.set_edge_weight_override(route)
    carrier = torch.zeros(1, 2, 2, 8)
    envelope = torch.zeros_like(carrier)
    envelope[:, 1, 0, 3] = 1.0
    phase = torch.zeros_like(carrier)
    confidence = torch.ones_like(carrier)

    target_only, _ = synapse(carrier, phase, confidence, envelope=envelope)
    envelope[:, 0, 1, 1] = 1.0
    with_delayed_source, _ = synapse(
        carrier, phase, confidence, envelope=envelope
    )

    torch.testing.assert_close(target_only, torch.zeros_like(target_only))
    assert with_delayed_source.abs().sum() > 0


def test_event_route_checkpointing_preserves_forward_and_gradients() -> None:
    kwargs = {
        "n_bands": 2,
        "n_channels": 2,
        "d_max": 1,
        "slow_d_max": 1,
        "n_delay_experts": 1,
        "expert_topk": 1,
        "fixed_delay_steps": 0.5,
        "event_native_transport": True,
        "matched_transport_control": True,
    }
    plain = DelayPhaseGraphSynapse(**kwargs, checkpoint_event_routes=False).train()
    rematerialized = DelayPhaseGraphSynapse(
        **kwargs, checkpoint_event_routes=True
    ).train()
    rematerialized.load_state_dict(plain.state_dict())
    carrier_plain = torch.randn(2, 2, 2, 12, requires_grad=True)
    carrier_checkpoint = carrier_plain.detach().clone().requires_grad_(True)
    envelope = torch.randn(2, 2, 2, 12)
    phase = torch.randn(2, 2, 2, 12)
    confidence = torch.sigmoid(torch.randn(2, 2, 2, 12))

    plain_current, _ = plain(
        carrier_plain, phase, confidence, envelope=envelope
    )
    checkpoint_current, _ = rematerialized(
        carrier_checkpoint, phase, confidence, envelope=envelope
    )
    plain_current.square().mean().backward()
    checkpoint_current.square().mean().backward()

    torch.testing.assert_close(plain_current, checkpoint_current)
    torch.testing.assert_close(carrier_plain.grad, carrier_checkpoint.grad)
    torch.testing.assert_close(
        plain.edge_logits.grad, rematerialized.edge_logits.grad
    )


def test_v5_native_readout_has_no_route_energy_or_temporal_stat_feature() -> None:
    model = _v5_model().eval()

    output = model(x=torch.randn(2, 3, 64))

    assert model.spike_accumulator is not None
    assert model.readout.in_features == model.spike_accumulator.out_features
    assert output["aux"]["temporal_pyramid_bins"].numel() == 0
    assert bool(output["aux"]["event_native_transport"])
    assert bool(output["aux"]["snn_native_readout"])
    assert bool(output["aux"]["decoder_is_spiking"])
    spikes = output["aux"]["binary_spike_layers"]
    assert torch.all((spikes == 0) | (spikes == 1))
    assert torch.isfinite(output["logits"]).all()


def test_v5_all_classification_activity_vanishes_when_delay_edges_are_zero() -> None:
    model = _v5_model().eval()
    model.synapse.set_edge_weight_override(
        torch.zeros_like(model.synapse.route_mask)
    )

    first = model(x=torch.randn(2, 3, 64))
    second = model(x=10.0 * torch.randn(2, 3, 64))

    torch.testing.assert_close(first["aux"]["current"], torch.zeros_like(first["aux"]["current"]))
    torch.testing.assert_close(first["logits"], second["logits"])


def test_v5_matched_ann_and_snn_have_equal_parameter_capacity() -> None:
    snn = _v5_model(decoder_mode="snn")
    ann = _v5_model(decoder_mode="matched_ann")

    assert sum(parameter.numel() for parameter in snn.parameters()) == sum(
        parameter.numel() for parameter in ann.parameters()
    )


def test_v5_contract_rejects_dense_temporal_statistics() -> None:
    with pytest.raises(ValueError, match="broadcast delayed statistics"):
        _v5_model(delayed_stat_channels=1)


def test_v5_protocol_controls_change_only_declared_mechanism() -> None:
    protocol = load_yaml(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiments"
        / "v50_event_native_gate.yaml"
    )
    paired = {row["name"]: row for row in protocol["paired_gate_variants"]}
    fixed = paired["fixed_audited_delay"]
    zero = paired["matched_zero_delay"]
    ignored = {"name", "fixed_delay_steps"}
    assert {key: value for key, value in fixed.items() if key not in ignored} == {
        key: value for key, value in zero.items() if key not in ignored
    }
    assert fixed["fixed_delay_steps"] is None
    assert zero["fixed_delay_steps"] == 0.0

    stages = {row["name"]: row for row in protocol["five_stages"]}
    ann = stages["matched_ann_dynamics"]
    timing = stages["event_timing_shuffle"]
    assert ann["decoder_mode"] == "matched_ann"
    assert ann["event_timing_intervention"] == "none"
    assert timing["decoder_mode"] == "snn"
    assert timing["event_timing_intervention"] == "shuffle"
    assert ann["reuse_audited_route_gain"]
    assert timing["reuse_audited_route_gain"]


def test_event_threshold_is_fitted_only_from_supplied_training_loader() -> None:
    rng = np.random.default_rng(101)
    x = rng.normal(size=(8, 3, 64)).astype(np.float32)
    y = np.arange(8, dtype=np.int64) % 2
    loader = make_loader(x, y, None, None, batch_size=2, shuffle=False)
    model = _v5_model().eval()

    thresholds = _fit_training_event_thresholds(
        model, loader, "cpu", quantile=0.90, max_samples_per_band=10_000
    )
    output = model(x=torch.from_numpy(x[:2]))

    assert len(thresholds) == 3
    assert all(value > 0 for value in thresholds)
    assert bool(model.event_encoder.envelope_threshold_ready)
    density = float(output["aux"]["envelope_event_density"])
    assert 0.02 <= density <= 0.25


def test_representation_checkpoint_cannot_replace_fold_event_thresholds() -> None:
    source = _v5_model()
    target = _v5_model()
    source.set_train_fitted_event_thresholds(torch.tensor([0.01, 0.02, 0.03]))
    target.set_train_fitted_event_thresholds(torch.tensor([0.11, 0.12, 0.13]))
    expected = target.event_encoder.envelope_threshold.detach().clone()

    target.load_representation_state(source.state_dict())

    torch.testing.assert_close(target.event_encoder.envelope_threshold, expected)
    assert bool(target.event_encoder.envelope_threshold_ready)
