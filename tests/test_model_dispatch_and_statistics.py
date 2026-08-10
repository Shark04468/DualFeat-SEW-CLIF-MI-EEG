from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dpc_snn.analysis.statistics import mixed_effects_or_fallback, paired_differences
from dpc_snn.analysis.fractional_delay import recover_fractional_delays
from dpc_snn.experiments.runners import _fair_hparam_candidates
from dpc_snn.models.dpc_snn import DPCSNN
from dpc_snn.models.delay_phase_synapse import DelayPhaseGraphSynapse
from dpc_snn.models.eegnet import EEGNet, _max_norm_weight
from dpc_snn.models.encoder import resize_phase
from dpc_snn.models.eeg_frontend import (
    AnchoredSpatialProjection,
    BandLogCovarianceBranch,
    BandSpecificAnchoredSpatialProjection,
    DelayedDirectionalMoments,
    LearnableAnalyticFilterBank,
    _stable_log_covariance,
)
from dpc_snn.models.lif import CLIFLayer, CausalChannelNorm, SEWCLIFBlock
from dpc_snn.models.hurdle_delay import hurdle_delay_posterior
from dpc_snn.preprocessing.standardize import euclidean_alignment_matrix
from dpc_snn.training.forward import forward_batch
from dpc_snn.training.latency import build_prefix_feature_batch
from dpc_snn.training.train import train_model
from dpc_snn.utils.metrics import (
    classification_metrics,
    exact_mcnemar_p,
    paired_prediction_comparison,
)


def test_dpc_snn_maps_all_transport_bands_only_after_delay() -> None:
    model = DPCSNN(
        n_classes=2,
        n_channels=3,
        n_bands=3,
        hidden_channels=4,
        timesteps=32,
        graph_timesteps=32,
        sfreq=64.0,
        task_tmin=0.0,
        task_tmax=1.0,
        latent_nodes=3,
        snn_channels=4,
        temporal_pool_channels=2,
        temporal_pool_bins=4,
        d_max=2,
        slow_d_max=2,
        band_edges_hz=[[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]],
        delay_evidence_band_edges_hz=[[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]],
        classification_band_edges_hz=[[2.0, 5.0], [5.0, 8.0]],
        decoder_layers=1,
        n_delay_experts=2,
        delay_expert_topk=1,
    )
    mapping = model.classification_band_map()
    assert mapping.shape == (2, 3)
    assert torch.all(mapping >= 0)
    assert torch.allclose(mapping.sum(dim=1), torch.ones(2), atol=1e-6)
    assert model.current_encoder[0].in_channels == 2 * 2 * 3
    output = model(
        x=torch.randn(2, 3, 64),
        delay_evidence_x=torch.randn(2, 3, 64),
    )
    assert output["logits"].shape == (2, 2)
    assert int(output["aux"]["transport_band_count"]) == 3
    assert int(output["aux"]["classification_band_count"]) == 2


def test_v41_preserves_transport_pairs_and_uses_one_frozen_physical_basis() -> None:
    model = DPCSNN(
        n_classes=2,
        n_channels=3,
        n_bands=3,
        hidden_channels=4,
        timesteps=32,
        graph_timesteps=32,
        sfreq=64.0,
        task_tmin=0.0,
        task_tmax=1.0,
        latent_nodes=3,
        snn_channels=4,
        temporal_pool_channels=2,
        temporal_pool_bins=4,
        d_max=2,
        slow_d_max=2,
        band_edges_hz=[[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]],
        delay_evidence_band_edges_hz=[[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]],
        classification_band_edges_hz=[[2.0, 5.0], [5.0, 8.0]],
        decoder_layers=3,
        n_delay_experts=2,
        delay_expert_topk=1,
        freeze_shared_physical_basis=True,
        preserve_transport_band_pairs=True,
        delayed_stat_channels=2,
        architecture_version="dpc_snn_v4_1_matched_delay_repair",
    )

    assert model.current_encoder[0].in_channels == 3 * 3 * (3 + 2)
    torch.testing.assert_close(model.spatial.weight(), model.evidence_spatial.weight())
    assert not any(parameter.requires_grad for parameter in model.spatial.parameters())
    output = model(x=torch.randn(2, 3, 64), delay_evidence_x=torch.randn(2, 3, 64))
    assert bool(output["aux"]["transport_pairs_preserved_to_snn"])
    assert output["aux"]["delayed_pair_statistics"].shape == (2, 3, 3, 2)
    assert model.export_learned_parameters()["architecture_version"] == (
        "dpc_snn_v4_1_matched_delay_repair"
    )


def test_matched_zero_changes_only_conditional_lag_transport() -> None:
    kwargs = dict(
        n_bands=1,
        n_channels=2,
        d_max=2,
        slow_d_max=2,
        n_delay_experts=2,
        use_fixed_fold_posterior=True,
        matched_transport_control=True,
    )
    audited = DelayPhaseGraphSynapse(**kwargs)
    zero = DelayPhaseGraphSynapse(**kwargs, fixed_delay_steps=0.0)
    zero.load_state_dict(audited.state_dict(), strict=True)
    route = torch.zeros_like(audited.route_mask)
    route[0, 0, 0, 1] = 0.8
    positive = torch.full((*route.shape, 3), 1.0 / 3.0)
    positive[0, 0, 0, 1] = torch.tensor([0.0, 0.8, 0.2])
    for synapse in (audited, zero):
        synapse.load_fold_local_evidence_prior(route, positive)
        synapse.begin_task_training()

    torch.testing.assert_close(
        audited.delay_route_probability_expert(),
        zero.delay_route_probability_expert(),
    )
    torch.testing.assert_close(audited.edge_weight_expert(), zero.edge_weight_expert())
    torch.testing.assert_close(
        audited.delay_confidence_expert(audited.delay_prob_expert()),
        zero.delay_confidence_expert(zero.delay_prob_expert()),
    )
    torch.testing.assert_close(
        audited.effective_phase_pref_expert(),
        zero.effective_phase_pref_expert(),
    )
    assert not torch.allclose(audited.delay_prob_expert(), zero.delay_prob_expert())
    assert not bool(audited.phase_residual_enabled)
    assert not bool(zero.phase_residual_enabled)


def test_representation_checkpoint_cannot_replace_fold_local_prior() -> None:
    source = DPCSNN(n_classes=2, n_channels=3, n_bands=2, latent_nodes=3, graph_timesteps=8)
    target = DPCSNN(n_classes=2, n_channels=3, n_bands=2, latent_nodes=3, graph_timesteps=8)
    source_route = torch.full_like(source.synapse.route_mask, 0.7) * source.synapse.route_mask
    target_route = torch.full_like(target.synapse.route_mask, 0.3) * target.synapse.route_mask
    positive = torch.full(
        (*source.synapse.route_mask.shape, source.synapse.d_max + 1),
        1.0 / (source.synapse.d_max + 1),
    )
    source_connectivity = torch.full_like(source.synapse.route_mask, 0.8)
    target_connectivity = torch.full_like(target.synapse.route_mask, 0.2)
    source.load_fold_local_evidence_prior(
        source_route, positive, connectivity_prior=source_connectivity
    )
    target.load_fold_local_evidence_prior(
        target_route, positive, connectivity_prior=target_connectivity
    )

    target.load_representation_state(source.state_dict())

    torch.testing.assert_close(target.synapse.fold_route_prior, target_route)
    torch.testing.assert_close(
        target.synapse.connectivity_prior_ema,
        target_connectivity * target.synapse.route_mask,
    )
    assert bool(target.synapse.fold_evidence_ready)


def test_online_psi_direction_matches_offline_target_source_convention() -> None:
    trials, bands, nodes, steps = 8, 8, 2, 256
    sample_rate = 128.0
    frequencies = torch.linspace(6.0, 34.0, bands)
    time = torch.arange(steps) / sample_rate
    source = torch.exp(
        1j * 2.0 * torch.pi * frequencies[:, None] * time[None]
    )
    analytic = torch.empty(trials, bands, nodes, steps, dtype=torch.complex64)
    analytic[:, :, 0] = source
    analytic[:, :, 1, :3] = 0.0
    analytic[:, :, 1, 3:] = source[:, :-3]

    prior = DelayPhaseGraphSynapse._directed_connectivity_prior(
        torch.angle(analytic),
        torch.ones_like(analytic.real),
        n_output_bands=2,
    )

    assert prior[0, 0, 1, 0] > prior[0, 0, 0, 1]


def test_causal_channel_norm_has_finite_gradient_for_zero_current() -> None:
    norm = CausalChannelNorm(channels=4)
    current = torch.zeros(3, 4, requires_grad=True)
    norm(current).sum().backward()
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()


class _TinyClassifier(torch.nn.Module):
    def __init__(self, nonfinite: bool = False):
        super().__init__()
        self.linear = torch.nn.Linear(8, 2)
        self.nonfinite = nonfinite

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        logits = self.linear(x.flatten(1))
        if self.nonfinite:
            logits = logits * torch.tensor(float("nan"), device=logits.device)
        return {"logits": logits, "aux": {}}


def test_eegnet_initialization_does_not_pollute_batchnorm_and_scales_kernels() -> None:
    model = EEGNet(n_channels=3, n_classes=2, samples=250, sfreq=250.0)

    assert model.temporal.kernel_size == (1, 125)
    assert model.sep_depth.kernel_size == (1, 31)
    torch.testing.assert_close(model.bn1.running_mean, torch.zeros_like(model.bn1.running_mean))
    torch.testing.assert_close(model.bn1.running_var, torch.ones_like(model.bn1.running_var))
    constrained = _max_norm_weight(torch.full((2, 1, 3, 1), 10.0), 1.0)
    assert torch.all(constrained.flatten(1).norm(dim=1) <= 1.0 + 1e-6)


def _tiny_training_data() -> dict[str, np.ndarray]:
    return {
        "X": np.random.default_rng(0).normal(size=(8, 2, 4)).astype(np.float32),
        "y": np.asarray([0, 1] * 4, dtype=np.int64),
    }


def test_dpc_snn_accepts_batches_with_x_and_features() -> None:
    model = DPCSNN(n_classes=4, n_channels=3, n_bands=2, timesteps=4, d_max=8)
    batch = {
        "x": torch.randn(5, 3, 32),
        "amplitude": torch.rand(5, 2, 3, 32),
        "phase": torch.randn(5, 2, 3, 32),
    }

    out = forward_batch(model, batch)

    assert out["logits"].shape == (5, 4)
    assert out["aux"]["hidden_spikes"].shape[-1] == model.graph_timesteps == 500


def test_dpc_snn_builds_continuous_frontend_before_spiking() -> None:
    from dpc_snn.models.build import build_model

    model = build_model(
        "dpc_snn",
        {
            "n_channels": 3,
            "n_classes": 2,
            "n_bands": 2,
            "timesteps": 4,
            "d_max": 1,
            "band_edges_hz": [[8.0, 13.0], [13.0, 30.0]],
        },
    )

    assert model.filterbank.n_bands == 2
    assert not hasattr(model, "encoder")


def test_delay_calibration_selects_complete_cross_band_mechanism() -> None:
    model = DPCSNN(n_classes=2, n_channels=3, n_bands=2, latent_nodes=3)

    model.set_calibration_mode("delay")

    assert model.synapse.fast_delay_logits.requires_grad
    assert model.synapse.fast_fraction_raw.requires_grad
    assert model.synapse.edge_logits.requires_grad
    assert model.synapse.edge_existence_logits.requires_grad
    assert model.synapse.delay_confidence_logits.requires_grad
    assert not model.synapse.phase_pref.requires_grad

    model.set_calibration_mode("phase_delay")
    assert model.synapse.phase_pref.requires_grad
    assert model.phase_confidence_raw.requires_grad


def test_e14_dpc_budget_balances_architecture_and_optimizer_dimensions() -> None:
    candidates = _fair_hparam_candidates("dpc_snn", 20)

    architectures = {(candidate["hidden_channels"], candidate["d_max"]) for candidate in candidates}
    learning_rates = {candidate["training"]["lr"] for candidate in candidates}
    weight_decays = {candidate["training"]["weight_decay"] for candidate in candidates}
    optimizer_pairs = {
        (candidate["training"]["lr"], candidate["training"]["weight_decay"])
        for candidate in candidates
    }

    assert len(architectures) == 20
    assert learning_rates == {3e-4, 1e-3, 3e-3}
    assert weight_decays == {0.0, 1e-4, 1e-3}
    assert len(optimizer_pairs) == 9


def test_few_shot_finetune_freezes_batchnorm_running_statistics(tmp_path) -> None:
    from dpc_snn.models.eegnet import EEGNet
    from dpc_snn.training.adaptation import supervised_finetune

    model = EEGNet(n_channels=3, n_classes=2, samples=128)
    before = model.bn1.running_mean.detach().clone()
    data = {
        "X": np.random.default_rng(0).normal(size=(8, 3, 128)).astype(np.float32),
        "y": np.asarray([0, 1] * 4, dtype=np.int64),
    }

    supervised_finetune(
        model,
        data,
        data,
        {
            "device": "cpu",
            "n_classes": 2,
            "training": {"calibration_epochs": 1, "batch_size": 4, "calibration_freeze_batchnorm": True},
        },
        tmp_path,
        mode="full",
    )

    torch.testing.assert_close(model.bn1.running_mean, before)


def test_checkpoint_model_builder_honours_saved_resolved_model_config() -> None:
    from dpc_snn.experiments.runners import _clone_model_with_checkpoint_audit

    data = {"X": np.zeros((4, 3, 32), dtype=np.float32), "y": np.asarray([0, 1, 0, 1]), "sfreq": 250.0}
    _, _, run_cfg, _ = _clone_model_with_checkpoint_audit(
        "dpc_snn",
        data,
        {
            "resolved_model_config": {"n_channels": 3, "n_classes": 2, "n_bands": 2, "band_edges_hz": [[2.0, 4.0], [4.0, 6.0]], "delay_evidence_band_edges_hz": [[2.0, 4.0], [4.0, 6.0]], "timesteps": 5, "graph_timesteps": 5, "preserve_graph_rate": True, "d_max": 2, "hidden_channels": 11},
        },
    )
    model, _, _, _ = _clone_model_with_checkpoint_audit("dpc_snn", data, run_cfg)

    assert model.timesteps == 5
    assert model.synapse.d_max == 2
    assert model.readout.in_features == 11


def test_checkpoint_clone_loads_saved_baseline_timing_automatically(tmp_path: Path) -> None:
    from dpc_snn.experiments.runners import _clone_model_with_checkpoint_audit

    data = {
        "X": np.zeros((4, 3, 250), dtype=np.float32),
        "y": np.asarray([0, 1, 0, 1]),
        "sfreq": 125.0,
        "epoch_tmin": -1.0,
        "epoch_tmax": 1.0,
    }
    resolved = {
        "n_channels": 3,
        "n_classes": 2,
        "n_bands": 2,
        "band_edges_hz": [[6.0, 10.0], [14.0, 20.0]],
        "delay_evidence_band_edges_hz": [[6.0, 10.0], [14.0, 20.0]],
        "graph_rate_hz": 96.0,
        "graph_timesteps": 96,
        "timesteps": 96,
        "preserve_graph_rate": True,
        "latent_nodes": 3,
        "snn_channels": 8,
        "epoch_tmin": -1.0,
        "task_tmin": 0.0,
        "task_tmax": 1.0,
    }
    source_model, _, _, _ = _clone_model_with_checkpoint_audit(
        "dpc_snn", data, {"resolved_model_config": resolved}
    )
    checkpoint = tmp_path / "model_checkpoint.pt"
    torch.save(source_model.state_dict(), checkpoint)
    (tmp_path / "run_config.json").write_text(
        json.dumps({"resolved_model_config": resolved}), encoding="utf-8"
    )

    restored, _, _, audit = _clone_model_with_checkpoint_audit(
        "dpc_snn", data, {"device": "cpu"}, checkpoint
    )

    assert restored.epoch_tmin == -1.0
    assert restored.task_tmin == 0.0
    assert audit["checkpoint_config_loaded"] is True
    assert audit["loaded_param_ratio"] == pytest.approx(1.0)


def test_prefix_features_do_not_depend_on_future_samples() -> None:
    raw = torch.randn(2, 3, 128)
    batch_a = {"x": raw.clone(), "y": torch.tensor([0, 1])}
    batch_b = {"x": raw.clone(), "y": torch.tensor([0, 1])}
    batch_b["x"][..., 64:] += 100.0

    prefix_a = build_prefix_feature_batch(batch_a, 64, sfreq=250.0, bands={"mu": [8.0, 13.0], "beta": [13.0, 30.0]})
    prefix_b = build_prefix_feature_batch(batch_b, 64, sfreq=250.0, bands={"mu": [8.0, 13.0], "beta": [13.0, 30.0]})

    torch.testing.assert_close(prefix_a["x"], prefix_b["x"])
    torch.testing.assert_close(prefix_a["amplitude"], prefix_b["amplitude"])
    torch.testing.assert_close(prefix_a["phase"], prefix_b["phase"])


def test_edge_occlusion_override_does_not_rethreshold_remaining_edges() -> None:
    synapse = DelayPhaseGraphSynapse(n_bands=2, n_channels=3, d_max=2, graph_sparsity=0.35)
    before = synapse.edge_weight().detach().clone()
    multiplier = torch.ones_like(before)
    multiplier.reshape(-1)[1] = 0.0
    synapse.set_edge_weight_override(multiplier)
    after = synapse.edge_weight().detach()

    assert after.reshape(-1)[1] == 0.0
    torch.testing.assert_close(after[multiplier.bool()], before[multiplier.bool()])


def test_phase_resampling_preserves_wraparound() -> None:
    phase = torch.tensor([[[np.pi - 0.1, -np.pi + 0.1]]], dtype=torch.float32)

    resized = resize_phase(phase, 3)

    assert abs(float(resized[..., 1])) > 3.0


def test_delay_moves_source_spike_and_source_phase_together() -> None:
    synapse = DelayPhaseGraphSynapse(
        n_bands=1,
        n_channels=2,
        d_max=1,
        graph_sparsity=0.0,
        delay_gate_init=20.0,
        n_delay_experts=1,
        expert_topk=1,
    )
    with torch.no_grad():
        synapse.edge_logits.fill_(2.0)
        synapse.delay_logits.fill_(-20.0)
        synapse.delay_logits[..., 1] = 20.0
        synapse.delay_confidence_logits.fill_(20.0)
        synapse.fast_fraction_raw.fill_(-30.0)
        synapse.slow_modulation_raw.fill_(-30.0)
        synapse.phase_pref.zero_()
        route = torch.zeros_like(synapse.route_mask)
        route[0, 0, 0, 1] = 1.0
        synapse.set_edge_weight_override(route)

    spikes = torch.zeros(1, 1, 2, 2)
    spikes[0, 0, 1, 0] = 1.0
    phase = torch.zeros_like(spikes)
    phase[0, 0, 1, 1] = np.pi

    _, aux = synapse(spikes, phase, torch.ones_like(spikes))

    assert float(aux["delayed_current"][0, 0, 0, 0, 1]) > 0.0
    torch.testing.assert_close(aux["delayed_current"][0, 0, 0, 0, 0], torch.tensor(0.0))


def test_delay_distribution_is_specific_to_source_target_band_pair() -> None:
    synapse = DelayPhaseGraphSynapse(n_bands=3, n_channels=2, d_max=2)

    assert synapse.delay_prob().shape == (3, 3, 2, 2, 3)
    assert synapse.learned_delay().shape == (3, 3, 2, 2)


def test_low_amplitude_phase_confidence_suppresses_delayed_current() -> None:
    synapse = DelayPhaseGraphSynapse(n_bands=1, n_channels=2, d_max=1, graph_sparsity=0.0)
    signal = torch.ones(1, 1, 2, 4)
    phase = torch.zeros_like(signal)

    current_high, _ = synapse(signal, phase, torch.ones_like(signal))
    current_low, _ = synapse(signal, phase, torch.zeros_like(signal))

    assert current_high.abs().sum() > 0
    torch.testing.assert_close(current_low, torch.zeros_like(current_low))


def test_within_band_phase_gate_controls_routed_current_magnitude() -> None:
    synapse = DelayPhaseGraphSynapse(
        n_bands=1,
        n_channels=2,
        d_max=1,
        delay_gate_init=20.0,
        force_zero_delay=True,
        n_delay_experts=1,
    ).eval()
    with torch.no_grad():
        synapse.edge_logits.fill_(2.0)
        synapse.delay_confidence_logits.fill_(20.0)
        synapse.slow_modulation_raw.fill_(-30.0)
        route = torch.zeros_like(synapse.route_mask)
        route[0, 0, 0, 1] = 1.0
        synapse.set_edge_weight_override(route)
    carrier = torch.ones(1, 1, 2, 8)
    aligned_phase = torch.zeros_like(carrier)
    opposed_phase = aligned_phase.clone()
    opposed_phase[:, :, 1] = torch.pi

    aligned, _ = synapse(carrier, aligned_phase, torch.ones_like(carrier))
    opposed, _ = synapse(carrier, opposed_phase, torch.ones_like(carrier))

    assert aligned.abs().sum() > opposed.abs().sum() * 2.0


def test_fractional_delay_recovers_quarter_half_and_three_quarter_samples() -> None:
    rows = recover_fractional_delays(steps=120)

    assert all(row["passed"] for row in rows)
    assert max(float(row["absolute_error_samples"]) for row in rows) <= 0.03
    assert {row["recovery_objective"] for row in rows} == {"delayed_signal_fit"}


def test_joint_training_progress_stops_router_warmup_noise() -> None:
    synapse = DelayPhaseGraphSynapse(
        n_bands=1,
        n_channels=2,
        d_max=2,
        n_delay_experts=2,
        expert_topk=1,
        router_warmup_fraction=0.35,
    )

    synapse.set_training_progress(0.8, anneal_lag=True)
    synapse.set_router_training_progress(0.8)

    assert float(synapse.training_progress_state) == pytest.approx(0.8)
    assert synapse.delay_temperature == pytest.approx(
        0.2 * synapse.initial_delay_temperature
        + 0.8 * synapse.min_delay_temperature
    )


def test_gcc_phat_recovers_directed_positive_delay() -> None:
    torch.manual_seed(3)
    delay = 5
    source = torch.randn(24, 1, 1, 128)
    target = torch.zeros_like(source)
    target[..., delay:] = source[..., :-delay]
    carrier = torch.cat((target, source), dim=2)

    evidence = DelayPhaseGraphSynapse._gcc_phat_evidence(carrier, n_delays=9)

    assert int(evidence[0, 0, 0, 1].argmax()) == delay
    assert float(evidence[0, 0, 0, 1, delay]) > 0.95


def test_fractional_delay_cannot_exceed_declared_maximum() -> None:
    synapse = DelayPhaseGraphSynapse(n_bands=1, n_channels=2, d_max=2)
    with torch.no_grad():
        synapse.fast_delay_logits.fill_(-20.0)
        synapse.fast_delay_logits[..., -1] = 20.0
        synapse.fast_fraction_raw.fill_(20.0)

    assert float(synapse.learned_fast_delay().max()) <= 2.0
    assert float(synapse.learned_delay_map().max()) <= 2.0


def test_envelope_uses_mandatory_delay_routes_without_a_classifier_bypass() -> None:
    synapse = DelayPhaseGraphSynapse(n_bands=1, n_channels=3, d_max=2)
    carrier = torch.zeros(2, 1, 3, 16)
    envelope = torch.rand_like(carrier) * 10.0

    current, _ = synapse(
        carrier,
        torch.zeros_like(carrier),
        torch.ones_like(carrier),
        envelope=envelope,
    )

    assert float(current.abs().sum()) > 0.0
    synapse.set_edge_weight_override(torch.zeros_like(synapse.route_mask))
    blocked, _ = synapse(
        carrier,
        torch.zeros_like(carrier),
        torch.ones_like(carrier),
        envelope=envelope,
    )
    torch.testing.assert_close(blocked, torch.zeros_like(blocked))


def test_delay_anchor_kl_detects_posterior_drift() -> None:
    synapse = DelayPhaseGraphSynapse(n_bands=1, n_channels=2, d_max=2)
    synapse.capture_delay_anchor()
    carrier = torch.randn(2, 1, 2, 16)
    with torch.no_grad():
        synapse.fast_delay_logits[..., 2] = 10.0
    synapse.eval()

    _, aux = synapse(carrier, torch.zeros_like(carrier), torch.ones_like(carrier))

    assert float(aux["delay_anchor_kl"]) > 0.1


def test_gate_structure_is_an_upper_bound_not_a_route_quota() -> None:
    synapse = DelayPhaseGraphSynapse(
        n_bands=1, n_channels=3, d_max=2, graph_sparsity=0.35
    )
    carrier = torch.randn(2, 1, 3, 16)
    with torch.no_grad():
        synapse.fast_delay_logits.fill_(-20.0)
        synapse.fast_delay_logits[..., 1] = 20.0
        synapse.delay_confidence_logits.fill_(20.0)
        synapse.edge_existence_logits.fill_(-20.0)
    synapse.eval()
    _, empty = synapse(
        carrier, torch.zeros_like(carrier), torch.ones_like(carrier)
    )

    with torch.no_grad():
        synapse.edge_existence_logits.fill_(20.0)
    _, dense = synapse(
        carrier, torch.zeros_like(carrier), torch.ones_like(carrier)
    )

    assert float(empty["delay_gate_structure_loss"]) == 0.0
    assert float(dense["delay_gate_structure_loss"]) > 0.1


def test_coarse_posterior_unwraps_short_carrier_alias() -> None:
    synapse = DelayPhaseGraphSynapse(
        n_bands=1, n_channels=2, d_max=11, slow_d_max=8, slow_downsample=4
    )
    carrier = torch.full((1, 1, 2, 2, 12), 1e-5)
    carrier[..., 2] = 0.8
    carrier[..., 10] = 0.2
    carrier = carrier / carrier.sum(dim=-1, keepdim=True)
    coarse = torch.full((1, 1, 2, 2, 3), 1e-4)
    coarse[..., 2] = 1.0
    coarse = coarse / coarse.sum(dim=-1, keepdim=True)
    synapse.coarse_reliability_ema.fill_(1.0)

    total = synapse._total_delay_prob(carrier, coarse)

    assert int(total[0, 0, 0, 1].argmax()) == 10


def test_cross_band_route_preserves_pair_axis_and_never_relabels_carrier() -> None:
    synapse = DelayPhaseGraphSynapse(
        n_bands=2, n_channels=2, d_max=1, graph_sparsity=0.0, delay_gate_init=20.0,
        n_delay_experts=1, expert_topk=1,
    )
    with torch.no_grad():
        synapse.edge_logits.fill_(2.0)
        synapse.fast_delay_logits.fill_(-20.0)
        synapse.fast_delay_logits[..., 1] = 20.0
        synapse.delay_confidence_logits.fill_(20.0)
        synapse.fast_fraction_raw.fill_(-30.0)
        synapse.slow_modulation_raw.fill_(-30.0)
        route = torch.zeros_like(synapse.route_mask)
        route[1, 0, 0, 1] = 1.0
        synapse.set_edge_weight_override(route)
    carrier = torch.zeros(1, 2, 2, 3)
    carrier[0, 0, 1, 0] = 1.0
    envelope = torch.zeros_like(carrier)
    envelope[0, 0, 1, 0] = 1.0
    envelope[0, 1, 0] = 1.0

    carrier_only, _ = synapse(
        carrier, torch.zeros_like(carrier), torch.ones_like(carrier),
        envelope=torch.zeros_like(carrier),
    )
    interaction, _ = synapse(
        carrier, torch.zeros_like(carrier), torch.ones_like(carrier),
        envelope=envelope,
    )

    torch.testing.assert_close(carrier_only, torch.zeros_like(carrier_only))
    assert interaction.shape == (1, 2, 2, 2, 3)
    assert float(interaction[0, 1, 0, 0, 1]) > 0.0
    torch.testing.assert_close(interaction[0, 0, 1], torch.zeros_like(interaction[0, 0, 1]))


def test_within_band_control_masks_only_cross_band_routes() -> None:
    synapse = DelayPhaseGraphSynapse(
        n_bands=2, n_channels=2, use_cross_band_routes=False
    )

    assert float(synapse.route_mask[1, 0].sum()) == 0.0
    assert float(synapse.route_mask[0, 0, 0, 1]) == 1.0


def test_shared_representation_load_preserves_variant_route_mask() -> None:
    full = DPCSNN(
        n_classes=2,
        n_channels=3,
        n_bands=2,
        graph_timesteps=32,
        timesteps=32,
        latent_nodes=3,
        snn_channels=8,
        band_edges_hz=[[8.0, 13.0], [13.0, 30.0]],
    )
    within = DPCSNN(
        n_classes=2,
        n_channels=3,
        n_bands=2,
        graph_timesteps=32,
        timesteps=32,
        latent_nodes=3,
        snn_channels=8,
        band_edges_hz=[[8.0, 13.0], [13.0, 30.0]],
        use_cross_band_routes=False,
    )
    expected_mask = within.synapse.route_mask.clone()

    within.load_representation_state(full.state_dict())

    torch.testing.assert_close(within.synapse.route_mask, expected_mask)
    assert float(within.synapse.route_mask[1, 0].sum()) == 0.0
    torch.testing.assert_close(within.spatial.delta, full.spatial.delta)


def test_dpc_hparam_budget_covers_architecture_grid_without_zero_delay() -> None:
    candidates = _fair_hparam_candidates("dpc_snn", budget=20)

    assert {item["hidden_channels"] for item in candidates} == {24, 32, 48, 64, 96}
    assert {item["d_max"] for item in candidates} == {2, 4, 8, 12}
    assert len({(item["hidden_channels"], item["d_max"]) for item in candidates}) == 20


def test_euclidean_alignment_whitens_training_reference_covariance() -> None:
    rng = np.random.default_rng(0)
    mixing = np.asarray([[2.0, 0.5, 0.0], [0.2, 1.0, 0.3], [0.0, 0.1, 0.5]], dtype=np.float32)
    x = np.einsum("ij,njt->nit", mixing, rng.normal(size=(20, 3, 200)).astype(np.float32))

    alignment = euclidean_alignment_matrix(x)
    aligned = np.einsum("ij,njt->nit", alignment, x)
    aligned = aligned - aligned.mean(axis=-1, keepdims=True)
    covariance = np.einsum("nct,ndt->ncd", aligned, aligned).mean(axis=0) / (aligned.shape[-1] - 1)

    np.testing.assert_allclose(covariance, np.eye(3), atol=2e-3)


def test_covariance_branch_cannot_bypass_delay_graph() -> None:
    torch.manual_seed(0)
    model = DPCSNN(
        n_classes=2,
        n_channels=3,
        n_bands=2,
        timesteps=8,
        graph_timesteps=16,
        latent_nodes=3,
        snn_channels=8,
        band_edges_hz=[[8.0, 13.0], [13.0, 30.0]],
    )
    model.eval()
    model.synapse.set_edge_weight_override(torch.zeros_like(model.synapse.route_mask))

    logits_a = model(x=torch.randn(2, 3, 64))["logits"]
    logits_b = model(x=torch.randn(2, 3, 64) * 5.0)["logits"]

    torch.testing.assert_close(logits_a, logits_b)


def test_coordinate_spatial_projection_uses_distributed_motor_anchors() -> None:
    projection = AnchoredSpatialProjection(n_channels=22, n_nodes=16)

    assert projection.node_coordinates.shape == (16, 2)
    assert torch.all((projection.anchors > 0.05).sum(dim=1) > 1)
    assert float(projection.node_coordinates[:, 1].abs().mean()) < 0.5


def test_band_specific_low_rank_projection_and_covariance_are_trainable() -> None:
    projection = BandSpecificAnchoredSpatialProjection(
        n_channels=6, n_nodes=4, n_bands=3, rank=2
    )
    x = torch.randn(2, 3, 6, 32)
    nodes = projection(x)
    covariance = BandLogCovarianceBranch(3, 4, 4)
    context = covariance(nodes)
    (context.square().mean() + projection.orthogonality_loss()).backward()

    assert nodes.shape == (2, 3, 4, 32)
    assert context.shape == (2, 12)
    assert projection.band_coefficients.grad is not None
    assert projection.weight().shape == (3, 4, 6)


def test_band_log_covariance_is_stable_for_repeated_eigenvalues() -> None:
    covariance = BandLogCovarianceBranch(3, 4, 4)
    x = torch.zeros(2, 3, 4, 32, requires_grad=True)

    context = covariance(x)
    context.square().mean().backward()

    assert torch.isfinite(context).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_band_log_covariance_handles_rank_deficiency_and_large_scale() -> None:
    covariance = BandLogCovarianceBranch(3, 4, 4)
    source = (torch.randn(2, 3, 1, 32) * 1e10).requires_grad_()
    x = source.repeat(1, 1, 4, 1)

    context = covariance(x)
    context.square().mean().backward()

    assert torch.isfinite(context).all()
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()


def test_matrix_log_covariance_gradient_matches_finite_differences() -> None:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(1, 2, 3, 8, generator=generator, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda value: _stable_log_covariance(value)[0],
        (x,),
        eps=1e-6,
        atol=2e-4,
        rtol=2e-3,
    )


def test_learnable_filterbank_cannot_cross_graph_nyquist_guard() -> None:
    filterbank = LearnableAnalyticFilterBank(
        250.0,
        [[30.0, 40.0]],
        max_high_hz=61.5,
    )
    with torch.no_grad():
        filterbank.center_shift_raw.fill_(100.0)
        filterbank.bandwidth_raw.fill_(100.0)

    center, width, _ = filterbank.band_parameters()

    assert float(center + width / 2.0) <= 61.5 + 1e-6
    assert float(width) <= 15.0 + 1e-6


def test_baseline_scale_prevents_cancelled_node_amplification() -> None:
    latent = torch.ones(1, 1, 2, 8, dtype=torch.complex64)
    latent[:, :, 0, :4] = 0.0

    baseline = DPCSNN._baseline_scale(latent, baseline_end=4)

    assert float(baseline[0, 0, 0, 0]) >= 0.05 - 1e-6
    assert float(baseline[0, 0, 1, 0]) == pytest.approx(1.0)


def test_training_writes_atomic_epoch_checkpoint_and_completed_status(tmp_path: Path) -> None:
    data = _tiny_training_data()
    result = train_model(
        _TinyClassifier(),
        data,
        data,
        {
            "device": "cpu",
            "n_classes": 2,
            "model_name": "tiny",
            "training": {"epochs": 1, "batch_size": 4, "patience": 1},
        },
        tmp_path,
    )

    status = json.loads((tmp_path / "training_status.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(tmp_path / "last_epoch_checkpoint.pt", weights_only=False)
    assert status["status"] == "completed"
    assert checkpoint["epoch"] == 1
    assert (tmp_path / "best_model_checkpoint.pt").exists()
    assert result["metrics"]["evaluation_split"] == "validation"


def test_training_rejects_nonfinite_logits_and_records_failure(tmp_path: Path) -> None:
    data = _tiny_training_data()

    with pytest.raises(FloatingPointError, match="Non-finite logits"):
        train_model(
            _TinyClassifier(nonfinite=True),
            data,
            data,
            {
                "device": "cpu",
                "n_classes": 2,
                "model_name": "tiny",
                "training": {"epochs": 1, "batch_size": 4},
            },
            tmp_path,
        )

    status = json.loads((tmp_path / "training_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["error_type"] == "FloatingPointError"
    assert "epoch=1, batch=0" in status["error"]


def test_clif_has_per_channel_decay_and_causal_channel_normalization() -> None:
    neuron = CLIFLayer(
        decay=0.9,
        channels=4,
        learnable_decay=True,
        causal_channel_norm=True,
    )
    spikes, membrane = neuron(torch.randn(3, 4, 12, requires_grad=True))
    membrane.square().mean().backward()

    assert spikes.shape == (3, 4, 12)
    assert neuron.decay.shape == (4,)
    assert neuron.decay_raw.grad is not None
    assert torch.isfinite(neuron.decay_raw.grad).all()
    assert float(neuron.decay_raw.grad.abs().max()) < 100.0


def test_clif_prefix_is_invariant_to_future_samples() -> None:
    torch.manual_seed(9)
    neuron = CLIFLayer(decay=0.9, channels=4, causal_channel_norm=True).eval()
    current_a = torch.randn(2, 4, 20)
    current_b = current_a.clone()
    current_b[..., 10:] += 100.0

    spikes_a, membrane_a = neuron(current_a)
    spikes_b, membrane_b = neuron(current_b)

    torch.testing.assert_close(spikes_a[..., :10], spikes_b[..., :10])
    torch.testing.assert_close(membrane_a[..., :10], membrane_b[..., :10])


def test_causal_channel_norm_is_non_centering_unit_rms() -> None:
    neuron = CLIFLayer(decay=0.9, channels=4, causal_channel_norm=True)
    current = torch.rand(3, 4) + 0.1

    normalized = neuron.current_norm(current)

    torch.testing.assert_close(
        normalized.square().mean(-1).sqrt(),
        torch.ones(3),
    )
    assert torch.all(normalized > 0.0)


def test_low_rank_delay_experts_use_fewer_parameters_than_full_routes() -> None:
    synapse = DelayPhaseGraphSynapse(n_bands=7, n_channels=16, d_max=8, n_delay_experts=4)
    low_rank = sum(parameter.numel() for parameter in synapse.delay_parameters())
    full_route_fast_delay_only = 7 * 7 * 16 * 16 * 9

    assert low_rank < full_route_fast_delay_only
    assert synapse.fast_delay_logits.shape == (4, 7, 7, 9)
    assert synapse.delay_prob().shape == (7, 7, 16, 16, 9)


def test_hurdle_delay_separates_null_from_conditional_positive_posterior() -> None:
    positive = hurdle_delay_posterior(torch.tensor([[0.0, 4.0, 1.0]]))
    null = hurdle_delay_posterior(torch.tensor([[4.0, 0.0, 1.0]]))

    assert float(positive.route_probability[0]) > 0.9
    assert float(null.route_probability[0]) < 0.1
    assert float(positive.positive_probability[0, 0]) == 0.0
    torch.testing.assert_close(positive.positive_probability.sum(-1), torch.ones(1))


def test_hurdle_force_zero_is_capacity_matched_control_only() -> None:
    score = torch.tensor([[0.0, 4.0, 1.0]])
    full = hurdle_delay_posterior(score)
    output = hurdle_delay_posterior(score, force_zero_delay=True)

    torch.testing.assert_close(output.positive_probability, torch.tensor([[1.0, 0.0, 0.0]]))
    torch.testing.assert_close(output.route_probability, full.route_probability)
    torch.testing.assert_close(output.route_log_odds, full.route_log_odds)


def test_route_temperature_is_independent_of_lag_annealing() -> None:
    synapse = DelayPhaseGraphSynapse(n_bands=1, n_channels=2, d_max=2)
    before = synapse.delay_route_probability_expert().detach().clone()

    synapse.set_training_progress(1.0, anneal_lag=True)
    after = synapse.delay_route_probability_expert().detach()

    assert synapse.delay_temperature == pytest.approx(synapse.min_delay_temperature)
    torch.testing.assert_close(before, after)
    assert float(synapse.training_progress_state) == 0.0


def test_router_warmup_progress_does_not_restart_during_joint_annealing() -> None:
    synapse = DelayPhaseGraphSynapse(n_bands=1, n_channels=2, d_max=2)
    synapse.set_training_progress(1.0, anneal_lag=False)

    synapse.set_training_progress(0.0, anneal_lag=True)

    assert float(synapse.training_progress_state) == 1.0
    assert synapse.delay_temperature == pytest.approx(synapse.initial_delay_temperature)


def test_fold_local_prior_is_loaded_into_route_and_lag_posteriors() -> None:
    synapse = DelayPhaseGraphSynapse(
        n_bands=1, n_channels=2, d_max=2, n_delay_experts=1
    )
    route = torch.zeros_like(synapse.route_mask)
    route[0, 0, 0, 1] = 0.95
    positive = torch.full((*route.shape, 3), 1.0 / 3.0)
    positive[0, 0, 0, 1] = torch.tensor([0.05, 0.9, 0.05])
    fraction = torch.zeros_like(route)
    fraction[0, 0, 0, 1] = 0.4

    synapse.load_fold_local_evidence_prior(route, positive, fraction)

    assert bool(synapse.fold_evidence_ready)
    assert float(synapse.delay_route_probability_expert()[0, 0, 0, 0, 1]) > 0.9
    assert int(synapse.fast_delay_prob_expert()[0, 0, 0, 0, 1].argmax()) == 1
    assert float(synapse.fold_fraction_target[0, 0, 0, 1]) == pytest.approx(0.4)
    synapse.set_evidence_updates(True)
    assert synapse.evidence_updates_enabled is False


def test_fixed_fold_posterior_is_exact_and_independent_of_random_logits() -> None:
    synapse = DelayPhaseGraphSynapse(
        n_bands=1,
        n_channels=2,
        d_max=2,
        slow_d_max=2,
        n_delay_experts=2,
        use_fixed_fold_posterior=True,
    )
    route = torch.zeros_like(synapse.route_mask)
    route[0, 0, 0, 1] = 0.8
    positive = torch.full((*route.shape, 3), 1.0 / 3.0)
    positive[0, 0, 0, 1] = torch.tensor([0.1, 0.7, 0.2])
    fraction = torch.zeros_like(route)
    fraction[0, 0, 0, 1] = 0.4
    synapse.load_fold_local_evidence_prior(route, positive, fraction)

    with torch.no_grad():
        synapse.fast_delay_logits.normal_(mean=0.0, std=20.0)
        synapse.edge_existence_logits.normal_(mean=0.0, std=20.0)
        synapse.fast_fraction_raw.normal_(mean=0.0, std=20.0)

    expected_route = torch.where(
        synapse.route_mask > 0,
        route.clamp_min(synapse.rejected_route_prior_floor),
        torch.zeros_like(route),
    )
    torch.testing.assert_close(
        synapse.delay_route_probability_expert(),
        expected_route[None].expand(synapse.n_delay_experts, *route.shape),
    )
    torch.testing.assert_close(
        synapse.fast_delay_prob_expert(),
        (positive * synapse.route_mask[..., None])[None].expand(
            synapse.n_delay_experts, *positive.shape
        ),
    )
    torch.testing.assert_close(
        synapse.combined_fraction_expert(),
        fraction[None].expand(synapse.n_delay_experts, *fraction.shape),
    )
    assert int(synapse.slow_delay_prob_expert()[0, 0, 0, 0, 1].argmax()) == 0


def test_delay_gate_matches_forward_route_weight_without_squaring_route() -> None:
    synapse = DelayPhaseGraphSynapse(
        n_bands=1, n_channels=2, d_max=2, n_delay_experts=1,
        use_fixed_fold_posterior=True,
    )
    route = torch.zeros_like(synapse.route_mask)
    route[0, 0, 0, 1] = 0.5
    positive = torch.full((*route.shape, 3), 1.0 / 3.0)
    synapse.load_fold_local_evidence_prior(route, positive)

    probability = synapse.delay_prob_expert()
    expected = (
        synapse.delay_confidence_expert(probability)
        * synapse.delay_route_probability_expert()
    ).mean(0)

    torch.testing.assert_close(synapse.delay_gate(), expected)


def test_fixed_delay_applies_same_physical_lag_to_slow_envelope() -> None:
    synapse = DelayPhaseGraphSynapse(
        n_bands=1,
        n_channels=2,
        d_max=4,
        slow_d_max=2,
        slow_downsample=4,
        n_delay_experts=1,
        fixed_delay_steps=1.0,
    )

    assert int(synapse.fast_delay_prob_expert()[0, 0, 0, 0, 1].argmax()) == 1
    assert int(synapse.slow_delay_prob_expert()[0, 0, 0, 0, 1].argmax()) == 0
    assert float(synapse.slow_fraction_expert()[0, 0, 0, 0, 1]) == pytest.approx(0.25)


def test_route_lag_and_fraction_objectives_have_separate_gradients() -> None:
    synapse = DelayPhaseGraphSynapse(
        n_bands=1, n_channels=2, d_max=2, n_delay_experts=1, expert_topk=1
    )
    route = torch.zeros_like(synapse.route_mask)
    route[0, 0, 0, 1] = 0.95
    positive = torch.full((*route.shape, 3), 1.0 / 3.0)
    positive[0, 0, 0, 1] = torch.tensor([0.05, 0.9, 0.05])
    fraction = torch.zeros_like(route)
    fraction[0, 0, 0, 1] = 0.4
    synapse.load_fold_local_evidence_prior(route, positive, fraction)
    synapse.eval()
    carrier = torch.randn(4, 1, 2, 32)

    _, aux = synapse(carrier, torch.zeros_like(carrier), torch.ones_like(carrier))
    loss = (
        aux["route_prior_loss"]
        + aux["positive_delay_kl_loss"]
        + aux["fractional_delay_loss"]
    )
    loss.backward()

    assert synapse.edge_existence_logits.grad is not None
    assert synapse.fast_delay_logits.grad is not None
    assert synapse.fast_fraction_raw.grad is not None
    assert float(synapse.edge_existence_logits.grad.abs().sum()) > 0.0
    assert float(synapse.fast_delay_logits.grad.abs().sum()) > 0.0
    assert float(synapse.fast_fraction_raw.grad.abs().sum()) > 0.0


def test_fixed_delay_control_supports_subsample_transport_without_bypass() -> None:
    synapse = DelayPhaseGraphSynapse(
        n_bands=1, n_channels=2, d_max=2, n_delay_experts=1,
        fixed_delay_steps=0.5,
    )

    probability = synapse.fast_delay_prob_expert()
    fraction = synapse.combined_fraction_expert()

    assert int(probability[0, 0, 0, 0, 1].argmax()) == 0
    assert float(fraction[0, 0, 0, 0, 1]) == pytest.approx(0.5)


def test_cp_route_factor_can_select_one_source_target_interaction() -> None:
    synapse = DelayPhaseGraphSynapse(
        n_bands=2, n_channels=3, d_max=2, n_delay_experts=1, route_rank=2
    )
    with torch.no_grad():
        synapse.edge_logits.zero_()
        synapse.edge_target_logits.zero_()
        synapse.edge_source_logits.zero_()
        synapse.edge_rank_logits.zero_()
        synapse.edge_target_logits[0, 1, 0, 0] = 1.0
        synapse.edge_source_logits[0, 0, 2, 0] = 1.0
        synapse.edge_rank_logits[0, 1, 0, 0] = 2.0

    logits = synapse._edge_logits_full()[0]

    assert float(logits[1, 0, 0, 2]) > 1.0
    assert torch.count_nonzero(logits) == 1


def test_group_phase_slope_identifies_shared_delay_across_bands() -> None:
    timestep = 0.01
    delay = 4
    frequencies = torch.tensor([8.0, 12.0, 18.0])
    time = torch.arange(200) * timestep
    source = 2.0 * torch.pi * frequencies[:, None] * time[None]
    target = source - 2.0 * torch.pi * frequencies[:, None] * (delay * timestep)
    phase = torch.stack((target, source), dim=1)[None].repeat(12, 1, 1, 1)
    synapse = DelayPhaseGraphSynapse(
        n_bands=3, n_channels=2, d_max=7, timestep_seconds=timestep
    )

    evidence = synapse._phase_slope_evidence(
        phase, torch.ones_like(phase), frequencies, n_delays=8
    )

    for band in range(3):
        assert int(evidence[band, band, 0, 1].argmax()) == delay


def test_group_phase_slope_is_invariant_to_unknown_phase_intercept() -> None:
    timestep = 0.01
    delay = 3
    frequencies = torch.tensor([7.0, 9.0, 11.0, 14.0, 18.0, 23.0, 29.0, 36.0])
    time = torch.arange(300) * timestep
    source = 2.0 * torch.pi * frequencies[:, None] * time[None]
    target = source - 2.0 * torch.pi * frequencies[:, None] * (delay * timestep) + 1.17
    phase = torch.stack((target, source), dim=1)[None].repeat(8, 1, 1, 1)
    synapse = DelayPhaseGraphSynapse(
        n_bands=2, n_channels=2, d_max=7, timestep_seconds=timestep
    )

    evidence = synapse._phase_slope_evidence(
        phase, torch.ones_like(phase), frequencies, n_delays=8
    )

    assert int(evidence[0, 0, 0, 1].argmax()) == delay
    assert int(evidence[1, 1, 0, 1].argmax()) == delay


def test_sparse_route_selection_does_not_force_declared_density() -> None:
    synapse = DelayPhaseGraphSynapse(
        n_bands=2, n_channels=4, d_max=2, graph_sparsity=0.25, n_delay_experts=2
    )

    with torch.no_grad():
        synapse.edge_existence_logits.fill_(-20.0)
    selection = synapse.effective_edge_selection_expert().detach()

    torch.testing.assert_close(selection, torch.zeros_like(selection))


def test_cross_band_pac_does_not_subtract_target_carrier_phase() -> None:
    synapse = DelayPhaseGraphSynapse(
        n_bands=2,
        n_channels=2,
        d_max=1,
        graph_sparsity=0.0,
        delay_gate_init=20.0,
        n_delay_experts=1,
        expert_topk=1,
    )
    synapse.eval()
    with torch.no_grad():
        synapse.edge_logits.fill_(2.0)
        synapse.fast_delay_logits.fill_(-20.0)
        synapse.fast_delay_logits[..., 0] = 20.0
        synapse.delay_confidence_logits.fill_(20.0)
        synapse.fast_fraction_raw.fill_(-30.0)
        synapse.slow_modulation_raw.fill_(-30.0)
        route = torch.zeros_like(synapse.route_mask)
        route[1, 0, 0, 1] = 1.0
        synapse.set_edge_weight_override(route)
    carrier = torch.zeros(1, 2, 2, 8)
    carrier[:, 0, 1] = 1.0
    confidence = torch.ones_like(carrier)
    phase_a = torch.zeros_like(carrier)
    phase_b = phase_a.clone()
    phase_b[:, 1, 0] = torch.linspace(-torch.pi, torch.pi, 8)

    current_a, _ = synapse(carrier, phase_a, confidence)
    current_b, _ = synapse(carrier, phase_b, confidence)

    torch.testing.assert_close(current_a[:, 1], current_b[:, 1])


def test_pointwise_sew_clif_cannot_fire_before_input_arrives() -> None:
    block = SEWCLIFBlock(channels=4)
    spikes = torch.zeros(2, 4, 16)
    spikes[..., 7] = 1.0

    output, _ = block(spikes)

    torch.testing.assert_close(output[..., :7], torch.zeros_like(output[..., :7]))


def test_sew_decoder_exports_only_binary_respikes_and_matching_membrane() -> None:
    model = DPCSNN(
        n_classes=2,
        n_channels=3,
        n_bands=2,
        graph_timesteps=64,
        timesteps=64,
        latent_nodes=3,
        snn_channels=8,
        decoder_layers=3,
        band_edges_hz=[[6.0, 10.0], [14.0, 20.0]],
        delay_evidence_band_edges_hz=[[6.0, 10.0], [14.0, 20.0]],
        task_tmax=1.0,
    ).eval()

    output = model(x=torch.randn(2, 3, 128))
    aux = output["aux"]

    assert len(model.spiking_blocks) == 2
    assert model.output_lif is not None
    assert aux["binary_spike_layers"].shape[1] == 4
    assert torch.all(
        (aux["binary_spike_layers"] == 0.0)
        | (aux["binary_spike_layers"] == 1.0)
    )
    assert torch.all((aux["hidden_spikes"] == 0.0) | (aux["hidden_spikes"] == 1.0))
    torch.testing.assert_close(aux["hidden_spikes"], aux["binary_spike_layers"][:, -1])
    assert aux["membrane"].shape == aux["hidden_spikes"].shape


def test_delayed_directional_moments_retain_post_delay_time_location() -> None:
    moments = DelayedDirectionalMoments()
    early = torch.zeros(1, 1, 1, 2, 16)
    late = torch.zeros_like(early)
    early[..., 3] = 1.0
    late[..., 11] = 1.0

    early_stats = moments(early)
    late_stats = moments(late)

    assert late_stats[..., 0].item() > early_stats[..., 0].item()
    assert torch.isfinite(early_stats).all()
    assert torch.isfinite(late_stats).all()


def test_v38_cumulative_readouts_and_fine_pyramid_are_delay_routed() -> None:
    model = DPCSNN(
        n_classes=3,
        n_channels=3,
        n_bands=2,
        graph_timesteps=96,
        timesteps=96,
        temporal_pool_bins=64,
        latent_nodes=3,
        snn_channels=8,
        temporal_pool_channels=4,
        band_edges_hz=[[8.0, 13.0], [13.0, 30.0]],
        task_tmax=1.0,
        cumulative_readout_seconds=(0.5, 1.0, 2.0, 4.0),
    )

    out = model(x=torch.randn(2, 3, 128))

    assert model.temporal_pool_bins == (1, 2, 4, 8, 16, 32, 64)
    assert tuple(out["aux"]["cumulative_readout_seconds"].tolist()) == (0.5, 1.0)
    assert out["aux"]["cumulative_logits"].shape == (2, 2, 3)


def test_v39_model_preserves_baseline_for_signed_erd_ers() -> None:
    model = DPCSNN(
        n_classes=2,
        n_channels=3,
        n_bands=2,
        sfreq=128.0,
        epoch_tmin=-1.0,
        task_tmin=0.0,
        task_tmax=1.0,
        graph_timesteps=128,
        timesteps=128,
        latent_nodes=3,
        snn_channels=8,
        band_edges_hz=[[6.0, 10.0], [14.0, 20.0]],
        cumulative_readout_seconds=(),
    ).eval()

    output = model(x=torch.randn(2, 3, 256))

    assert float(output["aux"]["baseline_samples"]) == 128.0
    assert torch.isfinite(output["aux"]["signed_erd_ers_mean"])
    assert output["aux"]["cumulative_logits"].shape[1] == 0


def test_v39_uses_fixed_training_gain_without_trialwise_rescaling() -> None:
    model = DPCSNN(
        n_classes=2,
        n_channels=3,
        n_bands=2,
        sfreq=128.0,
        task_tmax=1.0,
        graph_timesteps=128,
        timesteps=128,
        latent_nodes=3,
        snn_channels=8,
        band_edges_hz=[[6.0, 10.0], [14.0, 20.0]],
        cumulative_readout_seconds=(),
    ).eval()
    model.set_train_fitted_route_rms(0.25)
    sample = torch.randn(1, 3, 128)

    alone = model(x=sample)["aux"]["current"]
    batched = model(x=torch.cat((sample, 100.0 * torch.randn_like(sample))))["aux"]["current"]

    torch.testing.assert_close(alone[0], batched[0])
    assert float(model.train_route_rms) == pytest.approx(0.25)


def test_graph_rate_requires_transition_above_highest_band() -> None:
    with pytest.raises(ValueError, match="anti-alias transition"):
        DPCSNN(
            n_classes=2,
            n_channels=3,
            n_bands=2,
            graph_timesteps=320,
            task_tmax=4.0,
            band_edges_hz=[[8.0, 13.0], [30.0, 40.0]],
        )


def test_context_router_is_sparse_and_trial_dependent() -> None:
    synapse = DelayPhaseGraphSynapse(
        n_bands=2, n_channels=3, d_max=1, n_delay_experts=4, expert_topk=2, context_dim=6
    )
    with torch.no_grad():
        synapse.expert_router.weight.copy_(torch.eye(4, 6))
    context = torch.tensor([[5.0, 4.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 5.0, 4.0, 0.0, 0.0]])

    mixture = synapse.expert_mixture(context, batch_size=2)

    assert torch.all((mixture > 0).sum(dim=1) == 2)
    assert not torch.equal(mixture[0], mixture[1])
    torch.testing.assert_close(mixture.sum(dim=1), torch.ones(2))


def test_router_warmup_explores_every_expert() -> None:
    torch.manual_seed(7)
    synapse = DelayPhaseGraphSynapse(
        n_bands=2, n_channels=3, d_max=1, n_delay_experts=4, expert_topk=2, context_dim=6
    )
    synapse.train()

    mixture = synapse.expert_mixture(torch.zeros(128, 6), batch_size=128, explore=True)
    usage = (mixture > 0).float().sum(0)

    assert torch.all(usage > 0)
    assert torch.all((mixture > 0).sum(1) == 2)


def test_v38_uses_fixed_narrow_delay_evidence_and_fixed_spatial_projector() -> None:
    model = DPCSNN(
        n_classes=3,
        n_channels=3,
        n_bands=2,
        graph_timesteps=96,
        timesteps=96,
        latent_nodes=3,
        snn_channels=8,
        temporal_pool_channels=4,
        band_edges_hz=[[8.0, 13.0], [13.0, 30.0]],
        task_tmax=1.0,
        cumulative_readout_seconds=(0.5, 1.0, 2.0, 4.0),
    )

    out = model(x=torch.randn(2, 3, 128))
    assert model.delay_evidence_filterbank.n_bands == 12
    assert all(not parameter.requires_grad for parameter in model.delay_evidence_filterbank.parameters())
    assert all(not parameter.requires_grad for parameter in model.evidence_spatial.parameters())
    assert tuple(out["aux"]["delay_evidence_geometry"].tolist()) == (12.0, 3.0)
    assert "online_route_log_odds" in out["aux"]
    assert "delay_log_bayes_factor" not in out["aux"]
    assert out["aux"]["cumulative_logits"].shape[1] == 2
    assert out["aux"]["connectivity_prior"].shape == (2, 2, 3, 3)


def test_sparse_expert_router_keeps_dense_recovery_gradients() -> None:
    synapse = DelayPhaseGraphSynapse(
        n_bands=2, n_channels=3, d_max=1, n_delay_experts=4, expert_topk=1, context_dim=6
    )
    context = torch.randn(5, 6)

    mixture = synapse.expert_mixture(context, batch_size=5)
    (mixture[:, 0].sum()).backward()

    assert torch.all((mixture.detach() > 0).sum(dim=1) == 1)
    assert synapse.expert_router.weight.grad is not None
    assert torch.all(synapse.expert_router.weight.grad.abs().sum(dim=1) > 0)


def test_v34_temporal_pyramid_receives_only_delay_routed_activity() -> None:
    model = DPCSNN(
        n_classes=2, n_channels=3, n_bands=2, graph_timesteps=32, timesteps=8,
        preserve_graph_rate=True, temporal_pool_bins=8, latent_nodes=3, snn_channels=8,
        n_delay_experts=2, delay_expert_topk=1, band_edges_hz=[[8.0, 13.0], [13.0, 30.0]],
    ).eval()
    model.synapse.set_edge_weight_override(torch.zeros_like(model.synapse.route_mask))
    first = model(x=torch.randn(2, 3, 128))
    second = model(x=torch.randn(2, 3, 128) * 5.0)

    assert first["aux"]["hidden_spikes"].shape[-1] == 32
    assert first["aux"]["temporal_pyramid_bins"].tolist() == [1.0, 2.0, 4.0, 8.0]
    torch.testing.assert_close(first["logits"], second["logits"])


def test_zero_delay_control_preserves_parameter_capacity() -> None:
    full = DPCSNN(n_classes=2, n_channels=3, n_bands=2, latent_nodes=3, graph_timesteps=8)
    control = DPCSNN(
        n_classes=2, n_channels=3, n_bands=2, latent_nodes=3, graph_timesteps=8,
        force_zero_delay=True,
    )

    assert sum(parameter.numel() for parameter in full.parameters()) == sum(
        parameter.numel() for parameter in control.parameters()
    )
    torch.testing.assert_close(control.synapse.learned_delay(), torch.zeros_like(control.synapse.learned_delay()))
    torch.testing.assert_close(
        control.synapse.fast_delay_prob_expert()
        * (1.0 - control.synapse.route_mask)[None, ..., None],
        torch.zeros_like(control.synapse.fast_delay_prob_expert()),
    )


def test_v38_training_stage_freezes_posterior_but_trains_router() -> None:
    model = DPCSNN(n_classes=2, n_channels=3, n_bands=2, latent_nodes=3, graph_timesteps=8)

    model.set_training_stage("representation")
    assert not any(parameter.requires_grad for parameter in model.residual_delay_parameters())
    assert all(parameter.requires_grad for parameter in model.synapse.expert_router.parameters())
    assert model.readout.weight.requires_grad

    model.set_training_stage("joint")
    assert all(parameter.requires_grad for parameter in model.residual_delay_parameters())
    model.capture_delay_anchor()
    model.begin_task_training()
    assert model.synapse.delay_temperature == model.synapse.initial_delay_temperature
    model.set_training_progress(1.0)
    assert model.synapse.delay_temperature == pytest.approx(model.synapse.min_delay_temperature)
    assert model.delay_anchor_kl_scale == 0.0
    restored = DPCSNN(n_classes=2, n_channels=3, n_bands=2, latent_nodes=3, graph_timesteps=8)
    restored.load_state_dict(model.state_dict())
    assert restored.synapse.delay_temperature == pytest.approx(model.synapse.min_delay_temperature)

    fixed = DPCSNN(
        n_classes=2,
        n_channels=3,
        n_bands=2,
        latent_nodes=3,
        graph_timesteps=8,
        freeze_delay_posterior=True,
    )
    fixed.set_training_stage("joint")
    assert not any(parameter.requires_grad for parameter in fixed.synapse.posterior_parameters())
    assert fixed.synapse.edge_logits.requires_grad
    assert fixed.synapse.delay_confidence_logits.requires_grad
    assert fixed.synapse.phase_pref.requires_grad


def test_v42_representation_and_joint_never_update_tau_and_theta_together() -> None:
    model = DPCSNN(
        n_classes=2,
        n_channels=3,
        n_bands=2,
        latent_nodes=3,
        graph_timesteps=8,
        freeze_delay_after_pretrain=True,
    )
    posterior = model.synapse.posterior_parameters()

    model.begin_task_training()
    model.set_training_stage("representation")
    assert bool(model.synapse.phase_residual_enabled)
    assert not any(parameter.requires_grad for parameter in posterior)
    assert model.synapse.phase_pref.requires_grad

    model.set_training_stage("joint")
    assert not any(parameter.requires_grad for parameter in posterior)
    assert model.synapse.phase_pref.requires_grad


def test_v42_representation_checkpoint_cannot_replace_phase_stage_state() -> None:
    source = DPCSNN(
        n_classes=2, n_channels=3, n_bands=2, latent_nodes=3, graph_timesteps=8
    )
    source.begin_task_training()
    target = DPCSNN(
        n_classes=2, n_channels=3, n_bands=2, latent_nodes=3, graph_timesteps=8
    )

    target.load_representation_state(source.state_dict())

    assert bool(source.synapse.phase_residual_enabled)
    assert not bool(target.synapse.phase_residual_enabled)


def test_v42_shared_filterbank_is_identical_and_frozen() -> None:
    bands = [[6.0, 10.0], [14.0, 20.0]]
    model = DPCSNN(
        n_classes=2,
        n_channels=3,
        n_bands=2,
        latent_nodes=3,
        graph_timesteps=64,
        band_edges_hz=bands,
        delay_evidence_band_edges_hz=bands,
        freeze_shared_filterbank=True,
        task_tmax=1.0,
    )

    for carrier, evidence in zip(
        model.filterbank.parameters(), model.delay_evidence_filterbank.parameters()
    ):
        torch.testing.assert_close(carrier, evidence)
        assert not carrier.requires_grad
        assert not evidence.requires_grad


def test_v42_keeps_128_bins_and_directional_features_after_transport() -> None:
    bands = [[6.0, 10.0], [14.0, 20.0]]
    model = DPCSNN(
        n_classes=2,
        n_channels=2,
        n_bands=2,
        latent_nodes=2,
        graph_timesteps=128,
        temporal_pool_bins=128,
        snn_channels=8,
        temporal_pool_channels=2,
        band_edges_hz=bands,
        delay_evidence_band_edges_hz=bands,
        preserve_transport_band_pairs=True,
        delayed_stat_channels=2,
        delayed_directional_moments=True,
        task_tmax=1.0,
    )

    assert model.temporal_pool_bins[-1] == 128
    expected_pair_features = 2 * 2 * (2 + 2 + DelayedDirectionalMoments.out_features)
    assert model.current_encoder[0].in_channels == expected_pair_features


def test_v42_identifier_rejects_coordinate_or_delay_contract_drift() -> None:
    with pytest.raises(ValueError, match="freeze_shared_physical_basis"):
        DPCSNN(
            n_classes=2,
            n_channels=2,
            n_bands=1,
            architecture_version="dpc_snn_v4_2_identifiable_decoder",
        )


def test_rejected_route_floor_has_nonvanishing_reopening_gradient() -> None:
    synapse = DelayPhaseGraphSynapse(
        n_bands=1,
        n_channels=2,
        d_max=2,
        n_delay_experts=1,
        rejected_route_prior_floor=0.02,
    )
    route = torch.zeros_like(synapse.route_mask)
    positive = torch.full((*route.shape, 3), 1.0 / 3.0)
    synapse.load_fold_local_evidence_prior(route, positive)
    with torch.no_grad():
        for name in (
            "edge_existence_logits",
            "edge_existence_target_logits",
            "edge_existence_source_logits",
            "edge_existence_rank_logits",
        ):
            getattr(synapse, name).zero_()

    probability = synapse.delay_route_probability_expert()[0, 0, 0, 0, 1]
    probability.backward()

    gradient = synapse.edge_existence_logits.grad
    assert probability.item() == pytest.approx(0.02, abs=1e-5)
    assert gradient is not None
    assert gradient[0, 0, 0].abs() > 1e-2


def test_delay_pretraining_fixes_phase_intercept_and_excludes_router() -> None:
    model = DPCSNN(n_classes=2, n_channels=3, n_bands=2, latent_nodes=3, graph_timesteps=8)
    selected = {id(parameter) for parameter in model.delay_pretraining_parameters()}

    assert id(model.synapse.edge_logits) in selected
    assert id(model.synapse.edge_existence_logits) in selected
    assert id(model.synapse.delay_confidence_logits) in selected
    assert id(model.synapse.fast_delay_logits) in selected
    assert id(model.synapse.phase_pref) not in selected
    assert all(id(parameter) not in selected for parameter in model.synapse.expert_router.parameters())
    assert not bool(model.synapse.phase_residual_enabled)
    model.begin_task_training()
    assert bool(model.synapse.phase_residual_enabled)


def test_delay_pretraining_loss_actually_supervises_edge_and_confidence() -> None:
    model = DPCSNN(
        n_classes=2,
        n_channels=3,
        n_bands=2,
        latent_nodes=3,
        graph_timesteps=64,
        snn_channels=8,
        band_edges_hz=[[6.0, 10.0], [14.0, 20.0]],
        delay_evidence_band_edges_hz=[[6.0, 10.0], [14.0, 20.0]],
    )
    route = model.synapse.route_mask.clone()
    positive = torch.ones(*route.shape, model.synapse.d_max + 1)
    positive = positive / positive.shape[-1]
    model.load_fold_local_evidence_prior(route, positive, torch.zeros_like(route))
    output = model(x=torch.randn(2, 3, 64))

    model.delay_pretraining_loss(output["aux"]).backward()

    assert model.synapse.edge_logits.grad is not None
    assert torch.isfinite(model.synapse.edge_logits.grad).all()
    assert model.synapse.edge_logits.grad.abs().sum() > 0
    assert model.synapse.delay_confidence_logits.grad is not None
    assert model.synapse.delay_confidence_logits.grad.abs().sum() > 0


def test_route_gain_has_no_train_eval_running_state() -> None:
    torch.manual_seed(4)
    model = DPCSNN(
        n_classes=2, n_channels=3, n_bands=2, latent_nodes=3, graph_timesteps=16,
        snn_channels=8, reference_augmentation_prob=0.0, n_delay_experts=2,
    )
    model.begin_task_training()
    x = torch.randn(2, 3, 64)
    model.train()
    train_aux = model(x=x)["aux"]
    model.eval()
    eval_aux = model(x=x)["aux"]

    torch.testing.assert_close(train_aux["current"], eval_aux["current"])
    torch.testing.assert_close(train_aux["hidden_spikes"], eval_aux["hidden_spikes"])


def test_paired_differences_pair_by_dataset_protocol_subject_seed() -> None:
    rows = [
        {"dataset": "a", "protocol": "p", "subject": "s1", "seed": 0, "model": "dpc_snn", "accuracy": 0.5},
        {"dataset": "a", "protocol": "p", "subject": "s1", "seed": 0, "model": "control", "accuracy": 0.4},
        {"dataset": "b", "protocol": "p", "subject": "s1", "seed": 0, "model": "dpc_snn", "accuracy": 0.9},
    ]

    diffs = paired_differences(rows, "dpc_snn", "control", "accuracy")

    np.testing.assert_allclose(diffs, np.asarray([0.1]))


def test_classification_metrics_include_balanced_accuracy() -> None:
    metrics = classification_metrics(np.asarray([0, 0, 1, 1]), np.asarray([0, 0, 0, 1]), n_classes=2)

    assert metrics["accuracy"] == 0.75
    assert metrics["balanced_accuracy"] == 0.75


def test_paired_prediction_comparison_reports_exact_mcnemar_counts() -> None:
    labels = np.zeros(5, dtype=np.int64)
    first = np.asarray([0, 0, 0, 0, 1])
    second = np.asarray([1, 1, 1, 0, 0])

    comparison = paired_prediction_comparison(labels, first, second)

    assert comparison["first_only_correct"] == 3
    assert comparison["second_only_correct"] == 1
    assert comparison["discordant_predictions"] == 4
    assert comparison["exact_mcnemar_p"] == pytest.approx(0.625)
    assert exact_mcnemar_p(12, 3) == pytest.approx(0.03515625)


def test_mixed_effects_accepts_csv_string_metrics() -> None:
    rows = [
        {"dataset": "d", "subject": "s1", "model": "dpc_snn", "accuracy": "0.75"},
        {"dataset": "d", "subject": "s1", "model": "eegnet", "accuracy": "0.50"},
    ]

    result = mixed_effects_or_fallback(rows, metric="accuracy")

    assert result
