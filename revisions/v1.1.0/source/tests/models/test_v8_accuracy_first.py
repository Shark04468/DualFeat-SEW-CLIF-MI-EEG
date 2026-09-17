from __future__ import annotations

import ast
import inspect
from pathlib import Path

import torch
import yaml

import dpc_snn.models.build as build_module
from dpc_snn.models.build import build_model
from dpc_snn.models.v62_snn_decoder import CausalDepthwiseSEWBlock
from dpc_snn.models.v62_spatial import BCI2A_CHANNEL_NAMES
from dpc_snn.models.v8_accuracy_first import V8AccuracyFirstModel


TEST_BANDS = ((6.0, 10.0), (10.0, 14.0), (14.0, 20.0))
ROOT = Path(__file__).resolve().parents[2]


def _model(kind: str = "ann", residual_mode: str = "sew_add") -> V8AccuracyFirstModel:
    return V8AccuracyFirstModel(
        n_bands=3,
        n_latent_nodes=4,
        band_edges_hz=TEST_BANDS,
        task_tmax=1.0,
        analytic_taps=33,
        envelope_taps=9,
        fast_decimation=1,
        spatial_rank=2,
        temporal_dilations=(1, 2),
        temporal_depth=1,
        decoder_kind=kind,
        decoder_residual_mode=residual_mode,
        decoder_layers=1,
        decoder_channels=8,
        endpoint_seconds=(0.5, 1.0),
        statistical_spatial_filters=2,
        statistical_features=12,
        statistical_segments=2,
        covariance_features_per_band=2,
        fusion_features=16,
        dropout=0.0,
        parameter_ceiling=50_000,
    )


def test_v8_physical_basis_is_frozen_identity_and_separate_from_classifier_nodes() -> None:
    model = _model()
    weight = model.physical_basis.weight()
    identity = torch.eye(22)[None].expand(3, -1, -1)
    assert torch.equal(weight, identity)
    assert model.physical_basis.frozen
    assert not model.classifier_spatial.frozen
    assert model.physical_basis.n_nodes == 22
    assert model.classifier_spatial.n_nodes == 4

    fingerprint = model.physical_frontend_fingerprint()
    with torch.no_grad():
        model.classifier_spatial.shared_delta.add_(0.1)
    assert model.physical_frontend_fingerprint() == fingerprint
    model.architecture_version = "downstream_only_test_revision"
    assert model.physical_frontend_fingerprint() == fingerprint


def test_v8_forward_has_causal_prefixes_and_finite_regularizers() -> None:
    torch.manual_seed(7)
    model = _model("ann").eval()
    x = torch.randn(2, 22, 500) * 1e-5
    with torch.no_grad():
        output = model(x)
    assert output["logits"].shape == (2, 4)
    assert output["prefix_logits"].shape == (2, 2, 4)
    assert torch.isfinite(output["prefix_logits"]).all()
    assert torch.isfinite(output["spatial_orthogonality_loss"])
    assert torch.isfinite(output["statistical_orthogonality_loss"])
    assert output["aux"]["physical_fast"].shape == (2, 3, 22, 250)
    assert output["aux"]["latent_fast"].shape == (2, 3, 4, 250)
    assert output["aux"]["binary_spikes"] == ()


def test_v8_half_second_readout_cannot_see_later_task_samples() -> None:
    torch.manual_seed(11)
    model = _model("ann").eval()
    first = torch.randn(1, 22, 500) * 1e-5
    second = first.clone()
    # Epoch is -1..1 s. Samples from 0.5 s onward are unavailable to the
    # registered 0.5 s endpoint and must not change its logits.
    second[..., 375:] += torch.randn_like(second[..., 375:]) * 1e-4
    with torch.no_grad():
        first_logits = model(first)["prefix_logits"][:, 0]
        second_logits = model(second)["prefix_logits"][:, 0]
    assert torch.equal(first_logits, second_logits)


def test_v8_clif_counts_only_binary_threshold_outputs() -> None:
    torch.manual_seed(13)
    model = _model("clif").eval()
    with torch.no_grad():
        output = model(torch.randn(1, 22, 500) * 1e-5)
    binary = output["aux"]["binary_spikes"]
    assert len(binary) == 3
    assert all(bool(((value == 0) | (value == 1)).all()) for value in binary)
    assert torch.isfinite(output["firing_rate_loss"])
    assert output["aux"]["final_membrane"] is not None


def test_decoder_plain_and_sew_add_differ_only_by_the_residual_merge() -> None:
    torch.manual_seed(17)
    plain = CausalDepthwiseSEWBlock(
        4,
        kind="ann",
        decays=(0.65, 0.9),
        threshold=1.0,
        residual_mode="plain",
    ).eval()
    sew = CausalDepthwiseSEWBlock(
        4,
        kind="ann",
        decays=(0.65, 0.9),
        threshold=1.0,
        residual_mode="sew_add",
    ).eval()
    sew.load_state_dict(plain.state_dict())
    activity = torch.randn(2, 4, 16)
    with torch.no_grad():
        plain_merged, plain_branch, plain_membrane = plain(activity)
        sew_merged, sew_branch, sew_membrane = sew(activity)
    assert torch.equal(plain_merged, plain_branch)
    assert torch.equal(sew_merged, activity + sew_branch)
    assert torch.equal(plain_branch, sew_branch)
    assert torch.equal(plain_membrane, sew_membrane)


def test_v8_e4_decoder_controls_have_identical_trainable_capacity() -> None:
    variants = (
        _model("ann", "sew_add"),
        _model("plif", "plain"),
        _model("clif", "plain"),
        _model("clif", "sew_add"),
    )
    counts = {model.trainable_parameter_count for model in variants}
    signatures = {
        tuple(sorted(tuple(parameter.shape) for parameter in model.parameters()))
        for model in variants
    }
    assert len(counts) == 1
    assert len(signatures) == 1
    assert variants[2].decoder is not None
    assert variants[2].decoder.decoder_residual_mode == "plain"
    assert variants[3].decoder is not None
    assert variants[3].decoder.decoder_residual_mode == "sew_add"


def test_v8_build_dispatch_uses_registered_configuration() -> None:
    model = build_model(
        "v8_accuracy_first",
        {
            "n_channels": 22,
            "n_classes": 4,
            "channel_names": list(BCI2A_CHANNEL_NAMES),
            "n_bands": 3,
            "n_latent_nodes": 4,
            "band_edges_hz": [list(value) for value in TEST_BANDS],
            "task_tmax": 1.0,
            "analytic_taps": 33,
            "envelope_taps": 9,
            "fast_decimation": 1,
            "spatial_rank": 2,
            "temporal_dilations": [1, 2],
            "temporal_depth": 1,
            "decoder_kind": "plif",
            "decoder_layers": 1,
            "decoder_channels": 8,
            "endpoint_seconds": [0.5, 1.0],
            "statistical_spatial_filters": 2,
            "statistical_features": 12,
            "statistical_segments": 2,
            "covariance_features_per_band": 2,
            "fusion_features": 16,
            "dropout": 0.0,
            "parameter_ceiling": 50_000,
        },
    )
    assert isinstance(model, V8AccuracyFirstModel)
    assert model.decoder is not None
    assert model.decoder.decoder_kind == "plif"


def test_v8_build_dispatch_forwards_every_delay_stage_switch() -> None:
    model = build_model(
        "v8_accuracy_first",
        {
            "n_channels": 22,
            "n_classes": 4,
            "n_bands": 3,
            "n_latent_nodes": 4,
            "band_edges_hz": [list(value) for value in TEST_BANDS],
            "task_tmax": 1.0,
            "analytic_taps": 33,
            "envelope_taps": 9,
            "fast_decimation": 1,
            "spatial_rank": 2,
            "temporal_dilations": [1, 2],
            "temporal_depth": 1,
            "decoder_kind": "ann",
            "decoder_layers": 1,
            "decoder_channels": 8,
            "endpoint_seconds": [0.5, 1.0],
            "statistical_spatial_filters": 2,
            "statistical_features": 12,
            "statistical_segments": 2,
            "covariance_features_per_band": 2,
            "delay_auxiliary_enabled": True,
            "delay_signal_mode": "slow_envelope",
            "delay_contextual_residual_enabled": True,
            "delay_contextual_residual_bound": 0.375,
            "delay_phase_residual_enabled": True,
            "delay_phase_residual_bound": 0.125,
            "fusion_features": 16,
            "dropout": 0.0,
            "parameter_ceiling": 80_000,
        },
    )
    assert model.delay_auxiliary is not None
    assert model.delay_auxiliary.signal_mode == "slow_envelope"
    assert model.delay_auxiliary.contextual_residual_enabled
    assert model.delay_auxiliary.contextual_residual_bound == 0.375
    assert model.delay_auxiliary.phase_residual_enabled
    assert model.delay_auxiliary.phase_residual_bound == 0.125


def test_v8_build_dispatch_covers_the_full_constructor_signature() -> None:
    tree = ast.parse(inspect.getsource(build_module.build_model))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "V8AccuracyFirstModel"
    ]
    assert len(calls) == 1
    forwarded = {keyword.arg for keyword in calls[0].keywords}
    expected = set(inspect.signature(V8AccuracyFirstModel).parameters) - {"self"}
    assert forwarded == expected


def test_v8_experiment_model_configs_expose_runner_endpoint_metadata() -> None:
    for name in (
        "v8_accuracy_first.yaml",
        "v8_information_anchor.yaml",
        "v8_information_anchor_trainable_spatial.yaml",
        "v8_information_fbc_spatial.yaml",
    ):
        config = yaml.safe_load(
            (ROOT / "configs" / "models" / name).read_text(encoding="utf-8")
        )
        assert config["endpoint_seconds"][-1] == config["task_tmax"]
        model = build_model("v8_accuracy_first", config)
        assert tuple(config["endpoint_seconds"]) == model.endpoint_seconds


def test_v8_linear_information_anchor_preserves_carrier_log_variance() -> None:
    torch.manual_seed(23)
    model = V8AccuracyFirstModel(
        n_bands=3,
        n_latent_nodes=4,
        band_edges_hz=TEST_BANDS,
        task_tmax=1.0,
        analytic_taps=33,
        envelope_taps=9,
        fast_decimation=1,
        spatial_rank=2,
        endpoint_seconds=(0.5, 1.0),
        statistical_spatial_filters=22,
        statistical_segments=2,
        statistical_components=("carrier_log_variance",),
        statistical_encoder_kind="fbc",
        statistical_variance_transform="log",
        statistical_spatial_initialization="identity",
        statistical_spatial_trainable=False,
        use_covariance_branch=False,
        use_temporal_branch=False,
        fusion_kind="linear",
        dropout=0.0,
        parameter_ceiling=50_000,
    ).eval()
    assert model.statistical_branch is not None
    expected_weight = torch.eye(22)[None].expand(3, -1, -1)
    assert torch.equal(model.statistical_branch.weight(), expected_weight)
    assert not model.statistical_branch.spatial_weight.requires_grad
    assert model.fusion_kind == "linear"
    assert model.classifier.in_features == 3 * 22 * 2

    fast = torch.complex(torch.randn(2, 3, 22, 250), torch.randn(2, 3, 22, 250))
    slow = torch.randn(2, 3, 22, 125)
    with torch.no_grad():
        output = model.forward_rate_features(fast, slow)
    observed = output["aux"]["statistical_output"].raw_statistics
    expected = []
    for stop in (125, 250):
        chunks = torch.tensor_split(fast.real[..., :stop], 2, dim=-1)
        expected.append(
            torch.stack(
                [chunk.var(dim=-1, unbiased=False).clamp(1e-6, 1e6).log() for chunk in chunks],
                dim=-1,
            ).flatten(1)
        )
    assert torch.allclose(observed, torch.stack(expected, dim=1))
