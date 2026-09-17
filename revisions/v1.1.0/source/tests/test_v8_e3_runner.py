from __future__ import annotations

import torch

from dpc_snn.models.v8_accuracy_first import V8AccuracyFirstModel
from scripts.run_v8_e3_static_delay import (
    _freeze_delay_only,
    _load_parent_state,
)


def _model(delay: bool) -> V8AccuracyFirstModel:
    return V8AccuracyFirstModel(
        n_bands=2,
        n_latent_nodes=4,
        band_edges_hz=((6, 10), (10, 14)),
        sfreq=32,
        epoch_tmin=-0.5,
        task_tmin=0.0,
        task_tmax=1.0,
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
        delay_auxiliary_enabled=delay,
        delay_maximum_routes=2,
        delay_maximum_samples=2,
        delay_fusion_initial=0.0,
        fusion_features=16,
        dropout=0.0,
        parameter_ceiling=100_000,
    )


def test_e3_loads_all_parent_state_and_freezes_only_delay_scalar(tmp_path) -> None:
    parent = _model(False)
    path = tmp_path / "parent.pt"
    torch.save(parent.state_dict(), path)
    child = _model(True)
    digest = _load_parent_state(child, path)
    assert len(digest) == 64
    for name, value in parent.state_dict().items():
        assert torch.equal(child.state_dict()[name], value)
    _freeze_delay_only(child)
    assert [name for name, value in child.named_parameters() if value.requires_grad] == [
        "delay_fusion_raw"
    ]


def test_e3_matched_zero_is_a_nonzero_routed_control_not_parent_reproduction() -> None:
    model = _model(True).eval()
    probability = torch.tensor([[0.0, 1.0, 0.0]])
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
    with torch.no_grad():
        model.delay_fusion_raw.fill_(0.2)
        fast = torch.complex(torch.randn(2, 2, 22, 16), torch.randn(2, 2, 22, 16))
        slow = torch.randn(2, 2, 22, 8)
        off = model.forward_rate_features(fast, slow, delay_override="off")
        zero = model.forward_rate_features(fast, slow, delay_override="zero")
    assert not torch.equal(off["logits"], zero["logits"])
    assert torch.count_nonzero(zero["aux"]["delay_auxiliary"].physical_current) > 0
