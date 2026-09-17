from __future__ import annotations

import torch

from dpc_snn.models.v8_delay_residual_expert import (
    build_v8_delay_residual_expert,
)


def _prior() -> dict[str, torch.Tensor]:
    return {
        "source_band": torch.tensor([0]),
        "source_node": torch.tensor([0]),
        "target_band": torch.tensor([0]),
        "target_node": torch.tensor([1]),
        "delay_probability": torch.tensor([[0.0, 1.0, 0.0]]),
        "fractional_target": torch.tensor([0.25]),
        "route_weight": torch.tensor([1.0]),
        "route_confidence": torch.tensor([1.0]),
        "phase_preference": torch.tensor([0.0]),
        "amplitude_scale": torch.tensor([1.0]),
    }


def _model(variant: str):
    model = build_v8_delay_residual_expert(
        variant,
        n_bands=2,
        n_nodes=3,
        maximum_routes=4,
        maximum_delay=2,
        signal_mode="slow_envelope",
        decoder_channels=8,
        decoder_layers=1,
        sfreq=5.0,
        temporal_decimation=1,
        endpoint_seconds=(1.0, 2.0, 4.0),
        readout_features=12,
        dropout=0.0,
    )
    model.load_fold_prior(**_prior())
    model.set_input_gain(torch.ones(2, 3))
    return model


def test_full_and_matched_zero_delay_currents_are_both_nonzero() -> None:
    model = _model("sew_clif").eval()
    real = torch.randn(2, 2, 3, 20)
    fast = torch.complex(real, torch.randn_like(real))
    slow = torch.randn(2, 2, 3, 10)
    with torch.no_grad():
        full = model(fast, slow, delay_override="full")
        zero = model(fast, slow, delay_override="zero")
    assert full.logits.shape == (2, 4)
    assert full.prefix_logits.shape == (2, 3, 4)
    assert torch.count_nonzero(full.physical_current) > 0
    assert torch.count_nonzero(zero.physical_current) > 0
    assert not torch.equal(full.physical_current, zero.physical_current)
    assert full.binary_spikes


def test_matched_ann_and_sew_clif_have_identical_parameter_shapes() -> None:
    ann = _model("ann_sew")
    snn = _model("sew_clif")
    assert ann.parameter_count == snn.parameter_count
    assert sorted(parameter.shape for parameter in ann.parameters()) == sorted(
        parameter.shape for parameter in snn.parameters()
    )
