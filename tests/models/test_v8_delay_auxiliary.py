from __future__ import annotations

import numpy as np
import pytest
import torch

from dpc_snn.experiments.v8_protocol import (
    assert_delay_control_contract,
    mapping_sha256,
)
from dpc_snn.models.v8_accuracy_first import V8AccuracyFirstModel
from dpc_snn.models.v8_delay_auxiliary import SparsePhysicalDelayAuxiliary


def _prior(maximum_delay: int = 2) -> dict[str, torch.Tensor]:
    probability = torch.zeros(2, maximum_delay + 1)
    probability[0, 1] = 1.0
    probability[1, 2] = 1.0
    return {
        "source_band": torch.tensor([0, 1]),
        "source_node": torch.tensor([0, 2]),
        "target_band": torch.tensor([0, 1]),
        "target_node": torch.tensor([1, 1]),
        "delay_probability": probability,
        "fractional_target": torch.tensor([0.25, 0.5]),
        "route_weight": torch.tensor([1.0, -0.75]),
        "route_confidence": torch.tensor([0.8, 0.9]),
        "phase_preference": torch.tensor([0.0, 0.2]),
        "amplitude_scale": torch.tensor([0.5, 0.75]),
    }


def test_sparse_delay_zero_intervention_retains_nonzero_matched_transport() -> None:
    torch.manual_seed(3)
    module = SparsePhysicalDelayAuxiliary(
        2, 3, maximum_routes=4, maximum_delay=2
    )
    module.load_fold_prior(**_prior())
    physical = torch.complex(torch.randn(2, 2, 3, 20), torch.randn(2, 2, 3, 20))
    fingerprint = module.prior_fingerprint()
    zero = module(physical, override="zero")
    off = module(physical, override="off")
    full = module(physical, override="full")

    assert torch.count_nonzero(zero.physical_current) > 0
    assert torch.count_nonzero(off.physical_current) == 0
    assert not torch.equal(zero.physical_current, off.physical_current)
    assert torch.count_nonzero(full.physical_current) > 0
    assert not torch.equal(full.physical_current, zero.physical_current)
    torch.testing.assert_close(
        zero.transport_probability,
        torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    torch.testing.assert_close(zero.transport_fractional_delay, torch.zeros(2))
    torch.testing.assert_close(full.transport_probability, module.delay_probability[:2])
    torch.testing.assert_close(full.transport_fractional_delay, module.fractional_target[:2])
    assert module.prior_fingerprint() == fingerprint
    assert zero.active_routes == full.active_routes == 2


def test_point_zero_prior_makes_full_and_zero_transport_bit_exact() -> None:
    prior = _prior()
    probability = torch.zeros_like(prior["delay_probability"])
    probability[:, 0] = 1.0
    prior["delay_probability"] = probability
    prior["fractional_target"] = torch.zeros(2)
    module = SparsePhysicalDelayAuxiliary(2, 3, maximum_routes=4, maximum_delay=2)
    module.load_fold_prior(**prior)
    physical = torch.complex(torch.randn(2, 2, 3, 20), torch.randn(2, 2, 3, 20))
    full = module(physical, override="full")
    zero = module(physical, override="zero")
    torch.testing.assert_close(full.physical_current, zero.physical_current, rtol=0, atol=0)
    torch.testing.assert_close(full.route_current, zero.route_current, rtol=0, atol=0)


def test_sparse_delay_is_causal_and_rejects_cross_band_routes_by_default() -> None:
    torch.manual_seed(5)
    module = SparsePhysicalDelayAuxiliary(
        2, 3, maximum_routes=4, maximum_delay=2
    )
    module.load_fold_prior(**_prior())
    first = torch.complex(torch.randn(1, 2, 3, 20), torch.randn(1, 2, 3, 20))
    second = first.clone()
    second[..., 12:] += torch.complex(
        torch.randn_like(second.real[..., 12:]),
        torch.randn_like(second.real[..., 12:]),
    )
    with torch.no_grad():
        first_current = module(first, override="full").physical_current[..., :12]
        second_current = module(second, override="full").physical_current[..., :12]
    assert torch.equal(first_current, second_current)

    invalid = _prior()
    invalid["target_band"] = torch.tensor([1, 1])
    with pytest.raises(ValueError, match="cross-band"):
        SparsePhysicalDelayAuxiliary(
            2, 3, maximum_routes=4, maximum_delay=2
        ).load_fold_prior(**invalid)


def test_slow_envelope_delay_is_real_causal_and_matched_zero_is_exact() -> None:
    torch.manual_seed(9)
    module = SparsePhysicalDelayAuxiliary(
        2,
        3,
        maximum_routes=4,
        maximum_delay=2,
        signal_mode="slow_envelope",
    )
    module.load_fold_prior(**_prior())
    fast = torch.complex(torch.randn(2, 2, 3, 24), torch.randn(2, 2, 3, 24))
    slow = torch.randn(2, 2, 3, 12)
    zero = module(fast, slow, override="zero")
    full = module(fast, slow, override="full")
    assert zero.signal_mode == "slow_envelope"
    assert torch.count_nonzero(zero.physical_current) > 0
    assert torch.count_nonzero(full.physical_current) > 0
    assert not full.route_current.is_complex()

    changed_fast = fast.clone()
    changed_slow = slow.clone()
    changed_fast[..., 16:] += 3.0
    changed_slow[..., 8:] -= 2.0
    prefix = module(fast, slow, override="full").physical_current[..., :16]
    changed_prefix = module(
        changed_fast, changed_slow, override="full"
    ).physical_current[..., :16]
    torch.testing.assert_close(prefix, changed_prefix, rtol=0.0, atol=0.0)


def test_contextual_and_phase_residuals_preserve_zero_control_and_receive_gradients() -> None:
    torch.manual_seed(11)
    fast = torch.complex(torch.randn(2, 2, 3, 24), torch.randn(2, 2, 3, 24))
    slow = torch.randn(2, 2, 3, 12)
    contextual = SparsePhysicalDelayAuxiliary(
        2,
        3,
        maximum_routes=4,
        maximum_delay=2,
        signal_mode="slow_envelope",
        contextual_residual_enabled=True,
    )
    contextual.load_fold_prior(**_prior())
    with torch.no_grad():
        contextual.context_route_raw[:2].fill_(0.2)
    contextual_zero = contextual(fast, slow, override="zero")
    contextual_full = contextual(fast, slow, override="full")
    assert torch.count_nonzero(contextual_zero.physical_current) > 0
    assert torch.count_nonzero(contextual_zero.dynamic_delay_residual) == 0
    assert torch.count_nonzero(contextual_full.dynamic_delay_residual) > 0
    contextual_full.physical_current.square().mean().backward()
    assert contextual.context_route_raw.grad is not None
    assert torch.isfinite(contextual.context_route_raw.grad).all()

    phase = SparsePhysicalDelayAuxiliary(
        2,
        3,
        maximum_routes=4,
        maximum_delay=2,
        signal_mode="fast_phase",
        phase_residual_enabled=True,
    )
    phase.load_fold_prior(**_prior())
    with torch.no_grad():
        phase.phase_residual_raw[:2].fill_(0.3)
    phase_zero = phase(fast, slow, override="zero")
    phase_full = phase(fast, slow, override="full")
    assert torch.count_nonzero(phase_zero.physical_current) > 0
    assert torch.count_nonzero(phase_full.phase_residual) == 2
    phase_full.physical_current.square().mean().backward()
    assert phase.phase_residual_raw.grad is not None
    assert torch.isfinite(phase.phase_residual_raw.grad).all()


def test_cross_band_delay_requires_and_retains_true_band_pair_axes() -> None:
    prior = _prior()
    prior["target_band"] = torch.tensor([1, 0])
    module = SparsePhysicalDelayAuxiliary(
        2,
        3,
        maximum_routes=4,
        maximum_delay=2,
        allow_cross_band=True,
        signal_mode="slow_envelope",
    )
    module.load_fold_prior(**prior)
    assert not torch.equal(module.source_band[:2], module.target_band[:2])


def _integrated_model() -> V8AccuracyFirstModel:
    model = V8AccuracyFirstModel(
        n_bands=3,
        n_latent_nodes=4,
        band_edges_hz=((6.0, 10.0), (10.0, 14.0), (14.0, 20.0)),
        task_tmax=1.0,
        analytic_taps=33,
        envelope_taps=9,
        fast_decimation=1,
        spatial_rank=2,
        temporal_dilations=(1, 2),
        temporal_depth=1,
        decoder_kind="ann",
        decoder_layers=1,
        decoder_channels=8,
        endpoint_seconds=(0.5, 1.0),
        statistical_spatial_filters=2,
        statistical_features=12,
        statistical_segments=2,
        covariance_features_per_band=2,
        delay_auxiliary_enabled=True,
        delay_maximum_routes=4,
        delay_maximum_samples=2,
        fusion_features=16,
        dropout=0.0,
        parameter_ceiling=50_000,
    )
    probability = torch.zeros(2, 3)
    probability[:, 1] = 1.0
    model.load_fold_delay_prior(
        source_band=torch.tensor([0, 1]),
        source_node=torch.tensor([7, 11]),
        target_band=torch.tensor([0, 1]),
        target_node=torch.tensor([11, 7]),
        delay_probability=probability,
        fractional_target=torch.tensor([0.25, 0.5]),
        route_weight=torch.tensor([1.0, -1.0]),
        route_confidence=torch.tensor([0.8, 0.9]),
        amplitude_scale=torch.tensor([1e-5, 1e-5]),
    )
    return model


def test_integrated_matched_zero_is_nonzero_and_state_is_frozen() -> None:
    torch.manual_seed(17)
    model = _integrated_model().eval()
    x = torch.randn(2, 22, 500) * 1e-5
    before = mapping_sha256(model.state_dict())
    with torch.no_grad():
        off_output = model(x, delay_override="off")
        zero_output = model(x, delay_override="zero")
        full_output = model(x, delay_override="full")
    after = mapping_sha256(model.state_dict())
    off = off_output["prefix_logits"]
    zero = zero_output["prefix_logits"]
    full = full_output["prefix_logits"]
    zero_delay = zero_output["aux"]["delay_auxiliary"]
    full_delay = full_output["aux"]["delay_auxiliary"]
    assert zero_delay is not None and full_delay is not None
    assert not torch.equal(off, zero)
    assert not torch.equal(full, zero)
    routing_fingerprint = model.delay_auxiliary.routing_fingerprint()
    assert_delay_control_contract(
        full_current=full_delay.physical_current.numpy(),
        zero_current=zero_delay.physical_current.numpy(),
        full_delay_probability=full_delay.transport_probability.numpy(),
        zero_delay_probability=zero_delay.transport_probability.numpy(),
        full_fractional_delay=full_delay.transport_fractional_delay.numpy(),
        zero_fractional_delay=zero_delay.transport_fractional_delay.numpy(),
        full_routing_fingerprint=routing_fingerprint,
        zero_routing_fingerprint=routing_fingerprint,
        shared_state_before=before,
        shared_state_after=after,
    )
    assert np.isfinite(full.numpy()).all()
