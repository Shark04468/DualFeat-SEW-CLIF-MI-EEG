from __future__ import annotations

import math

import torch

from dpc_snn.models.coupled_dual_delay import CoupledDualDelay
from dpc_snn.models.dasp_snn_v62 import _BoundedPhasePairAdapter


def _phase_pair_delay() -> CoupledDualDelay:
    delay = CoupledDualDelay(
        n_bands=1,
        n_nodes=2,
        route_rank=1,
        delay_rank=1,
        fast_max_delay=2,
        slow_max_delay=4,
        cross_band_enabled=False,
        residual_route_scale=1.0,
        slow_residual_route_scale=1.0,
        fast_residual_route_scale=0.0,
        phase_pair_current_enabled=True,
    )
    route = torch.zeros(1, 1, 2, 2)
    route[0, 0, 1, 0] = 0.90
    posterior = torch.zeros(1, 1, 2, 2, 5)
    posterior[..., 0] = 1.0
    posterior[0, 0, 1, 0] = 0.0
    posterior[0, 0, 1, 0, 2] = 1.0
    fraction = torch.zeros(1, 1, 2, 2)
    delay.load_fold_local_slow_prior(route, posterior, fraction)
    delay.load_fold_local_phase_amplitude_scale(torch.ones(1, 2))
    return delay


def _known_lag_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    fast_steps = 64
    fast_time = torch.arange(fast_steps, dtype=torch.float32)
    source = torch.exp(1j * (math.pi / 8.0) * fast_time)
    target = torch.zeros_like(source)
    target[4:] = source[:-4]
    fast = torch.zeros(1, 1, 2, fast_steps, dtype=torch.complex64)
    fast[0, 0, 0] = source
    fast[0, 0, 1] = target
    slow = torch.zeros(1, 1, 2, fast_steps // 2)
    return fast, slow


def test_phase_pair_current_is_exact_zero_for_matched_zero() -> None:
    delay = _phase_pair_delay()
    fast, slow = _known_lag_inputs()
    output = delay(
        fast,
        slow,
        slow_delay_override="zero",
        fast_delay_override="zero",
    )
    current = output.slow_phase_pair_delay_contrast_current
    assert current is not None
    assert torch.equal(current, torch.zeros_like(current))


def test_phase_pair_current_recovers_known_delayed_alignment() -> None:
    delay = _phase_pair_delay()
    fast, slow = _known_lag_inputs()
    output = delay(
        fast,
        slow,
        slow_delay_override="learned",
        fast_delay_override="zero",
    )
    current = output.slow_phase_pair_delay_contrast_current
    assert current is not None
    assert current.shape == slow.shape
    assert float(current[0, 0, 1, 8:].mean()) > 0.10
    assert torch.equal(current[0, 0, 0], torch.zeros_like(current[0, 0, 0]))


def test_phase_pair_adapter_has_no_zero_input_bypass_and_receives_gradient() -> None:
    adapter = _BoundedPhasePairAdapter(
        2,
        3,
        bound=0.25,
        initial_scale=0.05,
    )
    zero = torch.zeros(4, 2, 3, 10)
    assert torch.equal(adapter(zero), zero)

    current = torch.randn(4, 2, 3, 10)
    adapter(current).square().mean().backward()
    assert adapter.raw_scale.grad is not None
    assert bool(torch.isfinite(adapter.raw_scale.grad).all())
    assert float(adapter.raw_scale.grad.abs().sum()) > 0.0
