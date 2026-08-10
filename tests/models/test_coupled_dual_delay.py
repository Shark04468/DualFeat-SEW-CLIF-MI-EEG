from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.models.coupled_dual_delay import (  # noqa: E402
    CoupledDualDelay,
    fractional_causal_shift,
    posterior_fractional_causal_shift,
    upsample_delay_posterior_2x,
)


def test_fractional_delay_values_and_gradient() -> None:
    source = torch.arange(8, dtype=torch.float64).view(1, 1, -1)
    delay = torch.tensor([0.25], dtype=torch.float64, requires_grad=True)

    shifted = fractional_causal_shift(source, delay, max_delay=4)
    expected = 0.75 * source + 0.25 * torch.cat(
        (torch.zeros_like(source[..., :1]), source[..., :-1]), dim=-1
    )

    torch.testing.assert_close(shifted, expected)
    shifted.sum().backward()
    assert delay.grad is not None
    assert torch.isfinite(delay.grad).all()
    assert float(delay.grad.abs().sum()) > 0.0


@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75])
def test_fractional_delay_matches_two_tap_definition(fraction: float) -> None:
    source = torch.randn(2, 3, 17)
    delay = torch.full((3,), 2.0 + fraction)
    shifted = fractional_causal_shift(source, delay, max_delay=5)
    d2 = torch.nn.functional.pad(source[..., :-2], (2, 0))
    d3 = torch.nn.functional.pad(source[..., :-3], (3, 0))
    torch.testing.assert_close(shifted, (1.0 - fraction) * d2 + fraction * d3)


def test_posterior_fractional_delay_is_a_true_routewise_mixture() -> None:
    source = torch.arange(8, dtype=torch.float64).view(1, 1, -1)
    probability = torch.tensor([[0.25, 0.0, 0.75]], dtype=torch.float64)
    fraction = torch.tensor([0.5], dtype=torch.float64)

    shifted = posterior_fractional_causal_shift(
        source,
        probability,
        fraction,
        max_delay=2,
    )
    d0 = fractional_causal_shift(source, torch.tensor([0.5]), max_delay=2)
    d2 = fractional_causal_shift(source, torch.tensor([2.0]), max_delay=2)
    torch.testing.assert_close(shifted, 0.25 * d0 + 0.75 * d2)


@pytest.mark.parametrize(
    ("probability", "fraction", "expected"),
    [
        ([1.0, 0.0, 0.0], 0.25, [0.5, 0.5, 0.0, 0.0, 0.0]),
        ([0.0, 1.0, 0.0], 0.50, [0.0, 0.0, 0.0, 1.0, 0.0]),
        ([0.0, 0.0, 1.0], 0.75, [0.0, 0.0, 0.0, 0.0, 1.0]),
    ],
)
def test_slow_posterior_maps_exactly_to_fast_grid(
    probability: list[float],
    fraction: float,
    expected: list[float],
) -> None:
    transformed = upsample_delay_posterior_2x(
        torch.tensor([probability]),
        torch.tensor([fraction]),
    )
    torch.testing.assert_close(transformed, torch.tensor([expected]))


def test_coupled_routes_are_nonseparable_and_fast_is_within_band() -> None:
    torch.manual_seed(3)
    module = CoupledDualDelay(n_bands=3, n_nodes=3, route_rank=3, delay_rank=2)
    parameters = module._route_parameters()
    matrix = parameters["slow_weight"].reshape(9, 9)

    assert int(torch.linalg.matrix_rank(matrix, tol=1e-6)) > 1
    off_diagonal = 1.0 - torch.eye(3)[:, :, None, None]
    assert torch.count_nonzero(parameters["fast_weight"] * off_diagonal) == 0


def test_target_reference_modulates_source_but_cannot_bypass_it() -> None:
    torch.manual_seed(5)
    module = CoupledDualDelay(
        n_bands=2,
        n_nodes=2,
        route_rank=2,
        delay_rank=2,
        fast_max_delay=2,
        slow_max_delay=3,
    )
    slow = torch.randn(2, 2, 2, 20)
    fast = torch.complex(torch.randn(2, 2, 2, 40), torch.randn(2, 2, 2, 40))
    target = slow.flip(-1)
    first = module(fast, slow, slow_target_reference=target)
    second = module(fast, slow, slow_target_reference=target.roll(1, dims=0))

    assert not torch.allclose(first.slow_current, second.slow_current)

    zero_source = module(
        torch.zeros_like(fast),
        torch.zeros_like(slow),
        fast_target_reference=fast,
        slow_target_reference=slow,
    )
    assert torch.count_nonzero(zero_source.fast_current) == 0
    assert torch.count_nonzero(zero_source.slow_current) == 0
    assert torch.count_nonzero(zero_source.slow_interaction_current) == 0
    assert torch.count_nonzero(zero_source.slow_delay_contrast_current) == 0
    assert torch.count_nonzero(zero_source.fused_current) == 0


def test_exported_target_interaction_is_nonzero_and_source_mandatory() -> None:
    torch.manual_seed(71)
    module = CoupledDualDelay(
        n_bands=2,
        n_nodes=3,
        route_rank=2,
        delay_rank=2,
        cross_band_enabled=False,
        target_interaction_bound=0.5,
    )
    with torch.no_grad():
        module.target_interaction.band.fill_(0.5)
    fast = torch.complex(torch.randn(2, 2, 3, 16), torch.randn(2, 2, 3, 16))
    slow = torch.randn(2, 2, 3, 8)
    target = torch.randn_like(slow)
    active = module(
        fast,
        slow,
        slow_target_reference=target,
        delay_override="zero",
    )
    no_source = module(
        torch.zeros_like(fast),
        torch.zeros_like(slow),
        slow_target_reference=target,
        delay_override="zero",
    )
    assert torch.count_nonzero(active.slow_interaction_current) > 0
    assert torch.count_nonzero(no_source.slow_interaction_current) == 0


def test_delay_contrast_is_exactly_zero_only_for_locked_zero() -> None:
    torch.manual_seed(73)
    module = CoupledDualDelay(
        n_bands=2,
        n_nodes=3,
        route_rank=2,
        delay_rank=2,
        initial_delay_fraction=0.25,
    )
    fast = torch.complex(torch.randn(2, 2, 3, 16), torch.randn(2, 2, 3, 16))
    slow = torch.randn(2, 2, 3, 8)
    learned = module(fast, slow, delay_override="learned")
    zero = module(fast, slow, delay_override="zero")
    assert torch.count_nonzero(learned.slow_delay_contrast_current) > 0
    assert torch.count_nonzero(zero.slow_delay_contrast_current) == 0


@pytest.mark.parametrize("cross_band", [False, True])
def test_route_delay_contrast_retains_route_axes_and_zero_control(
    cross_band: bool,
) -> None:
    torch.manual_seed(74)
    module = CoupledDualDelay(
        n_bands=2,
        n_nodes=3,
        route_rank=2,
        delay_rank=2,
        initial_delay_fraction=0.25,
        cross_band_enabled=cross_band,
        retain_slow_route_delay_contrast=True,
    )
    fast = torch.complex(torch.randn(2, 2, 3, 16), torch.randn(2, 2, 3, 16))
    slow = torch.randn(2, 2, 3, 8)

    learned = module(fast, slow, delay_override="learned")
    zero = module(fast, slow, delay_override="zero")

    expected_shape = (
        (2, 2, 2, 3, 3, 8) if cross_band else (2, 2, 3, 3, 8)
    )
    assert learned.slow_route_delay_contrast_current is not None
    assert learned.slow_route_delay_contrast_current.shape == expected_shape
    assert torch.count_nonzero(learned.slow_route_delay_contrast_current) > 0
    assert zero.slow_route_delay_contrast_current is None


def test_fold_route_contrast_excludes_identity_and_rejected_floor() -> None:
    module = CoupledDualDelay(
        n_bands=1,
        n_nodes=3,
        route_rank=1,
        delay_rank=1,
        slow_max_delay=2,
        cross_band_enabled=False,
        retain_slow_route_delay_contrast=True,
        slow_residual_route_scale=1.0,
        target_interaction_bound=0.0,
    )
    shape = (1, 1, 3, 3)
    route = torch.zeros(shape)
    route[0, 0, 0, 1] = 0.8
    probability = torch.zeros(*shape, 3)
    probability[..., 1] = 1.0
    fraction = torch.zeros(shape)
    module.load_fold_local_slow_prior(route, probability, fraction)
    slow = torch.zeros(1, 1, 3, 8)
    slow[0, 0, 1, 2] = 1.0
    fast = torch.zeros(1, 1, 3, 16, dtype=torch.complex64)

    output = module(fast, slow, delay_override="learned")

    contrast = output.slow_route_delay_contrast_current
    assert contrast is not None
    assert torch.count_nonzero(contrast[0, 0, 0, 1]) > 0
    delayed_source = fractional_causal_shift(
        slow[:, 0, 1:2],
        torch.tensor([1.0]),
        max_delay=2,
    )
    corrected_route_strength = (0.8 - 0.02) / (1.0 - 2.0 * 0.02)
    torch.testing.assert_close(
        contrast[:, 0, 0, 1],
        corrected_route_strength * (delayed_source[:, 0] - slow[:, 0, 1]),
    )
    accepted = contrast.clone()
    accepted[0, 0, 0, 1] = 0.0
    assert torch.count_nonzero(accepted) == 0
    diagonal = torch.arange(3)
    assert torch.count_nonzero(contrast[0, 0, diagonal, diagonal]) == 0
    assert torch.count_nonzero(module.fold_slow_expected_delay[0, 0, diagonal, diagonal]) == 0


def test_locked_zero_changes_delay_samples_only_and_keeps_memory_bounded() -> None:
    torch.manual_seed(7)
    module = CoupledDualDelay(
        n_bands=2,
        n_nodes=3,
        route_rank=2,
        delay_rank=2,
        fast_max_delay=2,
        slow_max_delay=4,
    )
    slow = torch.randn(2, 2, 3, 16)
    fast = torch.complex(torch.randn(2, 2, 3, 32), torch.randn(2, 2, 3, 32))
    broadband = torch.randn(2, 3, 32)

    learned = module(
        fast, slow, broadband_source=broadband, delay_override="learned"
    )
    locked_zero = module(
        fast, slow, broadband_source=broadband, delay_override="zero"
    )

    for field in ("slow_route_weight", "fast_route_weight", "slow_gate", "fast_gate"):
        torch.testing.assert_close(getattr(learned, field), getattr(locked_zero, field))
    assert not torch.allclose(learned.fused_current, locked_zero.fused_current)
    assert learned.broadband_current is not None
    assert locked_zero.broadband_current is not None
    assert not torch.allclose(learned.broadband_current, locked_zero.broadband_current)
    assert module.last_max_intermediate_elements <= 2 * (2 * 3) * 32


def test_legacy_slow_broadband_does_not_fabricate_identity_delay() -> None:
    module = CoupledDualDelay(
        n_bands=2,
        n_nodes=2,
        route_rank=2,
        delay_rank=2,
        fast_max_delay=2,
        slow_max_delay=2,
        cross_band_enabled=False,
        identity_backbone=True,
        slow_residual_route_scale=0.0,
        fast_residual_route_scale=0.0,
    )
    shape = (2, 2, 2, 2)
    route = torch.zeros(shape)
    route[:, :, 0, 1] = 0.8
    probability = torch.zeros(*shape, 3)
    probability[..., 1] = 1.0
    fraction = torch.full(shape, 0.25)
    module.load_fold_local_slow_prior(route, probability, fraction)
    fast = torch.zeros(1, 2, 2, 24, dtype=torch.complex64)
    slow = torch.zeros(1, 2, 2, 12)
    broadband = torch.arange(24, dtype=torch.float32).view(1, 1, -1).repeat(1, 2, 1)

    full = module(
        fast,
        slow,
        broadband_source=broadband,
        slow_delay_override="learned",
        fast_delay_override="zero",
        broadband_delay_source="slow",
    )
    zero = module(
        fast,
        slow,
        broadband_source=broadband,
        slow_delay_override="zero",
        fast_delay_override="zero",
        broadband_delay_source="slow",
    )

    assert full.broadband_delay_source == zero.broadband_delay_source == "slow"
    assert full.broadband_current is not None and zero.broadband_current is not None
    torch.testing.assert_close(full.broadband_current, zero.broadband_current)
    torch.testing.assert_close(full.broadband_current, broadband)
    torch.testing.assert_close(full.slow_route_weight, zero.slow_route_weight)
    torch.testing.assert_close(full.slow_gate, zero.slow_gate)


def test_legacy_bandwise_carrier_keeps_audited_self_routes_at_zero() -> None:
    module = CoupledDualDelay(
        n_bands=2,
        n_nodes=2,
        route_rank=2,
        delay_rank=2,
        fast_max_delay=2,
        slow_max_delay=2,
        cross_band_enabled=False,
        identity_backbone=True,
        slow_residual_route_scale=0.0,
        fast_residual_route_scale=0.0,
    )
    shape = (2, 2, 2, 2)
    route = torch.zeros(shape)
    route[0, 0, 0, 1] = 0.8
    route[1, 1, 0, 1] = 0.8
    probability = torch.zeros(*shape, 3)
    probability[..., 1] = 1.0
    probability[0, 0] = 0.0
    probability[0, 0, ..., 0] = 1.0
    fraction = torch.full(shape, 0.25)
    module.load_fold_local_slow_prior(route, probability, fraction)
    carrier = torch.arange(48, dtype=torch.float32).reshape(1, 2, 2, 12)
    fast = torch.complex(carrier, torch.zeros_like(carrier))
    slow = torch.zeros(1, 2, 2, 6)
    broadband = carrier.sum(dim=1) / (2.0**0.5)

    full = module(
        fast,
        slow,
        broadband_source=broadband,
        slow_delay_override="learned",
        fast_delay_override="zero",
        broadband_delay_source="slow_bandwise_pr",
    )
    zero = module(
        fast,
        slow,
        broadband_source=broadband,
        slow_delay_override="zero",
        fast_delay_override="zero",
        broadband_delay_source="slow_bandwise_pr",
    )

    assert full.broadband_current is not None and zero.broadband_current is not None
    assert full.broadband_current.shape == broadband.shape
    torch.testing.assert_close(zero.broadband_current, broadband)
    torch.testing.assert_close(full.broadband_current, broadband)
    assert full.broadband_delay_source == zero.broadband_delay_source == "slow_bandwise_pr"


def test_sparse_sourcewise_carrier_delays_only_sources_with_outgoing_evidence() -> None:
    module = CoupledDualDelay(
        n_bands=2,
        n_nodes=2,
        route_rank=2,
        delay_rank=2,
        fast_max_delay=2,
        slow_max_delay=2,
        cross_band_enabled=False,
        identity_backbone=True,
        slow_residual_route_scale=0.0,
        fast_residual_route_scale=0.0,
    )
    shape = (2, 2, 2, 2)
    route = torch.zeros(shape)
    route[0, 0, 0, 1] = 0.8
    route[1, 1, 0, 1] = 0.8
    probability = torch.zeros(*shape, 3)
    probability[..., 1] = 1.0
    fraction = torch.full(shape, 0.25)
    module.load_fold_local_slow_prior(route, probability, fraction)
    fast = torch.zeros(1, 2, 2, 12, dtype=torch.complex64)
    slow = torch.zeros(1, 2, 2, 6)
    broadband = torch.arange(24, dtype=torch.float32).reshape(1, 2, 12)

    full = module(
        fast,
        slow,
        broadband_source=broadband,
        slow_delay_override="learned",
        fast_delay_override="zero",
        broadband_delay_source="slow_sourcewise_point",
    )
    zero = module(
        fast,
        slow,
        broadband_source=broadband,
        slow_delay_override="zero",
        fast_delay_override="zero",
        broadband_delay_source="slow_sourcewise_point",
    )

    expected = broadband.clone()
    expected[:, 1:] = fractional_causal_shift(
        broadband[:, 1:],
        torch.tensor([2.5]),
        max_delay=4,
    )
    assert full.broadband_current is not None and zero.broadband_current is not None
    torch.testing.assert_close(full.broadband_current, expected)
    torch.testing.assert_close(zero.broadband_current, broadband)
    band = torch.arange(2)
    assert float(module.fold_slow_expected_delay[band, band, 0, 0].max()) == 0.0
    assert full.broadband_delay_source == "slow_sourcewise_point"


def test_sparse_route_residual_preserves_identity_and_source_target_direction() -> None:
    module = CoupledDualDelay(
        n_bands=1,
        n_nodes=2,
        route_rank=1,
        delay_rank=1,
        fast_max_delay=2,
        slow_max_delay=2,
        cross_band_enabled=False,
        identity_backbone=True,
        slow_residual_route_scale=0.0,
        fast_residual_route_scale=0.0,
        slow_carrier_residual_scale=0.05,
    )
    shape = (1, 1, 2, 2)
    route = torch.zeros(shape)
    route[0, 0, 0, 1] = 0.8
    probability = torch.zeros(*shape, 3)
    probability[..., 1] = 1.0
    fraction = torch.full(shape, 0.25)
    module.load_fold_local_slow_prior(route, probability, fraction)

    carrier = torch.zeros(1, 1, 2, 12)
    carrier[0, 0, 1, 2] = 1.0
    fast = torch.complex(carrier, torch.zeros_like(carrier))
    slow = torch.zeros(1, 1, 2, 6)
    broadband = torch.randn(1, 2, 12)
    full = module(
        fast,
        slow,
        broadband_source=broadband,
        slow_delay_override="learned",
        fast_delay_override="zero",
        broadband_delay_source="slow_route_residual",
    )
    zero = module(
        fast,
        slow,
        broadband_source=broadband,
        slow_delay_override="zero",
        fast_delay_override="zero",
        broadband_delay_source="slow_route_residual",
    )

    delayed_source = fractional_causal_shift(
        carrier[:, 0, 1:2],
        torch.tensor([2.5]),
        max_delay=4,
    )
    corrected_route_strength = (0.8 - 0.02) / (1.0 - 2.0 * 0.02)
    expected = broadband.clone()
    expected[:, 0] += (
        0.05
        * corrected_route_strength
        * (delayed_source[:, 0] - carrier[:, 0, 1])
        / (2.0**0.5)
    )
    assert full.broadband_current is not None and zero.broadband_current is not None
    torch.testing.assert_close(full.broadband_current, expected)
    torch.testing.assert_close(full.broadband_current[:, 1], broadband[:, 1])
    torch.testing.assert_close(zero.broadband_current, broadband)
    torch.testing.assert_close(full.slow_route_weight, zero.slow_route_weight)
    torch.testing.assert_close(full.slow_gate, zero.slow_gate)
    assert full.broadband_delay_source == zero.broadband_delay_source == "slow_route_residual"


def test_fold_prior_full_and_zero_share_routes_and_only_full_uses_posterior() -> None:
    torch.manual_seed(79)
    module = CoupledDualDelay(
        n_bands=2,
        n_nodes=3,
        route_rank=2,
        delay_rank=2,
        fast_max_delay=2,
        slow_max_delay=4,
        cross_band_enabled=False,
    )
    shape = (2, 2, 3, 3)
    route = torch.zeros(shape)
    route[:, :, 0, 1] = 0.9
    probability = torch.zeros(*shape, 5)
    probability[..., 2] = 1.0
    fraction = torch.full(shape, 0.25)
    module.load_fold_local_slow_prior(route, probability, fraction)
    slow = torch.randn(2, 2, 3, 12)
    fast = torch.complex(torch.randn(2, 2, 3, 24), torch.randn(2, 2, 3, 24))

    full = module(
        fast,
        slow,
        slow_delay_override="learned",
        fast_delay_override="zero",
    )
    zero = module(
        fast,
        slow,
        slow_delay_override="zero",
        fast_delay_override="zero",
    )

    torch.testing.assert_close(full.slow_route_weight, zero.slow_route_weight)
    torch.testing.assert_close(full.slow_gate, zero.slow_gate)
    torch.testing.assert_close(full.fast_route_weight, zero.fast_route_weight)
    assert full.slow_posterior_used is True
    assert zero.slow_posterior_used is False
    assert not torch.allclose(full.slow_current, zero.slow_current)
    assert float(module.fold_slow_route_prior.min()) == pytest.approx(0.02)
    band = torch.arange(2)
    node = torch.arange(3)
    same_band_self_delay = module.fold_slow_expected_delay[
        band[:, None], band[:, None], node[None], node[None]
    ]
    assert torch.count_nonzero(same_band_self_delay) == 0


def test_slow_and_fast_delay_controls_are_independent_and_cross_band_is_exact() -> None:
    torch.manual_seed(9)
    module = CoupledDualDelay(
        n_bands=2,
        n_nodes=3,
        route_rank=2,
        delay_rank=2,
        fast_max_delay=2,
        slow_max_delay=4,
        cross_band_enabled=False,
    )
    slow = torch.randn(1, 2, 3, 10)
    fast = torch.complex(torch.randn(1, 2, 3, 20), torch.randn(1, 2, 3, 20))
    mixed = module(
        fast,
        slow,
        slow_delay_override="learned",
        fast_delay_override="zero",
    )
    parameters = module._route_parameters()
    off_diagonal = 1.0 - torch.eye(2)[:, :, None, None]

    assert mixed.slow_delay_override == "learned"
    assert mixed.fast_delay_override == "zero"
    assert torch.count_nonzero(parameters["slow_weight"] * off_diagonal) == 0
    assert torch.count_nonzero(parameters["slow_gate"] * off_diagonal) == 0
    assert torch.count_nonzero(parameters["slow_delay"] * off_diagonal) == 0


def test_mandatory_identity_backbone_preserves_scale_and_starts_near_zero_lag() -> None:
    torch.manual_seed(10)
    module = CoupledDualDelay(
        n_bands=2,
        n_nodes=3,
        route_rank=2,
        delay_rank=2,
        fast_max_delay=2,
        slow_max_delay=4,
        identity_backbone=True,
        residual_route_scale=0.0,
        initial_delay_fraction=0.05,
    )
    parameters = module._route_parameters()
    identity = torch.eye(2)[:, :, None, None] * torch.eye(3)[None, None]
    slow_diagonal = parameters["slow_weight"][identity.bool()]
    fast_diagonal = parameters["fast_weight"][identity.bool()]

    torch.testing.assert_close(slow_diagonal, torch.full_like(slow_diagonal, 6**0.5))
    torch.testing.assert_close(fast_diagonal, torch.full_like(fast_diagonal, 3**0.5))
    assert float(parameters["slow_delay"][identity.bool()].mean()) < 0.3
    assert float(parameters["fast_delay"][identity.bool()].mean()) < 0.15

    slow = torch.zeros(2, 2, 3, 16)
    fast = torch.zeros(2, 2, 3, 32, dtype=torch.complex64)
    broadband = torch.randn(2, 3, 32)
    output = module(
        fast,
        slow,
        broadband_source=broadband,
        delay_override="zero",
    )
    assert output.broadband_current is not None
    torch.testing.assert_close(output.broadband_current, broadband)
    assert torch.equal(output.broadband_current, broadband)


def test_fast_and_slow_residual_route_scales_are_independent() -> None:
    module = CoupledDualDelay(
        n_bands=2,
        n_nodes=3,
        route_rank=2,
        delay_rank=2,
        residual_route_scale=0.0,
        slow_residual_route_scale=0.2,
        fast_residual_route_scale=0.0,
    )
    parameters = module._route_parameters()
    band_diagonal = torch.eye(2)[:, :, None, None]
    node_diagonal = torch.eye(3)[None, None]
    identity = band_diagonal * node_diagonal
    expected_fast = (3**0.5) * identity

    assert torch.count_nonzero(parameters["slow_weight"] - (6**0.5) * identity) > 0
    torch.testing.assert_close(parameters["fast_weight"], expected_fast)


def test_vectorized_zero_delay_matches_route_equations() -> None:
    torch.manual_seed(11)
    module = CoupledDualDelay(
        n_bands=2,
        n_nodes=3,
        route_rank=2,
        delay_rank=2,
        fast_max_delay=2,
        slow_max_delay=4,
    )
    slow = torch.randn(2, 2, 3, 9)
    fast = torch.complex(torch.randn(2, 2, 3, 18), torch.randn(2, 2, 3, 18))
    output = module(fast, slow, delay_override="zero")
    parameters = module._route_parameters()

    expected_slow = torch.empty_like(output.slow_current)
    for n in range(2):
        for a in range(2):
            for i in range(3):
                total = torch.zeros(9)
                for b in range(2):
                    for j in range(3):
                        interaction = 1.0 + parameters["eta"][a, b, i, j] * torch.tanh(
                            slow[n, a, i]
                        )
                        total = total + (
                            parameters["slow_weight"][a, b, i, j]
                            * slow[n, b, j]
                            * interaction
                        )
                expected_slow[n, a, i] = total / (2 * 3) ** 0.5

    expected_fast = torch.empty_like(output.fast_current)
    for n in range(2):
        for band in range(2):
            strength = module.phase_bound * torch.sigmoid(module.phase_strength_raw[band])
            for i in range(3):
                total = torch.zeros(18)
                target = fast[n, band, i]
                for j in range(3):
                    source = fast[n, band, j]
                    product = source.abs() * target.abs()
                    confidence = product / (product + 1e-4)
                    evidence = confidence * torch.cos(
                        torch.angle(source)
                        - torch.angle(target)
                        - module.phase_preference[band, i, j]
                    )
                    total = total + (
                        parameters["fast_weight"][band, band, i, j]
                        * source.real
                        * (1.0 + strength * torch.tanh(evidence))
                    )
                expected_fast[n, band, i] = total / 3**0.5

    torch.testing.assert_close(output.slow_current, expected_slow, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(output.fast_current, expected_fast, rtol=1e-5, atol=1e-6)


def test_metadata_context_is_rejected() -> None:
    module = CoupledDualDelay(n_bands=2, n_nodes=2)
    slow = torch.zeros(1, 2, 2, 4)
    fast = torch.zeros(1, 2, 2, 8, dtype=torch.complex64)
    with pytest.raises(TypeError, match="EEG-derived tensor"):
        module(fast, slow, context={"subject": "A01"})  # type: ignore[arg-type]
