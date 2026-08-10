from __future__ import annotations

import torch

from dpc_snn.analysis.delay_pair_features import (
    accepted_within_band_routes,
    accepted_cross_band_routes,
    complex_phase_pair_contrasts,
    cross_band_delay_contrasts,
    delay_pair_contrasts,
    fixed_feature_scales,
    phase_gated_current_contrasts,
    pooled_route_features,
)


def _prior(delay: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    route = torch.zeros(1, 1, 2, 2)
    route[0, 0, 1, 0] = 0.98
    posterior = torch.zeros(1, 1, 2, 2, 4)
    posterior[0, 0, 1, 0, delay] = 1.0
    fraction = torch.zeros_like(route)
    return route, posterior, fraction


def _signals() -> tuple[torch.Tensor, torch.Tensor]:
    time = torch.arange(48, dtype=torch.float32)
    source = torch.exp(1j * (torch.pi / 4.0) * time)
    target = torch.zeros_like(source)
    target[2:] = source[:-2]
    analytic = torch.zeros(2, 1, 2, 48, dtype=torch.complex64)
    analytic[:, 0, 0] = source
    analytic[:, 0, 1] = target
    envelope = analytic.abs()
    return analytic, envelope


def _cross_band_prior(delay: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    route = torch.zeros(2, 2, 2, 2)
    route[1, 0, 1, 0] = 0.98
    posterior = torch.zeros(2, 2, 2, 2, 4)
    posterior[..., 0] = 1.0
    posterior[1, 0, 1, 0] = 0.0
    posterior[1, 0, 1, 0, delay] = 1.0
    return route, posterior, torch.zeros_like(route)


def _cross_band_signals() -> tuple[torch.Tensor, torch.Tensor]:
    time = torch.arange(48, dtype=torch.float32)
    analytic = torch.zeros(2, 2, 2, 48, dtype=torch.complex64)
    analytic[:, 0, 0] = torch.exp(1j * (torch.pi / 4.0) * time)
    analytic[:, 1, 1] = 1.0 + 0.0j
    envelope = torch.zeros(2, 2, 2, 48)
    envelope[:, 0, 0] = torch.sin(torch.pi * time / 16.0)
    envelope[:, 1, 1] = torch.cos(torch.pi * time / 20.0)
    return analytic, envelope


def test_all_delay_pair_features_are_exactly_zero_for_matched_zero() -> None:
    analytic, envelope = _signals()
    route, posterior, fraction = _prior(delay=0)
    routes = accepted_within_band_routes(route, posterior, fraction)
    analytic_scale, envelope_scale = fixed_feature_scales(analytic, envelope)

    contrasts = delay_pair_contrasts(
        analytic,
        envelope,
        routes,
        analytic_scale=analytic_scale,
        envelope_scale=envelope_scale,
    )
    assert routes.n_routes == 1
    for value in contrasts.values():
        torch.testing.assert_close(value, torch.zeros_like(value), atol=0, rtol=0)


def test_phase_pair_detects_known_lag_that_envelope_shift_cannot_identify() -> None:
    analytic, envelope = _signals()
    route, posterior, fraction = _prior(delay=2)
    routes = accepted_within_band_routes(route, posterior, fraction)
    analytic_scale, envelope_scale = fixed_feature_scales(analytic, envelope)

    contrasts = delay_pair_contrasts(
        analytic,
        envelope,
        routes,
        analytic_scale=analytic_scale,
        envelope_scale=envelope_scale,
    )
    assert float(contrasts["phase_pair"][..., 4:].mean()) > 0.15
    torch.testing.assert_close(
        contrasts["source_only"][..., 4:],
        torch.zeros_like(contrasts["source_only"][..., 4:]),
        atol=1e-6,
        rtol=0,
    )


def test_phase_gated_currents_are_zero_safe_and_recover_known_lag() -> None:
    analytic, envelope = _signals()
    analytic_scale, envelope_scale = fixed_feature_scales(analytic, envelope)

    zero_route, zero_posterior, zero_fraction = _prior(delay=0)
    zero = phase_gated_current_contrasts(
        analytic,
        envelope,
        accepted_within_band_routes(
            zero_route,
            zero_posterior,
            zero_fraction,
        ),
        analytic_scale=analytic_scale,
        envelope_scale=envelope_scale,
    )
    for value in zero.values():
        torch.testing.assert_close(value, torch.zeros_like(value), atol=0, rtol=0)

    route, posterior, fraction = _prior(delay=2)
    delayed = phase_gated_current_contrasts(
        analytic,
        envelope,
        accepted_within_band_routes(route, posterior, fraction),
        analytic_scale=analytic_scale,
        envelope_scale=envelope_scale,
    )
    assert float(delayed["pure_phase_gate"][..., 4:].mean()) > 0.15
    assert float(delayed["phase_gated_amplitude"][..., 4:].mean()) > 0.15
    assert float(delayed["phase_gated_signed_envelope"][..., 4:].mean()) > 0.10


def test_complex_phase_pair_is_zero_safe_and_imaginary_part_flips() -> None:
    analytic, envelope = _signals()
    analytic_scale, _ = fixed_feature_scales(analytic, envelope)
    zero_route, zero_posterior, zero_fraction = _prior(delay=0)
    zero = complex_phase_pair_contrasts(
        analytic,
        envelope,
        accepted_within_band_routes(
            zero_route,
            zero_posterior,
            zero_fraction,
        ),
        analytic_scale=analytic_scale,
    )
    for value in zero.values():
        torch.testing.assert_close(value, torch.zeros_like(value), atol=0, rtol=0)

    route, posterior, fraction = _prior(delay=2)
    routes = accepted_within_band_routes(route, posterior, fraction)
    forward = complex_phase_pair_contrasts(
        analytic,
        envelope,
        routes,
        analytic_scale=analytic_scale,
    )
    reversed_phase = complex_phase_pair_contrasts(
        analytic.conj(),
        envelope,
        routes,
        analytic_scale=analytic_scale,
    )
    torch.testing.assert_close(
        reversed_phase["real_phase_pair"],
        forward["real_phase_pair"],
    )
    torch.testing.assert_close(
        reversed_phase["imaginary_phase_pair"],
        -forward["imaginary_phase_pair"],
    )
    assert float(forward["imaginary_phase_pair"][..., 4:].abs().mean()) > 0.15


def test_cross_band_pac_is_zero_safe_and_delay_sensitive() -> None:
    analytic, envelope = _cross_band_signals()
    analytic_scale, envelope_scale = fixed_feature_scales(analytic, envelope)
    zero_route, zero_posterior, zero_fraction = _cross_band_prior(delay=0)
    zero = cross_band_delay_contrasts(
        analytic,
        envelope,
        accepted_cross_band_routes(
            zero_route,
            zero_posterior,
            zero_fraction,
        ),
        analytic_scale=analytic_scale,
        envelope_scale=envelope_scale,
    )
    for value in zero.values():
        torch.testing.assert_close(value, torch.zeros_like(value), atol=0, rtol=0)

    route, posterior, fraction = _cross_band_prior(delay=2)
    delayed = cross_band_delay_contrasts(
        analytic,
        envelope,
        accepted_cross_band_routes(route, posterior, fraction),
        analytic_scale=analytic_scale,
        envelope_scale=envelope_scale,
    )
    pac_magnitude = torch.sqrt(
        delayed["cross_band_pac_real"].square()
        + delayed["cross_band_pac_imaginary"].square()
    )
    assert float(pac_magnitude[..., 4:].mean()) > 0.15


def test_route_pooling_retains_statistic_route_and_time_bin_axes() -> None:
    contrast = torch.arange(2 * 3 * 32, dtype=torch.float32).reshape(2, 3, 32)
    features = pooled_route_features(contrast, bins=8)

    assert features.shape == (2, 2, 3, 8)
    assert bool(torch.isfinite(features).all())
