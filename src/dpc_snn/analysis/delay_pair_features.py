"""Fold-fixed route features that compare audited delay with matched zero."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dpc_snn.models.coupled_dual_delay import posterior_fractional_causal_shift


@dataclass(frozen=True)
class WithinBandRouteBank:
    band: torch.Tensor
    target_node: torch.Tensor
    source_node: torch.Tensor
    weight: torch.Tensor
    delay_probability: torch.Tensor
    fractional_delay: torch.Tensor

    @property
    def n_routes(self) -> int:
        return int(self.band.numel())

    @property
    def max_delay(self) -> int:
        return int(self.delay_probability.shape[-1] - 1)

    def to(self, device: torch.device | str) -> WithinBandRouteBank:
        return WithinBandRouteBank(
            band=self.band.to(device),
            target_node=self.target_node.to(device),
            source_node=self.source_node.to(device),
            weight=self.weight.to(device),
            delay_probability=self.delay_probability.to(device),
            fractional_delay=self.fractional_delay.to(device),
        )


@dataclass(frozen=True)
class CrossBandRouteBank:
    target_band: torch.Tensor
    source_band: torch.Tensor
    target_node: torch.Tensor
    source_node: torch.Tensor
    weight: torch.Tensor
    delay_probability: torch.Tensor
    fractional_delay: torch.Tensor

    @property
    def n_routes(self) -> int:
        return int(self.target_band.numel())

    @property
    def max_delay(self) -> int:
        return int(self.delay_probability.shape[-1] - 1)

    def to(self, device: torch.device | str) -> CrossBandRouteBank:
        return CrossBandRouteBank(
            target_band=self.target_band.to(device),
            source_band=self.source_band.to(device),
            target_node=self.target_node.to(device),
            source_node=self.source_node.to(device),
            weight=self.weight.to(device),
            delay_probability=self.delay_probability.to(device),
            fractional_delay=self.fractional_delay.to(device),
        )


def accepted_within_band_routes(
    route_probability: torch.Tensor,
    delay_probability: torch.Tensor,
    fractional_delay: torch.Tensor,
    *,
    threshold: float = 0.5,
    rejected_floor: float = 0.02,
) -> WithinBandRouteBank:
    """Extract accepted non-self routes without collapsing their identities."""

    route = torch.as_tensor(route_probability, dtype=torch.float32)
    posterior = torch.as_tensor(delay_probability, dtype=torch.float32)
    fraction = torch.as_tensor(fractional_delay, dtype=torch.float32)
    if route.ndim != 4 or route.shape[0] != route.shape[1] or route.shape[2] != route.shape[3]:
        raise ValueError("route probability must have shape [B, B, K, K]")
    if posterior.shape[:-1] != route.shape or posterior.shape[-1] < 2:
        raise ValueError("delay probability must append a non-trivial lag axis")
    if fraction.shape != route.shape:
        raise ValueError("fractional delay must match the route axes")
    if not 0.0 <= rejected_floor < threshold < 1.0:
        raise ValueError("route floor and acceptance threshold are inconsistent")
    if not all(
        bool(torch.isfinite(value).all()) for value in (route, posterior, fraction)
    ):
        raise ValueError("route-bank tensors must be finite")

    bands = route.shape[0]
    nodes = route.shape[2]
    band = torch.arange(bands)
    within = route[band, band]
    mask = within > float(threshold)
    mask &= ~torch.eye(nodes, dtype=torch.bool)[None]
    selected = torch.nonzero(mask, as_tuple=False)
    if selected.numel() == 0:
        raise ValueError("no accepted non-self within-band routes")
    selected_band, target_node, source_node = selected.unbind(dim=1)
    strength = (
        (within[selected_band, target_node, source_node] - float(rejected_floor))
        / (1.0 - 2.0 * float(rejected_floor))
    ).clamp(0.0, 1.0)
    selected_posterior = posterior[
        selected_band,
        selected_band,
        target_node,
        source_node,
    ]
    selected_posterior = selected_posterior / selected_posterior.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    return WithinBandRouteBank(
        band=selected_band,
        target_node=target_node,
        source_node=source_node,
        weight=strength,
        delay_probability=selected_posterior,
        fractional_delay=fraction[
            selected_band,
            selected_band,
            target_node,
            source_node,
        ],
    )


def accepted_cross_band_routes(
    route_probability: torch.Tensor,
    delay_probability: torch.Tensor,
    fractional_delay: torch.Tensor,
    *,
    threshold: float = 0.5,
    rejected_floor: float = 0.02,
) -> CrossBandRouteBank:
    """Extract accepted source-band-to-target-band routes only."""

    route = torch.as_tensor(route_probability, dtype=torch.float32)
    posterior = torch.as_tensor(delay_probability, dtype=torch.float32)
    fraction = torch.as_tensor(fractional_delay, dtype=torch.float32)
    if route.ndim != 4 or route.shape[0] != route.shape[1] or route.shape[2] != route.shape[3]:
        raise ValueError("route probability must have shape [B, B, K, K]")
    if posterior.shape[:-1] != route.shape or posterior.shape[-1] < 2:
        raise ValueError("delay probability must append a non-trivial lag axis")
    if fraction.shape != route.shape:
        raise ValueError("fractional delay must match the route axes")
    if not 0.0 <= rejected_floor < threshold < 1.0:
        raise ValueError("route floor and acceptance threshold are inconsistent")
    band_mask = ~torch.eye(route.shape[0], dtype=torch.bool)[:, :, None, None]
    selected = torch.nonzero((route > float(threshold)) & band_mask, as_tuple=False)
    if selected.numel() == 0:
        raise ValueError("no accepted cross-band routes")
    target_band, source_band, target_node, source_node = selected.unbind(dim=1)
    strength = (
        (
            route[target_band, source_band, target_node, source_node]
            - float(rejected_floor)
        )
        / (1.0 - 2.0 * float(rejected_floor))
    ).clamp(0.0, 1.0)
    selected_posterior = posterior[
        target_band,
        source_band,
        target_node,
        source_node,
    ]
    selected_posterior = selected_posterior / selected_posterior.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    return CrossBandRouteBank(
        target_band=target_band,
        source_band=source_band,
        target_node=target_node,
        source_node=source_node,
        weight=strength,
        delay_probability=selected_posterior,
        fractional_delay=fraction[
            target_band,
            source_band,
            target_node,
            source_node,
        ],
    )


def fixed_feature_scales(
    analytic: torch.Tensor,
    envelope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit amplitude scales on an outer-training fold only."""

    _validate_signal_bank(analytic, envelope)
    analytic_scale = analytic.abs().median(dim=-1).values.median(dim=0).values
    envelope_scale = envelope.abs().median(dim=-1).values.median(dim=0).values
    return analytic_scale.clamp_min(1e-6), envelope_scale.clamp_min(1e-6)


def _validate_signal_bank(analytic: torch.Tensor, envelope: torch.Tensor) -> None:
    if analytic.ndim != 4 or envelope.shape != analytic.shape:
        raise ValueError("analytic and envelope tensors must align as [N, B, K, T]")
    if not analytic.is_complex() or envelope.is_complex():
        raise ValueError("analytic must be complex and envelope must be real")
    if not bool(torch.isfinite(analytic).all()) or not bool(torch.isfinite(envelope).all()):
        raise ValueError("delay-pair feature inputs must be finite")


def delay_pair_contrasts(
    analytic: torch.Tensor,
    envelope: torch.Tensor,
    routes: WithinBandRouteBank,
    *,
    analytic_scale: torch.Tensor,
    envelope_scale: torch.Tensor,
    target_interaction: float = 0.10,
) -> dict[str, torch.Tensor]:
    """Return source-only, envelope-pair, and phase-pair delay contrasts."""

    _validate_signal_bank(analytic, envelope)
    if routes.n_routes < 1:
        raise ValueError("delay-pair features require at least one route")
    if analytic.shape[1:3] != tuple(analytic_scale.shape):
        raise ValueError("analytic scale does not match band/node axes")
    if envelope.shape[1:3] != tuple(envelope_scale.shape):
        raise ValueError("envelope scale does not match band/node axes")
    if target_interaction < 0.0:
        raise ValueError("target interaction must be non-negative")

    active = routes.to(analytic.device)
    band = active.band
    target_node = active.target_node
    source_node = active.source_node
    weight = active.weight[None, :, None].to(envelope)

    envelope_scale = envelope_scale.to(envelope)
    normalized_envelope = torch.log1p(envelope.abs() / envelope_scale[None, :, :, None])
    source_envelope = normalized_envelope[:, band, source_node]
    target_envelope = normalized_envelope[:, band, target_node]
    delayed_envelope = posterior_fractional_causal_shift(
        source_envelope,
        active.delay_probability,
        active.fractional_delay,
        max_delay=active.max_delay,
    )
    envelope_delta = delayed_envelope - source_envelope
    source_only = weight * envelope_delta * (
        1.0 + float(target_interaction) * torch.tanh(target_envelope)
    )
    envelope_pair = weight * target_envelope * envelope_delta

    source_analytic = analytic[:, band, source_node]
    target_analytic = analytic[:, band, target_node]
    delayed_analytic = posterior_fractional_causal_shift(
        source_analytic,
        active.delay_probability,
        active.fractional_delay,
        max_delay=active.max_delay,
    )
    analytic_scale = analytic_scale.to(analytic.real)
    target_scale = analytic_scale[band, target_node][None, :, None]
    source_scale = analytic_scale[band, source_node][None, :, None]

    def phase_coupling(source: torch.Tensor) -> torch.Tensor:
        target_amplitude = target_analytic.abs()
        source_amplitude = source.abs()
        target_confidence = target_amplitude / (target_amplitude + target_scale)
        source_confidence = source_amplitude / (source_amplitude + source_scale)
        unit_target = target_analytic / target_amplitude.clamp_min(1e-6)
        unit_source = source / source_amplitude.clamp_min(1e-6)
        return (
            target_confidence
            * source_confidence
            * (unit_target * unit_source.conj()).real
        )

    phase_pair = weight * (
        phase_coupling(delayed_analytic) - phase_coupling(source_analytic)
    )
    causal_mask = (
        torch.arange(analytic.shape[-1], device=analytic.device) > active.max_delay
    ).to(envelope.dtype)
    return {
        "source_only": source_only * causal_mask,
        "envelope_pair": envelope_pair * causal_mask,
        "phase_pair": phase_pair * causal_mask,
    }


def phase_gated_current_contrasts(
    analytic: torch.Tensor,
    envelope: torch.Tensor,
    routes: WithinBandRouteBank,
    *,
    analytic_scale: torch.Tensor,
    envelope_scale: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compare a pure phase gate with physically dimensioned gated currents."""

    _validate_signal_bank(analytic, envelope)
    if routes.n_routes < 1:
        raise ValueError("phase-gated currents require at least one route")
    if analytic.shape[1:3] != tuple(analytic_scale.shape):
        raise ValueError("analytic scale does not match band/node axes")
    if envelope.shape[1:3] != tuple(envelope_scale.shape):
        raise ValueError("envelope scale does not match band/node axes")

    active = routes.to(analytic.device)
    band = active.band
    target_node = active.target_node
    source_node = active.source_node
    weight = active.weight[None, :, None].to(analytic.real)
    source = analytic[:, band, source_node]
    target = analytic[:, band, target_node]
    delayed = posterior_fractional_causal_shift(
        source,
        active.delay_probability,
        active.fractional_delay,
        max_delay=active.max_delay,
    )
    analytic_scale = analytic_scale.to(analytic.real)
    source_scale = analytic_scale[band, source_node][None, :, None]
    target_scale = analytic_scale[band, target_node][None, :, None]

    def coupling(candidate: torch.Tensor) -> torch.Tensor:
        source_amplitude = candidate.abs()
        target_amplitude = target.abs()
        source_confidence = source_amplitude / (source_amplitude + source_scale)
        target_confidence = target_amplitude / (target_amplitude + target_scale)
        source_unit = candidate / source_amplitude.clamp_min(1e-6)
        target_unit = target / target_amplitude.clamp_min(1e-6)
        return (
            source_confidence
            * target_confidence
            * (target_unit * source_unit.conj()).real
        )

    delayed_gate = coupling(delayed)
    zero_gate = coupling(source)
    normalized_delayed_carrier = delayed.real / source_scale
    normalized_zero_carrier = source.real / source_scale
    normalized_delayed_amplitude = delayed.abs() / source_scale
    normalized_zero_amplitude = source.abs() / source_scale

    envelope_scale = envelope_scale.to(envelope)
    source_envelope = envelope[:, band, source_node]
    normalized_envelope = torch.sign(source_envelope) * torch.log1p(
        source_envelope.abs()
        / envelope_scale[band, source_node][None, :, None]
    )
    delayed_envelope = posterior_fractional_causal_shift(
        normalized_envelope,
        active.delay_probability,
        active.fractional_delay,
        max_delay=active.max_delay,
    )
    causal_mask = (
        torch.arange(analytic.shape[-1], device=analytic.device) > active.max_delay
    ).to(analytic.real.dtype)

    def contrast(delayed_value: torch.Tensor, zero_value: torch.Tensor) -> torch.Tensor:
        return weight * (
            delayed_value * delayed_gate - zero_value * zero_gate
        ) * causal_mask

    return {
        "pure_phase_gate": weight * (delayed_gate - zero_gate) * causal_mask,
        "phase_gated_carrier": contrast(
            normalized_delayed_carrier,
            normalized_zero_carrier,
        ),
        "phase_gated_amplitude": contrast(
            normalized_delayed_amplitude,
            normalized_zero_amplitude,
        ),
        "phase_gated_signed_envelope": contrast(
            delayed_envelope,
            normalized_envelope,
        ),
    }


def complex_phase_pair_contrasts(
    analytic: torch.Tensor,
    envelope: torch.Tensor,
    routes: WithinBandRouteBank,
    *,
    analytic_scale: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return real and directional imaginary parts of delayed phase coupling."""

    _validate_signal_bank(analytic, envelope)
    if routes.n_routes < 1:
        raise ValueError("complex phase pairs require at least one route")
    if analytic.shape[1:3] != tuple(analytic_scale.shape):
        raise ValueError("analytic scale does not match band/node axes")
    active = routes.to(analytic.device)
    band = active.band
    target_node = active.target_node
    source_node = active.source_node
    source = analytic[:, band, source_node]
    target = analytic[:, band, target_node]
    delayed = posterior_fractional_causal_shift(
        source,
        active.delay_probability,
        active.fractional_delay,
        max_delay=active.max_delay,
    )
    scale = analytic_scale.to(analytic.real)
    source_scale = scale[band, source_node][None, :, None]
    target_scale = scale[band, target_node][None, :, None]

    def coupling(candidate: torch.Tensor) -> torch.Tensor:
        source_amplitude = candidate.abs()
        target_amplitude = target.abs()
        source_confidence = source_amplitude / (source_amplitude + source_scale)
        target_confidence = target_amplitude / (target_amplitude + target_scale)
        source_unit = candidate / source_amplitude.clamp_min(1e-6)
        target_unit = target / target_amplitude.clamp_min(1e-6)
        return (
            source_confidence
            * target_confidence
            * target_unit
            * source_unit.conj()
        )

    delta = coupling(delayed) - coupling(source)
    weight = active.weight[None, :, None].to(delta.real)
    causal_mask = (
        torch.arange(analytic.shape[-1], device=analytic.device) > active.max_delay
    ).to(delta.real.dtype)
    return {
        "real_phase_pair": weight * delta.real * causal_mask,
        "imaginary_phase_pair": weight * delta.imag * causal_mask,
    }


def cross_band_delay_contrasts(
    analytic: torch.Tensor,
    envelope: torch.Tensor,
    routes: CrossBandRouteBank,
    *,
    analytic_scale: torch.Tensor,
    envelope_scale: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return zero-safe cross-band envelope and direction-aware PAC contrasts."""

    _validate_signal_bank(analytic, envelope)
    if routes.n_routes < 1:
        raise ValueError("cross-band features require at least one route")
    if analytic.shape[1:3] != tuple(analytic_scale.shape):
        raise ValueError("analytic scale does not match band/node axes")
    if envelope.shape[1:3] != tuple(envelope_scale.shape):
        raise ValueError("envelope scale does not match band/node axes")
    active = routes.to(analytic.device)
    target_band = active.target_band
    source_band = active.source_band
    target_node = active.target_node
    source_node = active.source_node
    weight = active.weight[None, :, None].to(analytic.real)

    analytic_scale = analytic_scale.to(analytic.real)
    source_analytic = analytic[:, source_band, source_node]
    target_analytic = analytic[:, target_band, target_node]
    delayed_analytic = posterior_fractional_causal_shift(
        source_analytic,
        active.delay_probability,
        active.fractional_delay,
        max_delay=active.max_delay,
    )
    source_scale = analytic_scale[source_band, source_node][None, :, None]
    target_scale = analytic_scale[target_band, target_node][None, :, None]
    source_amplitude = source_analytic.abs()
    delayed_amplitude = delayed_analytic.abs()
    target_amplitude = target_analytic.abs()
    source_confidence = source_amplitude / (source_amplitude + source_scale)
    delayed_confidence = delayed_amplitude / (delayed_amplitude + source_scale)
    target_confidence = target_amplitude / (target_amplitude + target_scale)
    source_unit = source_analytic / source_amplitude.clamp_min(1e-6)
    delayed_unit = delayed_analytic / delayed_amplitude.clamp_min(1e-6)
    target_unit = target_analytic / target_amplitude.clamp_min(1e-6)

    envelope_scale = envelope_scale.to(envelope)
    source_envelope = envelope[:, source_band, source_node]
    target_envelope = envelope[:, target_band, target_node]

    def signed_normalize(
        value: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sign(value) * torch.log1p(value.abs() / scale)

    normalized_source_envelope = signed_normalize(
        source_envelope,
        envelope_scale[source_band, source_node][None, :, None],
    )
    normalized_target_envelope = signed_normalize(
        target_envelope,
        envelope_scale[target_band, target_node][None, :, None],
    )
    delayed_envelope = posterior_fractional_causal_shift(
        normalized_source_envelope,
        active.delay_probability,
        active.fractional_delay,
        max_delay=active.max_delay,
    )
    envelope_pair = weight * normalized_target_envelope * (
        delayed_envelope - normalized_source_envelope
    )

    low_source_to_high_target = (source_band < target_band)[None, :, None]
    low_phase_to_high_amplitude = target_confidence * (
        delayed_confidence * delayed_unit - source_confidence * source_unit
    )
    high_envelope_to_low_phase = target_confidence * target_unit * (
        delayed_amplitude / source_scale - source_amplitude / source_scale
    )
    pac = torch.where(
        low_source_to_high_target,
        low_phase_to_high_amplitude,
        high_envelope_to_low_phase,
    )
    causal_mask = (
        torch.arange(analytic.shape[-1], device=analytic.device) > active.max_delay
    ).to(analytic.real.dtype)
    return {
        "cross_band_envelope_pair": envelope_pair * causal_mask,
        "cross_band_pac_real": weight * pac.real * causal_mask,
        "cross_band_pac_imaginary": weight * pac.imag * causal_mask,
    }


def pooled_route_features(contrast: torch.Tensor, *, bins: int) -> torch.Tensor:
    """Pool a route-time contrast while retaining statistic/route/bin axes."""

    if contrast.ndim != 3:
        raise ValueError("route contrast must have shape [N, routes, T]")
    if bins <= 0 or bins > contrast.shape[-1]:
        raise ValueError("invalid temporal bin count")
    mean = F.adaptive_avg_pool1d(contrast, int(bins))
    mean_square = F.adaptive_avg_pool1d(contrast.square(), int(bins))
    rms = torch.where(
        mean_square == 0,
        torch.zeros_like(mean_square),
        torch.sqrt(mean_square + 1e-8) - 1e-4,
    )
    return torch.stack((mean, rms), dim=1)
