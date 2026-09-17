"""Coupled fast/slow mandatory delay transport for DASP-SNN V6.2-R1."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .v62_filterbank import causal_linear_upsample_2x


DelayOverride = Literal["learned", "zero"]
BroadbandDelaySource = Literal[
    "fast",
    "slow",
    "slow_bandwise_pr",
    "slow_sourcewise_point",
    "slow_route_residual",
]


@dataclass(frozen=True)
class DualDelayOutput:
    fast_current: torch.Tensor
    slow_current: torch.Tensor
    slow_interaction_current: torch.Tensor
    slow_delay_contrast_current: torch.Tensor
    fused_current: torch.Tensor
    broadband_current: torch.Tensor | None
    slow_route_weight: torch.Tensor
    fast_route_weight: torch.Tensor
    slow_delay_samples: torch.Tensor
    fast_delay_samples: torch.Tensor
    slow_gate: torch.Tensor
    fast_gate: torch.Tensor
    slow_delay_override: DelayOverride
    fast_delay_override: DelayOverride
    broadband_delay_source: BroadbandDelaySource
    slow_posterior_used: bool
    slow_route_delay_contrast_current: torch.Tensor | None = None
    slow_phase_pair_delay_contrast_current: torch.Tensor | None = None


def _orientation_and_unit_node(node: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return gauge-fixed node factors and the scale removed from each rank."""

    norm = node.square().sum(dim=(0, 1), keepdim=True).sqrt().clamp_min(1e-8)
    unit = node / norm
    flattened = unit.detach().permute(2, 0, 1).flatten(1)
    pivot = flattened.abs().argmax(dim=1)
    orientation = torch.sign(flattened.gather(1, pivot[:, None])).view(1, 1, -1)
    orientation = torch.where(orientation == 0, torch.ones_like(orientation), orientation)
    return unit * orientation, norm.flatten() * orientation.flatten()


class CoupledField(nn.Module):
    """Rank-R band-pair by node-pair field with a registered gauge."""

    def __init__(
        self,
        n_bands: int,
        n_nodes: int,
        rank: int,
        *,
        positive_band: bool = False,
        init_scale: float = 0.08,
    ) -> None:
        super().__init__()
        self.n_bands = int(n_bands)
        self.n_nodes = int(n_nodes)
        self.rank = int(rank)
        self.positive_band = bool(positive_band)
        if self.rank < 1:
            raise ValueError("coupled-field rank must be positive")
        self.band = nn.Parameter(
            torch.randn(self.n_bands, self.n_bands, self.rank) * float(init_scale)
        )
        self.node = nn.Parameter(
            torch.randn(self.n_nodes, self.n_nodes, self.rank) * float(init_scale)
        )
        self.rank_scale_raw = nn.Parameter(torch.full((self.rank,), -1.0))

    def factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        node, removed_scale = _orientation_and_unit_node(self.node)
        band = F.softplus(self.band) if self.positive_band else self.band
        scale = F.softplus(self.rank_scale_raw) * removed_scale
        return band * scale, node

    def forward(self) -> torch.Tensor:
        band, node = self.factors()
        return torch.einsum("abr,ijr->abij", band, node)


def fractional_causal_shift(
    source: torch.Tensor,
    delay_samples: torch.Tensor,
    *,
    max_delay: int,
) -> torch.Tensor:
    """Apply differentiable two-tap causal fractional delay route-wise.

    Args:
        source: ``[N, routes, T]`` real or complex source signals.
        delay_samples: ``[routes]`` or ``[N, routes]`` in ``[0, max_delay]``.
    """

    if source.ndim != 3:
        raise ValueError("fractional delay source must have shape [N, routes, T]")
    delay = torch.as_tensor(delay_samples, device=source.device, dtype=source.real.dtype)
    if delay.ndim == 1:
        delay = delay[None].expand(source.shape[0], -1)
    if delay.shape != source.shape[:2]:
        raise ValueError("fractional delay does not match the batch/route axes")
    delay = delay.clamp(0.0, float(max_delay))
    lower = torch.floor(delay).to(torch.long)
    fraction = delay - lower.to(delay.dtype)
    time = torch.arange(source.shape[-1], device=source.device)

    def gather(integer_delay: torch.Tensor) -> torch.Tensor:
        index = time.view(1, 1, -1) - integer_delay[..., None]
        valid = index >= 0
        value = torch.gather(source, -1, index.clamp_min(0).expand_as(source))
        return value * valid.to(value.dtype)

    lower_value = gather(lower)
    upper_value = gather((lower + 1).clamp_max(max_delay))
    fraction = fraction[..., None].to(lower_value.real)
    return lower_value * (1.0 - fraction) + upper_value * fraction


def posterior_fractional_causal_shift(
    source: torch.Tensor,
    base_probability: torch.Tensor,
    fractional_target: torch.Tensor,
    *,
    max_delay: int,
) -> torch.Tensor:
    """Apply a route-wise posterior mixture of causal fractional delays."""

    if source.ndim != 3:
        raise ValueError("posterior delay source must have shape [N, routes, T]")
    probability = torch.as_tensor(
        base_probability, device=source.device, dtype=source.real.dtype
    )
    fraction = torch.as_tensor(
        fractional_target, device=source.device, dtype=source.real.dtype
    )
    expected_tail = int(max_delay) + 1
    if probability.ndim == 2:
        probability = probability[None].expand(source.shape[0], -1, -1)
    if fraction.ndim == 1:
        fraction = fraction[None].expand(source.shape[0], -1)
    if probability.shape != (*source.shape[:2], expected_tail):
        raise ValueError("delay posterior does not match the batch/route axes")
    if fraction.shape != source.shape[:2]:
        raise ValueError("fractional target does not match the batch/route axes")
    if not bool(torch.isfinite(probability).all()) or not bool(torch.isfinite(fraction).all()):
        raise ValueError("delay posterior and fractional target must be finite")
    if bool((probability < 0).any()) or bool((fraction < 0).any()) or bool(
        (fraction > 1).any()
    ):
        raise ValueError("delay posterior or fractional target is outside its support")
    probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    time = torch.arange(source.shape[-1], device=source.device)
    delay = torch.arange(expected_tail, device=source.device)
    index = time.view(1, 1, 1, -1) - delay.view(1, 1, -1, 1)
    valid = index >= 0
    expanded = source[:, :, None, :].expand(-1, -1, expected_tail, -1)
    lower = torch.gather(expanded, -1, index.clamp_min(0).expand_as(expanded))
    lower = lower * valid.to(lower.dtype)
    upper_delay = (delay + 1).clamp_max(int(max_delay))
    upper_index = time.view(1, 1, 1, -1) - upper_delay.view(1, 1, -1, 1)
    upper_valid = upper_index >= 0
    upper = torch.gather(expanded, -1, upper_index.clamp_min(0).expand_as(expanded))
    upper = upper * upper_valid.to(upper.dtype)
    fraction = fraction[:, :, None, None]
    mixed = lower * (1.0 - fraction) + upper * fraction
    return (probability[..., None].to(mixed.real) * mixed).sum(dim=2)


def upsample_delay_posterior_2x(
    base_probability: torch.Tensor,
    fractional_target: torch.Tensor,
) -> torch.Tensor:
    """Map a slow-grid fractional posterior onto the exact 2x faster lag grid."""

    probability = torch.as_tensor(base_probability)
    fraction = torch.as_tensor(
        fractional_target,
        device=probability.device,
        dtype=probability.dtype,
    )
    if probability.ndim < 1 or fraction.shape != probability.shape[:-1]:
        raise ValueError("2x posterior and fractional target axes do not match")
    if probability.shape[-1] < 1:
        raise ValueError("2x posterior requires at least one lag candidate")
    if not bool(torch.isfinite(probability).all()) or not bool(torch.isfinite(fraction).all()):
        raise ValueError("2x posterior inputs must be finite")
    if bool((probability < 0).any()) or bool((fraction < 0).any()) or bool(
        (fraction > 1).any()
    ):
        raise ValueError("2x posterior inputs are outside their support")

    probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    slow_max_delay = probability.shape[-1] - 1
    fast_max_delay = 2 * slow_max_delay
    slow_lag = torch.arange(
        slow_max_delay + 1,
        device=probability.device,
        dtype=probability.dtype,
    )
    continuous_fast_lag = (
        2.0 * slow_lag + 2.0 * fraction[..., None]
    ).clamp_max(float(fast_max_delay))
    lower = torch.floor(continuous_fast_lag).to(torch.long)
    upper = (lower + 1).clamp_max(fast_max_delay)
    upper_weight = continuous_fast_lag - lower.to(continuous_fast_lag.dtype)
    fast_probability = probability.new_zeros(*probability.shape[:-1], fast_max_delay + 1)
    fast_probability.scatter_add_(-1, lower, probability * (1.0 - upper_weight))
    fast_probability.scatter_add_(-1, upper, probability * upper_weight)
    return fast_probability / fast_probability.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class CoupledDualDelay(nn.Module):
    """Mandatory target-conditioned dual-rate route transport.

    Fast carrier transport is within-band. Slow envelope transport contains
    every source-band to target-band pair. Route-time tensors are normally
    reduced one target node at a time, but an optional delay-contrast tensor
    retains route identity for a strictly post-delay low-rank readout.
    """

    def __init__(
        self,
        n_bands: int = 12,
        n_nodes: int = 16,
        route_rank: int = 4,
        delay_rank: int = 4,
        fast_max_delay: int = 4,
        slow_max_delay: int = 16,
        fast_initial_contribution: float = 0.10,
        phase_bound: float = 0.15,
        target_interaction_bound: float = 0.50,
        context_features: int = 0,
        dynamic_fast_bound: float = 1.0,
        dynamic_slow_bound: float = 2.0,
        cross_band_enabled: bool = True,
        identity_backbone: bool = True,
        residual_route_scale: float = 0.10,
        slow_residual_route_scale: float | None = None,
        fast_residual_route_scale: float | None = None,
        initial_delay_fraction: float = 0.05,
        rejected_route_prior_floor: float = 0.02,
        slow_carrier_residual_scale: float = 0.05,
        retain_slow_route_delay_contrast: bool = False,
        phase_pair_current_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.n_bands = int(n_bands)
        self.n_nodes = int(n_nodes)
        self.fast_max_delay = int(fast_max_delay)
        self.slow_max_delay = int(slow_max_delay)
        self.phase_bound = float(phase_bound)
        self.target_interaction_bound = float(target_interaction_bound)
        self.context_features = int(context_features)
        self.dynamic_fast_bound = float(dynamic_fast_bound)
        self.dynamic_slow_bound = float(dynamic_slow_bound)
        self.cross_band_enabled = bool(cross_band_enabled)
        self.identity_backbone = bool(identity_backbone)
        self.residual_route_scale = float(residual_route_scale)
        self.rejected_route_prior_floor = float(rejected_route_prior_floor)
        self.slow_carrier_residual_scale = float(slow_carrier_residual_scale)
        self.retain_slow_route_delay_contrast = bool(
            retain_slow_route_delay_contrast
        )
        self.phase_pair_current_enabled = bool(phase_pair_current_enabled)
        if not 0.0 < self.rejected_route_prior_floor < 0.5:
            raise ValueError("rejected route prior floor must lie in (0, 0.5)")
        if not 0.0 <= self.slow_carrier_residual_scale <= 1.0:
            raise ValueError("slow carrier residual scale must lie in [0, 1]")
        if not 0.0 <= self.residual_route_scale <= 1.0:
            raise ValueError("residual route scale must lie in [0, 1]")
        self.slow_residual_route_scale = float(
            self.residual_route_scale
            if slow_residual_route_scale is None
            else slow_residual_route_scale
        )
        self.fast_residual_route_scale = float(
            self.residual_route_scale
            if fast_residual_route_scale is None
            else fast_residual_route_scale
        )
        if not 0.0 <= self.slow_residual_route_scale <= 1.0:
            raise ValueError("slow residual route scale must lie in [0, 1]")
        if not 0.0 <= self.fast_residual_route_scale <= 1.0:
            raise ValueError("fast residual route scale must lie in [0, 1]")
        if not 0.0 < float(initial_delay_fraction) < 1.0:
            raise ValueError("initial delay fraction must lie strictly in (0, 1)")
        if self.fast_max_delay < 0 or self.slow_max_delay < 0:
            raise ValueError("maximum delays must be non-negative")

        self.slow_amplitude = CoupledField(
            self.n_bands, self.n_nodes, route_rank, positive_band=True
        )
        self.fast_amplitude = CoupledField(
            self.n_bands, self.n_nodes, route_rank, positive_band=True
        )
        self.slow_gate_field = CoupledField(self.n_bands, self.n_nodes, route_rank)
        self.fast_gate_field = CoupledField(self.n_bands, self.n_nodes, route_rank)
        self.slow_delay_field = CoupledField(self.n_bands, self.n_nodes, delay_rank)
        self.fast_delay_field = CoupledField(self.n_bands, self.n_nodes, delay_rank)
        self.target_interaction = CoupledField(self.n_bands, self.n_nodes, route_rank)
        self.phase_preference = nn.Parameter(
            torch.zeros(self.n_bands, self.n_nodes, self.n_nodes)
        )
        self.phase_strength_raw = nn.Parameter(torch.full((self.n_bands,), -2.0))
        gate_logit = torch.logit(torch.tensor(0.8))
        self.slow_gate_bias = nn.Parameter(gate_logit.clone())
        self.fast_gate_bias = nn.Parameter(gate_logit.clone())
        delay_bias = torch.logit(torch.tensor(float(initial_delay_fraction)))
        self.slow_delay_bias = nn.Parameter(delay_bias.clone())
        self.fast_delay_bias = nn.Parameter(delay_bias.clone())
        contribution = min(max(float(fast_initial_contribution) / 0.25, 1e-4), 1 - 1e-4)
        self.fast_contribution_raw = nn.Parameter(
            torch.full((self.n_bands,), float(torch.logit(torch.tensor(contribution))))
        )
        if self.context_features > 0:
            self.fast_context = nn.Linear(self.context_features, delay_rank, bias=False)
            self.slow_context = nn.Linear(self.context_features, delay_rank, bias=False)
            nn.init.zeros_(self.fast_context.weight)
            nn.init.zeros_(self.slow_context.weight)
        else:
            self.fast_context = None
            self.slow_context = None
        route_shape = (self.n_bands, self.n_bands, self.n_nodes, self.n_nodes)
        self.register_buffer("fold_slow_route_prior", torch.full(route_shape, 0.5))
        self.register_buffer(
            "fold_slow_positive_delay_prior",
            torch.zeros(*route_shape, self.slow_max_delay + 1),
        )
        self.fold_slow_positive_delay_prior[..., 0] = 1.0
        self.register_buffer("fold_slow_fraction_target", torch.zeros(route_shape))
        self.register_buffer("fold_slow_expected_delay", torch.zeros(route_shape))
        self.register_buffer("fold_slow_prior_ready", torch.tensor(False))
        self.register_buffer(
            "fold_phase_amplitude_scale",
            torch.ones(self.n_bands, self.n_nodes),
        )
        self.register_buffer("fold_phase_amplitude_scale_ready", torch.tensor(False))
        self.fold_slow_route_residual_enabled = False
        self.fold_slow_delay_residual_enabled = False
        self.last_max_intermediate_elements = 0

    def load_fold_local_phase_amplitude_scale(self, scale: torch.Tensor) -> None:
        """Load a positive analytic-amplitude scale fitted on the training fold."""

        value = torch.as_tensor(
            scale,
            dtype=self.fold_phase_amplitude_scale.dtype,
        ).clone()
        if value.shape != self.fold_phase_amplitude_scale.shape:
            raise ValueError("fold-local phase scale has an incompatible shape")
        if not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
            raise ValueError("fold-local phase scale must be finite and positive")
        self.fold_phase_amplitude_scale.copy_(
            value.to(self.fold_phase_amplitude_scale.device)
        )
        self.fold_phase_amplitude_scale_ready.fill_(True)

    def load_fold_local_slow_prior(
        self,
        route_probability: torch.Tensor,
        positive_delay_probability: torch.Tensor,
        fractional_delay_target: torch.Tensor,
    ) -> None:
        """Load one training-fold-only route and conditional delay posterior."""

        route = torch.as_tensor(
            route_probability, dtype=self.fold_slow_route_prior.dtype
        ).clone()
        positive = torch.as_tensor(
            positive_delay_probability,
            dtype=self.fold_slow_positive_delay_prior.dtype,
        ).clone()
        fraction = torch.as_tensor(
            fractional_delay_target,
            dtype=self.fold_slow_fraction_target.dtype,
        ).clone()
        if route.shape != self.fold_slow_route_prior.shape:
            raise ValueError("fold-local route prior has an incompatible shape")
        if positive.shape != self.fold_slow_positive_delay_prior.shape:
            raise ValueError("fold-local positive-delay posterior has an incompatible shape")
        if fraction.shape != self.fold_slow_fraction_target.shape:
            raise ValueError("fold-local fractional target has an incompatible shape")
        if not bool(torch.isfinite(route).all()) or not bool(torch.isfinite(positive).all()):
            raise ValueError("fold-local route/delay prior must be finite")
        if not bool(torch.isfinite(fraction).all()):
            raise ValueError("fold-local fractional target must be finite")
        if bool((route < 0).any()) or bool((route > 1).any()):
            raise ValueError("fold-local route prior must lie in [0, 1]")
        if bool((positive < 0).any()) or bool((fraction < 0).any()) or bool(
            (fraction > 1).any()
        ):
            raise ValueError("fold-local delay posterior is outside its support")
        positive = positive / positive.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # Identity transport has no estimable self-edge lag. Keep every
        # same-band self-route at exact d=0; accepted j-to-i evidence remains on
        # its original non-self route and must never be copied onto the much
        # stronger identity backbone.
        off_diagonal = ~torch.eye(self.n_nodes, dtype=torch.bool)
        global_weight = route * off_diagonal[None, None]
        if not bool((global_weight.sum() > 0)):
            raise ValueError("fold-local prior contains no accepted non-self route")
        band = torch.arange(self.n_bands)
        node = torch.arange(self.n_nodes)
        positive[band[:, None], band[:, None], node[None], node[None]] = 0.0
        positive[band[:, None], band[:, None], node[None], node[None], 0] = 1.0
        fraction[band[:, None], band[:, None], node[None], node[None]] = 0.0

        route = route.clamp(
            self.rejected_route_prior_floor,
            1.0 - self.rejected_route_prior_floor,
        )
        delay_index = torch.arange(self.slow_max_delay + 1, dtype=positive.dtype)
        expected = torch.sum(positive * delay_index, dim=-1)
        expected = expected + fraction * (1.0 - positive[..., -1])
        self.fold_slow_route_prior.copy_(route.to(self.fold_slow_route_prior.device))
        self.fold_slow_positive_delay_prior.copy_(
            positive.to(self.fold_slow_positive_delay_prior.device)
        )
        self.fold_slow_fraction_target.copy_(
            fraction.to(self.fold_slow_fraction_target.device)
        )
        self.fold_slow_expected_delay.copy_(expected.to(self.fold_slow_expected_delay.device))
        self.fold_slow_prior_ready.fill_(True)

    def set_fold_slow_prior_residuals(
        self,
        *,
        route: bool,
        delay: bool,
    ) -> None:
        if (route or delay) and not bool(self.fold_slow_prior_ready):
            raise RuntimeError("cannot enable prior residuals before loading a fold prior")
        self.fold_slow_route_residual_enabled = bool(route)
        self.fold_slow_delay_residual_enabled = bool(delay)

    def _route_parameters(self) -> dict[str, torch.Tensor]:
        slow_residual = self.slow_amplitude()
        fast_residual = self.fast_amplitude()
        slow_gate_residual = self.slow_gate_field()
        slow_gate = torch.sigmoid(slow_gate_residual + self.slow_gate_bias)
        fast_gate = torch.sigmoid(self.fast_gate_field() + self.fast_gate_bias)
        if bool(self.fold_slow_prior_ready):
            if self.fold_slow_route_residual_enabled:
                prior_logit = torch.logit(
                    self.fold_slow_route_prior.clamp(
                        self.rejected_route_prior_floor,
                        1.0 - self.rejected_route_prior_floor,
                    )
                )
                slow_gate = torch.sigmoid(prior_logit + slow_gate_residual)
            else:
                slow_gate = self.fold_slow_route_prior
        if bool(self.fold_slow_prior_ready) and not self.fold_slow_route_residual_enabled:
            slow_weight = self.slow_residual_route_scale * slow_gate
        else:
            slow_weight = self.slow_residual_route_scale * slow_residual * slow_gate
        fast_weight = self.fast_residual_route_scale * fast_residual * fast_gate
        slow_delay = self.slow_max_delay * torch.sigmoid(
            self.slow_delay_field() + self.slow_delay_bias
        )
        if bool(self.fold_slow_prior_ready):
            if self.fold_slow_delay_residual_enabled:
                slow_delay = (
                    self.fold_slow_expected_delay
                    + self.dynamic_slow_bound * torch.tanh(self.slow_delay_field())
                ).clamp(0.0, float(self.slow_max_delay))
            else:
                slow_delay = self.fold_slow_expected_delay
        fast_delay = self.fast_max_delay * torch.sigmoid(
            self.fast_delay_field() + self.fast_delay_bias
        )
        eta = self.target_interaction_bound * torch.tanh(self.target_interaction())
        diagonal = torch.eye(
            self.n_bands, device=fast_weight.device, dtype=fast_weight.dtype
        )[:, :, None, None]
        node_diagonal = torch.eye(
            self.n_nodes, device=fast_weight.device, dtype=fast_weight.dtype
        )[None, None]
        identity = diagonal * node_diagonal
        if self.identity_backbone:
            slow_weight = slow_weight + math.sqrt(self.n_bands * self.n_nodes) * identity
            fast_weight = fast_weight + math.sqrt(self.n_nodes) * identity
        if not self.cross_band_enabled:
            slow_weight = slow_weight * diagonal
            slow_gate = slow_gate * diagonal
            slow_delay = slow_delay * diagonal
            eta = eta * diagonal
        return {
            "slow_weight": slow_weight,
            "fast_weight": fast_weight * diagonal,
            "slow_gate": slow_gate,
            "fast_gate": fast_gate * diagonal,
            "slow_delay": slow_delay,
            "fast_delay": fast_delay * diagonal,
            "eta": eta,
        }

    def _slow_route_contrast_weight(
        self,
        parameters: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return non-self route weights without the rejected-prior floor."""

        if bool(self.fold_slow_prior_ready) and not self.fold_slow_route_residual_enabled:
            weight = (
                self.fold_slow_route_prior - self.rejected_route_prior_floor
            ).clamp_min(0.0) / (1.0 - 2.0 * self.rejected_route_prior_floor)
            weight = self.slow_residual_route_scale * weight.clamp_max(1.0)
        else:
            weight = parameters["slow_weight"]
        band_identity = torch.eye(
            self.n_bands,
            device=weight.device,
            dtype=weight.dtype,
        )[:, :, None, None]
        node_identity = torch.eye(
            self.n_nodes,
            device=weight.device,
            dtype=weight.dtype,
        )[None, None]
        weight = weight * (1.0 - band_identity * node_identity)
        if not self.cross_band_enabled:
            weight = weight * band_identity
        return weight

    def _slow_phase_pair_contrast_current(
        self,
        fast_source: torch.Tensor,
        parameters: dict[str, torch.Tensor],
        *,
        slow_delay_used: torch.Tensor,
        slow_residual: torch.Tensor | None,
        delay_override: DelayOverride,
        use_fold_posterior: bool,
    ) -> torch.Tensor:
        """Aggregate accepted delayed phase-pair contrasts on the slow grid."""

        if not bool(self.fold_slow_prior_ready):
            raise RuntimeError("phase-pair current requires a fold-local route prior")
        if not bool(self.fold_phase_amplitude_scale_ready):
            raise RuntimeError(
                "phase-pair current requires an outer-training-fold amplitude scale"
            )
        if fast_source.shape[-1] % 2:
            raise ValueError("phase-pair current requires an exact 2:1 fast/slow ratio")
        aligned = fast_source[..., ::2]
        route_weight = self._slow_route_contrast_weight(parameters)
        scale = self.fold_phase_amplitude_scale.to(aligned.real)
        causal_mask = (
            torch.arange(aligned.shape[-1], device=aligned.device)
            > self.slow_max_delay
        ).to(aligned.real.dtype)
        band_outputs: list[torch.Tensor] = []
        for band in range(self.n_bands):
            node_outputs: list[torch.Tensor] = []
            source = aligned[:, band]
            for target_node in range(self.n_nodes):
                delay = slow_delay_used[band, band, target_node]
                if slow_residual is not None:
                    delay = delay[None] + slow_residual[
                        :, band, band, target_node
                    ]
                delay = self._override_delay(delay, delay_override)
                posterior = self.fold_slow_positive_delay_prior[
                    band, band, target_node
                ]
                fraction = self.fold_slow_fraction_target[
                    band, band, target_node
                ]
                if use_fold_posterior:
                    delayed = posterior_fractional_causal_shift(
                        source,
                        posterior,
                        fraction,
                        max_delay=self.slow_max_delay,
                    )
                else:
                    delayed = fractional_causal_shift(
                        source,
                        delay,
                        max_delay=self.slow_max_delay,
                    )
                target = aligned[:, band, target_node][:, None]
                target_amplitude = target.abs()
                target_scale = scale[band, target_node]
                target_confidence = target_amplitude / (
                    target_amplitude + target_scale
                )
                target_unit = target / target_amplitude.clamp_min(1e-6)

                def coupling(candidate: torch.Tensor) -> torch.Tensor:
                    amplitude = candidate.abs()
                    source_scale = scale[band][None, :, None]
                    confidence = amplitude / (amplitude + source_scale)
                    unit = candidate / amplitude.clamp_min(1e-6)
                    return (
                        target_confidence
                        * confidence
                        * (target_unit * unit.conj()).real
                    )

                weight = route_weight[band, band, target_node]
                normalization = weight.square().sum().sqrt().clamp_min(1.0)
                contrast = weight[None, :, None] * (
                    coupling(delayed) - coupling(source)
                )
                node_outputs.append(
                    contrast.sum(dim=1) * causal_mask / normalization
                )
            band_outputs.append(torch.stack(node_outputs, dim=1))
        current = torch.stack(band_outputs, dim=1)
        if current.is_complex() or not bool(torch.isfinite(current).all()):
            raise FloatingPointError("phase-pair delay current is invalid")
        return current

    def _context_residual(
        self,
        context: torch.Tensor | None,
        field: CoupledField,
        layer: nn.Linear | None,
        bound: float,
    ) -> torch.Tensor | None:
        if context is None:
            return None
        if not isinstance(context, torch.Tensor):
            raise TypeError("trial context must be an EEG-derived tensor, never metadata")
        if layer is None or context.ndim != 2 or context.shape[1] != self.context_features:
            raise ValueError("trial context has an incompatible shape")
        _, node = field.factors()
        band = torch.tanh(field.band)
        coefficient = torch.tanh(layer(context))
        residual = torch.einsum("nr,abr,ijr->nabij", coefficient, band, node)
        return float(bound) * torch.tanh(residual)

    @staticmethod
    def _override_delay(delay: torch.Tensor, override: DelayOverride) -> torch.Tensor:
        if override == "learned":
            return delay
        if override == "zero":
            return torch.zeros_like(delay)
        raise ValueError("delay_override must be 'learned' or 'zero'")

    def _zero_slow_currents(
        self,
        slow_source: torch.Tensor,
        slow_target: torch.Tensor,
        parameters: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slow_weight = parameters["slow_weight"]
        eta = parameters["eta"]
        slow_base = torch.einsum("abij,nbjt->nait", slow_weight, slow_source)
        slow_interaction = torch.einsum(
            "abij,nbjt->nait", slow_weight * eta, slow_source
        )
        normalization = math.sqrt(self.n_bands * self.n_nodes)
        slow_interaction_current = (
            torch.tanh(slow_target) * slow_interaction
        ) / normalization
        return slow_base / normalization + slow_interaction_current, slow_interaction_current

    def _zero_delay_currents(
        self,
        fast_source: torch.Tensor,
        slow_source: torch.Tensor,
        fast_target: torch.Tensor,
        slow_target: torch.Tensor,
        parameters: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact vectorization of the route equations when every lag is zero."""

        slow_current, slow_interaction_current = self._zero_slow_currents(
            slow_source,
            slow_target,
            parameters,
        )

        index = torch.arange(self.n_bands, device=fast_source.device)
        fast_weight = parameters["fast_weight"][index, index]
        source = fast_source[:, :, None, :, :]
        target = fast_target[:, :, :, None, :]
        amplitude_product = source.abs() * target.abs()
        confidence = amplitude_product / (amplitude_product + 1e-4)
        preference = self.phase_preference[None, :, :, :, None]
        phase_difference = torch.angle(source) - torch.angle(target)
        phase_evidence = confidence * torch.cos(phase_difference - preference)
        strength = (
            self.phase_bound
            * torch.sigmoid(self.phase_strength_raw)[None, :, None, None, None]
        )
        multiplier = 1.0 + strength * torch.tanh(phase_evidence)
        fast_current = (
            fast_weight[None, :, :, :, None] * source.real * multiplier
        ).sum(dim=3) / math.sqrt(self.n_nodes)
        self.last_max_intermediate_elements = max(
            int(fast_source.numel()), int(slow_source.numel())
        )
        return fast_current, slow_current, slow_interaction_current

    def _broadband_current(
        self,
        source: torch.Tensor,
        parameters: dict[str, torch.Tensor],
        *,
        fast_delay_used: torch.Tensor,
        fast_residual: torch.Tensor | None,
        delay_override: DelayOverride,
    ) -> torch.Tensor:
        """Transport a wideband carrier with the band-averaged fast route."""

        if source.ndim != 3 or source.shape[1] != self.n_nodes or source.is_complex():
            raise ValueError("broadband delay source must be real [N, K, T]")
        index = torch.arange(self.n_bands, device=source.device)
        weight = parameters["fast_weight"][index, index].mean(dim=0)
        delay = fast_delay_used[index, index].mean(dim=0)
        context_delay = None
        if fast_residual is not None:
            context_delay = fast_residual[:, index, index].mean(dim=1)
        if delay_override == "zero":
            if self.identity_backbone and self.fast_residual_route_scale == 0.0:
                return source
            return torch.einsum("ij,njt->nit", weight, source) / math.sqrt(
                self.n_nodes
            )

        node_outputs = []
        for target_node in range(self.n_nodes):
            selected_delay = delay[target_node]
            if context_delay is not None:
                selected_delay = selected_delay[None] + context_delay[:, target_node]
            delayed = fractional_causal_shift(
                source,
                self._override_delay(selected_delay, delay_override),
                max_delay=self.fast_max_delay,
            )
            self.last_max_intermediate_elements = max(
                self.last_max_intermediate_elements, int(delayed.numel())
            )
            current = (
                weight[target_node][None, :, None] * delayed
            ).sum(dim=1) / math.sqrt(self.n_nodes)
            node_outputs.append(current)
        return torch.stack(node_outputs, dim=1)

    def _slow_broadband_current(
        self,
        source: torch.Tensor,
        parameters: dict[str, torch.Tensor],
        *,
        slow_delay_used: torch.Tensor,
        slow_residual: torch.Tensor | None,
        delay_override: DelayOverride,
    ) -> torch.Tensor:
        """Transport the exact ATC carrier with the active audited slow route."""

        if source.ndim != 3 or source.shape[1] != self.n_nodes or source.is_complex():
            raise ValueError("broadband delay source must be real [N, K, T]")
        index = torch.arange(self.n_bands, device=source.device)
        if self.cross_band_enabled:
            route_weight = parameters["slow_weight"].reshape(
                self.n_bands * self.n_bands,
                self.n_nodes,
                self.n_nodes,
            )
            route_delay = slow_delay_used.reshape_as(route_weight)
            posterior = self.fold_slow_positive_delay_prior.reshape(
                self.n_bands * self.n_bands,
                self.n_nodes,
                self.n_nodes,
                self.slow_max_delay + 1,
            )
            fraction = self.fold_slow_fraction_target.reshape_as(route_weight)
            context_delay = (
                None
                if slow_residual is None
                else slow_residual.reshape(
                    source.shape[0],
                    self.n_bands * self.n_bands,
                    self.n_nodes,
                    self.n_nodes,
                )
            )
        else:
            route_weight = parameters["slow_weight"][index, index]
            route_delay = slow_delay_used[index, index]
            posterior = self.fold_slow_positive_delay_prior[index, index]
            fraction = self.fold_slow_fraction_target[index, index]
            context_delay = (
                None if slow_residual is None else slow_residual[:, index, index]
            )

        # There are B identity band routes. Averaging over B preserves an exact
        # identity carrier under the zero-delay intervention while retaining
        # all accepted cross-band contributions when they are enabled.
        weight = route_weight.sum(dim=0) / float(self.n_bands)
        normalization = math.sqrt(self.n_bands * self.n_nodes)
        if delay_override == "zero":
            return torch.einsum("ij,njt->nit", weight, source) / normalization

        route_strength = route_weight.abs()
        strength_sum = route_strength.sum(dim=0).clamp_min(1e-8)
        use_fold_posterior = bool(
            self.fold_slow_prior_ready
            and not self.fold_slow_delay_residual_enabled
            and slow_residual is None
        )
        if use_fold_posterior:
            fast_posterior = upsample_delay_posterior_2x(posterior, fraction)
            aggregate_posterior = (
                fast_posterior * route_strength[..., None]
            ).sum(dim=0) / strength_sum[..., None]
        else:
            if context_delay is None:
                aggregate_delay = (
                    route_delay * route_strength
                ).sum(dim=0) / strength_sum
            else:
                aggregate_delay = (
                    (route_delay[None] + context_delay) * route_strength[None]
                ).sum(dim=1) / strength_sum[None]
            aggregate_delay = 2.0 * aggregate_delay

        node_outputs = []
        fast_max_delay = 2 * self.slow_max_delay
        for target_node in range(self.n_nodes):
            if use_fold_posterior:
                delayed = posterior_fractional_causal_shift(
                    source,
                    aggregate_posterior[target_node],
                    torch.zeros(
                        self.n_nodes,
                        device=source.device,
                        dtype=source.dtype,
                    ),
                    max_delay=fast_max_delay,
                )
                intermediate = delayed.numel() * (fast_max_delay + 1)
            else:
                selected_delay = (
                    aggregate_delay[target_node]
                    if aggregate_delay.ndim == 2
                    else aggregate_delay[:, target_node]
                )
                delayed = fractional_causal_shift(
                    source,
                    selected_delay,
                    max_delay=fast_max_delay,
                )
                intermediate = delayed.numel()
            self.last_max_intermediate_elements = max(
                self.last_max_intermediate_elements,
                int(intermediate),
            )
            current = (
                weight[target_node][None, :, None] * delayed
            ).sum(dim=1) / normalization
            node_outputs.append(current)
        return torch.stack(node_outputs, dim=1)

    def _slow_sourcewise_point_carrier_current(
        self,
        source: torch.Tensor,
        *,
        slow_delay_used: torch.Tensor,
        slow_residual: torch.Tensor | None,
        delay_override: DelayOverride,
    ) -> torch.Tensor:
        """Apply sparse outgoing-edge point delays to exact carrier sources."""

        if source.ndim != 3 or source.shape[1] != self.n_nodes or source.is_complex():
            raise ValueError("sourcewise carrier source must be real [N, K, T]")
        if delay_override == "zero":
            self.last_max_intermediate_elements = max(
                self.last_max_intermediate_elements,
                int(source.numel()),
            )
            return source

        node_off_diagonal = ~torch.eye(
            self.n_nodes,
            device=source.device,
            dtype=torch.bool,
        )
        route_strength = (
            self.fold_slow_route_prior - self.rejected_route_prior_floor
        ).clamp_min(0.0) / (1.0 - 2.0 * self.rejected_route_prior_floor)
        route_strength = route_strength.clamp_max(1.0)
        band_delay = []
        band_weight = []
        for band in range(self.n_bands):
            weight = route_strength[band, band] * node_off_diagonal
            denominator = weight.sum(dim=0).clamp_min(1e-8)
            band_delay.append(
                (slow_delay_used[band, band] * weight).sum(dim=0) / denominator
            )
            band_weight.append(weight.sum(dim=0))
        band_delay = torch.stack(band_delay, dim=0)
        active = torch.stack(band_weight, dim=0) > 0.0
        denominator = active.sum(dim=0).clamp_min(1)
        point_delay = (band_delay * active).sum(dim=0) / denominator
        point_delay = torch.where(
            active.any(dim=0),
            point_delay,
            torch.zeros_like(point_delay),
        )
        if slow_residual is not None:
            context_delay = []
            for band in range(self.n_bands):
                weight = route_strength[band, band] * node_off_diagonal
                weight_sum = weight.sum(dim=0).clamp_min(1e-8)
                context_delay.append(
                    (slow_residual[:, band, band] * weight[None]).sum(dim=1)
                    / weight_sum[None]
                )
            context_delay = torch.stack(context_delay, dim=1)
            point_delay = point_delay[None] + (
                context_delay * active[None]
            ).sum(dim=1) / denominator[None]
        delayed = fractional_causal_shift(
            source,
            2.0 * point_delay,
            max_delay=2 * self.slow_max_delay,
        )
        self.last_max_intermediate_elements = max(
            self.last_max_intermediate_elements,
            int(delayed.numel()),
        )
        return delayed

    def _slow_route_residual_carrier_current(
        self,
        source: torch.Tensor,
        broadband_source: torch.Tensor,
        parameters: dict[str, torch.Tensor],
        *,
        slow_delay_used: torch.Tensor,
        slow_residual: torch.Tensor | None,
        delay_override: DelayOverride,
    ) -> torch.Tensor:
        """Add sparse band-route delay contrasts to an identity self-route.

        The exact broadband carrier enters through a registered zero-lag
        self-route. Every non-self contribution is a source-to-target contrast
        ``D_tau(z_bj) - D_0(z_bj)``. Consequently, the locked-zero intervention
        cancels every residual exactly without changing route weights or gain.
        """

        if source.ndim != 4 or source.shape[1:3] != (self.n_bands, self.n_nodes):
            raise ValueError("route-residual source must have shape [N, B, K, T]")
        if not source.is_complex():
            raise ValueError("route-residual source must be an analytic carrier")
        if (
            broadband_source.ndim != 3
            or broadband_source.shape
            != (source.shape[0], self.n_nodes, source.shape[-1])
            or broadband_source.is_complex()
        ):
            raise ValueError("route-residual broadband source must be real [N, K, T]")

        identity_current = fractional_causal_shift(
            broadband_source,
            broadband_source.new_zeros(self.n_nodes),
            max_delay=2 * self.slow_max_delay,
        )
        if delay_override == "zero" or self.slow_carrier_residual_scale == 0.0:
            self.last_max_intermediate_elements = max(
                self.last_max_intermediate_elements,
                int(identity_current.numel()),
            )
            return identity_current

        if bool(self.fold_slow_prior_ready):
            route_strength = (
                self.fold_slow_route_prior - self.rejected_route_prior_floor
            ).clamp_min(0.0) / (1.0 - 2.0 * self.rejected_route_prior_floor)
            route_strength = route_strength.clamp_max(1.0)
        else:
            route_strength = parameters["slow_gate"]
        node_off_diagonal = ~torch.eye(
            self.n_nodes,
            device=source.device,
            dtype=torch.bool,
        )
        route_strength = route_strength * node_off_diagonal[None, None]

        analytic_real = source.real
        node_outputs: list[torch.Tensor] = []
        for target_node in range(self.n_nodes):
            if self.cross_band_enabled:
                selected_source = (
                    analytic_real[:, None]
                    .expand(-1, self.n_bands, -1, -1, -1)
                    .reshape(
                        source.shape[0],
                        self.n_bands * self.n_bands * self.n_nodes,
                        -1,
                    )
                )
                delay = slow_delay_used[:, :, target_node, :].reshape(-1)
                weight = route_strength[:, :, target_node, :].reshape(-1)
                context_delay = (
                    None
                    if slow_residual is None
                    else slow_residual[:, :, :, target_node, :].reshape(
                        source.shape[0], -1
                    )
                )
                normalization = math.sqrt(
                    self.n_bands * self.n_bands * self.n_nodes
                )
            else:
                band = torch.arange(self.n_bands, device=source.device)
                selected_source = analytic_real.reshape(
                    source.shape[0], self.n_bands * self.n_nodes, -1
                )
                delay = slow_delay_used[band, band, target_node, :].reshape(-1)
                weight = route_strength[band, band, target_node, :].reshape(-1)
                context_delay = (
                    None
                    if slow_residual is None
                    else slow_residual[:, band, band, target_node, :].reshape(
                        source.shape[0], -1
                    )
                )
                normalization = math.sqrt(self.n_bands * self.n_nodes)
            if context_delay is not None:
                delay = delay[None] + context_delay
            delayed = fractional_causal_shift(
                selected_source,
                2.0 * delay,
                max_delay=2 * self.slow_max_delay,
            )
            self.last_max_intermediate_elements = max(
                self.last_max_intermediate_elements,
                int(delayed.numel()),
            )
            residual = (
                weight[None, :, None] * (delayed - selected_source)
            ).sum(dim=1) / normalization
            node_outputs.append(residual)
        route_residual = torch.stack(node_outputs, dim=1)
        return identity_current + self.slow_carrier_residual_scale * route_residual

    def _slow_bandwise_carrier_current(
        self,
        source: torch.Tensor,
        broadband_source: torch.Tensor,
        *,
        slow_delay_used: torch.Tensor,
        slow_residual: torch.Tensor | None,
        delay_override: DelayOverride,
    ) -> torch.Tensor:
        """Causally delay a perfect-reconstruction carrier decomposition.

        The analytic bands and their exact causal complement sum to the
        registered ATC carrier. Every component is delayed in the full branch;
        replacing the posterior by delta-0 therefore recovers the parent input.
        """

        if source.ndim != 4 or source.shape[1:3] != (self.n_bands, self.n_nodes):
            raise ValueError("bandwise carrier source must have shape [N, B, K, T]")
        if (
            broadband_source.ndim != 3
            or broadband_source.shape != (source.shape[0], self.n_nodes, source.shape[-1])
            or broadband_source.is_complex()
        ):
            raise ValueError("perfect-reconstruction carrier must align with [N, K, T]")
        if source.is_complex():
            source = source.real
        batch = source.shape[0]
        time = source.shape[-1]
        use_fold_posterior = bool(
            self.fold_slow_prior_ready
            and not self.fold_slow_delay_residual_enabled
            and slow_residual is None
            and delay_override == "learned"
        )
        fast_max_delay = 2 * self.slow_max_delay
        zero_band_synthesis = source.sum(dim=1) / math.sqrt(self.n_bands)
        complement = broadband_source - zero_band_synthesis
        if delay_override == "zero":
            self.last_max_intermediate_elements = max(
                self.last_max_intermediate_elements,
                int(source.numel() + complement.numel()),
            )
            return zero_band_synthesis + complement

        node = torch.arange(self.n_nodes, device=source.device)
        posterior = torch.stack(
            [
                self.fold_slow_positive_delay_prior[band, band, node, node]
                for band in range(self.n_bands)
            ],
            dim=0,
        )
        fraction = torch.stack(
            [
                self.fold_slow_fraction_target[band, band, node, node]
                for band in range(self.n_bands)
            ],
            dim=0,
        )

        if use_fold_posterior:
            fast_posterior = upsample_delay_posterior_2x(posterior, fraction)

            def fixed_posterior_shift(
                value: torch.Tensor,
                probability: torch.Tensor,
            ) -> torch.Tensor:
                routes = value.shape[1]
                kernel = probability.reshape(routes, -1).flip(-1)[:, None]
                return F.conv1d(
                    F.pad(value, (fast_max_delay, 0)),
                    kernel.to(value),
                    groups=routes,
                )

            delayed_bands = fixed_posterior_shift(
                source.reshape(batch, self.n_bands * self.n_nodes, time),
                fast_posterior,
            ).reshape_as(source)
            complement_posterior = fast_posterior.mean(dim=0)
            delayed_complement = fixed_posterior_shift(
                complement,
                complement_posterior,
            )
            intermediate = (
                delayed_bands.numel()
                + delayed_complement.numel()
                + fast_posterior.numel()
            )
        else:
            delay = torch.stack(
                [
                    slow_delay_used[band, band, node, node]
                    for band in range(self.n_bands)
                ],
                dim=0,
            )
            if slow_residual is not None:
                context_delay = torch.stack(
                    [
                        slow_residual[:, band, band, node, node]
                        for band in range(self.n_bands)
                    ],
                    dim=1,
                )
                delay = delay[None] + context_delay
            flat_delay = 2.0 * delay.reshape(*delay.shape[:-2], -1)
            delayed_bands = fractional_causal_shift(
                source.reshape(batch, self.n_bands * self.n_nodes, time),
                flat_delay,
                max_delay=fast_max_delay,
            ).reshape_as(source)
            delayed_complement = fractional_causal_shift(
                complement,
                2.0 * delay.mean(dim=-2),
                max_delay=fast_max_delay,
            )
            intermediate = delayed_bands.numel() + delayed_complement.numel()

        self.last_max_intermediate_elements = max(
            self.last_max_intermediate_elements,
            int(intermediate),
        )
        return delayed_bands.sum(dim=1) / math.sqrt(self.n_bands) + delayed_complement

    def forward(
        self,
        fast_source: torch.Tensor,
        slow_source: torch.Tensor,
        *,
        broadband_source: torch.Tensor | None = None,
        fast_target_reference: torch.Tensor | None = None,
        slow_target_reference: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        delay_override: DelayOverride | None = None,
        slow_delay_override: DelayOverride | None = None,
        fast_delay_override: DelayOverride | None = None,
        broadband_delay_source: BroadbandDelaySource = "fast",
    ) -> DualDelayOutput:
        if fast_source.ndim != 4 or not fast_source.is_complex():
            raise ValueError("fast delay source must be complex [N, B, K, T]")
        if slow_source.ndim != 4 or slow_source.is_complex():
            raise ValueError("slow delay source must be real [N, B, K, T]")
        if fast_source.shape[1:3] != (self.n_bands, self.n_nodes):
            raise ValueError("fast delay source has incompatible band/node axes")
        if slow_source.shape[1:3] != (self.n_bands, self.n_nodes):
            raise ValueError("slow delay source has incompatible band/node axes")
        if fast_source.shape[0] != slow_source.shape[0]:
            raise ValueError("fast and slow delay sources require the same batch")
        if (
            self.phase_pair_current_enabled
            and fast_source.shape[-1] != 2 * slow_source.shape[-1]
        ):
            raise ValueError("fast and slow delay sources require an exact 2:1 rate ratio")
        if broadband_source is not None and (
            broadband_source.ndim != 3
            or broadband_source.shape[:2] != (fast_source.shape[0], self.n_nodes)
            or broadband_source.shape[-1] != fast_source.shape[-1]
            or broadband_source.is_complex()
        ):
            raise ValueError("broadband source must be real and align with [N, K, T_fast]")
        fast_target = fast_source if fast_target_reference is None else fast_target_reference
        slow_target = slow_source if slow_target_reference is None else slow_target_reference
        if fast_target.shape != fast_source.shape or slow_target.shape != slow_source.shape:
            raise ValueError("target references must match their source tensors")
        if self.phase_pair_current_enabled and (
            not bool(self.fold_slow_prior_ready)
            or not bool(self.fold_phase_amplitude_scale_ready)
        ):
            raise RuntimeError(
                "phase-pair current requires fold-local route and amplitude evidence"
            )
        common_override: DelayOverride = "learned" if delay_override is None else delay_override
        selected_slow_override = slow_delay_override or common_override
        selected_fast_override = fast_delay_override or common_override
        if selected_slow_override not in {"learned", "zero"}:
            raise ValueError("slow_delay_override must be 'learned' or 'zero'")
        if selected_fast_override not in {"learned", "zero"}:
            raise ValueError("fast_delay_override must be 'learned' or 'zero'")
        if broadband_delay_source not in {
            "fast",
            "slow",
            "slow_bandwise_pr",
            "slow_sourcewise_point",
            "slow_route_residual",
        }:
            raise ValueError(
                "unsupported broadband_delay_source"
            )

        parameters = self._route_parameters()
        slow_route_contrast_weight = (
            self._slow_route_contrast_weight(parameters)
            if self.retain_slow_route_delay_contrast
            else None
        )
        slow_residual = self._context_residual(
            context,
            self.slow_delay_field,
            self.slow_context,
            self.dynamic_slow_bound,
        )
        fast_residual = self._context_residual(
            context,
            self.fast_delay_field,
            self.fast_context,
            self.dynamic_fast_bound,
        )
        batch = slow_source.shape[0]
        slow_flat = slow_source.reshape(batch, self.n_bands * self.n_nodes, -1)
        slow_targets: list[torch.Tensor] = []
        slow_interaction_targets: list[torch.Tensor] = []
        slow_route_contrast_targets: list[torch.Tensor] = []
        max_intermediate = 0
        slow_delay_used = parameters["slow_delay"]
        fast_delay_used = parameters["fast_delay"]
        if selected_slow_override == "zero" and selected_fast_override == "zero":
            fast_current, slow_current, slow_interaction_current = self._zero_delay_currents(
                fast_source,
                slow_source,
                fast_target,
                slow_target,
                parameters,
            )
            if broadband_source is None:
                broadband_current = None
            elif broadband_delay_source == "slow_route_residual":
                broadband_current = self._slow_route_residual_carrier_current(
                    fast_source,
                    broadband_source,
                    parameters,
                    slow_delay_used=slow_delay_used,
                    slow_residual=slow_residual,
                    delay_override=selected_slow_override,
                )
            elif broadband_delay_source == "slow_sourcewise_point":
                broadband_current = self._slow_sourcewise_point_carrier_current(
                    broadband_source,
                    slow_delay_used=slow_delay_used,
                    slow_residual=slow_residual,
                    delay_override=selected_slow_override,
                )
            elif broadband_delay_source == "slow_bandwise_pr":
                broadband_current = self._slow_bandwise_carrier_current(
                    fast_source,
                    broadband_source,
                    slow_delay_used=slow_delay_used,
                    slow_residual=slow_residual,
                    delay_override=selected_slow_override,
                )
            elif broadband_delay_source == "slow":
                broadband_current = self._slow_broadband_current(
                    broadband_source,
                    parameters,
                    slow_delay_used=slow_delay_used,
                    slow_residual=slow_residual,
                    delay_override=selected_slow_override,
                )
            else:
                broadband_current = self._broadband_current(
                    broadband_source,
                    parameters,
                    fast_delay_used=fast_delay_used,
                    fast_residual=fast_residual,
                    delay_override=selected_fast_override,
                )
            slow_at_fast_rate = causal_linear_upsample_2x(slow_current)
            if slow_at_fast_rate.shape[-1] != fast_current.shape[-1]:
                raise ValueError("fast and causally upsampled slow paths are not timestamp aligned")
            contribution = 0.25 * torch.sigmoid(self.fast_contribution_raw)
            fused = slow_at_fast_rate + contribution[None, :, None, None] * fast_current
            return DualDelayOutput(
                fast_current=fast_current,
                slow_current=slow_current,
                slow_interaction_current=slow_interaction_current,
                slow_delay_contrast_current=torch.zeros_like(slow_current),
                fused_current=fused,
                broadband_current=broadband_current,
                slow_route_weight=parameters["slow_weight"],
                fast_route_weight=parameters["fast_weight"],
                slow_delay_samples=slow_delay_used,
                fast_delay_samples=fast_delay_used,
                slow_gate=parameters["slow_gate"],
                fast_gate=parameters["fast_gate"],
                slow_delay_override=selected_slow_override,
                fast_delay_override=selected_fast_override,
                broadband_delay_source=broadband_delay_source,
                slow_posterior_used=False,
                slow_phase_pair_delay_contrast_current=(
                    torch.zeros_like(slow_source)
                    if self.phase_pair_current_enabled
                    else None
                ),
            )
        use_fold_posterior = bool(
            self.fold_slow_prior_ready
            and not self.fold_slow_delay_residual_enabled
            and slow_residual is None
            and selected_slow_override == "learned"
        )
        slow_phase_pair_delay_contrast_current = (
            self._slow_phase_pair_contrast_current(
                fast_source,
                parameters,
                slow_delay_used=slow_delay_used,
                slow_residual=slow_residual,
                delay_override=selected_slow_override,
                use_fold_posterior=use_fold_posterior,
            )
            if self.phase_pair_current_enabled
            else None
        )
        for target_band in range(self.n_bands):
            node_outputs = []
            node_interaction_outputs = []
            node_route_contrasts: list[torch.Tensor] = []
            for target_node in range(self.n_nodes):
                if self.cross_band_enabled:
                    selected_source = slow_flat
                    delay = slow_delay_used[target_band, :, target_node, :].reshape(-1)
                    if slow_residual is not None:
                        delay = delay[None] + slow_residual[
                            :, target_band, :, target_node, :
                        ].reshape(batch, -1)
                    weight = parameters["slow_weight"][
                        target_band, :, target_node, :
                    ].reshape(1, -1, 1)
                    contrast_weight = (
                        None
                        if slow_route_contrast_weight is None
                        else slow_route_contrast_weight[
                            target_band, :, target_node, :
                        ].reshape(1, -1, 1)
                    )
                    eta = parameters["eta"][
                        target_band, :, target_node, :
                    ].reshape(1, -1, 1)
                    posterior = self.fold_slow_positive_delay_prior[
                        target_band, :, target_node, :
                    ].reshape(-1, self.slow_max_delay + 1)
                    fraction = self.fold_slow_fraction_target[
                        target_band, :, target_node, :
                    ].reshape(-1)
                else:
                    # Off-band weights are exactly zero, so selecting the one
                    # active source band is algebraically identical and avoids
                    # twelvefold redundant route shifting in the first E3 gate.
                    selected_source = slow_source[:, target_band]
                    delay = slow_delay_used[
                        target_band, target_band, target_node, :
                    ]
                    if slow_residual is not None:
                        delay = delay[None] + slow_residual[
                            :, target_band, target_band, target_node, :
                        ]
                    weight = parameters["slow_weight"][
                        target_band, target_band, target_node, :
                    ].reshape(1, -1, 1)
                    contrast_weight = (
                        None
                        if slow_route_contrast_weight is None
                        else slow_route_contrast_weight[
                            target_band, target_band, target_node, :
                        ].reshape(1, -1, 1)
                    )
                    eta = parameters["eta"][
                        target_band, target_band, target_node, :
                    ].reshape(1, -1, 1)
                    posterior = self.fold_slow_positive_delay_prior[
                        target_band, target_band, target_node, :
                    ]
                    fraction = self.fold_slow_fraction_target[
                        target_band, target_band, target_node, :
                    ]
                delay = self._override_delay(delay, selected_slow_override)
                if use_fold_posterior:
                    delayed = posterior_fractional_causal_shift(
                        selected_source,
                        posterior,
                        fraction,
                        max_delay=self.slow_max_delay,
                    )
                else:
                    delayed = fractional_causal_shift(
                        selected_source, delay, max_delay=self.slow_max_delay
                    )
                posterior_elements = (
                    delayed.numel() * (self.slow_max_delay + 1)
                    if use_fold_posterior
                    else delayed.numel()
                )
                max_intermediate = max(max_intermediate, posterior_elements)
                target = slow_target[:, target_band, target_node, :][:, None, :]
                interaction = 1.0 + eta * torch.tanh(target)
                base_current = (weight * delayed).sum(dim=1)
                interaction_current = (
                    weight * delayed * (interaction - 1.0)
                ).sum(dim=1)
                normalization = math.sqrt(self.n_bands * self.n_nodes)
                node_outputs.append((base_current + interaction_current) / normalization)
                node_interaction_outputs.append(interaction_current / normalization)
                if self.retain_slow_route_delay_contrast:
                    if contrast_weight is None:
                        raise RuntimeError("route contrast weights were not initialized")
                    contrast_normalization = (
                        contrast_weight.square()
                        .sum(dim=1, keepdim=True)
                        .sqrt()
                        .clamp_min(1.0)
                    )
                    route_contrast = (
                        contrast_weight
                        * (delayed - selected_source)
                        * interaction
                        / contrast_normalization
                    )
                    if route_contrast.is_complex():
                        raise ValueError(
                            "slow route delay contrast must be a real envelope current"
                        )
                    if self.cross_band_enabled:
                        route_contrast = route_contrast.reshape(
                            batch,
                            self.n_bands,
                            self.n_nodes,
                            route_contrast.shape[-1],
                        )
                    node_route_contrasts.append(route_contrast)
            slow_targets.append(torch.stack(node_outputs, dim=1))
            slow_interaction_targets.append(
                torch.stack(node_interaction_outputs, dim=1)
            )
            if self.retain_slow_route_delay_contrast:
                route_target_axis = 2 if self.cross_band_enabled else 1
                slow_route_contrast_targets.append(
                    torch.stack(node_route_contrasts, dim=route_target_axis)
                )
        slow_current = torch.stack(slow_targets, dim=1)
        slow_interaction_current = torch.stack(slow_interaction_targets, dim=1)
        slow_route_delay_contrast_current = (
            torch.stack(slow_route_contrast_targets, dim=1)
            if self.retain_slow_route_delay_contrast
            else None
        )
        zero_slow_current, _ = self._zero_slow_currents(
            slow_source,
            slow_target,
            parameters,
        )
        slow_delay_contrast_current = slow_current - zero_slow_current

        fast_targets: list[torch.Tensor] = []
        for band in range(self.n_bands):
            source = fast_source[:, band]
            node_outputs = []
            for target_node in range(self.n_nodes):
                delay = fast_delay_used[band, band, target_node, :]
                if fast_residual is not None:
                    delay = delay[None] + fast_residual[
                        :, band, band, target_node, :
                    ]
                delay = self._override_delay(delay, selected_fast_override)
                delayed = fractional_causal_shift(
                    source, delay, max_delay=self.fast_max_delay
                )
                max_intermediate = max(max_intermediate, delayed.numel())
                target = fast_target[:, band, target_node, :][:, None, :]
                confidence = (
                    delayed.abs() * target.abs()
                    / (delayed.abs() * target.abs() + 1e-4)
                )
                phase_difference = torch.angle(delayed) - torch.angle(target)
                preference = self.phase_preference[band, target_node][None, :, None]
                phase_evidence = confidence * torch.cos(phase_difference - preference)
                strength = self.phase_bound * torch.sigmoid(self.phase_strength_raw[band])
                phase_multiplier = 1.0 + strength * torch.tanh(phase_evidence)
                weight = parameters["fast_weight"][
                    band, band, target_node, :
                ].reshape(1, -1, 1)
                current = (weight * delayed.real * phase_multiplier).sum(dim=1)
                node_outputs.append(current / math.sqrt(self.n_nodes))
            fast_targets.append(torch.stack(node_outputs, dim=1))
        fast_current = torch.stack(fast_targets, dim=1)
        self.last_max_intermediate_elements = int(max_intermediate)
        if broadband_source is None:
            broadband_current = None
        elif broadband_delay_source == "slow_route_residual":
            broadband_current = self._slow_route_residual_carrier_current(
                fast_source,
                broadband_source,
                parameters,
                slow_delay_used=slow_delay_used,
                slow_residual=slow_residual,
                delay_override=selected_slow_override,
            )
        elif broadband_delay_source == "slow_sourcewise_point":
            broadband_current = self._slow_sourcewise_point_carrier_current(
                broadband_source,
                slow_delay_used=slow_delay_used,
                slow_residual=slow_residual,
                delay_override=selected_slow_override,
            )
        elif broadband_delay_source == "slow_bandwise_pr":
            broadband_current = self._slow_bandwise_carrier_current(
                fast_source,
                broadband_source,
                slow_delay_used=slow_delay_used,
                slow_residual=slow_residual,
                delay_override=selected_slow_override,
            )
        elif broadband_delay_source == "slow":
            broadband_current = self._slow_broadband_current(
                broadband_source,
                parameters,
                slow_delay_used=slow_delay_used,
                slow_residual=slow_residual,
                delay_override=selected_slow_override,
            )
        else:
            broadband_current = self._broadband_current(
                broadband_source,
                parameters,
                fast_delay_used=fast_delay_used,
                fast_residual=fast_residual,
                delay_override=selected_fast_override,
            )

        slow_at_fast_rate = causal_linear_upsample_2x(slow_current)
        if slow_at_fast_rate.shape[-1] != fast_current.shape[-1]:
            raise ValueError("fast and causally upsampled slow paths are not timestamp aligned")
        contribution = 0.25 * torch.sigmoid(self.fast_contribution_raw)
        fused = slow_at_fast_rate + contribution[None, :, None, None] * fast_current
        return DualDelayOutput(
            fast_current=fast_current,
            slow_current=slow_current,
            slow_interaction_current=slow_interaction_current,
            slow_delay_contrast_current=slow_delay_contrast_current,
            fused_current=fused,
            broadband_current=broadband_current,
            slow_route_weight=parameters["slow_weight"],
            fast_route_weight=parameters["fast_weight"],
            slow_delay_samples=slow_delay_used,
            fast_delay_samples=fast_delay_used,
            slow_gate=parameters["slow_gate"],
            fast_gate=parameters["fast_gate"],
            slow_delay_override=selected_slow_override,
            fast_delay_override=selected_fast_override,
            broadband_delay_source=broadband_delay_source,
            slow_posterior_used=use_fold_posterior,
            slow_route_delay_contrast_current=slow_route_delay_contrast_current,
            slow_phase_pair_delay_contrast_current=(
                slow_phase_pair_delay_contrast_current
            ),
        )
