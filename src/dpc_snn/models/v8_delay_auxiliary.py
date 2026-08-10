"""Sparse physical-node delay auxiliary for the V8 accuracy-first model."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Literal

import torch
from torch import nn

from .coupled_dual_delay import posterior_fractional_causal_shift
from .v62_filterbank import causal_linear_upsample_2x


V8DelayOverride = Literal["off", "zero", "full"]
V8DelaySignalMode = Literal["slow_envelope", "fast_phase", "dual"]


@dataclass(frozen=True)
class V8DelayAuxiliaryOutput:
    physical_current: torch.Tensor
    route_current: torch.Tensor
    delayed_source: torch.Tensor
    override: V8DelayOverride
    active_routes: int
    signal_mode: V8DelaySignalMode
    dynamic_delay_residual: torch.Tensor
    phase_residual: torch.Tensor
    transport_probability: torch.Tensor
    transport_fractional_delay: torch.Tensor

    @property
    def route_contrast(self) -> torch.Tensor:
        """Backward-compatible alias for artifacts produced before routed-current r2."""

        return self.route_current


class SparsePhysicalDelayAuxiliary(nn.Module):
    """Fold-local sparse source-band/node to target-band/node delay routes.

    The prior is stored as checkpoint buffers, not trainable logits.  A locked
    zero intervention keeps every route, confidence, weight, phase preference,
    and amplitude scale unchanged and replaces only the delay posterior.
    """

    def __init__(
        self,
        n_bands: int,
        n_nodes: int,
        *,
        maximum_routes: int = 256,
        maximum_delay: int = 8,
        allow_cross_band: bool = False,
        signal_mode: V8DelaySignalMode = "fast_phase",
        contextual_residual_enabled: bool = False,
        contextual_residual_bound: float = 0.5,
        phase_residual_enabled: bool = False,
        phase_residual_bound: float = 0.25,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_bands = int(n_bands)
        self.n_nodes = int(n_nodes)
        self.maximum_routes = int(maximum_routes)
        self.maximum_delay = int(maximum_delay)
        self.allow_cross_band = bool(allow_cross_band)
        self.signal_mode = signal_mode
        self.contextual_residual_enabled = bool(contextual_residual_enabled)
        self.contextual_residual_bound = float(contextual_residual_bound)
        self.phase_residual_enabled = bool(phase_residual_enabled)
        self.phase_residual_bound = float(phase_residual_bound)
        self.eps = float(eps)
        if self.n_bands < 1 or self.n_nodes < 1:
            raise ValueError("delay auxiliary band/node counts must be positive")
        if self.maximum_routes < 1 or self.maximum_delay < 1:
            raise ValueError("delay auxiliary route and lag capacities must be positive")
        if self.signal_mode not in {"slow_envelope", "fast_phase", "dual"}:
            raise ValueError("unsupported V8 delay signal mode")
        if self.contextual_residual_bound <= 0.0 or self.phase_residual_bound <= 0.0:
            raise ValueError("delay residual bounds must be positive")

        self.register_buffer("source_band", torch.zeros(self.maximum_routes, dtype=torch.long))
        self.register_buffer("source_node", torch.zeros(self.maximum_routes, dtype=torch.long))
        self.register_buffer("target_band", torch.zeros(self.maximum_routes, dtype=torch.long))
        self.register_buffer("target_node", torch.zeros(self.maximum_routes, dtype=torch.long))
        self.register_buffer("route_weight", torch.zeros(self.maximum_routes))
        self.register_buffer("route_confidence", torch.zeros(self.maximum_routes))
        self.register_buffer("phase_preference", torch.zeros(self.maximum_routes))
        self.register_buffer("amplitude_scale", torch.ones(self.maximum_routes))
        self.register_buffer(
            "delay_probability",
            torch.zeros(self.maximum_routes, self.maximum_delay + 1),
        )
        self.register_buffer("fractional_target", torch.zeros(self.maximum_routes))
        self.register_buffer("active_routes", torch.tensor(0, dtype=torch.long))
        self.register_buffer("prior_ready", torch.tensor(False))
        if self.contextual_residual_enabled:
            self.context_weight = nn.Parameter(torch.tensor([0.25, -0.25, 0.10, -0.10]))
            self.context_bias = nn.Parameter(torch.zeros(()))
            self.context_route_raw = nn.Parameter(torch.zeros(self.maximum_routes))
        else:
            self.register_parameter("context_weight", None)
            self.register_parameter("context_bias", None)
            self.register_parameter("context_route_raw", None)
        if self.phase_residual_enabled:
            self.phase_residual_raw = nn.Parameter(torch.zeros(self.maximum_routes))
        else:
            self.register_parameter("phase_residual_raw", None)

    def load_fold_prior(
        self,
        *,
        source_band: torch.Tensor,
        source_node: torch.Tensor,
        target_band: torch.Tensor,
        target_node: torch.Tensor,
        delay_probability: torch.Tensor,
        fractional_target: torch.Tensor,
        route_weight: torch.Tensor,
        route_confidence: torch.Tensor,
        phase_preference: torch.Tensor | None = None,
        amplitude_scale: torch.Tensor | None = None,
    ) -> None:
        """Load one immutable inner-training-fold route prior."""

        source_band = torch.as_tensor(source_band, dtype=torch.long).flatten()
        source_node = torch.as_tensor(source_node, dtype=torch.long).flatten()
        target_band = torch.as_tensor(target_band, dtype=torch.long).flatten()
        target_node = torch.as_tensor(target_node, dtype=torch.long).flatten()
        route_weight = torch.as_tensor(route_weight, dtype=torch.float32).flatten()
        route_confidence = torch.as_tensor(route_confidence, dtype=torch.float32).flatten()
        fractional_target = torch.as_tensor(fractional_target, dtype=torch.float32).flatten()
        probability = torch.as_tensor(delay_probability, dtype=torch.float32)
        count = int(source_band.numel())
        vectors = {
            "source_node": source_node,
            "target_band": target_band,
            "target_node": target_node,
            "route_weight": route_weight,
            "route_confidence": route_confidence,
            "fractional_target": fractional_target,
        }
        if count < 1 or count > self.maximum_routes:
            raise ValueError(
                f"delay prior route count must be in [1, {self.maximum_routes}], got {count}"
            )
        if any(value.numel() != count for value in vectors.values()):
            raise ValueError("delay prior route vectors have inconsistent lengths")
        if probability.shape != (count, self.maximum_delay + 1):
            raise ValueError("delay prior posterior has an incompatible shape")
        if bool((source_band < 0).any()) or bool((source_band >= self.n_bands).any()):
            raise ValueError("delay prior source band is outside the configured bank")
        if bool((target_band < 0).any()) or bool((target_band >= self.n_bands).any()):
            raise ValueError("delay prior target band is outside the configured bank")
        if bool((source_node < 0).any()) or bool((source_node >= self.n_nodes).any()):
            raise ValueError("delay prior source node is outside the physical basis")
        if bool((target_node < 0).any()) or bool((target_node >= self.n_nodes).any()):
            raise ValueError("delay prior target node is outside the physical basis")
        if not self.allow_cross_band and not torch.equal(source_band, target_band):
            raise ValueError("cross-band routes are disabled for this V8 delay stage")
        if bool(((source_band == target_band) & (source_node == target_node)).any()):
            raise ValueError("delay auxiliary prior must not contain identity self-routes")
        finite = (
            torch.isfinite(probability).all()
            and torch.isfinite(fractional_target).all()
            and torch.isfinite(route_weight).all()
            and torch.isfinite(route_confidence).all()
        )
        if not bool(finite):
            raise ValueError("delay prior contains NaN or Inf")
        if bool((probability < 0).any()) or bool((probability.sum(dim=-1) <= 0).any()):
            raise ValueError("delay prior posterior must be non-negative and non-empty")
        if bool(((fractional_target < 0) | (fractional_target > 1)).any()):
            raise ValueError("fractional delay target must lie in [0, 1]")
        if bool((route_confidence <= 0).any()):
            raise ValueError("delay route confidence must be strictly positive")

        if phase_preference is None:
            phase = torch.zeros(count)
        else:
            phase = torch.as_tensor(phase_preference, dtype=torch.float32).flatten()
            if phase.numel() != count or not bool(torch.isfinite(phase).all()):
                raise ValueError("phase preference must be finite for every route")
        if amplitude_scale is None:
            amplitude = torch.ones(count)
        else:
            amplitude = torch.as_tensor(amplitude_scale, dtype=torch.float32).flatten()
            if (
                amplitude.numel() != count
                or not bool(torch.isfinite(amplitude).all())
                or bool((amplitude <= 0).any())
            ):
                raise ValueError("amplitude scale must be finite and positive for every route")

        for buffer in (
            self.source_band,
            self.source_node,
            self.target_band,
            self.target_node,
            self.route_weight,
            self.route_confidence,
            self.phase_preference,
            self.fractional_target,
        ):
            buffer.zero_()
        self.amplitude_scale.fill_(1.0)
        self.delay_probability.zero_()
        self.source_band[:count].copy_(source_band)
        self.source_node[:count].copy_(source_node)
        self.target_band[:count].copy_(target_band)
        self.target_node[:count].copy_(target_node)
        self.route_weight[:count].copy_(route_weight)
        self.route_confidence[:count].copy_(route_confidence)
        self.phase_preference[:count].copy_(phase)
        self.amplitude_scale[:count].copy_(amplitude)
        normalized = probability / probability.sum(dim=-1, keepdim=True)
        self.delay_probability[:count].copy_(normalized)
        self.fractional_target[:count].copy_(fractional_target)
        self.active_routes.fill_(count)
        self.prior_ready.fill_(True)

    def prior_fingerprint(self) -> str:
        if not bool(self.prior_ready):
            raise RuntimeError("delay prior has not been loaded")
        count = int(self.active_routes)
        digest = hashlib.sha256()
        digest.update(str((self.n_bands, self.n_nodes, count, self.maximum_delay)).encode("ascii"))
        for name in (
            "source_band",
            "source_node",
            "target_band",
            "target_node",
            "route_weight",
            "route_confidence",
            "phase_preference",
            "amplitude_scale",
            "delay_probability",
            "fractional_target",
        ):
            value = getattr(self, name)[:count].detach().cpu().contiguous()
            digest.update(name.encode("ascii"))
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def routing_fingerprint(self) -> str:
        """Hash every fixed route property except the intervened delay operator."""

        if not bool(self.prior_ready):
            raise RuntimeError("delay prior has not been loaded")
        count = int(self.active_routes)
        digest = hashlib.sha256()
        digest.update(str((self.n_bands, self.n_nodes, count)).encode("ascii"))
        for name in (
            "source_band",
            "source_node",
            "target_band",
            "target_node",
            "route_weight",
            "route_confidence",
            "phase_preference",
            "amplitude_scale",
        ):
            value = getattr(self, name)[:count].detach().cpu().contiguous()
            digest.update(name.encode("ascii"))
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def _aggregate(self, route_current: torch.Tensor, count: int) -> torch.Tensor:
        target = self.target_band[:count] * self.n_nodes + self.target_node[:count]
        weight = self.route_weight[:count] * self.route_confidence[:count]
        weighted = route_current * weight.to(route_current)[None, :, None]
        output = route_current.new_zeros(
            route_current.shape[0], self.n_bands * self.n_nodes, route_current.shape[-1]
        )
        index = target.to(route_current.device)[None, :, None].expand_as(weighted)
        output.scatter_add_(1, index, weighted)
        norm = route_current.real.new_zeros(self.n_bands * self.n_nodes)
        norm.scatter_add_(0, target.to(norm.device), weight.to(norm).square())
        output = output / norm.sqrt().clamp_min(self.eps)[None, :, None]
        return output.reshape(
            route_current.shape[0], self.n_bands, self.n_nodes, route_current.shape[-1]
        )

    def _contextual_residual(
        self,
        source_envelope: torch.Tensor,
        target_envelope: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        if not self.contextual_residual_enabled:
            return source_envelope.new_zeros(source_envelope.shape)
        if (
            self.context_weight is None
            or self.context_bias is None
            or self.context_route_raw is None
        ):
            raise RuntimeError("contextual delay residual parameters are missing")
        source_history = torch.cumsum(source_envelope, dim=-1)
        target_history = torch.cumsum(target_envelope, dim=-1)
        denominator = torch.arange(
            1,
            source_envelope.shape[-1] + 1,
            device=source_envelope.device,
            dtype=source_envelope.dtype,
        )
        source_history = source_history / denominator
        target_history = target_history / denominator
        features = torch.stack(
            (
                source_envelope,
                target_envelope,
                source_history,
                target_history,
            ),
            dim=-1,
        )
        context = torch.tanh(
            torch.einsum("nrtf,f->nrt", features, self.context_weight.to(features))
            + self.context_bias.to(features)
        )
        route_scale = torch.tanh(self.context_route_raw[:count]).to(context)
        return self.contextual_residual_bound * route_scale[None, :, None] * context

    def _posterior_shift(
        self,
        source: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        count = source.shape[1]
        if residual is None or not self.contextual_residual_enabled:
            return posterior_fractional_causal_shift(
                source,
                self.delay_probability[:count],
                self.fractional_target[:count],
                max_delay=self.maximum_delay,
            )
        if residual.shape != source.shape:
            raise ValueError("dynamic delay residual must match source route/time axes")
        probability = self.delay_probability[:count].to(source.real)
        probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        candidate = torch.arange(
            self.maximum_delay + 1,
            device=source.device,
            dtype=source.real.dtype,
        )
        base = candidate[None, :, None] + self.fractional_target[:count].to(source.real)[
            :, None, None
        ]
        delay = (base[None] + residual[:, :, None]).clamp(
            0.0, float(self.maximum_delay)
        )
        lower = torch.floor(delay).long()
        fraction = delay - lower.to(delay.dtype)
        time = torch.arange(source.shape[-1], device=source.device)
        expanded = source[:, :, None, :].expand(
            -1, -1, self.maximum_delay + 1, -1
        )

        def gather(integer_delay: torch.Tensor) -> torch.Tensor:
            index = time[None, None, None, :] - integer_delay
            valid = index >= 0
            value = torch.gather(expanded, -1, index.clamp_min(0))
            return value * valid.to(value.dtype)

        lower_value = gather(lower)
        upper_value = gather((lower + 1).clamp_max(self.maximum_delay))
        mixed = lower_value * (1.0 - fraction).to(lower_value.real) + upper_value * fraction.to(
            upper_value.real
        )
        return (probability[None, :, :, None] * mixed).sum(dim=2)

    def forward(
        self,
        physical_fast: torch.Tensor,
        physical_slow: torch.Tensor | None = None,
        *,
        override: V8DelayOverride = "full",
    ) -> V8DelayAuxiliaryOutput:
        if not bool(self.prior_ready):
            raise RuntimeError("delay auxiliary cannot run before a fold-local prior is loaded")
        if override not in {"off", "zero", "full"}:
            raise ValueError("delay override must be 'off', 'zero', or 'full'")
        if physical_fast.ndim != 4 or not physical_fast.is_complex():
            raise ValueError("delay auxiliary expects complex physical [N, B, K, T]")
        if physical_fast.shape[1:3] != (self.n_bands, self.n_nodes):
            raise ValueError("delay auxiliary received incompatible physical axes")
        if self.signal_mode in {"slow_envelope", "dual"}:
            if physical_slow is None or physical_slow.ndim != 4 or physical_slow.is_complex():
                raise ValueError("slow/dual delay requires real physical slow features")
            if physical_slow.shape[:-1] != physical_fast.shape[:-1] or (
                physical_slow.shape[-1] * 2 != physical_fast.shape[-1]
            ):
                raise ValueError("physical slow features are not aligned 2:1 with fast features")
        count = int(self.active_routes)
        source_band = self.source_band[:count]
        source_node = self.source_node[:count]
        target_band = self.target_band[:count]
        target_node = self.target_node[:count]
        source_fast = physical_fast[:, source_band, source_node]
        target_fast = physical_fast[:, target_band, target_node]
        if physical_slow is None:
            source_envelope = source_fast.real.new_zeros(source_fast.shape)
            target_envelope = target_fast.real.new_zeros(target_fast.shape)
        else:
            upsampled_envelope = causal_linear_upsample_2x(physical_slow)
            source_envelope = upsampled_envelope[:, source_band, source_node]
            target_envelope = upsampled_envelope[:, target_band, target_node]
        dynamic_residual = self._contextual_residual(
            source_envelope,
            target_envelope,
            count,
        )
        if override in {"off", "zero"}:
            delayed_fast = source_fast
            delayed_envelope = source_envelope
            dynamic_residual = torch.zeros_like(dynamic_residual)
            transport_probability = self.delay_probability[:count].new_zeros(
                count, self.maximum_delay + 1
            )
            transport_probability[:, 0] = 1.0
            transport_fractional_delay = self.fractional_target[:count].new_zeros(count)
        else:
            residual = dynamic_residual if self.contextual_residual_enabled else None
            delayed_fast = self._posterior_shift(source_fast, residual)
            delayed_envelope = self._posterior_shift(source_envelope, residual)
            transport_probability = self.delay_probability[:count]
            transport_probability = transport_probability / transport_probability.sum(
                dim=-1, keepdim=True
            ).clamp_min(self.eps)
            transport_fractional_delay = self.fractional_target[:count]

        amplitude_scale = self.amplitude_scale[:count].to(source_fast.real)[None, :, None]
        preference = self.phase_preference[:count].to(source_fast.real)
        if self.phase_residual_enabled:
            if self.phase_residual_raw is None:
                raise RuntimeError("phase residual parameters are missing")
            phase_residual = self.phase_residual_bound * torch.tanh(
                self.phase_residual_raw[:count]
            ).to(source_fast.real)
        else:
            phase_residual = preference.new_zeros(preference.shape)
        preference = (preference + phase_residual)[None, :, None]

        def phase_current(value: torch.Tensor) -> torch.Tensor:
            source_confidence = value.abs() / (value.abs() + amplitude_scale)
            target_confidence = target_fast.abs() / (target_fast.abs() + amplitude_scale)
            phase_gate = source_confidence * target_confidence * torch.cos(
                torch.angle(target_fast) - torch.angle(value) - preference
            )
            return value.real * phase_gate

        fast_current = phase_current(delayed_fast)
        envelope_current = delayed_envelope * torch.sigmoid(target_envelope)
        if self.signal_mode == "slow_envelope":
            route_current = envelope_current
            delayed_source = delayed_envelope
        elif self.signal_mode == "fast_phase":
            route_current = fast_current
            delayed_source = delayed_fast
        else:
            route_current = 0.5 * (envelope_current + fast_current)
            delayed_source = delayed_fast
        if override == "off":
            route_current = torch.zeros_like(route_current)
        physical_current = self._aggregate(route_current, count)
        return V8DelayAuxiliaryOutput(
            physical_current=physical_current,
            route_current=route_current,
            delayed_source=delayed_source,
            override=override,
            active_routes=count,
            signal_mode=self.signal_mode,
            dynamic_delay_residual=dynamic_residual,
            phase_residual=phase_residual,
            transport_probability=transport_probability,
            transport_fractional_delay=transport_fractional_delay,
        )
