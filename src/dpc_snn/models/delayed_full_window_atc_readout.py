"""Full-window ATCNet adapter after mandatory delay transport."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DelayedFullWindowATCOutput:
    logits: torch.Tensor
    synthesized_carrier: torch.Tensor
    delayed_statistics_logits: torch.Tensor | None
    delayed_route_statistics_logits: torch.Tensor | None
    delayed_statistics_gate: torch.Tensor | None


class DelayedTensorCPClassifier(nn.Module):
    """Low-capacity tensor readout preserving statistic/band/node/time axes."""

    def __init__(
        self,
        n_bands: int,
        n_nodes: int,
        n_bins: int,
        n_classes: int,
        rank: int,
        n_statistics: int = 2,
    ) -> None:
        super().__init__()
        self.shape = (int(n_statistics), int(n_bands), int(n_nodes), int(n_bins))
        self.rank = int(rank)
        if self.rank <= 0:
            raise ValueError("delayed tensor CP rank must be positive")
        generator = torch.Generator(device="cpu").manual_seed(0)
        self.statistic_factor = nn.Parameter(torch.empty(self.rank, self.shape[0]))
        self.band_factor = nn.Parameter(torch.empty(self.rank, self.shape[1]))
        self.node_factor = nn.Parameter(torch.empty(self.rank, self.shape[2]))
        self.time_factor = nn.Parameter(torch.empty(self.rank, self.shape[3]))
        for parameter in (
            self.statistic_factor,
            self.band_factor,
            self.node_factor,
            self.time_factor,
        ):
            nn.init.normal_(parameter, generator=generator)
        self.class_factor = nn.Parameter(torch.zeros(int(n_classes), self.rank))
        self.bias = nn.Parameter(torch.zeros(int(n_classes)))
        self.register_buffer(
            "output_gain",
            torch.tensor(math.sqrt(float(self.shape[1] * self.shape[2]))),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 5 or tuple(features.shape[1:]) != self.shape:
            raise ValueError(
                "tensor CP statistics must have shape [N, 2, bands, nodes, bins]"
            )
        factors = tuple(
            F.normalize(parameter, dim=1, eps=1e-6)
            for parameter in (
                self.statistic_factor,
                self.band_factor,
                self.node_factor,
                self.time_factor,
            )
        )
        components = torch.einsum(
            "nmbkt,rm,rb,rk,rt->nr",
            features,
            *factors,
        )
        return F.linear(components * self.output_gain.to(components), self.class_factor, self.bias)


class DelayedRouteTensorCPClassifier(nn.Module):
    """Low-rank readout retaining source/target band and node identities."""

    def __init__(
        self,
        n_bands: int,
        n_nodes: int,
        n_bins: int,
        n_classes: int,
        rank: int,
        n_statistics: int = 2,
    ) -> None:
        super().__init__()
        self.n_statistics = int(n_statistics)
        self.n_bands = int(n_bands)
        self.n_nodes = int(n_nodes)
        self.n_bins = int(n_bins)
        self.rank = int(rank)
        if self.rank <= 0:
            raise ValueError("delayed route tensor CP rank must be positive")
        generator = torch.Generator(device="cpu").manual_seed(0)
        factor_sizes = (
            self.n_statistics,
            self.n_bands,
            self.n_bands,
            self.n_nodes,
            self.n_nodes,
            self.n_bins,
        )
        self.statistic_factor = nn.Parameter(torch.empty(self.rank, factor_sizes[0]))
        self.target_band_factor = nn.Parameter(torch.empty(self.rank, factor_sizes[1]))
        self.source_band_factor = nn.Parameter(torch.empty(self.rank, factor_sizes[2]))
        self.target_node_factor = nn.Parameter(torch.empty(self.rank, factor_sizes[3]))
        self.source_node_factor = nn.Parameter(torch.empty(self.rank, factor_sizes[4]))
        self.time_factor = nn.Parameter(torch.empty(self.rank, factor_sizes[5]))
        for parameter in (
            self.statistic_factor,
            self.target_band_factor,
            self.source_band_factor,
            self.target_node_factor,
            self.source_node_factor,
            self.time_factor,
        ):
            nn.init.normal_(parameter, generator=generator)
        self.class_factor = nn.Parameter(torch.zeros(int(n_classes), self.rank))
        self.bias = nn.Parameter(torch.zeros(int(n_classes)))
        self.register_buffer(
            "output_gain",
            torch.tensor(math.sqrt(float(self.n_bands * self.n_nodes))),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        factors = tuple(
            F.normalize(parameter, dim=1, eps=1e-6)
            for parameter in (
                self.statistic_factor,
                self.target_band_factor,
                self.source_band_factor,
                self.target_node_factor,
                self.source_node_factor,
                self.time_factor,
            )
        )
        within_shape = (
            self.n_statistics,
            self.n_bands,
            self.n_nodes,
            self.n_nodes,
            self.n_bins,
        )
        cross_shape = (
            self.n_statistics,
            self.n_bands,
            self.n_bands,
            self.n_nodes,
            self.n_nodes,
            self.n_bins,
        )
        if features.ndim == 6 and tuple(features.shape[1:]) == within_shape:
            components = torch.einsum(
                "nmaijt,rm,ra,ra,ri,rj,rt->nr",
                features,
                *factors,
            )
        elif features.ndim == 7 and tuple(features.shape[1:]) == cross_shape:
            components = torch.einsum(
                "nmabijt,rm,ra,rb,ri,rj,rt->nr",
                features,
                *factors,
            )
        else:
            raise ValueError(
                "route statistics must retain statistic, band-pair, node-pair, "
                "and time-bin axes"
            )
        return F.linear(
            components * self.output_gain.to(components),
            self.class_factor,
            self.bias,
        )


class DelayedFullWindowATCReadout(nn.Module):
    """Feed a transported carrier to a locked official ATCNet implementation.

    This adapter is for the Gate-A full-window comparison. Earlier endpoint
    slots repeat the final logit and must not be interpreted as early-decision
    predictions. E4 uses the causal decoder for valid prefix metrics.
    """

    prefix_semantics = "final_only_repeated_not_for_early_decision"
    supports_broadband_fusion = True

    def __init__(
        self,
        core: nn.Module,
        n_bands: int,
        n_nodes: int,
        n_classes: int,
        *,
        endpoint_samples: Sequence[int],
        node_reconstruction: torch.Tensor | None = None,
        delayed_residual_bound: float = 0.0,
        delayed_fusion_mode: str = "additive",
        delayed_statistics_bins: int = 0,
        delayed_statistics_interaction_gain: float = 0.0,
        delayed_statistics_uncertainty_gate: bool = False,
        delayed_statistics_cp_rank: int = 0,
        delayed_statistics_directional_moments: bool = False,
        delayed_route_statistics_bins: int = 0,
        delayed_route_statistics_cp_rank: int = 0,
    ) -> None:
        super().__init__()
        self.core = core
        self.n_bands = int(n_bands)
        self.n_nodes = int(n_nodes)
        self.n_classes = int(n_classes)
        self.endpoint_samples = tuple(int(value) for value in endpoint_samples)
        if not self.endpoint_samples:
            raise ValueError("ATC endpoints must not be empty")
        self.delayed_residual_bound = float(delayed_residual_bound)
        if self.delayed_residual_bound < 0.0:
            raise ValueError("delayed ATC residual bound must be non-negative")
        self.delayed_fusion_mode = str(delayed_fusion_mode).lower()
        if self.delayed_fusion_mode not in {"additive", "amplitude_modulation"}:
            raise ValueError(
                "delayed ATC fusion mode must be 'additive' or 'amplitude_modulation'"
            )
        self.delayed_residual_mix_raw = nn.Parameter(torch.zeros(()))
        self.delayed_statistics_bins = int(delayed_statistics_bins)
        self.delayed_statistics_interaction_gain = float(
            delayed_statistics_interaction_gain
        )
        self.delayed_statistics_directional_moments = bool(
            delayed_statistics_directional_moments
        )
        if self.delayed_statistics_interaction_gain < 0.0:
            raise ValueError("delayed statistics interaction gain must be non-negative")
        if self.delayed_statistics_bins < 0:
            raise ValueError("delayed statistics bins must be non-negative")
        if self.delayed_statistics_bins:
            if int(delayed_statistics_cp_rank) > 0:
                self.delayed_statistics_classifier = DelayedTensorCPClassifier(
                    self.n_bands,
                    self.n_nodes,
                    self.delayed_statistics_bins,
                    self.n_classes,
                    int(delayed_statistics_cp_rank),
                    n_statistics=(
                        4 if self.delayed_statistics_directional_moments else 2
                    ),
                )
            else:
                feature_count = (
                    (4 if self.delayed_statistics_directional_moments else 2)
                    * self.n_bands
                    * self.n_nodes
                    * self.delayed_statistics_bins
                )
                self.delayed_statistics_classifier = nn.Linear(
                    feature_count,
                    self.n_classes,
                )
                nn.init.zeros_(self.delayed_statistics_classifier.weight)
                nn.init.zeros_(self.delayed_statistics_classifier.bias)
            if bool(delayed_statistics_uncertainty_gate):
                self.delayed_statistics_gate_bias = nn.Parameter(torch.tensor(1.0))
                self.delayed_statistics_gate_slope_raw = nn.Parameter(
                    torch.tensor(0.5413248546)
                )
            else:
                self.register_parameter("delayed_statistics_gate_bias", None)
                self.register_parameter("delayed_statistics_gate_slope_raw", None)
        else:
            self.delayed_statistics_classifier = None
            self.register_parameter("delayed_statistics_gate_bias", None)
            self.register_parameter("delayed_statistics_gate_slope_raw", None)
        self.delayed_route_statistics_bins = int(delayed_route_statistics_bins)
        if self.delayed_route_statistics_bins < 0:
            raise ValueError("delayed route statistics bins must be non-negative")
        if self.delayed_route_statistics_bins:
            self.delayed_route_statistics_classifier = (
                DelayedRouteTensorCPClassifier(
                    self.n_bands,
                    self.n_nodes,
                    self.delayed_route_statistics_bins,
                    self.n_classes,
                    int(delayed_route_statistics_cp_rank),
                )
            )
        else:
            self.delayed_route_statistics_classifier = None
        if node_reconstruction is None:
            self.register_buffer("node_reconstruction", None)
        else:
            reconstruction = torch.as_tensor(node_reconstruction, dtype=torch.float32)
            if reconstruction.ndim != 2 or reconstruction.shape[1] != self.n_nodes:
                raise ValueError("full-window ATC reconstruction must have shape [C, K]")
            if not bool(torch.isfinite(reconstruction).all()):
                raise ValueError("ATC reconstruction must be finite")
            self.register_buffer("node_reconstruction", reconstruction)

    def _synthesize(self, delayed_carrier: torch.Tensor) -> torch.Tensor:
        if delayed_carrier.ndim == 4:
            if delayed_carrier.shape[1:3] != (self.n_bands, self.n_nodes):
                raise ValueError("delayed ATC band/node axes do not match the model")
            carrier = delayed_carrier.sum(dim=1) / math.sqrt(self.n_bands)
        elif delayed_carrier.ndim == 3:
            if delayed_carrier.shape[1] != self.n_nodes:
                raise ValueError("delayed ATC node axis does not match the model")
            carrier = delayed_carrier
        else:
            raise ValueError("delayed ATC input must be [N, B, K, T] or [N, K, T]")
        if carrier.is_complex():
            raise ValueError("delayed ATC input must be real")
        if self.node_reconstruction is None:
            return carrier
        return torch.einsum(
            "ck,nkt->nct", self.node_reconstruction.to(carrier), carrier
        )

    @property
    def delayed_residual_scale(self) -> torch.Tensor:
        return self.delayed_residual_bound * torch.tanh(self.delayed_residual_mix_raw)

    def _delayed_statistics(
        self,
        delayed: torch.Tensor,
        *,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.delayed_statistics_classifier is None:
            return None
        if delayed.ndim != 4 or delayed.shape[1:3] != (self.n_bands, self.n_nodes):
            raise ValueError("delayed statistics require [N, B, K, T] current")
        flat = delayed.reshape(delayed.shape[0], self.n_bands * self.n_nodes, -1)
        mean = F.adaptive_avg_pool1d(flat, self.delayed_statistics_bins)
        mean_square = F.adaptive_avg_pool1d(
            flat.square(), self.delayed_statistics_bins
        )
        rms = torch.where(
            mean_square == 0,
            torch.zeros_like(mean_square),
            torch.sqrt(mean_square + 1e-8) - 1e-4,
        )
        # Fixed signed compression preserves fold-fitted amplitude while
        # preventing a few high-current routes from dominating the residual.
        mean = torch.sign(mean) * torch.log1p(mean.abs())
        rms = torch.log1p(rms)
        statistic_parts = [mean, rms]
        if self.delayed_statistics_directional_moments:
            if reference is None or reference.shape != delayed.shape:
                raise ValueError(
                    "directional delay statistics require an aligned delayed reference"
                )
            reference_flat = reference.reshape_as(flat)
            cross = F.adaptive_avg_pool1d(
                reference_flat * flat,
                self.delayed_statistics_bins,
            )
            previous = torch.cat(
                (reference_flat[..., :1], reference_flat[..., :-1]),
                dim=-1,
            )
            causal_flux = F.adaptive_avg_pool1d(
                (reference_flat - previous) * flat,
                self.delayed_statistics_bins,
            )
            cross = torch.sign(cross) * torch.log1p(cross.abs())
            causal_flux = torch.sign(causal_flux) * torch.log1p(causal_flux.abs())
            statistic_parts.extend((cross, causal_flux))
        features = torch.stack(
            tuple(
                part.reshape(
                    delayed.shape[0],
                    self.n_bands,
                    self.n_nodes,
                    self.delayed_statistics_bins,
                )
                for part in statistic_parts
            ),
            dim=1,
        )
        if isinstance(self.delayed_statistics_classifier, nn.Linear):
            features = features.flatten(1)
        return self.delayed_statistics_classifier(features)

    def _delayed_route_statistics(
        self,
        route_contrast: torch.Tensor | None,
        *,
        reference_logits: torch.Tensor,
    ) -> torch.Tensor | None:
        classifier = self.delayed_route_statistics_classifier
        if classifier is None:
            return None
        if route_contrast is None:
            return torch.zeros_like(reference_logits)
        if route_contrast.ndim == 5:
            expected = (self.n_bands, self.n_nodes, self.n_nodes)
            if tuple(route_contrast.shape[1:4]) != expected:
                raise ValueError("within-band route contrast axes do not match the model")
            route_shape = tuple(route_contrast.shape[1:-1])
        elif route_contrast.ndim == 6:
            expected = (
                self.n_bands,
                self.n_bands,
                self.n_nodes,
                self.n_nodes,
            )
            if tuple(route_contrast.shape[1:5]) != expected:
                raise ValueError("cross-band route contrast axes do not match the model")
            route_shape = tuple(route_contrast.shape[1:-1])
        else:
            raise ValueError(
                "route contrast must be [N,A,I,J,T] or [N,A,B,I,J,T]"
            )
        flat = route_contrast.reshape(route_contrast.shape[0], -1, route_contrast.shape[-1])
        mean = F.adaptive_avg_pool1d(flat, self.delayed_route_statistics_bins)
        mean_square = F.adaptive_avg_pool1d(
            flat.square(), self.delayed_route_statistics_bins
        )
        rms = torch.where(
            mean_square == 0,
            torch.zeros_like(mean_square),
            torch.sqrt(mean_square + 1e-8) - 1e-4,
        )
        mean = torch.sign(mean) * torch.log1p(mean.abs())
        rms = torch.log1p(rms)
        features = torch.stack(
            tuple(
                part.reshape(
                    route_contrast.shape[0],
                    *route_shape,
                    self.delayed_route_statistics_bins,
                )
                for part in (mean, rms)
            ),
            dim=1,
        )
        return classifier(features)

    def forward(
        self,
        delayed_carrier: torch.Tensor,
        *,
        broadband_carrier: torch.Tensor | None = None,
        statistics_carrier: torch.Tensor | None = None,
        route_statistics_carrier: torch.Tensor | None = None,
        phase_pair_carrier: torch.Tensor | None = None,
    ) -> DelayedFullWindowATCOutput:
        residual = self._synthesize(delayed_carrier)
        if broadband_carrier is None:
            synthesized = residual
        else:
            base = self._synthesize(broadband_carrier)
            if base.shape != residual.shape:
                raise ValueError("broadband and multiband ATC carriers must align")
            scale = self.delayed_residual_scale.to(residual)
            if self.delayed_fusion_mode == "additive":
                synthesized = base + scale * residual
            else:
                synthesized = base * (1.0 + scale * torch.tanh(residual))
        if phase_pair_carrier is not None:
            phase_pair = self._synthesize(phase_pair_carrier)
            if phase_pair.shape != synthesized.shape:
                raise ValueError("phase-pair and ATC carriers must align")
            synthesized = synthesized + phase_pair
        result = self.core(synthesized)
        logits = result["logits"] if isinstance(result, dict) else result
        if logits.ndim != 2 or logits.shape[1] != self.n_classes:
            raise RuntimeError("official ATC core returned an invalid logit tensor")
        statistics_input = delayed_carrier
        statistics_reference = None
        if statistics_carrier is not None and self.delayed_statistics_interaction_gain > 0.0:
            if statistics_carrier.shape != delayed_carrier.shape:
                raise ValueError("interaction statistics must align with delayed current")
            statistics_input = (
                self.delayed_statistics_interaction_gain * statistics_carrier
            )
            statistics_reference = delayed_carrier
        delayed_statistics_logits = self._delayed_statistics(
            statistics_input,
            reference=statistics_reference,
        )
        delayed_statistics_gate = None
        if delayed_statistics_logits is not None:
            if self.delayed_statistics_gate_bias is not None:
                top_two = torch.topk(logits.detach(), k=2, dim=1).values
                margin = top_two[:, 0] - top_two[:, 1]
                slope = F.softplus(self.delayed_statistics_gate_slope_raw)
                delayed_statistics_gate = torch.sigmoid(
                    self.delayed_statistics_gate_bias - slope * margin
                )
                delayed_statistics_logits = (
                    delayed_statistics_logits * delayed_statistics_gate[:, None]
                )
            logits = logits + delayed_statistics_logits
        delayed_route_statistics_logits = self._delayed_route_statistics(
            route_statistics_carrier,
            reference_logits=logits,
        )
        if delayed_route_statistics_logits is not None:
            logits = logits + delayed_route_statistics_logits
        endpoints = logits[:, None].expand(-1, len(self.endpoint_samples), -1)
        return DelayedFullWindowATCOutput(
            logits=endpoints,
            synthesized_carrier=synthesized,
            delayed_statistics_logits=delayed_statistics_logits,
            delayed_route_statistics_logits=delayed_route_statistics_logits,
            delayed_statistics_gate=delayed_statistics_gate,
        )
