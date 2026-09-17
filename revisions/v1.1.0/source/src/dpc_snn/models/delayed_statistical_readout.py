"""FBC-style cumulative statistics computed strictly after delay transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DelayedStatisticalOutput:
    encoded: torch.Tensor
    raw_statistics: torch.Tensor
    carrier_spatial_signals: torch.Tensor
    envelope_spatial_signals: torch.Tensor


class DelayedStatisticalReadout(nn.Module):
    """Band-specific spatial filters followed by cumulative mean/log-variance.

    The input is the aggregated mandatory-delay current. No carrier, envelope,
    covariance, or class feature can enter this module before route transport.
    """

    def __init__(
        self,
        n_bands: int,
        n_nodes: int,
        *,
        spatial_filters: int = 8,
        endpoint_samples: Sequence[int] = (125, 250, 375, 500),
        output_features: int = 96,
        compression_scale: float = 0.10,
        dropout: float = 0.25,
        components: Sequence[str] = (
            "carrier_log_variance",
            "envelope_mean",
            "envelope_log_variance",
        ),
        encoder_kind: str = "linear",
        temporal_segments: int = 1,
        variance_transform: str = "log1p",
        post_spatial_activation: str = "none",
        spatial_initialization: str = "random",
        trainable_spatial: bool = True,
    ) -> None:
        super().__init__()
        self.n_bands = int(n_bands)
        self.n_nodes = int(n_nodes)
        self.spatial_filters = int(spatial_filters)
        self.endpoint_samples = tuple(int(value) for value in endpoint_samples)
        self.compression_scale = float(compression_scale)
        self.components = tuple(str(value) for value in components)
        self.encoder_kind = str(encoder_kind).lower()
        self.temporal_segments = int(temporal_segments)
        self.variance_transform = str(variance_transform).lower()
        self.post_spatial_activation = str(post_spatial_activation).lower()
        self.spatial_initialization = str(spatial_initialization).lower()
        allowed = {
            "carrier_log_variance",
            "envelope_mean",
            "envelope_log_variance",
        }
        if not self.components or len(set(self.components)) != len(self.components):
            raise ValueError("statistical components must be non-empty and unique")
        if not set(self.components) <= allowed:
            raise ValueError("unknown delayed statistical component")
        if self.encoder_kind not in {"fbc", "identity", "linear"}:
            raise ValueError("statistical encoder kind must be 'fbc', 'identity', or 'linear'")
        if self.post_spatial_activation not in {"fbc_scb", "none"}:
            raise ValueError("post-spatial activation must be 'fbc_scb' or 'none'")
        if self.spatial_initialization not in {"random", "identity"}:
            raise ValueError("spatial initialization must be 'random' or 'identity'")
        if self.spatial_filters < 1:
            raise ValueError("statistical spatial filters must be positive")
        if self.temporal_segments < 1:
            raise ValueError("statistical temporal segments must be positive")
        if self.variance_transform not in {"log", "log1p"}:
            raise ValueError("variance transform must be 'log' or 'log1p'")
        if not self.endpoint_samples or tuple(sorted(self.endpoint_samples)) != (
            self.endpoint_samples
        ):
            raise ValueError("statistical endpoints must be non-empty and increasing")
        if self.compression_scale <= 0.0:
            raise ValueError("statistical compression scale must be positive")

        if self.spatial_initialization == "identity":
            if self.spatial_filters != self.n_nodes:
                raise ValueError(
                    "identity statistical initialization requires one filter per node"
                )
            initial = torch.eye(self.n_nodes)[None].expand(
                self.n_bands, -1, -1
            ).clone()
        else:
            generator = torch.Generator().manual_seed(41_731)
            initial = torch.randn(
                self.n_bands,
                self.spatial_filters,
                self.n_nodes,
                generator=generator,
            )
            initial = F.normalize(initial, p=2, dim=-1)
        self.spatial_weight = nn.Parameter(initial, requires_grad=bool(trainable_spatial))
        self.raw_features = (
            len(self.components)
            * self.n_bands
            * self.spatial_filters
            * self.temporal_segments
        )
        if self.post_spatial_activation == "fbc_scb":
            self.carrier_norm = nn.BatchNorm1d(self.n_bands * self.spatial_filters)
        else:
            self.carrier_norm = None
        if self.encoder_kind in {"fbc", "identity"}:
            self.output_features = self.raw_features
            if self.encoder_kind == "fbc":
                self.encoder = nn.Identity()
            else:
                self.encoder = nn.Sequential(
                    nn.LayerNorm(self.raw_features, elementwise_affine=False),
                    nn.Dropout(float(dropout)),
                )
        else:
            self.output_features = int(output_features)
            self.encoder = nn.Sequential(
                nn.LayerNorm(self.raw_features, elementwise_affine=False),
                nn.Linear(self.raw_features, self.output_features, bias=False),
                nn.LayerNorm(self.output_features, elementwise_affine=False),
                nn.ELU(),
                nn.Dropout(float(dropout)),
            )

    def weight(self) -> torch.Tensor:
        return F.normalize(self.spatial_weight, p=2, dim=-1)

    def orthogonality_loss(self) -> torch.Tensor:
        weight = self.weight()
        gram = weight @ weight.transpose(-1, -2)
        identity = torch.eye(
            self.spatial_filters, device=weight.device, dtype=weight.dtype
        )
        return (gram - identity).square().mean()

    def _statistics(
        self,
        carrier: torch.Tensor,
        envelope: torch.Tensor,
        fast_stop: int,
    ) -> torch.Tensor:
        slow_stop = (int(fast_stop) + 1) // 2
        carrier_prefix = carrier[..., :fast_stop]
        envelope_prefix = envelope[..., :slow_stop]
        if carrier_prefix.shape[-1] < self.temporal_segments:
            raise ValueError("carrier prefix is shorter than the statistical segmentation")
        if envelope_prefix.shape[-1] < self.temporal_segments:
            raise ValueError("envelope prefix is shorter than the statistical segmentation")
        carrier_chunks = torch.tensor_split(
            carrier_prefix, self.temporal_segments, dim=-1
        )
        envelope_chunks = torch.tensor_split(
            envelope_prefix, self.temporal_segments, dim=-1
        )
        carrier_variance = torch.stack(
            [chunk.var(dim=-1, unbiased=False) for chunk in carrier_chunks], dim=-1
        )
        envelope_mean = torch.stack(
            [chunk.mean(dim=-1) for chunk in envelope_chunks], dim=-1
        )
        envelope_variance = torch.stack(
            [chunk.var(dim=-1, unbiased=False) for chunk in envelope_chunks], dim=-1
        )
        scale = self.compression_scale
        compressed_envelope_mean = torch.sign(envelope_mean) * torch.log1p(
            envelope_mean.abs() / scale
        )
        if self.variance_transform == "log":
            carrier_log_variance = carrier_variance.clamp(1e-6, 1e6).log()
            envelope_log_variance = envelope_variance.clamp(1e-6, 1e6).log()
        else:
            carrier_log_variance = torch.log1p(carrier_variance / (scale * scale))
            envelope_log_variance = torch.log1p(envelope_variance / (scale * scale))
        values = {
            "carrier_log_variance": carrier_log_variance,
            "envelope_mean": compressed_envelope_mean,
            "envelope_log_variance": envelope_log_variance,
        }
        return torch.cat([values[name].flatten(1) for name in self.components], dim=1)

    def forward(
        self,
        delayed_current: torch.Tensor,
        *,
        fast_current: torch.Tensor | None = None,
        slow_current: torch.Tensor | None = None,
    ) -> DelayedStatisticalOutput:
        if delayed_current.ndim != 4 or delayed_current.shape[1:3] != (
            self.n_bands,
            self.n_nodes,
        ):
            raise ValueError("delayed statistics expect [N, configured B, K, T]")
        if self.endpoint_samples[-1] > delayed_current.shape[-1]:
            raise ValueError("statistical endpoint exceeds delayed sequence length")
        carrier = delayed_current if fast_current is None else fast_current
        envelope = delayed_current[..., ::2] if slow_current is None else slow_current
        if carrier.shape != delayed_current.shape:
            raise ValueError("fast delayed current must align with the fused 125 Hz sequence")
        if envelope.shape[:-1] != delayed_current.shape[:-1] or envelope.shape[-1] * 2 != (
            delayed_current.shape[-1]
        ):
            raise ValueError("slow delayed current must align at exactly half the fast rate")
        weight = self.weight().to(delayed_current)
        carrier_projected = torch.einsum(
            "bfk,nbkt->nbft", weight, carrier
        )
        if self.carrier_norm is not None:
            shape = carrier_projected.shape
            carrier_flat = carrier_projected.reshape(shape[0], shape[1] * shape[2], shape[3])
            carrier_flat = self.carrier_norm(carrier_flat)
            carrier_flat = carrier_flat * torch.sigmoid(carrier_flat)
            carrier_projected = carrier_flat.reshape(shape)
        envelope_projected = torch.einsum(
            "bfk,nbkt->nbft", weight, envelope
        )
        raw = torch.stack(
            [
                self._statistics(carrier_projected, envelope_projected, stop)
                for stop in self.endpoint_samples
            ],
            dim=1,
        )
        encoded = self.encoder(raw)
        return DelayedStatisticalOutput(
            encoded=encoded,
            raw_statistics=raw,
            carrier_spatial_signals=carrier_projected,
            envelope_spatial_signals=envelope_projected,
        )
