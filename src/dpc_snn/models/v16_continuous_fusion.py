"""Continuous dual-view fusion used by the E16 information-sufficiency gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class ContinuousTemporalBlock(nn.Module):
    """Pre-normalized depthwise temporal convolution with a residual path."""

    def __init__(self, channels: int, *, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        if channels < 1 or kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("channels must be positive and kernel_size must be positive odd")
        self.norm = nn.LayerNorm(channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(channels, 2 * channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(2 * channels, channels, kernel_size=1, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        residual = sequence
        value = self.norm(sequence).transpose(1, 2)
        value = self.pointwise(self.depthwise(value)).transpose(1, 2)
        return residual + value


@dataclass(frozen=True)
class V16ContinuousOutput:
    logits: torch.Tensor
    prefix_logits: torch.Tensor
    fused_sequence: torch.Tensor


class V16ContinuousFusion(nn.Module):
    """Structure-preserving continuous fusion of frozen ATCNet/FBCNet views.

    FBCNet's 288 features are retained as nine bands by 32 spatial filters.
    No teacher logits enter the forward path; they are reporting anchors only
    during E16-A.
    """

    architecture_version = "dpc_snn_v16_continuous_fusion_r1"

    def __init__(
        self,
        *,
        atc_features: int = 32,
        atc_steps: int = 18,
        fbc_bands: int = 9,
        fbc_spatial_features: int = 32,
        fbc_steps: int = 4,
        hidden_channels: int = 64,
        band_channels: int = 8,
        temporal_layers: int = 2,
        n_classes: int = 4,
        dropout: float = 0.1,
        endpoint_steps: Sequence[int] = (3, 5, 9, 18),
    ) -> None:
        super().__init__()
        dimensions = (
            atc_features,
            atc_steps,
            fbc_bands,
            fbc_spatial_features,
            fbc_steps,
            hidden_channels,
            band_channels,
            temporal_layers,
            n_classes,
        )
        if min(dimensions) < 1:
            raise ValueError("all V16 dimensions must be positive")
        endpoints = tuple(int(step) for step in endpoint_steps)
        if not endpoints or tuple(sorted(set(endpoints))) != endpoints:
            raise ValueError("endpoint_steps must be non-empty, unique, and increasing")
        if endpoints[-1] != int(atc_steps) or endpoints[0] < 1:
            raise ValueError("endpoint_steps must terminate at atc_steps")

        self.atc_features = int(atc_features)
        self.atc_steps = int(atc_steps)
        self.fbc_bands = int(fbc_bands)
        self.fbc_spatial_features = int(fbc_spatial_features)
        self.fbc_steps = int(fbc_steps)
        self.hidden_channels = int(hidden_channels)
        self.endpoint_steps = endpoints

        self.atc_norm = nn.LayerNorm(self.atc_features)
        self.atc_projection = nn.Linear(self.atc_features, self.hidden_channels, bias=False)
        self.atc_temporal = ContinuousTemporalBlock(
            self.hidden_channels, dropout=float(dropout)
        )

        self.fbc_norm = nn.LayerNorm(self.fbc_spatial_features)
        self.fbc_band_weight = nn.Parameter(
            torch.empty(self.fbc_bands, self.fbc_spatial_features, int(band_channels))
        )
        nn.init.xavier_uniform_(self.fbc_band_weight)
        self.fbc_projection = nn.Linear(
            self.fbc_bands * int(band_channels), self.hidden_channels, bias=False
        )
        self.fbc_temporal = ContinuousTemporalBlock(
            self.hidden_channels, dropout=float(dropout)
        )

        self.interaction = nn.Sequential(
            nn.LayerNorm(4 * self.hidden_channels),
            nn.Linear(4 * self.hidden_channels, self.hidden_channels, bias=False),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.fusion_norm = nn.LayerNorm(self.hidden_channels)
        self.fusion_temporal = nn.Sequential(
            *[
                ContinuousTemporalBlock(self.hidden_channels, dropout=float(dropout))
                for _ in range(int(temporal_layers))
            ]
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(3 * self.hidden_channels),
            nn.Linear(3 * self.hidden_channels, self.hidden_channels),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_channels, int(n_classes)),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate(self, atc: torch.Tensor, fbc: torch.Tensor) -> None:
        if atc.ndim != 3 or atc.shape[1:] != (self.atc_steps, self.atc_features):
            raise ValueError(
                f"ATC sequence must have shape [N, {self.atc_steps}, {self.atc_features}]"
            )
        expected_fbc = self.fbc_bands * self.fbc_spatial_features
        if fbc.ndim != 3 or fbc.shape[1:] != (self.fbc_steps, expected_fbc):
            raise ValueError(
                f"FBC sequence must have shape [N, {self.fbc_steps}, {expected_fbc}]"
            )
        if atc.shape[0] != fbc.shape[0]:
            raise ValueError("ATC and FBC batches must be aligned")

    def encode(self, atc_sequence: torch.Tensor, fbc_sequence: torch.Tensor) -> torch.Tensor:
        self._validate(atc_sequence, fbc_sequence)
        atc = self.atc_temporal(self.atc_projection(self.atc_norm(atc_sequence)))

        fbc = fbc_sequence.reshape(
            fbc_sequence.shape[0],
            self.fbc_steps,
            self.fbc_bands,
            self.fbc_spatial_features,
        )
        fbc = self.fbc_norm(fbc)
        fbc = torch.einsum("ntbf,bfh->ntbh", fbc, self.fbc_band_weight)
        fbc = self.fbc_projection(fbc.flatten(start_dim=2))
        fbc = self.fbc_temporal(fbc)
        fbc = F.interpolate(
            fbc.transpose(1, 2), size=self.atc_steps, mode="linear", align_corners=False
        ).transpose(1, 2)

        interactions = torch.cat((atc, fbc, atc * fbc, torch.abs(atc - fbc)), dim=-1)
        fused = 0.5 * (atc + fbc) + self.interaction(interactions)
        return self.fusion_temporal(self.fusion_norm(fused))

    def _prefix_readout(self, sequence: torch.Tensor, stop: int) -> torch.Tensor:
        prefix = sequence[:, :stop]
        summary = torch.cat(
            (prefix.mean(dim=1), prefix.std(dim=1, unbiased=False), prefix[:, -1]), dim=1
        )
        return self.readout(summary)

    def forward(
        self, atc_sequence: torch.Tensor, fbc_sequence: torch.Tensor
    ) -> V16ContinuousOutput:
        fused = self.encode(atc_sequence, fbc_sequence)
        prefix_logits = torch.stack(
            [self._prefix_readout(fused, stop) for stop in self.endpoint_steps], dim=1
        )
        return V16ContinuousOutput(
            logits=prefix_logits[:, -1],
            prefix_logits=prefix_logits,
            fused_sequence=fused,
        )
