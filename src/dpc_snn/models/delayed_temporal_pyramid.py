"""Causal temporal pyramid applied only to aggregated delayed currents."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class BandSharedCausalConvolution(nn.Module):
    """One causal temporal kernel per band, shared across physical nodes."""

    def __init__(self, n_bands: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.n_bands = int(n_bands)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        if self.kernel_size < 1 or self.dilation < 1:
            raise ValueError("kernel_size and dilation must be positive")
        self.weight = nn.Parameter(
            torch.zeros(self.n_bands, 1, self.kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(self.n_bands))
        with torch.no_grad():
            self.weight[:, 0, -1] = 1.0

    @property
    def left_padding(self) -> int:
        return self.dilation * (self.kernel_size - 1)

    def forward(self, delayed: torch.Tensor) -> torch.Tensor:
        if delayed.ndim != 4 or delayed.shape[1] != self.n_bands:
            raise ValueError("band-shared convolution expects delayed [N, B, K, T]")
        n_trials, n_bands, n_nodes, n_time = delayed.shape
        work = delayed.permute(0, 2, 1, 3).reshape(n_trials * n_nodes, n_bands, n_time)
        work = F.pad(work, (self.left_padding, 0))
        output = F.conv1d(
            work,
            self.weight,
            self.bias,
            dilation=self.dilation,
            groups=self.n_bands,
        )
        return output.reshape(n_trials, n_nodes, n_bands, n_time).permute(0, 2, 1, 3)


class _DelayedPyramidBlock(nn.Module):
    def __init__(
        self,
        n_bands: int,
        n_nodes: int,
        dilations: Sequence[int],
        kernel_size: int,
        scale_attention: bool,
    ) -> None:
        super().__init__()
        self.n_bands = int(n_bands)
        self.n_nodes = int(n_nodes)
        self.branches = nn.ModuleList(
            [
                BandSharedCausalConvolution(self.n_bands, kernel_size, int(dilation))
                for dilation in dilations
            ]
        )
        self.scale_logits = nn.Parameter(torch.zeros(len(self.branches)))
        channels = self.n_bands * self.n_nodes
        bottleneck = max(16, min(40, channels // 4))
        self.pointwise_in = nn.Conv1d(channels, bottleneck, kernel_size=1, bias=False)
        self.pointwise_out = nn.Conv1d(bottleneck, channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.pointwise_in.weight, a=5**0.5)
        nn.init.zeros_(self.pointwise_out.weight)
        self.scale_attention = bool(scale_attention)
        self.attention = (
            nn.Conv1d(channels, len(self.branches), kernel_size=1)
            if self.scale_attention
            else None
        )
        if self.attention is not None:
            nn.init.zeros_(self.attention.weight)
            nn.init.zeros_(self.attention.bias)

    def forward(self, delayed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        branch_values = torch.stack([branch(delayed) for branch in self.branches], dim=1)
        base_weight = F.softplus(self.scale_logits)
        base_weight = base_weight / base_weight.sum().clamp_min(1e-8)
        if self.attention is None:
            weight = base_weight.view(1, -1, 1, 1, 1)
        else:
            flat = delayed.flatten(1, 2)
            local = torch.sigmoid(self.attention(flat))
            local = local / local.sum(dim=1, keepdim=True).clamp_min(1e-8)
            weight = (
                local[:, :, None, None, :]
                * base_weight.view(1, -1, 1, 1, 1)
            )
            weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
        fused = (branch_values * weight).sum(dim=1)
        mixed = self.pointwise_out(F.elu(self.pointwise_in(fused.flatten(1, 2))))
        mixed = mixed.reshape_as(delayed)
        return delayed + F.elu(mixed), weight


class DelayedTemporalPyramid(nn.Module):
    """Multi-scale causal feature pyramid after mandatory route aggregation."""

    def __init__(
        self,
        n_bands: int = 12,
        n_nodes: int = 16,
        dilations: Sequence[int] = (1, 2, 4, 8),
        kernel_size: int = 5,
        depth: int = 2,
        scale_attention: bool = True,
    ) -> None:
        super().__init__()
        if not dilations:
            raise ValueError("the delayed temporal pyramid needs at least one scale")
        if int(depth) < 1:
            raise ValueError("temporal pyramid depth must be positive")
        self.n_bands = int(n_bands)
        self.n_nodes = int(n_nodes)
        self.blocks = nn.ModuleList(
            [
                _DelayedPyramidBlock(
                    self.n_bands,
                    self.n_nodes,
                    dilations,
                    int(kernel_size),
                    bool(scale_attention),
                )
                for _ in range(int(depth))
            ]
        )

    def forward(self, delayed_current: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if delayed_current.ndim != 4:
            raise ValueError(
                "temporal pyramid accepts only aggregated delayed [N, B, K, T] currents"
            )
        output = delayed_current
        weights = []
        for block in self.blocks:
            output, scale_weight = block(output)
            weights.append(scale_weight)
        return output, weights
