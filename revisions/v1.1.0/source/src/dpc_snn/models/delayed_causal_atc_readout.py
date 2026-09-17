"""Causal ATC-style temporal decoding after mandatory delay transport."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class _MaxNormConv2d(nn.Conv2d):
    def __init__(self, *args: object, max_norm: float, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.max_norm = float(max_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.renorm(self.weight, p=2, dim=0, maxnorm=self.max_norm)
        return F.conv2d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class _MaxNormLinear(nn.Linear):
    def __init__(self, *args: object, max_norm: float, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.max_norm = float(max_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.renorm(self.weight, p=2, dim=0, maxnorm=self.max_norm)
        return F.linear(x, weight, self.bias)


class _CausalConv1d(nn.Conv1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = (self.kernel_size[0] - 1) * self.dilation[0]
        return super().forward(F.pad(x, (padding, 0)))


class _CausalATCFrontEnd(nn.Module):
    """ATCNet convolutional block with strictly left-sided temporal padding."""

    def __init__(
        self,
        n_nodes: int,
        *,
        temporal_filters: int,
        depth_multiplier: int,
        temporal_kernel: int,
        first_pool: int,
        second_pool: int,
        dropout: float,
    ) -> None:
        super().__init__()
        channels = int(temporal_filters) * int(depth_multiplier)
        self.temporal_kernel = int(temporal_kernel)
        self.second_kernel = 16
        self.first_pool = int(first_pool)
        self.second_pool = int(second_pool)
        self.temporal_conv = nn.Conv2d(
            1,
            int(temporal_filters),
            (1, self.temporal_kernel),
            bias=False,
        )
        self.temporal_norm = nn.BatchNorm2d(int(temporal_filters))
        self.spatial_conv = _MaxNormConv2d(
            int(temporal_filters),
            channels,
            (int(n_nodes), 1),
            groups=int(temporal_filters),
            bias=False,
            max_norm=1.0,
        )
        self.spatial_norm = nn.BatchNorm2d(channels)
        self.first_pooling = nn.AvgPool2d((1, self.first_pool))
        self.first_dropout = nn.Dropout(float(dropout))
        self.temporal_refinement = nn.Conv2d(
            channels,
            channels,
            (1, self.second_kernel),
            bias=False,
        )
        self.refinement_norm = nn.BatchNorm2d(channels)
        self.second_pooling = nn.AvgPool2d((1, self.second_pool))
        self.second_dropout = nn.Dropout(float(dropout))

    @property
    def total_pooling_stride(self) -> int:
        return self.first_pool * self.second_pool

    def output_steps(self, input_samples: int) -> int:
        return (int(input_samples) // self.first_pool) // self.second_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("causal ATC front end expects [N, nodes, T]")
        x = x[:, None]
        x = F.pad(x, (self.temporal_kernel - 1, 0, 0, 0))
        x = self.temporal_norm(self.temporal_conv(x))
        x = self.spatial_conv(x)
        x = F.elu(self.spatial_norm(x))
        x = self.first_dropout(self.first_pooling(x))
        x = F.pad(x, (self.second_kernel - 1, 0, 0, 0))
        x = self.temporal_refinement(x)
        x = F.elu(self.refinement_norm(x))
        x = self.second_dropout(self.second_pooling(x))
        return x.squeeze(2).transpose(1, 2)


class _CausalSelfAttention(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        key_features: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.heads = int(heads)
        self.key_features = int(key_features)
        self.norm = nn.LayerNorm(int(features))
        self.query = nn.Linear(int(features), self.heads * self.key_features)
        self.key = nn.Linear(int(features), self.heads * self.key_features)
        self.value = nn.Linear(int(features), self.heads * self.key_features)
        self.output = nn.Linear(self.heads * self.key_features, int(features))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        normalized = self.norm(x)
        batch, steps, _ = normalized.shape

        def heads(value: torch.Tensor) -> torch.Tensor:
            return value.view(batch, steps, self.heads, self.key_features).transpose(1, 2)

        query = heads(self.query(normalized))
        key = heads(self.key(normalized))
        value = heads(self.value(normalized))
        score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.key_features)
        causal_mask = torch.ones(
            steps, steps, device=x.device, dtype=torch.bool
        ).triu(diagonal=1)
        score = score.masked_fill(causal_mask[None, None], torch.finfo(score.dtype).min)
        probability = torch.softmax(score, dim=-1)
        attended = torch.matmul(probability, value)
        attended = attended.transpose(1, 2).reshape(batch, steps, -1)
        return residual + self.dropout(self.output(attended))


class _CausalTCNBlock(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        kernel: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.first = _CausalConv1d(
            int(features),
            int(features),
            kernel_size=int(kernel),
            dilation=int(dilation),
        )
        self.first_norm = nn.BatchNorm1d(int(features))
        self.first_dropout = nn.Dropout(float(dropout))
        self.second = _CausalConv1d(
            int(features),
            int(features),
            kernel_size=int(kernel),
            dilation=int(dilation),
        )
        self.second_norm = nn.BatchNorm1d(int(features))
        self.second_dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.first_dropout(F.elu(self.first_norm(self.first(x))))
        x = self.second_dropout(F.elu(self.second_norm(self.second(x))))
        return F.elu(residual + x)


class _CausalATCHead(nn.Module):
    def __init__(
        self,
        features: int,
        n_classes: int,
        *,
        key_features: int,
        heads: int,
        attention_dropout: float,
        tcn_depth: int,
        tcn_kernel: int,
        tcn_dropout: float,
    ) -> None:
        super().__init__()
        self.attention = _CausalSelfAttention(
            int(features),
            key_features=int(key_features),
            heads=int(heads),
            dropout=float(attention_dropout),
        )
        self.tcn = nn.ModuleList(
            [
                _CausalTCNBlock(
                    int(features),
                    kernel=int(tcn_kernel),
                    dilation=2**depth,
                    dropout=float(tcn_dropout),
                )
                for depth in range(int(tcn_depth))
            ]
        )
        self.classifier = _MaxNormLinear(
            int(features), int(n_classes), max_norm=0.25
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x).transpose(1, 2)
        for block in self.tcn:
            x = block(x)
        return self.classifier(x[..., -1])


@dataclass(frozen=True)
class DelayedCausalATCOutput:
    logits: torch.Tensor
    synthesized_carrier: torch.Tensor


class DelayedCausalATCReadout(nn.Module):
    """ATCNet-derived temporal classifier fed only delayed carrier currents.

    Equal-weight band synthesis occurs after delay transport. Each endpoint is
    evaluated from its own prefix, so neither batch normalization nor attention
    can expose an early-decision logit to later EEG samples.
    """

    def __init__(
        self,
        n_bands: int,
        n_nodes: int,
        n_classes: int,
        *,
        endpoint_samples: Sequence[int],
        node_reconstruction: torch.Tensor | None = None,
        temporal_filters: int = 16,
        depth_multiplier: int = 2,
        temporal_kernel: int = 64,
        first_pool: int = 8,
        second_pool: int = 7,
        convolution_dropout: float = 0.3,
        key_features: int = 8,
        attention_heads: int = 2,
        attention_dropout: float = 0.5,
        tcn_depth: int = 2,
        tcn_kernel: int = 4,
        tcn_dropout: float = 0.3,
        windows: int = 5,
    ) -> None:
        super().__init__()
        self.n_bands = int(n_bands)
        self.n_nodes = int(n_nodes)
        self.n_classes = int(n_classes)
        self.endpoint_samples = tuple(int(value) for value in endpoint_samples)
        self.windows = int(windows)
        if not self.endpoint_samples or tuple(sorted(self.endpoint_samples)) != (
            self.endpoint_samples
        ):
            raise ValueError("ATC endpoints must be non-empty and increasing")
        if self.windows < 1:
            raise ValueError("ATC window count must be positive")
        if node_reconstruction is None:
            self.register_buffer("node_reconstruction", None)
            frontend_nodes = self.n_nodes
        else:
            reconstruction = torch.as_tensor(node_reconstruction, dtype=torch.float32)
            valid = (
                reconstruction.ndim == 2
                and reconstruction.shape[1] == self.n_nodes
            ) or (
                reconstruction.ndim == 3
                and reconstruction.shape[0] == self.n_bands
                and reconstruction.shape[2] == self.n_nodes
            )
            if not valid:
                raise ValueError(
                    "ATC node reconstruction must have shape [C, K] or [B, C, K]"
                )
            if reconstruction.shape[-1] != self.n_nodes:
                raise ValueError("ATC node reconstruction has an incompatible node axis")
            if not bool(torch.isfinite(reconstruction).all()):
                raise ValueError("ATC node reconstruction must be finite")
            self.register_buffer("node_reconstruction", reconstruction)
            frontend_nodes = int(reconstruction.shape[-2])
        features = int(temporal_filters) * int(depth_multiplier)
        self.frontend = _CausalATCFrontEnd(
            frontend_nodes,
            temporal_filters=int(temporal_filters),
            depth_multiplier=int(depth_multiplier),
            temporal_kernel=int(temporal_kernel),
            first_pool=int(first_pool),
            second_pool=int(second_pool),
            dropout=float(convolution_dropout),
        )
        if min(self.frontend.output_steps(stop) for stop in self.endpoint_samples) < 1:
            raise ValueError("ATC endpoint is too short for the registered pooling strides")
        self.heads = nn.ModuleList(
            [
                _CausalATCHead(
                    features,
                    self.n_classes,
                    key_features=int(key_features),
                    heads=int(attention_heads),
                    attention_dropout=float(attention_dropout),
                    tcn_depth=int(tcn_depth),
                    tcn_kernel=int(tcn_kernel),
                    tcn_dropout=float(tcn_dropout),
                )
                for _ in range(self.windows)
            ]
        )
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _classify_prefix(self, sequence: torch.Tensor) -> torch.Tensor:
        active_windows = min(self.windows, sequence.shape[1])
        window_length = sequence.shape[1] - active_windows + 1
        logits = [
            self.heads[index](sequence[:, index : index + window_length])
            for index in range(active_windows)
        ]
        return torch.stack(logits, dim=0).mean(dim=0)

    def forward(self, delayed_carrier: torch.Tensor) -> DelayedCausalATCOutput:
        if delayed_carrier.ndim == 4:
            if delayed_carrier.shape[1:3] != (self.n_bands, self.n_nodes):
                raise ValueError("delayed ATC band/node axes do not match the model")
        elif delayed_carrier.ndim == 3:
            if delayed_carrier.shape[1] != self.n_nodes:
                raise ValueError("delayed ATC node axis does not match the model")
        else:
            raise ValueError("delayed ATC readout expects [N, B, K, T] or [N, K, T]")
        if delayed_carrier.is_complex():
            raise ValueError("delayed ATC readout requires real transported current")
        if self.endpoint_samples[-1] > delayed_carrier.shape[-1]:
            raise ValueError("ATC endpoint exceeds the delayed carrier sequence")
        if delayed_carrier.ndim == 3:
            if self.node_reconstruction is None:
                synthesized = delayed_carrier
            elif self.node_reconstruction.ndim == 2:
                synthesized = torch.einsum(
                    "ck,nkt->nct",
                    self.node_reconstruction.to(delayed_carrier),
                    delayed_carrier,
                )
            else:
                raise ValueError(
                    "wideband ATC input requires one shared [C, K] reconstruction"
                )
        elif self.node_reconstruction is None:
            synthesized = delayed_carrier.sum(dim=1) / math.sqrt(self.n_bands)
        elif self.node_reconstruction.ndim == 2:
            synthesized = torch.einsum(
                "ck,nbkt->nct",
                self.node_reconstruction.to(delayed_carrier),
                delayed_carrier,
            ) / math.sqrt(self.n_bands)
        else:
            synthesized = torch.einsum(
                "bck,nbkt->nct",
                self.node_reconstruction.to(delayed_carrier),
                delayed_carrier,
            ) / math.sqrt(self.n_bands)
        sequence = self.frontend(synthesized)
        endpoint_logits = [
            self._classify_prefix(
                sequence[:, : self.frontend.output_steps(stop)]
            )
            for stop in self.endpoint_samples
        ]
        return DelayedCausalATCOutput(
            logits=torch.stack(endpoint_logits, dim=1),
            synthesized_carrier=synthesized,
        )
