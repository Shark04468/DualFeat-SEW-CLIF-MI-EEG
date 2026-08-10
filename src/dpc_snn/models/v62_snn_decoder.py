"""Heterogeneous causal SNN decoder for DASP-SNN V6.2-R1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .lif import CLIFLayer
from .surrogate import spike_fn


DecoderKind = Literal["clif", "plif", "ann"]
ResidualMode = Literal["plain", "sew_add"]


def _channel_decays(channels: int, decays: Sequence[float]) -> torch.Tensor:
    if not decays:
        raise ValueError("at least one decoder decay is required")
    values = torch.as_tensor(tuple(float(value) for value in decays), dtype=torch.float32)
    if bool(((values <= 0.0) | (values >= 1.0)).any()):
        raise ValueError("decoder decays must lie strictly between zero and one")
    index = torch.arange(int(channels)) * len(values) // int(channels)
    return values[index.clamp_max(len(values) - 1)]


class HeterogeneousCLIF(nn.Module):
    def __init__(self, channels: int, decays: Sequence[float], threshold: float) -> None:
        super().__init__()
        initial = _channel_decays(channels, decays)
        self.layer = CLIFLayer(
            decay=0.9,
            threshold=float(threshold),
            channels=int(channels),
            learnable_decay=True,
            membrane_norm_groups=0,
            causal_channel_norm=False,
        )
        with torch.no_grad():
            normalized = ((initial - 0.02) / 0.96).clamp(1e-5, 1.0 - 1e-5)
            self.layer.decay_raw.copy_(torch.logit(normalized))

    def forward(self, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.layer(current)


class MatchedPLIF(nn.Module):
    """PLIF control using the same ``decay*u + current`` charging as CLIF."""

    def __init__(self, channels: int, decays: Sequence[float], threshold: float) -> None:
        super().__init__()
        initial = _channel_decays(channels, decays)
        self.decay_raw = nn.Parameter(torch.logit(initial))
        self.threshold = float(threshold)

    @property
    def decay(self) -> torch.Tensor:
        return torch.sigmoid(self.decay_raw).clamp(0.02, 0.98)

    def forward(self, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        membrane = torch.zeros_like(current[..., 0])
        decay = self.decay.to(current)[None]
        spikes = []
        membranes = []
        for step in range(current.shape[-1]):
            membrane = decay * membrane + current[..., step]
            spike = spike_fn(membrane - self.threshold)
            membrane = membrane - spike * self.threshold
            spikes.append(spike)
            membranes.append(membrane)
        return torch.stack(spikes, dim=-1), torch.stack(membranes, dim=-1)


class HeterogeneousANN(nn.Module):
    def __init__(self, channels: int, decays: Sequence[float]) -> None:
        super().__init__()
        initial = _channel_decays(channels, decays)
        self.decay_raw = nn.Parameter(torch.logit(initial))

    @property
    def decay(self) -> torch.Tensor:
        return torch.sigmoid(self.decay_raw).clamp(0.02, 0.98)

    def forward(self, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        steps = current.shape[-1]
        lag = torch.arange(steps - 1, -1, -1, device=current.device, dtype=current.dtype)
        kernel = self.decay.to(current)[:, None, None].pow(lag[None, None, :])
        state = F.conv1d(
            F.pad(current, (steps - 1, 0)),
            kernel,
            groups=current.shape[1],
        )
        return torch.tanh(state), state


def _make_state_layer(
    kind: DecoderKind,
    channels: int,
    decays: Sequence[float],
    threshold: float,
) -> nn.Module:
    if kind == "clif":
        return HeterogeneousCLIF(channels, decays, threshold)
    if kind == "plif":
        return MatchedPLIF(channels, decays, threshold)
    if kind == "ann":
        return HeterogeneousANN(channels, decays)
    raise ValueError("decoder kind must be 'clif', 'plif', or 'ann'")


class CausalDepthwiseSEWBlock(nn.Module):
    """Causal temporal branch with an explicit plain or SEW-ADD merge."""

    def __init__(
        self,
        channels: int,
        *,
        kind: DecoderKind,
        decays: Sequence[float],
        threshold: float,
        residual_mode: ResidualMode = "sew_add",
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        if residual_mode not in ("plain", "sew_add"):
            raise ValueError("decoder residual mode must be 'plain' or 'sew_add'")
        self.residual_mode: ResidualMode = residual_mode
        self.depthwise = nn.Conv1d(
            self.channels,
            self.channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            groups=self.channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(self.channels, self.channels, kernel_size=1, bias=False)
        self.neuron = _make_state_layer(kind, self.channels, decays, threshold)
        nn.init.zeros_(self.depthwise.weight)
        with torch.no_grad():
            self.depthwise.weight[:, 0, -1] = 1.0
        nn.init.normal_(self.pointwise.weight, mean=0.0, std=0.02)

    @property
    def left_padding(self) -> int:
        return (self.kernel_size - 1) * self.dilation

    def forward(
        self, activity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        branch_current = self.depthwise(F.pad(activity, (self.left_padding, 0)))
        branch, membrane = self.neuron(self.pointwise(branch_current))
        merged = branch if self.residual_mode == "plain" else activity + branch
        return merged, branch, membrane


@dataclass(frozen=True)
class DecoderOutput:
    logits: torch.Tensor
    final_spikes: torch.Tensor
    final_membrane: torch.Tensor
    binary_spikes: tuple[torch.Tensor, ...]
    residual_activities: tuple[torch.Tensor, ...]
    endpoint_features: torch.Tensor
    firing_rate_loss: torch.Tensor


class V62SNNDecoder(nn.Module):
    """Signed-population CLIF/SEW decoder with causal prefix readout."""

    def __init__(
        self,
        n_bands: int = 12,
        n_nodes: int = 16,
        n_classes: int = 4,
        snn_channels: int = 64,
        decoder_kind: DecoderKind = "clif",
        decoder_residual_mode: ResidualMode = "sew_add",
        decoder_layers: int = 2,
        decays: Sequence[float] = (0.65, 0.90, 0.975),
        threshold: float = 1.0,
        sfreq: float = 125.0,
        endpoint_seconds: Sequence[float] = (1.0, 2.0, 3.0, 4.0),
        readout_features: int = 96,
        dropout: float = 0.25,
        firing_rate_target: float = 0.12,
    ) -> None:
        super().__init__()
        self.n_bands = int(n_bands)
        self.n_nodes = int(n_nodes)
        self.n_classes = int(n_classes)
        self.snn_channels = int(snn_channels)
        self.decoder_kind: DecoderKind = decoder_kind
        if decoder_residual_mode not in ("plain", "sew_add"):
            raise ValueError("decoder residual mode must be 'plain' or 'sew_add'")
        self.decoder_residual_mode: ResidualMode = decoder_residual_mode
        if self.snn_channels % 2:
            raise ValueError("snn_channels must be even for signed populations")
        if int(decoder_layers) < 0:
            raise ValueError("decoder_layers must be non-negative")
        self.signed_channels = self.snn_channels // 2
        self.sfreq = float(sfreq)
        self.endpoint_samples = tuple(
            int(round(float(seconds) * self.sfreq)) for seconds in endpoint_seconds
        )
        if tuple(sorted(self.endpoint_samples)) != self.endpoint_samples:
            raise ValueError("endpoint_seconds must be increasing")
        self.firing_rate_target = float(firing_rate_target)
        self.current_encoder = nn.Conv1d(
            self.n_bands * self.n_nodes,
            self.signed_channels,
            kernel_size=1,
            bias=False,
        )
        self.stem = _make_state_layer(
            self.decoder_kind, self.snn_channels, decays, threshold
        )
        self.blocks = nn.ModuleList(
            [
                CausalDepthwiseSEWBlock(
                    self.snn_channels,
                    kind=self.decoder_kind,
                    decays=decays,
                    threshold=threshold,
                    residual_mode=self.decoder_residual_mode,
                    kernel_size=3,
                    dilation=2**index,
                )
                for index in range(int(decoder_layers))
            ]
        )
        self.respike = _make_state_layer(
            self.decoder_kind, self.snn_channels, decays, threshold
        )
        statistic_features = 4 * self.snn_channels
        self.readout = nn.Sequential(
            nn.Linear(statistic_features, int(readout_features), bias=False),
            nn.LayerNorm(int(readout_features), elementwise_affine=False),
            nn.ELU(),
            nn.Dropout(float(dropout)),
        )
        self.classifier = nn.Linear(int(readout_features), self.n_classes)

    def encode_current(
        self,
        delayed_features: torch.Tensor,
        geometry_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if delayed_features.ndim != 4 or delayed_features.shape[1:3] != (
            self.n_bands,
            self.n_nodes,
        ):
            raise ValueError("current encoder expects delayed [N, B, K, T]")
        signed = self.current_encoder(delayed_features.flatten(1, 2))
        if geometry_scale is not None:
            if geometry_scale.shape != signed.shape:
                raise ValueError("geometry modulation does not match signed current")
            signed = signed * geometry_scale
        return torch.cat((F.relu(signed), F.relu(-signed)), dim=1)

    @staticmethod
    def _prefix_statistics(
        spikes: torch.Tensor, membrane: torch.Tensor, stop: int
    ) -> torch.Tensor:
        spikes = spikes[..., :stop]
        membrane = membrane[..., :stop]
        return torch.cat(
            (
                spikes.mean(dim=-1),
                spikes.std(dim=-1, unbiased=False),
                membrane.mean(dim=-1),
                membrane.std(dim=-1, unbiased=False),
            ),
            dim=1,
        )

    def forward(
        self,
        delayed_features: torch.Tensor,
        geometry_scale: torch.Tensor | None = None,
    ) -> DecoderOutput:
        current = self.encode_current(delayed_features, geometry_scale)
        activity, stem_membrane = self.stem(current)
        binary: list[torch.Tensor] = []
        residual_activities: list[torch.Tensor] = []
        if self.decoder_kind != "ann":
            binary.append(activity)
        for block in self.blocks:
            activity, branch, _ = block(activity)
            residual_activities.append(activity)
            if self.decoder_kind != "ann":
                binary.append(branch)
        final_activity, final_membrane = self.respike(activity)
        if self.decoder_kind != "ann":
            binary.append(final_activity)
        endpoint_features = []
        logits = []
        for stop in self.endpoint_samples:
            if stop > final_activity.shape[-1]:
                raise ValueError("decoder endpoint exceeds the available causal sequence")
            features = self._prefix_statistics(final_activity, final_membrane, stop)
            encoded = self.readout(features)
            endpoint_features.append(encoded)
            logits.append(self.classifier(encoded))
        if binary:
            firing_rate_loss = torch.stack(
                [
                    (spikes.mean() - self.firing_rate_target).square()
                    for spikes in binary
                ]
            ).mean()
        else:
            firing_rate_loss = current.new_zeros(())
        return DecoderOutput(
            logits=torch.stack(logits, dim=1),
            final_spikes=final_activity,
            final_membrane=final_membrane,
            binary_spikes=tuple(binary),
            residual_activities=tuple(residual_activities),
            endpoint_features=torch.stack(endpoint_features, dim=1),
            firing_rate_loss=firing_rate_loss,
        )
