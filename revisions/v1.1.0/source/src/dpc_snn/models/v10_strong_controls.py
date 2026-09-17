"""Strong recurrent ANN controls for the V9 dual-feature SNN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .v9_dual_feature_student import V9DualFeatureStudent


ControlKind = Literal["tcn", "gru", "lstm"]


class V10DualFeatureFusion(nn.Module):
    """The exact V9 fusion front-end, separated from its state decoder."""

    def __init__(
        self,
        *,
        atc_features: int = 32,
        atc_steps: int = 18,
        fbc_features: int = 288,
        fbc_steps: int = 4,
        hidden_channels: int = 64,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.atc_features = int(atc_features)
        self.atc_steps = int(atc_steps)
        self.fbc_features = int(fbc_features)
        self.fbc_steps = int(fbc_steps)
        self.hidden_channels = int(hidden_channels)
        self.atc_projection = nn.Linear(self.atc_features, self.hidden_channels, bias=False)
        self.fbc_projection = nn.Linear(self.fbc_features, self.hidden_channels, bias=False)
        self.interaction = nn.Sequential(
            nn.LayerNorm(4 * self.hidden_channels),
            nn.Linear(4 * self.hidden_channels, self.hidden_channels, bias=False),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.fusion_norm = nn.LayerNorm(self.hidden_channels)

    def forward(self, atc_sequence: torch.Tensor, fbc_sequence: torch.Tensor) -> torch.Tensor:
        if atc_sequence.ndim != 3 or atc_sequence.shape[1:] != (
            self.atc_steps,
            self.atc_features,
        ):
            raise ValueError("ATC feature shape does not match the V10 fusion contract")
        if fbc_sequence.ndim != 3 or fbc_sequence.shape[1:] != (
            self.fbc_steps,
            self.fbc_features,
        ):
            raise ValueError("FBC feature shape does not match the V10 fusion contract")
        if atc_sequence.shape[0] != fbc_sequence.shape[0]:
            raise ValueError("ATC and FBC feature batches are not aligned")
        atc = self.atc_projection(atc_sequence)
        fbc = self.fbc_projection(fbc_sequence)
        fbc = F.interpolate(
            fbc.transpose(1, 2),
            size=self.atc_steps,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        interactions = torch.cat((atc, fbc, atc * fbc, torch.abs(atc - fbc)), dim=-1)
        return self.fusion_norm(0.5 * (atc + fbc) + self.interaction(interactions))


class _CausalConv1d(nn.Conv1d):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__(
            channels,
            channels,
            kernel_size=int(kernel_size),
            dilation=int(dilation),
            padding=0,
            bias=False,
        )
        self.left_padding = (int(kernel_size) - 1) * int(dilation)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(sequence, (self.left_padding, 0)))


class _CausalTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(channels, 3, dilation)
        self.conv2 = _CausalConv1d(channels, 3, dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(float(dropout))

    @staticmethod
    def _normalize(sequence: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        return norm(sequence.transpose(1, 2)).transpose(1, 2)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        residual = sequence
        sequence = self.conv1(sequence)
        sequence = self.dropout(F.gelu(self._normalize(sequence, self.norm1)))
        sequence = self.conv2(sequence)
        sequence = self.dropout(F.gelu(self._normalize(sequence, self.norm2)))
        return F.gelu(sequence + residual)


class _TCNDecoder(nn.Module):
    def __init__(
        self,
        input_features: int,
        n_classes: int,
        *,
        width: int = 40,
        layers: int = 2,
        readout_features: int = 96,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(input_features, width, 1, bias=False)
        self.blocks = nn.ModuleList(
            [_CausalTCNBlock(width, 2**index, dropout) for index in range(int(layers))]
        )
        self.readout = nn.Sequential(
            nn.Linear(3 * width, readout_features, bias=False),
            nn.LayerNorm(readout_features, elementwise_affine=False),
            nn.ELU(),
            nn.Dropout(float(dropout)),
        )
        self.classifier = nn.Linear(readout_features, n_classes)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.input_projection(sequence.transpose(1, 2))
        for block in self.blocks:
            state = block(state)
        statistics = torch.cat(
            (
                state.mean(dim=-1),
                state.std(dim=-1, unbiased=False),
                state[..., -1],
            ),
            dim=1,
        )
        return self.classifier(self.readout(statistics)), state


class _RecurrentDecoder(nn.Module):
    def __init__(
        self,
        kind: Literal["gru", "lstm"],
        input_features: int,
        n_classes: int,
        *,
        hidden_features: int,
        readout_features: int = 96,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        recurrent = nn.GRU if kind == "gru" else nn.LSTM
        self.recurrent = recurrent(
            input_size=input_features,
            hidden_size=int(hidden_features),
            num_layers=1,
            batch_first=True,
        )
        self.readout = nn.Sequential(
            nn.Linear(3 * int(hidden_features), readout_features, bias=False),
            nn.LayerNorm(readout_features, elementwise_affine=False),
            nn.ELU(),
            nn.Dropout(float(dropout)),
        )
        self.classifier = nn.Linear(readout_features, n_classes)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state, _ = self.recurrent(sequence)
        statistics = torch.cat(
            (
                state.mean(dim=1),
                state.std(dim=1, unbiased=False),
                state[:, -1],
            ),
            dim=1,
        )
        return self.classifier(self.readout(statistics)), state.transpose(1, 2)


@dataclass(frozen=True)
class V10ControlOutput:
    logits: torch.Tensor
    firing_rate_loss: torch.Tensor
    fused_sequence: torch.Tensor
    final_activity: torch.Tensor
    final_membrane: torch.Tensor
    binary_spikes: tuple[torch.Tensor, ...]


class V10StrongANNControl(nn.Module):
    """A parameter-matched strong ANN sequence decoder on the V9 features."""

    architecture_version = "dpc_snn_v10_strong_ann_control_r1"

    def __init__(
        self,
        kind: ControlKind,
        *,
        n_classes: int = 4,
        fusion_features: int = 64,
        readout_features: int = 96,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.kind: ControlKind = kind
        self.fusion = V10DualFeatureFusion(
            hidden_channels=int(fusion_features), dropout=float(dropout)
        )
        if kind == "tcn":
            self.decoder: nn.Module = _TCNDecoder(
                fusion_features,
                n_classes,
                width=40,
                layers=2,
                readout_features=readout_features,
                dropout=dropout,
            )
        elif kind == "gru":
            self.decoder = _RecurrentDecoder(
                "gru",
                fusion_features,
                n_classes,
                hidden_features=56,
                readout_features=readout_features,
                dropout=dropout,
            )
        elif kind == "lstm":
            self.decoder = _RecurrentDecoder(
                "lstm",
                fusion_features,
                n_classes,
                hidden_features=48,
                readout_features=readout_features,
                dropout=dropout,
            )
        else:
            raise ValueError("strong ANN control kind must be tcn, gru, or lstm")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(
        self, atc_sequence: torch.Tensor, fbc_sequence: torch.Tensor
    ) -> V10ControlOutput:
        fused = self.fusion(atc_sequence, fbc_sequence)
        logits, activity = self.decoder(fused)
        zero = logits.new_zeros(())
        return V10ControlOutput(
            logits=logits,
            firing_rate_loss=zero,
            fused_sequence=fused,
            final_activity=activity,
            final_membrane=activity,
            binary_spikes=(),
        )


V10_STRONG_CONTROL_MODELS = (
    "ann_leaky_sew",
    "ann_tcn",
    "ann_gru",
    "ann_lstm",
    "sew_clif",
)


def build_v10_strong_control(name: str, **kwargs: object) -> nn.Module:
    normalized = str(name)
    if normalized == "ann_leaky_sew":
        return V9DualFeatureStudent(
            decoder_kind="ann", residual_mode="sew_add", **kwargs
        )
    if normalized == "sew_clif":
        return V9DualFeatureStudent(
            decoder_kind="clif", residual_mode="sew_add", **kwargs
        )
    if normalized.startswith("ann_"):
        return V10StrongANNControl(normalized.removeprefix("ann_"), **kwargs)
    raise KeyError(f"unknown V10 strong control: {name}")
