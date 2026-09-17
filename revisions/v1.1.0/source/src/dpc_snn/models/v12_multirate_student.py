"""V12 native-rate dual-view spiking students."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .v62_snn_decoder import DecoderKind, V62SNNDecoder
from .v9_dual_feature_student import V9DualFeatureStudent


V12Architecture = Literal["interpolated", "static_gate", "dual_rate"]


@dataclass(frozen=True)
class V12StudentOutput:
    logits: torch.Tensor
    endpoint_logits: torch.Tensor
    atc_logits: torch.Tensor
    fbc_logits: torch.Tensor
    firing_rate_loss: torch.Tensor
    binary_spikes: tuple[torch.Tensor, ...]


class V12InterpolatedBranchStudent(nn.Module):
    """V9 interpolation retained while exposing branch-specific KD heads."""

    architecture_version = "dpc_snn_v12_interpolated_branch_r1"

    def __init__(self, *, decoder_kind: DecoderKind = "clif") -> None:
        super().__init__()
        self.backbone = V9DualFeatureStudent(
            decoder_kind=decoder_kind,
            residual_mode="sew_add",
            decoder_layers=2,
        )
        hidden = self.backbone.hidden_channels
        self.atc_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 4))
        self.fbc_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 4))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self, atc_sequence: torch.Tensor, fbc_sequence: torch.Tensor
    ) -> V12StudentOutput:
        output = self.backbone(atc_sequence, fbc_sequence)
        atc = self.backbone.atc_projection(atc_sequence).mean(dim=1)
        fbc = self.backbone.fbc_projection(fbc_sequence).mean(dim=1)
        return V12StudentOutput(
            logits=output.logits,
            endpoint_logits=output.logits[:, None],
            atc_logits=self.atc_head(atc),
            fbc_logits=self.fbc_head(fbc),
            firing_rate_loss=output.firing_rate_loss,
            binary_spikes=output.binary_spikes,
        )


class V12StaticGateBackbone(V9DualFeatureStudent):
    """V9 interpolation with a trial-independent channel-wise branch gate."""

    architecture_version = "dpc_snn_v12_static_gate_backbone_r1"

    def __init__(self, *, decoder_kind: DecoderKind = "clif") -> None:
        super().__init__(
            decoder_kind=decoder_kind,
            residual_mode="sew_add",
            decoder_layers=2,
        )
        self.branch_gate_logit = nn.Parameter(torch.zeros(self.hidden_channels))

    def fuse(self, atc_sequence: torch.Tensor, fbc_sequence: torch.Tensor) -> torch.Tensor:
        if atc_sequence.ndim != 3 or atc_sequence.shape[1:] != (
            self.atc_steps,
            self.atc_features,
        ):
            raise ValueError("ATC feature shape mismatch")
        if fbc_sequence.ndim != 3 or fbc_sequence.shape[1:] != (
            self.fbc_steps,
            self.fbc_features,
        ):
            raise ValueError("FBC feature shape mismatch")
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
        gate = torch.sigmoid(self.branch_gate_logit)[None, None, :]
        fused = gate * atc + (1.0 - gate) * fbc + self.interaction(interactions)
        return self.fusion_norm(fused)


class V12StaticGateStudent(nn.Module):
    """Low-capacity static ATC/FBC fusion selected by E11B."""

    architecture_version = "dpc_snn_v12_static_gate_student_r1"

    def __init__(self, *, decoder_kind: DecoderKind = "clif") -> None:
        super().__init__()
        self.backbone = V12StaticGateBackbone(decoder_kind=decoder_kind)
        hidden = self.backbone.hidden_channels
        self.atc_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 4))
        self.fbc_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 4))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self, atc_sequence: torch.Tensor, fbc_sequence: torch.Tensor
    ) -> V12StudentOutput:
        output = self.backbone(atc_sequence, fbc_sequence)
        atc = self.backbone.atc_projection(atc_sequence).mean(dim=1)
        fbc = self.backbone.fbc_projection(fbc_sequence).mean(dim=1)
        return V12StudentOutput(
            logits=output.logits,
            endpoint_logits=output.logits[:, None],
            atc_logits=self.atc_head(atc),
            fbc_logits=self.fbc_head(fbc),
            firing_rate_loss=output.firing_rate_loss,
            binary_spikes=output.binary_spikes,
        )


class V12DualRateStudent(nn.Module):
    """Process ATC carrier and FBC variance at their native temporal rates."""

    architecture_version = "dpc_snn_v12_dual_rate_r1"

    def __init__(
        self,
        *,
        decoder_kind: DecoderKind = "clif",
        branch_channels: int = 32,
        branch_readout_features: int = 48,
        fusion_features: int = 64,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if branch_channels % 2:
            raise ValueError("branch_channels must be even")
        common = {
            "n_classes": 4,
            "snn_channels": int(branch_channels),
            "decoder_kind": decoder_kind,
            "decoder_residual_mode": "sew_add",
            "decoder_layers": 2,
            "endpoint_seconds": (1.0, 2.0, 3.0, 4.0),
            "readout_features": int(branch_readout_features),
            "dropout": float(dropout),
        }
        self.atc_decoder = V62SNNDecoder(
            n_bands=1,
            n_nodes=32,
            sfreq=18.0 / 4.0,
            **common,
        )
        self.fbc_decoder = V62SNNDecoder(
            n_bands=1,
            n_nodes=288,
            sfreq=1.0,
            **common,
        )
        interaction_features = 4 * int(branch_readout_features)
        self.fusion = nn.Sequential(
            nn.LayerNorm(interaction_features),
            nn.Linear(interaction_features, int(fusion_features), bias=False),
            nn.ELU(),
            nn.Dropout(float(dropout)),
        )
        self.classifier = nn.Linear(int(fusion_features), 4)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self, atc_sequence: torch.Tensor, fbc_sequence: torch.Tensor
    ) -> V12StudentOutput:
        if atc_sequence.ndim != 3 or atc_sequence.shape[1:] != (18, 32):
            raise ValueError("ATC input must have shape [N, 18, 32]")
        if fbc_sequence.ndim != 3 or fbc_sequence.shape[1:] != (4, 288):
            raise ValueError("FBC input must have shape [N, 4, 288]")
        if atc_sequence.shape[0] != fbc_sequence.shape[0]:
            raise ValueError("ATC and FBC batches are not aligned")
        atc = self.atc_decoder(atc_sequence.transpose(1, 2).unsqueeze(1))
        fbc = self.fbc_decoder(fbc_sequence.transpose(1, 2).unsqueeze(1))
        interactions = torch.cat(
            (
                atc.endpoint_features,
                fbc.endpoint_features,
                atc.endpoint_features * fbc.endpoint_features,
                torch.abs(atc.endpoint_features - fbc.endpoint_features),
            ),
            dim=-1,
        )
        endpoint_logits = self.classifier(self.fusion(interactions))
        return V12StudentOutput(
            logits=endpoint_logits[:, -1],
            endpoint_logits=endpoint_logits,
            atc_logits=atc.logits[:, -1],
            fbc_logits=fbc.logits[:, -1],
            firing_rate_loss=0.5 * (atc.firing_rate_loss + fbc.firing_rate_loss),
            binary_spikes=(*atc.binary_spikes, *fbc.binary_spikes),
        )


V12_MODEL_ARCHITECTURES: dict[str, V12Architecture] = {
    "interpolated_branch_kd": "interpolated",
    "interpolated_static_gate_kd": "static_gate",
    "dual_rate_branch_kd": "dual_rate",
    "dual_rate_branch_temporal": "dual_rate",
}


def build_v12_student(variant: str, *, decoder_kind: DecoderKind = "clif") -> nn.Module:
    try:
        architecture = V12_MODEL_ARCHITECTURES[str(variant)]
    except KeyError as exc:
        raise KeyError(f"unknown V12 student variant: {variant}") from exc
    if architecture == "interpolated":
        return V12InterpolatedBranchStudent(decoder_kind=decoder_kind)
    if architecture == "static_gate":
        return V12StaticGateStudent(decoder_kind=decoder_kind)
    return V12DualRateStudent(decoder_kind=decoder_kind)
