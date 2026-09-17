"""Matched ANN/SNN students for frozen ATCNet and FBCNet features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .v62_snn_decoder import DecoderKind, ResidualMode, V62SNNDecoder


FusionMode = Literal["interaction", "simple", "atc_only", "fbc_only"]


class V9FBCFeatureBackbone(nn.Module):
    """Expose the exact pre-classifier FBCNet log-variance sequence.

    The official FBCNet produces 288 band-specific spatial filters and four
    non-overlapping temporal variance windows.  This wrapper does not alter
    the official forward computation; it only retains the [N, 4, 288]
    representation before flattening.
    """

    architecture_version = "dpc_snn_v9_fbc_feature_backbone_r1"

    def __init__(self, official_core: nn.Module) -> None:
        super().__init__()
        required = ("scb", "temporalLayer", "lastLayer", "strideFactor", "nBands", "m")
        missing = [name for name in required if not hasattr(official_core, name)]
        if missing:
            raise TypeError(f"official FBCNet core is missing required attributes: {missing}")
        self.official_core = official_core

    def continuous_sequence(self, carrier: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if carrier.ndim != 5:
            raise ValueError("FBCNet carrier must have shape [N, 1, C, T, bands]")
        features = torch.squeeze(carrier.permute(0, 4, 2, 3, 1), dim=4)
        features = self.official_core.scb(features)
        stride = int(self.official_core.strideFactor)
        if features.shape[-1] % stride:
            raise ValueError("FBCNet time axis is not divisible by strideFactor")
        features = features.reshape(
            *features.shape[:2], stride, features.shape[-1] // stride
        )
        temporal = self.official_core.temporalLayer(features)
        if temporal.ndim != 4 or temporal.shape[-1] != 1:
            raise RuntimeError("official FBCNet temporal layer returned an unexpected shape")
        sequence = temporal.squeeze(-1).transpose(1, 2)
        logits = self.official_core.lastLayer(torch.flatten(temporal, start_dim=1))
        return sequence, logits

    def forward(self, carrier: torch.Tensor) -> dict[str, torch.Tensor]:
        sequence, logits = self.continuous_sequence(carrier)
        return {"logits": logits, "continuous_sequence": sequence}


@dataclass(frozen=True)
class V9DualFeatureOutput:
    logits: torch.Tensor
    firing_rate_loss: torch.Tensor
    fused_sequence: torch.Tensor
    final_activity: torch.Tensor
    final_membrane: torch.Tensor
    binary_spikes: tuple[torch.Tensor, ...]


class V9DualFeatureStudent(nn.Module):
    """Fuse complementary frozen EEG features before matched ANN/SNN dynamics."""

    architecture_version = "dpc_snn_v9_dual_feature_student_r1"

    def __init__(
        self,
        *,
        atc_features: int = 32,
        atc_steps: int = 18,
        fbc_features: int = 288,
        fbc_steps: int = 4,
        hidden_channels: int = 64,
        n_classes: int = 4,
        decoder_kind: DecoderKind = "clif",
        residual_mode: ResidualMode = "sew_add",
        decoder_layers: int = 2,
        decays: Sequence[float] = (0.65, 0.90, 0.975),
        threshold: float = 1.0,
        task_seconds: float = 4.0,
        readout_features: int = 96,
        dropout: float = 0.25,
        firing_rate_target: float = 0.12,
        fusion_mode: FusionMode = "interaction",
    ) -> None:
        super().__init__()
        self.atc_features = int(atc_features)
        self.atc_steps = int(atc_steps)
        self.fbc_features = int(fbc_features)
        self.fbc_steps = int(fbc_steps)
        self.hidden_channels = int(hidden_channels)
        self.n_classes = int(n_classes)
        self.decoder_kind: DecoderKind = decoder_kind
        self.residual_mode: ResidualMode = residual_mode
        if fusion_mode not in ("interaction", "simple", "atc_only", "fbc_only"):
            raise ValueError("unknown dual-feature fusion mode")
        self.fusion_mode: FusionMode = fusion_mode
        if min(
            self.atc_features,
            self.atc_steps,
            self.fbc_features,
            self.fbc_steps,
            self.hidden_channels,
        ) < 1:
            raise ValueError("dual-feature dimensions must be positive")
        if self.hidden_channels % 2:
            raise ValueError("hidden_channels must be even for signed populations")
        if task_seconds <= 0.0:
            raise ValueError("task_seconds must be positive")

        self.atc_projection = nn.Linear(self.atc_features, self.hidden_channels, bias=False)
        self.fbc_projection = nn.Linear(self.fbc_features, self.hidden_channels, bias=False)
        self.interaction = nn.Sequential(
            nn.LayerNorm(4 * self.hidden_channels),
            nn.Linear(4 * self.hidden_channels, self.hidden_channels, bias=False),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.fusion_norm = nn.LayerNorm(self.hidden_channels)
        self.decoder = V62SNNDecoder(
            n_bands=1,
            n_nodes=self.hidden_channels,
            n_classes=self.n_classes,
            snn_channels=self.hidden_channels,
            decoder_kind=self.decoder_kind,
            decoder_residual_mode=self.residual_mode,
            decoder_layers=int(decoder_layers),
            decays=decays,
            threshold=float(threshold),
            sfreq=self.atc_steps / float(task_seconds),
            endpoint_seconds=(float(task_seconds),),
            readout_features=int(readout_features),
            dropout=float(dropout),
            firing_rate_target=float(firing_rate_target),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @property
    def is_spiking(self) -> bool:
        return self.decoder_kind != "ann"

    def fuse(self, atc_sequence: torch.Tensor, fbc_sequence: torch.Tensor) -> torch.Tensor:
        if atc_sequence.ndim != 3 or atc_sequence.shape[1:] != (
            self.atc_steps,
            self.atc_features,
        ):
            raise ValueError(
                "ATC feature shape mismatch: expected "
                f"[N, {self.atc_steps}, {self.atc_features}]"
            )
        if fbc_sequence.ndim != 3 or fbc_sequence.shape[1:] != (
            self.fbc_steps,
            self.fbc_features,
        ):
            raise ValueError(
                "FBC feature shape mismatch: expected "
                f"[N, {self.fbc_steps}, {self.fbc_features}]"
            )
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
        if self.fusion_mode == "interaction":
            fusion_features = torch.cat(
                (atc, fbc, atc * fbc, torch.abs(atc - fbc)), dim=-1
            )
            base = 0.5 * (atc + fbc)
        elif self.fusion_mode == "simple":
            # Repeating the two branch features keeps the interaction MLP input
            # width and parameter count identical without adding multiplicative
            # or difference features.
            fusion_features = torch.cat((atc, fbc, atc, fbc), dim=-1)
            base = 0.5 * (atc + fbc)
        elif self.fusion_mode == "atc_only":
            zeros = torch.zeros_like(atc)
            fusion_features = torch.cat((atc, zeros, atc, zeros), dim=-1)
            base = atc
        else:
            zeros = torch.zeros_like(fbc)
            fusion_features = torch.cat((zeros, fbc, zeros, fbc), dim=-1)
            base = fbc
        fused = base + self.interaction(fusion_features)
        return self.fusion_norm(fused)

    def forward(
        self, atc_sequence: torch.Tensor, fbc_sequence: torch.Tensor
    ) -> V9DualFeatureOutput:
        fused = self.fuse(atc_sequence, fbc_sequence)
        decoded = self.decoder(fused.transpose(1, 2).unsqueeze(1))
        return V9DualFeatureOutput(
            logits=decoded.logits[:, -1],
            firing_rate_loss=decoded.firing_rate_loss,
            fused_sequence=fused,
            final_activity=decoded.final_spikes,
            final_membrane=decoded.final_membrane,
            binary_spikes=decoded.binary_spikes,
        )


V9_DUAL_FEATURE_MODEL_VARIANTS: dict[
    str, tuple[Literal["ann", "plif", "clif"], Literal["plain", "sew_add"]]
] = {
    "ann_plain": ("ann", "plain"),
    "plif_plain": ("plif", "plain"),
    "clif_plain": ("clif", "plain"),
    "ann_sew": ("ann", "sew_add"),
    "sew_clif": ("clif", "sew_add"),
}


def build_v9_dual_feature_student(variant: str, **kwargs: object) -> V9DualFeatureStudent:
    try:
        decoder_kind, residual_mode = V9_DUAL_FEATURE_MODEL_VARIANTS[str(variant)]
    except KeyError as exc:
        raise KeyError(f"unknown V9 dual-feature model variant: {variant}") from exc
    return V9DualFeatureStudent(
        decoder_kind=decoder_kind,
        residual_mode=residual_mode,
        **kwargs,
    )
