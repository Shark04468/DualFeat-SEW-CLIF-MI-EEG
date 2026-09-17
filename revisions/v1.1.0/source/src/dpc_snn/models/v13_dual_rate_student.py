"""Parameter-matched native-rate dual-branch SNN students."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from .v12_multirate_student import V12StudentOutput
from .v62_snn_decoder import DecoderKind, DecoderOutput, V62SNNDecoder


V13Variant = Literal["dual_snn_logit_mean", "dual_snn_feature_late"]


class _V13DualRateBase(nn.Module):
    def __init__(
        self,
        *,
        decoder_kind: DecoderKind,
        branch_channels: int,
        branch_readout_features: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if branch_channels % 2:
            raise ValueError("branch channels must be even")
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

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _branches(
        self, atc_sequence: torch.Tensor, fbc_sequence: torch.Tensor
    ) -> tuple[DecoderOutput, DecoderOutput]:
        if atc_sequence.ndim != 3 or atc_sequence.shape[1:] != (18, 32):
            raise ValueError("ATC input must have shape [N, 18, 32]")
        if fbc_sequence.ndim != 3 or fbc_sequence.shape[1:] != (4, 288):
            raise ValueError("FBC input must have shape [N, 4, 288]")
        if atc_sequence.shape[0] != fbc_sequence.shape[0]:
            raise ValueError("ATC and FBC batches are not aligned")
        atc = self.atc_decoder(atc_sequence.transpose(1, 2).unsqueeze(1))
        fbc = self.fbc_decoder(fbc_sequence.transpose(1, 2).unsqueeze(1))
        return atc, fbc

    @staticmethod
    def _output(
        atc: DecoderOutput,
        fbc: DecoderOutput,
        endpoint_logits: torch.Tensor,
    ) -> V12StudentOutput:
        return V12StudentOutput(
            logits=endpoint_logits[:, -1],
            endpoint_logits=endpoint_logits,
            atc_logits=atc.logits[:, -1],
            fbc_logits=fbc.logits[:, -1],
            firing_rate_loss=0.5 * (atc.firing_rate_loss + fbc.firing_rate_loss),
            binary_spikes=(*atc.binary_spikes, *fbc.binary_spikes),
        )


class V13DualSNNLogitMean(_V13DualRateBase):
    """Two native-rate SNN branches with exact equal-logit fusion."""

    architecture_version = "dpc_snn_v13_dual_snn_logit_mean_r1"

    def __init__(self, *, decoder_kind: DecoderKind = "clif") -> None:
        super().__init__(
            decoder_kind=decoder_kind,
            branch_channels=48,
            branch_readout_features=144,
            dropout=0.25,
        )

    def forward(
        self, atc_sequence: torch.Tensor, fbc_sequence: torch.Tensor
    ) -> V12StudentOutput:
        atc, fbc = self._branches(atc_sequence, fbc_sequence)
        return self._output(atc, fbc, 0.5 * (atc.logits + fbc.logits))


class V13DualSNNFeatureLate(_V13DualRateBase):
    """Fuse only native-rate SNN endpoint states after both branches spike."""

    architecture_version = "dpc_snn_v13_dual_snn_feature_late_r1"

    def __init__(self, *, decoder_kind: DecoderKind = "clif") -> None:
        super().__init__(
            decoder_kind=decoder_kind,
            branch_channels=72,
            branch_readout_features=48,
            dropout=0.25,
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(4 * 48),
            nn.Linear(4 * 48, 64, bias=False),
            nn.ELU(),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Linear(64, 4)

    def forward(
        self, atc_sequence: torch.Tensor, fbc_sequence: torch.Tensor
    ) -> V12StudentOutput:
        atc, fbc = self._branches(atc_sequence, fbc_sequence)
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
        return self._output(atc, fbc, endpoint_logits)


V13_MODEL_VARIANTS: dict[str, type[nn.Module]] = {
    "dual_snn_logit_mean": V13DualSNNLogitMean,
    "dual_snn_feature_late": V13DualSNNFeatureLate,
}


def build_v13_student(
    variant: str, *, decoder_kind: DecoderKind = "clif"
) -> nn.Module:
    try:
        model_type = V13_MODEL_VARIANTS[str(variant)]
    except KeyError as exc:
        raise KeyError(f"unknown V13 student variant: {variant}") from exc
    return model_type(decoder_kind=decoder_kind)
