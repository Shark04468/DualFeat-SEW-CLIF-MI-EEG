"""Frozen shared-SNN backbone with native-rate residual SNN experts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .v62_snn_decoder import DecoderOutput, V62SNNDecoder
from .v9_dual_feature_student import V9DualFeatureStudent


V14_VARIANTS = (
    "r0_shared_replay",
    "r1_atc_residual",
    "r2_fbc_residual",
    "r3_dual_residual",
    "r4_generic_residual",
)


@dataclass(frozen=True)
class V14ResidualOutput:
    logits: torch.Tensor
    shared_logits: torch.Tensor
    atc_residual_logits: torch.Tensor
    fbc_residual_logits: torch.Tensor
    generic_residual_logits: torch.Tensor
    firing_rate_loss: torch.Tensor
    binary_spikes: tuple[torch.Tensor, ...]


def _center(logits: torch.Tensor) -> torch.Tensor:
    return logits - logits.mean(dim=-1, keepdim=True)


def _expert(
    *, n_nodes: int, sfreq: float, channels: int, readout_features: int
) -> V62SNNDecoder:
    return V62SNNDecoder(
        n_bands=1,
        n_nodes=n_nodes,
        n_classes=4,
        snn_channels=channels,
        decoder_kind="clif",
        decoder_residual_mode="sew_add",
        decoder_layers=2,
        sfreq=sfreq,
        endpoint_seconds=(1.0, 2.0, 3.0, 4.0),
        readout_features=readout_features,
        dropout=0.25,
    )


class V14SharedResidualStudent(nn.Module):
    """Add bounded class-wise SNN residuals to an exact frozen V9 replay."""

    architecture_version = "dpc_snn_v14_shared_residual_r1"

    def __init__(self, variant: str) -> None:
        super().__init__()
        if variant not in V14_VARIANTS:
            raise KeyError(f"unknown V14 residual variant: {variant}")
        self.variant = variant
        self.shared = V9DualFeatureStudent(
            decoder_kind="clif",
            residual_mode="sew_add",
            decoder_layers=2,
        )
        for parameter in self.shared.parameters():
            parameter.requires_grad_(False)
        self.atc_expert = (
            _expert(n_nodes=32, sfreq=18.0 / 4.0, channels=24, readout_features=32)
            if variant in ("r1_atc_residual", "r3_dual_residual")
            else None
        )
        self.fbc_expert = (
            _expert(n_nodes=288, sfreq=1.0, channels=24, readout_features=32)
            if variant in ("r2_fbc_residual", "r3_dual_residual")
            else None
        )
        self.generic_expert = (
            _expert(n_nodes=64, sfreq=18.0 / 4.0, channels=32, readout_features=73)
            if variant == "r4_generic_residual"
            else None
        )
        initial_gate_logit = -2.1972245773362196
        self.atc_gate_logit = (
            nn.Parameter(torch.full((4,), initial_gate_logit))
            if self.atc_expert is not None
            else None
        )
        self.fbc_gate_logit = (
            nn.Parameter(torch.full((4,), initial_gate_logit))
            if self.fbc_expert is not None
            else None
        )
        self.generic_gate_logit = (
            nn.Parameter(torch.full((4,), initial_gate_logit))
            if self.generic_expert is not None
            else None
        )
        self.shared.eval()

    def train(self, mode: bool = True) -> V14SharedResidualStudent:
        super().train(mode)
        self.shared.eval()
        return self

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )

    def load_shared_state(self, state: dict[str, torch.Tensor]) -> None:
        self.shared.load_state_dict(state, strict=True)
        self.shared.eval()

    @staticmethod
    def _residual(decoder: DecoderOutput | None, reference: torch.Tensor) -> torch.Tensor:
        if decoder is None:
            return reference.new_zeros(reference.shape)
        return _center(decoder.logits[:, -1])

    @staticmethod
    def _gate(logit: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
        if logit is None:
            return reference.new_zeros((1, reference.shape[-1]))
        return 0.5 * torch.sigmoid(logit)[None, :]

    def forward(
        self, atc_sequence: torch.Tensor, fbc_sequence: torch.Tensor
    ) -> V14ResidualOutput:
        with torch.no_grad():
            shared = self.shared(atc_sequence, fbc_sequence)
        atc = (
            self.atc_expert(atc_sequence.transpose(1, 2).unsqueeze(1))
            if self.atc_expert is not None
            else None
        )
        fbc = (
            self.fbc_expert(fbc_sequence.transpose(1, 2).unsqueeze(1))
            if self.fbc_expert is not None
            else None
        )
        generic = (
            self.generic_expert(shared.fused_sequence.transpose(1, 2).unsqueeze(1))
            if self.generic_expert is not None
            else None
        )
        atc_residual = self._residual(atc, shared.logits)
        fbc_residual = self._residual(fbc, shared.logits)
        generic_residual = self._residual(generic, shared.logits)
        logits = (
            shared.logits
            + self._gate(self.atc_gate_logit, shared.logits) * atc_residual
            + self._gate(self.fbc_gate_logit, shared.logits) * fbc_residual
            + self._gate(self.generic_gate_logit, shared.logits) * generic_residual
        )
        experts = tuple(value for value in (atc, fbc, generic) if value is not None)
        binary_spikes = tuple(
            spike for expert in experts for spike in expert.binary_spikes
        )
        firing_rate_loss = (
            torch.stack([expert.firing_rate_loss for expert in experts]).mean()
            if experts
            else shared.logits.new_zeros(())
        )
        return V14ResidualOutput(
            logits=logits,
            shared_logits=shared.logits,
            atc_residual_logits=atc_residual,
            fbc_residual_logits=fbc_residual,
            generic_residual_logits=generic_residual,
            firing_rate_loss=firing_rate_loss,
            binary_spikes=binary_spikes,
        )

    def gate_values(self) -> dict[str, torch.Tensor]:
        reference = next(self.shared.parameters())
        zero = reference.new_zeros(4)
        return {
            "atc": (
                self._gate(self.atc_gate_logit, reference.new_zeros(1, 4)).squeeze(0)
                if self.atc_gate_logit is not None
                else zero
            ),
            "fbc": (
                self._gate(self.fbc_gate_logit, reference.new_zeros(1, 4)).squeeze(0)
                if self.fbc_gate_logit is not None
                else zero
            ),
            "generic": (
                self._gate(self.generic_gate_logit, reference.new_zeros(1, 4)).squeeze(0)
                if self.generic_gate_logit is not None
                else zero
            ),
        }


def build_v14_student(variant: str) -> V14SharedResidualStudent:
    return V14SharedResidualStudent(variant)
