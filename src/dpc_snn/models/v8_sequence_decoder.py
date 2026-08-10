"""Matched ANN/SNN decoders for the frozen official ATCNet sequence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import nn

from .v62_snn_decoder import DecoderKind, ResidualMode, V62SNNDecoder


@dataclass(frozen=True)
class V8SequenceDecoderOutput:
    logits: torch.Tensor
    firing_rate_loss: torch.Tensor
    final_activity: torch.Tensor
    final_membrane: torch.Tensor
    binary_spikes: tuple[torch.Tensor, ...]


class V8ATCSequenceDecoder(nn.Module):
    """Decode the exact post-convolution ATC sequence with matched dynamics.

    The official ATC convolutional stem emits 18 time steps with 32 features
    for a four-second BCI2a trial.  This adapter preserves that sequence and
    changes only the recurrent state equation used by the decoder.
    """

    architecture_version = "dpc_snn_v8_atc_sequence_decoder_r1"

    def __init__(
        self,
        *,
        input_features: int = 32,
        expected_steps: int = 18,
        n_classes: int = 4,
        hidden_channels: int = 64,
        decoder_kind: DecoderKind = "clif",
        residual_mode: ResidualMode = "plain",
        decoder_layers: int = 2,
        decays: Sequence[float] = (0.65, 0.90, 0.975),
        threshold: float = 1.0,
        task_seconds: float = 4.0,
        readout_features: int = 96,
        dropout: float = 0.25,
        firing_rate_target: float = 0.12,
    ) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.expected_steps = int(expected_steps)
        self.n_classes = int(n_classes)
        self.decoder_kind: DecoderKind = decoder_kind
        self.residual_mode: ResidualMode = residual_mode
        if self.input_features < 1 or self.expected_steps < 2:
            raise ValueError("ATC sequence dimensions must be positive")
        if task_seconds <= 0.0:
            raise ValueError("task_seconds must be positive")
        sequence_rate = self.expected_steps / float(task_seconds)
        self.decoder = V62SNNDecoder(
            n_bands=1,
            n_nodes=self.input_features,
            n_classes=self.n_classes,
            snn_channels=int(hidden_channels),
            decoder_kind=self.decoder_kind,
            decoder_residual_mode=self.residual_mode,
            decoder_layers=int(decoder_layers),
            decays=decays,
            threshold=float(threshold),
            sfreq=sequence_rate,
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

    def forward(self, sequence: torch.Tensor) -> V8SequenceDecoderOutput:
        if sequence.ndim != 3:
            raise ValueError("ATC sequence decoder expects [batch, time, features]")
        if sequence.shape[1:] != (self.expected_steps, self.input_features):
            raise ValueError(
                "ATC sequence shape mismatch: expected "
                f"[N, {self.expected_steps}, {self.input_features}], got {tuple(sequence.shape)}"
            )
        decoded = self.decoder(sequence.transpose(1, 2).unsqueeze(1))
        return V8SequenceDecoderOutput(
            logits=decoded.logits[:, -1],
            firing_rate_loss=decoded.firing_rate_loss,
            final_activity=decoded.final_spikes,
            final_membrane=decoded.final_membrane,
            binary_spikes=decoded.binary_spikes,
        )


V8_SEQUENCE_DECODER_VARIANTS: dict[
    str, tuple[Literal["ann", "plif", "clif"], Literal["plain", "sew_add"]]
] = {
    "ann_plain": ("ann", "plain"),
    "ann_sew": ("ann", "sew_add"),
    "plif_plain": ("plif", "plain"),
    "clif_plain": ("clif", "plain"),
    "sew_clif": ("clif", "sew_add"),
}


def build_v8_sequence_decoder(variant: str, **kwargs: object) -> V8ATCSequenceDecoder:
    try:
        decoder_kind, residual_mode = V8_SEQUENCE_DECODER_VARIANTS[str(variant)]
    except KeyError as exc:
        raise KeyError(f"unknown V8 sequence decoder variant: {variant}") from exc
    return V8ATCSequenceDecoder(
        decoder_kind=decoder_kind,
        residual_mode=residual_mode,
        **kwargs,
    )
