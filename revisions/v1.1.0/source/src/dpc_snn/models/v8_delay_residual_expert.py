"""Matched ANN/SNN experts driven only by audited routed delay currents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch
from torch import nn

from .v62_snn_decoder import DecoderKind, ResidualMode, V62SNNDecoder
from .v8_delay_auxiliary import (
    SparsePhysicalDelayAuxiliary,
    V8DelayAuxiliaryOutput,
    V8DelayOverride,
    V8DelaySignalMode,
)


@dataclass(frozen=True)
class V8DelayResidualExpertOutput:
    logits: torch.Tensor
    prefix_logits: torch.Tensor
    firing_rate_loss: torch.Tensor
    physical_current: torch.Tensor
    delay: V8DelayAuxiliaryOutput
    binary_spikes: tuple[torch.Tensor, ...]
    final_activity: torch.Tensor
    final_membrane: torch.Tensor


class V8DelayResidualExpert(nn.Module):
    """Classify the same sparse routes under audited or point-zero lag transport."""

    architecture_version = "dpc_snn_v8_delay_residual_expert_r2_matched_transport"

    def __init__(
        self,
        *,
        n_bands: int = 12,
        n_nodes: int = 22,
        n_classes: int = 4,
        maximum_routes: int = 256,
        maximum_delay: int = 8,
        allow_cross_band: bool = False,
        signal_mode: V8DelaySignalMode = "slow_envelope",
        decoder_kind: DecoderKind = "clif",
        residual_mode: ResidualMode = "sew_add",
        decoder_channels: int = 64,
        decoder_layers: int = 2,
        decays: Sequence[float] = (0.65, 0.90, 0.975),
        threshold: float = 1.0,
        sfreq: float = 125.0,
        temporal_decimation: int = 2,
        endpoint_seconds: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
        readout_features: int = 96,
        dropout: float = 0.25,
        firing_rate_target: float = 0.08,
    ) -> None:
        super().__init__()
        self.n_bands = int(n_bands)
        self.n_nodes = int(n_nodes)
        self.n_classes = int(n_classes)
        self.decoder_kind: DecoderKind = decoder_kind
        self.residual_mode: ResidualMode = residual_mode
        self.temporal_decimation = int(temporal_decimation)
        if self.temporal_decimation < 1:
            raise ValueError("delay expert temporal decimation must be positive")
        if signal_mode != "slow_envelope" and self.temporal_decimation != 1:
            raise ValueError("fast-phase delay currents must retain the full sample rate")
        self.delay = SparsePhysicalDelayAuxiliary(
            self.n_bands,
            self.n_nodes,
            maximum_routes=int(maximum_routes),
            maximum_delay=int(maximum_delay),
            allow_cross_band=bool(allow_cross_band),
            signal_mode=signal_mode,
            contextual_residual_enabled=False,
            phase_residual_enabled=False,
        )
        self.decoder = V62SNNDecoder(
            n_bands=self.n_bands,
            n_nodes=self.n_nodes,
            n_classes=self.n_classes,
            snn_channels=int(decoder_channels),
            decoder_kind=self.decoder_kind,
            decoder_residual_mode=self.residual_mode,
            decoder_layers=int(decoder_layers),
            decays=decays,
            threshold=float(threshold),
            sfreq=float(sfreq) / self.temporal_decimation,
            endpoint_seconds=endpoint_seconds,
            readout_features=int(readout_features),
            dropout=float(dropout),
            firing_rate_target=float(firing_rate_target),
        )
        self.register_buffer("input_gain", torch.ones(self.n_bands, self.n_nodes))
        self.register_buffer("input_gain_ready", torch.tensor(False))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @property
    def is_spiking(self) -> bool:
        return self.decoder_kind != "ann"

    def load_fold_prior(self, **prior: torch.Tensor) -> None:
        self.delay.load_fold_prior(**prior)

    def set_input_gain(self, gain: torch.Tensor) -> None:
        value = torch.as_tensor(gain, dtype=self.input_gain.dtype)
        if value.shape != self.input_gain.shape:
            raise ValueError("delay expert input gain has an incompatible shape")
        if not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
            raise ValueError("delay expert input gain must be finite and positive")
        self.input_gain.copy_(value)
        self.input_gain_ready.fill_(True)

    def delay_current(
        self,
        physical_fast: torch.Tensor,
        physical_slow: torch.Tensor,
        *,
        delay_override: V8DelayOverride,
    ) -> tuple[torch.Tensor, V8DelayAuxiliaryOutput]:
        delayed = self.delay(
            physical_fast,
            physical_slow,
            override=delay_override,
        )
        current = delayed.physical_current * self.input_gain.to(delayed.physical_current)[
            None, :, :, None
        ]
        current = current[..., :: self.temporal_decimation]
        return current, delayed

    def forward(
        self,
        physical_fast: torch.Tensor,
        physical_slow: torch.Tensor,
        *,
        delay_override: V8DelayOverride = "full",
    ) -> V8DelayResidualExpertOutput:
        if not bool(self.input_gain_ready):
            raise RuntimeError("delay expert input gain has not been fitted")
        current, delayed = self.delay_current(
            physical_fast,
            physical_slow,
            delay_override=delay_override,
        )
        decoded = self.decoder(current)
        return V8DelayResidualExpertOutput(
            logits=decoded.logits[:, -1],
            prefix_logits=decoded.logits,
            firing_rate_loss=decoded.firing_rate_loss,
            physical_current=current,
            delay=delayed,
            binary_spikes=decoded.binary_spikes,
            final_activity=decoded.final_spikes,
            final_membrane=decoded.final_membrane,
        )


V8_DELAY_EXPERT_VARIANTS: dict[
    str, tuple[Literal["ann", "clif"], Literal["sew_add"]]
] = {
    "ann_sew": ("ann", "sew_add"),
    "sew_clif": ("clif", "sew_add"),
}


def build_v8_delay_residual_expert(
    variant: str, **kwargs: Any
) -> V8DelayResidualExpert:
    try:
        decoder_kind, residual_mode = V8_DELAY_EXPERT_VARIANTS[str(variant)]
    except KeyError as exc:
        raise KeyError(f"unknown V8 delay expert variant: {variant}") from exc
    return V8DelayResidualExpert(
        decoder_kind=decoder_kind,
        residual_mode=residual_mode,
        **kwargs,
    )
