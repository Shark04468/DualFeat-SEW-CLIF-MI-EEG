"""Sparse low-rank residual delay on the V12 temporal carrier."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import nn

from .v12_multirate_student import V12StudentOutput, build_v12_student


DelayControl = Literal["full", "zero", "shuffled"]


@dataclass(frozen=True)
class V13DelayOutput:
    logits: torch.Tensor
    endpoint_logits: torch.Tensor
    atc_logits: torch.Tensor
    fbc_logits: torch.Tensor
    firing_rate_loss: torch.Tensor
    binary_spikes: tuple[torch.Tensor, ...]
    route_probability: torch.Tensor
    lag_posterior: torch.Tensor
    nonzero_delay_mass: torch.Tensor
    delay_residual_rms: torch.Tensor


class SparseLowRankDelayResidual(nn.Module):
    """Route a sparse carrier residual through causal integer-delay experts."""

    def __init__(
        self,
        *,
        channels: int = 32,
        max_lag: int = 4,
        delay_experts: int = 3,
        mixing_rank: int = 8,
        permutation_seed: int = 13_013,
    ) -> None:
        super().__init__()
        if min(channels, max_lag, delay_experts, mixing_rank) < 1:
            raise ValueError("delay residual dimensions must be positive")
        self.channels = int(channels)
        self.max_lag = int(max_lag)
        self.delay_experts = int(delay_experts)
        self.mixing_rank = int(mixing_rank)
        self.expert_lag_logits = nn.Parameter(
            torch.zeros(self.delay_experts, self.max_lag + 1)
        )
        with torch.no_grad():
            self.expert_lag_logits[:, 0] = 1.0
            for expert in range(self.delay_experts):
                preferred = 1 + expert % self.max_lag
                self.expert_lag_logits[expert, preferred] = 0.75
        self.router_logits = nn.Parameter(torch.zeros(self.channels, self.delay_experts))
        self.route_logits = nn.Parameter(torch.full((self.channels,), -2.0))
        self.mix_left = nn.Parameter(torch.empty(self.channels, self.mixing_rank))
        self.mix_right = nn.Parameter(torch.empty(self.channels, self.mixing_rank))
        nn.init.orthogonal_(self.mix_left)
        nn.init.orthogonal_(self.mix_right)
        self.residual_scale_logit = nn.Parameter(torch.tensor(-2.0))
        permutation = torch.randperm(
            self.channels, generator=torch.Generator().manual_seed(permutation_seed)
        )
        self.register_buffer("shuffle_permutation", permutation, persistent=True)

    def posterior(self, control: DelayControl) -> torch.Tensor:
        expert = torch.softmax(self.expert_lag_logits, dim=-1)
        router = torch.softmax(self.router_logits, dim=-1)
        posterior = router @ expert
        if control == "full":
            return posterior
        if control == "zero":
            value = torch.zeros_like(posterior)
            value[:, 0] = 1.0
            return value
        if control == "shuffled":
            return posterior[self.shuffle_permutation]
        raise ValueError(f"unknown delay control: {control}")

    def forward(
        self, sequence: torch.Tensor, *, control: DelayControl = "full"
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if sequence.ndim != 3 or sequence.shape[-1] != self.channels:
            raise ValueError("delay residual expects [N, T, channels]")
        candidates = [sequence]
        for lag in range(1, self.max_lag + 1):
            candidates.append(
                torch.cat((torch.zeros_like(sequence[:, :lag]), sequence[:, :-lag]), dim=1)
            )
        candidate = torch.stack(candidates, dim=-1)
        posterior = self.posterior(control).to(sequence)
        delayed = torch.einsum("ntcd,cd->ntc", candidate, posterior)
        mixing = (self.mix_left @ self.mix_right.transpose(0, 1)) / math.sqrt(
            self.mixing_rank
        )
        residual = torch.einsum("ntc,oc->nto", delayed, mixing.to(sequence))
        route = torch.sigmoid(self.route_logits).to(sequence)
        scale = torch.sigmoid(self.residual_scale_logit).to(sequence)
        residual = scale * route[None, None, :] * residual
        return sequence + residual, {
            "route_probability": route,
            "lag_posterior": posterior,
            "nonzero_delay_mass": posterior[:, 1:].sum(dim=-1).mean(),
            "delay_residual_rms": residual.square().mean().sqrt(),
        }


class V13DelayResidualStudent(nn.Module):
    """Frozen or trainable V12 student augmented by an exact-control delay branch."""

    architecture_version = "dpc_snn_v13_sparse_delay_residual_r1"

    def __init__(
        self,
        *,
        base_variant: str = "dual_rate_branch_temporal",
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.base_variant = str(base_variant)
        self.base = build_v12_student(self.base_variant)
        self.delay = SparseLowRankDelayResidual()
        if freeze_base:
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def delay_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.delay.parameters())

    def load_base_state(self, state: dict[str, torch.Tensor]) -> None:
        self.base.load_state_dict(state, strict=True)

    def forward(
        self,
        atc_sequence: torch.Tensor,
        fbc_sequence: torch.Tensor,
        *,
        control: DelayControl = "full",
    ) -> V13DelayOutput:
        routed, diagnostics = self.delay(atc_sequence, control=control)
        output: V12StudentOutput = self.base(routed, fbc_sequence)
        return V13DelayOutput(
            logits=output.logits,
            endpoint_logits=output.endpoint_logits,
            atc_logits=output.atc_logits,
            fbc_logits=output.fbc_logits,
            firing_rate_loss=output.firing_rate_loss,
            binary_spikes=output.binary_spikes,
            route_probability=diagnostics["route_probability"],
            lag_posterior=diagnostics["lag_posterior"],
            nonzero_delay_mass=diagnostics["nonzero_delay_mass"],
            delay_residual_rms=diagnostics["delay_residual_rms"],
        )
