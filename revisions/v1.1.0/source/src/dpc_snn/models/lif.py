"""Leaky integrate-and-fire neuron modules."""

from __future__ import annotations

import torch
from torch import nn

from .surrogate import spike_fn


def _inverse_sigmoid(value: float) -> float:
    value = min(max(float(value), 1e-5), 1.0 - 1e-5)
    return float(torch.logit(torch.tensor(value)))


class CausalRMSNorm(nn.Module):
    """Per-time RMS normalization without channel centering.

    The operation only uses the current time step, so future samples cannot
    affect the recurrent state.  Keeping the channel mean is important for the
    signed dual-population code: subtracting it creates artificial negative
    current in an otherwise non-negative population.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.channels))

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        if current.shape[-1] != self.channels:
            raise ValueError("CausalRMSNorm received an incompatible channel axis")
        eps_squared = self.eps * self.eps
        scale = current.square().mean(dim=-1, keepdim=True).clamp_min(eps_squared).sqrt()
        return current / scale * self.weight.to(current)


# Compatibility alias for old checkpoints/imports.  Its semantics are now the
# scientifically valid non-centering RMS normalization above.
CausalChannelNorm = CausalRMSNorm


class LIFLayer(nn.Module):
    def __init__(self, tau_mem: float = 0.9, threshold: float = 1.0, reset: float = 0.0):
        super().__init__()
        self.tau_mem = float(tau_mem)
        self.threshold = float(threshold)
        self.reset = float(reset)

    def forward(self, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run LIF dynamics over the last dimension.

        Args:
            current: input current, shape [..., time].
        """

        mem = torch.zeros_like(current[..., 0])
        spikes = []
        mems = []
        for t in range(current.shape[-1]):
            mem = self.tau_mem * mem + current[..., t]
            spk = spike_fn(mem - self.threshold)
            mem = mem * (1.0 - spk) + self.reset * spk
            spikes.append(spk)
            mems.append(mem)
        return torch.stack(spikes, dim=-1), torch.stack(mems, dim=-1)


class CLIFLayer(nn.Module):
    """Complementary LIF dynamics for tensors shaped [N, C, T]."""

    def __init__(
        self,
        decay: float = 0.9,
        threshold: float = 1.0,
        channels: int | None = None,
        learnable_decay: bool = True,
        membrane_norm_groups: int = 0,
        causal_channel_norm: bool = False,
    ):
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError("CLIF decay must lie in [0, 1)")
        self.channels = 1 if channels is None else int(channels)
        decay_value = _inverse_sigmoid((float(decay) - 0.02) / 0.96)
        raw = torch.full((self.channels,), decay_value)
        if learnable_decay:
            self.decay_raw = nn.Parameter(raw)
        else:
            self.register_buffer("decay_raw", raw)
        self.threshold = float(threshold)
        # The legacy argument is accepted for checkpoint/config compatibility,
        # but full-trajectory GroupNorm is intentionally not constructed.
        self.causal_channel_norm = bool(causal_channel_norm or membrane_norm_groups > 0)
        self.current_norm = (
            CausalRMSNorm(self.channels)
            if self.causal_channel_norm and channels is not None
            else None
        )

    @property
    def decay(self) -> torch.Tensor:
        # Keep gradients away from exact 0/1 where recurrent credit assignment
        # becomes either memoryless or numerically brittle.
        return 0.02 + 0.96 * torch.sigmoid(self.decay_raw)

    def _decay_for(self, membrane: torch.Tensor) -> torch.Tensor:
        decay = self.decay.to(device=membrane.device, dtype=membrane.dtype)
        if decay.numel() == 1:
            return decay
        if membrane.shape[-1] != decay.numel():
            raise ValueError(
                f"CLIF expected {decay.numel()} channels, received {membrane.shape[-1]}"
            )
        return decay.view(*([1] * (membrane.ndim - 1)), -1)

    def forward(self, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if current.ndim != 3 and self.current_norm is not None:
            raise ValueError("Causal channel normalization expects CLIF current [N, C, T]")
        membrane = torch.zeros_like(current[..., 0])
        complement = torch.zeros_like(membrane)
        spikes = []
        membranes = []
        for step in range(current.shape[-1]):
            decay = self._decay_for(membrane)
            step_current = current[..., step]
            if self.current_norm is not None:
                step_current = self.current_norm(step_current)
            membrane = decay * membrane + step_current
            spike = spike_fn(membrane - self.threshold)
            complement = complement * torch.sigmoid((1.0 - decay) * membrane) + spike
            membrane = membrane - spike * (self.threshold + torch.sigmoid(complement))
            spikes.append(spike)
            membranes.append(membrane)
        return torch.stack(spikes, dim=-1), torch.stack(membranes, dim=-1)


class PLIFLayer(nn.Module):
    """Parametric LIF with a learnable per-channel inverse time constant."""

    def __init__(
        self,
        decay: float = 0.9,
        threshold: float = 1.0,
        channels: int | None = None,
        learnable_decay: bool = True,
        causal_channel_norm: bool = False,
    ):
        super().__init__()
        self.channels = 1 if channels is None else int(channels)
        raw = torch.full((self.channels,), _inverse_sigmoid(1.0 - float(decay)))
        if learnable_decay:
            self.inverse_tau_raw = nn.Parameter(raw)
        else:
            self.register_buffer("inverse_tau_raw", raw)
        self.threshold = float(threshold)
        self.current_norm = (
            CausalRMSNorm(self.channels)
            if causal_channel_norm and channels is not None
            else None
        )

    @property
    def inverse_tau(self) -> torch.Tensor:
        return torch.sigmoid(self.inverse_tau_raw).clamp(1e-3, 1.0 - 1e-3)

    def forward(self, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        membrane = torch.zeros_like(current[..., 0])
        inverse_tau = self.inverse_tau.to(current)
        if inverse_tau.numel() > 1:
            inverse_tau = inverse_tau.view(*([1] * (membrane.ndim - 1)), -1)
        spikes = []
        membranes = []
        for step in range(current.shape[-1]):
            step_current = current[..., step]
            if self.current_norm is not None:
                step_current = self.current_norm(step_current)
            membrane = membrane + inverse_tau * (step_current - membrane)
            spike = spike_fn(membrane - self.threshold)
            membrane = membrane * (1.0 - spike)
            spikes.append(spike)
            membranes.append(membrane)
        return torch.stack(spikes, -1), torch.stack(membranes, -1)


class CausalLeakyANNLayer(nn.Module):
    """Parameter-matched non-spiking control for CLIF/PLIF dynamics.

    The layer has the same per-channel learnable decay count as CLIF but no
    threshold, reset, complement state, or surrogate gradient.  It is used
    only as a scientific control with the same delayed event input and readout.
    """

    def __init__(
        self,
        decay: float = 0.9,
        channels: int | None = None,
        learnable_decay: bool = True,
        causal_channel_norm: bool = False,
    ):
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError("causal ANN decay must lie in [0, 1)")
        self.channels = 1 if channels is None else int(channels)
        raw = torch.full((self.channels,), _inverse_sigmoid(float(decay)))
        if learnable_decay:
            self.decay_raw = nn.Parameter(raw)
        else:
            self.register_buffer("decay_raw", raw)
        self.current_norm = (
            CausalRMSNorm(self.channels)
            if causal_channel_norm and channels is not None
            else None
        )

    @property
    def decay(self) -> torch.Tensor:
        return torch.sigmoid(self.decay_raw).clamp(1e-3, 1.0 - 1e-3)

    def forward(self, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = torch.zeros_like(current[..., 0])
        decay = self.decay.to(current)
        if decay.numel() > 1:
            decay = decay.view(*([1] * (state.ndim - 1)), -1)
        activations = []
        states = []
        for step in range(current.shape[-1]):
            step_current = current[..., step]
            if self.current_norm is not None:
                step_current = self.current_norm(step_current)
            state = decay * state + (1.0 - decay) * step_current
            activations.append(torch.tanh(state))
            states.append(state)
        return torch.stack(activations, -1), torch.stack(states, -1)


def _timescale_splits(channels: int, groups: int) -> tuple[int, ...]:
    groups = max(1, min(int(groups), int(channels)))
    base, remainder = divmod(int(channels), groups)
    return tuple(base + (index < remainder) for index in range(groups))


class MultiTimescaleStateLayer(nn.Module):
    """Parallel fast/medium/slow causal dynamics over channel groups."""

    def __init__(
        self,
        channels: int,
        decays: tuple[float, ...] = (0.65, 0.90, 0.975),
        threshold: float = 1.0,
        neuron_type: str = "clif",
        causal_channel_norm: bool = False,
    ):
        super().__init__()
        if not decays:
            raise ValueError("at least one decoder timescale is required")
        n_groups = min(int(channels), len(decays))
        self.splits = _timescale_splits(channels, n_groups)
        self.initial_decays = tuple(float(value) for value in decays[:n_groups])
        neuron_type = neuron_type.lower()
        modules: list[nn.Module] = []
        for split, decay in zip(self.splits, self.initial_decays):
            if neuron_type == "clif":
                module = CLIFLayer(
                    decay=decay,
                    threshold=threshold,
                    channels=split,
                    learnable_decay=True,
                    membrane_norm_groups=0,
                    causal_channel_norm=causal_channel_norm,
                )
            elif neuron_type == "plif":
                module = PLIFLayer(
                    decay=decay,
                    threshold=threshold,
                    channels=split,
                    learnable_decay=True,
                    causal_channel_norm=causal_channel_norm,
                )
            elif neuron_type == "ann":
                module = CausalLeakyANNLayer(
                    decay=decay,
                    channels=split,
                    learnable_decay=True,
                    causal_channel_norm=causal_channel_norm,
                )
            else:
                raise ValueError("neuron_type must be 'clif', 'plif', or 'ann'")
            modules.append(module)
        self.groups = nn.ModuleList(modules)
        self.neuron_type = neuron_type

    def forward(self, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if current.ndim != 3 or current.shape[1] != sum(self.splits):
            raise ValueError("multi-timescale layer expects [N, configured C, T]")
        outputs = []
        states = []
        for chunk, module in zip(torch.split(current, self.splits, dim=1), self.groups):
            output, state = module(chunk)
            outputs.append(output)
            states.append(state)
        return torch.cat(outputs, dim=1), torch.cat(states, dim=1)


class MultiTimescaleSEWBlock(nn.Module):
    """SEW residual block whose branch contains grouped causal dynamics."""

    def __init__(
        self,
        channels: int,
        decays: tuple[float, ...] = (0.65, 0.90, 0.975),
        threshold: float = 1.0,
        connect_function: str = "ADD",
        neuron_type: str = "clif",
        causal_channel_norm: bool = False,
    ):
        super().__init__()
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.neuron = MultiTimescaleStateLayer(
            channels,
            decays=decays,
            threshold=threshold,
            neuron_type=neuron_type,
            causal_channel_norm=causal_channel_norm,
        )
        self.connect_function = connect_function.upper()
        if self.connect_function not in {"ADD", "AND", "IAND"}:
            raise ValueError("SEW connect_function must be ADD, AND, or IAND")
        nn.init.normal_(self.pointwise.weight, mean=0.0, std=0.02)

    def forward_with_state(
        self, activity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        branch, state = self.neuron(self.pointwise(activity))
        if self.connect_function == "ADD":
            output = activity + branch
        elif self.connect_function == "AND":
            output = activity * branch
        else:
            output = (1.0 - activity) * branch
        return output, branch, state

    def forward(self, activity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output, _, state = self.forward_with_state(activity)
        return output, state


class SEWCLIFBlock(nn.Module):
    """Pointwise spike-element-wise residual block with no temporal shift."""

    def __init__(
        self,
        channels: int,
        decay: float = 0.9,
        threshold: float = 1.0,
        membrane_norm_groups: int = 8,
        causal_channel_norm: bool = False,
        connect_function: str = "ADD",
        neuron_type: str = "clif",
    ):
        super().__init__()
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        neuron_type = neuron_type.lower()
        if neuron_type == "clif":
            self.neuron = CLIFLayer(
                decay=decay,
                threshold=threshold,
                channels=channels,
                learnable_decay=True,
                membrane_norm_groups=membrane_norm_groups,
                causal_channel_norm=causal_channel_norm,
            )
        elif neuron_type == "plif":
            self.neuron = PLIFLayer(
                decay=decay,
                threshold=threshold,
                channels=channels,
                learnable_decay=True,
                causal_channel_norm=causal_channel_norm,
            )
        else:
            raise ValueError(f"Unknown spiking neuron type: {neuron_type}")
        self.connect_function = connect_function.upper()
        if self.connect_function not in {"ADD", "AND", "IAND"}:
            raise ValueError("SEW connect_function must be ADD, AND, or IAND")
        nn.init.normal_(self.pointwise.weight, mean=0.0, std=0.02)

    def forward_with_state(
        self, spikes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return residual activation, binary branch spikes, and branch membrane."""

        residual, membrane = self.neuron(self.pointwise(spikes))
        if self.connect_function == "ADD":
            output = spikes + residual
        elif self.connect_function == "AND":
            output = spikes * residual
        else:
            output = (1.0 - spikes) * residual
        return output, residual, membrane

    def forward(self, spikes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output, _, membrane = self.forward_with_state(spikes)
        return output, membrane


class DenseLIFBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, tau_mem: float = 0.9, threshold: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.lif = LIFLayer(tau_mem=tau_mem, threshold=threshold)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, features, time]
        current = self.linear(x.transpose(1, 2)).transpose(1, 2)
        return self.lif(current)
