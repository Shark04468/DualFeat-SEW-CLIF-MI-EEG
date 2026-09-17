"""Vanilla rate-coded LIF-SNN baseline."""

from __future__ import annotations

import torch
from torch import nn

from .encoder import resize_time
from .lif import DenseLIFBlock
from .surrogate import spike_fn


class VanillaSNN(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, hidden: int = 64, timesteps: int = 16):
        super().__init__()
        self.timesteps = timesteps
        self.block = DenseLIFBlock(n_channels, hidden)
        self.readout = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor, **_: torch.Tensor) -> dict[str, torch.Tensor]:
        x = resize_time(x, self.timesteps)
        xmin = x.amin(dim=-1, keepdim=True)
        xmax = x.amax(dim=-1, keepdim=True)
        prob = (x - xmin) / (xmax - xmin).clamp_min(1e-6)
        spikes = spike_fn(prob - 0.5)
        hidden_spikes, mem = self.block(spikes)
        logits = self.readout(hidden_spikes.mean(dim=-1))
        return {"logits": logits, "aux": {"input_spikes": spikes, "hidden_spikes": hidden_spikes, "membrane": mem}}

    def regularization_loss(self, aux: dict[str, torch.Tensor]) -> torch.Tensor:
        return 0.001 * aux["hidden_spikes"].mean()

