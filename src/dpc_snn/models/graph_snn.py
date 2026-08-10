"""Graph SNN baseline without learnable delay and phase coupling."""

from __future__ import annotations

import math

import torch
from torch import nn

from .encoder import resize_time
from .lif import LIFLayer
from .surrogate import spike_fn


class GraphSNNNoDelay(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, hidden_channels: int = 32, timesteps: int = 16):
        super().__init__()
        self.timesteps = timesteps
        self.adj = nn.Parameter(torch.empty(n_channels, n_channels))
        nn.init.xavier_uniform_(self.adj)
        self.register_buffer("self_mask", 1.0 - torch.eye(n_channels))
        self.lif = LIFLayer()
        self.feature = nn.Sequential(nn.Linear(n_channels, hidden_channels), nn.ELU())
        self.readout = nn.Linear(hidden_channels, n_classes)

    def forward(self, x: torch.Tensor, **_: torch.Tensor) -> dict[str, torch.Tensor]:
        x = resize_time(x, self.timesteps)
        xmin = x.amin(dim=-1, keepdim=True)
        xmax = x.amax(dim=-1, keepdim=True)
        spikes = spike_fn((x - xmin) / (xmax - xmin).clamp_min(1e-6) - 0.5)
        weight = torch.tanh(self.adj) * self.self_mask
        current = torch.einsum("ij,njt->nit", weight, spikes) / math.sqrt(x.shape[1])
        hidden_spikes, mem = self.lif(current)
        logits = self.readout(self.feature(hidden_spikes.mean(dim=-1)))
        return {"logits": logits, "aux": {"input_spikes": spikes, "hidden_spikes": hidden_spikes, "edge_weight": weight, "membrane": mem}}

    def regularization_loss(self, aux: dict[str, torch.Tensor]) -> torch.Tensor:
        return 0.001 * aux["hidden_spikes"].mean() + 0.0001 * aux["edge_weight"].abs().mean()

