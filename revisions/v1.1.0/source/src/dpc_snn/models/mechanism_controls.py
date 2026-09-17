"""Non-SNN delay/phase mechanism controls for E15."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .encoder import resize_time


class DelayGraphANN(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, timesteps: int = 16, d_max: int = 16, hidden: int = 64):
        super().__init__()
        self.timesteps = timesteps
        self.d_max = d_max
        self.adj = nn.Parameter(torch.empty(n_channels, n_channels))
        self.delay_logits = nn.Parameter(torch.zeros(n_channels, n_channels, d_max + 1))
        nn.init.xavier_uniform_(self.adj)
        self.mlp = nn.Sequential(nn.Linear(n_channels * timesteps, hidden), nn.ELU(), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor, **_: torch.Tensor) -> dict[str, torch.Tensor]:
        x = resize_time(x, self.timesteps)
        delayed = []
        for d in range(self.d_max + 1):
            delayed.append(x if d == 0 else F.pad(x[..., :-d], (d, 0)))
        stack = torch.stack(delayed, dim=2)  # [N, C, D, T]
        prob = F.softmax(self.delay_logits, dim=-1)
        mixed = torch.einsum("njdt,ijd->nit", stack, prob)
        weight = torch.tanh(self.adj)
        out = torch.einsum("ij,njt->nit", weight, mixed)
        return {"logits": self.mlp(out.flatten(1)), "aux": {"edge_weight": weight}}


class PhaseGatedGNN(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, n_bands: int = 2, timesteps: int = 16, hidden: int = 64):
        super().__init__()
        self.timesteps = timesteps
        self.adj = nn.Parameter(torch.empty(n_bands, n_channels, n_channels))
        self.phase_pref = nn.Parameter(torch.zeros(n_bands, n_channels, n_channels))
        nn.init.xavier_uniform_(self.adj)
        self.mlp = nn.Sequential(nn.Linear(n_bands * n_channels * timesteps, hidden), nn.ELU(), nn.Linear(hidden, n_classes))

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor, **_: torch.Tensor) -> dict[str, torch.Tensor]:
        amp = resize_time(amplitude, self.timesteps)
        ph = resize_time(phase, self.timesteps)
        gate = torch.sigmoid(2.0 * torch.cos(ph.unsqueeze(3) - ph.unsqueeze(2) - self.phase_pref[None, :, :, :, None]))
        weight = torch.tanh(self.adj)
        out = torch.einsum("bij,nbjt,nbijt->nbit", weight, amp, gate)
        return {"logits": self.mlp(out.flatten(1)), "aux": {"edge_weight": weight, "phase_pref": self.phase_pref}}


class DilatedTCN(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, hidden, kernel_size=5, padding=2, dilation=1),
            nn.ELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2),
            nn.ELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=8, dilation=4),
            nn.ELU(),
        )
        self.readout = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor, **_: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.net(x).mean(dim=-1)
        return {"logits": self.readout(feat), "aux": {}}


class ComplexValuedCNN(nn.Module):
    """A practical complex-control approximation using amplitude and phase channels."""

    def __init__(self, n_channels: int, n_classes: int, n_bands: int = 2, hidden: int = 32):
        super().__init__()
        in_ch = 2 * n_bands * n_channels
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=9, padding=4),
            nn.ELU(),
            nn.Conv1d(hidden, hidden, kernel_size=9, padding=4),
            nn.ELU(),
        )
        self.readout = nn.Linear(hidden, n_classes)

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor, **_: torch.Tensor) -> dict[str, torch.Tensor]:
        real = amplitude * torch.cos(phase)
        imag = amplitude * torch.sin(phase)
        x = torch.cat([real, imag], dim=1).flatten(1, 2)
        feat = self.net(x).mean(dim=-1)
        return {"logits": self.readout(feat), "aux": {}}

