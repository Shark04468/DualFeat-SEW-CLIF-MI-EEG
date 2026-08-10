"""Spike encoders for band amplitude/phase features."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .surrogate import spike_fn


def resize_time(x: torch.Tensor, steps: int, mode: str = "linear") -> torch.Tensor:
    if x.shape[-1] == steps:
        return x
    shape = x.shape
    flat = x.reshape(-1, 1, shape[-1])
    resized = F.interpolate(flat, size=steps, mode=mode, align_corners=False)
    return resized.reshape(*shape[:-1], steps)


def resize_phase(phase: torch.Tensor, steps: int) -> torch.Tensor:
    """Resample wrapped phase through its unit-circle representation."""
    if phase.shape[-1] == steps:
        return phase
    real = resize_time(torch.cos(phase), steps)
    imag = resize_time(torch.sin(phase), steps)
    return torch.atan2(imag, real)


class PhaseAwareSpikeEncoder(nn.Module):
    def __init__(
        self,
        n_bands: int,
        n_channels: int,
        timesteps: int,
        threshold: float = 0.5,
        deterministic: bool = True,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.n_channels = n_channels
        self.timesteps = timesteps
        self.threshold = threshold
        self.deterministic = deterministic
        self.alpha = nn.Parameter(torch.ones(n_bands, n_channels))
        self.beta = nn.Parameter(torch.ones(n_bands, n_channels))
        self.theta = nn.Parameter(torch.zeros(n_bands, n_channels))
        # Trainable band/channel centers preserve between-trial amplitude
        # differences while still giving the spike threshold a stable origin.
        self.amp_center = nn.Parameter(torch.zeros(n_bands, n_channels))

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        amp = resize_time(amplitude, self.timesteps)
        ph = resize_phase(phase, self.timesteps)
        # Per-trial min-max normalisation erases the absolute band-power
        # changes that carry MI ERD/ERS information.  The raw EEG has already
        # been standardised from training data; log amplitude keeps that
        # information while controlling the Hilbert-envelope tail.
        amp_feature = torch.log1p(amp.clamp_min(0.0))
        logits = (
            self.alpha[None, :, :, None] * (amp_feature - self.amp_center[None, :, :, None])
            + self.beta[None, :, :, None] * torch.cos(ph - self.theta[None, :, :, None])
        )
        prob = torch.sigmoid(logits)
        if self.deterministic:
            spikes = spike_fn(prob - self.threshold)
        else:
            spikes = torch.bernoulli(prob)
        return spikes, prob
