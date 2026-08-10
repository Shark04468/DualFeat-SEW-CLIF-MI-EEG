"""Causal variance/covariance modulation from mandatory delayed currents."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _causal_mean_complete(x: torch.Tensor, window: int) -> torch.Tensor:
    """Trailing mean, masked until the complete causal window is available."""

    if x.ndim != 3:
        raise ValueError("causal rolling mean expects [N, C, T]")
    channels = x.shape[1]
    kernel = torch.ones(channels, 1, window, device=x.device, dtype=x.dtype) / window
    mean = F.conv1d(F.pad(x, (window - 1, 0)), kernel, groups=channels)
    if window > 1:
        mask = torch.arange(x.shape[-1], device=x.device) >= window - 1
        mean = mean * mask.to(mean.dtype)[None, None, :]
    return mean


class CausalDelayedGeometryFiLM(nn.Module):
    """Bounded current modulation from delayed trailing EEG geometry.

    No eigendecomposition or matrix logarithm is used.  Zero delayed current
    yields exactly zero geometry features and therefore a neutral scale of one.
    """

    def __init__(
        self,
        n_bands: int = 12,
        n_nodes: int = 16,
        output_channels: int = 32,
        sfreq: float = 125.0,
        windows_seconds: Sequence[float] = (0.25, 0.5, 1.0),
        covariance_components: int = 6,
        covariance_shrinkage: float = 0.10,
        compression_scale: float = 0.10,
        max_modulation: float = 0.25,
        hidden_features: int = 24,
    ) -> None:
        super().__init__()
        self.n_bands = int(n_bands)
        self.n_nodes = int(n_nodes)
        self.output_channels = int(output_channels)
        self.sfreq = float(sfreq)
        self.windows = tuple(max(2, int(round(float(value) * self.sfreq))) for value in windows_seconds)
        if len(set(self.windows)) != len(self.windows):
            raise ValueError("geometry windows must map to distinct sample counts")
        self.covariance_components = int(covariance_components)
        if not 1 <= self.covariance_components <= self.n_nodes:
            raise ValueError("covariance_components must lie in [1, n_nodes]")
        if not 0.0 <= covariance_shrinkage <= 1.0:
            raise ValueError("covariance_shrinkage must lie in [0, 1]")
        self.covariance_shrinkage = float(covariance_shrinkage)
        self.compression_scale = float(compression_scale)
        self.max_modulation = float(max_modulation)
        self.projection = nn.Parameter(
            torch.randn(self.n_bands, self.covariance_components, self.n_nodes) * 0.05
        )
        triu = torch.triu_indices(self.covariance_components, self.covariance_components)
        self.register_buffer("triu_row", triu[0])
        self.register_buffer("triu_col", triu[1])
        features_per_band = len(self.windows) * (
            self.n_nodes + self.triu_row.numel()
        )
        in_features = self.n_bands * features_per_band
        self.modulation = nn.Sequential(
            nn.Conv1d(in_features, int(hidden_features), kernel_size=1, bias=False),
            nn.ELU(),
            nn.Conv1d(int(hidden_features), self.output_channels, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.modulation[-1].weight)

    def projection_weight(self) -> torch.Tensor:
        return F.normalize(self.projection, p=2, dim=-1)

    def _window_features(
        self,
        delayed: torch.Tensor,
        projected: torch.Tensor,
        window: int,
    ) -> torch.Tensor:
        n_trials, n_bands, n_nodes, n_time = delayed.shape
        flat = delayed.reshape(n_trials, n_bands * n_nodes, n_time)
        mean = _causal_mean_complete(flat, window)
        mean_square = _causal_mean_complete(flat.square(), window)
        variance = (mean_square - mean.square()).clamp_min(0.0)
        log_variance = torch.log1p(variance / max(self.compression_scale, 1e-8))
        log_variance = log_variance.reshape(n_trials, n_bands, n_nodes, n_time)

        components = self.covariance_components
        projected_flat = projected.reshape(n_trials, n_bands * components, n_time)
        projected_mean = _causal_mean_complete(projected_flat, window).reshape(
            n_trials, n_bands, components, n_time
        )
        outer = projected[:, :, :, None, :] * projected[:, :, None, :, :]
        outer_flat = outer.reshape(n_trials, n_bands * components * components, n_time)
        covariance = _causal_mean_complete(outer_flat, window).reshape_as(outer)
        covariance = covariance - projected_mean[:, :, :, None] * projected_mean[:, :, None]
        diagonal_mean = covariance.diagonal(dim1=2, dim2=3).mean(dim=-1)
        eye = torch.eye(components, device=delayed.device, dtype=delayed.dtype)
        covariance = (1.0 - self.covariance_shrinkage) * covariance + (
            self.covariance_shrinkage
            * diagonal_mean[:, :, None, None, :]
            * eye[None, None, :, :, None]
        )
        upper = covariance[:, :, self.triu_row, self.triu_col, :]
        compressed = torch.sign(upper) * torch.log1p(
            upper.abs() / max(self.compression_scale, 1e-8)
        )
        return torch.cat((log_variance, compressed), dim=2)

    def forward(self, delayed_current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if delayed_current.ndim != 4 or delayed_current.shape[1:3] != (
            self.n_bands,
            self.n_nodes,
        ):
            raise ValueError("delayed geometry expects [N, configured B, K, T]")
        projected = torch.einsum(
            "bgk,nbkt->nbgt", self.projection_weight().to(delayed_current), delayed_current
        )
        features = torch.cat(
            [
                self._window_features(delayed_current, projected, window)
                for window in self.windows
            ],
            dim=2,
        )
        flat = features.flatten(1, 2)
        scale = 1.0 + self.max_modulation * torch.tanh(self.modulation(flat))
        return scale, features
