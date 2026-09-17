"""Continuous EEG front-end for the V3 delay-phase model."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(value.clamp_min(1e-4)))


class LearnableAnalyticFilterBank(nn.Module):
    """Smooth learnable band-pass masks followed by an analytic transform."""

    def __init__(
        self,
        sfreq: float,
        band_edges_hz: list[list[float]],
        max_center_shift_hz: float = 1.5,
        min_bandwidth_hz: float = 1.5,
        transition_hz: float = 1.0,
        padding_seconds: float = 0.5,
        max_high_hz: float | None = None,
        max_bandwidth_scale: float = 1.5,
    ):
        super().__init__()
        edges = torch.as_tensor(band_edges_hz, dtype=torch.float32)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("band_edges_hz must have shape [n_bands, 2]")
        centers = edges.mean(dim=1)
        widths = edges[:, 1] - edges[:, 0]
        if torch.any(widths <= 0):
            raise ValueError("Every EEG band must have positive width")
        self.sfreq = float(sfreq)
        self.max_center_shift_hz = float(max_center_shift_hz)
        self.min_bandwidth_hz = float(min_bandwidth_hz)
        self.transition_hz = float(transition_hz)
        self.padding_seconds = float(padding_seconds)
        self.max_high_hz = float(self.sfreq / 2.0 - 1e-3 if max_high_hz is None else max_high_hz)
        self.max_bandwidth_scale = float(max_bandwidth_scale)
        if self.max_high_hz <= 1.0 or self.max_high_hz >= self.sfreq / 2.0:
            raise ValueError("max_high_hz must lie strictly between 1 Hz and Nyquist")
        if self.max_high_hz <= 1.0 + self.min_bandwidth_hz:
            raise ValueError("Anti-alias range is too narrow for the minimum bandwidth")
        if torch.any(edges[:, 0] < 1.0):
            raise ValueError("Initial EEG bands must start at or above 1 Hz")
        if torch.any(edges[:, 1] > self.max_high_hz + 1e-6):
            raise ValueError("Initial EEG bands exceed the anti-alias high-frequency limit")
        if self.max_bandwidth_scale < 1.0:
            raise ValueError("max_bandwidth_scale must be at least one")
        self.register_buffer("initial_centers_hz", centers)
        self.register_buffer("initial_widths_hz", widths)
        self.center_shift_raw = nn.Parameter(torch.zeros_like(centers))
        self.bandwidth_raw = nn.Parameter(_inverse_softplus(widths - self.min_bandwidth_hz))
        self.band_gain_raw = nn.Parameter(torch.zeros_like(centers))

    @property
    def n_bands(self) -> int:
        return int(self.initial_centers_hz.numel())

    def band_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        center = self.initial_centers_hz + self.max_center_shift_hz * torch.tanh(
            self.center_shift_raw
        )
        center = center.clamp(
            min=1.0 + self.min_bandwidth_hz / 2.0,
            max=self.max_high_hz - self.min_bandwidth_hz / 2.0,
        )
        unconstrained_width = self.min_bandwidth_hz + F.softplus(self.bandwidth_raw)
        physiological_limit = self.initial_widths_hz * self.max_bandwidth_scale
        anti_alias_limit = 2.0 * (self.max_high_hz - center).clamp_min(self.min_bandwidth_hz / 2.0)
        width = torch.minimum(
            unconstrained_width,
            torch.minimum(physiological_limit, anti_alias_limit),
        )
        gain = F.softplus(self.band_gain_raw) / math.log(2.0)
        return center, width, gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_time = x.shape[-1]
        padding = min(max(0, int(round(self.padding_seconds * self.sfreq))), original_time - 1)
        if padding:
            x = F.pad(x, (padding, padding), mode="reflect")
        n_time = x.shape[-1]
        freq = torch.fft.fftfreq(n_time, d=1.0 / self.sfreq, device=x.device, dtype=x.dtype)
        abs_freq = freq.abs()[None, :]
        center, width, gain = self.band_parameters()
        low = (center - width / 2.0).clamp_min(1.0)[:, None]
        high = (center + width / 2.0).clamp_max(self.max_high_hz)[:, None]
        sharpness = 4.0 / max(self.transition_hz, 1e-3)
        mask = torch.sigmoid(sharpness * (abs_freq - low)) * torch.sigmoid(
            sharpness * (high - abs_freq)
        )
        mask = mask * gain[:, None]

        hilbert = torch.zeros_like(freq)
        hilbert[freq > 0] = 2.0
        hilbert[freq == 0] = 1.0
        if n_time % 2 == 0:
            hilbert[freq.abs() == self.sfreq / 2.0] = 1.0
        spectrum = torch.fft.fft(x, dim=-1)
        analytic_spectrum = (
            spectrum[:, None, :, :] * mask[None, :, None, :] * hilbert[None, None, None, :]
        )
        analytic = torch.fft.ifft(analytic_spectrum, dim=-1)
        if padding:
            analytic = analytic[..., padding : padding + original_time]
        return analytic


BCI2A_COORDINATES = torch.tensor(
    [
        [0.00, 1.00],
        [-0.52, 0.62],
        [-0.26, 0.66],
        [0.00, 0.70],
        [0.26, 0.66],
        [0.52, 0.62],
        [-0.82, 0.18],
        [-0.52, 0.20],
        [-0.26, 0.22],
        [0.00, 0.24],
        [0.26, 0.22],
        [0.52, 0.20],
        [0.82, 0.18],
        [-0.50, -0.26],
        [-0.25, -0.28],
        [0.00, -0.30],
        [0.25, -0.28],
        [0.50, -0.26],
        [-0.24, -0.68],
        [0.00, -0.72],
        [0.24, -0.68],
        [0.00, -1.00],
    ],
    dtype=torch.float32,
)

BCI2A_MOTOR_INDICES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17)


class AnchoredSpatialProjection(nn.Module):
    """Coordinate-aware full-channel spatial filters centred on motor regions."""

    def __init__(
        self,
        n_channels: int,
        n_nodes: int,
        max_deviation: float = 0.75,
        anchor_sigma: float = 0.32,
        identity_init: bool = False,
        electrode_coordinates: list[list[float]] | torch.Tensor | None = None,
        anchor_indices: list[int] | torch.Tensor | None = None,
    ):
        super().__init__()
        if n_nodes > n_channels:
            raise ValueError("n_nodes cannot exceed n_channels")
        self.n_channels = int(n_channels)
        self.n_nodes = int(n_nodes)
        self.max_deviation = float(max_deviation)
        if electrode_coordinates is not None:
            coordinates = torch.as_tensor(electrode_coordinates, dtype=torch.float32)
            if coordinates.ndim != 2 or coordinates.shape[0] != n_channels:
                raise ValueError("electrode_coordinates must have shape [n_channels, 2 or 3]")
            if coordinates.shape[1] not in {2, 3} or not bool(torch.isfinite(coordinates).all()):
                raise ValueError("electrode_coordinates must contain finite 2D or 3D positions")
            coordinates = coordinates - coordinates.mean(dim=0, keepdim=True)
            scale = coordinates.square().sum(dim=-1).sqrt().amax().clamp_min(1e-6)
            coordinates = coordinates / scale
            if anchor_indices is None:
                preferred = list(range(n_channels))
            else:
                preferred = [int(index) for index in torch.as_tensor(anchor_indices).flatten()]
                if len(set(preferred)) != len(preferred):
                    raise ValueError("anchor_indices must be unique")
                if not preferred or min(preferred) < 0 or max(preferred) >= n_channels:
                    raise ValueError("anchor_indices contain an out-of-range channel")
        elif n_channels == len(BCI2A_COORDINATES):
            coordinates = BCI2A_COORDINATES.clone()
            preferred = list(BCI2A_MOTOR_INDICES)
        else:
            coordinates = torch.stack(
                (torch.linspace(-1.0, 1.0, n_channels), torch.zeros(n_channels)), dim=1
            )
            preferred = list(range(n_channels))
        if n_nodes <= len(preferred):
            node_indices = preferred[:n_nodes]
        else:
            remaining = [index for index in range(n_channels) if index not in preferred]
            node_indices = (preferred + remaining)[:n_nodes]
        node_coordinates = coordinates[torch.as_tensor(node_indices, dtype=torch.long)]
        if identity_init:
            if n_nodes != n_channels:
                raise ValueError("identity spatial projection requires n_nodes == n_channels")
            anchors = torch.eye(n_channels, dtype=coordinates.dtype)
            node_coordinates = coordinates.clone()
        else:
            distance2 = (node_coordinates[:, None, :] - coordinates[None, :, :]).square().sum(-1)
            anchors = torch.exp(-distance2 / (2.0 * float(anchor_sigma) ** 2))
            anchors = F.normalize(anchors, p=2, dim=-1)
        self.register_buffer("anchors", anchors)
        self.register_buffer("electrode_coordinates", coordinates)
        self.register_buffer("node_coordinates", node_coordinates)
        self.delta = nn.Parameter(torch.zeros_like(anchors))

    def weight(self) -> torch.Tensor:
        return F.normalize(self.anchors + self.max_deviation * torch.tanh(self.delta), p=2, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("kc,nbct->nbkt", self.weight().to(x.dtype), x)

    def project_raw(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("kc,nct->nkt", self.weight().to(x.dtype), x)

    def orthogonality_loss(self) -> torch.Tensor:
        weight = self.weight()
        gram = weight @ weight.transpose(0, 1)
        return (
            (gram - torch.eye(self.n_nodes, device=gram.device, dtype=gram.dtype)).square().mean()
        )


class BandSpecificAnchoredSpatialProjection(nn.Module):
    """Motor-cortex anchors plus a low-rank band-specific spatial residual."""

    def __init__(
        self,
        n_channels: int,
        n_nodes: int,
        n_bands: int,
        rank: int = 4,
        max_deviation: float = 0.75,
        max_band_deviation: float = 0.25,
        identity_init: bool = False,
        electrode_coordinates: list[list[float]] | torch.Tensor | None = None,
        anchor_indices: list[int] | torch.Tensor | None = None,
    ):
        super().__init__()
        self.base = AnchoredSpatialProjection(
            n_channels,
            n_nodes,
            max_deviation=max_deviation,
            identity_init=identity_init,
            electrode_coordinates=electrode_coordinates,
            anchor_indices=anchor_indices,
        )
        self.n_bands = int(n_bands)
        self.n_nodes = int(n_nodes)
        self.n_channels = int(n_channels)
        self.rank = max(1, int(rank))
        self.max_band_deviation = float(max_band_deviation)
        self.band_coefficients = nn.Parameter(torch.zeros(self.n_bands, self.rank))
        self.band_basis = nn.Parameter(torch.empty(self.rank, self.n_nodes, self.n_channels))
        nn.init.normal_(self.band_basis, mean=0.0, std=0.02)

    @property
    def node_coordinates(self) -> torch.Tensor:
        return self.base.node_coordinates

    @property
    def delta(self) -> nn.Parameter:
        """Backward-compatible access to the shared spatial residual."""

        return self.base.delta

    @property
    def electrode_coordinates(self) -> torch.Tensor:
        return self.base.electrode_coordinates

    def weight(self) -> torch.Tensor:
        residual = torch.einsum("br,rkc->bkc", self.band_coefficients, self.band_basis)
        weight = self.base.weight()[None] + self.max_band_deviation * torch.tanh(residual)
        return F.normalize(weight, p=2, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.n_bands:
            raise ValueError(f"Expected {self.n_bands} bands, received {x.shape[1]}")
        return torch.einsum("bkc,nbct->nbkt", self.weight().to(x.dtype), x)

    def orthogonality_loss(self) -> torch.Tensor:
        weight = self.weight()
        gram = weight @ weight.transpose(-1, -2)
        eye = torch.eye(self.n_nodes, device=gram.device, dtype=gram.dtype)
        return (gram - eye).square().mean()


class _SymmetricMatrixLog(torch.autograd.Function):
    """Matrix logarithm with a stable derivative at repeated eigenvalues."""

    @staticmethod
    def forward(ctx, matrix: torch.Tensor) -> torch.Tensor:
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        floor = torch.finfo(matrix.dtype).eps
        eigenvalues = eigenvalues.clamp_min(floor)
        ctx.save_for_backward(eigenvalues, eigenvectors)
        return (eigenvectors * eigenvalues.log().unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        eigenvalues, eigenvectors = ctx.saved_tensors
        grad_output = 0.5 * (grad_output + grad_output.transpose(-1, -2))
        rotated_grad = eigenvectors.transpose(-1, -2) @ grad_output @ eigenvectors

        left = eigenvalues.unsqueeze(-1)
        right = eigenvalues.unsqueeze(-2)
        difference = left - right
        log_difference = left.log() - right.log()
        tolerance = torch.finfo(eigenvalues.dtype).eps ** 0.5 * torch.maximum(left, right)
        # The divided difference of log tends to 1/lambda as the two
        # eigenvalues coincide. The symmetric mean is stable near coincidence.
        divided_difference = torch.where(
            difference.abs() > tolerance,
            log_difference / difference,
            2.0 / (left + right),
        )
        grad_matrix = (
            eigenvectors @ (divided_difference * rotated_grad) @ eigenvectors.transpose(-1, -2)
        )
        return (0.5 * (grad_matrix + grad_matrix.transpose(-1, -2)),)


def _stable_log_covariance(x_nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return trace-normalized matrix-log covariance and log variance.

    Scaling each trial/band by one scalar before forming the covariance is
    algebraically cancelled by trace normalization, but prevents float32
    overflow. The decomposition is kept in float64 because these small EEG
    covariance matrices can be rank deficient after CSD projection.
    """

    output_dtype = x_nodes.dtype
    work = x_nodes.to(torch.float64)
    if not bool(torch.isfinite(work).all()):
        raise FloatingPointError("Log-covariance input contains NaN or Inf")
    centered = work - work.mean(dim=-1, keepdim=True)
    log_variance = centered.square().mean(dim=-1).clamp_min(1e-12).log()
    scale = centered.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    scaled = centered / scale.detach()
    covariance = scaled @ scaled.transpose(-1, -2) / max(1, x_nodes.shape[-1] - 1)
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    trace = covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).clamp_min(1e-12)
    covariance = covariance / trace.unsqueeze(-1)
    eye = torch.eye(covariance.shape[-1], device=x_nodes.device, dtype=work.dtype)
    log_covariance = _SymmetricMatrixLog.apply(covariance + 1e-4 * eye)
    return log_covariance.to(output_dtype), log_variance.to(output_dtype)


class LogCovarianceBranch(nn.Module):
    """Compact log-covariance spatial context on anchored latent nodes."""

    def __init__(self, n_nodes: int, out_features: int):
        super().__init__()
        indices = torch.triu_indices(n_nodes, n_nodes)
        self.register_buffer("triu_row", indices[0])
        self.register_buffer("triu_col", indices[1])
        self.proj = nn.Sequential(
            nn.Linear(n_nodes * (n_nodes + 1) // 2, out_features),
            nn.LayerNorm(out_features),
            nn.ELU(),
        )

    def forward(self, x_nodes: torch.Tensor) -> torch.Tensor:
        log_covariance, _ = _stable_log_covariance(x_nodes)
        features = log_covariance[:, self.triu_row, self.triu_col]
        return self.proj(features)


class BandLogCovarianceBranch(nn.Module):
    """Per-band log-covariance and log-variance context.

    The output is a compact trial context for routing and multiplicative
    modulation. It is never an additive classifier input.
    """

    def __init__(self, n_bands: int, n_nodes: int, out_features_per_band: int):
        super().__init__()
        self.n_bands = int(n_bands)
        self.n_nodes = int(n_nodes)
        indices = torch.triu_indices(n_nodes, n_nodes)
        self.register_buffer("triu_row", indices[0])
        self.register_buffer("triu_col", indices[1])
        covariance_features = n_nodes * (n_nodes + 1) // 2
        self.proj = nn.Sequential(
            nn.Linear(covariance_features + n_nodes, out_features_per_band),
            nn.LayerNorm(out_features_per_band),
            nn.ELU(),
        )

    def forward(self, x_nodes: torch.Tensor) -> torch.Tensor:
        if x_nodes.ndim != 4 or x_nodes.shape[1] != self.n_bands:
            raise ValueError("Band covariance expects [N, B, K, T]")
        log_covariance, log_variance = _stable_log_covariance(x_nodes)
        upper = log_covariance[..., self.triu_row, self.triu_col]
        features = torch.cat((upper, log_variance), dim=-1)
        return self.proj(features).flatten(1)


class DelayedBandPairStatistics(nn.Module):
    """Shared log-covariance/log-variance encoder for delayed band pairs.

    Input is the mandatory delayed transport ``[N, Bt, Bs, K, T]``.  Statistics
    are computed independently for every target/source band pair, so no
    undelayed covariance feature or early band aggregation can reach the
    classifier.
    """

    def __init__(self, n_nodes: int, out_features_per_pair: int = 4):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.out_features_per_pair = int(out_features_per_pair)
        indices = torch.triu_indices(self.n_nodes, self.n_nodes)
        self.register_buffer("triu_row", indices[0])
        self.register_buffer("triu_col", indices[1])
        covariance_features = self.n_nodes * (self.n_nodes + 1) // 2
        self.proj = nn.Sequential(
            nn.Linear(covariance_features + self.n_nodes, self.out_features_per_pair),
            nn.LayerNorm(self.out_features_per_pair),
            nn.ELU(),
        )

    def forward(self, delayed: torch.Tensor) -> torch.Tensor:
        if delayed.ndim != 5 or delayed.shape[-2] != self.n_nodes:
            raise ValueError("Delayed pair statistics expect [N, Bt, Bs, K, T]")
        shape = delayed.shape
        pair_nodes = delayed.reshape(-1, self.n_nodes, shape[-1])
        log_covariance, log_variance = _stable_log_covariance(pair_nodes)
        upper = log_covariance[..., self.triu_row, self.triu_col]
        encoded = self.proj(torch.cat((upper, log_variance), dim=-1))
        return encoded.reshape(shape[0], shape[1], shape[2], self.out_features_per_pair)


class DelayedDirectionalMoments(nn.Module):
    """Fixed shift-sensitive moments for each directed delayed band pair."""

    out_features = 4

    def forward(self, delayed: torch.Tensor) -> torch.Tensor:
        if delayed.ndim != 5:
            raise ValueError("Delayed directional moments expect [N, Bt, Bs, K, T]")
        energy = delayed.square().mean(dim=3)
        total = energy.sum(dim=-1).clamp_min(1e-8)
        time = torch.linspace(
            -1.0, 1.0, delayed.shape[-1], device=delayed.device, dtype=delayed.dtype
        )
        centroid = (energy * time).sum(dim=-1) / total
        centered_time = time - centroid[..., None]
        spread = (energy * centered_time.square()).sum(dim=-1) / total
        skew = (energy * centered_time.pow(3)).sum(dim=-1) / (
            total * spread.clamp_min(1e-6).pow(1.5)
        )
        difference = delayed[..., 1:] - delayed[..., :-1]
        flux = (delayed[..., 1:] * difference).mean(dim=(-2, -1))
        power = delayed.square().mean(dim=(-2, -1)).clamp_min(1e-8)
        flux = flux / power
        return torch.stack((centroid, spread, skew, flux), dim=-1)
