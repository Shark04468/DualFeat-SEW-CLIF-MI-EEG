"""Gauge-controlled physical spatial basis for DASP-SNN V6.2-R1."""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .eeg_frontend import BCI2A_COORDINATES, BCI2A_MOTOR_INDICES


BCI2A_CHANNEL_NAMES: tuple[str, ...] = (
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
)


def _normalized_coordinates(coordinates: torch.Tensor) -> torch.Tensor:
    coordinates = torch.as_tensor(coordinates, dtype=torch.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] not in {2, 3}:
        raise ValueError("electrode coordinates must have shape [channels, 2 or 3]")
    if not bool(torch.isfinite(coordinates).all()):
        raise ValueError("electrode coordinates must be finite")
    coordinates = coordinates - coordinates.mean(dim=0, keepdim=True)
    return coordinates / coordinates.norm(dim=-1).amax().clamp_min(1e-8)


class CoupledSpatialBasis(nn.Module):
    """One physical node coordinate system with bounded band residuals.

    The shared basis is common to every band.  Band-specific variation is a
    bounded low-rank residual followed by row normalization.  The same frozen
    module state must be supplied to fold-local evidence and online transport.
    """

    def __init__(
        self,
        channel_names: Sequence[str],
        n_bands: int = 12,
        n_nodes: int = 16,
        *,
        electrode_coordinates: Sequence[Sequence[float]] | torch.Tensor | None = None,
        anchor_indices: Sequence[int] | None = None,
        residual_rank: int = 4,
        anchor_sigma: float = 0.32,
        exact_sensor_basis: bool = False,
        max_shared_deviation: float = 0.25,
        max_band_deviation: float = 0.15,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.channel_names = tuple(str(name) for name in channel_names)
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("channel_names must be unique and ordered")
        self.n_channels = len(self.channel_names)
        self.n_bands = int(n_bands)
        self.n_nodes = int(n_nodes)
        self.residual_rank = int(residual_rank)
        if not 1 <= self.n_nodes <= self.n_channels:
            raise ValueError("n_nodes must lie between one and n_channels")
        if self.residual_rank < 1:
            raise ValueError("residual_rank must be positive")

        if electrode_coordinates is None:
            if self.channel_names != BCI2A_CHANNEL_NAMES:
                raise ValueError(
                    "explicit electrode coordinates are required for non-canonical channel order"
                )
            coordinates = BCI2A_COORDINATES.clone()
        else:
            coordinates = torch.as_tensor(electrode_coordinates, dtype=torch.float32)
            if coordinates.shape[0] != self.n_channels:
                raise ValueError("electrode coordinates do not match channel_names")
        coordinates = _normalized_coordinates(coordinates)

        if anchor_indices is None:
            if self.channel_names == BCI2A_CHANNEL_NAMES:
                preferred = list(BCI2A_MOTOR_INDICES)
            else:
                raise ValueError("anchor_indices are required for non-canonical channel order")
        else:
            preferred = [int(index) for index in anchor_indices]
        if len(set(preferred)) != len(preferred):
            raise ValueError("anchor_indices must be unique")
        if not preferred or min(preferred) < 0 or max(preferred) >= self.n_channels:
            raise ValueError("anchor_indices contain an out-of-range channel")
        remaining = [index for index in range(self.n_channels) if index not in preferred]
        selected = (preferred + remaining)[: self.n_nodes]
        node_coordinates = coordinates[torch.as_tensor(selected, dtype=torch.long)]
        if bool(exact_sensor_basis):
            anchors = F.one_hot(
                torch.as_tensor(selected, dtype=torch.long),
                num_classes=self.n_channels,
            ).to(torch.float32)
        else:
            distance2 = (
                node_coordinates[:, None, :] - coordinates[None, :, :]
            ).square().sum(dim=-1)
            anchors = torch.exp(-distance2 / (2.0 * float(anchor_sigma) ** 2))
            anchors = F.normalize(anchors, p=2, dim=-1)

        self.max_shared_deviation = float(max_shared_deviation)
        self.max_band_deviation = float(max_band_deviation)
        self.register_buffer("electrode_coordinates", coordinates)
        self.register_buffer("node_coordinates", node_coordinates)
        self.register_buffer("anchors", anchors)
        self.shared_delta = nn.Parameter(torch.zeros_like(anchors))
        self.band_coefficients = nn.Parameter(torch.zeros(self.n_bands, self.residual_rank))
        basis_generator = torch.Generator().manual_seed(17_291)
        self.band_basis = nn.Parameter(
            torch.randn(
                self.residual_rank,
                self.n_nodes,
                self.n_channels,
                generator=basis_generator,
            )
            * 0.02
        )
        if not trainable:
            self.freeze()

    def validate_channel_names(self, channel_names: Sequence[str]) -> None:
        observed = tuple(str(name) for name in channel_names)
        if observed != self.channel_names:
            raise ValueError(
                "EEG channel order differs from the frozen physical basis: "
                f"expected {self.channel_names}, received {observed}"
            )

    def _gauge_fixed_band_basis(self) -> torch.Tensor:
        basis = self.band_basis
        norm = basis.square().sum(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
        normalized = basis / norm
        flattened = normalized.detach().abs().flatten(1)
        pivot = flattened.argmax(dim=1)
        signed_flat = normalized.detach().flatten(1)
        orientation = torch.sign(signed_flat.gather(1, pivot[:, None])).view(-1, 1, 1)
        orientation = torch.where(orientation == 0, torch.ones_like(orientation), orientation)
        return normalized * orientation

    def weight(self) -> torch.Tensor:
        shared = self.shared_weight()
        basis = self._gauge_fixed_band_basis()
        coefficients = torch.tanh(self.band_coefficients)
        residual = torch.einsum("br,rkc->bkc", coefficients, basis)
        weight = shared[None] + self.max_band_deviation * residual
        return F.normalize(weight, p=2, dim=-1)

    def shared_weight(self) -> torch.Tensor:
        shared = self.anchors + self.max_shared_deviation * torch.tanh(self.shared_delta)
        return F.normalize(shared, p=2, dim=-1)

    def forward(self, analytic: torch.Tensor) -> torch.Tensor:
        if analytic.ndim != 4 or analytic.shape[1] != self.n_bands:
            raise ValueError("spatial basis expects [N, configured bands, C, T]")
        if analytic.shape[2] != self.n_channels:
            raise ValueError("spatial basis received an incompatible channel axis")
        return torch.einsum("bkc,nbct->nbkt", self.weight().to(analytic), analytic)

    def project_real(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.n_channels:
            raise ValueError("real projection expects [N, configured channels, T]")
        shared = self.shared_weight()
        return torch.einsum("kc,nct->nkt", shared.to(x), x)

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @property
    def frozen(self) -> bool:
        return all(not parameter.requires_grad for parameter in self.parameters())

    def orthogonality_loss(self) -> torch.Tensor:
        weight = self.weight()
        gram = weight @ weight.transpose(-1, -2)
        eye = torch.eye(self.n_nodes, device=gram.device, dtype=gram.dtype)
        return (gram - eye).square().mean()

    def state_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps(self.channel_names).encode("utf-8"))
        for name, value in sorted(self.state_dict().items()):
            digest.update(name.encode("utf-8"))
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()
