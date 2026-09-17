"""Shared physical-electrode geometry for motor-imagery datasets."""

from __future__ import annotations

from typing import Any

import numpy as np


MOTOR_REGION_ANCHORS = (
    "FC5",
    "FC3",
    "FC1",
    "FC2",
    "FC4",
    "FC6",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
)


def electrode_coordinates(epochs: Any) -> np.ndarray:
    """Return finite three-dimensional coordinates in epoch channel order."""

    montage = epochs.get_montage()
    if montage is None:
        raise ValueError("EEG epochs do not contain an electrode montage")
    positions = montage.get_positions().get("ch_pos", {})
    missing = [name for name in epochs.ch_names if name not in positions]
    if missing:
        raise ValueError(f"EEG montage is missing coordinates for {missing}")
    coordinates = np.asarray([positions[name] for name in epochs.ch_names], dtype=np.float32)
    if coordinates.shape != (len(epochs.ch_names), 3) or not np.isfinite(coordinates).all():
        raise ValueError("Electrode coordinates must be finite [channels, 3]")
    return coordinates


def motor_anchor_indices(channel_names: list[str], minimum: int = 16) -> np.ndarray:
    """Select declared bilateral motor-region anchors in channel order."""

    lookup = {name.upper(): index for index, name in enumerate(channel_names)}
    indices = [lookup[name.upper()] for name in MOTOR_REGION_ANCHORS if name.upper() in lookup]
    if len(indices) < int(minimum):
        raise ValueError(
            f"EEG montage exposes {len(indices)} motor-region anchors; at least {minimum} required"
        )
    return np.asarray(indices, dtype=np.int64)
