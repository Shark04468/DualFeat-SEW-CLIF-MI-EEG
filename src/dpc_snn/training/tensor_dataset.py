"""Torch dataset wrappers for trial arrays."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class TrialTensorDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        amplitude: np.ndarray | None = None,
        phase: np.ndarray | None = None,
        delay_evidence_x: np.ndarray | None = None,
        meta: dict[str, Any] | None = None,
    ):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.amplitude = torch.as_tensor(amplitude, dtype=torch.float32) if amplitude is not None else None
        self.phase = torch.as_tensor(phase, dtype=torch.float32) if phase is not None else None
        self.delay_evidence_x = (
            torch.as_tensor(delay_evidence_x, dtype=torch.float32)
            if delay_evidence_x is not None
            else None
        )
        self.meta = meta or {}

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {"x": self.x[idx], "y": self.y[idx]}
        if self.amplitude is not None:
            item["amplitude"] = self.amplitude[idx]
        if self.phase is not None:
            item["phase"] = self.phase[idx]
        if self.delay_evidence_x is not None:
            item["delay_evidence_x"] = self.delay_evidence_x[idx]
        return item


def batch_to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
