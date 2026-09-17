"""Torch helpers kept isolated so non-training scripts can run without torch."""

from __future__ import annotations

from typing import Any

from .imports import optional_import


def torch_module() -> Any:
    return optional_import("torch", "model training")


def resolve_device(device: str) -> str:
    torch = torch_module()
    if device == "cuda_if_available":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def count_parameters(model: Any, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        params = [p for p in params if p.requires_grad]
    return int(sum(p.numel() for p in params))

