"""Model dispatch helpers for mixed raw/feature inputs."""

from __future__ import annotations

import inspect
from typing import Any

import torch


def _accepted_forward_kwargs(model: Any, candidates: dict[str, torch.Tensor | None]) -> dict[str, torch.Tensor]:
    provided = {key: value for key, value in candidates.items() if value is not None}
    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError, AttributeError):
        return provided
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
    if accepts_kwargs:
        return provided
    return {key: value for key, value in provided.items() if key in params}


def forward_batch(model: Any, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    x = batch.get("x")
    amplitude = batch.get("amplitude")
    phase = batch.get("phase")
    delay_evidence_x = batch.get("delay_evidence_x")
    kwargs = _accepted_forward_kwargs(
        model,
        {
            "x": x,
            "amplitude": amplitude,
            "phase": phase,
            "delay_evidence_x": delay_evidence_x,
        },
    )
    if kwargs:
        return model(**kwargs)
    if x is not None:
        return model(x)
    if amplitude is not None and phase is not None:
        return model(amplitude, phase)
    else:
        raise KeyError("Batch must contain x or amplitude/phase")


def regularization_loss(model: Any, aux: dict[str, torch.Tensor]) -> torch.Tensor:
    if hasattr(model, "regularization_loss"):
        return model.regularization_loss(aux)
    if aux:
        first = next(iter(aux.values()))
        if isinstance(first, torch.Tensor):
            return torch.zeros((), device=first.device)
    return torch.zeros(())
