"""Evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dpc_snn.utils.metrics import classification_metrics, confusion_matrix

from .forward import forward_batch
from .tensor_dataset import batch_to_device


@torch.no_grad()
def predict(model: Any, loader: DataLoader, device: str) -> dict[str, Any]:
    model.eval()
    y_true = []
    y_pred = []
    logits_all = []
    aux_values: dict[str, list[float]] = {}
    expert_usage_sum: torch.Tensor | None = None
    expert_usage_count = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        out = forward_batch(model, batch)
        logits = out["logits"]
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Evaluation logits contain NaN or Inf")
        pred = logits.argmax(dim=1)
        y_true.append(batch["y"].detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())
        logits_all.append(logits.detach().cpu().numpy())
        for key in ["delay_alignment_loss", "delay_entropy", "phase_confidence_mean", "delay_gate"]:
            value = out.get("aux", {}).get(key)
            if isinstance(value, torch.Tensor):
                aux_values.setdefault(key, []).append(float(value.detach().float().mean().cpu()))
        mixture = out.get("aux", {}).get("expert_mixture")
        if isinstance(mixture, torch.Tensor):
            batch_usage = mixture.detach().float().sum(dim=0).cpu()
            expert_usage_sum = batch_usage if expert_usage_sum is None else expert_usage_sum + batch_usage
            expert_usage_count += int(mixture.shape[0])
    result = {
        "y_true": np.concatenate(y_true),
        "y_pred": np.concatenate(y_pred),
        "logits": np.concatenate(logits_all),
    }
    result.update({key: float(np.mean(values)) for key, values in aux_values.items()})
    if expert_usage_sum is not None and expert_usage_count:
        for expert, usage in enumerate(expert_usage_sum / expert_usage_count):
            result[f"expert_usage_{expert}"] = float(usage)
    return result


def evaluate(model: Any, loader: DataLoader, device: str, n_classes: int | None = None) -> dict[str, Any]:
    pred = predict(model, loader, device)
    metrics = classification_metrics(pred["y_true"], pred["y_pred"], n_classes=n_classes)
    metrics["confusion_matrix"] = confusion_matrix(pred["y_true"], pred["y_pred"], n_classes=n_classes)
    return {**metrics, **pred}
