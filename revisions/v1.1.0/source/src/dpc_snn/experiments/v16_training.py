"""Training utilities for the E16 continuous information-sufficiency gate."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dpc_snn.models.v16_continuous_fusion import V16ContinuousFusion
from dpc_snn.utils.metrics import classification_metrics


@dataclass
class V16ContinuousFit:
    model: V16ContinuousFusion
    history: list[dict[str, float | int]]
    best_epoch: int
    best_metric: float
    best_accuracy: float
    best_state: dict[str, torch.Tensor]
    last_state: dict[str, torch.Tensor]
    optimizer_steps: int
    elapsed_seconds: float


def seed_v16(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _validated(
    atc_sequence: np.ndarray, fbc_sequence: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    atc = np.ascontiguousarray(atc_sequence, dtype=np.float32)
    fbc = np.ascontiguousarray(fbc_sequence, dtype=np.float32)
    y = np.ascontiguousarray(labels, dtype=np.int64)
    if atc.ndim != 3 or atc.shape[1:] != (18, 32):
        raise ValueError("ATC sequence must have shape [N, 18, 32]")
    if fbc.ndim != 3 or fbc.shape[1:] != (4, 288):
        raise ValueError("FBC sequence must have shape [N, 4, 288]")
    if y.shape != (atc.shape[0],) or fbc.shape[0] != atc.shape[0]:
        raise ValueError("features and labels are not aligned")
    if np.any(y < 0) or np.any(y >= 4):
        raise ValueError("labels must lie in [0, 3]")
    if not np.isfinite(atc).all() or not np.isfinite(fbc).all():
        raise FloatingPointError("features contain non-finite values")
    return atc, fbc, y


def _loader(
    atc: np.ndarray,
    fbc: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    atc, fbc, labels = _validated(atc, fbc, labels)
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(torch.from_numpy(atc), torch.from_numpy(fbc), torch.from_numpy(labels)),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


@torch.no_grad()
def predict_v16_continuous(
    model: V16ContinuousFusion,
    atc: np.ndarray,
    fbc: np.ndarray,
    labels: np.ndarray,
    *,
    device: str,
    batch_size: int = 64,
) -> dict[str, Any]:
    model.eval().to(device)
    loader = _loader(atc, fbc, labels, batch_size=batch_size, shuffle=False, seed=0)
    logits: list[np.ndarray] = []
    prefix_logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch_atc, batch_fbc, batch_y in loader:
        output = model(
            batch_atc.to(device, non_blocking=True),
            batch_fbc.to(device, non_blocking=True),
        )
        if not bool(torch.isfinite(output.prefix_logits).all()):
            raise FloatingPointError("V16 continuous model produced non-finite logits")
        logits.append(output.logits.float().cpu().numpy())
        prefix_logits.append(output.prefix_logits.float().cpu().numpy())
        targets.append(batch_y.numpy())
    all_logits = np.concatenate(logits)
    all_prefix = np.concatenate(prefix_logits)
    all_labels = np.concatenate(targets)
    metrics = classification_metrics(all_labels, all_logits.argmax(axis=1), n_classes=4)
    prefix_accuracy = [
        float(np.mean(all_prefix[:, index].argmax(axis=1) == all_labels))
        for index in range(all_prefix.shape[1])
    ]
    return {
        **metrics,
        "logits": all_logits,
        "prefix_logits": all_prefix,
        "labels": all_labels,
        "prefix_accuracy": prefix_accuracy,
    }


def fit_v16_continuous(
    *,
    atc_train: np.ndarray,
    fbc_train: np.ndarray,
    y_train: np.ndarray,
    atc_validation: np.ndarray | None,
    fbc_validation: np.ndarray | None,
    y_validation: np.ndarray | None,
    device: str,
    seed: int,
    epochs: int = 120,
    patience: int = 20,
    fixed_epoch: int | None = None,
    scheduler_epochs: int | None = None,
    batch_size: int = 48,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    model_kwargs: dict[str, Any] | None = None,
    run_label: str = "",
) -> V16ContinuousFit:
    validation_values = (atc_validation, fbc_validation, y_validation)
    has_validation = all(value is not None for value in validation_values)
    if has_validation != any(value is not None for value in validation_values):
        raise ValueError("all validation arrays must be supplied together")
    if not has_validation and fixed_epoch is None:
        raise ValueError("training without validation requires fixed_epoch")
    if fixed_epoch is not None and int(fixed_epoch) < 1:
        raise ValueError("fixed_epoch must be positive")

    seed_v16(seed)
    model = V16ContinuousFusion(**(model_kwargs or {})).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(scheduler_epochs or epochs), 1),
        eta_min=float(learning_rate) * 0.05,
    )
    loader = _loader(
        atc_train, fbc_train, y_train, batch_size=batch_size, shuffle=True, seed=seed
    )
    maximum_epochs = int(fixed_epoch or epochs)
    best_metric = -math.inf
    best_accuracy = 0.0
    best_epoch = 0
    best_state = _state_cpu(model)
    history: list[dict[str, float | int]] = []
    optimizer_steps = 0
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch_atc, batch_fbc, batch_y in loader:
            batch_atc = batch_atc.to(device, non_blocking=True)
            batch_fbc = batch_fbc.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            output = model(batch_atc, batch_fbc)
            loss = F.cross_entropy(output.logits, batch_y)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("V16 continuous loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer_steps += 1
            total_loss += float(loss.detach()) * batch_y.shape[0]
            total_examples += batch_y.shape[0]
        scheduler.step()

        row: dict[str, float | int] = {
            "epoch": epoch,
            "loss": total_loss / max(total_examples, 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if has_validation:
            evaluation = predict_v16_continuous(
                model,
                atc_validation,
                fbc_validation,
                y_validation,
                device=device,
                batch_size=batch_size,
            )
            metric = float(evaluation["kappa"])
            accuracy = float(evaluation["accuracy"])
            row.update({"validation_kappa": metric, "validation_accuracy": accuracy})
            if (metric, accuracy, -epoch) > (best_metric, best_accuracy, -best_epoch):
                best_metric = metric
                best_accuracy = accuracy
                best_epoch = epoch
                best_state = _state_cpu(model)
                stale = 0
            else:
                stale += 1
        else:
            best_epoch = epoch
            best_state = _state_cpu(model)
        history.append(row)
        if run_label and (epoch == 1 or epoch % 10 == 0 or epoch == maximum_epochs):
            print(
                "__V16_A_PROGRESS__ "
                f"run={run_label} epoch={epoch}/{maximum_epochs} "
                f"loss={row['loss']:.6f} "
                f"val_kappa={row.get('validation_kappa', float('nan')):.6f}",
                flush=True,
            )
        if has_validation and epoch >= 10 and stale >= int(patience):
            break

    last_state = _state_cpu(model)
    if has_validation:
        model.load_state_dict(deepcopy(best_state), strict=True)
    else:
        best_metric = float("nan")
        best_accuracy = float("nan")
    return V16ContinuousFit(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_metric=best_metric,
        best_accuracy=best_accuracy,
        best_state=best_state,
        last_state=last_state,
        optimizer_steps=optimizer_steps,
        elapsed_seconds=time.perf_counter() - started,
    )
