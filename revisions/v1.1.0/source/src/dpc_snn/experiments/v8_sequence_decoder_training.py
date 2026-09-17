"""Training utilities for matched decoders on frozen ATCNet sequences."""

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

from dpc_snn.models.v8_sequence_decoder import (
    V8ATCSequenceDecoder,
    build_v8_sequence_decoder,
)
from dpc_snn.utils.metrics import classification_metrics


@dataclass
class V8SequenceDecoderFit:
    model: V8ATCSequenceDecoder
    history: list[dict[str, float | int]]
    best_epoch: int
    best_metric: float
    best_accuracy: float
    best_state: dict[str, torch.Tensor]
    last_state: dict[str, torch.Tensor]
    optimizer_steps: int
    elapsed_seconds: float


def seed_v8_sequence_decoder(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _loader(
    sequence: np.ndarray,
    labels: np.ndarray,
    teacher_logits: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    x = np.ascontiguousarray(sequence, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    teacher = np.ascontiguousarray(teacher_logits, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (18, 32):
        raise ValueError("frozen ATC sequence must have shape [trials, 18, 32]")
    if (
        y.shape != (x.shape[0],)
        or teacher.ndim != 2
        or teacher.shape[0] != x.shape[0]
        or teacher.shape[1] < 2
        or np.any(y < 0)
        or np.any(y >= teacher.shape[1])
    ):
        raise ValueError("sequence, label, and teacher arrays are not aligned")
    if not np.isfinite(x).all() or not np.isfinite(teacher).all():
        raise FloatingPointError("sequence decoder data contain non-finite values")
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(teacher)),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def predict_v8_sequence_decoder(
    model: V8ATCSequenceDecoder,
    sequence: np.ndarray,
    labels: np.ndarray,
    teacher_logits: np.ndarray,
    *,
    device: str,
    batch_size: int = 64,
) -> dict[str, Any]:
    model.eval().to(device)
    loader = _loader(
        sequence,
        labels,
        teacher_logits,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    )
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    rates: list[float] = []
    for batch_x, batch_y, _ in loader:
        output = model(batch_x.to(device, non_blocking=True))
        if not bool(torch.isfinite(output.logits).all()):
            raise FloatingPointError("sequence decoder produced non-finite logits")
        logits.append(output.logits.float().cpu().numpy())
        targets.append(batch_y.numpy())
        if output.binary_spikes:
            rates.append(
                float(
                    torch.stack([spikes.float().mean() for spikes in output.binary_spikes])
                    .mean()
                    .cpu()
                )
            )
    all_logits = np.concatenate(logits)
    all_labels = np.concatenate(targets)
    prediction = all_logits.argmax(axis=1)
    metrics = classification_metrics(
        all_labels, prediction, n_classes=int(all_logits.shape[1])
    )
    return {
        **metrics,
        "logits": all_logits,
        "labels": all_labels,
        "mean_firing_rate": float(np.mean(rates)) if rates else 0.0,
    }


def fit_v8_sequence_decoder(
    variant: str,
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    teacher_train: np.ndarray,
    x_validation: np.ndarray | None,
    y_validation: np.ndarray | None,
    teacher_validation: np.ndarray | None,
    device: str,
    seed: int,
    epochs: int = 200,
    patience: int = 40,
    fixed_epoch: int | None = None,
    scheduler_epochs: int | None = None,
    batch_size: int = 48,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    distillation_weight: float = 0.35,
    distillation_temperature: float = 2.0,
    firing_rate_weight: float = 0.02,
    model_kwargs: dict[str, Any] | None = None,
    run_label: str = "",
) -> V8SequenceDecoderFit:
    if not 0.0 <= distillation_weight <= 1.0:
        raise ValueError("distillation_weight must lie in [0, 1]")
    if distillation_temperature <= 0.0 or firing_rate_weight < 0.0:
        raise ValueError("distillation temperature and firing-rate weight are invalid")
    if fixed_epoch is not None and int(fixed_epoch) < 1:
        raise ValueError("fixed_epoch must be positive")
    has_validation = x_validation is not None
    if has_validation != (y_validation is not None and teacher_validation is not None):
        raise ValueError("validation sequence, labels, and teacher logits must be supplied together")
    if not has_validation and fixed_epoch is None:
        raise ValueError("training without validation requires a fixed epoch")

    seed_v8_sequence_decoder(seed)
    model = build_v8_sequence_decoder(variant, **(model_kwargs or {})).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
        betas=(0.9, 0.999),
    )
    horizon = int(scheduler_epochs or epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(horizon, 1), eta_min=float(learning_rate) * 0.05
    )
    train_loader = _loader(
        x_train,
        y_train,
        teacher_train,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
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
    temperature = float(distillation_temperature)

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch_x, batch_y, batch_teacher in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_teacher = batch_teacher.to(device, non_blocking=True)
            output = model(batch_x)
            hard_loss = F.cross_entropy(output.logits, batch_y)
            soft_loss = F.kl_div(
                F.log_softmax(output.logits / temperature, dim=1),
                F.softmax(batch_teacher / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature * temperature)
            loss = (
                (1.0 - float(distillation_weight)) * hard_loss
                + float(distillation_weight) * soft_loss
                + float(firing_rate_weight) * output.firing_rate_loss
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("sequence decoder loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer_steps += 1
            total_loss += float(loss.detach()) * batch_x.shape[0]
            total_examples += batch_x.shape[0]
        scheduler.step()

        row: dict[str, float | int] = {
            "epoch": epoch,
            "loss": total_loss / max(total_examples, 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if has_validation:
            evaluation = predict_v8_sequence_decoder(
                model,
                x_validation,
                y_validation,
                teacher_validation,
                device=device,
                batch_size=batch_size,
            )
            metric = float(evaluation["kappa"])
            accuracy = float(evaluation["accuracy"])
            row.update(
                {
                    "validation_kappa": metric,
                    "validation_accuracy": accuracy,
                    "validation_firing_rate": float(evaluation["mean_firing_rate"]),
                }
            )
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
            validation_value = row.get("validation_kappa", float("nan"))
            print(
                "__V8_DECODER_PROGRESS__ "
                f"run={run_label} epoch={epoch}/{maximum_epochs} "
                f"loss={row['loss']:.6f} val_kappa={validation_value:.6f}",
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
    return V8SequenceDecoderFit(
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
