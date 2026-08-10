"""Two-stage training for frozen-shared V14 residual SNN experts."""

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

from dpc_snn.models.v14_shared_residual_student import (
    V14ResidualOutput,
    V14SharedResidualStudent,
    build_v14_student,
)
from dpc_snn.utils.metrics import classification_metrics


@dataclass
class V14Fit:
    model: V14SharedResidualStudent
    history: list[dict[str, Any]]
    best_epoch: int
    best_metric: float
    best_accuracy: float
    best_state: dict[str, torch.Tensor]
    last_state: dict[str, torch.Tensor]
    optimizer_steps: int
    elapsed_seconds: float


def seed_v14(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _loader(
    atc: np.ndarray,
    fbc: np.ndarray,
    labels: np.ndarray,
    equal_teacher: np.ndarray,
    atc_teacher: np.ndarray,
    fbc_teacher: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    arrays = [
        np.ascontiguousarray(atc, dtype=np.float32),
        np.ascontiguousarray(fbc, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.ascontiguousarray(equal_teacher, dtype=np.float32),
        np.ascontiguousarray(atc_teacher, dtype=np.float32),
        np.ascontiguousarray(fbc_teacher, dtype=np.float32),
    ]
    samples = arrays[0].shape[0]
    if arrays[0].shape != (samples, 18, 32) or arrays[1].shape != (
        samples,
        4,
        288,
    ):
        raise ValueError("V14 feature shapes are invalid")
    if arrays[2].shape != (samples,) or any(
        value.shape != (samples, 4) for value in arrays[3:]
    ):
        raise ValueError("V14 labels or teacher logits are misaligned")
    if any(not np.isfinite(value).all() for value in arrays if value.dtype.kind == "f"):
        raise FloatingPointError("V14 inputs contain non-finite values")
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(*(torch.from_numpy(value) for value in arrays)),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _center(logits: torch.Tensor) -> torch.Tensor:
    return logits - logits.mean(dim=-1, keepdim=True)


def _residual_loss(
    model: V14SharedResidualStudent,
    output: V14ResidualOutput,
    equal_teacher: torch.Tensor,
    atc_teacher: torch.Tensor,
    fbc_teacher: torch.Tensor,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    if model.atc_expert is not None:
        target = _center(atc_teacher - output.shared_logits)
        losses.append(F.smooth_l1_loss(output.atc_residual_logits, target))
    if model.fbc_expert is not None:
        target = _center(fbc_teacher - output.shared_logits)
        losses.append(F.smooth_l1_loss(output.fbc_residual_logits, target))
    if model.generic_expert is not None:
        target = _center(equal_teacher - output.shared_logits)
        losses.append(F.smooth_l1_loss(output.generic_residual_logits, target))
    if not losses:
        return output.logits.new_zeros(())
    return torch.stack(losses).mean()


def _kd(student: torch.Tensor, teacher: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(student / temperature, dim=-1),
        F.softmax(teacher / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature * temperature)


@torch.no_grad()
def predict_v14(
    model: V14SharedResidualStudent,
    atc: np.ndarray,
    fbc: np.ndarray,
    labels: np.ndarray,
    equal_teacher: np.ndarray,
    atc_teacher: np.ndarray,
    fbc_teacher: np.ndarray,
    *,
    device: str,
    batch_size: int = 64,
) -> dict[str, Any]:
    model.eval().to(device)
    loader = _loader(
        atc,
        fbc,
        labels,
        equal_teacher,
        atc_teacher,
        fbc_teacher,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    )
    collected: dict[str, list[np.ndarray]] = {
        "logits": [],
        "shared_logits": [],
        "atc_residual_logits": [],
        "fbc_residual_logits": [],
        "generic_residual_logits": [],
    }
    targets: list[np.ndarray] = []
    rates: list[float] = []
    for batch in loader:
        batch_atc, batch_fbc, batch_labels, *_ = batch
        output = model(
            batch_atc.to(device, non_blocking=True),
            batch_fbc.to(device, non_blocking=True),
        )
        for name in collected:
            value = getattr(output, name)
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"V14 produced non-finite {name}")
            collected[name].append(value.float().cpu().numpy())
        targets.append(batch_labels.numpy())
        if output.binary_spikes:
            rates.append(
                float(
                    torch.stack(
                        [spikes.float().mean() for spikes in output.binary_spikes]
                    )
                    .mean()
                    .cpu()
                )
            )
    values = {name: np.concatenate(parts) for name, parts in collected.items()}
    all_labels = np.concatenate(targets)
    metrics = classification_metrics(
        all_labels, values["logits"].argmax(axis=1), n_classes=4
    )
    gates = {
        name: value.detach().cpu().numpy()
        for name, value in model.gate_values().items()
    }
    return {
        **metrics,
        **values,
        "labels": all_labels,
        "mean_firing_rate": float(np.mean(rates)) if rates else 0.0,
        "gates": gates,
    }


def fit_v14(
    variant: str,
    shared_state: dict[str, torch.Tensor],
    *,
    atc_train: np.ndarray,
    fbc_train: np.ndarray,
    y_train: np.ndarray,
    equal_teacher_train: np.ndarray,
    atc_teacher_train: np.ndarray,
    fbc_teacher_train: np.ndarray,
    atc_validation: np.ndarray | None,
    fbc_validation: np.ndarray | None,
    y_validation: np.ndarray | None,
    equal_teacher_validation: np.ndarray | None,
    atc_teacher_validation: np.ndarray | None,
    fbc_teacher_validation: np.ndarray | None,
    device: str,
    seed: int,
    epochs: int = 100,
    patience: int = 20,
    fixed_epoch: int | None = None,
    pretrain_epochs: int = 10,
    batch_size: int = 48,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    distillation_temperature: float = 2.0,
    firing_rate_weight: float = 0.01,
    run_label: str = "",
) -> V14Fit:
    if variant == "r0_shared_replay":
        raise ValueError("R0 is a replay control and must not be trained")
    if fixed_epoch is not None and fixed_epoch < 1:
        raise ValueError("fixed epoch must be positive")
    if pretrain_epochs < 0 or distillation_temperature <= 0.0:
        raise ValueError("invalid V14 training schedule")
    validation_values = (
        atc_validation,
        fbc_validation,
        y_validation,
        equal_teacher_validation,
        atc_teacher_validation,
        fbc_teacher_validation,
    )
    has_validation = all(value is not None for value in validation_values)
    if has_validation != any(value is not None for value in validation_values):
        raise ValueError("all V14 validation arrays must be supplied together")
    if not has_validation and fixed_epoch is None:
        raise ValueError("V14 outer training requires a fixed epoch")

    seed_v14(seed)
    model = build_v14_student(variant)
    model.load_shared_state(shared_state)
    model.to(device)
    parameters = [value for value in model.parameters() if value.requires_grad]
    if not parameters:
        raise RuntimeError("V14 trainable variant has no expert parameters")
    train_loader = _loader(
        atc_train,
        fbc_train,
        y_train,
        equal_teacher_train,
        atc_teacher_train,
        fbc_teacher_train,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    history: list[dict[str, Any]] = []
    optimizer_steps = 0
    started = time.perf_counter()

    if pretrain_epochs:
        optimizer = torch.optim.AdamW(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=pretrain_epochs, eta_min=learning_rate * 0.1
        )
        for epoch in range(1, pretrain_epochs + 1):
            model.train()
            total = 0.0
            examples = 0
            for batch in train_loader:
                batch_atc, batch_fbc, _, equal, atc_teacher, fbc_teacher = (
                    value.to(device, non_blocking=True) for value in batch
                )
                output = model(batch_atc, batch_fbc)
                residual = _residual_loss(
                    model, output, equal, atc_teacher, fbc_teacher
                )
                loss = residual + firing_rate_weight * output.firing_rate_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                optimizer.step()
                optimizer_steps += 1
                count = batch_atc.shape[0]
                total += float(loss.detach()) * count
                examples += count
            scheduler.step()
            history.append(
                {
                    "phase": "residual_pretrain",
                    "epoch": epoch,
                    "loss": total / max(examples, 1),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )

    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay
    )
    maximum_epochs = int(fixed_epoch or epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(epochs), 1), eta_min=learning_rate * 0.05
    )
    best_metric = -math.inf
    best_accuracy = 0.0
    best_epoch = 0
    best_state = _state_cpu(model)
    stale = 0
    temperature = float(distillation_temperature)
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        totals = {"loss": 0.0, "hard": 0.0, "kd": 0.0, "residual": 0.0}
        examples = 0
        for batch in train_loader:
            batch_atc, batch_fbc, batch_y, equal, atc_teacher, fbc_teacher = (
                value.to(device, non_blocking=True) for value in batch
            )
            output = model(batch_atc, batch_fbc)
            hard = F.cross_entropy(output.logits, batch_y)
            soft = _kd(output.logits, equal, temperature)
            residual = _residual_loss(model, output, equal, atc_teacher, fbc_teacher)
            loss = (
                0.50 * hard
                + 0.35 * soft
                + 0.15 * residual
                + firing_rate_weight * output.firing_rate_loss
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("V14 loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            optimizer_steps += 1
            count = batch_atc.shape[0]
            totals["loss"] += float(loss.detach()) * count
            totals["hard"] += float(hard.detach()) * count
            totals["kd"] += float(soft.detach()) * count
            totals["residual"] += float(residual.detach()) * count
            examples += count
        scheduler.step()
        row: dict[str, Any] = {
            "phase": "residual_classification",
            "epoch": epoch,
            **{name: value / max(examples, 1) for name, value in totals.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if has_validation:
            evaluation = predict_v14(
                model,
                atc_validation,
                fbc_validation,
                y_validation,
                equal_teacher_validation,
                atc_teacher_validation,
                fbc_teacher_validation,
                device=device,
                batch_size=batch_size,
            )
            metric = float(evaluation["kappa"])
            accuracy = float(evaluation["accuracy"])
            row.update(
                {"validation_kappa": metric, "validation_accuracy": accuracy}
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
            print(
                "__V14_PROGRESS__ "
                f"run={run_label} epoch={epoch}/{maximum_epochs} "
                f"loss={row['loss']:.6f} "
                f"val_kappa={row.get('validation_kappa', float('nan')):.6f}",
                flush=True,
            )
        if has_validation and epoch >= 10 and stale >= patience:
            break

    last_state = _state_cpu(model)
    if has_validation:
        model.load_state_dict(deepcopy(best_state), strict=True)
    else:
        best_metric = float("nan")
        best_accuracy = float("nan")
    return V14Fit(
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
