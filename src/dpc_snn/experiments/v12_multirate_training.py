"""Leakage-safe training for V12 multi-rate dual-view students."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import random
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dpc_snn.models.v12_multirate_student import (
    V12_MODEL_ARCHITECTURES,
    V12StudentOutput,
    build_v12_student,
)
from dpc_snn.models.v62_snn_decoder import DecoderKind
from dpc_snn.utils.metrics import classification_metrics


@dataclass(frozen=True)
class V12Objective:
    final_kd_weight: float
    branch_kd_weight: float
    temporal_kd_weight: float

    def __post_init__(self) -> None:
        weights = (
            self.final_kd_weight,
            self.branch_kd_weight,
            self.temporal_kd_weight,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("V12 objective weights must be finite and non-negative")
        if sum(weights) > 1.0 + 1e-12:
            raise ValueError("V12 supervised loss weights exceed one")

    @property
    def hard_weight(self) -> float:
        return 1.0 - (
            self.final_kd_weight
            + self.branch_kd_weight
            + self.temporal_kd_weight
        )


V12_OBJECTIVES: dict[str, V12Objective] = {
    "interpolated_branch_kd": V12Objective(0.35, 0.15, 0.0),
    "interpolated_static_gate_kd": V12Objective(0.35, 0.15, 0.0),
    "dual_rate_branch_kd": V12Objective(0.35, 0.15, 0.0),
    "dual_rate_branch_temporal": V12Objective(0.35, 0.15, 0.05),
}


V12_E12_OBJECTIVES: dict[str, V12Objective] = {
    "o0_current": V12Objective(0.35, 0.15, 0.0),
    "o1_fused_070": V12Objective(0.70, 0.0, 0.0),
    "o2_fused_080": V12Objective(0.80, 0.0, 0.0),
    "o3_light_branch": V12Objective(0.60, 0.10, 0.0),
}


@dataclass
class V12Fit:
    model: nn.Module
    history: list[dict[str, float | int]]
    best_epoch: int
    best_metric: float
    best_accuracy: float
    best_state: dict[str, torch.Tensor]
    last_state: dict[str, torch.Tensor]
    optimizer_steps: int
    elapsed_seconds: float


def seed_v12(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _features(
    atc_sequence: np.ndarray, fbc_sequence: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    atc = np.ascontiguousarray(atc_sequence, dtype=np.float32)
    fbc = np.ascontiguousarray(fbc_sequence, dtype=np.float32)
    if atc.ndim != 3 or atc.shape[1:] != (18, 32):
        raise ValueError("ATC sequence must have shape [N, 18, 32]")
    if fbc.ndim != 3 or fbc.shape[1:] != (4, 288):
        raise ValueError("FBC sequence must have shape [N, 4, 288]")
    if atc.shape[0] != fbc.shape[0] or not np.isfinite(atc).all() or not np.isfinite(
        fbc
    ).all():
        raise ValueError("V12 feature arrays are invalid or misaligned")
    return atc, fbc


def _loader(
    atc_sequence: np.ndarray,
    fbc_sequence: np.ndarray,
    labels: np.ndarray,
    equal_teacher: np.ndarray,
    atc_teacher: np.ndarray,
    fbc_teacher: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    atc, fbc = _features(atc_sequence, fbc_sequence)
    y = np.asarray(labels, dtype=np.int64)
    teachers = [
        np.ascontiguousarray(value, dtype=np.float32)
        for value in (equal_teacher, atc_teacher, fbc_teacher)
    ]
    if y.shape != (atc.shape[0],) or any(
        value.shape != (atc.shape[0], 4) or not np.isfinite(value).all()
        for value in teachers
    ):
        raise ValueError("V12 labels or teacher logits are misaligned")
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(
            torch.from_numpy(atc),
            torch.from_numpy(fbc),
            torch.from_numpy(y),
            *(torch.from_numpy(value) for value in teachers),
        ),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _kd(student: torch.Tensor, teacher: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(student / temperature, dim=-1),
        F.softmax(teacher / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature * temperature)


def _loss(
    output: V12StudentOutput,
    labels: torch.Tensor,
    equal_teacher: torch.Tensor,
    atc_teacher: torch.Tensor,
    fbc_teacher: torch.Tensor,
    *,
    objective: V12Objective,
    temperature: float,
    firing_rate_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    hard = F.cross_entropy(output.logits, labels)
    final_kd = _kd(output.logits, equal_teacher, temperature)
    branch_kd = 0.5 * (
        _kd(output.atc_logits, atc_teacher, temperature)
        + _kd(output.fbc_logits, fbc_teacher, temperature)
    )
    temporal_kd = output.logits.new_zeros(())
    if objective.temporal_kd_weight:
        if output.endpoint_logits.shape[1] < 4:
            raise RuntimeError("temporal KD requires four causal endpoint logits")
        prefix = output.endpoint_logits[:, 1:-1]
        target = equal_teacher[:, None, :].expand_as(prefix)
        temporal_kd = _kd(
            prefix.reshape(-1, prefix.shape[-1]),
            target.reshape(-1, target.shape[-1]),
            temperature,
        )
    total = (
        objective.hard_weight * hard
        + objective.final_kd_weight * final_kd
        + objective.branch_kd_weight * branch_kd
        + objective.temporal_kd_weight * temporal_kd
        + float(firing_rate_weight) * output.firing_rate_loss
    )
    return total, {
        "hard_loss": hard,
        "final_kd_loss": final_kd,
        "branch_kd_loss": branch_kd,
        "temporal_kd_loss": temporal_kd,
        "firing_rate_loss": output.firing_rate_loss,
    }


@torch.no_grad()
def predict_v12(
    model: nn.Module,
    atc_sequence: np.ndarray,
    fbc_sequence: np.ndarray,
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
        atc_sequence,
        fbc_sequence,
        labels,
        equal_teacher,
        atc_teacher,
        fbc_teacher,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    )
    logits: list[np.ndarray] = []
    endpoint_logits: list[np.ndarray] = []
    atc_logits: list[np.ndarray] = []
    fbc_logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    rates: list[float] = []
    for batch_atc, batch_fbc, batch_y, *_ in loader:
        output = model(
            batch_atc.to(device, non_blocking=True),
            batch_fbc.to(device, non_blocking=True),
        )
        if not bool(torch.isfinite(output.logits).all()):
            raise FloatingPointError("V12 student produced non-finite logits")
        logits.append(output.logits.float().cpu().numpy())
        endpoint_logits.append(output.endpoint_logits.float().cpu().numpy())
        atc_logits.append(output.atc_logits.float().cpu().numpy())
        fbc_logits.append(output.fbc_logits.float().cpu().numpy())
        targets.append(batch_y.numpy())
        if output.binary_spikes:
            rates.append(
                float(
                    torch.stack([value.float().mean() for value in output.binary_spikes])
                    .mean()
                    .cpu()
                )
            )
    all_logits = np.concatenate(logits)
    all_endpoints = np.concatenate(endpoint_logits)
    all_atc_logits = np.concatenate(atc_logits)
    all_fbc_logits = np.concatenate(fbc_logits)
    all_labels = np.concatenate(targets)
    metrics = classification_metrics(all_labels, all_logits.argmax(axis=1), n_classes=4)
    return {
        **metrics,
        "logits": all_logits,
        "endpoint_logits": all_endpoints,
        "atc_logits": all_atc_logits,
        "fbc_logits": all_fbc_logits,
        "labels": all_labels,
        "mean_firing_rate": float(np.mean(rates)) if rates else 0.0,
    }


def fit_v12(
    variant: str,
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
    epochs: int = 160,
    patience: int = 30,
    fixed_epoch: int | None = None,
    scheduler_epochs: int | None = None,
    batch_size: int = 48,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    distillation_temperature: float = 2.0,
    firing_rate_weight: float = 0.01,
    decoder_kind: DecoderKind = "clif",
    objective_override: V12Objective | None = None,
    model_builder: Callable[[DecoderKind], nn.Module] | None = None,
    run_label: str = "",
) -> V12Fit:
    if variant not in V12_MODEL_ARCHITECTURES or variant not in V12_OBJECTIVES:
        raise KeyError(f"unknown V12 experiment variant: {variant}")
    if distillation_temperature <= 0.0 or firing_rate_weight < 0.0:
        raise ValueError("invalid V12 objective hyperparameters")
    if fixed_epoch is not None and int(fixed_epoch) < 1:
        raise ValueError("fixed_epoch must be positive")
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
        raise ValueError("all V12 validation arrays must be supplied together")
    if not has_validation and fixed_epoch is None:
        raise ValueError("V12 training without validation requires fixed_epoch")

    seed_v12(seed)
    model = (
        model_builder(decoder_kind)
        if model_builder is not None
        else build_v12_student(variant, decoder_kind=decoder_kind)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(scheduler_epochs or epochs), 1),
        eta_min=float(learning_rate) * 0.05,
    )
    loader = _loader(
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
    maximum_epochs = int(fixed_epoch or epochs)
    objective = objective_override or V12_OBJECTIVES[variant]
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
        totals = {
            "loss": 0.0,
            "hard_loss": 0.0,
            "final_kd_loss": 0.0,
            "branch_kd_loss": 0.0,
            "temporal_kd_loss": 0.0,
            "firing_rate_loss": 0.0,
        }
        examples = 0
        for batch in loader:
            batch_atc, batch_fbc, batch_y, equal, atc_teacher, fbc_teacher = (
                value.to(device, non_blocking=True) for value in batch
            )
            output = model(batch_atc, batch_fbc)
            loss, components = _loss(
                output,
                batch_y,
                equal,
                atc_teacher,
                fbc_teacher,
                objective=objective,
                temperature=float(distillation_temperature),
                firing_rate_weight=float(firing_rate_weight),
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("V12 loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer_steps += 1
            count = batch_atc.shape[0]
            totals["loss"] += float(loss.detach()) * count
            for name, value in components.items():
                totals[name] += float(value.detach()) * count
            examples += count
        scheduler.step()
        row: dict[str, float | int] = {
            "epoch": epoch,
            **{name: value / max(examples, 1) for name, value in totals.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if has_validation:
            evaluation = predict_v12(
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
            print(
                "__V12_PROGRESS__ "
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
    return V12Fit(
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
