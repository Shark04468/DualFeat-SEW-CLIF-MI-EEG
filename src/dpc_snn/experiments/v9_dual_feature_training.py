"""Leakage-safe training for dual-feature matched ANN/SNN students."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import random
import time
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dpc_snn.models.v9_dual_feature_student import (
    V9DualFeatureStudent,
    build_v9_dual_feature_student,
)
from dpc_snn.utils.metrics import classification_metrics


@dataclass(frozen=True)
class V9VariantSpec:
    model_variant: str
    objective: Literal["ce", "kd"]


V9_DUAL_FEATURE_EXPERIMENT_VARIANTS: dict[str, V9VariantSpec] = {
    "ann_plain_ce": V9VariantSpec("ann_plain", "ce"),
    "plif_plain_ce": V9VariantSpec("plif_plain", "ce"),
    "clif_plain_ce": V9VariantSpec("clif_plain", "ce"),
    "ann_sew_ce": V9VariantSpec("ann_sew", "ce"),
    "sew_clif_ce": V9VariantSpec("sew_clif", "ce"),
    "ann_sew_kd": V9VariantSpec("ann_sew", "kd"),
    "sew_clif_kd": V9VariantSpec("sew_clif", "kd"),
}


@dataclass(frozen=True)
class FrozenFeatureStandardizer:
    atc_mean: np.ndarray
    atc_scale: np.ndarray
    fbc_mean: np.ndarray
    fbc_scale: np.ndarray

    def transform(
        self, atc_sequence: np.ndarray, fbc_sequence: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        atc, fbc = _validated_features(atc_sequence, fbc_sequence)
        transformed_atc = (atc - self.atc_mean) / self.atc_scale
        transformed_fbc = (fbc - self.fbc_mean) / self.fbc_scale
        if not np.isfinite(transformed_atc).all() or not np.isfinite(transformed_fbc).all():
            raise FloatingPointError("standardized frozen features contain non-finite values")
        return (
            np.ascontiguousarray(transformed_atc, dtype=np.float32),
            np.ascontiguousarray(transformed_fbc, dtype=np.float32),
        )

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "atc_mean": self.atc_mean.reshape(-1).tolist(),
            "atc_scale": self.atc_scale.reshape(-1).tolist(),
            "fbc_mean": self.fbc_mean.reshape(-1).tolist(),
            "fbc_scale": self.fbc_scale.reshape(-1).tolist(),
        }


@dataclass
class V9DualFeatureFit:
    model: V9DualFeatureStudent
    history: list[dict[str, float | int]]
    best_epoch: int
    best_metric: float
    best_accuracy: float
    best_state: dict[str, torch.Tensor]
    last_state: dict[str, torch.Tensor]
    optimizer_steps: int
    elapsed_seconds: float


def seed_v9_dual_feature(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _validated_features(
    atc_sequence: np.ndarray, fbc_sequence: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    atc = np.ascontiguousarray(atc_sequence, dtype=np.float32)
    fbc = np.ascontiguousarray(fbc_sequence, dtype=np.float32)
    if atc.ndim != 3 or atc.shape[1:] != (18, 32):
        raise ValueError("frozen ATC sequence must have shape [trials, 18, 32]")
    if fbc.ndim != 3 or fbc.shape[1:] != (4, 288):
        raise ValueError("frozen FBC sequence must have shape [trials, 4, 288]")
    if atc.shape[0] != fbc.shape[0]:
        raise ValueError("ATC and FBC feature arrays are not aligned")
    if not np.isfinite(atc).all() or not np.isfinite(fbc).all():
        raise FloatingPointError("frozen feature arrays contain non-finite values")
    return atc, fbc


def fit_feature_standardizer(
    atc_sequence: np.ndarray,
    fbc_sequence: np.ndarray,
    *,
    minimum_scale: float = 1e-5,
) -> FrozenFeatureStandardizer:
    if minimum_scale <= 0.0:
        raise ValueError("minimum feature scale must be positive")
    atc, fbc = _validated_features(atc_sequence, fbc_sequence)

    def moments(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = array.mean(axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
        scale = array.std(axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
        return mean, np.maximum(scale, np.float32(minimum_scale))

    atc_mean, atc_scale = moments(atc)
    fbc_mean, fbc_scale = moments(fbc)
    return FrozenFeatureStandardizer(atc_mean, atc_scale, fbc_mean, fbc_scale)


def equal_probability_teacher(
    atc_logits: np.ndarray, fbc_logits: np.ndarray
) -> np.ndarray:
    atc = torch.as_tensor(np.asarray(atc_logits, dtype=np.float32))
    fbc = torch.as_tensor(np.asarray(fbc_logits, dtype=np.float32))
    if atc.ndim != 2 or atc.shape != fbc.shape or atc.shape[1] < 2:
        raise ValueError("ATC and FBC teacher logits are not aligned")
    if not bool(torch.isfinite(atc).all()) or not bool(torch.isfinite(fbc).all()):
        raise FloatingPointError("teacher logits contain non-finite values")
    probability = 0.5 * torch.softmax(atc, dim=1) + 0.5 * torch.softmax(fbc, dim=1)
    return probability.clamp_min(1e-8).log().numpy().astype(np.float32)


def _state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _loader(
    atc_sequence: np.ndarray,
    fbc_sequence: np.ndarray,
    labels: np.ndarray,
    teacher_log_probability: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    atc, fbc = _validated_features(atc_sequence, fbc_sequence)
    y = np.asarray(labels, dtype=np.int64)
    teacher = np.ascontiguousarray(teacher_log_probability, dtype=np.float32)
    if (
        y.shape != (atc.shape[0],)
        or teacher.ndim != 2
        or teacher.shape[0] != atc.shape[0]
        or teacher.shape[1] < 2
        or np.any(y < 0)
        or np.any(y >= teacher.shape[1])
    ):
        raise ValueError("frozen features, labels, and teacher targets are not aligned")
    if not np.isfinite(teacher).all():
        raise FloatingPointError("teacher targets contain non-finite values")
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(
            torch.from_numpy(atc),
            torch.from_numpy(fbc),
            torch.from_numpy(y),
            torch.from_numpy(teacher),
        ),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def predict_v9_dual_feature(
    model: V9DualFeatureStudent,
    atc_sequence: np.ndarray,
    fbc_sequence: np.ndarray,
    labels: np.ndarray,
    teacher_log_probability: np.ndarray,
    *,
    device: str,
    batch_size: int = 64,
) -> dict[str, Any]:
    model.eval().to(device)
    loader = _loader(
        atc_sequence,
        fbc_sequence,
        labels,
        teacher_log_probability,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    )
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    rates: list[float] = []
    for batch_atc, batch_fbc, batch_y, _ in loader:
        output = model(
            batch_atc.to(device, non_blocking=True),
            batch_fbc.to(device, non_blocking=True),
        )
        if not bool(torch.isfinite(output.logits).all()):
            raise FloatingPointError("dual-feature student produced non-finite logits")
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
    metrics = classification_metrics(
        all_labels,
        all_logits.argmax(axis=1),
        n_classes=int(model.n_classes),
    )
    return {
        **metrics,
        "logits": all_logits,
        "labels": all_labels,
        "mean_firing_rate": float(np.mean(rates)) if rates else 0.0,
    }


def fit_v9_dual_feature(
    variant: str,
    *,
    atc_train: np.ndarray,
    fbc_train: np.ndarray,
    y_train: np.ndarray,
    teacher_train: np.ndarray,
    atc_validation: np.ndarray | None,
    fbc_validation: np.ndarray | None,
    y_validation: np.ndarray | None,
    teacher_validation: np.ndarray | None,
    device: str,
    seed: int,
    epochs: int = 160,
    patience: int = 30,
    fixed_epoch: int | None = None,
    scheduler_epochs: int | None = None,
    batch_size: int = 48,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    distillation_weight: float = 0.35,
    distillation_temperature: float = 2.0,
    firing_rate_weight: float = 0.01,
    model_kwargs: dict[str, Any] | None = None,
    run_label: str = "",
) -> V9DualFeatureFit:
    try:
        specification = V9_DUAL_FEATURE_EXPERIMENT_VARIANTS[str(variant)]
    except KeyError as exc:
        raise KeyError(f"unknown V9 dual-feature experiment variant: {variant}") from exc
    if not 0.0 <= distillation_weight <= 1.0:
        raise ValueError("distillation_weight must lie in [0, 1]")
    if distillation_temperature <= 0.0 or firing_rate_weight < 0.0:
        raise ValueError("distillation temperature and firing-rate weight are invalid")
    if fixed_epoch is not None and int(fixed_epoch) < 1:
        raise ValueError("fixed_epoch must be positive")
    validation_values = (
        atc_validation,
        fbc_validation,
        y_validation,
        teacher_validation,
    )
    has_validation = all(value is not None for value in validation_values)
    if has_validation != any(value is not None for value in validation_values):
        raise ValueError("all validation arrays must be supplied together")
    if not has_validation and fixed_epoch is None:
        raise ValueError("training without validation requires a fixed epoch")

    seed_v9_dual_feature(seed)
    model = build_v9_dual_feature_student(
        specification.model_variant, **(model_kwargs or {})
    ).to(device)
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
        atc_train,
        fbc_train,
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
    kd_weight = float(distillation_weight) if specification.objective == "kd" else 0.0

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        total_loss = 0.0
        total_hard = 0.0
        total_soft = 0.0
        total_examples = 0
        for batch_atc, batch_fbc, batch_y, batch_teacher in train_loader:
            batch_atc = batch_atc.to(device, non_blocking=True)
            batch_fbc = batch_fbc.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_teacher = batch_teacher.to(device, non_blocking=True)
            output = model(batch_atc, batch_fbc)
            hard_loss = F.cross_entropy(output.logits, batch_y)
            soft_loss = F.kl_div(
                F.log_softmax(output.logits / temperature, dim=1),
                F.softmax(batch_teacher / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature * temperature)
            loss = (
                (1.0 - kd_weight) * hard_loss
                + kd_weight * soft_loss
                + float(firing_rate_weight) * output.firing_rate_loss
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("dual-feature student loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer_steps += 1
            batch_examples = batch_atc.shape[0]
            total_loss += float(loss.detach()) * batch_examples
            total_hard += float(hard_loss.detach()) * batch_examples
            total_soft += float(soft_loss.detach()) * batch_examples
            total_examples += batch_examples
        scheduler.step()

        row: dict[str, float | int] = {
            "epoch": epoch,
            "loss": total_loss / max(total_examples, 1),
            "hard_loss": total_hard / max(total_examples, 1),
            "soft_loss": total_soft / max(total_examples, 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if has_validation:
            evaluation = predict_v9_dual_feature(
                model,
                atc_validation,
                fbc_validation,
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
                "__V9_DUAL_PROGRESS__ "
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
    return V9DualFeatureFit(
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
