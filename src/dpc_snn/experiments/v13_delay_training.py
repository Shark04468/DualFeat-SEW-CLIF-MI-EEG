"""Training and counterfactual evaluation for the V13 residual delay."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dpc_snn.experiments.v12_multirate_training import seed_v12
from dpc_snn.models.v13_delay_residual import DelayControl, V13DelayResidualStudent
from dpc_snn.utils.metrics import classification_metrics


@dataclass
class V13Fit:
    model: V13DelayResidualStudent
    history: list[dict[str, float | int]]
    best_epoch: int
    best_metric: float
    best_accuracy: float
    best_state: dict[str, torch.Tensor]
    last_state: dict[str, torch.Tensor]
    optimizer_steps: int
    elapsed_seconds: float


def _loader(
    atc: np.ndarray,
    fbc: np.ndarray,
    labels: np.ndarray,
    teacher: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    atc_value = np.ascontiguousarray(atc, dtype=np.float32)
    fbc_value = np.ascontiguousarray(fbc, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    target = np.ascontiguousarray(teacher, dtype=np.float32)
    if (
        atc_value.ndim != 3
        or atc_value.shape[1:] != (18, 32)
        or fbc_value.shape != (atc_value.shape[0], 4, 288)
        or y.shape != (atc_value.shape[0],)
        or target.shape != (atc_value.shape[0], 4)
    ):
        raise ValueError("V13 arrays are misaligned")
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(
            torch.from_numpy(atc_value),
            torch.from_numpy(fbc_value),
            torch.from_numpy(y),
            torch.from_numpy(target),
        ),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


@torch.no_grad()
def predict_v13(
    model: V13DelayResidualStudent,
    atc: np.ndarray,
    fbc: np.ndarray,
    labels: np.ndarray,
    teacher: np.ndarray,
    *,
    control: DelayControl,
    device: str,
    batch_size: int = 64,
) -> dict[str, Any]:
    model.eval().to(device)
    loader = _loader(
        atc,
        fbc,
        labels,
        teacher,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    )
    logits: list[np.ndarray] = []
    endpoints: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    rates: list[float] = []
    nonzero: list[float] = []
    residual_rms: list[float] = []
    route: np.ndarray | None = None
    posterior: np.ndarray | None = None
    for batch_atc, batch_fbc, batch_y, _ in loader:
        output = model(
            batch_atc.to(device, non_blocking=True),
            batch_fbc.to(device, non_blocking=True),
            control=control,
        )
        logits.append(output.logits.float().cpu().numpy())
        endpoints.append(output.endpoint_logits.float().cpu().numpy())
        targets.append(batch_y.numpy())
        nonzero.append(float(output.nonzero_delay_mass.cpu()))
        residual_rms.append(float(output.delay_residual_rms.cpu()))
        route = output.route_probability.float().cpu().numpy()
        posterior = output.lag_posterior.float().cpu().numpy()
        if output.binary_spikes:
            rates.append(
                float(
                    torch.stack([value.float().mean() for value in output.binary_spikes])
                    .mean()
                    .cpu()
                )
            )
    all_logits = np.concatenate(logits)
    all_labels = np.concatenate(targets)
    metrics = classification_metrics(all_labels, all_logits.argmax(axis=1), n_classes=4)
    assert route is not None and posterior is not None
    return {
        **metrics,
        "logits": all_logits,
        "endpoint_logits": np.concatenate(endpoints),
        "labels": all_labels,
        "mean_firing_rate": float(np.mean(rates)) if rates else 0.0,
        "mean_nonzero_delay_mass": float(np.mean(nonzero)),
        "mean_delay_residual_rms": float(np.mean(residual_rms)),
        "route_probability": route,
        "lag_posterior": posterior,
    }


def fit_v13(
    *,
    base_variant: str,
    base_state: dict[str, torch.Tensor],
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
    epochs: int = 100,
    patience: int = 20,
    fixed_epoch: int | None = None,
    scheduler_epochs: int | None = None,
    batch_size: int = 48,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    kd_weight: float = 0.35,
    temperature: float = 2.0,
    route_l0_weight: float = 1e-3,
    firing_rate_weight: float = 0.01,
    run_label: str = "",
) -> V13Fit:
    values = (atc_validation, fbc_validation, y_validation, teacher_validation)
    has_validation = all(value is not None for value in values)
    if has_validation != any(value is not None for value in values):
        raise ValueError("all V13 validation arrays must be supplied together")
    if not has_validation and fixed_epoch is None:
        raise ValueError("V13 training without validation requires fixed_epoch")
    if not 0.0 <= kd_weight <= 1.0 or min(temperature, learning_rate) <= 0.0:
        raise ValueError("invalid V13 optimization settings")
    seed_v12(seed)
    model = V13DelayResidualStudent(base_variant=base_variant, freeze_base=True)
    model.load_base_state(base_state)
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(learning_rate), weight_decay=float(weight_decay)
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
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        total = 0.0
        total_examples = 0
        for batch_atc, batch_fbc, batch_y, batch_teacher in loader:
            batch_atc = batch_atc.to(device, non_blocking=True)
            batch_fbc = batch_fbc.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_teacher = batch_teacher.to(device, non_blocking=True)
            output = model(batch_atc, batch_fbc, control="full")
            hard = F.cross_entropy(output.logits, batch_y)
            soft = F.kl_div(
                F.log_softmax(output.logits / temperature, dim=1),
                F.softmax(batch_teacher / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature * temperature)
            route_penalty = output.route_probability.mean()
            loss = (
                (1.0 - kd_weight) * hard
                + kd_weight * soft
                + float(route_l0_weight) * route_penalty
                + float(firing_rate_weight) * output.firing_rate_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            optimizer_steps += 1
            total += float(loss.detach()) * batch_atc.shape[0]
            total_examples += batch_atc.shape[0]
        scheduler.step()
        row: dict[str, float | int] = {
            "epoch": epoch,
            "loss": total / max(total_examples, 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if has_validation:
            evaluation = predict_v13(
                model,
                atc_validation,
                fbc_validation,
                y_validation,
                teacher_validation,
                control="full",
                device=device,
                batch_size=batch_size,
            )
            metric = float(evaluation["kappa"])
            accuracy = float(evaluation["accuracy"])
            row.update(
                {
                    "validation_kappa": metric,
                    "validation_accuracy": accuracy,
                    "validation_nonzero_delay_mass": float(
                        evaluation["mean_nonzero_delay_mass"]
                    ),
                    "validation_delay_residual_rms": float(
                        evaluation["mean_delay_residual_rms"]
                    ),
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
                "__V13_PROGRESS__ "
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
    return V13Fit(
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
