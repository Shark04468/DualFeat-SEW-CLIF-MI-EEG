"""Training utilities for matched routed-delay residual experts."""

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

from dpc_snn.experiments.v8_fusion import entropy_residual_probability
from dpc_snn.experiments.v8_training import V8CachedRates
from dpc_snn.models.v8_delay_auxiliary import V8DelayOverride
from dpc_snn.models.v8_delay_residual_expert import V8DelayResidualExpert
from dpc_snn.utils.metrics import classification_metrics


@dataclass
class V8DelayExpertFit:
    model: V8DelayResidualExpert
    history: list[dict[str, float | int]]
    best_epoch: int
    best_metric: float
    best_accuracy: float
    best_state: dict[str, torch.Tensor]
    last_state: dict[str, torch.Tensor]
    optimizer_steps: int
    elapsed_seconds: float


def seed_v8_delay_expert(seed: int) -> None:
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
    rates: V8CachedRates,
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    y = np.asarray(labels, dtype=np.int64)
    if y.shape != (rates.fast.shape[0],):
        raise ValueError("delay expert rates and labels are not aligned")
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(rates.fast, rates.slow, torch.from_numpy(y)),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def fit_delay_expert_input_gain(
    model: V8DelayResidualExpert,
    rates: V8CachedRates,
    *,
    device: str,
    batch_size: int = 4,
) -> torch.Tensor:
    """Fit one fold-fixed band/node gain from full delay currents only."""

    model.eval().to(device)
    model.input_gain.fill_(1.0)
    model.input_gain_ready.fill_(False)
    energy = torch.zeros(model.n_bands, model.n_nodes, dtype=torch.float64)
    observations = 0
    dummy = np.zeros(rates.fast.shape[0], dtype=np.int64)
    for fast, slow, _ in _loader(
        rates, dummy, batch_size=batch_size, shuffle=False, seed=0
    ):
        delayed = model.delay(
            fast.to(device, non_blocking=True),
            slow.to(device, non_blocking=True),
            override="full",
        )
        current = delayed.physical_current[..., :: model.temporal_decimation].double()
        energy += current.square().sum(dim=(0, 3)).cpu()
        observations += current.shape[0] * current.shape[-1]
    if observations < 1:
        raise ValueError("delay expert gain fitting received no observations")
    rms = (energy / observations).sqrt()
    finite = rms[torch.isfinite(rms) & (rms > 0)]
    if finite.numel() == 0:
        raise RuntimeError("audited delay routes generated no non-zero current")
    floor = max(float(finite.median()) * 0.05, torch.finfo(torch.float32).tiny)
    gain = rms.clamp_min(floor).reciprocal().float()
    model.set_input_gain(gain)
    return gain.cpu()


@torch.no_grad()
def predict_v8_delay_expert(
    model: V8DelayResidualExpert,
    rates: V8CachedRates,
    labels: np.ndarray,
    *,
    delay_override: V8DelayOverride,
    device: str,
    batch_size: int = 16,
) -> dict[str, Any]:
    model.eval().to(device)
    loader = _loader(rates, labels, batch_size=batch_size, shuffle=False, seed=0)
    logits: list[np.ndarray] = []
    prefix: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    firing_rates: list[float] = []
    current_energy = 0.0
    current_values = 0
    for fast, slow, batch_y in loader:
        output = model(
            fast.to(device, non_blocking=True),
            slow.to(device, non_blocking=True),
            delay_override=delay_override,
        )
        if not bool(torch.isfinite(output.logits).all()):
            raise FloatingPointError("delay expert produced non-finite logits")
        logits.append(output.logits.float().cpu().numpy())
        prefix.append(output.prefix_logits.float().cpu().numpy())
        targets.append(batch_y.numpy())
        current_energy += float(output.physical_current.float().square().sum().cpu())
        current_values += output.physical_current.numel()
        if output.binary_spikes:
            firing_rates.append(
                float(
                    torch.stack([value.float().mean() for value in output.binary_spikes])
                    .mean()
                    .cpu()
                )
            )
    all_logits = np.concatenate(logits)
    all_labels = np.concatenate(targets)
    metrics = classification_metrics(all_labels, all_logits.argmax(axis=1), n_classes=4)
    return {
        **metrics,
        "logits": all_logits,
        "prefix_logits": np.concatenate(prefix),
        "labels": all_labels,
        "mean_firing_rate": float(np.mean(firing_rates)) if firing_rates else 0.0,
        "current_rms": math.sqrt(current_energy / max(current_values, 1)),
    }


def fit_v8_delay_expert(
    model: V8DelayResidualExpert,
    rates: V8CachedRates,
    labels: np.ndarray,
    *,
    validation_rates: V8CachedRates | None,
    validation_labels: np.ndarray | None,
    validation_anchor_probability: np.ndarray | None = None,
    maximum_residual_weight: float | None = None,
    delay_override: V8DelayOverride,
    device: str,
    seed: int,
    epochs: int = 80,
    patience: int = 15,
    minimum_epochs: int = 10,
    fixed_epoch: int | None = None,
    scheduler_epochs: int | None = None,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    firing_rate_weight: float = 0.02,
    run_label: str = "",
) -> V8DelayExpertFit:
    has_validation = validation_rates is not None
    if has_validation != (validation_labels is not None):
        raise ValueError("validation rates and labels must be supplied together")
    has_fused_validation = validation_anchor_probability is not None
    if has_fused_validation != (maximum_residual_weight is not None):
        raise ValueError(
            "validation anchor probability and residual weight must be supplied together"
        )
    if has_fused_validation and not has_validation:
        raise ValueError("fused checkpoint selection requires validation data")
    if has_fused_validation:
        anchor = np.asarray(validation_anchor_probability, dtype=np.float32)
        expected = (int(np.asarray(validation_labels).size), model.n_classes)
        if anchor.shape != expected:
            raise ValueError(
                f"validation anchor probability has shape {anchor.shape}; expected {expected}"
            )
    else:
        anchor = None
    if not has_validation and fixed_epoch is None:
        raise ValueError("fixed-epoch training is required without validation")
    if int(gradient_accumulation_steps) < 1:
        raise ValueError("gradient accumulation steps must be positive")

    seed_v8_delay_expert(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    horizon = int(scheduler_epochs or epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(horizon, 1), eta_min=float(learning_rate) * 0.05
    )
    loader = _loader(rates, labels, batch_size=batch_size, shuffle=True, seed=seed)
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
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_examples = 0
        batches = len(loader)
        for index, (fast, slow, batch_y) in enumerate(loader, start=1):
            batch_y = batch_y.to(device, non_blocking=True)
            output = model(
                fast.to(device, non_blocking=True),
                slow.to(device, non_blocking=True),
                delay_override=delay_override,
            )
            loss = F.cross_entropy(output.logits, batch_y) + float(
                firing_rate_weight
            ) * output.firing_rate_loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("delay expert loss became non-finite")
            (loss / int(gradient_accumulation_steps)).backward()
            if index % int(gradient_accumulation_steps) == 0 or index == batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
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
            evaluation = predict_v8_delay_expert(
                model,
                validation_rates,
                validation_labels,
                delay_override=delay_override,
                device=device,
                batch_size=batch_size,
            )
            expert_metric = float(evaluation["kappa"])
            expert_accuracy = float(evaluation["accuracy"])
            decoder_probability = torch.softmax(
                torch.from_numpy(evaluation["logits"]).float(), dim=1
            ).numpy()
            validation_targets = np.asarray(evaluation["labels"], dtype=np.int64)
            expert_nll = float(
                -np.log(
                    np.clip(
                        decoder_probability[
                            np.arange(validation_targets.size), validation_targets
                        ],
                        1e-12,
                        1.0,
                    )
                ).mean()
            )
            if anchor is not None:
                fused_probability, _ = entropy_residual_probability(
                    anchor,
                    decoder_probability,
                    maximum_weight=float(maximum_residual_weight),
                )
                fused = classification_metrics(
                    evaluation["labels"],
                    fused_probability.argmax(axis=1),
                    n_classes=model.n_classes,
                )
                metric = float(fused["kappa"])
                accuracy = float(fused["accuracy"])
                validation_nll = float(
                    -np.log(
                        np.clip(
                            fused_probability[
                                np.arange(validation_targets.size), validation_targets
                            ],
                            1e-12,
                            1.0,
                        )
                    ).mean()
                )
            else:
                metric = expert_metric
                accuracy = expert_accuracy
                validation_nll = expert_nll
            row.update(
                {
                    "validation_kappa": metric,
                    "validation_accuracy": accuracy,
                    "validation_nll": validation_nll,
                    "validation_expert_kappa": expert_metric,
                    "validation_expert_accuracy": expert_accuracy,
                    "validation_expert_nll": expert_nll,
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
                "__V8_DELAY_EXPERT_PROGRESS__ "
                f"run={run_label} epoch={epoch}/{maximum_epochs} "
                f"loss={row['loss']:.6f} val_kappa={row.get('validation_kappa', float('nan')):.6f}",
                flush=True,
            )
        if has_validation and epoch >= int(minimum_epochs) and stale >= int(patience):
            break

    last_state = _state_cpu(model)
    if has_validation:
        model.load_state_dict(deepcopy(best_state), strict=True)
    else:
        best_metric = float("nan")
        best_accuracy = float("nan")
    return V8DelayExpertFit(
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
