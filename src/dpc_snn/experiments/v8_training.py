"""Leakage-safe cached-rate training utilities for V8."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import random
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dpc_snn.models.v8_accuracy_first import V8AccuracyFirstModel
from dpc_snn.models.v8_delay_auxiliary import V8DelayOverride
from dpc_snn.utils.metrics import classification_metrics


@dataclass(frozen=True)
class V8CachedRates:
    """Frozen physical analytic rates before all trainable V8 representation layers."""

    fast: torch.Tensor
    slow: torch.Tensor
    gain: torch.Tensor
    physical_frontend_fingerprint: str

    def subset(self, indices: np.ndarray | Sequence[int]) -> "V8CachedRates":
        selected = torch.as_tensor(np.asarray(indices), dtype=torch.long)
        return V8CachedRates(
            fast=self.fast.index_select(0, selected),
            slow=self.slow.index_select(0, selected),
            gain=self.gain,
            physical_frontend_fingerprint=self.physical_frontend_fingerprint,
        )


@dataclass
class V8FitResult:
    model: V8AccuracyFirstModel
    history: list[dict[str, float | int]]
    best_epoch: int
    best_metric: float
    best_accuracy: float
    best_state: dict[str, torch.Tensor]
    last_state: dict[str, torch.Tensor]
    optimizer_steps: int
    elapsed_seconds: float


def seed_v8(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _raw_loader(x: np.ndarray, batch_size: int) -> DataLoader:
    array = np.ascontiguousarray(np.asarray(x), dtype=np.float32)
    return DataLoader(
        TensorDataset(torch.from_numpy(array)),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def fit_v8_physical_gain(
    model: V8AccuracyFirstModel,
    x_inner_train: np.ndarray,
    *,
    device: str,
    batch_size: int = 16,
) -> torch.Tensor:
    """Fit inverse physical carrier RMS using only the active training partition."""

    if not model.physical_basis.frozen or model.physical_basis.n_nodes != model.n_channels:
        raise ValueError("V8 gain fitting requires the frozen exact physical sensor basis")
    model.eval().to(device)
    model.set_training_gain(torch.ones(model.n_bands, model.n_channels))
    energy = torch.zeros(model.n_bands, model.n_channels, dtype=torch.float64)
    observations = 0
    for (batch_x,) in _raw_loader(x_inner_train, batch_size):
        prepared = model._prepare_raw(batch_x.to(device, non_blocking=True))
        analytic = model.filterbank(prepared)
        physical = model.physical_basis(analytic)
        rates = model.resampler(
            physical,
            epoch_tmin=model.epoch_tmin,
            task_tmin=model.task_tmin,
            task_tmax=model.task_tmax,
        )
        energy += (
            rates.fast.to(torch.complex128).abs().square().sum(dim=(0, 3)).cpu()
        )
        observations += rates.fast.shape[0] * rates.fast.shape[-1]
    if observations < 1:
        raise ValueError("V8 gain fitting received no training observations")
    rms = (energy / observations).sqrt()
    finite = rms[torch.isfinite(rms) & (rms > 0)]
    if finite.numel() == 0:
        raise ValueError("V8 physical carrier has no finite non-zero RMS")
    floor = max(float(finite.median()) * 1e-3, torch.finfo(torch.float32).tiny)
    gain = rms.clamp_min(floor).reciprocal().float()
    if not bool(torch.isfinite(gain).all()) or bool((gain <= 0).any()):
        raise FloatingPointError("V8 fold-local gain is not finite and positive")
    model.set_training_gain(gain)
    return gain.cpu()


@torch.no_grad()
def cache_v8_physical_rates(
    model: V8AccuracyFirstModel,
    x: np.ndarray,
    *,
    device: str,
    batch_size: int = 16,
) -> V8CachedRates:
    """Cache only frozen physical rates; trainable spatial/temporal layers stay live."""

    if not model.physical_basis.frozen or not bool(model.resampler.training_gain_ready):
        raise ValueError("V8 caching requires a frozen basis and fitted training gain")
    model.eval().to(device)
    fast: list[torch.Tensor] = []
    slow: list[torch.Tensor] = []
    for (batch_x,) in _raw_loader(x, batch_size):
        prepared = model._prepare_raw(batch_x.to(device, non_blocking=True))
        analytic = model.filterbank(prepared)
        physical = model.physical_basis(analytic)
        rates = model.resampler(
            physical,
            epoch_tmin=model.epoch_tmin,
            task_tmin=model.task_tmin,
            task_tmax=model.task_tmax,
        )
        fast.append(rates.fast.to(torch.complex64).cpu())
        slow.append(rates.slow.float().cpu())
    cached = V8CachedRates(
        fast=torch.cat(fast),
        slow=torch.cat(slow),
        gain=model.resampler.training_gain.detach().cpu().clone(),
        physical_frontend_fingerprint=model.physical_frontend_fingerprint(),
    )
    _validate_rates(cached, model=model)
    return cached


def fit_v8_gain_from_cached_rates(
    base_rates: V8CachedRates,
    train_indices: np.ndarray | Sequence[int],
) -> torch.Tensor:
    """Fit inverse carrier RMS from a unit-gain physical-rate cache."""

    _validate_rates(base_rates)
    if not torch.equal(base_rates.gain, torch.ones_like(base_rates.gain)):
        raise ValueError("fold gains must be fitted from a unit-gain V8 cache")
    indices = torch.as_tensor(np.asarray(train_indices), dtype=torch.long)
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("V8 cached gain fitting requires non-empty one-dimensional indices")
    selected = base_rates.fast.index_select(0, indices)
    rms = selected.to(torch.complex128).abs().square().mean(dim=(0, 3)).sqrt()
    finite = rms[torch.isfinite(rms) & (rms > 0)]
    if finite.numel() == 0:
        raise ValueError("V8 cached carrier has no finite non-zero RMS")
    floor = max(float(finite.median()) * 1e-3, torch.finfo(torch.float32).tiny)
    gain = rms.clamp_min(floor).reciprocal().float()
    if not bool(torch.isfinite(gain).all()) or bool((gain <= 0).any()):
        raise FloatingPointError("V8 cached fold gain is not finite and positive")
    return gain


def apply_v8_fold_gain(
    base_rates: V8CachedRates,
    gain: torch.Tensor,
) -> V8CachedRates:
    """Apply a fold-local positive gain without recomputing scale-invariant envelopes.

    The slow stream is baseline-relative log amplitude computed in physical
    units before carrier gain.  It is therefore exactly invariant to the
    fold-local carrier gain, including in the guarded near-zero region.
    """

    _validate_rates(base_rates)
    gain = torch.as_tensor(gain, dtype=torch.float32)
    if gain.shape != base_rates.gain.shape or not bool(torch.isfinite(gain).all()):
        raise ValueError("V8 fold gain has incompatible shape or non-finite values")
    if bool((gain <= 0).any()):
        raise ValueError("V8 fold gain must be strictly positive")
    return V8CachedRates(
        fast=base_rates.fast * gain[None, :, :, None],
        slow=base_rates.slow,
        gain=gain,
        physical_frontend_fingerprint=base_rates.physical_frontend_fingerprint,
    )


def save_v8_rates(path: str | Any, rates: V8CachedRates) -> None:
    torch.save(
        {
            "schema": "dpc-snn-v8-physical-rates/v2-gain-invariant-envelope",
            "fast": rates.fast,
            "slow": rates.slow,
            "gain": rates.gain,
            "physical_frontend_fingerprint": rates.physical_frontend_fingerprint,
        },
        path,
    )


def load_v8_rates(path: str | Any, model: V8AccuracyFirstModel) -> V8CachedRates:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "dpc-snn-v8-physical-rates/v2-gain-invariant-envelope":
        raise RuntimeError("unsupported V8 physical-rate cache schema")
    cached = V8CachedRates(
        fast=payload["fast"].to(torch.complex64),
        slow=payload["slow"].float(),
        gain=payload["gain"].float(),
        physical_frontend_fingerprint=str(payload["physical_frontend_fingerprint"]),
    )
    _validate_rates(cached, model=model)
    model.set_training_gain(cached.gain)
    return cached


def _validate_rates(
    rates: V8CachedRates,
    *,
    model: V8AccuracyFirstModel | None = None,
) -> None:
    if rates.fast.ndim != 4 or not rates.fast.is_complex():
        raise ValueError("V8 fast cache must be complex [N, B, C, T]")
    if rates.slow.ndim != 4 or rates.slow.is_complex():
        raise ValueError("V8 slow cache must be real [N, B, C, T/2]")
    if rates.fast.shape[:-1] != rates.slow.shape[:-1]:
        raise ValueError("V8 fast and slow cache axes differ")
    if rates.fast.shape[-1] != 2 * rates.slow.shape[-1]:
        raise ValueError("V8 cached rates do not have an exact 2:1 ratio")
    if rates.gain.shape != rates.fast.shape[1:3]:
        raise ValueError("V8 cached gain does not match band/channel axes")
    if not bool(torch.isfinite(rates.fast.real).all()) or not bool(
        torch.isfinite(rates.fast.imag).all()
    ):
        raise FloatingPointError("V8 fast cache contains non-finite values")
    if not bool(torch.isfinite(rates.slow).all()) or not bool(torch.isfinite(rates.gain).all()):
        raise FloatingPointError("V8 slow cache or gain contains non-finite values")
    if model is not None:
        if rates.fast.shape[1:3] != (model.n_bands, model.n_channels):
            raise ValueError("V8 cache does not match the active model")
        if rates.physical_frontend_fingerprint != model.physical_frontend_fingerprint():
            raise RuntimeError("V8 cache physical-front-end fingerprint mismatch")


def paired_v8_rate_reconstruction(
    fast: torch.Tensor,
    slow: torch.Tensor,
    labels: torch.Tensor,
    *,
    segments: int = 8,
    probability: float = 0.5,
    carrier_scale_range: Sequence[float] = (0.9, 1.1),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same-class aligned segment recombination within the current train batch."""

    if probability <= 0.0 or torch.rand((), device=fast.device) >= float(probability):
        return fast, slow
    if fast.shape[-1] != 2 * slow.shape[-1]:
        raise ValueError("V8 reconstruction requires aligned 2:1 rates")
    mixed_fast = fast.clone()
    mixed_slow = slow.clone()
    boundaries = torch.linspace(0, slow.shape[-1], int(segments) + 1, device=slow.device)
    boundaries = boundaries.round().long()
    for class_value in torch.unique(labels):
        indices = torch.where(labels == class_value)[0]
        if indices.numel() < 2:
            continue
        for segment in range(int(segments)):
            donors = indices[torch.randperm(indices.numel(), device=indices.device)]
            slow_start = int(boundaries[segment])
            slow_stop = int(boundaries[segment + 1])
            fast_start, fast_stop = 2 * slow_start, 2 * slow_stop
            mixed_fast[indices, ..., fast_start:fast_stop] = fast[
                donors, ..., fast_start:fast_stop
            ]
            mixed_slow[indices, ..., slow_start:slow_stop] = slow[
                donors, ..., slow_start:slow_stop
            ]
    low, high = (float(value) for value in carrier_scale_range)
    if not 0.0 < low <= high:
        raise ValueError("carrier scale range must be positive and ordered")
    scale = torch.empty(
        (fast.shape[0], 1, 1, 1), device=fast.device, dtype=fast.real.dtype
    ).uniform_(low, high)
    return mixed_fast * scale, mixed_slow


def _rates_loader(
    rates: V8CachedRates,
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    _validate_rates(rates)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    if rates.fast.shape[0] != y.shape[0]:
        raise ValueError("V8 rates and labels have different trial counts")
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(rates.fast, rates.slow, y),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _loss(
    output: dict[str, Any],
    labels: torch.Tensor,
    *,
    endpoint_weights: Sequence[float],
    endpoint_loss_weight: float,
    firing_rate_weight: float,
    spatial_orthogonality_weight: float,
    statistical_orthogonality_weight: float,
    label_smoothing: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    primary = F.cross_entropy(
        output["logits"], labels, label_smoothing=float(label_smoothing)
    )
    prefix = output["prefix_logits"]
    weights = torch.as_tensor(tuple(endpoint_weights), device=prefix.device, dtype=prefix.dtype)
    if weights.shape != (prefix.shape[1],) or bool((weights < 0).any()):
        raise ValueError("endpoint loss weights do not match V8 prefix endpoints")
    positive = weights.sum()
    if float(positive) > 0.0:
        endpoint_losses = torch.stack(
            [
                F.cross_entropy(
                    prefix[:, index], labels, label_smoothing=float(label_smoothing)
                )
                for index in range(prefix.shape[1])
            ]
        )
        endpoint = (endpoint_losses * weights).sum() / positive
    else:
        endpoint = primary.new_zeros(())
    firing = output["firing_rate_loss"]
    spatial = output["spatial_orthogonality_loss"]
    statistical = output["statistical_orthogonality_loss"]
    total = (
        primary
        + float(endpoint_loss_weight) * endpoint
        + float(firing_rate_weight) * firing
        + float(spatial_orthogonality_weight) * spatial
        + float(statistical_orthogonality_weight) * statistical
    )
    return total, {
        "primary": primary,
        "endpoint": endpoint,
        "firing": firing,
        "spatial": spatial,
        "statistical": statistical,
    }


def _scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_updates: int,
    warmup_updates: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def scale(update: int) -> float:
        current = update + 1
        if current <= warmup_updates:
            return current / max(1, warmup_updates)
        progress = (current - warmup_updates) / max(1, total_updates - warmup_updates)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


@torch.no_grad()
def predict_v8(
    model: V8AccuracyFirstModel,
    rates: V8CachedRates,
    labels: np.ndarray,
    *,
    device: str,
    batch_size: int,
    delay_override: V8DelayOverride = "off",
) -> dict[str, Any]:
    _validate_rates(rates, model=model)
    model.eval().to(device)
    logits: list[np.ndarray] = []
    prefix_logits: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    spike_sum = 0.0
    spike_elements = 0
    final_activity_nonzero = 0
    final_activity_elements = 0
    final_activity_absolute_sum = 0.0
    delay_current_energy = 0.0
    delay_current_elements = 0
    for fast, slow, batch_y in _rates_loader(
        rates, labels, batch_size=batch_size, shuffle=False, seed=0
    ):
        output = model.forward_rate_features(
            fast.to(device, non_blocking=True),
            slow.to(device, non_blocking=True),
            delay_override=delay_override,
        )
        if not bool(torch.isfinite(output["logits"]).all()):
            raise FloatingPointError("V8 prediction produced non-finite logits")
        logits.append(output["logits"].float().cpu().numpy())
        prefix_logits.append(output["prefix_logits"].float().cpu().numpy())
        observed.append(batch_y.numpy())
        for spikes in output["aux"]["binary_spikes"]:
            spike_sum += float(spikes.detach().float().sum().cpu())
            spike_elements += spikes.numel()
        final_activity = output["aux"]["final_spikes"]
        if final_activity is not None:
            detached = final_activity.detach().float()
            final_activity_nonzero += int((detached.abs() > 1e-8).sum().cpu())
            final_activity_elements += detached.numel()
            final_activity_absolute_sum += float(detached.abs().sum().cpu())
        delay_output = output["aux"]["delay_auxiliary"]
        if delay_override == "off":
            if delay_output is not None:
                raise RuntimeError("off delay prediction unexpectedly returned delay output")
        else:
            if delay_output is None:
                raise RuntimeError("active delay prediction did not return routed current")
            current = delay_output.physical_current.detach().float()
            delay_current_energy += float(current.square().sum().cpu())
            delay_current_elements += current.numel()
    all_logits = np.concatenate(logits)
    all_prefix = np.concatenate(prefix_logits)
    all_labels = np.concatenate(observed)
    prediction = all_logits.argmax(axis=1)
    endpoint_metrics = [
        classification_metrics(
            all_labels,
            all_prefix[:, index].argmax(axis=1),
            n_classes=int(model.n_classes),
        )
        for index in range(all_prefix.shape[1])
    ]
    return {
        "logits": all_logits,
        "prefix_logits": all_prefix,
        "labels": all_labels,
        "pred": prediction,
        "endpoint_metrics": endpoint_metrics,
        "binary_spike_rate": (
            float(spike_sum / spike_elements) if spike_elements else None
        ),
        "final_activity_nonzero_rate": (
            float(final_activity_nonzero / final_activity_elements)
            if final_activity_elements
            else None
        ),
        "final_activity_absolute_mean": (
            float(final_activity_absolute_sum / final_activity_elements)
            if final_activity_elements
            else None
        ),
        "delay_current_rms": (
            float(math.sqrt(delay_current_energy / delay_current_elements))
            if delay_current_elements
            else None
        ),
        **classification_metrics(all_labels, prediction, n_classes=int(model.n_classes)),
    }


def fit_v8(
    model: V8AccuracyFirstModel,
    train_rates: V8CachedRates,
    train_labels: np.ndarray,
    *,
    validation_rates: V8CachedRates | None,
    validation_labels: np.ndarray | None,
    device: str,
    seed: int,
    epochs: int,
    patience: int,
    minimum_epochs: int,
    batch_size: int,
    accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    max_gradient_norm: float,
    warmup_fraction: float,
    endpoint_weights: Sequence[float],
    endpoint_loss_weight: float,
    firing_rate_weight: float,
    spatial_orthogonality_weight: float,
    statistical_orthogonality_weight: float,
    label_smoothing: float,
    augmentation: dict[str, Any] | None,
    fixed_epoch: int | None = None,
    scheduler_epochs: int | None = None,
    delay_override: V8DelayOverride = "off",
    include_initial_validation: bool = False,
    run_label: str = "",
) -> V8FitResult:
    """Fit V8 on cached physical rates with update-aligned accumulation."""

    seed_v8(seed)
    _validate_rates(train_rates, model=model)
    if validation_rates is not None:
        _validate_rates(validation_rates, model=model)
        if validation_labels is None:
            raise ValueError("validation labels are required with validation rates")
        if not torch.equal(train_rates.gain, validation_rates.gain):
            raise ValueError("training and validation rates use different fold-local gains")
    elif validation_labels is not None:
        raise ValueError("validation rates are required with validation labels")
    actual_epochs = int(fixed_epoch if fixed_epoch is not None else epochs)
    if actual_epochs < 1 or int(accumulation_steps) < 1:
        raise ValueError("epochs and accumulation steps must be positive")
    schedule_horizon = int(scheduler_epochs if scheduler_epochs is not None else actual_epochs)
    if schedule_horizon < actual_epochs:
        raise ValueError("scheduler horizon may not end before the executed epochs")
    model.set_training_gain(train_rates.gain)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    train_loader = _rates_loader(
        train_rates,
        train_labels,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    updates_per_epoch = math.ceil(len(train_loader) / int(accumulation_steps))
    total_updates = schedule_horizon * updates_per_epoch
    scheduler = _scheduler(
        optimizer,
        total_updates=total_updates,
        warmup_updates=max(1, int(round(total_updates * float(warmup_fraction)))),
    )
    best_metric = -math.inf
    best_accuracy = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    optimizer_steps = 0
    stale_epochs = 0
    started = time.perf_counter()
    augmentation = dict(augmentation or {})
    if include_initial_validation:
        if validation_rates is None or validation_labels is None:
            raise ValueError("initial validation requires validation rates and labels")
        initial = predict_v8(
            model,
            validation_rates,
            validation_labels,
            device=device,
            batch_size=batch_size,
            delay_override=delay_override,
        )
        best_metric = float(initial["kappa"])
        best_accuracy = float(initial["accuracy"])
        best_epoch = 0
        best_state = deepcopy(
            {name: value.detach().cpu() for name, value in model.state_dict().items()}
        )

    for epoch in range(1, actual_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums = {name: 0.0 for name in ("loss", "primary", "endpoint", "firing", "spatial", "statistical")}
        examples = 0
        for batch_index, (fast, slow, batch_y) in enumerate(train_loader):
            fast = fast.to(device, non_blocking=True)
            slow = slow.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            if augmentation.get("enabled", False):
                fast, slow = paired_v8_rate_reconstruction(
                    fast,
                    slow,
                    batch_y,
                    segments=int(augmentation.get("segments", 8)),
                    probability=float(augmentation.get("probability", 0.5)),
                    carrier_scale_range=augmentation.get("carrier_scale_range", (0.9, 1.1)),
                )
            output = model.forward_rate_features(
                fast,
                slow,
                delay_override=delay_override,
            )
            loss, components = _loss(
                output,
                batch_y,
                endpoint_weights=endpoint_weights,
                endpoint_loss_weight=endpoint_loss_weight,
                firing_rate_weight=firing_rate_weight,
                spatial_orthogonality_weight=spatial_orthogonality_weight,
                statistical_orthogonality_weight=statistical_orthogonality_weight,
                label_smoothing=label_smoothing,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("V8 training loss is non-finite")
            window_start = (batch_index // int(accumulation_steps)) * int(accumulation_steps)
            window_size = min(int(accumulation_steps), len(train_loader) - window_start)
            (loss / window_size).backward()
            batch_examples = batch_y.numel()
            examples += batch_examples
            sums["loss"] += float(loss.detach().cpu()) * batch_examples
            for name, value in components.items():
                sums[name] += float(value.detach().cpu()) * batch_examples
            if (batch_index + 1) % int(accumulation_steps) == 0 or batch_index + 1 == len(
                train_loader
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_gradient_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_steps += 1

        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": sums["loss"] / max(1, examples),
            "train_primary_loss": sums["primary"] / max(1, examples),
            "train_endpoint_loss": sums["endpoint"] / max(1, examples),
            "train_firing_loss": sums["firing"] / max(1, examples),
            "train_spatial_orthogonality": sums["spatial"] / max(1, examples),
            "train_statistical_orthogonality": sums["statistical"] / max(1, examples),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "optimizer_steps": optimizer_steps,
        }
        if validation_rates is not None and validation_labels is not None:
            evaluation = predict_v8(
                model,
                validation_rates,
                validation_labels,
                device=device,
                batch_size=batch_size,
                delay_override=delay_override,
            )
            row.update(
                {
                    "validation_accuracy": evaluation["accuracy"],
                    "validation_kappa": evaluation["kappa"],
                    "validation_macro_f1": evaluation["macro_f1"],
                }
            )
            metric = float(evaluation["kappa"])
            accuracy = float(evaluation["accuracy"])
            improved = metric > best_metric + 1e-12 or (
                abs(metric - best_metric) <= 1e-12 and accuracy > best_accuracy + 1e-12
            )
            if improved:
                best_metric = metric
                best_accuracy = accuracy
                best_epoch = epoch
                best_state = deepcopy(
                    {name: value.detach().cpu() for name, value in model.state_dict().items()}
                )
                stale_epochs = 0
            else:
                stale_epochs += 1
        else:
            best_metric = math.nan
            best_accuracy = math.nan
            best_epoch = epoch
            best_state = deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )
        history.append(row)
        print(
            "__V8_TRAIN_PROGRESS__ "
            f"run={run_label} epoch={epoch}/{actual_epochs} "
            f"loss={row['train_loss']:.6f} "
            f"val_kappa={row.get('validation_kappa', float('nan')):.6f}",
            flush=True,
        )
        if (
            validation_rates is not None
            and fixed_epoch is None
            and epoch >= int(minimum_epochs)
            and stale_epochs >= int(patience)
        ):
            break
    if best_state is None:
        raise RuntimeError("V8 training did not produce a checkpoint")
    last_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    model.load_state_dict(best_state)
    return V8FitResult(
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
