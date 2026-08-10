"""Unified preprocessing and training utilities for V6.2 neural baselines."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dpc_snn.baselines.neural import build_v62_neural_baseline
from dpc_snn.utils.metrics import classification_metrics


FBCNET_BANDS = tuple((low, low + 4) for low in range(4, 40, 4))

BASELINE_OPTIMIZERS: dict[str, dict[str, Any]] = {
    "eegnet": {"lr": 1e-3, "weight_decay": 0.0, "beta1": 0.9, "batch_size": 64, "schedule": False},
    "fbcnet": {"lr": 1e-3, "weight_decay": 0.0, "beta1": 0.9, "batch_size": 16, "schedule": False},
    "atcnet": {"lr": 9e-4, "weight_decay": 1e-3, "beta1": 0.5, "batch_size": 48, "schedule": True},
    "tcformer": {"lr": 9e-4, "weight_decay": 1e-3, "beta1": 0.5, "batch_size": 48, "schedule": True},
    "eeg_conformer": {"lr": 2e-4, "weight_decay": 0.0, "beta1": 0.5, "batch_size": 48, "schedule": True},
    "mi_snn_plif": {"lr": 1e-2, "weight_decay": 0.0, "beta1": 0.9, "batch_size": 16, "schedule": False},
    "bfatcnet": {"lr": 9e-4, "weight_decay": 1e-3, "beta1": 0.9, "batch_size": 48, "schedule": True},
}


@dataclass(frozen=True)
class FixedGain:
    values: np.ndarray
    clip: float


@dataclass
class FitResult:
    model: nn.Module
    history: list[dict[str, float | int]]
    best_epoch: int
    best_metric: float
    best_state: dict[str, torch.Tensor]
    last_state: dict[str, torch.Tensor]
    optimizer_steps: int
    elapsed_seconds: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def task_carrier(
    x: np.ndarray,
    *,
    sfreq: float = 250.0,
    epoch_tmin: float = -1.0,
    task_start: float = 0.0,
    task_stop: float = 4.0,
) -> np.ndarray:
    """CAR, pre-cue baseline correction, then an exact half-open task crop."""

    array = np.asarray(x, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("baseline carrier expects [trials, channels, samples]")
    times = float(epoch_tmin) + np.arange(array.shape[-1], dtype=np.float64) / float(sfreq)
    baseline = (times >= -1.0) & (times < 0.0)
    task = (times >= float(task_start)) & (times < float(task_stop))
    expected = int(round((task_stop - task_start) * sfreq))
    if int(baseline.sum()) != int(round(sfreq)) or int(task.sum()) != expected:
        raise ValueError(
            f"epoch timestamps do not contain the required baseline/task samples: "
            f"baseline={baseline.sum()}, task={task.sum()}, expected_task={expected}"
        )
    referenced = array - array.mean(axis=1, keepdims=True)
    corrected = referenced - referenced[..., baseline].mean(axis=-1, keepdims=True)
    return np.ascontiguousarray(corrected[..., task], dtype=np.float32)


def fit_fixed_gain(x_train: np.ndarray, clip: float = 12.0) -> FixedGain:
    array = np.asarray(x_train, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] == 0:
        raise ValueError("fixed gain requires non-empty [trials, channels, time]")
    rms = np.sqrt(np.mean(np.square(array), axis=(0, 2), keepdims=True))
    finite = rms[np.isfinite(rms) & (rms > 0)]
    if finite.size == 0:
        raise ValueError("training carrier has no finite non-zero channel gain")
    floor = max(float(np.median(finite)) * 1e-3, np.finfo(np.float32).tiny)
    return FixedGain(values=np.maximum(rms, floor).astype(np.float32), clip=float(clip))


def apply_fixed_gain(x: np.ndarray, gain: FixedGain) -> np.ndarray:
    normalized = np.asarray(x, dtype=np.float32) / gain.values
    return np.ascontiguousarray(np.clip(normalized, -gain.clip, gain.clip), dtype=np.float32)


def fbcnet_filterbank(x: np.ndarray, sfreq: float = 250.0) -> np.ndarray:
    """Reproduce the official 9-band Chebyshev-II causal filter bank."""

    from scipy import signal

    array = np.asarray(x, dtype=np.float32)
    output = np.empty((*array.shape, len(FBCNET_BANDS)), dtype=np.float32)
    nyquist = float(sfreq) / 2.0
    for index, (low, high) in enumerate(FBCNET_BANDS):
        passband = [low / nyquist, high / nyquist]
        stopband = [(low - 2) / nyquist, (high + 2) / nyquist]
        order, stop = signal.cheb2ord(passband, stopband, 3, 30)
        numerator, denominator = signal.cheby2(order, 30, stop, btype="bandpass")
        output[..., index] = signal.lfilter(
            numerator, denominator, array, axis=-1
        ).astype(np.float32)
    return np.ascontiguousarray(output[:, None, ...], dtype=np.float32)


def prepare_model_input(name: str, x: np.ndarray, sfreq: float = 250.0) -> np.ndarray:
    if name == "fbcnet":
        return fbcnet_filterbank(x, sfreq=sfreq)
    if name == "eegnet":
        from scipy import signal

        start = int(round(0.5 * sfreq))
        stop = int(round(2.5 * sfreq))
        cropped = np.asarray(x, dtype=np.float32)[..., start:stop]
        resampled = signal.resample_poly(cropped, up=64, down=125, axis=-1)
        if resampled.shape[-1] != 256:
            raise RuntimeError(f"EEGNet official view must contain 256 samples, got {resampled.shape}")
        return np.ascontiguousarray(resampled, dtype=np.float32)
    if name == "eeg_conformer":
        from scipy import signal

        sos = signal.butter(4, (4.0, 40.0), btype="bandpass", fs=sfreq, output="sos")
        filtered = signal.sosfiltfilt(sos, np.asarray(x, dtype=np.float32), axis=-1)
        return np.ascontiguousarray(filtered, dtype=np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def segment_reconstruction(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    segments: int = 8,
    probability: float = 0.5,
    noise_std: float = 0.01,
    scale_range: tuple[float, float] = (0.9, 1.1),
) -> torch.Tensor:
    """Class-conditional segment recombination using current training samples only."""

    if probability <= 0.0 or torch.rand((), device=x.device) >= probability:
        return x
    time_axis = x.ndim - 2 if x.ndim == 5 else x.ndim - 1
    length = x.shape[time_axis]
    boundaries = torch.linspace(0, length, segments + 1, device=x.device).round().long()
    mixed = x.clone()
    for class_value in torch.unique(y):
        indices = torch.where(y == class_value)[0]
        if indices.numel() < 2:
            continue
        for segment in range(segments):
            donors = indices[torch.randperm(indices.numel(), device=x.device)]
            target_slice = [slice(None)] * x.ndim
            donor_slice = [slice(None)] * x.ndim
            target_slice[0] = indices
            donor_slice[0] = donors
            start = int(boundaries[segment])
            stop = int(boundaries[segment + 1])
            target_slice[time_axis] = slice(start, stop)
            donor_slice[time_axis] = slice(start, stop)
            mixed[tuple(target_slice)] = x[tuple(donor_slice)]
    scale_shape = [x.shape[0]] + [1] * (x.ndim - 1)
    low, high = scale_range
    scale = torch.empty(scale_shape, device=x.device, dtype=x.dtype).uniform_(low, high)
    if noise_std > 0.0:
        mixed = mixed + torch.randn_like(mixed) * float(noise_std)
    return mixed * scale


def _loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    dataset = TensorDataset(torch.from_numpy(x), torch.as_tensor(y, dtype=torch.long))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    output = model(x)
    logits = output["logits"] if isinstance(output, dict) else output
    if logits.ndim != 2:
        raise RuntimeError(f"baseline logits must have shape [batch, classes], got {logits.shape}")
    return logits


@torch.no_grad()
def predict_baseline(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    loader = _loader(x, y, batch_size=batch_size, shuffle=False, seed=0)
    labels: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        output = _logits(model, batch_x)
        if not torch.isfinite(output).all():
            raise FloatingPointError("baseline evaluation produced non-finite logits")
        logits.append(output.float().cpu().numpy())
        labels.append(batch_y.numpy())
    all_logits = np.concatenate(logits)
    all_labels = np.concatenate(labels)
    prediction = all_logits.argmax(axis=1)
    return {
        "logits": all_logits,
        "labels": all_labels,
        "pred": prediction,
        **classification_metrics(all_labels, prediction, n_classes=all_logits.shape[1]),
    }


@torch.no_grad()
def canonicalize_evaluation_state(
    name: str,
    model: nn.Module,
    x_reference: np.ndarray,
    *,
    device: str,
) -> bool:
    """Apply official forward-time constraints before sealing a checkpoint."""

    if name != "fbcnet":
        return False
    reference = np.asarray(x_reference, dtype=np.float32)
    if reference.shape[0] == 0:
        raise ValueError("FBCNet state canonicalization requires a training example")
    model.eval()
    _logits(model, torch.from_numpy(reference[:1]).to(device))
    return True


def _scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    enabled: bool,
    epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if not enabled:
        return None

    def scale(epoch: int) -> float:
        current = epoch + 1
        if current <= warmup_epochs:
            return current / max(1, warmup_epochs)
        progress = (current - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def fit_baseline(
    name: str,
    *,
    source_root: str | Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray | None,
    y_validation: np.ndarray | None,
    device: str,
    seed: int,
    epochs: int,
    patience: int,
    augmentation: dict[str, Any] | None = None,
    fixed_epoch: int | None = None,
    scheduler_epochs: int | None = None,
    run_label: str = "",
    n_channels: int = 22,
    n_classes: int = 4,
    physical_batch_size: int | None = None,
    effective_batch_size: int | None = None,
) -> FitResult:
    """Fit one fold, or a fixed-epoch final model when validation is omitted."""

    seed_everything(seed)
    settings = dict(BASELINE_OPTIMIZERS[name])
    batch_size = int(
        settings["batch_size"] if physical_batch_size is None else physical_batch_size
    )
    effective_batch = int(
        batch_size if effective_batch_size is None else effective_batch_size
    )
    if batch_size <= 0 or effective_batch <= 0:
        raise ValueError("physical and effective batch sizes must be positive")
    if effective_batch < batch_size or effective_batch % batch_size != 0:
        raise ValueError(
            "effective batch size must be an integer multiple of physical batch size"
        )
    accumulation_steps = effective_batch // batch_size
    actual_epochs = int(fixed_epoch if fixed_epoch is not None else epochs)
    schedule_horizon = int(scheduler_epochs if scheduler_epochs is not None else actual_epochs)
    if schedule_horizon < actual_epochs:
        raise ValueError("baseline scheduler horizon may not end before executed epochs")
    model = build_v62_neural_baseline(
        name,
        source_root=source_root,
        n_channels=int(n_channels),
        n_classes=int(n_classes),
        samples=int(x_train.shape[-2] if x_train.ndim == 5 else x_train.shape[-1]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["lr"]),
        betas=(float(settings["beta1"]), 0.999),
        weight_decay=float(settings["weight_decay"]),
    )
    scheduler = _scheduler(
        optimizer,
        enabled=bool(settings["schedule"]),
        epochs=schedule_horizon,
        warmup_epochs=min(20, max(1, schedule_horizon // 5)),
    )
    loader = _loader(x_train, y_train, batch_size=batch_size, shuffle=True, seed=seed)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    history: list[dict[str, float | int]] = []
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_metric = -float("inf")
    best_epoch = 0
    stale = 0
    optimizer_steps = 0
    started = time.perf_counter()
    augmentation = augmentation or {}
    for epoch in range(1, actual_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_samples = 0
        pending_x: list[torch.Tensor] = []
        pending_y: list[torch.Tensor] = []
        for batch_index, (physical_x, physical_y) in enumerate(loader):
            pending_x.append(physical_x)
            pending_y.append(physical_y)
            group_end = (batch_index + 1) % accumulation_steps == 0
            final_batch = batch_index + 1 == len(loader)
            if not (group_end or final_batch):
                continue
            batch_x = torch.cat(pending_x, dim=0).to(device, non_blocking=True)
            batch_y = torch.cat(pending_y, dim=0).to(device, non_blocking=True)
            pending_x.clear()
            pending_y.clear()
            if augmentation.get("enabled", False):
                batch_x = segment_reconstruction(
                    batch_x,
                    batch_y,
                    segments=int(augmentation.get("segments", 8)),
                    probability=float(augmentation.get("probability", 0.5)),
                    noise_std=float(augmentation.get("noise_std", 0.01)),
                    scale_range=tuple(augmentation.get("scale_range", (0.9, 1.1))),
                )
            group_samples = int(batch_y.shape[0])
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, group_samples, batch_size):
                stop = min(start + batch_size, group_samples)
                logits = _logits(model, batch_x[start:stop])
                loss = criterion(logits, batch_y[start:stop])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite {name} loss at epoch {epoch}")
                micro_samples = stop - start
                (loss * (micro_samples / group_samples)).backward()
                epoch_loss_sum += float(loss.detach().cpu()) * micro_samples
                epoch_samples += micro_samples
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0, error_if_nonfinite=True
            )
            optimizer.step()
            optimizer_steps += 1
        if scheduler is not None:
            scheduler.step()
        row: dict[str, float | int] = {
            "epoch": epoch,
            "loss": float(epoch_loss_sum / max(1, epoch_samples)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "optimizer_steps": optimizer_steps,
            "physical_batch_size": batch_size,
            "effective_batch_size": effective_batch,
        }
        if x_validation is not None and y_validation is not None:
            evaluation = predict_baseline(
                model,
                x_validation,
                y_validation,
                device=device,
                batch_size=batch_size,
            )
            row.update(
                val_accuracy=float(evaluation["accuracy"]),
                val_balanced_accuracy=float(evaluation["balanced_accuracy"]),
                val_kappa=float(evaluation["kappa"]),
                val_macro_f1=float(evaluation["macro_f1"]),
            )
            metric = float(evaluation["kappa"])
        else:
            metric = -float(row["loss"])
        history.append(row)
        print(
            "__V62_BASELINE_PROGRESS__ "
            f"run={run_label or name} epoch={epoch}/{actual_epochs} "
            f"loss={float(row['loss']):.6f} "
            f"val_kappa={float(row.get('val_kappa', float('nan'))):.6f}",
            flush=True,
        )
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if x_validation is not None and stale >= patience:
            break
    canonicalize_evaluation_state(name, model, x_train, device=device)
    last_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state if x_validation is not None else last_state)
    return FitResult(
        model=model,
        history=history,
        best_epoch=best_epoch if x_validation is not None else actual_epochs,
        best_metric=best_metric,
        best_state=best_state if x_validation is not None else last_state,
        last_state=last_state,
        optimizer_steps=optimizer_steps,
        elapsed_seconds=time.perf_counter() - started,
    )


def clone_cpu_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return deepcopy({key: value.detach().cpu() for key, value in state.items()})
