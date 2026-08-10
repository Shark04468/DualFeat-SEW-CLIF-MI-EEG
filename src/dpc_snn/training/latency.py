"""Latency, spike-statistic, and early-decision measurement."""

from __future__ import annotations

import time

import numpy as np
import torch

from dpc_snn.preprocessing.hilbert import band_amplitude_phase
from dpc_snn.training.forward import forward_batch
from dpc_snn.training.tensor_dataset import batch_to_device
from dpc_snn.utils.metrics import classification_metrics


@torch.no_grad()
def measure_inference_latency(
    model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    device: str,
    warmup: int = 5,
    repeat: int = 30,
) -> dict[str, float]:
    model.to(device)
    model.eval()
    prepared = [batch_to_device(batch, device) for batch in batches]
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for i in range(min(warmup, len(prepared))):
        _ = forward_batch(model, prepared[i])
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    times = []
    for i in range(repeat):
        batch = prepared[i % len(prepared)]
        start = time.perf_counter()
        _ = forward_batch(model, batch)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    peak_memory_mb = 0.0
    if device.startswith("cuda"):
        peak_memory_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    return {
        "device": device,
        "latency_ms_mean": float(np.mean(times)),
        "latency_ms_std": float(np.std(times)),
        "peak_memory_mb": peak_memory_mb,
    }


@torch.no_grad()
def collect_spike_statistics(model: torch.nn.Module, batch: dict[str, torch.Tensor], device: str) -> dict[str, float]:
    model.to(device)
    model.eval()
    batch = batch_to_device(batch, device)
    out = forward_batch(model, batch)
    aux = out.get("aux", {})
    rows: dict[str, float] = {}
    for key in ["input_spikes", "hidden_spikes"]:
        if key in aux:
            arr = aux[key].detach().float()
            rows[f"{key}_rate"] = float(arr.mean().cpu())
            rows[f"{key}_count"] = float(arr.sum().cpu())
    if "edge_weight" in aux and "input_spikes" in aux:
        active_edges = (aux["edge_weight"].detach().abs() > 1e-6).sum().item()
        rows["active_edges"] = float(active_edges)
        rows["synops_proxy"] = float(active_edges * aux["input_spikes"].detach().float().mean().cpu() * aux["input_spikes"].shape[-1])
    return rows


@torch.no_grad()
def build_prefix_feature_batch(
    batch: dict[str, torch.Tensor],
    samples: int,
    sfreq: float,
    bands: dict[str, list[float]],
) -> dict[str, torch.Tensor]:
    """Rebuild band features solely from a raw EEG prefix.

    This deliberately does not slice full-trial Hilbert features: doing so
    leaks future samples into nominal early-decision measurements.
    """

    if "x" not in batch or "y" not in batch:
        raise KeyError("Early-decision batches require raw x and y tensors")
    x = batch["x"][..., :samples].detach().cpu()
    amplitude, phase, _ = band_amplitude_phase(x.numpy(), sfreq, bands)
    return {
        "x": x,
        "y": batch["y"].detach().cpu(),
        "amplitude": torch.from_numpy(amplitude),
        "phase": torch.from_numpy(phase),
    }


@torch.no_grad()
def decision_latency_auc(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: str,
    fractions: list[float] | None = None,
    n_classes: int | None = None,
    sfreq: float = 250.0,
    bands: dict[str, list[float]] | None = None,
) -> tuple[list[dict[str, float]], float]:
    fractions = fractions or [0.125, 0.25, 0.375, 0.5, 0.75, 1.0]
    model.to(device)
    model.eval()
    bands = bands or {"mu": [8.0, 13.0], "beta": [13.0, 30.0]}
    rows = []
    full_t = batch["x"].shape[-1]
    y_true = batch["y"].detach().cpu().numpy()
    for frac in fractions:
        t = max(4, int(round(full_t * frac)))
        partial = build_prefix_feature_batch(batch, t, sfreq=sfreq, bands=bands)
        partial = batch_to_device(partial, device)
        logits = forward_batch(model, partial)["logits"]
        pred = logits.argmax(dim=1).detach().cpu().numpy()
        metrics = classification_metrics(y_true, pred, n_classes=n_classes)
        rows.append({"fraction": frac, "samples": t, **metrics})
    x = np.asarray([r["fraction"] for r in rows], dtype=float)
    y = np.asarray([r["accuracy"] for r in rows], dtype=float)
    auc = float(np.trapz(y, x) / max(x[-1] - x[0], 1e-8))
    return rows, auc
