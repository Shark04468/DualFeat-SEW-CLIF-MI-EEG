"""Supervised and unlabeled calibration routines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dpc_snn.training.evaluate import evaluate
from dpc_snn.training.forward import forward_batch, regularization_loss
from dpc_snn.training.tensor_dataset import TrialTensorDataset, batch_to_device
from dpc_snn.utils.io import ensure_dir, write_csv, write_json
from dpc_snn.utils.torch import count_parameters

from .calibration import set_trainable_by_mode
from .train import (
    _assert_finite,
    _assert_finite_parameters,
    _atomic_torch_save,
    make_loader,
)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, strict: bool = False) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    if not strict:
        current = model.state_dict()
        state = {
            key: value
            for key, value in state.items()
            if key in current and tuple(current[key].shape) == tuple(value.shape)
        }
    model.load_state_dict(state, strict=strict)


def supervised_finetune(
    model: torch.nn.Module,
    calib_data: dict[str, np.ndarray],
    eval_data: dict[str, np.ndarray],
    cfg: dict[str, Any],
    output_dir: str | Path,
    mode: str,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    device = cfg.get("device", "cpu")
    model.to(device)
    trainable = set_trainable_by_mode(model, mode)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError(f"No trainable parameters for calibration mode {mode}")
    train_cfg = cfg.get("training", cfg)
    epochs = int(train_cfg.get("calibration_epochs", max(3, min(20, int(train_cfg.get("epochs", 10))))))
    batch_size = int(train_cfg.get("batch_size", 64))
    optimizer = torch.optim.AdamW(params, lr=float(train_cfg.get("calibration_lr", train_cfg.get("lr", 1e-3))))
    loader = make_loader(
        calib_data["X"],
        calib_data["y"],
        calib_data.get("amplitude"),
        calib_data.get("phase"),
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    eval_loader = make_loader(
        eval_data["X"],
        eval_data["y"],
        eval_data.get("amplitude"),
        eval_data.get("phase"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    rows = []
    freeze_batchnorm = bool(train_cfg.get("calibration_freeze_batchnorm", True))
    for epoch in range(1, epochs + 1):
        model.train()
        if freeze_batchnorm:
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
        losses = []
        for batch_index, batch in enumerate(loader):
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = forward_batch(model, batch)
            _assert_finite("calibration logits", out["logits"], epoch, batch_index)
            loss = F.cross_entropy(out["logits"], batch["y"]) + regularization_loss(model, out.get("aux", {})).to(device)
            _assert_finite("calibration loss", loss, epoch, batch_index)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(train_cfg.get("grad_clip", 5.0)), error_if_nonfinite=True)
            optimizer.step()
            _assert_finite_parameters(model, epoch, batch_index)
            losses.append(float(loss.detach().cpu()))
        rows.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else np.nan})
    metrics = evaluate(model, eval_loader, device=device, n_classes=int(cfg.get("n_classes", 4)))
    summary = {
        "mode": mode,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "kappa": metrics["kappa"],
        "macro_f1": metrics["macro_f1"],
        "params": count_parameters(model, trainable_only=False),
        "trainable_params": trainable,
    }
    write_csv(output_dir / "calibration_history.csv", rows)
    write_json(output_dir / "metrics.json", summary)
    _atomic_torch_save(model.state_dict(), output_dir / "model_checkpoint.pt")
    return {**summary, "y_true": metrics["y_true"], "y_pred": metrics["y_pred"]}


def unlabeled_adapt(
    model: torch.nn.Module,
    unlabeled_data: dict[str, np.ndarray],
    eval_data: dict[str, np.ndarray],
    cfg: dict[str, Any],
    output_dir: str | Path,
    method: str,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    device = cfg.get("device", "cpu")
    model.to(device)
    mode = cfg.get("calibration_mode", "phase_delay")
    trainable = set_trainable_by_mode(model, mode)
    params = [p for p in model.parameters() if p.requires_grad]
    if method == "bn_adaptation":
        params = []
    train_cfg = cfg.get("training", cfg)
    batch_size = int(train_cfg.get("batch_size", 64))
    epochs = int(train_cfg.get("unlabeled_epochs", max(3, min(15, int(train_cfg.get("epochs", 10))))))
    threshold = float(train_cfg.get("pseudo_label_threshold", 0.75))
    consistency_std = float(train_cfg.get("consistency_noise_std", 0.03))
    loader = DataLoader(
        TrialTensorDataset(
            unlabeled_data["X"],
            np.zeros(len(unlabeled_data["X"]), dtype=np.int64),
            unlabeled_data.get("amplitude"),
            unlabeled_data.get("phase"),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    eval_loader = make_loader(
        eval_data["X"],
        eval_data["y"],
        eval_data.get("amplitude"),
        eval_data.get("phase"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    optimizer = torch.optim.AdamW(params, lr=float(train_cfg.get("unlabeled_lr", train_cfg.get("lr", 1e-3)))) if params else None
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        confidences = []
        kept = []
        for batch_index, batch in enumerate(loader):
            batch = batch_to_device(batch, device)
            if method == "bn_adaptation":
                with torch.no_grad():
                    logits = forward_batch(model, batch)["logits"]
                    confidences.append(float(torch.softmax(logits, dim=1).max(dim=1).values.mean().detach().cpu()))
                continue
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            out = forward_batch(model, batch)
            logits = out["logits"]
            _assert_finite("unlabeled adaptation logits", logits, epoch, batch_index)
            prob = torch.softmax(logits, dim=1)
            if method == "entropy":
                loss = -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=1).mean()
            elif method == "pseudo_label":
                conf, pseudo = prob.max(dim=1)
                mask = conf >= threshold
                kept.append(float(mask.float().mean().detach().cpu()))
                if mask.any():
                    loss = F.cross_entropy(logits[mask], pseudo[mask])
                else:
                    loss = -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=1).mean() * 0.0
            elif method == "consistency":
                noisy = dict(batch)
                noisy["x"] = batch["x"] + consistency_std * torch.randn_like(batch["x"])
                if "amplitude" in batch:
                    amp_scale = batch["amplitude"].detach().std(dim=-1, keepdim=True).clamp_min(1e-3)
                    noisy["amplitude"] = (batch["amplitude"] + consistency_std * amp_scale * torch.randn_like(batch["amplitude"])).clamp_min(0.0)
                if "phase" in batch:
                    noisy["phase"] = batch["phase"] + consistency_std * torch.randn_like(batch["phase"])
                logits_noisy = forward_batch(model, noisy)["logits"]
                loss = F.kl_div(
                    F.log_softmax(logits_noisy, dim=1),
                    prob.detach(),
                    reduction="batchmean",
                )
            else:
                raise ValueError(f"Unknown unlabeled adaptation method: {method}")
            loss = loss + regularization_loss(model, out.get("aux", {})).to(device)
            _assert_finite("unlabeled adaptation loss", loss, epoch, batch_index)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(train_cfg.get("grad_clip", 5.0)), error_if_nonfinite=True)
            optimizer.step()
            _assert_finite_parameters(model, epoch, batch_index)
            losses.append(float(loss.detach().cpu()))
            confidences.append(float(prob.max(dim=1).values.mean().detach().cpu()))
        rows.append(
            {
                "epoch": epoch,
                "method": method,
                "loss": float(np.mean(losses)) if losses else 0.0,
                "mean_confidence": float(np.mean(confidences)) if confidences else np.nan,
                "pseudo_keep_fraction": float(np.mean(kept)) if kept else np.nan,
            }
        )
    metrics = evaluate(model, eval_loader, device=device, n_classes=int(cfg.get("n_classes", 4)))
    summary = {
        "method": method,
        "calibration_mode": mode,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "kappa": metrics["kappa"],
        "macro_f1": metrics["macro_f1"],
        "params": count_parameters(model, trainable_only=False),
        "trainable_params": trainable,
    }
    write_csv(output_dir / "unlabeled_history.csv", rows)
    write_json(output_dir / "metrics.json", summary)
    _atomic_torch_save(model.state_dict(), output_dir / "model_checkpoint.pt")
    return {**summary, "history": rows}
