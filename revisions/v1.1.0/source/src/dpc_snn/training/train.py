"""Training loop used by experiment runners."""

from __future__ import annotations

import os
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dpc_snn.utils.io import ensure_dir, save_npy, save_npz, write_csv, write_json
from dpc_snn.utils.metrics import classification_metrics
from dpc_snn.utils.torch import count_parameters

from .evaluate import evaluate
from .forward import forward_batch, regularization_loss
from .tensor_dataset import TrialTensorDataset, batch_to_device


def _assert_finite(name: str, value: torch.Tensor, epoch: int, batch: int) -> None:
    if bool(torch.isfinite(value).all()):
        return
    finite = value.detach()[torch.isfinite(value.detach())]
    finite_range = (
        "no finite values"
        if finite.numel() == 0
        else f"finite_min={float(finite.min()):.6g}, finite_max={float(finite.max()):.6g}"
    )
    raise FloatingPointError(
        f"Non-finite {name} at epoch={epoch}, batch={batch}; {finite_range}"
    )


def _assert_finite_parameters(model: nn.Module, epoch: int, batch: int) -> None:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError(
                f"Non-finite parameter {name} after optimizer step at epoch={epoch}, batch={batch}"
            )


def _atomic_torch_save(payload: Any, path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_status(status: str, **extra: Any) -> dict[str, Any]:
    return {
        "status": status,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        **extra,
    }


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    amplitude: np.ndarray | None,
    phase: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    delay_evidence_x: np.ndarray | None = None,
) -> DataLoader:
    ds = TrialTensorDataset(
        x=x,
        y=y,
        amplitude=amplitude,
        phase=phase,
        delay_evidence_x=delay_evidence_x,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def _pretrain_delay(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    epochs: int,
    learning_rate: float,
    grad_clip: float,
    gradient_accumulation_steps: int = 1,
) -> list[dict[str, float]]:
    if epochs <= 0 or not hasattr(model, "delay_pretraining_parameters"):
        return []
    parameters = list(model.delay_pretraining_parameters())
    if not parameters:
        return []
    requires_grad = {id(parameter): parameter.requires_grad for parameter in model.parameters()}
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in parameters:
        parameter.requires_grad = True
    if hasattr(model, "set_delay_evidence_updates"):
        model.set_delay_evidence_updates(True)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
    history = []
    for epoch in range(1, epochs + 1):
        if hasattr(model, "synapse") and hasattr(model.synapse, "begin_evidence_epoch"):
            model.synapse.begin_evidence_epoch()
        if hasattr(model, "set_training_progress"):
            model.set_training_progress((epoch - 1) / max(1, epochs - 1), anneal_lag=False)
        # Keep unrelated BatchNorm buffers and reference augmentation frozen;
        # only the synapse needs training mode for evidence EMA updates.
        model.eval()
        if hasattr(model, "synapse"):
            model.synapse.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
            batch = batch_to_device(batch, device)
            out = forward_batch(model, batch)
            loss = model.delay_pretraining_loss(out.get("aux", {}))
            _assert_finite("delay pretraining loss", loss, epoch, batch_index)
            (loss / gradient_accumulation_steps).backward()
            should_step = (
                (batch_index + 1) % gradient_accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    max_norm=grad_clip,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                _assert_finite_parameters(model, epoch, batch_index)
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        if hasattr(model, "synapse") and hasattr(model.synapse, "end_evidence_epoch"):
            model.synapse.end_evidence_epoch()
        history.append(
            {
                "epoch": float(epoch),
                "delay_pretrain_loss": float(np.mean(losses)) if losses else np.nan,
            }
        )
    if hasattr(model, "capture_delay_anchor"):
        model.capture_delay_anchor()
    if hasattr(model, "set_delay_evidence_updates"):
        model.set_delay_evidence_updates(False)
    for parameter in model.parameters():
        parameter.requires_grad = requires_grad[id(parameter)]
    return history


@torch.no_grad()
def _fit_training_route_gain(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Fit one delayed-current RMS on the inner-training fold only."""

    if not hasattr(model, "set_train_fitted_route_rms"):
        return 1.0
    was_training = model.training
    model.eval()
    power = 0.0
    count = 0.0
    for batch in loader:
        output = forward_batch(model, batch_to_device(batch, device))
        aux = output.get("aux", {})
        power += float(aux["route_power_sum"].detach().cpu())
        count += float(aux["route_element_count"].detach().cpu())
    fitted = float(np.sqrt(power / max(count, 1.0)))
    if fitted < 1e-6:
        raise RuntimeError(
            "Inner-training delayed current collapsed before gain fitting; refusing to amplify numerical noise."
        )
    model.set_train_fitted_route_rms(fitted)
    model.train(was_training)
    return fitted


@torch.no_grad()
def _fit_training_event_thresholds(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    quantile: float,
    max_samples_per_band: int = 200_000,
) -> list[float]:
    """Fit per-band envelope-delta thresholds on inner-training trials only."""

    if not hasattr(model, "event_envelope_deltas") or getattr(
        model, "event_encoder", None
    ) is None:
        return []
    if not 0.5 <= float(quantile) < 1.0:
        raise ValueError("event threshold quantile must lie in [0.5, 1.0)")
    was_training = model.training
    model.eval()
    sampled: list[list[torch.Tensor]] | None = None
    per_batch_budget = max(
        256, int(max_samples_per_band) // max(1, len(loader))
    )
    for batch in loader:
        batch = batch_to_device(batch, device)
        delta = model.event_envelope_deltas(batch["x"])
        flat = delta.permute(1, 0, 2, 3).reshape(delta.shape[1], -1).cpu()
        if sampled is None:
            sampled = [[] for _ in range(flat.shape[0])]
        take = min(flat.shape[1], per_batch_budget)
        if take < flat.shape[1]:
            indices = torch.linspace(0, flat.shape[1] - 1, steps=take).long()
            flat = flat.index_select(1, indices)
        for band in range(flat.shape[0]):
            sampled[band].append(flat[band])
    if not sampled or any(not values for values in sampled):
        raise RuntimeError("inner-training event threshold fit received no samples")
    thresholds = torch.stack(
        [
            torch.quantile(torch.cat(values), float(quantile)).clamp_min(1e-5)
            for values in sampled
        ]
    )
    model.set_train_fitted_event_thresholds(thresholds)
    model.train(was_training)
    return [float(value) for value in thresholds]


def _train_model_impl(
    model: nn.Module,
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray],
    cfg: dict[str, Any],
    output_dir: str | Path,
    test_data: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    device = cfg.get("device", "cpu")
    model.to(device)
    train_cfg = cfg.get("training", cfg)
    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 0))
    epochs = int(train_cfg.get("epochs", 100))
    patience = int(train_cfg.get("patience", max(epochs, 1)))
    learning_rate = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    gradient_accumulation_steps = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
    tet_loss_weight = float(train_cfg.get("tet_loss_weight", 0.0))
    criterion = nn.CrossEntropyLoss()
    initial_checkpoint = train_cfg.get("initial_checkpoint")
    if initial_checkpoint:
        state = torch.load(initial_checkpoint, map_location=device, weights_only=True)
        if hasattr(model, "load_representation_state"):
            model.load_representation_state(state)
        else:
            model.load_state_dict(state, strict=True)
    train_loader = make_loader(
        train_data["X"],
        train_data["y"],
        train_data.get("amplitude"),
        train_data.get("phase"),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        delay_evidence_x=train_data.get("delay_evidence_X"),
    )
    val_loader = make_loader(
        val_data["X"],
        val_data["y"],
        val_data.get("amplitude"),
        val_data.get("phase"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        delay_evidence_x=val_data.get("delay_evidence_X"),
    )

    fixed_event_thresholds = train_cfg.get("fixed_event_thresholds")
    if fixed_event_thresholds is None:
        fitted_event_thresholds = _fit_training_event_thresholds(
            model,
            train_loader,
            device,
            quantile=float(train_cfg.get("event_threshold_quantile", 0.90)),
            max_samples_per_band=int(
                train_cfg.get("event_threshold_max_samples_per_band", 200_000)
            ),
        )
    elif hasattr(model, "set_train_fitted_event_thresholds"):
        fitted_event_thresholds = [float(value) for value in fixed_event_thresholds]
        model.set_train_fitted_event_thresholds(
            torch.as_tensor(fitted_event_thresholds)
        )
    else:
        raise ValueError(
            "fixed_event_thresholds were supplied to a model without event encoding"
        )

    delay_pretrain_history = _pretrain_delay(
        model,
        train_loader,
        device,
        epochs=int(train_cfg.get("delay_pretrain_epochs", 0)),
        learning_rate=float(train_cfg.get("delay_pretrain_lr", 5e-3)),
        grad_clip=float(train_cfg.get("grad_clip", 5.0)),
        gradient_accumulation_steps=int(
            train_cfg.get(
                "delay_pretrain_gradient_accumulation_steps",
                gradient_accumulation_steps,
            )
        ),
    )
    write_csv(output_dir / "delay_pretrain_history.csv", delay_pretrain_history)
    if hasattr(model, "begin_task_training"):
        model.begin_task_training()
    fixed_route_rms = train_cfg.get("fixed_route_rms")
    if fixed_route_rms is None:
        fitted_route_rms = _fit_training_route_gain(model, train_loader, device)
    else:
        fitted_route_rms = float(fixed_route_rms)
        if not hasattr(model, "set_train_fitted_route_rms"):
            raise ValueError("fixed_route_rms was supplied to a model without fixed-gain support")
        model.set_train_fitted_route_rms(fitted_route_rms)
    if hasattr(model, "set_training_stage"):
        model.set_training_stage("joint")
    parameters = (
        model.optimizer_parameter_groups(learning_rate, weight_decay)
        if hasattr(model, "optimizer_parameter_groups")
        else model.parameters()
    )
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)

    best_metric = -np.inf
    best_state = None
    stale = 0
    history = []
    optimizer_step_count = 0
    start = time.perf_counter()
    representation_warmup_epochs = min(
        epochs,
        max(0, int(train_cfg.get("representation_warmup_epochs", 0))),
    )
    joint_epochs = epochs - representation_warmup_epochs
    minimum_joint_epochs = int(train_cfg.get("minimum_lag_anneal_joint_epochs", 0))
    if minimum_joint_epochs > 0 and joint_epochs > 0 and joint_epochs < minimum_joint_epochs and not bool(train_cfg.get("allow_short_joint_smoke", False)):
        raise ValueError(
            f"Lag annealing requires at least {minimum_joint_epochs} joint epochs; received {joint_epochs}."
        )
    for epoch in range(1, epochs + 1):
        stage = "representation" if epoch <= representation_warmup_epochs else "joint"
        if hasattr(model, "set_training_stage"):
            model.set_training_stage(stage)
        if hasattr(model, "set_training_progress"):
            if stage == "representation":
                progress = (epoch - 1) / max(1, representation_warmup_epochs - 1)
                model.set_training_progress(progress, anneal_lag=False)
            else:
                joint_epoch = max(0, epoch - representation_warmup_epochs - 1)
                joint_total = max(1, joint_epochs - 1)
                joint_progress = joint_epoch / joint_total
                model.set_training_progress(joint_progress, anneal_lag=True)
                # With no representation stage, the router otherwise remains
                # at progress zero and injects warmup noise for every joint
                # epoch. Do not restart router warmup when representation
                # training has already completed it.
                if (
                    representation_warmup_epochs == 0
                    and hasattr(model, "set_router_training_progress")
                ):
                    model.set_router_training_progress(joint_progress)
        model.train()
        losses = []
        tet_losses = []
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader):
            batch = batch_to_device(batch, device)
            out = forward_batch(model, batch)
            _assert_finite("logits", out["logits"], epoch, batch_index)
            cls_loss = criterion(out["logits"], batch["y"])
            reg_loss = regularization_loss(model, out.get("aux", {})).to(device)
            _assert_finite("classification loss", cls_loss, epoch, batch_index)
            _assert_finite("regularization loss", reg_loss, epoch, batch_index)
            cumulative_logits = out.get("aux", {}).get("cumulative_logits")
            if (
                tet_loss_weight > 0.0
                and isinstance(cumulative_logits, torch.Tensor)
                and cumulative_logits.shape[1] > 0
            ):
                # Every prefix is supervised by the actual trial label. This
                # avoids treating short local windows as complete examples and
                # avoids self-distillation from a potentially wrong final logit.
                prefix_losses = [
                    criterion(cumulative_logits[:, index], batch["y"])
                    for index in range(cumulative_logits.shape[1])
                ]
                tet_loss = torch.stack(prefix_losses).mean()
            else:
                tet_loss = cls_loss.new_zeros(())
            loss = cls_loss + reg_loss + tet_loss_weight * tet_loss
            _assert_finite("total loss", loss, epoch, batch_index)
            (loss / gradient_accumulation_steps).backward()
            should_step = (batch_index + 1) % gradient_accumulation_steps == 0 or batch_index + 1 == len(train_loader)
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(train_cfg.get("grad_clip", 5.0)),
                    error_if_nonfinite=True,
                )
                optimizer.step()
                optimizer_step_count += 1
                _assert_finite_parameters(model, epoch, batch_index)
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            tet_losses.append(float(tet_loss.detach().cpu()))
        val_eval = evaluate(model, val_loader, device=device, n_classes=int(cfg.get("n_classes", cfg.get("model", {}).get("n_classes", 4))))
        row = {
            "epoch": epoch,
            "training_stage": stage,
            "delay_temperature": float(getattr(getattr(model, "synapse", None), "delay_temperature", np.nan)),
            "delay_anchor_kl_scale": float(getattr(model, "delay_anchor_kl_scale", np.nan)),
            "loss": float(np.mean(losses)) if losses else np.nan,
            "tet_loss": float(np.mean(tet_losses)) if tet_losses else np.nan,
            "cumulative_aux_loss": float(np.mean(tet_losses)) if tet_losses else np.nan,
            "val_accuracy": val_eval["accuracy"],
            "val_kappa": val_eval["kappa"],
            "val_macro_f1": val_eval["macro_f1"],
            "val_delay_alignment_loss": val_eval.get("delay_alignment_loss", np.nan),
            "val_delay_entropy": val_eval.get("delay_entropy", np.nan),
        }
        history.append(row)
        write_csv(output_dir / "history.csv", history)
        print(
            "__DS_PROGRESS__ "
            f"epoch={epoch} stage={stage} val_accuracy={row['val_accuracy']:.6f} "
            f"val_kappa={row['val_kappa']:.6f} delay_temperature={row['delay_temperature']:.6f}",
            flush=True,
        )
        metric = row["val_kappa"] if np.isfinite(row["val_kappa"]) else row["val_accuracy"]
        if str(cfg.get("experiment_id", "")) in {"E1", "E2"} and np.isfinite(row["val_delay_alignment_loss"]):
            weight = float(cfg.get("delay_recovery_selection_weight", 1.0))
            metric = metric - weight * row["val_delay_alignment_loss"]
        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        _atomic_torch_save(
            {
                "schema_version": 1,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_state": best_state,
                "best_metric": best_metric,
                "stale": stale,
                "history": history,
            },
            output_dir / "last_epoch_checkpoint.pt",
        )
        if best_state is not None and metric == best_metric:
            _atomic_torch_save(best_state, output_dir / "best_model_checkpoint.pt")
        if stale >= patience:
            break

    if best_state is not None:
        if bool(train_cfg.get("select_last_checkpoint", False)):
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(best_state)

    elapsed = time.perf_counter() - start
    write_csv(output_dir / "history.csv", history)
    # Checkpoints are only interpretable together with their exact training
    # configuration and normalization statistics.
    write_json(output_dir / "run_config.json", cfg)
    if "standardize_mean" in train_data and "standardize_std" in train_data:
        save_npz(
            output_dir / "preprocessing_stats.npz",
            mean=np.asarray(train_data["standardize_mean"], dtype=np.float32),
            std=np.asarray(train_data["standardize_std"], dtype=np.float32),
        )
    if (
        "delay_evidence_standardize_mean" in train_data
        and "delay_evidence_standardize_std" in train_data
    ):
        save_npz(
            output_dir / "delay_evidence_preprocessing_stats.npz",
            mean=np.asarray(train_data["delay_evidence_standardize_mean"], dtype=np.float32),
            std=np.asarray(train_data["delay_evidence_standardize_std"], dtype=np.float32),
        )
    _atomic_torch_save(model.state_dict(), output_dir / "model_checkpoint.pt")
    n_classes = int(cfg.get("n_classes", cfg.get("model", {}).get("n_classes", 4)))
    selection_eval = evaluate(model, val_loader, device=device, n_classes=n_classes)
    if test_data is None:
        final_eval = selection_eval
        evaluation_split = "validation"
    else:
        test_loader = make_loader(
            test_data["X"],
            test_data["y"],
            test_data.get("amplitude"),
            test_data.get("phase"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            delay_evidence_x=test_data.get("delay_evidence_X"),
        )
        final_eval = evaluate(model, test_loader, device=device, n_classes=n_classes)
        evaluation_split = "heldout_test"
    metrics = {
        "model": cfg.get("model_name", cfg.get("model", {}).get("model", "")),
        "implementation_id": getattr(
            model,
            "implementation_id",
            cfg.get("resolved_model_config", {}).get("architecture_version", ""),
        ),
        "dataset": cfg.get("dataset_name", ""),
        "protocol": cfg.get("protocol", cfg.get("experiment", {}).get("name", "")),
        "subject": cfg.get("subject", ""),
        "seed": cfg.get("seed", ""),
        "accuracy": final_eval["accuracy"],
        "balanced_accuracy": final_eval.get("balanced_accuracy", np.nan),
        "kappa": final_eval["kappa"],
        "macro_f1": final_eval["macro_f1"],
        "params": count_parameters(model, trainable_only=False),
        "trainable_params": count_parameters(model, trainable_only=True),
        "train_seconds": elapsed,
        "evaluation_split": evaluation_split,
        "selection_accuracy": selection_eval["accuracy"],
        "selection_balanced_accuracy": selection_eval.get("balanced_accuracy", np.nan),
        "selection_kappa": selection_eval["kappa"],
        "selection_macro_f1": selection_eval["macro_f1"],
        "train_fitted_route_rms": fitted_route_rms,
        "train_fitted_event_thresholds": fitted_event_thresholds,
        "optimizer_steps": int(optimizer_step_count),
        "effective_batch_size": int(batch_size * gradient_accumulation_steps),
    }
    protocol_metadata = cfg.get("protocol_metadata", {})
    if isinstance(protocol_metadata, dict):
        for key in (
            "selection_session",
            "selection_split",
            "evaluation_session",
            "heldout_test_accessed",
            "evidence_role",
            "frozen_architecture_id",
            "evidence_space",
            "evidence_audit_path",
        ):
            if key in protocol_metadata:
                metrics[key] = protocol_metadata[key]
    for key, value in final_eval.items():
        if key.startswith("expert_usage_"):
            metrics[key] = value
    learned = model.export_learned_parameters() if hasattr(model, "export_learned_parameters") else None
    if learned is not None and "delay_prob" in learned and "edge_selection" in learned:
        probability = learned["delay_prob"].float()
        selection = learned["edge_selection"].float()
        selected_mass = selection.sum().clamp_min(1e-6)
        null_probability = learned.get("delay_null_probability")
        if null_probability is None:
            null_probability = probability[..., 0]
        metrics["selected_null_route_mass"] = float(
            (null_probability.float() * selection).sum() / selected_mass
        )
        metrics["selected_base_zero_transport_mass"] = float(
            (probability[..., 0] * selection).sum() / selected_mass
        )
        metrics["selected_mean_delay_steps"] = float(
            (learned["delay_latent"].float() * selection).sum() / selected_mass
        )
        metrics["selected_nonzero_delay_fraction"] = float(
            ((learned["delay_latent"].float() > 0.05).float() * selection).sum()
            / selected_mass
        )
        metrics["selected_exact_zero_delay_fraction"] = float(
            ((learned["delay_latent"].float() <= 0.05).float() * selection).sum()
            / selected_mass
        )
        metrics["selected_zero_delay_mass"] = metrics[
            "selected_exact_zero_delay_fraction"
        ]
        available = float(getattr(getattr(model, "synapse", None), "route_mask", selection).sum())
        metrics["effective_route_density"] = float(selection.sum() / max(available, 1.0))
    write_json(output_dir / "metrics.json", metrics)
    write_csv(output_dir / "metrics.csv", [metrics])
    write_csv(
        output_dir / "selection_metrics.csv",
        [
            {
                "accuracy": selection_eval["accuracy"],
                "kappa": selection_eval["kappa"],
                "macro_f1": selection_eval["macro_f1"],
                "selection_split": "validation",
            }
        ],
    )
    save_npy(output_dir / "confusion_matrix.npy", final_eval["confusion_matrix"])
    save_npy(output_dir / "selection_y_true.npy", selection_eval["y_true"])
    save_npy(output_dir / "selection_y_pred.npy", selection_eval["y_pred"])
    save_npy(output_dir / "selection_logits.npy", selection_eval["logits"])
    save_npy(output_dir / "evaluation_y_true.npy", final_eval["y_true"])
    save_npy(output_dir / "evaluation_y_pred.npy", final_eval["y_pred"])
    save_npy(output_dir / "evaluation_logits.npy", final_eval["logits"])
    if learned is not None:
        arrays = {k: v.numpy() if hasattr(v, "numpy") else np.asarray(v) for k, v in learned.items()}
        save_npz(output_dir / "learned_params.npz", **arrays)
    return {
        "metrics": metrics,
        "history": history,
        "output_dir": str(output_dir),
        "y_true": final_eval["y_true"],
        "y_pred": final_eval["y_pred"],
        "logits": final_eval["logits"],
        "selection_y_true": selection_eval["y_true"],
        "selection_y_pred": selection_eval["y_pred"],
        "selection_logits": selection_eval["logits"],
    }


def train_model(
    model: nn.Module,
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray],
    cfg: dict[str, Any],
    output_dir: str | Path,
    test_data: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Train with durable lifecycle metadata while preserving the exception."""

    output_dir = ensure_dir(output_dir)
    status_path = output_dir / "training_status.json"
    write_json(status_path, _runtime_status("running", started_at=time.time()))
    try:
        result = _train_model_impl(
            model,
            train_data,
            val_data,
            cfg,
            output_dir,
            test_data=test_data,
        )
    except BaseException as exc:
        write_json(
            status_path,
            _runtime_status(
                "failed",
                failed_at=time.time(),
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            ),
        )
        raise
    write_json(status_path, _runtime_status("completed", completed_at=time.time()))
    return result


def evaluate_predictions_to_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: dict[str, Any],
    n_classes: int | None = None,
) -> dict[str, Any]:
    return {**metadata, **classification_metrics(y_true, y_pred, n_classes=n_classes)}
