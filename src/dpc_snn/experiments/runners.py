"""Concrete E0-E24 runners."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from dpc_snn.analysis.connectivity import (
    debiased_weighted_phase_lag_index,
    imaginary_coherence,
    matrix_correlation,
    phase_locking_value,
)
from dpc_snn.analysis.erd_ers import erd_ers
from dpc_snn.analysis.explainability import edge_jaccard
from dpc_snn.analysis.fairness import audit_row
from dpc_snn.analysis.robustness import add_gaussian_noise, channel_dropout, temporal_jitter
from dpc_snn.analysis.statistics import fdr_bh, mixed_effects_or_fallback, wilcoxon_signed_rank
from dpc_snn.data.async_windows import make_pseudo_async_from_trials
from dpc_snn.data.bci2a import load_processed_npz, subject_session_split
from dpc_snn.data.moabb import load_moabb_motor_imagery
from dpc_snn.data.physionet import physionet_task_mapping
from dpc_snn.data.splits import k_shot_indices, leave_one_subject_out, stratified_split_indices
from dpc_snn.data.synthetic import (
    SyntheticDelayPhaseConfig,
    generate_delay_phase_dataset,
    label_shuffle,
    phase_surrogate,
    source_to_scalp_mix,
    time_reverse,
)
from dpc_snn.preprocessing.reference import apply_reference
from dpc_snn.preprocessing.spikes import phase_aware_code
from dpc_snn.preprocessing.standardize import apply_channelwise_zscore, standardize_train_test
from dpc_snn.utils.imports import has_module
from dpc_snn.utils.io import ensure_dir, save_npy, write_csv, write_json
from dpc_snn.utils.metrics import classification_metrics

from .common import (
    DEFAULT_BANDS,
    _feature_bands_from_cfg,
    add_features,
    build_model_for_data,
    dependency_snapshot,
    load_referenced_yaml,
    needs_input,
    resolve_dataset_cfg,
    resolve_model_cfg,
    split_train_val,
    subset,
    train_split,
    write_run_manifest,
)
from .contracts import validate_experiment_contract


def _selected_target_subjects(cfg: dict[str, Any]) -> set[str] | None:
    configured = cfg.get("target_subjects")
    environment = os.environ.get("DPC_TARGET_SUBJECTS", "").strip()
    if environment:
        configured = [value.strip() for value in environment.split(",") if value.strip()]
    if not configured:
        return None
    return {str(value) for value in configured}


def _standardize_pair(train: dict[str, Any], test: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    x_train, x_test, stats = standardize_train_test(train["X"], test["X"])
    out_train = dict(train)
    out_test = dict(test)
    out_train["X"] = x_train
    out_test["X"] = x_test
    out_train["standardize_mean"] = stats["mean"]
    out_train["standardize_std"] = stats["std"]
    return out_train, out_test


def _crop_to_common_task_window(data: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Preserve the declared baseline and task interval before splitting.

    DPC-SNN uses the pre-cue interval only to fit trial-relative ERD/ERS.  The
    classifier still receives the task interval selected inside the model.
    """
    if "epoch_tmin" not in data or "X" not in data:
        return data
    model_cfg = resolve_model_cfg(cfg, "dpc_snn")
    epoch_tmin = float(data["epoch_tmin"])
    sfreq = float(data.get("sfreq", 250.0))
    n_time = int(np.asarray(data["X"]).shape[-1])
    epoch_tmax = float(data.get("epoch_tmax", epoch_tmin + n_time / sfreq))
    task_tmin = max(epoch_tmin, float(model_cfg.get("task_tmin", epoch_tmin)))
    task_tmax = min(epoch_tmax, float(model_cfg.get("task_tmax", epoch_tmax)))
    if task_tmax <= task_tmin:
        raise ValueError(f"Invalid common task window {task_tmin}..{task_tmax} for epoch {epoch_tmin}..{epoch_tmax}")
    baseline_tmin = max(
        epoch_tmin,
        min(task_tmin, float(model_cfg.get("baseline_tmin", task_tmin))),
    )
    start = max(0, min(n_time - 1, int(round((baseline_tmin - epoch_tmin) * sfreq))))
    stop = max(start + 1, min(n_time, int(round((task_tmax - epoch_tmin) * sfreq))))
    out = dict(data)
    out["X"] = np.asarray(data["X"])[..., start:stop]
    out["epoch_tmin"] = baseline_tmin
    out["epoch_tmax"] = baseline_tmin + (stop - start) / sfreq
    return out


def _apply_physical_eeg_space(
    data: dict[str, Any], cfg: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    """Apply a fit-free sensor-space reference before learned spatial filters."""

    model_cfg = resolve_model_cfg(cfg, "dpc_snn")
    mode = str(model_cfg.get("physical_reference", "none")).lower()
    if mode in {"none", "original"}:
        return dict(data), "original_sensor_space"
    out = dict(data)
    if mode in {"car", "common_average"}:
        from dpc_snn.preprocessing.reference import common_average_reference

        out["X"] = common_average_reference(np.asarray(data["X"]))
        return out, "common_average_before_physical_projection"
    if mode == "csd":
        from dpc_snn.analysis.evidence_space import current_source_density

        channel_names = data.get("ch_names")
        if channel_names is None or len(channel_names) != np.asarray(data["X"]).shape[1]:
            raise ValueError("CSD preprocessing requires one valid EEG channel name per sensor")
        out["X"] = current_source_density(
            np.asarray(data["X"]),
            [str(name) for name in channel_names],
            float(data.get("sfreq", 250.0)),
        )
        return out, "spherical_spline_csd_before_physical_projection"
    raise ValueError(f"Unknown physical_reference mode: {mode}")


def _split_protocol_train_validation(
    train_raw: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Fit an inner split and preprocessing using the training session only."""

    train_raw = _crop_to_common_task_window(train_raw, cfg)
    train_raw, physical_space = _apply_physical_eeg_space(train_raw, cfg)
    train_idx, validation_idx = stratified_split_indices(
        np.asarray(train_raw["y"]),
        val_fraction=float(cfg.get("inner_validation_fraction", 0.15)),
        seed=int(cfg.get("seed", 0)),
    )
    if train_idx.size == 0 or validation_idx.size == 0:
        raise ValueError("Protocol training data cannot form non-empty train and validation splits.")
    inner_train = subset(train_raw, train_idx)
    inner_validation = subset(train_raw, validation_idx)
    x_train, x_validation, stats = standardize_train_test(inner_train["X"], inner_validation["X"])
    inner_train["X"] = x_train
    inner_validation["X"] = x_validation
    # Persist the exact fit-on-training-session transform with every checkpoint
    # so later held-out/session latency evaluation can reconstruct its input.
    inner_train["standardize_mean"] = stats["mean"]
    inner_train["standardize_std"] = stats["std"]
    protocol_metadata = cfg.get("protocol_metadata", {})
    selection_split = (
        str(protocol_metadata.get("evaluation_split", "session_T_inner_validation"))
        if isinstance(protocol_metadata, dict)
        else "session_T_inner_validation"
    )
    return (
        inner_train,
        inner_validation,
        {
            "selection_split": str(
                protocol_metadata.get(
                    "selection_split_label", "session_T_stratified_inner_validation"
                )
            ),
            "inner_train_trials": int(train_idx.size),
            "inner_validation_trials": int(validation_idx.size),
            "evaluation_split": selection_split,
            "heldout_test_accessed": False,
            "physical_eeg_space": physical_space,
            "baseline_preserved": float(inner_train.get("epoch_tmin", 0.0)) < float(resolve_model_cfg(cfg, "dpc_snn").get("task_tmin", 0.0)),
        },
    )


def _split_protocol_train_validation_test(
    train_raw: dict[str, Any],
    test_raw: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Apply T-fitted preprocessing to an explicitly requested test session."""

    inner_train, inner_validation, audit = _split_protocol_train_validation(train_raw, cfg)
    test_raw = _crop_to_common_task_window(test_raw, cfg)
    test_raw, test_physical_space = _apply_physical_eeg_space(test_raw, cfg)
    if test_physical_space != audit.get("physical_eeg_space"):
        raise ValueError("Training and held-out sessions used different physical EEG spaces")
    heldout_test = dict(test_raw)
    heldout_test["X"] = apply_channelwise_zscore(
        test_raw["X"],
        inner_train["standardize_mean"],
        inner_train["standardize_std"],
    )
    protocol_metadata = cfg.get("protocol_metadata", {})
    evaluation_split = (
        str(protocol_metadata.get("evaluation_split", "heldout_session_E"))
        if isinstance(protocol_metadata, dict)
        else "heldout_session_E"
    )
    audit = {
        **audit,
        "evaluation_split": evaluation_split,
        "heldout_test_accessed": True,
        "heldout_test_trials": int(len(test_raw["y"])),
    }
    return (
        inner_train,
        inner_validation,
        heldout_test,
        audit,
    )


def _load_processed_if_exists(root: str | Path) -> dict[str, Any] | None:
    root = Path(root)
    if root.exists():
        return load_processed_npz(root)
    return None


def _load_async_window_npz(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    if root.is_dir():
        files = sorted(root.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No async window .npz files found under {root}")
        parts = [_load_async_window_npz(path) for path in files]
        merged: dict[str, Any] = {}
        n_keys = {
            "X",
            "y_binary",
            "y_mi",
            "window_start",
            "event_id",
            "event_start",
            "event_uid",
            "onset_latency_sec",
            "subject",
            "session",
            "recording_id",
        }
        for key in n_keys:
            if all(key in part for part in parts):
                merged[key] = np.concatenate([np.asarray(part[key]) for part in parts], axis=0)
        merged["sfreq"] = parts[0].get("sfreq", 250.0)
        merged["dataset_name"] = root.name
        return merged
    with np.load(root, allow_pickle=True) as npz:
        required = {"X", "y_binary", "y_mi", "window_start"}
        missing = required.difference(npz.files)
        if missing:
            raise ValueError(f"{root} missing async window arrays: {sorted(missing)}")
        out = {key: np.asarray(npz[key]) for key in npz.files}
    out["X"] = np.asarray(out["X"], dtype=np.float32)
    out["y_binary"] = np.asarray(out["y_binary"], dtype=np.int64)
    out["y_mi"] = np.asarray(out["y_mi"], dtype=np.int64)
    out["window_start"] = np.asarray(out["window_start"], dtype=np.int64)
    if "onset_latency_sec" not in out:
        out["onset_latency_sec"] = np.full(len(out["y_binary"]), np.nan, dtype=np.float32)
    if "event_id" not in out:
        out["event_id"] = np.full(len(out["y_binary"]), -1, dtype=np.int64)
    if "event_start" not in out:
        out["event_start"] = np.full(len(out["y_binary"]), -1, dtype=np.int64)
    if "recording_id" not in out:
        out["recording_id"] = np.asarray([root.stem] * len(out["y_binary"]))
    if "event_uid" not in out:
        out["event_uid"] = np.asarray(
            [f"{recording_id}:{event_id}" for recording_id, event_id in zip(out["recording_id"], out["event_id"], strict=False)],
            dtype=str,
        )
    out["sfreq"] = float(np.asarray(out.get("sfreq", 250.0)).item())
    out["dataset_name"] = str(out.get("dataset_name", root.stem))
    return out


def _split_async_recording_train_validation_test(
    data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Hold out complete continuous recordings for validation and testing.

    Randomly splitting overlapping windows would place nearly identical windows
    in training and evaluation. Each subject therefore contributes earlier
    recordings to training, the penultimate recording to selection, and the
    final recording to final evaluation.
    """

    if "recording_id" not in data or "subject" not in data:
        raise ValueError("Real asynchronous data must include subject and recording_id metadata.")
    recordings = np.asarray(data["recording_id"]).astype(str)
    subjects = np.asarray(data["subject"]).astype(str)
    train_recordings: list[str] = []
    validation_recordings: list[str] = []
    test_recordings: list[str] = []
    for subject in sorted(set(subjects)):
        subject_recordings = sorted(set(recordings[subjects == subject]))
        if len(subject_recordings) < 3:
            raise ValueError(
                f"Subject {subject} has only {len(subject_recordings)} continuous recordings; "
                "at least three are required for run-level train/validation/test isolation."
            )
        train_recordings.extend(subject_recordings[:-2])
        validation_recordings.append(subject_recordings[-2])
        test_recordings.append(subject_recordings[-1])
    train_idx = np.where(np.isin(recordings, train_recordings))[0]
    validation_idx = np.where(np.isin(recordings, validation_recordings))[0]
    test_idx = np.where(np.isin(recordings, test_recordings))[0]
    if min(train_idx.size, validation_idx.size, test_idx.size) == 0:
        raise ValueError("Could not form non-empty asynchronous train/validation/test recording splits.")
    train = subset(data, train_idx)
    validation = subset(data, validation_idx)
    test = subset(data, test_idx)
    if len(np.unique(train["y"])) < 2 or len(np.unique(validation["y"])) < 2 or len(np.unique(test["y"])) < 2:
        raise ValueError("Every asynchronous recording split must contain both classes.")
    x_train, x_validation, stats = standardize_train_test(train["X"], validation["X"])
    train["X"] = x_train
    validation["X"] = x_validation
    test["X"] = apply_channelwise_zscore(test["X"], stats["mean"], stats["std"])
    return (
        train,
        validation,
        test,
        {
            "selection_split": "heldout_recording_validation",
            "test_split": "heldout_recording_test",
            "train_recordings": train_recordings,
            "validation_recordings": validation_recordings,
            "test_recordings": test_recordings,
        },
    )


def _load_external_processed(cfg: dict[str, Any], key: str) -> dict[str, Any] | None:
    dcfg = resolve_dataset_cfg(cfg, key=key)
    root_value = dcfg.get("processed_root") or (dcfg.get("root") if dcfg.get("kind") == "processed_npz" else "")
    root = Path(root_value) if root_value else Path("__missing__")
    if root.exists():
        data = load_processed_npz(root)
        data["dataset_name"] = dcfg.get("name", key.replace("_config", ""))
        return data
    return None


def _clone_model_from_checkpoint(model_name: str, data: dict[str, Any], cfg: dict[str, Any], checkpoint: str | Path | None = None):
    model, featured, run_cfg, _ = _clone_model_with_checkpoint_audit(model_name, data, cfg, checkpoint)
    return model, featured, run_cfg


def _metric_row(dataset: str, protocol: str, subject: str, model: str, seed: Any, metrics: dict[str, Any], **extra: Any) -> dict[str, Any]:
    keys = [
        "accuracy",
        "balanced_accuracy",
        "kappa",
        "macro_f1",
        "params",
        "trainable_params",
        "train_seconds",
        "evaluation_split",
        "classical_implementation",
        "implementation_id",
        "spike_rate",
        "synops_proxy",
        "cpu_latency_ms",
        "gpu_latency_ms",
        "peak_memory_mb",
        "delay_corr",
        "delay_mae",
        "edge_stability",
        "occlusion_drop",
    ]
    row = {"dataset": dataset, "protocol": protocol, "subject": subject, "model": model, "seed": seed}
    for key in keys:
        if key in metrics:
            row[key] = metrics[key]
    for key, value in metrics.items():
        if key.startswith("expert_usage_") or key in {
            "selected_zero_delay_mass",
            "selected_null_route_mass",
            "selected_base_zero_transport_mass",
            "selected_exact_zero_delay_fraction",
            "selected_mean_delay_steps",
            "selected_nonzero_delay_fraction",
            "effective_route_density",
        }:
            row[key] = value
    row.update(extra)
    return row


def _model_output_name(model_name: str) -> str:
    return model_name.lower().replace("+", "_").replace("-", "_")


def _normalize_model_name(model_name: str) -> str:
    name = model_name.lower().replace("-", "_")
    aliases = {
        "csp": "csp_lda",
        "csplda": "csp_lda",
        "csp_lda": "csp_lda",
        "graph_snn": "graph_snn_no_delay",
    }
    return aliases.get(name, name)


def _benchmark_model_names(
    cfg: dict[str, Any],
    include_classical: bool = True,
    configured_models: list[str] | None = None,
) -> list[str]:
    configured = configured_models or cfg.get("benchmark_models") or cfg.get("models")
    names: list[str] = []
    if configured:
        names.extend(str(item) for item in configured)
    model_cfg = cfg.get("model", {})
    baselines_path = model_cfg.get("baselines_config") if isinstance(model_cfg, dict) else None
    if not names and baselines_path:
        for item in load_referenced_yaml(baselines_path).get("baselines", []):
            if isinstance(item, dict) and item.get("name"):
                names.append(str(item["name"]))
    if not names:
        names = ["dpc_snn", "eegnet", "vanilla_snn", "graph_snn_no_delay", "csp_lda"]
    if "dpc_snn" not in {_normalize_model_name(name) for name in names}:
        names.insert(0, "dpc_snn")
    out: list[str] = []
    for name in names:
        norm = _normalize_model_name(name)
        if norm == "csp_lda" and not include_classical:
            continue
        if norm not in out:
            out.append(norm)
    return out


def _checkpoint_load_audit(model: Any, checkpoint: str | Path) -> dict[str, Any]:
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    current = model.state_dict()
    matched = {
        key: value
        for key, value in state.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    model.load_state_dict(matched, strict=False)
    loaded_params = int(sum(value.numel() for value in matched.values()))
    model_params = int(sum(value.numel() for value in current.values()))
    checkpoint_params = int(sum(value.numel() for value in state.values()))
    return {
        "checkpoint_tensors": len(state),
        "model_tensors": len(current),
        "matched_tensors": len(matched),
        "checkpoint_params": checkpoint_params,
        "model_params": model_params,
        "loaded_params": loaded_params,
        "loaded_param_ratio": float(loaded_params / max(1, model_params)),
    }


def _clone_model_with_checkpoint_audit(
    model_name: str,
    data: dict[str, Any],
    cfg: dict[str, Any],
    checkpoint: str | Path | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    from dpc_snn.utils.torch import resolve_device

    checkpoint_cfg: dict[str, Any] = {}
    checkpoint_config_path: Path | None = None
    if checkpoint is not None:
        checkpoint_config_path = Path(checkpoint).parent / "run_config.json"
        if checkpoint_config_path.exists():
            checkpoint_cfg = json.loads(checkpoint_config_path.read_text(encoding="utf-8"))
    structure_cfg = checkpoint_cfg or cfg
    featured = add_features(data, _feature_bands_from_cfg(structure_cfg))
    resolved_model_cfg = structure_cfg.get("resolved_model_config")
    model = build_model_for_data(
        model_name,
        featured,
        structure_cfg,
        copy.deepcopy(resolved_model_cfg) if isinstance(resolved_model_cfg, dict) else None,
    )
    audit = {
        "checkpoint_tensors": 0,
        "model_tensors": len(model.state_dict()),
        "matched_tensors": 0,
        "checkpoint_params": 0,
        "model_params": int(sum(value.numel() for value in model.state_dict().values())),
        "loaded_params": 0,
        "loaded_param_ratio": 0.0,
        "checkpoint_config_loaded": bool(checkpoint_cfg),
        "checkpoint_config_path": str(checkpoint_config_path or ""),
    }
    if checkpoint:
        audit = {**audit, **_checkpoint_load_audit(model, checkpoint)}
    run_cfg = copy.deepcopy(cfg)
    if checkpoint_cfg:
        run_cfg["resolved_model_config"] = copy.deepcopy(
            checkpoint_cfg.get("resolved_model_config", {})
        )
    run_cfg["device"] = resolve_device(str(cfg.get("device", "cpu")))
    run_cfg["n_classes"] = int(np.max(data["y"]) + 1)
    run_cfg["model_name"] = model_name
    run_cfg["dataset_name"] = str(data.get("dataset_name", cfg.get("dataset_name", "")))
    return model, featured, run_cfg, audit


def _evaluate_model_on_data(model: Any, data: dict[str, Any], cfg: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Any]:
    from dpc_snn.training.evaluate import evaluate
    from dpc_snn.training.train import make_loader
    from dpc_snn.utils.torch import count_parameters, resolve_device

    featured = add_features(data, _feature_bands_from_cfg(cfg))
    run_cfg = copy.deepcopy(cfg)
    device = resolve_device(str(cfg.get("device", "cpu")))
    run_cfg["device"] = device
    n_classes = int(getattr(model, "n_classes", 0) or cfg.get("n_classes", 0) or (np.max(data["y"]) + 1))
    train_cfg = cfg.get("training", cfg)
    loader = make_loader(
        featured["X"],
        featured["y"],
        featured.get("amplitude"),
        featured.get("phase"),
        batch_size=int(train_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    model.to(device)
    metrics = evaluate(model, loader, device=device, n_classes=n_classes)
    summary = {
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics.get("balanced_accuracy", np.nan),
        "kappa": metrics["kappa"],
        "macro_f1": metrics["macro_f1"],
        "params": count_parameters(model, trainable_only=False),
        "trainable_params": count_parameters(model, trainable_only=True),
    }
    if output_dir is not None:
        output_dir = ensure_dir(output_dir)
        write_json(output_dir / "metrics.json", summary)
        write_csv(output_dir / "metrics.csv", [summary])
        save_npy(output_dir / "confusion_matrix.npy", metrics["confusion_matrix"])
    return {**summary, "y_true": metrics["y_true"], "y_pred": metrics["y_pred"]}


def _evaluate_checkpoint_on_data(
    model_name: str,
    data: dict[str, Any],
    cfg: dict[str, Any],
    checkpoint: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    model, _, _, audit = _clone_model_with_checkpoint_audit(model_name, data, cfg, checkpoint)
    result = _evaluate_model_on_data(model, data, cfg, output_dir)
    summary = {key: value for key, value in result.items() if key not in {"y_true", "y_pred"}}
    summary.update({f"checkpoint_{key}": value for key, value in audit.items()})
    write_json(output_dir / "metrics.json", summary)
    write_csv(output_dir / "metrics.csv", [summary])
    return {**result, **summary}


def _evaluate_classical_baseline(
    model_name: str,
    train: dict[str, Any],
    test: dict[str, Any],
    cfg: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    from dpc_snn.baselines.classical import FBCSPLDA, RiemannianLogisticRegression

    output_dir = ensure_dir(output_dir)
    baseline_cfg = cfg.get("baseline", {})
    if model_name == "csp_lda":
        model = FBCSPLDA(
            sfreq=float(train["sfreq"]),
            bands=_feature_bands_from_cfg(cfg),
            n_components=int(baseline_cfg.get("csp_components", 4)),
        )
        implementation = "filter_bank_one_vs_rest_csp_shrinkage_lda"
    elif model_name == "riemann_lr":
        model = RiemannianLogisticRegression(
            covariance_estimator=str(baseline_cfg.get("riemann_covariance_estimator", "oas")),
            c=float(baseline_cfg.get("riemann_logreg_c", 1.0)),
            max_iter=int(baseline_cfg.get("riemann_max_iter", 1000)),
        )
        implementation = "pyriemann_riemannian_tangent_space_logistic_regression"
    else:
        raise KeyError(f"Unsupported classical baseline: {model_name}")
    model.fit(train["X"], train["y"])
    y_pred = model.predict(test["X"])
    n_classes = int(max(np.max(train["y"]), np.max(test["y"])) + 1)
    metrics = classification_metrics(test["y"], y_pred, n_classes=n_classes)
    summary = {
        **metrics,
        "params": 0,
        "trainable_params": 0,
        "train_seconds": np.nan,
        "evaluation_split": "heldout_test",
        "classical_implementation": implementation,
    }
    write_json(output_dir / "metrics.json", summary)
    write_csv(output_dir / "metrics.csv", [summary])
    return {**summary, "y_true": np.asarray(test["y"]), "y_pred": y_pred}


def _run_train_test_model(
    model_name: str,
    train_raw: dict[str, Any],
    test_raw: dict[str, Any],
    cfg: dict[str, Any],
    output_dir: str | Path,
    dataset_name: str,
    protocol: str,
    subject: str,
    model_cfg: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    run_cfg = copy.deepcopy(cfg)
    selection_session = extra.get("selection_session", extra.get("train_session", "T"))
    evaluation_session = extra.get("evaluation_session", extra.get("test_session", "E"))
    evaluation_split = extra.get("evaluation_split", "heldout_session_E")
    run_cfg.update(
        {
            "dataset_name": dataset_name,
            "protocol": protocol,
            "subject": subject,
            "protocol_metadata": {
                "selection_session": selection_session,
                "selection_split": "inner_validation",
                "selection_split_label": extra.get(
                    "selection_split_label", "session_T_stratified_inner_validation"
                ),
                "evaluation_session": evaluation_session,
                "evaluation_split": evaluation_split,
                "heldout_test_accessed": True,
            },
        }
    )
    run_cfg["protocol_metadata"].update(
        {
            key: extra[key]
            for key in (
                "evidence_role",
                "frozen_architecture_id",
                "evidence_space",
                "evidence_audit_path",
            )
            if key in extra
        }
    )
    train, validation, test, split_audit = _split_protocol_train_validation_test(
        copy.deepcopy(train_raw), copy.deepcopy(test_raw), run_cfg
    )
    model_name = _normalize_model_name(model_name)
    model_dir = ensure_dir(Path(output_dir) / _model_output_name(model_name))
    if model_name == "csp_lda":
        result = _evaluate_classical_baseline(model_name, train, test, run_cfg, model_dir)
        metrics = result
    elif model_name == "riemann_lr":
        result = _evaluate_classical_baseline(model_name, train, test, run_cfg, model_dir)
        metrics = result
    else:
        result = train_split(
            model_name,
            train,
            validation,
            run_cfg,
            model_dir,
            model_cfg or resolve_model_cfg(run_cfg, model_name),
            test=test,
        )
        metrics = result["metrics"]
    row_extra = {**split_audit, **extra}
    return _metric_row(
        dataset_name,
        protocol,
        subject,
        model_name,
        run_cfg.get("seed", 0),
        metrics,
        status="completed",
        evaluated_on_heldout_test=True,
        **row_extra,
    )


def _run_train_validation_model(
    model_name: str,
    train_raw: dict[str, Any],
    cfg: dict[str, Any],
    output_dir: str | Path,
    dataset_name: str,
    protocol: str,
    subject: str,
    model_cfg: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Train/select on Session T without loading or evaluating Session E."""

    run_cfg = copy.deepcopy(cfg)
    run_cfg.update(
        {
            "dataset_name": dataset_name,
            "protocol": protocol,
            "subject": subject,
            "protocol_metadata": {
                "selection_session": "T",
                "selection_split": "inner_validation",
                "evaluation_session": None,
                "heldout_test_accessed": False,
            },
        }
    )
    run_cfg["protocol_metadata"].update(
        {
            "evidence_role": extra.get("evidence_role", "training_selection_only"),
            **{
                key: extra[key]
                for key in ("evidence_space", "evidence_audit_path")
                if key in extra
            },
        }
    )
    train, validation, split_audit = _split_protocol_train_validation(
        copy.deepcopy(train_raw), run_cfg
    )
    model_name = _normalize_model_name(model_name)
    model_dir = ensure_dir(Path(output_dir) / _model_output_name(model_name))
    if model_name in {"csp_lda", "riemann_lr"}:
        metrics = _evaluate_classical_baseline(
            model_name, train, validation, run_cfg, model_dir
        )
        metrics["evaluation_split"] = "validation"
    else:
        result = train_split(
            model_name,
            train,
            validation,
            run_cfg,
            model_dir,
            model_cfg or resolve_model_cfg(run_cfg, model_name),
            test=None,
        )
        metrics = result["metrics"]
    row_extra = {**split_audit, **extra}
    return _metric_row(
        dataset_name,
        protocol,
        subject,
        model_name,
        run_cfg.get("seed", 0),
        metrics,
        status="completed",
        evaluated_on_heldout_test=False,
        **row_extra,
    )


def _common_channel_alignment(
    source: dict[str, Any],
    target: dict[str, Any],
    min_common: int = 3,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_names = source.get("ch_names")
    target_names = target.get("ch_names")
    if source_names is None or target_names is None or len(source_names) == 0 or len(target_names) == 0:
        return source, target, {"status": "missing_ch_names", "common_channels": 0}
    source_norm = {_canonical_eeg_channel_name(name): idx for idx, name in enumerate(source_names)}
    target_norm = {_canonical_eeg_channel_name(name): idx for idx, name in enumerate(target_names)}
    if len(source_norm) != len(source_names) or len(target_norm) != len(target_names):
        return source, target, {"status": "ambiguous_channel_names", "common_channels": 0}
    common = [name for name in source_norm if name in target_norm]
    if len(common) < min_common:
        return source, target, {"status": "insufficient_common_channels", "common_channels": len(common)}
    source_idx = np.asarray([source_norm[name] for name in common], dtype=int)
    target_idx = np.asarray([target_norm[name] for name in common], dtype=int)
    out_source = dict(source)
    out_target = dict(target)
    out_source["X"] = source["X"][:, source_idx, :]
    out_target["X"] = target["X"][:, target_idx, :]
    out_source["ch_names"] = common
    out_target["ch_names"] = common
    return out_source, out_target, {"status": "aligned_by_name", "common_channels": len(common), "channels": common}


def _canonical_eeg_channel_name(name: Any) -> str:
    """Normalize non-semantic display differences in standard EEG channel labels."""

    value = str(name).strip().upper()
    value = re.sub(r"^EEG[ _-]*", "", value)
    value = value.rstrip(".")
    return re.sub(r"[^A-Z0-9]", "", value)


def _analysis_scope_root(output_dir: str | Path, cfg: dict[str, Any], key: str = "analysis_results_root") -> Path:
    value = cfg.get(key) or cfg.get("results_root")
    return Path(value) if value else Path(output_dir).parent


def _connectivity_rows(matrices: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for name, mat in matrices.items():
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                rows.append({"metric": name, "source_channel": i, "target_channel": j, "value": float(mat[i, j])})
    return rows


def _collapse_band_pair_matrix(value: np.ndarray, diagonal_only: bool = False) -> np.ndarray:
    """Collapse [target_band, source_band, channel, channel] without losing channels."""
    value = np.asarray(value)
    if value.ndim == 3:
        return value.mean(axis=0)
    if value.ndim != 4:
        raise ValueError(f"Expected 3D/4D learned matrix, got shape {value.shape}")
    if diagonal_only:
        return np.stack([value[b, b] for b in range(min(value.shape[:2]))]).mean(axis=0)
    return value.mean(axis=(0, 1))


def _require_torch(output_dir: str | Path, cfg: dict[str, Any], context: str) -> dict[str, Any] | None:
    if has_module("torch"):
        return None
    return needs_input(output_dir, cfg, f"{context} requires PyTorch. Install with: pip install -e \".[all]\"")


def run_environment_audit(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    data_roots = {
        "raw": str(Path("data/raw").resolve()),
        "processed": str(Path("data/processed").resolve()),
    }
    payload = {
        "status": "completed",
        "dependencies": dependency_snapshot(),
        "data_roots": data_roots,
        "config": cfg,
    }
    write_json(output_dir / "environment.yaml", payload)
    write_json(output_dir / "dataset_manifest.json", {"data_roots": data_roots, "note": "Fill hashes after dataset download."})
    write_csv(output_dir / "preprocess_check.csv", [{"check": "processed_npz_schema", "status": "pending_until_data_present"}])
    write_json(output_dir / "seed_config.yaml", {"seed": cfg.get("seed", 0), "seeds": cfg.get("seeds", [0, 1, 2, 3, 4])})
    write_run_manifest(output_dir, cfg, "completed")
    return payload


def _synthetic_cfg_from_runner(cfg: dict[str, Any]) -> SyntheticDelayPhaseConfig:
    dcfg = resolve_dataset_cfg(cfg, key="synthetic_config")
    return SyntheticDelayPhaseConfig(
        n_trials=int(dcfg.get("n_trials", 512)),
        n_channels=int(dcfg.get("n_channels", 22)),
        n_time=int(dcfg.get("n_time", 256)),
        n_classes=int(dcfg.get("n_classes", 4)),
        sfreq=float(dcfg.get("sfreq", 250)),
        max_delay=int(dcfg.get("max_delay", 16)),
        noise_std=float(dcfg.get("noise_std", 0.35)),
        style_std=float(dcfg.get("style_std", 0.15)),
        phase_jitter=float(dcfg.get("phase_jitter", 0.25)),
        seed=int(cfg.get("seed", dcfg.get("seed", 0))),
        bands=tuple(float(value) for value in dcfg.get("carrier_frequencies_hz", (10.0, 18.0))),
    )


def _synthetic_delay_model_cfg(cfg: dict[str, Any], data: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """Resolve V3 graph-delay bins from a raw-sample synthetic target."""

    model_cfg = resolve_model_cfg(cfg, "dpc_snn")
    n_time = int(np.asarray(data["X"]).shape[-1])
    max_delay_samples = float(np.max(np.asarray(data["delay_gt"])))
    sfreq = float(data.get("sfreq", 250.0))
    # Never choose graph resolution from the delay bins themselves.  Doing so
    # previously reduced E1 to 31.25 Hz and aliased its 20 Hz carrier, making
    # 32-sample delays indistinguishable from shorter periodic lags.  The
    # fractional-delay synapse no longer requires GT delays to land on integers.
    graph_rate_hz = min(sfreq, max(125.0, float(model_cfg.get("graph_rate_hz", 125.0))))
    graph_timesteps = max(2, int(round((n_time / sfreq) * graph_rate_hz)))
    raw_samples_per_step = float(n_time) / graph_timesteps
    model_cfg["graph_timesteps"] = graph_timesteps
    model_cfg["graph_rate_hz"] = sfreq / raw_samples_per_step
    model_cfg["d_max"] = max(1, int(math.ceil(max_delay_samples / raw_samples_per_step)))
    # Mechanism recovery needs a one-to-one latent/sensor graph and bands that
    # match the generator's 10/18 Hz oscillators; compression is evaluated in
    # the separate scalp-mixing control rather than silently changing targets.
    model_cfg["latent_nodes"] = int(np.asarray(data["X"]).shape[1])
    model_cfg["spatial_max_deviation"] = 0.0
    model_cfg["identity_spatial_projection"] = True
    model_cfg["euclidean_alignment"] = False
    model_cfg["reference_augmentation_prob"] = 0.0
    model_cfg["band_edges_hz"] = [[8.0, 12.0], [16.0, 20.0]]
    model_cfg["n_bands"] = 2
    return model_cfg, raw_samples_per_step


def run_synthetic_recovery(cfg: dict[str, Any], output_dir: str | Path, mode: str = "normal") -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    torch_status = _require_torch(output_dir, cfg, f"E1/E2/E3 synthetic training ({mode})")
    if torch_status:
        return torch_status
    scfg = _synthetic_cfg_from_runner(cfg)
    data = generate_delay_phase_dataset(scfg)
    if mode == "scalp_mixing":
        data = source_to_scalp_mix(data, seed=int(cfg.get("seed", 0)))
        save_npy(output_dir / "mixing_matrix.npy", data["mixing_matrix"])
        save_npy(output_dir / "source_delay_gt.npy", data["delay_gt"])
    elif mode == "phase_surrogate":
        data["X"] = phase_surrogate(data["X"], seed=int(cfg.get("seed", 0)))
    elif mode == "label_shuffle":
        # Use an independent random stream. Reusing the generator seed can
        # create deterministic residual correspondence with its label shuffle.
        data["y"] = label_shuffle(data["y"], seed=int(cfg.get("seed", 0)) + 1_000_003)
    elif mode == "time_reversal":
        data = time_reverse(data)
    save_npy(output_dir / "delay_gt.npy", data["delay_gt"])
    save_npy(output_dir / "delay_gt_global.npy", data["delay_gt"])
    save_npy(output_dir / "edge_gt.npy", data["edge_gt"])

    train, val = split_train_val(data, seed=int(cfg.get("seed", 0)))
    model_cfg, raw_samples_per_step = _synthetic_delay_model_cfg(cfg, data)
    result = train_split("dpc_snn", train, val, cfg, output_dir, model_cfg)

    learned_path = output_dir / "learned_params.npz"
    corr = float("nan")
    mae = float("nan")
    mean_corr = float("nan")
    mean_mae = float("nan")
    if learned_path.exists():
        with np.load(learned_path) as learned:
            delay_estimator = "posterior_map" if "delay_map" in learned else "posterior_mean"
            delay_key = "delay_map" if "delay_map" in learned else "delay"
            delay_learned_steps = _collapse_band_pair_matrix(
                learned[delay_key], diagonal_only=True
            )
            delay_mean_steps = _collapse_band_pair_matrix(
                learned["delay"], diagonal_only=True
            )
            delay_probability = np.asarray(learned["delay_prob"])
            null_probability = np.asarray(
                learned["delay_null_probability"]
                if "delay_null_probability" in learned
                else delay_probability[..., 0]
            )
            diagonal_null_probability = np.stack(
                [null_probability[band, band] for band in range(null_probability.shape[0])]
            ).mean(axis=0)
            diagonal_base_zero_probability = np.stack(
                [
                    delay_probability[band, band, ..., 0]
                    for band in range(delay_probability.shape[0])
                ]
            ).mean(axis=0)
            active_edge_mask = np.asarray(data["edge_gt"], dtype=bool)
            null_route_mass = float(
                diagonal_null_probability[active_edge_mask].mean()
            )
            base_zero_transport_mass = float(
                diagonal_base_zero_probability[active_edge_mask].mean()
            )
            selection_key = "edge_selection" if "edge_selection" in learned else "delay_gate"
            selection = np.asarray(learned[selection_key])
            route_available = np.ones(selection.shape, dtype=bool)
            for band in range(selection.shape[0]):
                np.fill_diagonal(route_available[band, band], False)
            if not bool(model_cfg.get("use_cross_band_routes", True)):
                route_available &= np.eye(selection.shape[0], dtype=bool)[:, :, None, None]
            route_density = float(selection.sum() / max(1, route_available.sum()))
        delay_learned = delay_learned_steps * raw_samples_per_step
        delay_mean = delay_mean_steps * raw_samples_per_step
        gt = np.asarray(data["delay_gt"], dtype=np.float32)
        active_edge_mask = np.asarray(data["edge_gt"], dtype=bool)
        corr = matrix_correlation(gt, delay_learned, mask=active_edge_mask)
        mae = float(np.abs(gt[active_edge_mask] - delay_learned[active_edge_mask]).mean())
        mean_corr = matrix_correlation(gt, delay_mean, mask=active_edge_mask)
        mean_mae = float(np.abs(gt[active_edge_mask] - delay_mean[active_edge_mask]).mean())
        zero_delay_mass = float(
            (delay_learned_steps[active_edge_mask] <= 0.05).mean()
        )
        save_npy(output_dir / "delay_learned.npy", delay_learned)
        save_npy(output_dir / "delay_learned_steps.npy", delay_learned_steps)
        save_npy(output_dir / "delay_learned_mean.npy", delay_mean)
    else:
        zero_delay_mass = float("nan")
        null_route_mass = float("nan")
        base_zero_transport_mass = float("nan")
        route_density = float("nan")
    row = {
        **result["metrics"],
        "dataset": "synthetic_delay_phase",
        "protocol": mode,
        "model": "dpc_snn",
        "seed": cfg.get("seed", 0),
        "delay_corr": corr,
        "delay_corr_global": corr,
        "delay_mae": mae,
        "delay_mae_global": mae,
        "delay_corr_posterior_mean": mean_corr,
        "delay_mae_posterior_mean": mean_mae,
        "delay_target_scope": "single_global_active_edge_graph_raw_samples",
        "delay_raw_samples_per_graph_step": raw_samples_per_step,
        "delay_dmax_graph_steps": int(model_cfg["d_max"]),
        "delay_estimator": delay_estimator if learned_path.exists() else "unavailable",
        "zero_delay_mass": zero_delay_mass,
        "null_route_mass": null_route_mass,
        "base_zero_transport_mass": base_zero_transport_mass,
        "effective_route_density": route_density,
        # Backward-compatible aliases retained for older table loaders.
        "delay_raw_samples_per_snn_step": raw_samples_per_step,
        "delay_dmax_snn_steps": int(model_cfg["d_max"]),
    }
    write_json(
        output_dir / "delay_evaluation_contract.json",
        {
            "primary_delay_metric": "delay_corr_global",
            "delay_estimator": "posterior_map",
            "target_scope": "single_global_active_edge_graph",
            "ground_truth_unit": "raw_eeg_samples",
            "model_unit": "compressed_snn_steps",
            "raw_samples_per_snn_step": raw_samples_per_step,
            "active_edges": int(np.asarray(data["edge_gt"], dtype=bool).sum()),
            "evaluation_mask": "active_edge_gt_only",
        },
    )
    if mode == "scalp_mixing":
        row["delay_target_scope"] = "source_graph_reference_not_identifiable_from_mixed_scalp"
        row["delay_corr"] = np.nan
        row["delay_mae"] = np.nan
        write_csv(output_dir / "common_source_control.csv", [row])
        save_npy(output_dir / "scalp_delay_learned.npy", delay_learned if learned_path.exists() else np.zeros_like(data["delay_gt"]))
    out_name = "synthetic_results.csv" if mode == "normal" else f"{mode}_results.csv"
    write_csv(output_dir / out_name, [row])
    write_csv(output_dir / "delay_corr.csv", [{"delay_corr": row["delay_corr"], "delay_mae": row["delay_mae"]}])
    write_run_manifest(output_dir, cfg, "completed", {"metrics": row})
    return row


def run_scalp_mixing(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    return run_synthetic_recovery(cfg, output_dir, mode="scalp_mixing")


def run_negative_controls(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    phase_row = run_synthetic_recovery(cfg, output_dir / "phase_surrogate", mode="phase_surrogate")
    label_row = run_synthetic_recovery(cfg, output_dir / "label_shuffle", mode="label_shuffle")
    reversal_row = run_synthetic_recovery(cfg, output_dir / "time_reversal", mode="time_reversal")
    write_csv(output_dir / "phase_surrogate_results.csv", [phase_row])
    write_csv(output_dir / "label_shuffle_results.csv", [label_row])
    write_csv(output_dir / "time_reversal_results.csv", [reversal_row])
    write_run_manifest(
        output_dir,
        cfg,
        "completed",
        {"phase_surrogate": phase_row, "label_shuffle": label_row, "time_reversal": reversal_row},
    )
    return {"phase_surrogate": phase_row, "label_shuffle": label_row, "time_reversal": reversal_row}


def run_preprocessing_qc(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    data = generate_delay_phase_dataset(_synthetic_cfg_from_runner(cfg))
    featured = add_features(data, DEFAULT_BANDS)
    spikes = phase_aware_code(featured["amplitude"][:8], featured["phase"][:8])
    rows = []
    for band_idx, band in enumerate(featured["band_names"]):
        rows.append(
            {
                "band": str(band),
                "amplitude_mean": float(featured["amplitude"][:, band_idx].mean()),
                "amplitude_std": float(featured["amplitude"][:, band_idx].std()),
                "phase_min": float(featured["phase"][:, band_idx].min()),
                "phase_max": float(featured["phase"][:, band_idx].max()),
                "spike_rate": float(spikes[:, band_idx].mean()),
            }
        )
    write_csv(output_dir / "preprocess_qc.csv", rows)
    write_csv(output_dir / "spike_raster.csv", _spike_rows(spikes[0]))
    write_csv(output_dir / "phase_encoder_debug.csv", rows)
    write_run_manifest(output_dir, cfg, "completed")
    return {"rows": rows}


def _spike_rows(spike: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for b in range(spike.shape[0]):
        for c in range(spike.shape[1]):
            active = np.where(spike[b, c] > 0)[0]
            rows.append({"band": b, "channel": c, "spike_indices": " ".join(map(str, active.tolist()))})
    return rows


def _load_bci_data_or_status(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any] | None:
    dcfg = resolve_dataset_cfg(cfg, key="bci2a_config")
    root = Path(dcfg.get("root", "data/processed/bci2a"))
    if not root.exists():
        needs_input(output_dir, cfg, f"BCI2a processed data not found at {root}. Prepare NPZ files first.")
        return None
    return load_processed_npz(root)


def _target_session_unlabeled_eval_indices(target_data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Define E9 target adaptation/evaluation partitions without target labels."""

    sessions = np.asarray(target_data.get("session", np.asarray([]))).astype(str)
    unlabeled_idx = np.where(sessions == "T")[0]
    evaluation_idx = np.where(sessions == "E")[0]
    if unlabeled_idx.size == 0 or evaluation_idx.size == 0:
        raise ValueError("Target data requires non-empty T and E sessions for unlabeled adaptation.")
    return unlabeled_idx, evaluation_idx


def run_bci2a_subject_dependent(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    data = _load_bci_data_or_status(cfg, output_dir)
    if data is None:
        return {"status": "needs_input"}
    torch_status = _require_torch(output_dir, cfg, "E5 subject-dependent training")
    if torch_status:
        return torch_status
    rows = []
    confusion_matrices = []
    subjects = sorted(set(np.asarray(data["subject"]).astype(str)))
    for subject in subjects:
        train_raw, test_raw = subject_session_split(data, subject=subject, train_session="T", test_session="E")
        row = _run_train_test_model(
                "dpc_snn",
                train_raw,
                test_raw,
                cfg,
                output_dir / f"subject_{subject}",
                dataset_name="bci2a",
                protocol="subject_dependent",
                subject=subject,
                train_session="T",
                test_session="E",
            )
        rows.append(row)
        cm_path = output_dir / f"subject_{subject}" / "dpc_snn" / "confusion_matrix.npy"
        if not cm_path.exists():
            raise RuntimeError(f"E5 missing per-subject confusion matrix at {cm_path}")
        confusion_matrices.append(np.load(cm_path))
    write_csv(output_dir / "main_bci2a_subject_dependent.csv", rows)
    save_npy(output_dir / "confusion_matrices.npy", np.stack(confusion_matrices, axis=0))
    write_run_manifest(output_dir, cfg, "completed", {"n_subjects": len(rows)})
    return {"rows": rows}


def run_bci2a_cross_session(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    data = _load_bci_data_or_status(cfg, output_dir)
    if data is None:
        return {"status": "needs_input"}
    torch_status = _require_torch(output_dir, cfg, "E6 cross-session training")
    if torch_status:
        return torch_status
    rows = []
    subjects = sorted(set(np.asarray(data["subject"]).astype(str)))
    for subject in subjects:
        for train_session, test_session, direction in [("T", "E", "T_to_E"), ("E", "T", "E_to_T")]:
            train_raw, test_raw = subject_session_split(data, subject=subject, train_session=train_session, test_session=test_session)
            rows.append(
                _run_train_test_model(
                    "dpc_snn",
                    train_raw,
                    test_raw,
                    cfg,
                    output_dir / f"subject_{subject}_{direction}",
                    dataset_name="bci2a",
                    protocol="cross_session",
                    subject=subject,
                    direction=direction,
                    train_session=train_session,
                    test_session=test_session,
                )
            )
    gap_rows = []
    for subject in subjects:
        subj_rows = {row["direction"]: row for row in rows if row["subject"] == subject}
        if {"T_to_E", "E_to_T"}.issubset(subj_rows):
            gap_rows.append(
                {
                    "subject": subject,
                    "accuracy_T_to_E": subj_rows["T_to_E"].get("accuracy", np.nan),
                    "accuracy_E_to_T": subj_rows["E_to_T"].get("accuracy", np.nan),
                    "accuracy_gap_E_to_T_minus_T_to_E": float(subj_rows["E_to_T"].get("accuracy", np.nan)) - float(subj_rows["T_to_E"].get("accuracy", np.nan)),
                    "kappa_T_to_E": subj_rows["T_to_E"].get("kappa", np.nan),
                    "kappa_E_to_T": subj_rows["E_to_T"].get("kappa", np.nan),
                }
            )
    write_csv(output_dir / "cross_session_results.csv", rows)
    write_csv(output_dir / "session_gap.csv", gap_rows)
    write_run_manifest(output_dir, cfg, "completed" if rows else "needs_input", {"n_subjects": len(subjects)})
    return {"rows": rows, "session_gap": gap_rows}


def run_bci2a_loso(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    data = _load_bci_data_or_status(cfg, output_dir)
    if data is None:
        return {"status": "needs_input"}
    torch_status = _require_torch(output_dir, cfg, "E7 LOSO training")
    if torch_status:
        return torch_status
    rows = []
    for train_idx, test_idx, target in leave_one_subject_out(np.asarray(data["subject"]).astype(str)):
        train_raw = subset(data, train_idx)
        test_raw = subset(data, test_idx)
        rows.append(
            _run_train_test_model(
                "dpc_snn",
                train_raw,
                test_raw,
                cfg,
                output_dir / f"target_{target}",
                dataset_name="bci2a",
                protocol="loso",
                subject=target,
                selection_session="source_subjects",
                evaluation_session="heldout_target_subject",
                selection_split_label="source_subjects_inner_validation",
                evaluation_split="heldout_target_subject",
            )
        )
    write_csv(output_dir / "loso_results.csv", rows)
    write_csv(output_dir / "target_subject_breakdown.csv", rows)
    write_run_manifest(output_dir, cfg, "completed", {"n_targets": len(rows)})
    return {"rows": rows}


def run_few_shot_calibration(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    data = _load_bci_data_or_status(cfg, output_dir)
    if data is None:
        return {"status": "needs_input"}
    data = _crop_to_common_task_window(data, cfg)
    torch_status = _require_torch(output_dir, cfg, "E8 few-shot calibration")
    if torch_status:
        return torch_status
    rows = []
    param_rows = []
    settings = cfg.get("few_shot", {})
    calibration_ks = [int(value) for value in settings.get("calibration_ks", [1, 5, 10, 20, 50])]
    draw_seeds = [int(value) for value in settings.get("draw_seeds", cfg.get("seeds", [0, 1, 2, 3, 4]))]
    model_modes = {
        "dpc_snn": ["source_only", "readout", "delay", "phase_delay", "full"],
        "eegnet": ["source_only", "readout", "full"],
    }
    requested_models = [str(value) for value in settings.get("models", ["dpc_snn", "eegnet"])]
    unsupported = sorted(set(requested_models).difference(model_modes))
    if unsupported:
        raise ValueError(f"Unsupported E8 confirmatory models: {unsupported}")
    if not calibration_ks or not draw_seeds or not requested_models:
        return needs_input(output_dir, cfg, "E8 requires calibration K values, draw seeds, and at least one model.")
    selected_targets = _selected_target_subjects(cfg)
    for train_idx, test_idx, target in leave_one_subject_out(np.asarray(data["subject"]).astype(str)):
        if selected_targets is not None and str(target) not in selected_targets:
            continue
        del test_idx
        target_calibration_raw, target_evaluation_raw = subject_session_split(data, target, train_session="T", test_session="E")
        source = subset(data, train_idx)
        source, physical_space = _apply_physical_eeg_space(source, cfg)
        target_calibration_raw, calibration_space = _apply_physical_eeg_space(
            target_calibration_raw, cfg
        )
        target_evaluation_raw, evaluation_space = _apply_physical_eeg_space(
            target_evaluation_raw, cfg
        )
        if len({physical_space, calibration_space, evaluation_space}) != 1:
            raise ValueError("E8 source, calibration, and evaluation data used different EEG spaces")
        source_train_idx, source_val_idx = stratified_split_indices(source["y"], val_fraction=0.15, seed=int(cfg.get("seed", 0)))
        train_source = subset(source, source_train_idx)
        val_source = subset(source, source_val_idx)
        x_train, x_val, stats = standardize_train_test(train_source["X"], val_source["X"])
        train_source["X"], val_source["X"] = x_train, x_val
        train_source["standardize_mean"] = stats["mean"]
        train_source["standardize_std"] = stats["std"]
        target_calibration_raw["X"] = apply_channelwise_zscore(target_calibration_raw["X"], stats["mean"], stats["std"])
        target_evaluation_raw["X"] = apply_channelwise_zscore(target_evaluation_raw["X"], stats["mean"], stats["std"])
        for model_name in requested_models:
            pretrain_cfg = copy.deepcopy(cfg)
            pretrain_cfg["seed"] = int(cfg.get("seed", 0))
            pretrain_result = train_split(
                model_name,
                copy.deepcopy(train_source),
                copy.deepcopy(val_source),
                pretrain_cfg,
                output_dir / f"target_{target}" / model_name / "source_pretrain",
            )
            checkpoint = Path(pretrain_result["output_dir"]) / "model_checkpoint.pt"
            checkpoint_cfg = json.loads((Path(pretrain_result["output_dir"]) / "run_config.json").read_text(encoding="utf-8"))
            source_only = _evaluate_checkpoint_on_data(
                model_name,
                target_evaluation_raw,
                checkpoint_cfg,
                checkpoint,
                output_dir / f"target_{target}" / model_name / "source_only_eval",
            )
            for draw_seed in draw_seeds:
                for k in calibration_ks:
                    shot_idx = k_shot_indices(target_calibration_raw["y"], k=k, seed=draw_seed)
                    if shot_idx.size == 0:
                        continue
                    calib = subset(target_calibration_raw, shot_idx)
                    common = {
                        "calibration_k": k,
                        "calibration_draw_seed": draw_seed,
                        "target_calibration_session": "T",
                        "target_evaluation_session": "E",
                        "calibration_evaluation_overlap": False,
                        "evaluated_on_target": True,
                        "evaluated_on_heldout_test": True,
                        "calibration_trials": int(shot_idx.size),
                        "target_eval_trials": int(len(target_evaluation_raw["y"])),
                        "source_checkpoint": str(checkpoint),
                        "physical_eeg_space": physical_space,
                    }
                    rows.append(
                        _metric_row(
                            "bci2a",
                            "target_T_calibration_to_target_E",
                            target,
                            model_name,
                            checkpoint_cfg["seed"],
                            source_only,
                            calibration_mode="source_only",
                            **common,
                        )
                    )
                    param_rows.append(
                        {
                            "subject": target,
                            "model": model_name,
                            "calibration_k": k,
                            "calibration_draw_seed": draw_seed,
                            "mode": "source_only",
                            "trainable_params": 0,
                        }
                    )
                    for mode in model_modes[model_name][1:]:
                        from dpc_snn.training.adaptation import supervised_finetune

                        model, calib_feat, run_cfg = _clone_model_from_checkpoint(model_name, calib, checkpoint_cfg, checkpoint)
                        eval_feat = add_features(target_evaluation_raw, _feature_bands_from_cfg(run_cfg))
                        result = supervised_finetune(
                            model,
                            calib_feat,
                            eval_feat,
                            run_cfg,
                            output_dir / f"target_{target}" / model_name / f"draw_{draw_seed}" / f"k{k}_{mode}",
                            mode=mode,
                        )
                        rows.append(
                            _metric_row(
                                "bci2a",
                                "target_T_calibration_to_target_E",
                                target,
                                model_name,
                                checkpoint_cfg["seed"],
                                result,
                                calibration_mode=mode,
                                **common,
                            )
                        )
                        param_rows.append(
                            {
                                "subject": target,
                                "model": model_name,
                                "calibration_k": k,
                                "calibration_draw_seed": draw_seed,
                                "mode": mode,
                                "trainable_params": result["trainable_params"],
                            }
                        )
                    write_json(
                        output_dir / f"target_{target}" / f"draw_{draw_seed}_k{k}_calibration_manifest.json",
                        {
                            "calibration_indices_within_session_T": shot_idx.tolist(),
                            "target_calibration_session": "T",
                            "target_evaluation_session": "E",
                            "target_evaluation_trials": int(len(target_evaluation_raw["y"])),
                            "calibration_evaluation_overlap": False,
                            "models": requested_models,
                            "model_modes": model_modes,
                        },
                    )
    write_csv(output_dir / "few_shot_results.csv", rows)
    write_csv(output_dir / "calibration_param_count.csv", param_rows)
    write_run_manifest(
        output_dir,
        cfg,
        "completed" if rows else "needs_input",
        {
            "protocol": "target_session_T_calibration_to_untouched_target_session_E",
            "models": requested_models,
            "calibration_ks": calibration_ks,
            "draw_seeds": draw_seeds,
        },
    )
    return {"rows": rows}


def _fairness_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    baseline_path = cfg.get("model", {}).get("baselines_config") if isinstance(cfg.get("model"), dict) else None
    from_file = load_referenced_yaml(baseline_path).get("fairness", {}) if baseline_path else {}
    explicit = cfg.get("fairness", {})
    return {**from_file, **explicit}


def _fair_hparam_candidates(model_name: str, budget: int) -> list[dict[str, Any]]:
    """Return deterministic, genuinely executable inner-validation candidates."""

    architectures: list[dict[str, Any]]
    if model_name == "dpc_snn":
        architectures = [
            {"hidden_channels": value, "d_max": delay}
            for value in [24, 32, 48, 64, 96]
            for delay in [2, 4, 8, 12]
        ]
    elif model_name == "eegnet":
        architectures = [{"f1": value, "depth_multiplier": depth, "dropout": drop} for value in [4, 8, 12] for depth in [1, 2] for drop in [0.1, 0.25]]
    elif model_name == "vanilla_snn":
        architectures = [{"hidden_channels": value, "timesteps": steps} for value in [24, 48, 64, 96] for steps in [8, 16]]
    elif model_name == "graph_snn_no_delay":
        architectures = [{"hidden_channels": value, "timesteps": steps} for value in [16, 32, 48, 64] for steps in [8, 16]]
    elif model_name == "csp_lda":
        candidates = [{"baseline": {"csp_components": value}} for value in range(1, 23)]
        if budget > len(candidates):
            raise ValueError(f"Candidate grid for {model_name} has only {len(candidates)} unique trials, below requested budget {budget}.")
        return _balanced_candidate_subset(candidates, budget)
    elif model_name == "riemann_lr":
        candidates = [
            {"baseline": {"riemann_covariance_estimator": estimator, "riemann_logreg_c": value}}
            for estimator in ["oas", "lwf", "scm"]
            for value in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
        ]
        if budget > len(candidates):
            raise ValueError(f"Candidate grid for {model_name} has only {len(candidates)} unique trials, below requested budget {budget}.")
        return _balanced_candidate_subset(candidates, budget)
    else:
        raise KeyError(f"No fair-tuning candidate grid for {model_name}")
    candidates = []
    for architecture in architectures:
        for lr in [3e-4, 1e-3, 3e-3]:
            for weight_decay in [0.0, 1e-4, 1e-3]:
                candidate = copy.deepcopy(architecture)
                candidate["training"] = {"lr": lr, "weight_decay": weight_decay}
                candidates.append(candidate)
    if budget < 1:
        raise ValueError("fairness.max_hparam_trials must be at least one")
    if len(candidates) < budget:
        raise ValueError(f"Candidate grid for {model_name} has only {len(candidates)} unique trials, below requested budget {budget}.")
    return _balanced_candidate_subset(candidates, budget)


def _flatten_candidate(candidate: dict[str, Any], prefix: str = "") -> dict[str, str]:
    flattened: dict[str, str] = {}
    for key, value in sorted(candidate.items()):
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_candidate(value, name))
        else:
            flattened[name] = json.dumps(value, sort_keys=True)
    architecture = {key: value for key, value in candidate.items() if key not in {"training", "baseline"}}
    optimiser = {key: value for key, value in candidate.items() if key in {"training", "baseline"}}
    flattened["__architecture__"] = json.dumps(architecture, sort_keys=True)
    flattened["__optimiser__"] = json.dumps(optimiser, sort_keys=True)
    return flattened


def _balanced_candidate_subset(candidates: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Greedily balance architecture and optimiser marginals over a fixed budget."""
    if budget < 1:
        raise ValueError("Candidate budget must be at least one")
    if budget >= len(candidates):
        return candidates
    signatures = [_flatten_candidate(candidate) for candidate in candidates]
    counts: dict[tuple[str, str], int] = {}
    selected: list[int] = []
    remaining = set(range(len(candidates)))
    for _ in range(budget):
        best_index = max(
            remaining,
            key=lambda index: (
                sum(1.0 / (1.0 + counts.get((field, value), 0)) for field, value in signatures[index].items()),
                -index,
            ),
        )
        selected.append(best_index)
        remaining.remove(best_index)
        for field, value in signatures[best_index].items():
            counts[(field, value)] = counts.get((field, value), 0) + 1
    return [candidates[index] for index in selected]


def _apply_hparam_candidate(cfg: dict[str, Any], model_name: str, candidate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    run_cfg = copy.deepcopy(cfg)
    run_cfg["training"] = {**run_cfg.get("training", {}), **candidate.get("training", {})}
    run_cfg["baseline"] = {**run_cfg.get("baseline", {}), **candidate.get("baseline", {})}
    model_cfg = resolve_model_cfg(run_cfg, model_name)
    model_cfg.update({key: value for key, value in candidate.items() if key not in {"training", "baseline"}})
    return run_cfg, model_cfg


def _selection_score(metrics: dict[str, Any]) -> float:
    kappa = float(metrics.get("kappa", np.nan))
    return kappa if np.isfinite(kappa) else float(metrics.get("accuracy", -np.inf))


def run_baseline_fairness(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    data = _load_bci_data_or_status(cfg, output_dir)
    if data is None:
        return {"status": "needs_input"}
    torch_status = _require_torch(output_dir, cfg, "E14 baseline fairness audit")
    if torch_status:
        return torch_status
    models = _benchmark_model_names(cfg, include_classical=True)
    fairness = _fairness_settings(cfg)
    budget = int(fairness.get("max_hparam_trials", cfg.get("baseline_hparam_budget", 20)))
    tuning_epochs = min(
        int(cfg.get("training", {}).get("epochs", 150)),
        int(fairness.get("tuning_epochs", cfg.get("training", {}).get("epochs", 150))),
    )
    seeds = [int(seed) for seed in fairness.get("seeds", cfg.get("seeds", [0, 1, 2, 3, 4]))]
    if not seeds:
        return needs_input(output_dir, cfg, "E14 requires at least one training seed.")
    tune_seed = int(fairness.get("tuning_seed", seeds[0]))
    rows = []
    tuning_rows = []
    budget_rows = []
    subjects = sorted(set(np.asarray(data["subject"]).astype(str)))
    selected_targets = _selected_target_subjects(cfg)
    if selected_targets is not None:
        subjects = [subject for subject in subjects if subject in selected_targets]
    for subject in subjects:
        train_raw, test_raw = subject_session_split(data, subject=subject, train_session="T", test_session="E")
        for model_name in models:
            candidates = _fair_hparam_candidates(model_name, budget)
            tune_cfg_base = copy.deepcopy(cfg)
            tune_cfg_base["seed"] = tune_seed
            train, validation, split_audit = _split_protocol_train_validation(
                copy.deepcopy(train_raw), tune_cfg_base
            )
            trial_metrics: list[tuple[float, int, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
            for trial_id, candidate in enumerate(candidates):
                candidate_cfg, model_cfg = _apply_hparam_candidate(tune_cfg_base, model_name, candidate)
                candidate_cfg["training"]["epochs"] = tuning_epochs
                trial_dir = output_dir / "hparam_search" / f"subject_{subject}" / model_name / f"trial_{trial_id:02d}"
                try:
                    if model_name in {"csp_lda", "riemann_lr"}:
                        result = _evaluate_classical_baseline(model_name, train, validation, candidate_cfg, trial_dir)
                        metrics = result
                    else:
                        result = train_split(model_name, copy.deepcopy(train), copy.deepcopy(validation), candidate_cfg, trial_dir, model_cfg)
                        metrics = result["metrics"]
                    score = _selection_score(metrics)
                    trial_metrics.append((score, trial_id, candidate, candidate_cfg, model_cfg))
                    tuning_rows.append(
                        {
                            "dataset": "bci2a",
                            "subject": subject,
                            "model": model_name,
                            "tuning_seed": tune_seed,
                            "trial_id": trial_id,
                            "candidate": json.dumps(candidate, sort_keys=True),
                            "selection_score": score,
                            "selection_split": "inner_validation_on_session_T",
                            "selection_max_epochs": tuning_epochs,
                            "evaluated_on_heldout_test": False,
                            "status": "completed",
                            **{f"selection_{key}": value for key, value in metrics.items() if key in {"accuracy", "balanced_accuracy", "kappa", "macro_f1", "params"}},
                            **split_audit,
                        }
                    )
                except Exception as exc:
                    tuning_rows.append(
                        {
                            "dataset": "bci2a",
                            "subject": subject,
                            "model": model_name,
                            "tuning_seed": tune_seed,
                            "trial_id": trial_id,
                            "candidate": json.dumps(candidate, sort_keys=True),
                            "selection_split": "inner_validation_on_session_T",
                            "selection_max_epochs": tuning_epochs,
                            "evaluated_on_heldout_test": False,
                            "status": "failed",
                            "error": repr(exc),
                            **split_audit,
                        }
                    )
            if not trial_metrics:
                for seed in seeds:
                    rows.append(
                        {
                            "dataset": "bci2a",
                            "protocol": "baseline_fairness",
                            "subject": subject,
                            "model": model_name,
                            "seed": seed,
                            "status": "failed",
                            "hparam_budget": budget,
                            "hparam_trials_executed": budget,
                            "error": "All inner-validation hyperparameter trials failed.",
                        }
                    )
                continue
            _, selected_trial, selected_candidate, _, selected_model_cfg = max(trial_metrics, key=lambda item: (item[0], -item[1]))
            budget_rows.append(
                {
                    "dataset": "bci2a",
                    "subject": subject,
                    "model": model_name,
                    "max_hparam_trials": budget,
                    "executed_hparam_trials": len(candidates),
                    "successful_hparam_trials": len(trial_metrics),
                    "tuning_seed": tune_seed,
                    "selected_trial": selected_trial,
                    "selected_candidate": json.dumps(selected_candidate, sort_keys=True),
                    "selection_split": "inner_validation_on_session_T",
                    "selection_max_epochs": tuning_epochs,
                }
            )
            for seed in seeds:
                final_cfg, final_model_cfg = _apply_hparam_candidate(cfg, model_name, selected_candidate)
                final_cfg["seed"] = seed
                physical_mode = str(
                    resolve_model_cfg(final_cfg, "dpc_snn").get(
                        "physical_reference", "none"
                    )
                ).lower()
                preprocess_name = (
                    f"baseline_task_window_{physical_mode}_train_zscore_learnable_analytic_filterbank"
                    if model_name == "dpc_snn"
                    else f"baseline_task_window_{physical_mode}_shared_train_zscore"
                )
                try:
                    metric = _run_train_test_model(
                        model_name,
                        train_raw,
                        test_raw,
                        final_cfg,
                        output_dir / f"subject_{subject}" / f"seed_{seed}",
                        dataset_name="bci2a",
                        protocol="baseline_fairness",
                        subject=subject,
                        model_cfg=final_model_cfg,
                        split="subject_T_to_E",
                        preprocess=preprocess_name,
                        hparam_tuning_split="inner_validation_on_session_T",
                        hparam_budget=budget,
                        hparam_trials_executed=len(candidates),
                        hparam_trials_successful=len(trial_metrics),
                        hparam_selection_max_epochs=tuning_epochs,
                        tuning_seed=tune_seed,
                        selected_hparam_trial=selected_trial,
                        selected_hparams=json.dumps(selected_candidate, sort_keys=True),
                    )
                    rows.append(
                        {
                            **audit_row(
                                metric["model"],
                                {
                                    "seed": seed,
                                    "hparam_budget": budget,
                                    "split": "subject_T_to_E",
                                    "preprocess": preprocess_name,
                                    "status": "completed",
                                },
                                metric,
                            ),
                            **metric,
                        }
                    )
                except Exception as exc:
                    failed = audit_row(
                        model_name,
                        {
                            "seed": seed,
                            "hparam_budget": budget,
                            "split": "subject_T_to_E",
                            "preprocess": preprocess_name,
                            "status": "failed",
                            "notes": repr(exc),
                        },
                        {"status": "failed"},
                    )
                    failed.update({"dataset": "bci2a", "protocol": "baseline_fairness", "subject": subject, "error": repr(exc)})
                    rows.append(failed)
    write_csv(output_dir / "baseline_audit.csv", rows)
    write_csv(output_dir / "hparam_trials.csv", tuning_rows)
    write_csv(output_dir / "hparam_budget.csv", budget_rows)
    write_csv(output_dir / "runtime_summary.csv", rows)
    completed_models = {row.get("model", "") for row in rows if row.get("status") == "completed"}
    failures = [row for row in rows if row.get("status") == "failed"]
    expected_rows = len(subjects) * len(models) * len(seeds)
    status = "completed" if len(rows) == expected_rows and len(completed_models) == len(models) and not failures else "needs_input"
    write_run_manifest(
        output_dir,
        cfg,
        status,
        {"models": models, "seeds": seeds, "hparam_budget": budget, "tuning_epochs": tuning_epochs, "completed_models": sorted(completed_models), "failed_rows": len(failures)},
    )
    return {"rows": rows, "status": status}


def _parameter_count_for_config(model_name: str, data: dict[str, Any], cfg: dict[str, Any], model_cfg: dict[str, Any]) -> int:
    model = build_model_for_data(model_name, add_features(data, _feature_bands_from_cfg(cfg)), cfg, copy.deepcopy(model_cfg))
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _parameter_matched_control_config(
    model_name: str,
    data: dict[str, Any],
    cfg: dict[str, Any],
    target_params: int,
) -> tuple[dict[str, Any], int]:
    base = resolve_model_cfg(cfg, model_name)
    if model_name == "dpc_snn":
        return base, _parameter_count_for_config(model_name, data, cfg, base)
    candidates = []
    for hidden in range(4, 257, 4):
        candidate = copy.deepcopy(base)
        candidate["hidden_channels"] = hidden
        candidates.append(candidate)
    counts = [
        _parameter_count_for_config(model_name, data, cfg, candidate)
        for candidate in candidates
    ]
    best_index = min(range(len(candidates)), key=lambda index: abs(counts[index] - target_params))
    return candidates[best_index], counts[best_index]


def run_mechanism_controls(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    torch_status = _require_torch(output_dir, cfg, "E15 mechanism controls")
    if torch_status:
        return torch_status
    data = generate_delay_phase_dataset(_synthetic_cfg_from_runner(cfg))
    seeds = [int(seed) for seed in cfg.get("mechanism_controls", {}).get("seeds", cfg.get("seeds", [0, 1, 2, 3, 4]))]
    if not seeds:
        return needs_input(output_dir, cfg, "E15 requires at least one independent training seed.")
    rows = []
    match_rows = []
    model_names = ["delay_graph_ann", "phase_gated_gnn", "dilated_tcn", "complex_cnn", "dpc_snn"]
    for seed in seeds:
        seed_cfg = copy.deepcopy(cfg)
        seed_cfg["seed"] = seed
        train, val = split_train_val(data, seed=seed)
        dpc_cfg = resolve_model_cfg(seed_cfg, "dpc_snn")
        target_params = _parameter_count_for_config("dpc_snn", train, seed_cfg, dpc_cfg)
        for model_name in model_names:
            model_cfg, matched_params = _parameter_matched_control_config(model_name, train, seed_cfg, target_params)
            result = train_split(
                model_name,
                copy.deepcopy(train),
                copy.deepcopy(val),
                seed_cfg,
                output_dir / f"seed_{seed}" / model_name,
                model_cfg,
            )
            actual_params = int(result["metrics"]["params"])
            delta = actual_params - target_params
            row = {
                **result["metrics"],
                "dataset": "synthetic_delay_phase",
                "protocol": "mechanism_control_validation",
                "subject": "all",
                "model": model_name,
                "seed": seed,
                "target_dpc_params": target_params,
                "candidate_params_before_training": matched_params,
                "parameter_delta": delta,
                "relative_parameter_delta": float(delta / max(1, target_params)),
                "matching_method": "nearest_hidden_width_over_grid_4_to_256_step_4",
                "evaluation_split": "validation",
            }
            rows.append(row)
            match_rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "params": actual_params,
                    "target_dpc_params": target_params,
                    "parameter_delta": delta,
                    "relative_parameter_delta": float(delta / max(1, target_params)),
                    "matching_method": "nearest_hidden_width_over_grid_4_to_256_step_4",
                    "model_config": json.dumps(model_cfg, sort_keys=True),
                }
            )
    write_csv(output_dir / "mechanism_controls.csv", rows)
    write_csv(output_dir / "matched_params.csv", match_rows)
    write_run_manifest(output_dir, cfg, "completed", {"seeds": seeds, "matching_method": "nearest_hidden_width_over_grid_4_to_256_step_4"})
    return {"rows": rows}


def run_ablation_core(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    torch_status = _require_torch(output_dir, cfg, "E16 ablation")
    if torch_status:
        return torch_status
    data = generate_delay_phase_dataset(_synthetic_cfg_from_runner(cfg))
    seeds = [int(seed) for seed in cfg.get("synthetic_analysis", {}).get("seeds", cfg.get("seeds", [0, 1, 2, 3, 4]))]
    variants = {
        "full": {},
        "wo_delay": {"force_zero_delay": True},
        "wo_cross_band_delay": {"use_cross_band_routes": False},
        "wo_covariance": {"use_covariance": False},
        "wo_phase_confidence": {"use_phase_confidence": False},
        "wo_alignment": {"euclidean_alignment": False},
        "low_graph_sparsity": {"graph_sparsity": 0.05},
        "dense_graph": {"graph_sparsity": 1.0},
        "short_timesteps": {"timesteps": 32},
        "long_timesteps": {"timesteps": 256},
    }
    rows = []
    for seed in seeds:
        seed_cfg = copy.deepcopy(cfg)
        seed_cfg["seed"] = seed
        train, val = split_train_val(data, seed=seed)
        for variant, override in variants.items():
            model_cfg = resolve_model_cfg(seed_cfg, "dpc_snn")
            model_cfg.update(override)
            result = train_split("dpc_snn", copy.deepcopy(train), copy.deepcopy(val), seed_cfg, output_dir / f"seed_{seed}" / variant, model_cfg)
            rows.append({**result["metrics"], "dataset": "synthetic_delay_phase", "protocol": "ablation_validation", "subject": "all", "variant": variant, "model": "dpc_snn", "seed": seed, "evaluation_split": "validation"})
    write_csv(output_dir / "ablation_core.csv", rows)
    write_run_manifest(output_dir, cfg, "completed", {"seeds": seeds, "evaluation_split": "validation"})
    return {"rows": rows}


def run_sensitivity(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    torch_status = _require_torch(output_dir, cfg, "E17 sensitivity")
    if torch_status:
        return torch_status
    data = generate_delay_phase_dataset(_synthetic_cfg_from_runner(cfg))
    seeds = [int(seed) for seed in cfg.get("synthetic_analysis", {}).get("seeds", cfg.get("seeds", [0, 1, 2, 3, 4]))]
    dmax_rows = []
    ts_rows = []
    graph_rate_rows = []
    band_rows = []
    band_configs = {
        "mu": {"mu": [8.0, 13.0]},
        "beta": {"beta": [13.0, 30.0]},
        "mu_beta": DEFAULT_BANDS,
        "alpha_mu_beta": {"alpha": [6.0, 8.0], "mu": [8.0, 13.0], "beta": [13.0, 30.0]},
    }
    for seed in seeds:
        seed_cfg = copy.deepcopy(cfg)
        seed_cfg["seed"] = seed
        train, val = split_train_val(data, seed=seed)
        for dmax in [0, 2, 4, 8, 12, 16]:
            model_cfg = resolve_model_cfg(seed_cfg, "dpc_snn")
            model_cfg["d_max"] = dmax
            result = train_split("dpc_snn", copy.deepcopy(train), copy.deepcopy(val), seed_cfg, output_dir / f"seed_{seed}" / f"dmax_{dmax}", model_cfg)
            dmax_rows.append({**result["metrics"], "dataset": "synthetic_delay_phase", "protocol": "sensitivity_dmax_validation", "subject": "all", "model": "dpc_snn", "seed": seed, "sweep": "d_max", "value": dmax, "evaluation_split": "validation"})
        for steps in [32, 64, 128, 256]:
            model_cfg = resolve_model_cfg(seed_cfg, "dpc_snn")
            model_cfg["timesteps"] = steps
            result = train_split("dpc_snn", copy.deepcopy(train), copy.deepcopy(val), seed_cfg, output_dir / f"seed_{seed}" / f"timesteps_{steps}", model_cfg)
            ts_rows.append({**result["metrics"], "dataset": "synthetic_delay_phase", "protocol": "sensitivity_timesteps_validation", "subject": "all", "model": "dpc_snn", "seed": seed, "sweep": "timesteps", "value": steps, "evaluation_split": "validation"})
        for graph_rate in [40.0, 64.0, 80.0, 100.0]:
            model_cfg = resolve_model_cfg(seed_cfg, "dpc_snn")
            model_cfg["graph_rate_hz"] = graph_rate
            result = train_split(
                "dpc_snn",
                copy.deepcopy(train),
                copy.deepcopy(val),
                seed_cfg,
                output_dir / f"seed_{seed}" / f"graph_rate_{graph_rate:g}",
                model_cfg,
            )
            graph_rate_rows.append(
                {
                    **result["metrics"],
                    "dataset": "synthetic_delay_phase",
                    "protocol": "sensitivity_graph_rate_validation",
                    "subject": "all",
                    "model": "dpc_snn",
                    "seed": seed,
                    "sweep": "graph_rate_hz",
                    "value": graph_rate,
                    "evaluation_split": "validation",
                }
            )
        for name, bands in band_configs.items():
            band_cfg = copy.deepcopy(seed_cfg)
            band_cfg["feature_bands"] = bands
            model_cfg = resolve_model_cfg(band_cfg, "dpc_snn")
            model_cfg["band_edges_hz"] = [[float(edge[0]), float(edge[1])] for edge in bands.values()]
            model_cfg["n_bands"] = len(bands)
            result = train_split("dpc_snn", copy.deepcopy(train), copy.deepcopy(val), band_cfg, output_dir / f"seed_{seed}" / f"bands_{name}", model_cfg)
            band_rows.append(
                {
                    **result["metrics"],
                    "dataset": "synthetic_delay_phase",
                    "protocol": "band_ablation_validation",
                    "subject": "all",
                    "model": "dpc_snn",
                    "seed": seed,
                    "sweep": "feature_bands",
                    "value": name,
                    "bands": json.dumps(bands, sort_keys=True),
                    "evaluation_split": "validation",
                }
            )
    write_csv(output_dir / "sensitivity_dmax.csv", dmax_rows)
    write_csv(output_dir / "sensitivity_timesteps.csv", ts_rows)
    write_csv(output_dir / "sensitivity_graph_rate.csv", graph_rate_rows)
    write_csv(output_dir / "band_ablation.csv", band_rows)
    write_run_manifest(output_dir, cfg, "completed", {"seeds": seeds, "evaluation_split": "validation"})
    return {"dmax": dmax_rows, "timesteps": ts_rows, "graph_rate": graph_rate_rows, "bands": band_rows}


def run_neurophysiology(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    dcfg = resolve_dataset_cfg(cfg, key="bci2a_config")
    root = Path(dcfg.get("root", "data/processed/bci2a"))
    if not root.exists():
        return needs_input(output_dir, cfg, "E18 requires real BCI2a epochs with recorded pre-cue timing metadata.")
    data = load_processed_npz(root)
    epoch_tmin = data.get("epoch_tmin")
    epoch_tmax = data.get("epoch_tmax")
    if epoch_tmin is None or epoch_tmax is None or float(epoch_tmin) >= 0.0 or float(epoch_tmax) <= 0.0:
        return needs_input(
            output_dir,
            cfg,
            "E18 ERD/ERS requires BCI2a epochs spanning a pre-cue baseline and imagery interval. "
            "Re-export with scripts/prepare_bci2a_moabb.py --tmin -1 --tmax 4.",
        )
    sfreq = float(data["sfreq"])
    baseline_slice = slice(0, int(round(-float(epoch_tmin) * sfreq)))
    imagery_slice = slice(int(round(-float(epoch_tmin) * sfreq)), data["X"].shape[-1])
    rows = erd_ers(data["X"], data["y"], sfreq, (8.0, 13.0), baseline_slice, imagery_slice)
    write_csv(output_dir / "erd_ers_topomap.csv", rows)
    checkpoint_root = _analysis_scope_root(output_dir, cfg, key="interpretation_checkpoint_root") / "E14"
    learned_paths = sorted(checkpoint_root.glob("**/dpc_snn/learned_params.npz"))
    if not learned_paths:
        return needs_input(output_dir, cfg, f"E18 requires DPC-SNN learned parameters under {checkpoint_root}.")
    edge_params = []
    delay_params = []
    checkpoint_metadata = []
    for learned_path in learned_paths:
        run_cfg_path = learned_path.parent / "run_config.json"
        stats_path = learned_path.parent / "preprocessing_stats.npz"
        if not run_cfg_path.exists() or not stats_path.exists():
            continue
        run_cfg = json.loads(run_cfg_path.read_text(encoding="utf-8"))
        if str(run_cfg.get("experiment_id", "")) != "E14" or str(run_cfg.get("model_name", "")) != "dpc_snn":
            continue
        with np.load(learned_path) as learned:
            edge_params.append(np.asarray(learned["edge_weight"], dtype=np.float32))
            delay_params.append(np.asarray(learned["delay"], dtype=np.float32))
        checkpoint_metadata.append(
            {
                "learned_parameters": str(learned_path),
                "checkpoint": str(learned_path.parent / "model_checkpoint.pt"),
                "subject": str(run_cfg.get("subject", "")),
                "seed": run_cfg.get("seed"),
                "evaluation_split": "heldout_session_E",
            }
        )
    expected_subjects = set(np.asarray(data["subject"]).astype(str))
    observed_subjects = {str(item["subject"]) for item in checkpoint_metadata}
    if not edge_params or not expected_subjects.issubset(observed_subjects):
        return needs_input(
            output_dir,
            cfg,
            "E18 requires reproducible E14 DPC-SNN checkpoints with preprocessing statistics for every BCI2a subject. "
            "Rerun E14 after the strict protocol repair.",
        )
    save_npy(
        output_dir / "edge_importance.npy",
        np.mean([_collapse_band_pair_matrix(np.abs(value)) for value in edge_params], axis=0),
    )
    save_npy(
        output_dir / "delay_matrix.npy",
        np.mean([_collapse_band_pair_matrix(value) for value in delay_params], axis=0),
    )
    write_json(
        output_dir / "model_interpretation_manifest.json",
        {
            "checkpoint_source_experiment": "E14",
            "pre_cue_baseline_sec": [float(epoch_tmin), 0.0],
            "imagery_sec": [0.0, float(epoch_tmax)],
            "checkpoint_metadata": checkpoint_metadata,
        },
    )
    plv = phase_locking_value(data["X"])
    dwpli = debiased_weighted_phase_lag_index(data["X"])
    imcoh = imaginary_coherence(data["X"])
    conn_rows = _connectivity_rows({"plv": plv, "dwpli": dwpli, "imaginary_coherence": imcoh})
    write_csv(output_dir / "connectivity_metrics.csv", conn_rows)
    write_run_manifest(output_dir, cfg, "completed", {"n_checkpoints": len(checkpoint_metadata), "checkpoint_source_experiment": "E14"})
    return {"rows": rows}


def run_explainability_sanity(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    torch_status = _require_torch(output_dir, cfg, "E19 explainability sanity")
    if torch_status:
        return torch_status
    data = _load_bci_data_or_status(cfg, output_dir)
    if data is None:
        return {"status": "needs_input"}
    seeds = list(cfg.get("explainability", {}).get("seeds", cfg.get("seeds", [0, 1, 2])))[:5]
    if len(seeds) < 2:
        return needs_input(output_dir, cfg, "E19 requires at least two independently trained seeds per subject.")
    import torch

    edges_by_subject: dict[str, list[tuple[int, np.ndarray]]] = {}
    occlusion_rows = []
    subjects = sorted(set(np.asarray(data["subject"]).astype(str)))
    selected_targets = _selected_target_subjects(cfg)
    if selected_targets is not None:
        subjects = [subject for subject in subjects if subject in selected_targets]
    for subject in subjects:
        train_raw, test_raw = subject_session_split(data, subject=subject, train_session="T", test_session="E")
        for seed in seeds:
            seed_cfg = copy.deepcopy(cfg)
            seed_cfg["seed"] = int(seed)
            train, validation, test, split_audit = _split_protocol_train_validation_test(
                copy.deepcopy(train_raw), copy.deepcopy(test_raw), seed_cfg
            )
            result = train_split(
                "dpc_snn",
                train,
                validation,
                seed_cfg,
                output_dir / f"subject_{subject}" / f"seed_{seed}" / "dpc_snn",
                test=test,
            )
            learned_path = Path(result["output_dir"]) / "learned_params.npz"
            checkpoint = Path(result["output_dir"]) / "model_checkpoint.pt"
            if not learned_path.exists() or not checkpoint.exists():
                raise RuntimeError(f"E19 missing learned DPC-SNN artifacts for subject={subject}, seed={seed}")
            with np.load(learned_path) as z:
                edge_key = "edge_weight_latent" if "edge_weight_latent" in z.files else "edge_weight"
                edge_full = np.abs(np.asarray(z[edge_key], dtype=np.float32))
                edge_selection = np.asarray(
                    z["edge_selection"]
                    if "edge_selection" in z.files
                    else edge_full > 0.0,
                    dtype=np.float32,
                )
                edge_full = edge_full * (edge_selection > 0.0)
            edges_by_subject.setdefault(subject, []).append((int(seed), edge_full))
            baseline = result["metrics"]
            flat = edge_full.reshape(-1)
            candidates = np.flatnonzero(edge_selection.reshape(-1) > 0.0)
            if candidates.size == 0:
                raise RuntimeError(
                    f"E19 found no accepted routes for subject={subject}, seed={seed}"
                )
            k = min(
                int(cfg.get("explainability", {}).get("occlusion_topk", 20)),
                int(candidates.size),
            )
            top_idx = candidates[np.argsort(flat[candidates])[-k:]]
            rng = np.random.default_rng(int(seed))
            random_idx = rng.choice(candidates, size=k, replace=False)
            for label, chosen in [("top_edge", top_idx), ("random_edge", random_idx)]:
                occ_model, _, _, audit = _clone_model_with_checkpoint_audit("dpc_snn", test, seed_cfg, checkpoint)
                if not hasattr(occ_model, "synapse") or not hasattr(occ_model.synapse, "set_edge_weight_override"):
                    raise RuntimeError("E19 requires a DPC-SNN synapse with a fixed edge-weight override.")
                multiplier = torch.ones_like(occ_model.synapse.route_mask)
                multiplier.reshape(-1)[chosen] = 0.0
                occ_model.synapse.set_edge_weight_override(multiplier)
                occ = _evaluate_model_on_data(
                    occ_model,
                    test,
                    seed_cfg,
                    output_dir / f"subject_{subject}" / f"seed_{seed}" / f"{label}_occlusion",
                )
                occlusion_rows.append(
                    {
                        "dataset": "bci2a",
                        "protocol": "subject_T_to_E_heldout_test",
                        "subject": subject,
                        "seed": seed,
                        "occlusion": label,
                        "k_edges": int(k),
                        "baseline_accuracy": baseline["accuracy"],
                        "accuracy": occ["accuracy"],
                        "kappa": occ["kappa"],
                        "macro_f1": occ["macro_f1"],
                        "occlusion_drop": float(baseline["accuracy"] - occ["accuracy"]),
                        "checkpoint_loaded_param_ratio": audit["loaded_param_ratio"],
                        "evaluated_on_heldout_test": True,
                        "occlusion_protocol": "accepted_route_matched_fixed_edge_multiplier",
                        **split_audit,
                    }
                )
    rows = []
    for subject, edges in edges_by_subject.items():
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                seed_i, edge_i = edges[i]
                seed_j, edge_j = edges[j]
                rows.append(
                    {
                        "dataset": "bci2a",
                        "protocol": "subject_T_to_E_heldout_test",
                        "subject": subject,
                        "seed_i": seed_i,
                        "seed_j": seed_j,
                        "topk_jaccard": edge_jaccard(edge_i, edge_j, k=min(20, edge_i.size)),
                        "stability_scope": "within_subject_across_independent_seeds",
                    }
                )
    if not rows:
        return needs_input(output_dir, cfg, "E19 did not produce at least two completed seed fits for any subject.")
    write_csv(output_dir / "edge_stability.csv", rows)
    write_csv(output_dir / "explainability_sanity.csv", rows)
    write_csv(output_dir / "occlusion_results.csv", occlusion_rows)
    write_run_manifest(output_dir, cfg, "completed", {"seeds": seeds, "data": "real_bci2a", "subjects": subjects})
    return {"rows": rows, "occlusion": occlusion_rows}


def run_reference_sensitivity(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    torch_status = _require_torch(output_dir, cfg, "E20 reference sensitivity")
    if torch_status:
        return torch_status
    data = _load_bci_data_or_status(cfg, output_dir)
    if data is None:
        return {"status": "needs_input"}
    modes = ["original", "car", "csd"]
    rows = []
    volume_rows = []
    for subject in sorted(set(np.asarray(data["subject"]).astype(str))):
        train_raw, test_raw = subject_session_split(data, subject=subject, train_session="T", test_session="E")
        for mode in modes:
            if mode == "csd":
                csd_cfg = copy.deepcopy(cfg)
                csd_cfg["model"] = {
                    **csd_cfg.get("model", {}),
                    "physical_reference": "csd",
                }
                train_ref, _ = _apply_physical_eeg_space(train_raw, csd_cfg)
                test_ref, _ = _apply_physical_eeg_space(test_raw, csd_cfg)
            else:
                train_ref = dict(train_raw)
                test_ref = dict(test_raw)
                train_ref["X"] = apply_reference(train_raw["X"], mode)
                test_ref["X"] = apply_reference(test_raw["X"], mode)
            reference_cfg = copy.deepcopy(cfg)
            reference_cfg["model"] = {
                **reference_cfg.get("model", {}),
                "physical_reference": "none",
            }
            metric = _run_train_test_model(
                "dpc_snn",
                train_ref,
                test_ref,
                reference_cfg,
                output_dir / f"subject_{subject}" / mode,
                dataset_name="bci2a",
                protocol="reference_sensitivity",
                subject=subject,
                reference=mode,
                physical_eeg_space=f"explicit_{mode}_reference",
                connectivity_data="source_training_session_T",
            )
            rows.append(metric)
            plv = phase_locking_value(train_ref["X"])
            dwpli = debiased_weighted_phase_lag_index(train_ref["X"])
            imcoh = imaginary_coherence(train_ref["X"])
            learned_path = output_dir / f"subject_{subject}" / mode / "dpc_snn" / "learned_params.npz"
            edge_corr = {"plv_corr": np.nan, "dwpli_corr": np.nan, "imaginary_coherence_corr": np.nan}
            if learned_path.exists():
                with np.load(learned_path) as z:
                    edge = _collapse_band_pair_matrix(
                        np.abs(np.asarray(z["edge_weight"], dtype=np.float32))
                    )
                off_diagonal = ~np.eye(edge.shape[0], dtype=bool)
                edge_corr = {
                    "plv_corr": matrix_correlation(edge, plv, mask=off_diagonal),
                    "dwpli_corr": matrix_correlation(edge, dwpli, mask=off_diagonal),
                    "imaginary_coherence_corr": matrix_correlation(edge, imcoh, mask=off_diagonal),
                }
            volume_rows.append(
                {
                    "dataset": "bci2a",
                    "subject": subject,
                    "reference": mode,
                    "connectivity_data": "source_training_session_T",
                    "evaluated_on_heldout_test": True,
                    **edge_corr,
                }
            )
    write_csv(output_dir / "reference_sensitivity.csv", rows)
    write_csv(output_dir / "volume_conduction_checks.csv", volume_rows)
    write_run_manifest(output_dir, cfg, "completed", {"references": modes, "data": "real_bci2a", "protocol": "subject_T_to_E"})
    return {"rows": rows}


def run_efficiency_latency(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    scan_root = _analysis_scope_root(output_dir, cfg, key="latency_checkpoint_root")
    checkpoint_experiment = str(cfg.get("latency", {}).get("checkpoint_experiment", "E14"))
    checkpoint_root = scan_root / checkpoint_experiment
    checkpoints = sorted(checkpoint_root.glob("**/dpc_snn/model_checkpoint.pt"))
    if not checkpoints:
        return needs_input(output_dir, cfg, f"E21 requires DPC-SNN checkpoints from {checkpoint_experiment} under {checkpoint_root}.")
    try:
        import torch
        from torch.utils.data import DataLoader
        from dpc_snn.training.latency import collect_spike_statistics, decision_latency_auc, measure_inference_latency
        from dpc_snn.training.tensor_dataset import TrialTensorDataset
        from dpc_snn.utils.torch import resolve_device
    except ImportError as exc:
        return needs_input(output_dir, cfg, f"E21 latency measurement requires torch: {exc!r}")
    data = _load_bci_data_or_status(cfg, output_dir)
    if data is None:
        return {"status": "needs_input"}
    spike_rows = []
    latency_rows = []
    auc_rows_all = []
    checkpoint_audit_rows = []
    min_ratio = float(cfg.get("latency", {}).get("min_loaded_param_ratio", 0.95))
    max_checkpoints = cfg.get("latency", {}).get("max_checkpoints")
    if max_checkpoints is not None:
        checkpoints = checkpoints[: int(max_checkpoints)]
    for checkpoint in checkpoints:
        run_cfg_path = checkpoint.parent / "run_config.json"
        stats_path = checkpoint.parent / "preprocessing_stats.npz"
        if not run_cfg_path.exists() or not stats_path.exists():
            checkpoint_audit_rows.append(
                {"checkpoint": str(checkpoint), "status": "skipped_missing_reproducibility_metadata"}
            )
            continue
        run_cfg = json.loads(run_cfg_path.read_text(encoding="utf-8"))
        if str(run_cfg.get("experiment_id", "")) != checkpoint_experiment or str(run_cfg.get("model_name", "")) != "dpc_snn":
            checkpoint_audit_rows.append(
                {"checkpoint": str(checkpoint), "status": "skipped_wrong_experiment_or_model"}
            )
            continue
        subject = str(run_cfg.get("subject", ""))
        if not subject:
            checkpoint_audit_rows.append({"checkpoint": str(checkpoint), "status": "skipped_missing_subject"})
            continue
        try:
            _, test_raw = subject_session_split(data, subject=subject, train_session="T", test_session="E")
        except ValueError as exc:
            checkpoint_audit_rows.append({"checkpoint": str(checkpoint), "subject": subject, "status": "skipped_missing_heldout_session", "error": repr(exc)})
            continue
        test_raw = _crop_to_common_task_window(test_raw, run_cfg)
        test_raw, physical_space = _apply_physical_eeg_space(test_raw, run_cfg)
        with np.load(stats_path) as stats:
            test = dict(test_raw)
            test["X"] = apply_channelwise_zscore(test_raw["X"], stats["mean"], stats["std"])
        model, featured, checkpoint_cfg, audit = _clone_model_with_checkpoint_audit("dpc_snn", test, run_cfg, checkpoint)
        ds = TrialTensorDataset(featured["X"], featured["y"], featured.get("amplitude"), featured.get("phase"))
        loader = DataLoader(ds, batch_size=min(16, len(ds)), shuffle=False)
        first_batch = next(iter(loader))
        batches = [first_batch]
        audit_row_payload = {
            "checkpoint": str(checkpoint),
            "subject": subject,
            "source_experiment": checkpoint_experiment,
            "physical_eeg_space": physical_space,
            **audit,
        }
        if audit["loaded_param_ratio"] < min_ratio:
            checkpoint_audit_rows.append({**audit_row_payload, "status": "skipped_incompatible"})
            continue
        checkpoint_audit_rows.append({**audit_row_payload, "status": "included"})
        device = resolve_device(str(checkpoint_cfg.get("device", "cpu")))
        cpu_latency = measure_inference_latency(model, batches, "cpu", repeat=int(cfg.get("latency", {}).get("repeat", 10)))
        latency_rows.append({"checkpoint": str(checkpoint), "subject": subject, "source_experiment": checkpoint_experiment, **cpu_latency})
        if torch.cuda.is_available() and device.startswith("cuda"):
            gpu_latency = measure_inference_latency(model, batches, device, repeat=int(cfg.get("latency", {}).get("repeat", 10)))
            latency_rows.append({"checkpoint": str(checkpoint), "subject": subject, "source_experiment": checkpoint_experiment, **gpu_latency})
        spike = collect_spike_statistics(model, first_batch, "cpu")
        spike_rows.append({"checkpoint": str(checkpoint), "subject": subject, "source_experiment": checkpoint_experiment, **spike})
        auc_rows, auc = decision_latency_auc(
            model,
            first_batch,
            "cpu",
            n_classes=int(checkpoint_cfg.get("n_classes", 4)),
            sfreq=float(test.get("sfreq", 250.0)),
            bands=_feature_bands_from_cfg(checkpoint_cfg),
        )
        for row in auc_rows:
            auc_rows_all.append(
                {
                    "checkpoint": str(checkpoint),
                    "subject": subject,
                    "source_experiment": checkpoint_experiment,
                    "evaluation_split": "heldout_session_E",
                    "feature_protocol": "prefix_only_recomputed_fft_hilbert",
                    "decision_auc": auc,
                    **row,
                }
            )
    write_csv(output_dir / "spike_stats.csv", spike_rows)
    write_csv(output_dir / "synops_proxy.csv", spike_rows)
    write_csv(output_dir / "latency_cpu_gpu.csv", latency_rows)
    write_csv(output_dir / "decision_auc.csv", auc_rows_all)
    write_csv(output_dir / "latency_checkpoint_audit.csv", checkpoint_audit_rows)
    if not latency_rows:
        return needs_input(output_dir, cfg, f"E21 found no reproducible {checkpoint_experiment} DPC-SNN checkpoints compatible with held-out BCI2a evaluation.")
    write_run_manifest(output_dir, cfg, "completed", {"n_checkpoints": len(checkpoints), "n_included_checkpoints": len({row["checkpoint"] for row in latency_rows}), "checkpoint_root": str(checkpoint_root), "source_experiment": checkpoint_experiment, "feature_protocol": "prefix_only_recomputed_fft_hilbert"})
    return {"spike_stats": spike_rows, "latency": latency_rows}


def run_artifact_robustness(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    data = _load_bci_data_or_status(cfg, output_dir)
    if data is None:
        return {"status": "needs_input"}
    torch_status = _require_torch(output_dir, cfg, "E22 real EEG artifact robustness")
    if torch_status:
        return torch_status
    rows = []
    artifact_cfg = {
        "gaussian_noise": {"std": float(cfg.get("artifact", {}).get("gaussian_std", 0.25))},
        "channel_dropout": {"drop_fraction": float(cfg.get("artifact", {}).get("drop_fraction", 0.1))},
        "temporal_jitter": {"max_shift": int(cfg.get("artifact", {}).get("max_shift", 4))},
    }
    for subject in sorted(set(np.asarray(data["subject"]).astype(str))):
        train_raw, test_raw = subject_session_split(data, subject=subject, train_session="T", test_session="E")
        train, validation, test, split_audit = _split_protocol_train_validation_test(
            copy.deepcopy(train_raw), copy.deepcopy(test_raw), cfg
        )
        result = train_split(
            "dpc_snn",
            train,
            validation,
            cfg,
            output_dir / f"subject_{subject}_clean_train",
            test=test,
        )
        checkpoint = Path(result["output_dir"]) / "model_checkpoint.pt"
        clean = result["metrics"]
        rows.append(
            _metric_row(
                "bci2a",
                "artifact_robustness",
                subject,
                "dpc_snn",
                cfg.get("seed", 0),
                clean,
                artifact="clean",
                severity=0.0,
                n_trials=int(test["X"].shape[0]),
                n_channels=int(test["X"].shape[1]),
                n_time=int(test["X"].shape[2]),
                evaluated_on_heldout_test=True,
                **split_audit,
            )
        )
        perturbations = [
            ("gaussian_noise", add_gaussian_noise(test["X"], artifact_cfg["gaussian_noise"]["std"], seed=int(cfg.get("seed", 0))), artifact_cfg["gaussian_noise"]["std"]),
            ("channel_dropout", channel_dropout(test["X"], artifact_cfg["channel_dropout"]["drop_fraction"], seed=int(cfg.get("seed", 0))), artifact_cfg["channel_dropout"]["drop_fraction"]),
            ("temporal_jitter", temporal_jitter(test["X"], artifact_cfg["temporal_jitter"]["max_shift"], seed=int(cfg.get("seed", 0))), artifact_cfg["temporal_jitter"]["max_shift"]),
        ]
        for name, perturbed_x, severity in perturbations:
            perturbed = dict(test)
            perturbed["X"] = perturbed_x
            metrics = _evaluate_checkpoint_on_data("dpc_snn", perturbed, cfg, checkpoint, output_dir / f"subject_{subject}_{name}_eval")
            rows.append(
                _metric_row(
                    "bci2a",
                    "artifact_robustness",
                    subject,
                    "dpc_snn",
                    cfg.get("seed", 0),
                    metrics,
                    artifact=name,
                    severity=severity,
                    clean_accuracy=clean["accuracy"],
                    accuracy_drop=float(clean["accuracy"] - metrics["accuracy"]),
                    n_trials=int(perturbed_x.shape[0]),
                    n_channels=int(perturbed_x.shape[1]),
                    n_time=int(perturbed_x.shape[2]),
                    evaluated_on_heldout_test=True,
                    **split_audit,
                )
            )
    write_csv(output_dir / "artifact_robustness.csv", rows)
    write_json(output_dir / "artifact_config.json", artifact_cfg)
    write_run_manifest(output_dir, cfg, "completed" if rows else "needs_input")
    return {"rows": rows}


def run_low_channel(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    torch_status = _require_torch(output_dir, cfg, "E23 low-channel simulation")
    if torch_status:
        return torch_status
    data = _load_bci_data_or_status(cfg, output_dir)
    if data is None:
        return {"status": "needs_input"}
    channel_names = data.get("ch_names")
    if channel_names is None:
        return needs_input(output_dir, cfg, "E23 requires named real BCI2a EEG channels to define portable subsets.")
    normalized = {_canonical_eeg_channel_name(name): index for index, name in enumerate(channel_names)}
    required = {"C3", "CZ", "C4"}
    if not required.issubset(normalized):
        return needs_input(
            output_dir,
            cfg,
            f"E23 requires C3, Cz, and C4 channel names; available normalized channels are {sorted(normalized)}.",
        )
    subsets = {
        "C3_Cz_C4": [normalized["C3"], normalized["CZ"], normalized["C4"]],
        "C3_C4": [normalized["C3"], normalized["C4"]],
    }
    rows = []
    for subject in sorted(set(np.asarray(data["subject"]).astype(str))):
        train_raw, test_raw = subject_session_split(data, subject=subject, train_session="T", test_session="E")
        train_raw, train_space = _apply_physical_eeg_space(train_raw, cfg)
        test_raw, test_space = _apply_physical_eeg_space(test_raw, cfg)
        if train_space != test_space:
            raise ValueError("E23 training and held-out data used different EEG spaces")
        subset_cfg = copy.deepcopy(cfg)
        subset_cfg["model"] = {
            **subset_cfg.get("model", {}),
            "physical_reference": "none",
        }
        for name, channels in subsets.items():
            train = dict(train_raw)
            test = dict(test_raw)
            train["X"] = train_raw["X"][:, channels, :]
            test["X"] = test_raw["X"][:, channels, :]
            selected_names = [str(channel_names[index]) for index in channels]
            train["ch_names"] = selected_names
            test["ch_names"] = selected_names
            metric = _run_train_test_model(
                "dpc_snn",
                train,
                test,
                subset_cfg,
                output_dir / f"subject_{subject}" / name,
                dataset_name="bci2a",
                protocol="portable_low_channel_subject_T_to_E",
                subject=subject,
                channel_subset=name,
                channels=" ".join(selected_names),
                evaluated_on_real_bci2a=True,
                physical_eeg_space=f"{train_space}_then_channel_subset",
            )
            rows.append(metric)
    write_csv(output_dir / "low_channel_results.csv", rows)
    write_json(
        output_dir / "channel_subset_config.json",
        {
            name: {"indices": indices, "names": [str(channel_names[index]) for index in indices]}
            for name, indices in subsets.items()
        },
    )
    write_run_manifest(output_dir, cfg, "completed" if rows else "needs_input", {"data": "real_bci2a", "protocol": "subject_T_to_E"})
    return {"rows": rows}


def run_asynchronous_bci(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    torch_status = _require_torch(output_dir, cfg, "E13 asynchronous BCI")
    if torch_status:
        return torch_status
    dcfg = resolve_dataset_cfg(cfg, key="physionet_config")
    async_cfg = cfg.get("async", {})
    async_root = Path(async_cfg.get("root", dcfg.get("async_root", "data/processed/async_physionet")))
    if async_root.exists():
        windows = _load_async_window_npz(async_root)
        source_label = str(async_root)
        source_type = "real_async"
    else:
        if not bool(async_cfg.get("allow_pseudo_from_trials", False)):
            return needs_input(
                output_dir,
                cfg,
                f"E13 requires real continuous PhysioNet window NPZ data at {async_root}. "
                "Run scripts/prepare_physionet_async.py first. "
                "Set async.allow_pseudo_from_trials=true only for smoke tests.",
            )
        bci_root = Path(resolve_dataset_cfg(cfg, key="bci2a_config").get("root", "data/processed/bci2a"))
        data = load_processed_npz(bci_root) if bci_root.exists() else None
        if data is None:
            data = generate_delay_phase_dataset(_synthetic_cfg_from_runner(cfg))
            data["dataset_name"] = "synthetic_pseudo_async"
        max_trials = async_cfg.get("max_pseudo_trials")
        if max_trials and len(data["y"]) > int(max_trials):
            rng = np.random.default_rng(int(cfg.get("seed", 0)))
            chosen = np.asarray(sorted(rng.choice(len(data["y"]), size=int(max_trials), replace=False)), dtype=int)
            data = subset(data, chosen)
        windows = make_pseudo_async_from_trials(
            data["X"],
            data["y"],
            float(data.get("sfreq", 250.0)),
            window_sec=float(async_cfg.get("window_sec", 1.0)),
            step_sec=float(async_cfg.get("step_sec", 0.25)),
            seed=int(cfg.get("seed", 0)),
        )
        windows["dataset_name"] = data.get("dataset_name", "bci2a_pseudo_async")
        source_label = str(windows["dataset_name"])
        source_type = "pseudo_async_smoke"
    required_real = {"recording_id", "subject", "event_id", "event_uid", "event_start"}
    if source_type == "real_async" and not required_real.issubset(windows):
        missing = sorted(required_real.difference(windows))
        return needs_input(
            output_dir,
            cfg,
            f"E13 real asynchronous input is missing required recording metadata {missing}. "
            "Regenerate it with scripts/prepare_physionet_async.py.",
        )
    binary = {
        "X": windows["X"],
        "y": windows["y_binary"],
        "sfreq": windows.get("sfreq", 250.0),
        "dataset_name": windows.get("dataset_name", "async"),
        "window_index": np.arange(len(windows["y_binary"]), dtype=int),
        "window_start": windows.get("window_start", np.arange(len(windows["y_binary"]), dtype=int)),
        "event_id": windows.get("event_id", np.full(len(windows["y_binary"]), -1, dtype=np.int64)),
        "event_uid": windows.get("event_uid", np.asarray(["pseudo:-1"] * len(windows["y_binary"]))),
        "event_start": windows.get("event_start", np.full(len(windows["y_binary"]), -1, dtype=np.int64)),
        "onset_latency_sec": windows.get("onset_latency_sec", np.full(len(windows["y_binary"]), np.nan)),
        "subject": windows.get("subject", np.asarray(["pseudo"] * len(windows["y_binary"]))),
        "recording_id": windows.get("recording_id", np.asarray(["pseudo"] * len(windows["y_binary"]))),
    }
    if len(np.unique(binary["y"])) < 2:
        return needs_input(output_dir, cfg, "E13 rest-vs-MI detection requires both rest and MI windows.")
    mi_mask = windows["y_mi"] >= 0
    mi_raw_labels = np.asarray(windows["y_mi"][mi_mask], dtype=int)
    mi_unique = sorted(np.unique(mi_raw_labels).tolist())
    mi_label_map = {int(label): idx for idx, label in enumerate(mi_unique)}
    mi_y = np.asarray([mi_label_map[int(label)] for label in mi_raw_labels], dtype=np.int64)
    mi = {
        "X": windows["X"][mi_mask],
        "y": mi_y,
        "sfreq": windows.get("sfreq", 250.0),
        "dataset_name": windows.get("dataset_name", "async"),
        "window_index": np.arange(len(windows["y_mi"]), dtype=int)[mi_mask],
        "event_id": binary["event_id"][mi_mask],
        "event_uid": binary["event_uid"][mi_mask],
        "event_start": binary["event_start"][mi_mask],
        "onset_latency_sec": binary["onset_latency_sec"][mi_mask],
        "subject": binary["subject"][mi_mask],
        "recording_id": binary["recording_id"][mi_mask],
    }
    if source_type != "real_async":
        return needs_input(output_dir, cfg, "E13 pseudo-async output is only a smoke test and cannot satisfy the real continuous-data protocol.")
    try:
        train_bin, val_bin, test_bin, split_audit = _split_async_recording_train_validation_test(binary)
    except ValueError as exc:
        return needs_input(output_dir, cfg, f"E13 cannot form leakage-free recording splits: {exc}")
    bin_result = train_split("dpc_snn", train_bin, val_bin, cfg, output_dir / "rest_vs_mi", test=test_bin)
    false_activations = float(((bin_result["y_pred"] == 1) & (bin_result["y_true"] == 0)).sum())
    rest_minutes = max(
        1e-6,
        float((bin_result["y_true"] == 0).sum()) * float(async_cfg.get("step_sec", 0.25)) / 60.0,
    )
    detected_latencies = []
    missed_events = 0
    for event_uid in sorted(set(np.asarray(test_bin["event_uid"])[np.asarray(test_bin["y"]) == 1])):
        event_mask = np.asarray(test_bin["event_uid"]) == event_uid
        positive = np.asarray(bin_result["y_pred"])[event_mask] == 1
        if positive.any():
            detected_latencies.append(float(np.nanmin(np.asarray(test_bin["onset_latency_sec"])[event_mask][positive])))
        else:
            missed_events += 1
    rows = [
        _metric_row(
            str(binary["dataset_name"]),
            "asynchronous_rest_vs_mi",
            "all",
            "dpc_snn",
            cfg.get("seed", 0),
            bin_result["metrics"],
            false_activation_per_min=false_activations / rest_minutes,
            mean_detection_latency_sec=float(np.mean(detected_latencies)) if detected_latencies else np.nan,
            missed_mi_intervals=missed_events,
            source=source_type,
            evaluated_on_heldout_test=True,
            **split_audit,
        )
    ]
    if len(np.unique(mi["y"])) > 1 and len(mi["y"]) >= 8:
        try:
            train_mi, val_mi, test_mi, mi_split_audit = _split_async_recording_train_validation_test(mi)
            mi_result = train_split("dpc_snn", train_mi, val_mi, cfg, output_dir / "mi_classifier", test=test_mi)
            rows.append(
                _metric_row(
                    str(mi["dataset_name"]),
                    "asynchronous_mi_classification",
                    "all",
                    "dpc_snn",
                    cfg.get("seed", 0),
                    mi_result["metrics"],
                    source=source_type,
                    evaluated_on_heldout_test=True,
                    **mi_split_audit,
                )
            )
        except ValueError as exc:
            rows.append(
                {
                    "dataset": str(mi["dataset_name"]),
                    "protocol": "asynchronous_mi_classification",
                    "subject": "all",
                    "model": "dpc_snn",
                    "seed": cfg.get("seed", 0),
                    "status": "needs_input",
                    "error": f"Leakage-free MI split unavailable: {exc}",
                }
            )
    validation_predictions = {
        int(idx): (int(true), int(pred))
        for idx, true, pred in zip(test_bin["window_index"], bin_result["y_true"], bin_result["y_pred"], strict=False)
    }
    pred_rows = []
    for idx, (start, y_bin, y_mi) in enumerate(zip(windows["window_start"], windows["y_binary"], windows["y_mi"], strict=False)):
        pred_true, pred = validation_predictions.get(idx, (None, None))
        pred_rows.append(
            {
                "window_index": idx,
                "window_start": int(start),
                "event_uid": str(binary["event_uid"][idx]),
                "true_binary": int(y_bin),
                "true_mi": int(y_mi),
                "onset_latency_sec": float(windows.get("onset_latency_sec", np.full(len(windows["y_binary"]), np.nan))[idx]),
                "prediction_split": "heldout_recording_test" if idx in validation_predictions else "not_evaluated",
                "eval_true_binary": pred_true,
                "pred_binary": pred,
            }
        )
    write_csv(output_dir / "async_detection_results.csv", rows)
    write_csv(output_dir / "sliding_window_predictions.csv", pred_rows)
    write_json(output_dir / "mi_label_mapping.json", {str(key): value for key, value in mi_label_map.items()})
    write_run_manifest(
        output_dir,
        cfg,
        "completed",
        {"source": source_label, "source_type": source_type, "split_audit": split_audit},
    )
    return {"rows": rows}


def run_physionet_validation(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    data = _load_external_processed(cfg, "physionet_config")
    if data is None:
        dcfg = resolve_dataset_cfg(cfg, key="physionet_config")
        processed = Path(dcfg.get("processed_root", "data/processed/physionet_eegmmi"))
        if processed.exists():
            data = load_processed_npz(processed)
            data["dataset_name"] = "physionet_eegmmi"
        else:
            write_json(output_dir / "task_mapping.json", physionet_task_mapping())
            return needs_input(output_dir, cfg, f"PhysioNet processed NPZ files not found at {processed}. Run scripts/prepare_physionet.py or provide NPZ files.")
    torch_status = _require_torch(output_dir, cfg, "E10 PhysioNet validation")
    if torch_status:
        return torch_status
    rows = _run_subjectwise_random_split(data, cfg, output_dir, dataset_name="physionet_eegmmi", protocol="physionet_external")
    write_csv(output_dir / "physionet_results.csv", rows)
    write_json(output_dir / "task_mapping.json", physionet_task_mapping())
    write_run_manifest(output_dir, cfg, "completed" if rows else "needs_input")
    return {"rows": rows}


def run_moabb_mini(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    dcfg = resolve_dataset_cfg(cfg, key="moabb_config")
    processed_root = Path(dcfg.get("processed_root", "data/processed/moabb_mini"))
    datasets: list[dict[str, Any]] = []
    if processed_root.exists():
        for path in sorted(processed_root.glob("*.npz")):
            item = load_processed_npz(path)
            item["dataset_name"] = path.stem
            datasets.append(item)
    else:
        try:
            datasets = load_moabb_motor_imagery(
                dcfg.get("datasets", []),
                max_subjects=dcfg.get("max_subjects"),
                tmin=float(dcfg.get("tmin", 0.0)),
                tmax=float(dcfg.get("tmax", 4.0)),
                include_rest=bool(dcfg.get("include_rest", False)),
                resample=dcfg.get("resample"),
                n_classes=dcfg.get("n_classes"),
                events=dcfg.get("events"),
            )
        except Exception as exc:
            return needs_input(output_dir, cfg, f"MOABB mini benchmark needs processed NPZ files or installed MOABB datasets: {exc!r}")
    torch_status = _require_torch(output_dir, cfg, "E11 MOABB mini benchmark")
    if torch_status:
        return torch_status
    rows = []
    requested_protocol = str(dcfg.get("protocol", "within_session"))
    if requested_protocol != "within_session":
        return needs_input(
            output_dir,
            cfg,
            f"Unsupported MOABB protocol {requested_protocol!r}; only explicit within_session evaluation is implemented.",
        )
    models = _benchmark_model_names(
        cfg,
        include_classical=True,
        configured_models=dcfg.get("pipelines"),
    )
    for item in datasets:
        rows.extend(
            _run_moabb_within_session_split(
                item,
                cfg,
                output_dir / str(item.get("name", item.get("dataset_name", "dataset"))),
                dataset_name=str(item.get("name", item.get("dataset_name", "moabb"))),
                model_names=models,
            )
        )
    write_csv(output_dir / "moabb_mini_results.csv", rows)
    rank_rows = _average_rank_rows(rows)
    write_csv(output_dir / "average_rank.csv", rank_rows)
    best_rows = _per_subject_best_rows(rows)
    write_csv(output_dir / "per_subject_best.csv", best_rows)
    completed_models = {row.get("model", "") for row in rows if row.get("status", "completed") == "completed"}
    failures = [row for row in rows if row.get("status") == "failed"]
    status = "completed" if set(models).issubset(completed_models) and not failures else "needs_input"
    dataset_manifest = [
        {
            "name": str(item.get("name", item.get("dataset_name", "moabb"))),
            "n_trials": int(len(item["y"])),
            "n_channels": int(np.asarray(item["X"]).shape[1]),
            "n_time": int(np.asarray(item["X"]).shape[2]),
            "sfreq": float(item["sfreq"]),
            "resample_sfreq": float(item.get("resample_sfreq", item["sfreq"])),
            "n_classes": int(item.get("n_classes", dcfg.get("n_classes", 0))),
            "task_events": [str(event) for event in item.get("task_events", dcfg.get("events", []))],
            "label_map": {str(key): int(value) for key, value in item.get("label_map", {}).items()},
            "skipped_subjects": item.get("skipped_subjects", []),
        }
        for item in datasets
    ]
    write_json(output_dir / "dataset_manifest.json", {"datasets": dataset_manifest, "models": models})
    write_run_manifest(
        output_dir,
        cfg,
        status,
        {"models": models, "completed_models": sorted(completed_models), "failed_rows": len(failures)},
    )
    return {"rows": rows, "average_rank": rank_rows, "status": status}


def run_cross_dataset_fewshot(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    source = _load_bci_data_or_status(cfg, output_dir)
    target = _load_external_processed(cfg, "physionet_config")
    if source is None or target is None:
        return needs_input(output_dir, cfg, "Cross-dataset few-shot requires source BCI2a NPZ and target PhysioNet/OpenBMI-style NPZ.")
    torch_status = _require_torch(output_dir, cfg, "E12 cross-dataset few-shot")
    if torch_status:
        return torch_status
    source = _crop_to_common_task_window(source, cfg)
    target = _crop_to_common_task_window(target, cfg)
    source, source_space = _apply_physical_eeg_space(source, cfg)
    target, target_space = _apply_physical_eeg_space(target, cfg)
    if source_space != target_space:
        raise ValueError("E12 source and target data used different physical EEG spaces")
    source, target, channel_audit = _common_channel_alignment(source, target, min_common=int(cfg.get("transfer_min_common_channels", 3)))
    if channel_audit["status"] != "aligned_by_name":
        return needs_input(
            output_dir,
            cfg,
            f"E12 cross-dataset transfer requires named common channels. Alignment status: {channel_audit}",
        )
    source_train_idx, source_val_idx = stratified_split_indices(source["y"], val_fraction=0.15, seed=int(cfg.get("seed", 0)))
    source_train = subset(source, source_train_idx)
    source_val = subset(source, source_val_idx)
    source_train, source_val = _standardize_pair(source_train, source_val)
    pretrain = train_split("dpc_snn", source_train, source_val, cfg, output_dir / "source_pretrain")
    checkpoint = Path(pretrain["output_dir"]) / "model_checkpoint.pt"
    _, _, _, transfer_audit = _clone_model_with_checkpoint_audit("dpc_snn", target, cfg, checkpoint)
    min_ratio = float(cfg.get("transfer_min_loaded_param_ratio", 0.25))
    if transfer_audit["loaded_param_ratio"] < min_ratio:
        return needs_input(
            output_dir,
            cfg,
            f"E12 checkpoint is not structurally compatible with target after channel alignment: loaded_param_ratio={transfer_audit['loaded_param_ratio']:.4f}, required>={min_ratio:.4f}.",
        )
    rows = []
    for train_idx, test_idx, target_subject in leave_one_subject_out(np.asarray(target["subject"]).astype(str)):
        target_subject_data = subset(target, test_idx)
        for k in [1, 5, 10, 20, 50]:
            shot_idx = k_shot_indices(target_subject_data["y"], k=k, seed=int(cfg.get("seed", 0)))
            eval_idx = np.setdiff1d(np.arange(len(target_subject_data["y"])), shot_idx)
            if shot_idx.size == 0 or eval_idx.size == 0:
                continue
            calib = subset(target_subject_data, shot_idx)
            eval_data = subset(target_subject_data, eval_idx)
            calib, eval_data = _standardize_pair(calib, eval_data)
            for mode in ["readout", "phase_delay", "full"]:
                from dpc_snn.training.adaptation import supervised_finetune

                model, calib_feat, run_cfg = _clone_model_from_checkpoint("dpc_snn", calib, cfg, checkpoint)
                eval_feat = add_features(eval_data, _feature_bands_from_cfg(cfg))
                result = supervised_finetune(model, calib_feat, eval_feat, run_cfg, output_dir / f"target_{target_subject}_k{k}_{mode}", mode=mode)
                rows.append(
                    _metric_row(
                        "bci2a_to_physionet",
                        "cross_dataset_fewshot",
                        target_subject,
                        "dpc_snn",
                        cfg.get("seed", 0),
                        result,
                        calibration_k=k,
                        calibration_mode=mode,
                        common_channels=channel_audit["common_channels"],
                        channel_alignment=channel_audit["status"],
                        checkpoint_loaded_param_ratio=transfer_audit["loaded_param_ratio"],
                        physical_eeg_space=source_space,
                    )
                )
    write_csv(output_dir / "cross_dataset_fewshot.csv", rows)
    write_json(output_dir / "channel_alignment.json", channel_audit)
    write_run_manifest(output_dir, cfg, "completed" if rows else "needs_input", {"channel_alignment": channel_audit, "checkpoint_audit": transfer_audit})
    return {"rows": rows}


def _failed_model_row(
    *,
    dataset_name: str,
    protocol: str,
    subject: str,
    model_name: str,
    cfg: dict[str, Any],
    error: Exception,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "protocol": protocol,
        "subject": subject,
        "model": _normalize_model_name(model_name),
        "seed": cfg.get("seed", 0),
        "status": "failed",
        "error": repr(error),
        **extra,
    }


def _run_subjectwise_random_split(
    data: dict[str, Any],
    cfg: dict[str, Any],
    output_dir: str | Path,
    dataset_name: str,
    protocol: str,
    model_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    output_dir = ensure_dir(output_dir)
    subjects = np.asarray(data.get("subject", np.asarray(["all"] * len(data["y"])))).astype(str)
    model_names = model_names or ["dpc_snn"]
    rows = []
    for subject in sorted(set(subjects)):
        idx = np.where(subjects == subject)[0]
        if idx.size < 8 or len(np.unique(data["y"][idx])) < 2:
            continue
        subject_data = subset(data, idx)
        train_idx, test_idx = stratified_split_indices(subject_data["y"], val_fraction=0.3, seed=int(cfg.get("seed", 0)))
        train = subset(subject_data, train_idx)
        test = subset(subject_data, test_idx)
        for model_name in model_names:
            try:
                rows.append(
                    _run_train_test_model(
                        model_name,
                        train,
                        test,
                        cfg,
                        output_dir / f"subject_{subject}",
                        dataset_name=dataset_name,
                        protocol=protocol,
                        subject=subject,
                    )
                )
            except Exception as exc:
                rows.append(
                    _failed_model_row(
                        dataset_name=dataset_name,
                        protocol=protocol,
                        subject=subject,
                        model_name=model_name,
                        cfg=cfg,
                        error=exc,
                    )
                )
    return rows


def _run_moabb_within_session_split(
    data: dict[str, Any],
    cfg: dict[str, Any],
    output_dir: str | Path,
    dataset_name: str,
    model_names: list[str],
) -> list[dict[str, Any]]:
    """Evaluate each model within the same MOABB subject-session recording.

    This is deliberately separate from the generic subjectwise splitter.  MOABB
    sessions can differ materially in acquisition state; pooling them before a
    random split would violate the declared within-session protocol.
    """

    output_dir = ensure_dir(output_dir)
    subjects = np.asarray(data.get("subject", np.asarray(["unknown"] * len(data["y"])))).astype(str)
    sessions = np.asarray(data.get("session", np.asarray(["unknown"] * len(data["y"])))).astype(str)
    if np.any(sessions == "unknown"):
        raise ValueError("MOABB within-session evaluation requires non-empty session metadata for every trial.")
    rows = []
    for subject in sorted(set(subjects)):
        for session in sorted(set(sessions[subjects == subject])):
            idx = np.where((subjects == subject) & (sessions == session))[0]
            if idx.size < 8 or len(np.unique(np.asarray(data["y"])[idx])) < 2:
                continue
            group = subset(data, idx)
            train_idx, test_idx = stratified_split_indices(
                group["y"], val_fraction=0.3, seed=int(cfg.get("seed", 0))
            )
            if train_idx.size == 0 or test_idx.size == 0:
                continue
            train = subset(group, train_idx)
            test = subset(group, test_idx)
            for model_name in model_names:
                try:
                    rows.append(
                        _run_train_test_model(
                            model_name,
                            train,
                            test,
                            cfg,
                            output_dir / f"subject_{subject}_session_{session}",
                            dataset_name=dataset_name,
                            protocol="moabb_within_session",
                            subject=subject,
                            session=session,
                            moabb_protocol="within_session",
                            selection_session=session,
                            evaluation_session=session,
                            selection_split_label="within_session_inner_validation",
                            evaluation_split="within_session_heldout_trials",
                        )
                    )
                except Exception as exc:
                    rows.append(
                        _failed_model_row(
                            dataset_name=dataset_name,
                            protocol="moabb_within_session",
                            subject=subject,
                            model_name=model_name,
                            cfg=cfg,
                            error=exc,
                            session=session,
                            moabb_protocol="within_session",
                        )
                    )
    return rows


def _average_rank_rows(rows: list[dict[str, Any]], metric: str = "accuracy") -> list[dict[str, Any]]:
    """Average ranks within matched dataset-subject-session evaluation units."""

    grouped: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for row in rows:
        if row.get("status", "completed") != "completed":
            continue
        value = _as_finite_float(row.get(metric))
        if value is None:
            continue
        key = (
            str(row.get("dataset", "")),
            str(row.get("protocol", "")),
            str(row.get("subject", "")),
            str(row.get("session", "")),
        )
        grouped.setdefault(key, {})[str(row.get("model", ""))] = value
    ranks: dict[str, list[float]] = {}
    metrics: dict[str, list[float]] = {}
    for values in grouped.values():
        ordered = sorted(values.items(), key=lambda item: item[1], reverse=True)
        index = 0
        while index < len(ordered):
            end = index + 1
            while end < len(ordered) and np.isclose(ordered[end][1], ordered[index][1]):
                end += 1
            rank = float((index + 1 + end) / 2.0)
            for model, value in ordered[index:end]:
                ranks.setdefault(model, []).append(rank)
                metrics.setdefault(model, []).append(value)
            index = end
    return [
        {
            "model": model,
            "average_rank": float(np.mean(ranks[model])),
            "n_evaluation_units": int(len(ranks[model])),
            f"mean_{metric}": float(np.mean(metrics[model])),
        }
        for model in sorted(ranks, key=lambda name: (float(np.mean(ranks[name])), name))
    ]


def _as_finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _per_subject_best_rows(rows: list[dict[str, Any]], metric: str = "accuracy") -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status", "completed") != "completed" or _as_finite_float(row.get(metric)) is None:
            continue
        by_key.setdefault(
            (str(row.get("dataset", "")), str(row.get("subject", "")), str(row.get("model", ""))),
            [],
        ).append(row)
    by_subject: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (dataset, subject, model), values in by_key.items():
        by_subject.setdefault((dataset, subject), []).append(
            {
                "model": model,
                metric: float(np.mean([float(row[metric]) for row in values])),
                "n_sessions": len({str(row.get("session", "")) for row in values}),
            }
        )
    out = []
    for (dataset, subject), values in by_subject.items():
        if not values:
            continue
        best = max(values, key=lambda row: float(row[metric]))
        out.append(
            {
                "dataset": dataset,
                "subject": subject,
                "best_model": best["model"],
                metric: best[metric],
                "n_sessions": best["n_sessions"],
            }
        )
    return out


def run_unlabeled_adaptation(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    data = _load_bci_data_or_status(cfg, output_dir)
    if data is None:
        return {"status": "needs_input"}
    torch_status = _require_torch(output_dir, cfg, "E9 unlabeled adaptation")
    if torch_status:
        return torch_status
    data = _crop_to_common_task_window(data, cfg)
    rows = []
    confidence_rows = []
    methods = ["entropy", "consistency", "pseudo_label"]
    for train_idx, test_idx, target in leave_one_subject_out(np.asarray(data["subject"]).astype(str)):
        source = subset(data, train_idx)
        target_raw = subset(data, test_idx)
        source, source_space = _apply_physical_eeg_space(source, cfg)
        target_raw, target_space = _apply_physical_eeg_space(target_raw, cfg)
        if source_space != target_space:
            raise ValueError("E9 source and target data used different physical EEG spaces")
        source_train_idx, source_val_idx = stratified_split_indices(source["y"], val_fraction=0.15, seed=int(cfg.get("seed", 0)))
        train_source = subset(source, source_train_idx)
        val_source = subset(source, source_val_idx)
        x_train, x_val, stats = standardize_train_test(train_source["X"], val_source["X"])
        train_source["X"], val_source["X"] = x_train, x_val
        pretrain_result = train_split("dpc_snn", train_source, val_source, cfg, output_dir / f"target_{target}_pretrain")
        checkpoint = Path(pretrain_result["output_dir"]) / "model_checkpoint.pt"
        try:
            unlabeled_idx, eval_idx = _target_session_unlabeled_eval_indices(target_raw)
        except ValueError:
            return needs_input(
                output_dir,
                cfg,
                f"E9 target subject {target} requires separate T (unlabeled adaptation) and E (labelled evaluation) sessions.",
            )
        # Selection is session-defined before target labels are ever read.
        unlabeled = subset(target_raw, unlabeled_idx)
        eval_data = subset(target_raw, eval_idx)
        unlabeled["X"] = apply_channelwise_zscore(unlabeled["X"], stats["mean"], stats["std"])
        eval_data["X"] = apply_channelwise_zscore(eval_data["X"], stats["mean"], stats["std"])
        source_only = _evaluate_checkpoint_on_data(
            "dpc_snn",
            eval_data,
            cfg,
            checkpoint,
            output_dir / f"target_{target}_source_only_eval",
        )
        rows.append(
            _metric_row(
                "bci2a",
                "unlabeled_adaptation",
                target,
                "dpc_snn",
                cfg.get("seed", 0),
                source_only,
                adaptation_method="source_only",
                calibration_mode="none",
                evaluated_on_target=True,
                unlabeled_trials=int(unlabeled_idx.size),
                target_eval_trials=int(eval_idx.size),
                target_unlabeled_session="T",
                target_evaluation_session="E",
                target_labels_used_for_split=False,
                checkpoint_loaded_param_ratio=source_only.get("checkpoint_loaded_param_ratio", np.nan),
                physical_eeg_space=source_space,
            )
        )
        for method in methods:
            from dpc_snn.training.adaptation import unlabeled_adapt

            model, unlabeled_feat, run_cfg = _clone_model_from_checkpoint("dpc_snn", unlabeled, cfg, checkpoint)
            eval_feat = add_features(eval_data, _feature_bands_from_cfg(cfg))
            result = unlabeled_adapt(
                model,
                unlabeled_feat,
                eval_feat,
                run_cfg,
                output_dir / f"target_{target}_{method}",
                method=method,
            )
            rows.append(
                _metric_row(
                    "bci2a",
                    "unlabeled_adaptation",
                    target,
                    "dpc_snn",
                    cfg.get("seed", 0),
                    result,
                    adaptation_method=method,
                    calibration_mode=result["calibration_mode"],
                    evaluated_on_target=True,
                    unlabeled_trials=int(unlabeled_idx.size),
                    target_eval_trials=int(eval_idx.size),
                    target_unlabeled_session="T",
                    target_evaluation_session="E",
                    target_labels_used_for_split=False,
                    physical_eeg_space=source_space,
                )
            )
            for hist in result.get("history", []):
                confidence_rows.append({"subject": target, **hist})
    write_csv(output_dir / "unlabeled_adaptation_results.csv", rows)
    write_csv(output_dir / "calibration_confidence.csv", confidence_rows)
    write_run_manifest(
        output_dir,
        cfg,
        "completed" if rows else "needs_input",
        {"methods": methods, "target_protocol": "target_T_unlabeled_adaptation_to_target_E_heldout_evaluation"},
    )
    return {"rows": rows}


def _read_completed_accuracy_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            if not raw or str(raw.get("status", "completed")).lower() != "completed":
                continue
            accuracy = _as_finite_float(raw.get("accuracy"))
            if accuracy is None:
                continue
            row = dict(raw)
            row["accuracy"] = accuracy
            rows.append(row)
    return rows


def _aggregate_subject_condition_rows(
    rows: list[dict[str, Any]],
    condition_field: str,
    extra_strata: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[float]] = {}
    for row in rows:
        key = (
            str(row.get("dataset", "unknown")),
            str(row.get("protocol", "unknown")),
            str(row.get("subject", "unknown")),
            str(row.get(condition_field, "")),
            *(str(row.get(field, "")) for field in extra_strata),
        )
        grouped.setdefault(key, []).append(float(row["accuracy"]))
    aggregated = []
    for key, values in grouped.items():
        dataset, protocol, subject, condition, *extra = key
        row = {
            "dataset": dataset,
            "protocol": protocol,
            "subject": subject,
            condition_field: condition,
            "accuracy": float(np.mean(values)),
            "n_repeats": int(len(values)),
        }
        row.update({field: value for field, value in zip(extra_strata, extra, strict=False)})
        aggregated.append(row)
    return aggregated


def _hierarchical_model_tests(
    experiment: str,
    path: Path,
    extra_strata: tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _read_completed_accuracy_rows(path)
    extra_strata = tuple(field for field in extra_strata if any(str(row.get(field, "")) for row in rows))
    aggregated = _aggregate_subject_condition_rows(rows, "model", extra_strata)
    tests = []
    mixed_rows = []
    strata = sorted(
        {
            tuple([row["dataset"], row["protocol"], *[str(row[field]) for field in extra_strata]])
            for row in aggregated
        }
    )
    for stratum in strata:
        dataset, protocol, *extra = stratum
        subset_rows = [
            row
            for row in aggregated
            if row["dataset"] == dataset
            and row["protocol"] == protocol
            and all(str(row[field]) == value for field, value in zip(extra_strata, extra, strict=False))
        ]
        models = sorted({str(row["model"]) for row in subset_rows})
        if "dpc_snn" not in models:
            continue
        by_subject: dict[str, dict[str, float]] = {}
        for row in subset_rows:
            by_subject.setdefault(str(row["subject"]), {})[str(row["model"])] = float(row["accuracy"])
        for model in models:
            if model == "dpc_snn":
                continue
            diffs = np.asarray(
                [values["dpc_snn"] - values[model] for values in by_subject.values() if "dpc_snn" in values and model in values],
                dtype=float,
            )
            stat = wilcoxon_signed_rank(diffs)
            stat.update(
                {
                    "experiment": experiment,
                    "dataset": dataset,
                    "protocol": protocol,
                    "comparison_type": "primary_model",
                    "model_a": "dpc_snn",
                    "model_b": model,
                    "condition_field": "",
                    "condition_a": "",
                    "condition_b": "",
                    "metric": "accuracy",
                    "mean_difference": float(np.mean(diffs)) if diffs.size else np.nan,
                    "median_difference": float(np.median(diffs)) if diffs.size else np.nan,
                    "wins": int((diffs > 0).sum()),
                    "ties": int(np.isclose(diffs, 0.0).sum()),
                    "losses": int((diffs < 0).sum()),
                    **{field: value for field, value in zip(extra_strata, extra, strict=False)},
                }
            )
            tests.append(stat)
        if len(models) >= 2:
            for row in mixed_effects_or_fallback(subset_rows, metric="accuracy"):
                mixed_rows.append(
                    {
                        "experiment": experiment,
                        "dataset": dataset,
                        "protocol": protocol,
                        **{field: value for field, value in zip(extra_strata, extra, strict=False)},
                        **row,
                    }
                )
    return tests, mixed_rows


def _hierarchical_condition_tests(
    experiment: str,
    path: Path,
    condition_field: str,
    reference: str,
) -> list[dict[str, Any]]:
    rows = _read_completed_accuracy_rows(path)
    extra_strata = tuple(
        field
        for field in ("calibration_k", "model")
        if any(str(row.get(field, "")) for row in rows)
    )
    aggregated = _aggregate_subject_condition_rows(rows, condition_field, extra_strata)
    tests = []
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in aggregated:
        key = tuple([row["dataset"], row["protocol"], *[str(row[field]) for field in extra_strata]])
        groups.setdefault(key, []).append(row)
    for key, values in groups.items():
        dataset, protocol, *extra = key
        by_subject: dict[str, dict[str, float]] = {}
        for row in values:
            by_subject.setdefault(str(row["subject"]), {})[str(row[condition_field])] = float(row["accuracy"])
        for condition in sorted({str(row[condition_field]) for row in values}.difference({reference})):
            diffs = np.asarray(
                [conditions[condition] - conditions[reference] for conditions in by_subject.values() if condition in conditions and reference in conditions],
                dtype=float,
            )
            stat = wilcoxon_signed_rank(diffs)
            compared_model = str(dict(zip(extra_strata, extra, strict=False)).get("model", "dpc_snn"))
            stat.update(
                {
                    "experiment": experiment,
                    "dataset": dataset,
                    "protocol": protocol,
                    "comparison_type": "secondary_condition",
                    "model_a": compared_model,
                    "model_b": compared_model,
                    "condition_field": condition_field,
                    "condition_a": reference,
                    "condition_b": condition,
                    "metric": "accuracy",
                    "mean_difference": float(np.mean(diffs)) if diffs.size else np.nan,
                    "median_difference": float(np.median(diffs)) if diffs.size else np.nan,
                    "wins": int((diffs > 0).sum()),
                    "ties": int(np.isclose(diffs, 0.0).sum()),
                    "losses": int((diffs < 0).sum()),
                    **{field: value for field, value in zip(extra_strata, extra, strict=False)},
                }
            )
            tests.append(stat)
    return tests


def _apply_stratified_fdr(rows: list[dict[str, Any]]) -> None:
    families: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        families.setdefault((str(row["experiment"]), str(row["comparison_type"])), []).append(index)
    for indices in families.values():
        adjusted = fdr_bh([rows[index].get("p_value", np.nan) for index in indices])
        for index, p_fdr in zip(indices, adjusted, strict=False):
            rows[index]["p_fdr"] = p_fdr


def run_final_statistics(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Run experiment-stratified subject-level statistics.

    Distinct datasets, protocols, sessions, K values, and experiment families
    are never pooled as if they were independent observations.
    """

    output_dir = ensure_dir(output_dir)
    scan_root = Path(cfg.get("statistics_results_root", Path(output_dir).parent))
    primary_specs = {
        "E11": "moabb_mini_results.csv",
        "E14": "baseline_audit.csv",
    }
    condition_specs = {
        "E8": ("few_shot_results.csv", "calibration_mode", "source_only"),
        "E9": ("unlabeled_adaptation_results.csv", "adaptation_method", "source_only"),
        "E12": ("cross_dataset_fewshot.csv", "calibration_mode", "readout"),
    }
    tests = []
    mixed_rows = []
    scanned_csv: list[str] = []
    for experiment, filename in primary_specs.items():
        path = scan_root / experiment / filename
        if not path.exists():
            continue
        scanned_csv.append(str(path))
        experiment_tests, experiment_mixed = _hierarchical_model_tests(experiment, path)
        tests.extend(experiment_tests)
        mixed_rows.extend(experiment_mixed)
    e8_path = scan_root / "E8" / "few_shot_results.csv"
    if e8_path.exists():
        experiment_tests, experiment_mixed = _hierarchical_model_tests(
            "E8",
            e8_path,
            extra_strata=("calibration_k", "calibration_mode"),
        )
        tests.extend(experiment_tests)
        mixed_rows.extend(experiment_mixed)
    for experiment, (filename, condition_field, reference) in condition_specs.items():
        path = scan_root / experiment / filename
        if not path.exists():
            continue
        scanned_csv.append(str(path))
        tests.extend(_hierarchical_condition_tests(experiment, path, condition_field, reference))
    _apply_stratified_fdr(tests)
    write_csv(output_dir / "stat_tests.csv", tests)
    write_csv(output_dir / "mixed_effects_results.csv", mixed_rows)
    write_json(
        output_dir / "statistics_protocol.json",
        {
            "primary_model_comparisons": primary_specs,
            "secondary_condition_comparisons": condition_specs,
            "unit": "subject-level mean over available sessions and seeds",
            "strata": ["experiment", "dataset", "protocol"],
            "fdr_family": ["experiment", "comparison_type"],
            "excluded_from_primary_pooling": ["different datasets", "different protocols", "different calibration K values", "synthetic single-run controls"],
        },
    )
    status_rows = []
    for path in sorted(set(scan_root.glob("**/runner_status.json")) | set(scan_root.glob("**/run_manifest.json"))):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            status_rows.append({"source": str(path), "status": "failed", "error": repr(exc)})
            continue
        status = str(payload.get("status", "")).lower()
        if status in {"failed", "needs_input", "partial"}:
            status_rows.append(
                {
                    "source": str(path),
                    "status": status,
                    "message": payload.get("message", ""),
                    "experiment_id": payload.get("experiment_id", ""),
                }
            )
    failure_text = "# Failure Cases\n\n"
    if status_rows:
        for row in status_rows:
            failure_text += f"- source={row.get('source', '')} status={row.get('status', '')} experiment={row.get('experiment_id', '')} message={row.get('message', '')} error={row.get('error', '')}\n"
    else:
        failure_text += "No failed, partial, or missing-input experiment manifests were found.\n"
    (output_dir / "failure_cases.md").write_text(failure_text, encoding="utf-8")
    experiment_ids = [f"E{i}" for i in range(25)]
    reproduce_sh = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'CONFIG="${1:-configs/experiments/all_experiments.yaml}"',
            'OUT="${2:-runs/reproduce_all}"',
            'mkdir -p "$OUT"',
            *[f'python scripts/run_experiment.py --config "$CONFIG" --experiment {exp_id} --output "$OUT/{exp_id}"' for exp_id in experiment_ids],
            'python scripts/reproduce_figures.py --results "$OUT" --figures "$OUT/figures"',
            "",
        ]
    )
    (output_dir / "reproduce_all.sh").write_text(reproduce_sh, encoding="utf-8")
    reproduce_ps1 = "\n".join(
        [
            'param([string]$Config = "configs/experiments/all_experiments.yaml", [string]$Out = "runs/reproduce_all")',
            '$ErrorActionPreference = "Stop"',
            "New-Item -ItemType Directory -Force -Path $Out | Out-Null",
            *[f'python scripts/run_experiment.py --config $Config --experiment {exp_id} --output (Join-Path $Out "{exp_id}")' for exp_id in experiment_ids],
            'python scripts/reproduce_figures.py --results $Out --figures (Join-Path $Out "figures")',
            "",
        ]
    )
    (output_dir / "reproduce_all.ps1").write_text(reproduce_ps1, encoding="utf-8")
    write_run_manifest(
        output_dir,
        cfg,
        "completed" if tests else "partial",
        {
            "statistics_results_root": str(scan_root),
            "scanned_csv": scanned_csv,
            "status_failures": status_rows,
            "statistics_protocol": "experiment_stratified_subject_level_v2",
        },
    )
    return {"tests": tests, "mixed_effects": mixed_rows}


RUNNERS = {
    "environment_audit": run_environment_audit,
    "synthetic_recovery": run_synthetic_recovery,
    "scalp_mixing": run_scalp_mixing,
    "negative_controls": run_negative_controls,
    "preprocessing_qc": run_preprocessing_qc,
    "bci2a_subject_dependent": run_bci2a_subject_dependent,
    "bci2a_cross_session": run_bci2a_cross_session,
    "bci2a_loso": run_bci2a_loso,
    "few_shot_calibration": run_few_shot_calibration,
    "unlabeled_adaptation": run_unlabeled_adaptation,
    "physionet_validation": run_physionet_validation,
    "moabb_mini": run_moabb_mini,
    "cross_dataset_fewshot": run_cross_dataset_fewshot,
    "asynchronous_bci": run_asynchronous_bci,
    "baseline_fairness": run_baseline_fairness,
    "mechanism_controls": run_mechanism_controls,
    "ablation_core": run_ablation_core,
    "sensitivity": run_sensitivity,
    "neurophysiology": run_neurophysiology,
    "explainability_sanity": run_explainability_sanity,
    "reference_sensitivity": run_reference_sensitivity,
    "efficiency_latency": run_efficiency_latency,
    "artifact_robustness": run_artifact_robustness,
    "low_channel": run_low_channel,
    "final_statistics": run_final_statistics,
}


def run_registered_experiment(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    runner_name = cfg.get("experiment", {}).get("runner")
    if runner_name not in RUNNERS:
        known = ", ".join(sorted(RUNNERS))
        raise KeyError(f"Unknown runner {runner_name!r}. Known runners: {known}")
    result = RUNNERS[runner_name](cfg, output_dir)
    validate_experiment_contract(cfg, output_dir, result if isinstance(result, dict) else {})
    summary_path = Path(output_dir) / "summary.json"
    ensure_dir(summary_path.parent)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    return result
