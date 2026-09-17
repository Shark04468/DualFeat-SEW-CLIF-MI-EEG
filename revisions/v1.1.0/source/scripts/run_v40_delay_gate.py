#!/usr/bin/env python
"""Run the V4.2 Subject-1 identifiable-delay gate without Session E access."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import signal
import sys
import traceback
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.config import load_experiment_config, load_yaml  # noqa: E402
from dpc_snn.data.bci2a import load_processed_npz, subject_session_data  # noqa: E402
from dpc_snn.data.splits import stratified_split_indices  # noqa: E402
from dpc_snn.experiments.common import (  # noqa: E402
    resolve_dataset_cfg,
    resolve_model_cfg,
    subset,
    train_split,
)
from dpc_snn.experiments.runners import (  # noqa: E402
    _apply_physical_eeg_space,
    _crop_to_common_task_window,
)
from dpc_snn.preprocessing.standardize import standardize_train_test  # noqa: E402
from dpc_snn.utils.io import ensure_dir, write_csv, write_json  # noqa: E402


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment_manifest(device: str) -> dict[str, Any]:
    dependency_names = (
        "numpy",
        "pandas",
        "PyYAML",
        "torch",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "statsmodels",
        "pyriemann",
        "mne",
        "moabb",
    )
    installed_packages: dict[str, str | None] = {}
    for package in dependency_names:
        try:
            installed_packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            installed_packages[package] = None
    manifest: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "requested_device": str(device),
        "requirements_sha256": _file_sha256(ROOT / "requirements.txt"),
        "pyproject_sha256": _file_sha256(ROOT / "pyproject.toml"),
        "installed_packages": installed_packages,
    }
    if torch.cuda.is_available():
        manifest["cuda_device_name"] = torch.cuda.get_device_name(0)
        manifest["cuda_capability"] = list(torch.cuda.get_device_capability(0))
    return manifest


def _source_fingerprint(protocol_path: Path, base_config_path: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((ROOT / "src").rglob("*.py"))
    paths.extend(sorted((ROOT / "configs").rglob("*.yaml")))
    paths.extend(
        [
            ROOT / "configs" / "models" / "dpc_snn.yaml",
            ROOT / "requirements.txt",
            ROOT / "pyproject.toml",
            base_config_path,
            protocol_path,
        ]
    )
    paths.extend(sorted((ROOT / "scripts").glob("*.py")))
    paths = sorted(set(paths))
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _data_fingerprint(*datasets: dict[str, Any]) -> str:
    """Bind resumptions to the exact standardized carrier/evidence fold."""

    digest = hashlib.sha256()
    for split_index, data in enumerate(datasets):
        digest.update(str(split_index).encode("ascii"))
        for key in ("X", "delay_evidence_X", "y"):
            value = np.ascontiguousarray(data[key])
            digest.update(key.encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.tobytes())
    return digest.hexdigest()


def _repeated_stratified_folds(
    y: np.ndarray,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic repeated folds without a scikit-learn runtime dependency."""

    labels = np.asarray(y)
    all_indices = np.arange(labels.size, dtype=int)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng(seed + 104729 * repeat)
        class_folds: dict[Any, list[np.ndarray]] = {}
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            rng.shuffle(indices)
            class_folds[label] = list(np.array_split(indices, n_splits))
        for fold in range(n_splits):
            val_idx = np.sort(
                np.concatenate([class_folds[label][fold] for label in class_folds])
            )
            train_idx = np.setdiff1d(all_indices, val_idx, assume_unique=True)
            if train_idx.size == 0 or val_idx.size == 0:
                raise ValueError("Repeated stratified CV produced an empty fold")
            splits.append((train_idx, val_idx))
    return splits


def _model_cfg(base_cfg: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    model_cfg = resolve_model_cfg(base_cfg, "dpc_snn")
    transport_edges = copy.deepcopy(protocol["transport_band_edges_hz"])
    model_cfg.update(
        {
            "architecture_version": protocol.get(
                "architecture_version", "dpc_snn_v4_2_identifiable_decoder"
            ),
            "n_bands": len(transport_edges),
            "band_edges_hz": transport_edges,
            "delay_evidence_band_edges_hz": transport_edges,
            "classification_band_edges_hz": copy.deepcopy(
                protocol["classification_band_edges_hz"]
            ),
            "baseline_tmin": -1.0,
            "delay_evidence_reference": "csd",
            "euclidean_alignment": False,
            "fold_local_evidence_prior": True,
            "require_valid_fold_delay_evidence": True,
            "fold_prior_min_accepted_edges": 1,
            "fold_prior_min_surrogate_support_drop": 0.0,
            "fold_prior_max_reversal_support_mae": 0.05,
            "reference_augmentation_prob": 0.0,
            "cumulative_readout_seconds": [],
            "freeze_shared_physical_basis": True,
            "freeze_shared_filterbank": True,
            "preserve_transport_band_pairs": True,
            "delayed_stat_channels": 4,
            "delayed_directional_moments": True,
            "temporal_pool_bins": 128,
            "rejected_route_prior_floor": 0.02,
            "freeze_delay_after_pretrain": True,
            "causal_channel_norm": True,
            "decoder_layers": 3,
        }
    )
    model_cfg.update(copy.deepcopy(protocol.get("model_overrides", {})))
    return model_cfg


def _reference_space(
    cropped: dict[str, Any], model_cfg: dict[str, Any], reference: str
) -> tuple[dict[str, Any], str]:
    cfg = {"model": {**model_cfg, "physical_reference": reference}}
    return _apply_physical_eeg_space(cropped, cfg)


def _prepare_fold(
    carrier: dict[str, Any],
    evidence: dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train = subset(carrier, train_idx)
    val = subset(carrier, val_idx)
    train_x, val_x, carrier_stats = standardize_train_test(train["X"], val["X"])
    evidence_train = subset(evidence, train_idx)
    evidence_val = subset(evidence, val_idx)
    evidence_train_x, evidence_val_x, evidence_stats = standardize_train_test(
        evidence_train["X"], evidence_val["X"]
    )
    train["X"] = train_x
    val["X"] = val_x
    train["delay_evidence_X"] = evidence_train_x
    val["delay_evidence_X"] = evidence_val_x
    train["standardize_mean"] = carrier_stats["mean"]
    train["standardize_std"] = carrier_stats["std"]
    train["delay_evidence_standardize_mean"] = evidence_stats["mean"]
    train["delay_evidence_standardize_std"] = evidence_stats["std"]
    return train, val


def _training_cfg(
    cfg: dict[str, Any],
    *,
    epochs: int,
    representation_epochs: int,
    delay_pretrain_epochs: int,
    batch_size: int,
    accumulation: int,
    learned_delay: bool,
) -> dict[str, Any]:
    run_cfg = copy.deepcopy(cfg)
    run_cfg["experiment_id"] = str(
        cfg.get("_gate_experiment_id", "V42_IDENTIFIABLE_DECODER")
    )
    run_cfg["protocol"] = str(
        cfg.get("_gate_protocol", "v42_repeated_session_T_identifiable_delay_gate")
    )
    run_cfg["training"] = {
        **run_cfg.get("training", {}),
        "epochs": epochs,
        "patience": epochs + 1,
        "representation_warmup_epochs": representation_epochs if learned_delay else 0,
        "delay_pretrain_epochs": delay_pretrain_epochs if learned_delay else 0,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "tet_loss_weight": 0.0,
        "select_last_checkpoint": True,
        "minimum_lag_anneal_joint_epochs": 10 if learned_delay else 0,
    }
    run_cfg["training"].update(
        copy.deepcopy(cfg.get("_gate_training_overrides", {}))
    )
    return run_cfg


def _load_completed(
    run_dir: Path, expected_resume_contract: dict[str, Any]
) -> dict[str, Any] | None:
    status_path = run_dir / "training_status.json"
    metrics_path = run_dir / "metrics.json"
    contract_path = run_dir / "resume_contract.json"
    required = [
        run_dir / "selection_y_true.npy",
        run_dir / "selection_y_pred.npy",
        run_dir / "selection_logits.npy",
    ]
    if (
        not status_path.exists()
        or not metrics_path.exists()
        or not contract_path.exists()
        or not all(path.exists() for path in required)
    ):
        return None
    saved_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if saved_contract != expected_resume_contract:
        return None
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        return None
    return {
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
        "selection_y_true": np.load(required[0]),
        "selection_y_pred": np.load(required[1]),
        "selection_logits": np.load(required[2]),
    }


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_variant(
    *,
    train: dict[str, Any],
    val: dict[str, Any],
    base_cfg: dict[str, Any],
    base_model_cfg: dict[str, Any],
    stage: dict[str, Any],
    run_dir: Path,
    prior_path: Path,
    args: argparse.Namespace,
    carrier_name: str,
    split_label: str,
    source_fingerprint: str,
    data_fingerprint: str,
    environment_manifest: dict[str, Any],
    fixed_route_rms: float | None = None,
    fixed_event_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    learned_delay = not bool(stage["freeze_delay_posterior"])
    run_cfg = _training_cfg(
        base_cfg,
        epochs=args.epochs,
        representation_epochs=args.representation_epochs,
        delay_pretrain_epochs=args.delay_pretrain_epochs,
        batch_size=args.batch_size,
        accumulation=args.gradient_accumulation_steps,
        learned_delay=learned_delay,
    )
    run_cfg["seed"] = args.seed
    run_cfg["subject"] = str(args.subject)
    run_cfg["device"] = args.device
    if fixed_route_rms is not None:
        run_cfg["training"]["fixed_route_rms"] = float(fixed_route_rms)
    if fixed_event_thresholds is not None:
        run_cfg["training"]["fixed_event_thresholds"] = [
            float(value) for value in fixed_event_thresholds
        ]
    run_cfg["protocol_metadata"] = {
        "selection_session": "T",
        "selection_split": split_label,
        "evaluation_session": "T",
        "heldout_test_accessed": False,
        "evidence_role": "inner_train_CSD_delay_evidence_only",
        "evidence_space": "spherical_spline_CSD",
        "carrier_space": carrier_name,
        "checkpoint_rule": f"last_epoch_{args.epochs}",
    }
    model_cfg = copy.deepcopy(base_model_cfg)
    orchestration_keys = {"name", "reuse_audited_route_gain"}
    model_cfg.update(
        {
            key: value
            for key, value in stage.items()
            if key not in orchestration_keys
        }
    )
    model_cfg["matched_transport_control"] = bool(
        stage.get(
            "matched_transport_control",
            str(stage["name"])
            in {"matched_zero_delay", "fixed_audited_delay"},
        )
    )
    # Fold-prior caches are scientific artifacts. Any implementation/config
    # change that invalidates a run must also invalidate the cached prior.
    model_cfg["fold_prior_algorithm_fingerprint"] = source_fingerprint
    model_cfg["physical_reference"] = carrier_name
    model_cfg["fold_local_evidence_prior_path"] = str(prior_path)
    initial_checkpoint_value = run_cfg.get("training", {}).get("initial_checkpoint")
    initial_checkpoint = (
        Path(initial_checkpoint_value) if initial_checkpoint_value else None
    )
    resolved_run_payload = {
        "run_cfg": run_cfg,
        "model_cfg": model_cfg,
    }
    resume_contract = {
        "source_fingerprint": source_fingerprint,
        "data_fingerprint": data_fingerprint,
        "architecture_version": model_cfg["architecture_version"],
        "subject": str(args.subject),
        "seed": int(args.seed),
        "split": split_label,
        "carrier_space": carrier_name,
        "variant": str(stage["name"]),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "checkpoint_rule": f"last_epoch_{args.epochs}",
        "fixed_route_rms": None if fixed_route_rms is None else float(fixed_route_rms),
        "fixed_event_thresholds": (
            None
            if fixed_event_thresholds is None
            else [float(value) for value in fixed_event_thresholds]
        ),
        "resolved_run_config_sha256": hashlib.sha256(
            json.dumps(
                resolved_run_payload,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest(),
        "model_config_sha256": hashlib.sha256(
            json.dumps(model_cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "initial_checkpoint_sha256": _file_sha256(initial_checkpoint),
        "fold_prior_sha256": _file_sha256(prior_path),
        "environment": environment_manifest,
        "environment_manifest_sha256": hashlib.sha256(
            json.dumps(
                environment_manifest,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest(),
    }
    completed = _load_completed(run_dir, resume_contract)
    if completed is not None:
        return completed
    ensure_dir(run_dir)
    write_json(run_dir / "resume_contract.json", resume_contract)
    result = train_split(
        "dpc_snn",
        train,
        val,
        run_cfg,
        run_dir,
        model_cfg=model_cfg,
    )
    resume_contract["fold_prior_sha256"] = _file_sha256(prior_path)
    write_json(run_dir / "resume_contract.json", resume_contract)
    return result


def _paired_cv(
    *,
    carrier_spaces: dict[str, dict[str, Any]],
    evidence: dict[str, Any],
    base_cfg: dict[str, Any],
    base_model_cfg: dict[str, Any],
    protocol: dict[str, Any],
    output: Path,
    args: argparse.Namespace,
    source_fingerprint: str,
    environment_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, float]:
    splits = _repeated_stratified_folds(
        np.asarray(evidence["y"]),
        args.cv_folds,
        args.cv_repeats,
        args.seed,
    )
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for carrier_name, carrier in carrier_spaces.items():
        for split_index, (train_idx, val_idx) in enumerate(splits):
            repeat = split_index // args.cv_folds
            fold = split_index % args.cv_folds
            train, val = _prepare_fold(carrier, evidence, train_idx, val_idx)
            fold_data_fingerprint = _data_fingerprint(train, val)
            prior_path = output / "fold_priors" / f"repeat_{repeat}_fold_{fold}.npz"
            stages = sorted(
                protocol["paired_gate_variants"],
                key=lambda value: 0 if value["name"] == "fixed_audited_delay" else 1,
            )
            shared_route_rms: float | None = None
            shared_event_thresholds: list[float] | None = None
            for stage in stages:
                name = str(stage["name"])
                needs_event_thresholds = bool(
                    base_model_cfg.get("event_native_transport", False)
                )
                if name == "matched_zero_delay" and (
                    shared_route_rms is None
                    or (needs_event_thresholds and shared_event_thresholds is None)
                ):
                    raise RuntimeError(
                        "Matched zero-delay must reuse audited gain and event thresholds"
                    )
                run_dir = output / "paired_cv" / carrier_name / f"repeat_{repeat}" / f"fold_{fold}" / name
                result = _run_variant(
                    train=train,
                    val=val,
                    base_cfg=base_cfg,
                    base_model_cfg=base_model_cfg,
                    stage=stage,
                    run_dir=run_dir,
                    prior_path=prior_path,
                    args=args,
                    carrier_name=carrier_name,
                    split_label="repeated_session_T_validation",
                    source_fingerprint=source_fingerprint,
                    data_fingerprint=fold_data_fingerprint,
                    environment_manifest=environment_manifest,
                    fixed_route_rms=(
                        shared_route_rms if name == "matched_zero_delay" else None
                    ),
                    fixed_event_thresholds=(
                        shared_event_thresholds
                        if name == "matched_zero_delay" and needs_event_thresholds
                        else None
                    ),
                )
                metrics = result["metrics"]
                if name == "fixed_audited_delay":
                    shared_route_rms = float(metrics["train_fitted_route_rms"])
                    shared_event_thresholds = [
                        float(value)
                        for value in metrics.get(
                            "train_fitted_event_thresholds", []
                        )
                    ]
                    if needs_event_thresholds and not shared_event_thresholds:
                        raise RuntimeError(
                            "Audited event-native run did not export fitted thresholds"
                        )
                metric_rows.append(
                    {
                        "carrier_space": carrier_name,
                        "repeat": repeat,
                        "fold": fold,
                        "variant": name,
                        **metrics,
                    }
                )
                logits = np.asarray(result["selection_logits"])
                y_true = np.asarray(result["selection_y_true"])
                y_pred = np.asarray(result["selection_y_pred"])
                if not np.array_equal(y_true, np.asarray(val["y"])):
                    raise RuntimeError("Saved validation predictions lost fold trial order")
                for row_index, trial_index in enumerate(val_idx):
                    prediction_rows.append(
                        {
                            "carrier_space": carrier_name,
                            "repeat": repeat,
                            "fold": fold,
                            "variant": name,
                            "trial_index": int(trial_index),
                            "y_true": int(y_true[row_index]),
                            "y_pred": int(y_pred[row_index]),
                            **{
                                f"logit_{class_index}": float(logits[row_index, class_index])
                                for class_index in range(logits.shape[1])
                            },
                        }
                    )
                write_csv(output / "paired_cv_metrics.csv", metric_rows)
                write_csv(output / "paired_trial_predictions.csv", prediction_rows)
                _release_cuda()

    summaries: list[dict[str, Any]] = []
    best_carrier = ""
    best_delta = -np.inf
    for carrier_name in carrier_spaces:
        per_fold: list[float] = []
        for split_index in range(len(splits)):
            repeat = split_index // args.cv_folds
            fold = split_index % args.cv_folds
            selected = [
                row
                for row in metric_rows
                if row["carrier_space"] == carrier_name
                and row["repeat"] == repeat
                and row["fold"] == fold
            ]
            accuracy = {str(row["variant"]): float(row["accuracy"]) for row in selected}
            per_fold.append(accuracy["fixed_audited_delay"] - accuracy["matched_zero_delay"])
        delta = float(np.mean(per_fold))
        summary = {
            "carrier_space": carrier_name,
            "folds": len(per_fold),
            "mean_fixed_minus_zero": delta,
            "median_fixed_minus_zero": float(np.median(per_fold)),
            "positive_folds": int(np.sum(np.asarray(per_fold) > 0.0)),
            "nonnegative_folds": int(np.sum(np.asarray(per_fold) >= 0.0)),
            "mechanism_gate_passed": bool(delta > 0.0),
        }
        summaries.append(summary)
        if delta > best_delta:
            best_delta = delta
            best_carrier = carrier_name
    write_csv(output / "paired_gate_summary.csv", summaries)
    return metric_rows, prediction_rows, best_carrier, best_delta


def _five_stage(
    *,
    carrier: dict[str, Any],
    carrier_name: str,
    evidence: dict[str, Any],
    base_cfg: dict[str, Any],
    base_model_cfg: dict[str, Any],
    protocol: dict[str, Any],
    output: Path,
    args: argparse.Namespace,
    source_fingerprint: str,
    environment_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    train_idx, val_idx = stratified_split_indices(
        np.asarray(carrier["y"]),
        val_fraction=args.validation_fraction,
        seed=args.seed,
    )
    train, val = _prepare_fold(carrier, evidence, train_idx, val_idx)
    fold_data_fingerprint = _data_fingerprint(train, val)
    rows: list[dict[str, Any]] = []
    stage_order = sorted(
        enumerate(protocol["five_stages"]),
        key=lambda item: (
            0 if item[1]["name"] == "fixed_audited_delay"
            else 1 if item[1]["name"] == "matched_zero_delay"
            else 2 + item[0]
        ),
    )
    shared_route_rms: float | None = None
    shared_event_thresholds: list[float] | None = None
    for stage_index, stage in stage_order:
        name = str(stage["name"])
        reuse_audited_gain = bool(
            stage.get("reuse_audited_route_gain", name == "matched_zero_delay")
        )
        needs_event_thresholds = bool(
            base_model_cfg.get("event_native_transport", False)
        )
        if reuse_audited_gain and (
            shared_route_rms is None
            or (needs_event_thresholds and shared_event_thresholds is None)
        ):
            raise RuntimeError(
                f"{name} must reuse audited gain and event thresholds"
            )
        result = _run_variant(
            train=train,
            val=val,
            base_cfg=base_cfg,
            base_model_cfg=base_model_cfg,
            stage=stage,
            run_dir=output / "five_stage" / name,
            prior_path=output / "five_stage" / "fold_prior.npz",
            args=args,
            carrier_name=carrier_name,
            split_label="session_T_fixed_inner_validation",
            source_fingerprint=source_fingerprint,
            data_fingerprint=fold_data_fingerprint,
            environment_manifest=environment_manifest,
            fixed_route_rms=(
                shared_route_rms if reuse_audited_gain else None
            ),
            fixed_event_thresholds=(
                shared_event_thresholds
                if reuse_audited_gain and needs_event_thresholds
                else None
            ),
        )
        metrics = result["metrics"]
        if name == "fixed_audited_delay":
            shared_route_rms = float(metrics["train_fitted_route_rms"])
            shared_event_thresholds = [
                float(value)
                for value in metrics.get("train_fitted_event_thresholds", [])
            ]
            if base_model_cfg.get("event_native_transport") and not shared_event_thresholds:
                raise RuntimeError(
                    "Audited event-native run did not export fitted thresholds"
                )
        expert_usage = [
            float(value) for key, value in metrics.items() if key.startswith("expert_usage_")
        ]
        rows.append(
            {
                "stage_index": stage_index,
                "stage": name,
                "carrier_space": carrier_name,
                "effective_experts_ge_5pct": int(np.sum(np.asarray(expert_usage) >= 0.05)),
                **metrics,
            }
        )
        write_csv(output / "five_stage_results.csv", rows)
        _release_cuda()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/all_experiments.yaml")
    parser.add_argument(
        "--protocol-config",
        default="configs/experiments/v42_identifiable_decoder.yaml",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--subject", default="1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--representation-epochs", type=int, default=6)
    parser.add_argument("--delay-pretrain-epochs", type=int, default=8)
    # Exact two-layer backward peaks near 28 GiB on the 32 GiB target GPU.
    # Batch 2 remains within the measured envelope and halves recurrent steps.
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--cv-repeats", type=int, default=2)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--carrier-spaces", default="car,original")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=("all", "paired_cv", "five_stage"), default="all")
    parser.add_argument("--five-stage-carrier", choices=("car", "original"), default="car")
    args = parser.parse_args()

    if args.cv_folds < 2 or args.cv_repeats < 2:
        raise ValueError("The confirmatory development gate requires repeated CV with at least two repeats")
    protocol_path = Path(args.protocol_config)
    if not protocol_path.is_absolute():
        protocol_path = (ROOT / protocol_path).resolve()
    base_config_path = Path(args.config)
    if not base_config_path.is_absolute():
        base_config_path = (ROOT / base_config_path).resolve()
    protocol = load_yaml(protocol_path)
    output = ensure_dir(args.output)
    fingerprint = _source_fingerprint(protocol_path, base_config_path)
    environment = _environment_manifest(args.device)
    experiment_id = str(protocol.get("experiment_id", "V42_IDENTIFIABLE_DECODER"))
    protocol_name = str(
        protocol.get("protocol", "v42_repeated_session_T_identifiable_delay_gate")
    )
    manifest_path = output / "run_manifest.json"
    manifest = {
        "experiment_id": experiment_id,
        "protocol": protocol_name,
        "subject": str(args.subject),
        "seed": args.seed,
        "epochs": args.epochs,
        "checkpoint_rule": "fixed_last_epoch",
        "source_fingerprint": fingerprint,
        "environment": environment,
        "heldout_session_E_accessed": False,
    }
    write_json(manifest_path, {**manifest, "status": "running"})

    def record_failure(exc_type, exc, tb) -> None:
        write_json(
            manifest_path,
            {
                **manifest,
                "status": "failed",
                "error_type": exc_type.__name__,
                "error": str(exc),
                "traceback": "".join(traceback.format_exception(exc_type, exc, tb)),
            },
        )
        sys.__excepthook__(exc_type, exc, tb)

    def record_interrupt(signum, _frame) -> None:
        write_json(manifest_path, {**manifest, "status": "interrupted", "signal": int(signum)})
        raise SystemExit(128 + int(signum))

    sys.excepthook = record_failure
    signal.signal(signal.SIGTERM, record_interrupt)
    signal.signal(signal.SIGINT, record_interrupt)

    cfg = load_experiment_config(args.config, "E14", overrides={"device": args.device})
    cfg["_gate_experiment_id"] = experiment_id
    cfg["_gate_protocol"] = protocol_name
    cfg["_gate_training_overrides"] = copy.deepcopy(
        protocol.get("training_overrides", {})
    )
    base_model_cfg = _model_cfg(cfg, protocol)
    dataset_cfg = resolve_dataset_cfg(cfg, key="bci2a_config")
    data = load_processed_npz(dataset_cfg["root"])
    session_t = subject_session_data(data, args.subject, session="T")
    prep_cfg = {"model": base_model_cfg}
    cropped = _crop_to_common_task_window(session_t, prep_cfg)
    evidence, evidence_space = _reference_space(cropped, base_model_cfg, "csd")
    if "CSD" not in evidence_space.upper():
        raise RuntimeError("Delay evidence was not transformed to CSD")
    requested_carriers = [value.strip().lower() for value in args.carrier_spaces.split(",") if value.strip()]
    invalid = sorted(set(requested_carriers).difference({"car", "original"}))
    if invalid:
        raise ValueError(f"Unknown carrier spaces: {invalid}")
    carrier_spaces = {
        name: _reference_space(cropped, base_model_cfg, name)[0]
        for name in requested_carriers
    }
    write_json(
        output / "protocol_contract.json",
        {
            **manifest,
            "carrier_spaces": requested_carriers,
            "delay_evidence_space": evidence_space,
            "transport_bands": len(protocol["transport_band_edges_hz"]),
            "classification_bands": len(protocol["classification_band_edges_hz"]),
            "all_classification_information_mandatory_delayed": True,
            "heldout_session_E_accessed": False,
        },
    )

    selected_carrier = args.five_stage_carrier
    gate_delta = float("nan")
    if args.mode in {"all", "paired_cv"}:
        _, _, selected_carrier, gate_delta = _paired_cv(
            carrier_spaces=carrier_spaces,
            evidence=evidence,
            base_cfg=cfg,
            base_model_cfg=base_model_cfg,
            protocol=protocol,
            output=output,
            args=args,
            source_fingerprint=fingerprint,
            environment_manifest=environment,
        )
    if args.mode == "all" and not gate_delta > 0.0:
        write_json(
            manifest_path,
            {
                **manifest,
                "status": "mechanism_gate_failed",
                "selected_carrier": selected_carrier,
                "mean_fixed_minus_zero": gate_delta,
                "five_stage_started": False,
            },
        )
        return
    if args.mode in {"all", "five_stage"}:
        if selected_carrier not in carrier_spaces:
            carrier_spaces[selected_carrier] = _reference_space(cropped, base_model_cfg, selected_carrier)[0]
        _five_stage(
            carrier=carrier_spaces[selected_carrier],
            carrier_name=selected_carrier,
            evidence=evidence,
            base_cfg=cfg,
            base_model_cfg=base_model_cfg,
            protocol=protocol,
            output=output,
            args=args,
            source_fingerprint=fingerprint,
            environment_manifest=environment,
        )
    write_json(
        manifest_path,
        {
            **manifest,
            "status": "completed",
            "selected_carrier": selected_carrier,
            "mean_fixed_minus_zero": gate_delta,
        },
    )


if __name__ == "__main__":
    main()
