#!/usr/bin/env python3
"""Run the V6.2-R1 zero-delay ANN scaffold and its pre-registered gate."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.baselines.neural import verify_official_source_locks  # noqa: E402
from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    assert_t_e_isolation,
    build_run_fingerprint,
    file_sha256,
    session_t_run_grouped_folds,
    validate_resume_fingerprint,
    validate_run_artifact_manifest,
    validate_trial_metadata,
    write_fingerprint_manifest,
    write_run_artifact_manifest,
    write_trial_predictions,
)
from dpc_snn.experiments.v62_scaffold import (  # noqa: E402
    build_zero_scaffold,
    cache_official_fbc_rate_features,
    cache_analytic_fixed_channel_gain_rate_features,
    cache_rate_features,
    fit_official_fbc_channel_gain,
    fit_official_atc_scaffold_parity,
    fit_projected_training_gain,
    fit_zero_scaffold,
    predict_scaffold,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402


SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".md", ".sh"}
RUN_REQUIRED_FILES = (
    "manifest.json",
    "resolved_config.yaml",
    "source_fingerprint.json",
    "split_manifest.json",
    "augmentation_manifest.json",
    "frontend_manifest.json",
    "history.csv",
    "predictions.npz",
    "predictions.csv",
    "metrics.json",
    "best.pt",
    "last.pt",
    "runtime_status.json",
    "stdout.log",
    "stderr.log",
)


def _parse_csv(value: str, cast: Any = str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for root_name in ("src", "scripts", "configs"):
        for path in sorted((ROOT / root_name).rglob("*")):
            if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES:
                result[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    for name in (
        "pyproject.toml",
        "requirements.txt",
        "README.md",
        "PLAN.md",
        "CHECKLIST.md",
    ):
        path = ROOT / name
        if path.is_file():
            result[name] = file_sha256(path)
    return result


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "torch", "scipy", "mne", "moabb"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "storage_root": os.environ.get("DPC_SNN_STORAGE_ROOT"),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def _metadata_rows(
    data: dict[str, Any], indices: np.ndarray | None = None
) -> list[dict[str, Any]]:
    labels = np.asarray(data["y"])
    selected = range(len(labels)) if indices is None else (int(index) for index in indices)
    return [
        {
            "dataset": str(data.get("dataset_name", "bci2a")),
            "subject": np.asarray(data["subject"])[index],
            "session": np.asarray(data["session"])[index],
            "run": np.asarray(data["run"])[index],
            "trial_id": np.asarray(data["trial_id"])[index],
            "class": int(labels[index]),
            "sfreq": float(data["sfreq"]),
            "ch_names": data["ch_names"],
            "epoch_tmin": float(data["epoch_tmin"]),
            "epoch_tmax": float(data["epoch_tmax"]),
        }
        for index in selected
    ]


def _subject_file(data_root: Path, subject: int) -> Path:
    candidates = [data_root / f"A{subject:02d}.npz", data_root / f"A{subject:02d}_all.npz"]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(data_root.glob(f"A{subject:02d}*.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"cannot uniquely resolve subject {subject} under {data_root}")
    return matches[0]


def _session_indices(
    data: dict[str, Any], session: str
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    indices = np.asarray(
        [
            index
            for index, value in enumerate(np.asarray(data["session"]))
            if str(value) == session
        ],
        dtype=np.int64,
    )
    if indices.size != 288:
        raise RuntimeError(f"expected 288 trials in Session {session}, got {indices.size}")
    rows = validate_trial_metadata(_metadata_rows(data, indices))
    return indices, np.asarray(data["y"])[indices], rows


def _probabilities(logits: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.from_numpy(logits).float(), dim=1).numpy()


def _completed_run(run_dir: Path, fingerprint: dict[str, Any]) -> bool:
    if not (run_dir / "manifest.json").is_file():
        return False
    validate_resume_fingerprint(run_dir / "source_fingerprint.json", fingerprint)
    validate_run_artifact_manifest(
        run_dir,
        required_files=RUN_REQUIRED_FILES,
        verify_hashes=True,
        verify_prediction_schema=True,
    )
    return True


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _gate_report(
    *,
    baseline_summary: Path,
    scaffold_rows: list[dict[str, Any]],
    subjects: list[int],
    seeds: list[int],
    maximum_gap_pp: float,
) -> dict[str, Any]:
    if not baseline_summary.is_file():
        return {
            "status": "pending",
            "reason": f"baseline summary is not available: {baseline_summary}",
        }
    required_pairs = {(subject, seed) for subject in subjects for seed in seeds}
    grouped: dict[str, list[float]] = {}
    grouped_pairs: dict[str, set[tuple[int, int]]] = {}
    for row in _load_csv(baseline_summary):
        if row.get("status") != "completed":
            continue
        pair = (int(row["subject"]), int(row["seed"]))
        if pair not in required_pairs:
            continue
        grouped.setdefault(row["model"], []).append(float(row["accuracy"]))
        grouped_pairs.setdefault(row["model"], set()).add(pair)
    complete = {
        model: values
        for model, values in grouped.items()
        if grouped_pairs.get(model) == required_pairs and len(values) == len(required_pairs)
    }
    scaffold_pairs = {(int(row["subject"]), int(row["seed"])) for row in scaffold_rows}
    if not complete or scaffold_pairs != required_pairs:
        return {
            "status": "pending",
            "reason": "E1 or E2 does not yet contain every required subject-seed run",
            "complete_baseline_models": sorted(complete),
            "scaffold_pairs": sorted(scaffold_pairs),
        }
    baseline_means = {model: float(np.mean(values)) for model, values in complete.items()}
    strongest_model = max(baseline_means, key=baseline_means.get)
    strongest_accuracy = baseline_means[strongest_model]
    scaffold_accuracy = float(np.mean([float(row["accuracy"]) for row in scaffold_rows]))
    gap_pp = 100.0 * (strongest_accuracy - scaffold_accuracy)
    passed = gap_pp <= float(maximum_gap_pp)
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "strongest_baseline_model": strongest_model,
        "strongest_baseline_accuracy": strongest_accuracy,
        "baseline_model_means": baseline_means,
        "zero_scaffold_accuracy": scaffold_accuracy,
        "gap_pp": gap_pp,
        "maximum_gap_pp": float(maximum_gap_pp),
        "aggregation": "mean_over_matching_subject_seed_runs",
        "next_stage_allowed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--source-root", default="")
    parser.add_argument(
        "--config", default="configs/experiments/v62_e2_zero_scaffold.yaml"
    )
    parser.add_argument("--subjects", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--cv-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="run Session-T cross-validation without reading or evaluating Session-E labels",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    configure_cache_env()
    data_root = Path(args.data).resolve()
    output = ensure_dir(args.output)
    baseline_summary = Path(args.baseline_summary).resolve()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    subjects = _parse_csv(args.subjects, int) if args.subjects else list(config["subjects"])
    seeds = _parse_csv(args.seeds, int) if args.seeds else list(config["seeds"])
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    source = _source_hashes()
    environment = _environment()
    cv_epochs = int(args.cv_epochs or config["selection"]["max_epochs"])
    patience = int(args.patience or config["selection"]["patience"])
    n_splits = int(config["selection"]["n_splits"])
    maximum_folds = min(n_splits, int(args.max_folds or n_splits))
    minimum_epochs = int(config["selection"]["minimum_epochs"])
    minimum_final_epochs = int(config["selection"]["minimum_final_epochs"])
    training = dict(config["training"])
    augmentation = {
        "enabled": bool(config["augmentation"]["enabled"]),
        "parent_scope": config["augmentation"]["parent_scope"],
        "segments": int(config["augmentation"]["paired_fast_slow_segments"]),
        "probability": float(config["augmentation"]["probability"]),
        "carrier_scale_range": config["augmentation"]["carrier_scale_range"],
        "noise_std": float(config["augmentation"].get("noise_std", 0.0)),
    }
    model_config = dict(config["model"])
    official_source_root = args.source_root or None
    if str(model_config.get("atc_readout_variant", "causal")).lower() == (
        "official_full_window"
    ):
        if official_source_root is None:
            raise ValueError("official_full_window ATC requires --source-root")
        source["official_source_locks"] = verify_official_source_locks(
            official_source_root
        )
    frontend_kind = str(config["preprocessing"].get("frontend", "analytic_fir"))
    campaign_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for subject in subjects:
        subject_path = _subject_file(data_root, subject)
        data = load_processed_npz(subject_path)
        t_indices, y_t, metadata_t = _session_indices(data, "T")
        if args.selection_only:
            e_indices = None
            y_e_metadata_only = None
            metadata_e = None
            session_values = np.asarray(data["session"])
            if int(np.count_nonzero(session_values.astype(str) == "E")) != 288:
                raise RuntimeError("selection-only seal expected 288 Session-E trials")
        else:
            e_indices, y_e_metadata_only, metadata_e = _session_indices(data, "E")
            assert_t_e_isolation(metadata_t, metadata_e)
        x_t = np.asarray(data["X"])[t_indices]
        folds = session_t_run_grouped_folds(
            metadata_t,
            n_splits=n_splits,
            seed=int(config["selection_seed"]),
            shuffle=True,
        )[:maximum_folds]
        split_manifest = {
            "subject": subject,
            "selection_session": "T",
            "evaluation_session": "E",
            "method": "run_grouped",
            "n_splits": n_splits,
            "executed_folds": maximum_folds,
            "folds": [
                {
                    "fold": fold,
                    "train_trial_ids": [metadata_t[int(i)]["trial_id"] for i in train],
                    "validation_trial_ids": [
                        metadata_t[int(i)]["trial_id"] for i in validation
                    ],
                    "validation_runs": sorted(
                        {metadata_t[int(i)]["run"] for i in validation}
                    ),
                }
                for fold, (train, validation) in enumerate(folds)
            ],
        }
        resolved = {
            **{key: value for key, value in config.items() if key != "seeds"},
            "active_subject": subject,
            "execution_mode": "selection_only" if args.selection_only else "t_to_e",
            "selection": {
                **config["selection"],
                "max_epochs": cv_epochs,
                "patience": patience,
                "executed_folds": maximum_folds,
            },
        }
        selection_fingerprint = build_run_fingerprint(
            resolved_config=resolved,
            source=source,
            data={subject_path.name: file_sha256(subject_path)},
            split=split_manifest,
            augmentation=augmentation,
            prior={"policy": "zero_delay_no_evidence_prior"},
            checkpoint={
                "policy": (
                    "six_fold_t_oof_only"
                    if args.selection_only
                    else "six_fold_t_median_best_epoch_then_full_t_retrain"
                ),
                "session_e_checkpoint_selection": False,
                "session_e_accessed": False,
            },
            environment=environment,
        )
        selection_dir = ensure_dir(output / f"subject_{subject:02d}" / "selection")
        fingerprint_path = selection_dir / "source_fingerprint.json"
        if fingerprint_path.is_file():
            validate_resume_fingerprint(fingerprint_path, selection_fingerprint)
        else:
            write_fingerprint_manifest(fingerprint_path, selection_fingerprint)
            write_json(selection_dir / "split_manifest.json", split_manifest)
            write_json(selection_dir / "augmentation_manifest.json", augmentation)
            (selection_dir / "resolved_config.yaml").write_text(
                yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
            )

        best_epochs: list[int] = []
        fold_rows: list[dict[str, Any]] = []
        oof_parts: list[tuple[np.ndarray, dict[str, Any]]] = []
        for fold_index, (train_indices, validation_indices) in enumerate(folds):
            fold_dir = ensure_dir(selection_dir / f"fold_{fold_index}")
            result_path = fold_dir / "result.json"
            prediction_path = fold_dir / "validation_predictions.npz"
            if result_path.is_file() and prediction_path.is_file() and (
                fold_dir / "best.pt"
            ).is_file():
                fold_result = read_json(result_path)
                with np.load(prediction_path, allow_pickle=False) as archive:
                    fold_prediction = {key: archive[key] for key in archive.files}
            else:
                fold_seed = int(config["selection_seed"]) + fold_index * 1009
                model = build_zero_scaffold(
                    seed=fold_seed,
                    model_config=model_config,
                    official_source_root=official_source_root,
                )
                if frontend_kind in {
                    "official_fbc_chebyshev_ii",
                    "analytic_fir_fixed_channel_gain",
                }:
                    gain = fit_official_fbc_channel_gain(
                        x_t[train_indices],
                        sfreq=float(data["sfreq"]),
                        epoch_tmin=float(data["epoch_tmin"]),
                        clip=float(config["preprocessing"]["clip_after_gain"]),
                    )
                    cache_function = (
                        cache_official_fbc_rate_features
                        if frontend_kind == "official_fbc_chebyshev_ii"
                        else cache_analytic_fixed_channel_gain_rate_features
                    )
                    train_rates = cache_function(
                        model, x_t[train_indices], gain, device=args.device,
                        batch_size=max(8, int(training["batch_size"])),
                    )
                    validation_rates = cache_function(
                        model, x_t[validation_indices], gain, device=args.device,
                        batch_size=max(8, int(training["batch_size"])),
                    )
                    gain_values = gain.values
                else:
                    gain = fit_projected_training_gain(
                        model,
                        x_t[train_indices],
                        device=args.device,
                        batch_size=max(8, int(training["batch_size"])),
                    )
                    train_rates = cache_rate_features(
                        model,
                        x_t[train_indices],
                        device=args.device,
                        batch_size=max(8, int(training["batch_size"])),
                    )
                    validation_rates = cache_rate_features(
                        model,
                        x_t[validation_indices],
                        device=args.device,
                        batch_size=max(8, int(training["batch_size"])),
                    )
                    gain_values = gain.numpy()
                if bool(training.get("official_core_parity_mode", False)):
                    fit = fit_official_atc_scaffold_parity(
                        model,
                        source_root=str(official_source_root),
                        x_train_raw=x_t[train_indices],
                        y_train=y_t[train_indices],
                        channel_gain=gain,
                        x_validation_raw=x_t[validation_indices],
                        y_validation=y_t[validation_indices],
                        sfreq=float(data["sfreq"]),
                        epoch_tmin=float(data["epoch_tmin"]),
                        device=args.device,
                        seed=fold_seed,
                        epochs=cv_epochs,
                        patience=patience,
                        augmentation=augmentation,
                        run_label=f"zero_ann:S{subject}:fold{fold_index}",
                    )
                else:
                    fit = fit_zero_scaffold(
                        model,
                        train_rates=train_rates,
                        y_train=y_t[train_indices],
                        validation_rates=validation_rates,
                        y_validation=y_t[validation_indices],
                        device=args.device,
                        seed=fold_seed,
                        epochs=cv_epochs,
                        patience=patience,
                        minimum_epochs=minimum_epochs,
                        batch_size=int(training["batch_size"]),
                        accumulation_steps=int(training["gradient_accumulation_steps"]),
                        learning_rate=float(training["learning_rate"]),
                        weight_decay=float(training["weight_decay"]),
                        auxiliary_weights=training["auxiliary_endpoint_weights"],
                        beta1=float(training.get("beta1", 0.9)),
                        scheduler_warmup_epochs=int(
                            training.get("scheduler_warmup_epochs", 10)
                        ),
                        use_scheduler=bool(training.get("schedule", True)),
                        scheduler_step_unit=str(
                            training.get("scheduler_step_unit", "update")
                        ),
                        reseed_before_training=bool(
                            training.get("reseed_before_training", True)
                        ),
                        augmentation=augmentation,
                        run_label=f"zero_ann:S{subject}:fold{fold_index}",
                    )
                evaluation = predict_scaffold(
                    fit.model,
                    validation_rates,
                    y_t[validation_indices],
                    device=args.device,
                    batch_size=max(16, int(training["batch_size"])),
                )
                fold_result = {
                    "fold": fold_index,
                    "best_epoch": fit.best_epoch,
                    "best_selection_metric": fit.best_metric,
                    "accuracy": evaluation["accuracy"],
                    "balanced_accuracy": evaluation["balanced_accuracy"],
                    "kappa": evaluation["kappa"],
                    "macro_f1": evaluation["macro_f1"],
                    "optimizer_steps": fit.optimizer_steps,
                    "elapsed_seconds": fit.elapsed_seconds,
                    "gain": np.asarray(gain_values).reshape(-1).tolist(),
                    "frontend_fingerprint": train_rates.frontend_fingerprint,
                }
                fold_prediction = {
                    "indices": validation_indices.astype(np.int64),
                    "logits": evaluation["logits"].astype(np.float32),
                    "labels": evaluation["labels"].astype(np.int64),
                }
                torch.save(fit.best_state, fold_dir / "best.pt")
                torch.save(fit.last_state, fold_dir / "last.pt")
                write_csv(fold_dir / "history.csv", fit.history)
                write_json(result_path, fold_result)
                np.savez_compressed(prediction_path, **fold_prediction)
                del model, train_rates, validation_rates, fit
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            best_epochs.append(int(fold_result["best_epoch"]))
            fold_rows.append(fold_result)
            oof_parts.append((fold_prediction["indices"], fold_prediction))
            print(
                "__V62_E2_FOLD_DONE__ "
                f"subject={subject} fold={fold_index} "
                f"best_epoch={fold_result['best_epoch']} kappa={fold_result['kappa']:.6f}",
                flush=True,
            )

        selected_epoch = min(
            cv_epochs,
            max(minimum_final_epochs, int(math.ceil(float(np.median(best_epochs))))),
        )
        selection_summary = {
            "model": "dasp_snn_v62_zero_ann",
            "subject": subject,
            "best_epochs": best_epochs,
            "selected_final_epoch": selected_epoch,
            "cv_accuracy_mean": float(np.mean([row["accuracy"] for row in fold_rows])),
            "cv_kappa_mean": float(np.mean([row["kappa"] for row in fold_rows])),
            "cv_macro_f1_mean": float(np.mean([row["macro_f1"] for row in fold_rows])),
            "session_e_accessed": False,
        }
        selection_rows.append(selection_summary)
        write_json(selection_dir / "selection_summary.json", selection_summary)
        write_csv(selection_dir / "fold_metrics.csv", fold_rows)
        oof_indices = np.concatenate([part[0] for part in oof_parts])
        oof_logits = np.concatenate([part[1]["logits"] for part in oof_parts])
        order = np.argsort(oof_indices)
        oof_indices = oof_indices[order]
        oof_logits = oof_logits[order]
        oof_probabilities = _probabilities(oof_logits)
        write_trial_predictions(
            selection_dir,
            logits=oof_logits,
            probabilities=oof_probabilities,
            pred=oof_probabilities.argmax(axis=1),
            label=y_t[oof_indices],
            subject=[metadata_t[int(index)]["subject"] for index in oof_indices],
            session="T",
            run=[metadata_t[int(index)]["run"] for index in oof_indices],
            trial_id=[metadata_t[int(index)]["trial_id"] for index in oof_indices],
            seed=int(config["selection_seed"]),
            model="dasp_snn_v62_zero_ann_selection_oof",
        )

        if args.selection_only:
            del data, x_t
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            continue

        frontend_template = build_zero_scaffold(
            seed=0,
            model_config=model_config,
            official_source_root=official_source_root,
        )
        if frontend_kind in {
            "official_fbc_chebyshev_ii",
            "analytic_fir_fixed_channel_gain",
        }:
            final_gain = fit_official_fbc_channel_gain(
                x_t,
                sfreq=float(data["sfreq"]),
                epoch_tmin=float(data["epoch_tmin"]),
                clip=float(config["preprocessing"]["clip_after_gain"]),
            )
            cache_function = (
                cache_official_fbc_rate_features
                if frontend_kind == "official_fbc_chebyshev_ii"
                else cache_analytic_fixed_channel_gain_rate_features
            )
            final_t_rates = cache_function(
                frontend_template, x_t, final_gain, device=args.device,
                batch_size=max(8, int(training["batch_size"])),
            )
            final_gain_values = final_gain.values
        else:
            final_gain = fit_projected_training_gain(
                frontend_template,
                x_t,
                device=args.device,
                batch_size=max(8, int(training["batch_size"])),
            )
            final_t_rates = cache_rate_features(
                frontend_template,
                x_t,
                device=args.device,
                batch_size=max(8, int(training["batch_size"])),
            )
            final_gain_values = final_gain.numpy()
        final_e_rates = None
        if e_indices is None or y_e_metadata_only is None or metadata_e is None:
            raise RuntimeError("Session-E remained sealed outside selection-only mode")
        metadata_e_live = metadata_e
        for seed in seeds:
            model = build_zero_scaffold(
                seed=seed,
                model_config=model_config,
                official_source_root=official_source_root,
            )
            if model.frontend_fingerprint() != final_t_rates.frontend_fingerprint:
                raise RuntimeError("seed changed the pre-registered frozen physical front end")
            model.set_training_gain(final_t_rates.gain)
            run_dir = ensure_dir(output / f"subject_{subject:02d}" / f"seed_{seed}")
            run_resolved = {
                **resolved,
                "active_seed": seed,
                "selected_final_epoch": selected_epoch,
            }
            run_fingerprint = build_run_fingerprint(
                resolved_config=run_resolved,
                source=source,
                data={subject_path.name: file_sha256(subject_path)},
                split=split_manifest,
                augmentation=augmentation,
                prior={"policy": "zero_delay_no_evidence_prior"},
                checkpoint={
                    "selection_fingerprint": selection_fingerprint["combined_sha256"],
                    "selected_final_epoch": selected_epoch,
                    "selection_session": "T",
                    "evaluation_session_checkpoint_selection": False,
                },
                environment=environment,
            )
            if _completed_run(run_dir, run_fingerprint):
                metrics = read_json(run_dir / "metrics.json")
                campaign_rows.append(metrics)
                print(f"__V62_E2_RESUME_SKIP__ subject={subject} seed={seed}", flush=True)
                continue
            if (run_dir / "source_fingerprint.json").is_file():
                validate_resume_fingerprint(
                    run_dir / "source_fingerprint.json", run_fingerprint
                )
            write_json(
                run_dir / "runtime_status.json",
                {"status": "running", "started_at": time.time()},
            )
            write_fingerprint_manifest(run_dir / "source_fingerprint.json", run_fingerprint)
            write_json(run_dir / "split_manifest.json", split_manifest)
            write_json(run_dir / "augmentation_manifest.json", augmentation)
            write_json(
                run_dir / "frontend_manifest.json",
                {
                    "frontend_fingerprint": final_t_rates.frontend_fingerprint,
                    "gain": np.asarray(final_gain_values).reshape(-1).tolist(),
                    "frontend_kind": frontend_kind,
                    "gain_fit_session": "T",
                    "cache_boundary": "pre_delay_dual_rate_features",
                },
            )
            (run_dir / "resolved_config.yaml").write_text(
                yaml.safe_dump(run_resolved, sort_keys=False), encoding="utf-8"
            )
            if bool(training.get("official_core_parity_mode", False)):
                fit = fit_official_atc_scaffold_parity(
                    model,
                    source_root=str(official_source_root),
                    x_train_raw=x_t,
                    y_train=y_t,
                    channel_gain=final_gain,
                    x_validation_raw=None,
                    y_validation=None,
                    sfreq=float(data["sfreq"]),
                    epoch_tmin=float(data["epoch_tmin"]),
                    device=args.device,
                    seed=seed,
                    epochs=selected_epoch,
                    patience=selected_epoch,
                    augmentation=augmentation,
                    fixed_epoch=selected_epoch,
                    run_label=f"zero_ann:S{subject}:seed{seed}:final",
                )
            else:
                fit = fit_zero_scaffold(
                    model,
                    train_rates=final_t_rates,
                    y_train=y_t,
                    validation_rates=None,
                    y_validation=None,
                    device=args.device,
                    seed=seed,
                    epochs=selected_epoch,
                    patience=selected_epoch,
                    minimum_epochs=selected_epoch,
                    batch_size=int(training["batch_size"]),
                    accumulation_steps=int(training["gradient_accumulation_steps"]),
                    learning_rate=float(training["learning_rate"]),
                    weight_decay=float(training["weight_decay"]),
                    auxiliary_weights=training["auxiliary_endpoint_weights"],
                    beta1=float(training.get("beta1", 0.9)),
                    scheduler_warmup_epochs=int(
                        training.get("scheduler_warmup_epochs", 10)
                    ),
                    use_scheduler=bool(training.get("schedule", True)),
                    scheduler_step_unit=str(
                        training.get("scheduler_step_unit", "update")
                    ),
                    reseed_before_training=bool(
                        training.get("reseed_before_training", True)
                    ),
                    augmentation=augmentation,
                    fixed_epoch=selected_epoch,
                    run_label=f"zero_ann:S{subject}:seed{seed}:final",
                )
            if final_e_rates is None:
                x_e = np.asarray(data["X"])[e_indices]
                if not np.array_equal(np.asarray(data["y"])[e_indices], y_e_metadata_only):
                    raise RuntimeError("Session E labels changed before final evaluation")
                if frontend_kind in {
                    "official_fbc_chebyshev_ii",
                    "analytic_fir_fixed_channel_gain",
                }:
                    cache_function = (
                        cache_official_fbc_rate_features
                        if frontend_kind == "official_fbc_chebyshev_ii"
                        else cache_analytic_fixed_channel_gain_rate_features
                    )
                    final_e_rates = cache_function(
                        fit.model, x_e, final_gain, device=args.device,
                        batch_size=max(8, int(training["batch_size"])),
                    )
                else:
                    final_e_rates = cache_rate_features(
                        fit.model,
                        x_e,
                        device=args.device,
                        batch_size=max(8, int(training["batch_size"])),
                    )
            evaluation = predict_scaffold(
                fit.model,
                final_e_rates,
                y_e_metadata_only,
                device=args.device,
                batch_size=max(16, int(training["batch_size"])),
            )
            metrics = {
                "status": "completed",
                "stage": "E2",
                "model": "dasp_snn_v62_zero_ann",
                "implementation_id": "dasp_snn_v6_2_r1_zero_delay_ann_scaffold",
                "subject": subject,
                "seed": seed,
                "accuracy": evaluation["accuracy"],
                "balanced_accuracy": evaluation["balanced_accuracy"],
                "kappa": evaluation["kappa"],
                "macro_f1": evaluation["macro_f1"],
                "cv_accuracy_mean": selection_summary["cv_accuracy_mean"],
                "cv_kappa_mean": selection_summary["cv_kappa_mean"],
                "selected_final_epoch": selected_epoch,
                "parameters": fit.model.parameter_count,
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in fit.model.parameters()
                    if parameter.requires_grad
                ),
                "optimizer_steps": fit.optimizer_steps,
                "train_seconds": fit.elapsed_seconds,
                "selection_session": "T",
                "evaluation_session": "E",
                "heldout_e_selected_checkpoint": False,
                "delay_override": "zero",
                "mandatory_delay_bottleneck": True,
                "frontend_fingerprint": final_t_rates.frontend_fingerprint,
                "run_fingerprint": run_fingerprint["combined_sha256"],
            }
            write_csv(run_dir / "history.csv", fit.history)
            write_json(run_dir / "metrics.json", metrics)
            torch.save(fit.last_state, run_dir / "best.pt")
            torch.save(fit.last_state, run_dir / "last.pt")
            probabilities = _probabilities(evaluation["logits"])
            write_trial_predictions(
                run_dir,
                logits=evaluation["logits"],
                probabilities=probabilities,
                pred=evaluation["pred"],
                label=evaluation["labels"],
                subject=[row["subject"] for row in metadata_e_live],
                session=[row["session"] for row in metadata_e_live],
                run=[row["run"] for row in metadata_e_live],
                trial_id=[row["trial_id"] for row in metadata_e_live],
                seed=seed,
                model="dasp_snn_v62_zero_ann",
            )
            write_json(
                run_dir / "runtime_status.json",
                {"status": "completed", "completed_at": time.time()},
            )
            (run_dir / "stdout.log").write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
            (run_dir / "stderr.log").write_text("", encoding="utf-8")
            write_run_artifact_manifest(run_dir, required_files=RUN_REQUIRED_FILES)
            campaign_rows.append(metrics)
            write_csv(output / "summary.csv", campaign_rows)
            print(
                "__V62_E2_RUN_DONE__ "
                f"subject={subject} seed={seed} accuracy={metrics['accuracy']:.6f} "
                f"kappa={metrics['kappa']:.6f}",
                flush=True,
            )
            del model, fit
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        del data, x_t, final_t_rates, frontend_template
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if args.selection_only:
        write_csv(output / "selection_summary.csv", selection_rows)
        status = {
            "status": "completed",
            "mode": "selection_only",
            "subjects": subjects,
            "selection_runs": len(selection_rows),
            "session_e_accessed": False,
        }
        write_json(output / "campaign_status.json", status)
        print(json.dumps(status, indent=2))
        return

    write_csv(output / "summary.csv", campaign_rows)
    gate = _gate_report(
        baseline_summary=baseline_summary,
        scaffold_rows=campaign_rows,
        subjects=subjects,
        seeds=seeds,
        maximum_gap_pp=float(config["gate"]["maximum_gap_pp"]),
    )
    write_json(output / "gate_report.json", gate)
    write_json(
        output / "campaign_status.json",
        {
            "status": "completed",
            "subjects": subjects,
            "seeds": seeds,
            "runs": len(campaign_rows),
            "gate": gate,
        },
    )
    print(json.dumps({"status": "completed", "runs": len(campaign_rows), "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
