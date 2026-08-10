#!/usr/bin/env python3
"""Run source verification and unified V6.2-R1 strong-baseline experiments."""

from __future__ import annotations

import argparse
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

from dpc_snn.baselines.neural import (  # noqa: E402
    SOURCE_LOCKS,
    V62_NEURAL_BASELINES,
    build_v62_neural_baseline,
    verify_official_source_locks,
)
from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v62_baselines import (  # noqa: E402
    BASELINE_OPTIMIZERS,
    apply_fixed_gain,
    fit_baseline,
    fit_fixed_gain,
    predict_baseline,
    prepare_model_input,
    task_carrier,
)
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
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402


SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".md", ".sh"}
RUN_REQUIRED_FILES = (
    "manifest.json",
    "resolved_config.yaml",
    "source_fingerprint.json",
    "split_manifest.json",
    "augmentation_manifest.json",
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


def _source_hashes() -> dict[str, Any]:
    project: dict[str, str] = {}
    for root_name in ("src", "scripts", "configs"):
        for path in sorted((ROOT / root_name).rglob("*")):
            if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES:
                project[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    for name in ("pyproject.toml", "requirements.txt", "README.md", "PLAN.md", "CHECKLIST.md"):
        path = ROOT / name
        if path.is_file():
            project[name] = file_sha256(path)
    return {
        "project": project,
        "official_source_locks": {
            name: {"commit": commit, "archive_sha256": archive}
            for name, (commit, archive) in SOURCE_LOCKS.items()
        },
    }


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "torch", "scipy", "einops", "mne", "moabb"):
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


def _metadata_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    labels = np.asarray(data["y"])
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
        for index in range(len(labels))
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


def _session(data: dict[str, Any], session: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rows = validate_trial_metadata(_metadata_rows(data))
    indices = np.asarray([index for index, row in enumerate(rows) if row["session"] == session])
    if indices.size != 288:
        raise RuntimeError(f"expected 288 trials in Session {session}, got {indices.size}")
    return (
        np.asarray(data["X"])[indices],
        np.asarray(data["y"])[indices],
        [rows[int(index)] for index in indices],
    )


def _probabilities(logits: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.from_numpy(logits).float(), dim=1).numpy()


def _source_smoke(output: Path, source_root: Path, device: str) -> dict[str, Any]:
    report: dict[str, Any] = {"status": "passed", "models": {}}
    for name in V62_NEURAL_BASELINES:
        torch.manual_seed(0)
        samples = 256 if name == "eegnet" else 1000
        model = build_v62_neural_baseline(name, source_root=source_root, samples=samples).to(device)
        if name == "fbcnet":
            x = torch.randn(2, 1, 22, 1000, 9, device=device)
        else:
            x = torch.randn(2, 22, samples, device=device)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = model(x)
        loss = result["logits"].square().mean()
        loss.backward()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        report["models"][name] = {
            "implementation_id": getattr(model, "implementation_id", ""),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "output_shape": list(result["logits"].shape),
            "finite_output": bool(torch.isfinite(result["logits"]).all()),
            "finite_gradients": all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_memory_mib": (
                torch.cuda.max_memory_allocated() / (1024**2) if device.startswith("cuda") else 0.0
            ),
        }
        if not report["models"][name]["finite_output"] or not report["models"][name][
            "finite_gradients"
        ]:
            raise FloatingPointError(f"official-source smoke failed for {name}")
        del model, x, result, loss
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    write_json(output / "source_verification.json", report)
    return report


def _selection_fingerprint(
    *,
    resolved: dict[str, Any],
    source: dict[str, Any],
    data_path: Path,
    split_manifest: dict[str, Any],
    augmentation: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    return build_run_fingerprint(
        resolved_config=resolved,
        source=source,
        data={data_path.name: file_sha256(data_path)},
        split=split_manifest,
        augmentation=augmentation,
        prior={"policy": "none"},
        checkpoint={
            "policy": "six_fold_session_t_median_best_epoch_then_full_t_retrain",
            "session_e_checkpoint_selection": False,
        },
        environment=environment,
    )


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/experiments/v62_e1_baselines.yaml")
    parser.add_argument("--models", default="")
    parser.add_argument("--subjects", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--cv-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-smoke-only", action="store_true")
    args = parser.parse_args()

    configure_cache_env()
    data_root = Path(args.data).resolve()
    source_root = Path(args.source_root).resolve()
    output = ensure_dir(args.output)
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    models = _parse_csv(args.models) if args.models else list(config["models"])
    subjects = _parse_csv(args.subjects, int) if args.subjects else list(config["subjects"])
    seeds = _parse_csv(args.seeds, int) if args.seeds else list(config["seeds"])
    unknown = sorted(set(models) - set(V62_NEURAL_BASELINES))
    if unknown:
        raise ValueError(f"unknown baseline models: {unknown}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    source_locks = verify_official_source_locks(source_root)
    source = _source_hashes()
    environment = _environment()
    write_json(output / "official_source_locks.json", source_locks)
    smoke = _source_smoke(output, source_root, args.device)
    if args.source_smoke_only:
        print(json.dumps(smoke, indent=2))
        return

    cv_epochs = int(args.cv_epochs or config["selection"]["max_epochs"])
    patience = int(args.patience or config["selection"]["patience"])
    n_splits = int(config["selection"]["n_splits"])
    maximum_folds = min(n_splits, int(args.max_folds or n_splits))
    minimum_final_epochs = int(config["selection"]["minimum_final_epochs"])
    augmentation = dict(config["augmentation"])
    campaign_rows: list[dict[str, Any]] = []

    for name in models:
        for subject in subjects:
            subject_path = _subject_file(data_root, subject)
            data = load_processed_npz(subject_path)
            x_t_raw, y_t, metadata_t = _session(data, "T")
            _, y_e_metadata_only, metadata_e = _session(data, "E")
            assert_t_e_isolation(metadata_t, metadata_e)
            carrier_t = task_carrier(
                x_t_raw,
                sfreq=float(data["sfreq"]),
                epoch_tmin=float(data["epoch_tmin"]),
            )
            folds = session_t_run_grouped_folds(
                metadata_t,
                n_splits=n_splits,
                seed=0,
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
                        "fold": index,
                        "train_trial_ids": [metadata_t[int(i)]["trial_id"] for i in train],
                        "validation_trial_ids": [metadata_t[int(i)]["trial_id"] for i in validation],
                        "validation_runs": sorted({metadata_t[int(i)]["run"] for i in validation}),
                    }
                    for index, (train, validation) in enumerate(folds)
                ],
            }
            resolved = {
                **{key: value for key, value in config.items() if key != "seeds"},
                "active_model": name,
                "active_subject": subject,
                "selection": {
                    **config["selection"],
                    "max_epochs": cv_epochs,
                    "patience": patience,
                    "executed_folds": maximum_folds,
                },
                "optimizer": BASELINE_OPTIMIZERS[name],
            }
            fingerprint = _selection_fingerprint(
                resolved=resolved,
                source=source,
                data_path=subject_path,
                split_manifest=split_manifest,
                augmentation=augmentation,
                environment=environment,
            )
            selection_dir = ensure_dir(output / name / f"subject_{subject:02d}" / "selection")
            fingerprint_path = selection_dir / "source_fingerprint.json"
            if fingerprint_path.is_file():
                validate_resume_fingerprint(fingerprint_path, fingerprint)
            else:
                write_fingerprint_manifest(fingerprint_path, fingerprint)
                write_json(selection_dir / "split_manifest.json", split_manifest)
                write_json(selection_dir / "augmentation_manifest.json", augmentation)
                (selection_dir / "resolved_config.yaml").write_text(
                    yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
                )

            best_epochs: list[int] = []
            cv_rows: list[dict[str, Any]] = []
            oof_parts: list[tuple[np.ndarray, dict[str, Any]]] = []
            for fold_index, (train_indices, validation_indices) in enumerate(folds):
                fold_dir = ensure_dir(selection_dir / f"fold_{fold_index}")
                result_path = fold_dir / "result.json"
                prediction_path = fold_dir / "validation_predictions.npz"
                if result_path.is_file() and prediction_path.is_file() and (fold_dir / "best.pt").is_file():
                    fold_result = read_json(result_path)
                    with np.load(prediction_path, allow_pickle=False) as archive:
                        fold_prediction = {key: archive[key] for key in archive.files}
                else:
                    gain = fit_fixed_gain(carrier_t[train_indices], clip=float(config["preprocessing"]["clip_after_gain"]))
                    x_train = prepare_model_input(
                        name,
                        apply_fixed_gain(carrier_t[train_indices], gain),
                        sfreq=float(data["sfreq"]),
                    )
                    x_validation = prepare_model_input(
                        name,
                        apply_fixed_gain(carrier_t[validation_indices], gain),
                        sfreq=float(data["sfreq"]),
                    )
                    fit = fit_baseline(
                        name,
                        source_root=source_root,
                        x_train=x_train,
                        y_train=y_t[train_indices],
                        x_validation=x_validation,
                        y_validation=y_t[validation_indices],
                        device=args.device,
                        seed=int(config["selection_seed"]) + fold_index * 1009,
                        epochs=cv_epochs,
                        patience=patience,
                        augmentation=augmentation,
                        run_label=f"{name}:S{subject}:fold{fold_index}",
                    )
                    evaluation = predict_baseline(
                        fit.model,
                        x_validation,
                        y_t[validation_indices],
                        device=args.device,
                        batch_size=int(BASELINE_OPTIMIZERS[name]["batch_size"]),
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
                        "gain": gain.values.reshape(-1).tolist(),
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
                best_epochs.append(int(fold_result["best_epoch"]))
                cv_rows.append(fold_result)
                oof_parts.append((fold_prediction["indices"], fold_prediction))
                print(
                    "__V62_E1_FOLD_DONE__ "
                    f"model={name} subject={subject} fold={fold_index} "
                    f"best_epoch={fold_result['best_epoch']} kappa={fold_result['kappa']:.6f}",
                    flush=True,
                )

            selected_epoch = min(
                cv_epochs,
                max(minimum_final_epochs, int(math.ceil(float(np.median(best_epochs))))),
            )
            selection_summary = {
                "model": name,
                "subject": subject,
                "best_epochs": best_epochs,
                "selected_final_epoch": selected_epoch,
                "cv_accuracy_mean": float(np.mean([row["accuracy"] for row in cv_rows])),
                "cv_kappa_mean": float(np.mean([row["kappa"] for row in cv_rows])),
                "cv_macro_f1_mean": float(np.mean([row["macro_f1"] for row in cv_rows])),
            }
            write_json(selection_dir / "selection_summary.json", selection_summary)
            write_csv(selection_dir / "fold_metrics.csv", cv_rows)
            oof_indices = np.concatenate([part[0] for part in oof_parts])
            oof_logits = np.concatenate([part[1]["logits"] for part in oof_parts])
            order = np.argsort(oof_indices)
            oof_indices = oof_indices[order]
            oof_logits = oof_logits[order]
            oof_labels = y_t[oof_indices]
            oof_probabilities = _probabilities(oof_logits)
            write_trial_predictions(
                selection_dir,
                logits=oof_logits,
                probabilities=oof_probabilities,
                pred=oof_probabilities.argmax(axis=1),
                label=oof_labels,
                subject=[metadata_t[int(index)]["subject"] for index in oof_indices],
                session="T",
                run=[metadata_t[int(index)]["run"] for index in oof_indices],
                trial_id=[metadata_t[int(index)]["trial_id"] for index in oof_indices],
                seed=int(config["selection_seed"]),
                model=f"{name}_selection_oof",
            )

            final_gain = fit_fixed_gain(carrier_t, clip=float(config["preprocessing"]["clip_after_gain"]))
            x_t = prepare_model_input(
                name, apply_fixed_gain(carrier_t, final_gain), sfreq=float(data["sfreq"])
            )
            for seed in seeds:
                run_dir = ensure_dir(output / name / f"subject_{subject:02d}" / f"seed_{seed}")
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
                    prior={"policy": "none"},
                    checkpoint={
                        "selection_fingerprint": fingerprint["combined_sha256"],
                        "selected_final_epoch": selected_epoch,
                        "selection_session": "T",
                        "evaluation_session_checkpoint_selection": False,
                    },
                    environment=environment,
                )
                if _completed_run(run_dir, run_fingerprint):
                    metrics = read_json(run_dir / "metrics.json")
                    campaign_rows.append(metrics)
                    print(f"__V62_E1_RESUME_SKIP__ model={name} subject={subject} seed={seed}", flush=True)
                    continue
                if (run_dir / "source_fingerprint.json").is_file():
                    validate_resume_fingerprint(run_dir / "source_fingerprint.json", run_fingerprint)
                write_json(
                    run_dir / "runtime_status.json",
                    {"status": "running", "started_at": time.time()},
                )
                write_fingerprint_manifest(run_dir / "source_fingerprint.json", run_fingerprint)
                write_json(run_dir / "split_manifest.json", split_manifest)
                write_json(run_dir / "augmentation_manifest.json", augmentation)
                (run_dir / "resolved_config.yaml").write_text(
                    yaml.safe_dump(run_resolved, sort_keys=False), encoding="utf-8"
                )
                fit = fit_baseline(
                    name,
                    source_root=source_root,
                    x_train=x_t,
                    y_train=y_t,
                    x_validation=None,
                    y_validation=None,
                    device=args.device,
                    seed=seed,
                    epochs=selected_epoch,
                    patience=selected_epoch,
                    augmentation=augmentation,
                    fixed_epoch=selected_epoch,
                    run_label=f"{name}:S{subject}:seed{seed}:final",
                )
                # Session E is materialized only after the Session-T model is fixed.
                x_e_raw, y_e, metadata_e_live = _session(data, "E")
                if not np.array_equal(y_e, y_e_metadata_only):
                    raise RuntimeError("Session E labels changed between metadata and evaluation access")
                carrier_e = task_carrier(
                    x_e_raw,
                    sfreq=float(data["sfreq"]),
                    epoch_tmin=float(data["epoch_tmin"]),
                )
                x_e = prepare_model_input(
                    name,
                    apply_fixed_gain(carrier_e, final_gain),
                    sfreq=float(data["sfreq"]),
                )
                evaluation = predict_baseline(
                    fit.model,
                    x_e,
                    y_e,
                    device=args.device,
                    batch_size=int(BASELINE_OPTIMIZERS[name]["batch_size"]),
                )
                params = sum(parameter.numel() for parameter in fit.model.parameters())
                metrics = {
                    "status": "completed",
                    "stage": "E1",
                    "model": name,
                    "implementation_id": getattr(fit.model, "implementation_id", ""),
                    "source_fidelity": config["source_fidelity"][name],
                    "subject": subject,
                    "seed": seed,
                    "accuracy": evaluation["accuracy"],
                    "balanced_accuracy": evaluation["balanced_accuracy"],
                    "kappa": evaluation["kappa"],
                    "macro_f1": evaluation["macro_f1"],
                    "cv_accuracy_mean": selection_summary["cv_accuracy_mean"],
                    "cv_kappa_mean": selection_summary["cv_kappa_mean"],
                    "selected_final_epoch": selected_epoch,
                    "parameters": params,
                    "optimizer_steps": fit.optimizer_steps,
                    "train_seconds": fit.elapsed_seconds,
                    "selection_session": "T",
                    "evaluation_session": "E",
                    "heldout_e_selected_checkpoint": False,
                    "run_fingerprint": run_fingerprint["combined_sha256"],
                }
                history = [
                    {"phase": "selection", **row}
                    for fold, rows in enumerate(cv_rows)
                    for row in [{"fold": fold, **rows}]
                ] + [{"phase": "final_train", "fold": "", **row} for row in fit.history]
                write_csv(run_dir / "history.csv", history)
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
                    model=name,
                )
                write_json(
                    run_dir / "runtime_status.json",
                    {"status": "completed", "completed_at": time.time()},
                )
                (run_dir / "stdout.log").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
                (run_dir / "stderr.log").write_text("", encoding="utf-8")
                write_run_artifact_manifest(run_dir, required_files=RUN_REQUIRED_FILES)
                campaign_rows.append(metrics)
                write_csv(output / "summary.csv", campaign_rows)
                print(
                    "__V62_E1_RUN_DONE__ "
                    f"model={name} subject={subject} seed={seed} "
                    f"accuracy={metrics['accuracy']:.6f} kappa={metrics['kappa']:.6f}",
                    flush=True,
                )
            del data, carrier_t, x_t
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    write_csv(output / "summary.csv", campaign_rows)
    write_json(
        output / "campaign_status.json",
        {
            "status": "completed",
            "models": models,
            "subjects": subjects,
            "seeds": seeds,
            "runs": len(campaign_rows),
        },
    )
    print(json.dumps({"status": "completed", "runs": len(campaign_rows)}, indent=2))


if __name__ == "__main__":
    main()
