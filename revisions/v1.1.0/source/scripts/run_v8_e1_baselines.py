#!/usr/bin/env python3
"""Run the locked Session-T-only V8 strong-baseline campaign."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
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
    file_sha256,
    session_t_run_grouped_folds,
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
    write_trial_predictions,
)
from dpc_snn.experiments.v8_baselines import (  # noqa: E402
    merge_oof_predictions,
    nested_run_grouped_indices,
    rank_screening_models,
    session_t_development_view,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    build_v8_run_fingerprint,
    collect_source_tree_manifest,
    source_tree_digest,
    validate_v8_resume_fingerprint,
    v8_heldout_lock_manifest,
    write_v8_fingerprint,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402


EXTERNAL_SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".toml", ".txt"}
RUN_FILES = (
    "manifest.json",
    "resolved_config.yaml",
    "source_fingerprint.json",
    "source_tree_manifest.json",
    "data_access_manifest.json",
    "split_manifest.json",
    "augmentation_manifest.json",
    "metrics.json",
    "predictions.npz",
    "predictions.csv",
    "runtime_status.json",
    "stdout.log",
    "stderr.log",
)
FOLD_FILES = (
    "best.pt",
    "last.pt",
    "history.csv",
    "result.json",
    "outer_test_predictions.npz",
    "selection_best.pt",
    "selection_last.pt",
    "selection_history.csv",
    "selection_result.json",
    "selection_predictions.npz",
)


def _csv(value: str, cast: Any = str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _subject_file(data_root: Path, subject: int) -> Path:
    candidates = (data_root / f"A{subject:02d}.npz", data_root / f"A{subject:02d}_all.npz")
    present = [path for path in candidates if path.is_file()]
    if len(present) == 1:
        return present[0]
    matches = sorted(data_root.glob(f"A{subject:02d}*.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"cannot uniquely resolve subject {subject} under {data_root}")
    return matches[0]


def _environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
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
        "cuda_version": torch.version.cuda,
        "storage_root": os.environ.get("DPC_SNN_STORAGE_ROOT"),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def _external_source_manifest(source_root: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for source_name in SOURCE_LOCKS:
        base = source_root / source_name
        if not base.is_dir():
            raise FileNotFoundError(f"missing pinned source tree: {base}")
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTERNAL_SOURCE_SUFFIXES:
                continue
            relative_parts = path.relative_to(base).parts
            if any(part in {".git", "__pycache__"} for part in relative_parts):
                continue
            key = f"external/{source_name}/{path.relative_to(base).as_posix()}"
            manifest[key] = file_sha256(path)
    if not manifest:
        raise RuntimeError("external baseline source manifest is empty")
    return manifest


def _source_smoke(output: Path, source_root: Path, device: str) -> dict[str, Any]:
    report: dict[str, Any] = {"status": "passed", "models": {}}
    for name in V62_NEURAL_BASELINES:
        torch.manual_seed(0)
        samples = 256 if name == "eegnet" else 1000
        model = build_v62_neural_baseline(name, source_root=source_root, samples=samples).to(device)
        x = (
            torch.randn(2, 1, 22, 1000, 9, device=device)
            if name == "fbcnet"
            else torch.randn(2, 22, samples, device=device)
        )
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = model(x)
        logits = result["logits"]
        logits.square().mean().backward()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        finite_gradients = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        row = {
            "implementation_id": getattr(model, "implementation_id", ""),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "output_shape": list(logits.shape),
            "finite_output": bool(torch.isfinite(logits).all()),
            "finite_gradients": finite_gradients,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_memory_mib": (
                torch.cuda.max_memory_allocated() / (1024**2)
                if device.startswith("cuda")
                else 0.0
            ),
        }
        if not row["finite_output"] or not row["finite_gradients"]:
            raise FloatingPointError(f"baseline smoke failed for {name}")
        report["models"][name] = row
        del model, x, result, logits
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    write_json(output / "source_verification.json", report)
    return report


def _required_files(n_folds: int) -> tuple[str, ...]:
    return RUN_FILES + tuple(
        f"fold_{fold}/{name}" for fold in range(n_folds) for name in FOLD_FILES
    )


def _probabilities(logits: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.from_numpy(logits).float(), dim=1).numpy()


def _load_fold(
    fold_dir: Path,
    *,
    expected_indices: np.ndarray,
    expected_labels: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    paths = [fold_dir / name for name in FOLD_FILES]
    if not all(path.is_file() for path in paths):
        return None
    result = read_json(fold_dir / "result.json")
    with np.load(fold_dir / "outer_test_predictions.npz", allow_pickle=False) as archive:
        prediction = {key: archive[key] for key in archive.files}
    if set(prediction) != {"indices", "logits", "labels"}:
        return None
    if not np.array_equal(np.asarray(prediction["indices"]), expected_indices):
        return None
    if not np.array_equal(np.asarray(prediction["labels"]), expected_labels):
        return None
    logits = np.asarray(prediction["logits"])
    if logits.shape != (expected_indices.size, 4) or not np.isfinite(logits).all():
        return None
    return result, prediction


def _run_one(
    *,
    output: Path,
    config: dict[str, Any],
    source_root: Path,
    source_tree: dict[str, str],
    environment: dict[str, Any],
    official_locks: dict[str, Any],
    data_root: Path,
    model_name: str,
    subject: int,
    seed: int,
    cv_epochs: int,
    patience: int,
    maximum_folds: int,
    device: str,
) -> dict[str, Any]:
    subject_path = _subject_file(data_root, subject)
    data = load_processed_npz(subject_path)
    x_t_raw, y_t, metadata_t, access_manifest = session_t_development_view(data)
    carrier_t = task_carrier(
        x_t_raw,
        sfreq=float(data["sfreq"]),
        epoch_tmin=float(data["epoch_tmin"]),
    )
    n_splits = int(config["selection"]["n_splits"])
    folds = session_t_run_grouped_folds(
        metadata_t,
        n_splits=n_splits,
        seed=int(config["selection"]["split_seed"]),
        shuffle=True,
    )[:maximum_folds]
    if maximum_folds != n_splits:
        raise RuntimeError("formal V8 E1 requires all preregistered Session-T folds")
    nested_folds = [
        (
            train,
            outer_test,
            *nested_run_grouped_indices(metadata_t, train, outer_test),
        )
        for train, outer_test in folds
    ]
    split_manifest = {
        "stage": "development",
        "subject": subject,
        "session": "T",
        "method": "nested_run_grouped_oof",
        "n_splits": n_splits,
        "split_seed": int(config["selection"]["split_seed"]),
        "folds": [
            {
                "fold": fold,
                "outer_train_trial_ids": [
                    metadata_t[int(index)]["trial_id"] for index in outer_train
                ],
                "outer_test_trial_ids": [
                    metadata_t[int(index)]["trial_id"] for index in outer_test
                ],
                "inner_train_trial_ids": [
                    metadata_t[int(index)]["trial_id"] for index in inner_train
                ],
                "inner_validation_trial_ids": [
                    metadata_t[int(index)]["trial_id"] for index in inner_validation
                ],
                "outer_train_runs": sorted(
                    {metadata_t[int(index)]["run"] for index in outer_train}
                ),
                "outer_test_run": metadata_t[int(outer_test[0])]["run"],
                "inner_validation_run": inner_run,
            }
            for fold, (
                outer_train,
                outer_test,
                inner_train,
                inner_validation,
                inner_run,
            ) in enumerate(nested_folds)
        ],
    }
    augmentation = dict(config["augmentation"])
    resolved = {
        **config,
        "active_model": model_name,
        "active_subject": subject,
        "active_seed": seed,
        "selection": {
            **config["selection"],
            "max_epochs": cv_epochs,
            "patience": patience,
            "executed_folds": maximum_folds,
        },
        "optimizer": BASELINE_OPTIMIZERS[model_name],
    }
    fingerprint = build_v8_run_fingerprint(
        resolved_run_config=resolved,
        source_tree=source_tree,
        data={subject_path.name: file_sha256(subject_path), "access": access_manifest},
        split=split_manifest,
        augmentation=augmentation,
        prior={"policy": "none"},
        checkpoint={
            "policy": (
                "inner-run best validation kappa -> fixed-epoch outer-train retrain -> "
                "single outer-run test"
            ),
            "evaluation_scope": "nested Session-T outer-run OOF only",
            "session_e_accessed": False,
        },
        environment={**environment, "official_source_locks": official_locks},
    )
    run_dir = ensure_dir(
        output / model_name / f"subject_{subject:02d}" / f"seed_{seed}"
    )
    fingerprint_path = run_dir / "source_fingerprint.json"
    if (run_dir / "manifest.json").is_file():
        validate_v8_resume_fingerprint(fingerprint_path, fingerprint)
        validate_run_artifact_manifest(
            run_dir,
            required_files=_required_files(maximum_folds),
            verify_hashes=True,
            verify_prediction_schema=True,
        )
        return read_json(run_dir / "metrics.json")
    if fingerprint_path.is_file():
        validate_v8_resume_fingerprint(fingerprint_path, fingerprint)
    else:
        write_v8_fingerprint(fingerprint_path, fingerprint)
        write_json(run_dir / "source_tree_manifest.json", source_tree)
        write_json(run_dir / "data_access_manifest.json", access_manifest)
        write_json(run_dir / "split_manifest.json", split_manifest)
        write_json(run_dir / "augmentation_manifest.json", augmentation)
        (run_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )
    write_json(
        run_dir / "runtime_status.json",
        {"status": "running", "started_at": time.time(), "session_e_accessed": False},
    )

    fold_rows: list[dict[str, Any]] = []
    oof_parts: list[dict[str, np.ndarray]] = []
    total_seconds = 0.0
    total_steps = 0
    parameters = 0
    implementation_id = ""
    for fold_index, (
        outer_train_indices,
        outer_test_indices,
        inner_train_indices,
        inner_validation_indices,
        inner_validation_run,
    ) in enumerate(nested_folds):
        fold_dir = ensure_dir(run_dir / f"fold_{fold_index}")
        cached = _load_fold(
            fold_dir,
            expected_indices=outer_test_indices,
            expected_labels=y_t[outer_test_indices],
        )
        if cached is not None:
            result, prediction = cached
        else:
            selection_gain = fit_fixed_gain(
                carrier_t[inner_train_indices],
                clip=float(config["preprocessing"]["clip_after_gain"]),
            )
            x_inner_train = prepare_model_input(
                model_name,
                apply_fixed_gain(carrier_t[inner_train_indices], selection_gain),
                sfreq=float(data["sfreq"]),
            )
            x_inner_validation = prepare_model_input(
                model_name,
                apply_fixed_gain(carrier_t[inner_validation_indices], selection_gain),
                sfreq=float(data["sfreq"]),
            )
            fold_seed = int(seed) * 100_003 + fold_index * 1_009
            selection_fit = fit_baseline(
                model_name,
                source_root=source_root,
                x_train=x_inner_train,
                y_train=y_t[inner_train_indices],
                x_validation=x_inner_validation,
                y_validation=y_t[inner_validation_indices],
                device=device,
                seed=fold_seed,
                epochs=cv_epochs,
                patience=patience,
                augmentation=augmentation,
                run_label=(
                    f"V8-E1-select:{model_name}:S{subject}:seed{seed}:fold{fold_index}"
                ),
            )
            selection_evaluation = predict_baseline(
                selection_fit.model,
                x_inner_validation,
                y_t[inner_validation_indices],
                device=device,
                batch_size=int(BASELINE_OPTIMIZERS[model_name]["batch_size"]),
            )
            selected_epoch = max(
                int(config["selection"]["minimum_outer_retrain_epochs"]),
                int(selection_fit.best_epoch),
            )
            selected_epoch = min(selected_epoch, cv_epochs)
            selection_result = {
                "fold": fold_index,
                "fold_seed": fold_seed,
                "inner_validation_run": inner_validation_run,
                "best_epoch": selection_fit.best_epoch,
                "selected_outer_retrain_epoch": selected_epoch,
                "best_selection_metric": selection_fit.best_metric,
                "accuracy": selection_evaluation["accuracy"],
                "balanced_accuracy": selection_evaluation["balanced_accuracy"],
                "kappa": selection_evaluation["kappa"],
                "macro_f1": selection_evaluation["macro_f1"],
                "optimizer_steps": selection_fit.optimizer_steps,
                "elapsed_seconds": selection_fit.elapsed_seconds,
                "gain": selection_gain.values.reshape(-1).tolist(),
            }
            torch.save(selection_fit.best_state, fold_dir / "selection_best.pt")
            torch.save(selection_fit.last_state, fold_dir / "selection_last.pt")
            write_csv(fold_dir / "selection_history.csv", selection_fit.history)
            write_json(fold_dir / "selection_result.json", selection_result)
            np.savez_compressed(
                fold_dir / "selection_predictions.npz",
                indices=inner_validation_indices.astype(np.int64),
                logits=selection_evaluation["logits"].astype(np.float32),
                labels=selection_evaluation["labels"].astype(np.int64),
            )

            outer_gain = fit_fixed_gain(
                carrier_t[outer_train_indices],
                clip=float(config["preprocessing"]["clip_after_gain"]),
            )
            x_outer_train = prepare_model_input(
                model_name,
                apply_fixed_gain(carrier_t[outer_train_indices], outer_gain),
                sfreq=float(data["sfreq"]),
            )
            x_outer_test = prepare_model_input(
                model_name,
                apply_fixed_gain(carrier_t[outer_test_indices], outer_gain),
                sfreq=float(data["sfreq"]),
            )
            outer_seed = fold_seed + 1_000_003
            outer_fit = fit_baseline(
                model_name,
                source_root=source_root,
                x_train=x_outer_train,
                y_train=y_t[outer_train_indices],
                x_validation=None,
                y_validation=None,
                device=device,
                seed=outer_seed,
                epochs=selected_epoch,
                patience=selected_epoch,
                augmentation=augmentation,
                fixed_epoch=selected_epoch,
                scheduler_epochs=cv_epochs,
                run_label=f"V8-E1-outer:{model_name}:S{subject}:seed{seed}:fold{fold_index}",
            )
            evaluation = predict_baseline(
                outer_fit.model,
                x_outer_test,
                y_t[outer_test_indices],
                device=device,
                batch_size=int(BASELINE_OPTIMIZERS[model_name]["batch_size"]),
            )
            parameters = sum(parameter.numel() for parameter in outer_fit.model.parameters())
            implementation_id = getattr(outer_fit.model, "implementation_id", "")
            result = {
                "fold": fold_index,
                "fold_seed": fold_seed,
                "outer_seed": outer_seed,
                "inner_validation_run": inner_validation_run,
                "inner_best_epoch": selection_fit.best_epoch,
                "selected_outer_retrain_epoch": selected_epoch,
                "inner_best_selection_metric": selection_fit.best_metric,
                "accuracy": evaluation["accuracy"],
                "balanced_accuracy": evaluation["balanced_accuracy"],
                "kappa": evaluation["kappa"],
                "macro_f1": evaluation["macro_f1"],
                "selection_optimizer_steps": selection_fit.optimizer_steps,
                "outer_optimizer_steps": outer_fit.optimizer_steps,
                "optimizer_steps": selection_fit.optimizer_steps + outer_fit.optimizer_steps,
                "selection_elapsed_seconds": selection_fit.elapsed_seconds,
                "outer_elapsed_seconds": outer_fit.elapsed_seconds,
                "elapsed_seconds": selection_fit.elapsed_seconds + outer_fit.elapsed_seconds,
                "parameters": parameters,
                "implementation_id": implementation_id,
                "outer_gain": outer_gain.values.reshape(-1).tolist(),
            }
            prediction = {
                "indices": outer_test_indices.astype(np.int64),
                "logits": evaluation["logits"].astype(np.float32),
                "labels": evaluation["labels"].astype(np.int64),
            }
            torch.save(outer_fit.last_state, fold_dir / "best.pt")
            torch.save(outer_fit.last_state, fold_dir / "last.pt")
            write_csv(fold_dir / "history.csv", outer_fit.history)
            write_json(fold_dir / "result.json", result)
            np.savez_compressed(fold_dir / "outer_test_predictions.npz", **prediction)
            del (
                selection_fit,
                selection_evaluation,
                outer_fit,
                evaluation,
                x_inner_train,
                x_inner_validation,
                x_outer_train,
                x_outer_test,
            )
        fold_rows.append(result)
        oof_parts.append(prediction)
        total_seconds += float(result["elapsed_seconds"])
        total_steps += int(result["optimizer_steps"])
        parameters = int(result["parameters"])
        implementation_id = str(result["implementation_id"])
        print(
            "__V8_E1_FOLD_DONE__ "
            f"model={model_name} subject={subject} seed={seed} fold={fold_index} "
            f"kappa={float(result['kappa']):.6f}",
            flush=True,
        )

    indices, logits, labels = merge_oof_predictions(oof_parts, labels=y_t)
    probabilities = _probabilities(logits)
    predictions = probabilities.argmax(axis=1)
    pooled = classification_metrics(labels, predictions, n_classes=4)
    write_trial_predictions(
        run_dir,
        logits=logits,
        probabilities=probabilities,
        pred=predictions,
        label=labels,
        subject=[metadata_t[int(index)]["subject"] for index in indices],
        session="T",
        run=[metadata_t[int(index)]["run"] for index in indices],
        trial_id=[metadata_t[int(index)]["trial_id"] for index in indices],
        seed=seed,
        model=f"{model_name}_v8_e1_oof",
    )
    metrics = {
        "status": "completed",
        "stage": "E1",
        "protocol": "bci2a_session_t_nested_six_fold_oof",
        "model": model_name,
        "implementation_id": implementation_id,
        "source_fidelity": config["source_fidelity"][model_name],
        "subject": subject,
        "seed": seed,
        "accuracy": pooled["accuracy"],
        "balanced_accuracy": pooled["balanced_accuracy"],
        "kappa": pooled["kappa"],
        "macro_f1": pooled["macro_f1"],
        "fold_accuracy_mean": float(np.mean([row["accuracy"] for row in fold_rows])),
        "fold_kappa_mean": float(np.mean([row["kappa"] for row in fold_rows])),
        "parameters": parameters,
        "optimizer_steps": total_steps,
        "train_seconds": total_seconds,
        "selection_session": "T",
        "evaluation_scope": "Session-T nested outer-run OOF test",
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
        "run_fingerprint": fingerprint["combined_sha256"],
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "runtime_status.json",
        {
            "status": "completed",
            "completed_at": time.time(),
            "session_e_accessed": False,
        },
    )
    (run_dir / "stdout.log").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    write_run_artifact_manifest(run_dir, required_files=_required_files(maximum_folds))
    del data, x_t_raw, y_t, carrier_t, logits, probabilities
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/experiments/v8_e1_baselines.yaml")
    parser.add_argument("--models", default="")
    parser.add_argument("--subjects", default="")
    parser.add_argument("--screening-seed", type=int, default=None)
    parser.add_argument("--confirmation-seeds", default="")
    parser.add_argument("--confirmation-top-k", type=int, default=None)
    parser.add_argument("--cv-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-smoke-only", action="store_true")
    args = parser.parse_args()

    configure_cache_env()
    output = ensure_dir(Path(args.output).resolve())
    data_root = Path(args.data).resolve()
    source_root = Path(args.source_root).resolve()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("stage") != "development":
        raise RuntimeError("V8 E1 must run under the locked development stage")
    if bool(config["data_access"].get("heldout_session_e_accessed")):
        raise RuntimeError("V8 E1 config may not unlock Session E")
    models = _csv(args.models) if args.models else list(config["models"])
    subjects = _csv(args.subjects, int) if args.subjects else list(config["subjects"])
    unknown = sorted(set(models) - set(V62_NEURAL_BASELINES))
    if unknown:
        raise ValueError(f"unknown baseline models: {unknown}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    official_locks = verify_official_source_locks(source_root)
    project_sources = collect_source_tree_manifest(ROOT)
    source_tree = dict(sorted({**project_sources, **_external_source_manifest(source_root)}.items()))
    write_json(output / "source_tree_manifest.json", source_tree)
    write_json(
        output / "source_tree_summary.json",
        {"files": len(source_tree), "sha256": source_tree_digest(source_tree)},
    )
    write_json(output / "official_source_locks.json", official_locks)
    write_json(output / "heldout_lock_manifest.json", v8_heldout_lock_manifest())
    _source_smoke(output, source_root, args.device)
    if args.source_smoke_only:
        print(json.dumps({"status": "passed", "source_smoke_only": True}, indent=2))
        return

    screening_seed = int(
        args.screening_seed if args.screening_seed is not None else config["screening_seed"]
    )
    confirmation_seeds = (
        _csv(args.confirmation_seeds, int)
        if args.confirmation_seeds
        else list(config["confirmation_seeds"])
    )
    top_k = int(
        args.confirmation_top_k
        if args.confirmation_top_k is not None
        else config["confirmation_top_k"]
    )
    if screening_seed not in confirmation_seeds:
        raise ValueError("confirmation seeds must include the screening seed")
    if not 1 <= top_k <= len(models):
        raise ValueError("confirmation_top_k must select at least one available model")
    cv_epochs = int(args.cv_epochs or config["selection"]["max_epochs"])
    patience = int(args.patience or config["selection"]["patience"])
    n_splits = int(config["selection"]["n_splits"])
    maximum_folds = min(n_splits, int(args.max_folds or n_splits))
    environment = _environment()
    rows: list[dict[str, Any]] = []

    for model_name in models:
        for subject in subjects:
            row = _run_one(
                output=output,
                config=config,
                source_root=source_root,
                source_tree=source_tree,
                environment=environment,
                official_locks=official_locks,
                data_root=data_root,
                model_name=model_name,
                subject=int(subject),
                seed=screening_seed,
                cv_epochs=cv_epochs,
                patience=patience,
                maximum_folds=maximum_folds,
                device=args.device,
            )
            rows.append(row)
            write_csv(output / "summary.csv", rows)

    ranking = rank_screening_models(rows, subjects=subjects, screening_seed=screening_seed)
    write_csv(output / "screening_leaderboard.csv", ranking)
    selected_models = [row["model"] for row in ranking[:top_k]]
    write_json(
        output / "confirmation_selection.json",
        {
            "selection_scope": "Session-T OOF only",
            "ranking_metric": config["selection"]["ranking_metric"],
            "top_k": top_k,
            "models": selected_models,
            "confirmation_seeds": confirmation_seeds,
            "session_e_accessed": False,
        },
    )
    for model_name in selected_models:
        for seed in confirmation_seeds:
            if int(seed) == screening_seed:
                continue
            for subject in subjects:
                row = _run_one(
                    output=output,
                    config=config,
                    source_root=source_root,
                    source_tree=source_tree,
                    environment=environment,
                    official_locks=official_locks,
                    data_root=data_root,
                    model_name=model_name,
                    subject=int(subject),
                    seed=int(seed),
                    cv_epochs=cv_epochs,
                    patience=patience,
                    maximum_folds=maximum_folds,
                    device=args.device,
                )
                rows.append(row)
                write_csv(output / "summary.csv", rows)
    rows = sorted(rows, key=lambda row: (row["model"], int(row["subject"]), int(row["seed"])))
    write_csv(output / "summary.csv", rows)
    write_json(
        output / "campaign_status.json",
        {
            "status": "completed",
            "stage": "E1",
            "protocol": config["protocol"],
            "screened_models": models,
            "subjects": subjects,
            "screening_seed": screening_seed,
            "confirmed_models": selected_models,
            "confirmation_seeds": confirmation_seeds,
            "runs": len(rows),
            "session_e_accessed": False,
            "source_tree_sha256": source_tree_digest(source_tree),
        },
    )
    print(json.dumps({"status": "completed", "runs": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
