#!/usr/bin/env python3
"""Run all frozen E1 baselines under the same BCI2a T-to-E contract."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.baselines.neural import verify_official_source_locks  # noqa: E402
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
    sha256_fingerprint,
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
    write_trial_predictions,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    build_v8_run_fingerprint,
    collect_source_tree_manifest,
    mapping_sha256,
    source_tree_digest,
    validate_v8_freeze_manifest,
    validate_v8_resume_fingerprint,
    write_v8_fingerprint,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v8_e1_baselines import (  # noqa: E402
    _environment,
    _external_source_manifest,
)
from scripts.run_v8_e2_zero_delay import _subject_file  # noqa: E402
from scripts.run_v8_e6_bci2a_frozen import _metadata, _session_indices  # noqa: E402


RUN_FILES = (
    "manifest.json",
    "resolved_config.yaml",
    "source_fingerprint.json",
    "source_tree_manifest.json",
    "data_access_manifest.json",
    "augmentation_manifest.json",
    "history.csv",
    "best.pt",
    "last.pt",
    "metrics.json",
    "predictions.npz",
    "predictions.csv",
    "state_audit.json",
    "runtime_status.json",
    "stdout.log",
    "stderr.log",
)
CAMPAIGN_FILES = (
    "manifest.json",
    "campaign_status.json",
    "summary.csv",
    "resolved_campaign.yaml",
    "source_tree_manifest.json",
    "freeze_manifest.json",
    "official_source_locks.json",
)


def _csv(value: str, cast: Any = str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _probabilities(logits: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.from_numpy(logits).float(), dim=1).numpy()


def _state_digest(model: torch.nn.Module) -> str:
    return sha256_fingerprint(mapping_sha256(model.state_dict()))


def _run_one(
    *,
    output: Path,
    source_root: Path,
    subject_path: Path,
    data: dict[str, Any],
    t_indices: np.ndarray,
    e_indices: np.ndarray,
    metadata_t: list[dict[str, Any]],
    freeze: dict[str, Any],
    source_tree: dict[str, str],
    environment: dict[str, Any],
    official_locks: dict[str, Any],
    model_name: str,
    subject: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    baseline = freeze["baselines"]
    protocol = dict(baseline["protocol_config"])
    final_epoch = int(baseline["fixed_epochs"][model_name])
    scheduler_horizon = int(protocol["selection"]["max_epochs"])
    augmentation = dict(protocol["augmentation"])
    run_seed = int(seed) * 100_003 + int(subject) * 1_009
    resolved = {
        "stage": "E6",
        "protocol": "bci2a_session_t_train_session_e_single_evaluation",
        "freeze_sha256": freeze["combined_sha256"],
        "model": model_name,
        "subject": int(subject),
        "seed": int(seed),
        "run_seed": run_seed,
        "final_epoch": final_epoch,
        "scheduler_horizon": scheduler_horizon,
        "optimizer": BASELINE_OPTIMIZERS[model_name],
        "preprocessing": protocol["preprocessing"],
        "augmentation": augmentation,
    }
    split = {
        "train_session": "T",
        "evaluation_session": "E",
        "train_trial_ids": [row["trial_id"] for row in metadata_t],
        "evaluation_trial_ids_sha256": sha256_fingerprint(
            [str(np.asarray(data["trial_id"])[index]) for index in e_indices]
        ),
        "evaluation_labels_used_for_checkpoint_selection": False,
    }
    fingerprint = build_v8_run_fingerprint(
        resolved_run_config=resolved,
        source_tree=source_tree,
        data={subject_path.name: file_sha256(subject_path)},
        split=split,
        augmentation=augmentation,
        prior={"policy": "none"},
        checkpoint={
            "freeze_sha256": freeze["combined_sha256"],
            "fixed_epoch": final_epoch,
            "scheduler_horizon": scheduler_horizon,
            "session_e_checkpoint_selection": False,
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
            required_files=RUN_FILES,
            verify_hashes=True,
            verify_prediction_schema=True,
        )
        return read_json(run_dir / "metrics.json")
    if fingerprint_path.is_file():
        validate_v8_resume_fingerprint(fingerprint_path, fingerprint)
    else:
        write_v8_fingerprint(fingerprint_path, fingerprint)
        write_json(run_dir / "source_tree_manifest.json", source_tree)
        write_json(run_dir / "augmentation_manifest.json", augmentation)
        (run_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )
    write_json(
        run_dir / "runtime_status.json",
        {
            "status": "training_session_t",
            "started_at": time.time(),
            "session_e_accessed": False,
        },
    )

    sfreq = float(np.asarray(data["sfreq"]).item())
    epoch_tmin = float(np.asarray(data["epoch_tmin"]).item())
    x_t_raw = np.asarray(data["X"], dtype=np.float32)[t_indices]
    y_t = np.asarray(data["y"], dtype=np.int64)[t_indices]
    carrier_t = task_carrier(x_t_raw, sfreq=sfreq, epoch_tmin=epoch_tmin)
    gain = fit_fixed_gain(
        carrier_t, clip=float(protocol["preprocessing"]["clip_after_gain"])
    )
    x_t = prepare_model_input(
        model_name, apply_fixed_gain(carrier_t, gain), sfreq=sfreq
    )
    fit = fit_baseline(
        model_name,
        source_root=source_root,
        x_train=x_t,
        y_train=y_t,
        x_validation=None,
        y_validation=None,
        device=device,
        seed=run_seed,
        epochs=final_epoch,
        patience=final_epoch,
        augmentation=augmentation,
        fixed_epoch=final_epoch,
        scheduler_epochs=scheduler_horizon,
        run_label=f"V8-E6:{model_name}:S{subject}:seed{seed}",
    )
    model = fit.model
    state_before_e = _state_digest(model)
    torch.save(fit.last_state, run_dir / "best.pt")
    torch.save(fit.last_state, run_dir / "last.pt")

    # Evaluation labels and signals enter only after the fixed checkpoint exists.
    metadata_e = _metadata(data, e_indices, stage="bci2a_evaluation", role="evaluation")
    x_e_raw = np.asarray(data["X"], dtype=np.float32)[e_indices]
    y_e = np.asarray(data["y"], dtype=np.int64)[e_indices]
    carrier_e = task_carrier(x_e_raw, sfreq=sfreq, epoch_tmin=epoch_tmin)
    x_e = prepare_model_input(
        model_name, apply_fixed_gain(carrier_e, gain), sfreq=sfreq
    )
    evaluation = predict_baseline(
        model,
        x_e,
        y_e,
        device=device,
        batch_size=int(BASELINE_OPTIMIZERS[model_name]["batch_size"]),
    )
    state_after_e = _state_digest(model)
    if state_before_e != state_after_e:
        raise RuntimeError("baseline state changed while evaluating Session E")

    probabilities = _probabilities(evaluation["logits"])
    write_trial_predictions(
        run_dir,
        logits=evaluation["logits"],
        probabilities=probabilities,
        pred=evaluation["pred"],
        label=evaluation["labels"],
        subject=[row["subject"] for row in metadata_e],
        session="E",
        run=[row["run"] for row in metadata_e],
        trial_id=[row["trial_id"] for row in metadata_e],
        seed=seed,
        model=f"v8_e6_{model_name}",
    )
    write_csv(run_dir / "history.csv", fit.history)
    access = {
        "stage": "bci2a_evaluation",
        "train_session": "T",
        "train_trials": len(metadata_t),
        "evaluation_session": "E",
        "evaluation_trials": len(metadata_e),
        "session_e_first_used_after_checkpoint_sha256": state_before_e,
        "session_e_checkpoint_selection": False,
        "session_e_gradient_updates": False,
        "openbmi_s2_accessed": False,
    }
    write_json(run_dir / "data_access_manifest.json", access)
    write_json(
        run_dir / "state_audit.json",
        {
            "checkpoint_before_session_e": state_before_e,
            "state_after_session_e": state_after_e,
            "identical": True,
        },
    )
    metrics = {
        "status": "completed",
        "stage": "E6_BASELINE",
        "protocol": "bci2a_session_t_train_session_e_single_evaluation",
        "model": model_name,
        "implementation_id": getattr(model, "implementation_id", ""),
        "subject": int(subject),
        "seed": int(seed),
        "accuracy": evaluation["accuracy"],
        "balanced_accuracy": evaluation["balanced_accuracy"],
        "kappa": evaluation["kappa"],
        "macro_f1": evaluation["macro_f1"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "final_epoch": final_epoch,
        "optimizer_steps": fit.optimizer_steps,
        "train_seconds": fit.elapsed_seconds,
        "selection_session": "T development artifacts",
        "evaluation_session": "E",
        "heldout_e_selected_checkpoint": False,
        "freeze_sha256": freeze["combined_sha256"],
        "run_fingerprint": fingerprint["combined_sha256"],
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "runtime_status.json",
        {"status": "completed", "completed_at": time.time(), "session_e_accessed": True},
    )
    (run_dir / "stdout.log").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    write_run_artifact_manifest(run_dir, required_files=RUN_FILES)
    return metrics


def _baseline_subject_worker_command(
    *,
    args: argparse.Namespace,
    output: Path,
    subject: int,
    models: list[str],
    seeds: list[int],
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--data",
        str(Path(args.data).resolve()),
        "--source-root",
        str(Path(args.source_root).resolve()),
        "--freeze",
        str(Path(args.freeze).resolve()),
        "--output",
        str(output),
        "--config",
        str(Path(args.config).resolve()),
        "--device",
        str(args.device),
        "--worker-subject",
        str(subject),
        "--worker-models",
        ",".join(models),
        "--worker-seeds",
        ",".join(str(seed) for seed in seeds),
    ]


def _run_baseline_subject_worker(
    command: list[str], result_paths: list[Path]
) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    worker_threads = int(environment.get("DPC_SNN_E6_BASELINE_WORKER_THREADS", "8"))
    if worker_threads < 1 or worker_threads > 32:
        raise ValueError("DPC_SNN_E6_BASELINE_WORKER_THREADS must be between 1 and 32")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = str(worker_threads)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "E6 baseline subject worker failed with code "
            f"{completed.returncode}: {' '.join(command)}\n{completed.stderr[-8000:]}"
        )
    missing = [str(path) for path in result_paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            "E6 baseline subject worker missed results: " + ", ".join(missing)
        )
    return [read_json(path) for path in result_paths]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default="configs/experiments/v8_e6_baselines_frozen.yaml"
    )
    parser.add_argument("--models", default="")
    parser.add_argument("--subjects", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-subject", type=int, default=None)
    parser.add_argument("--worker-models", default="")
    parser.add_argument("--worker-seeds", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()

    configure_cache_env()
    output = ensure_dir(Path(args.output).resolve())
    data_root = Path(args.data).resolve()
    source_root = Path(args.source_root).resolve()
    config = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    project_sources = collect_source_tree_manifest(ROOT)
    project_digest = source_tree_digest(project_sources)
    freeze = validate_v8_freeze_manifest(
        Path(args.freeze).resolve(), expected_source_tree_sha256=project_digest
    )
    official_locks = verify_official_source_locks(source_root)
    if official_locks != freeze["baselines"]["official_source_locks"]:
        raise RuntimeError("official baseline sources differ from the frozen E1 sources")
    source_tree = dict(
        sorted({**project_sources, **_external_source_manifest(source_root)}.items())
    )
    models = _csv(args.models) if args.models else list(config["models"])
    subjects = _csv(args.subjects, int) if args.subjects else list(config["subjects"])
    seeds = _csv(args.seeds, int) if args.seeds else list(config["seeds"])
    if models != list(freeze["baselines"]["models"]):
        if not args.canary:
            raise RuntimeError("formal E6 must run every frozen baseline")
    workers = int(args.workers)
    if workers < 1 or workers > 8:
        raise ValueError("V8 E6 baseline workers must be between 1 and 8")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    if args.worker_subject is not None:
        worker_subject = int(args.worker_subject)
        worker_models = _csv(args.worker_models)
        worker_seeds = _csv(args.worker_seeds, int)
        if (
            worker_subject not in config["subjects"]
            or not worker_models
            or not worker_seeds
            or set(worker_models).difference(models)
            or set(worker_seeds).difference(config["seeds"])
        ):
            raise ValueError("invalid E6 baseline subject-worker contract")
        subject_path = _subject_file(data_root, worker_subject)
        data = load_processed_npz(subject_path)
        t_indices = _session_indices(data, "T")
        e_indices = _session_indices(data, "E")
        metadata_t = _metadata(
            data, t_indices, stage="bci2a_evaluation", role="training"
        )
        worker_environment = _environment()
        worker_rows = []
        for model_name in worker_models:
            for seed in worker_seeds:
                worker_rows.append(
                    _run_one(
                        output=output,
                        source_root=source_root,
                        subject_path=subject_path,
                        data=data,
                        t_indices=t_indices,
                        e_indices=e_indices,
                        metadata_t=metadata_t,
                        freeze=freeze,
                        source_tree=source_tree,
                        environment=worker_environment,
                        official_locks=official_locks,
                        model_name=model_name,
                        subject=worker_subject,
                        seed=int(seed),
                        device=args.device,
                    )
                )
        print(json.dumps({"status": "completed", "runs": len(worker_rows)}, indent=2))
        return
    full_contract = (
        models == list(config["models"])
        and subjects == list(config["subjects"])
        and seeds == list(config["seeds"])
    )
    if not full_contract and not args.canary:
        raise RuntimeError("partial E6 baseline execution requires --canary")
    write_json(output / "source_tree_manifest.json", source_tree)
    write_json(output / "freeze_manifest.json", freeze)
    write_json(output / "official_source_locks.json", official_locks)
    (output / "resolved_campaign.yaml").write_text(
        yaml.safe_dump(
            {
                **config,
                "active_models": models,
                "active_subjects": subjects,
                "active_seeds": seeds,
                "freeze_sha256": freeze["combined_sha256"],
                "canary": bool(args.canary),
                "parallel_workers": workers,
                "worker_cpu_threads": (
                    int(os.environ.get("DPC_SNN_E6_BASELINE_WORKER_THREADS", "8"))
                    if workers > 1
                    else None
                ),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    environment = _environment()
    rows: list[dict[str, Any]] = []
    if workers == 1:
        for subject in subjects:
            subject_path = _subject_file(data_root, int(subject))
            data = load_processed_npz(subject_path)
            t_indices = _session_indices(data, "T")
            e_indices = _session_indices(data, "E")
            metadata_t = _metadata(
                data, t_indices, stage="bci2a_evaluation", role="training"
            )
            for model_name in models:
                for seed in seeds:
                    row = _run_one(
                        output=output,
                        source_root=source_root,
                        subject_path=subject_path,
                        data=data,
                        t_indices=t_indices,
                        e_indices=e_indices,
                        metadata_t=metadata_t,
                        freeze=freeze,
                        source_tree=source_tree,
                        environment=environment,
                        official_locks=official_locks,
                        model_name=model_name,
                        subject=int(subject),
                        seed=int(seed),
                        device=args.device,
                    )
                    rows.append(row)
                    write_csv(output / "summary.csv", rows)
                    print(
                        "__V8_E6_BASELINE_DONE__ "
                        f"model={model_name} subject={subject} seed={seed} "
                        f"accuracy={float(row['accuracy']):.6f}",
                        flush=True,
                    )
            del data
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for subject in subjects:
                result_paths = [
                    output
                    / model_name
                    / f"subject_{int(subject):02d}"
                    / f"seed_{int(seed)}"
                    / "metrics.json"
                    for model_name in models
                    for seed in seeds
                ]
                command = _baseline_subject_worker_command(
                    args=args,
                    output=output,
                    subject=int(subject),
                    models=models,
                    seeds=seeds,
                )
                futures[executor.submit(
                    _run_baseline_subject_worker, command, result_paths
                )] = int(subject)
            for future in as_completed(futures):
                rows.extend(future.result())
    rows = sorted(
        rows, key=lambda row: (str(row["model"]), int(row["subject"]), int(row["seed"]))
    )
    write_csv(output / "summary.csv", rows)
    status = {
        "status": "completed" if full_contract else "canary_completed",
        "stage": "E6_BASELINES",
        "protocol": config["protocol"],
        "models": models,
        "subjects": subjects,
        "seeds": seeds,
        "runs": len(rows),
        "full_registered_contract": full_contract,
        "freeze_sha256": freeze["combined_sha256"],
        "session_e_accessed": True,
        "session_e_used_for_selection": False,
        "openbmi_s2_accessed": False,
        "parallel_workers": workers,
        "worker_cpu_threads": (
            int(os.environ.get("DPC_SNN_E6_BASELINE_WORKER_THREADS", "8"))
            if workers > 1
            else None
        ),
    }
    write_json(output / "campaign_status.json", status)
    write_run_artifact_manifest(output, required_files=CAMPAIGN_FILES)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
