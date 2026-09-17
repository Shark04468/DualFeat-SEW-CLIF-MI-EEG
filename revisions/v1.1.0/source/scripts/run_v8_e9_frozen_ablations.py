#!/usr/bin/env python3
"""Run preregistered frozen V8 structural ablations on BCI2a T-to-E."""

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

from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
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
from dpc_snn.experiments.v8_training import (  # noqa: E402
    cache_v8_physical_rates,
    fit_v8,
    fit_v8_physical_gain,
    predict_v8,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v8_e2_zero_delay import _environment, _fit_kwargs, _subject_file  # noqa: E402
from scripts.run_v8_e6_bci2a_frozen import (  # noqa: E402
    _build_model,
    _metadata,
    _probabilities,
    _session_indices,
)


RUN_FILES = (
    "manifest.json",
    "resolved_config.yaml",
    "source_fingerprint.json",
    "source_tree_manifest.json",
    "data_access_manifest.json",
    "history.csv",
    "best.pt",
    "last.pt",
    "metrics.json",
    "predictions.npz",
    "predictions.csv",
    "prefix_predictions.npz",
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
)


def _csv(value: str, cast: Any = str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _state_digest(model: torch.nn.Module) -> str:
    return sha256_fingerprint(mapping_sha256(model.state_dict()))


def _ablation_contract(
    freeze: dict[str, Any], config: dict[str, Any], variant: str
) -> tuple[dict[str, Any], dict[str, Any], int]:
    registry = dict(config["retrained_single_variable_ablations"])
    if variant not in registry:
        raise ValueError(f"unknown frozen E9 ablation {variant!r}")
    spec = dict(registry[variant])
    model = {
        **dict(freeze["architecture"]["model_config"]),
        **dict(spec["model_overrides"]),
        "delay_auxiliary_enabled": False,
    }
    augmentation = {
        **dict(freeze["augmentation"]),
        **dict(spec["augmentation_overrides"]),
    }
    if not any(
        bool(model.get(name))
        for name in (
            "use_statistical_branch",
            "use_covariance_branch",
            "use_temporal_branch",
        )
    ):
        raise RuntimeError(f"E9 ablation {variant!r} removed every feature branch")
    return model, augmentation, int(freeze["checkpoint_rule"]["final_epoch"])


def _run_one(
    *,
    output: Path,
    subject_path: Path,
    data: dict[str, Any],
    t_indices: np.ndarray,
    e_indices: np.ndarray,
    metadata_t: list[dict[str, Any]],
    freeze: dict[str, Any],
    config: dict[str, Any],
    source_tree: dict[str, str],
    environment: dict[str, Any],
    e6_campaign_sha256: str,
    variant: str,
    subject: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    model_config, augmentation, final_epoch = _ablation_contract(
        freeze, config, variant
    )
    training = dict(freeze["training"])
    scheduler_horizon = int(freeze["checkpoint_rule"]["scheduler_horizon"])
    run_seed = int(seed) * 100_003 + int(subject) * 1_009
    resolved = {
        "stage": "E9",
        "protocol": config["protocol"],
        "freeze_sha256": freeze["combined_sha256"],
        "variant": variant,
        "subject": int(subject),
        "seed": int(seed),
        "run_seed": run_seed,
        "model": model_config,
        "training": training,
        "augmentation": augmentation,
        "final_epoch": final_epoch,
        "scheduler_horizon": scheduler_horizon,
        "delay": {"enabled": False, "reason": "structural core ablation"},
        "reference_e6_campaign_sha256": e6_campaign_sha256,
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
        prior={"policy": "none", "delay_enabled": False},
        checkpoint={
            "freeze_sha256": freeze["combined_sha256"],
            "fixed_epoch": final_epoch,
            "scheduler_horizon": scheduler_horizon,
            "session_e_checkpoint_selection": False,
            "reference_e6_campaign_sha256": e6_campaign_sha256,
        },
        environment=environment,
    )
    run_dir = ensure_dir(
        output / variant / f"subject_{subject:02d}" / f"seed_{seed}"
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

    x_t = np.asarray(data["X"], dtype=np.float32)[t_indices]
    y_t = np.asarray(data["y"], dtype=np.int64)[t_indices]
    model = _build_model(model_config, seed=run_seed)
    fit_v8_physical_gain(model, x_t, device=device, batch_size=16)
    train_rates = cache_v8_physical_rates(model, x_t, device=device, batch_size=16)
    fit = fit_v8(
        model,
        train_rates,
        y_t,
        validation_rates=None,
        validation_labels=None,
        device=device,
        seed=run_seed,
        epochs=final_epoch,
        patience=final_epoch,
        minimum_epochs=final_epoch,
        fixed_epoch=final_epoch,
        scheduler_epochs=scheduler_horizon,
        run_label=f"V8-E9:{variant}:S{subject}:seed{seed}",
        **_fit_kwargs(training, augmentation),
    )
    model = fit.model
    torch.save(fit.best_state, run_dir / "best.pt")
    torch.save(fit.last_state, run_dir / "last.pt")
    checkpoint_before_e = _state_digest(model)

    # Session-E labels/features are materialized only after the fixed checkpoint exists.
    metadata_e = _metadata(
        data, e_indices, stage="bci2a_evaluation", role="evaluation"
    )
    x_e = np.asarray(data["X"], dtype=np.float32)[e_indices]
    y_e = np.asarray(data["y"], dtype=np.int64)[e_indices]
    evaluation_rates = cache_v8_physical_rates(
        model, x_e, device=device, batch_size=16
    )
    evaluation = predict_v8(
        model,
        evaluation_rates,
        y_e,
        device=device,
        batch_size=int(training["batch_size"]),
        delay_override="off",
    )
    state_after_e = _state_digest(model)
    if checkpoint_before_e != state_after_e:
        raise RuntimeError("E9 model state changed during held-out evaluation")

    probability = _probabilities(evaluation["logits"])
    write_trial_predictions(
        run_dir,
        logits=evaluation["logits"],
        probabilities=probability,
        pred=evaluation["pred"],
        label=evaluation["labels"],
        subject=[row["subject"] for row in metadata_e],
        session="E",
        run=[row["run"] for row in metadata_e],
        trial_id=[row["trial_id"] for row in metadata_e],
        seed=seed,
        model=f"v8_e9_{variant}",
    )
    np.savez_compressed(
        run_dir / "prefix_predictions.npz",
        logits=evaluation["prefix_logits"].astype(np.float32),
        labels=evaluation["labels"].astype(np.int64),
        endpoint_seconds=np.asarray(model_config["endpoint_seconds"], dtype=np.float32),
    )
    write_csv(run_dir / "history.csv", fit.history)
    write_json(
        run_dir / "data_access_manifest.json",
        {
            "stage": "bci2a_evaluation",
            "train_session": "T",
            "evaluation_session": "E",
            "session_e_first_used_after_checkpoint_sha256": checkpoint_before_e,
            "session_e_checkpoint_selection": False,
            "session_e_gradient_updates": False,
            "openbmi_s2_accessed": False,
        },
    )
    write_json(
        run_dir / "state_audit.json",
        {
            "checkpoint_before_session_e": checkpoint_before_e,
            "state_after_session_e": state_after_e,
            "identical": True,
        },
    )
    metrics = {
        "status": "completed",
        "stage": "E9",
        "protocol": config["protocol"],
        "variant": variant,
        "subject": int(subject),
        "seed": int(seed),
        "accuracy": evaluation["accuracy"],
        "balanced_accuracy": evaluation["balanced_accuracy"],
        "kappa": evaluation["kappa"],
        "macro_f1": evaluation["macro_f1"],
        "endpoint_metrics": evaluation["endpoint_metrics"],
        "binary_spike_rate": evaluation["binary_spike_rate"],
        "final_activity_nonzero_rate": evaluation["final_activity_nonzero_rate"],
        "parameters": model.parameter_count,
        "trainable_parameters": model.trainable_parameter_count,
        "final_epoch": final_epoch,
        "optimizer_steps": fit.optimizer_steps,
        "train_seconds": fit.elapsed_seconds,
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


def _e9_subject_worker_command(
    *,
    args: argparse.Namespace,
    output: Path,
    subject: int,
    variants: list[str],
    seeds: list[int],
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--data",
        str(Path(args.data).resolve()),
        "--freeze",
        str(Path(args.freeze).resolve()),
        "--e6",
        str(Path(args.e6).resolve()),
        "--e6-audit",
        str(Path(args.e6_audit).resolve()),
        "--output",
        str(output),
        "--config",
        str(Path(args.config).resolve()),
        "--device",
        str(args.device),
        "--worker-subject",
        str(subject),
        "--worker-variants",
        ",".join(variants),
        "--worker-seeds",
        ",".join(str(seed) for seed in seeds),
    ]


def _run_e9_subject_worker(
    command: list[str], result_paths: list[Path]
) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    worker_threads = int(environment.get("DPC_SNN_E9_WORKER_THREADS", "8"))
    if worker_threads < 1 or worker_threads > 32:
        raise ValueError("DPC_SNN_E9_WORKER_THREADS must be between 1 and 32")
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
            "E9 subject worker failed with code "
            f"{completed.returncode}: {' '.join(command)}\n{completed.stderr[-8000:]}"
        )
    missing = [str(path) for path in result_paths if not path.is_file()]
    if missing:
        raise RuntimeError("E9 subject worker missed results: " + ", ".join(missing))
    return [read_json(path) for path in result_paths]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--e6", required=True)
    parser.add_argument("--e6-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default="configs/experiments/v8_e9_frozen_ablation.yaml"
    )
    parser.add_argument("--subjects", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--variants", default="")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-subject", type=int, default=None)
    parser.add_argument("--worker-variants", default="")
    parser.add_argument("--worker-seeds", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()

    configure_cache_env()
    output = ensure_dir(Path(args.output).resolve())
    data_root = Path(args.data).resolve()
    config = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    source_tree = collect_source_tree_manifest(ROOT)
    source_digest = source_tree_digest(source_tree)
    freeze = validate_v8_freeze_manifest(
        Path(args.freeze).resolve(), expected_source_tree_sha256=source_digest
    )
    e6 = Path(args.e6).resolve()
    e6_status = read_json(e6 / "campaign_status.json")
    e6_campaign_sha256 = file_sha256(e6 / "campaign_status.json")
    e6_audit = read_json(Path(args.e6_audit).resolve() / "audit_report.json")
    if (
        e6_status.get("status") != "completed"
        or not bool(e6_status.get("full_registered_contract"))
        or e6_status.get("freeze_sha256") != freeze["combined_sha256"]
        or e6_audit.get("status") != "passed"
        or e6_audit.get("freeze_sha256") != freeze["combined_sha256"]
    ):
        raise RuntimeError("E9 requires the complete passed E6 campaign and audit")

    registry = list(config["retrained_single_variable_ablations"])
    subjects = _csv(args.subjects, int) if args.subjects else list(config["subjects"])
    seeds = _csv(args.seeds, int) if args.seeds else list(config["seeds"])
    variants = _csv(args.variants) if args.variants else registry
    if set(variants) - set(registry):
        raise ValueError("E9 requested an unregistered ablation")
    workers = int(args.workers)
    if workers < 1 or workers > 8:
        raise ValueError("V8 E9 workers must be between 1 and 8")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    if args.worker_subject is not None:
        worker_subject = int(args.worker_subject)
        worker_variants = _csv(args.worker_variants)
        worker_seeds = _csv(args.worker_seeds, int)
        if (
            worker_subject not in config["subjects"]
            or not worker_variants
            or not worker_seeds
            or set(worker_variants).difference(registry)
            or set(worker_seeds).difference(config["seeds"])
        ):
            raise ValueError("invalid E9 subject-worker contract")
        subject_path = _subject_file(data_root, worker_subject)
        data = load_processed_npz(subject_path)
        t_indices = _session_indices(data, "T")
        e_indices = _session_indices(data, "E")
        metadata_t = _metadata(
            data, t_indices, stage="bci2a_evaluation", role="training"
        )
        worker_environment = _environment()
        worker_rows = []
        for variant in worker_variants:
            for seed in worker_seeds:
                worker_rows.append(
                    _run_one(
                        output=output,
                        subject_path=subject_path,
                        data=data,
                        t_indices=t_indices,
                        e_indices=e_indices,
                        metadata_t=metadata_t,
                        freeze=freeze,
                        config=config,
                        source_tree=source_tree,
                        environment=worker_environment,
                        e6_campaign_sha256=e6_campaign_sha256,
                        variant=variant,
                        subject=worker_subject,
                        seed=int(seed),
                        device=args.device,
                    )
                )
        print(json.dumps({"status": "completed", "runs": len(worker_rows)}, indent=2))
        return
    full_contract = (
        subjects == list(config["subjects"])
        and seeds == list(config["seeds"])
        and variants == registry
    )
    if not full_contract and not args.canary:
        raise RuntimeError("partial E9 execution requires --canary")

    write_json(output / "source_tree_manifest.json", source_tree)
    write_json(output / "freeze_manifest.json", freeze)
    (output / "resolved_campaign.yaml").write_text(
        yaml.safe_dump(
            {
                **config,
                "active_subjects": subjects,
                "active_seeds": seeds,
                "active_variants": variants,
                "freeze_sha256": freeze["combined_sha256"],
                "canary": bool(args.canary),
                "parallel_workers": workers,
                "worker_cpu_threads": (
                    int(os.environ.get("DPC_SNN_E9_WORKER_THREADS", "8"))
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
            for variant in variants:
                for seed in seeds:
                    row = _run_one(
                        output=output,
                        subject_path=subject_path,
                        data=data,
                        t_indices=t_indices,
                        e_indices=e_indices,
                        metadata_t=metadata_t,
                        freeze=freeze,
                        config=config,
                        source_tree=source_tree,
                        environment=environment,
                        e6_campaign_sha256=e6_campaign_sha256,
                        variant=variant,
                        subject=int(subject),
                        seed=int(seed),
                        device=args.device,
                    )
                    rows.append(row)
                    write_csv(output / "summary.csv", rows)
                    print(
                        "__V8_E9_RUN_DONE__ "
                        f"variant={variant} subject={subject} seed={seed} "
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
                    / variant
                    / f"subject_{int(subject):02d}"
                    / f"seed_{int(seed)}"
                    / "metrics.json"
                    for variant in variants
                    for seed in seeds
                ]
                command = _e9_subject_worker_command(
                    args=args,
                    output=output,
                    subject=int(subject),
                    variants=variants,
                    seeds=seeds,
                )
                futures[
                    executor.submit(_run_e9_subject_worker, command, result_paths)
                ] = int(subject)
            for future in as_completed(futures):
                rows.extend(future.result())
    rows.sort(key=lambda row: (str(row["variant"]), int(row["subject"]), int(row["seed"])))
    write_csv(output / "summary.csv", rows)
    status = {
        "status": "completed" if full_contract else "canary_completed",
        "stage": "E9",
        "protocol": config["protocol"],
        "subjects": subjects,
        "seeds": seeds,
        "variants": variants,
        "runs": len(rows),
        "full_registered_contract": full_contract,
        "freeze_sha256": freeze["combined_sha256"],
        "reference_e6": str(e6),
        "session_e_accessed": True,
        "session_e_used_for_selection": False,
        "openbmi_s2_accessed": False,
        "parallel_workers": workers,
        "worker_cpu_threads": (
            int(os.environ.get("DPC_SNN_E9_WORKER_THREADS", "8"))
            if workers > 1
            else None
        ),
    }
    write_json(output / "campaign_status.json", status)
    write_run_artifact_manifest(output, required_files=CAMPAIGN_FILES)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
