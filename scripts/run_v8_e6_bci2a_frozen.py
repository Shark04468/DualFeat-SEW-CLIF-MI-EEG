#!/usr/bin/env python3
"""Run the frozen V8 BCI2a Session-T to Session-E benchmark exactly once."""

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
    validate_trial_metadata,
    write_run_artifact_manifest,
    write_trial_predictions,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    assert_v8_data_access,
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
    seed_v8,
)
from dpc_snn.models.build import build_model  # noqa: E402
from dpc_snn.models.v8_accuracy_first import V8AccuracyFirstModel  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v8_e2_zero_delay import _environment, _fit_kwargs, _subject_file  # noqa: E402
from scripts.run_v8_e3_static_delay import (  # noqa: E402
    _build_delay_model,
    _freeze_delay_only,
    _load_or_fit_prior,
    _load_parent_state,
)


BASE_RUN_FILES = (
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
DELAY_RUN_FILES = (
    "base_history.csv",
    "base.pt",
    "residual_history.csv",
    "matched_zero_predictions.npz",
    "matched_zero_predictions.csv",
    "matched_zero_prefix_predictions.npz",
    "full_session_t_prior/manifest.json",
    "full_session_t_prior/prior.npz",
    "full_session_t_prior/evidence.npz",
    "full_session_t_prior/fingerprint.json",
    "full_session_t_prior/summary.json",
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


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else value


def _session_indices(data: dict[str, Any], session: str) -> np.ndarray:
    indices = np.flatnonzero(np.asarray(data["session"]).astype(str) == str(session))
    if indices.size != 288:
        raise RuntimeError(f"expected 288 BCI2a Session-{session} trials, got {indices.size}")
    return indices


def _metadata(
    data: dict[str, Any], indices: np.ndarray, *, stage: str, role: str
) -> list[dict[str, Any]]:
    labels = np.asarray(data["y"], dtype=np.int64)[indices]
    sessions = np.asarray(data["session"]).astype(str)[indices]
    rows = validate_trial_metadata(
        [
            {
                "dataset": str(_scalar(data.get("dataset_name", "bci2a"))),
                "subject": str(np.asarray(data["subject"]).astype(str)[index]),
                "session": str(np.asarray(data["session"]).astype(str)[index]),
                "run": str(np.asarray(data["run"]).astype(str)[index]),
                "trial_id": str(np.asarray(data["trial_id"]).astype(str)[index]),
                "class": int(np.asarray(data["y"])[index]),
                "sfreq": float(_scalar(data["sfreq"])),
                "ch_names": [str(name) for name in data["ch_names"]],
                "epoch_tmin": float(_scalar(data["epoch_tmin"])),
                "epoch_tmax": float(_scalar(data["epoch_tmax"])),
            }
            for index in indices
        ],
        allowed_sessions=("T", "E"),
    )
    assert_v8_data_access(rows, stage=stage, role=role)
    if len(set(row["trial_id"] for row in rows)) != len(rows):
        raise RuntimeError(f"Session-{sessions[0]} trial identifiers are not unique")
    classes, counts = np.unique(labels, return_counts=True)
    if classes.tolist() != [0, 1, 2, 3] or len(set(counts.tolist())) != 1:
        raise RuntimeError(f"Session-{sessions[0]} is not balanced four-class data")
    return rows


def _build_model(config: dict[str, Any], *, seed: int) -> V8AccuracyFirstModel:
    seed_v8(seed)
    model = build_model("v8_accuracy_first", config)
    if not isinstance(model, V8AccuracyFirstModel):
        raise TypeError("frozen V8 factory returned an unexpected model")
    if model.delay_auxiliary_enabled or model.delay_auxiliary is not None:
        raise RuntimeError("base E6 model must not silently enable delay")
    return model


def _probabilities(logits: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.from_numpy(logits).float(), dim=1).numpy()


def _required_files(delay_enabled: bool) -> tuple[str, ...]:
    return BASE_RUN_FILES + (DELAY_RUN_FILES if delay_enabled else ())


def _state_digest(model: V8AccuracyFirstModel) -> str:
    return sha256_fingerprint(mapping_sha256(model.state_dict()))


def _resolved_variant(
    freeze: dict[str, Any], variant: str
) -> tuple[dict[str, Any], int, bool]:
    architecture = freeze["architecture"]
    checkpoint = freeze["checkpoint_rule"]
    if variant == "frozen_primary":
        return (
            dict(architecture["model_config"]),
            int(checkpoint["final_epoch"]),
            bool(architecture["delay"]["enabled"]),
        )
    if variant == "matched_ann":
        return (
            dict(architecture["matched_ann_control_config"]),
            int(checkpoint["matched_ann_final_epoch"]),
            False,
        )
    raise ValueError(f"unknown frozen E6 variant {variant!r}")


def _run_one(
    *,
    output: Path,
    subject_path: Path,
    data: dict[str, Any],
    t_indices: np.ndarray,
    e_indices: np.ndarray,
    metadata_t: list[dict[str, Any]],
    freeze: dict[str, Any],
    source_tree: dict[str, str],
    environment: dict[str, Any],
    variant: str,
    subject: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    model_config, final_epoch, delay_enabled = _resolved_variant(freeze, variant)
    if variant == "matched_ann" and freeze["architecture"]["primary_variant"] == "ann_residual":
        raise RuntimeError("matched ANN duplicates an ANN primary and must be omitted")
    training = dict(freeze["training"])
    augmentation = dict(freeze["augmentation"])
    scheduler_horizon = int(freeze["checkpoint_rule"]["scheduler_horizon"])
    run_seed = int(seed) * 100_003 + int(subject) * 1_009 + (
        0 if variant == "frozen_primary" else 7_000_027
    )
    resolved = {
        "stage": "E6",
        "protocol": "bci2a_session_t_train_session_e_single_evaluation",
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
        "delay": freeze["architecture"]["delay"] if delay_enabled else {"enabled": False, "mode": "off"},
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
        prior=(
            freeze["architecture"]["delay"]
            if delay_enabled
            else {"policy": "none", "delay_enabled": False}
        ),
        checkpoint={
            "freeze_sha256": freeze["combined_sha256"],
            "fixed_epoch": final_epoch,
            "scheduler_horizon": scheduler_horizon,
            "session_e_checkpoint_selection": False,
        },
        environment=environment,
    )
    run_dir = ensure_dir(
        output / variant / f"subject_{subject:02d}" / f"seed_{seed}"
    )
    required = _required_files(delay_enabled)
    fingerprint_path = run_dir / "source_fingerprint.json"
    if (run_dir / "manifest.json").is_file():
        validate_v8_resume_fingerprint(fingerprint_path, fingerprint)
        validate_run_artifact_manifest(
            run_dir,
            required_files=required,
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
    base_model = _build_model(model_config, seed=run_seed)
    gain = fit_v8_physical_gain(base_model, x_t, device=device, batch_size=16)
    train_rates = cache_v8_physical_rates(base_model, x_t, device=device, batch_size=16)
    fit = fit_v8(
        base_model,
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
        run_label=f"V8-E6:{variant}:S{subject}:seed{seed}:base",
        **_fit_kwargs(training, augmentation),
    )
    model = fit.model
    base_history = fit.history
    residual_history: list[dict[str, Any]] = []
    base_seconds = float(fit.elapsed_seconds)
    residual_seconds = 0.0
    optimizer_steps = int(fit.optimizer_steps)
    prior_summary: dict[str, Any] | None = None
    torch.save(fit.last_state, run_dir / "base.pt") if delay_enabled else None

    if delay_enabled:
        delay_frozen = freeze["architecture"]["delay"]
        delay_contract = dict(delay_frozen["config"])
        delay_model = _build_delay_model(
            model_config, dict(delay_contract["delay"]), seed=run_seed + 17
        )
        _load_parent_state(delay_model, run_dir / "base.pt")
        prior, prior_summary = _load_or_fit_prior(
            run_dir / "full_session_t_prior",
            scope="all_session_t_before_single_session_e_evaluation",
            trial_ids=[row["trial_id"] for row in metadata_t],
            rates=train_rates,
            gain=gain,
            model_config=model_config,
            delay_config=dict(delay_contract["delay"]),
            source_sha256=source_tree_digest(source_tree),
            seed=run_seed + 211,
        )
        delay_model.load_fold_delay_prior(**prior)
        _freeze_delay_only(delay_model)
        residual_epoch = int(delay_frozen["residual_epoch"])
        if residual_epoch > 0:
            residual_fit = fit_v8(
                delay_model,
                train_rates,
                y_t,
                validation_rates=None,
                validation_labels=None,
                device=device,
                seed=run_seed + 17,
                epochs=residual_epoch,
                patience=residual_epoch,
                minimum_epochs=residual_epoch,
                fixed_epoch=residual_epoch,
                scheduler_epochs=int(delay_contract["selection"]["max_residual_epochs"]),
                delay_override="full",
                run_label=f"V8-E6:{variant}:S{subject}:seed{seed}:delay",
                **_fit_kwargs(
                    dict(delay_contract["training"]),
                    dict(delay_contract["augmentation"]),
                ),
            )
            model = residual_fit.model
            residual_history = residual_fit.history
            residual_seconds = float(residual_fit.elapsed_seconds)
            optimizer_steps += int(residual_fit.optimizer_steps)
        else:
            model = delay_model.to(device)
            residual_history = [{"epoch": 0, "status": "registered_exact_null"}]

    # Held-out labels are first materialized only after the fixed checkpoint exists.
    state_before_e = _state_digest(model)
    checkpoint = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    torch.save(checkpoint, run_dir / "best.pt")
    torch.save(checkpoint, run_dir / "last.pt")
    metadata_e = _metadata(data, e_indices, stage="bci2a_evaluation", role="evaluation")
    x_e = np.asarray(data["X"], dtype=np.float32)[e_indices]
    y_e = np.asarray(data["y"], dtype=np.int64)[e_indices]
    model.set_training_gain(gain)
    evaluation_rates = cache_v8_physical_rates(
        model, x_e, device=device, batch_size=16
    )
    evaluation = predict_v8(
        model,
        evaluation_rates,
        y_e,
        device=device,
        batch_size=int(training["batch_size"]),
        delay_override="full" if delay_enabled else "off",
    )
    state_after_e = _state_digest(model)
    if state_before_e != state_after_e:
        raise RuntimeError("frozen model state changed while evaluating Session E")

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
        model=f"v8_e6_{variant}",
    )
    np.savez_compressed(
        run_dir / "prefix_predictions.npz",
        logits=evaluation["prefix_logits"].astype(np.float32),
        labels=evaluation["labels"].astype(np.int64),
        endpoint_seconds=np.asarray(model_config["endpoint_seconds"], dtype=np.float32),
    )

    matched_zero_metrics: dict[str, Any] | None = None
    if delay_enabled:
        matched_zero = predict_v8(
            model,
            evaluation_rates,
            y_e,
            device=device,
            batch_size=int(training["batch_size"]),
            delay_override="zero",
        )
        if _state_digest(model) != state_before_e:
            raise RuntimeError("matched-zero evaluation changed the frozen model state")
        zero_probability = _probabilities(matched_zero["logits"])
        write_trial_predictions(
            run_dir,
            logits=matched_zero["logits"],
            probabilities=zero_probability,
            pred=matched_zero["pred"],
            label=matched_zero["labels"],
            subject=[row["subject"] for row in metadata_e],
            session="E",
            run=[row["run"] for row in metadata_e],
            trial_id=[row["trial_id"] for row in metadata_e],
            seed=seed,
            model="v8_e6_matched_zero",
            basename="matched_zero_predictions",
        )
        np.savez_compressed(
            run_dir / "matched_zero_prefix_predictions.npz",
            logits=matched_zero["prefix_logits"].astype(np.float32),
            labels=matched_zero["labels"].astype(np.int64),
            endpoint_seconds=np.asarray(model_config["endpoint_seconds"], dtype=np.float32),
        )
        matched_zero_metrics = classification_metrics(
            matched_zero["labels"], matched_zero["pred"], n_classes=4
        )

    write_csv(run_dir / "history.csv", base_history + residual_history)
    if delay_enabled:
        write_csv(run_dir / "base_history.csv", base_history)
        write_csv(run_dir / "residual_history.csv", residual_history)
    access_manifest = {
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
    write_json(run_dir / "data_access_manifest.json", access_manifest)
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
        "stage": "E6",
        "protocol": "bci2a_session_t_train_session_e_single_evaluation",
        "variant": variant,
        "primary_variant": freeze["architecture"]["primary_variant"],
        "subject": int(subject),
        "seed": int(seed),
        "accuracy": evaluation["accuracy"],
        "balanced_accuracy": evaluation["balanced_accuracy"],
        "kappa": evaluation["kappa"],
        "macro_f1": evaluation["macro_f1"],
        "endpoint_metrics": evaluation["endpoint_metrics"],
        "binary_spike_rate": evaluation["binary_spike_rate"],
        "final_activity_nonzero_rate": evaluation["final_activity_nonzero_rate"],
        "final_activity_absolute_mean": evaluation["final_activity_absolute_mean"],
        "parameters": model.parameter_count,
        "trainable_parameters": model.trainable_parameter_count,
        "final_epoch": final_epoch,
        "delay_enabled": delay_enabled,
        "delay_prior_routes": (
            prior_summary.get("selected_sparse_routes") if prior_summary else None
        ),
        "matched_zero_accuracy": (
            matched_zero_metrics["accuracy"] if matched_zero_metrics else None
        ),
        "full_minus_zero_accuracy": (
            float(evaluation["accuracy"] - matched_zero_metrics["accuracy"])
            if matched_zero_metrics
            else None
        ),
        "optimizer_steps": optimizer_steps,
        "train_seconds": base_seconds + residual_seconds,
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
    write_run_artifact_manifest(run_dir, required_files=required)
    return metrics


def _subject_worker_command(
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


def _run_subject_worker(
    command: list[str], result_paths: list[Path]
) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    worker_threads = int(environment.get("DPC_SNN_E6_WORKER_THREADS", "8"))
    if worker_threads < 1 or worker_threads > 32:
        raise ValueError("DPC_SNN_E6_WORKER_THREADS must be between 1 and 32")
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
            "E6 subject worker failed with code "
            f"{completed.returncode}: {' '.join(command)}\n{completed.stderr[-8000:]}"
        )
    missing = [str(path) for path in result_paths if not path.is_file()]
    if missing:
        raise RuntimeError("E6 subject worker missed results: " + ", ".join(missing))
    return [read_json(path) for path in result_paths]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default="configs/experiments/v8_e6_bci2a_frozen.yaml"
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
    if config.get("stage") != "bci2a_evaluation":
        raise RuntimeError("E6 config has the wrong stage")
    source_tree = collect_source_tree_manifest(ROOT)
    source_digest = source_tree_digest(source_tree)
    freeze = validate_v8_freeze_manifest(
        Path(args.freeze).resolve(), expected_source_tree_sha256=source_digest
    )
    subjects = _csv(args.subjects, int) if args.subjects else list(config["subjects"])
    seeds = _csv(args.seeds, int) if args.seeds else list(config["seeds"])
    variants = _csv(args.variants) if args.variants else list(config["variants"])
    if freeze["architecture"]["primary_variant"] == "ann_residual":
        variants = [name for name in variants if name != "matched_ann"]
    unknown = sorted(set(variants) - {"frozen_primary", "matched_ann"})
    if unknown:
        raise ValueError(f"unknown E6 variants: {unknown}")
    workers = int(args.workers)
    if workers < 1 or workers > 8:
        raise ValueError("V8 E6 workers must be between 1 and 8")
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
            or set(worker_variants).difference(variants)
            or set(worker_seeds).difference(config["seeds"])
        ):
            raise ValueError("invalid E6 subject-worker contract")
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
                        source_tree=source_tree,
                        environment=worker_environment,
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
        and variants
        == (
            ["frozen_primary"]
            if freeze["architecture"]["primary_variant"] == "ann_residual"
            else list(config["variants"])
        )
    )
    if not full_contract and not args.canary:
        raise RuntimeError("partial E6 execution requires --canary")
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
                    int(os.environ.get("DPC_SNN_E6_WORKER_THREADS", "8"))
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
            # E metadata and labels are intentionally not materialized here.
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
                        source_tree=source_tree,
                        environment=environment,
                        variant=variant,
                        subject=int(subject),
                        seed=int(seed),
                        device=args.device,
                    )
                    rows.append(row)
                    write_csv(output / "summary.csv", rows)
                    print(
                        "__V8_E6_RUN_DONE__ "
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
                command = _subject_worker_command(
                    args=args,
                    output=output,
                    subject=int(subject),
                    variants=variants,
                    seeds=seeds,
                )
                futures[executor.submit(
                    _run_subject_worker, command, result_paths
                )] = int(subject)
            for future in as_completed(futures):
                rows.extend(future.result())
    rows = sorted(
        rows, key=lambda row: (str(row["variant"]), int(row["subject"]), int(row["seed"]))
    )
    write_csv(output / "summary.csv", rows)
    status = {
        "status": "completed" if full_contract else "canary_completed",
        "stage": "E6",
        "protocol": config["protocol"],
        "subjects": subjects,
        "seeds": seeds,
        "variants": variants,
        "runs": len(rows),
        "full_registered_contract": full_contract,
        "freeze_sha256": freeze["combined_sha256"],
        "session_e_accessed": True,
        "session_e_used_for_selection": False,
        "openbmi_s2_accessed": False,
        "parallel_workers": workers,
        "worker_cpu_threads": (
            int(os.environ.get("DPC_SNN_E6_WORKER_THREADS", "8"))
            if workers > 1
            else None
        ),
    }
    write_json(output / "campaign_status.json", status)
    write_run_artifact_manifest(output, required_files=CAMPAIGN_FILES)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
