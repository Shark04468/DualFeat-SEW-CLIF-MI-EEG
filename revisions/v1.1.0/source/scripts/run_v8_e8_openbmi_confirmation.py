#!/usr/bin/env python3
"""Train on OpenBMI S1 and evaluate locked S2 with the frozen V8 protocol."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
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

from dpc_snn.data.openbmi import load_openbmi_subject  # noqa: E402
from dpc_snn.data.v8_openbmi import prepare_openbmi_v8_view  # noqa: E402
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
    validate_v8_external_unlock_manifest,
    validate_v8_freeze_manifest,
    validate_v8_resume_fingerprint,
    write_v8_fingerprint,
)
from dpc_snn.experiments.v8_training import (  # noqa: E402
    V8CachedRates,
    cache_v8_physical_rates,
    fit_v8,
    fit_v8_physical_gain,
    predict_v8,
    seed_v8,
)
from dpc_snn.models.build import build_model  # noqa: E402
from dpc_snn.models.v8_accuracy_first import V8AccuracyFirstModel  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v8_e2_zero_delay import _environment, _fit_kwargs  # noqa: E402
from scripts.run_v8_e3_static_delay import (  # noqa: E402
    _build_delay_model,
    _freeze_delay_only,
    _load_or_fit_prior,
    _load_parent_state,
)


BASE_FILES = (
    "manifest.json",
    "training_fingerprint.json",
    "source_fingerprint.json",
    "resolved_config.yaml",
    "s1_access_manifest.json",
    "s2_access_manifest.json",
    "base_history.csv",
    "history.csv",
    "base.pt",
    "best.pt",
    "last.pt",
    "metrics.json",
    "predictions.npz",
    "predictions.csv",
    "prefix_predictions.npz",
    "state_audit.json",
    "runtime_status.json",
)
DELAY_TRAINING_FILES = (
    "residual_history.csv",
    "full_s1_prior/manifest.json",
    "full_s1_prior/prior.npz",
    "full_s1_prior/evidence.npz",
    "full_s1_prior/fingerprint.json",
    "full_s1_prior/summary.json",
)
DELAY_EVALUATION_FILES = (
    "matched_zero_predictions.npz",
    "matched_zero_predictions.csv",
    "matched_zero_prefix_predictions.npz",
)
DELAY_FILES = DELAY_TRAINING_FILES + DELAY_EVALUATION_FILES
CAMPAIGN_FILES = (
    "manifest.json",
    "campaign_status.json",
    "summary.csv",
    "resolved_campaign.yaml",
    "source_tree_manifest.json",
    "freeze_manifest.json",
    "external_unlock_manifest.json",
)


def _csv(value: str, cast: Any = str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _array_digest(x: np.ndarray, y: np.ndarray, trial_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for array in (np.asarray(x), np.asarray(y)):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    digest.update("\n".join(trial_ids).encode("utf-8"))
    return digest.hexdigest()


def _probabilities(logits: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.from_numpy(logits).float(), dim=1).numpy()


def _state_digest(model: V8AccuracyFirstModel) -> str:
    return sha256_fingerprint(mapping_sha256(model.state_dict()))


def _arm_contract(
    freeze: dict[str, Any], unlock: dict[str, Any], arm: str
) -> tuple[dict[str, Any], int, bool]:
    if arm == "frozen_primary":
        model_config = dict(freeze["architecture"]["model_config"])
        epoch = int(unlock["training"]["primary_final_epoch"])
        delay_enabled = bool(freeze["architecture"]["delay"]["enabled"])
    elif arm == "matched_ann":
        model_config = dict(freeze["architecture"]["matched_ann_control_config"])
        epoch = int(unlock["training"]["matched_ann_final_epoch"])
        delay_enabled = False
    else:
        raise ValueError(f"unknown E8 arm {arm!r}")
    model_config["n_classes"] = 2
    return model_config, epoch, delay_enabled


def _build_base(config: dict[str, Any], seed: int) -> V8AccuracyFirstModel:
    seed_v8(seed)
    model = build_model("v8_accuracy_first", config)
    if not isinstance(model, V8AccuracyFirstModel):
        raise TypeError("E8 model factory returned an unexpected model")
    if model.delay_auxiliary_enabled or model.delay_auxiliary is not None:
        raise RuntimeError("E8 base model silently enabled delay")
    return model


def _training_required(delay_enabled: bool) -> tuple[str, ...]:
    values = (
        "training_fingerprint.json",
        "resolved_config.yaml",
        "s1_access_manifest.json",
        "base_history.csv",
        "history.csv",
        "base.pt",
        "best.pt",
        "last.pt",
        "runtime_status.json",
    )
    if delay_enabled:
        return values + DELAY_TRAINING_FILES
    return values


def _train_one(
    *,
    output: Path,
    freeze: dict[str, Any],
    unlock: dict[str, Any],
    source_tree: dict[str, str],
    environment: dict[str, Any],
    arm: str,
    subject: int,
    seed: int,
    train_rates: V8CachedRates,
    gain: torch.Tensor,
    y_s1: np.ndarray,
    metadata_s1: list[dict[str, Any]],
    s1_manifest: dict[str, Any],
    s1_digest: str,
    device: str,
) -> Path:
    model_config, final_epoch, delay_enabled = _arm_contract(freeze, unlock, arm)
    training = dict(freeze["training"])
    augmentation = dict(freeze["augmentation"])
    scheduler_horizon = int(unlock["training"]["scheduler_horizon"])
    run_seed = int(seed) * 100_003 + int(subject) * 1_009 + (
        0 if arm == "frozen_primary" else 7_000_027
    )
    resolved = {
        "stage": "E8_training",
        "arm": arm,
        "subject": subject,
        "seed": seed,
        "run_seed": run_seed,
        "model": model_config,
        "final_epoch": final_epoch,
        "scheduler_horizon": scheduler_horizon,
        "training": training,
        "augmentation": augmentation,
        "delay": freeze["architecture"]["delay"] if delay_enabled else {"enabled": False},
        "external_unlock_sha256": unlock["combined_sha256"],
    }
    fingerprint = build_v8_run_fingerprint(
        resolved_run_config=resolved,
        source_tree=source_tree,
        data={"openbmi_s1_sha256": s1_digest, "access": s1_manifest},
        split={
            "train_session": "S1",
            "train_trial_ids": [row["trial_id"] for row in metadata_s1],
            "evaluation_session_unopened": "S2",
        },
        augmentation=augmentation,
        prior=(
            freeze["architecture"]["delay"]
            if delay_enabled
            else {"policy": "none", "delay_enabled": False}
        ),
        checkpoint={
            "fixed_epoch": final_epoch,
            "scheduler_horizon": scheduler_horizon,
            "s2_checkpoint_selection": False,
        },
        environment=environment,
    )
    run_dir = ensure_dir(output / arm / f"subject_{subject:02d}" / f"seed_{seed}")
    fingerprint_path = run_dir / "training_fingerprint.json"
    training_required = _training_required(delay_enabled)
    if all((run_dir / name).is_file() for name in training_required):
        validate_v8_resume_fingerprint(fingerprint_path, fingerprint)
        return run_dir
    if fingerprint_path.is_file():
        validate_v8_resume_fingerprint(fingerprint_path, fingerprint)
    else:
        write_v8_fingerprint(fingerprint_path, fingerprint)
        (run_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )
        write_json(run_dir / "s1_access_manifest.json", s1_manifest)
    write_json(
        run_dir / "runtime_status.json",
        {
            "status": "training_s1_s2_unopened",
            "started_at": time.time(),
            "openbmi_s2_accessed": False,
        },
    )
    model = _build_base(model_config, run_seed)
    fit = fit_v8(
        model,
        train_rates,
        y_s1,
        validation_rates=None,
        validation_labels=None,
        device=device,
        seed=run_seed,
        epochs=final_epoch,
        patience=final_epoch,
        minimum_epochs=final_epoch,
        fixed_epoch=final_epoch,
        scheduler_epochs=scheduler_horizon,
        run_label=f"V8-E8:{arm}:S{subject}:seed{seed}:base",
        **_fit_kwargs(training, augmentation),
    )
    model = fit.model
    base_history = fit.history
    residual_history: list[dict[str, Any]] = []
    torch.save(fit.last_state, run_dir / "base.pt")
    if delay_enabled:
        delay_contract = dict(freeze["architecture"]["delay"]["config"])
        delay_model = _build_delay_model(
            model_config, dict(delay_contract["delay"]), seed=run_seed + 17
        )
        _load_parent_state(delay_model, run_dir / "base.pt")
        prior, _ = _load_or_fit_prior(
            run_dir / "full_s1_prior",
            scope="all_openbmi_s1_before_s2",
            trial_ids=[row["trial_id"] for row in metadata_s1],
            rates=train_rates,
            gain=gain,
            model_config=model_config,
            delay_config=dict(delay_contract["delay"]),
            source_sha256=source_tree_digest(source_tree),
            seed=run_seed + 211,
        )
        delay_model.load_fold_delay_prior(**prior)
        _freeze_delay_only(delay_model)
        residual_epoch = int(freeze["architecture"]["delay"]["residual_epoch"])
        if residual_epoch > 0:
            residual_fit = fit_v8(
                delay_model,
                train_rates,
                y_s1,
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
                run_label=f"V8-E8:{arm}:S{subject}:seed{seed}:delay",
                **_fit_kwargs(
                    dict(delay_contract["training"]),
                    dict(delay_contract["augmentation"]),
                ),
            )
            model = residual_fit.model
            residual_history = residual_fit.history
        else:
            model = delay_model
            residual_history = [{"epoch": 0, "status": "registered_exact_null"}]
        write_csv(run_dir / "residual_history.csv", residual_history)
    checkpoint = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    torch.save(checkpoint, run_dir / "best.pt")
    torch.save(checkpoint, run_dir / "last.pt")
    write_csv(run_dir / "base_history.csv", base_history)
    write_csv(run_dir / "history.csv", base_history + residual_history)
    write_json(
        run_dir / "runtime_status.json",
        {
            "status": "training_completed_s2_unopened",
            "completed_at": time.time(),
            "checkpoint_sha256": file_sha256(run_dir / "best.pt"),
            "openbmi_s2_accessed": False,
        },
    )
    return run_dir


def _evaluate_one(
    *,
    run_dir: Path,
    freeze: dict[str, Any],
    unlock: dict[str, Any],
    source_tree: dict[str, str],
    environment: dict[str, Any],
    arm: str,
    subject: int,
    seed: int,
    evaluation_rates: V8CachedRates,
    y_s2: np.ndarray,
    metadata_s2: list[dict[str, Any]],
    s1_digest: str,
    s2_digest: str,
    s2_manifest: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    model_config, final_epoch, delay_enabled = _arm_contract(freeze, unlock, arm)
    final_fingerprint = build_v8_run_fingerprint(
        resolved_run_config=yaml.safe_load(
            (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
        ),
        source_tree=source_tree,
        data={
            "openbmi_s1_sha256": s1_digest,
            "openbmi_s2_sha256": s2_digest,
            "s2_access": s2_manifest,
        },
        split={
            "train_session": "S1",
            "evaluation_session": "S2",
            "evaluation_trial_ids": [row["trial_id"] for row in metadata_s2],
        },
        augmentation=freeze["augmentation"],
        prior=(
            freeze["architecture"]["delay"]
            if delay_enabled
            else {"policy": "none", "delay_enabled": False}
        ),
        checkpoint={
            "training_fingerprint": read_json(run_dir / "training_fingerprint.json")[
                "combined_sha256"
            ],
            "checkpoint_sha256": file_sha256(run_dir / "best.pt"),
            "fixed_epoch": final_epoch,
            "s2_checkpoint_selection": False,
        },
        environment=environment,
    )
    final_path = run_dir / "source_fingerprint.json"
    required = BASE_FILES + (DELAY_FILES if delay_enabled else ())
    if (run_dir / "manifest.json").is_file():
        validate_v8_resume_fingerprint(final_path, final_fingerprint)
        validate_run_artifact_manifest(
            run_dir,
            required_files=required,
            verify_hashes=True,
            verify_prediction_schema=True,
        )
        return read_json(run_dir / "metrics.json")
    if final_path.is_file():
        validate_v8_resume_fingerprint(final_path, final_fingerprint)
    else:
        write_v8_fingerprint(final_path, final_fingerprint)
    model, _, _ = _arm_contract(freeze, unlock, arm)
    if delay_enabled:
        delay = dict(freeze["architecture"]["delay"])
        built = _build_delay_model(model, dict(delay["config"]["delay"]), seed=0)
        override = "full"
    else:
        built = _build_base(model, 0)
        override = "off"
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    built.load_state_dict(state, strict=True)
    before = _state_digest(built)
    evaluation = predict_v8(
        built,
        evaluation_rates,
        y_s2,
        device=device,
        batch_size=int(freeze["training"]["batch_size"]),
        delay_override=override,
    )
    after = _state_digest(built)
    if before != after:
        raise RuntimeError("E8 model state changed while evaluating OpenBMI S2")
    probabilities = _probabilities(evaluation["logits"])
    write_trial_predictions(
        run_dir,
        logits=evaluation["logits"],
        probabilities=probabilities,
        pred=evaluation["pred"],
        label=evaluation["labels"],
        subject=[row["subject"] for row in metadata_s2],
        session="S2",
        run=[row["run"] for row in metadata_s2],
        trial_id=[row["trial_id"] for row in metadata_s2],
        seed=seed,
        model=f"v8_e8_{arm}",
    )
    np.savez_compressed(
        run_dir / "prefix_predictions.npz",
        logits=evaluation["prefix_logits"].astype(np.float32),
        labels=evaluation["labels"].astype(np.int64),
        endpoint_seconds=np.asarray(model_config["endpoint_seconds"], dtype=np.float32),
    )
    matched_zero_accuracy: float | None = None
    if delay_enabled:
        matched_zero = predict_v8(
            built,
            evaluation_rates,
            y_s2,
            device=device,
            batch_size=int(freeze["training"]["batch_size"]),
            delay_override="zero",
        )
        if _state_digest(built) != before:
            raise RuntimeError("E8 matched-zero evaluation changed the frozen model")
        zero_probabilities = _probabilities(matched_zero["logits"])
        write_trial_predictions(
            run_dir,
            logits=matched_zero["logits"],
            probabilities=zero_probabilities,
            pred=matched_zero["pred"],
            label=matched_zero["labels"],
            subject=[row["subject"] for row in metadata_s2],
            session="S2",
            run=[row["run"] for row in metadata_s2],
            trial_id=[row["trial_id"] for row in metadata_s2],
            seed=seed,
            model="v8_e8_matched_zero",
            basename="matched_zero_predictions",
        )
        np.savez_compressed(
            run_dir / "matched_zero_prefix_predictions.npz",
            logits=matched_zero["prefix_logits"].astype(np.float32),
            labels=matched_zero["labels"].astype(np.int64),
            endpoint_seconds=np.asarray(model_config["endpoint_seconds"], dtype=np.float32),
        )
        matched_zero_accuracy = float(matched_zero["accuracy"])
    write_json(run_dir / "s2_access_manifest.json", s2_manifest)
    write_json(
        run_dir / "state_audit.json",
        {"before_s2": before, "after_s2": after, "identical": True, "model_updates": False},
    )
    metrics = {
        "status": "completed",
        "stage": "E8",
        "protocol": "openbmi_s1_train_s2_single_evaluation",
        "arm": arm,
        "subject": subject,
        "seed": seed,
        "accuracy": evaluation["accuracy"],
        "balanced_accuracy": evaluation["balanced_accuracy"],
        "kappa": evaluation["kappa"],
        "macro_f1": evaluation["macro_f1"],
        "endpoint_metrics": evaluation["endpoint_metrics"],
        "binary_spike_rate": evaluation["binary_spike_rate"],
        "final_activity_nonzero_rate": evaluation["final_activity_nonzero_rate"],
        "matched_zero_accuracy": matched_zero_accuracy,
        "full_minus_zero_accuracy": (
            float(evaluation["accuracy"] - matched_zero_accuracy)
            if matched_zero_accuracy is not None
            else None
        ),
        "parameters": built.parameter_count,
        "final_epoch": final_epoch,
        "train_session": "S1",
        "evaluation_session": "S2",
        "s2_checkpoint_selection": False,
        "s2_gradient_updates": False,
        "freeze_sha256": freeze["combined_sha256"],
        "external_unlock_sha256": unlock["combined_sha256"],
        "run_fingerprint": final_fingerprint["combined_sha256"],
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "runtime_status.json",
        {"status": "completed", "completed_at": time.time(), "openbmi_s2_accessed": True},
    )
    write_run_artifact_manifest(run_dir, required_files=required)
    return metrics


def _run_subject_contract(
    *,
    output: Path,
    freeze: dict[str, Any],
    unlock: dict[str, Any],
    source_tree: dict[str, str],
    environment: dict[str, Any],
    subject: int,
    seeds: list[int],
    arms: list[str],
    channels: list[str],
    device: str,
) -> list[dict[str, Any]]:
    # S1 is the only session opened before every subject-specific checkpoint exists.
    s1_raw = load_openbmi_subject(int(subject), sessions=("S1",), resample=1000.0)
    x_s1, y_s1, metadata_s1, s1_manifest = prepare_openbmi_v8_view(
        s1_raw,
        channel_names=channels,
        target_sfreq=250.0,
        role="training",
    )
    s1_digest = _array_digest(x_s1, y_s1, [row["trial_id"] for row in metadata_s1])
    reference_config, _, _ = _arm_contract(freeze, unlock, arms[0])
    reference = _build_base(reference_config, 0)
    gain = fit_v8_physical_gain(reference, x_s1, device=device, batch_size=16)
    train_rates = cache_v8_physical_rates(
        reference, x_s1, device=device, batch_size=16
    )
    run_dirs: dict[tuple[str, int], Path] = {}
    for arm in arms:
        for seed in seeds:
            run_dirs[(arm, int(seed))] = _train_one(
                output=output,
                freeze=freeze,
                unlock=unlock,
                source_tree=source_tree,
                environment=environment,
                arm=arm,
                subject=int(subject),
                seed=int(seed),
                train_rates=train_rates,
                gain=gain,
                y_s1=y_s1,
                metadata_s1=metadata_s1,
                s1_manifest=s1_manifest,
                s1_digest=s1_digest,
                device=device,
            )

    # S2 is opened once only after every requested checkpoint for this subject exists.
    s2_raw = load_openbmi_subject(int(subject), sessions=("S2",), resample=1000.0)
    x_s2, y_s2, metadata_s2, s2_manifest = prepare_openbmi_v8_view(
        s2_raw,
        channel_names=channels,
        target_sfreq=250.0,
        role="evaluation",
    )
    s2_digest = _array_digest(x_s2, y_s2, [row["trial_id"] for row in metadata_s2])
    reference.set_training_gain(gain)
    evaluation_rates = cache_v8_physical_rates(
        reference, x_s2, device=device, batch_size=16
    )
    rows: list[dict[str, Any]] = []
    for arm in arms:
        for seed in seeds:
            row = _evaluate_one(
                run_dir=run_dirs[(arm, int(seed))],
                freeze=freeze,
                unlock=unlock,
                source_tree=source_tree,
                environment=environment,
                arm=arm,
                subject=int(subject),
                seed=int(seed),
                evaluation_rates=evaluation_rates,
                y_s2=y_s2,
                metadata_s2=metadata_s2,
                s1_digest=s1_digest,
                s2_digest=s2_digest,
                s2_manifest=s2_manifest,
                device=device,
            )
            rows.append(row)
            print(
                "__V8_E8_RUN_DONE__ "
                f"arm={arm} subject={subject} seed={seed} "
                f"accuracy={float(row['accuracy']):.6f}",
                flush=True,
            )
    del s1_raw, s2_raw, reference, train_rates, evaluation_rates
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows


def _e8_subject_worker_command(
    *,
    args: argparse.Namespace,
    output: Path,
    subject: int,
    arms: list[str],
    seeds: list[int],
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--freeze",
        str(Path(args.freeze).resolve()),
        "--unlock",
        str(Path(args.unlock).resolve()),
        "--output",
        str(output),
        "--device",
        str(args.device),
        "--worker-subject",
        str(subject),
        "--worker-arms",
        ",".join(arms),
        "--worker-seeds",
        ",".join(str(seed) for seed in seeds),
    ]


def _run_e8_subject_worker(
    command: list[str], result_paths: list[Path]
) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    worker_threads = int(environment.get("DPC_SNN_E8_WORKER_THREADS", "8"))
    if worker_threads < 1 or worker_threads > 32:
        raise ValueError("DPC_SNN_E8_WORKER_THREADS must be between 1 and 32")
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
            "E8 subject worker failed with code "
            f"{completed.returncode}: {' '.join(command)}\n{completed.stderr[-8000:]}"
        )
    missing = [str(path) for path in result_paths if not path.is_file()]
    if missing:
        raise RuntimeError("E8 subject worker missed results: " + ", ".join(missing))
    return [read_json(path) for path in result_paths]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--unlock", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subjects", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--arms", default="")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-subject", type=int, default=None)
    parser.add_argument("--worker-arms", default="")
    parser.add_argument("--worker-seeds", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()

    configure_cache_env()
    output = ensure_dir(Path(args.output).resolve())
    source_tree = collect_source_tree_manifest(ROOT)
    source_digest = source_tree_digest(source_tree)
    freeze = validate_v8_freeze_manifest(
        Path(args.freeze).resolve(), expected_source_tree_sha256=source_digest
    )
    unlock = validate_v8_external_unlock_manifest(
        Path(args.unlock).resolve(),
        expected_source_tree_sha256=source_digest,
        expected_parent_freeze_sha256=freeze["combined_sha256"],
    )
    subjects = (
        _csv(args.subjects, int)
        if args.subjects
        else list(unlock["dataset"]["confirmatory_subjects"])
    )
    seeds = _csv(args.seeds, int) if args.seeds else list(unlock["dataset"]["seeds"])
    arms = _csv(args.arms) if args.arms else list(unlock["analysis_plan"]["arms"])
    if freeze["architecture"]["primary_variant"] == "ann_residual":
        arms = [arm for arm in arms if arm != "matched_ann"]
    workers = int(args.workers)
    if workers < 1 or workers > 8:
        raise ValueError("V8 E8 workers must be between 1 and 8")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    channels = list(unlock["architecture_adaptation"]["ordered_channels"])
    if args.worker_subject is not None:
        worker_subject = int(args.worker_subject)
        worker_arms = _csv(args.worker_arms)
        worker_seeds = _csv(args.worker_seeds, int)
        allowed_arms = (
            ["frozen_primary"]
            if freeze["architecture"]["primary_variant"] == "ann_residual"
            else list(unlock["analysis_plan"]["arms"])
        )
        if (
            worker_subject not in unlock["dataset"]["confirmatory_subjects"]
            or not worker_arms
            or not worker_seeds
            or set(worker_arms).difference(allowed_arms)
            or set(worker_seeds).difference(unlock["dataset"]["seeds"])
        ):
            raise ValueError("invalid E8 subject-worker contract")
        worker_rows = _run_subject_contract(
            output=output,
            freeze=freeze,
            unlock=unlock,
            source_tree=source_tree,
            environment=_environment(),
            subject=worker_subject,
            seeds=worker_seeds,
            arms=worker_arms,
            channels=channels,
            device=args.device,
        )
        print(json.dumps({"status": "completed", "runs": len(worker_rows)}, indent=2))
        return
    full_contract = (
        subjects == list(unlock["dataset"]["confirmatory_subjects"])
        and seeds == list(unlock["dataset"]["seeds"])
        and arms
        == (
            ["frozen_primary"]
            if freeze["architecture"]["primary_variant"] == "ann_residual"
            else list(unlock["analysis_plan"]["arms"])
        )
    )
    if not full_contract and not args.canary:
        raise RuntimeError("partial E8 execution requires --canary")
    environment = _environment()
    write_json(output / "source_tree_manifest.json", source_tree)
    write_json(output / "freeze_manifest.json", freeze)
    write_json(output / "external_unlock_manifest.json", unlock)
    (output / "resolved_campaign.yaml").write_text(
        yaml.safe_dump(
            {
                "stage": "E8",
                "subjects": subjects,
                "seeds": seeds,
                "arms": arms,
                "freeze_sha256": freeze["combined_sha256"],
                "external_unlock_sha256": unlock["combined_sha256"],
                "canary": bool(args.canary),
                "parallel_workers": workers,
                "worker_cpu_threads": (
                    int(os.environ.get("DPC_SNN_E8_WORKER_THREADS", "8"))
                    if workers > 1
                    else None
                ),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    if workers == 1:
        for subject in subjects:
            rows.extend(
                _run_subject_contract(
                    output=output,
                    freeze=freeze,
                    unlock=unlock,
                    source_tree=source_tree,
                    environment=environment,
                    subject=int(subject),
                    seeds=seeds,
                    arms=arms,
                    channels=channels,
                    device=args.device,
                )
            )
            write_csv(output / "summary.csv", rows)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for subject in subjects:
                result_paths = [
                    output
                    / arm
                    / f"subject_{int(subject):02d}"
                    / f"seed_{int(seed)}"
                    / "metrics.json"
                    for arm in arms
                    for seed in seeds
                ]
                command = _e8_subject_worker_command(
                    args=args,
                    output=output,
                    subject=int(subject),
                    arms=arms,
                    seeds=seeds,
                )
                futures[
                    executor.submit(_run_e8_subject_worker, command, result_paths)
                ] = int(subject)
            for future in as_completed(futures):
                rows.extend(future.result())
    rows = sorted(rows, key=lambda row: (row["arm"], row["subject"], row["seed"]))
    write_csv(output / "summary.csv", rows)
    status = {
        "status": "completed" if full_contract else "canary_completed",
        "stage": "E8",
        "protocol": "openbmi_s1_train_s2_single_evaluation",
        "subjects": subjects,
        "seeds": seeds,
        "arms": arms,
        "runs": len(rows),
        "full_registered_contract": full_contract,
        "freeze_sha256": freeze["combined_sha256"],
        "external_unlock_sha256": unlock["combined_sha256"],
        "openbmi_s2_accessed": True,
        "s2_used_for_selection": False,
        "s2_gradient_updates": False,
        "parallel_workers": workers,
        "worker_cpu_threads": (
            int(os.environ.get("DPC_SNN_E8_WORKER_THREADS", "8"))
            if workers > 1
            else None
        ),
    }
    write_json(output / "campaign_status.json", status)
    write_run_artifact_manifest(output, required_files=CAMPAIGN_FILES)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
