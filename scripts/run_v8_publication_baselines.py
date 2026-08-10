#!/usr/bin/env python3
"""Run equal-budget post-hoc baselines on complete BCI2a and OpenBMI protocols."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.analysis.v8_utility import multiclass_calibration_metrics  # noqa: E402
from dpc_snn.baselines.neural import (  # noqa: E402
    SOURCE_LOCKS,
    build_v62_neural_baseline,
    verify_official_source_locks,
)
from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.data.openbmi import load_openbmi_subject  # noqa: E402
from dpc_snn.data.v8_openbmi import prepare_openbmi_v8_view  # noqa: E402
from dpc_snn.experiments.v62_baselines import (  # noqa: E402
    BASELINE_OPTIMIZERS,
    FixedGain,
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
    validate_v8_resume_fingerprint,
    write_v8_fingerprint,
)
from dpc_snn.experiments.v8_publication_baselines import (  # noqa: E402
    FREQUENCY_OCCLUSIONS_HZ,
    PUBLICATION_BASELINES,
    REGION_CHANNELS,
    array_sha256,
    fft_band_occlusion,
    input_gradient_channel_saliency,
    softmax_probabilities,
    validate_region_partition,
    zero_reference_region,
)
from dpc_snn.utils.io import (  # noqa: E402
    ensure_dir,
    read_json,
    save_npz,
    write_csv,
    write_json,
)
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402


TRAIN_FILES = (
    "manifest.json",
    "source_fingerprint.json",
    "resolved_config.yaml",
    "training_data_manifest.json",
    "gain.npz",
    "history.csv",
    "checkpoint.pt",
    "training_metrics.json",
    "state_audit.json",
    "runtime_status.json",
)
EVALUATION_FILES = (
    "manifest.json",
    "source_fingerprint.json",
    "resolved_config.yaml",
    "evaluation_data_manifest.json",
    "metrics.json",
    "predictions.npz",
    "predictions.csv",
    "perturbations.npz",
    "perturbation_metrics.csv",
    "saliency.npz",
    "state_audit.json",
    "runtime_status.json",
)


def _csv_values(value: str, cast: Any = str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


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
        "threads": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def _external_source_manifest(source_root: Path) -> dict[str, str]:
    allowed = {".py", ".txt", ".yaml", ".yml", ".json"}
    manifest: dict[str, str] = {}
    for source_name in SOURCE_LOCKS:
        base = source_root / source_name
        if not base.is_dir():
            raise FileNotFoundError(f"missing pinned source tree: {base}")
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in allowed:
                continue
            relative = path.relative_to(base)
            if any(part in {".git", "__pycache__"} for part in relative.parts):
                continue
            manifest[f"external/{source_name}/{relative.as_posix()}"] = file_sha256(path)
    if not manifest:
        raise RuntimeError("external baseline source manifest is empty")
    return manifest


def _state_digest(model: torch.nn.Module) -> str:
    return sha256_fingerprint(mapping_sha256(model.state_dict()))


def _atomic_torch_save(payload: Any, path: Path) -> None:
    ensure_dir(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _subject_file(data_root: Path, subject: int) -> Path:
    candidates = (data_root / f"A{subject:02d}.npz", data_root / f"A{subject:02d}_all.npz")
    present = [path for path in candidates if path.is_file()]
    if len(present) == 1:
        return present[0]
    matches = sorted(data_root.glob(f"A{subject:02d}*.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"cannot uniquely resolve subject {subject} under {data_root}")
    return matches[0]


def _align_channels(
    x: np.ndarray, source_names: list[str] | None, requested_names: list[str]
) -> tuple[np.ndarray, list[int]]:
    if source_names is None:
        raise ValueError("publication baselines require explicit EEG channel names")
    normalized = {str(name).strip().upper(): index for index, name in enumerate(source_names)}
    if len(normalized) != len(source_names):
        raise ValueError("dataset channel names are not unique")
    missing = [name for name in requested_names if name.strip().upper() not in normalized]
    if missing:
        raise ValueError(f"dataset is missing frozen channels: {missing}")
    indices = [normalized[name.strip().upper()] for name in requested_names]
    return np.ascontiguousarray(np.asarray(x)[:, indices], dtype=np.float32), indices


def _bci2a_view(
    *,
    root: Path,
    subject: int,
    session: str,
    role: str,
    channel_names: list[str],
    expected_trials: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    path = _subject_file(root, subject)
    data = load_processed_npz(path)
    sessions = np.asarray(data["session"]).astype(str)
    if set(sessions.tolist()) != {session}:
        raise ValueError(f"{path} must contain only Session {session}, got {sorted(set(sessions))}")
    x, indices = _align_channels(
        np.asarray(data["X"], dtype=np.float32), data.get("ch_names"), channel_names
    )
    y = np.asarray(data["y"], dtype=np.int64)
    if y.size != expected_trials or x.shape[0] != expected_trials:
        raise ValueError(
            f"BCI2a Subject {subject} Session {session} has {y.size} trials; "
            f"expected {expected_trials}"
        )
    sfreq = float(data["sfreq"])
    epoch_tmin = float(data.get("epoch_tmin", -1.0))
    epoch_tmax = epoch_tmin + x.shape[-1] / sfreq
    runs = np.asarray(
        data.get("run", [f"{session}_run_{index // 48 + 1}" for index in range(y.size)])
    ).astype(str)
    trial_ids = np.asarray(
        data.get(
            "trial_id",
            [f"BCI2a-S{subject:02d}-{session}-trial{index:03d}" for index in range(y.size)],
        )
    ).astype(str)
    rows = validate_trial_metadata(
        [
            {
                "dataset": "bci2a",
                "subject": str(subject),
                "session": session,
                "run": runs[index],
                "trial_id": trial_ids[index],
                "class": int(y[index]),
                "sfreq": sfreq,
                "ch_names": channel_names,
                "epoch_tmin": epoch_tmin,
                "epoch_tmax": epoch_tmax,
            }
            for index in range(y.size)
        ],
        allowed_sessions=("T", "E"),
    )
    assert_v8_data_access(
        rows,
        stage="bci2a_evaluation",
        role="evaluation" if role == "evaluation" else "training",
    )
    manifest = {
        "dataset": "bci2a",
        "subject": subject,
        "session": session,
        "role": role,
        "trials": int(y.size),
        "sfreq": sfreq,
        "epoch_tmin": epoch_tmin,
        "epoch_tmax": epoch_tmax,
        "channel_names": channel_names,
        "source_channel_indices": indices,
        "source_file": path.name,
        "source_file_sha256": file_sha256(path),
        "signal_sha256": array_sha256(x),
        "label_sha256": array_sha256(y),
        "trial_ids_sha256": sha256_fingerprint(trial_ids.tolist()),
    }
    identity = {
        "source_file": path.name,
        "source_file_sha256": manifest["source_file_sha256"],
        "signal_sha256": manifest["signal_sha256"],
        "label_sha256": manifest["label_sha256"],
    }
    return x, y, rows, manifest, identity


def _openbmi_view(
    *,
    subject: int,
    session: str,
    role: str,
    dataset_config: dict[str, Any],
    channel_names: list[str],
    expected_trials: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    raw = load_openbmi_subject(
        subject,
        sessions=(session,),
        resample=float(dataset_config["source_sfreq"]),
    )
    x, y, rows, adapter_manifest = prepare_openbmi_v8_view(
        raw,
        channel_names=channel_names,
        input_adapter=dataset_config["channel_adapter"],
        target_sfreq=float(dataset_config["target_sfreq"]),
        stage="openbmi_confirmation",
        role="evaluation" if role == "evaluation" else "training",
    )
    if y.size != expected_trials:
        raise ValueError(
            f"OpenBMI Subject {subject} {session} has {y.size} trials; expected {expected_trials}"
        )
    manifest = {
        **adapter_manifest,
        "subject": subject,
        "signal_sha256": array_sha256(x),
        "label_sha256": array_sha256(y),
        "trial_ids_sha256": sha256_fingerprint([row["trial_id"] for row in rows]),
    }
    identity = {
        "dataset": "OpenBMI_Lee2019_MI",
        "subject": subject,
        "session": session,
        "signal_sha256": manifest["signal_sha256"],
        "label_sha256": manifest["label_sha256"],
        "adapter_sha256": sha256_fingerprint(adapter_manifest),
    }
    return x, y, rows, manifest, identity


def _load_view(
    args: argparse.Namespace,
    config: dict[str, Any],
    *,
    subject: int,
    role: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    dataset = args.dataset
    dataset_config = dict(config["datasets"][dataset])
    channel_names = [str(value) for value in config["channel_names"]]
    if role == "training":
        session = str(dataset_config["train_session"])
        expected = int(dataset_config["expected_train_trials"])
    else:
        session = str(dataset_config["evaluation_session"])
        expected = int(dataset_config["expected_evaluation_trials"])
    if dataset == "bci2a":
        root = Path(args.bci2a_train_root if role == "training" else args.bci2a_eval_root)
        return _bci2a_view(
            root=root,
            subject=subject,
            session=session,
            role=role,
            channel_names=channel_names,
            expected_trials=expected,
        )
    return _openbmi_view(
        subject=subject,
        session=session,
        role=role,
        dataset_config=dataset_config,
        channel_names=channel_names,
        expected_trials=expected,
    )


def _scope(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    dataset_config = dict(config["datasets"][args.dataset])
    if args.canary:
        canary = dict(config["canary"])
        subjects = [int(value) for value in canary[f"{args.dataset}_subjects"]]
        models = [str(value) for value in canary["models"]]
        seeds = [int(value) for value in canary["seeds"]]
        epochs = int(canary["fixed_epochs"])
    else:
        subjects = [int(value) for value in dataset_config["subjects"]]
        models = [str(value) for value in config["models"]]
        seeds = [int(value) for value in config["seeds"]]
        epochs = int(config["fixed_epochs"])
    subjects = _csv_values(args.subjects, int) or subjects
    models = _csv_values(args.models, str) or models
    seeds = _csv_values(args.seeds, int) or seeds
    epochs = int(args.fixed_epochs or epochs)
    if len(subjects) != len(set(subjects)) or len(models) != len(set(models)) or len(seeds) != len(set(seeds)):
        raise ValueError("subjects, models, and seeds must be unique")
    if any(model not in PUBLICATION_BASELINES for model in models):
        raise ValueError(f"models must be selected from {list(PUBLICATION_BASELINES)}")
    if any(model not in BASELINE_OPTIMIZERS for model in models):
        raise ValueError("one or more models have no locked optimizer settings")
    if not subjects or not models or not seeds or epochs <= 0:
        raise ValueError("campaign scope must be non-empty and use positive epochs")
    return {"subjects": subjects, "models": models, "seeds": seeds, "fixed_epochs": epochs}


def _dataset_output(args: argparse.Namespace) -> Path:
    root = Path(args.output_root)
    if args.canary:
        root = root / "canary"
    return ensure_dir(root / args.dataset)


def _init_contract(
    args: argparse.Namespace, config: dict[str, Any], scope: dict[str, Any]
) -> dict[str, Any]:
    output = _dataset_output(args)
    source_root = Path(args.source_root)
    official_locks = verify_official_source_locks(source_root)
    source_manifest = collect_source_tree_manifest(ROOT)
    external_manifest = _external_source_manifest(source_root)
    environment = _environment()
    body = {
        "schema": "dpc-snn-v8-posthoc-publication-contract/v1",
        "posthoc_explanatory": True,
        "dataset": args.dataset,
        "scope": scope,
        "resolved_config": config,
        "config_file_sha256": file_sha256(args.config),
        "source_tree_sha256": source_tree_digest(source_manifest),
        "external_source_sha256": sha256_fingerprint(external_manifest),
        "official_source_locks": official_locks,
        "environment": environment,
        "paths": {
            "source_root": str(source_root.resolve()),
            "bci2a_train_root": str(Path(args.bci2a_train_root).resolve()),
            "bci2a_eval_root": str(Path(args.bci2a_eval_root).resolve()),
            "storage_root": str(Path(args.storage_root).resolve()),
        },
    }
    contract = {**body, "combined_sha256": sha256_fingerprint(body)}
    contract_path = output / "contract.json"
    if contract_path.is_file():
        saved = read_json(contract_path)
        if saved != contract:
            raise RuntimeError("existing publication campaign contract differs from current inputs")
    else:
        write_json(contract_path, contract)
        write_json(output / "source_tree_manifest.json", source_manifest)
        write_json(output / "external_source_manifest.json", external_manifest)
        write_json(output / "official_source_locks.json", official_locks)
        (output / "resolved_campaign.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
    return contract


def _load_contract(output: Path) -> dict[str, Any]:
    contract = read_json(output / "contract.json")
    body = {key: value for key, value in contract.items() if key != "combined_sha256"}
    if contract.get("combined_sha256") != sha256_fingerprint(body):
        raise RuntimeError("publication campaign contract digest is invalid")
    return contract


def _run_base(output: Path, model: str, subject: int, seed: int) -> Path:
    return output / "runs" / model / f"subject_{subject:02d}" / f"seed_{seed}"


def _training_fingerprint(
    *,
    contract: dict[str, Any],
    source_tree: dict[str, str],
    resolved: dict[str, Any],
    data_identity: dict[str, Any],
    rows: list[dict[str, Any]],
    augmentation: dict[str, Any],
) -> dict[str, Any]:
    return build_v8_run_fingerprint(
        resolved_run_config=resolved,
        source_tree=source_tree,
        data=data_identity,
        split={
            "role": "training",
            "sessions": sorted({str(row["session"]) for row in rows}),
            "trial_ids_sha256": sha256_fingerprint([row["trial_id"] for row in rows]),
            "heldout_loaded": False,
        },
        augmentation=augmentation,
        prior={"policy": "none"},
        checkpoint={
            "selection": "fixed_last_epoch",
            "heldout_checkpoint_selection": False,
            "contract_sha256": contract["combined_sha256"],
        },
        environment=contract["environment"],
    )


def _train_subject(
    args: argparse.Namespace,
    config: dict[str, Any],
    scope: dict[str, Any],
    output: Path,
    contract: dict[str, Any],
    subject: int,
) -> list[dict[str, Any]]:
    source_tree = read_json(output / "source_tree_manifest.json")
    x_raw, y, rows, data_manifest, data_identity = _load_view(
        args, config, subject=subject, role="training"
    )
    sfreq = float(rows[0]["sfreq"])
    carrier = task_carrier(
        x_raw, sfreq=sfreq, epoch_tmin=float(rows[0]["epoch_tmin"])
    )
    gain = fit_fixed_gain(carrier, clip=float(config["preprocessing"]["clip_after_gain"]))
    normalized = apply_fixed_gain(carrier, gain)
    prepared = {
        model: prepare_model_input(model, normalized, sfreq=sfreq)
        for model in scope["models"]
    }
    records: list[dict[str, Any]] = []
    dataset_config = dict(config["datasets"][args.dataset])
    augmentation = dict(config["augmentation"])
    for model_name in scope["models"]:
        physical_batch = int(config["physical_batch_size"][model_name])
        effective_batch = int(config["effective_batch_size"])
        expected_steps = int(np.ceil(y.size / effective_batch)) * int(scope["fixed_epochs"])
        for seed in scope["seeds"]:
            run_seed = int(seed) * 100_003 + int(subject) * 1_009 + (
                500_000 if args.dataset == "openbmi" else 0
            )
            resolved = {
                "stage": "posthoc_publication_baseline_training",
                "dataset": args.dataset,
                "model": model_name,
                "subject": int(subject),
                "seed": int(seed),
                "run_seed": run_seed,
                "fixed_epochs": int(scope["fixed_epochs"]),
                "physical_batch_size": physical_batch,
                "effective_batch_size": effective_batch,
                "optimizer": BASELINE_OPTIMIZERS[model_name],
                "preprocessing": config["preprocessing"],
                "augmentation": augmentation,
                "n_channels": len(config["channel_names"]),
                "n_classes": int(dataset_config["n_classes"]),
                "checkpoint_state_canonicalization": (
                    "official_forward_max_norm_applied_on_training_reference_before_seal"
                    if model_name == "fbcnet"
                    else "none"
                ),
            }
            fingerprint = _training_fingerprint(
                contract=contract,
                source_tree=source_tree,
                resolved=resolved,
                data_identity=data_identity,
                rows=rows,
                augmentation=augmentation,
            )
            training_dir = ensure_dir(_run_base(output, model_name, subject, seed) / "training")
            fingerprint_path = training_dir / "source_fingerprint.json"
            if (training_dir / "manifest.json").is_file():
                validate_v8_resume_fingerprint(fingerprint_path, fingerprint)
                validate_run_artifact_manifest(
                    training_dir, required_files=TRAIN_FILES, verify_hashes=True
                )
                metrics = read_json(training_dir / "training_metrics.json")
                records.append(metrics)
                continue
            if fingerprint_path.is_file():
                validate_v8_resume_fingerprint(fingerprint_path, fingerprint)
            else:
                write_v8_fingerprint(fingerprint_path, fingerprint)
            (training_dir / "resolved_config.yaml").write_text(
                yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
            )
            write_json(training_dir / "training_data_manifest.json", data_manifest)
            save_npz(training_dir / "gain.npz", values=gain.values, clip=np.asarray(gain.clip))
            training_started_at = time.time()
            write_json(
                training_dir / "runtime_status.json",
                {
                    "status": "training",
                    "started_at": training_started_at,
                    "heldout_session_loaded": False,
                },
            )
            fit = fit_baseline(
                model_name,
                source_root=args.source_root,
                x_train=prepared[model_name],
                y_train=y,
                x_validation=None,
                y_validation=None,
                device=args.device,
                seed=run_seed,
                epochs=int(scope["fixed_epochs"]),
                patience=int(scope["fixed_epochs"]),
                augmentation=augmentation,
                fixed_epoch=int(scope["fixed_epochs"]),
                scheduler_epochs=int(scope["fixed_epochs"]),
                run_label=f"PUB:{args.dataset}:{model_name}:S{subject}:seed{seed}",
                n_channels=len(config["channel_names"]),
                n_classes=int(dataset_config["n_classes"]),
                physical_batch_size=physical_batch,
                effective_batch_size=effective_batch,
            )
            if fit.optimizer_steps != expected_steps:
                raise RuntimeError(
                    f"optimizer-step mismatch for {model_name}: "
                    f"{fit.optimizer_steps} != {expected_steps}"
                )
            state_digest = _state_digest(fit.model)
            checkpoint = {
                "state_dict": fit.last_state,
                "model": model_name,
                "subject": int(subject),
                "seed": int(seed),
                "run_seed": run_seed,
                "n_channels": len(config["channel_names"]),
                "n_classes": int(dataset_config["n_classes"]),
                "samples": int(
                    prepared[model_name].shape[-2]
                    if prepared[model_name].ndim == 5
                    else prepared[model_name].shape[-1]
                ),
                "state_sha256": state_digest,
                "training_fingerprint_sha256": fingerprint["combined_sha256"],
                "contract_sha256": contract["combined_sha256"],
            }
            _atomic_torch_save(checkpoint, training_dir / "checkpoint.pt")
            write_csv(training_dir / "history.csv", fit.history)
            metrics = {
                "dataset": args.dataset,
                "model": model_name,
                "subject": int(subject),
                "seed": int(seed),
                "fixed_epochs": int(scope["fixed_epochs"]),
                "physical_batch_size": physical_batch,
                "effective_batch_size": effective_batch,
                "optimizer_steps": fit.optimizer_steps,
                "expected_optimizer_steps": expected_steps,
                "final_training_loss": float(fit.history[-1]["loss"]),
                "elapsed_seconds": float(fit.elapsed_seconds),
                "parameter_count": int(sum(parameter.numel() for parameter in fit.model.parameters())),
                "heldout_session_loaded": False,
                "checkpoint_state_sha256": state_digest,
                "checkpoint_state_canonicalized": model_name == "fbcnet",
            }
            write_json(training_dir / "training_metrics.json", metrics)
            write_json(
                training_dir / "state_audit.json",
                {
                    "live_state_sha256": state_digest,
                    "checkpoint_state_sha256": checkpoint["state_sha256"],
                    "identical": True,
                },
            )
            write_json(
                training_dir / "runtime_status.json",
                {
                    "status": "training_completed_checkpoint_sealed",
                    "started_at": training_started_at,
                    "completed_at": time.time(),
                    "heldout_session_loaded": False,
                },
            )
            write_run_artifact_manifest(training_dir, required_files=TRAIN_FILES)
            records.append(metrics)
            del fit, checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return records


def _load_gain(path: Path) -> FixedGain:
    with np.load(path, allow_pickle=False) as archive:
        return FixedGain(
            values=np.asarray(archive["values"], dtype=np.float32),
            clip=float(np.asarray(archive["clip"]).item()),
        )


def _gain_digest(gain: FixedGain) -> str:
    return sha256_fingerprint(
        {"values_sha256": array_sha256(gain.values), "clip": float(gain.clip)}
    )


def _validate_barrier(output: Path, contract: dict[str, Any]) -> dict[str, Any]:
    barrier = read_json(output / "checkpoint_barrier.json")
    body = {key: value for key, value in barrier.items() if key != "combined_sha256"}
    if barrier.get("combined_sha256") != sha256_fingerprint(body):
        raise RuntimeError("checkpoint barrier digest is invalid")
    if barrier.get("contract_sha256") != contract["combined_sha256"]:
        raise RuntimeError("checkpoint barrier belongs to another contract")
    for record in barrier["records"]:
        checkpoint = output / record["checkpoint_path"]
        manifest = output / record["training_manifest_path"]
        if file_sha256(checkpoint) != record["checkpoint_file_sha256"]:
            raise RuntimeError(f"checkpoint changed after barrier: {checkpoint}")
        if file_sha256(manifest) != record["training_manifest_file_sha256"]:
            raise RuntimeError(f"training manifest changed after barrier: {manifest}")
    return barrier


def _build_barrier(
    output: Path, contract: dict[str, Any], scope: dict[str, Any]
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for model in scope["models"]:
        for subject in scope["subjects"]:
            for seed in scope["seeds"]:
                training_dir = _run_base(output, model, subject, seed) / "training"
                validate_run_artifact_manifest(
                    training_dir, required_files=TRAIN_FILES, verify_hashes=True
                )
                checkpoint_path = training_dir / "checkpoint.pt"
                manifest_path = training_dir / "manifest.json"
                records.append(
                    {
                        "model": model,
                        "subject": int(subject),
                        "seed": int(seed),
                        "checkpoint_path": checkpoint_path.relative_to(output).as_posix(),
                        "checkpoint_file_sha256": file_sha256(checkpoint_path),
                        "training_manifest_path": manifest_path.relative_to(output).as_posix(),
                        "training_manifest_file_sha256": file_sha256(manifest_path),
                    }
                )
    core = {
        "schema": "dpc-snn-v8-posthoc-publication-checkpoint-barrier/v1",
        "dataset": contract["dataset"],
        "contract_sha256": contract["combined_sha256"],
        "scope": scope,
        "expected_runs": len(scope["models"]) * len(scope["subjects"]) * len(scope["seeds"]),
        "all_training_complete_before_evaluation": True,
        "records": records,
    }
    path = output / "checkpoint_barrier.json"
    if path.is_file():
        saved = read_json(path)
        saved_core = {
            key: value
            for key, value in saved.items()
            if key not in {"sealed_at", "combined_sha256"}
        }
        if saved_core != core:
            raise RuntimeError("existing checkpoint barrier differs from sealed checkpoints")
        return _validate_barrier(output, contract)
    body = {**core, "sealed_at": time.time()}
    barrier = {**body, "combined_sha256": sha256_fingerprint(body)}
    write_json(path, barrier)
    return _validate_barrier(output, contract)


def _evaluation_fingerprint(
    *,
    contract: dict[str, Any],
    barrier: dict[str, Any],
    source_tree: dict[str, str],
    resolved: dict[str, Any],
    data_identity: dict[str, Any],
    rows: list[dict[str, Any]],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    return build_v8_run_fingerprint(
        resolved_run_config=resolved,
        source_tree=source_tree,
        data=data_identity,
        split={
            "role": "evaluation",
            "sessions": sorted({str(row["session"]) for row in rows}),
            "trial_ids_sha256": sha256_fingerprint([row["trial_id"] for row in rows]),
            "checkpoint_selection_from_evaluation": False,
        },
        augmentation={"enabled": False, "evaluation_only": True},
        prior={"policy": "none"},
        checkpoint={
            "checkpoint_file_sha256": checkpoint_sha256,
            "checkpoint_barrier_sha256": barrier["combined_sha256"],
            "contract_sha256": contract["combined_sha256"],
        },
        environment=contract["environment"],
    )


def _evaluate_subject(
    args: argparse.Namespace,
    config: dict[str, Any],
    scope: dict[str, Any],
    output: Path,
    contract: dict[str, Any],
    subject: int,
) -> list[dict[str, Any]]:
    barrier = _validate_barrier(output, contract)
    source_tree = read_json(output / "source_tree_manifest.json")
    x_raw, y, rows, data_manifest, data_identity = _load_view(
        args, config, subject=subject, role="evaluation"
    )
    sfreq = float(rows[0]["sfreq"])
    carrier = task_carrier(
        x_raw, sfreq=sfreq, epoch_tmin=float(rows[0]["epoch_tmin"])
    )
    channel_names = [str(value) for value in config["channel_names"]]
    validate_region_partition(channel_names)
    dataset_config = dict(config["datasets"][args.dataset])
    records: list[dict[str, Any]] = []
    for model_name in scope["models"]:
        reference_training = _run_base(output, model_name, subject, scope["seeds"][0]) / "training"
        reference_gain = _load_gain(reference_training / "gain.npz")
        reference_gain_sha = _gain_digest(reference_gain)
        normalized = apply_fixed_gain(carrier, reference_gain)
        nominal_input = prepare_model_input(model_name, normalized, sfreq=sfreq)
        frequency_inputs = {
            name: prepare_model_input(
                model_name,
                apply_fixed_gain(
                    fft_band_occlusion(carrier, sfreq=sfreq, low_hz=band[0], high_hz=band[1]),
                    reference_gain,
                ),
                sfreq=sfreq,
            )
            for name, band in FREQUENCY_OCCLUSIONS_HZ.items()
        }
        region_inputs = {
            name: prepare_model_input(
                model_name,
                zero_reference_region(
                    normalized, channel_names=channel_names, region=name
                ),
                sfreq=sfreq,
            )
            for name in REGION_CHANNELS
        }
        for seed in scope["seeds"]:
            base = _run_base(output, model_name, subject, seed)
            training_dir = base / "training"
            evaluation_dir = ensure_dir(base / "evaluation")
            gain_path = training_dir / "gain.npz"
            if _gain_digest(_load_gain(gain_path)) != reference_gain_sha:
                raise RuntimeError("training-derived gain differs across paired seeds/models")
            checkpoint_path = training_dir / "checkpoint.pt"
            checkpoint_sha = file_sha256(checkpoint_path)
            resolved = {
                "stage": "posthoc_publication_baseline_evaluation",
                "dataset": args.dataset,
                "model": model_name,
                "subject": int(subject),
                "seed": int(seed),
                "evaluation_session": dataset_config["evaluation_session"],
                "frequency_occlusions_hz": FREQUENCY_OCCLUSIONS_HZ,
                "region_partition": REGION_CHANNELS,
                "saliency": "absolute_input_gradient_times_input_for_true_class",
                "checkpoint_barrier_sha256": barrier["combined_sha256"],
            }
            fingerprint = _evaluation_fingerprint(
                contract=contract,
                barrier=barrier,
                source_tree=source_tree,
                resolved=resolved,
                data_identity=data_identity,
                rows=rows,
                checkpoint_sha256=checkpoint_sha,
            )
            fingerprint_path = evaluation_dir / "source_fingerprint.json"
            if (evaluation_dir / "manifest.json").is_file():
                validate_v8_resume_fingerprint(fingerprint_path, fingerprint)
                validate_run_artifact_manifest(
                    evaluation_dir,
                    required_files=EVALUATION_FILES,
                    verify_hashes=True,
                    verify_prediction_schema=True,
                )
                records.append(read_json(evaluation_dir / "metrics.json"))
                continue
            if fingerprint_path.is_file():
                validate_v8_resume_fingerprint(fingerprint_path, fingerprint)
            else:
                write_v8_fingerprint(fingerprint_path, fingerprint)
            (evaluation_dir / "resolved_config.yaml").write_text(
                yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
            )
            write_json(evaluation_dir / "evaluation_data_manifest.json", data_manifest)
            evaluation_started_at = time.time()
            write_json(
                evaluation_dir / "runtime_status.json",
                {"status": "evaluating", "started_at": evaluation_started_at},
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if (
                checkpoint["contract_sha256"] != contract["combined_sha256"]
                or checkpoint["model"] != model_name
                or int(checkpoint["subject"]) != int(subject)
                or int(checkpoint["seed"]) != int(seed)
            ):
                raise RuntimeError("checkpoint metadata do not match evaluation run")
            model = build_v62_neural_baseline(
                model_name,
                source_root=args.source_root,
                n_channels=int(checkpoint["n_channels"]),
                n_classes=int(checkpoint["n_classes"]),
                samples=int(checkpoint["samples"]),
            ).to(args.device)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            before = _state_digest(model)
            if before != checkpoint["state_sha256"]:
                raise RuntimeError("loaded checkpoint state digest does not match training seal")
            batch_size = int(config["physical_batch_size"][model_name])
            nominal = predict_baseline(
                model, nominal_input, y, device=args.device, batch_size=batch_size
            )
            nominal_probability = softmax_probabilities(nominal["logits"])
            frequency_results: list[dict[str, Any]] = []
            frequency_logits: list[np.ndarray] = []
            frequency_probabilities: list[np.ndarray] = []
            frequency_predictions: list[np.ndarray] = []
            perturbation_rows: list[dict[str, Any]] = []
            for name, inputs in frequency_inputs.items():
                result = predict_baseline(
                    model, inputs, y, device=args.device, batch_size=batch_size
                )
                probability = softmax_probabilities(result["logits"])
                frequency_results.append(result)
                frequency_logits.append(result["logits"])
                frequency_probabilities.append(probability)
                frequency_predictions.append(result["pred"])
                perturbation_rows.append(
                    {
                        "kind": "frequency",
                        "name": name,
                        **{key: float(result[key]) for key in ("accuracy", "balanced_accuracy", "kappa", "macro_f1")},
                        "nominal_accuracy": float(nominal["accuracy"]),
                        "accuracy_drop": float(nominal["accuracy"] - result["accuracy"]),
                    }
                )
            region_logits: list[np.ndarray] = []
            region_probabilities: list[np.ndarray] = []
            region_predictions: list[np.ndarray] = []
            for name, inputs in region_inputs.items():
                result = predict_baseline(
                    model, inputs, y, device=args.device, batch_size=batch_size
                )
                probability = softmax_probabilities(result["logits"])
                region_logits.append(result["logits"])
                region_probabilities.append(probability)
                region_predictions.append(result["pred"])
                perturbation_rows.append(
                    {
                        "kind": "region",
                        "name": name,
                        **{key: float(result[key]) for key in ("accuracy", "balanced_accuracy", "kappa", "macro_f1")},
                        "nominal_accuracy": float(nominal["accuracy"]),
                        "accuracy_drop": float(nominal["accuracy"] - result["accuracy"]),
                    }
                )
            saliency_raw, saliency_normalized = input_gradient_channel_saliency(
                model,
                nominal_input,
                y,
                device=args.device,
                batch_size=batch_size,
                channel_axis=2 if nominal_input.ndim == 5 else 1,
            )
            after = _state_digest(model)
            if before != after:
                raise RuntimeError("model state changed during held-out evaluation")
            write_trial_predictions(
                evaluation_dir,
                logits=nominal["logits"],
                probabilities=nominal_probability,
                pred=nominal["pred"],
                label=y,
                subject=[row["subject"] for row in rows],
                session=[row["session"] for row in rows],
                run=[row["run"] for row in rows],
                trial_id=[row["trial_id"] for row in rows],
                seed=int(seed),
                model=model_name,
            )
            save_npz(
                evaluation_dir / "perturbations.npz",
                label=y,
                frequency_names=np.asarray(list(FREQUENCY_OCCLUSIONS_HZ)),
                frequency_logits=np.asarray(frequency_logits, dtype=np.float32),
                frequency_probabilities=np.asarray(frequency_probabilities, dtype=np.float32),
                frequency_pred=np.asarray(frequency_predictions, dtype=np.int64),
                region_names=np.asarray(list(REGION_CHANNELS)),
                region_logits=np.asarray(region_logits, dtype=np.float32),
                region_probabilities=np.asarray(region_probabilities, dtype=np.float32),
                region_pred=np.asarray(region_predictions, dtype=np.int64),
            )
            save_npz(
                evaluation_dir / "saliency.npz",
                channel_names=np.asarray(channel_names),
                raw=saliency_raw,
                normalized=saliency_normalized,
                label=y,
                trial_id=np.asarray([row["trial_id"] for row in rows]),
            )
            write_csv(evaluation_dir / "perturbation_metrics.csv", perturbation_rows)
            metrics = {
                "dataset": args.dataset,
                "model": model_name,
                "subject": int(subject),
                "seed": int(seed),
                **{
                    key: float(nominal[key])
                    for key in ("accuracy", "balanced_accuracy", "kappa", "macro_f1")
                },
                "calibration": multiclass_calibration_metrics(nominal["logits"], y),
                "n_trials": int(y.size),
                "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
                "checkpoint_file_sha256": checkpoint_sha,
                "checkpoint_barrier_sha256": barrier["combined_sha256"],
                "state_unchanged": True,
            }
            write_json(evaluation_dir / "metrics.json", metrics)
            write_json(
                evaluation_dir / "state_audit.json",
                {"before_sha256": before, "after_sha256": after, "identical": True},
            )
            write_json(
                evaluation_dir / "runtime_status.json",
                {
                    "status": "evaluation_completed",
                    "started_at": evaluation_started_at,
                    "completed_at": time.time(),
                },
            )
            write_run_artifact_manifest(evaluation_dir, required_files=EVALUATION_FILES)
            records.append(metrics)
            del model, checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return records


def _worker_command(args: argparse.Namespace, subject: int, phase: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(args.config),
        "--dataset",
        args.dataset,
        "--phase",
        phase,
        "--output-root",
        str(args.output_root),
        "--source-root",
        str(args.source_root),
        "--bci2a-train-root",
        str(args.bci2a_train_root),
        "--bci2a-eval-root",
        str(args.bci2a_eval_root),
        "--storage-root",
        str(args.storage_root),
        "--device",
        str(args.device),
        "--subjects",
        str(subject),
        "--models",
        str(args.models),
        "--seeds",
        str(args.seeds),
        "--worker-subject",
        str(subject),
    ]
    if args.fixed_epochs:
        command.extend(("--fixed-epochs", str(args.fixed_epochs)))
    if args.canary:
        command.append("--canary")
    return command


def _run_workers(
    args: argparse.Namespace,
    *,
    subjects: list[int],
    phase: str,
    output: Path,
) -> None:
    log_dir = ensure_dir(output / "workers")

    def execute(subject: int) -> int:
        stdout_path = log_dir / f"{phase}_subject_{subject:02d}.stdout.log"
        stderr_path = log_dir / f"{phase}_subject_{subject:02d}.stderr.log"
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
            "a", encoding="utf-8"
        ) as stderr:
            completed = subprocess.run(
                _worker_command(args, subject, phase),
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        if completed.returncode != 0:
            tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"{phase} worker failed for subject {subject}:\n{tail}")
        return subject

    with ThreadPoolExecutor(max_workers=min(int(args.workers), len(subjects))) as pool:
        futures = {pool.submit(execute, subject): subject for subject in subjects}
        for future in as_completed(futures):
            subject = future.result()
            print(f"__V8_PUBLICATION_WORKER_DONE__ phase={phase} subject={subject}", flush=True)


def _aggregate_nominal(output: Path, scope: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in scope["models"]:
        for subject in scope["subjects"]:
            for seed in scope["seeds"]:
                evaluation_dir = _run_base(output, model, subject, seed) / "evaluation"
                validate_run_artifact_manifest(
                    evaluation_dir,
                    required_files=EVALUATION_FILES,
                    verify_hashes=True,
                    verify_prediction_schema=True,
                )
                metrics = read_json(evaluation_dir / "metrics.json")
                rows.append(
                    {
                        "dataset": metrics["dataset"],
                        "model": model,
                        "subject": subject,
                        "seed": seed,
                        "accuracy": metrics["accuracy"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "kappa": metrics["kappa"],
                        "macro_f1": metrics["macro_f1"],
                        "negative_log_likelihood": metrics["calibration"]["negative_log_likelihood"],
                        "brier_score": metrics["calibration"]["brier_score"],
                        "ece": metrics["calibration"]["ece"],
                        "parameter_count": metrics["parameter_count"],
                    }
                )
    write_csv(output / "summary.csv", rows)
    summaries: list[dict[str, Any]] = []
    for model in scope["models"]:
        model_rows = [row for row in rows if row["model"] == model]
        subject_values = []
        for subject in scope["subjects"]:
            paired = [row for row in model_rows if int(row["subject"]) == int(subject)]
            subject_values.append(float(np.mean([float(row["accuracy"]) for row in paired])))
        summaries.append(
            {
                "model": model,
                "subject_macro_accuracy": float(np.mean(subject_values)),
                "subject_accuracy_std": float(np.std(subject_values, ddof=1)) if len(subject_values) > 1 else 0.0,
                "subjects": len(subject_values),
                "seeds_per_subject": len(scope["seeds"]),
                "parameter_count": int(model_rows[0]["parameter_count"]),
            }
        )
    write_csv(output / "model_summary.csv", summaries)
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/v8_publication_baselines.yaml")
    parser.add_argument("--dataset", choices=("bci2a", "openbmi"), required=True)
    parser.add_argument("--phase", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--bci2a-train-root", default="data/processed/bci2a_session_t")
    parser.add_argument("--bci2a-eval-root", default="data/processed/bci2a_session_e")
    parser.add_argument("--storage-root", default=os.environ.get("DPC_SNN_STORAGE_ROOT", "artifacts/storage"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--subjects", default="")
    parser.add_argument("--models", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--fixed-epochs", type=int, default=0)
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--worker-subject", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def main() -> None:
    args = _parse_args()
    configure_cache_env(args.storage_root)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    scope = _scope(args, config)
    output = _dataset_output(args)
    if args.worker_subject:
        contract = _load_contract(output)
        if int(args.worker_subject) not in scope["subjects"]:
            raise ValueError("worker subject is outside the resolved scope")
        if args.phase == "train":
            records = _train_subject(
                args, config, scope, output, contract, int(args.worker_subject)
            )
        elif args.phase == "evaluate":
            records = _evaluate_subject(
                args, config, scope, output, contract, int(args.worker_subject)
            )
        else:
            raise ValueError("worker phase must be train or evaluate")
        print(json.dumps({"status": "completed", "records": len(records)}, indent=2))
        return

    contract = _init_contract(args, config, scope)
    if args.phase in {"train", "all"}:
        _run_workers(args, subjects=scope["subjects"], phase="train", output=output)
        barrier = _build_barrier(output, contract, scope)
        write_json(
            output / "campaign_status.json",
            {
                "status": "training_complete_evaluation_unopened_by_this_phase",
                "dataset": args.dataset,
                "training_runs": barrier["expected_runs"],
                "barrier_sha256": barrier["combined_sha256"],
            },
        )
    else:
        _validate_barrier(output, contract)
    if args.phase in {"evaluate", "all"}:
        _run_workers(args, subjects=scope["subjects"], phase="evaluate", output=output)
        rows = _aggregate_nominal(output, scope)
        write_json(
            output / "campaign_status.json",
            {
                "status": "baseline_evaluation_complete_pending_fusion_analysis",
                "dataset": args.dataset,
                "completed_runs": len(rows),
                "expected_runs": len(scope["subjects"]) * len(scope["models"]) * len(scope["seeds"]),
                "posthoc_explanatory": True,
            },
        )
    print(json.dumps(read_json(output / "campaign_status.json"), indent=2))


if __name__ == "__main__":
    main()
