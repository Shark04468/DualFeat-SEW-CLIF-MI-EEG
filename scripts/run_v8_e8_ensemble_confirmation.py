#!/usr/bin/env python3
"""Run the frozen V8 ensemble OpenBMI S1-to-S2 external confirmation."""

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

from dpc_snn.analysis.v8_utility import multiclass_calibration_metrics  # noqa: E402
from dpc_snn.data.openbmi import load_openbmi_subject  # noqa: E402
from dpc_snn.data.v8_openbmi import (  # noqa: E402
    OPENBMI_FIXED_INPUT_ADAPTER,
    prepare_openbmi_v8_view,
)
from dpc_snn.experiments.v62_baselines import task_carrier  # noqa: E402
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    sha256_fingerprint,
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
    write_trial_predictions,
)
from dpc_snn.experiments.v8_ensemble import (  # noqa: E402
    validate_ensemble_freeze_contract,
)
from dpc_snn.experiments.v8_ensemble_followup import (  # noqa: E402
    component_state_digests,
    fit_ensemble_components,
    load_ensemble_components,
    load_fixed_gain,
    predict_ensemble_from_carrier,
    resolve_frozen_channel_basis,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    build_v8_run_fingerprint,
    collect_source_tree_manifest,
    source_tree_digest,
    validate_v8_external_unlock_manifest,
    validate_v8_freeze_manifest,
    validate_v8_resume_fingerprint,
    write_v8_fingerprint,
)
from dpc_snn.experiments.v8_statistics import (  # noqa: E402
    pair_subject_seed_rows,
    paired_delta_summary,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v8_e1_baselines import _environment  # noqa: E402


ARMS = (
    "primary",
    "matched_ann",
    "anchor",
    "atcnet",
    "fbcnet",
    "decoder_snn",
    "decoder_ann",
)
CHECKPOINTS = ("atcnet", "fbcnet", "sew_clif", "ann_sew")
TRAINING_FILES = (
    "training_fingerprint.json",
    "source_tree_manifest.json",
    "resolved_config.yaml",
    "s1_access_manifest.json",
    "gain.npz",
    "atcnet_history.csv",
    "fbcnet_history.csv",
    "sew_clif_history.csv",
    "ann_sew_history.csv",
    "atcnet.pt",
    "fbcnet.pt",
    "sew_clif.pt",
    "ann_sew.pt",
    "training_state_audit.json",
    "training_runtime_status.json",
)
FINAL_FILES = (
    "manifest.json",
    *TRAINING_FILES,
    "training_artifact_manifest.json",
    "source_fingerprint.json",
    "s2_access_manifest.json",
    "data_access_manifest.json",
    "state_audit.json",
    "metrics.json",
    "runtime_status.json",
) + tuple(f"{arm}_predictions.{suffix}" for arm in ARMS for suffix in ("npz", "csv"))
CAMPAIGN_FILES = (
    "manifest.json",
    "campaign_status.json",
    "checkpoint_barrier.json",
    "summary.csv",
    "model_summary.csv",
    "paired_subject_seed.csv",
    "source_tree_manifest.json",
    "resolved_campaign.yaml",
    "freeze_manifest.json",
    "external_unlock_manifest.json",
)


def _csv_values(value: str, cast: Any = str) -> list[Any]:
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


def _training_manifest(run_dir: Path) -> dict[str, Any]:
    payload = {
        "schema": "dpc-snn-v8-e8-training-checkpoint/v1",
        "files": {name: file_sha256(run_dir / name) for name in TRAINING_FILES},
    }
    payload["combined_sha256"] = sha256_fingerprint(payload)
    return payload


def _write_training_manifest(run_dir: Path) -> None:
    write_json(run_dir / "training_artifact_manifest.json", _training_manifest(run_dir))


def _validate_training(run_dir: Path) -> dict[str, Any]:
    manifest = read_json(run_dir / "training_artifact_manifest.json")
    if manifest.get("schema") != "dpc-snn-v8-e8-training-checkpoint/v1":
        raise RuntimeError(f"invalid E8 training manifest: {run_dir}")
    expected = {
        "schema": manifest["schema"],
        "files": {name: file_sha256(run_dir / name) for name in TRAINING_FILES},
    }
    expected["combined_sha256"] = sha256_fingerprint(expected)
    if manifest != expected:
        raise RuntimeError(f"E8 training artifacts changed after checkpointing: {run_dir}")
    return manifest


def _save_components(run_dir: Path, fit: Any) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, model in fit.components.as_dict().items():
        state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        torch.save(state, run_dir / f"{name}.pt")
        hashes[name] = sha256_fingerprint(
            {key: file_sha256(run_dir / f"{name}.pt") for key in ("checkpoint",)}
        )
        write_csv(run_dir / f"{name}_history.csv", fit.histories[name])
    np.savez_compressed(
        run_dir / "gain.npz",
        values=np.asarray(fit.gain.values, dtype=np.float32),
        clip=np.asarray(fit.gain.clip, dtype=np.float32),
    )
    return hashes


def _train_one(
    *,
    output: Path,
    source_root: Path,
    freeze: dict[str, Any],
    unlock: dict[str, Any],
    config: dict[str, Any],
    source_tree: dict[str, str],
    environment: dict[str, Any],
    subject: int,
    seed: int,
    x_s1: np.ndarray,
    y_s1: np.ndarray,
    metadata_s1: list[dict[str, Any]],
    s1_manifest: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    run_dir = ensure_dir(output / f"subject_{subject:02d}" / f"seed_{seed}")
    run_seed = int(seed) * 100_003 + int(subject) * 1_009
    decoder_seed = run_seed + 8_000_003
    component_rule = dict(freeze["checkpoint_rule"]["components"])
    model_config = dict(freeze["architecture"]["model_config"])
    resolved = {
        "stage": "E8_ENSEMBLE_TRAINING",
        "protocol": config["protocol"],
        "subject": int(subject),
        "seed": int(seed),
        "run_seed": run_seed,
        "decoder_seed": decoder_seed,
        "n_classes": 2,
        "architecture": model_config,
        "component_rule": component_rule,
        "preprocessing": freeze["preprocessing"],
        "augmentation": freeze["augmentation"],
        "external_unlock_sha256": unlock["combined_sha256"],
    }
    s1_digest = _array_digest(
        x_s1, y_s1, [str(row["trial_id"]) for row in metadata_s1]
    )
    fingerprint = build_v8_run_fingerprint(
        resolved_run_config=resolved,
        source_tree=source_tree,
        data={"openbmi_s1_sha256": s1_digest, "access": s1_manifest},
        split={
            "train_session": "S1",
            "train_trial_ids": [row["trial_id"] for row in metadata_s1],
            "evaluation_session_unopened": "S2",
        },
        augmentation=freeze["augmentation"],
        prior={"policy": "none", "delay_enabled": False},
        checkpoint={
            "components": component_rule,
            "all_checkpoints_before_any_s2": True,
        },
        environment=environment,
    )
    fingerprint_path = run_dir / "training_fingerprint.json"
    if (run_dir / "training_artifact_manifest.json").is_file():
        validate_v8_resume_fingerprint(fingerprint_path, fingerprint)
        manifest = _validate_training(run_dir)
        return {
            "subject": subject,
            "seed": seed,
            "training_manifest_sha256": manifest["combined_sha256"],
            "training_fingerprint": fingerprint["combined_sha256"],
        }
    if fingerprint_path.is_file():
        validate_v8_resume_fingerprint(fingerprint_path, fingerprint)
    else:
        write_v8_fingerprint(fingerprint_path, fingerprint)
    write_json(run_dir / "source_tree_manifest.json", source_tree)
    write_json(run_dir / "s1_access_manifest.json", s1_manifest)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    write_json(
        run_dir / "training_runtime_status.json",
        {
            "status": "training_s1_s2_unopened",
            "started_at": time.time(),
            "openbmi_s2_accessed": False,
        },
    )
    fit = fit_ensemble_components(
        x_s1,
        y_s1,
        sfreq=250.0,
        epoch_tmin=-1.0,
        source_root=source_root,
        model_config=model_config,
        component_rule=component_rule,
        preprocessing=dict(freeze["preprocessing"]),
        augmentation=dict(freeze["augmentation"]),
        n_classes=2,
        run_seed=run_seed,
        decoder_seed=decoder_seed,
        device=device,
        run_label=f"V8-E8-ensemble:S{subject}:seed{seed}",
    )
    state = component_state_digests(fit.components)
    _save_components(run_dir, fit)
    write_json(
        run_dir / "training_state_audit.json",
        {
            "checkpoint_state_sha256": state,
            "constraint_projection_counts": fit.constraint_projection_counts,
            "optimizer_steps": fit.optimizer_steps,
            "train_seconds": fit.train_seconds,
            "openbmi_s2_accessed": False,
        },
    )
    write_json(
        run_dir / "training_runtime_status.json",
        {
            "status": "training_completed_s2_unopened",
            "completed_at": time.time(),
            "openbmi_s2_accessed": False,
        },
    )
    _write_training_manifest(run_dir)
    manifest = _validate_training(run_dir)
    return {
        "subject": subject,
        "seed": seed,
        "training_manifest_sha256": manifest["combined_sha256"],
        "training_fingerprint": fingerprint["combined_sha256"],
    }


def _validate_barrier(
    path: Path,
    *,
    source_digest: str,
    unlock_sha256: str,
    subjects: list[int],
    seeds: list[int],
) -> dict[str, Any]:
    barrier = read_json(path)
    body = {key: value for key, value in barrier.items() if key != "combined_sha256"}
    if (
        barrier.get("schema") != "dpc-snn-v8-e8-checkpoint-barrier/v1"
        or barrier.get("source_tree_sha256") != source_digest
        or barrier.get("external_unlock_sha256") != unlock_sha256
        or barrier.get("subjects") != subjects
        or barrier.get("seeds") != seeds
        or barrier.get("all_training_complete_before_s2") is not True
        or barrier.get("combined_sha256") != sha256_fingerprint(body)
    ):
        raise RuntimeError("E8 checkpoint barrier is invalid")
    return barrier


def _evaluate_one(
    *,
    output: Path,
    source_root: Path,
    freeze: dict[str, Any],
    unlock: dict[str, Any],
    config: dict[str, Any],
    source_tree: dict[str, str],
    environment: dict[str, Any],
    barrier: dict[str, Any],
    subject: int,
    seed: int,
    x_s2: np.ndarray,
    y_s2: np.ndarray,
    metadata_s2: list[dict[str, Any]],
    s2_manifest: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    run_dir = output / f"subject_{subject:02d}" / f"seed_{seed}"
    training_manifest = _validate_training(run_dir)
    training_fingerprint = read_json(run_dir / "training_fingerprint.json")
    s2_digest = _array_digest(
        x_s2, y_s2, [str(row["trial_id"]) for row in metadata_s2]
    )
    final_resolved = {
        **yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8")),
        "stage": "E8_ENSEMBLE_CONFIRMATION",
        "checkpoint_barrier_sha256": barrier["combined_sha256"],
    }
    fingerprint = build_v8_run_fingerprint(
        resolved_run_config=final_resolved,
        source_tree=source_tree,
        data={
            "openbmi_s1_training_fingerprint": training_fingerprint["combined_sha256"],
            "openbmi_s2_sha256": s2_digest,
            "s2_access": s2_manifest,
        },
        split={
            "train_session": "S1",
            "evaluation_session": "S2",
            "evaluation_trial_ids": [row["trial_id"] for row in metadata_s2],
        },
        augmentation=freeze["augmentation"],
        prior={"policy": "none", "delay_enabled": False},
        checkpoint={
            "training_manifest_sha256": training_manifest["combined_sha256"],
            "checkpoint_barrier_sha256": barrier["combined_sha256"],
            "s2_checkpoint_selection": False,
        },
        environment=environment,
    )
    final_path = run_dir / "source_fingerprint.json"
    if (run_dir / "manifest.json").is_file():
        validate_v8_resume_fingerprint(final_path, fingerprint)
        validate_run_artifact_manifest(
            run_dir,
            required_files=FINAL_FILES,
            verify_hashes=True,
            verify_prediction_schema=True,
        )
        return read_json(run_dir / "metrics.json")
    if final_path.is_file():
        validate_v8_resume_fingerprint(final_path, fingerprint)
    else:
        write_v8_fingerprint(final_path, fingerprint)

    model_config = dict(freeze["architecture"]["model_config"])
    components = load_ensemble_components(
        run_dir,
        source_root=source_root,
        model_config=model_config,
        n_classes=2,
    ).to(device).eval()
    before = component_state_digests(components)
    expected_before = read_json(run_dir / "training_state_audit.json")[
        "checkpoint_state_sha256"
    ]
    if before != expected_before:
        raise RuntimeError("E8 loaded state differs from the pre-S2 checkpoint barrier")
    gain = load_fixed_gain(run_dir / "gain.npz")
    carrier = task_carrier(x_s2, sfreq=250.0, epoch_tmin=-1.0)
    prediction = predict_ensemble_from_carrier(
        components,
        carrier,
        y_s2,
        gain,
        sfreq=250.0,
        model_config=model_config,
        n_classes=2,
        device=device,
    )
    after = component_state_digests(components)
    if before != after:
        raise RuntimeError("E8 changed a checkpoint during OpenBMI S2 evaluation")

    for arm in ARMS:
        values = prediction["arms"][arm]
        write_trial_predictions(
            run_dir,
            logits=values["logits"],
            probabilities=values["probabilities"],
            pred=values["pred"],
            label=y_s2,
            subject=[row["subject"] for row in metadata_s2],
            session="S2",
            run=[row["run"] for row in metadata_s2],
            trial_id=[row["trial_id"] for row in metadata_s2],
            seed=seed,
            model=f"v8_e8_ensemble_{arm}",
            basename=f"{arm}_predictions",
        )
    arm_metrics: dict[str, Any] = {}
    for arm in ARMS:
        values = prediction["arms"][arm]
        arm_metrics[arm] = {
            **{
                key: values[key]
                for key in ("accuracy", "balanced_accuracy", "kappa", "macro_f1")
            },
            **multiclass_calibration_metrics(values["logits"], y_s2),
        }
    write_json(run_dir / "s2_access_manifest.json", s2_manifest)
    write_json(
        run_dir / "data_access_manifest.json",
        {
            "train_session": "S1",
            "evaluation_session": "S2",
            "all_training_complete_before_s2_barrier_sha256": barrier[
                "combined_sha256"
            ],
            "s2_checkpoint_selection": False,
            "s2_gradient_updates": False,
            "openbmi_s2_accessed": True,
        },
    )
    write_json(
        run_dir / "state_audit.json",
        {
            "checkpoint_before_s2": before,
            "state_after_s2": after,
            "identical": True,
            "model_updates": False,
        },
    )
    metrics = {
        "status": "completed",
        "stage": "E8_ENSEMBLE_CONFIRMATION",
        "protocol": config["protocol"],
        "subject": int(subject),
        "seed": int(seed),
        "accuracy": arm_metrics["primary"]["accuracy"],
        "balanced_accuracy": arm_metrics["primary"]["balanced_accuracy"],
        "kappa": arm_metrics["primary"]["kappa"],
        "macro_f1": arm_metrics["primary"]["macro_f1"],
        "arms": arm_metrics,
        "snn_mean_firing_rate": prediction["snn_mean_firing_rate"],
        "entropy_gate_mean": float(np.mean(prediction["primary_gate"])),
        "freeze_sha256": freeze["combined_sha256"],
        "external_unlock_sha256": unlock["combined_sha256"],
        "checkpoint_barrier_sha256": barrier["combined_sha256"],
        "run_fingerprint": fingerprint["combined_sha256"],
        "s2_selected_checkpoint": False,
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "runtime_status.json",
        {
            "status": "completed",
            "completed_at": time.time(),
            "openbmi_s2_accessed": True,
            "model_updates": False,
        },
    )
    write_run_artifact_manifest(run_dir, required_files=FINAL_FILES)
    return metrics


def _worker_command(
    args: argparse.Namespace,
    output: Path,
    *,
    phase: str,
    subject: int,
    seeds: list[int],
    barrier: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--source-root",
        str(Path(args.source_root).resolve()),
        "--freeze",
        str(Path(args.freeze).resolve()),
        "--e6-audit",
        str(Path(args.e6_audit).resolve()),
        "--e7",
        str(Path(args.e7).resolve()),
        "--unlock",
        str(Path(args.unlock).resolve()),
        "--output",
        str(output),
        "--config",
        str(Path(args.config).resolve()),
        "--dataset-config",
        str(Path(args.dataset_config).resolve()),
        "--device",
        str(args.device),
        "--worker-phase",
        phase,
        "--worker-subject",
        str(subject),
        "--worker-seeds",
        ",".join(str(seed) for seed in seeds),
    ]
    if barrier is not None:
        command.extend(("--barrier", str(barrier)))
    return command


def _run_worker(command: list[str], expected: list[Path]) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    threads = int(environment.get("DPC_SNN_E8_WORKER_THREADS", "8"))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = str(threads)
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
            f"E8 worker failed ({completed.returncode}): {' '.join(command)}\n"
            + completed.stderr[-8000:]
        )
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError("E8 worker missed outputs: " + ", ".join(missing))
    return [read_json(path) for path in expected]


def _run_parallel(
    args: argparse.Namespace,
    output: Path,
    *,
    phase: str,
    subjects: list[int],
    seeds: list[int],
    workers: int,
    barrier: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(int(workers), len(subjects))) as pool:
        futures = {}
        for subject in subjects:
            filename = "training_artifact_manifest.json" if phase == "train" else "metrics.json"
            expected = [
                output / f"subject_{subject:02d}" / f"seed_{seed}" / filename
                for seed in seeds
            ]
            future = pool.submit(
                _run_worker,
                _worker_command(
                    args,
                    output,
                    phase=phase,
                    subject=subject,
                    seeds=seeds,
                    barrier=barrier,
                ),
                expected,
            )
            futures[future] = subject
        for future in as_completed(futures):
            rows.extend(future.result())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--e6-audit", required=True)
    parser.add_argument("--e7", required=True)
    parser.add_argument("--unlock", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default="configs/experiments/v8_e8_ensemble_frozen.yaml"
    )
    parser.add_argument("--dataset-config", default="configs/datasets/openbmi.yaml")
    parser.add_argument("--subjects", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--canary-s1-only", action="store_true")
    parser.add_argument("--worker-phase", choices=("train", "evaluate"))
    parser.add_argument("--worker-subject", type=int)
    parser.add_argument("--worker-seeds", default="")
    parser.add_argument("--barrier", default="")
    args = parser.parse_args()

    configure_cache_env()
    output = ensure_dir(Path(args.output).resolve())
    source_root = Path(args.source_root).resolve()
    source_tree = collect_source_tree_manifest(ROOT)
    source_digest = source_tree_digest(source_tree)
    freeze = validate_ensemble_freeze_contract(
        validate_v8_freeze_manifest(Path(args.freeze).resolve())
    )
    unlock = validate_v8_external_unlock_manifest(
        Path(args.unlock).resolve(),
        expected_source_tree_sha256=source_digest,
        expected_parent_freeze_sha256=freeze["combined_sha256"],
    )
    e6_audit = read_json(Path(args.e6_audit).resolve() / "audit_report.json")
    e7_gate = read_json(Path(args.e7).resolve() / "gate_decision.json")
    if (
        e6_audit.get("status") != "passed"
        or e6_audit.get("runs_audited") != 45
        or e7_gate.get("passed") is not True
    ):
        raise RuntimeError("E8 requires passed E6 audit and E7 gate")
    config = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    dataset = yaml.safe_load(
        Path(args.dataset_config).resolve().read_text(encoding="utf-8")
    )
    expected_subjects = [int(value) for value in dataset["confirmatory_subjects"]]
    expected_seeds = [int(value) for value in config["seeds"]]
    channels = resolve_frozen_channel_basis(freeze)
    if unlock["architecture_adaptation"].get("ordered_channels") != channels:
        raise RuntimeError("E8 unlock sensor order differs from the frozen basis")
    input_adapter = unlock["architecture_adaptation"].get("input_channel_adapter")
    if input_adapter != OPENBMI_FIXED_INPUT_ADAPTER:
        raise RuntimeError("E8 unlock input-channel adapter is not the sealed projection")
    environment = _environment()

    if args.worker_phase:
        if args.worker_subject is None:
            raise RuntimeError("E8 worker requires --worker-subject")
        subject = int(args.worker_subject)
        seeds = _csv_values(args.worker_seeds, int)
        if args.worker_phase == "train":
            raw = load_openbmi_subject(subject, sessions=("S1",), resample=1000.0)
            x, y, metadata, access = prepare_openbmi_v8_view(
                raw,
                channel_names=channels,
                input_adapter=input_adapter,
                target_sfreq=250.0,
                stage="openbmi_confirmation",
                role="training",
            )
            rows = [
                _train_one(
                    output=output,
                    source_root=source_root,
                    freeze=freeze,
                    unlock=unlock,
                    config=config,
                    source_tree=source_tree,
                    environment=environment,
                    subject=subject,
                    seed=seed,
                    x_s1=x,
                    y_s1=y,
                    metadata_s1=metadata,
                    s1_manifest=access,
                    device=args.device,
                )
                for seed in seeds
            ]
        else:
            if not args.barrier:
                raise RuntimeError("E8 evaluation worker requires --barrier")
            barrier = _validate_barrier(
                Path(args.barrier),
                source_digest=source_digest,
                unlock_sha256=unlock["combined_sha256"],
                subjects=expected_subjects,
                seeds=expected_seeds,
            )
            raw = load_openbmi_subject(subject, sessions=("S2",), resample=1000.0)
            x, y, metadata, access = prepare_openbmi_v8_view(
                raw,
                channel_names=channels,
                input_adapter=input_adapter,
                target_sfreq=250.0,
                stage="openbmi_confirmation",
                role="evaluation",
            )
            rows = [
                _evaluate_one(
                    output=output,
                    source_root=source_root,
                    freeze=freeze,
                    unlock=unlock,
                    config=config,
                    source_tree=source_tree,
                    environment=environment,
                    barrier=barrier,
                    subject=subject,
                    seed=seed,
                    x_s2=x,
                    y_s2=y,
                    metadata_s2=metadata,
                    s2_manifest=access,
                    device=args.device,
                )
                for seed in seeds
            ]
        print(json.dumps({"status": "worker_completed", "phase": args.worker_phase, "runs": len(rows)}, indent=2))
        return

    subjects = _csv_values(args.subjects, int) or (
        [expected_subjects[0]] if args.canary_s1_only else expected_subjects
    )
    seeds = _csv_values(args.seeds, int) or ([0] if args.canary_s1_only else expected_seeds)
    if not args.canary_s1_only and (subjects != expected_subjects or seeds != expected_seeds):
        raise RuntimeError("formal E8 coverage must match the sealed external unlock")
    write_json(output / "source_tree_manifest.json", source_tree)
    (output / "resolved_campaign.yaml").write_text(
        yaml.safe_dump(
            {
                **config,
                "subjects": subjects,
                "seeds": seeds,
                "source_tree_sha256": source_digest,
                "external_unlock_sha256": unlock["combined_sha256"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    training_rows = _run_parallel(
        args,
        output,
        phase="train",
        subjects=subjects,
        seeds=seeds,
        workers=max(1, int(args.workers)),
    )
    if args.canary_s1_only:
        status = {
            "status": "s1_canary_completed_s2_unopened",
            "stage": "E8_ENSEMBLE_S1_CANARY",
            "runs": len(training_rows),
            "subjects": subjects,
            "seeds": seeds,
            "openbmi_s2_accessed": False,
        }
        write_json(output / "campaign_status.json", status)
        print(json.dumps(status, indent=2))
        return

    records = []
    for subject in subjects:
        for seed in seeds:
            run_dir = output / f"subject_{subject:02d}" / f"seed_{seed}"
            manifest = _validate_training(run_dir)
            records.append(
                {
                    "subject": subject,
                    "seed": seed,
                    "training_manifest_sha256": manifest["combined_sha256"],
                    "checkpoint_file_sha256": {
                        name: file_sha256(run_dir / f"{name}.pt") for name in CHECKPOINTS
                    },
                }
            )
    barrier_body = {
        "schema": "dpc-snn-v8-e8-checkpoint-barrier/v1",
        "created_at": time.time(),
        "source_tree_sha256": source_digest,
        "external_unlock_sha256": unlock["combined_sha256"],
        "subjects": subjects,
        "seeds": seeds,
        "runs": len(records),
        "all_training_complete_before_s2": True,
        "records": records,
    }
    barrier = {**barrier_body, "combined_sha256": sha256_fingerprint(barrier_body)}
    barrier_path = output / "checkpoint_barrier.json"
    write_json(barrier_path, barrier)
    _validate_barrier(
        barrier_path,
        source_digest=source_digest,
        unlock_sha256=unlock["combined_sha256"],
        subjects=subjects,
        seeds=seeds,
    )
    started = time.time()
    rows = _run_parallel(
        args,
        output,
        phase="evaluate",
        subjects=subjects,
        seeds=seeds,
        workers=max(1, int(args.workers)),
        barrier=barrier_path,
    )
    rows = sorted(rows, key=lambda row: (row["subject"], row["seed"]))
    summary_rows = []
    for row in rows:
        for arm in ARMS:
            summary_rows.append(
                {
                    "subject": row["subject"],
                    "seed": row["seed"],
                    "arm": arm,
                    **row["arms"][arm],
                }
            )
    write_csv(output / "summary.csv", summary_rows)
    model_summary = []
    for arm in ARMS:
        subject_means = []
        for subject in subjects:
            values = [
                float(row["arms"][arm]["accuracy"])
                for row in rows
                if int(row["subject"]) == subject
            ]
            subject_means.append(float(np.mean(values)))
        model_summary.append(
            {
                "arm": arm,
                "subject_macro_accuracy": float(np.mean(subject_means)),
                "subject_median_accuracy": float(np.median(subject_means)),
                "subject_standard_deviation": float(np.std(subject_means, ddof=1)),
                "minimum_subject_accuracy": float(np.min(subject_means)),
                "maximum_subject_accuracy": float(np.max(subject_means)),
            }
        )
    write_csv(output / "model_summary.csv", model_summary)
    paired_rows = []
    comparisons = {}
    primary = [
        {"subject": row["subject"], "seed": row["seed"], "accuracy": row["arms"]["primary"]["accuracy"]}
        for row in rows
    ]
    for index, arm in enumerate(("matched_ann", "anchor", "atcnet", "fbcnet")):
        comparator = [
            {"subject": row["subject"], "seed": row["seed"], "accuracy": row["arms"][arm]["accuracy"]}
            for row in rows
        ]
        pairs = pair_subject_seed_rows(comparator, primary, value="accuracy")
        comparisons[f"primary_minus_{arm}"] = paired_delta_summary(
            pairs, seed=20260820 + index
        )
        paired_rows.extend({"comparison": f"primary_minus_{arm}", **pair} for pair in pairs)
    write_csv(output / "paired_subject_seed.csv", paired_rows)
    status = {
        "status": "completed",
        "stage": "E8_ENSEMBLE_CONFIRMATION",
        "protocol": config["protocol"],
        "subjects": subjects,
        "seeds": seeds,
        "runs": len(rows),
        "full_registered_contract": True,
        "all_training_complete_before_s2": True,
        "checkpoint_barrier_sha256": barrier["combined_sha256"],
        "freeze_sha256": freeze["combined_sha256"],
        "external_unlock_sha256": unlock["combined_sha256"],
        "source_tree_sha256": source_digest,
        "comparisons": comparisons,
        "s2_checkpoint_selection": False,
        "s2_gradient_updates": False,
        "openbmi_s2_accessed": True,
        "elapsed_evaluation_seconds": time.time() - started,
    }
    write_json(output / "campaign_status.json", status)
    write_json(output / "freeze_manifest.json", freeze)
    write_json(output / "external_unlock_manifest.json", unlock)
    write_run_artifact_manifest(output, required_files=CAMPAIGN_FILES)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
