#!/usr/bin/env python3
"""Assemble one complete V7 E3 subject/seed OOF result from fold shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    write_trial_predictions,
)
from dpc_snn.experiments.v7_delay import paired_delay_gate  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import cohen_kappa, paired_prediction_comparison  # noqa: E402


def _parse_fold_ids(value: str) -> list[int]:
    fold_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not fold_ids or len(fold_ids) != len(set(fold_ids)):
        raise ValueError("expected unique comma-separated fold ids")
    return sorted(fold_ids)


def _subject_file(data_root: Path, subject: int) -> Path:
    candidates = sorted(data_root.glob(f"*A{subject:02d}*.npz"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one processed file for subject {subject}, found {len(candidates)}"
        )
    return candidates[0]


def _comparison_signature(resolved: dict[str, Any]) -> dict[str, Any]:
    training = resolved["training"]
    return {
        "stage": resolved["stage"],
        "protocol": resolved["protocol"],
        "architecture_version": resolved["architecture_version"],
        "max_epochs": resolved["max_epochs"],
        "patience": resolved["patience"],
        "fixed_epoch": resolved["fixed_epoch_no_outer_selection"],
        "scheduler_horizon_epochs": resolved["scheduler_horizon_epochs"],
        "training": {
            key: training.get(key)
            for key in (
                "batch_size",
                "gradient_accumulation_steps",
                "effective_batch_size",
                "learning_rate",
                "weight_decay",
                "beta1",
                "scheduler_warmup_epochs",
                "schedule",
                "train_readout",
                "auxiliary_endpoint_weights",
                "readout_training_scope",
                "atc_core_learning_rate_multiplier",
            )
        },
        "augmentation": resolved["augmentation"],
        "control": resolved["control"],
        "model_overrides": resolved["model_overrides"],
    }


def _fold_directories(
    roots: list[Path], *, subject: int, seed: int
) -> dict[int, Path]:
    folds: dict[int, Path] = {}
    for root in roots:
        campaign_status = root / "campaign_status.json"
        if not campaign_status.is_file():
            raise FileNotFoundError(f"missing shard campaign status: {campaign_status}")
        status = read_json(campaign_status)
        if status.get("status") != "completed" or bool(
            status.get("session_e_accessed", True)
        ):
            raise RuntimeError(f"invalid or Session-E-contaminated shard: {root}")
        seed_dir = root / f"subject_{subject:02d}" / f"seed_{seed}"
        for fold_dir in sorted(seed_dir.glob("fold_*")):
            fold = int(fold_dir.name.split("_", maxsplit=1)[1])
            if fold in folds:
                raise RuntimeError(f"duplicate fold {fold} across E3 shards")
            folds[fold] = fold_dir
    return folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--subject", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--expected-folds", default="0,1,2,3,4,5")
    parser.add_argument(
        "--config",
        default="configs/experiments/v7_e3_static_slow_within.yaml",
    )
    args = parser.parse_args()

    roots = [Path(value).resolve() for value in args.inputs]
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output = ensure_dir(output)
    expected_folds = _parse_fold_ids(args.expected_folds)
    folds = _fold_directories(roots, subject=args.subject, seed=args.seed)
    if sorted(folds) != expected_folds:
        raise RuntimeError(
            f"fold coverage mismatch: expected={expected_folds}, observed={sorted(folds)}"
        )

    subject_path = _subject_file(Path(args.data).resolve(), args.subject)
    data = load_processed_npz(subject_path)
    session = np.asarray(data["session"]).astype(str)
    t_indices = np.flatnonzero(session == "T")
    labels_t = np.asarray(data["y"], dtype=np.int64)[t_indices]
    runs_t = np.asarray(data["run"])[t_indices]
    trial_ids_t = np.asarray(data["trial_id"])[t_indices]
    subjects_t = np.asarray(data["subject"])[t_indices]

    prediction_parts: list[dict[str, np.ndarray]] = []
    fold_manifest: list[dict[str, Any]] = []
    reference_signature: dict[str, Any] | None = None
    reference_fingerprint_components: dict[str, str] | None = None
    amplitude_differences = []
    for fold, fold_dir in sorted(folds.items()):
        required = {
            "result": fold_dir / "result.json",
            "runtime_status": fold_dir / "runtime_status.json",
            "predictions": fold_dir / "paired_validation_predictions.npz",
            "resolved_config": fold_dir / "resolved_config.yaml",
            "source_fingerprint": fold_dir / "source_fingerprint.json",
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"fold {fold} is incomplete: {missing}")

        result = read_json(required["result"])
        runtime_status = read_json(required["runtime_status"])
        if result.get("status") != "completed" or runtime_status.get("status") != "completed":
            raise RuntimeError(f"fold {fold} did not complete")
        if (
            int(result["subject"]) != args.subject
            or int(result["seed"]) != args.seed
            or int(result["fold"]) != fold
        ):
            raise RuntimeError(f"fold {fold} result identity mismatch")
        required_invariants = {
            "session_e_accessed": False,
            "checkpoint_state_unchanged_by_control": True,
            "checkpoint_state_unchanged_by_full_evaluation": True,
            "delay_checkpoint_unchanged_by_training": True,
        }
        for key, expected in required_invariants.items():
            if result.get(key) is not expected:
                raise RuntimeError(f"fold {fold} failed invariant {key}")
        if not bool(result.get("fold_local_prior", {}).get("evidence_pipeline_passed")):
            raise RuntimeError(f"fold {fold} used an unvalidated delay prior")

        resolved = yaml.safe_load(required["resolved_config"].read_text(encoding="utf-8"))
        signature = _comparison_signature(resolved)
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise RuntimeError(f"fold {fold} comparison configuration differs")

        fingerprint = read_json(required["source_fingerprint"])
        stable_components = {
            key: str(fingerprint["components"][key])
            for key in ("source", "data", "augmentation", "environment")
        }
        if reference_fingerprint_components is None:
            reference_fingerprint_components = stable_components
        elif stable_components != reference_fingerprint_components:
            raise RuntimeError(f"fold {fold} stable fingerprint components differ")

        with np.load(required["predictions"], allow_pickle=False) as archive:
            part = {key: archive[key] for key in archive.files}
        indices = np.asarray(part["indices"], dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= len(labels_t)):
            raise RuntimeError(f"fold {fold} contains out-of-range trial indices")
        if not np.array_equal(np.asarray(part["labels"], dtype=np.int64), labels_t[indices]):
            raise RuntimeError(f"fold {fold} prediction labels differ from the dataset")
        prediction_parts.append(part)
        amplitude_differences.append(float(result["amplitude_relative_difference"]))
        fold_manifest.append(
            {
                "fold": fold,
                "root": str(fold_dir),
                "result_sha256": file_sha256(required["result"]),
                "predictions_sha256": file_sha256(required["predictions"]),
                "resolved_config_sha256": file_sha256(required["resolved_config"]),
                "source_fingerprint_sha256": file_sha256(required["source_fingerprint"]),
                "run_fingerprint": result["run_fingerprint"],
                "prior_sha256": result["fold_local_prior"]["prior_sha256"],
                "fixed_epoch_policy": resolved["fixed_epoch_selection_policy"],
            }
        )

    indices = np.concatenate([part["indices"] for part in prediction_parts])
    order = np.argsort(indices)
    indices = indices[order]
    expected_indices = np.arange(len(labels_t), dtype=np.int64)
    if not np.array_equal(indices, expected_indices):
        raise RuntimeError("fold shards do not form one complete, unique Session-T OOF set")
    full_logits = np.concatenate([part["full_logits"] for part in prediction_parts])[order]
    zero_logits = np.concatenate(
        [part["locked_zero_logits"] for part in prediction_parts]
    )[order]
    full_pred = full_logits.argmax(axis=1)
    zero_pred = zero_logits.argmax(axis=1)
    full_accuracy = float(np.mean(full_pred == labels_t))
    zero_accuracy = float(np.mean(zero_pred == labels_t))
    pair = {
        "stage": str(reference_signature["stage"]),
        "subject": args.subject,
        "seed": args.seed,
        "full_accuracy": full_accuracy,
        "locked_zero_accuracy": zero_accuracy,
        "delta_accuracy": full_accuracy - zero_accuracy,
        "full_kappa": cohen_kappa(labels_t, full_pred),
        "locked_zero_kappa": cohen_kappa(labels_t, zero_pred),
        "amplitude_relative_difference": max(amplitude_differences),
        "folds": len(expected_folds),
        "fold_ids": expected_folds,
        "complete_oof": True,
        "session_e_accessed": False,
        **paired_prediction_comparison(labels_t, full_pred, zero_pred),
    }

    seed_dir = ensure_dir(output / f"subject_{args.subject:02d}" / f"seed_{args.seed}")
    write_json(seed_dir / "paired_oof_metrics.json", pair)
    write_trial_predictions(
        seed_dir / "full_oof",
        logits=full_logits,
        probabilities=np.exp(full_logits - full_logits.max(axis=1, keepdims=True))
        / np.exp(full_logits - full_logits.max(axis=1, keepdims=True)).sum(
            axis=1, keepdims=True
        ),
        pred=full_pred,
        label=labels_t,
        subject=subjects_t,
        session="T",
        run=runs_t,
        trial_id=trial_ids_t,
        seed=args.seed,
        model=f"dasp_v7_{pair['stage']}_full",
    )
    write_trial_predictions(
        seed_dir / "locked_zero_oof",
        logits=zero_logits,
        probabilities=np.exp(zero_logits - zero_logits.max(axis=1, keepdims=True))
        / np.exp(zero_logits - zero_logits.max(axis=1, keepdims=True)).sum(
            axis=1, keepdims=True
        ),
        pred=zero_pred,
        label=labels_t,
        subject=subjects_t,
        session="T",
        run=runs_t,
        trial_id=trial_ids_t,
        seed=args.seed,
        model=f"dasp_v7_{pair['stage']}_locked_zero",
    )
    write_csv(output / "paired_summary.csv", [pair])

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gate = paired_delay_gate([pair], config)
    write_json(output / "gate_report.json", gate)
    write_json(
        output / "aggregation_manifest.json",
        {
            "status": "completed",
            "aggregation_type": "complete_oof_from_fold_shards",
            "aggregator": str(Path(__file__).resolve()),
            "aggregator_sha256": file_sha256(Path(__file__)),
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "data": str(subject_path),
            "data_sha256": file_sha256(subject_path),
            "comparison_signature": reference_signature,
            "stable_fingerprint_components": reference_fingerprint_components,
            "folds": fold_manifest,
            "trial_count": len(labels_t),
            "session_e_accessed": False,
        },
    )
    write_json(
        output / "campaign_status.json",
        {
            "status": "completed",
            "stage": pair["stage"],
            "subjects": [args.subject],
            "seeds": [args.seed],
            "session_e_accessed": False,
            "gate": gate,
        },
    )
    print(json.dumps({"paired_oof": pair, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
