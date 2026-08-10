#!/usr/bin/env python3
"""Aggregate the matched E12 distillation-objective pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v12_multirate_training import (  # noqa: E402
    V12_E12_OBJECTIVES,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402


ANCHOR = "v9_sew_clif_kd"
TEACHER = "equal_teacher"
CONTROL = "o0_current"


def _csv_int(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError("integer list must be non-empty and unique")
    return parsed


def _csv_objectives(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError("objective list must be non-empty and unique")
    unknown = set(parsed).difference(V12_E12_OBJECTIVES)
    if unknown or CONTROL not in parsed:
        raise ValueError(f"invalid E12 objectives: {sorted(unknown)}")
    return parsed


def _prediction(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "indices",
            "logits",
            "labels",
            "equal_teacher_logits",
            "atc_teacher_logits",
            "fbc_teacher_logits",
        }
        if set(archive.files) != required:
            raise RuntimeError(f"invalid E12 prediction archive: {path}")
        indices = np.asarray(archive["indices"], dtype=np.int64)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        logits = np.asarray(archive["logits"], dtype=np.float64)
        teacher = np.asarray(archive["equal_teacher_logits"], dtype=np.float64)
    if logits.shape != (indices.size, 4) or teacher.shape != logits.shape:
        raise RuntimeError(f"malformed E12 prediction archive: {path}")
    return indices, labels, logits, teacher


def _anchor(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        indices = np.asarray(archive["indices"], dtype=np.int64)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        logits = np.asarray(archive["logits"], dtype=np.float64)
    return indices, labels, logits


def _metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    prediction = logits.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "kappa": float(cohen_kappa_score(labels, prediction)),
    }


def _softmax(logits: np.ndarray, temperature: float = 2.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    probability = np.exp(scaled)
    return probability / probability.sum(axis=1, keepdims=True)


def _diagnostics(
    labels: np.ndarray, student: np.ndarray, teacher: np.ndarray
) -> dict[str, float]:
    student_prediction = student.argmax(axis=1)
    teacher_prediction = teacher.argmax(axis=1)
    student_correct = student_prediction == labels
    teacher_correct = teacher_prediction == labels
    teacher_probability = _softmax(teacher)
    student_probability = _softmax(student)
    kl = np.sum(
        teacher_probability
        * (
            np.log(np.clip(teacher_probability, 1e-12, 1.0))
            - np.log(np.clip(student_probability, 1e-12, 1.0))
        ),
        axis=1,
    )
    return {
        "teacher_student_kl": float(np.mean(kl)),
        "teacher_student_disagreement": float(
            np.mean(teacher_prediction != student_prediction)
        ),
        "teacher_rescue_rate": float(np.mean(teacher_correct & ~student_correct)),
        "student_rescue_rate": float(np.mean(student_correct & ~teacher_correct)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--anchor-root", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--subjects", default="1,3,8")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--folds", default="0,1,2,3,4,5")
    parser.add_argument("--objectives", default=",".join(V12_E12_OBJECTIVES))
    parser.add_argument("--expected-source-digest", required=True)
    parser.add_argument("--minimum-gain-pp", type=float, default=0.5)
    parser.add_argument("--minimum-positive-pairs", type=int, default=2)
    parser.add_argument("--maximum-regression-pp", type=float, default=1.0)
    parser.add_argument("--minimum-gap-recovery", type=float, default=0.4)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    anchor_root = Path(args.anchor_root).resolve()
    output = ensure_dir(Path(args.output).resolve() if args.output else root / "aggregate")
    subjects = _csv_int(args.subjects)
    seeds = _csv_int(args.seeds)
    folds = _csv_int(args.folds)
    objectives = _csv_objectives(args.objectives)
    fold_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    fingerprints: list[str] = []
    source_digests: list[str] = []

    for subject in subjects:
        for seed in seeds:
            seen: list[int] = []
            for fold in folds:
                fold_dir = root / f"subject_{subject:02d}" / f"seed_{seed}" / f"fold_{fold}"
                status = read_json(fold_dir / "campaign_status.json")
                replay = read_json(fold_dir / "reference_replay.json")
                if (
                    status.get("status") != "completed"
                    or status.get("session_e_accessed") is not False
                    or status.get("openbmi_s2_accessed") is not False
                    or status.get("objectives") != list(objectives)
                    or status.get("total_hpo_configurations") != 0
                    or replay.get("status") != "passed"
                ):
                    raise RuntimeError(f"incomplete or invalid E12 fold: {fold_dir}")
                source_digests.append(
                    str(read_json(fold_dir / "source_tree_summary.json")["sha256"])
                )
                fingerprints.append(
                    str(read_json(fold_dir / "run_fingerprint.json")["combined_sha256"])
                )
                predictions: dict[str, np.ndarray] = {}
                indices: np.ndarray | None = None
                labels: np.ndarray | None = None
                teacher: np.ndarray | None = None
                summary = pd.read_csv(fold_dir / "summary.csv")
                if set(summary["objective"]) != set(objectives):
                    raise RuntimeError(f"E12 summary objectives are incomplete: {fold_dir}")
                for objective in objectives:
                    saved_indices, saved_labels, logits, saved_teacher = _prediction(
                        fold_dir / objective / "outer_predictions.npz"
                    )
                    if indices is None:
                        indices, labels, teacher = saved_indices, saved_labels, saved_teacher
                    elif (
                        not np.array_equal(indices, saved_indices)
                        or not np.array_equal(labels, saved_labels)
                        or not np.array_equal(teacher, saved_teacher)
                    ):
                        raise RuntimeError(f"E12 predictions are not aligned: {fold_dir}")
                    predictions[objective] = logits
                    fold_rows.append(
                        {
                            "subject": subject,
                            "seed": seed,
                            "fold": fold,
                            "objective": objective,
                            **_metrics(saved_labels, logits),
                            **_diagnostics(saved_labels, logits, saved_teacher),
                        }
                    )
                assert indices is not None and labels is not None and teacher is not None
                anchor_indices, anchor_labels, anchor_logits = _anchor(
                    anchor_root
                    / f"subject_{subject:02d}"
                    / f"seed_{seed}"
                    / f"fold_{fold}"
                    / "sew_clif_kd"
                    / "outer_predictions.npz"
                )
                if not np.array_equal(indices, anchor_indices) or not np.array_equal(
                    labels, anchor_labels
                ):
                    raise RuntimeError("E12 and V9 anchor predictions are not aligned")
                predictions[ANCHOR] = anchor_logits
                predictions[TEACHER] = teacher
                seen.extend(indices.tolist())
                for offset, trial_index in enumerate(indices):
                    trial_rows.append(
                        {
                            "subject": subject,
                            "seed": seed,
                            "fold": fold,
                            "trial_index": int(trial_index),
                            "label": int(labels[offset]),
                            **{
                                f"prediction_{name}": int(logits[offset].argmax())
                                for name, logits in predictions.items()
                            },
                        }
                    )
            counts = Counter(seen)
            if len(counts) != 288 or set(counts.values()) != {1}:
                raise RuntimeError("E12 outer folds do not cover Session T exactly once")

    expected_runs = len(subjects) * len(seeds) * len(folds)
    if set(source_digests) != {args.expected_source_digest}:
        raise RuntimeError("E12 source snapshot mismatch")
    if len(set(fingerprints)) != expected_runs:
        raise RuntimeError("E12 fingerprints are duplicated or incomplete")

    trial_frame = pd.DataFrame(trial_rows).sort_values(
        ["subject", "seed", "trial_index"]
    )
    subject_seed_rows: list[dict[str, Any]] = []
    all_models = (*objectives, ANCHOR, TEACHER)
    for (subject, seed), group in trial_frame.groupby(["subject", "seed"], sort=True):
        labels = group["label"].to_numpy()
        for model in all_models:
            prediction = group[f"prediction_{model}"].to_numpy()
            subject_seed_rows.append(
                {
                    "subject": int(subject),
                    "seed": int(seed),
                    "model": model,
                    "accuracy": float(accuracy_score(labels, prediction)),
                    "kappa": float(cohen_kappa_score(labels, prediction)),
                }
            )
    subject_seed = pd.DataFrame(subject_seed_rows)
    model_summary = (
        subject_seed.groupby("model", as_index=False)
        .agg(
            subject_seed_pairs=("accuracy", "size"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            kappa_mean=("kappa", "mean"),
            kappa_std=("kappa", "std"),
        )
        .sort_values("accuracy_mean", ascending=False)
    )
    anchor_values = subject_seed.loc[subject_seed["model"] == ANCHOR].sort_values(
        ["subject", "seed"]
    )["accuracy"].to_numpy()
    teacher_values = subject_seed.loc[subject_seed["model"] == TEACHER].sort_values(
        ["subject", "seed"]
    )["accuracy"].to_numpy()
    teacher_gap = float(teacher_values.mean() - anchor_values.mean())
    comparison_rows: list[dict[str, Any]] = []
    for objective in objectives:
        values = subject_seed.loc[subject_seed["model"] == objective].sort_values(
            ["subject", "seed"]
        )["accuracy"].to_numpy()
        delta = 100.0 * (values - anchor_values)
        comparison_rows.append(
            {
                "objective": objective,
                "reference": ANCHOR,
                "mean_delta_pp": float(delta.mean()),
                "median_delta_pp": float(np.median(delta)),
                "minimum_delta_pp": float(delta.min()),
                "positive_pairs": int(np.sum(delta > 0.0)),
                "negative_pairs": int(np.sum(delta < 0.0)),
                "tied_pairs": int(np.sum(delta == 0.0)),
                "teacher_gap_recovery": (
                    float((values.mean() - anchor_values.mean()) / teacher_gap)
                    if teacher_gap > 0.0
                    else float("nan")
                ),
            }
        )
    comparisons = pd.DataFrame(comparison_rows).sort_values(
        ["mean_delta_pp", "median_delta_pp"], ascending=False
    )
    candidates = comparisons.loc[comparisons["objective"] != CONTROL]
    selected_row = candidates.iloc[0]
    promotion_passed = bool(
        float(selected_row["mean_delta_pp"]) >= float(args.minimum_gain_pp)
        and int(selected_row["positive_pairs"]) >= int(args.minimum_positive_pairs)
        and float(selected_row["minimum_delta_pp"])
        >= -float(args.maximum_regression_pp)
        and float(selected_row["teacher_gap_recovery"])
        >= float(args.minimum_gap_recovery)
    )
    promotion = {
        "status": "passed" if promotion_passed else "failed",
        "selected_objective": str(selected_row["objective"]),
        "mean_delta_pp": float(selected_row["mean_delta_pp"]),
        "median_delta_pp": float(selected_row["median_delta_pp"]),
        "minimum_delta_pp": float(selected_row["minimum_delta_pp"]),
        "positive_pairs": int(selected_row["positive_pairs"]),
        "teacher_gap_pp": 100.0 * teacher_gap,
        "teacher_gap_recovery": float(selected_row["teacher_gap_recovery"]),
        "criteria": {
            "minimum_mean_gain_pp": float(args.minimum_gain_pp),
            "minimum_positive_pairs": int(args.minimum_positive_pairs),
            "maximum_regression_pp": float(args.maximum_regression_pp),
            "minimum_teacher_gap_recovery": float(args.minimum_gap_recovery),
        },
    }

    per_class_rows: list[dict[str, Any]] = []
    for model in all_models:
        prediction = trial_frame[f"prediction_{model}"].to_numpy()
        labels = trial_frame["label"].to_numpy()
        for target in range(4):
            mask = labels == target
            per_class_rows.append(
                {
                    "model": model,
                    "class": target,
                    "trials": int(mask.sum()),
                    "accuracy": float(np.mean(prediction[mask] == labels[mask])),
                }
            )

    write_csv(output / "fold_metrics.csv", fold_rows)
    write_csv(output / "trial_predictions.csv", trial_frame.to_dict(orient="records"))
    write_csv(
        output / "subject_seed_metrics.csv", subject_seed.to_dict(orient="records")
    )
    write_csv(output / "model_summary.csv", model_summary.to_dict(orient="records"))
    write_csv(output / "paired_comparisons.csv", comparisons.to_dict(orient="records"))
    write_csv(output / "per_class_accuracy.csv", per_class_rows)
    audit = {
        "status": "passed",
        "expected_runs": expected_runs,
        "completed_runs": expected_runs,
        "subjects": list(subjects),
        "seeds": list(seeds),
        "folds": list(folds),
        "objectives": list(objectives),
        "source_tree_sha256": args.expected_source_digest,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "unique_run_fingerprints": len(set(fingerprints)),
        "reference_replay_passed_folds": expected_runs,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    write_json(output / "campaign_audit.json", audit)
    summary = {"status": "completed", "audit": audit, "promotion": promotion}
    write_json(output / "aggregate_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
