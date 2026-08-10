#!/usr/bin/env python3
"""Aggregate E16-A folds and make the preregistered progression decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.utils.io import read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402


def _csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--subjects", default="1,3,8")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--folds", default="0,1,2,3,4,5")
    parser.add_argument("--minimum-accuracy", type=float, default=0.885)
    parser.add_argument("--maximum-teacher-gap", type=float, default=0.005)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    subjects = _csv_ints(args.subjects)
    seeds = _csv_ints(args.seeds)
    folds = _csv_ints(args.folds)
    rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for subject in subjects:
        for seed in seeds:
            labels: list[np.ndarray] = []
            student: list[np.ndarray] = []
            teacher: list[np.ndarray] = []
            v9: list[np.ndarray] = []
            indices: list[np.ndarray] = []
            for fold in folds:
                fold_dir = root / f"subject_{subject:02d}" / f"seed_{seed}" / f"fold_{fold}"
                if not (fold_dir / "campaign_status.json").is_file():
                    missing.append(str(fold_dir))
                    continue
                status = read_json(fold_dir / "campaign_status.json")
                if status.get("status") != "completed" or status.get("session_e_accessed") is not False:
                    raise RuntimeError(f"invalid E16-A fold status: {fold_dir}")
                result = read_json(fold_dir / "result.json")
                rows.append(result)
                with np.load(fold_dir / "outer_predictions.npz", allow_pickle=False) as archive:
                    labels.append(archive["labels"])
                    indices.append(archive["indices"])
                    student.append(archive["logits"])
                    teacher.append(archive["teacher_logits"])
                    v9.append(archive["v9_logits"])
            if len(labels) != len(folds):
                continue
            all_indices = np.concatenate(indices)
            if len(np.unique(all_indices)) != len(all_indices):
                raise RuntimeError(f"outer folds overlap for subject {subject}, seed {seed}")
            all_labels = np.concatenate(labels)

            def metrics(values: list[np.ndarray]) -> dict[str, Any]:
                logits = np.concatenate(values)
                return classification_metrics(all_labels, logits.argmax(axis=1), n_classes=4)

            student_metrics = metrics(student)
            teacher_metrics = metrics(teacher)
            v9_metrics = metrics(v9)
            subject_rows.append(
                {
                    "subject": subject,
                    "seed": seed,
                    "trials": len(all_labels),
                    "student_accuracy": student_metrics["accuracy"],
                    "student_kappa": student_metrics["kappa"],
                    "teacher_accuracy": teacher_metrics["accuracy"],
                    "v9_accuracy": v9_metrics["accuracy"],
                    "delta_vs_teacher_pp": 100.0
                    * (student_metrics["accuracy"] - teacher_metrics["accuracy"]),
                    "delta_vs_v9_pp": 100.0
                    * (student_metrics["accuracy"] - v9_metrics["accuracy"]),
                }
            )
    write_csv(root / "fold_summary.csv", rows)
    write_csv(root / "subject_summary.csv", subject_rows)
    if missing:
        status = {
            "status": "incomplete",
            "completed_folds": len(rows),
            "required_folds": len(subjects) * len(seeds) * len(folds),
            "missing": missing,
        }
        write_json(root / "aggregate_status.json", status)
        print(json.dumps(status, indent=2))
        raise SystemExit(2)

    mean_student = float(np.mean([row["student_accuracy"] for row in subject_rows]))
    mean_teacher = float(np.mean([row["teacher_accuracy"] for row in subject_rows]))
    mean_v9 = float(np.mean([row["v9_accuracy"] for row in subject_rows]))
    accuracy_pass = mean_student >= float(args.minimum_accuracy)
    teacher_gap_pass = mean_teacher - mean_student <= float(args.maximum_teacher_gap)
    gate_passed = accuracy_pass or teacher_gap_pass
    decision = {
        "status": "completed",
        "stage": "E16-A-continuous-information-gate",
        "mean_student_accuracy": mean_student,
        "mean_teacher_accuracy": mean_teacher,
        "mean_v9_accuracy": mean_v9,
        "delta_vs_teacher_pp": 100.0 * (mean_student - mean_teacher),
        "delta_vs_v9_pp": 100.0 * (mean_student - mean_v9),
        "minimum_accuracy": float(args.minimum_accuracy),
        "maximum_teacher_gap": float(args.maximum_teacher_gap),
        "accuracy_pass": accuracy_pass,
        "teacher_gap_pass": teacher_gap_pass,
        "gate_passed": gate_passed,
        "next_stage": "E16-B-matched-ANN-SNN" if gate_passed else "stop-and-repair-front-end",
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    write_json(root / "gate_decision.json", decision)
    write_json(root / "aggregate_status.json", decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
