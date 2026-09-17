#!/usr/bin/env python3
"""Diagnose the V10 teacher-to-student gap from audited multi-seed predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


PRIMARY_MODELS = ("sew_clif", "ann_gru", "ann_tcn", "ann_lstm", "equal_teacher")


def _probabilities(frame: pd.DataFrame, model: str) -> np.ndarray:
    columns = [f"probability_{model}_{index}" for index in range(4)]
    if not set(columns).issubset(frame.columns):
        raise RuntimeError(f"missing probability columns for {model}")
    probabilities = frame[columns].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(probabilities)) or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-6
    ):
        raise RuntimeError(f"invalid probability simplex for {model}")
    return probabilities


def _ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == labels
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        lower, upper = boundaries[index : index + 2]
        mask = (confidence > lower) & (confidence <= upper)
        if index == 0:
            mask |= confidence == lower
        if np.any(mask):
            value += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return value


def _metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    prediction = probabilities.argmax(axis=1)
    one_hot = np.eye(4, dtype=np.float64)[labels]
    return {
        "accuracy": float(np.mean(prediction == labels)),
        "negative_log_likelihood": float(
            -np.log(np.clip(probabilities[np.arange(labels.size), labels], 1e-12, 1.0)).mean()
        ),
        "brier_score": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece": _ece(probabilities, labels),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    aggregate = Path(args.aggregate).resolve()
    output = Path(args.output).resolve() if args.output else aggregate / "e11_diagnosis"
    output.mkdir(parents=True, exist_ok=True)
    audit = json.loads((aggregate / "campaign_audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("session_e_accessed") is not False:
        raise RuntimeError("E11 requires a passed, Session-E-locked E10.2 aggregate")
    frame = pd.read_csv(aggregate / "trial_predictions.csv")
    expected_rows = len(audit["subjects"]) * len(audit["seeds"]) * 288
    if len(frame) != expected_rows:
        raise RuntimeError("E10.2 trial coverage is incomplete")
    if frame.duplicated(["subject", "seed", "trial_index"]).any():
        raise RuntimeError("duplicate subject-seed-trial rows in E10.2 predictions")

    metric_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    for (subject, seed), group in frame.groupby(["subject", "seed"], sort=True):
        labels = group["label"].to_numpy(dtype=np.int64)
        for model in PRIMARY_MODELS:
            probabilities = _probabilities(group, model)
            prediction = probabilities.argmax(axis=1)
            metric_rows.append(
                {
                    "subject": int(subject),
                    "seed": int(seed),
                    "model": model,
                    **_metrics(probabilities, labels),
                }
            )
            matrix = confusion_matrix(labels, prediction, labels=np.arange(4))
            for label in range(4):
                class_rows.append(
                    {
                        "subject": int(subject),
                        "seed": int(seed),
                        "model": model,
                        "class": label,
                        "support": int(matrix[label].sum()),
                        "recall": float(
                            matrix[label, label] / max(int(matrix[label].sum()), 1)
                        ),
                    }
                )
                for predicted in range(4):
                    confusion_rows.append(
                        {
                            "subject": int(subject),
                            "seed": int(seed),
                            "model": model,
                            "label": label,
                            "prediction": predicted,
                            "count": int(matrix[label, predicted]),
                        }
                    )

    metrics = pd.DataFrame(metric_rows)
    classes = pd.DataFrame(class_rows)
    confusions = pd.DataFrame(confusion_rows)
    subject_metrics = (
        metrics.groupby(["subject", "model"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            negative_log_likelihood=("negative_log_likelihood", "mean"),
            brier_score=("brier_score", "mean"),
            ece=("ece", "mean"),
            mean_confidence=("mean_confidence", "mean"),
        )
        .sort_values(["subject", "model"])
    )

    comparison_rows: list[dict[str, object]] = []
    for comparator in ("ann_gru", "ann_tcn", "ann_lstm", "equal_teacher"):
        for (subject, seed), group in frame.groupby(["subject", "seed"], sort=True):
            labels = group["label"].to_numpy(dtype=np.int64)
            snn = group["prediction_sew_clif"].to_numpy(dtype=np.int64)
            other = group[f"prediction_{comparator}"].to_numpy(dtype=np.int64)
            snn_correct = snn == labels
            other_correct = other == labels
            comparison_rows.append(
                {
                    "subject": int(subject),
                    "seed": int(seed),
                    "first": "sew_clif",
                    "second": comparator,
                    "both_correct": int(np.sum(snn_correct & other_correct)),
                    "snn_only_correct": int(np.sum(snn_correct & ~other_correct)),
                    "second_only_correct": int(np.sum(~snn_correct & other_correct)),
                    "both_wrong": int(np.sum(~snn_correct & ~other_correct)),
                    "prediction_disagreement": int(np.sum(snn != other)),
                    "oracle_accuracy": float(np.mean(snn_correct | other_correct)),
                }
            )
    comparisons = pd.DataFrame(comparison_rows)

    trial_difficulty_rows: list[dict[str, object]] = []
    for (subject, trial_index), group in frame.groupby(
        ["subject", "trial_index"], sort=True
    ):
        labels = group["label"].to_numpy(dtype=np.int64)
        if np.unique(labels).size != 1:
            raise RuntimeError("trial labels differ between seeds")
        row: dict[str, object] = {
            "subject": int(subject),
            "trial_index": int(trial_index),
            "label": int(labels[0]),
        }
        for model in PRIMARY_MODELS:
            prediction = group[f"prediction_{model}"].to_numpy(dtype=np.int64)
            row[f"correct_seeds_{model}"] = int(np.sum(prediction == labels))
        trial_difficulty_rows.append(row)
    difficulty = pd.DataFrame(trial_difficulty_rows)

    teacher_accuracy = float(
        subject_metrics.loc[subject_metrics["model"] == "equal_teacher", "accuracy"].mean()
    )
    snn_accuracy = float(
        subject_metrics.loc[subject_metrics["model"] == "sew_clif", "accuracy"].mean()
    )
    per_subject_gap: list[dict[str, float | int]] = []
    for subject in sorted(frame["subject"].unique()):
        rows = subject_metrics.loc[subject_metrics["subject"] == subject].set_index("model")
        per_subject_gap.append(
            {
                "subject": int(subject),
                "teacher_minus_snn_pp": 100.0
                * float(rows.loc["equal_teacher", "accuracy"] - rows.loc["sew_clif", "accuracy"]),
                "snn_minus_gru_pp": 100.0
                * float(rows.loc["sew_clif", "accuracy"] - rows.loc["ann_gru", "accuracy"]),
            }
        )

    metrics.to_csv(output / "subject_seed_metrics.csv", index=False)
    subject_metrics.to_csv(output / "subject_metrics.csv", index=False)
    classes.to_csv(output / "class_recall.csv", index=False)
    confusions.to_csv(output / "confusion_counts.csv", index=False)
    comparisons.to_csv(output / "paired_error_decomposition.csv", index=False)
    difficulty.to_csv(output / "trial_difficulty.csv", index=False)
    pd.DataFrame(per_subject_gap).to_csv(output / "subject_gap_summary.csv", index=False)
    capacity = pd.read_csv(aggregate / "end_to_end_capacity.csv")
    capacity.to_csv(output / "end_to_end_capacity.csv", index=False)

    summary = {
        "status": "completed",
        "teacher_accuracy": teacher_accuracy,
        "snn_accuracy": snn_accuracy,
        "teacher_minus_snn_pp": 100.0 * (teacher_accuracy - snn_accuracy),
        "per_subject_gap": per_subject_gap,
        "largest_gap_subject": int(
            max(per_subject_gap, key=lambda row: float(row["teacher_minus_snn_pp"]))[
                "subject"
            ]
        ),
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_campaign_audit_sha256": hashlib.sha256(
            (aggregate / "campaign_audit.json").read_bytes()
        ).hexdigest(),
        "session_e_accessed": False,
        "interpretation_scope": "descriptive Session-T development diagnosis",
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
