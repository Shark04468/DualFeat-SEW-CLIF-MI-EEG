#!/usr/bin/env python3
"""Aggregate and audit the V10 E1 strong-control screen."""

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
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import accuracy_score, cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.models.v10_strong_controls import V10_STRONG_CONTROL_MODELS  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402


ANCHORS = ("v9_ann_sew_kd", "v9_sew_clif_kd")


def _csv_int(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError("integer list must be non-empty and unique")
    return parsed


def _prediction(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"indices", "logits", "labels"}:
            raise RuntimeError(f"invalid prediction archive: {path}")
        indices = np.asarray(archive["indices"], dtype=np.int64)
        logits = np.asarray(archive["logits"], dtype=np.float32)
        labels = np.asarray(archive["labels"], dtype=np.int64)
    if logits.shape != (indices.size, 4) or labels.shape != indices.shape:
        raise RuntimeError(f"malformed prediction archive: {path}")
    return indices, labels, logits.argmax(axis=1)


def _metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "kappa": float(cohen_kappa_score(labels, prediction)),
    }


def _paired(delta: np.ndarray) -> dict[str, float | int]:
    nonzero = np.asarray(delta, dtype=np.float64)
    nonzero = nonzero[nonzero != 0.0]
    positive = int(np.sum(nonzero > 0.0))
    negative = int(np.sum(nonzero < 0.0))
    return {
        "positive_subjects": positive,
        "negative_subjects": negative,
        "ties": int(len(delta) - len(nonzero)),
        "sign_test_p_greater": (
            float(binomtest(positive, len(nonzero), 0.5, alternative="greater").pvalue)
            if len(nonzero)
            else 1.0
        ),
        "wilcoxon_p_greater": (
            float(wilcoxon(nonzero, alternative="greater").pvalue) if len(nonzero) else 1.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--anchor-root", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--subjects", default="1,3,8")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", default="0,1,2,3,4,5")
    parser.add_argument("--expected-source-digest", required=True)
    args = parser.parse_args()

    campaign_root = Path(args.root).resolve()
    anchor_root = Path(args.anchor_root).resolve()
    output = ensure_dir(Path(args.output).resolve() if args.output else campaign_root / "aggregate")
    subjects = _csv_int(args.subjects)
    folds = _csv_int(args.folds)
    models = tuple(V10_STRONG_CONTROL_MODELS)
    all_models = (*models, *ANCHORS)
    fold_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    hpo_rows: list[dict[str, Any]] = []
    fingerprints: list[str] = []
    source_digests: list[str] = []

    for subject in subjects:
        seen: list[int] = []
        for fold in folds:
            fold_dir = (
                campaign_root
                / f"subject_{subject:02d}"
                / f"seed_{args.seed}"
                / f"fold_{fold}"
            )
            status = read_json(fold_dir / "campaign_status.json")
            if status.get("status") != "completed" or status.get("session_e_accessed") is not False:
                raise RuntimeError(f"incomplete or unlocked V10 fold: {fold_dir}")
            if int(status.get("hpo_configurations_per_model", -1)) != 6:
                raise RuntimeError(f"V10 HPO budget differs from six configurations: {fold_dir}")
            source = read_json(fold_dir / "source_tree_summary.json")
            source_digests.append(str(source["sha256"]))
            fingerprint = read_json(fold_dir / "run_fingerprint.json")
            fingerprints.append(str(fingerprint["combined_sha256"]))
            capacity = read_json(fold_dir / "capacity_audit.json")
            if capacity.get("status") != "passed" or float(
                capacity["maximum_to_minimum_ratio"]
            ) > 1.05:
                raise RuntimeError(f"capacity audit failed: {fold_dir}")
            summary = pd.read_csv(fold_dir / "summary.csv")
            if set(summary["model"]) != set(models) or len(summary) != len(models):
                raise RuntimeError(f"V10 model set is incomplete: {fold_dir}")
            hpo = pd.read_csv(fold_dir / "hpo_results.csv")
            if len(hpo) != 6 * len(models):
                raise RuntimeError(f"V10 HPO row count is incomplete: {fold_dir}")
            for row in hpo.to_dict(orient="records"):
                hpo_rows.append({"subject": subject, "seed": args.seed, "fold": fold, **row})

            predictions: dict[str, np.ndarray] = {}
            indices: np.ndarray | None = None
            labels: np.ndarray | None = None
            for model in models:
                saved_indices, saved_labels, prediction = _prediction(
                    fold_dir / model / "outer_predictions.npz"
                )
                if indices is None:
                    indices, labels = saved_indices, saved_labels
                elif not np.array_equal(indices, saved_indices) or not np.array_equal(
                    labels, saved_labels
                ):
                    raise RuntimeError(f"V10 predictions are not aligned: {fold_dir}")
                predictions[model] = prediction
            assert indices is not None and labels is not None
            anchor_fold = (
                anchor_root
                / f"subject_{subject:02d}"
                / f"seed_{args.seed}"
                / f"fold_{fold}"
            )
            for anchor, directory in (
                ("v9_ann_sew_kd", "ann_sew_kd"),
                ("v9_sew_clif_kd", "sew_clif_kd"),
            ):
                saved_indices, saved_labels, prediction = _prediction(
                    anchor_fold / directory / "outer_predictions.npz"
                )
                if not np.array_equal(indices, saved_indices) or not np.array_equal(
                    labels, saved_labels
                ):
                    raise RuntimeError(f"V9/V10 predictions are not aligned: {fold_dir}")
                predictions[anchor] = prediction
            if indices.size != 48 or np.unique(indices).size != 48:
                raise RuntimeError(f"V10 outer fold is not 48 unique trials: {fold_dir}")
            seen.extend(indices.tolist())

            for model, prediction in predictions.items():
                row: dict[str, Any] = {
                    "subject": subject,
                    "seed": args.seed,
                    "fold": fold,
                    "model": model,
                    **_metrics(labels, prediction),
                }
                if model in models:
                    source_row = summary.loc[summary["model"] == model].iloc[0]
                    row.update(
                        {
                            "parameters": int(source_row["parameters"]),
                            "mean_firing_rate": float(source_row["mean_firing_rate"]),
                            "selected_learning_rate": float(
                                source_row["selected_learning_rate"]
                            ),
                            "selected_weight_decay": float(
                                source_row["selected_weight_decay"]
                            ),
                            "selected_epoch": int(source_row["selected_epoch"]),
                        }
                    )
                fold_rows.append(row)
            for offset, trial_index in enumerate(indices):
                row = {
                    "subject": subject,
                    "seed": args.seed,
                    "fold": fold,
                    "trial_index": int(trial_index),
                    "label": int(labels[offset]),
                }
                row.update(
                    {
                        f"prediction_{model}": int(prediction[offset])
                        for model, prediction in predictions.items()
                    }
                )
                trial_rows.append(row)
        counts = Counter(seen)
        if len(counts) != 288 or set(counts.values()) != {1}:
            raise RuntimeError(f"S{subject} does not cover Session T exactly once")

    expected_runs = len(subjects) * len(folds)
    if set(source_digests) != {args.expected_source_digest}:
        raise RuntimeError("V10 folds do not share the expected source snapshot")
    if len(set(fingerprints)) != expected_runs:
        raise RuntimeError("V10 run fingerprints are incomplete or duplicated")

    fold_frame = pd.DataFrame(fold_rows)
    trial_frame = pd.DataFrame(trial_rows)
    subject_rows: list[dict[str, Any]] = []
    for subject, group in trial_frame.groupby("subject", sort=True):
        labels = group["label"].to_numpy()
        for model in all_models:
            subject_rows.append(
                {
                    "subject": int(subject),
                    "model": model,
                    **_metrics(labels, group[f"prediction_{model}"].to_numpy()),
                }
            )
    subject_frame = pd.DataFrame(subject_rows)
    model_rows: list[dict[str, Any]] = []
    for model, group in subject_frame.groupby("model", sort=False):
        model_rows.append(
            {
                "model": model,
                "subjects": len(group),
                "accuracy_mean": float(group["accuracy"].mean()),
                "accuracy_std": float(group["accuracy"].std(ddof=1)),
                "kappa_mean": float(group["kappa"].mean()),
                "kappa_std": float(group["kappa"].std(ddof=1)),
            }
        )
    model_frame = pd.DataFrame(model_rows)

    paired_rows: list[dict[str, Any]] = []
    for second in ("ann_leaky_sew", "ann_tcn", "ann_gru", "ann_lstm", "v9_sew_clif_kd"):
        first_values = subject_frame.loc[
            subject_frame["model"] == "sew_clif", "accuracy"
        ].to_numpy()
        second_values = subject_frame.loc[
            subject_frame["model"] == second, "accuracy"
        ].to_numpy()
        delta = 100.0 * (first_values - second_values)
        fold_first = fold_frame.loc[fold_frame["model"] == "sew_clif", "accuracy"].to_numpy()
        fold_second = fold_frame.loc[fold_frame["model"] == second, "accuracy"].to_numpy()
        fold_delta = 100.0 * (fold_first - fold_second)
        paired_rows.append(
            {
                "first": "sew_clif",
                "second": second,
                "subjects": len(delta),
                "mean_delta_pp": float(delta.mean()),
                "median_delta_pp": float(np.median(delta)),
                "minimum_delta_pp": float(delta.min()),
                "maximum_delta_pp": float(delta.max()),
                "positive_folds": int(np.sum(fold_delta > 0.0)),
                "negative_folds": int(np.sum(fold_delta < 0.0)),
                "tied_folds": int(np.sum(fold_delta == 0.0)),
                **_paired(delta),
            }
        )
    paired_frame = pd.DataFrame(paired_rows)

    write_csv(output / "fold_metrics.csv", fold_frame.to_dict(orient="records"))
    write_csv(output / "subject_metrics.csv", subject_frame.to_dict(orient="records"))
    write_csv(output / "model_summary.csv", model_frame.to_dict(orient="records"))
    write_csv(output / "paired_comparisons.csv", paired_frame.to_dict(orient="records"))
    write_csv(output / "hpo_results.csv", hpo_rows)
    write_csv(output / "trial_predictions.csv", trial_frame.to_dict(orient="records"))
    audit = {
        "status": "passed",
        "expected_runs": expected_runs,
        "completed_runs": expected_runs,
        "subjects": list(subjects),
        "seed": int(args.seed),
        "folds": list(folds),
        "trial_rows": len(trial_frame),
        "source_tree_sha256": args.expected_source_digest,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "unique_run_fingerprints": len(set(fingerprints)),
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    write_json(output / "campaign_audit.json", audit)
    summary = {
        "status": "completed",
        "audit": audit,
        "models": model_frame.to_dict(orient="records"),
        "paired_comparisons": paired_frame.to_dict(orient="records"),
    }
    write_json(output / "aggregate_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
