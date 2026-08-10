#!/usr/bin/env python3
"""Aggregate the bounded native-rate dual-SNN E13 pilot."""

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

from dpc_snn.models.v13_dual_rate_student import V13_MODEL_VARIANTS  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402


ANCHOR = "v9_sew_clif_kd"
TEACHER = "equal_teacher"


def _csv_int(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError("integer list must be non-empty and unique")
    return parsed


def _csv_variants(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError("variant list must be non-empty and unique")
    unknown = set(parsed).difference(V13_MODEL_VARIANTS)
    if unknown:
        raise ValueError(f"unknown V13 variants: {sorted(unknown)}")
    return parsed


def _prediction(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "indices",
            "logits",
            "endpoint_logits",
            "atc_logits",
            "fbc_logits",
            "equal_teacher_logits",
            "labels",
        }
        if set(archive.files) != required:
            raise RuntimeError(f"invalid V13 prediction archive: {path}")
        values = tuple(np.asarray(archive[name]) for name in sorted(required))
        indices = np.asarray(archive["indices"], dtype=np.int64)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        logits = np.asarray(archive["logits"], dtype=np.float64)
        atc = np.asarray(archive["atc_logits"], dtype=np.float64)
        fbc = np.asarray(archive["fbc_logits"], dtype=np.float64)
        teacher = np.asarray(archive["equal_teacher_logits"], dtype=np.float64)
    if any(value.shape != (indices.size, 4) for value in (logits, atc, fbc, teacher)):
        raise RuntimeError(f"malformed V13 prediction archive: {path}")
    if not all(np.isfinite(value).all() for value in values):
        raise RuntimeError(f"non-finite V13 prediction archive: {path}")
    return indices, labels, logits, atc, fbc, teacher


def _anchor(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        indices = np.asarray(archive["indices"], dtype=np.int64)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        logits = np.asarray(archive["logits"], dtype=np.float64)
    return indices, labels, logits


def _metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "kappa": float(cohen_kappa_score(labels, prediction)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--anchor-root", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--subjects", default="1,3,8")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--folds", default="0,1,2,3,4,5")
    parser.add_argument("--variants", default=",".join(V13_MODEL_VARIANTS))
    parser.add_argument("--expected-source-digest", required=True)
    parser.add_argument("--minimum-gain-pp", type=float, default=0.5)
    parser.add_argument("--minimum-positive-pairs", type=int, default=2)
    parser.add_argument("--maximum-regression-pp", type=float, default=1.0)
    parser.add_argument("--minimum-gap-recovery", type=float, default=0.4)
    parser.add_argument("--minimum-fusion-gain-pp", type=float, default=0.5)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    anchor_root = Path(args.anchor_root).resolve()
    output = ensure_dir(Path(args.output).resolve() if args.output else root / "aggregate")
    subjects = _csv_int(args.subjects)
    seeds = _csv_int(args.seeds)
    folds = _csv_int(args.folds)
    variants = _csv_variants(args.variants)
    fold_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    fingerprints: list[str] = []
    source_digests: list[str] = []

    for subject in subjects:
        for seed in seeds:
            seen: list[int] = []
            for fold in folds:
                fold_dir = root / f"subject_{subject:02d}" / f"seed_{seed}" / f"fold_{fold}"
                status = read_json(fold_dir / "campaign_status.json")
                capacity = read_json(fold_dir / "capacity_audit.json")
                if (
                    status.get("status") != "completed"
                    or status.get("session_e_accessed") is not False
                    or status.get("openbmi_s2_accessed") is not False
                    or status.get("variants") != list(variants)
                    or int(status.get("total_hpo_configurations", -1)) > 12
                    or capacity.get("status") != "passed"
                    or capacity.get("raw_feature_bypass") is not False
                ):
                    raise RuntimeError(f"incomplete or invalid V13 fold: {fold_dir}")
                source_digests.append(
                    str(read_json(fold_dir / "source_tree_summary.json")["sha256"])
                )
                fingerprints.append(
                    str(read_json(fold_dir / "run_fingerprint.json")["combined_sha256"])
                )
                predictions: dict[str, np.ndarray] = {}
                branches: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                indices: np.ndarray | None = None
                labels: np.ndarray | None = None
                teacher: np.ndarray | None = None
                for variant in variants:
                    saved_indices, saved_labels, logits, atc, fbc, saved_teacher = (
                        _prediction(fold_dir / variant / "outer_predictions.npz")
                    )
                    if indices is None:
                        indices, labels, teacher = saved_indices, saved_labels, saved_teacher
                    elif (
                        not np.array_equal(indices, saved_indices)
                        or not np.array_equal(labels, saved_labels)
                        or not np.array_equal(teacher, saved_teacher)
                    ):
                        raise RuntimeError(f"V13 predictions are not aligned: {fold_dir}")
                    predictions[variant] = logits
                    branches[variant] = (atc, fbc)
                    fused_prediction = logits.argmax(axis=1)
                    atc_prediction = atc.argmax(axis=1)
                    fbc_prediction = fbc.argmax(axis=1)
                    fused_accuracy = float(np.mean(fused_prediction == saved_labels))
                    best_branch = max(
                        float(np.mean(atc_prediction == saved_labels)),
                        float(np.mean(fbc_prediction == saved_labels)),
                    )
                    fold_rows.append(
                        {
                            "subject": subject,
                            "seed": seed,
                            "fold": fold,
                            "variant": variant,
                            **_metrics(saved_labels, fused_prediction),
                            "atc_branch_accuracy": float(
                                np.mean(atc_prediction == saved_labels)
                            ),
                            "fbc_branch_accuracy": float(
                                np.mean(fbc_prediction == saved_labels)
                            ),
                            "fusion_gain_over_best_branch_pp": 100.0
                            * (fused_accuracy - best_branch),
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
                    raise RuntimeError("V13 and V9 anchor predictions are not aligned")
                predictions[ANCHOR] = anchor_logits
                predictions[TEACHER] = teacher
                seen.extend(indices.tolist())
                endpoints = pd.read_csv(fold_dir / "endpoint_metrics.csv")
                endpoint_rows.extend(endpoints.to_dict(orient="records"))
                for offset, trial_index in enumerate(indices):
                    row: dict[str, Any] = {
                        "subject": subject,
                        "seed": seed,
                        "fold": fold,
                        "trial_index": int(trial_index),
                        "label": int(labels[offset]),
                    }
                    for name, logits in predictions.items():
                        row[f"prediction_{name}"] = int(logits[offset].argmax())
                    for variant, (atc, fbc) in branches.items():
                        row[f"prediction_{variant}_atc"] = int(atc[offset].argmax())
                        row[f"prediction_{variant}_fbc"] = int(fbc[offset].argmax())
                    trial_rows.append(row)
            counts = Counter(seen)
            if len(counts) != 288 or set(counts.values()) != {1}:
                raise RuntimeError("V13 outer folds do not cover Session T exactly once")

    expected_runs = len(subjects) * len(seeds) * len(folds)
    if set(source_digests) != {args.expected_source_digest}:
        raise RuntimeError("V13 source snapshot mismatch")
    if len(set(fingerprints)) != expected_runs:
        raise RuntimeError("V13 fingerprints are duplicated or incomplete")

    trial_frame = pd.DataFrame(trial_rows).sort_values(
        ["subject", "seed", "trial_index"]
    )
    subject_seed_rows: list[dict[str, Any]] = []
    for (subject, seed), group in trial_frame.groupby(["subject", "seed"], sort=True):
        labels = group["label"].to_numpy()
        for model in (*variants, ANCHOR, TEACHER):
            prediction = group[f"prediction_{model}"].to_numpy()
            subject_seed_rows.append(
                {
                    "subject": int(subject),
                    "seed": int(seed),
                    "model": model,
                    **_metrics(labels, prediction),
                }
            )
        for variant in variants:
            atc_prediction = group[f"prediction_{variant}_atc"].to_numpy()
            fbc_prediction = group[f"prediction_{variant}_fbc"].to_numpy()
            fused_prediction = group[f"prediction_{variant}"].to_numpy()
            atc_accuracy = float(np.mean(atc_prediction == labels))
            fbc_accuracy = float(np.mean(fbc_prediction == labels))
            fused_accuracy = float(np.mean(fused_prediction == labels))
            subject_seed_rows.append(
                {
                    "subject": int(subject),
                    "seed": int(seed),
                    "model": f"{variant}_best_branch",
                    "accuracy": max(atc_accuracy, fbc_accuracy),
                    "kappa": float("nan"),
                }
            )
            subject_seed_rows.append(
                {
                    "subject": int(subject),
                    "seed": int(seed),
                    "model": f"{variant}_fusion_gain_pp",
                    "accuracy": 100.0 * (fused_accuracy - max(atc_accuracy, fbc_accuracy)),
                    "kappa": float("nan"),
                }
            )
    subject_seed = pd.DataFrame(subject_seed_rows)
    report_models = (*variants, ANCHOR, TEACHER)
    model_summary = (
        subject_seed.loc[subject_seed["model"].isin(report_models)]
        .groupby("model", as_index=False)
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
    for variant in variants:
        values = subject_seed.loc[subject_seed["model"] == variant].sort_values(
            ["subject", "seed"]
        )["accuracy"].to_numpy()
        branch_values = subject_seed.loc[
            subject_seed["model"] == f"{variant}_best_branch"
        ].sort_values(["subject", "seed"])["accuracy"].to_numpy()
        delta = 100.0 * (values - anchor_values)
        fusion_gain = 100.0 * (values - branch_values)
        comparison_rows.append(
            {
                "variant": variant,
                "reference": ANCHOR,
                "mean_delta_pp": float(delta.mean()),
                "median_delta_pp": float(np.median(delta)),
                "minimum_delta_pp": float(delta.min()),
                "positive_pairs": int(np.sum(delta > 0.0)),
                "negative_pairs": int(np.sum(delta < 0.0)),
                "mean_fusion_gain_over_best_branch_pp": float(fusion_gain.mean()),
                "minimum_fusion_gain_over_best_branch_pp": float(fusion_gain.min()),
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
    selected = comparisons.iloc[0]
    promotion_passed = bool(
        float(selected["mean_delta_pp"]) >= float(args.minimum_gain_pp)
        and int(selected["positive_pairs"]) >= int(args.minimum_positive_pairs)
        and float(selected["minimum_delta_pp"]) >= -float(args.maximum_regression_pp)
        and float(selected["teacher_gap_recovery"]) >= float(args.minimum_gap_recovery)
        and float(selected["mean_fusion_gain_over_best_branch_pp"])
        >= float(args.minimum_fusion_gain_pp)
    )
    promotion = {
        "status": "passed" if promotion_passed else "failed",
        "selected_variant": str(selected["variant"]),
        "mean_delta_pp": float(selected["mean_delta_pp"]),
        "median_delta_pp": float(selected["median_delta_pp"]),
        "minimum_delta_pp": float(selected["minimum_delta_pp"]),
        "positive_pairs": int(selected["positive_pairs"]),
        "mean_fusion_gain_over_best_branch_pp": float(
            selected["mean_fusion_gain_over_best_branch_pp"]
        ),
        "teacher_gap_pp": 100.0 * teacher_gap,
        "teacher_gap_recovery": float(selected["teacher_gap_recovery"]),
        "criteria": {
            "minimum_mean_gain_pp": float(args.minimum_gain_pp),
            "minimum_positive_pairs": int(args.minimum_positive_pairs),
            "maximum_regression_pp": float(args.maximum_regression_pp),
            "minimum_teacher_gap_recovery": float(args.minimum_gap_recovery),
            "minimum_fusion_gain_over_best_branch_pp": float(
                args.minimum_fusion_gain_pp
            ),
        },
    }

    per_class_rows: list[dict[str, Any]] = []
    labels = trial_frame["label"].to_numpy()
    for model in report_models:
        prediction = trial_frame[f"prediction_{model}"].to_numpy()
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
    write_csv(output / "endpoint_metrics.csv", endpoint_rows)
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
        "variants": list(variants),
        "source_tree_sha256": args.expected_source_digest,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "unique_run_fingerprints": len(set(fingerprints)),
        "parameter_match_threshold": 0.01,
        "raw_feature_bypass": False,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    write_json(output / "campaign_audit.json", audit)
    summary = {"status": "completed", "audit": audit, "promotion": promotion}
    write_json(output / "aggregate_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
