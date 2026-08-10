#!/usr/bin/env python3
"""Aggregate and audit the complete V9 E3 dual-feature development pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import accuracy_score, cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v9_dual_feature_training import (  # noqa: E402
    V9_DUAL_FEATURE_EXPERIMENT_VARIANTS,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402


TEACHERS = ("atc", "fbc", "equal")


def _csv_int(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError("integer list must be non-empty and unique")
    return parsed


def _metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "kappa": float(cohen_kappa_score(labels, prediction)),
    }


def _read_student_prediction(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"indices", "logits", "labels"}:
            raise RuntimeError(f"invalid student prediction archive: {path}")
        indices = np.asarray(archive["indices"], dtype=np.int64)
        logits = np.asarray(archive["logits"], dtype=np.float32)
        labels = np.asarray(archive["labels"], dtype=np.int64)
    if (
        indices.ndim != 1
        or labels.shape != indices.shape
        or logits.shape != (indices.size, 4)
        or not np.isfinite(logits).all()
    ):
        raise RuntimeError(f"malformed student predictions: {path}")
    return indices, labels, logits.argmax(axis=1)


def _greater_paired_tests(delta: np.ndarray) -> dict[str, float | int]:
    delta = np.asarray(delta, dtype=np.float64)
    nonzero = delta[delta != 0.0]
    positive = int(np.sum(nonzero > 0.0))
    negative = int(np.sum(nonzero < 0.0))
    sign_p = (
        float(binomtest(positive, positive + negative, 0.5, alternative="greater").pvalue)
        if positive + negative
        else 1.0
    )
    wilcoxon_p = (
        float(wilcoxon(nonzero, alternative="greater", method="auto").pvalue)
        if nonzero.size
        else 1.0
    )
    return {
        "positive_pairs": positive,
        "negative_pairs": negative,
        "ties": int(delta.size - nonzero.size),
        "sign_test_p_greater": sign_p,
        "wilcoxon_p_greater": wilcoxon_p,
    }


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--subjects", default="1,3,8")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--folds", default="0,1,2,3,4,5")
    parser.add_argument("--expected-source-digest", default="")
    args = parser.parse_args()

    campaign_root = Path(args.root).resolve()
    output = ensure_dir(Path(args.output).resolve() if args.output else campaign_root / "aggregate")
    subjects = _csv_int(args.subjects)
    seeds = _csv_int(args.seeds)
    folds = _csv_int(args.folds)
    variants = tuple(V9_DUAL_FEATURE_EXPERIMENT_VARIANTS)
    expected_runs = len(subjects) * len(seeds) * len(folds)

    fold_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    source_digests: list[str] = []
    fingerprints: list[str] = []
    replay_errors: list[float] = []

    for subject in subjects:
        for seed in seeds:
            seen_indices: list[int] = []
            for fold in folds:
                fold_dir = (
                    campaign_root
                    / f"subject_{subject:02d}"
                    / f"seed_{seed}"
                    / f"fold_{fold}"
                )
                status = read_json(fold_dir / "campaign_status.json")
                if status.get("status") != "completed":
                    raise RuntimeError(f"fold is not complete: {fold_dir}")
                if status.get("session_e_accessed") is not False:
                    raise RuntimeError(f"Session E lock is not intact: {fold_dir}")
                summary = pd.read_csv(fold_dir / "summary.csv")
                if set(summary["variant"]) != set(variants) or len(summary) != len(variants):
                    raise RuntimeError(f"variant set is incomplete: {fold_dir}")
                source_summary = read_json(fold_dir / "source_tree_summary.json")
                source_digests.append(str(source_summary["sha256"]))
                fingerprint = read_json(fold_dir / "run_fingerprint.json")
                fingerprints.append(str(fingerprint["combined_sha256"]))
                replay = read_json(fold_dir / "teacher_replay.json")
                if replay.get("status") != "passed":
                    raise RuntimeError(f"teacher replay did not pass: {fold_dir}")
                for branch in ("atcnet", "fbcnet"):
                    for key in (
                        "selection_max_abs_logit_error",
                        "outer_max_abs_logit_error",
                    ):
                        error = float(replay["branches"][branch][key])
                        replay_errors.append(error)
                        if error > 1e-5:
                            raise RuntimeError(f"teacher replay error exceeds tolerance: {fold_dir}")

                with np.load(fold_dir / "frozen_dual_feature_cache.npz") as cache:
                    indices = np.asarray(cache["outer_test_indices"], dtype=np.int64)
                    teacher_prediction = {
                        "atc": cache["atcnet_outer_test_logits"].argmax(axis=1),
                        "fbc": cache["fbcnet_outer_test_logits"].argmax(axis=1),
                        "equal": cache["teacher_outer_test"].argmax(axis=1),
                    }
                student_prediction: dict[str, np.ndarray] = {}
                labels: np.ndarray | None = None
                for variant in variants:
                    saved_indices, saved_labels, prediction = _read_student_prediction(
                        fold_dir / variant / "outer_predictions.npz"
                    )
                    if not np.array_equal(saved_indices, indices):
                        raise RuntimeError(f"student trial indices differ from cache: {fold_dir}")
                    if labels is None:
                        labels = saved_labels
                    elif not np.array_equal(labels, saved_labels):
                        raise RuntimeError(f"variant labels differ within fold: {fold_dir}")
                    student_prediction[variant] = prediction
                assert labels is not None
                if indices.size != 48 or np.unique(indices).size != indices.size:
                    raise RuntimeError(f"outer fold must contain 48 unique trials: {fold_dir}")
                seen_indices.extend(indices.tolist())

                for name, prediction in {**teacher_prediction, **student_prediction}.items():
                    metric = _metrics(labels, prediction)
                    row: dict[str, Any] = {
                        "subject": subject,
                        "seed": seed,
                        "fold": fold,
                        "model": name,
                        **metric,
                    }
                    if name in variants:
                        source = summary.loc[summary["variant"] == name].iloc[0]
                        row["mean_firing_rate"] = float(source["student_mean_firing_rate"])
                        row["selected_epoch"] = int(source["selected_epoch"])
                    fold_rows.append(row)

                for offset, trial_index in enumerate(indices):
                    row = {
                        "subject": subject,
                        "seed": seed,
                        "fold": fold,
                        "trial_index": int(trial_index),
                        "label": int(labels[offset]),
                    }
                    row.update(
                        {
                            f"prediction_{name}": int(prediction[offset])
                            for name, prediction in {
                                **teacher_prediction,
                                **student_prediction,
                            }.items()
                        }
                    )
                    trial_rows.append(row)

            counts = Counter(seen_indices)
            if len(counts) != 288 or set(counts.values()) != {1}:
                raise RuntimeError(
                    f"S{subject} seed{seed} does not cover 288 Session-T trials exactly once"
                )

    if len(set(source_digests)) != 1:
        raise RuntimeError("folds were produced by more than one source tree")
    source_digest = source_digests[0]
    if args.expected_source_digest and source_digest != args.expected_source_digest:
        raise RuntimeError("campaign source digest differs from the expected snapshot")
    if len(set(fingerprints)) != expected_runs:
        raise RuntimeError("formal folds do not have unique complete run fingerprints")

    fold_frame = pd.DataFrame(fold_rows)
    trial_frame = pd.DataFrame(trial_rows)
    subject_seed_rows: list[dict[str, Any]] = []
    for (subject, seed), group in trial_frame.groupby(["subject", "seed"], sort=True):
        labels = group["label"].to_numpy()
        for model in (*TEACHERS, *variants):
            prediction = group[f"prediction_{model}"].to_numpy()
            subject_seed_rows.append(
                {"subject": subject, "seed": seed, "model": model, **_metrics(labels, prediction)}
            )
    subject_seed_frame = pd.DataFrame(subject_seed_rows)

    model_rows: list[dict[str, Any]] = []
    for model, group in subject_seed_frame.groupby("model", sort=False):
        accuracy_mean, accuracy_std = _mean_std(group["accuracy"])
        kappa_mean, kappa_std = _mean_std(group["kappa"])
        model_rows.append(
            {
                "model": model,
                "subject_seed_pairs": len(group),
                "accuracy_mean": accuracy_mean,
                "accuracy_std": accuracy_std,
                "kappa_mean": kappa_mean,
                "kappa_std": kappa_std,
            }
        )
    model_frame = pd.DataFrame(model_rows)

    paired_rows: list[dict[str, Any]] = []
    for first, second, label in (
        ("sew_clif_ce", "ann_sew_ce", "SEW-CLIF CE vs matched ANN-SEW CE"),
        ("sew_clif_kd", "ann_sew_kd", "SEW-CLIF KD vs matched ANN-SEW KD"),
        ("sew_clif_kd", "atc", "SEW-CLIF KD vs frozen ATCNet"),
        ("sew_clif_kd", "equal", "SEW-CLIF KD vs equal teacher ensemble"),
    ):
        first_values = subject_seed_frame.loc[
            subject_seed_frame["model"] == first, "accuracy"
        ].to_numpy()
        second_values = subject_seed_frame.loc[
            subject_seed_frame["model"] == second, "accuracy"
        ].to_numpy()
        if first_values.shape != (len(subjects) * len(seeds),) or second_values.shape != (
            len(subjects) * len(seeds),
        ):
            raise RuntimeError("paired subject-seed metric vectors are incomplete")
        delta = 100.0 * (first_values - second_values)
        paired_rows.append(
            {
                "comparison": label,
                "first": first,
                "second": second,
                "pairs": delta.size,
                "mean_delta_pp": float(delta.mean()),
                "median_delta_pp": float(np.median(delta)),
                "min_delta_pp": float(delta.min()),
                "max_delta_pp": float(delta.max()),
                **_greater_paired_tests(delta),
            }
        )
    paired_frame = pd.DataFrame(paired_rows)

    write_csv(output / "fold_metrics.csv", fold_frame.to_dict(orient="records"))
    write_csv(
        output / "subject_seed_metrics.csv", subject_seed_frame.to_dict(orient="records")
    )
    write_csv(output / "model_summary.csv", model_frame.to_dict(orient="records"))
    write_csv(output / "paired_comparisons.csv", paired_frame.to_dict(orient="records"))
    write_csv(output / "trial_predictions.csv", trial_frame.to_dict(orient="records"))
    audit = {
        "status": "passed",
        "expected_runs": expected_runs,
        "completed_runs": expected_runs,
        "subjects": list(subjects),
        "seeds": list(seeds),
        "folds": list(folds),
        "trials_per_subject_seed": 288,
        "trial_rows": len(trial_frame),
        "source_tree_sha256": source_digest,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "unique_run_fingerprints": len(set(fingerprints)),
        "maximum_teacher_replay_error": max(replay_errors, default=0.0),
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
