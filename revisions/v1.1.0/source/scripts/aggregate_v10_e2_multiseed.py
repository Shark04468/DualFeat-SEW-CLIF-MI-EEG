#!/usr/bin/env python3
"""Aggregate and audit the V10 E2 multi-seed strong-control campaign."""

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
from scipy.special import softmax
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import accuracy_score, cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.models.v10_strong_controls import V10_STRONG_CONTROL_MODELS  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402


ANCHORS = ("v9_ann_sew_kd", "v9_sew_clif_kd")
TEACHER = "equal_teacher"
STRONG_ANNS = ("ann_tcn", "ann_gru", "ann_lstm")
ATCNET_PARAMETERS = 113_732
FBCNET_PARAMETERS = 11_812


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
        logits = np.asarray(archive["logits"], dtype=np.float64)
        labels = np.asarray(archive["labels"], dtype=np.int64)
    if logits.shape != (indices.size, 4) or labels.shape != indices.shape:
        raise RuntimeError(f"malformed prediction archive: {path}")
    return indices, labels, logits


def _metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    prediction = logits.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "kappa": float(cohen_kappa_score(labels, prediction)),
    }


def _paired(delta: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(delta, dtype=np.float64)
    nonzero = values[values != 0.0]
    positive = int(np.sum(nonzero > 0.0))
    negative = int(np.sum(nonzero < 0.0))
    return {
        "positive": positive,
        "negative": negative,
        "ties": int(values.size - nonzero.size),
        "sign_test_p_greater": (
            float(binomtest(positive, nonzero.size, 0.5, alternative="greater").pvalue)
            if nonzero.size
            else 1.0
        ),
        "wilcoxon_p_greater": (
            float(wilcoxon(nonzero, alternative="greater").pvalue)
            if nonzero.size
            else 1.0
        ),
    }


def _teacher_logits(cache_path: Path, expected_indices: np.ndarray) -> np.ndarray:
    with np.load(cache_path, allow_pickle=False) as archive:
        indices = np.asarray(archive["outer_test_indices"], dtype=np.int64)
        logits = np.asarray(archive["teacher_outer_test"], dtype=np.float64)
    if not np.array_equal(indices, expected_indices) or logits.shape != (
        expected_indices.size,
        4,
    ):
        raise RuntimeError(f"teacher cache is not aligned: {cache_path}")
    return logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--anchor-root", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--subjects", default="1,3,8")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--folds", default="0,1,2,3,4,5")
    parser.add_argument("--expected-source-digest", required=True)
    parser.add_argument("--noninferiority-margin-pp", type=float, default=-0.3)
    parser.add_argument("--stability-positive-pairs", type=int, default=5)
    parser.add_argument("--maximum-subject-deficit-pp", type=float, default=2.0)
    parser.add_argument("--superiority-median-pp", type=float, default=2.0)
    parser.add_argument("--superiority-positive-pairs", type=int, default=7)
    args = parser.parse_args()

    campaign_root = Path(args.root).resolve()
    anchor_root = Path(args.anchor_root).resolve()
    output = ensure_dir(
        Path(args.output).resolve() if args.output else campaign_root / "aggregate_multiseed"
    )
    subjects = _csv_int(args.subjects)
    seeds = _csv_int(args.seeds)
    folds = _csv_int(args.folds)
    models = tuple(V10_STRONG_CONTROL_MODELS)
    all_models = (*models, *ANCHORS, TEACHER)

    fold_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    hpo_rows: list[dict[str, Any]] = []
    capacity_rows: dict[str, dict[str, int]] = {}
    fingerprints: list[str] = []
    source_digests: list[str] = []

    for subject in subjects:
        for seed in seeds:
            seen: list[int] = []
            for fold in folds:
                fold_dir = (
                    campaign_root
                    / f"subject_{subject:02d}"
                    / f"seed_{seed}"
                    / f"fold_{fold}"
                )
                status = read_json(fold_dir / "campaign_status.json")
                if status.get("status") != "completed" or status.get(
                    "session_e_accessed"
                ) is not False:
                    raise RuntimeError(f"incomplete or unlocked V10 fold: {fold_dir}")
                if int(status.get("hpo_configurations_per_model", -1)) != 6:
                    raise RuntimeError(f"unequal V10 HPO budget: {fold_dir}")
                source_digests.append(
                    str(read_json(fold_dir / "source_tree_summary.json")["sha256"])
                )
                fingerprints.append(
                    str(read_json(fold_dir / "run_fingerprint.json")["combined_sha256"])
                )
                capacity = read_json(fold_dir / "capacity_audit.json")
                if capacity.get("status") != "passed" or float(
                    capacity["maximum_to_minimum_ratio"]
                ) > 1.05:
                    raise RuntimeError(f"capacity audit failed: {fold_dir}")
                for model, row in capacity["models"].items():
                    current = {
                        "decoder_parameters": int(row["parameters"]),
                        "frozen_frontend_parameters": ATCNET_PARAMETERS
                        + FBCNET_PARAMETERS,
                        "end_to_end_parameters": int(row["parameters"])
                        + ATCNET_PARAMETERS
                        + FBCNET_PARAMETERS,
                    }
                    if model in capacity_rows and capacity_rows[model] != current:
                        raise RuntimeError("capacity changed between V10 folds")
                    capacity_rows[model] = current

                summary = pd.read_csv(fold_dir / "summary.csv")
                if set(summary["model"]) != set(models) or len(summary) != len(models):
                    raise RuntimeError(f"V10 model set is incomplete: {fold_dir}")
                hpo = pd.read_csv(fold_dir / "hpo_results.csv")
                if len(hpo) != 6 * len(models):
                    raise RuntimeError(f"V10 HPO rows are incomplete: {fold_dir}")
                for row in hpo.to_dict(orient="records"):
                    hpo_rows.append(
                        {"subject": subject, "seed": seed, "fold": fold, **row}
                    )

                logits_by_model: dict[str, np.ndarray] = {}
                indices: np.ndarray | None = None
                labels: np.ndarray | None = None
                for model in models:
                    saved_indices, saved_labels, logits = _prediction(
                        fold_dir / model / "outer_predictions.npz"
                    )
                    if indices is None:
                        indices, labels = saved_indices, saved_labels
                    elif not np.array_equal(indices, saved_indices) or not np.array_equal(
                        labels, saved_labels
                    ):
                        raise RuntimeError(f"V10 predictions are not aligned: {fold_dir}")
                    logits_by_model[model] = logits
                assert indices is not None and labels is not None

                anchor_fold = (
                    anchor_root
                    / f"subject_{subject:02d}"
                    / f"seed_{seed}"
                    / f"fold_{fold}"
                )
                for anchor, directory in (
                    ("v9_ann_sew_kd", "ann_sew_kd"),
                    ("v9_sew_clif_kd", "sew_clif_kd"),
                ):
                    saved_indices, saved_labels, logits = _prediction(
                        anchor_fold / directory / "outer_predictions.npz"
                    )
                    if not np.array_equal(indices, saved_indices) or not np.array_equal(
                        labels, saved_labels
                    ):
                        raise RuntimeError(f"V9/V10 predictions are not aligned: {fold_dir}")
                    logits_by_model[anchor] = logits
                logits_by_model[TEACHER] = _teacher_logits(
                    anchor_fold / "frozen_dual_feature_cache.npz", indices
                )

                if indices.size != 48 or np.unique(indices).size != 48:
                    raise RuntimeError(f"outer fold is not 48 unique trials: {fold_dir}")
                seen.extend(indices.tolist())
                for model, logits in logits_by_model.items():
                    row: dict[str, Any] = {
                        "subject": subject,
                        "seed": seed,
                        "fold": fold,
                        "model": model,
                        **_metrics(labels, logits),
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

                probabilities = {
                    model: softmax(logits, axis=1)
                    for model, logits in logits_by_model.items()
                }
                for offset, trial_index in enumerate(indices):
                    row = {
                        "subject": subject,
                        "seed": seed,
                        "fold": fold,
                        "trial_index": int(trial_index),
                        "label": int(labels[offset]),
                    }
                    for model in all_models:
                        row[f"prediction_{model}"] = int(
                            logits_by_model[model][offset].argmax()
                        )
                        row[f"confidence_{model}"] = float(
                            probabilities[model][offset].max()
                        )
                        for class_index in range(4):
                            row[f"probability_{model}_{class_index}"] = float(
                                probabilities[model][offset, class_index]
                            )
                    trial_rows.append(row)
            counts = Counter(seen)
            if len(counts) != 288 or set(counts.values()) != {1}:
                raise RuntimeError(
                    f"S{subject} seed {seed} does not cover Session T exactly once"
                )

    expected_runs = len(subjects) * len(seeds) * len(folds)
    if set(source_digests) != {args.expected_source_digest}:
        raise RuntimeError("V10 folds do not share the expected source snapshot")
    if len(set(fingerprints)) != expected_runs:
        raise RuntimeError("V10 run fingerprints are incomplete or duplicated")

    fold_frame = pd.DataFrame(fold_rows)
    trial_frame = pd.DataFrame(trial_rows)
    subject_seed_rows: list[dict[str, Any]] = []
    for (subject, seed), group in trial_frame.groupby(["subject", "seed"], sort=True):
        labels = group["label"].to_numpy()
        for model in all_models:
            logits = np.log(
                np.clip(
                    group[
                        [f"probability_{model}_{index}" for index in range(4)]
                    ].to_numpy(),
                    1e-12,
                    1.0,
                )
            )
            subject_seed_rows.append(
                {
                    "subject": int(subject),
                    "seed": int(seed),
                    "model": model,
                    **_metrics(labels, logits),
                }
            )
    subject_seed_frame = pd.DataFrame(subject_seed_rows)
    subject_frame = (
        subject_seed_frame.groupby(["subject", "model"], as_index=False)[
            ["accuracy", "kappa"]
        ]
        .mean()
        .sort_values(["subject", "model"])
    )
    model_frame = (
        subject_frame.groupby("model", as_index=False)
        .agg(
            subjects=("subject", "size"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            kappa_mean=("kappa", "mean"),
            kappa_std=("kappa", "std"),
        )
    )

    paired_rows: list[dict[str, Any]] = []
    for second in (*models[:-1], "v9_sew_clif_kd", TEACHER):
        first_rows = subject_seed_frame.loc[
            subject_seed_frame["model"] == "sew_clif"
        ].sort_values(["subject", "seed"])
        second_rows = subject_seed_frame.loc[
            subject_seed_frame["model"] == second
        ].sort_values(["subject", "seed"])
        if not np.array_equal(
            first_rows[["subject", "seed"]].to_numpy(),
            second_rows[["subject", "seed"]].to_numpy(),
        ):
            raise RuntimeError("paired subject-seed rows are not aligned")
        pair_delta = 100.0 * (
            first_rows["accuracy"].to_numpy() - second_rows["accuracy"].to_numpy()
        )
        pair_table = first_rows[["subject", "seed"]].copy()
        pair_table["delta"] = pair_delta
        subject_delta = pair_table.groupby("subject")["delta"].mean().to_numpy()
        paired_rows.append(
            {
                "first": "sew_clif",
                "second": second,
                "subject_seed_pairs": int(pair_delta.size),
                "mean_delta_pp": float(pair_delta.mean()),
                "median_delta_pp": float(np.median(pair_delta)),
                "minimum_delta_pp": float(pair_delta.min()),
                "maximum_delta_pp": float(pair_delta.max()),
                "subject_mean_delta_pp": float(subject_delta.mean()),
                "minimum_subject_delta_pp": float(subject_delta.min()),
                **{f"pair_{key}": value for key, value in _paired(pair_delta).items()},
                **{
                    f"subject_{key}": value
                    for key, value in _paired(subject_delta).items()
                },
            }
        )
    paired_frame = pd.DataFrame(paired_rows)

    snn_pairs = subject_seed_frame.loc[
        subject_seed_frame["model"] == "sew_clif"
    ].sort_values(["subject", "seed"])
    ann_table = subject_seed_frame.loc[
        subject_seed_frame["model"].isin(STRONG_ANNS)
    ].pivot(index=["subject", "seed"], columns="model", values="accuracy")
    ann_best = ann_table.max(axis=1).sort_index().to_numpy()
    snn_values = snn_pairs["accuracy"].to_numpy()
    best_ann_delta = 100.0 * (snn_values - ann_best)
    best_ann_pair_table = snn_pairs[["subject", "seed"]].copy()
    best_ann_pair_table["delta"] = best_ann_delta
    best_ann_subject_delta = (
        best_ann_pair_table.groupby("subject")["delta"].mean().to_numpy()
    )

    gru = paired_frame.loc[paired_frame["second"] == "ann_gru"].iloc[0]
    stability_gate = {
        "status": "passed"
        if (
            float(gru["mean_delta_pp"]) >= float(args.noninferiority_margin_pp)
            and int(gru["pair_positive"]) >= int(args.stability_positive_pairs)
            and float(gru["minimum_subject_delta_pp"])
            >= -float(args.maximum_subject_deficit_pp)
        )
        else "failed",
        "primary_comparator": "ann_gru",
        "mean_delta_pp": float(gru["mean_delta_pp"]),
        "minimum_subject_delta_pp": float(gru["minimum_subject_delta_pp"]),
        "positive_subject_seed_pairs": int(gru["pair_positive"]),
        "criteria": {
            "mean_delta_pp_minimum": float(args.noninferiority_margin_pp),
            "positive_subject_seed_pairs_minimum": int(args.stability_positive_pairs),
            "minimum_subject_delta_pp": -float(args.maximum_subject_deficit_pp),
        },
    }
    superiority_gate = {
        "status": "passed"
        if (
            float(np.median(best_ann_delta)) >= float(args.superiority_median_pp)
            and int(np.sum(best_ann_delta > 0.0))
            >= int(args.superiority_positive_pairs)
        )
        else "failed",
        "comparison": "sew_clif_minus_pairwise_best_strong_ann",
        "mean_delta_pp": float(best_ann_delta.mean()),
        "median_delta_pp": float(np.median(best_ann_delta)),
        "minimum_subject_delta_pp": float(best_ann_subject_delta.min()),
        "positive_subject_seed_pairs": int(np.sum(best_ann_delta > 0.0)),
        "criteria": {
            "median_delta_pp_minimum": float(args.superiority_median_pp),
            "positive_subject_seed_pairs_minimum": int(
                args.superiority_positive_pairs
            ),
        },
    }

    capacity_frame = pd.DataFrame(
        [
            {"model": model, **row}
            for model, row in sorted(capacity_rows.items())
        ]
        + [
            {
                "model": TEACHER,
                "decoder_parameters": 0,
                "frozen_frontend_parameters": ATCNET_PARAMETERS + FBCNET_PARAMETERS,
                "end_to_end_parameters": ATCNET_PARAMETERS + FBCNET_PARAMETERS,
            }
        ]
    )

    write_csv(output / "fold_metrics.csv", fold_frame.to_dict(orient="records"))
    write_csv(
        output / "subject_seed_metrics.csv",
        subject_seed_frame.to_dict(orient="records"),
    )
    write_csv(output / "subject_metrics.csv", subject_frame.to_dict(orient="records"))
    write_csv(output / "model_summary.csv", model_frame.to_dict(orient="records"))
    write_csv(output / "paired_comparisons.csv", paired_frame.to_dict(orient="records"))
    write_csv(output / "hpo_results.csv", hpo_rows)
    write_csv(output / "trial_predictions.csv", trial_frame.to_dict(orient="records"))
    write_csv(output / "end_to_end_capacity.csv", capacity_frame.to_dict(orient="records"))
    audit = {
        "status": "passed",
        "expected_runs": expected_runs,
        "completed_runs": expected_runs,
        "subjects": list(subjects),
        "seeds": list(seeds),
        "folds": list(folds),
        "trial_rows": len(trial_frame),
        "source_tree_sha256": args.expected_source_digest,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "unique_run_fingerprints": len(set(fingerprints)),
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
        "inference_unit": "subject after averaging seeds",
    }
    write_json(output / "campaign_audit.json", audit)
    summary = {
        "status": "completed",
        "audit": audit,
        "stability_gate": stability_gate,
        "title_superiority_gate": superiority_gate,
        "models": model_frame.to_dict(orient="records"),
        "paired_comparisons": paired_frame.to_dict(orient="records"),
    }
    write_json(output / "aggregate_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
