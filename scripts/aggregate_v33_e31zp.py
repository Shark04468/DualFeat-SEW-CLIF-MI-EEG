#!/usr/bin/env python3
"""Aggregate objective-pure E31-ZP curves and dataset heterogeneity."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402


METRICS = ("accuracy", "balanced_accuracy", "kappa", "macro_f1")


def _design(rows: list[dict[str, Any]], gain_key: str) -> tuple[np.ndarray, np.ndarray]:
    datasets = sorted({str(row["dataset"]) for row in rows})
    reference = datasets[0]
    matrix = []
    response = []
    for row in rows:
        matrix.append(
            [
                1.0,
                float(np.log2(float(row["examples_per_class"]))),
                *[
                    float(str(row["dataset"]) == dataset)
                    for dataset in datasets
                    if dataset != reference
                ],
            ]
        )
        response.append(float(row[gain_key]))
    return np.asarray(matrix, dtype=np.float64), np.asarray(response, dtype=np.float64)


def _common_slope(rows: list[dict[str, Any]], gain_key: str) -> float:
    matrix, response = _design(rows, gain_key)
    return float(np.linalg.lstsq(matrix, response, rcond=None)[0][1])


def _dataset_slope(rows: list[dict[str, Any]], gain_key: str, dataset: str) -> float:
    selected = [row for row in rows if row["dataset"] == dataset]
    x = np.asarray(
        [np.log2(float(row["examples_per_class"])) for row in selected], dtype=float
    )
    y = np.asarray([float(row[gain_key]) for row in selected], dtype=float)
    return float(np.linalg.lstsq(np.column_stack((np.ones_like(x), x)), y, rcond=None)[0][1])


def _bootstrap(
    rows: list[dict[str, Any]], repetitions: int, seed: int = 33_031
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    datasets = sorted({str(row["dataset"]) for row in rows})
    by_dataset: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset"]), {}).setdefault(
            int(row["subject"]), []
        ).append(row)
    generator = np.random.default_rng(seed)
    common = []
    original = []
    per_dataset = {dataset: [] for dataset in datasets}
    for _ in range(int(repetitions)):
        sample: list[dict[str, Any]] = []
        for dataset in datasets:
            subjects = by_dataset[dataset]
            keys = sorted(subjects)
            selected = generator.choice(keys, size=len(keys), replace=True)
            for subject in selected:
                sample.extend(subjects[int(subject)])
        common.append(_common_slope(sample, "gain_accuracy"))
        original.append(_common_slope(sample, "original_gain_accuracy"))
        for dataset in datasets:
            per_dataset[dataset].append(
                _dataset_slope(sample, "gain_accuracy", dataset)
            )
    return (
        np.asarray(common),
        {key: np.asarray(value) for key, value in per_dataset.items()},
        np.asarray(original),
    )


def _holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * pvalues[name]))
        adjusted[name] = running
    return adjusted


def _prediction_identity(parent_path: Path, child_path: Path) -> None:
    with np.load(parent_path, allow_pickle=False) as parent, np.load(
        child_path, allow_pickle=False
    ) as child:
        for key in ("label", "trial_id"):
            if not np.array_equal(parent[key], child[key]):
                raise RuntimeError(f"E31-ZP prediction identity mismatch: {child_path} {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(
        (ROOT / "configs" / "experiments" / "v33_e31_zero_penalty.yaml").read_text(
            encoding="utf-8"
        )
    )
    root = Path(args.input).resolve()
    parent = Path(args.parent).resolve()
    output = ensure_dir(Path(args.output).resolve())
    variant_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for dataset, dataset_config in config["datasets"].items():
        for subject_value in dataset_config["subjects"]:
            subject = int(subject_value)
            status = read_json(
                root / dataset / f"subject_{subject:02d}" / "evaluation_status.json"
            )
            if status.get("status") != "completed":
                raise RuntimeError(f"incomplete E31-ZP evaluation: {dataset} {subject}")
            for seed_value in config["seeds"]:
                seed = int(seed_value)
                for budget in config["budgets"][dataset]:
                    parent_budget = (
                        parent
                        / dataset
                        / f"subject_{subject:02d}"
                        / f"seed_{seed}"
                        / f"budget_{budget}"
                    )
                    child_budget = (
                        root
                        / dataset
                        / f"subject_{subject:02d}"
                        / f"seed_{seed}"
                        / f"budget_{budget}"
                    )
                    ann = read_json(
                        parent_budget / "ann_sew_ce" / "evaluation" / "metrics.json"
                    )
                    original = read_json(
                        parent_budget / "sew_clif_ce" / "evaluation" / "metrics.json"
                    )
                    zero = read_json(
                        child_budget
                        / "sew_clif_ce_fr0"
                        / "evaluation"
                        / "metrics.json"
                    )
                    ann_predictions = (
                        parent_budget / "ann_sew_ce" / "evaluation" / "predictions.npz"
                    )
                    zero_predictions = (
                        child_budget
                        / "sew_clif_ce_fr0"
                        / "evaluation"
                        / "predictions.npz"
                    )
                    _prediction_identity(ann_predictions, zero_predictions)
                    examples = float(zero["examples_per_class"])
                    for variant, values in (
                        ("ann_sew_ce", ann),
                        ("sew_clif_ce_fr001", original),
                        ("sew_clif_ce_fr0", zero),
                    ):
                        variant_rows.append(
                            {
                                "dataset": dataset,
                                "subject": subject,
                                "seed": seed,
                                "budget": budget,
                                "examples_per_class": examples,
                                "variant": variant,
                                **{metric: float(values[metric]) for metric in METRICS},
                            }
                        )
                    paired_rows.append(
                        {
                            "dataset": dataset,
                            "subject": subject,
                            "seed": seed,
                            "budget": budget,
                            "examples_per_class": examples,
                            **{
                                f"ann_{metric}": float(ann[metric])
                                for metric in METRICS
                            },
                            **{
                                f"zero_snn_{metric}": float(zero[metric])
                                for metric in METRICS
                            },
                            **{
                                f"original_snn_{metric}": float(original[metric])
                                for metric in METRICS
                            },
                            **{
                                f"gain_{metric}": float(zero[metric]) - float(ann[metric])
                                for metric in METRICS
                            },
                            **{
                                f"original_gain_{metric}": float(original[metric])
                                - float(ann[metric])
                                for metric in METRICS
                            },
                        }
                    )
    subject_rows: list[dict[str, Any]] = []
    subject_keys = sorted(
        {
            (row["dataset"], row["subject"], row["budget"])
            for row in paired_rows
        }
    )
    numeric_keys = [
        key
        for key in paired_rows[0]
        if key not in {"dataset", "subject", "seed", "budget"}
    ]
    for dataset, subject, budget in subject_keys:
        selected = [
            row
            for row in paired_rows
            if (row["dataset"], row["subject"], row["budget"])
            == (dataset, subject, budget)
        ]
        subject_rows.append(
            {
                "dataset": dataset,
                "subject": subject,
                "budget": budget,
                **{
                    key: float(np.mean([float(row[key]) for row in selected]))
                    for key in numeric_keys
                },
            }
        )
    repetitions = int(config["analysis"]["subject_cluster_bootstrap_repetitions"])
    common_bootstrap, dataset_bootstrap, original_bootstrap = _bootstrap(
        subject_rows, repetitions
    )
    observed = _common_slope(subject_rows, "gain_accuracy")
    original_observed = _common_slope(subject_rows, "original_gain_accuracy")
    lower, upper = np.quantile(common_bootstrap, [0.025, 0.975])
    dataset_results: dict[str, Any] = {}
    for dataset, samples in dataset_bootstrap.items():
        dataset_results[dataset] = {
            "slope_gain_per_log2_example": _dataset_slope(
                subject_rows, "gain_accuracy", dataset
            ),
            "slope_pp_per_doubling": 100.0
            * _dataset_slope(subject_rows, "gain_accuracy", dataset),
            "bootstrap_95_ci": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        }
    pairwise_p: dict[str, float] = {}
    pairwise: dict[str, Any] = {}
    datasets = sorted(dataset_bootstrap)
    for index, first in enumerate(datasets):
        for second in datasets[index + 1 :]:
            difference = dataset_bootstrap[first] - dataset_bootstrap[second]
            name = f"{first}_minus_{second}"
            pvalue = min(
                1.0,
                2.0
                * min(
                    (np.count_nonzero(difference <= 0.0) + 1) / (repetitions + 1),
                    (np.count_nonzero(difference >= 0.0) + 1) / (repetitions + 1),
                ),
            )
            pairwise_p[name] = float(pvalue)
            pairwise[name] = {
                "mean_slope_difference": float(difference.mean()),
                "bootstrap_95_ci": [
                    float(value) for value in np.quantile(difference, [0.025, 0.975])
                ],
                "two_sided_bootstrap_p": float(pvalue),
            }
    for name, adjusted in _holm(pairwise_p).items():
        pairwise[name]["holm_p"] = adjusted
    if upper < 0.0:
        status = "strong_support"
    elif observed < 0.0:
        status = "partial_support"
    else:
        status = "not_supported"
    slope_change = common_bootstrap - original_bootstrap
    decision = {
        "schema": "dpc-snn-e31zp-decision/v1",
        "status": status,
        "slope_gain_per_log2_example": observed,
        "slope_pp_per_doubling": 100.0 * observed,
        "bootstrap_95_ci": [float(lower), float(upper)],
        "one_sided_bootstrap_p": float(
            (np.count_nonzero(common_bootstrap >= 0.0) + 1) / (repetitions + 1)
        ),
        "original_fr001_slope_gain_per_log2_example": original_observed,
        "zero_minus_original_slope": float(observed - original_observed),
        "zero_minus_original_slope_95_ci": [
            float(value) for value in np.quantile(slope_change, [0.025, 0.975])
        ],
        "dataset_slopes": dataset_results,
        "dataset_slope_pairwise": pairwise,
        "bootstrap_repetitions": repetitions,
        "interpretation_scope": "frozen_supervised_representation_decoder_label_efficiency",
        "not_end_to_end_sample_efficiency": True,
    }
    write_csv(output / "variant_metrics.csv", variant_rows)
    write_csv(output / "paired_seed_metrics.csv", paired_rows)
    write_csv(output / "subject_learning_curves.csv", subject_rows)
    write_json(output / "decision.json", decision)
    print(yaml.safe_dump(decision, sort_keys=False))


if __name__ == "__main__":
    main()
