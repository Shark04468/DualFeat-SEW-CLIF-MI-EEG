#!/usr/bin/env python3
"""Aggregate V31 paired curves and test the predeclared cross-dataset slope."""

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

from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402


def _design(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    datasets = sorted({str(row["dataset"]) for row in rows})
    reference = datasets[0]
    x_rows: list[list[float]] = []
    y: list[float] = []
    for row in rows:
        x_rows.append(
            [
                1.0,
                float(np.log(float(row["examples_per_class"]))),
                *[float(row["dataset"] == value) for value in datasets if value != reference],
            ]
        )
        y.append(float(row["gain"]))
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y, dtype=np.float64)


def _slope(rows: list[dict[str, Any]]) -> float:
    x, y = _design(rows)
    if x.shape[0] <= x.shape[1]:
        raise RuntimeError("insufficient V31 rows for the fixed-effect slope")
    return float(np.linalg.lstsq(x, y, rcond=None)[0][1])


def _cluster_bootstrap(
    rows: list[dict[str, Any]], repetitions: int, seed: int = 31_031
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    by_dataset: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset"]), {}).setdefault(
            int(row["subject"]), []
        ).append(row)
    values: list[float] = []
    for _ in range(int(repetitions)):
        sample: list[dict[str, Any]] = []
        for subjects in by_dataset.values():
            keys = sorted(subjects)
            selected = generator.choice(keys, size=len(keys), replace=True)
            for subject in selected:
                sample.extend(subjects[int(subject)])
        values.append(_slope(sample))
    return np.asarray(values, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.input).resolve()
    output = ensure_dir(Path(args.output).resolve())
    config = yaml.safe_load(
        (ROOT / "configs" / "experiments" / "v31_decoder_learning_curve.yaml").read_text(
            encoding="utf-8"
        )
    )
    metric_rows: list[dict[str, Any]] = []
    for dataset, dataset_config in config["datasets"].items():
        for subject in dataset_config["subjects"]:
            subject_root = root / dataset / f"subject_{int(subject):02d}"
            status_path = subject_root / "evaluation_status.json"
            if not status_path.is_file() or read_json(status_path).get("status") != "completed":
                raise RuntimeError(f"V31 evaluation is incomplete: {status_path}")
            for seed in config["seeds"]:
                seed_root = subject_root / f"seed_{int(seed)}"
                for budget_dir in sorted(seed_root.glob("budget_*")):
                    subset = read_json(budget_dir / "subset.json")
                    for variant in config["variants"]:
                        metrics = read_json(
                            budget_dir / str(variant) / "evaluation" / "metrics.json"
                        )
                        metric_rows.append(
                            {
                                "dataset": dataset,
                                "subject": int(subject),
                                "seed": int(seed),
                                "budget": subset["label"],
                                "examples_per_class": float(subset["examples_per_class"]),
                                "variant": str(variant),
                                "accuracy": float(metrics["accuracy"]),
                                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                                "kappa": float(metrics["kappa"]),
                                "macro_f1": float(metrics["macro_f1"]),
                                "mean_firing_rate": float(metrics["mean_firing_rate"]),
                            }
                        )
    write_csv(output / "variant_metrics.csv", metric_rows)

    lookup = {
        (row["dataset"], row["subject"], row["seed"], row["budget"], row["variant"]): row
        for row in metric_rows
    }
    paired: list[dict[str, Any]] = []
    keys = sorted(
        {
            (row["dataset"], row["subject"], row["seed"], row["budget"])
            for row in metric_rows
        }
    )
    for dataset, subject, seed, budget in keys:
        ann = lookup[(dataset, subject, seed, budget, "ann_sew_ce")]
        snn = lookup[(dataset, subject, seed, budget, "sew_clif_ce")]
        paired.append(
            {
                "dataset": dataset,
                "subject": subject,
                "seed": seed,
                "budget": budget,
                "examples_per_class": ann["examples_per_class"],
                "ann_accuracy": ann["accuracy"],
                "snn_accuracy": snn["accuracy"],
                "gain": snn["accuracy"] - ann["accuracy"],
            }
        )
    write_csv(output / "paired_seed_metrics.csv", paired)

    subject_rows: list[dict[str, Any]] = []
    subject_keys = sorted(
        {(row["dataset"], row["subject"], row["budget"]) for row in paired}
    )
    for dataset, subject, budget in subject_keys:
        values = [
            row
            for row in paired
            if (row["dataset"], row["subject"], row["budget"])
            == (dataset, subject, budget)
        ]
        subject_rows.append(
            {
                "dataset": dataset,
                "subject": subject,
                "budget": budget,
                "examples_per_class": float(
                    np.mean([row["examples_per_class"] for row in values])
                ),
                "ann_accuracy": float(np.mean([row["ann_accuracy"] for row in values])),
                "snn_accuracy": float(np.mean([row["snn_accuracy"] for row in values])),
                "gain": float(np.mean([row["gain"] for row in values])),
            }
        )
    write_csv(output / "subject_learning_curves.csv", subject_rows)

    observed = _slope(subject_rows)
    repetitions = int(config["primary_analysis"]["subject_cluster_bootstrap_repetitions"])
    bootstrap = _cluster_bootstrap(subject_rows, repetitions)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    dataset_slopes = {
        dataset: _slope([row for row in subject_rows if row["dataset"] == dataset])
        for dataset in sorted(config["datasets"])
    }
    decision = {
        "schema": "dpc-snn-v31-learning-curve-decision/v1",
        "status": "pass" if float(upper) < 0.0 else "fail",
        "slope_accuracy_gain_per_log_example": observed,
        "bootstrap_95_ci": [float(lower), float(upper)],
        "one_sided_bootstrap_p": float((np.count_nonzero(bootstrap >= 0.0) + 1) / (repetitions + 1)),
        "bootstrap_repetitions": repetitions,
        "dataset_slopes": dataset_slopes,
        "interpretation_scope": "frozen_supervised_representation_decoder_label_efficiency",
        "not_end_to_end_sample_efficiency": True,
    }
    write_json(output / "decision.json", decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
