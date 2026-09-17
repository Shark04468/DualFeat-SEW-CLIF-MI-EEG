#!/usr/bin/env python3
"""Aggregate E29-ZP or E30-ZP against read-only parent controls."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import wilcoxon
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402


METRICS = ("accuracy", "balanced_accuracy", "kappa", "macro_f1")


def _paths(
    dataset: str, root: Path, parent: Path, subject: int, seed: int
) -> dict[str, Path]:
    if dataset == "openbmi":
        parent_seed = parent / f"subject_{subject:02d}" / f"seed_{seed}"
        child = root / f"subject_{subject:02d}" / f"seed_{seed}" / "sew_clif_ce_fr0"
        return {
            "ann_metrics": parent_seed / "ann_sew_ce" / "evaluation" / "metrics.json",
            "ann_predictions": parent_seed / "ann_sew_ce" / "evaluation" / "predictions.npz",
            "old_metrics": parent_seed / "sew_clif_ce" / "evaluation" / "metrics.json",
            "zero_metrics": child / "evaluation" / "metrics.json",
            "zero_predictions": child / "evaluation" / "predictions.npz",
        }
    parent_seed = parent / f"subject_{subject:02d}" / f"seed_{seed}" / "students"
    child = (
        root
        / f"subject_{subject:02d}"
        / f"seed_{seed}"
        / "students"
        / "sew_clif_ce_fr0"
    )
    return {
        "ann_metrics": parent_seed / "ann_sew_ce" / "evaluation" / "metrics.json",
        "ann_predictions": parent_seed / "ann_sew_ce" / "evaluation" / "predictions.npz",
        "old_metrics": parent_seed / "sew_clif_ce" / "evaluation" / "metrics.json",
        "zero_metrics": child / "evaluation" / "metrics.json",
        "zero_predictions": child / "evaluation" / "predictions.npz",
    }


def _identity(ann: Path, zero: Path) -> None:
    with np.load(ann, allow_pickle=False) as first, np.load(
        zero, allow_pickle=False
    ) as second:
        for key in ("label", "trial_id"):
            if not np.array_equal(first[key], second[key]):
                raise RuntimeError(f"paired prediction identity mismatch: {zero} {key}")


def _metric_summary(values: np.ndarray, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    nonzero = values[values != 0.0]
    pvalue = (
        float(wilcoxon(nonzero, alternative="greater", method="auto").pvalue)
        if nonzero.size
        else 1.0
    )
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, values.size, size=(int(bootstrap_samples), values.size)
    )
    bootstrap = values[indices].mean(axis=1)
    return {
        "mean_delta": float(values.mean()),
        "mean_delta_pp": 100.0 * float(values.mean()),
        "median_delta_pp": 100.0 * float(np.median(values)),
        "positive_subjects": int(np.count_nonzero(values > 0.0)),
        "negative_subjects": int(np.count_nonzero(values < 0.0)),
        "ties": int(np.count_nonzero(values == 0.0)),
        "subject_bootstrap_95_ci_pp": [
            100.0 * float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
        ],
        "one_sided_wilcoxon_p": pvalue,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("openbmi", "bnci2014_004"), required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    args = parser.parse_args()
    config_name = (
        "v33_e29_zero_penalty.yaml"
        if args.dataset == "openbmi"
        else "v33_e30_zero_penalty.yaml"
    )
    config = yaml.safe_load(
        (ROOT / "configs" / "experiments" / config_name).read_text(encoding="utf-8")
    )
    subjects = (
        config["subjects"]
        if args.dataset == "openbmi"
        else config["dataset"]["subjects"]
    )
    root = Path(args.root).resolve()
    parent = Path(args.parent).resolve()
    output = ensure_dir(Path(args.output).resolve())
    rows: list[dict[str, Any]] = []
    for subject_value in subjects:
        subject = int(subject_value)
        status = read_json(root / f"subject_{subject:02d}" / "evaluation_status.json")
        if status.get("status") != "completed":
            raise RuntimeError(f"incomplete objective-pure evaluation: {subject}")
        for seed_value in config["seeds"]:
            seed = int(seed_value)
            paths = _paths(args.dataset, root, parent, subject, seed)
            _identity(paths["ann_predictions"], paths["zero_predictions"])
            ann = read_json(paths["ann_metrics"])
            old = read_json(paths["old_metrics"])
            zero = read_json(paths["zero_metrics"])
            rows.append(
                {
                    "dataset": args.dataset,
                    "subject": subject,
                    "seed": seed,
                    **{f"ann_{key}": float(ann[key]) for key in METRICS},
                    **{f"old_snn_{key}": float(old[key]) for key in METRICS},
                    **{f"zero_snn_{key}": float(zero[key]) for key in METRICS},
                    **{
                        f"zero_gain_{key}": float(zero[key]) - float(ann[key])
                        for key in METRICS
                    },
                    **{
                        f"old_gain_{key}": float(old[key]) - float(ann[key])
                        for key in METRICS
                    },
                }
            )
    subject_rows: list[dict[str, Any]] = []
    numeric_keys = [key for key in rows[0] if key not in {"dataset", "subject", "seed"}]
    for subject_value in subjects:
        subject = int(subject_value)
        selected = [row for row in rows if row["subject"] == subject]
        subject_rows.append(
            {
                "dataset": args.dataset,
                "subject": subject,
                **{
                    key: float(np.mean([float(row[key]) for row in selected]))
                    for key in numeric_keys
                },
            }
        )
    comparisons = {
        metric: _metric_summary(
            np.asarray([row[f"zero_gain_{metric}"] for row in subject_rows]),
            int(args.bootstrap_samples),
            33_000 + index,
        )
        for index, metric in enumerate(METRICS)
    }
    decision = {
        "schema": f"dpc-snn-{args.dataset}-objective-pure-decision/v1",
        "status": "completed",
        "dataset": args.dataset,
        "subjects": len(subjects),
        "seeds": len(config["seeds"]),
        "mean_accuracy": {
            "ann": float(np.mean([row["ann_accuracy"] for row in subject_rows])),
            "old_snn_fr001": float(
                np.mean([row["old_snn_accuracy"] for row in subject_rows])
            ),
            "zero_snn_fr0": float(
                np.mean([row["zero_snn_accuracy"] for row in subject_rows])
            ),
        },
        "comparisons": comparisons,
        "historical_evaluation_exposure": True,
        "interpretation": "retrospective_objective_pure_replication",
    }
    write_csv(output / "subject_seed_metrics.csv", rows)
    write_csv(output / "subject_metrics.csv", subject_rows)
    write_json(output / "decision.json", decision)
    print(yaml.safe_dump(decision, sort_keys=False))


if __name__ == "__main__":
    main()
