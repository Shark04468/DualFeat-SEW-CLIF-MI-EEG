#!/usr/bin/env python3
"""Recompute E28-E30 supplementary metrics from sealed trial predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.utils.io import ensure_dir, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402


METRICS = ("accuracy", "balanced_accuracy", "kappa", "macro_f1")
MODELS = ("ann_sew_ce", "sew_clif_ce")


def _metrics(labels: np.ndarray, logits: np.ndarray, n_classes: int) -> dict[str, float]:
    values = classification_metrics(labels, logits.argmax(axis=1), n_classes=n_classes)
    return {key: float(values[key]) for key in METRICS}


def _e28_fold(root: Path, fallback: Path, subject: int, seed: int, fold: int) -> Path:
    relative = Path(f"subject_{subject:02d}") / f"seed_{seed}" / f"fold_{fold}"
    for candidate_root in (root, fallback):
        candidate = candidate_root / relative
        if all((candidate / model / "outer_predictions.npz").is_file() for model in MODELS):
            return candidate
    raise FileNotFoundError(f"missing E28 fold: {relative}")


def _load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        label_key = "labels" if "labels" in archive.files else "label"
        index_key = "indices" if "indices" in archive.files else None
        labels = np.asarray(archive[label_key], dtype=np.int64)
        logits = np.asarray(archive["logits"], dtype=np.float32)
        indices = (
            np.asarray(archive[index_key], dtype=np.int64)
            if index_key is not None
            else np.arange(labels.size, dtype=np.int64)
        )
    return indices, labels, logits


def _e28_rows(root: Path, fallback: Path) -> list[dict[str, Any]]:
    rows = []
    for subject in range(1, 10):
        for seed in range(3):
            model_parts: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
                model: [] for model in MODELS
            }
            for fold in range(6):
                fold_root = _e28_fold(root, fallback, subject, seed, fold)
                for model in MODELS:
                    model_parts[model].append(
                        _load_prediction(fold_root / model / "outer_predictions.npz")
                    )
            reference_indices = None
            reference_labels = None
            for model in MODELS:
                indices = np.concatenate([part[0] for part in model_parts[model]])
                order = np.argsort(indices)
                labels = np.concatenate([part[1] for part in model_parts[model]])[order]
                logits = np.concatenate([part[2] for part in model_parts[model]])[order]
                if reference_indices is None:
                    reference_indices = indices[order]
                    reference_labels = labels
                elif not np.array_equal(reference_indices, indices[order]) or not np.array_equal(
                    reference_labels, labels
                ):
                    raise RuntimeError("E28 paired OOF predictions are not aligned")
                rows.append(
                    {
                        "dataset": "bci2a",
                        "experiment": "E28",
                        "subject": subject,
                        "seed": seed,
                        "model": model,
                        **_metrics(labels, logits, 4),
                    }
                )
    return rows


def _external_rows(dataset: str, root: Path, subjects: int, seeds: int) -> list[dict[str, Any]]:
    rows = []
    for subject in range(1, subjects + 1):
        for seed in range(seeds):
            for model in MODELS:
                if dataset == "openbmi":
                    path = (
                        root
                        / f"subject_{subject:02d}"
                        / f"seed_{seed}"
                        / model
                        / "evaluation"
                        / "predictions.npz"
                    )
                else:
                    path = (
                        root
                        / f"subject_{subject:02d}"
                        / f"seed_{seed}"
                        / "students"
                        / model
                        / "evaluation"
                        / "predictions.npz"
                    )
                _, labels, logits = _load_prediction(path)
                rows.append(
                    {
                        "dataset": dataset,
                        "experiment": "E29" if dataset == "openbmi" else "E30",
                        "subject": subject,
                        "seed": seed,
                        "model": model,
                        **_metrics(labels, logits, 2),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e28-root", required=True)
    parser.add_argument("--e28-fallback", required=True)
    parser.add_argument("--e29-root", required=True)
    parser.add_argument("--e30-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = _e28_rows(Path(args.e28_root), Path(args.e28_fallback))
    rows.extend(_external_rows("openbmi", Path(args.e29_root), 54, 5))
    rows.extend(_external_rows("bnci2014_004", Path(args.e30_root), 9, 5))
    subject_rows: list[dict[str, Any]] = []
    keys = sorted({(row["dataset"], row["experiment"], row["subject"], row["model"]) for row in rows})
    for dataset, experiment, subject, model in keys:
        selected = [
            row
            for row in rows
            if (row["dataset"], row["experiment"], row["subject"], row["model"])
            == (dataset, experiment, subject, model)
        ]
        subject_rows.append(
            {
                "dataset": dataset,
                "experiment": experiment,
                "subject": subject,
                "model": model,
                **{
                    metric: float(np.mean([float(row[metric]) for row in selected]))
                    for metric in METRICS
                },
            }
        )
    summary_rows: list[dict[str, Any]] = []
    for dataset, experiment, model in sorted(
        {(row["dataset"], row["experiment"], row["model"]) for row in subject_rows}
    ):
        selected = [
            row
            for row in subject_rows
            if (row["dataset"], row["experiment"], row["model"])
            == (dataset, experiment, model)
        ]
        summary_rows.append(
            {
                "dataset": dataset,
                "experiment": experiment,
                "model": model,
                "subjects": len(selected),
                **{
                    metric: float(np.mean([float(row[metric]) for row in selected]))
                    for metric in METRICS
                },
            }
        )
    output = ensure_dir(Path(args.output).resolve())
    write_csv(output / "subject_seed_metrics.csv", rows)
    write_csv(output / "subject_metrics.csv", subject_rows)
    write_csv(output / "supplementary_metric_summary.csv", summary_rows)
    write_json(
        output / "status.json",
        {
            "status": "completed",
            "subject_seed_rows": len(rows),
            "subject_rows": len(subject_rows),
            "summary_rows": len(summary_rows),
            "aggregation": "seed_mean_then_subject_macro_mean",
        },
    )


if __name__ == "__main__":
    main()
