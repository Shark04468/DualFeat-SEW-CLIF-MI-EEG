#!/usr/bin/env python3
"""Aggregate V12 bounded multi-rate architecture experiments."""

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

from dpc_snn.models.v12_multirate_student import V12_MODEL_ARCHITECTURES  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402


ANCHOR = "v9_sew_clif_kd"


def _csv_int(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError("integer list must be non-empty and unique")
    return parsed


def _csv_variants(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError("variant list must be non-empty and unique")
    unknown = set(parsed).difference(V12_MODEL_ARCHITECTURES)
    if unknown:
        raise ValueError(f"unknown V12 variants: {sorted(unknown)}")
    return parsed


def _prediction(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"indices", "logits", "endpoint_logits", "labels"}
        if set(archive.files) != required:
            raise RuntimeError(f"invalid V12 prediction archive: {path}")
        indices = np.asarray(archive["indices"], dtype=np.int64)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        logits = np.asarray(archive["logits"], dtype=np.float64)
        endpoints = np.asarray(archive["endpoint_logits"], dtype=np.float64)
    if logits.shape != (indices.size, 4) or endpoints.shape[0] != indices.size:
        raise RuntimeError(f"malformed V12 prediction archive: {path}")
    return indices, labels, logits, endpoints


def _anchor(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        indices = np.asarray(archive["indices"], dtype=np.int64)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        logits = np.asarray(archive["logits"], dtype=np.float64)
    return indices, labels, logits


def _metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    prediction = logits.argmax(axis=1)
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
    parser.add_argument("--variants", default=",".join(V12_MODEL_ARCHITECTURES))
    parser.add_argument("--expected-source-digest", required=True)
    parser.add_argument("--minimum-gain-pp", type=float, default=0.5)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    anchor_root = Path(args.anchor_root).resolve()
    output = ensure_dir(Path(args.output).resolve() if args.output else root / "aggregate")
    subjects = _csv_int(args.subjects)
    seeds = _csv_int(args.seeds)
    folds = _csv_int(args.folds)
    variants = _csv_variants(args.variants)
    rows: list[dict[str, Any]] = []
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
                if status.get("status") != "completed" or status.get(
                    "session_e_accessed"
                ) is not False:
                    raise RuntimeError(f"incomplete or unlocked V12 fold: {fold_dir}")
                if int(status.get("total_hpo_configurations", -1)) > 12:
                    raise RuntimeError("V12 bounded search exceeded 12 configurations")
                source_digests.append(
                    str(read_json(fold_dir / "source_tree_summary.json")["sha256"])
                )
                fingerprints.append(
                    str(read_json(fold_dir / "run_fingerprint.json")["combined_sha256"])
                )
                summary = pd.read_csv(fold_dir / "summary.csv")
                if set(summary["variant"]) != set(variants):
                    raise RuntimeError(f"V12 variants are incomplete: {fold_dir}")
                predictions: dict[str, np.ndarray] = {}
                indices: np.ndarray | None = None
                labels: np.ndarray | None = None
                for variant in variants:
                    saved_indices, saved_labels, logits, endpoints = _prediction(
                        fold_dir / variant / "outer_predictions.npz"
                    )
                    if indices is None:
                        indices, labels = saved_indices, saved_labels
                    elif not np.array_equal(indices, saved_indices) or not np.array_equal(
                        labels, saved_labels
                    ):
                        raise RuntimeError(f"V12 predictions are not aligned: {fold_dir}")
                    predictions[variant] = logits
                    for endpoint in range(endpoints.shape[1]):
                        endpoint_seconds = (
                            4.0 if endpoints.shape[1] == 1 else float(endpoint + 1)
                        )
                        endpoint_rows.append(
                            {
                                "subject": subject,
                                "seed": seed,
                                "fold": fold,
                                "variant": variant,
                                "endpoint": endpoint + 1,
                                "endpoint_seconds": endpoint_seconds,
                                **_metrics(saved_labels, endpoints[:, endpoint]),
                            }
                        )
                assert indices is not None and labels is not None
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
                    raise RuntimeError("V12 and V9 anchor predictions are not aligned")
                predictions[ANCHOR] = anchor_logits
                seen.extend(indices.tolist())
                for variant, logits in predictions.items():
                    rows.append(
                        {
                            "subject": subject,
                            "seed": seed,
                            "fold": fold,
                            "variant": variant,
                            **_metrics(labels, logits),
                        }
                    )
                for offset, trial_index in enumerate(indices):
                    trial_rows.append(
                        {
                            "subject": subject,
                            "seed": seed,
                            "fold": fold,
                            "trial_index": int(trial_index),
                            "label": int(labels[offset]),
                            **{
                                f"prediction_{variant}": int(logits[offset].argmax())
                                for variant, logits in predictions.items()
                            },
                        }
                    )
            counts = Counter(seen)
            if len(counts) != 288 or set(counts.values()) != {1}:
                raise RuntimeError("V12 outer folds do not cover Session T exactly once")

    expected_runs = len(subjects) * len(seeds) * len(folds)
    if set(source_digests) != {args.expected_source_digest}:
        raise RuntimeError("V12 source snapshot mismatch")
    if len(set(fingerprints)) != expected_runs:
        raise RuntimeError("V12 fingerprints are duplicated or incomplete")

    fold_frame = pd.DataFrame(rows)
    trial_frame = pd.DataFrame(trial_rows)
    subject_seed_rows: list[dict[str, Any]] = []
    all_variants = (*variants, ANCHOR)
    for (subject, seed), group in trial_frame.groupby(["subject", "seed"], sort=True):
        labels = group["label"].to_numpy()
        for variant in all_variants:
            prediction = group[f"prediction_{variant}"].to_numpy()
            subject_seed_rows.append(
                {
                    "subject": int(subject),
                    "seed": int(seed),
                    "variant": variant,
                    "accuracy": float(accuracy_score(labels, prediction)),
                    "kappa": float(cohen_kappa_score(labels, prediction)),
                }
            )
    subject_seed = pd.DataFrame(subject_seed_rows)
    model_summary = (
        subject_seed.groupby("variant", as_index=False)
        .agg(
            subject_seed_pairs=("accuracy", "size"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            kappa_mean=("kappa", "mean"),
            kappa_std=("kappa", "std"),
        )
        .sort_values("accuracy_mean", ascending=False)
    )
    anchor_values = subject_seed.loc[
        subject_seed["variant"] == ANCHOR
    ].sort_values(["subject", "seed"])["accuracy"].to_numpy()
    comparison_rows: list[dict[str, Any]] = []
    for variant in variants:
        values = subject_seed.loc[
            subject_seed["variant"] == variant
        ].sort_values(["subject", "seed"])["accuracy"].to_numpy()
        delta = 100.0 * (values - anchor_values)
        comparison_rows.append(
            {
                "variant": variant,
                "reference": ANCHOR,
                "mean_delta_pp": float(delta.mean()),
                "median_delta_pp": float(np.median(delta)),
                "minimum_delta_pp": float(delta.min()),
                "positive_pairs": int(np.sum(delta > 0.0)),
                "negative_pairs": int(np.sum(delta < 0.0)),
                "tied_pairs": int(np.sum(delta == 0.0)),
            }
        )
    comparisons = pd.DataFrame(comparison_rows).sort_values(
        ["mean_delta_pp", "median_delta_pp"], ascending=False
    )
    selected = str(comparisons.iloc[0]["variant"])
    selected_row = comparisons.iloc[0]
    required_positive = 2 if len(seeds) == 1 else 6
    promotion = {
        "status": "passed"
        if (
            float(selected_row["mean_delta_pp"]) >= float(args.minimum_gain_pp)
            and int(selected_row["positive_pairs"]) >= required_positive
        )
        else "failed",
        "selected_variant": selected,
        "mean_delta_pp": float(selected_row["mean_delta_pp"]),
        "median_delta_pp": float(selected_row["median_delta_pp"]),
        "positive_pairs": int(selected_row["positive_pairs"]),
        "criteria": {
            "minimum_mean_gain_pp": float(args.minimum_gain_pp),
            "minimum_positive_pairs": required_positive,
        },
    }

    write_csv(output / "fold_metrics.csv", fold_frame.to_dict(orient="records"))
    write_csv(output / "endpoint_metrics.csv", endpoint_rows)
    write_csv(output / "trial_predictions.csv", trial_frame.to_dict(orient="records"))
    write_csv(
        output / "subject_seed_metrics.csv", subject_seed.to_dict(orient="records")
    )
    write_csv(output / "model_summary.csv", model_summary.to_dict(orient="records"))
    write_csv(output / "paired_comparisons.csv", comparisons.to_dict(orient="records"))
    audit = {
        "status": "passed",
        "expected_runs": expected_runs,
        "completed_runs": expected_runs,
        "subjects": list(subjects),
        "seeds": list(seeds),
        "folds": list(folds),
        "source_tree_sha256": args.expected_source_digest,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "unique_run_fingerprints": len(set(fingerprints)),
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    write_json(output / "campaign_audit.json", audit)
    summary = {"status": "completed", "audit": audit, "promotion": promotion}
    write_json(output / "aggregate_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
