#!/usr/bin/env python3
"""Aggregate matched ANN/PLIF/CLIF necessity and utility controls."""

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

from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402


KINDS = ("ann", "plif", "clif")


def _csv_int(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError("integer list must be non-empty and unique")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--base-variant", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--subjects", default="1,3,8")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--folds", default="0,1,2,3,4,5")
    parser.add_argument("--expected-source-digest", required=True)
    parser.add_argument("--noninferiority-margin-pp", type=float, default=-0.3)
    parser.add_argument("--utility-gain-pp", type=float, default=0.5)
    parser.add_argument("--maximum-firing-rate", type=float, default=0.25)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = ensure_dir(Path(args.output).resolve() if args.output else root / "aggregate")
    subjects = _csv_int(args.subjects)
    seeds = _csv_int(args.seeds)
    folds = _csv_int(args.folds)
    models = {kind: f"{args.base_variant}__{kind}" for kind in KINDS}
    fold_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
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
                    raise RuntimeError(f"invalid E14 fold: {fold_dir}")
                expected_specs = {
                    (args.base_variant, kind) for kind in KINDS
                }
                found_specs = {
                    (str(row["variant"]), str(row["decoder_kind"]))
                    for row in status["model_specs"]
                }
                if found_specs != expected_specs:
                    raise RuntimeError(f"E14 neuron controls are incomplete: {fold_dir}")
                source_digests.append(
                    str(read_json(fold_dir / "source_tree_summary.json")["sha256"])
                )
                fingerprints.append(
                    str(read_json(fold_dir / "run_fingerprint.json")["combined_sha256"])
                )
                summary = pd.read_csv(fold_dir / "summary.csv")
                endpoint = pd.read_csv(fold_dir / "endpoint_metrics.csv")
                robustness = pd.read_csv(fold_dir / "robustness_metrics.csv")
                for row in summary.to_dict(orient="records"):
                    fold_rows.append(row)
                endpoint_rows.extend(endpoint.to_dict(orient="records"))
                robustness_rows.extend(robustness.to_dict(orient="records"))

                indices: np.ndarray | None = None
                labels: np.ndarray | None = None
                predictions: dict[str, np.ndarray] = {}
                for kind, model in models.items():
                    with np.load(
                        fold_dir / model / "outer_predictions.npz", allow_pickle=False
                    ) as archive:
                        saved_indices = np.asarray(archive["indices"], dtype=np.int64)
                        saved_labels = np.asarray(archive["labels"], dtype=np.int64)
                        logits = np.asarray(archive["logits"], dtype=np.float64)
                    if indices is None:
                        indices, labels = saved_indices, saved_labels
                    elif not np.array_equal(indices, saved_indices) or not np.array_equal(
                        labels, saved_labels
                    ):
                        raise RuntimeError(f"E14 predictions are not aligned: {fold_dir}")
                    predictions[kind] = logits.argmax(axis=1)
                assert indices is not None and labels is not None
                seen.extend(indices.tolist())
                for offset, trial_index in enumerate(indices):
                    trial_rows.append(
                        {
                            "subject": subject,
                            "seed": seed,
                            "fold": fold,
                            "trial_index": int(trial_index),
                            "label": int(labels[offset]),
                            **{
                                f"prediction_{kind}": int(prediction[offset])
                                for kind, prediction in predictions.items()
                            },
                        }
                    )
            counts = Counter(seen)
            if len(counts) != 288 or set(counts.values()) != {1}:
                raise RuntimeError("E14 folds do not cover Session T exactly once")

    expected_runs = len(subjects) * len(seeds) * len(folds)
    if set(source_digests) != {args.expected_source_digest}:
        raise RuntimeError("E14 source snapshot mismatch")
    if len(set(fingerprints)) != expected_runs:
        raise RuntimeError("E14 fingerprints are duplicated or incomplete")

    trial_frame = pd.DataFrame(trial_rows)
    subject_seed_rows: list[dict[str, Any]] = []
    for (subject, seed), group in trial_frame.groupby(["subject", "seed"], sort=True):
        labels = group["label"].to_numpy()
        for kind in KINDS:
            prediction = group[f"prediction_{kind}"].to_numpy()
            subject_seed_rows.append(
                {
                    "subject": int(subject),
                    "seed": int(seed),
                    "decoder_kind": kind,
                    "accuracy": float(accuracy_score(labels, prediction)),
                    "kappa": float(cohen_kappa_score(labels, prediction)),
                }
            )
    subject_seed = pd.DataFrame(subject_seed_rows)
    subject = (
        subject_seed.groupby(["subject", "decoder_kind"], as_index=False)[
            ["accuracy", "kappa"]
        ].mean()
    )
    model_summary = (
        subject.groupby("decoder_kind", as_index=False)
        .agg(
            subjects=("subject", "size"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            kappa_mean=("kappa", "mean"),
            kappa_std=("kappa", "std"),
        )
        .sort_values("accuracy_mean", ascending=False)
    )
    pair_rows: list[dict[str, Any]] = []
    clif = subject_seed.loc[subject_seed["decoder_kind"] == "clif"].sort_values(
        ["subject", "seed"]
    )["accuracy"].to_numpy()
    for second in ("ann", "plif"):
        other = subject_seed.loc[subject_seed["decoder_kind"] == second].sort_values(
            ["subject", "seed"]
        )["accuracy"].to_numpy()
        delta = 100.0 * (clif - other)
        pair_rows.append(
            {
                "first": "clif",
                "second": second,
                "mean_delta_pp": float(delta.mean()),
                "median_delta_pp": float(np.median(delta)),
                "positive_pairs": int(np.sum(delta > 0.0)),
                "negative_pairs": int(np.sum(delta < 0.0)),
                "tied_pairs": int(np.sum(delta == 0.0)),
            }
        )
    pairs = pd.DataFrame(pair_rows)

    endpoint = pd.DataFrame(endpoint_rows)
    endpoint_summary = (
        endpoint.groupby(["decoder_kind", "endpoint_seconds"], as_index=False)["accuracy"]
        .mean()
        .sort_values(["endpoint_seconds", "decoder_kind"])
    )
    robustness = pd.DataFrame(robustness_rows)
    robustness_summary = (
        robustness.groupby(["decoder_kind", "perturbation"], as_index=False)["accuracy"]
        .mean()
    )
    early = endpoint_summary.loc[endpoint_summary["endpoint_seconds"] < 4.0]
    early_pivot = early.pivot(
        index="endpoint_seconds", columns="decoder_kind", values="accuracy"
    )
    early_gain = (
        100.0 * float((early_pivot["clif"] - early_pivot["ann"]).max())
        if not early_pivot.empty
        else float("-inf")
    )
    robust = robustness_summary.loc[
        robustness_summary["perturbation"] != "nominal"
    ].groupby("decoder_kind")["accuracy"].mean()
    robustness_gain = 100.0 * float(robust["clif"] - robust["ann"])
    fold_frame = pd.DataFrame(fold_rows)
    firing_rate = float(
        fold_frame.loc[fold_frame["decoder_kind"] == "clif", "mean_firing_rate"].mean()
    )
    clif_ann = pairs.loc[pairs["second"] == "ann"].iloc[0]
    accuracy_noninferior = float(clif_ann["mean_delta_pp"]) >= float(
        args.noninferiority_margin_pp
    )
    utility = {
        "early_decision": early_gain >= float(args.utility_gain_pp),
        "robustness": robustness_gain >= float(args.utility_gain_pp),
        "binary_activity_sparsity": firing_rate <= float(args.maximum_firing_rate),
    }
    gate = {
        "status": "passed" if accuracy_noninferior and any(utility.values()) else "failed",
        "clif_minus_ann_mean_pp": float(clif_ann["mean_delta_pp"]),
        "maximum_early_decision_gain_pp": early_gain,
        "mean_robustness_gain_pp": robustness_gain,
        "clif_mean_firing_rate": firing_rate,
        "accuracy_noninferior": accuracy_noninferior,
        "utility_wins": utility,
        "criteria": {
            "noninferiority_margin_pp": float(args.noninferiority_margin_pp),
            "utility_gain_pp": float(args.utility_gain_pp),
            "maximum_firing_rate": float(args.maximum_firing_rate),
        },
    }

    write_csv(output / "fold_metrics.csv", fold_rows)
    write_csv(output / "endpoint_metrics.csv", endpoint_rows)
    write_csv(output / "robustness_metrics.csv", robustness_rows)
    write_csv(output / "trial_predictions.csv", trial_frame.to_dict(orient="records"))
    write_csv(
        output / "subject_seed_metrics.csv", subject_seed.to_dict(orient="records")
    )
    write_csv(output / "subject_metrics.csv", subject.to_dict(orient="records"))
    write_csv(output / "model_summary.csv", model_summary.to_dict(orient="records"))
    write_csv(output / "paired_comparisons.csv", pairs.to_dict(orient="records"))
    write_csv(
        output / "endpoint_summary.csv", endpoint_summary.to_dict(orient="records")
    )
    write_csv(
        output / "robustness_summary.csv", robustness_summary.to_dict(orient="records")
    )
    audit = {
        "status": "passed",
        "expected_runs": expected_runs,
        "completed_runs": expected_runs,
        "source_tree_sha256": args.expected_source_digest,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
        "gpu_latency_scope": "cached-feature decoder on conventional CUDA hardware",
    }
    write_json(output / "campaign_audit.json", audit)
    summary = {"status": "completed", "audit": audit, "snn_necessity_gate": gate}
    write_json(output / "aggregate_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
