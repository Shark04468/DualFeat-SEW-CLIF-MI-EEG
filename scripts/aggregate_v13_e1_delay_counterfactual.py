#!/usr/bin/env python3
"""Aggregate exact-checkpoint V13 full/zero/shuffled delay controls."""

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


CONTROLS = ("full", "zero", "shuffled")


def _csv_int(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError("integer list must be non-empty and unique")
    return parsed


def _prediction(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"indices", "logits", "endpoint_logits", "labels"}:
            raise RuntimeError(f"invalid V13 prediction archive: {path}")
        return (
            np.asarray(archive["indices"], dtype=np.int64),
            np.asarray(archive["labels"], dtype=np.int64),
            np.asarray(archive["logits"], dtype=np.float64),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--subjects", default="1,3,8")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--folds", default="0,1,2,3,4,5")
    parser.add_argument("--expected-source-digest", required=True)
    parser.add_argument("--minimum-median-gain-pp", type=float, default=0.5)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = ensure_dir(Path(args.output).resolve() if args.output else root / "aggregate")
    subjects = _csv_int(args.subjects)
    seeds = _csv_int(args.seeds)
    folds = _csv_int(args.folds)
    fold_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    fingerprints: list[str] = []
    source_digests: list[str] = []

    for subject in subjects:
        for seed in seeds:
            seen: list[int] = []
            for fold in folds:
                fold_dir = root / f"subject_{subject:02d}" / f"seed_{seed}" / f"fold_{fold}"
                status = read_json(fold_dir / "campaign_status.json")
                if (
                    status.get("status") != "completed"
                    or status.get("same_checkpoint_for_all_controls") is not True
                    or status.get("only_lag_posterior_intervened") is not True
                    or status.get("session_e_accessed") is not False
                ):
                    raise RuntimeError(f"invalid V13 control semantics: {fold_dir}")
                source_digests.append(
                    str(read_json(fold_dir / "source_tree_summary.json")["sha256"])
                )
                fingerprints.append(
                    str(read_json(fold_dir / "run_fingerprint.json")["combined_sha256"])
                )
                summary = pd.read_csv(fold_dir / "summary.csv")
                if set(summary["control"]) != set(CONTROLS):
                    raise RuntimeError(f"V13 controls are incomplete: {fold_dir}")
                predictions: dict[str, np.ndarray] = {}
                indices: np.ndarray | None = None
                labels: np.ndarray | None = None
                for control in CONTROLS:
                    saved_indices, saved_labels, logits = _prediction(
                        fold_dir / control / "outer_predictions.npz"
                    )
                    if indices is None:
                        indices, labels = saved_indices, saved_labels
                    elif not np.array_equal(indices, saved_indices) or not np.array_equal(
                        labels, saved_labels
                    ):
                        raise RuntimeError(f"V13 controls are not aligned: {fold_dir}")
                    predictions[control] = logits.argmax(axis=1)
                    source = summary.loc[summary["control"] == control].iloc[0]
                    fold_rows.append(
                        {
                            "subject": subject,
                            "seed": seed,
                            "fold": fold,
                            "control": control,
                            "accuracy": float(source["accuracy"]),
                            "kappa": float(source["kappa"]),
                            "mean_nonzero_delay_mass": float(
                                source["mean_nonzero_delay_mass"]
                            ),
                            "mean_delay_residual_rms": float(
                                source["mean_delay_residual_rms"]
                            ),
                        }
                    )
                assert indices is not None and labels is not None
                seen.extend(indices.tolist())
                with np.load(fold_dir / "delay_diagnostics.npz", allow_pickle=False) as archive:
                    route = np.asarray(archive["route_probability"], dtype=np.float64)
                    posterior = np.asarray(archive["lag_posterior"], dtype=np.float64)
                for channel in range(route.size):
                    route_rows.append(
                        {
                            "subject": subject,
                            "seed": seed,
                            "fold": fold,
                            "channel": channel,
                            "route_probability": float(route[channel]),
                            "nonzero_delay_mass": float(posterior[channel, 1:].sum()),
                            "expected_lag": float(
                                posterior[channel]
                                @ np.arange(posterior.shape[1], dtype=np.float64)
                            ),
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
                                f"prediction_{control}": int(prediction[offset])
                                for control, prediction in predictions.items()
                            },
                        }
                    )
            counts = Counter(seen)
            if len(counts) != 288 or set(counts.values()) != {1}:
                raise RuntimeError("V13 folds do not cover Session T exactly once")

    expected_runs = len(subjects) * len(seeds) * len(folds)
    if set(source_digests) != {args.expected_source_digest}:
        raise RuntimeError("V13 source snapshot mismatch")
    if len(set(fingerprints)) != expected_runs:
        raise RuntimeError("V13 fingerprints are duplicated or incomplete")
    trial_frame = pd.DataFrame(trial_rows)
    subject_seed_rows: list[dict[str, Any]] = []
    for (subject, seed), group in trial_frame.groupby(["subject", "seed"], sort=True):
        labels = group["label"].to_numpy()
        for control in CONTROLS:
            prediction = group[f"prediction_{control}"].to_numpy()
            subject_seed_rows.append(
                {
                    "subject": int(subject),
                    "seed": int(seed),
                    "control": control,
                    "accuracy": float(accuracy_score(labels, prediction)),
                    "kappa": float(cohen_kappa_score(labels, prediction)),
                }
            )
    subject_seed = pd.DataFrame(subject_seed_rows)
    comparisons: list[dict[str, Any]] = []
    full = subject_seed.loc[subject_seed["control"] == "full"].sort_values(
        ["subject", "seed"]
    )["accuracy"].to_numpy()
    for second in ("zero", "shuffled"):
        other = subject_seed.loc[subject_seed["control"] == second].sort_values(
            ["subject", "seed"]
        )["accuracy"].to_numpy()
        delta = 100.0 * (full - other)
        comparisons.append(
            {
                "first": "full",
                "second": second,
                "mean_delta_pp": float(delta.mean()),
                "median_delta_pp": float(np.median(delta)),
                "minimum_delta_pp": float(delta.min()),
                "positive_pairs": int(np.sum(delta > 0.0)),
                "negative_pairs": int(np.sum(delta < 0.0)),
                "tied_pairs": int(np.sum(delta == 0.0)),
            }
        )
    comparison_frame = pd.DataFrame(comparisons)
    zero = comparison_frame.loc[comparison_frame["second"] == "zero"].iloc[0]
    required_positive = 2 if len(seeds) == 1 else 6
    gate = {
        "status": "passed"
        if (
            float(zero["median_delta_pp"]) >= float(args.minimum_median_gain_pp)
            and int(zero["positive_pairs"]) >= required_positive
        )
        else "failed",
        "median_full_minus_zero_pp": float(zero["median_delta_pp"]),
        "mean_full_minus_zero_pp": float(zero["mean_delta_pp"]),
        "positive_pairs": int(zero["positive_pairs"]),
        "criteria": {
            "minimum_median_gain_pp": float(args.minimum_median_gain_pp),
            "minimum_positive_pairs": required_positive,
        },
    }

    write_csv(output / "fold_metrics.csv", fold_rows)
    write_csv(output / "trial_predictions.csv", trial_frame.to_dict(orient="records"))
    write_csv(output / "route_statistics.csv", route_rows)
    write_csv(
        output / "subject_seed_metrics.csv", subject_seed.to_dict(orient="records")
    )
    write_csv(
        output / "paired_comparisons.csv", comparison_frame.to_dict(orient="records")
    )
    audit = {
        "status": "passed",
        "expected_runs": expected_runs,
        "completed_runs": expected_runs,
        "source_tree_sha256": args.expected_source_digest,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "same_checkpoint_for_all_controls": True,
        "only_lag_posterior_intervened": True,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    write_json(output / "campaign_audit.json", audit)
    summary = {"status": "completed", "audit": audit, "delay_gate": gate}
    write_json(output / "aggregate_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
