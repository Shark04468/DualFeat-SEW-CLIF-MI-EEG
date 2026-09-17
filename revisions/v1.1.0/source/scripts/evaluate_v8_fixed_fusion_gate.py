#!/usr/bin/env python3
"""Evaluate the pre-specified equal-probability ATCNet/FBCNet fusion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import file_sha256  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from scripts.analyze_v8_e1_fusion import _accuracy, _aligned, _load  # noqa: E402


DEFAULT_SUBJECTS = (1, 3, 8)
DEFAULT_SEEDS = (0, 1, 2)
FIXED_ANCHOR_WEIGHT = 0.5
MINIMUM_MEDIAN_DELTA_PP = 0.5
MINIMUM_POSITIVE_PAIRS = 7
MINIMUM_MEAN_DELTA_PP = 0.5


def _csv(value: str, cast: Any = str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _validate_run_metadata(run_dir: Path, *, subject: int, seed: int) -> Mapping[str, Any]:
    metrics = read_json(run_dir / "metrics.json")
    expected = {
        "status": "completed",
        "stage": "E1",
        "protocol": "bci2a_session_t_nested_six_fold_oof",
        "subject": int(subject),
        "seed": int(seed),
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    mismatches = {
        key: {"expected": value, "actual": metrics.get(key)}
        for key, value in expected.items()
        if metrics.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"invalid E1 run metadata in {run_dir}: {mismatches}")
    fingerprint = metrics.get("run_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise RuntimeError(f"missing run fingerprint in {run_dir}")
    return metrics


def two_way_cluster_bootstrap_mean_ci(
    delta_pp: np.ndarray,
    *,
    samples: int = 20_000,
    seed: int = 20_260_718,
) -> tuple[float, float]:
    """Bootstrap crossed subject and seed clusters, not individual trials."""

    values = np.asarray(delta_pp, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("delta_pp must be a subject-by-seed matrix")
    if samples < 1:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        subject_index = rng.integers(0, values.shape[0], size=values.shape[0])
        seed_index = rng.integers(0, values.shape[1], size=values.shape[1])
        means[index] = values[np.ix_(subject_index, seed_index)].mean()
    lower, upper = np.quantile(means, (0.025, 0.975))
    return float(lower), float(upper)


def fixed_fusion_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_median_delta_pp: float = MINIMUM_MEDIAN_DELTA_PP,
    minimum_positive_pairs: int = MINIMUM_POSITIVE_PAIRS,
    minimum_mean_delta_pp: float = MINIMUM_MEAN_DELTA_PP,
) -> dict[str, Any]:
    deltas = np.asarray([float(row["delta_pp"]) for row in rows], dtype=np.float64)
    if deltas.size == 0:
        raise ValueError("at least one paired result is required")
    positive_pairs = int(np.sum(deltas > 0.0))
    mean_delta = float(deltas.mean())
    median_delta = float(np.median(deltas))
    checks = {
        "mean_delta": mean_delta >= float(minimum_mean_delta_pp),
        "median_delta": median_delta >= float(minimum_median_delta_pp),
        "positive_pairs": positive_pairs >= int(minimum_positive_pairs),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "minimum_mean_delta_pp": float(minimum_mean_delta_pp),
            "minimum_median_delta_pp": float(minimum_median_delta_pp),
            "minimum_positive_pairs": int(minimum_positive_pairs),
        },
        "mean_delta_pp": mean_delta,
        "median_delta_pp": median_delta,
        "positive_pairs": positive_pairs,
        "total_pairs": int(deltas.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--anchor", default="atcnet")
    parser.add_argument("--branch", default="fbcnet")
    parser.add_argument("--subjects", default="1,3,8")
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()

    root = Path(args.e1_root).resolve()
    output = ensure_dir(Path(args.output).resolve())
    subjects = _csv(args.subjects, int)
    seeds = _csv(args.seeds, int)
    expected_pairs = len(subjects) * len(seeds)
    if expected_pairs != 9:
        raise RuntimeError("the locked development gate requires exactly 3 subjects x 3 seeds")

    rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    delta_matrix = np.empty((len(subjects), len(seeds)), dtype=np.float64)
    for subject_index, subject in enumerate(subjects):
        for seed_index, seed in enumerate(seeds):
            anchor_dir = root / args.anchor / f"subject_{subject:02d}" / f"seed_{seed}"
            branch_dir = root / args.branch / f"subject_{subject:02d}" / f"seed_{seed}"
            anchor_metrics = _validate_run_metadata(anchor_dir, subject=subject, seed=seed)
            branch_metrics = _validate_run_metadata(branch_dir, subject=subject, seed=seed)
            anchor_path = anchor_dir / "predictions.npz"
            branch_path = branch_dir / "predictions.npz"
            anchor = _load(anchor_path)
            branch = _load(branch_path)
            _aligned(anchor, branch)
            label = anchor["label"]
            anchor_prediction = anchor["probabilities"].argmax(1)
            branch_prediction = branch["probabilities"].argmax(1)
            fused_probability = (
                FIXED_ANCHOR_WEIGHT * anchor["probabilities"]
                + (1.0 - FIXED_ANCHOR_WEIGHT) * branch["probabilities"]
            )
            fused_prediction = fused_probability.argmax(1)
            anchor_accuracy = _accuracy(label, anchor_prediction)
            branch_accuracy = _accuracy(label, branch_prediction)
            fused_accuracy = _accuracy(label, fused_prediction)
            delta_pp = 100.0 * (fused_accuracy - anchor_accuracy)
            delta_matrix[subject_index, seed_index] = delta_pp
            rows.append(
                {
                    "subject": subject,
                    "seed": seed,
                    "anchor": args.anchor,
                    "branch": args.branch,
                    "anchor_weight": FIXED_ANCHOR_WEIGHT,
                    "anchor_accuracy": anchor_accuracy,
                    "branch_accuracy": branch_accuracy,
                    "fixed_fusion_accuracy": fused_accuracy,
                    "delta_pp": delta_pp,
                    "prediction_disagreements": int(
                        np.sum(anchor_prediction != branch_prediction)
                    ),
                    "session_e_accessed": False,
                }
            )
            key = f"subject_{subject:02d}/seed_{seed}"
            provenance[key] = {
                "anchor_run_fingerprint": anchor_metrics["run_fingerprint"],
                "branch_run_fingerprint": branch_metrics["run_fingerprint"],
                "anchor_predictions_sha256": file_sha256(anchor_path),
                "branch_predictions_sha256": file_sha256(branch_path),
            }

    gate = fixed_fusion_gate(rows)
    ci_low, ci_high = two_way_cluster_bootstrap_mean_ci(delta_matrix)
    aggregate = {
        **gate,
        "anchor": args.anchor,
        "branch": args.branch,
        "anchor_weight": FIXED_ANCHOR_WEIGHT,
        "subjects": subjects,
        "seeds": seeds,
        "anchor_macro_accuracy": float(np.mean([row["anchor_accuracy"] for row in rows])),
        "branch_macro_accuracy": float(np.mean([row["branch_accuracy"] for row in rows])),
        "fixed_fusion_macro_accuracy": float(
            np.mean([row["fixed_fusion_accuracy"] for row in rows])
        ),
        "mean_delta_two_way_cluster_bootstrap_95_ci_pp": [ci_low, ci_high],
        "bootstrap_note": "exploratory uncertainty only; the preregistered gate does not use it",
        "session_e_accessed": False,
    }
    payload = {
        "protocol": "bci2a_session_t_nested_six_fold_oof_fixed_equal_fusion_gate",
        "selection": "none; anchor weight fixed to 0.5 before seeds 1 and 2 completed",
        "aggregate": aggregate,
        "pairs": rows,
        "provenance": provenance,
    }
    write_csv(output / "per_pair.csv", rows)
    write_json(output / "gate.json", payload)
    print(json.dumps({"status": "completed", "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
