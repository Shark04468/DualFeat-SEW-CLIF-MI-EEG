#!/usr/bin/env python3
"""Aggregate the prespecified E15 fold-local residual calibration experiment."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v15_residual_calibration import E15_METHODS  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import accuracy, cohen_kappa  # noqa: E402


CONTROL = "r0_shared_replay"
PRIMARY = "c2_atc_rms"
CAPACITY_CONTROL = "c3_generic_rms"
TEACHER = "equal_teacher"


def _csv_int(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError("integer list must be non-empty and unique")
    return parsed


def _prediction(path: Path) -> dict[str, np.ndarray]:
    required = {
        "indices",
        "labels",
        "logits",
        "shared_logits",
        "residual_logits",
        "selected_alpha",
        "equal_teacher_logits",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise RuntimeError(f"invalid E15 prediction archive: {path}")
        values = {name: np.asarray(archive[name]) for name in required}
    samples = values["indices"].size
    if values["labels"].shape != (samples,) or any(
        values[name].shape != (samples, 4)
        for name in ("logits", "shared_logits", "residual_logits", "equal_teacher_logits")
    ):
        raise RuntimeError(f"malformed E15 prediction archive: {path}")
    if values["selected_alpha"].shape != (1,) or any(
        not value.size or not np.isfinite(value).all() for value in values.values()
    ):
        raise RuntimeError(f"non-finite E15 prediction archive: {path}")
    return values


def _metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy(labels, prediction),
        "kappa": cohen_kappa(labels, prediction, n_classes=4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--subjects", default="1,3,8")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--folds", default="0,1,2,3,4,5")
    parser.add_argument("--expected-source-digest", required=True)
    parser.add_argument("--minimum-gain-pp", type=float, default=0.5)
    parser.add_argument("--minimum-positive-pairs", type=int, default=2)
    parser.add_argument("--maximum-regression-pp", type=float, default=1.0)
    parser.add_argument("--minimum-capacity-adjusted-gain-pp", type=float, default=0.3)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = ensure_dir(Path(args.output).resolve() if args.output else root / "aggregate")
    subjects = _csv_int(args.subjects)
    seeds = _csv_int(args.seeds)
    folds = _csv_int(args.folds)
    trial_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    source_digests: list[str] = []
    fingerprints: list[str] = []
    replay_passes = 0

    for subject in subjects:
        for seed in seeds:
            seen: list[int] = []
            for fold in folds:
                fold_dir = root / f"subject_{subject:02d}" / f"seed_{seed}" / f"fold_{fold}"
                status = read_json(fold_dir / "campaign_status.json")
                replay = read_json(fold_dir / "v14_replay_audit.json")
                if (
                    status.get("status") != "completed"
                    or tuple(status.get("methods", ())) != E15_METHODS
                    or status.get("v14_replay") != "passed"
                    or status.get("outer_labels_used_for_selection") is not False
                    or status.get("session_e_accessed") is not False
                    or status.get("openbmi_s2_accessed") is not False
                    or replay.get("status") != "passed"
                ):
                    raise RuntimeError(f"incomplete or invalid E15 fold: {fold_dir}")
                replay_passes += 1
                source_digests.append(
                    str(read_json(fold_dir / "source_tree_summary.json")["sha256"])
                )
                fingerprints.append(
                    str(read_json(fold_dir / "run_fingerprint.json")["combined_sha256"])
                )
                summary = pd.read_csv(fold_dir / "summary.csv")
                if tuple(summary["method"]) != E15_METHODS:
                    raise RuntimeError(f"E15 summary method order is invalid: {fold_dir}")
                archives = {
                    method: _prediction(fold_dir / method / "outer_predictions.npz")
                    for method in E15_METHODS
                }
                reference = archives[CONTROL]
                indices = reference["indices"].astype(np.int64)
                labels = reference["labels"].astype(np.int64)
                teacher = reference["equal_teacher_logits"]
                if not np.array_equal(reference["logits"], reference["shared_logits"]):
                    raise RuntimeError("E15 R0 is not an exact shared replay")
                for method, archive in archives.items():
                    if (
                        not np.array_equal(indices, archive["indices"])
                        or not np.array_equal(labels, archive["labels"])
                        or not np.array_equal(teacher, archive["equal_teacher_logits"])
                        or not np.array_equal(
                            reference["shared_logits"], archive["shared_logits"]
                        )
                    ):
                        raise RuntimeError(f"E15 archives are not aligned: {fold_dir}")
                    prediction = archive["logits"].argmax(axis=1)
                    shared_prediction = archive["shared_logits"].argmax(axis=1)
                    correct = prediction == labels
                    shared_correct = shared_prediction == labels
                    fold_rows.append(
                        {
                            "subject": subject,
                            "seed": seed,
                            "fold": fold,
                            "method": method,
                            **_metrics(labels, prediction),
                            "rescue_rate": float(np.mean(correct & ~shared_correct)),
                            "damage_rate": float(np.mean(~correct & shared_correct)),
                        }
                    )
                    alpha_rows.append(
                        {
                            "subject": subject,
                            "seed": seed,
                            "fold": fold,
                            "method": method,
                            "selected_alpha": float(archive["selected_alpha"][0]),
                        }
                    )
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
                                f"prediction_{method}": int(
                                    archive["logits"][offset].argmax()
                                )
                                for method, archive in archives.items()
                            },
                            f"prediction_{TEACHER}": int(teacher[offset].argmax()),
                        }
                    )
            counts = Counter(seen)
            if len(counts) != 288 or set(counts.values()) != {1}:
                raise RuntimeError("E15 folds do not cover Session T exactly once")

    expected_runs = len(subjects) * len(seeds) * len(folds)
    if set(source_digests) != {args.expected_source_digest}:
        raise RuntimeError("E15 source snapshot mismatch")
    if len(set(fingerprints)) != expected_runs or replay_passes != expected_runs:
        raise RuntimeError("E15 fingerprints or V14 replays are incomplete")

    trials = pd.DataFrame(trial_rows).sort_values(["subject", "seed", "trial_index"])
    subject_rows: list[dict[str, Any]] = []
    for (subject, seed), group in trials.groupby(["subject", "seed"], sort=True):
        labels = group["label"].to_numpy()
        shared_prediction = group[f"prediction_{CONTROL}"].to_numpy()
        shared_correct = shared_prediction == labels
        for model in (*E15_METHODS, TEACHER):
            prediction = group[f"prediction_{model}"].to_numpy()
            correct = prediction == labels
            subject_rows.append(
                {
                    "subject": int(subject),
                    "seed": int(seed),
                    "model": model,
                    **_metrics(labels, prediction),
                    "rescue_rate": float(np.mean(correct & ~shared_correct)),
                    "damage_rate": float(np.mean(~correct & shared_correct)),
                    "net_rescue_pp": 100.0
                    * float(np.mean(correct & ~shared_correct) - np.mean(~correct & shared_correct)),
                }
            )
    subject_frame = pd.DataFrame(subject_rows)
    model_summary = (
        subject_frame.groupby("model", as_index=False)
        .agg(
            subject_seed_pairs=("accuracy", "size"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            kappa_mean=("kappa", "mean"),
            kappa_std=("kappa", "std"),
            rescue_rate_mean=("rescue_rate", "mean"),
            damage_rate_mean=("damage_rate", "mean"),
            net_rescue_pp_mean=("net_rescue_pp", "mean"),
        )
        .sort_values("accuracy_mean", ascending=False)
    )
    control_values = subject_frame.loc[subject_frame["model"] == CONTROL].sort_values(
        ["subject", "seed"]
    )["accuracy"].to_numpy()
    comparison_rows: list[dict[str, Any]] = []
    for method in E15_METHODS[1:]:
        values = subject_frame.loc[subject_frame["model"] == method].sort_values(
            ["subject", "seed"]
        )["accuracy"].to_numpy()
        delta = 100.0 * (values - control_values)
        comparison_rows.append(
            {
                "method": method,
                "reference": CONTROL,
                "mean_delta_pp": float(delta.mean()),
                "median_delta_pp": float(np.median(delta)),
                "minimum_delta_pp": float(delta.min()),
                "positive_pairs": int(np.sum(delta > 0.0)),
                "negative_pairs": int(np.sum(delta < 0.0)),
            }
        )
    comparisons = pd.DataFrame(comparison_rows).sort_values(
        ["mean_delta_pp", "median_delta_pp"], ascending=False
    )
    primary_row = comparisons.loc[comparisons["method"] == PRIMARY].iloc[0]
    capacity_row = comparisons.loc[comparisons["method"] == CAPACITY_CONTROL].iloc[0]
    primary_net_rescue = float(
        subject_frame.loc[subject_frame["model"] == PRIMARY]["net_rescue_pp"].mean()
    )
    capacity_adjusted_gain = float(
        primary_row["mean_delta_pp"] - capacity_row["mean_delta_pp"]
    )
    nonzero_alpha_folds = int(
        np.sum(
            (pd.DataFrame(alpha_rows)["method"] == PRIMARY)
            & (pd.DataFrame(alpha_rows)["selected_alpha"] > 0.0)
        )
    )
    mechanism_passed = bool(
        float(primary_row["mean_delta_pp"]) >= float(args.minimum_gain_pp)
        and int(primary_row["positive_pairs"]) >= int(args.minimum_positive_pairs)
        and float(primary_row["minimum_delta_pp"]) >= -float(args.maximum_regression_pp)
        and primary_net_rescue > 0.0
    )
    architecture_passed = bool(
        mechanism_passed
        and capacity_adjusted_gain >= float(args.minimum_capacity_adjusted_gain_pp)
    )
    promotion = {
        "status": "passed" if architecture_passed else "failed",
        "mechanism_gate_status": "passed" if mechanism_passed else "failed",
        "primary_method": PRIMARY,
        "best_method": str(comparisons.iloc[0]["method"]),
        "mean_delta_pp": float(primary_row["mean_delta_pp"]),
        "median_delta_pp": float(primary_row["median_delta_pp"]),
        "minimum_delta_pp": float(primary_row["minimum_delta_pp"]),
        "positive_pairs": int(primary_row["positive_pairs"]),
        "negative_pairs": int(primary_row["negative_pairs"]),
        "primary_net_rescue_pp": primary_net_rescue,
        "capacity_control_delta_pp": float(capacity_row["mean_delta_pp"]),
        "capacity_adjusted_gain_pp": capacity_adjusted_gain,
        "nonzero_alpha_folds": nonzero_alpha_folds,
        "criteria": {
            "minimum_mean_gain_pp": float(args.minimum_gain_pp),
            "minimum_positive_pairs": int(args.minimum_positive_pairs),
            "maximum_regression_pp": float(args.maximum_regression_pp),
            "positive_net_rescue_required": True,
            "minimum_capacity_adjusted_gain_pp": float(
                args.minimum_capacity_adjusted_gain_pp
            ),
        },
    }

    write_csv(output / "fold_metrics.csv", fold_rows)
    write_csv(output / "selected_alphas.csv", alpha_rows)
    write_csv(output / "trial_predictions.csv", trials.to_dict(orient="records"))
    write_csv(output / "subject_seed_metrics.csv", subject_rows)
    write_csv(output / "model_summary.csv", model_summary.to_dict(orient="records"))
    write_csv(output / "paired_comparisons.csv", comparisons.to_dict(orient="records"))
    audit = {
        "status": "passed",
        "expected_runs": expected_runs,
        "completed_runs": expected_runs,
        "subjects": list(subjects),
        "seeds": list(seeds),
        "folds": list(folds),
        "methods": list(E15_METHODS),
        "source_tree_sha256": args.expected_source_digest,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "unique_run_fingerprints": len(set(fingerprints)),
        "exact_v14_replay_passed_folds": replay_passes,
        "outer_labels_used_for_selection": False,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    write_json(output / "campaign_audit.json", audit)
    summary = {"status": "completed", "audit": audit, "promotion": promotion}
    write_json(output / "aggregate_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
