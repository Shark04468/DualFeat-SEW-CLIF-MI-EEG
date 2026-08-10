#!/usr/bin/env python3
"""Aggregate the V14 frozen-shared residual-expert pilot."""

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

from dpc_snn.models.v14_shared_residual_student import V14_VARIANTS  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402


CONTROL = "r0_shared_replay"
PRIMARY = "r3_dual_residual"
CAPACITY_CONTROL = "r4_generic_residual"
TEACHER = "equal_teacher"


def _csv_int(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError("integer list must be non-empty and unique")
    return parsed


def _csv_variants(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if parsed != V14_VARIANTS:
        raise ValueError("formal V14 aggregation requires prespecified R0-R4 order")
    return parsed


def _prediction(path: Path) -> dict[str, np.ndarray]:
    required = {
        "indices",
        "labels",
        "logits",
        "shared_logits",
        "atc_residual_logits",
        "fbc_residual_logits",
        "generic_residual_logits",
        "equal_teacher_logits",
        "atc_teacher_logits",
        "fbc_teacher_logits",
        "atc_gate",
        "fbc_gate",
        "generic_gate",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise RuntimeError(f"invalid V14 prediction archive: {path}")
        values = {name: np.asarray(archive[name]) for name in required}
    samples = values["indices"].size
    if values["labels"].shape != (samples,) or any(
        values[name].shape != (samples, 4)
        for name in (
            "logits",
            "shared_logits",
            "atc_residual_logits",
            "fbc_residual_logits",
            "generic_residual_logits",
            "equal_teacher_logits",
            "atc_teacher_logits",
            "fbc_teacher_logits",
        )
    ):
        raise RuntimeError(f"malformed V14 prediction archive: {path}")
    if any(values[name].shape != (4,) for name in ("atc_gate", "fbc_gate", "generic_gate")):
        raise RuntimeError(f"malformed V14 gate archive: {path}")
    if any(not np.isfinite(value).all() for value in values.values()):
        raise RuntimeError(f"non-finite V14 prediction archive: {path}")
    return values


def _metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "kappa": float(cohen_kappa_score(labels, prediction)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--subjects", default="1,3,8")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--folds", default="0,1,2,3,4,5")
    parser.add_argument("--variants", default=",".join(V14_VARIANTS))
    parser.add_argument("--expected-source-digest", required=True)
    parser.add_argument("--minimum-gain-pp", type=float, default=0.5)
    parser.add_argument("--minimum-positive-pairs", type=int, default=2)
    parser.add_argument("--maximum-regression-pp", type=float, default=1.0)
    parser.add_argument("--minimum-gap-recovery", type=float, default=0.4)
    parser.add_argument("--minimum-gain-over-capacity-control-pp", type=float, default=0.3)
    parser.add_argument("--maximum-class-regression-pp", type=float, default=2.0)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = ensure_dir(Path(args.output).resolve() if args.output else root / "aggregate")
    subjects = _csv_int(args.subjects)
    seeds = _csv_int(args.seeds)
    folds = _csv_int(args.folds)
    variants = _csv_variants(args.variants)
    fold_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    fingerprints: list[str] = []
    source_digests: list[str] = []
    replay_passes = 0

    for subject in subjects:
        for seed in seeds:
            seen: list[int] = []
            for fold in folds:
                fold_dir = root / f"subject_{subject:02d}" / f"seed_{seed}" / f"fold_{fold}"
                status = read_json(fold_dir / "campaign_status.json")
                replay = read_json(fold_dir / "shared_replay.json")
                capacity = read_json(fold_dir / "capacity_audit.json")
                if (
                    status.get("status") != "completed"
                    or status.get("variants") != list(variants)
                    or status.get("session_e_accessed") is not False
                    or status.get("openbmi_s2_accessed") is not False
                    or int(status.get("total_hpo_configurations", -1)) > 10
                    or replay.get("status") != "passed"
                    or capacity.get("status") != "passed"
                    or capacity.get("raw_feature_classifier_bypass") is not False
                    or int(capacity.get("r3_r4_total_parameter_difference", -1)) > 4
                ):
                    raise RuntimeError(f"incomplete or invalid V14 fold: {fold_dir}")
                replay_passes += 1
                source_digests.append(
                    str(read_json(fold_dir / "source_tree_summary.json")["sha256"])
                )
                fingerprints.append(
                    str(read_json(fold_dir / "run_fingerprint.json")["combined_sha256"])
                )
                summary = pd.read_csv(fold_dir / "summary.csv")
                if tuple(summary["variant"]) != variants:
                    raise RuntimeError(f"V14 summary variants are invalid: {fold_dir}")
                archives = {
                    variant: _prediction(fold_dir / variant / "outer_predictions.npz")
                    for variant in variants
                }
                reference = archives[CONTROL]
                indices = reference["indices"].astype(np.int64)
                labels = reference["labels"].astype(np.int64)
                teacher = reference["equal_teacher_logits"]
                if not np.array_equal(reference["logits"], reference["shared_logits"]):
                    raise RuntimeError("V14 R0 does not equal its shared logits")
                for variant, archive in archives.items():
                    if (
                        not np.array_equal(indices, archive["indices"])
                        or not np.array_equal(labels, archive["labels"])
                        or not np.array_equal(teacher, archive["equal_teacher_logits"])
                        or not np.array_equal(
                            reference["shared_logits"], archive["shared_logits"]
                        )
                    ):
                        raise RuntimeError(f"V14 archives are not aligned: {fold_dir}")
                    prediction = archive["logits"].argmax(axis=1)
                    shared_prediction = archive["shared_logits"].argmax(axis=1)
                    final_correct = prediction == labels
                    shared_correct = shared_prediction == labels
                    fold_rows.append(
                        {
                            "subject": subject,
                            "seed": seed,
                            "fold": fold,
                            "variant": variant,
                            **_metrics(labels, prediction),
                            "rescue_rate": float(np.mean(final_correct & ~shared_correct)),
                            "damage_rate": float(np.mean(~final_correct & shared_correct)),
                        }
                    )
                    for gate_name in ("atc", "fbc", "generic"):
                        for class_index, value in enumerate(archive[f"{gate_name}_gate"]):
                            gate_rows.append(
                                {
                                    "subject": subject,
                                    "seed": seed,
                                    "fold": fold,
                                    "variant": variant,
                                    "gate": gate_name,
                                    "class": class_index,
                                    "value": float(value),
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
                                f"prediction_{variant}": int(
                                    archive["logits"][offset].argmax()
                                )
                                for variant, archive in archives.items()
                            },
                            f"prediction_{TEACHER}": int(teacher[offset].argmax()),
                        }
                    )
            counts = Counter(seen)
            if len(counts) != 288 or set(counts.values()) != {1}:
                raise RuntimeError("V14 outer folds do not cover Session T exactly once")

    expected_runs = len(subjects) * len(seeds) * len(folds)
    if set(source_digests) != {args.expected_source_digest}:
        raise RuntimeError("V14 source snapshot mismatch")
    if len(set(fingerprints)) != expected_runs or replay_passes != expected_runs:
        raise RuntimeError("V14 fingerprints or replays are incomplete")

    trial_frame = pd.DataFrame(trial_rows).sort_values(
        ["subject", "seed", "trial_index"]
    )
    subject_seed_rows: list[dict[str, Any]] = []
    for (subject, seed), group in trial_frame.groupby(["subject", "seed"], sort=True):
        labels = group["label"].to_numpy()
        shared_prediction = group[f"prediction_{CONTROL}"].to_numpy()
        shared_correct = shared_prediction == labels
        for model in (*variants, TEACHER):
            prediction = group[f"prediction_{model}"].to_numpy()
            correct = prediction == labels
            subject_seed_rows.append(
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
    subject_seed = pd.DataFrame(subject_seed_rows)
    model_summary = (
        subject_seed.groupby("model", as_index=False)
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
    control_values = subject_seed.loc[subject_seed["model"] == CONTROL].sort_values(
        ["subject", "seed"]
    )["accuracy"].to_numpy()
    teacher_values = subject_seed.loc[subject_seed["model"] == TEACHER].sort_values(
        ["subject", "seed"]
    )["accuracy"].to_numpy()
    teacher_gap = float(teacher_values.mean() - control_values.mean())
    comparison_rows: list[dict[str, Any]] = []
    for variant in variants[1:]:
        values = subject_seed.loc[subject_seed["model"] == variant].sort_values(
            ["subject", "seed"]
        )["accuracy"].to_numpy()
        delta = 100.0 * (values - control_values)
        comparison_rows.append(
            {
                "variant": variant,
                "reference": CONTROL,
                "mean_delta_pp": float(delta.mean()),
                "median_delta_pp": float(np.median(delta)),
                "minimum_delta_pp": float(delta.min()),
                "positive_pairs": int(np.sum(delta > 0.0)),
                "negative_pairs": int(np.sum(delta < 0.0)),
                "teacher_gap_recovery": (
                    float((values.mean() - control_values.mean()) / teacher_gap)
                    if teacher_gap > 0.0
                    else float("nan")
                ),
            }
        )
    comparisons = pd.DataFrame(comparison_rows).sort_values(
        ["mean_delta_pp", "median_delta_pp"], ascending=False
    )

    labels = trial_frame["label"].to_numpy()
    per_class_rows: list[dict[str, Any]] = []
    for model in (*variants, TEACHER):
        prediction = trial_frame[f"prediction_{model}"].to_numpy()
        for target in range(4):
            mask = labels == target
            per_class_rows.append(
                {
                    "model": model,
                    "class": target,
                    "trials": int(mask.sum()),
                    "accuracy": float(np.mean(prediction[mask] == labels[mask])),
                }
            )
    per_class = pd.DataFrame(per_class_rows)
    control_class = per_class.loc[per_class["model"] == CONTROL].sort_values("class")[
        "accuracy"
    ].to_numpy()
    primary_class = per_class.loc[per_class["model"] == PRIMARY].sort_values("class")[
        "accuracy"
    ].to_numpy()
    minimum_class_delta_pp = float((100.0 * (primary_class - control_class)).min())

    primary_row = comparisons.loc[comparisons["variant"] == PRIMARY].iloc[0]
    capacity_row = comparisons.loc[
        comparisons["variant"] == CAPACITY_CONTROL
    ].iloc[0]
    primary_accuracy = subject_seed.loc[subject_seed["model"] == PRIMARY][
        "accuracy"
    ].mean()
    capacity_accuracy = subject_seed.loc[
        subject_seed["model"] == CAPACITY_CONTROL
    ]["accuracy"].mean()
    gain_over_capacity_pp = 100.0 * float(primary_accuracy - capacity_accuracy)
    primary_net_rescue = float(
        subject_seed.loc[subject_seed["model"] == PRIMARY]["net_rescue_pp"].mean()
    )
    active_gates = pd.DataFrame(gate_rows)
    active_gates = active_gates.loc[
        (active_gates["variant"] == PRIMARY)
        & (active_gates["gate"].isin(("atc", "fbc")))
    ]
    primary_gate_mean = float(active_gates["value"].mean())
    promotion_passed = bool(
        float(primary_row["mean_delta_pp"]) >= float(args.minimum_gain_pp)
        and int(primary_row["positive_pairs"]) >= int(args.minimum_positive_pairs)
        and float(primary_row["minimum_delta_pp"]) >= -float(args.maximum_regression_pp)
        and float(primary_row["teacher_gap_recovery"]) >= float(args.minimum_gap_recovery)
        and gain_over_capacity_pp >= float(args.minimum_gain_over_capacity_control_pp)
        and primary_net_rescue > 0.0
        and primary_gate_mean > 0.05
        and minimum_class_delta_pp >= -float(args.maximum_class_regression_pp)
    )
    promotion = {
        "status": "passed" if promotion_passed else "failed",
        "primary_variant": PRIMARY,
        "best_variant": str(comparisons.iloc[0]["variant"]),
        "mean_delta_pp": float(primary_row["mean_delta_pp"]),
        "median_delta_pp": float(primary_row["median_delta_pp"]),
        "minimum_delta_pp": float(primary_row["minimum_delta_pp"]),
        "positive_pairs": int(primary_row["positive_pairs"]),
        "teacher_gap_pp": 100.0 * teacher_gap,
        "teacher_gap_recovery": float(primary_row["teacher_gap_recovery"]),
        "gain_over_capacity_control_pp": gain_over_capacity_pp,
        "primary_net_rescue_pp": primary_net_rescue,
        "primary_gate_mean": primary_gate_mean,
        "minimum_class_delta_pp": minimum_class_delta_pp,
        "capacity_control_delta_pp": float(capacity_row["mean_delta_pp"]),
        "criteria": {
            "minimum_mean_gain_pp": float(args.minimum_gain_pp),
            "minimum_positive_pairs": int(args.minimum_positive_pairs),
            "maximum_regression_pp": float(args.maximum_regression_pp),
            "minimum_teacher_gap_recovery": float(args.minimum_gap_recovery),
            "minimum_gain_over_capacity_control_pp": float(
                args.minimum_gain_over_capacity_control_pp
            ),
            "maximum_class_regression_pp": float(args.maximum_class_regression_pp),
            "positive_net_rescue_required": True,
            "gate_mean_must_exceed_initial_0.05": True,
        },
    }

    write_csv(output / "fold_metrics.csv", fold_rows)
    write_csv(output / "gate_values.csv", gate_rows)
    write_csv(output / "trial_predictions.csv", trial_frame.to_dict(orient="records"))
    write_csv(
        output / "subject_seed_metrics.csv", subject_seed.to_dict(orient="records")
    )
    write_csv(output / "model_summary.csv", model_summary.to_dict(orient="records"))
    write_csv(output / "paired_comparisons.csv", comparisons.to_dict(orient="records"))
    write_csv(output / "per_class_accuracy.csv", per_class_rows)
    audit = {
        "status": "passed",
        "expected_runs": expected_runs,
        "completed_runs": expected_runs,
        "subjects": list(subjects),
        "seeds": list(seeds),
        "folds": list(folds),
        "variants": list(variants),
        "source_tree_sha256": args.expected_source_digest,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "unique_run_fingerprints": len(set(fingerprints)),
        "exact_shared_replay_passed_folds": replay_passes,
        "r3_r4_parameter_difference": 4,
        "raw_feature_classifier_bypass": False,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    write_json(output / "campaign_audit.json", audit)
    summary = {"status": "completed", "audit": audit, "promotion": promotion}
    write_json(output / "aggregate_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
