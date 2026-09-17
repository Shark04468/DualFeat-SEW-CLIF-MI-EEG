#!/usr/bin/env python3
"""Independently audit frozen V8 ensemble E7 utility artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.analysis.v8_utility import (  # noqa: E402
    LogitCalibrator,
    multiclass_calibration_metrics,
    stratified_kshot_indices,
    utility_win_summary,
)
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    source_tree_digest,
)
from dpc_snn.experiments.v8_statistics import (  # noqa: E402
    pair_subject_seed_rows,
    paired_delta_summary,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from scripts.run_v8_e7_ensemble_utility import (  # noqa: E402
    ARMS,
    CAMPAIGN_FILES,
    RUN_FILES,
)


AUDIT_FILES = (
    "manifest.json",
    "audit_report.json",
    "recomputed_summary.csv",
    "recomputed_paired_subject_seed.csv",
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        values = list(csv.DictReader(handle))
    if not values:
        raise RuntimeError(f"empty E7 table: {path}")
    return values


def _close(first: Any, second: Any, tolerance: float = 2e-7) -> bool:
    return bool(np.isclose(float(first), float(second), rtol=0.0, atol=tolerance))


def _e6_logits(e6_run: Path, arm: str) -> tuple[np.ndarray, np.ndarray]:
    filename = "predictions.npz" if arm == "primary" else "matched_ann_predictions.npz"
    with np.load(e6_run / filename, allow_pickle=False) as archive:
        return np.asarray(archive["logits"]), np.asarray(archive["label"])


def _audit_run(run: Path, e6_run: Path) -> dict[str, Any]:
    validate_run_artifact_manifest(run, required_files=RUN_FILES, verify_hashes=True)
    metrics = read_json(run / "metrics.json")
    state = read_json(run / "state_audit.json")
    runtime = read_json(run / "runtime_status.json")
    replay = read_json(run / "clean_replay.json")
    if (
        metrics.get("status") != "completed"
        or metrics.get("model_updates") is not False
        or state.get("identical") is not True
        or state.get("model_updates") is not False
        or state.get("checkpoint_before_utility") != state.get("state_after_utility")
        or runtime.get("status") != "completed"
        or runtime.get("model_updates") is not False
        or runtime.get("openbmi_s2_accessed") is not False
        or replay.get("all_predictions_identical") is not True
        or max(float(value) for value in replay["maximum_probability_error"].values()) > 1e-5
    ):
        raise RuntimeError(f"invalid frozen E7 state contract: {run}")

    calibration = _rows(run / "calibration.csv")
    early = _rows(run / "early_decision.csv")
    robustness = _rows(run / "robustness.csv")
    kshot = _rows(run / "kshot.csv")
    if len(calibration) != 2 or len(early) != 8 or len(robustness) != 14 or len(kshot) != 100:
        raise RuntimeError(f"E7 row coverage differs from the registered utility grid: {run}")

    for row in calibration:
        logits, labels = _e6_logits(e6_run, row["arm"])
        recomputed = multiclass_calibration_metrics(logits, labels, n_bins=15)
        for field, value in recomputed.items():
            if not _close(row[field], value):
                raise RuntimeError(f"E7 calibration drift at {run}: {row['arm']} {field}")

    with np.load(run / "early_predictions.npz", allow_pickle=False) as archive:
        early_arms = archive["arms"].astype(str).tolist()
        endpoints = archive["endpoint_seconds"]
        early_logits = archive["logits"]
        labels = archive["labels"]
    if early_arms != list(ARMS) or early_logits.shape[:2] != (4, 2):
        raise RuntimeError(f"invalid E7 early prediction axes: {run}")
    for endpoint_index, endpoint in enumerate(endpoints):
        for arm_index, arm in enumerate(ARMS):
            row = next(
                item
                for item in early
                if item["arm"] == arm
                and np.isclose(float(item["endpoint_seconds"]), float(endpoint))
            )
            pred = early_logits[endpoint_index, arm_index].argmax(axis=1)
            recomputed = classification_metrics(labels, pred, n_classes=4)
            for field in ("accuracy", "balanced_accuracy", "kappa", "macro_f1"):
                if not _close(row[field], recomputed[field]):
                    raise RuntimeError(f"E7 early metric drift at {run}: {arm} {endpoint}")

    with np.load(run / "robustness_predictions.npz", allow_pickle=False) as archive:
        robust_arms = archive["arms"].astype(str).tolist()
        conditions = archive["conditions"].astype(str).tolist()
        robust_logits = archive["logits"]
        robust_labels = archive["labels"]
    if robust_arms != list(ARMS) or robust_logits.shape[:2] != (7, 2):
        raise RuntimeError(f"invalid E7 robustness prediction axes: {run}")
    for condition_index, condition in enumerate(conditions):
        for arm_index, arm in enumerate(ARMS):
            row = next(
                item
                for item in robustness
                if item["arm"] == arm and item["condition"] == condition
            )
            pred = robust_logits[condition_index, arm_index].argmax(axis=1)
            recomputed = classification_metrics(robust_labels, pred, n_classes=4)
            for field in ("accuracy", "balanced_accuracy", "kappa", "macro_f1"):
                if not _close(row[field], recomputed[field]):
                    raise RuntimeError(f"E7 robustness drift at {run}: {arm} {condition}")

    for row in kshot:
        logits, labels = _e6_logits(e6_run, row["arm"])
        calibration_indices, evaluation_indices = stratified_kshot_indices(
            labels,
            k_per_class=int(row["k_per_class"]),
            seed=int(row["split_seed"]),
        )
        calibrator = LogitCalibrator(
            log_temperature=float(row["log_temperature"]),
            bias=np.asarray(json.loads(row["centered_bias"]), dtype=np.float64),
        )
        calibrated = multiclass_calibration_metrics(
            calibrator.apply(logits[evaluation_indices]), labels[evaluation_indices], n_bins=15
        )
        raw = multiclass_calibration_metrics(
            logits[evaluation_indices], labels[evaluation_indices], n_bins=15
        )
        checks = {
            "raw_accuracy": raw["accuracy"],
            "calibrated_accuracy": calibrated["accuracy"],
            "calibrated_nll": calibrated["negative_log_likelihood"],
            "calibrated_ece": calibrated["ece"],
            "calibration_trials": calibration_indices.size,
            "evaluation_trials": evaluation_indices.size,
        }
        for field, value in checks.items():
            if not _close(row[field], value):
                raise RuntimeError(f"E7 K-shot drift at {run}: {row['arm']} {field}")

    operation = read_json(run / "operation_proxy.json")
    reduction = 1.0 - float(
        operation["primary"]["activity_weighted_decoder_events"]
    ) / float(operation["matched_ann"]["activity_weighted_decoder_events"])
    if (
        not _close(operation["reduction"], reduction)
        or operation["primary"].get("hardware_energy_claim_allowed") is not False
        or operation["matched_ann"].get("hardware_energy_claim_allowed") is not False
    ):
        raise RuntimeError(f"invalid E7 operation proxy: {run}")

    primary_endpoint = 1.0
    for arm in ARMS:
        arm_early = next(
            row
            for row in early
            if row["arm"] == arm
            and np.isclose(float(row["endpoint_seconds"]), primary_endpoint)
        )
        arm_robust = [
            row
            for row in robustness
            if row["arm"] == arm and row["condition"] != "clean"
        ]
        arm_kshot = [row for row in kshot if row["arm"] == arm]
        expected = {
            "final_accuracy": read_json(e6_run / "metrics.json")["arms"][arm]["accuracy"],
            "early_accuracy": arm_early["accuracy"],
            "robustness_mean_accuracy": np.mean(
                [float(row["accuracy"]) for row in arm_robust]
            ),
            "kshot_mean_accuracy": np.mean(
                [float(row["calibrated_accuracy"]) for row in arm_kshot]
            ),
        }
        for field, value in expected.items():
            if not _close(metrics["arms"][arm][field], value):
                raise RuntimeError(f"E7 summary drift at {run}: {arm} {field}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e7", required=True)
    parser.add_argument("--e6", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    e7 = Path(args.e7).resolve()
    e6 = Path(args.e6).resolve()
    output = ensure_dir(Path(args.output).resolve())
    validate_run_artifact_manifest(e7, required_files=CAMPAIGN_FILES, verify_hashes=True)
    status = read_json(e7 / "campaign_status.json")
    if (
        status.get("status") != "completed"
        or status.get("full_registered_contract") is not True
        or status.get("runs") != 45
        or status.get("openbmi_s2_accessed") is not False
    ):
        raise RuntimeError("E7 audit requires the full 9-subject x 5-seed campaign")
    active_digest = source_tree_digest(collect_source_tree_manifest(ROOT))
    if source_tree_digest(read_json(e7 / "source_tree_manifest.json")) != active_digest:
        raise RuntimeError("active source differs from the E7 campaign source")
    rows = []
    for subject in range(1, 10):
        for seed in range(5):
            rows.append(
                _audit_run(
                    e7 / f"subject_{subject:02d}" / f"seed_{seed}",
                    e6 / f"subject_{subject:02d}" / f"seed_{seed}",
                )
            )
    recomputed_rows = []
    for row in rows:
        recomputed_rows.append(
            {
                "subject": row["subject"],
                "seed": row["seed"],
                **{
                    f"{arm}_{field}": value
                    for arm in ARMS
                    for field, value in row["arms"][arm].items()
                },
                "operation_proxy_reduction": row["operation_proxy_reduction"],
            }
        )
    write_csv(output / "recomputed_summary.csv", recomputed_rows)
    pair_rows = []
    summaries = {}
    for index, field in enumerate(
        ("final_accuracy", "early_accuracy", "robustness_mean_accuracy", "kshot_mean_accuracy")
    ):
        ann = [
            {"subject": row["subject"], "seed": row["seed"], "value": row["arms"]["matched_ann"][field]}
            for row in rows
        ]
        snn = [
            {"subject": row["subject"], "seed": row["seed"], "value": row["arms"]["primary"][field]}
            for row in rows
        ]
        pairs = pair_subject_seed_rows(ann, snn, value="value")
        summaries[field] = paired_delta_summary(pairs, seed=20260803 + index)
        pair_rows.extend({"metric": field, **pair} for pair in pairs)
    write_csv(output / "recomputed_paired_subject_seed.csv", pair_rows)
    operation_reduction = float(np.mean([row["operation_proxy_reduction"] for row in rows]))
    utility = utility_win_summary(
        final_gain_pp=100.0 * summaries["final_accuracy"]["subject_macro_mean_delta"],
        early_gain_pp=100.0 * summaries["early_accuracy"]["subject_macro_mean_delta"],
        robustness_gain_pp=100.0 * summaries["robustness_mean_accuracy"]["subject_macro_mean_delta"],
        kshot_gain_pp=100.0 * summaries["kshot_mean_accuracy"]["subject_macro_mean_delta"],
        operation_reduction=operation_reduction,
    )
    saved_gate = read_json(e7 / "gate_decision.json")
    if saved_gate.get("utility") != utility or saved_gate.get("paired_summaries") != summaries:
        raise RuntimeError("E7 campaign gate differs from independent recomputation")
    report = {
        "status": "passed",
        "stage": "E7_ENSEMBLE_AUDIT",
        "runs_audited": 45,
        "all_artifact_hashes_verified": True,
        "all_metrics_recomputed": True,
        "all_calibrators_replayed": True,
        "all_model_states_unchanged": True,
        "gate_recomputed": True,
        "gate_passed": bool(utility["passed"]),
        "utility": utility,
        "paired_summaries": summaries,
        "operation_proxy_reduction": operation_reduction,
        "hardware_energy_claim_allowed": False,
        "openbmi_s2_accessed": False,
    }
    write_json(output / "audit_report.json", report)
    write_run_artifact_manifest(output, required_files=AUDIT_FILES)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
