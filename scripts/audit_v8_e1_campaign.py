#!/usr/bin/env python3
"""Audit every artifact and nested trial identity in a completed V8 E1 campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    validate_prediction_schema,
    validate_run_artifact_manifest,
)
from dpc_snn.experiments.v8_campaign_audit import (  # noqa: E402
    V8CampaignAuditError,
    expected_e1_run_keys,
    index_e1_summary_rows,
    validate_e1_campaign_contract,
    validate_nested_trial_sets,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402


RUN_FILES = (
    "manifest.json",
    "resolved_config.yaml",
    "source_fingerprint.json",
    "source_tree_manifest.json",
    "data_access_manifest.json",
    "split_manifest.json",
    "augmentation_manifest.json",
    "metrics.json",
    "predictions.npz",
    "predictions.csv",
    "runtime_status.json",
    "stdout.log",
    "stderr.log",
)
FOLD_FILES = (
    "best.pt",
    "last.pt",
    "history.csv",
    "result.json",
    "outer_test_predictions.npz",
    "selection_best.pt",
    "selection_last.pt",
    "selection_history.csv",
    "selection_result.json",
    "selection_predictions.npz",
)


def _required_files(n_folds: int) -> tuple[str, ...]:
    return RUN_FILES + tuple(
        f"fold_{fold}/{name}" for fold in range(n_folds) for name in FOLD_FILES
    )


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise V8CampaignAuditError(f"CSV is empty: {path}")
    return rows


def _archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def _same_float(first: Any, second: Any, *, atol: float = 1e-12) -> bool:
    return math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=atol)


def _audit_run(
    campaign: Path,
    key: tuple[str, int, int],
    summary_row: dict[str, str],
    *,
    expected_source_digest: str,
) -> dict[str, Any]:
    model, subject, seed = key
    run_dir = campaign / model / f"subject_{subject:02d}" / f"seed_{seed}"
    split = read_json(run_dir / "split_manifest.json")
    folds = list(split["folds"])
    n_folds = int(split["n_splits"])
    if len(folds) != n_folds or [int(fold["fold"]) for fold in folds] != list(range(n_folds)):
        raise V8CampaignAuditError(f"invalid fold enumeration in {run_dir}")
    validate_run_artifact_manifest(
        run_dir,
        required_files=_required_files(n_folds),
        verify_hashes=True,
        verify_prediction_schema=True,
    )
    validate_prediction_schema(run_dir / "predictions.npz", run_dir / "predictions.csv")
    source_manifest = read_json(run_dir / "source_tree_manifest.json")
    campaign_manifest = read_json(campaign / "source_tree_manifest.json")
    if source_manifest != campaign_manifest:
        raise V8CampaignAuditError(f"run source manifest differs from campaign: {run_dir}")
    fingerprint = read_json(run_dir / "source_fingerprint.json")
    metrics = read_json(run_dir / "metrics.json")
    runtime = read_json(run_dir / "runtime_status.json")
    if str(fingerprint["combined_sha256"]) != str(metrics["run_fingerprint"]):
        raise V8CampaignAuditError(f"metric/fingerprint mismatch in {run_dir}")
    if metrics.get("status") != "completed" or runtime.get("status") != "completed":
        raise V8CampaignAuditError(f"run is not completed: {run_dir}")
    if metrics.get("session_e_accessed") or metrics.get("openbmi_s2_accessed"):
        raise V8CampaignAuditError(f"held-out data flag is set in {run_dir}")
    if (str(metrics["model"]), int(metrics["subject"]), int(metrics["seed"])) != key:
        raise V8CampaignAuditError(f"metric identity mismatch in {run_dir}")

    prediction = _archive(run_dir / "predictions.npz")
    count = int(prediction["label"].size)
    if count < 1 or len(set(map(str, prediction["trial_id"]))) != count:
        raise V8CampaignAuditError(f"prediction trial identities are empty or duplicated: {run_dir}")
    if set(map(str, prediction["session"])) != {"T"}:
        raise V8CampaignAuditError(f"non-Session-T prediction in {run_dir}")
    if set(map(int, prediction["subject"])) != {subject} or set(map(int, prediction["seed"])) != {
        seed
    }:
        raise V8CampaignAuditError(f"prediction subject/seed mismatch in {run_dir}")
    if set(map(str, prediction["model"])) != {f"{model}_v8_e1_oof"}:
        raise V8CampaignAuditError(f"prediction model mismatch in {run_dir}")
    recomputed = classification_metrics(
        prediction["label"], prediction["pred"], n_classes=4
    )
    for field in ("accuracy", "balanced_accuracy", "kappa", "macro_f1"):
        if not _same_float(recomputed[field], metrics[field]):
            raise V8CampaignAuditError(f"recomputed {field} differs in {run_dir}")
        if not _same_float(metrics[field], summary_row[field]):
            raise V8CampaignAuditError(f"summary {field} differs in {run_dir}")

    full_trial_ids = np.asarray(prediction["trial_id"]).astype(str)
    full_labels = np.asarray(prediction["label"], dtype=np.int64)
    observed_outer_ids: list[str] = []
    for fold_index, fold in enumerate(folds):
        validate_nested_trial_sets(fold)
        fold_dir = run_dir / f"fold_{fold_index}"
        outer = _archive(fold_dir / "outer_test_predictions.npz")
        selection = _archive(fold_dir / "selection_predictions.npz")
        outer_indices = np.asarray(outer["indices"], dtype=np.int64)
        selection_indices = np.asarray(selection["indices"], dtype=np.int64)
        if np.intersect1d(outer_indices, selection_indices).size:
            raise V8CampaignAuditError(f"inner validation overlaps outer test in {fold_dir}")
        expected_outer_ids = set(map(str, fold["outer_test_trial_ids"]))
        expected_selection_ids = set(map(str, fold["inner_validation_trial_ids"]))
        if set(full_trial_ids[outer_indices]) != expected_outer_ids:
            raise V8CampaignAuditError(f"outer prediction identities differ in {fold_dir}")
        if set(full_trial_ids[selection_indices]) != expected_selection_ids:
            raise V8CampaignAuditError(f"inner prediction identities differ in {fold_dir}")
        if not np.array_equal(outer["labels"], full_labels[outer_indices]):
            raise V8CampaignAuditError(f"outer prediction labels differ in {fold_dir}")
        if not np.array_equal(selection["labels"], full_labels[selection_indices]):
            raise V8CampaignAuditError(f"selection prediction labels differ in {fold_dir}")
        result = read_json(fold_dir / "result.json")
        selection_result = read_json(fold_dir / "selection_result.json")
        if int(result["selected_outer_retrain_epoch"]) != int(
            selection_result["selected_outer_retrain_epoch"]
        ):
            raise V8CampaignAuditError(f"selected epoch differs between fold records: {fold_dir}")
        if str(result["inner_validation_run"]) != str(fold["inner_validation_run"]):
            raise V8CampaignAuditError(f"inner validation run differs in {fold_dir}")
        observed_outer_ids.extend(map(str, fold["outer_test_trial_ids"]))
    if len(observed_outer_ids) != count or set(observed_outer_ids) != set(full_trial_ids):
        raise V8CampaignAuditError(f"outer folds do not exactly cover all trials in {run_dir}")

    return {
        "model": model,
        "subject": subject,
        "seed": seed,
        "trials": count,
        "folds": n_folds,
        "accuracy": float(metrics["accuracy"]),
        "kappa": float(metrics["kappa"]),
        "macro_f1": float(metrics["macro_f1"]),
        "run_fingerprint": str(metrics["run_fingerprint"]),
        "source_tree_sha256": expected_source_digest,
        "artifact_hashes_valid": True,
        "trial_identity_valid": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--config", default="configs/experiments/v8_e1_baselines.yaml")
    args = parser.parse_args()
    campaign = Path(args.campaign).resolve()
    output = ensure_dir(Path(args.output).resolve() if args.output else campaign / "audit")
    status = read_json(campaign / "campaign_status.json")
    selection = read_json(campaign / "confirmation_selection.json")
    config = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    if status.get("status") != "completed" or status.get("stage") != "E1":
        raise V8CampaignAuditError("E1 campaign must be completed before audit")
    if status.get("protocol") != "bci2a_session_t_nested_six_fold_oof":
        raise V8CampaignAuditError("E1 campaign protocol is not the locked nested protocol")
    if status.get("session_e_accessed") or selection.get("session_e_accessed"):
        raise V8CampaignAuditError("E1 campaign reports held-out Session-E access")
    validate_e1_campaign_contract(status, selection, config)
    source_summary = read_json(campaign / "source_tree_summary.json")
    source_digest = str(source_summary["sha256"])
    if source_digest != str(status["source_tree_sha256"]):
        raise V8CampaignAuditError("campaign source digest differs from completion status")

    expected = expected_e1_run_keys(status, selection)
    summary_rows = _csv_rows(campaign / "summary.csv")
    indexed = index_e1_summary_rows(summary_rows, expected)
    if int(status["runs"]) != len(expected):
        raise V8CampaignAuditError("campaign run count differs from required coverage")
    audited = [
        _audit_run(campaign, key, indexed[key], expected_source_digest=source_digest)
        for key in sorted(expected)
    ]
    report = {
        "status": "passed",
        "stage": "E1_AUDIT",
        "protocol": status["protocol"],
        "runs": len(audited),
        "screened_models": status["screened_models"],
        "confirmed_models": status["confirmed_models"],
        "subjects": status["subjects"],
        "confirmation_seeds": status["confirmation_seeds"],
        "source_tree_sha256": source_digest,
        "artifact_hashes_valid": True,
        "prediction_schema_valid": True,
        "nested_trial_identity_valid": True,
        "heldout_session_e_accessed": False,
    }
    write_csv(output / "audited_runs.csv", audited)
    write_json(output / "audit_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
