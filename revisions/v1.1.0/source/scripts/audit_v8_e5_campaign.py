#!/usr/bin/env python3
"""Independently audit the bounded V8 E5 inner-validation search."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_baselines import session_t_development_view  # noqa: E402
from dpc_snn.experiments.v8_hpo import (  # noqa: E402
    generate_balanced_v8_candidates,
    rank_v8_hpo_candidates,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from scripts.run_v8_e2_zero_delay import _nested_folds, _subject_file  # noqa: E402
from scripts.run_v8_e5_bounded_hpo import FOLD_REQUIRED_FILES  # noqa: E402


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"E5 audit found an empty CSV: {path}")
    return rows


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default="configs/experiments/v8_e5_bounded_hpo.yaml"
    )
    args = parser.parse_args()

    campaign = Path(args.campaign).resolve()
    output = ensure_dir(Path(args.output).resolve())
    data_root = Path(args.data).resolve()
    config = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    candidates = generate_balanced_v8_candidates(config)
    candidate_index = {row["candidate_id"]: row for row in candidates}
    subjects = list(config["subjects"])
    stages = list(config["successive_halving"])
    status = read_json(campaign / "campaign_status.json")
    if (
        status.get("status") != "completed"
        or status.get("stage") != "E5"
        or status.get("protocol") != config["protocol"]
        or int(status.get("registered_candidates", -1)) != len(candidates)
        or int(status.get("active_candidates", -1)) != len(candidates)
        or status.get("subjects") != subjects
        or status.get("completed_stages") != stages
        or status.get("selection_role") != "inner_validation_only"
        or status.get("outer_test_predictions_created")
        or status.get("session_e_accessed")
        or not status.get("full_registered_contract")
    ):
        raise RuntimeError("E5 campaign status violates the registered full contract")
    top_required = [
        "manifest.json",
        "campaign_status.json",
        "summary.csv",
        "candidate_ledger.json",
        "capacity_audit.csv",
        "source_tree_manifest.json",
        "source_tree_summary.json",
        "heldout_lock_manifest.json",
        "shared_cache_provenance.json",
        "resolved_campaign.yaml",
        "promotion_ledger.json",
        *[f"{name}_ranking.csv" for name in stages],
        "selected_candidate.json",
        "selected_model.yaml",
        "selected_e2_config.yaml",
    ]
    validate_run_artifact_manifest(
        campaign,
        required_files=top_required,
        verify_hashes=True,
        verify_prediction_schema=False,
    )
    ledger = read_json(campaign / "candidate_ledger.json")
    if (
        int(ledger.get("registered_count", -1)) != len(candidates)
        or int(ledger.get("active_count", -1)) != len(candidates)
        or ledger.get("candidates") != candidates
        or not ledger.get("balanced_complete_factorial")
    ):
        raise RuntimeError("E5 candidate ledger differs from the registered factorial")
    promotions = read_json(campaign / "promotion_ledger.json")
    if len(promotions) != len(stages):
        raise RuntimeError("E5 promotion ledger has the wrong number of stages")

    subject_data: dict[int, dict[str, Any]] = {}
    for subject in subjects:
        data = load_processed_npz(_subject_file(data_root, int(subject)))
        _, y, metadata, access = session_t_development_view(data)
        if access.get("session_e_accessed"):
            raise RuntimeError("E5 audit data view accessed Session E")
        nested = _nested_folds(
            metadata,
            n_splits=int(config["selection"]["n_splits"]),
            split_seed=int(config["selection"]["split_seed"]),
        )
        subject_data[int(subject)] = {"y": y, "metadata": metadata, "nested": nested}

    summary_rows = _csv_rows(campaign / "summary.csv")
    summary_coverage = {
        (row["search_stage"], row["candidate_id"], int(row["subject"]), int(row["fold"]))
        for row in summary_rows
    }
    expected_coverage: set[tuple[str, str, int, int]] = set()
    active_ids = [row["candidate_id"] for row in candidates]
    audited_rows: list[dict[str, Any]] = []
    ranking_audit: dict[str, Any] = {}
    for stage_index, stage_name in enumerate(stages):
        stage_config = dict(config["successive_halving"][stage_name])
        folds = [int(value) for value in stage_config["active_folds"]]
        stage_rows = []
        for candidate_id in active_ids:
            if candidate_id not in candidate_index:
                raise RuntimeError(f"unknown promoted E5 candidate {candidate_id}")
            for subject in subjects:
                bundle = subject_data[int(subject)]
                for fold in folds:
                    expected_coverage.add((stage_name, candidate_id, int(subject), fold))
                    _, outer_test, _, inner_validation, _ = bundle["nested"][fold]
                    directory = (
                        campaign
                        / stage_name
                        / candidate_id
                        / f"subject_{int(subject):02d}"
                        / f"fold_{fold}"
                    )
                    validate_run_artifact_manifest(
                        directory,
                        required_files=FOLD_REQUIRED_FILES,
                        verify_hashes=True,
                        verify_prediction_schema=False,
                    )
                    result = read_json(directory / "result.json")
                    if (
                        result.get("status") != "completed"
                        or result.get("stage") != "E5"
                        or result.get("search_stage") != stage_name
                        or result.get("candidate_id") != candidate_id
                        or int(result.get("subject")) != int(subject)
                        or int(result.get("fold")) != fold
                        or result.get("evaluation_role") != "inner_validation"
                        or int(result.get("outer_test_trials_evaluated", -1)) != 0
                        or int(result.get("session_e_trials_evaluated", -1)) != 0
                    ):
                        raise RuntimeError(f"invalid E5 fold identity under {directory}")
                    with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
                        indices = archive["indices"].astype(np.int64)
                        labels = archive["labels"].astype(np.int64)
                        logits = archive["logits"].astype(np.float32)
                    if not np.array_equal(indices, inner_validation):
                        raise RuntimeError(f"E5 predictions are not inner validation: {directory}")
                    if np.intersect1d(indices, outer_test).size:
                        raise RuntimeError(f"E5 predictions overlap outer test: {directory}")
                    if not np.array_equal(labels, bundle["y"][inner_validation]):
                        raise RuntimeError(f"E5 prediction labels changed: {directory}")
                    metrics = classification_metrics(labels, logits.argmax(axis=1), n_classes=4)
                    if not _close(result["validation_accuracy"], metrics["accuracy"]):
                        raise RuntimeError(f"E5 accuracy does not reproduce: {directory}")
                    if not _close(result["validation_kappa"], metrics["kappa"]):
                        raise RuntimeError(f"E5 kappa does not reproduce: {directory}")
                    stage_rows.append(result)
                    audited_rows.append(result)
        recomputed = rank_v8_hpo_candidates(
            stage_rows,
            candidate_ids=active_ids,
            subjects=subjects,
            folds=folds,
        )
        recorded = _csv_rows(campaign / f"{stage_name}_ranking.csv")
        if [row["candidate_id"] for row in recorded] != [
            row["candidate_id"] for row in recomputed
        ]:
            raise RuntimeError(f"E5 {stage_name} ranking order does not reproduce")
        for left, right in zip(recorded, recomputed, strict=True):
            for key in (
                "subject_macro_mean_validation_kappa",
                "subject_macro_mean_validation_accuracy",
            ):
                if not _close(float(left[key]), float(right[key])):
                    raise RuntimeError(f"E5 {stage_name} ranking metric drift for {key}")
        promoted = [
            row["candidate_id"]
            for row in recomputed[: int(stage_config["promote_top_k"])]
        ]
        promotion = promotions[stage_index]
        if (
            promotion.get("stage") != stage_name
            or promotion.get("input_candidates") != active_ids
            or promotion.get("promoted_candidates") != promoted
            or promotion.get("outer_test_accessed")
        ):
            raise RuntimeError(f"E5 {stage_name} promotion ledger does not reproduce")
        ranking_audit[stage_name] = {
            "input_candidates": len(active_ids),
            "promoted_candidates": promoted,
            "rows": len(stage_rows),
        }
        active_ids = promoted

    if summary_coverage != expected_coverage or len(summary_rows) != len(expected_coverage):
        raise RuntimeError("E5 summary coverage is incomplete or duplicated")
    if len(active_ids) != 1 or status.get("selected_candidate_id") != active_ids[0]:
        raise RuntimeError("E5 selected candidate does not equal the final promotion")
    selected = read_json(campaign / "selected_candidate.json")
    if selected.get("candidate_id") != active_ids[0] or selected.get(
        "outer_test_accessed"
    ):
        raise RuntimeError("E5 selected candidate provenance is invalid")
    selected_e2 = yaml.safe_load(
        (campaign / "selected_e2_config.yaml").read_text(encoding="utf-8")
    )
    if (
        selected_e2.get("required_confirmation_variants")
        != ["hpo_selected_full_ann"]
        or selected_e2.get("hpo_provenance", {}).get(
            "outer_test_accessed_for_selection"
        )
        or selected_e2.get("data_access", {}).get("heldout_session_e_accessed")
    ):
        raise RuntimeError("E5 selected E2 config violates held-out selection rules")

    report = {
        "status": "passed",
        "stage": "E5_AUDIT",
        "registered_candidates": len(candidates),
        "audited_fold_results": len(audited_rows),
        "expected_fold_results": 234,
        "selected_candidate_id": active_ids[0],
        "ranking_audit": ranking_audit,
        "outer_test_predictions_created": False,
        "session_e_accessed": False,
        "all_child_hashes_verified": True,
        "all_metrics_recomputed": True,
    }
    if len(audited_rows) != int(report["expected_fold_results"]):
        raise RuntimeError(
            f"E5 audited {len(audited_rows)} folds, expected {report['expected_fold_results']}"
        )
    write_json(output / "audit_report.json", report)
    write_run_artifact_manifest(
        output, required_files=("manifest.json", "audit_report.json")
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
