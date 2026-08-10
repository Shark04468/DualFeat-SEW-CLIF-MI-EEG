#!/usr/bin/env python3
"""Independently verify the fold-local delay-prior prerequisite for E3/E5."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    sha256_fingerprint,
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_evidence_stability import (  # noqa: E402
    evaluate_v8_evidence_seed_stability,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    source_tree_digest,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402


CONFIG_SCHEMA = "dpc-snn-v8-e3-delay-prior-feasibility-gate/v1"
ACCEPTED_FILES = (
    "manifest.json",
    "fingerprint.json",
    "prior.npz",
    "evidence.npz",
    "summary.json",
)
REJECTED_FILES = (
    "manifest.json",
    "fingerprint.json",
    "rejected_evidence.npz",
    "rejected_summary.json",
)
OUTPUT_FILES = (
    "manifest.json",
    "feasibility_decision.json",
    "per_subject_fold.csv",
    "input_manifest.json",
    "resolved_config.yaml",
    "source_tree_manifest.json",
)


def _float_equal(first: Any, second: Any, *, atol: float = 1e-12) -> bool:
    left = float(first)
    right = float(second)
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right)
    return math.isclose(left, right, rel_tol=0.0, abs_tol=atol)


def _assert_stability_equal(
    stored: Mapping[str, Any], recomputed: Mapping[str, Any]
) -> None:
    exact_fields = (
        "schema",
        "passed",
        "replicates",
        "consensus_edge_count",
        "consensus_edge_indices",
        "criteria",
        "thresholds",
        "heldout_session_e_accessed",
        "classifier_training",
    )
    for field in exact_fields:
        if stored.get(field) != recomputed.get(field):
            raise RuntimeError(f"stored prior stability field differs: {field}")
    scalar_fields = (
        "replicate_pass_fraction",
        "consensus_frequency",
        "median_pairwise_edge_jaccard",
        "minimum_pairwise_edge_jaccard",
        "median_pairwise_delay_correlation",
    )
    for field in scalar_fields:
        if not _float_equal(stored.get(field), recomputed.get(field)):
            raise RuntimeError(f"stored prior stability scalar differs: {field}")
    stored_controls = dict(stored.get("control_pass_fraction", {}))
    recomputed_controls = dict(recomputed.get("control_pass_fraction", {}))
    if set(stored_controls) != set(recomputed_controls) or any(
        not _float_equal(stored_controls[name], recomputed_controls[name])
        for name in stored_controls
    ):
        raise RuntimeError("stored prior control pass fractions differ")


def _replicate_arrays(
    archive: Mapping[str, np.ndarray], replicates: int
) -> list[dict[str, np.ndarray]]:
    result: list[dict[str, np.ndarray]] = []
    for index in range(replicates):
        prefix = f"replicate_{index}__"
        values = {
            name[len(prefix) :]: np.asarray(value)
            for name, value in archive.items()
            if name.startswith(prefix)
        }
        if "signed_node_accepted" not in values or "signed_node_delay_mean" not in values:
            raise RuntimeError(f"prior evidence replicate {index} is incomplete")
        result.append(values)
    unexpected = {
        name.split("__", 1)[0]
        for name in archive
        if name.startswith("replicate_")
    } - {f"replicate_{index}" for index in range(replicates)}
    if unexpected:
        raise RuntimeError("prior evidence contains unregistered replicate indices")
    return result


def _load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def evaluate_prior_feasibility(
    *, prior_root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    expected = config["expected"]
    thresholds = dict(config["stability_thresholds"])
    subjects = [int(value) for value in expected["subjects"]]
    folds = [int(value) for value in expected["folds"]]
    expected_count = len(subjects) * len(folds)
    rows: list[dict[str, Any]] = []
    input_entries: list[dict[str, Any]] = []

    for subject in subjects:
        for fold in folds:
            directory = prior_root / f"subject_{subject:02d}" / f"fold_{fold}" / "inner"
            manifest = read_json(directory / "manifest.json")
            status = str(manifest.get("status"))
            if status == "completed":
                required = ACCEPTED_FILES
                summary_name = "summary.json"
                evidence_name = "evidence.npz"
                prior_status = "accepted"
            elif status == "rejected":
                required = REJECTED_FILES
                summary_name = "rejected_summary.json"
                evidence_name = "rejected_evidence.npz"
                prior_status = "rejected"
            else:
                raise RuntimeError(f"prior artifact has invalid status at {directory}")
            validate_run_artifact_manifest(
                directory,
                required_files=required,
                verify_hashes=True,
                verify_prediction_schema=False,
            )
            summary = read_json(directory / summary_name)
            fingerprint = read_json(directory / "fingerprint.json")
            combined = fingerprint.pop("combined_sha256", None)
            if combined != sha256_fingerprint(fingerprint):
                raise RuntimeError(f"prior fingerprint is invalid at {directory}")
            if summary.get("input_fingerprint") != combined or summary.get(
                "input_payload"
            ) != fingerprint:
                raise RuntimeError(f"prior summary/fingerprint mismatch at {directory}")
            payload = dict(summary["input_payload"])
            trial_ids = [str(value) for value in payload.get("trial_ids", [])]
            if (
                summary.get("schema") != "dpc-snn-v8-fold-delay-prior-ensemble/v1"
                or summary.get("analytic_representation")
                != expected["analytic_representation"]
                or int(summary.get("ensemble_replicates", -1))
                != int(expected["ensemble_replicates"])
                or payload.get("scope") != expected["scope"]
                or payload.get("source_tree_sha256")
                != expected["producer_source_tree_sha256"]
                or payload.get("heldout_data_accessed") is not False
                or not trial_ids
                or any(":T:" not in trial_id or ":E:" in trial_id for trial_id in trial_ids)
                or summary.get("heldout_session_e_accessed") is not False
                or summary.get("classifier_training") is not False
            ):
                raise RuntimeError(f"prior identity or held-out lock changed at {directory}")
            stored_thresholds = dict(summary["stability_gate"]["thresholds"])
            if stored_thresholds != thresholds:
                raise RuntimeError(f"prior stability thresholds changed at {directory}")
            archive = _load_archive(directory / evidence_name)
            replicates = int(expected["ensemble_replicates"])
            evidence = _replicate_arrays(archive, replicates)
            replicate_summaries = list(summary.get("replicate_summaries", []))
            if len(replicate_summaries) != replicates:
                raise RuntimeError(f"prior replicate summaries are incomplete at {directory}")
            recomputed, pairwise = evaluate_v8_evidence_seed_stability(
                replicate_summaries,
                evidence,
                minimum_replicates=int(thresholds["minimum_replicates"]),
                minimum_pass_fraction=float(thresholds["minimum_pass_fraction"]),
                minimum_consensus_frequency=float(
                    thresholds["minimum_consensus_frequency"]
                ),
                minimum_consensus_edges=int(thresholds["minimum_consensus_edges"]),
                minimum_median_edge_jaccard=float(
                    thresholds["minimum_median_edge_jaccard"]
                ),
                minimum_edge_jaccard_floor=float(
                    thresholds["minimum_edge_jaccard_floor"]
                ),
                minimum_median_delay_correlation=float(
                    thresholds["minimum_median_delay_correlation"]
                ),
            )
            _assert_stability_equal(summary["stability_gate"], recomputed)
            if len(pairwise) != math.comb(replicates, 2):
                raise RuntimeError("prior pairwise stability coverage is incomplete")
            passed = bool(recomputed["passed"])
            if (prior_status == "accepted") != passed:
                raise RuntimeError("prior artifact status disagrees with recomputed gate")
            failed_criteria = sorted(
                name for name, value in recomputed["criteria"].items() if not bool(value)
            )
            row = {
                "subject": subject,
                "fold": fold,
                "prior_status": prior_status,
                "passed": passed,
                "replicate_pass_fraction": recomputed["replicate_pass_fraction"],
                "consensus_edge_count": recomputed["consensus_edge_count"],
                "median_pairwise_edge_jaccard": recomputed[
                    "median_pairwise_edge_jaccard"
                ],
                "minimum_pairwise_edge_jaccard": recomputed[
                    "minimum_pairwise_edge_jaccard"
                ],
                "median_pairwise_delay_correlation": recomputed[
                    "median_pairwise_delay_correlation"
                ],
                "failed_criteria": ";".join(failed_criteria),
                "summary_sha256": file_sha256(directory / summary_name),
                "manifest_sha256": file_sha256(directory / "manifest.json"),
            }
            for name, value in sorted(recomputed["control_pass_fraction"].items()):
                row[f"control_{name}_pass_fraction"] = value
            rows.append(row)
            input_entries.append(
                {
                    "subject": subject,
                    "fold": fold,
                    "directory": str(directory),
                    "manifest_sha256": row["manifest_sha256"],
                    "summary_sha256": row["summary_sha256"],
                    "evidence_sha256": file_sha256(directory / evidence_name),
                    "input_fingerprint": combined,
                }
            )

    if len(rows) != expected_count or len(
        {(int(row["subject"]), int(row["fold"])) for row in rows}
    ) != expected_count:
        raise RuntimeError("prior feasibility coverage is incomplete")
    all_passed = all(bool(row["passed"]) for row in rows)
    if not bool(expected["require_all_subject_folds_pass"]):
        raise RuntimeError("this gate must fail closed on every registered subject-fold")
    decision = {
        "status": "completed",
        "stage": "E3_DELAY_PRIOR_FEASIBILITY_GATE",
        "protocol": config["protocol"],
        "passed": all_passed,
        "decision": config["decision_policy"]["pass" if all_passed else "fail"],
        "registered_subject_folds": expected_count,
        "passing_subject_folds": sum(bool(row["passed"]) for row in rows),
        "failed_subject_folds": [
            {"subject": int(row["subject"]), "fold": int(row["fold"])}
            for row in rows
            if not bool(row["passed"])
        ],
        "producer_source_tree_sha256": expected["producer_source_tree_sha256"],
        "metrics_recomputed_from_replicate_arrays": True,
        "threshold_relaxation_after_observation": False,
        "classifier_training_authorized": all_passed,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    return rows, decision, {"prior_artifacts": input_entries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config",
        default="configs/experiments/v8_e3_prior_feasibility_gate.yaml",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA:
        raise RuntimeError("E3 prior-feasibility config schema changed")
    if config.get("data_access") != {
        "allowed_session": "T",
        "heldout_session_e_accessed": False,
        "openbmi_session_s2_accessed": False,
    }:
        raise RuntimeError("E3 prior-feasibility held-out lock changed")
    prior_root = Path(args.prior_root).resolve()
    rows, decision, inputs = evaluate_prior_feasibility(
        prior_root=prior_root,
        config=config,
    )
    output = ensure_dir(Path(args.output).resolve())
    source_tree = collect_source_tree_manifest(ROOT)
    write_csv(output / "per_subject_fold.csv", rows)
    write_json(output / "feasibility_decision.json", decision)
    write_json(output / "source_tree_manifest.json", source_tree)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    write_json(
        output / "input_manifest.json",
        {
            **inputs,
            "config_sha256": file_sha256(config_path),
            "evaluator_source_tree_sha256": source_tree_digest(source_tree),
        },
    )
    write_run_artifact_manifest(output, required_files=OUTPUT_FILES)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
