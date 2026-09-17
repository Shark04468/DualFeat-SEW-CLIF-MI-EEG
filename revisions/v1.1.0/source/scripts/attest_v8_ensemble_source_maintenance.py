#!/usr/bin/env python3
"""Attest a non-scientific E6 runner/audit maintenance update."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import file_sha256  # noqa: E402
from dpc_snn.experiments.v8_maintenance import (  # noqa: E402
    build_ensemble_maintenance_manifest,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    source_tree_digest,
    validate_v8_freeze_manifest,
)
from dpc_snn.utils.io import read_json, write_json  # noqa: E402


EXPECTED_CHANGED = [
    "scripts/audit_v8_e6_ensemble.py",
    "scripts/run_v8_e6_ensemble_frozen.py",
    "tests/test_v8_ensemble.py",
]
EXPECTED_ADDED = [
    "scripts/attest_v8_ensemble_source_maintenance.py",
    "src/dpc_snn/experiments/v8_maintenance.py",
]


def _validate_split_audit(path: Path) -> dict[str, object]:
    audit = read_json(path)
    digest = str(audit.pop("combined_sha256"))
    canonical = json.dumps(audit, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != digest:
        raise RuntimeError("data split audit digest is invalid")
    if (
        audit.get("status") != "passed"
        or audit.get("freeze_precedes_e_materialization") is not True
        or len(audit.get("subjects", [])) != 9
        or not all(row.get("passed") is True for row in audit["subjects"])
    ):
        raise RuntimeError("data split audit did not pass for all nine subjects")
    return {**audit, "combined_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-freeze", required=True)
    parser.add_argument("--parent-source-manifest", required=True)
    parser.add_argument("--data-split-audit", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--prospective-e6-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    parent_freeze_path = Path(args.parent_freeze).resolve()
    parent_source_path = Path(args.parent_source_manifest).resolve()
    split_audit_path = Path(args.data_split_audit).resolve()
    evaluation_root = Path(args.evaluation_root).resolve()
    prospective_e6_output = Path(args.prospective_e6_output).resolve()
    output = Path(args.output).resolve()
    if prospective_e6_output.exists():
        raise RuntimeError("E6 output exists before source-maintenance attestation")

    parent_freeze = validate_v8_freeze_manifest(parent_freeze_path)
    parent_source = read_json(parent_source_path)
    current_source = collect_source_tree_manifest(ROOT)
    parent_digest = source_tree_digest(parent_source)
    current_digest = source_tree_digest(current_source)
    if parent_digest != parent_freeze["source_tree_sha256"]:
        raise RuntimeError("parent source manifest differs from the architecture freeze")

    parent_names = set(parent_source)
    current_names = set(current_source)
    changed = sorted(
        name
        for name in parent_names & current_names
        if parent_source[name] != current_source[name]
    )
    added = sorted(current_names - parent_names)
    removed = sorted(parent_names - current_names)
    if changed != EXPECTED_CHANGED or added != EXPECTED_ADDED or removed:
        raise RuntimeError(
            f"unexpected source delta: changed={changed}, added={added}, removed={removed}"
        )

    _validate_split_audit(split_audit_path)
    evaluation_files = sorted(evaluation_root.glob("A??.npz"))
    if len(evaluation_files) != 9:
        raise RuntimeError("evaluation root must contain exactly nine subject files")
    freeze_time = datetime.fromisoformat(parent_freeze["created_at_utc"]).timestamp()
    if any(path.stat().st_mtime <= freeze_time for path in evaluation_files):
        raise RuntimeError("Session E was materialized before the parent architecture freeze")

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_freeze_sha256": parent_freeze["combined_sha256"],
        "parent_source_tree_sha256": parent_digest,
        "current_source_tree_sha256": current_digest,
        "source_delta": {
            "changed": changed,
            "added": added,
            "removed": removed,
        },
        "scientific_invariants": {
            "architecture_files_unchanged": True,
            "model_math_unchanged": True,
            "training_budget_unchanged": True,
            "preprocessing_unchanged": True,
            "analysis_gate_unchanged": True,
            "maintenance_scope": "data_identity_fingerprint_and_access_audit_only",
        },
        "heldout_access": {
            "architecture_frozen_before_session_e_materialization": True,
            "session_e_materialized_after_parent_freeze": True,
            "session_e_use_before_attestation": "deterministic_preprocessing_and_exact_equivalence_audit_only",
            "session_e_predictions_or_metrics_computed": False,
            "session_e_used_for_model_or_checkpoint_selection": False,
            "openbmi_s2_accessed": False,
        },
        "artifact_hashes": {
            "parent_freeze_manifest": file_sha256(parent_freeze_path),
            "parent_source_manifest": file_sha256(parent_source_path),
            "data_split_audit": file_sha256(split_audit_path),
            **{
                f"session_e/{path.name}": file_sha256(path)
                for path in evaluation_files
            },
        },
    }
    manifest = build_ensemble_maintenance_manifest(payload)
    write_json(output, manifest)
    print(
        json.dumps(
            {
                "status": "passed",
                "parent_freeze_sha256": manifest["parent_freeze_sha256"],
                "current_source_tree_sha256": current_digest,
                "maintenance_sha256": manifest["combined_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
