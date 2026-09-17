"""Source-maintenance lineage for a frozen V8 architecture."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dpc_snn.experiments.v62_protocol import sha256_fingerprint


V8_ENSEMBLE_MAINTENANCE_SCHEMA = "dpc-snn-v8-ensemble-source-maintenance/v1"


def build_ensemble_maintenance_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.pop("combined_sha256", None)
    body.setdefault("schema", V8_ENSEMBLE_MAINTENANCE_SCHEMA)
    return validate_ensemble_maintenance_manifest(
        {**body, "combined_sha256": sha256_fingerprint(body)}
    )


def validate_ensemble_maintenance_manifest(
    payload: Mapping[str, Any],
    *,
    expected_parent_freeze_sha256: str | None = None,
    expected_current_source_tree_sha256: str | None = None,
) -> dict[str, Any]:
    value = dict(payload)
    required = {
        "schema",
        "created_at_utc",
        "parent_freeze_sha256",
        "parent_source_tree_sha256",
        "current_source_tree_sha256",
        "source_delta",
        "scientific_invariants",
        "heldout_access",
        "artifact_hashes",
        "combined_sha256",
    }
    if set(value) != required:
        raise RuntimeError("V8 maintenance manifest fields are incomplete")
    if value["schema"] != V8_ENSEMBLE_MAINTENANCE_SCHEMA:
        raise RuntimeError("unsupported V8 maintenance schema")
    if expected_parent_freeze_sha256 is not None and value[
        "parent_freeze_sha256"
    ] != str(expected_parent_freeze_sha256):
        raise RuntimeError("maintenance manifest references another architecture freeze")
    if expected_current_source_tree_sha256 is not None and value[
        "current_source_tree_sha256"
    ] != str(expected_current_source_tree_sha256):
        raise RuntimeError("active source tree differs from the maintenance manifest")
    delta = value["source_delta"]
    if (
        not isinstance(delta, Mapping)
        or delta.get("removed") != []
        or delta.get("changed")
        != [
            "scripts/audit_v8_e6_ensemble.py",
            "scripts/run_v8_e6_ensemble_frozen.py",
            "tests/test_v8_ensemble.py",
        ]
        or delta.get("added")
        != [
            "scripts/attest_v8_ensemble_source_maintenance.py",
            "src/dpc_snn/experiments/v8_maintenance.py",
        ]
    ):
        raise RuntimeError("source delta exceeds the registered maintenance scope")
    invariants = value["scientific_invariants"]
    if (
        not isinstance(invariants, Mapping)
        or invariants.get("architecture_files_unchanged") is not True
        or invariants.get("model_math_unchanged") is not True
        or invariants.get("training_budget_unchanged") is not True
        or invariants.get("preprocessing_unchanged") is not True
        or invariants.get("analysis_gate_unchanged") is not True
        or invariants.get("maintenance_scope")
        != "data_identity_fingerprint_and_access_audit_only"
    ):
        raise RuntimeError("maintenance changed a frozen scientific invariant")
    heldout = value["heldout_access"]
    if (
        not isinstance(heldout, Mapping)
        or heldout.get("architecture_frozen_before_session_e_materialization") is not True
        or heldout.get("session_e_materialized_after_parent_freeze") is not True
        or heldout.get("session_e_use_before_attestation")
        != "deterministic_preprocessing_and_exact_equivalence_audit_only"
        or heldout.get("session_e_predictions_or_metrics_computed") is not False
        or heldout.get("session_e_used_for_model_or_checkpoint_selection") is not False
        or heldout.get("openbmi_s2_accessed") is not False
    ):
        raise RuntimeError("maintenance held-out access lineage is invalid")
    hashes = value["artifact_hashes"]
    if not isinstance(hashes, Mapping) or not hashes:
        raise RuntimeError("maintenance manifest has no artifact hashes")
    for name, digest in hashes.items():
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f"invalid maintenance artifact digest for {name!r}")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise RuntimeError(
                f"maintenance artifact digest for {name!r} is not hexadecimal"
            ) from exc
    expected = sha256_fingerprint(
        {key: item for key, item in value.items() if key != "combined_sha256"}
    )
    if value["combined_sha256"] != expected:
        raise RuntimeError("maintenance manifest digest does not match its contents")
    return value
