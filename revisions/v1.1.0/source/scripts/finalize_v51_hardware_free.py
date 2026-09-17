#!/usr/bin/env python
"""Finalize a V5.1 campaign after its preregistered stop decision."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.utils.io import ensure_dir, read_json, write_json  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decision(directory: Path) -> dict[str, Any]:
    path = directory / "stage_decision.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing stage decision: {path}")
    payload = read_json(path)
    if bool(payload.get("classifier_training", False)):
        raise RuntimeError(f"Classifier training occurred before gates passed: {path}")
    if bool(payload.get("heldout_session_S2_accessed", False)):
        raise RuntimeError(f"Held-out Session S2 was accessed before freeze: {path}")
    return payload


def _artifact_index(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            rows.append(
                {
                    "root": str(root),
                    "relative_path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-dir", required=True)
    parser.add_argument("--semisynthetic-dir", required=True)
    parser.add_argument("--real-evidence-dir", required=True)
    parser.add_argument("--snapshot-root", action="append", default=[])
    parser.add_argument("--superseded-dir", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    synthetic_dir = Path(args.synthetic_dir)
    semisynthetic_dir = Path(args.semisynthetic_dir)
    real_evidence_dir = Path(args.real_evidence_dir)
    output = ensure_dir(args.output)
    synthetic = _decision(synthetic_dir)
    semisynthetic = _decision(semisynthetic_dir)
    real_evidence = _decision(real_evidence_dir)

    if synthetic.get("status") != "passed":
        raise RuntimeError("Isolated synthetic stage did not pass its preregistered point gate")
    mechanism_passed = bool(semisynthetic.get("pre_registered_point_gate_passed", False))
    evidence_passed = bool(real_evidence.get("passed", False))
    downstream_allowed = mechanism_passed and evidence_passed
    if downstream_allowed:
        raise RuntimeError(
            "Both development gates passed; this stop finalizer cannot replace classification"
        )

    snapshot_roots = [Path(value) for value in args.snapshot_root]
    snapshot_paths = sorted(
        path for root in snapshot_roots for path in root.glob("*/source_snapshot.tar.gz")
    )
    if snapshot_roots and not snapshot_paths:
        raise FileNotFoundError("No source snapshots found under requested snapshot roots")
    indexed_roots = [synthetic_dir, semisynthetic_dir, real_evidence_dir, *snapshot_roots]
    artifacts = _artifact_index(indexed_roots)
    write_json(output / "artifact_sha256_manifest.json", {"artifacts": artifacts})

    superseded = []
    for value in args.superseded_dir:
        directory = Path(value)
        marker = {
            "status": "superseded",
            "reason": "metric_or_split_scope_was_inconsistent_with_the_declared_node_delay_contract",
            "replacement_semisynthetic": str(semisynthetic_dir),
            "replacement_real_evidence": str(real_evidence_dir),
        }
        write_json(directory / "SUPERSEDED.json", marker)
        superseded.append(str(directory))

    final = {
        "campaign_status": "completed_with_preregistered_stop",
        "classifier_training_performed": False,
        "heldout_session_S2_accessed": False,
        "confirmatory_subjects_accessed": False,
        "external_validation_accessed": False,
        "gate_results": {
            "isolated_synthetic": synthetic,
            "real_background_injection": semisynthetic,
            "natural_session_S1_evidence": real_evidence,
        },
        "downstream_allowed": False,
        "skipped_by_design": {
            "classification_pilot": "Gate 2 and Gate 3 did not both pass",
            "session_S2_confirmation": "architecture was not eligible for freeze",
            "external_dataset_confirmation": "development mechanism gate failed",
            "strong_baseline_comparison": "no eligible DPC-SNN classifier result",
            "trained_model_efficiency": "no scientifically eligible trained checkpoint",
        },
        "claim_boundary": {
            "supported": [
                "delay estimator recovers strong isolated synthetic delays at or above 4 ms",
                "failure boundary under realistic background mixing",
                "lack of stable natural EEG routes under the locked estimator",
            ],
            "unsupported": [
                "classification superiority",
                "held-out generalization",
                "neuromorphic energy advantage",
                "stable biological propagation-delay discovery",
            ],
        },
        "source_snapshot_count": len(snapshot_paths),
        "source_snapshots": [str(path) for path in snapshot_paths],
        "superseded_directories": superseded,
        "artifact_manifest": str(output / "artifact_sha256_manifest.json"),
    }
    write_json(output / "campaign_decision.json", final)


if __name__ == "__main__":
    main()
