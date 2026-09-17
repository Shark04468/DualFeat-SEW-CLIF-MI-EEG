#!/usr/bin/env python
"""Validate and merge classifier-free V5.1 real-evidence subject shards."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.config import load_yaml  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--protocol-config", default="configs/experiments/v51_hardware_free_plan.yaml"
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output = ensure_dir(args.output)
    shard_dirs = sorted(path.parent for path in input_root.glob("*/audit_manifest.json"))
    if not shard_dirs:
        raise FileNotFoundError(f"No V5.1 real-evidence shards found under {input_root}")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    fingerprints: set[str] = set()
    experiments: set[str] = set()
    expected_subjects: set[int] = set()
    for shard in shard_dirs:
        exit_path = shard / "exit_code.txt"
        if not exit_path.exists() or exit_path.read_text(encoding="utf-8").strip() != "0":
            raise RuntimeError(f"Shard did not exit successfully: {shard}")
        manifest = read_json(shard / "audit_manifest.json")
        if manifest["stage"] != "real":
            raise RuntimeError(f"Shard is not a real-evidence audit: {shard}")
        fingerprints.add(str(manifest["source_fingerprint"]))
        experiments.add(str(manifest["experiment_id"]))
        expected_subjects.update(int(subject) for subject in manifest["subjects"])
        shard_rows = _read_csv(shard / "real_evidence_subject_summary.csv")
        expected_rows = len(manifest["subjects"]) * len(manifest["spaces"])
        if len(shard_rows) != expected_rows:
            raise RuntimeError(
                f"Shard {shard} contains {len(shard_rows)} rows, expected {expected_rows}"
            )
        for row in shard_rows:
            key = (int(row["subject"]), str(row["evidence_space"]))
            if key in seen:
                raise RuntimeError(f"Duplicate subject/space across shards: {key}")
            seen.add(key)
            rows.append(row)

    if len(fingerprints) != 1 or len(experiments) != 1:
        raise RuntimeError("Shard source fingerprint or experiment identity differs")
    rows.sort(key=lambda row: (int(row["subject"]), str(row["evidence_space"])))
    write_csv(output / "real_evidence_subject_summary.csv", rows)

    protocol = load_yaml(ROOT / args.protocol_config)
    required = int(protocol["real_evidence_gate"]["minimum_passing_development_subjects"])
    spaces = sorted({str(row["evidence_space"]) for row in rows})
    passing_subjects_by_space = {
        space: sorted(
            int(row["subject"])
            for row in rows
            if str(row["evidence_space"]) == space and _as_bool(row["subject_gate_passed"])
        )
        for space in spaces
    }
    passing_spaces = sorted(
        space for space, subjects in passing_subjects_by_space.items() if len(subjects) >= required
    )
    per_space_rows = [
        {
            "evidence_space": space,
            "passing_subject_count": len(passing_subjects_by_space[space]),
            "required_subject_count": required,
            "passed": len(passing_subjects_by_space[space]) >= required,
            "passing_subjects": ";".join(map(str, passing_subjects_by_space[space])),
            "median_accepted_evidence_edges": statistics.median(
                int(float(row["accepted_evidence_edges"]))
                for row in rows
                if str(row["evidence_space"]) == space
            ),
        }
        for space in spaces
    ]
    write_csv(output / "real_evidence_space_summary.csv", per_space_rows)
    decision = {
        "status": "passed" if passing_spaces else "failed",
        "passed": bool(passing_spaces),
        "passing_evidence_spaces": passing_spaces,
        "passing_subjects_by_evidence_space": passing_subjects_by_space,
        "required_subject_count": required,
        "selection_rule": "one_fixed_evidence_space_must_pass_across_subjects",
        "classifier_training": False,
        "heldout_session_S2_accessed": False,
        "source_fingerprint": next(iter(fingerprints)),
        "subjects": sorted(expected_subjects),
        "shards": [str(path) for path in shard_dirs],
    }
    write_json(output / "stage_decision.json", decision)
    write_json(
        output / "merged_manifest.json",
        {
            "experiment_id": next(iter(experiments)),
            "source_fingerprint": next(iter(fingerprints)),
            "shard_count": len(shard_dirs),
            "row_count": len(rows),
            "unique_subject_space_count": len(seen),
            "classifier_training": False,
            "heldout_session_S2_accessed": False,
        },
    )


if __name__ == "__main__":
    main()
