#!/usr/bin/env python3
"""Freeze the genuinely blind BNCI2014-004 protocol before any data access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import file_sha256, sha256_fingerprint  # noqa: E402
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    source_tree_digest,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_json  # noqa: E402


def _existing_files(storage_root: Path, role: str) -> list[str]:
    matches: set[Path] = set()
    for pattern in (f"B??{role}.mat",):
        matches.update(path for path in storage_root.rglob(pattern) if path.is_file())
    return sorted(str(path.resolve()) for path in matches)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e29-decision", required=True)
    parser.add_argument("--storage-root", required=True)
    parser.add_argument(
        "--config", default="configs/experiments/v30_bnci2014_004_blind.yaml"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"V30 freeze output must be new: {output}")
    decision_path = Path(args.e29_decision).resolve()
    decision = read_json(decision_path)
    if (
        decision.get("status") != "pass"
        or decision.get("authorized_next_stage") != "P30_external_blind_dataset"
    ):
        raise RuntimeError("E29 did not authorize P30")

    config_path = (ROOT / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    blindness = dict(config["blindness"])
    if (
        config.get("status") != "project_level_blind_external_confirmation"
        or blindness.get("evaluation_sessions_previously_accessed_by_project") is not False
    ):
        raise RuntimeError("V30 project-level blindness declaration is invalid")
    storage_root = Path(args.storage_root).resolve()
    existing_training = _existing_files(storage_root, "T")
    existing_evaluation = _existing_files(storage_root, "E")
    if existing_evaluation:
        raise RuntimeError(
            "Refusing a project-level blind freeze because BNCI2014-004 evaluation "
            "files already exist: "
            + json.dumps(existing_evaluation[:20])
        )

    ensure_dir(output)
    source_tree = collect_source_tree_manifest(ROOT)
    payload = {
        "schema": "dpc-snn-v30-bnci2014-004-freeze/v1",
        "status": "frozen_before_any_bnci2014_004_evaluation_access",
        "config": config,
        "config_sha256": file_sha256(config_path),
        "source_tree_sha256": source_tree_digest(source_tree),
        "e29_gate_decision": str(decision_path),
        "e29_gate_sha256": file_sha256(decision_path),
        "storage_root": str(storage_root),
        "training_files_present_before_freeze": existing_training,
        "evaluation_files_present_before_freeze": existing_evaluation,
        "training_sessions_accessed": bool(existing_training),
        "evaluation_sessions_accessed": False,
        "evaluation_gradient_updates_allowed": False,
    }
    payload["combined_sha256"] = sha256_fingerprint(payload)
    write_json(output / "freeze_manifest.json", payload)
    write_json(
        output / "freeze_summary.json",
        {
            "status": "frozen",
            "freeze_sha256": payload["combined_sha256"],
            "source_tree_sha256": payload["source_tree_sha256"],
            "training_files_present_before_freeze": len(existing_training),
            "evaluation_files_present_before_freeze": 0,
            "evaluation_sessions_accessed": False,
        },
    )
    print(json.dumps(read_json(output / "freeze_summary.json"), indent=2))


if __name__ == "__main__":
    main()
