"""Freeze a retrospective BNCI2014-004 recovery run before local evaluation access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v8_protocol import collect_source_tree_manifest, source_tree_digest
from dpc_snn.experiments.v62_protocol import file_sha256, sha256_fingerprint
from dpc_snn.utils.io import ensure_dir, write_json


def _existing_files(storage_root: Path, role: str) -> list[str]:
    matches = {
        path.resolve()
        for pattern in (f"B??{role}.mat",)
        for path in storage_root.rglob(pattern)
        if path.is_file()
    }
    return sorted(str(path) for path in matches)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "v30_bnci2014_004_recovery.yaml"),
    )
    parser.add_argument("--storage-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"recovery freeze output must be new: {output}")
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    disclosure = config.get("blindness", {})
    if (
        config.get("status") != "recovered_parent_replication"
        or disclosure.get("evaluation_sessions_previously_accessed_by_project") is not True
        or disclosure.get("confirmatory_status") != "retrospective_recovery_not_blind"
    ):
        raise RuntimeError("recovery config must explicitly disclose prior project access")
    storage_root = Path(args.storage_root).resolve()
    existing_training = _existing_files(storage_root, "T")
    existing_evaluation = _existing_files(storage_root, "E")
    if existing_evaluation:
        raise RuntimeError(
            "recovery freeze must precede evaluation download on this instance: "
            + json.dumps(existing_evaluation[:20])
        )
    source_tree = collect_source_tree_manifest(ROOT)
    payload = {
        "schema": "dpc-snn-v30-recovery-freeze/v1",
        "status": "frozen_before_recovery_run_evaluation_access",
        "scientific_status": "retrospective_recovery_not_blind",
        "project_had_historical_evaluation_access": True,
        "evaluation_unavailable_to_this_recovery_run_before_barrier": True,
        "config": config,
        "config_sha256": file_sha256(config_path),
        "source_tree_sha256": source_tree_digest(source_tree),
        "storage_root": str(storage_root),
        "training_files_present_before_freeze": existing_training,
        "evaluation_files_present_before_freeze": [],
        "evaluation_gradient_updates_allowed": False,
    }
    payload["combined_sha256"] = sha256_fingerprint(payload)
    ensure_dir(output)
    write_json(output / "freeze_manifest.json", payload)
    print(json.dumps({"status": payload["status"], "sha256": payload["combined_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
