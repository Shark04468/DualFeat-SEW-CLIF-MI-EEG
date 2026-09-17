#!/usr/bin/env python3
"""Seal the V27 read-only utility protocol against the passed V25 campaign."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import file_sha256, sha256_fingerprint  # noqa: E402
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    source_tree_digest,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-freeze", required=True)
    parser.add_argument("--e26-audit", required=True)
    parser.add_argument("--config", default="configs/experiments/v27_bci2a_utility.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"V27 freeze output must be new: {output}")
    ensure_dir(output)
    parent = read_json(args.parent_freeze)
    audit = read_json(args.e26_audit)
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if audit.get("status") != "completed" or audit.get("decision", {}).get("status") != "pass":
        raise RuntimeError("V27 requires the passed independent E26 audit")
    if audit.get("freeze_sha256") != parent.get("combined_sha256"):
        raise RuntimeError("E26 audit and parent freeze are not bound to each other")
    source_tree = collect_source_tree_manifest(ROOT)
    payload = {
        "schema_version": 1,
        "architecture_id": parent["architecture_id"],
        "protocol": config["protocol"],
        "subjects": list(range(1, 10)),
        "seeds": list(range(5)),
        "arms": ["sew_clif_ce", "ann_plain_ce"],
        "parent_v25_freeze_sha256": parent["combined_sha256"],
        "parent_e26_audit_sha256": file_sha256(args.e26_audit),
        "source_tree_sha256": source_tree_digest(source_tree),
        "config_sha256": file_sha256(config_path),
        "utility_config": config,
        "historical_data_exposure": parent["historical_data_exposure"],
        "claim_scope": "posthoc frozen utility; no Session-E model updates or selection",
        "created_at": time.time(),
    }
    payload["combined_sha256"] = sha256_fingerprint(payload)
    write_json(output / "freeze_manifest.json", payload)
    write_json(output / "source_tree_manifest.json", source_tree)
    write_json(
        output / "freeze_summary.json",
        {
            "status": "frozen",
            "combined_sha256": payload["combined_sha256"],
            "source_tree_sha256": payload["source_tree_sha256"],
            "units": 45,
        },
    )
    print(payload["combined_sha256"])


if __name__ == "__main__":
    main()
