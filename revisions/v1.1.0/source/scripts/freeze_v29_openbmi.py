#!/usr/bin/env python3
"""Seal the retrospective OpenBMI S1-to-S2 replication before current S2 access."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e28-decision", required=True)
    parser.add_argument("--publication-root", required=True)
    parser.add_argument(
        "--config", default="configs/experiments/v29_openbmi_replication.yaml"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"V29 freeze output must be new: {output}")
    ensure_dir(output)
    decision_path = Path(args.e28_decision).resolve()
    decision = read_json(decision_path)
    if decision.get("status") != "pass" or decision.get("authorized_next_stage") != "E29_openbmi_replication":
        raise RuntimeError("E28 did not authorize E29")
    publication = Path(args.publication_root).resolve()
    publication_audit = read_json(publication / "audit" / "audit_report.json")
    if publication_audit.get("status") != "passed" or publication_audit.get("issues"):
        raise RuntimeError("publication baseline checkpoint source is not independently clean")
    barrier_path = publication / "checkpoint_barrier.json"
    barrier = read_json(barrier_path)
    if (
        barrier.get("schema")
        != "dpc-snn-v8-posthoc-publication-checkpoint-barrier/v1"
        or barrier.get("all_training_complete_before_evaluation") is not True
        or len(barrier.get("records", [])) != int(barrier.get("expected_runs", -1))
    ):
        raise RuntimeError("publication baseline checkpoint barrier is not sealed")
    config_path = (ROOT / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "retrospective_external_replication_after_historical_s2_exposure":
        raise RuntimeError("V29 historical-exposure disclosure is missing")
    source_tree = collect_source_tree_manifest(ROOT)
    payload = {
        "schema": "dpc-snn-v29-openbmi-freeze/v1",
        "status": "frozen_before_current_campaign_s2_access",
        "config": config,
        "config_sha256": file_sha256(config_path),
        "source_tree_sha256": source_tree_digest(source_tree),
        "e28_gate_decision": str(decision_path),
        "e28_gate_sha256": file_sha256(decision_path),
        "publication_root": str(publication),
        "publication_checkpoint_barrier_sha256": file_sha256(barrier_path),
        "publication_checkpoint_contract_sha256": barrier["combined_sha256"],
        "current_campaign_openbmi_s2_accessed": False,
        "current_campaign_s2_gradient_updates_allowed": False,
        "historical_exposure": config["historical_exposure"],
    }
    payload["combined_sha256"] = sha256_fingerprint(payload)
    write_json(output / "freeze_manifest.json", payload)
    write_json(
        output / "freeze_summary.json",
        {
            "status": "frozen",
            "freeze_sha256": payload["combined_sha256"],
            "source_tree_sha256": payload["source_tree_sha256"],
            "current_campaign_openbmi_s2_accessed": False,
        },
    )
    print(json.dumps(read_json(output / "freeze_summary.json"), indent=2))


if __name__ == "__main__":
    main()
