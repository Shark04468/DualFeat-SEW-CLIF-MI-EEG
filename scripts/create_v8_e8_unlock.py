#!/usr/bin/env python3
"""Freeze the OpenBMI external protocol before Session S2 is opened."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    source_tree_digest,
    validate_v8_freeze_manifest,
    write_v8_external_unlock_manifest,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--e6-audit", required=True)
    parser.add_argument("--e7", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-config", default="configs/datasets/openbmi.yaml")
    parser.add_argument(
        "--experiment-config", default="configs/experiments/v8_e8_openbmi_frozen.yaml"
    )
    args = parser.parse_args()

    output = ensure_dir(Path(args.output).resolve())
    source_tree = collect_source_tree_manifest(ROOT)
    source_digest = source_tree_digest(source_tree)
    freeze_path = Path(args.freeze).resolve()
    freeze = validate_v8_freeze_manifest(
        freeze_path, expected_source_tree_sha256=source_digest
    )
    e6_audit_path = Path(args.e6_audit).resolve() / "audit_report.json"
    e7_status_path = Path(args.e7).resolve() / "campaign_status.json"
    e7_gate_path = Path(args.e7).resolve() / "gate_decision.json"
    e6_audit = read_json(e6_audit_path)
    e7_status = read_json(e7_status_path)
    if (
        e6_audit.get("status") != "passed"
        or e6_audit.get("freeze_sha256") != freeze["combined_sha256"]
        or e7_status.get("status") != "completed"
        or e7_status.get("freeze_sha256") != freeze["combined_sha256"]
        or e7_status.get("openbmi_s2_accessed") is not False
    ):
        raise RuntimeError("E8 unlock requires passed E6 audit and completed frozen E7")
    dataset = yaml.safe_load(Path(args.dataset_config).resolve().read_text(encoding="utf-8"))
    experiment = yaml.safe_load(
        Path(args.experiment_config).resolve().read_text(encoding="utf-8")
    )
    if dataset.get("confirmatory_test_session") != "S2" or experiment.get(
        "stage"
    ) != "openbmi_confirmation":
        raise RuntimeError("OpenBMI external protocol config is inconsistent")
    channels = list(freeze["architecture"]["model_config"]["channel_names"])
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_tree_sha256": source_digest,
        "parent_freeze_sha256": freeze["combined_sha256"],
        "architecture_adaptation": {
            "permitted_change": "four_class_head_to_binary_head_only",
            "source_classes": 4,
            "target_classes": 2,
            "target_sfreq": 250,
            "ordered_channels": channels,
            "anti_aliasing": "scipy.signal.resample_poly_polyphase_FIR",
        },
        "training": {
            "primary_final_epoch": freeze["checkpoint_rule"]["final_epoch"],
            "matched_ann_final_epoch": freeze["checkpoint_rule"][
                "matched_ann_final_epoch"
            ],
            "scheduler_horizon": freeze["checkpoint_rule"]["scheduler_horizon"],
            "optimizer": freeze["training"],
            "augmentation": freeze["augmentation"],
            "delay": freeze["architecture"]["delay"],
            "selection_data": "BCI2a Session T only",
        },
        "dataset": {
            "name": dataset["name"],
            "train_session": "S1",
            "evaluation_session": "S2",
            "confirmatory_subjects": dataset["confirmatory_subjects"],
            "excluded_development_subjects": dataset["development_subjects"],
            "seeds": experiment["seeds"],
        },
        "analysis_plan": {
            **dict(experiment["analysis"]),
            "arms": experiment["arms"],
            "matched_ann_omitted_when_primary_is_ann": True,
        },
        "evidence_hashes": {
            "parent_freeze": file_sha256(freeze_path),
            "e6_audit": file_sha256(e6_audit_path),
            "e7_status": file_sha256(e7_status_path),
            "e7_gate": file_sha256(e7_gate_path),
            "dataset_config": file_sha256(Path(args.dataset_config).resolve()),
            "experiment_config": file_sha256(Path(args.experiment_config).resolve()),
        },
        "heldout_access": {
            "openbmi_s2_accessed_before_unlock": False,
            "s2_checkpoint_selection_allowed": False,
            "s2_gradient_updates_allowed": False,
        },
    }
    write_json(output / "source_tree_manifest.json", source_tree)
    write_v8_external_unlock_manifest(output / "external_unlock_manifest.json", payload)
    unlock = read_json(output / "external_unlock_manifest.json")
    summary = {
        "status": "completed",
        "external_unlock_sha256": unlock["combined_sha256"],
        "parent_freeze_sha256": freeze["combined_sha256"],
        "confirmatory_subjects": len(dataset["confirmatory_subjects"]),
        "seeds": experiment["seeds"],
        "openbmi_s2_accessed": False,
    }
    write_json(output / "unlock_summary.json", summary)
    write_run_artifact_manifest(
        output,
        required_files=(
            "manifest.json",
            "external_unlock_manifest.json",
            "unlock_summary.json",
            "source_tree_manifest.json",
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
