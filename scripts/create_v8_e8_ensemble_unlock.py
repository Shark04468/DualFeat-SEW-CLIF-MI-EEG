#!/usr/bin/env python3
"""Seal the selected V8 ensemble OpenBMI protocol before Session S2 access."""

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
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
)
from dpc_snn.data.v8_openbmi import OPENBMI_FIXED_INPUT_ADAPTER  # noqa: E402
from dpc_snn.experiments.v8_ensemble import (  # noqa: E402
    validate_ensemble_freeze_contract,
)
from dpc_snn.experiments.v8_ensemble_followup import (  # noqa: E402
    resolve_frozen_channel_basis,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    source_tree_digest,
    validate_v8_freeze_manifest,
    write_v8_external_unlock_manifest,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_json  # noqa: E402
from scripts.run_v8_e7_ensemble_utility import CAMPAIGN_FILES as E7_FILES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--e6-audit", required=True)
    parser.add_argument("--e7", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-config", default="configs/datasets/openbmi.yaml")
    parser.add_argument(
        "--experiment-config",
        default="configs/experiments/v8_e8_ensemble_frozen.yaml",
    )
    args = parser.parse_args()

    output = ensure_dir(Path(args.output).resolve())
    source_tree = collect_source_tree_manifest(ROOT)
    source_digest = source_tree_digest(source_tree)
    freeze_path = Path(args.freeze).resolve()
    freeze = validate_ensemble_freeze_contract(validate_v8_freeze_manifest(freeze_path))
    e6_audit_path = Path(args.e6_audit).resolve() / "audit_report.json"
    e7_root = Path(args.e7).resolve()
    validate_run_artifact_manifest(e7_root, required_files=E7_FILES, verify_hashes=True)
    e6_audit = read_json(e6_audit_path)
    e7_status = read_json(e7_root / "campaign_status.json")
    e7_gate = read_json(e7_root / "gate_decision.json")
    if (
        e6_audit.get("status") != "passed"
        or e6_audit.get("runs_audited") != 45
        or e6_audit.get("freeze_sha256") != freeze["combined_sha256"]
        or e7_status.get("status") != "completed"
        or e7_status.get("full_registered_contract") is not True
        or e7_status.get("freeze_sha256") != freeze["combined_sha256"]
        or e7_status.get("openbmi_s2_accessed") is not False
        or e7_gate.get("passed") is not True
    ):
        raise RuntimeError("E8 unlock requires passed full E6 and E7 ensemble gates")
    dataset_path = Path(args.dataset_config).resolve()
    experiment_path = Path(args.experiment_config).resolve()
    dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    adaptation = experiment.get("architecture_adaptation", {})
    if (
        dataset.get("confirmatory_test_session") != "S2"
        or experiment.get("stage") != "openbmi_confirmation"
        or experiment.get("protocol")
        != "openbmi_s1_train_all_then_s2_single_evaluation"
        or adaptation.get("permitted_change")
        != "four_class_head_to_binary_head_only"
        or adaptation.get("channel_policy")
        != "frozen_22_sensor_basis_with_fixed_fcz_linear_interpolation"
        or adaptation.get("input_adapter") != OPENBMI_FIXED_INPUT_ADAPTER
    ):
        raise RuntimeError("OpenBMI ensemble confirmation config is inconsistent")
    channels = resolve_frozen_channel_basis(freeze)
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
            "input_channel_adapter": OPENBMI_FIXED_INPUT_ADAPTER,
            "anti_aliasing": "scipy.signal.resample_poly_polyphase_FIR",
            "ensemble_rule_changed": False,
        },
        "training": {
            "components": freeze["checkpoint_rule"]["components"],
            "optimizer": freeze["training"],
            "augmentation": freeze["augmentation"],
            "delay": freeze["architecture"]["delay"],
            "all_checkpoints_before_any_s2": True,
            "selection_data": "BCI2a Session T and OpenBMI S1 only",
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
        },
        "evidence_hashes": {
            "parent_freeze": file_sha256(freeze_path),
            "e6_audit": file_sha256(e6_audit_path),
            "e7_manifest": file_sha256(e7_root / "manifest.json"),
            "e7_status": file_sha256(e7_root / "campaign_status.json"),
            "e7_gate": file_sha256(e7_root / "gate_decision.json"),
            "dataset_config": file_sha256(dataset_path),
            "experiment_config": file_sha256(experiment_path),
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
        "stage": "E8_ENSEMBLE_UNLOCK",
        "external_unlock_sha256": unlock["combined_sha256"],
        "parent_freeze_sha256": freeze["combined_sha256"],
        "source_tree_sha256": source_digest,
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
