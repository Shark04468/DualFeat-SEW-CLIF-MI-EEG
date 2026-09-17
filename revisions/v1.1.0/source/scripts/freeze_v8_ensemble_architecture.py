#!/usr/bin/env python3
"""Freeze the selected ATCNet/FBCNet plus SEW-CLIF V8 branch before Session E."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.baselines.neural import verify_official_source_locks  # noqa: E402
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    sha256_fingerprint,
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    source_tree_digest,
    write_v8_freeze_manifest,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_json  # noqa: E402


FREEZE_FILES = (
    "manifest.json",
    "freeze_manifest.json",
    "freeze_summary.json",
    "resolved_freeze_config.yaml",
    "source_tree_manifest.json",
    "official_source_locks.json",
)


def _passed_development_run(root: Path, *, name: str) -> dict[str, Any]:
    metrics = read_json(root / "metrics.json")
    if str(metrics.get("status")) != "passed":
        raise RuntimeError(f"{name} did not pass: {root}")
    if bool(metrics.get("session_e_accessed")) or bool(metrics.get("openbmi_s2_accessed")):
        raise RuntimeError(f"{name} accessed held-out data before the freeze")
    return metrics


def _validate_declared_manifest(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "manifest.json")
    required = tuple(str(value) for value in manifest["required_files"])
    return validate_run_artifact_manifest(
        root,
        required_files=required,
        verify_hashes=True,
        verify_prediction_schema=False,
    )


def _development_epochs(
    *, e1: Path, fbc: Path, e4: Path
) -> tuple[dict[str, int], dict[str, Any]]:
    identities = [
        (subject, seed, fold)
        for subject in (1, 3, 8)
        for seed in (0, 1, 2)
        for fold in range(6)
    ]
    values: dict[str, list[int]] = {
        "atcnet": [],
        "fbcnet": [],
        "sew_clif": [],
        "ann_sew": [],
    }
    for subject, seed, fold in identities:
        atc_result = read_json(
            e1
            / "atcnet"
            / f"subject_{subject:02d}"
            / f"seed_{seed}"
            / f"fold_{fold}"
            / "result.json"
        )
        fbc_result = read_json(
            fbc
            / "fbcnet"
            / f"subject_{subject:02d}"
            / f"seed_{seed}"
            / f"fold_{fold}"
            / "result.json"
        )
        values["atcnet"].append(int(atc_result["selected_outer_retrain_epoch"]))
        values["fbcnet"].append(int(fbc_result["selected_outer_retrain_epoch"]))
        fold_root = e4 / f"formal_s{subject}_seed{seed}_fold{fold}"
        for variant in ("sew_clif", "ann_sew"):
            values[variant].append(
                int(read_json(fold_root / variant / "metrics.json")["selected_epoch"])
            )
    if any(len(items) != 54 or any(value < 1 for value in items) for items in values.values()):
        raise RuntimeError("development epoch evidence is incomplete")
    selected = {
        name: int(math.floor(float(statistics.median(items)) + 0.5))
        for name, items in values.items()
    }
    evidence = {
        name: {
            "count": len(items),
            "minimum": min(items),
            "maximum": max(items),
            "median": float(statistics.median(items)),
            "round_half_up": selected[name],
        }
        for name, items in values.items()
    }
    return selected, evidence


def _validate_selection_contract(
    config: dict[str, Any],
    *,
    e4_report: Path,
    e4_lock: Path,
    e1: Path,
    e1_audit: Path,
    fbc: Path,
    e2_gate: Path,
    e4: Path,
    e4_gate: Path,
    feasibility: Path,
    final_gate: Path,
    feasibility_config: Path,
    architecture_gate_config: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if config.get("selected_branch") != "confirmed_e4_sequence_residual":
        raise RuntimeError("the recovered final branch is not the confirmed E4 fallback")
    basis = dict(config["selection_basis"])
    required_basis = {
        "delay_prior_feasibility_passed": False,
        "passing_subjects": [1],
        "failing_subjects": [3, 8],
        "e5_formal_hpo_completed": False,
        "delay_branch_promoted": False,
        "e4_fallback_gate_passed": True,
        "e4_confirmation_pairs": 6,
        "e4_positive_pairs": 4,
    }
    for field, expected in required_basis.items():
        if basis.get(field) != expected:
            raise RuntimeError(f"restored selection basis drifted at {field!r}")
    if basis.get("provenance_status") != "original_hash_verified_development_artifacts":
        raise RuntimeError("the freeze must bind the original development artifacts")
    if basis.get("raw_development_fold_artifacts_available_on_current_instance") is not True:
        raise RuntimeError("the original fold artifacts are required on the freeze instance")

    e1_status = read_json(e1 / "campaign_status.json")
    audit = read_json(e1_audit / "audit_report.json")
    fbc_status = read_json(fbc / "campaign_status.json")
    if (
        e1_status.get("status") != "completed"
        or e1_status.get("runs") != 27
        or e1_status.get("screened_models")
        != [
            "eegnet",
            "fbcnet",
            "atcnet",
            "tcformer",
            "eeg_conformer",
            "mi_snn_plif",
            "bfatcnet",
        ]
        or e1_status.get("session_e_accessed") is not False
        or audit.get("status") != "passed"
        or audit.get("artifact_hashes_valid") is not True
        or audit.get("prediction_schema_valid") is not True
        or audit.get("nested_trial_identity_valid") is not True
        or fbc_status.get("status") != "completed"
        or fbc_status.get("runs") != 9
        or fbc_status.get("session_e_accessed") is not False
    ):
        raise RuntimeError("E1 ATCNet/FBCNet evidence is incomplete or unaudited")
    fixed_fusion = read_json(e2_gate / "gate.json")
    aggregate = fixed_fusion.get("aggregate", {})
    if (
        aggregate.get("passed") is not True
        or aggregate.get("positive_pairs") != 9
        or float(aggregate.get("fixed_fusion_macro_accuracy", 0.0))
        != 0.8885030864197532
        or aggregate.get("session_e_accessed") is not False
    ):
        raise RuntimeError("the fixed ATCNet/FBCNet anchor did not pass its locked gate")

    _validate_declared_manifest(feasibility)
    feasibility_decision = read_json(feasibility / "feasibility_decision.json")
    if (
        feasibility_decision.get("passed") is not False
        or feasibility_decision.get("decision")
        != "reject_delay_branch_and_use_predeclared_e4_fallback"
        or feasibility_decision.get("passing_subject_folds") != 1
        or feasibility_decision.get("threshold_relaxation_after_observation") is not False
        or feasibility_decision.get("session_e_accessed") is not False
    ):
        raise RuntimeError("the delay-prior feasibility decision is inconsistent")
    _validate_declared_manifest(final_gate)
    final_decision = read_json(final_gate / "decision.json")
    if (
        final_decision.get("selected_branch") != "confirmed_e4_sequence_residual"
        or final_decision.get("selection_reason")
        != "registered_delay_prior_feasibility_failed"
        or final_decision.get("e4_fallback_gate_passed") is not True
        or final_decision.get("post_session_e_selection") is not False
        or final_decision.get("session_e_accessed") is not False
    ):
        raise RuntimeError("the final architecture gate did not select the E4 fallback")

    architecture = dict(config["architecture"])
    if architecture.get("primary_decoder") != "sew_clif":
        raise RuntimeError("the selected decoder must be SEW-CLIF")
    if architecture.get("matched_ann_decoder") != "ann_sew":
        raise RuntimeError("the matched control must be ANN-SEW")
    if architecture.get("delay") != {
        "enabled": False,
        "mode": "off",
        "status": "failed_development_prerequisite_retained_as_future_ablation_only",
    }:
        raise RuntimeError("the failed delay branch must remain disabled in E6")
    residual = dict(architecture["residual_fusion"])
    if residual != {
        "mode": "anchor_entropy",
        "maximum_decoder_weight": 0.05,
        "equation": "p_final=(1-g)*p_anchor+g*p_decoder;g=0.05*H(p_anchor)/log(4)",
        "learned_calibration": False,
    }:
        raise RuntimeError("the entropy residual rule differs from the confirmed E4 rule")

    checkpoint = dict(config["checkpoint_rule"])
    expected_epochs = {
        "selection_data": "BCI2a Session T development only",
        "atcnet_final_epoch": 79,
        "atcnet_scheduler_horizon": 300,
        "fbcnet_final_epoch": 70,
        "fbcnet_scheduler_horizon": 300,
        "decoder_final_epoch": 20,
        "decoder_scheduler_horizon": 120,
        "heldout_checkpoint_selection": False,
    }
    if checkpoint != expected_epochs:
        raise RuntimeError("fixed all-T checkpoint rules drifted from development medians")

    report_text = e4_report.read_text(encoding="utf-8")
    missing = [
        fragment
        for fragment in config["required_e4_report_fragments"]
        if str(fragment) not in report_text
    ]
    if missing:
        raise RuntimeError(f"E4 decision report is missing locked evidence: {missing}")
    lock = yaml.safe_load(e4_lock.read_text(encoding="utf-8"))
    if (
        lock.get("selection", {}).get("selected_snn_variant") != "sew_clif"
        or lock.get("selection", {}).get("matched_ann_control") != "ann_sew"
        or float(lock.get("selection", {}).get("maximum_decoder_weight", -1.0)) != 0.05
        or lock.get("data_access", {}).get("heldout_session_e_accessed") is not False
    ):
        raise RuntimeError("E4 confirmation lock differs from the selected branch")
    original_e4_gate = read_json(e4_gate / "gate.json")
    if (
        original_e4_gate.get("passed") is not True
        or original_e4_gate.get("selected_snn_variant") != "sew_clif"
        or original_e4_gate.get("matched_ann_control") != "ann_sew"
        or original_e4_gate.get("confirmation_pairs") != 6
        or original_e4_gate.get("positive_subject_seed_pairs") != 4
        or float(original_e4_gate.get("macro_snn_accuracy", 0.0)) != 0.890625
        or original_e4_gate.get("session_e_accessed") is not False
    ):
        raise RuntimeError("original E4 confirmation gate differs from the locked result")

    selected_epochs, epoch_evidence = _development_epochs(e1=e1, fbc=fbc, e4=e4)
    if selected_epochs != {
        "atcnet": checkpoint["atcnet_final_epoch"],
        "fbcnet": checkpoint["fbcnet_final_epoch"],
        "sew_clif": checkpoint["decoder_final_epoch"],
        "ann_sew": checkpoint["decoder_final_epoch"],
    }:
        raise RuntimeError("fixed epochs do not match the original 54-fold medians")

    feasibility = yaml.safe_load(feasibility_config.read_text(encoding="utf-8"))
    if (
        feasibility.get("expected", {}).get("subjects") != [1, 3, 8]
        or feasibility.get("expected", {}).get("require_all_subject_folds_pass") is not True
        or feasibility.get("decision_policy", {}).get("fail")
        != "reject_delay_branch_and_use_predeclared_e4_fallback"
        or feasibility.get("decision_policy", {}).get(
            "threshold_relaxation_after_observation"
        )
        is not False
    ):
        raise RuntimeError("delay feasibility gate no longer encodes the fail-closed policy")
    architecture_gate = yaml.safe_load(
        architecture_gate_config.read_text(encoding="utf-8")
    )
    if architecture_gate.get("selection_policy", {}).get("priority") != [
        "full_delay_routed_snn",
        "matched_zero_routed_snn",
        "confirmed_e4_sequence_residual",
    ]:
        raise RuntimeError("final branch priority differs from the preregistered order")
    return architecture, checkpoint, basis, epoch_evidence


def _artifact_hashes(paths: dict[str, Path]) -> dict[str, str]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("freeze evidence is incomplete: " + ", ".join(missing))
    return {name: file_sha256(path) for name, path in sorted(paths.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0", required=True)
    parser.add_argument("--e0", required=True)
    parser.add_argument("--e1", required=True)
    parser.add_argument("--e1-audit", required=True)
    parser.add_argument("--fbc", required=True)
    parser.add_argument("--e2-gate", required=True)
    parser.add_argument("--e4", required=True)
    parser.add_argument("--e4-gate", required=True)
    parser.add_argument("--feasibility", required=True)
    parser.add_argument("--final-gate", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument(
        "--config", default="configs/experiments/v8_e6_ensemble_freeze.yaml"
    )
    parser.add_argument(
        "--e4-report", default="reports/v8_e4_sequence_decoder_decision_20260719.md"
    )
    parser.add_argument(
        "--e4-lock", default="configs/experiments/v8_e4_entropy_residual_confirmation.yaml"
    )
    parser.add_argument(
        "--feasibility-config",
        default="configs/experiments/v8_e3_prior_feasibility_gate.yaml",
    )
    parser.add_argument(
        "--architecture-gate-config",
        default="configs/experiments/v8_e3_final_architecture_gate.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = ensure_dir(Path(args.output).resolve())
    p0 = Path(args.p0).resolve()
    e0 = Path(args.e0).resolve()
    e1 = Path(args.e1).resolve()
    e1_audit = Path(args.e1_audit).resolve()
    fbc = Path(args.fbc).resolve()
    e2_gate = Path(args.e2_gate).resolve()
    e4 = Path(args.e4).resolve()
    e4_gate = Path(args.e4_gate).resolve()
    feasibility = Path(args.feasibility).resolve()
    final_gate = Path(args.final_gate).resolve()
    source_root = Path(args.source_root).resolve()
    config_path = Path(args.config).resolve()
    e4_report = Path(args.e4_report).resolve()
    e4_lock = Path(args.e4_lock).resolve()
    feasibility_config = Path(args.feasibility_config).resolve()
    architecture_gate_config = Path(args.architecture_gate_config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    p0_metrics = _passed_development_run(p0, name="P0")
    e0_metrics = _passed_development_run(e0, name="E0")
    p0_reference = read_json(e0 / "p0_reference.json")
    p0_fingerprint = read_json(p0 / "source_fingerprint.json")
    if p0_reference.get("combined_sha256") != p0_fingerprint.get("combined_sha256"):
        raise RuntimeError("E0 is not bound to the supplied P0 protocol run")

    architecture, checkpoint, selection_basis, epoch_evidence = (
        _validate_selection_contract(
            config,
            e4_report=e4_report,
            e4_lock=e4_lock,
            e1=e1,
            e1_audit=e1_audit,
            fbc=fbc,
            e2_gate=e2_gate,
            e4=e4,
            e4_gate=e4_gate,
            feasibility=feasibility,
            final_gate=final_gate,
            feasibility_config=feasibility_config,
            architecture_gate_config=architecture_gate_config,
        )
    )
    official_locks = verify_official_source_locks(source_root)
    source_tree = collect_source_tree_manifest(ROOT)
    source_digest = source_tree_digest(source_tree)
    write_json(output / "source_tree_manifest.json", source_tree)
    write_json(output / "official_source_locks.json", official_locks)
    (output / "resolved_freeze_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    evidence_paths = {
        "p0_manifest": p0 / "manifest.json",
        "p0_metrics": p0 / "metrics.json",
        "p0_source_fingerprint": p0 / "source_fingerprint.json",
        "e0_manifest": e0 / "manifest.json",
        "e0_metrics": e0 / "metrics.json",
        "e0_p0_reference": e0 / "p0_reference.json",
        "e1_campaign_status": e1 / "campaign_status.json",
        "e1_summary": e1 / "summary.csv",
        "e1_audit": e1_audit / "audit_report.json",
        "fbc_campaign_status": fbc / "campaign_status.json",
        "fbc_summary": fbc / "summary.csv",
        "e2_fixed_fusion_gate": e2_gate / "gate.json",
        "e4_confirmation_gate": e4_gate / "gate.json",
        "e4_confirmation_predictions": e4_gate / "predictions.npz",
        "e3_feasibility_manifest": feasibility / "manifest.json",
        "e3_feasibility_decision": feasibility / "feasibility_decision.json",
        "final_architecture_manifest": final_gate / "manifest.json",
        "final_architecture_decision": final_gate / "decision.json",
        "e4_decision_report": e4_report,
        "e4_confirmation_lock": e4_lock,
        "delay_prior_feasibility_config": feasibility_config,
        "final_architecture_gate_config": architecture_gate_config,
        "ensemble_freeze_config": config_path,
    }
    artifact_hashes = _artifact_hashes(evidence_paths)
    artifact_hashes["official_source_locks_payload"] = sha256_fingerprint(official_locks)

    model_config = {
        "architecture_id": "v8_atc_fbc_entropy_residual_sequence_ensemble_r1",
        **architecture,
    }
    ann_config = {
        **model_config,
        "primary_decoder": "ann_sew",
        "matched_control_for": "sew_clif",
    }
    freeze_payload = {
        "freeze_scope": "pre_E6_architecture_and_analysis",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_tree_sha256": source_digest,
        "architecture": {
            "model_config": model_config,
            "matched_ann_control_config": ann_config,
            "primary_variant": "sew_clif",
            "delay": {"enabled": False, "mode": "off"},
        },
        "training": {
            "baseline_optimizers": "registered_v62_baseline_optimizers",
            "sequence_decoder": dict(architecture["decoder"]),
            "all_session_t_retraining": True,
            "session_e_gradient_updates": False,
        },
        "preprocessing": dict(config["preprocessing"]),
        "augmentation": dict(config["augmentation"]),
        "checkpoint_rule": {
            "selection_data": checkpoint["selection_data"],
            "final_epoch": checkpoint["decoder_final_epoch"],
            "scheduler_horizon": checkpoint["decoder_scheduler_horizon"],
            "matched_ann_final_epoch": checkpoint["decoder_final_epoch"],
            "heldout_checkpoint_selection": False,
            "components": checkpoint,
        },
        "analysis_plan": dict(config["analysis_plan"]),
        "development_evidence": {
            **selection_basis,
            "p0_status": p0_metrics["status"],
            "e0_status": e0_metrics["status"],
            "fixed_epoch_evidence": epoch_evidence,
            "all_original_gate_manifests_hash_verified": True,
        },
        "baselines": {
            "models": ["atcnet", "fbcnet"],
            "strongest_development_model": "atcnet",
            "fixed_epochs": {"atcnet": 79, "fbcnet": 70},
            "scheduler_horizons": {"atcnet": 300, "fbcnet": 300},
            "official_source_locks": official_locks,
            "protocol_config": {
                "preprocessing": dict(config["preprocessing"]),
                "augmentation": dict(config["augmentation"]),
            },
        },
        "heldout_access": {
            "bci2a_session_e_accessed_before_freeze": False,
            "openbmi_session_s2_accessed_before_freeze": False,
            "status": "sealed_until_validated_freeze",
        },
        "artifact_hashes": artifact_hashes,
    }
    write_v8_freeze_manifest(output / "freeze_manifest.json", freeze_payload)
    freeze = read_json(output / "freeze_manifest.json")
    summary = {
        "status": "passed",
        "stage": "V8_ENSEMBLE_PRE_E6_FREEZE",
        "selected_branch": config["selected_branch"],
        "source_tree_sha256": source_digest,
        "freeze_sha256": freeze["combined_sha256"],
        "delay_enabled": False,
        "raw_development_fold_artifacts_available": True,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    write_json(output / "freeze_summary.json", summary)
    write_run_artifact_manifest(output, required_files=FREEZE_FILES)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
