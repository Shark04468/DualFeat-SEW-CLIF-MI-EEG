#!/usr/bin/env python3
"""Create the only manifest that may unlock V8 held-out evaluation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
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
    write_v8_freeze_manifest,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_json  # noqa: E402


FREEZE_FILES = (
    "manifest.json",
    "freeze_manifest.json",
    "freeze_summary.json",
    "source_tree_manifest.json",
)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty development summary: {path}")
    return rows


def _require_report(path: Path, *, expected: str = "passed") -> dict[str, Any]:
    report = read_json(path)
    if str(report.get("status")) != expected:
        raise RuntimeError(f"development gate did not pass: {path}")
    if bool(report.get("session_e_accessed")) or bool(
        report.get("openbmi_s2_accessed")
    ):
        raise RuntimeError(f"development report accessed held-out data: {path}")
    return report


def _median_epoch(values: list[int]) -> int:
    if not values or any(int(value) < 1 for value in values):
        raise RuntimeError("fixed-epoch selection requires positive development epochs")
    return int(math.floor(float(np.median(values)) + 0.5))


def _epochs_from_campaign(
    campaign: Path,
    *,
    name: str,
    row_field: str,
    expected_folds: int = 6,
) -> tuple[int, dict[str, Any]]:
    rows = [row for row in _csv_rows(campaign / "summary.csv") if row[row_field] == name]
    if not rows:
        raise RuntimeError(f"no development rows for {row_field}={name!r}")
    epochs: list[int] = []
    runs: list[dict[str, Any]] = []
    for row in rows:
        subject, seed = int(row["subject"]), int(row["seed"])
        run = campaign / name / f"subject_{subject:02d}" / f"seed_{seed}"
        run_epochs: list[int] = []
        for fold in range(expected_folds):
            result = read_json(run / f"fold_{fold}" / "result.json")
            value = int(result["selected_outer_retrain_epoch"])
            run_epochs.append(value)
            epochs.append(value)
        runs.append({"subject": subject, "seed": seed, "epochs": run_epochs})
    return _median_epoch(epochs), {
        "rule": "round_half_up(median(all registered Session-T fold-selected epochs))",
        "values": len(epochs),
        "minimum": min(epochs),
        "maximum": max(epochs),
        "median": float(np.median(epochs)),
        "runs": runs,
    }


def _residual_epoch(e3: Path) -> tuple[int, dict[str, Any]]:
    rows = _csv_rows(e3 / "summary.csv")
    epochs: list[int] = []
    for row in rows:
        run = e3 / f"subject_{int(row['subject']):02d}" / f"seed_{int(row['seed'])}"
        for fold in range(6):
            epochs.append(
                int(read_json(run / f"fold_{fold}" / "result.json")["selected_residual_epoch"])
            )
    # Epoch zero is a valid registered choice for a residual that remains at the exact null.
    if not epochs or any(value < 0 for value in epochs):
        raise RuntimeError("invalid E3 residual-epoch evidence")
    selected = int(math.floor(float(np.median(epochs)) + 0.5))
    return selected, {
        "rule": "round_half_up(median(all registered Session-T residual epochs))",
        "values": len(epochs),
        "minimum": min(epochs),
        "maximum": max(epochs),
        "median": float(np.median(epochs)),
    }


def _hashes(paths: dict[str, Path]) -> dict[str, str]:
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
    parser.add_argument("--e5", required=True)
    parser.add_argument("--e5-audit", required=True)
    parser.add_argument("--selected-e2", required=True)
    parser.add_argument("--selected-e2-gate", required=True)
    parser.add_argument("--e4", required=True)
    parser.add_argument("--e4-gate", required=True)
    parser.add_argument("--e3", default="")
    parser.add_argument("--e3-gate", default="")
    parser.add_argument("--e3-sequence", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = ensure_dir(Path(args.output).resolve())
    p0 = Path(args.p0).resolve()
    e0 = Path(args.e0).resolve()
    e1 = Path(args.e1).resolve()
    e1_audit = Path(args.e1_audit).resolve()
    e5 = Path(args.e5).resolve()
    e5_audit = Path(args.e5_audit).resolve()
    e2 = Path(args.selected_e2).resolve()
    e2_gate_root = Path(args.selected_e2_gate).resolve()
    e4 = Path(args.e4).resolve()
    e4_gate_root = Path(args.e4_gate).resolve()
    e3 = Path(args.e3).resolve() if args.e3 else None
    e3_gate_root = Path(args.e3_gate).resolve() if args.e3_gate else None
    e3_sequence = Path(args.e3_sequence).resolve() if args.e3_sequence else None

    _require_report(p0 / "metrics.json")
    _require_report(e0 / "metrics.json")
    p0_fingerprint = read_json(p0 / "source_fingerprint.json")
    e0_fingerprint = read_json(e0 / "source_fingerprint.json")
    p0_reference = read_json(e0 / "p0_reference.json")
    if p0_reference.get("combined_sha256") != p0_fingerprint.get("combined_sha256"):
        raise RuntimeError("E0 is not bound to the supplied P0 preregistration")
    _require_report(e1_audit / "audit_report.json")
    _require_report(e5_audit / "audit_report.json")
    e2_gate = _require_report(e2_gate_root / "gate_decision.json", expected="completed")
    if not bool(e2_gate.get("passed")):
        raise RuntimeError("selected E2 backbone failed the preregistered accuracy gate")
    e4_gate = _require_report(e4_gate_root / "gate_decision.json", expected="completed")
    e4_status = _require_report(e4 / "campaign_status.json", expected="completed")
    if not bool(e4_status.get("full_registered_contract")):
        raise RuntimeError("E4 is not the complete registered four-arm campaign")

    selected_model_path = e5 / "selected_model.yaml"
    selected_e2_config_path = e5 / "selected_e2_config.yaml"
    selected_model = yaml.safe_load(selected_model_path.read_text(encoding="utf-8"))
    selected_e2_config = yaml.safe_load(
        selected_e2_config_path.read_text(encoding="utf-8")
    )
    e4_config = yaml.safe_load((e4 / "resolved_campaign.yaml").read_text(encoding="utf-8"))
    if e4_config.get("subjects") != [1, 3, 8] or e4_config.get("seeds") != [0, 1, 2]:
        raise RuntimeError("E4 development coverage differs from the freeze contract")
    for field in ("training", "preprocessing", "augmentation"):
        if e4_config.get(field) != selected_e2_config.get(field):
            raise RuntimeError(
                f"E4 {field} drifted from the E5-selected E2 contract"
            )

    snn_passed = bool(e4_gate.get("passed"))
    primary_variant = (
        str(e4_gate["gate"]["selected_variant"]) if snn_passed else "ann_residual"
    )
    variants = dict(e4_config["variants"])
    if primary_variant not in variants or "ann_residual" not in variants:
        raise RuntimeError("E4 gate selected an unregistered decoder")
    resolved_model = {**selected_model, **dict(variants[primary_variant])}
    ann_model = {**selected_model, **dict(variants["ann_residual"])}

    primary_epoch, primary_epoch_evidence = _epochs_from_campaign(
        e4, name=primary_variant, row_field="variant"
    )
    ann_epoch, ann_epoch_evidence = _epochs_from_campaign(
        e4, name="ann_residual", row_field="variant"
    )

    delay_gate: dict[str, Any] | None = None
    delay_enabled = False
    residual_epoch = 0
    residual_epoch_evidence: dict[str, Any] | None = None
    frozen_delay_config: dict[str, Any] | None = None
    selected_delay_stage: str | None = None
    e3_sequence_status: dict[str, Any] | None = None
    e3_cross_band_gate_passed = False
    if e3_sequence is not None:
        e3_sequence_status = _require_report(
            e3_sequence / "sequence_status.json", expected="completed"
        )
        if e3_sequence_status.get("stage") != "E3_SEQUENCE":
            raise RuntimeError("invalid E3 sequence status")
        selected_delay_stage = e3_sequence_status.get("selected_delay_stage")
        e3_cross_band_gate_passed = any(
            row.get("stage") == "static_slow_cross_band" and bool(row.get("passed"))
            for row in e3_sequence_status.get("results", [])
        )
        if bool(selected_delay_stage) != bool(e3 is not None and e3_gate_root is not None):
            raise RuntimeError(
                "E3 selected campaign/gate do not match the sequence promotion result"
            )
    if e3 is not None or e3_gate_root is not None:
        if e3 is None or e3_gate_root is None:
            raise RuntimeError("E3 campaign and gate must be supplied together")
        delay_gate = _require_report(
            e3_gate_root / "gate_decision.json", expected="completed"
        )
        selected_delay_stage = str(delay_gate.get("delay_stage", ""))
        if not selected_delay_stage:
            raise RuntimeError("E3 gate does not identify its delay stage")
        if e3_sequence_status is not None and selected_delay_stage != str(
            e3_sequence_status.get("selected_delay_stage")
        ):
            raise RuntimeError("E3 gate differs from the sequence-selected stage")
        if bool(delay_gate.get("passed")) and primary_variant == "ann_residual":
            delay_enabled = True
            residual_epoch, residual_epoch_evidence = _residual_epoch(e3)
            first_e3_row = _csv_rows(e3 / "summary.csv")[0]
            first_e3_run = (
                e3
                / f"subject_{int(first_e3_row['subject']):02d}"
                / f"seed_{int(first_e3_row['seed'])}"
                / "resolved_config.yaml"
            )
            resolved_e3 = yaml.safe_load(first_e3_run.read_text(encoding="utf-8"))
            frozen_delay_config = {
                "delay": dict(resolved_e3["delay"]),
                "training": dict(resolved_e3["training"]),
                "augmentation": dict(resolved_e3["augmentation"]),
                "selection": dict(resolved_e3["selection"]),
            }

    e1_status = _require_report(e1 / "campaign_status.json", expected="completed")
    baseline_selection = read_json(e1 / "confirmation_selection.json")
    baseline_models = list(e1_status["screened_models"])
    e1_source_tree = read_json(e1 / "source_tree_manifest.json")
    e1_config_path = ROOT / "configs" / "experiments" / "v8_e1_baselines.yaml"
    if e1_source_tree.get("configs/experiments/v8_e1_baselines.yaml") != file_sha256(
        e1_config_path
    ):
        raise RuntimeError("active E1 baseline config differs from the audited campaign")
    baseline_protocol_config = yaml.safe_load(e1_config_path.read_text(encoding="utf-8"))
    official_source_locks = read_json(e1 / "official_source_locks.json")
    baseline_epochs: dict[str, int] = {}
    baseline_epoch_evidence: dict[str, Any] = {}
    for model in baseline_models:
        epoch, evidence = _epochs_from_campaign(e1, name=str(model), row_field="model")
        baseline_epochs[str(model)] = epoch
        baseline_epoch_evidence[str(model)] = evidence

    source_tree = collect_source_tree_manifest(ROOT)
    source_digest = source_tree_digest(source_tree)
    if read_json(p0 / "source_tree_manifest.json") != source_tree:
        raise RuntimeError("P0 source tree differs from the architecture-freeze source tree")
    if (
        p0_fingerprint.get("components", {}).get("source_tree") != source_digest
        or e0_fingerprint.get("components", {}).get("source_tree") != source_digest
    ):
        raise RuntimeError("P0/E0 source fingerprints differ from the freeze source tree")
    write_json(output / "source_tree_manifest.json", source_tree)
    evidence_paths = {
        "p0_manifest": p0 / "manifest.json",
        "p0_metrics": p0 / "metrics.json",
        "p0_protocol_manifest": p0 / "protocol_manifest.json",
        "p0_source_fingerprint": p0 / "source_fingerprint.json",
        "e0_manifest": e0 / "manifest.json",
        "e0_metrics": e0 / "metrics.json",
        "e0_p0_reference": e0 / "p0_reference.json",
        "e0_source_fingerprint": e0 / "source_fingerprint.json",
        "e1_campaign_status": e1 / "campaign_status.json",
        "e1_audit": e1_audit / "audit_report.json",
        "e1_confirmation_selection": e1 / "confirmation_selection.json",
        "e1_source_tree_manifest": e1 / "source_tree_manifest.json",
        "e1_official_source_locks": e1 / "official_source_locks.json",
        "e5_campaign_status": e5 / "campaign_status.json",
        "e5_audit": e5_audit / "audit_report.json",
        "e5_selected_candidate": e5 / "selected_candidate.json",
        "e5_selected_model": selected_model_path,
        "e5_selected_e2_config": selected_e2_config_path,
        "selected_e2_campaign_status": e2 / "campaign_status.json",
        "selected_e2_gate": e2_gate_root / "gate_decision.json",
        "e4_campaign_status": e4 / "campaign_status.json",
        "e4_gate": e4_gate_root / "gate_decision.json",
    }
    if e3 is not None and e3_gate_root is not None:
        evidence_paths.update(
            {
                "e3_campaign_status": e3 / "campaign_status.json",
                "e3_gate": e3_gate_root / "gate_decision.json",
            }
        )
    if e3_sequence is not None and e3_sequence_status is not None:
        evidence_paths["e3_sequence_status"] = e3_sequence / "sequence_status.json"
        for row in e3_sequence_status.get("results", []):
            stage = str(row["stage"])
            campaign = Path(str(row["campaign"])).resolve()
            gate = Path(str(row["gate"])).resolve()
            if e3_sequence not in campaign.parents or e3_sequence not in gate.parents:
                raise RuntimeError("E3 sequence evidence escaped its registered root")
            evidence_paths[f"e3_{stage}_campaign_status"] = (
                campaign / "campaign_status.json"
            )
            evidence_paths[f"e3_{stage}_gate"] = gate / "gate_decision.json"
    e7_config_path = ROOT / "configs" / "experiments" / "v8_e7_utility.yaml"
    e9_config_path = ROOT / "configs" / "experiments" / "v8_e9_frozen_ablation.yaml"
    evidence_paths["e7_preregistered_config"] = e7_config_path
    evidence_paths["e9_preregistered_config"] = e9_config_path

    payload = {
        "freeze_scope": "pre_E6_architecture_and_analysis",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_tree_sha256": source_digest,
        "architecture": {
            "name": "v8_accuracy_first",
            "primary_variant": primary_variant,
            "model_config": resolved_model,
            "matched_ann_control_config": ann_model,
            "snn_gate_passed": snn_passed,
            "delay": {
                "enabled": delay_enabled,
                "mode": selected_delay_stage if delay_enabled else "off",
                "residual_epoch": residual_epoch,
                "gate_passed": bool(delay_gate and delay_gate.get("passed")),
                "config": frozen_delay_config,
                "exclusion_reason": (
                    None
                    if delay_enabled
                    else (
                        "E3 gate did not pass"
                        if not delay_gate or not delay_gate.get("passed")
                        else "E3 was validated on ANN; the selected SNN combination was not validated"
                    )
                ),
            },
        },
        "training": dict(e4_config["training"]),
        "preprocessing": dict(e4_config["preprocessing"]),
        "augmentation": dict(e4_config["augmentation"]),
        "checkpoint_rule": {
            "policy": "fixed epoch retrain on all Session T; evaluate Session E once",
            "selection_data": "BCI2a Session T development only",
            "final_epoch": primary_epoch,
            "scheduler_horizon": int(e4_config["selection"]["max_epochs"]),
            "heldout_checkpoint_selection": False,
            "epoch_evidence": primary_epoch_evidence,
            "matched_ann_final_epoch": ann_epoch,
            "matched_ann_epoch_evidence": ann_epoch_evidence,
            "delay_residual_epoch_evidence": residual_epoch_evidence,
        },
        "analysis_plan": {
            "primary_metric": "subject_macro_accuracy",
            "secondary_metrics": [
                "balanced_accuracy",
                "macro_f1",
                "cohen_kappa",
                "0.5_1_2_4_second_accuracy",
            ],
            "subject_is_inferential_unit": True,
            "seeds_are_paired_optimization_repeats": True,
            "post_E_tuning_allowed": False,
            "uncertainty": "subject-blocked bootstrap and paired subject summaries",
            "multiplicity": "Holm correction for secondary claim families",
            "e7_preregistered_config_sha256": file_sha256(e7_config_path),
            "e9_preregistered_config_sha256": file_sha256(e9_config_path),
        },
        "development_evidence": {
            "selected_e2_variant": e2_gate["v8_variant"],
            "selected_e2_gate_passed": True,
            "e4_gate_passed": snn_passed,
            "e4_selected_variant": primary_variant,
            "e3_gate": delay_gate,
            "e3_sequence": e3_sequence_status,
            "e3_cross_band_gate_passed": e3_cross_band_gate_passed,
            "e5_candidate": read_json(e5 / "selected_candidate.json")["candidate_id"],
            "heldout_metrics_used_for_selection": False,
        },
        "baselines": {
            "models": baseline_models,
            "strongest_development_model": baseline_selection["models"][0],
            "fixed_epochs": baseline_epochs,
            "epoch_evidence": baseline_epoch_evidence,
            "protocol_config": baseline_protocol_config,
            "official_source_locks": official_source_locks,
            "same_T_to_E_protocol_required": True,
        },
        "heldout_access": {
            "bci2a_session_e_accessed_before_freeze": False,
            "openbmi_session_s2_accessed_before_freeze": False,
            "status": "sealed_until_validated_freeze",
        },
        "artifact_hashes": _hashes(evidence_paths),
    }
    write_v8_freeze_manifest(output / "freeze_manifest.json", payload)
    freeze = read_json(output / "freeze_manifest.json")
    summary = {
        "status": "completed",
        "freeze_sha256": freeze["combined_sha256"],
        "source_tree_sha256": source_digest,
        "primary_variant": primary_variant,
        "final_epoch": primary_epoch,
        "delay_enabled": delay_enabled,
        "baseline_models": baseline_models,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    write_json(output / "freeze_summary.json", summary)
    write_run_artifact_manifest(output, required_files=FREEZE_FILES)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
