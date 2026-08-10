#!/usr/bin/env python3
"""Independently audit the bounded V8 E5 matched-transport HPO campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_anchor import (  # noqa: E402
    load_fused_e1_selection_anchor,
    probability_from_logits,
)
from dpc_snn.experiments.v8_delay_hpo import (  # noqa: E402
    apply_delay_hpo_candidate,
    enumerate_delay_hpo_candidates,
    rank_delay_hpo_candidates,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    file_sha256,
    sha256_fingerprint,
    source_tree_digest,
)
from dpc_snn.experiments.v8_fusion import entropy_residual_probability  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from scripts.run_v8_e5_matched_transport_hpo import (  # noqa: E402
    CANDIDATE_FILES,
    CONFIG_SCHEMA,
    _candidate_dir,
)


AUDIT_FILES = (
    "manifest.json",
    "audit_report.json",
    "campaign_manifest.json",
    "source_tree_manifest.json",
)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty E5 audit CSV: {path}")
    return rows


def _assert_ranking_matches(
    expected: Sequence[Mapping[str, Any]], stored: Sequence[Mapping[str, Any]]
) -> None:
    if len(expected) != len(stored):
        raise RuntimeError("stored E5 ranking has the wrong length")
    numeric = (
        "subject_macro_mean_validation_kappa",
        "subject_macro_mean_validation_accuracy",
        "subject_macro_mean_validation_nll",
        "parameters",
        "median_best_epoch",
        "subjects",
        "folds_per_subject",
        "validation_rows",
        "rank",
    )
    for expected_row, stored_row in zip(expected, stored, strict=True):
        if str(stored_row["candidate_id"]) != str(expected_row["candidate_id"]):
            raise RuntimeError("stored E5 candidate ranking order is incorrect")
        for field in numeric:
            if not math.isclose(
                float(stored_row[field]),
                float(expected_row[field]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(f"stored E5 ranking differs at {field}")


def _audit_validation_predictions(
    directory: Path,
    *,
    contract: Mapping[str, Any],
    metrics: Mapping[str, Any],
    maximum_weight: float,
) -> None:
    path = directory / "validation_predictions.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    required = {
        "indices",
        "labels",
        "anchor_probability",
        "expert_logits",
        "expert_probability",
        "fused_probability",
        "residual_gate",
    }
    if set(values) != required:
        raise RuntimeError(f"invalid E5 validation prediction schema: {path}")
    indices = np.asarray(values["indices"], dtype=np.int64)
    labels = np.asarray(values["labels"], dtype=np.int64)
    anchor = np.asarray(values["anchor_probability"], dtype=np.float32)
    logits = np.asarray(values["expert_logits"], dtype=np.float32)
    expert = np.asarray(values["expert_probability"], dtype=np.float32)
    fused = np.asarray(values["fused_probability"], dtype=np.float32)
    gate = np.asarray(values["residual_gate"], dtype=np.float32)
    if (
        indices.ndim != 1
        or indices.size == 0
        or np.unique(indices).size != indices.size
        or labels.shape != indices.shape
        or anchor.shape != (indices.size, 4)
        or logits.shape != anchor.shape
        or expert.shape != anchor.shape
        or fused.shape != anchor.shape
        or gate.shape != indices.shape
        or np.any(labels < 0)
        or np.any(labels >= 4)
        or not all(
            np.isfinite(value).all()
            for value in (anchor, logits, expert, fused, gate)
        )
    ):
        raise RuntimeError(f"invalid E5 validation prediction values: {path}")
    shared = contract.get("shared_input_sha256", {})
    if sha256_fingerprint(indices.tolist()) != shared.get(
        "inner_validation_indices"
    ):
        raise RuntimeError("E5 validation indices differ from the candidate contract")
    atc_path = Path(str(shared.get("atc_selection_path", "")))
    fbc_path = Path(str(shared.get("fbc_selection_path", "")))
    if (
        not atc_path.is_file()
        or not fbc_path.is_file()
        or file_sha256(atc_path) != shared.get("atc_selection_predictions")
        or file_sha256(fbc_path) != shared.get("fbc_selection_predictions")
    ):
        raise RuntimeError("E5 fold-local anchor source is missing or changed")
    source_anchor = load_fused_e1_selection_anchor(
        atc_path,
        fbc_path,
        expected_indices=indices,
        expected_labels=labels,
    )
    if not np.allclose(anchor, source_anchor, rtol=0.0, atol=1e-7):
        raise RuntimeError("E5 stored anchor is not the fold-local E1 selection anchor")
    recomputed_expert = probability_from_logits(logits)
    recomputed_fused, recomputed_gate = entropy_residual_probability(
        anchor,
        recomputed_expert,
        maximum_weight=float(maximum_weight),
    )
    if not np.allclose(expert, recomputed_expert, rtol=0.0, atol=1e-7):
        raise RuntimeError("E5 expert probabilities do not reproduce from logits")
    if not np.allclose(fused, recomputed_fused, rtol=0.0, atol=1e-7) or not np.allclose(
        gate, recomputed_gate, rtol=0.0, atol=1e-7
    ):
        raise RuntimeError("E5 fused validation probabilities do not reproduce")
    fused_metrics = classification_metrics(labels, fused.argmax(axis=1), n_classes=4)
    expert_metrics = classification_metrics(labels, expert.argmax(axis=1), n_classes=4)
    nll = float(
        -np.log(
            np.clip(fused[np.arange(labels.size), labels], 1e-12, 1.0)
        ).mean()
    )
    checks = {
        "validation_kappa": float(fused_metrics["kappa"]),
        "validation_accuracy": float(fused_metrics["accuracy"]),
        "validation_nll": nll,
        "validation_expert_kappa": float(expert_metrics["kappa"]),
        "validation_expert_accuracy": float(expert_metrics["accuracy"]),
    }
    for field, expected in checks.items():
        if not math.isclose(
            float(metrics[field]), expected, rel_tol=0.0, abs_tol=1e-7
        ):
            raise RuntimeError(f"E5 validation prediction audit differs at {field}")
    history = _csv_rows(directory / "history.csv")
    best_epoch = int(metrics["best_epoch"])
    matches = [row for row in history if int(row["epoch"]) == best_epoch]
    if len(matches) != 1:
        raise RuntimeError("E5 best epoch is absent or duplicated in history")
    for field in ("validation_kappa", "validation_accuracy", "validation_nll"):
        if not math.isclose(
            float(matches[0][field]), float(metrics[field]), rel_tol=0.0, abs_tol=1e-7
        ):
            raise RuntimeError(f"E5 best history row differs at {field}")


def audit_campaign(
    campaign: Path,
    *,
    config: Mapping[str, Any],
    base_config: Mapping[str, Any],
    source_digest: str,
) -> dict[str, Any]:
    candidates = enumerate_delay_hpo_candidates(config)
    by_id = {str(row["candidate_id"]): dict(row) for row in candidates}
    subjects = [int(value) for value in config["subjects"]]
    campaign_contract = read_json(campaign / "campaign_contract.json")
    campaign_payload = {
        key: value
        for key, value in campaign_contract.items()
        if key != "combined_sha256"
    }
    if campaign_contract.get("combined_sha256") != sha256_fingerprint(
        campaign_payload
    ):
        raise RuntimeError("E5 campaign contract fingerprint is invalid")
    if campaign_contract.get("source_tree_sha256") != source_digest:
        raise RuntimeError("E5 campaign source digest differs from the audit source")
    if campaign_contract.get("candidate_order") != list(by_id):
        raise RuntimeError("E5 campaign candidate order differs from the registered factorial")
    if campaign_contract.get("candidate_payload_sha256") != sha256_fingerprint(candidates):
        raise RuntimeError("E5 campaign candidate payload changed")

    active_ids = list(by_id)
    audited_candidate_runs = 0
    stage_reports = []
    for stage, stage_config in config["successive_halving"].items():
        folds = [int(value) for value in stage_config["active_folds"]]
        rows = []
        for subject in subjects:
            for fold in folds:
                for candidate_id in active_ids:
                    directory = _candidate_dir(
                        campaign, stage, subject, fold, candidate_id
                    )
                    validate_run_artifact_manifest(
                        directory,
                        required_files=CANDIDATE_FILES,
                        verify_hashes=True,
                        verify_prediction_schema=False,
                    )
                    contract = read_json(directory / "candidate_contract.json")
                    metrics = read_json(directory / "metrics.json")
                    expected_identity = {
                        "stage": stage,
                        "candidate_id": candidate_id,
                        "subject": subject,
                        "fold": fold,
                    }
                    for field, value in expected_identity.items():
                        if metrics.get(field) != value:
                            raise RuntimeError(
                                f"E5 candidate metric identity differs at {directory}: {field}"
                            )
                    if contract.get("campaign_contract_sha256") != campaign_contract.get(
                        "combined_sha256"
                    ):
                        raise RuntimeError("E5 candidate is not bound to the campaign")
                    contract_payload = {
                        key: value
                        for key, value in contract.items()
                        if key != "combined_sha256"
                    }
                    if contract.get("combined_sha256") != sha256_fingerprint(
                        contract_payload
                    ):
                        raise RuntimeError("E5 candidate contract fingerprint is invalid")
                    if (
                        contract.get("stage") != stage
                        or contract.get("stage_config") != dict(stage_config)
                        or int(contract.get("subject", -1)) != subject
                        or int(contract.get("fold", -1)) != fold
                    ):
                        raise RuntimeError("E5 candidate fold contract is invalid")
                    if contract.get("candidate") != by_id[candidate_id]:
                        raise RuntimeError("E5 candidate contract differs from the ledger")
                    for payload in (contract, metrics):
                        if bool(payload.get("outer_test_metrics_used")) or bool(
                            payload.get("full_delay_metrics_used")
                        ):
                            raise RuntimeError("E5 selection consumed a forbidden metric")
                    if (
                        metrics.get("delay_override") != "zero"
                        or metrics.get("source_tree_sha256") != source_digest
                        or metrics.get("prior_stability_passed") is not True
                        or metrics.get("session_e_accessed") is not False
                        or metrics.get("openbmi_s2_accessed") is not False
                    ):
                        raise RuntimeError("E5 candidate scientific contract is invalid")
                    _audit_validation_predictions(
                        directory,
                        contract=contract,
                        metrics=metrics,
                        maximum_weight=float(
                            base_config["residual_fusion"]["maximum_expert_weight"]
                        ),
                    )
                    rows.append(metrics)
                    audited_candidate_runs += 1
        expected_ranking = rank_delay_hpo_candidates(
            rows,
            candidate_ids=active_ids,
            subjects=subjects,
            folds=folds,
        )
        _assert_ranking_matches(
            expected_ranking, _csv_rows(campaign / f"{stage}_ranking.csv")
        )
        promotion = read_json(campaign / f"{stage}_promotion.json")
        expected_promoted = [
            str(row["candidate_id"])
            for row in expected_ranking[: int(stage_config["promote_top_k"])]
        ]
        if (
            promotion.get("input_candidates") != active_ids
            or promotion.get("promoted_candidates") != expected_promoted
            or promotion.get("selection_uses_inner_validation_only") is not True
            or promotion.get("full_delay_metrics_used") is not False
            or promotion.get("outer_test_metrics_used") is not False
            or promotion.get("session_e_accessed") is not False
            or promotion.get("openbmi_s2_accessed") is not False
        ):
            raise RuntimeError(f"E5 promotion was not reproduced for {stage}")
        stage_reports.append(
            {
                "stage": stage,
                "input_candidates": len(active_ids),
                "folds": folds,
                "candidate_runs": len(rows),
                "promoted_candidates": expected_promoted,
            }
        )
        active_ids = expected_promoted

    if len(active_ids) != 1:
        raise RuntimeError("E5 audit did not reproduce one final candidate")
    selected_id = active_ids[0]
    selected = read_json(campaign / "selected_candidate.json")
    if (
        selected.get("candidate_id") != selected_id
        or selected.get("candidate") != by_id[selected_id]
        or selected.get("selection_uses_inner_validation_only") is not True
        or selected.get("full_delay_metrics_used") is not False
        or selected.get("outer_test_metrics_used") is not False
        or selected.get("session_e_accessed") is not False
        or selected.get("openbmi_s2_accessed") is not False
    ):
        raise RuntimeError("E5 selected-candidate record is invalid")
    selected_path = campaign / "selected_e3_config.yaml"
    if selected.get("selected_e3_config_sha256") != file_sha256(selected_path):
        raise RuntimeError("E5 selected config hash differs from the selected record")
    selected_config = yaml.safe_load(selected_path.read_text(encoding="utf-8"))
    expected_config = apply_delay_hpo_candidate(base_config, by_id[selected_id])
    expected_config["experiment_id"] = "V8_E3_MATCHED_TRANSPORT_HPO_SELECTED"
    expected_config["hpo_selection"] = {
        "campaign_contract_sha256": campaign_contract["combined_sha256"],
        "candidate_id": selected_id,
        "candidate": by_id[selected_id],
        "selection_protocol": "Session-T inner-validation only",
        "requires_fresh_formal_outer_oof": True,
    }
    if selected_config != expected_config:
        raise RuntimeError("E5 selected E3 config is not the exact winning candidate")
    status = read_json(campaign / "campaign_status.json")
    if (
        status.get("status") != "completed"
        or status.get("stage") != "E5_MATCHED_TRANSPORT_HPO"
        or int(status.get("unique_candidates", -1)) != len(candidates)
        or status.get("selected_candidate") != selected_id
        or status.get("selection_uses_inner_validation_only") is not True
        or status.get("requires_fresh_formal_e3_outer_oof") is not True
        or status.get("session_e_accessed") is not False
        or status.get("openbmi_s2_accessed") is not False
    ):
        raise RuntimeError("E5 campaign status is invalid")
    return {
        "status": "passed",
        "stage": "E5_MATCHED_TRANSPORT_HPO_AUDIT",
        "source_tree_sha256": source_digest,
        "unique_candidates": len(candidates),
        "audited_candidate_runs": audited_candidate_runs,
        "stages": stage_reports,
        "selected_candidate": selected_id,
        "selected_e3_config_sha256": file_sha256(selected_path),
        "rankings_recomputed": True,
        "candidate_manifests_and_hashes_verified": True,
        "selection_uses_inner_validation_only": True,
        "full_delay_metrics_used": False,
        "outer_test_metrics_used": False,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config",
        default="configs/experiments/v8_e5_matched_transport_hpo.yaml",
    )
    args = parser.parse_args()

    campaign = Path(args.campaign).resolve()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA:
        raise RuntimeError("E5 audit config schema changed")
    base_config_path = (ROOT / str(config["fixed_contract"]["base_config"])).resolve()
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    source_tree = collect_source_tree_manifest(ROOT)
    source_digest = source_tree_digest(source_tree)
    campaign_manifest = read_json(campaign / "manifest.json")
    validate_run_artifact_manifest(
        campaign,
        required_files=tuple(campaign_manifest["required_files"]),
        verify_hashes=True,
        verify_prediction_schema=False,
    )
    campaign_contract = read_json(campaign / "campaign_contract.json")
    if (
        campaign_contract.get("config_sha256") != file_sha256(config_path)
        or campaign_contract.get("base_e3_config_sha256")
        != file_sha256(base_config_path)
    ):
        raise RuntimeError("E5 audit inputs differ from the campaign")
    report = audit_campaign(
        campaign,
        config=config,
        base_config=base_config,
        source_digest=source_digest,
    )
    output = ensure_dir(Path(args.output).resolve())
    write_json(output / "audit_report.json", report)
    write_json(output / "source_tree_manifest.json", source_tree)
    write_json(
        output / "campaign_manifest.json",
        {
            "path": str(campaign),
            "manifest_sha256": file_sha256(campaign / "manifest.json"),
            "campaign_contract_sha256": campaign_contract["combined_sha256"],
        },
    )
    write_run_artifact_manifest(output, required_files=AUDIT_FILES)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
