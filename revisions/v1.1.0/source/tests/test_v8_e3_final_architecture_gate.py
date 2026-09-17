from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from dpc_snn.experiments.v62_protocol import (
    file_sha256,
    sha256_fingerprint,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_evidence_stability import (
    evaluate_v8_evidence_seed_stability,
)
from dpc_snn.utils.io import save_npz, write_csv, write_json
from scripts.evaluate_v8_e3_final_architecture_gate import evaluate_final_gate
from scripts.evaluate_v8_e3_prior_feasibility_gate import evaluate_prior_feasibility


SOURCE_DIGEST = "a" * 64
REPRESENTATION = "v8_csd_unwhitened_var_innovations_fixed_anatomical_regions"


def _feasibility_config(subjects: list[int]) -> dict:
    return {
        "schema": "dpc-snn-v8-e3-delay-prior-feasibility-gate/v1",
        "protocol": "bci2a_session_t_inner_train_fold_local_delay_evidence",
        "expected": {
            "subjects": subjects,
            "folds": [0],
            "scope": "inner_training_fold_only",
            "producer_source_tree_sha256": SOURCE_DIGEST,
            "analytic_representation": REPRESENTATION,
            "ensemble_replicates": 5,
            "require_all_subject_folds_pass": True,
        },
        "stability_thresholds": {
            "minimum_replicates": 5,
            "minimum_pass_fraction": 0.8,
            "minimum_consensus_frequency": 0.8,
            "minimum_consensus_edges": 3,
            "minimum_median_edge_jaccard": 0.5,
            "minimum_edge_jaccard_floor": 0.25,
            "minimum_median_delay_correlation": 0.5,
        },
        "decision_policy": {
            "pass": "authorize_e5_hpo_and_formal_e3",
            "fail": "reject_delay_branch_and_use_predeclared_e4_fallback",
        },
    }


def _replicate(*, passed: bool) -> tuple[dict, dict[str, np.ndarray]]:
    accepted = np.zeros((3, 3), dtype=np.uint8)
    accepted[0, 1] = 1
    accepted[1, 2] = 1
    accepted[2, 0] = 1
    delay = np.zeros((3, 3), dtype=np.float32)
    delay[0, 1] = 1.0
    delay[1, 2] = 2.0
    delay[2, 0] = 3.0
    checks = {
        "natural_nonzero_edges": True,
        "split_half_delay": passed,
        "bootstrap_frequency": True,
        "time_reversal": True,
        "phase_surrogate": True,
    }
    return (
        {"evidence_checks": checks, "evidence_pipeline_passed": passed},
        {"signed_node_accepted": accepted, "signed_node_delay_mean": delay},
    )


def _write_prior(root: Path, *, subject: int, passed: bool) -> Path:
    directory = root / f"subject_{subject:02d}" / "fold_0" / "inner"
    directory.mkdir(parents=True)
    replicates = [_replicate(passed=passed) for _ in range(5)]
    summaries = [item[0] for item in replicates]
    arrays = [item[1] for item in replicates]
    report, pairwise = evaluate_v8_evidence_seed_stability(summaries, arrays)
    payload = {
        "schema": "dpc-snn-v8-fold-delay-prior/v1",
        "scope": "inner_training_fold_only",
        "trial_ids": [f"bci2a:A{subject:02d}:T:0:{index:03d}" for index in range(8)],
        "source_tree_sha256": SOURCE_DIGEST,
        "heldout_data_accessed": False,
    }
    fingerprint = sha256_fingerprint(payload)
    summary = {
        "schema": "dpc-snn-v8-fold-delay-prior-ensemble/v1",
        "analytic_representation": REPRESENTATION,
        "ensemble_replicates": 5,
        "replicate_summaries": summaries,
        "stability_gate": report,
        "pairwise_stability": pairwise,
        "classifier_training": False,
        "heldout_session_e_accessed": False,
        "input_fingerprint": fingerprint,
        "input_payload": payload,
    }
    write_json(directory / "fingerprint.json", {**payload, "combined_sha256": fingerprint})
    archive = {
        f"replicate_{index}__{name}": value
        for index, values in enumerate(arrays)
        for name, value in values.items()
    }
    if passed:
        save_npz(directory / "prior.npz", route=np.ones((1,), dtype=np.float32))
        save_npz(directory / "evidence.npz", **archive)
        write_json(directory / "summary.json", summary)
        required = (
            "manifest.json",
            "fingerprint.json",
            "prior.npz",
            "evidence.npz",
            "summary.json",
        )
        status = "completed"
    else:
        summary["status"] = "rejected"
        summary["rejection_reason"] = "synthetic registered failure"
        save_npz(directory / "rejected_evidence.npz", **archive)
        write_json(directory / "rejected_summary.json", summary)
        required = (
            "manifest.json",
            "fingerprint.json",
            "rejected_evidence.npz",
            "rejected_summary.json",
        )
        status = "rejected"
    write_run_artifact_manifest(directory, required_files=required, status=status)
    return directory


def test_prior_feasibility_recomputes_acceptance_and_rejection(tmp_path: Path) -> None:
    root = tmp_path / "priors"
    _write_prior(root, subject=1, passed=True)
    _write_prior(root, subject=3, passed=False)
    rows, decision, inputs = evaluate_prior_feasibility(
        prior_root=root,
        config=_feasibility_config([1, 3]),
    )
    assert [bool(row["passed"]) for row in rows] == [True, False]
    assert decision["passed"] is False
    assert decision["classifier_training_authorized"] is False
    assert decision["failed_subject_folds"] == [{"subject": 3, "fold": 0}]
    assert len(inputs["prior_artifacts"]) == 2


def test_prior_feasibility_rejects_tampered_stability_claim(tmp_path: Path) -> None:
    root = tmp_path / "priors"
    directory = _write_prior(root, subject=1, passed=True)
    summary = yaml.safe_load((directory / "summary.json").read_text(encoding="utf-8"))
    summary["stability_gate"]["consensus_edge_count"] = 99
    write_json(directory / "summary.json", summary)
    write_run_artifact_manifest(
        directory,
        required_files=(
            "manifest.json",
            "fingerprint.json",
            "prior.npz",
            "evidence.npz",
            "summary.json",
        ),
    )
    with pytest.raises(RuntimeError, match="stability field differs"):
        evaluate_prior_feasibility(
            prior_root=root,
            config=_feasibility_config([1]),
        )


def _write_feasibility_gate(root: Path, *, passed: bool) -> Path:
    root.mkdir(parents=True)
    write_json(
        root / "feasibility_decision.json",
        {
            "status": "completed",
            "stage": "E3_DELAY_PRIOR_FEASIBILITY_GATE",
            "passed": passed,
            "registered_subject_folds": 1,
            "passing_subject_folds": int(passed),
            "metrics_recomputed_from_replicate_arrays": True,
            "threshold_relaxation_after_observation": False,
            "classifier_training_authorized": passed,
            "session_e_accessed": False,
            "openbmi_s2_accessed": False,
        },
    )
    write_csv(
        root / "per_subject_fold.csv",
        [{"subject": 1, "fold": 0, "passed": passed, "prior_status": "accepted" if passed else "rejected"}],
    )
    write_json(root / "input_manifest.json", {"synthetic": True})
    (root / "resolved_config.yaml").write_text("synthetic: true\n", encoding="utf-8")
    write_json(root / "source_tree_manifest.json", {"files": []})
    write_run_artifact_manifest(
        root,
        required_files=(
            "manifest.json",
            "feasibility_decision.json",
            "per_subject_fold.csv",
            "input_manifest.json",
            "resolved_config.yaml",
            "source_tree_manifest.json",
        ),
    )
    return root


def _write_e4_gate(path: Path) -> Path:
    row = {
        "subject": 1,
        "seed": 1,
        "snn_accuracy": 0.80,
        "matched_ann_accuracy": 0.79,
        "anchor_accuracy": 0.79,
        "snn_delta_vs_anchor_pp": 1.0,
        "snn_delta_vs_matched_ann_pp": 1.0,
        "snn_mean_firing_rate": 0.01,
        "positive_vs_anchor": True,
        "passes_anchor_floor": True,
        "passes_firing_interval": True,
    }
    write_json(
        path,
        {
            "schema": "dpc-snn-v8-e4-entropy-residual-result/v1",
            "passed": True,
            "selected_snn_variant": "sew_clif",
            "matched_ann_control": "ann_sew",
            "confirmation_pairs": 1,
            "positive_subject_seed_pairs": 1,
            "macro_snn_accuracy": 0.80,
            "macro_matched_ann_accuracy": 0.79,
            "macro_anchor_accuracy": 0.79,
            "macro_snn_delta_vs_matched_ann_pp": 1.0,
            "macro_snn_delta_vs_anchor_pp": 1.0,
            "criteria": {
                "macro_snn_vs_matched_ann": True,
                "macro_snn_vs_anchor": True,
                "minimum_positive_pairs": True,
                "all_pairs_above_anchor_floor": True,
                "all_pairs_have_nondegenerate_firing": True,
            },
            "thresholds": {
                "subjects": [1],
                "seeds": [1],
                "folds": [0],
                "macro_snn_vs_matched_ann_minimum_delta_pp": -0.3,
                "macro_snn_vs_anchor_minimum_delta_pp": 0.0,
                "minimum_positive_subject_seed_pairs": 1,
                "per_pair_snn_vs_anchor_floor_pp": -1.0,
                "firing_rate_interval": [0.005, 0.3],
            },
            "pair_rows": [row],
            "session_e_accessed": False,
            "openbmi_s2_accessed": False,
        },
    )
    return path


def _final_config() -> dict:
    config = yaml.safe_load(
        Path("configs/experiments/v8_e3_final_architecture_gate.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["expected"] = {
        "subjects": [1],
        "seeds": [0],
        "folds": [0],
        "subject_seed_pairs": 1,
        "primary_variant": "sew_clif",
        "matched_ann_variant": "ann_sew",
    }
    config["full_delay_branch"] = {
        **config["full_delay_branch"],
        "minimum_nonnegative_pairs_vs_anchor": 1,
    }
    config["zero_delay_branch"] = {
        **config["zero_delay_branch"],
        "minimum_nonnegative_pairs_vs_anchor": 1,
    }
    return config


def test_final_gate_fail_closes_to_confirmed_e4(tmp_path: Path) -> None:
    feasibility = _write_feasibility_gate(tmp_path / "feasibility", passed=False)
    e4 = _write_e4_gate(tmp_path / "e4.json")
    rows, decision = evaluate_final_gate(
        feasibility_gate=feasibility,
        e4_gate_path=e4,
        config=_final_config(),
    )
    assert rows[0]["diagnostic_scope"] == "prior_feasibility"
    assert decision["selected_branch"] == "confirmed_e4_sequence_residual"
    assert decision["e3_delay_gate_evaluated"] is False


def _write_formal_e3(tmp_path: Path) -> tuple[Path, Path]:
    campaign = tmp_path / "campaign"
    metric_dir = campaign / "subject_01" / "seed_0" / "fold_0" / "sew_clif"
    metric_dir.mkdir(parents=True)
    write_json(
        metric_dir / "metrics.json",
        {
            "variant": "sew_clif",
            "subject": 1,
            "seed": 0,
            "fold": 0,
            "full_firing_rate": 0.01,
            "session_e_accessed": False,
        },
    )
    write_json(
        campaign / "campaign_status.json",
        {
            "status": "completed",
            "stage": "E3_DELAY_RESIDUAL_CAMPAIGN",
            "protocol": "bci2a_session_t_nested_six_fold_oof",
            "source_tree_sha256": SOURCE_DIGEST,
            "subjects": [1],
            "seeds": [0],
            "folds": [0],
            "variants": ["ann_sew", "sew_clif"],
            "fold_runs": 1,
            "full_registered_contract": True,
            "canary": False,
            "session_e_accessed": False,
            "openbmi_s2_accessed": False,
        },
    )
    write_json(
        campaign / "campaign_contract.json",
        {"combined_sha256": "b" * 64, "config_sha256": "c" * 64},
    )
    write_run_artifact_manifest(
        campaign,
        required_files=(
            "manifest.json",
            "campaign_status.json",
            "campaign_contract.json",
            "subject_01/seed_0/fold_0/sew_clif/metrics.json",
        ),
    )

    gate = tmp_path / "e3_gate"
    gate.mkdir()
    write_json(
        gate / "gate_decision.json",
        {
            "status": "completed",
            "stage": "E3_MATCHED_TRANSPORT_GATE",
            "protocol": "bci2a_session_t_nested_six_fold_oof",
            "passed": True,
            "primary_variant": "sew_clif",
            "matched_ann_variant": "ann_sew",
            "fold_manifest_and_hash_validation": True,
            "trial_metrics_recomputed": True,
            "exact_oof_trials_per_pair": 288,
            "source_tree_sha256": SOURCE_DIGEST,
            "session_e_accessed": False,
            "openbmi_s2_accessed": False,
        },
    )
    write_json(
        gate / "campaign_manifest.json",
        {
            "manifest_sha256": file_sha256(campaign / "manifest.json"),
            "campaign_contract_sha256": "b" * 64,
            "config_sha256": "c" * 64,
        },
    )
    write_csv(
        gate / "per_subject_seed.csv",
        [
            {
                "variant": "ann_sew",
                "subject": 1,
                "seed": 0,
                "trials": 288,
                "anchor_accuracy": 0.80,
                "full_accuracy": 0.821,
                "matched_zero_accuracy": 0.812,
            },
            {
                "variant": "sew_clif",
                "subject": 1,
                "seed": 0,
                "trials": 288,
                "anchor_accuracy": 0.80,
                "full_accuracy": 0.82,
                "matched_zero_accuracy": 0.81,
            },
        ],
    )
    write_run_artifact_manifest(
        gate,
        required_files=(
            "manifest.json",
            "gate_decision.json",
            "campaign_manifest.json",
            "per_subject_seed.csv",
        ),
    )
    return campaign, gate


def test_final_gate_selects_full_only_after_all_compound_checks(tmp_path: Path) -> None:
    feasibility = _write_feasibility_gate(tmp_path / "feasibility", passed=True)
    e4 = _write_e4_gate(tmp_path / "e4.json")
    campaign, e3_gate = _write_formal_e3(tmp_path)
    _, decision = evaluate_final_gate(
        feasibility_gate=feasibility,
        campaign=campaign,
        e3_gate=e3_gate,
        e4_gate_path=e4,
        config=_final_config(),
    )
    assert decision["selected_branch"] == "full_delay_routed_snn"
    assert decision["full_delay_branch_passed"] is True


def test_final_gate_rejects_duplicate_e3_identity(tmp_path: Path) -> None:
    feasibility = _write_feasibility_gate(tmp_path / "feasibility", passed=True)
    e4 = _write_e4_gate(tmp_path / "e4.json")
    campaign, e3_gate = _write_formal_e3(tmp_path)
    with (e3_gate / "per_subject_seed.csv").open("a", encoding="utf-8") as handle:
        handle.write("ann_sew,1,0,288,0.8,0.821,0.812\n")
    write_run_artifact_manifest(
        e3_gate,
        required_files=(
            "manifest.json",
            "gate_decision.json",
            "campaign_manifest.json",
            "per_subject_seed.csv",
        ),
    )
    with pytest.raises(RuntimeError, match="duplicate, missing, or extra"):
        evaluate_final_gate(
            feasibility_gate=feasibility,
            campaign=campaign,
            e3_gate=e3_gate,
            e4_gate_path=e4,
            config=_final_config(),
        )
