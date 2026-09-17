from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from dpc_snn.experiments.v62_protocol import write_run_artifact_manifest
from dpc_snn.experiments.v8_anchor import (
    load_e1_selection_probability,
    probability_from_logits,
)
from dpc_snn.experiments.v8_delay_hpo import (
    apply_delay_hpo_candidate,
    enumerate_delay_hpo_candidates,
    rank_delay_hpo_candidates,
)
from dpc_snn.experiments.v8_fusion import entropy_residual_probability
from dpc_snn.experiments.v8_protocol import file_sha256, sha256_fingerprint
from dpc_snn.utils.io import write_csv, write_json
from dpc_snn.utils.metrics import classification_metrics
from scripts.audit_v8_e5_matched_transport_hpo import audit_campaign
from scripts.run_v8_e5_matched_transport_hpo import (
    CANDIDATE_FILES,
    _candidate_dir,
    _candidate_seed,
)


def _config() -> dict:
    return yaml.safe_load(
        Path(
            "configs/experiments/v8_e5_matched_transport_hpo.yaml"
        ).read_text(encoding="utf-8")
    )


def test_delay_hpo_is_the_complete_balanced_24_candidate_factorial() -> None:
    candidates = enumerate_delay_hpo_candidates(_config())
    assert len(candidates) == 24
    assert len({row["candidate_id"] for row in candidates}) == 24
    assert {row["decoder_channels"] for row in candidates} == {16, 32}
    assert {row["decoder_layers"] for row in candidates} == {1, 2}
    assert {row["temporal_decimation"] for row in candidates} == {1, 2}
    assert {row["learning_rate"] for row in candidates} == {0.0005, 0.001, 0.002}
    combinations = {
        (
            row["decoder_channels"],
            row["decoder_layers"],
            row["temporal_decimation"],
            row["learning_rate"],
        )
        for row in candidates
    }
    assert len(combinations) == 24
    assert enumerate_delay_hpo_candidates(_config()) == candidates


def test_selected_candidate_changes_only_registered_expert_and_optimizer_fields() -> None:
    base = yaml.safe_load(
        Path(
            "configs/experiments/v8_e3_matched_transport_campaign.yaml"
        ).read_text(encoding="utf-8")
    )
    candidate = enumerate_delay_hpo_candidates(_config())[0]
    selected = apply_delay_hpo_candidate(base, candidate)
    assert selected["expert"]["decoder_channels"] == candidate["decoder_channels"]
    assert selected["expert"]["decoder_layers"] == candidate["decoder_layers"]
    assert selected["expert"]["temporal_decimation"] == candidate[
        "temporal_decimation"
    ]
    assert selected["training"]["learning_rate"] == candidate["learning_rate"]
    for field in ("delay", "controls", "residual_fusion", "gate", "data_access"):
        assert selected[field] == base[field]


def test_delay_hpo_ranking_uses_subject_macro_validation_metrics() -> None:
    rows = []
    for candidate, values in {
        "stable": {1: [0.6, 0.6], 3: [0.6, 0.6]},
        "imbalanced": {1: [1.0, 1.0], 3: [0.1, 0.1]},
    }.items():
        for subject, kappas in values.items():
            for fold, kappa in enumerate(kappas):
                rows.append(
                    {
                        "candidate_id": candidate,
                        "subject": subject,
                        "fold": fold,
                        "validation_kappa": kappa,
                        "validation_accuracy": kappa,
                        "validation_nll": 1.0 - kappa,
                        "parameters": 100,
                        "best_epoch": 2,
                    }
                )
    ranking = rank_delay_hpo_candidates(
        rows,
        candidate_ids=["stable", "imbalanced"],
        subjects=[1, 3],
        folds=[0, 1],
    )
    assert ranking[0]["candidate_id"] == "stable"
    assert ranking[0]["subject_macro_mean_validation_kappa"] == pytest.approx(0.6)

    with pytest.raises(ValueError, match="coverage differs"):
        rank_delay_hpo_candidates(
            rows[:-1],
            candidate_ids=["stable", "imbalanced"],
            subjects=[1, 3],
            folds=[0, 1],
        )


def test_hpo_selection_explicitly_forbids_full_and_outer_metrics() -> None:
    config = _config()
    assert config["delay_override_for_selection"] == "zero"
    assert config["selection"]["outer_test_signals_forbidden"] is True
    assert config["selection"]["outer_test_labels_forbidden"] is True
    assert config["selection"]["full_delay_metrics_forbidden"] is True
    assert config["output_contract"][
        "requires_fresh_formal_e3_outer_oof_after_selection"
    ] is True


def test_hpo_uses_common_random_numbers_across_candidates() -> None:
    first = _candidate_seed(1, 2, 0)
    second = _candidate_seed(1, 2, 0)
    assert first == second
    assert first != _candidate_seed(1, 3, 0)
    assert first != _candidate_seed(1, 2, 1)


def test_fold_local_anchor_loader_reorders_and_rejects_wrong_fold(
    tmp_path: Path,
) -> None:
    path = tmp_path / "selection_predictions.npz"
    logits = np.asarray([[0.0, 0.0, 3.0, 0.0], [3.0, 0.0, 0.0, 0.0]])
    np.savez_compressed(
        path,
        indices=np.asarray([2, 0]),
        logits=logits,
        labels=np.asarray([2, 0]),
    )
    probability = load_e1_selection_probability(
        path,
        expected_indices=np.asarray([0, 2]),
        expected_labels=np.asarray([0, 2]),
    )
    assert probability.argmax(axis=1).tolist() == [0, 2]
    with pytest.raises(RuntimeError, match="active inner fold"):
        load_e1_selection_probability(
            path,
            expected_indices=np.asarray([0, 1]),
            expected_labels=np.asarray([0, 1]),
        )


def test_independent_hpo_audit_recomputes_ranking_and_selected_config(
    tmp_path: Path,
) -> None:
    config = _config()
    config["subjects"] = [1]
    config["maximum_unique_configurations"] = 2
    config["candidate_factors"] = {
        "decoder_channels": {"c16": 16, "c32": 32},
        "decoder_layers": {"l1": 1},
        "temporal_decimation": {"half_rate": 2},
        "learning_rate": {"lr_10e4": 0.001},
    }
    config["successive_halving"] = {
        "stage_1": {
            "active_folds": [0],
            "max_epochs": 1,
            "minimum_epochs": 1,
            "patience": 1,
            "promote_top_k": 1,
        }
    }
    base = yaml.safe_load(
        Path(
            "configs/experiments/v8_e3_matched_transport_campaign.yaml"
        ).read_text(encoding="utf-8")
    )
    candidates = enumerate_delay_hpo_candidates(config)
    source_digest = "a" * 64
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    campaign_contract = {
        "source_tree_sha256": source_digest,
        "candidate_order": [row["candidate_id"] for row in candidates],
        "candidate_payload_sha256": sha256_fingerprint(candidates),
    }
    campaign_contract["combined_sha256"] = sha256_fingerprint(campaign_contract)
    write_json(campaign / "campaign_contract.json", campaign_contract)
    validation_indices = np.asarray([10, 11, 12, 13], dtype=np.int64)
    validation_labels = np.asarray([0, 1, 2, 3], dtype=np.int64)
    anchor_logits = np.asarray(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    atc_source = tmp_path / "atc_selection_predictions.npz"
    fbc_source = tmp_path / "fbc_selection_predictions.npz"
    for source in (atc_source, fbc_source):
        np.savez_compressed(
            source,
            indices=validation_indices,
            logits=anchor_logits,
            labels=validation_labels,
        )
    anchor_probability = probability_from_logits(anchor_logits)
    rows = []
    for index, candidate in enumerate(candidates):
        candidate_id = candidate["candidate_id"]
        directory = _candidate_dir(campaign, "stage_1", 1, 0, candidate_id)
        directory.mkdir(parents=True)
        contract = {
            "campaign_contract_sha256": campaign_contract["combined_sha256"],
            "stage": "stage_1",
            "stage_config": config["successive_halving"]["stage_1"],
            "candidate": candidate,
            "subject": 1,
            "fold": 0,
            "shared_input_sha256": {
                "inner_validation_indices": sha256_fingerprint(
                    validation_indices.tolist()
                ),
                "atc_selection_path": str(atc_source),
                "fbc_selection_path": str(fbc_source),
                "atc_selection_predictions": file_sha256(atc_source),
                "fbc_selection_predictions": file_sha256(fbc_source),
            },
            "outer_test_metrics_used": False,
            "full_delay_metrics_used": False,
        }
        contract["combined_sha256"] = sha256_fingerprint(contract)
        expert_logits = anchor_logits.copy()
        expert_logits[:, (index + 1) % 4] += float(index)
        expert_probability = probability_from_logits(expert_logits)
        fused_probability, residual_gate = entropy_residual_probability(
            anchor_probability,
            expert_probability,
            maximum_weight=float(base["residual_fusion"]["maximum_expert_weight"]),
        )
        fused = classification_metrics(
            validation_labels, fused_probability.argmax(axis=1), n_classes=4
        )
        expert = classification_metrics(
            validation_labels, expert_probability.argmax(axis=1), n_classes=4
        )
        validation_nll = float(
            -np.log(
                fused_probability[
                    np.arange(validation_labels.size), validation_labels
                ]
            ).mean()
        )
        metrics = {
            "status": "completed",
            "stage": "stage_1",
            "candidate_id": candidate_id,
            "candidate": candidate,
            "subject": 1,
            "fold": 0,
            "best_epoch": 1,
            "validation_kappa": float(fused["kappa"]),
            "validation_accuracy": float(fused["accuracy"]),
            "validation_nll": validation_nll,
            "validation_expert_kappa": float(expert["kappa"]),
            "validation_expert_accuracy": float(expert["accuracy"]),
            "parameters": 100 + index,
            "delay_override": "zero",
            "outer_test_metrics_used": False,
            "full_delay_metrics_used": False,
            "prior_stability_passed": True,
            "source_tree_sha256": source_digest,
            "session_e_accessed": False,
            "openbmi_s2_accessed": False,
        }
        write_json(directory / "candidate_contract.json", contract)
        write_json(directory / "metrics.json", metrics)
        write_csv(
            directory / "history.csv",
            [
                {
                    "epoch": 1,
                    "loss": 1.0,
                    "validation_kappa": metrics["validation_kappa"],
                    "validation_accuracy": metrics["validation_accuracy"],
                    "validation_nll": metrics["validation_nll"],
                }
            ],
        )
        np.savez_compressed(
            directory / "validation_predictions.npz",
            indices=validation_indices,
            labels=validation_labels,
            anchor_probability=anchor_probability,
            expert_logits=expert_logits,
            expert_probability=expert_probability,
            fused_probability=fused_probability,
            residual_gate=residual_gate,
        )
        torch.save({}, directory / "best.pt")
        torch.save({}, directory / "last.pt")
        write_run_artifact_manifest(directory, required_files=CANDIDATE_FILES)
        rows.append(metrics)
    ranking = rank_delay_hpo_candidates(
        rows,
        candidate_ids=[row["candidate_id"] for row in candidates],
        subjects=[1],
        folds=[0],
    )
    write_csv(campaign / "stage_1_ranking.csv", ranking)
    selected_id = ranking[0]["candidate_id"]
    write_json(
        campaign / "stage_1_promotion.json",
        {
            "input_candidates": [row["candidate_id"] for row in candidates],
            "promoted_candidates": [selected_id],
            "selection_uses_inner_validation_only": True,
            "full_delay_metrics_used": False,
            "outer_test_metrics_used": False,
            "session_e_accessed": False,
            "openbmi_s2_accessed": False,
        },
    )
    selected_candidate = next(
        row for row in candidates if row["candidate_id"] == selected_id
    )
    selected_config = apply_delay_hpo_candidate(base, selected_candidate)
    selected_config["experiment_id"] = "V8_E3_MATCHED_TRANSPORT_HPO_SELECTED"
    selected_config["hpo_selection"] = {
        "campaign_contract_sha256": campaign_contract["combined_sha256"],
        "candidate_id": selected_id,
        "candidate": selected_candidate,
        "selection_protocol": "Session-T inner-validation only",
        "requires_fresh_formal_outer_oof": True,
    }
    selected_path = campaign / "selected_e3_config.yaml"
    selected_path.write_text(
        yaml.safe_dump(selected_config, sort_keys=False), encoding="utf-8"
    )
    write_json(
        campaign / "selected_candidate.json",
        {
            "candidate_id": selected_id,
            "candidate": selected_candidate,
            "selected_e3_config_sha256": file_sha256(selected_path),
            "selection_uses_inner_validation_only": True,
            "full_delay_metrics_used": False,
            "outer_test_metrics_used": False,
            "session_e_accessed": False,
            "openbmi_s2_accessed": False,
        },
    )
    write_json(
        campaign / "campaign_status.json",
        {
            "status": "completed",
            "stage": "E5_MATCHED_TRANSPORT_HPO",
            "unique_candidates": 2,
            "selected_candidate": selected_id,
            "selection_uses_inner_validation_only": True,
            "requires_fresh_formal_e3_outer_oof": True,
            "session_e_accessed": False,
            "openbmi_s2_accessed": False,
        },
    )
    report = audit_campaign(
        campaign,
        config=config,
        base_config=base,
        source_digest=source_digest,
    )
    assert report["status"] == "passed"
    assert report["selected_candidate"] == selected_id
    assert report["audited_candidate_runs"] == 2

    promotion = write_json(
        campaign / "stage_1_promotion.json",
        {
            "input_candidates": [row["candidate_id"] for row in candidates],
            "promoted_candidates": [ranking[1]["candidate_id"]],
            "selection_uses_inner_validation_only": True,
            "full_delay_metrics_used": False,
            "outer_test_metrics_used": False,
            "session_e_accessed": False,
            "openbmi_s2_accessed": False,
        },
    )
    assert promotion.is_file()
    with pytest.raises(RuntimeError, match="promotion was not reproduced"):
        audit_campaign(
            campaign,
            config=config,
            base_config=base,
            source_digest=source_digest,
        )
