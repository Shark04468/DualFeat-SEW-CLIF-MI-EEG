from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from dpc_snn.experiments.v8_ensemble import (
    entropy_residual_prediction,
    equal_probability_anchor,
    softmax_probability,
    state_digest,
    validate_ensemble_freeze_contract,
)
from dpc_snn.experiments.v62_protocol import sha256_fingerprint
from dpc_snn.experiments.v8_protocol import build_v8_freeze_manifest, mapping_sha256
from dpc_snn.experiments.v8_maintenance import (
    build_ensemble_maintenance_manifest,
    validate_ensemble_maintenance_manifest,
)
from scripts.audit_v8_e6_ensemble import (
    _checkpoint_digest,
    _validate_campaign_coverage,
)
from scripts.freeze_v8_ensemble_architecture import _development_epochs
from scripts.run_v8_e6_ensemble_frozen import (
    PREDICTION_BASES,
    _log_probability,
    _metadata,
    _cpu_state,
    _session_data_hashes,
    _state_component_digests,
)


def _freeze_payload() -> dict[str, object]:
    components = {
        "selection_data": "BCI2a Session T development only",
        "atcnet_final_epoch": 79,
        "atcnet_scheduler_horizon": 300,
        "fbcnet_final_epoch": 70,
        "fbcnet_scheduler_horizon": 300,
        "decoder_final_epoch": 20,
        "decoder_scheduler_horizon": 120,
        "heldout_checkpoint_selection": False,
    }
    model = {
        "architecture_id": "v8_atc_fbc_entropy_residual_sequence_ensemble_r1",
        "anchor": {
            "components": ["atcnet", "fbcnet"],
            "probability_weights": [0.5, 0.5],
        },
        "primary_decoder": "sew_clif",
        "matched_ann_decoder": "ann_sew",
        "residual_fusion": {
            "mode": "anchor_entropy",
            "maximum_decoder_weight": 0.05,
        },
    }
    return {
        "freeze_scope": "pre_E6_architecture_and_analysis",
        "created_at_utc": "2026-08-03T00:00:00+00:00",
        "source_tree_sha256": "a" * 64,
        "architecture": {
            "model_config": model,
            "matched_ann_control_config": {**model, "primary_decoder": "ann_sew"},
            "primary_variant": "sew_clif",
            "delay": {"enabled": False, "mode": "off"},
        },
        "training": {},
        "preprocessing": {},
        "augmentation": {},
        "checkpoint_rule": {
            "selection_data": "BCI2a Session T development only",
            "final_epoch": 20,
            "scheduler_horizon": 120,
            "matched_ann_final_epoch": 20,
            "heldout_checkpoint_selection": False,
            "components": components,
        },
        "analysis_plan": {
            "primary_metric": "subject_macro_accuracy",
            "subject_is_inferential_unit": True,
            "post_E_tuning_allowed": False,
        },
        "development_evidence": {},
        "baselines": {
            "models": ["atcnet", "fbcnet"],
            "official_source_locks": {},
        },
        "heldout_access": {
            "bci2a_session_e_accessed_before_freeze": False,
            "openbmi_session_s2_accessed_before_freeze": False,
            "status": "sealed_until_validated_freeze",
        },
        "artifact_hashes": {"e4_gate": "b" * 64},
    }


def test_ensemble_probability_flow_is_normalized_and_bounded() -> None:
    atc_logits = np.asarray([[2.0, 1.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]])
    fbc_logits = np.asarray([[1.0, 2.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    decoder_logits = np.asarray([[0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]])
    anchor = equal_probability_anchor(atc_logits, fbc_logits)
    result = entropy_residual_prediction(anchor, decoder_logits, maximum_weight=0.05)
    assert np.allclose(anchor.sum(axis=1), 1.0)
    assert np.allclose(result["probabilities"].sum(axis=1), 1.0)
    assert np.all((result["gate"] >= 0.0) & (result["gate"] <= 0.05))
    assert np.allclose(softmax_probability(_log_probability(anchor)), anchor, atol=1e-7)


def test_ensemble_state_digest_detects_parameter_change() -> None:
    model = torch.nn.Linear(3, 2)
    before = state_digest(model)
    components = _state_component_digests(model)
    assert set(components) == {"bias", "weight"}
    with torch.no_grad():
        model.weight[0, 0] += 1.0
    assert state_digest(model) != before
    assert _state_component_digests(model)["weight"] != components["weight"]
    saved = _cpu_state(model)
    assert all(value.device.type == "cpu" for value in saved.values())
    assert state_digest(model) == sha256_fingerprint(mapping_sha256(saved))


def test_ensemble_freeze_contract_rejects_delay_or_epoch_drift() -> None:
    freeze = build_v8_freeze_manifest(_freeze_payload())
    assert validate_ensemble_freeze_contract(freeze) == freeze
    delayed = json.loads(json.dumps(freeze))
    delayed["architecture"]["delay"] = {"enabled": True, "mode": "static"}
    with pytest.raises(RuntimeError, match="delay"):
        validate_ensemble_freeze_contract(delayed)
    changed = json.loads(json.dumps(freeze))
    changed["checkpoint_rule"]["components"]["atcnet_final_epoch"] = 80
    with pytest.raises(RuntimeError, match="checkpoint"):
        validate_ensemble_freeze_contract(changed)


def test_ensemble_freeze_yaml_preserves_delay_mode_as_string() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiments"
        / "v8_e6_ensemble_freeze.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["architecture"]["delay"] == {
        "enabled": False,
        "mode": "off",
        "status": "failed_development_prerequisite_retained_as_future_ablation_only",
    }


def test_session_data_hashes_keep_same_basename_distinct(tmp_path: Path) -> None:
    train = tmp_path / "session_t" / "A01.npz"
    evaluation = tmp_path / "session_e" / "A01.npz"
    train.parent.mkdir()
    evaluation.parent.mkdir()
    train.write_bytes(b"training")
    evaluation.write_bytes(b"evaluation")
    hashes = _session_data_hashes(train, evaluation)
    assert set(hashes) == {"session_t/A01.npz", "session_e/A01.npz"}
    assert hashes["session_t/A01.npz"] != hashes["session_e/A01.npz"]


def test_source_maintenance_manifest_rejects_scientific_drift() -> None:
    payload = {
        "created_at_utc": "2026-08-03T00:00:00+00:00",
        "parent_freeze_sha256": "a" * 64,
        "parent_source_tree_sha256": "b" * 64,
        "current_source_tree_sha256": "c" * 64,
        "source_delta": {
            "changed": [
                "scripts/audit_v8_e6_ensemble.py",
                "scripts/run_v8_e6_ensemble_frozen.py",
                "tests/test_v8_ensemble.py",
            ],
            "added": [
                "scripts/attest_v8_ensemble_source_maintenance.py",
                "src/dpc_snn/experiments/v8_maintenance.py",
            ],
            "removed": [],
        },
        "scientific_invariants": {
            "architecture_files_unchanged": True,
            "model_math_unchanged": True,
            "training_budget_unchanged": True,
            "preprocessing_unchanged": True,
            "analysis_gate_unchanged": True,
            "maintenance_scope": "data_identity_fingerprint_and_access_audit_only",
        },
        "heldout_access": {
            "architecture_frozen_before_session_e_materialization": True,
            "session_e_materialized_after_parent_freeze": True,
            "session_e_use_before_attestation": "deterministic_preprocessing_and_exact_equivalence_audit_only",
            "session_e_predictions_or_metrics_computed": False,
            "session_e_used_for_model_or_checkpoint_selection": False,
            "openbmi_s2_accessed": False,
        },
        "artifact_hashes": {"parent_freeze_manifest": "d" * 64},
    }
    manifest = build_ensemble_maintenance_manifest(payload)
    assert validate_ensemble_maintenance_manifest(
        manifest,
        expected_parent_freeze_sha256="a" * 64,
        expected_current_source_tree_sha256="c" * 64,
    ) == manifest
    drifted = json.loads(json.dumps(manifest))
    drifted["scientific_invariants"]["model_math_unchanged"] = False
    with pytest.raises(RuntimeError, match="scientific invariant"):
        validate_ensemble_maintenance_manifest(drifted)


def test_e6_audit_distinguishes_canary_and_formal_coverage() -> None:
    canary = {
        "status": "completed",
        "full_registered_contract": False,
        "runs": 1,
        "subjects": [1],
        "seeds": [0],
    }
    assert _validate_campaign_coverage(canary, canary=True) == ([1], [0], 1)
    with pytest.raises(RuntimeError, match="coverage"):
        _validate_campaign_coverage(canary, canary=False)


def test_e6_checkpoint_digest_matches_active_model(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 3)
    checkpoint = tmp_path / "model.pt"
    torch.save(_cpu_state(model), checkpoint)
    assert _checkpoint_digest(checkpoint) == state_digest(model)


def _write_json(path: Path, payload: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_development_epoch_rule_reconstructs_all_54_folds(tmp_path: Path) -> None:
    e1, fbc, e4 = tmp_path / "e1", tmp_path / "fbc", tmp_path / "e4"
    index = 0
    for subject in (1, 3, 8):
        for seed in (0, 1, 2):
            for fold in range(6):
                atc_epoch = 79 if index < 28 else 80
                fbc_epoch = 69 if index < 27 else 70
                decoder_epoch = 20 if index < 40 else 21
                _write_json(
                    e1
                    / "atcnet"
                    / f"subject_{subject:02d}"
                    / f"seed_{seed}"
                    / f"fold_{fold}"
                    / "result.json",
                    {"selected_outer_retrain_epoch": atc_epoch},
                )
                _write_json(
                    fbc
                    / "fbcnet"
                    / f"subject_{subject:02d}"
                    / f"seed_{seed}"
                    / f"fold_{fold}"
                    / "result.json",
                    {"selected_outer_retrain_epoch": fbc_epoch},
                )
                fold_root = e4 / f"formal_s{subject}_seed{seed}_fold{fold}"
                _write_json(
                    fold_root / "sew_clif" / "metrics.json",
                    {"selected_epoch": decoder_epoch},
                )
                _write_json(
                    fold_root / "ann_sew" / "metrics.json",
                    {"selected_epoch": decoder_epoch},
                )
                index += 1
    selected, evidence = _development_epochs(e1=e1, fbc=fbc, e4=e4)
    assert selected == {"atcnet": 79, "fbcnet": 70, "sew_clif": 20, "ann_sew": 20}
    assert all(row["count"] == 54 for row in evidence.values())


def _session_data(session: str) -> dict[str, np.ndarray]:
    labels = np.repeat(np.arange(4, dtype=np.int64), 72)
    return {
        "X": np.zeros((288, 22, 1250), dtype=np.float32),
        "y": labels,
        "subject": np.full(288, "1"),
        "session": np.full(288, session),
        "run": np.asarray([f"run_{index // 48}" for index in range(288)]),
        "trial_id": np.asarray([f"bci2a:A01:{session}:{index:03d}" for index in range(288)]),
        "sfreq": np.asarray(250.0),
        "ch_names": np.asarray([f"C{index}" for index in range(22)]),
        "epoch_tmin": np.asarray(-1.0),
        "epoch_tmax": np.asarray(4.0),
        "dataset_name": np.asarray("bci2a"),
    }


def test_e6_metadata_enforces_physical_session_split() -> None:
    assert len(_metadata(_session_data("T"), expected_session="T", role="training")) == 288
    assert len(_metadata(_session_data("E"), expected_session="E", role="evaluation")) == 288
    with pytest.raises(Exception):
        _metadata(_session_data("E"), expected_session="T", role="training")
    assert len(PREDICTION_BASES) == 7
