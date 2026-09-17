from __future__ import annotations

from copy import deepcopy

import pytest

from dpc_snn.experiments.v8_protocol import (
    V8FreezeManifestError,
    build_v8_external_unlock_manifest,
    build_v8_freeze_manifest,
    validate_v8_external_unlock_manifest,
    validate_v8_freeze_manifest,
)


def _payload() -> dict:
    digest = "a" * 64
    return {
        "freeze_scope": "pre_E6_architecture_and_analysis",
        "created_at_utc": "2026-07-18T12:00:00+00:00",
        "source_tree_sha256": digest,
        "architecture": {
            "name": "v8_accuracy_first",
            "primary_variant": "clif_plain",
            "model_config": {"name": "v8_accuracy_first", "decoder_kind": "clif"},
            "matched_ann_control_config": {
                "name": "v8_accuracy_first",
                "decoder_kind": "ann",
            },
            "snn_gate_passed": True,
            "delay": {
                "enabled": False,
                "mode": "off",
                "residual_epoch": 0,
                "gate_passed": False,
                "config": None,
                "exclusion_reason": "E3 gate did not pass",
            },
        },
        "training": {"batch_size": 4},
        "preprocessing": {"reference": "CAR"},
        "augmentation": {"enabled": True},
        "checkpoint_rule": {
            "selection_data": "BCI2a Session T development only",
            "final_epoch": 80,
            "scheduler_horizon": 200,
            "heldout_checkpoint_selection": False,
        },
        "analysis_plan": {
            "primary_metric": "subject_macro_accuracy",
            "subject_is_inferential_unit": True,
            "post_E_tuning_allowed": False,
        },
        "development_evidence": {"heldout_metrics_used_for_selection": False},
        "baselines": {"models": ["fbcnet"], "fixed_epochs": {"fbcnet": 100}},
        "heldout_access": {
            "bci2a_session_e_accessed_before_freeze": False,
            "openbmi_session_s2_accessed_before_freeze": False,
            "status": "sealed_until_validated_freeze",
        },
        "artifact_hashes": {"e5_selected_model": "b" * 64},
    }


def test_v8_freeze_round_trip_and_source_binding() -> None:
    freeze = build_v8_freeze_manifest(_payload())

    validated = validate_v8_freeze_manifest(
        freeze, expected_source_tree_sha256="a" * 64
    )

    assert validated["combined_sha256"] == freeze["combined_sha256"]
    assert validated["architecture"]["primary_variant"] == "clif_plain"


def test_v8_freeze_rejects_content_tampering() -> None:
    freeze = build_v8_freeze_manifest(_payload())
    tampered = deepcopy(freeze)
    tampered["checkpoint_rule"]["final_epoch"] = 81

    with pytest.raises(V8FreezeManifestError, match="digest"):
        validate_v8_freeze_manifest(tampered)


def test_v8_freeze_rejects_pre_freeze_heldout_access() -> None:
    payload = _payload()
    payload["heldout_access"]["bci2a_session_e_accessed_before_freeze"] = True

    with pytest.raises(V8FreezeManifestError, match="held-out"):
        build_v8_freeze_manifest(payload)


def test_v8_freeze_rejects_active_source_drift() -> None:
    freeze = build_v8_freeze_manifest(_payload())

    with pytest.raises(V8FreezeManifestError, match="source tree"):
        validate_v8_freeze_manifest(
            freeze, expected_source_tree_sha256="c" * 64
        )


def test_v8_freeze_rejects_inconsistent_delay_state() -> None:
    payload = _payload()
    payload["architecture"]["delay"]["enabled"] = True

    with pytest.raises(V8FreezeManifestError, match="delay state"):
        build_v8_freeze_manifest(payload)


def test_v8_external_unlock_is_bound_to_source_and_parent_freeze() -> None:
    payload = {
        "created_at_utc": "2026-07-18T12:00:00+00:00",
        "source_tree_sha256": "a" * 64,
        "parent_freeze_sha256": "b" * 64,
        "architecture_adaptation": {
            "permitted_change": "four_class_head_to_binary_head_only",
            "target_sfreq": 250,
            "ordered_channels": [f"C{index}" for index in range(22)],
        },
        "training": {"final_epoch": 80},
        "dataset": {
            "train_session": "S1",
            "evaluation_session": "S2",
            "confirmatory_subjects": [1, 2],
        },
        "analysis_plan": {"primary_metric": "subject_macro_accuracy"},
        "evidence_hashes": {"e6_audit": "c" * 64},
        "heldout_access": {
            "openbmi_s2_accessed_before_unlock": False,
            "s2_checkpoint_selection_allowed": False,
            "s2_gradient_updates_allowed": False,
        },
    }
    unlock = build_v8_external_unlock_manifest(payload)

    validated = validate_v8_external_unlock_manifest(
        unlock,
        expected_source_tree_sha256="a" * 64,
        expected_parent_freeze_sha256="b" * 64,
    )

    assert validated["combined_sha256"] == unlock["combined_sha256"]


def test_v8_external_unlock_rejects_s2_fitting() -> None:
    payload = {
        "created_at_utc": "2026-07-18T12:00:00+00:00",
        "source_tree_sha256": "a" * 64,
        "parent_freeze_sha256": "b" * 64,
        "architecture_adaptation": {
            "permitted_change": "four_class_head_to_binary_head_only",
            "target_sfreq": 250,
            "ordered_channels": [f"C{index}" for index in range(22)],
        },
        "training": {},
        "dataset": {
            "train_session": "S1",
            "evaluation_session": "S2",
            "confirmatory_subjects": [1],
        },
        "analysis_plan": {},
        "evidence_hashes": {"e6_audit": "c" * 64},
        "heldout_access": {
            "openbmi_s2_accessed_before_unlock": False,
            "s2_checkpoint_selection_allowed": True,
            "s2_gradient_updates_allowed": False,
        },
    }

    with pytest.raises(V8FreezeManifestError, match="accessed or authorized"):
        build_v8_external_unlock_manifest(payload)
