from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from dpc_snn.experiments.v8_protocol import (
    V8_FINGERPRINT_COMPONENTS,
    V8DataAccessError,
    V8ProtocolError,
    V8ResumeFingerprintMismatch,
    assert_delay_control_contract,
    assert_v8_data_access,
    build_v8_run_fingerprint,
    collect_source_tree_manifest,
    source_tree_digest,
    validate_v8_hpo_budget,
    validate_v8_resume_fingerprint,
    v8_heldout_lock_manifest,
)


def _metadata(dataset: str, session: str) -> list[dict[str, object]]:
    return [
        {
            "dataset": dataset,
            "subject": "1",
            "session": session,
            "run": "1",
            "trial_id": "trial-1",
            "class": 0,
            "sfreq": 250.0,
            "ch_names": ["C3", "Cz", "C4"],
            "epoch_tmin": -1.0,
            "epoch_tmax": 4.0,
        }
    ]


def _fingerprint(**overrides: object) -> dict[str, object]:
    payload = {
        "resolved_run_config": {"experiment_id": "V8-test", "seed": 0},
        "source_tree": {"src/model.py": "a" * 64},
        "data": {"A01.npz": "b" * 64},
        "split": {"fold": 0, "train": [0, 1], "validation": [2]},
        "augmentation": {"policy": "none"},
        "prior": {"policy": "none"},
        "checkpoint": {"policy": "fresh"},
        "environment": {"python": "3.12", "torch": "2.x"},
    }
    payload.update(overrides)
    return build_v8_run_fingerprint(**payload)


def _minimal_tree(root: Path) -> None:
    for directory in ("src", "scripts", "configs", "tests"):
        path = root / directory
        path.mkdir(parents=True)
        (path / f"{directory}.py").write_text(f"VALUE = {directory!r}\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (root / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    (root / "environment.yml").write_text("name: test\n", encoding="utf-8")


def test_source_tree_manifest_does_not_require_git_and_detects_every_source_change(
    tmp_path: Path,
) -> None:
    _minimal_tree(tmp_path)
    generated = tmp_path / "src" / "example.egg-info"
    generated.mkdir()
    (generated / "SOURCES.txt").write_text("machine-specific\n", encoding="utf-8")
    first = collect_source_tree_manifest(tmp_path)
    first_digest = source_tree_digest(first)
    assert ".git" not in "\n".join(first)
    assert set(first) >= {
        "src/src.py",
        "scripts/scripts.py",
        "configs/configs.py",
        "tests/tests.py",
        "pyproject.toml",
        "requirements.txt",
        "environment.yml",
    }
    assert "src/example.egg-info/SOURCES.txt" not in first

    (tmp_path / "src" / "src.py").write_text("VALUE = 'changed'\n", encoding="utf-8")
    second = collect_source_tree_manifest(tmp_path)
    assert source_tree_digest(second) != first_digest


def test_v8_resume_hashes_complete_run_contract() -> None:
    saved = _fingerprint()
    assert tuple(saved["components"]) == V8_FINGERPRINT_COMPONENTS
    assert validate_v8_resume_fingerprint(saved, _fingerprint())

    with pytest.raises(V8ResumeFingerprintMismatch, match="resolved_run_config"):
        validate_v8_resume_fingerprint(
            saved,
            _fingerprint(resolved_run_config={"experiment_id": "V8-test", "seed": 1}),
        )
    with pytest.raises(V8ResumeFingerprintMismatch, match="checkpoint"):
        validate_v8_resume_fingerprint(
            saved,
            _fingerprint(checkpoint={"policy": "warm", "sha256": "c" * 64}),
        )


def test_v8_development_lock_rejects_bci2a_e_and_all_openbmi_access() -> None:
    assert assert_v8_data_access(
        _metadata("bci2a", "T"), stage="development", role="training"
    )
    with pytest.raises(V8DataAccessError):
        assert_v8_data_access(
            _metadata("bci2a", "E"), stage="development", role="evaluation"
        )
    with pytest.raises(V8DataAccessError):
        assert_v8_data_access(
            _metadata("openbmi", "S1"), stage="development", role="training"
        )

    lock = v8_heldout_lock_manifest()
    assert {item["session"] for item in lock["locked"]} == {"E", "S2"}


def test_v8_formal_evaluation_stages_have_directional_session_access() -> None:
    assert assert_v8_data_access(
        _metadata("bci2a", "T"), stage="bci2a_evaluation", role="training"
    )
    assert assert_v8_data_access(
        _metadata("bci2a", "E"), stage="bci2a_evaluation", role="evaluation"
    )
    with pytest.raises(V8DataAccessError):
        assert_v8_data_access(
            _metadata("bci2a", "E"), stage="bci2a_evaluation", role="checkpoint_selection"
        )

    assert assert_v8_data_access(
        _metadata("openbmi", "S1"), stage="openbmi_confirmation", role="training"
    )
    assert assert_v8_data_access(
        _metadata("openbmi", "S2"), stage="openbmi_confirmation", role="evaluation"
    )
    with pytest.raises(V8DataAccessError):
        assert_v8_data_access(
            _metadata("openbmi", "S2"),
            stage="openbmi_confirmation",
            role="normalization",
        )


def test_matched_zero_changes_only_delay_operator_and_shared_state_stays_frozen() -> None:
    full_current = np.asarray([[1.0, -1.0], [0.2, 0.3]], dtype=np.float32)
    zero_current = np.asarray([[0.9, -0.8], [0.1, 0.4]], dtype=np.float32)
    full_probability = np.asarray([[0.0, 0.75, 0.25]], dtype=np.float32)
    zero_probability = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    state = {"frontend": "a" * 64, "decoder": "b" * 64}
    assert assert_delay_control_contract(
        full_current=full_current,
        zero_current=zero_current,
        full_delay_probability=full_probability,
        zero_delay_probability=zero_probability,
        full_fractional_delay=np.asarray([0.25]),
        zero_fractional_delay=np.asarray([0.0]),
        full_routing_fingerprint="a" * 64,
        zero_routing_fingerprint="a" * 64,
        shared_state_before=state,
        shared_state_after=dict(state),
    )
    with pytest.raises(V8ProtocolError, match="point mass"):
        assert_delay_control_contract(
            full_current=full_current,
            zero_current=zero_current,
            full_delay_probability=full_probability,
            zero_delay_probability=full_probability,
            full_fractional_delay=np.asarray([0.25]),
            zero_fractional_delay=np.asarray([0.25]),
            full_routing_fingerprint="a" * 64,
            zero_routing_fingerprint="a" * 64,
            shared_state_before=state,
            shared_state_after=state,
        )
    with pytest.raises(V8ProtocolError, match="state changed"):
        assert_delay_control_contract(
            full_current=full_current,
            zero_current=zero_current,
            full_delay_probability=full_probability,
            zero_delay_probability=zero_probability,
            full_fractional_delay=np.asarray([0.25]),
            zero_fractional_delay=np.asarray([0.0]),
            full_routing_fingerprint="a" * 64,
            zero_routing_fingerprint="a" * 64,
            shared_state_before=state,
            shared_state_after={**state, "decoder": "c" * 64},
        )
    with pytest.raises(V8ProtocolError, match="routing invariants"):
        assert_delay_control_contract(
            full_current=full_current,
            zero_current=zero_current,
            full_delay_probability=full_probability,
            zero_delay_probability=zero_probability,
            full_fractional_delay=np.asarray([0.25]),
            zero_fractional_delay=np.asarray([0.0]),
            full_routing_fingerprint="a" * 64,
            zero_routing_fingerprint="b" * 64,
            shared_state_before=state,
            shared_state_after=state,
        )


def test_v8_hpo_budget_is_bounded_and_deduplicated() -> None:
    assert validate_v8_hpo_budget([{"channels": value} for value in range(24)])
    with pytest.raises(V8ProtocolError, match="budget exceeded"):
        validate_v8_hpo_budget([{"channels": value} for value in range(25)])
    with pytest.raises(V8ProtocolError, match="duplicates"):
        validate_v8_hpo_budget([{"channels": 64}, {"channels": 64}])


def test_v8_preregistered_stage_semantics_match_formal_runners() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs" / "experiments" / "v8_p0_protocol.yaml").read_text(
            encoding="utf-8"
        )
    )
    expected = {
        "E0": "engineering_invariants",
        "E1": "locked_unified_baselines",
        "E2": "zero_delay_accuracy_scaffold",
        "E3": "optional_delay_auxiliary",
        "E4": "matched_ann_plif_clif_sew_clif_controls",
        "E5": "bounded_session_t_hpo",
        "E6": "frozen_bci2a_t_to_e_evaluation",
        "E7": "frozen_snn_utility_evaluation",
        "E8": "openbmi_external_confirmation",
        "E9": "frozen_ablation_robustness_and_statistics",
    }
    assert {stage: payload["name"] for stage, payload in config["stages"].items()} == expected
    assert config["execution_order"] == [
        "P0",
        "E0",
        "E1",
        "E2",
        "E5",
        "E2_selected",
        "E3",
        "E4",
        "architecture_freeze",
        "E6",
        "E7",
        "E8",
        "E9",
    ]
    assert len(config["stages"]["E1"]["models"]) == 7


def test_v8_heldout_unlock_requirements_match_evaluation_order() -> None:
    lock = v8_heldout_lock_manifest()
    assert lock["unlock_requires"]["bci2a_E"] == "signed V8 architecture-freeze manifest"
    assert "passed E6 audit" in lock["unlock_requires"]["openbmi_S2"]
    assert "completed E7 utility evaluation" in lock["unlock_requires"]["openbmi_S2"]


def test_v8_cloud_chain_reruns_p0_e0_and_fails_closed_on_e3_runtime_errors() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "cloud" / "run_v8_full_after_e2.sh").read_text(
        encoding="utf-8"
    )
    assert 'scripts/run_v8_p0.py' in script
    assert 'scripts/run_v8_e0_invariants.py' in script
    assert '--p0 "$P0" --e0 "$E0"' in script
    assert 'test -f "$E3_SEQUENCE/sequence_status.json"' in script
    assert 'set +e\n"$PY" scripts/run_v8_e3_sequence.py' not in script
