from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_script(
    "test_audit_v7_fold_delay_prior",
    ROOT / "scripts" / "audit_v7_fold_delay_prior.py",
)
RUNNER = _load_script(
    "test_run_v7_e3_delay",
    ROOT / "scripts" / "run_v7_e3_delay.py",
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _valid_prior(root: Path, scope: str = "class_3") -> None:
    scope_root = root / "fold_0" / scope
    scope_root.mkdir(parents=True)
    shape = (12, 12, 16, 16)
    np.savez_compressed(
        scope_root / "fold_local_evidence_prior.npz",
        route_probability=np.full(shape, 0.5, dtype=np.float32),
        positive_delay_probability=np.full(shape + (16,), 1.0 / 16, dtype=np.float32),
        fractional_delay_target=np.zeros(shape, dtype=np.float32),
    )
    _write_json(
        scope_root / "fold_local_evidence_prior.json",
        {
            "subject": 1,
            "fold": 0,
            "scope": scope,
            "fit_scope": "inner_training_fold_only",
            "evidence_pipeline_passed": True,
            "evidence_checks": {
                "natural_nonzero_edges": True,
                "split_half_delay": True,
                "bootstrap_frequency": True,
                "time_reversal": True,
                "phase_surrogate": True,
            },
            "session_e_accessed": False,
            "heldout_data_accessed": False,
            "analytic_representation": (
                "v7_online_car_baseline_mean_fold_gain_causal_filterbank_"
                "exact_sensor_fast_aligned_to_slow"
            ),
            "frontend_fingerprint": "frontend-test",
        },
    )
    _write_json(
        root / "manifest.json",
        {
            "status": "completed",
            "scientific_gate_eligible": True,
            "bootstrap_samples": 128,
            "scope_mode": "classes",
            "class_labels": [3],
            "subject": 1,
            "folds": [0],
            "session_e_accessed": False,
        },
    )


def test_class_only_audit_selects_only_requested_training_trials() -> None:
    labels = np.asarray([0, 1, 2, 3, 3, 0], dtype=np.int64)
    train = np.arange(len(labels), dtype=np.int64)

    scopes = AUDIT._audit_scopes(
        train,
        labels,
        scope_mode="classes",
        class_labels=[3],
    )

    assert [name for name, _ in scopes] == ["class_3"]
    np.testing.assert_array_equal(scopes[0][1], np.asarray([3, 4]))


def test_class_only_audit_rejects_missing_or_unavailable_labels() -> None:
    labels = np.asarray([0, 1, 2, 3], dtype=np.int64)
    train = np.arange(len(labels), dtype=np.int64)

    with pytest.raises(ValueError, match="requires --class-labels"):
        AUDIT._audit_scopes(train, labels, scope_mode="classes", class_labels=[])
    with pytest.raises(ValueError, match="absent from the fold"):
        AUDIT._audit_scopes(train, labels, scope_mode="classes", class_labels=[4])


def test_runner_loads_only_the_requested_gate_passing_class_prior(tmp_path: Path) -> None:
    _valid_prior(tmp_path)

    arrays, provenance = RUNNER._load_audited_fold_prior(
        tmp_path,
        subject=1,
        fold=0,
        scope="class_3",
    )

    assert set(arrays) >= {
        "route_probability",
        "positive_delay_probability",
        "fractional_delay_target",
    }
    assert provenance["scope"] == "class_3"
    assert provenance["policy"] == "fold_local_v7_online_matched_class_3_b128"


def test_runner_rejects_a_class_prior_when_any_registered_check_failed(
    tmp_path: Path,
) -> None:
    _valid_prior(tmp_path)
    summary_path = tmp_path / "fold_0" / "class_3" / "fold_local_evidence_prior.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["evidence_checks"]["time_reversal"] = False
    _write_json(summary_path, summary)

    with pytest.raises(RuntimeError, match="every registered evidence check"):
        RUNNER._load_audited_fold_prior(
            tmp_path,
            subject=1,
            fold=0,
            scope="class_3",
        )


def test_runner_rejects_manifest_that_does_not_cover_requested_class(
    tmp_path: Path,
) -> None:
    _valid_prior(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["class_labels"] = [2]
    _write_json(manifest_path, manifest)

    with pytest.raises(RuntimeError, match="class list"):
        RUNNER._load_audited_fold_prior(
            tmp_path,
            subject=1,
            fold=0,
            scope="class_3",
        )


def test_route_only_readout_contract_rejects_hidden_generic_statistics_head() -> None:
    valid = {
        "atc_delayed_statistics_bins": 0,
        "atc_delayed_route_statistics_bins": 16,
        "atc_delayed_route_statistics_cp_rank": 8,
    }
    RUNNER._validate_route_only_readout_config(valid)

    with pytest.raises(RuntimeError, match="generic delayed-statistics"):
        RUNNER._validate_route_only_readout_config(
            {**valid, "atc_delayed_statistics_bins": 16}
        )
    with pytest.raises(RuntimeError, match="route-statistics bins"):
        RUNNER._validate_route_only_readout_config(
            {**valid, "atc_delayed_route_statistics_bins": 0}
        )


def test_e3_model_overrides_reject_unregistered_keys() -> None:
    model_config: dict[str, object] = {}
    with pytest.raises(RuntimeError, match="unregistered E3 model overrides"):
        RUNNER._apply_registered_model_overrides(
            model_config,
            {"untracked_architecture_change": 1},
        )
