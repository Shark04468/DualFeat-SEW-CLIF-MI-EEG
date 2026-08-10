from __future__ import annotations

import numpy as np
import pytest
import yaml

from dpc_snn.experiments.v62_protocol import write_run_artifact_manifest
from dpc_snn.utils.io import write_json
from scripts.run_v8_e3_delay_residual_fold import (
    VARIANT_FILES,
    _load_completed_variant,
    _prepare_partial_output,
    _select_paired_epoch,
    capacity_audit,
)


def test_delay_fold_capacity_audit_matches_ann_and_snn() -> None:
    config = yaml.safe_load(
        open(
            "configs/experiments/v8_e3_delay_residual_canary.yaml",
            encoding="utf-8",
        )
    )
    audit = capacity_audit(["ann_sew", "sew_clif"], config)
    assert audit["status"] == "passed"
    assert audit["variants"]["ann_sew"]["parameters"] == audit["variants"]["sew_clif"][
        "parameters"
    ]


def test_region_delay_fold_capacity_audit_matches_ann_and_snn() -> None:
    config = yaml.safe_load(
        open(
            "configs/experiments/v8_e3_region_delay_residual_canary.yaml",
            encoding="utf-8",
        )
    )
    audit = capacity_audit(["ann_sew", "sew_clif"], config)
    assert audit["status"] == "passed"
    assert audit["variants"]["ann_sew"]["parameters"] == audit["variants"]["sew_clif"][
        "parameters"
    ]


def test_paired_checkpoint_rule_uses_mean_full_zero_validation_metric() -> None:
    full = [
        {"epoch": 1, "validation_kappa": 0.9, "validation_accuracy": 0.9},
        {"epoch": 2, "validation_kappa": 0.7, "validation_accuracy": 0.8},
        {"epoch": 3, "validation_kappa": 0.6, "validation_accuracy": 0.7},
    ]
    zero = [
        {"epoch": 1, "validation_kappa": 0.1, "validation_accuracy": 0.2},
        {"epoch": 2, "validation_kappa": 0.7, "validation_accuracy": 0.8},
        {"epoch": 3, "validation_kappa": 0.9, "validation_accuracy": 0.9},
    ]
    epoch, metrics = _select_paired_epoch(full, zero, minimum_epoch=1)
    assert epoch == 3
    assert metrics["mean_validation_kappa"] == 0.75


def test_partial_run_resume_requires_an_exact_contract(tmp_path) -> None:
    output = tmp_path / "run"
    contract = {"schema": "test/v1", "combined_sha256": "a" * 64}
    assert not _prepare_partial_output(output, contract)
    assert _prepare_partial_output(output, contract)
    with pytest.raises(RuntimeError, match="contract mismatch"):
        _prepare_partial_output(
            output,
            {"schema": "test/v1", "combined_sha256": "b" * 64},
        )


def test_completed_variant_resume_requires_manifest_and_trial_identity(tmp_path) -> None:
    directory = tmp_path / "ann_sew"
    directory.mkdir()
    indices = np.asarray([1, 3], dtype=np.int64)
    labels = np.asarray([0, 2], dtype=np.int64)
    trial_ids = np.asarray(["trial-1", "trial-3"])
    prediction = {
        "indices": indices,
        "labels": labels,
        "subject": np.asarray([1, 1]),
        "session": np.asarray(["T", "T"]),
        "run": np.asarray(["0", "0"]),
        "trial_id": trial_ids,
        "anchor_probability": np.full((2, 4), 0.25),
        "full_expert_logits": np.zeros((2, 4)),
        "matched_zero_expert_logits": np.zeros((2, 4)),
        "same_weight_zero_expert_logits": np.zeros((2, 4)),
        "full_probability": np.full((2, 4), 0.25),
        "matched_zero_probability": np.full((2, 4), 0.25),
        "same_weight_zero_probability": np.full((2, 4), 0.25),
    }
    np.savez_compressed(directory / "outer_predictions.npz", **prediction)
    write_json(directory / "metrics.json", {"variant": "ann_sew"})
    for name in VARIANT_FILES:
        path = directory / name
        if name in {"manifest.json", "metrics.json", "outer_predictions.npz"}:
            continue
        path.write_bytes(b"test")
    write_run_artifact_manifest(directory, required_files=VARIANT_FILES)
    loaded = _load_completed_variant(
        directory,
        expected_indices=indices,
        expected_labels=labels,
        expected_trial_ids=trial_ids,
    )
    assert loaded is not None and loaded[0]["variant"] == "ann_sew"
    with pytest.raises(RuntimeError, match="trial identities"):
        _load_completed_variant(
            directory,
            expected_indices=indices,
            expected_labels=labels,
            expected_trial_ids=np.asarray(["wrong", "trial-3"]),
        )
