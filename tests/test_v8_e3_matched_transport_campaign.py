from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from dpc_snn.experiments.v62_protocol import write_run_artifact_manifest
from dpc_snn.utils.io import write_json
from scripts.evaluate_v8_e3_matched_transport_gate import evaluate_campaign
from scripts.run_v8_e3_delay_residual_campaign import (
    _prepare_output,
    build_campaign_contract,
)
from scripts.run_v8_e3_delay_residual_fold import _import_prior_bundle
from scripts.run_v8_e3_static_delay import PRIOR_FILES


def _registered_config() -> dict:
    return yaml.safe_load(
        Path(
            "configs/experiments/v8_e3_matched_transport_campaign.yaml"
        ).read_text(encoding="utf-8")
    )


def test_formal_campaign_registers_complete_balanced_scope() -> None:
    config = _registered_config()
    assert config["subjects"] == [1, 3, 8]
    assert config["seeds"] == [0, 1, 2]
    assert config["folds"] == [0, 1, 2, 3, 4, 5]
    assert config["primary_variant"] == "sew_clif"
    assert config["matched_ann_variant"] == "ann_sew"
    assert config["delay"]["prior_seed_scope"] == (
        "subject_fold_fixed_across_classifier_seeds"
    )
    assert config["gate"]["expected_subject_seed_pairs"] == 9


def test_campaign_resume_contract_is_fail_closed(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    model_path = tmp_path / "model.yaml"
    config_path.write_text("schema: test\n", encoding="utf-8")
    model_path.write_text("model: test\n", encoding="utf-8")
    contract = build_campaign_contract(
        source_digest="a" * 64,
        config_path=config_path,
        model_config_path=model_path,
        inputs={"data": "b" * 64},
        subjects=[1],
        seeds=[0],
        folds=[0],
        variants=["ann_sew", "sew_clif"],
        device="cpu",
        canary=True,
    )
    output = tmp_path / "campaign"
    assert not _prepare_output(output, contract)
    assert _prepare_output(output, contract)
    changed = {**contract, "device": "cuda"}
    with pytest.raises(RuntimeError, match="exact resume contract"):
        _prepare_output(output, changed)


def test_prior_import_copies_only_a_hash_valid_complete_bundle(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for name in PRIOR_FILES:
        if name != "manifest.json":
            (source / name).write_bytes(name.encode("ascii"))
    write_run_artifact_manifest(source, required_files=PRIOR_FILES)
    destination = tmp_path / "destination"
    _import_prior_bundle(source, destination)
    assert all((destination / name).is_file() for name in PRIOR_FILES)
    (source / "summary.json").write_bytes(b"tampered")
    with pytest.raises(Exception):
        _import_prior_bundle(source, tmp_path / "rejected")


def _write_subject(data_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = 288
    labels = np.tile(np.arange(4, dtype=np.int64), count // 4)
    trial_ids = np.asarray([f"trial-{index:03d}" for index in range(count)])
    runs = np.asarray([str(index // 48) for index in range(count)])
    np.savez_compressed(
        data_root / "A01.npz",
        X=np.zeros((count, 1, 2), dtype=np.float32),
        y=labels,
        subject=np.full(count, "1"),
        session=np.full(count, "T"),
        run=runs,
        trial_id=trial_ids,
        sfreq=np.asarray(1.0),
        ch_names=np.asarray(["Cz"]),
        epoch_tmin=np.asarray(-1.0),
        epoch_tmax=np.asarray(4.0),
        dataset_name=np.asarray("bci2a"),
    )
    return labels, trial_ids, runs


def _probability(labels: np.ndarray, correct: np.ndarray) -> np.ndarray:
    prediction = (labels + 1) % 4
    prediction[correct] = labels[correct]
    probability = np.full((labels.size, 4), 0.02, dtype=np.float32)
    probability[np.arange(labels.size), prediction] = 0.94
    return probability


def _write_variant(
    directory: Path,
    *,
    variant: str,
    subject: int,
    seed: int,
    fold: int,
    indices: np.ndarray,
    labels: np.ndarray,
    trial_ids: np.ndarray,
    runs: np.ndarray,
    full_correct: np.ndarray,
    zero_correct: np.ndarray,
) -> None:
    variant_dir = directory / variant
    variant_dir.mkdir(parents=True)
    fold_labels = labels[indices]
    full = _probability(fold_labels, full_correct[indices])
    zero = _probability(fold_labels, zero_correct[indices])
    anchor = _probability(fold_labels, np.zeros(labels.size, dtype=bool)[indices])
    prediction = {
        "indices": indices,
        "labels": fold_labels,
        "subject": np.full(indices.size, subject, dtype=np.int64),
        "session": np.full(indices.size, "T"),
        "run": runs[indices],
        "trial_id": trial_ids[indices],
        "anchor_probability": anchor,
        "full_expert_logits": np.log(full),
        "matched_zero_expert_logits": np.log(zero),
        "same_weight_zero_expert_logits": np.log(zero),
        "full_probability": full,
        "matched_zero_probability": zero,
        "same_weight_zero_probability": zero,
    }
    np.savez_compressed(variant_dir / "outer_predictions.npz", **prediction)
    metric = {
        "variant": variant,
        "subject": subject,
        "seed": seed,
        "fold": fold,
        "session_e_accessed": False,
        "anchor_accuracy": float(np.mean(anchor.argmax(axis=1) == fold_labels)),
        "full_accuracy": float(np.mean(full.argmax(axis=1) == fold_labels)),
        "matched_zero_accuracy": float(np.mean(zero.argmax(axis=1) == fold_labels)),
        "same_weight_zero_accuracy": float(np.mean(zero.argmax(axis=1) == fold_labels)),
        "full_current_rms": 0.1,
        "zero_current_rms": 0.1,
        "full_optimizer_steps": 10,
        "zero_optimizer_steps": 10,
        "prior_audit_replicates": 5,
        "inner_prior_stability_passed": True,
        "outer_prior_stability_passed": True,
        "routing_sha256": "1" * 64,
        "prior_sha256": "2" * 64,
        "input_gain_sha256": "3" * 64,
        "delay_evidence_space": "csd_innovations_unwhitened_regions",
        "delay_transport_space": "car_anatomical_regions",
        "delay_transport_nodes": 9,
    }
    write_json(variant_dir / "metrics.json", metric)
    write_run_artifact_manifest(
        variant_dir,
        required_files=("manifest.json", "metrics.json", "outer_predictions.npz"),
    )


def _write_synthetic_campaign(
    root: Path,
    labels: np.ndarray,
    trial_ids: np.ndarray,
    runs: np.ndarray,
    source_digest: str,
) -> None:
    full_correct = np.zeros(labels.size, dtype=bool)
    full_correct[:216] = True
    zero_correct = np.zeros(labels.size, dtype=bool)
    zero_correct[:144] = True
    for fold in range(6):
        directory = root / "subject_01" / "seed_0" / f"fold_{fold}"
        directory.mkdir(parents=True)
        indices = np.arange(fold * 48, (fold + 1) * 48, dtype=np.int64)
        for variant in ("ann_sew", "sew_clif"):
            _write_variant(
                directory,
                variant=variant,
                subject=1,
                seed=0,
                fold=fold,
                indices=indices,
                labels=labels,
                trial_ids=trial_ids,
                runs=runs,
                full_correct=full_correct,
                zero_correct=zero_correct,
            )
        write_json(
            directory / "campaign_status.json",
            {
                "source_tree_sha256": source_digest,
                "session_e_accessed": False,
                "openbmi_s2_accessed": False,
            },
        )
        write_run_artifact_manifest(
            directory,
            required_files=("manifest.json", "campaign_status.json"),
        )


def _synthetic_config() -> dict:
    config = _registered_config()
    config["subjects"] = [1]
    config["seeds"] = [0]
    config["gate"] = {
        **config["gate"],
        "expected_subject_seed_pairs": 1,
        "minimum_positive_pairs": 1,
        "bootstrap_samples": 100,
    }
    return config


def test_independent_gate_recomputes_exact_oof_and_passes_registered_delta(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    labels, trial_ids, runs = _write_subject(data_root)
    campaign = tmp_path / "campaign"
    source_digest = "a" * 64
    _write_synthetic_campaign(campaign, labels, trial_ids, runs, source_digest)
    pair_rows, diagnostics, trial_rows, decision = evaluate_campaign(
        campaign=campaign,
        data_root=data_root,
        config=_synthetic_config(),
        source_digest=source_digest,
    )
    assert len(pair_rows) == 2
    assert len(diagnostics) == 2
    assert len(trial_rows) == 288
    assert decision["passed"] is True
    assert decision["gate"]["observed_median_gain_pp"] == pytest.approx(25.0)


def test_independent_gate_rejects_duplicate_oof_trial(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    labels, trial_ids, runs = _write_subject(data_root)
    campaign = tmp_path / "campaign"
    source_digest = "a" * 64
    _write_synthetic_campaign(campaign, labels, trial_ids, runs, source_digest)
    target = campaign / "subject_01" / "seed_0" / "fold_5" / "sew_clif"
    prediction = dict(np.load(target / "outer_predictions.npz", allow_pickle=False))
    prediction["indices"][0] = 0
    np.savez_compressed(target / "outer_predictions.npz", **prediction)
    write_run_artifact_manifest(
        target,
        required_files=("manifest.json", "metrics.json", "outer_predictions.npz"),
    )
    fold_dir = target.parent
    write_run_artifact_manifest(
        fold_dir,
        required_files=("manifest.json", "campaign_status.json"),
    )
    with pytest.raises(RuntimeError, match="source data|exact Session-T OOF coverage"):
        evaluate_campaign(
            campaign=campaign,
            data_root=data_root,
            config=_synthetic_config(),
            source_digest=source_digest,
        )
