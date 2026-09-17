from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import (
    ArtifactManifestError,
    FINGERPRINT_COMPONENTS,
    MetadataValidationError,
    PREDICTION_FIELDS,
    ResumeFingerprintMismatch,
    SessionLeakageError,
    build_run_fingerprint,
    canonical_json,
    session_t_run_grouped_folds,
    assert_t_e_isolation,
    validate_prediction_schema,
    validate_resume_fingerprint,
    validate_run_artifact_manifest,
    validate_trial_metadata,
    write_run_artifact_manifest,
    write_trial_predictions,
)
from dpc_snn.utils.io import write_json


def _metadata(*, sessions: tuple[str, ...] = ("T",), runs: int = 3) -> list[dict]:
    rows = []
    for session in sessions:
        for run in range(runs):
            for within_run in range(4):
                rows.append(
                    {
                        "dataset": "bci2a",
                        "subject": "A01",
                        "session": session,
                        "run": run,
                        "trial_id": f"{session}-{run}-{within_run}",
                        "class": within_run,
                        "sfreq": 250.0,
                        "ch_names": ["C3", "Cz", "C4"],
                        "epoch_tmin": 0.0,
                        "epoch_tmax": 4.0,
                    }
                )
    return rows


def _fingerprint_payloads() -> dict[str, object]:
    return {
        "resolved_config": {"training": {"epochs": 10, "lr": 1e-3}},
        "source": {"src/a.py": "a" * 64},
        "data": {"trials": np.arange(8, dtype=np.int16)},
        "split": {"fold": 0, "train": [0, 1], "validation": [2]},
        "augmentation": {"name": "none", "parents": []},
        "prior": {"kind": "none"},
        "checkpoint": {"rule": "fixed_epoch", "initial": None},
        "environment": {"python": "3.12", "numpy": "2.x"},
    }


def test_metadata_validation_is_strict_and_normalizes_columnar_input() -> None:
    rows = _metadata(runs=1)
    columnar = {
        field: [row[field] for row in rows]
        for field in ("subject", "session", "run", "trial_id", "class")
    }
    columnar.update(
        {
            "dataset": "bci2a",
            "sfreq": 250,
            "ch_names": ["C3", "Cz", "C4"],
            "epoch_tmin": 0,
            "epoch_tmax": 4,
        }
    )

    normalized = validate_trial_metadata(columnar)

    assert len(normalized) == 4
    assert normalized[0]["subject"] == "A01"
    assert normalized[0]["run"] == "0"
    assert normalized[0]["sfreq"] == 250.0
    assert normalized[0]["ch_names"] == ["C3", "Cz", "C4"]

    missing = dict(rows[0])
    missing.pop("trial_id")
    with pytest.raises(MetadataValidationError, match="trial_id"):
        validate_trial_metadata([missing])

    invalid_bounds = dict(rows[0], epoch_tmin=4.0, epoch_tmax=4.0)
    with pytest.raises(MetadataValidationError, match="epoch_tmin < epoch_tmax"):
        validate_trial_metadata([invalid_bounds])

    duplicate_channels = dict(rows[0], ch_names=["C3", "C3"])
    with pytest.raises(MetadataValidationError, match="duplicate channels"):
        validate_trial_metadata([duplicate_channels])


def test_session_t_run_grouped_folds_are_deterministic_and_group_isolated() -> None:
    metadata = _metadata(runs=6)

    first = session_t_run_grouped_folds(metadata, n_splits=3, seed=17)
    second = session_t_run_grouped_folds(metadata, n_splits=3, seed=17)

    assert len(first) == 3
    for (train_a, val_a), (train_b, val_b) in zip(first, second, strict=True):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(val_a, val_b)
        train_runs = {metadata[int(index)]["run"] for index in train_a}
        validation_runs = {metadata[int(index)]["run"] for index in val_a}
        assert train_runs.isdisjoint(validation_runs)
        assert set(train_a).isdisjoint(val_a)
        assert set(train_a) | set(val_a) == set(range(len(metadata)))

    validation_indices = np.concatenate([fold[1] for fold in first])
    assert sorted(validation_indices.tolist()) == list(range(len(metadata)))


def test_session_e_is_rejected_from_folds_and_every_fit_provenance_role() -> None:
    training = _metadata(sessions=("T",), runs=2)
    evaluation = _metadata(sessions=("E",), runs=2)
    assert assert_t_e_isolation(training, evaluation)

    with pytest.raises(SessionLeakageError, match="Session E leakage.*augmentation"):
        assert_t_e_isolation(
            training,
            evaluation,
            augmentation_parent_metadata=[evaluation[0]],
        )

    with pytest.raises(SessionLeakageError, match="overlapping physical trials"):
        assert_t_e_isolation(
            training[:-1],
            evaluation,
            validation_metadata=[training[0]],
        )

    with pytest.raises(SessionLeakageError, match="Session E leakage"):
        session_t_run_grouped_folds(training + evaluation, n_splits=2)


def test_canonical_fingerprint_is_order_independent_and_component_complete() -> None:
    payloads = _fingerprint_payloads()
    reordered = {
        **payloads,
        "resolved_config": {"training": {"lr": 1e-3, "epochs": 10}},
    }

    first = build_run_fingerprint(**payloads)
    second = build_run_fingerprint(**reordered)

    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})
    assert first == second
    assert tuple(first["components"]) == FINGERPRINT_COMPONENTS
    assert len(first["combined_sha256"]) == 64

    for component in FINGERPRINT_COMPONENTS:
        changed = dict(payloads)
        changed[component] = {"changed": component}
        candidate = build_run_fingerprint(**changed)
        assert candidate["components"][component] != first["components"][component]
        assert candidate["combined_sha256"] != first["combined_sha256"]


def test_resume_requires_the_complete_fingerprint_to_match() -> None:
    payloads = _fingerprint_payloads()
    saved = build_run_fingerprint(**payloads)
    current = build_run_fingerprint(**payloads)

    assert validate_resume_fingerprint(saved, current)

    changed = dict(payloads)
    changed["environment"] = {"python": "3.13", "numpy": "2.x"}
    with pytest.raises(ResumeFingerprintMismatch, match="environment"):
        validate_resume_fingerprint(saved, build_run_fingerprint(**changed))


def test_trial_prediction_recorder_writes_npz_and_csv_with_exact_schema(
    tmp_path: Path,
) -> None:
    logits = np.asarray([[0.1, 0.9], [2.0, -1.0]], dtype=np.float32)
    probabilities = np.asarray([[0.31, 0.69], [0.95, 0.05]], dtype=np.float32)

    paths = write_trial_predictions(
        tmp_path,
        logits=logits,
        probabilities=probabilities,
        pred=[1, 0],
        label=[1, 1],
        subject="A01",
        session="E",
        run=[4, 5],
        trial_id=["E-4-0", "E-5-0"],
        seed=17,
        model="dasp_snn_v62_r1",
    )

    with np.load(paths["npz"], allow_pickle=False) as archive:
        assert set(archive.files) == set(PREDICTION_FIELDS)
        assert archive["logits"].shape == (2, 2)
        assert archive["session"].tolist() == ["E", "E"]
        assert archive["model"].tolist() == ["dasp_snn_v62_r1"] * 2
    with paths["csv"].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(PREDICTION_FIELDS)
        assert len(list(reader)) == 2
    assert validate_prediction_schema(paths["npz"], paths["csv"]) == {
        "n_trials": 2,
        "n_classes": 2,
    }


def test_run_artifact_manifest_rejects_missing_or_modified_required_files(
    tmp_path: Path,
) -> None:
    write_trial_predictions(
        tmp_path,
        logits=[[1.0, 0.0]],
        probabilities=[[0.75, 0.25]],
        pred=[0],
        label=[0],
        subject="A01",
        session="E",
        run="4",
        trial_id="E-4-0",
        seed=0,
        model="dasp_snn_v62_r1",
    )
    write_json(tmp_path / "metrics.json", {"accuracy": 1.0})
    required = ("manifest.json", "metrics.json", "predictions.npz", "predictions.csv")

    manifest_path = write_run_artifact_manifest(tmp_path, required_files=required)

    assert manifest_path == tmp_path / "manifest.json"
    manifest = validate_run_artifact_manifest(
        tmp_path,
        required_files=required,
        verify_prediction_schema=True,
    )
    assert manifest["required_files"] == list(required)

    write_json(tmp_path / "metrics.json", {"accuracy": 0.0})
    with pytest.raises(ArtifactManifestError, match="size/hash mismatch"):
        validate_run_artifact_manifest(tmp_path, required_files=required)

    (tmp_path / "predictions.csv").unlink()
    with pytest.raises(ArtifactManifestError, match="missing required artifacts"):
        validate_run_artifact_manifest(tmp_path, required_files=required)
