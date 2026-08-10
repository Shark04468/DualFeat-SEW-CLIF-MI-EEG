#!/usr/bin/env python3
"""Materialize the V6.2-R1 P0 protocol and provenance gate."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    assert_t_e_isolation,
    build_run_fingerprint,
    file_sha256,
    session_t_run_grouped_folds,
    validate_trial_metadata,
    write_fingerprint_manifest,
    write_run_artifact_manifest,
)
from dpc_snn.utils.io import ensure_dir, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402


SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".md", ".sh"}


def _source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for root_name in ("src", "scripts", "configs"):
        for path in sorted((ROOT / root_name).rglob("*")):
            if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES:
                hashes[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    for name in ("pyproject.toml", "README.md", "PLAN.md", "CHECKLIST.md"):
        path = ROOT / name
        if path.exists():
            hashes[name] = file_sha256(path)
    return hashes


def _data_hashes(data_root: Path) -> dict[str, str]:
    return {
        path.name: file_sha256(path)
        for path in sorted(data_root.glob("*.npz"))
    }


def _environment() -> dict[str, object]:
    packages = {}
    for name in ("numpy", "torch", "scipy", "scikit-learn", "mne", "moabb"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "storage_root": os.environ.get("DPC_SNN_STORAGE_ROOT"),
        "cache_environment": {
            key: os.environ.get(key)
            for key in (
                "HF_HOME",
                "HF_DATASETS_CACHE",
                "TORCH_HOME",
                "MNE_DATA",
                "MOABB_DATA",
                "XDG_CACHE_HOME",
                "PIP_CACHE_DIR",
                "TMPDIR",
                "PYTHONPYCACHEPREFIX",
            )
        },
    }


def _metadata_rows(data: dict[str, object]) -> list[dict[str, object]]:
    labels = np.asarray(data["y"])
    return [
        {
            "dataset": str(data.get("dataset_name", "bci2a")),
            "subject": np.asarray(data["subject"])[index],
            "session": np.asarray(data["session"])[index],
            "run": np.asarray(data["run"])[index],
            "trial_id": np.asarray(data["trial_id"])[index],
            "class": int(labels[index]),
            "sfreq": float(data["sfreq"]),
            "ch_names": data["ch_names"],
            "epoch_tmin": float(data["epoch_tmin"]),
            "epoch_tmax": float(data["epoch_tmax"]),
        }
        for index in range(len(labels))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/processed/bci2a_v62",
        help="Directory containing the nine V6.2 BCI2a NPZ files.",
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/v62_p0_protocol.yaml",
    )
    parser.add_argument(
        "--output",
        default="runs/v62/P0_protocol",
    )
    args = parser.parse_args()

    storage = configure_cache_env()
    started = time.time()
    data_root = Path(args.data).resolve()
    output = ensure_dir(args.output)
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = load_processed_npz(data_root)
    required = {"run", "trial_id", "epoch_tmin", "epoch_tmax", "ch_names"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"P0 data is missing required metadata: {sorted(missing)}")
    rows = validate_trial_metadata(_metadata_rows(data))

    counts = Counter((row["subject"], row["session"]) for row in rows)
    expected_subjects = {str(value) for value in config["dataset"]["subjects"]}
    observed_subjects = {row["subject"] for row in rows}
    if observed_subjects != expected_subjects:
        raise ValueError(
            f"P0 expected subjects {sorted(expected_subjects)}, got {sorted(observed_subjects)}"
        )
    expected_trials = int(config["dataset"]["expected_trials_per_subject_session"])
    expected_keys = {(subject, session) for subject in expected_subjects for session in ("T", "E")}
    if set(counts) != expected_keys or any(counts[key] != expected_trials for key in expected_keys):
        raise ValueError(f"P0 subject/session counts are invalid: {counts}")

    split_manifest: dict[str, object] = {
        "method": "session_t_run_grouped",
        "n_splits": int(config["split"]["n_splits"]),
        "seed": int(config["split"]["seed"]),
        "subjects": {},
    }
    for subject in sorted(observed_subjects, key=int):
        training = [row for row in rows if row["subject"] == subject and row["session"] == "T"]
        evaluation = [row for row in rows if row["subject"] == subject and row["session"] == "E"]
        assert_t_e_isolation(training, evaluation)
        folds = session_t_run_grouped_folds(
            training,
            n_splits=int(config["split"]["n_splits"]),
            seed=int(config["split"]["seed"]),
            shuffle=bool(config["split"]["shuffle_groups"]),
        )
        split_manifest["subjects"][subject] = [
            {
                "fold": fold_index,
                "train_trial_ids": [training[int(index)]["trial_id"] for index in train],
                "validation_trial_ids": [training[int(index)]["trial_id"] for index in validation],
                "validation_runs": sorted({training[int(index)]["run"] for index in validation}),
            }
            for fold_index, (train, validation) in enumerate(folds)
        ]

    source = _source_hashes()
    data_hashes = _data_hashes(data_root)
    environment = _environment()
    fingerprint = build_run_fingerprint(
        resolved_config=config,
        source=source,
        data=data_hashes,
        split=split_manifest,
        augmentation={"policy": "none", "parents": []},
        prior={"policy": "none", "fold_local": True},
        checkpoint={"policy": "none", "resume": False},
        environment=environment,
    )

    shutil.copyfile(config_path, output / "resolved_config.yaml")
    write_json(output / "split_manifest.json", split_manifest)
    write_json(output / "augmentation_manifest.json", {"policy": "none", "parents": []})
    write_fingerprint_manifest(output / "source_fingerprint.json", fingerprint)
    metrics = {
        "status": "passed",
        "n_trials": len(rows),
        "n_subjects": len(observed_subjects),
        "n_sessions": 2,
        "trials_per_subject_session": expected_trials,
        "runs_per_subject_session": 6,
        "folds_per_subject": int(config["split"]["n_splits"]),
        "data_combined_sha256": fingerprint["components"]["data"],
        "source_combined_sha256": fingerprint["components"]["source"],
        "run_fingerprint": fingerprint["combined_sha256"],
        "storage_root": str(storage) if storage is not None else None,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "metrics.json", metrics)
    write_csv(output / "history.csv", [{"stage": "P0", **metrics}])
    write_json(
        output / "runtime_status.json",
        {"status": "completed", "started_at": started, "completed_at": time.time()},
    )
    (output / "stdout.log").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output / "stderr.log").write_text("", encoding="utf-8")
    write_run_artifact_manifest(
        output,
        required_files=(
            "manifest.json",
            "resolved_config.yaml",
            "source_fingerprint.json",
            "split_manifest.json",
            "augmentation_manifest.json",
            "history.csv",
            "metrics.json",
            "runtime_status.json",
            "stdout.log",
            "stderr.log",
        ),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

