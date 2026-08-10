#!/usr/bin/env python3
"""Materialize the exact E1 ATC OOF run as the V8 E2 accuracy anchor."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    build_v8_run_fingerprint,
    collect_source_tree_manifest,
    source_tree_digest,
    v8_heldout_lock_manifest,
    write_v8_fingerprint,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402


RUN_FILES = (
    "manifest.json",
    "resolved_config.yaml",
    "source_fingerprint.json",
    "source_reference.json",
    "heldout_lock_manifest.json",
    "metrics.json",
    "predictions.npz",
    "predictions.csv",
    "runtime_status.json",
    "stdout.log",
    "stderr.log",
)


def _link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        if file_sha256(target) != file_sha256(source):
            raise RuntimeError(f"existing anchor file differs from source: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _strongest_seed_zero_macro(rows: list[dict[str, str]], subjects: list[int]) -> tuple[str, float]:
    candidates: dict[str, list[float]] = {}
    wanted = {int(value) for value in subjects}
    for row in rows:
        if int(row["seed"]) != 0 or int(row["subject"]) not in wanted:
            continue
        candidates.setdefault(row["model"], []).append(float(row["accuracy"]))
    complete = {
        model: float(np.mean(values))
        for model, values in candidates.items()
        if len(values) == len(wanted)
    }
    if not complete:
        raise RuntimeError("E1 summary has no complete seed-0 three-subject baseline")
    model = max(complete, key=complete.get)
    return model, complete[model]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default="configs/experiments/v8_e2_atc_accuracy_anchor.yaml"
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("stage") != "development":
        raise RuntimeError("V8 ATC anchor must remain a development artifact")
    if bool(config["data_access"].get("heldout_session_e_accessed")):
        raise RuntimeError("V8 ATC anchor may not access Session E")
    subjects = [int(value) for value in config["subjects"]]
    seeds = [int(value) for value in config["seeds"]]
    e1_root = Path(args.e1_root).resolve()
    output = ensure_dir(Path(args.output).resolve())
    summary_path = e1_root / "summary.csv"
    baseline_rows = _summary_rows(summary_path)
    strongest_model, strongest_accuracy = _strongest_seed_zero_macro(
        baseline_rows, subjects
    )
    source_tree = collect_source_tree_manifest(ROOT)
    write_json(output / "source_tree_manifest.json", source_tree)
    write_json(
        output / "source_tree_summary.json",
        {"files": len(source_tree), "sha256": source_tree_digest(source_tree)},
    )
    write_json(output / "heldout_lock_manifest.json", v8_heldout_lock_manifest())

    rows: list[dict[str, Any]] = []
    for subject in subjects:
        for seed in seeds:
            source_dir = e1_root / "atcnet" / f"subject_{subject:02d}" / f"seed_{seed}"
            source_metrics = read_json(source_dir / "metrics.json")
            if source_metrics.get("status") != "completed":
                raise RuntimeError(f"incomplete source ATC run: {source_dir}")
            if source_metrics.get("protocol") != config["protocol"]:
                raise RuntimeError("ATC source protocol differs from the E2 anchor")
            if source_metrics.get("implementation_id") != config["source"]["implementation_id"]:
                raise RuntimeError("ATC source implementation is not the pinned official core")
            if source_metrics.get("evaluation_scope") != "Session-T nested outer-run OOF test":
                raise RuntimeError("ATC source is not Session-T nested OOF")
            if bool(source_metrics.get("session_e_accessed", True)):
                raise RuntimeError("ATC source accessed Session E")
            source_manifest = read_json(source_dir / "manifest.json")
            validate_run_artifact_manifest(
                source_dir,
                required_files=tuple(source_manifest["required_files"]),
                verify_hashes=True,
                verify_prediction_schema=True,
            )
            run_dir = ensure_dir(output / f"subject_{subject:02d}" / f"seed_{seed}")
            fold_suffixes = {
                "best.pt",
                "last.pt",
                "history.csv",
                "result.json",
                "outer_test_predictions.npz",
            }
            fold_files = tuple(
                name
                for name in source_manifest["required_files"]
                if name.startswith("fold_") and Path(name).name in fold_suffixes
            )
            if len(fold_files) != 6 * len(fold_suffixes):
                raise RuntimeError("ATC source does not contain six complete outer folds")
            linked_files = ("predictions.npz", "predictions.csv", *fold_files)
            source_files = {
                name: file_sha256(source_dir / name) for name in linked_files
            }
            resolved = {
                **config,
                "active_subject": subject,
                "active_seed": seed,
                "source_run_fingerprint": source_metrics["run_fingerprint"],
            }
            fingerprint = build_v8_run_fingerprint(
                resolved_run_config=resolved,
                source_tree=source_tree,
                data={"source_e1_metrics": file_sha256(source_dir / "metrics.json")},
                split={"source_split_manifest": file_sha256(source_dir / "split_manifest.json")},
                augmentation={"source_augmentation_manifest": file_sha256(source_dir / "augmentation_manifest.json")},
                prior={"policy": "none", "delay_auxiliary_enabled": False},
                checkpoint={"reuse_policy": config["source"]["reuse_policy"], **source_files},
                environment={"materialization": "no_training_no_inference"},
            )
            metrics = {
                **source_metrics,
                "stage": "E2",
                "model": "v8_atc_accuracy_anchor",
                "variant": "official_atc_ann_anchor",
                "source_stage": "E1",
                "source_model": "atcnet",
                "source_run_fingerprint": source_metrics["run_fingerprint"],
                "reuse_policy": config["source"]["reuse_policy"],
                "delay_auxiliary_enabled": False,
                "session_e_accessed": False,
                "openbmi_s2_accessed": False,
                "run_fingerprint": fingerprint["combined_sha256"],
            }
            write_v8_fingerprint(run_dir / "source_fingerprint.json", fingerprint)
            (run_dir / "resolved_config.yaml").write_text(
                yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
            )
            write_json(
                run_dir / "source_reference.json",
                {
                    "source_directory": str(source_dir),
                    "source_metrics_sha256": file_sha256(source_dir / "metrics.json"),
                    "source_run_fingerprint": source_metrics["run_fingerprint"],
                    "linked_files": source_files,
                    "new_training_performed": False,
                    "new_inference_performed": False,
                },
            )
            write_json(run_dir / "heldout_lock_manifest.json", v8_heldout_lock_manifest())
            for name in linked_files:
                (run_dir / name).parent.mkdir(parents=True, exist_ok=True)
                _link_or_copy(source_dir / name, run_dir / name)
            write_json(run_dir / "metrics.json", metrics)
            write_json(
                run_dir / "runtime_status.json",
                {"status": "completed", "completed_at": time.time(), "materialized": True},
            )
            (run_dir / "stdout.log").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            (run_dir / "stderr.log").write_text("", encoding="utf-8")
            write_run_artifact_manifest(
                run_dir, required_files=(*RUN_FILES, *fold_files)
            )
            rows.append(metrics)

    anchor_accuracy = float(np.mean([float(row["accuracy"]) for row in rows]))
    gap_pp = 100.0 * (strongest_accuracy - anchor_accuracy)
    gate = {
        "passed": bool(gap_pp <= float(config["gate"]["maximum_gap_pp"])),
        "strongest_e1_model": strongest_model,
        "strongest_e1_seed0_macro_accuracy": strongest_accuracy,
        "anchor_seed0_macro_accuracy": anchor_accuracy,
        "gap_pp": gap_pp,
        "maximum_gap_pp": float(config["gate"]["maximum_gap_pp"]),
        "result_reused_without_retraining": True,
        "session_e_accessed": False,
    }
    write_csv(output / "summary.csv", rows)
    write_json(output / "gate_report.json", gate)
    write_json(
        output / "campaign_status.json",
        {"status": "completed", "subjects": subjects, "seeds": seeds, "gate": gate},
    )
    print(json.dumps({"status": "completed", "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
