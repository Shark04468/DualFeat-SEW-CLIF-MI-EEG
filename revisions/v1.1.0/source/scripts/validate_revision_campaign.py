"""Validate every parent and reviewer barrier before declaring four-GPU completion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.utils.io import read_json, write_json


def _sealed(path: Path) -> dict:
    value = read_json(path)
    if value.get("status") != "sealed":
        raise RuntimeError(f"barrier is not sealed: {path}")
    return value


def _reviewer_expected(config: dict) -> int:
    total = 0
    for dataset_config in config["datasets"].values():
        for variant, spec in config["variants"].items():
            del variant
            budgets = (
                config["fixed_budget_controls"]
                if spec["budget_group"] == "fixed_budget_controls"
                else dataset_config["equal_update_budgets"]
            )
            total += len(dataset_config["subjects"]) * len(config["seeds"]) * len(budgets)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--reviewer-config", required=True)
    parser.add_argument("--binary-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    reviewer_config = yaml.safe_load(
        Path(args.reviewer_config).resolve().read_text(encoding="utf-8")
    )
    binary_config = yaml.safe_load(Path(args.binary_config).resolve().read_text(encoding="utf-8"))
    publication = root / "publication"
    for dataset in ("bci2a", "openbmi"):
        status = read_json(publication / dataset / "campaign_status.json")
        if status.get(
            "status"
        ) != "baseline_evaluation_complete_pending_fusion_analysis" or status.get(
            "completed_runs"
        ) != status.get("expected_runs"):
            raise RuntimeError(f"publication parent is incomplete: {dataset}")
    v30 = _sealed(root / "v30_recovered" / "checkpoint_barrier.json")
    v31 = _sealed(root / "v31_recovered" / "checkpoint_barrier.json")
    reviewer = _sealed(root / "reviewer_controls" / "CHECKPOINT_BARRIER.json")
    binary = _sealed(root / "bci2a_binary" / "CHECKPOINT_BARRIER.json")
    v30_config = yaml.safe_load(
        (ROOT / "configs" / "experiments" / "v30_bnci2014_004_recovery.yaml").read_text(
            encoding="utf-8"
        )
    )
    v30_expected = (
        len(v30_config["dataset"]["subjects"])
        * len(v30_config["seeds"])
        * (len(v30_config["teacher_models"]) + len(v30_config["variants"]))
    )
    if int(v30.get("expected_checkpoints", -1)) != v30_expected:
        raise RuntimeError("V30 checkpoint count is incomplete")
    v31_config = yaml.safe_load(
        (ROOT / "configs" / "experiments" / "v31_decoder_learning_curve.yaml").read_text(
            encoding="utf-8"
        )
    )
    v31_expected = 0
    for dataset, dataset_config in v31_config["datasets"].items():
        class_floor = {
            "bci2a": 72,
            "openbmi": 50,
            "bnci2014_004": 200,
        }
        numeric = sum(
            1
            for value in v31_config["numeric_budgets_per_class"]
            if int(value) < class_floor[dataset]
        )
        budgets = numeric + int(bool(v31_config["include_all_available"]))
        v31_expected += (
            len(dataset_config["subjects"])
            * len(v31_config["seeds"])
            * budgets
            * len(v31_config["variants"])
        )
    if int(v31.get("checkpoint_count", -1)) != v31_expected:
        raise RuntimeError(
            f"V31 checkpoint count mismatch: {v31.get('checkpoint_count')} != {v31_expected}"
        )
    reviewer_expected = _reviewer_expected(reviewer_config)
    if int(reviewer.get("checkpoint_count", -1)) != reviewer_expected:
        raise RuntimeError("reviewer-control checkpoint count is incomplete")
    binary_expected = (
        len(binary_config["dataset"]["subjects"])
        * len(binary_config["seeds"])
        * (len(binary_config["numeric_budgets_per_class"]) + 1)
        * len(binary_config["variants"])
    )
    if int(binary.get("checkpoint_count", -1)) != binary_expected:
        raise RuntimeError("REV-E5 checkpoint count is incomplete")
    expected_labels = sum(
        len(dataset_config["subjects"]) for dataset_config in reviewer_config["datasets"].values()
    )
    labels = list(
        (Path(reviewer_config["parents"]["training_labels"])).glob(
            "*/subject_*/training_labels.npz"
        )
    )
    if len(labels) != expected_labels:
        raise RuntimeError(f"recovery-label count mismatch: {len(labels)} != {expected_labels}")
    for path in labels:
        metadata = read_json(path.with_suffix(".json"))
        if metadata.get("schema") != "dpc-snn-recovery-training-labels/v2":
            raise RuntimeError(f"training label lacks trial-ID provenance: {path}")
    reviewer_summary = read_json(root / "reviewer_controls" / "aggregate" / "summary.json")
    binary_summary = read_json(root / "bci2a_binary" / "aggregate" / "summary.json")
    utility = read_json(root / "reviewer_controls" / "utility" / "utility_profile.json")
    if reviewer_summary.get("status") != "completed" or binary_summary.get("status") != "completed":
        raise RuntimeError("reviewer aggregation is incomplete")
    if utility.get("status") != "completed" or len(utility.get("profiles", [])) < 4:
        raise RuntimeError("REV-E4 utility profile is incomplete")
    gpu_name = str(utility.get("hardware", {}).get("gpu_name", ""))
    if "5090" not in gpu_name:
        raise RuntimeError(f"publication latency was not measured on RTX 5090: {gpu_name}")
    payload = {
        "schema": "dpc-snn-major-revision-readiness/v1",
        "status": "READY_FOR_MANUSCRIPT_REVISION",
        "publication_datasets": 2,
        "v30_checkpoints": v30_expected,
        "v31_checkpoints": v31_expected,
        "reviewer_checkpoints": reviewer_expected,
        "binary_checkpoints": binary_expected,
        "recovery_label_subjects": expected_labels,
        "latency_gpu": gpu_name,
        "scientific_scope": "retrospective recovered parent; no renewed blind-confirmation claim",
    }
    write_json(Path(args.output).resolve(), payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
