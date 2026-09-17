#!/usr/bin/env python3
"""Independently audit the V8 ensemble E9 ablation campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.analysis.v8_utility import multiclass_calibration_metrics  # noqa: E402
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    sha256_fingerprint,
    validate_prediction_schema,
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    mapping_sha256,
    source_tree_digest,
)
from dpc_snn.experiments.v8_statistics import (  # noqa: E402
    pair_subject_seed_rows,
    paired_delta_summary,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from scripts.run_v8_e9_ensemble_ablations import (  # noqa: E402
    ARMS,
    CAMPAIGN_FILES,
    PREDICTION_NAMES,
    RUN_FILES,
)


AUDIT_FILES = (
    "manifest.json",
    "audit_report.json",
    "paired_subject_seed.csv",
)


def _close(first: Any, second: Any, tolerance: float = 2e-7) -> bool:
    return bool(np.isclose(float(first), float(second), rtol=0.0, atol=tolerance))


def _state_digest(path: Path) -> str:
    state = torch.load(path, map_location="cpu", weights_only=True)
    return sha256_fingerprint(mapping_sha256(state))


def _prediction(run: Path, arm: str) -> dict[str, np.ndarray]:
    path = run / f"{PREDICTION_NAMES[arm]}.npz"
    validate_prediction_schema(path, path.with_suffix(".csv"))
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _audit_run(run: Path, e6_run: Path) -> dict[str, Any]:
    validate_run_artifact_manifest(
        run,
        required_files=RUN_FILES,
        verify_hashes=True,
        verify_prediction_schema=True,
    )
    metrics = read_json(run / "metrics.json")
    state = read_json(run / "state_audit.json")
    access = read_json(run / "data_access_manifest.json")
    runtime = read_json(run / "runtime_status.json")
    checkpoint = {
        name: _state_digest(run / f"{name}.pt")
        for name in ("atcnet", "fbcnet", "sew_clif", "ann_sew")
    }
    if (
        metrics.get("status") != "completed"
        or metrics.get("variant") != "no_augmentation"
        or metrics.get("post_e6_explanatory_ablation") is not True
        or state.get("identical") is not True
        or state.get("checkpoint_before_session_e") != state.get("state_after_session_e")
        or state.get("checkpoint_before_session_e") != checkpoint
        or access.get("session_e_checkpoint_selection") is not False
        or access.get("session_e_gradient_updates") is not False
        or access.get("openbmi_s2_gradient_updates") is not False
        or runtime.get("status") != "completed"
    ):
        raise RuntimeError(f"invalid E9 state/access contract: {run}")
    e6_metrics = read_json(e6_run / "metrics.json")
    if metrics.get("reference_e6_run_fingerprint") != e6_metrics.get("run_fingerprint"):
        raise RuntimeError(f"E9 references another E6 run: {run}")
    predictions = {arm: _prediction(run, arm) for arm in ARMS}
    primary = predictions["primary"]
    for arm, prediction in predictions.items():
        for field in ("subject", "session", "run", "trial_id", "label", "seed"):
            if not np.array_equal(primary[field], prediction[field]):
                raise RuntimeError(f"E9 prediction identity mismatch: {run} {arm} {field}")
        recomputed = {
            **classification_metrics(
                prediction["label"], prediction["pred"], n_classes=4
            ),
            **multiclass_calibration_metrics(prediction["logits"], prediction["label"]),
        }
        for field in (
            "accuracy",
            "balanced_accuracy",
            "kappa",
            "macro_f1",
            "negative_log_likelihood",
            "brier_score",
            "ece",
            "maximum_calibration_error",
        ):
            if not _close(recomputed[field], metrics["arms"][arm][field]):
                raise RuntimeError(f"E9 metric drift: {run} {arm} {field}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e9", required=True)
    parser.add_argument("--e6", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    e9 = Path(args.e9).resolve()
    e6 = Path(args.e6).resolve()
    output = ensure_dir(Path(args.output).resolve())
    validate_run_artifact_manifest(e9, required_files=CAMPAIGN_FILES, verify_hashes=True)
    status = read_json(e9 / "campaign_status.json")
    if (
        status.get("status") != "completed"
        or status.get("full_registered_contract") is not True
        or status.get("runs") != 45
        or status.get("session_e_checkpoint_selection") is not False
        or status.get("openbmi_s2_model_updates") is not False
    ):
        raise RuntimeError("E9 audit requires the full 9-subject x 5-seed campaign")
    active_digest = source_tree_digest(collect_source_tree_manifest(ROOT))
    if source_tree_digest(read_json(e9 / "source_tree_manifest.json")) != active_digest:
        raise RuntimeError("active source differs from the E9 campaign source")
    rows = []
    e6_rows = []
    for subject in range(1, 10):
        for seed in range(5):
            e6_run = e6 / f"subject_{subject:02d}" / f"seed_{seed}"
            rows.append(
                _audit_run(
                    e9 / "no_augmentation" / f"subject_{subject:02d}" / f"seed_{seed}",
                    e6_run,
                )
            )
            e6_rows.append(read_json(e6_run / "metrics.json"))
    paired_rows = []
    comparisons = {}
    no_aug = [
        {"subject": row["subject"], "seed": row["seed"], "accuracy": row["arms"]["primary"]["accuracy"]}
        for row in rows
    ]
    primary = [
        {"subject": row["subject"], "seed": row["seed"], "accuracy": row["arms"]["primary"]["accuracy"]}
        for row in e6_rows
    ]
    pairs = pair_subject_seed_rows(no_aug, primary, value="accuracy")
    key = "augmentation_contribution_e6_minus_no_augmentation"
    comparisons[key] = paired_delta_summary(pairs, seed=20260901)
    paired_rows.extend({"comparison": key, **pair} for pair in pairs)
    for index, arm in enumerate(("matched_ann", "anchor", "atcnet", "fbcnet", "decoder_snn", "decoder_ann")):
        comparator = [
            {"subject": row["subject"], "seed": row["seed"], "accuracy": row["arms"][arm]["accuracy"]}
            for row in e6_rows
        ]
        pairs = pair_subject_seed_rows(comparator, primary, value="accuracy")
        key = f"primary_minus_{arm}"
        comparisons[key] = paired_delta_summary(pairs, seed=20260902 + index)
        paired_rows.extend({"comparison": key, **pair} for pair in pairs)
    write_csv(output / "paired_subject_seed.csv", paired_rows)
    if read_json(e9 / "comparisons.json") != comparisons:
        raise RuntimeError("E9 campaign comparisons differ from independent recomputation")
    links = read_json(e9 / "evidence_links.json")
    for evidence in links.values():
        if file_sha256(Path(evidence["path"])) != evidence["sha256"]:
            raise RuntimeError(f"E9 linked evidence changed: {evidence['path']}")
    report = {
        "status": "passed",
        "stage": "E9_ENSEMBLE_ABLATION_AUDIT",
        "runs_audited": 45,
        "all_artifact_hashes_verified": True,
        "all_model_states_unchanged_during_session_e": True,
        "all_metrics_recomputed": True,
        "all_reused_controls_bound_to_e6": True,
        "all_linked_evidence_hashes_verified": True,
        "comparisons": comparisons,
        "post_e6_explanatory_ablation": True,
    }
    write_json(output / "audit_report.json", report)
    write_run_artifact_manifest(output, required_files=AUDIT_FILES)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
