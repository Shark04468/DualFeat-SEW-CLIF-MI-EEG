#!/usr/bin/env python3
"""Independently reconstruct and audit the frozen V8 ensemble E6 campaign."""

from __future__ import annotations

import argparse
import csv
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
    sha256_fingerprint,
    validate_prediction_schema,
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_maintenance import (  # noqa: E402
    validate_ensemble_maintenance_manifest,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    mapping_sha256,
    source_tree_digest,
    validate_v8_freeze_manifest,
    validate_v8_fingerprint,
)
from dpc_snn.experiments.v8_statistics import (  # noqa: E402
    pair_subject_seed_rows,
    paired_delta_summary,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import (  # noqa: E402
    classification_metrics,
    paired_prediction_comparison,
)
from scripts.run_v8_e6_ensemble_frozen import (  # noqa: E402
    PREDICTION_BASES,
    RUN_FILES,
)


AUDIT_FILES = (
    "manifest.json",
    "audit_report.json",
    "model_summary.csv",
    "paired_subject_seed.csv",
    "paired_trial_diagnostics.csv",
)
BASE_TO_ARM = {
    "predictions": "primary",
    "matched_ann_predictions": "matched_ann",
    "anchor_predictions": "anchor",
    "atcnet_predictions": "atcnet",
    "fbcnet_predictions": "fbcnet",
    "decoder_snn_predictions": "decoder_snn",
    "decoder_ann_predictions": "decoder_ann",
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty campaign table: {path}")
    return rows


def _close(first: float, second: float, tolerance: float = 1e-10) -> bool:
    return bool(np.isclose(float(first), float(second), rtol=0.0, atol=tolerance))


def _prediction(run: Path, basename: str) -> dict[str, np.ndarray]:
    npz = run / f"{basename}.npz"
    validate_prediction_schema(npz, run / f"{basename}.csv")
    with np.load(npz, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _identity(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> None:
    for field in ("subject", "session", "run", "trial_id", "label", "seed"):
        if not np.array_equal(first[field], second[field]):
            raise RuntimeError(f"paired E6 prediction identity differs at {field}")


def _recomputed_metrics(prediction: dict[str, np.ndarray]) -> dict[str, float]:
    labels = prediction["label"]
    pred = prediction["pred"]
    return {
        **classification_metrics(labels, pred, n_classes=4),
        **multiclass_calibration_metrics(prediction["logits"], labels),
    }


def _checkpoint_digest(path: Path) -> str:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"checkpoint is not a non-empty state dict: {path}")
    return sha256_fingerprint(mapping_sha256(state))


def _audit_run(
    run: Path,
    *,
    freeze_sha256: str,
    maintenance_sha256: str,
    source_tree_sha256: str,
) -> dict[str, Any]:
    validate_run_artifact_manifest(
        run,
        required_files=RUN_FILES,
        verify_hashes=True,
        verify_prediction_schema=True,
    )
    metrics = read_json(run / "metrics.json")
    validate_v8_fingerprint(read_json(run / "source_fingerprint.json"))
    if source_tree_digest(read_json(run / "source_tree_manifest.json")) != source_tree_sha256:
        raise RuntimeError(f"run source tree differs from the audited snapshot: {run}")
    if (
        metrics.get("status") != "completed"
        or metrics.get("stage") != "E6_ENSEMBLE"
        or metrics.get("freeze_sha256") != freeze_sha256
        or metrics.get("source_maintenance_sha256") != maintenance_sha256
        or metrics.get("heldout_e_selected_checkpoint") is not False
    ):
        raise RuntimeError(f"invalid E6 metrics contract: {run}")
    state = read_json(run / "state_audit.json")
    access = read_json(run / "data_access_manifest.json")
    identity = read_json(run / "data_identity.json")
    runtime = read_json(run / "runtime_status.json")
    checkpoint_hashes = {
        name: _checkpoint_digest(run / f"{name}.pt")
        for name in ("atcnet", "fbcnet", "sew_clif", "ann_sew")
    }
    if (
        state.get("identical") is not True
        or state.get("checkpoint_before_session_e") != state.get("state_after_session_e")
        or state.get("checkpoint_state_sha256")
        != state.get("checkpoint_before_session_e")
        or checkpoint_hashes != state.get("checkpoint_state_sha256")
        or state.get("changed_state_components") != {}
        or int(state.get("constraint_projection_counts", {}).get("fbcnet", 0)) < 1
        or access.get("session_e_arrays_first_loaded_after_all_checkpoint_hashes")
        != state.get("checkpoint_before_session_e")
        or access.get("session_e_byte_hash_computed_before_training")
        != identity.get("evaluation", {}).get("sha256")
        or access.get("session_e_checkpoint_selection") is not False
        or access.get("session_e_gradient_updates") is not False
        or access.get("evaluation_session") != "E"
        or access.get("train_session") != "T"
        or int(access.get("train_trials", 0)) != 288
        or int(access.get("evaluation_trials", 0)) != 288
        or access.get("openbmi_s2_accessed") is not False
        or runtime.get("status") != "completed"
        or runtime.get("session_e_arrays_loaded") is not True
        or runtime.get("session_e_identity_hash_computed") is not True
    ):
        raise RuntimeError(f"held-out state/access audit failed: {run}")
    if (
        identity.get("train", {}).get("logical_key")
        != f"session_t/A{int(metrics['subject']):02d}.npz"
        or identity.get("evaluation", {}).get("logical_key")
        != f"session_e/A{int(metrics['subject']):02d}.npz"
        or identity.get("train", {}).get("semantic_load_before_training") is not True
        or identity.get("evaluation", {}).get("byte_hash_before_training") is not True
        or identity.get("evaluation", {}).get("semantic_load_before_checkpoint") is not False
    ):
        raise RuntimeError(f"T/E data identity contract failed: {run}")

    predictions = {basename: _prediction(run, basename) for basename in PREDICTION_BASES}
    primary = predictions["predictions"]
    if primary["pred"].size != 288 or len(np.unique(primary["trial_id"])) != 288:
        raise RuntimeError(f"E6 run does not contain 288 unique Session-E trials: {run}")
    for prediction in predictions.values():
        _identity(primary, prediction)
    for basename, prediction in predictions.items():
        arm = BASE_TO_ARM[basename]
        recomputed = _recomputed_metrics(prediction)
        saved = metrics["arms"][arm]
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "kappa",
            "macro_f1",
            "negative_log_likelihood",
            "brier_score",
            "ece",
            "maximum_calibration_error",
        ):
            if not _close(recomputed[metric], saved[metric], tolerance=2e-7):
                raise RuntimeError(f"E6 metric drift at {run}, arm={arm}, metric={metric}")
    for metric in (
        "accuracy",
        "balanced_accuracy",
        "kappa",
        "macro_f1",
        "negative_log_likelihood",
        "ece",
    ):
        if not _close(metrics[metric], metrics["arms"]["primary"][metric], tolerance=2e-7):
            raise RuntimeError(f"top-level primary metric drift at {run}: {metric}")
    return {"metrics": metrics, "predictions": predictions}


def _validate_campaign_coverage(
    status: dict[str, Any], *, canary: bool
) -> tuple[list[int], list[int], int]:
    expected_subjects = [1] if canary else list(range(1, 10))
    expected_seeds = [0] if canary else list(range(5))
    expected_runs = len(expected_subjects) * len(expected_seeds)
    expected_full_contract = not canary
    if (
        status.get("status") != "completed"
        or status.get("full_registered_contract") is not expected_full_contract
        or status.get("runs") != expected_runs
        or status.get("subjects") != expected_subjects
        or status.get("seeds") != expected_seeds
    ):
        raise RuntimeError("E6 campaign coverage differs from the requested audit scope")
    return expected_subjects, expected_seeds, expected_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e6", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--maintenance", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()

    e6 = Path(args.e6).resolve()
    output = ensure_dir(Path(args.output).resolve())
    freeze = validate_v8_freeze_manifest(Path(args.freeze).resolve())
    active_source_digest = source_tree_digest(collect_source_tree_manifest(ROOT))
    maintenance = validate_ensemble_maintenance_manifest(
        read_json(Path(args.maintenance).resolve()),
        expected_parent_freeze_sha256=freeze["combined_sha256"],
        expected_current_source_tree_sha256=active_source_digest,
    )
    status = read_json(e6 / "campaign_status.json")
    expected_subjects, expected_seeds, expected_runs = _validate_campaign_coverage(
        status, canary=bool(args.canary)
    )
    if (
        status.get("freeze_sha256") != freeze["combined_sha256"]
        or status.get("source_maintenance_sha256") != maintenance["combined_sha256"]
        or status.get("session_e_used_for_selection") is not False
        or status.get("session_e_gradient_updates") is not False
        or status.get("openbmi_s2_accessed") is not False
    ):
        raise RuntimeError("E6 ensemble campaign is incomplete or not frozen")
    summary_rows = _rows(e6 / "summary.csv")
    expected = {
        (subject, seed) for subject in expected_subjects for seed in expected_seeds
    }
    observed = {(int(row["subject"]), int(row["seed"])) for row in summary_rows}
    if observed != expected or len(summary_rows) != expected_runs:
        raise RuntimeError("E6 summary coverage differs from the requested audit scope")
    summary_index = {
        (int(row["subject"]), int(row["seed"])): row for row in summary_rows
    }

    arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in BASE_TO_ARM.values()}
    prediction_index: dict[tuple[int, int], dict[str, dict[str, np.ndarray]]] = {}
    for subject, seed in sorted(expected):
        run = e6 / f"subject_{subject:02d}" / f"seed_{seed}"
        audited = _audit_run(
            run,
            freeze_sha256=freeze["combined_sha256"],
            maintenance_sha256=maintenance["combined_sha256"],
            source_tree_sha256=active_source_digest,
        )
        metrics = audited["metrics"]
        campaign_row = summary_index[(subject, seed)]
        summary_fields = {
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "kappa": metrics["kappa"],
            "macro_f1": metrics["macro_f1"],
            "negative_log_likelihood": metrics["negative_log_likelihood"],
            "ece": metrics["ece"],
            "anchor_accuracy": metrics["arms"]["anchor"]["accuracy"],
            "atcnet_accuracy": metrics["arms"]["atcnet"]["accuracy"],
            "fbcnet_accuracy": metrics["arms"]["fbcnet"]["accuracy"],
            "matched_ann_accuracy": metrics["arms"]["matched_ann"]["accuracy"],
            "snn_mean_firing_rate": metrics["snn_mean_firing_rate"],
        }
        for field, expected_value in summary_fields.items():
            if not _close(float(campaign_row[field]), expected_value, tolerance=2e-7):
                raise RuntimeError(
                    f"campaign summary drift at subject={subject}, seed={seed}, field={field}"
                )
        prediction_index[(subject, seed)] = {
            BASE_TO_ARM[basename]: prediction
            for basename, prediction in audited["predictions"].items()
        }
        for arm, values in metrics["arms"].items():
            arm_rows[arm].append(
                {"subject": subject, "seed": seed, "accuracy": values["accuracy"]}
            )

    comparisons = ("atcnet", "fbcnet", "anchor", "matched_ann")
    paired_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    comparison_summaries: dict[str, Any] = {}
    for index, comparator in enumerate(comparisons):
        paired = pair_subject_seed_rows(arm_rows[comparator], arm_rows["primary"])
        name = f"primary_minus_{comparator}"
        comparison_summaries[name] = paired_delta_summary(
            paired, seed=20260803 + index, bootstrap_samples=10_000
        )
        paired_rows.extend({"comparison": name, **row} for row in paired)
        for subject, seed in sorted(expected):
            first = prediction_index[(subject, seed)][comparator]
            second = prediction_index[(subject, seed)]["primary"]
            trial_rows.append(
                {
                    "comparison": name,
                    "subject": subject,
                    "seed": seed,
                    **paired_prediction_comparison(
                        second["label"], first["pred"], second["pred"]
                    ),
                }
            )

    model_summary: list[dict[str, Any]] = []
    for arm, rows in arm_rows.items():
        subject_means = [
            float(
                np.mean(
                    [
                        float(row["accuracy"])
                        for row in rows
                        if int(row["subject"]) == subject
                    ]
                )
            )
            for subject in expected_subjects
        ]
        model_summary.append(
            {
                "arm": arm,
                "subject_macro_accuracy": float(np.mean(subject_means)),
                "subject_median_accuracy": float(np.median(subject_means)),
                "subject_standard_deviation": (
                    float(np.std(subject_means, ddof=1))
                    if len(subject_means) > 1
                    else 0.0
                ),
                "minimum_subject_accuracy": float(np.min(subject_means)),
                "maximum_subject_accuracy": float(np.max(subject_means)),
            }
        )
    model_summary.sort(key=lambda row: (-row["subject_macro_accuracy"], row["arm"]))
    report = {
        "status": "passed",
        "stage": "E6_ENSEMBLE_AUDIT",
        "freeze_sha256": freeze["combined_sha256"],
        "source_maintenance_sha256": maintenance["combined_sha256"],
        "runs_audited": expected_runs,
        "canary": bool(args.canary),
        "prediction_arms_per_run": len(PREDICTION_BASES),
        "all_artifact_hashes_verified": True,
        "all_trial_identities_paired": True,
        "all_metrics_recomputed": True,
        "all_model_states_unchanged_during_session_e": True,
        "comparisons": comparison_summaries,
        "post_session_e_tuning_detected": False,
        "openbmi_s2_accessed": False,
    }
    write_csv(output / "model_summary.csv", model_summary)
    write_csv(output / "paired_subject_seed.csv", paired_rows)
    write_csv(output / "paired_trial_diagnostics.csv", trial_rows)
    write_json(output / "audit_report.json", report)
    write_run_artifact_manifest(output, required_files=AUDIT_FILES)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
