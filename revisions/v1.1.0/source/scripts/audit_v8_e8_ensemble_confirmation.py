#!/usr/bin/env python3
"""Independently audit the V8 ensemble OpenBMI external confirmation."""

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
from dpc_snn.data.v8_openbmi import OPENBMI_FIXED_INPUT_ADAPTER  # noqa: E402
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    sha256_fingerprint,
    validate_prediction_schema,
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_ensemble import (  # noqa: E402
    validate_ensemble_freeze_contract,
)
from dpc_snn.experiments.v8_ensemble_followup import (  # noqa: E402
    resolve_frozen_channel_basis,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    mapping_sha256,
    source_tree_digest,
    validate_v8_external_unlock_manifest,
    validate_v8_freeze_manifest,
)
from dpc_snn.experiments.v8_statistics import (  # noqa: E402
    pair_subject_seed_rows,
    paired_delta_summary,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from scripts.run_v8_e8_ensemble_confirmation import (  # noqa: E402
    ARMS,
    CAMPAIGN_FILES,
    CHECKPOINTS,
    FINAL_FILES,
    TRAINING_FILES,
)


AUDIT_FILES = (
    "manifest.json",
    "audit_report.json",
    "model_summary.csv",
    "paired_subject_seed.csv",
)


def _close(first: Any, second: Any, tolerance: float = 2e-7) -> bool:
    return bool(np.isclose(float(first), float(second), rtol=0.0, atol=tolerance))


def _state_digest(path: Path) -> str:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"invalid E8 checkpoint: {path}")
    return sha256_fingerprint(mapping_sha256(state))


def _prediction(run: Path, arm: str) -> dict[str, np.ndarray]:
    path = run / f"{arm}_predictions.npz"
    validate_prediction_schema(path, path.with_suffix(".csv"))
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _training_manifest(run: Path) -> dict[str, Any]:
    manifest = read_json(run / "training_artifact_manifest.json")
    body = {
        "schema": "dpc-snn-v8-e8-training-checkpoint/v1",
        "files": {name: file_sha256(run / name) for name in TRAINING_FILES},
    }
    expected = {**body, "combined_sha256": sha256_fingerprint(body)}
    if manifest != expected:
        raise RuntimeError(f"E8 training manifest drift: {run}")
    return manifest


def _validate_input_adapter_manifest(
    manifest: dict[str, Any],
    *,
    expected_channels: list[str],
    expected_adapter: dict[str, Any],
) -> None:
    recorded = manifest.get("input_adapter", {})
    if (
        manifest.get("channel_names") != expected_channels
        or recorded.get("policy") != expected_adapter["policy"]
        or recorded.get("applied_derived_channels")
        != expected_adapter["derived_channels"]
    ):
        raise RuntimeError("E8 data-access channel projection differs from the unlock")
    indices = manifest.get("source_channel_indices", [])
    fcz_index = expected_channels.index("FCz")
    if len(indices) != len(expected_channels) or indices[fcz_index] is not None:
        raise RuntimeError("E8 FCz was not recorded as a fixed derived channel")


def _audit_run(
    run: Path,
    *,
    subject: int,
    seed: int,
    barrier: dict[str, Any],
    barrier_record: dict[str, Any],
    unlock_sha256: str,
    expected_channels: list[str],
    expected_adapter: dict[str, Any],
) -> dict[str, Any]:
    validate_run_artifact_manifest(
        run,
        required_files=FINAL_FILES,
        verify_hashes=True,
        verify_prediction_schema=True,
    )
    training_manifest = _training_manifest(run)
    if training_manifest["combined_sha256"] != barrier_record["training_manifest_sha256"]:
        raise RuntimeError(f"E8 barrier training hash mismatch: {run}")
    for name in CHECKPOINTS:
        if file_sha256(run / f"{name}.pt") != barrier_record["checkpoint_file_sha256"][name]:
            raise RuntimeError(f"E8 barrier checkpoint hash mismatch: {run} {name}")
    training_runtime = read_json(run / "training_runtime_status.json")
    runtime = read_json(run / "runtime_status.json")
    access = read_json(run / "data_access_manifest.json")
    state = read_json(run / "state_audit.json")
    training_state = read_json(run / "training_state_audit.json")
    _validate_input_adapter_manifest(
        read_json(run / "s1_access_manifest.json"),
        expected_channels=expected_channels,
        expected_adapter=expected_adapter,
    )
    _validate_input_adapter_manifest(
        read_json(run / "s2_access_manifest.json"),
        expected_channels=expected_channels,
        expected_adapter=expected_adapter,
    )
    checkpoint_digests = {
        name: _state_digest(run / f"{name}.pt") for name in CHECKPOINTS
    }
    if (
        training_runtime.get("status") != "training_completed_s2_unopened"
        or training_runtime.get("openbmi_s2_accessed") is not False
        or float(training_runtime.get("completed_at", float("inf")))
        > float(barrier["created_at"])
        or runtime.get("status") != "completed"
        or runtime.get("model_updates") is not False
        or state.get("identical") is not True
        or state.get("model_updates") is not False
        or state.get("checkpoint_before_s2") != state.get("state_after_s2")
        or state.get("checkpoint_before_s2") != checkpoint_digests
        or training_state.get("checkpoint_state_sha256") != checkpoint_digests
        or access.get("all_training_complete_before_s2_barrier_sha256")
        != barrier["combined_sha256"]
        or access.get("s2_checkpoint_selection") is not False
        or access.get("s2_gradient_updates") is not False
    ):
        raise RuntimeError(f"E8 checkpoint/S2 state contract failed: {run}")
    metrics = read_json(run / "metrics.json")
    if (
        metrics.get("status") != "completed"
        or metrics.get("subject") != subject
        or metrics.get("seed") != seed
        or metrics.get("external_unlock_sha256") != unlock_sha256
        or metrics.get("checkpoint_barrier_sha256") != barrier["combined_sha256"]
        or metrics.get("s2_selected_checkpoint") is not False
    ):
        raise RuntimeError(f"invalid E8 metrics contract: {run}")
    predictions = {arm: _prediction(run, arm) for arm in ARMS}
    primary = predictions["primary"]
    if primary["label"].size == 0 or len(np.unique(primary["trial_id"])) != primary["label"].size:
        raise RuntimeError(f"invalid E8 trial coverage: {run}")
    for arm, prediction in predictions.items():
        for field in ("subject", "session", "run", "trial_id", "label", "seed"):
            if not np.array_equal(primary[field], prediction[field]):
                raise RuntimeError(f"E8 paired identity differs: {run} {arm} {field}")
        recomputed = {
            **classification_metrics(
                prediction["label"], prediction["pred"], n_classes=2
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
                raise RuntimeError(f"E8 metric drift: {run} {arm} {field}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e8", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    e8 = Path(args.e8).resolve()
    output = ensure_dir(Path(args.output).resolve())
    validate_run_artifact_manifest(e8, required_files=CAMPAIGN_FILES, verify_hashes=True)
    status = read_json(e8 / "campaign_status.json")
    if (
        status.get("status") != "completed"
        or status.get("full_registered_contract") is not True
        or status.get("runs") != 210
        or status.get("all_training_complete_before_s2") is not True
        or status.get("s2_checkpoint_selection") is not False
        or status.get("s2_gradient_updates") is not False
    ):
        raise RuntimeError("E8 audit requires the full sealed 42-subject x 5-seed campaign")
    active_digest = source_tree_digest(collect_source_tree_manifest(ROOT))
    if source_tree_digest(read_json(e8 / "source_tree_manifest.json")) != active_digest:
        raise RuntimeError("active source differs from the E8 campaign source")
    freeze = validate_ensemble_freeze_contract(
        validate_v8_freeze_manifest(e8 / "freeze_manifest.json")
    )
    unlock = validate_v8_external_unlock_manifest(
        e8 / "external_unlock_manifest.json",
        expected_source_tree_sha256=active_digest,
        expected_parent_freeze_sha256=freeze["combined_sha256"],
    )
    expected_channels = resolve_frozen_channel_basis(freeze)
    if unlock["architecture_adaptation"].get("ordered_channels") != expected_channels:
        raise RuntimeError("E8 unlock sensor order differs from the frozen basis")
    expected_adapter = unlock["architecture_adaptation"].get(
        "input_channel_adapter"
    )
    if expected_adapter != OPENBMI_FIXED_INPUT_ADAPTER:
        raise RuntimeError("E8 unlock input-channel adapter is not the sealed projection")
    barrier = read_json(e8 / "checkpoint_barrier.json")
    barrier_body = {key: value for key, value in barrier.items() if key != "combined_sha256"}
    if (
        barrier.get("combined_sha256") != sha256_fingerprint(barrier_body)
        or barrier.get("all_training_complete_before_s2") is not True
        or barrier.get("runs") != 210
        or barrier.get("external_unlock_sha256") != unlock["combined_sha256"]
    ):
        raise RuntimeError("E8 checkpoint barrier digest or coverage is invalid")
    records = {
        (int(row["subject"]), int(row["seed"])): row for row in barrier["records"]
    }
    expected_keys = {
        (int(subject), int(seed))
        for subject in barrier["subjects"]
        for seed in barrier["seeds"]
    }
    if set(records) != expected_keys:
        raise RuntimeError("E8 checkpoint barrier subject-seed coverage is incomplete")
    rows = []
    for subject, seed in sorted(expected_keys):
        rows.append(
            _audit_run(
                e8 / f"subject_{subject:02d}" / f"seed_{seed}",
                subject=subject,
                seed=seed,
                barrier=barrier,
                barrier_record=records[(subject, seed)],
                unlock_sha256=unlock["combined_sha256"],
                expected_channels=expected_channels,
                expected_adapter=expected_adapter,
            )
        )
    model_rows = []
    for arm in ARMS:
        subject_means = np.asarray(
            [
                np.mean(
                    [
                        float(row["arms"][arm]["accuracy"])
                        for row in rows
                        if int(row["subject"]) == subject
                    ]
                )
                for subject in barrier["subjects"]
            ],
            dtype=np.float64,
        )
        model_rows.append(
            {
                "arm": arm,
                "subject_macro_accuracy": float(subject_means.mean()),
                "subject_median_accuracy": float(np.median(subject_means)),
                "subject_standard_deviation": float(subject_means.std(ddof=1)),
                "minimum_subject_accuracy": float(subject_means.min()),
                "maximum_subject_accuracy": float(subject_means.max()),
            }
        )
    write_csv(output / "model_summary.csv", model_rows)
    primary = [
        {"subject": row["subject"], "seed": row["seed"], "accuracy": row["arms"]["primary"]["accuracy"]}
        for row in rows
    ]
    paired_rows = []
    comparisons = {}
    for index, arm in enumerate(("matched_ann", "anchor", "atcnet", "fbcnet")):
        comparator = [
            {"subject": row["subject"], "seed": row["seed"], "accuracy": row["arms"][arm]["accuracy"]}
            for row in rows
        ]
        pairs = pair_subject_seed_rows(comparator, primary, value="accuracy")
        key = f"primary_minus_{arm}"
        comparisons[key] = paired_delta_summary(pairs, seed=20260820 + index)
        paired_rows.extend({"comparison": key, **pair} for pair in pairs)
    write_csv(output / "paired_subject_seed.csv", paired_rows)
    if status.get("comparisons") != comparisons:
        raise RuntimeError("E8 campaign comparisons differ from independent recomputation")
    report = {
        "status": "passed",
        "stage": "E8_ENSEMBLE_AUDIT",
        "runs_audited": len(rows),
        "subjects": len(barrier["subjects"]),
        "seeds_per_subject": len(barrier["seeds"]),
        "all_artifact_hashes_verified": True,
        "all_training_completed_before_any_s2": True,
        "all_checkpoint_states_unchanged_during_s2": True,
        "all_trial_identities_paired": True,
        "all_metrics_recomputed": True,
        "checkpoint_barrier_sha256": barrier["combined_sha256"],
        "external_unlock_sha256": unlock["combined_sha256"],
        "comparisons": comparisons,
        "post_s2_tuning_detected": False,
    }
    write_json(output / "audit_report.json", report)
    write_run_artifact_manifest(output, required_files=AUDIT_FILES)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
