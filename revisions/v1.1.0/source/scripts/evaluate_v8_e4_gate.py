#!/usr/bin/env python3
"""Audit and evaluate the registered V8 E4 matched decoder gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_statistics import (  # noqa: E402
    matched_snn_gate_decision,
    pair_subject_seed_rows,
    paired_delta_summary,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from scripts.run_v8_e2_zero_delay import _required_files  # noqa: E402
from scripts.run_v8_e4_decoder_controls import EXPECTED_VARIANTS  # noqa: E402


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"summary is empty: {path}")
    return rows


def _subject_macro(rows: list[dict[str, Any]], key: str) -> float:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["subject"]), []).append(float(row[key]))
    if not grouped:
        raise RuntimeError(f"cannot aggregate empty E4 metric {key!r}")
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def _comparison_config(path: Path) -> str:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload.pop("active_variant", None)
    resolved_model = dict(payload["resolved_model"])
    resolved_model.pop("decoder_kind", None)
    resolved_model.pop("decoder_residual_mode", None)
    payload["resolved_model"] = resolved_model
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _audit_run(
    root: Path,
    *,
    variant: str,
    subject: int,
    seed: int,
    protocol: str,
    parameter_count: int,
) -> dict[str, Any]:
    run_dir = root / variant / f"subject_{subject:02d}" / f"seed_{seed}"
    validate_run_artifact_manifest(
        run_dir,
        required_files=_required_files(6),
        verify_hashes=True,
        verify_prediction_schema=True,
    )
    metrics = read_json(run_dir / "metrics.json")
    if (
        metrics.get("status") != "completed"
        or metrics.get("stage") != "E4"
        or metrics.get("protocol") != protocol
        or metrics.get("variant") != variant
        or int(metrics.get("subject")) != subject
        or int(metrics.get("seed")) != seed
    ):
        raise RuntimeError(f"invalid E4 metrics identity under {run_dir}")
    if metrics.get("session_e_accessed") or metrics.get("openbmi_s2_accessed"):
        raise RuntimeError(f"held-out data access was recorded under {run_dir}")
    if int(metrics["parameters"]) != int(parameter_count):
        raise RuntimeError(f"E4 parameter-count drift under {run_dir}")
    with np.load(run_dir / "predictions.npz", allow_pickle=False) as archive:
        labels = archive["label"].astype(np.int64)
        predictions = archive["pred"].astype(np.int64)
        trial_ids = archive["trial_id"].astype(str)
    recomputed = classification_metrics(labels, predictions, n_classes=4)
    if abs(float(metrics["accuracy"]) - float(recomputed["accuracy"])) > 1e-12:
        raise RuntimeError(f"E4 final accuracy does not reproduce under {run_dir}")
    with np.load(run_dir / "prefix_predictions.npz", allow_pickle=False) as archive:
        prefix_logits = archive["logits"].astype(np.float32)
        prefix_labels = archive["labels"].astype(np.int64)
        endpoint_seconds = archive["endpoint_seconds"].astype(np.float64)
    if not np.array_equal(labels, prefix_labels):
        raise RuntimeError(f"E4 prefix labels do not match final labels under {run_dir}")
    endpoint_metrics = [
        classification_metrics(
            labels, prefix_logits[:, index].argmax(axis=1), n_classes=4
        )
        for index in range(prefix_logits.shape[1])
    ]
    registered = list(metrics["endpoint_metrics"])
    if len(registered) != len(endpoint_metrics) or any(
        abs(float(left["accuracy"]) - float(right["accuracy"])) > 1e-12
        for left, right in zip(registered, endpoint_metrics, strict=True)
    ):
        raise RuntimeError(f"E4 endpoint metrics do not reproduce under {run_dir}")
    return {
        "variant": variant,
        "subject": subject,
        "seed": seed,
        "accuracy": float(metrics["accuracy"]),
        "endpoint_accuracy": [float(row["accuracy"]) for row in endpoint_metrics],
        "endpoint_seconds": endpoint_seconds.tolist(),
        "binary_spike_rate": metrics["binary_spike_rate"],
        "final_activity_nonzero_rate": metrics["final_activity_nonzero_rate"],
        "final_activity_absolute_mean": metrics["final_activity_absolute_mean"],
        "trial_ids": trial_ids.tolist(),
        "comparison_config": _comparison_config(run_dir / "resolved_config.yaml"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e4", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default="configs/experiments/v8_e4_decoder_controls.yaml"
    )
    args = parser.parse_args()

    root = Path(args.e4).resolve()
    output = ensure_dir(Path(args.output).resolve())
    config = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    status = read_json(root / "campaign_status.json")
    expected_subjects = list(config["subjects"])
    expected_seeds = list(config["seeds"])
    expected_runs = len(EXPECTED_VARIANTS) * len(expected_subjects) * len(expected_seeds)
    if (
        status.get("status") != "completed"
        or status.get("stage") != "E4"
        or status.get("protocol") != config["protocol"]
        or status.get("variants") != list(EXPECTED_VARIANTS)
        or status.get("subjects") != expected_subjects
        or status.get("seeds") != expected_seeds
        or int(status.get("runs", -1)) != expected_runs
        or not status.get("full_registered_contract")
    ):
        raise RuntimeError("E4 campaign status does not satisfy the exact registered contract")
    validate_run_artifact_manifest(
        root,
        required_files=(
            "manifest.json",
            "campaign_status.json",
            "summary.csv",
            "capacity_audit.json",
            "variant_smoke.json",
            "source_tree_manifest.json",
            "source_tree_summary.json",
            "heldout_lock_manifest.json",
            "shared_cache_provenance.json",
            "resolved_campaign.yaml",
        ),
        verify_hashes=True,
        verify_prediction_schema=False,
    )
    capacity = read_json(root / "capacity_audit.json")
    if capacity.get("status") != "passed" or set(capacity["variants"]) != set(
        EXPECTED_VARIANTS
    ):
        raise RuntimeError("E4 capacity audit is absent or incomplete")
    parameter_counts = {
        int(row["parameters"]) for row in capacity["variants"].values()
    }
    trainable_counts = {
        int(row["trainable_parameters"]) for row in capacity["variants"].values()
    }
    if len(parameter_counts) != 1 or len(trainable_counts) != 1:
        raise RuntimeError("E4 capacity controls are not parameter matched")
    parameter_count = next(iter(parameter_counts))

    summary = _rows(root / "summary.csv")
    coverage = {
        (row["variant"], int(row["subject"]), int(row["seed"])) for row in summary
    }
    expected_coverage = {
        (variant, subject, seed)
        for variant in EXPECTED_VARIANTS
        for subject in expected_subjects
        for seed in expected_seeds
    }
    if coverage != expected_coverage or len(summary) != expected_runs:
        raise RuntimeError("E4 summary coverage is incomplete or duplicated")

    records: dict[str, list[dict[str, Any]]] = {name: [] for name in EXPECTED_VARIANTS}
    for variant, subject, seed in sorted(expected_coverage):
        records[variant].append(
            _audit_run(
                root,
                variant=variant,
                subject=subject,
                seed=seed,
                protocol=config["protocol"],
                parameter_count=parameter_count,
            )
        )
    endpoint_seconds = records["ann_residual"][0]["endpoint_seconds"]
    if endpoint_seconds != [0.5, 1.0, 2.0, 4.0]:
        raise RuntimeError(f"unexpected E4 endpoint grid: {endpoint_seconds}")

    candidate_rows: list[dict[str, Any]] = []
    all_pair_rows: list[dict[str, Any]] = []
    ann = records["ann_residual"]
    ann_activity = _subject_macro(ann, "final_activity_nonzero_rate")
    for variant in ("plif_plain", "clif_plain", "sew_clif"):
        snn = records[variant]
        paired = pair_subject_seed_rows(ann, snn)
        for row in paired:
            all_pair_rows.append({"variant": variant, **row})
        final_summary = paired_delta_summary(paired, seed=20260718)
        endpoint_gains: list[float] = []
        for endpoint_index in range(len(endpoint_seconds)):
            ann_endpoint = [
                {**row, "endpoint_accuracy": row["endpoint_accuracy"][endpoint_index]}
                for row in ann
            ]
            snn_endpoint = [
                {**row, "endpoint_accuracy": row["endpoint_accuracy"][endpoint_index]}
                for row in snn
            ]
            endpoint_pairs = pair_subject_seed_rows(
                ann_endpoint, snn_endpoint, value="endpoint_accuracy"
            )
            endpoint_summary = paired_delta_summary(
                endpoint_pairs, seed=20260718 + endpoint_index
            )
            endpoint_gains.append(100.0 * endpoint_summary["subject_macro_mean_delta"])
        for ann_row, snn_row in zip(ann, snn, strict=True):
            if (
                ann_row["subject"] != snn_row["subject"]
                or ann_row["seed"] != snn_row["seed"]
                or ann_row["trial_ids"] != snn_row["trial_ids"]
                or ann_row["comparison_config"] != snn_row["comparison_config"]
            ):
                raise RuntimeError(f"E4 matched-control provenance mismatch for {variant}")
        candidate_rows.append(
            {
                "variant": variant,
                "final_gain_pp": 100.0 * final_summary["subject_macro_mean_delta"],
                "early_gain_pp": max(endpoint_gains[:-1]),
                "endpoint_gain_pp": endpoint_gains,
                "ann_final_activity_nonzero_rate": ann_activity,
                "snn_final_activity_nonzero_rate": _subject_macro(
                    snn, "final_activity_nonzero_rate"
                ),
                "snn_binary_spike_rate": _subject_macro(snn, "binary_spike_rate"),
                "paired_final_summary": final_summary,
            }
        )

    gate = dict(config["gate"])
    decision = matched_snn_gate_decision(
        candidate_rows,
        maximum_accuracy_gap_pp=float(gate["maximum_snn_accuracy_gap_pp"]),
        minimum_early_gain_pp=float(gate["minimum_early_endpoint_gain_pp"]),
        minimum_activity_density_reduction=float(
            gate["minimum_activity_density_reduction"]
        ),
        minimum_nondegenerate_activity_rate=float(
            gate["minimum_nondegenerate_snn_activity_rate"]
        ),
        maximum_sparse_activity_rate=float(gate["maximum_sparse_snn_activity_rate"]),
    )
    result = {
        "status": "completed",
        "stage": "E4_GATE",
        "protocol": config["protocol"],
        "development_selection_only": True,
        "parameter_count_per_arm": parameter_count,
        "trainable_parameter_count_per_arm": next(iter(trainable_counts)),
        "passed": decision["passed"],
        "decision": (
            "retain_selected_snn_for_freeze_candidate"
            if decision["passed"]
            else "do_not_claim_snn_necessity_route_to_bounded_e5_or_ann_backbone"
        ),
        "gate": decision,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    write_csv(output / "paired_subject_seed_results.csv", all_pair_rows)
    write_json(output / "gate_decision.json", result)
    write_run_artifact_manifest(
        output,
        required_files=(
            "manifest.json",
            "paired_subject_seed_results.csv",
            "gate_decision.json",
        ),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
