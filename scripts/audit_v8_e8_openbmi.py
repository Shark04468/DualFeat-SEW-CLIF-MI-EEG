#!/usr/bin/env python3
"""Audit frozen OpenBMI S1-to-S2 external confirmation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    validate_prediction_schema,
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    validate_v8_external_unlock_manifest,
    validate_v8_freeze_manifest,
)
from dpc_snn.experiments.v8_statistics import (  # noqa: E402
    pair_subject_seed_rows,
    paired_delta_summary,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty E8 summary: {path}")
    return rows


def _prediction(path: Path) -> dict[str, np.ndarray]:
    validate_prediction_schema(path, path.with_suffix(".csv"))
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _audit_run(run: Path, row: dict[str, str]) -> dict[str, np.ndarray]:
    manifest = read_json(run / "manifest.json")
    validate_run_artifact_manifest(
        run,
        required_files=manifest["required_files"],
        verify_hashes=True,
        verify_prediction_schema=True,
    )
    metrics = read_json(run / "metrics.json")
    prediction = _prediction(run / "predictions.npz")
    recomputed = classification_metrics(
        prediction["label"], prediction["pred"], n_classes=2
    )
    for field in ("accuracy", "balanced_accuracy", "kappa", "macro_f1"):
        if not np.isclose(
            float(metrics[field]), float(recomputed[field]), rtol=0.0, atol=1e-10
        ):
            raise RuntimeError(f"E8 metric drift at {run}: {field}")
    state = read_json(run / "state_audit.json")
    access = read_json(run / "s2_access_manifest.json")
    if (
        not state.get("identical")
        or metrics.get("s2_checkpoint_selection") is not False
        or metrics.get("s2_gradient_updates") is not False
        or access.get("sessions") != ["S2"]
        or set(prediction["session"].tolist()) != {"S2"}
        or metrics.get("run_fingerprint") != row["run_fingerprint"]
    ):
        raise RuntimeError(f"E8 held-out protocol audit failed at {run}")
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e8", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--unlock", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    e8 = Path(args.e8).resolve()
    output = ensure_dir(Path(args.output).resolve())
    freeze = validate_v8_freeze_manifest(Path(args.freeze).resolve())
    unlock = validate_v8_external_unlock_manifest(
        Path(args.unlock).resolve(),
        expected_parent_freeze_sha256=freeze["combined_sha256"],
    )
    status = read_json(e8 / "campaign_status.json")
    if (
        status.get("status") != "completed"
        or not status.get("full_registered_contract")
        or status.get("s2_used_for_selection") is not False
        or status.get("s2_gradient_updates") is not False
        or status.get("external_unlock_sha256") != unlock["combined_sha256"]
    ):
        raise RuntimeError("E8 campaign is incomplete or violates the external unlock")
    rows = _rows(e8 / "summary.csv")
    subjects = list(unlock["dataset"]["confirmatory_subjects"])
    seeds = list(unlock["dataset"]["seeds"])
    arms = list(status["arms"])
    expected = {
        (arm, int(subject), int(seed))
        for arm in arms
        for subject in subjects
        for seed in seeds
    }
    observed = {(row["arm"], int(row["subject"]), int(row["seed"])) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise RuntimeError("E8 external coverage is incomplete or duplicated")

    predictions: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}
    for row in rows:
        key = (row["arm"], int(row["subject"]), int(row["seed"]))
        run = e8 / key[0] / f"subject_{key[1]:02d}" / f"seed_{key[2]}"
        predictions[key] = _audit_run(run, row)
    primary = [row for row in rows if row["arm"] == "frozen_primary"]
    model_summary: list[dict[str, Any]] = []
    for arm in arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        subject_accuracy = [
            float(
                np.mean(
                    [
                        float(row["accuracy"])
                        for row in arm_rows
                        if int(row["subject"]) == int(subject)
                    ]
                )
            )
            for subject in subjects
        ]
        rng = np.random.default_rng(20260718)
        sampled = rng.integers(
            0, len(subject_accuracy), size=(20_000, len(subject_accuracy))
        )
        bootstrap = np.asarray(subject_accuracy)[sampled].mean(axis=1)
        model_summary.append(
            {
                "arm": arm,
                "subjects": len(subjects),
                "seeds_per_subject": len(seeds),
                "subject_macro_accuracy": float(np.mean(subject_accuracy)),
                "subject_median_accuracy": float(np.median(subject_accuracy)),
                "subject_ci95_low": float(np.quantile(bootstrap, 0.025)),
                "subject_ci95_high": float(np.quantile(bootstrap, 0.975)),
                "chance_level": 0.5,
            }
        )

    primary_minus_ann: dict[str, Any] | None = None
    pair_rows: list[dict[str, Any]] = []
    if "matched_ann" in arms:
        ann = [row for row in rows if row["arm"] == "matched_ann"]
        pair_rows = pair_subject_seed_rows(ann, primary)
        primary_minus_ann = paired_delta_summary(pair_rows, seed=20260719)
        for subject in subjects:
            for seed in seeds:
                first = predictions[("matched_ann", int(subject), int(seed))]
                second = predictions[("frozen_primary", int(subject), int(seed))]
                for field in ("subject", "session", "run", "trial_id", "label"):
                    if not np.array_equal(first[field], second[field]):
                        raise RuntimeError(f"E8 ANN/SNN identity mismatch for {field}")

    delay_summary: dict[str, Any] | None = None
    delay_pairs: list[dict[str, Any]] = []
    if freeze["architecture"]["delay"]["enabled"]:
        zero_rows: list[dict[str, Any]] = []
        for row in primary:
            subject, seed = int(row["subject"]), int(row["seed"])
            run = e8 / "frozen_primary" / f"subject_{subject:02d}" / f"seed_{seed}"
            zero = _prediction(run / "matched_zero_predictions.npz")
            full = predictions[("frozen_primary", subject, seed)]
            for field in ("subject", "session", "run", "trial_id", "label"):
                if not np.array_equal(zero[field], full[field]):
                    raise RuntimeError(f"E8 delay identity mismatch for {field}")
            zero_metric = classification_metrics(zero["label"], zero["pred"], n_classes=2)
            zero_rows.append(
                {"subject": subject, "seed": seed, "accuracy": zero_metric["accuracy"]}
            )
        delay_pairs = pair_subject_seed_rows(zero_rows, primary)
        delay_summary = paired_delta_summary(delay_pairs, seed=20260720)

    report = {
        "status": "passed",
        "stage": "E8_AUDIT",
        "freeze_sha256": freeze["combined_sha256"],
        "subjects": len(subjects),
        "seeds_per_subject": len(seeds),
        "runs_audited": len(rows),
        "all_hashes_verified": True,
        "all_metrics_recomputed": True,
        "all_s2_trial_identities_verified": True,
        "s2_used_for_selection": False,
        "s2_gradient_updates": False,
        "primary_minus_matched_ann": primary_minus_ann,
        "delay_full_minus_matched_zero": delay_summary,
        "external_unlock_sha256": unlock["combined_sha256"],
    }
    write_csv(output / "model_summary.csv", model_summary)
    if pair_rows:
        write_csv(output / "primary_vs_ann_subject_seed.csv", pair_rows)
    if delay_pairs:
        write_csv(output / "delay_full_vs_zero_subject_seed.csv", delay_pairs)
    write_json(output / "audit_report.json", report)
    write_run_artifact_manifest(
        output,
        required_files=("manifest.json", "audit_report.json", "model_summary.csv"),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
