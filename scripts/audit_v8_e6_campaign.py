#!/usr/bin/env python3
"""Independently audit and summarize the frozen E6 T-to-E campaigns."""

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
from dpc_snn.experiments.v8_protocol import validate_v8_freeze_manifest  # noqa: E402
from dpc_snn.experiments.v8_statistics import (  # noqa: E402
    pair_subject_seed_rows,
    paired_delta_summary,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import (  # noqa: E402
    classification_metrics,
    paired_prediction_comparison,
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty E6 summary: {path}")
    return rows


def _close(first: float, second: float, tolerance: float = 1e-10) -> bool:
    return bool(np.isclose(float(first), float(second), rtol=0.0, atol=tolerance))


def _prediction(path: Path) -> dict[str, np.ndarray]:
    validate_prediction_schema(path, path.with_suffix(".csv"))
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _audit_prediction(
    run: Path,
    metrics: dict[str, Any],
    *,
    model_field: str,
) -> dict[str, np.ndarray]:
    declared = read_json(run / "manifest.json")["required_files"]
    validate_run_artifact_manifest(
        run,
        required_files=tuple(declared),
        verify_hashes=True,
        verify_prediction_schema=True,
    )
    prediction = _prediction(run / "predictions.npz")
    recomputed = classification_metrics(
        prediction["label"], prediction["pred"], n_classes=4
    )
    for metric in ("accuracy", "balanced_accuracy", "kappa", "macro_f1"):
        if not _close(metrics[metric], recomputed[metric]):
            raise RuntimeError(f"E6 {model_field} metric drift at {run}: {metric}")
    state = read_json(run / "state_audit.json")
    access = read_json(run / "data_access_manifest.json")
    if (
        not bool(state.get("identical"))
        or state.get("checkpoint_before_session_e") != state.get("state_after_session_e")
        or access.get("session_e_checkpoint_selection") is not False
        or access.get("session_e_gradient_updates") is not False
        or access.get("evaluation_session") != "E"
    ):
        raise RuntimeError(f"E6 held-out state/access audit failed at {run}")
    if len(np.unique(prediction["trial_id"])) != prediction["trial_id"].size:
        raise RuntimeError(f"E6 duplicate held-out trial IDs at {run}")
    return prediction


def _identity(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> None:
    for field in ("subject", "session", "run", "trial_id", "label"):
        if not np.array_equal(first[field], second[field]):
            raise RuntimeError(f"E6 paired held-out identity mismatch for {field}")


def _paired_report(
    first_rows: list[dict[str, Any]],
    second_rows: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    paired = pair_subject_seed_rows(first_rows, second_rows)
    if len(paired) != 45:
        raise RuntimeError(f"E6 paired comparison requires 45 runs, got {len(paired)}")
    return {"pairs": paired, "summary": paired_delta_summary(paired, seed=seed)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e6", required=True)
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    e6 = Path(args.e6).resolve()
    baselines = Path(args.baselines).resolve()
    output = ensure_dir(Path(args.output).resolve())
    freeze = validate_v8_freeze_manifest(Path(args.freeze).resolve())
    e6_status = read_json(e6 / "campaign_status.json")
    baseline_status = read_json(baselines / "campaign_status.json")
    if (
        e6_status.get("status") != "completed"
        or baseline_status.get("status") != "completed"
        or not e6_status.get("full_registered_contract")
        or not baseline_status.get("full_registered_contract")
        or e6_status.get("freeze_sha256") != freeze["combined_sha256"]
        or baseline_status.get("freeze_sha256") != freeze["combined_sha256"]
        or e6_status.get("session_e_used_for_selection") is not False
        or baseline_status.get("session_e_used_for_selection") is not False
    ):
        raise RuntimeError("E6 campaigns are incomplete, unmatched or not frozen")

    subjects = list(range(1, 10))
    seeds = list(range(5))
    primary_rows = [
        row for row in _rows(e6 / "summary.csv") if row["variant"] == "frozen_primary"
    ]
    ann_rows = [
        row for row in _rows(e6 / "summary.csv") if row["variant"] == "matched_ann"
    ]
    baseline_rows = _rows(baselines / "summary.csv")
    expected_primary = {(subject, seed) for subject in subjects for seed in seeds}
    if {(int(row["subject"]), int(row["seed"])) for row in primary_rows} != expected_primary:
        raise RuntimeError("E6 primary coverage is not 9 subjects x 5 seeds")
    expected_baseline = {
        (model, subject, seed)
        for model in freeze["baselines"]["models"]
        for subject in subjects
        for seed in seeds
    }
    observed_baseline = {
        (row["model"], int(row["subject"]), int(row["seed"]))
        for row in baseline_rows
    }
    if observed_baseline != expected_baseline or len(baseline_rows) != len(
        expected_baseline
    ):
        raise RuntimeError("E6 baseline coverage is incomplete or duplicated")
    snn_primary = freeze["architecture"]["primary_variant"] != "ann_residual"
    if snn_primary and {
        (int(row["subject"]), int(row["seed"])) for row in ann_rows
    } != expected_primary:
        raise RuntimeError("E6 matched ANN coverage is incomplete")
    if not snn_primary and ann_rows:
        raise RuntimeError("E6 duplicated an ANN primary as a second ANN control")

    primary_predictions: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    ann_predictions: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    baseline_predictions: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}
    for row in primary_rows:
        key = (int(row["subject"]), int(row["seed"]))
        run = e6 / "frozen_primary" / f"subject_{key[0]:02d}" / f"seed_{key[1]}"
        metrics = read_json(run / "metrics.json")
        primary_predictions[key] = _audit_prediction(
            run, metrics, model_field="frozen_primary"
        )
    for row in ann_rows:
        key = (int(row["subject"]), int(row["seed"]))
        run = e6 / "matched_ann" / f"subject_{key[0]:02d}" / f"seed_{key[1]}"
        metrics = read_json(run / "metrics.json")
        ann_predictions[key] = _audit_prediction(run, metrics, model_field="matched_ann")
    for row in baseline_rows:
        key = (row["model"], int(row["subject"]), int(row["seed"]))
        run = baselines / key[0] / f"subject_{key[1]:02d}" / f"seed_{key[2]}"
        metrics = read_json(run / "metrics.json")
        baseline_predictions[key] = _audit_prediction(run, metrics, model_field=key[0])

    strongest = str(freeze["baselines"]["strongest_development_model"])
    strongest_rows = [row for row in baseline_rows if row["model"] == strongest]
    versus_strongest = _paired_report(
        strongest_rows, primary_rows, seed=20260718
    )
    trial_rows: list[dict[str, Any]] = []
    for subject, seed in sorted(expected_primary):
        baseline_prediction = baseline_predictions[(strongest, subject, seed)]
        primary_prediction = primary_predictions[(subject, seed)]
        _identity(baseline_prediction, primary_prediction)
        trial_rows.append(
            {
                "comparison": f"frozen_primary_minus_{strongest}",
                "subject": subject,
                "seed": seed,
                **paired_prediction_comparison(
                    primary_prediction["label"],
                    baseline_prediction["pred"],
                    primary_prediction["pred"],
                ),
            }
        )

    versus_ann: dict[str, Any] | None = None
    if snn_primary:
        versus_ann = _paired_report(ann_rows, primary_rows, seed=20260719)
        for subject, seed in sorted(expected_primary):
            ann_prediction = ann_predictions[(subject, seed)]
            primary_prediction = primary_predictions[(subject, seed)]
            _identity(ann_prediction, primary_prediction)
            trial_rows.append(
                {
                    "comparison": "frozen_primary_minus_matched_ann",
                    "subject": subject,
                    "seed": seed,
                    **paired_prediction_comparison(
                        primary_prediction["label"],
                        ann_prediction["pred"],
                        primary_prediction["pred"],
                    ),
                }
            )

    delay_report: dict[str, Any] | None = None
    if freeze["architecture"]["delay"]["enabled"]:
        zero_rows: list[dict[str, Any]] = []
        for row in primary_rows:
            subject, seed = int(row["subject"]), int(row["seed"])
            run = e6 / "frozen_primary" / f"subject_{subject:02d}" / f"seed_{seed}"
            zero = _prediction(run / "matched_zero_predictions.npz")
            full = primary_predictions[(subject, seed)]
            _identity(zero, full)
            zero_metric = classification_metrics(zero["label"], zero["pred"], n_classes=4)
            zero_rows.append(
                {"subject": subject, "seed": seed, "accuracy": zero_metric["accuracy"]}
            )
        delay_report = _paired_report(zero_rows, primary_rows, seed=20260720)

    model_summary: list[dict[str, Any]] = []
    arms: dict[str, list[dict[str, Any]]] = {"frozen_primary": primary_rows}
    if ann_rows:
        arms["matched_ann"] = ann_rows
    for model in freeze["baselines"]["models"]:
        arms[str(model)] = [row for row in baseline_rows if row["model"] == model]
    for model, rows in arms.items():
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
            for subject in subjects
        ]
        model_summary.append(
            {
                "model": model,
                "subject_macro_accuracy": float(np.mean(subject_means)),
                "subject_median_accuracy": float(np.median(subject_means)),
                "subject_standard_deviation": float(np.std(subject_means, ddof=1)),
                "minimum_subject_accuracy": float(np.min(subject_means)),
                "maximum_subject_accuracy": float(np.max(subject_means)),
            }
        )
    model_summary.sort(key=lambda row: (-row["subject_macro_accuracy"], row["model"]))

    report = {
        "status": "passed",
        "stage": "E6_AUDIT",
        "freeze_sha256": freeze["combined_sha256"],
        "primary_runs_audited": len(primary_rows),
        "matched_ann_runs_audited": len(ann_rows),
        "baseline_runs_audited": len(baseline_rows),
        "all_trial_identities_paired": True,
        "all_metrics_recomputed": True,
        "all_artifact_hashes_verified": True,
        "strongest_development_baseline": strongest,
        "primary_minus_strongest": versus_strongest["summary"],
        "primary_minus_matched_ann": versus_ann["summary"] if versus_ann else None,
        "delay_full_minus_matched_zero": (
            delay_report["summary"] if delay_report else None
        ),
        "post_session_e_tuning_detected": False,
        "openbmi_s2_accessed": False,
    }
    write_csv(output / "model_summary.csv", model_summary)
    write_csv(
        output / "primary_vs_strongest_subject_seed.csv", versus_strongest["pairs"]
    )
    if versus_ann:
        write_csv(output / "primary_vs_ann_subject_seed.csv", versus_ann["pairs"])
    if delay_report:
        write_csv(output / "delay_full_vs_zero_subject_seed.csv", delay_report["pairs"])
    write_csv(output / "paired_trial_diagnostics.csv", trial_rows)
    write_json(output / "audit_report.json", report)
    write_run_artifact_manifest(
        output,
        required_files=(
            "manifest.json",
            "audit_report.json",
            "model_summary.csv",
            "primary_vs_strongest_subject_seed.csv",
            "paired_trial_diagnostics.csv",
        ),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
