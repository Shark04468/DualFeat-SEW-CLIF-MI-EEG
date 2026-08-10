#!/usr/bin/env python3
"""Evaluate the preregistered V8 zero-delay versus strongest-baseline gate."""

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

from dpc_snn.experiments.v62_protocol import validate_prediction_schema  # noqa: E402
from dpc_snn.experiments.v8_statistics import (  # noqa: E402
    pair_subject_seed_rows,
    paired_delta_summary,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import paired_prediction_comparison  # noqa: E402


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"summary is empty: {path}")
    return rows


def _prediction(path: Path) -> dict[str, np.ndarray]:
    validate_prediction_schema(path, path.with_suffix(".csv"))
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _paired_trial_audit(
    baseline_root: Path,
    v8_root: Path,
    *,
    baseline_model: str,
    v8_variant: str,
    pairs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_trials = 0
    total_discordant = 0
    for pair in pairs:
        subject, seed = int(pair["subject"]), int(pair["seed"])
        baseline_path = (
            baseline_root
            / baseline_model
            / f"subject_{subject:02d}"
            / f"seed_{seed}"
            / "predictions.npz"
        )
        v8_path = (
            v8_root
            / v8_variant
            / f"subject_{subject:02d}"
            / f"seed_{seed}"
            / "predictions.npz"
        )
        baseline = _prediction(baseline_path)
        v8 = _prediction(v8_path)
        identity_fields = ("subject", "session", "run", "trial_id", "label")
        for field in identity_fields:
            if not np.array_equal(baseline[field], v8[field]):
                raise RuntimeError(
                    f"paired prediction identity mismatch at subject={subject}, seed={seed}, field={field}"
                )
        comparison = paired_prediction_comparison(
            baseline["label"], baseline["pred"], v8["pred"]
        )
        row = {
            "subject": subject,
            "seed": seed,
            "trials": int(baseline["label"].size),
            "baseline_accuracy": pair["first"],
            "v8_accuracy": pair["second"],
            "delta_v8_minus_baseline": pair["delta_second_minus_first"],
            **comparison,
        }
        rows.append(row)
        total_trials += row["trials"]
        total_discordant += int(comparison["discordant_predictions"])
    return rows, {
        "identity_match": True,
        "pairs": len(rows),
        "trials": total_trials,
        "discordant_predictions": total_discordant,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1", required=True)
    parser.add_argument("--e2", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--v8-variant", default="full_ann")
    parser.add_argument("--maximum-gap-pp", type=float, default=0.5)
    args = parser.parse_args()

    e1 = Path(args.e1).resolve()
    e2 = Path(args.e2).resolve()
    output = ensure_dir(Path(args.output).resolve())
    e1_status = read_json(e1 / "campaign_status.json")
    e2_status = read_json(e2 / "campaign_status.json")
    if e1_status.get("status") != "completed" or e2_status.get("status") != "completed":
        raise RuntimeError("E1 and E2 campaigns must both be complete before gate evaluation")
    if e1_status.get("protocol") != "bci2a_session_t_nested_six_fold_oof":
        raise RuntimeError("E1 is not the locked nested Session-T protocol")
    if e2_status.get("protocol") != "bci2a_session_t_nested_six_fold_oof":
        raise RuntimeError("E2 is not the locked nested Session-T protocol")
    baseline_selection = read_json(e1 / "confirmation_selection.json")
    baseline_model = str(baseline_selection["models"][0])
    v8_variant = str(args.v8_variant)
    if v8_variant not in set(e2_status["confirmed_variants"]):
        raise RuntimeError(f"V8 gate variant {v8_variant!r} was not confirmed across seeds")

    e1_rows = [row for row in _rows(e1 / "summary.csv") if row["model"] == baseline_model]
    e2_rows = [row for row in _rows(e2 / "summary.csv") if row["variant"] == v8_variant]
    paired = pair_subject_seed_rows(e1_rows, e2_rows)
    expected_pairs = len(e1_status["subjects"]) * len(e1_status["confirmation_seeds"])
    if len(paired) != expected_pairs:
        raise RuntimeError(f"expected {expected_pairs} subject-seed pairs, got {len(paired)}")
    summary = paired_delta_summary(paired, seed=20260718)
    trial_rows, trial_audit = _paired_trial_audit(
        e1,
        e2,
        baseline_model=baseline_model,
        v8_variant=v8_variant,
        pairs=paired,
    )
    threshold = -float(args.maximum_gap_pp) / 100.0
    passed = bool(summary["subject_macro_mean_delta"] >= threshold)
    decision = {
        "status": "completed",
        "stage": "E2_GATE",
        "baseline_model": baseline_model,
        "v8_variant": v8_variant,
        "metric": "subject-macro paired nested Session-T OOF accuracy",
        "maximum_allowed_gap_pp": float(args.maximum_gap_pp),
        "pass_threshold_delta": threshold,
        "passed": passed,
        "decision": "advance_to_E3" if passed else "route_to_bounded_E5_HPO_before_E3",
        "paired_summary": summary,
        "trial_pairing_audit": trial_audit,
        "session_e_accessed": False,
    }
    write_csv(output / "paired_subject_seed_results.csv", paired)
    write_csv(output / "paired_trial_comparisons.csv", trial_rows)
    write_json(output / "gate_decision.json", decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
