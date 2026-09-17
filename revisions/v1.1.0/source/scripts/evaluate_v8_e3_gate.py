#!/usr/bin/env python3
"""Evaluate one registered V8 delay-stage versus matched-zero gate."""

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
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import validate_prediction_schema  # noqa: E402
from dpc_snn.experiments.v8_statistics import (  # noqa: E402
    pair_subject_seed_rows,
    static_delay_gate_decision,
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
        return {name: archive[name] for name in archive.files}


def _trial_diagnostics(root: Path, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        subject, seed = int(pair["subject"]), int(pair["seed"])
        directory = root / f"subject_{subject:02d}" / f"seed_{seed}"
        zero = _prediction(directory / "zero_predictions.npz")
        full = _prediction(directory / "full_predictions.npz")
        for field in ("subject", "session", "run", "trial_id", "label", "seed"):
            if not np.array_equal(zero[field], full[field]):
                raise RuntimeError(
                    f"E3 full/zero identity mismatch: subject={subject}, seed={seed}, field={field}"
                )
        comparison = paired_prediction_comparison(zero["label"], zero["pred"], full["pred"])
        rows.append(
            {
                "subject": subject,
                "seed": seed,
                "zero_accuracy": pair["first"],
                "full_accuracy": pair["second"],
                "delta_full_minus_zero": pair["delta_second_minus_first"],
                **comparison,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e3", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/experiments/v8_e3_static_delay.yaml")
    parser.add_argument("--parent-variant", default="full_ann")
    parser.add_argument("--minimum-median-gain-pp", type=float, default=0.5)
    parser.add_argument("--minimum-positive-pairs", type=int, default=6)
    args = parser.parse_args()

    root = Path(args.e3).resolve()
    output = ensure_dir(Path(args.output).resolve())
    config = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    delay_stage = str(config["delay"]["stage"])
    status = read_json(root / "campaign_status.json")
    if (
        status.get("status") != "completed"
        or status.get("stage") != "E3"
        or status.get("protocol") != "bci2a_session_t_nested_six_fold_oof"
        or status.get("parent_variant") != str(args.parent_variant)
        or status.get("variant") != delay_stage
        or status.get("subjects") != config["subjects"]
        or status.get("seeds") != config["seeds"]
        or int(status.get("runs", -1)) != len(config["subjects"]) * len(config["seeds"])
        or bool(status.get("session_e_accessed"))
    ):
        raise RuntimeError("E3 campaign is incomplete, unlocked or uses the wrong protocol")
    rows = _rows(root / "summary.csv")
    zero = [
        {"subject": row["subject"], "seed": row["seed"], "accuracy": row["zero_accuracy"]}
        for row in rows
    ]
    full = [
        {"subject": row["subject"], "seed": row["seed"], "accuracy": row["full_accuracy"]}
        for row in rows
    ]
    paired = pair_subject_seed_rows(zero, full)
    expected_pairs = len(config["subjects"]) * len(config["seeds"])
    gate = static_delay_gate_decision(
        paired,
        expected_pairs=expected_pairs,
        minimum_median_gain_pp=float(args.minimum_median_gain_pp),
        minimum_positive_pairs=int(args.minimum_positive_pairs),
        seed=20260718,
    )
    trial_rows = _trial_diagnostics(root, paired)
    next_stage = {
        "static_slow_within_band": "static_slow_cross_band",
        "static_slow_cross_band": "static_fast_within_band",
        "static_fast_within_band": "contextual_slow_residual",
        "contextual_slow_residual": "phase_fast_residual",
        "phase_fast_residual": None,
    }
    if delay_stage not in next_stage:
        raise RuntimeError(f"unregistered E3 gate stage: {delay_stage!r}")
    if not gate["passed"]:
        promotion = "stop_delay_promotion_and_proceed_to_E4"
    elif next_stage[delay_stage] is None:
        promotion = "delay_sequence_completed"
    else:
        promotion = f"advance_to_{next_stage[delay_stage]}"
    decision = {
        "status": "completed",
        "stage": "E3_DELAY_GATE",
        "delay_stage": delay_stage,
        "parent_variant": str(args.parent_variant),
        **gate,
        "decision": promotion,
        "trial_diagnostics_are_inferential": False,
        "session_e_accessed": False,
    }
    write_csv(output / "paired_subject_seed_results.csv", paired)
    write_csv(output / "paired_trial_diagnostics.csv", trial_rows)
    write_json(output / "gate_decision.json", decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
