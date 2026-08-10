#!/usr/bin/env python3
"""Evaluate the pre-registered V8 E4 untouched-subject confirmation gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402


LOCK_SCHEMA = "dpc-snn-v8-e4-three-way-confirmation-lock/v1"
FUSION_RULE = "unweighted arithmetic mean of ATCNet, FBCNet, and decoder probabilities"


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise RuntimeError(f"expected a boolean value, received {value!r}")


def _int(value: Any) -> int:
    return int(str(value).strip())


def _float(value: Any) -> float:
    return float(str(value).strip())


def load_aggregate_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty aggregate CSV: {path}")
    return rows


def evaluate_confirmation(
    lock: Mapping[str, Any],
    aggregate_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if lock.get("schema") != LOCK_SCHEMA:
        raise RuntimeError("confirmation lock schema does not match the evaluator")
    if lock.get("fusion_rule") != FUSION_RULE:
        raise RuntimeError("confirmation fusion rule is not the locked equal-probability rule")
    for key in ("no_temperature_calibration", "no_weight_search"):
        if lock.get(key) is not True:
            raise RuntimeError(f"confirmation lock must set {key}=true")
    for key in ("session_e_accessed", "openbmi_s2_accessed"):
        if lock.get(key) is not False:
            raise RuntimeError(f"confirmation lock must set {key}=false")

    selected = str(lock["selected_snn_variant"])
    matched_ann = str(lock["matched_ann_control"])
    subjects = [_int(value) for value in lock["confirmation_subjects"]]
    seeds = [_int(value) for value in lock["confirmation_seeds"]]
    expected_pairs = [(subject, seed) for subject in subjects for seed in seeds]
    thresholds = lock["thresholds"]
    firing_low, firing_high = [
        _float(value) for value in thresholds["per_subject_mean_firing_rate_interval"]
    ]

    by_key: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    for row in aggregate_rows:
        key = (_int(row["subject"]), _int(row["seed"]), str(row["variant"]))
        if key in by_key:
            raise RuntimeError(f"duplicate confirmation aggregate row: {key}")
        by_key[key] = row

    pair_rows: list[dict[str, Any]] = []
    for subject, seed in expected_pairs:
        snn_key = (subject, seed, selected)
        ann_key = (subject, seed, matched_ann)
        if snn_key not in by_key or ann_key not in by_key:
            raise RuntimeError(f"missing locked confirmation pair S{subject}/seed{seed}")
        snn = by_key[snn_key]
        ann = by_key[ann_key]
        if _int(snn["folds"]) != 6 or _int(ann["folds"]) != 6:
            raise RuntimeError("confirmation requires complete six-fold OOF predictions")
        if _bool(snn["session_e_accessed"]) or _bool(ann["session_e_accessed"]):
            raise RuntimeError("Session E was accessed during confirmation")

        snn_accuracy = _float(snn["three_way_equal_accuracy"])
        ann_accuracy = _float(ann["three_way_equal_accuracy"])
        snn_anchor = _float(snn["anchor_atc_plus_fbc_accuracy"])
        ann_anchor = _float(ann["anchor_atc_plus_fbc_accuracy"])
        if abs(snn_anchor - ann_anchor) > 1e-12:
            raise RuntimeError("matched SNN and ANN rows use different accuracy anchors")
        firing_rate = _float(snn["mean_firing_rate"])
        delta_ann_pp = 100.0 * (snn_accuracy - ann_accuracy)
        delta_anchor_pp = 100.0 * (snn_accuracy - snn_anchor)
        pair_rows.append(
            {
                "subject": subject,
                "seed": seed,
                "selected_snn_variant": selected,
                "matched_ann_control": matched_ann,
                "snn_three_way_accuracy": snn_accuracy,
                "ann_three_way_accuracy": ann_accuracy,
                "atc_fbc_anchor_accuracy": snn_anchor,
                "snn_delta_vs_matched_ann_pp": delta_ann_pp,
                "snn_delta_vs_anchor_pp": delta_anchor_pp,
                "snn_mean_firing_rate": firing_rate,
                "passes_anchor_floor": delta_anchor_pp
                >= _float(thresholds["per_subject_snn_vs_anchor_floor_pp"]),
                "passes_firing_interval": firing_low <= firing_rate <= firing_high,
                "session_e_accessed": False,
            }
        )

    macro_snn = sum(row["snn_three_way_accuracy"] for row in pair_rows) / len(pair_rows)
    macro_ann = sum(row["ann_three_way_accuracy"] for row in pair_rows) / len(pair_rows)
    macro_anchor = sum(row["atc_fbc_anchor_accuracy"] for row in pair_rows) / len(
        pair_rows
    )
    macro_ann_delta_pp = 100.0 * (macro_snn - macro_ann)
    macro_anchor_delta_pp = 100.0 * (macro_snn - macro_anchor)
    criteria = {
        "macro_snn_vs_matched_ann": macro_ann_delta_pp
        >= _float(thresholds["macro_snn_vs_matched_ann_minimum_delta_pp"]),
        "macro_snn_vs_atc_fbc_anchor": macro_anchor_delta_pp
        >= _float(thresholds["macro_snn_vs_atc_fbc_anchor_minimum_delta_pp"]),
        "all_pairs_above_anchor_floor": all(
            bool(row["passes_anchor_floor"]) for row in pair_rows
        ),
        "all_pairs_have_nondegenerate_firing": all(
            bool(row["passes_firing_interval"]) for row in pair_rows
        ),
    }
    gate = {
        "schema": "dpc-snn-v8-e4-three-way-confirmation-result/v1",
        "passed": all(criteria.values()),
        "lock_schema": LOCK_SCHEMA,
        "lock_snapshot_digest": str(lock["snapshot_digest"]),
        "selected_snn_variant": selected,
        "matched_ann_control": matched_ann,
        "fusion_rule": FUSION_RULE,
        "confirmation_pairs": len(pair_rows),
        "macro_snn_three_way_accuracy": macro_snn,
        "macro_ann_three_way_accuracy": macro_ann,
        "macro_atc_fbc_anchor_accuracy": macro_anchor,
        "macro_snn_delta_vs_matched_ann_pp": macro_ann_delta_pp,
        "macro_snn_delta_vs_atc_fbc_anchor_pp": macro_anchor_delta_pp,
        "criteria": criteria,
        "thresholds": dict(thresholds),
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    return pair_rows, gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--aggregate-csv", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    lock_path = Path(args.lock).resolve()
    rows: list[dict[str, Any]] = []
    for value in args.aggregate_csv:
        rows.extend(load_aggregate_csv(Path(value).resolve()))
    pair_rows, gate = evaluate_confirmation(read_json(lock_path), rows)
    output = ensure_dir(Path(args.output).resolve())
    write_csv(output / "per_pair.csv", pair_rows)
    write_json(output / "gate.json", gate)
    write_json(output / "confirmation_lock.json", read_json(lock_path))
    print(json.dumps({"status": "completed", "pairs": pair_rows, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
