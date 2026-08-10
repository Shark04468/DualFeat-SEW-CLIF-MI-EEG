#!/usr/bin/env python3
"""Evaluate the frozen full-cohort E28 ANN/SNN control gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v28_controls import v28_control_gate  # noqa: E402
from dpc_snn.utils.io import ensure_dir, write_json  # noqa: E402


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    aggregate = Path(args.aggregate).resolve()
    output = ensure_dir(Path(args.output).resolve())
    audit = json.loads((aggregate / "campaign_audit.json").read_text(encoding="utf-8"))
    if audit.get("status") not in {"passed", "completed"} or audit.get("issues"):
        raise RuntimeError("E28 aggregate audit did not pass cleanly")
    comparisons = _rows(aggregate / "paired_comparisons.csv")
    match = [
        row
        for row in comparisons
        if row["first"] == "sew_clif_ce" and row["second"] == "ann_sew_ce"
    ]
    if len(match) != 1:
        raise RuntimeError("E28 matched ANN/SNN comparison is missing or duplicated")
    metrics = _rows(aggregate / "subject_seed_metrics.csv")
    by_key = {
        (int(row["subject"]), int(row["seed"]), row["model"]): float(row["accuracy"])
        for row in metrics
    }
    subjects = sorted({key[0] for key in by_key})
    seeds = sorted({key[1] for key in by_key})
    if subjects != list(range(1, 10)) or seeds != [0, 1, 2]:
        raise RuntimeError("E28 gate requires exactly 9 subjects and 3 seeds")
    subject_deltas = {
        subject: 100.0
        * sum(
            by_key[(subject, seed, "sew_clif_ce")]
            - by_key[(subject, seed, "ann_sew_ce")]
            for seed in seeds
        )
        / len(seeds)
        for subject in subjects
    }
    decision = v28_control_gate(
        match[0], worst_subject_mean_delta_pp=min(subject_deltas.values())
    )
    decision["subject_mean_delta_pp"] = subject_deltas
    decision["aggregate"] = str(aggregate)
    write_json(output / "gate_decision.json", decision)
    print(json.dumps(decision, indent=2))
    if decision["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
