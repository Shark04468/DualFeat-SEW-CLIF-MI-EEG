#!/usr/bin/env python
"""Evaluate the three-seed V3.5 synthetic promotion gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


def _number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for path in sorted(root.glob("seed_*/synthetic_results.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows.extend(csv.DictReader(handle))
    if len(rows) != 3:
        raise SystemExit(f"Expected three E1 rows, found {len(rows)}")
    summary: dict[str, object] = {
        "seeds": [int(row["seed"]) for row in rows],
        "median_delay_corr": statistics.median(_number(row, "delay_corr") for row in rows),
        "max_delay_mae": max(_number(row, "delay_mae") for row in rows),
        "max_zero_delay_mass": max(_number(row, "zero_delay_mass") for row in rows),
        "median_route_density": statistics.median(
            _number(row, "effective_route_density") for row in rows
        ),
    }
    checks = {
        "delay_corr": summary["median_delay_corr"] >= 0.60,
        "delay_mae": summary["max_delay_mae"] <= 2.5,
        "zero_delay_mass": summary["max_zero_delay_mass"] <= 0.70,
        "route_density": 0.15 <= summary["median_route_density"] <= 0.25,
    }
    control_patterns = {
        "scalp": "controls/seed_*/E2/scalp_mixing_results.csv",
        "phase": "controls/seed_*/E3/phase_surrogate_results.csv",
        "label": "controls/seed_*/E3/label_shuffle_results.csv",
        "reversal": "controls/seed_*/E3/time_reversal_results.csv",
    }
    controls: dict[str, list[dict[str, str]]] = {}
    for name, pattern in control_patterns.items():
        control_rows = []
        for path in sorted(root.glob(pattern)):
            with path.open(newline="", encoding="utf-8-sig") as handle:
                control_rows.extend(csv.DictReader(handle))
        controls[name] = control_rows
    if all(len(control_rows) == 3 for control_rows in controls.values()):
        normal_accuracy = statistics.median(_number(row, "accuracy") for row in rows)
        control_summary = {
            "median_scalp_delay_corr_reference": statistics.median(
                _number(row, "delay_corr_global") for row in controls["scalp"]
            ),
            "median_phase_surrogate_delay_corr": statistics.median(
                _number(row, "delay_corr") for row in controls["phase"]
            ),
            "median_time_reversal_delay_corr": statistics.median(
                _number(row, "delay_corr") for row in controls["reversal"]
            ),
            "median_label_shuffle_accuracy": statistics.median(
                _number(row, "accuracy") for row in controls["label"]
            ),
            "median_normal_accuracy": normal_accuracy,
            "median_phase_surrogate_accuracy": statistics.median(
                _number(row, "accuracy") for row in controls["phase"]
            ),
        }
        summary["controls"] = control_summary
        checks.update(
            {
                "common_source": control_summary["median_scalp_delay_corr_reference"]
                <= summary["median_delay_corr"] - 0.25,
                "phase_surrogate_delay": control_summary["median_phase_surrogate_delay_corr"]
                <= summary["median_delay_corr"] - 0.20,
                "time_reversal": control_summary["median_time_reversal_delay_corr"] >= 0.60,
                "label_shuffle": control_summary["median_label_shuffle_accuracy"] <= 0.35,
            }
        )
        summary["limitations"] = {
            "phase_surrogate_retains_classification": control_summary[
                "median_phase_surrogate_accuracy"
            ]
            >= normal_accuracy - 0.10,
            "interpretation": "Amplitude-spectrum information remains discriminative; classification is not predominantly phase-delay driven.",
        }
    summary["checks"] = checks
    passed = all(checks.values())
    has_limitation = bool(
        isinstance(summary.get("limitations"), dict)
        and summary["limitations"].get("phase_surrogate_retains_classification")
    )
    summary["status"] = "passed_with_limitation" if passed and has_limitation else "passed" if passed else "failed"
    summary["promotion_allowed"] = passed
    (root / "mechanism_gate.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
