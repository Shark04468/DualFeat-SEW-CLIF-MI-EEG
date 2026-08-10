#!/usr/bin/env python
"""Create uncertainty-aware summaries for the V5.1 synthetic power matrix."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.config import load_yaml  # noqa: E402
from dpc_snn.utils.io import ensure_dir, write_csv, write_json  # noqa: E402


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total**2))
        / denominator
    )
    return centre - radius, centre + radius


def _power_row(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(_as_bool(row["known_route_accepted"]) for row in rows)
    lower, upper = _wilson(successes, len(rows))
    return {
        "scope": label,
        "n": len(rows),
        "detected": successes,
        "power": successes / len(rows) if rows else float("nan"),
        "wilson_95_lower": lower,
        "wilson_95_upper": upper,
        "median_delay_mae_ms": float(
            np.median([float(row["known_delay_absolute_error_ms"]) for row in rows])
        )
        if rows
        else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--protocol-config", default="configs/experiments/v51_hardware_free_plan.yaml"
    )
    args = parser.parse_args()

    with Path(args.input).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = ensure_dir(args.output)
    protocol = load_yaml(ROOT / args.protocol_config)
    threshold = float(protocol["mechanism_power"]["minimum_direction_accuracy"])
    coupled = [row for row in rows if row["condition"] == "coupled"]
    grouped: list[dict[str, Any]] = []
    for strength in sorted({float(row["strength"]) for row in coupled}):
        for delay_ms in sorted({float(row["delay_ms"]) for row in coupled}):
            selected = [
                row
                for row in coupled
                if float(row["strength"]) == strength and float(row["delay_ms"]) == delay_ms
            ]
            grouped.append(
                {
                    "strength": strength,
                    "delay_ms": delay_ms,
                    **_power_row(f"strength={strength};delay_ms={delay_ms}", selected),
                }
            )
    write_csv(output / "power_by_strength_delay.csv", grouped)

    primary = [
        row for row in coupled if float(row["strength"]) >= 0.25 and float(row["delay_ms"]) >= 4.0
    ]
    primary_rows = [_power_row("primary_all", primary)]
    for strength in sorted({float(row["strength"]) for row in primary}):
        primary_rows.append(
            _power_row(
                f"primary_strength={strength}",
                [row for row in primary if float(row["strength"]) == strength],
            )
        )
    write_csv(output / "primary_power_uncertainty.csv", primary_rows)

    maximum_strength = max(float(row["strength"]) for row in coupled)
    reversal_rows: list[dict[str, Any]] = []
    for delay_ms in sorted({float(row["delay_ms"]) for row in coupled}):
        selected = [
            row
            for row in coupled
            if float(row["strength"]) == maximum_strength and float(row["delay_ms"]) == delay_ms
        ]
        reversal_rows.append(
            {
                "delay_ms": delay_ms,
                "n": len(selected),
                "normal_detection_rate": float(
                    np.mean([_as_bool(row["known_route_accepted"]) for row in selected])
                ),
                "time_reversal_transpose_rate": float(
                    np.mean([_as_bool(row["time_reversal_transpose_accepted"]) for row in selected])
                ),
                "phase_surrogate_false_positive_rate": float(
                    np.mean(
                        [_as_bool(row["phase_surrogate_known_route_accepted"]) for row in selected]
                    )
                ),
            }
        )
    write_csv(output / "negative_control_by_delay.csv", reversal_rows)

    primary_summary = primary_rows[0]
    interpretation = {
        "pre_registered_point_gate_passed": float(primary_summary["power"]) >= threshold,
        "confidence_qualified_gate_passed": float(primary_summary["wilson_95_lower"]) >= threshold,
        "minimum_direction_accuracy": threshold,
        "primary": primary_summary,
        "note": (
            "The confidence-qualified result is a robustness diagnostic added without changing "
            "the preregistered point-estimate decision."
        ),
    }
    write_json(output / "uncertainty_decision.json", interpretation)


if __name__ == "__main__":
    main()
