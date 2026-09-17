#!/usr/bin/env python
"""Validate and merge disjoint V5.1 OpenBMI audit shards."""

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
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402


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


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row_key(row: dict[str, Any]) -> tuple[int, str, int, float, float]:
    return (
        int(row["subject"]),
        str(row["evidence_space"]),
        int(row["injection_seed"]),
        float(row["strength"]),
        float(row["true_delay_ms"]),
    )


def _summary_row(scope: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    detected = np.asarray([_as_bool(row["known_direction_detected"]) for row in rows])
    errors = np.asarray(
        [float(row["known_delay_absolute_error_ms"]) for row in rows], dtype=np.float64
    )
    finite = np.isfinite(errors)
    joint = detected & finite & (errors <= 2.0)
    lower, upper = _wilson(int(detected.sum()), len(rows))
    joint_lower, joint_upper = _wilson(int(joint.sum()), len(rows))
    return {
        "scope": scope,
        "n": len(rows),
        "direction_detected": int(detected.sum()),
        "direction_accuracy": float(detected.mean()) if len(rows) else float("nan"),
        "direction_wilson_95_lower": lower,
        "direction_wilson_95_upper": upper,
        "joint_direction_delay_success": int(joint.sum()),
        "joint_direction_delay_accuracy": float(joint.mean()) if len(rows) else float("nan"),
        "joint_wilson_95_lower": joint_lower,
        "joint_wilson_95_upper": joint_upper,
        "median_delay_mae_ms_among_detected": float(np.median(errors[detected & finite]))
        if np.any(detected & finite)
        else float("nan"),
        "undetected_count": int((~detected).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--protocol-config", default="configs/experiments/v51_hardware_free_plan.yaml"
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output = ensure_dir(args.output)
    shard_dirs = sorted(path.parent for path in input_root.glob("*/audit_manifest.json"))
    if not shard_dirs:
        raise FileNotFoundError(f"No V5.1 audit shards found under {input_root}")

    manifests: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str, int, float, float]] = set()
    for shard in shard_dirs:
        exit_path = shard / "exit_code.txt"
        if not exit_path.exists() or exit_path.read_text(encoding="utf-8").strip() != "0":
            raise RuntimeError(f"Shard did not exit successfully: {shard}")
        manifest = read_json(shard / "audit_manifest.json")
        manifests.append(manifest)
        shard_rows = _read_csv(shard / "real_background_injection_power.csv")
        expected = (
            len(manifest["subjects"])
            * len(manifest["spaces"])
            * len(manifest["injection_seeds"])
            * len(manifest["injection_strengths"])
            * len(manifest["injection_delays_ms"])
        )
        if len(shard_rows) != expected:
            raise RuntimeError(
                f"Shard {shard} contains {len(shard_rows)} rows, expected {expected}"
            )
        for row in shard_rows:
            key = _row_key(row)
            if key in seen:
                raise RuntimeError(f"Duplicate scientific condition across shards: {key}")
            seen.add(key)
            rows.append(row)

    fingerprints = {str(manifest["source_fingerprint"]) for manifest in manifests}
    experiments = {str(manifest["experiment_id"]) for manifest in manifests}
    if len(fingerprints) != 1 or len(experiments) != 1:
        raise RuntimeError("Shard source fingerprint or experiment identity differs")
    rows.sort(key=_row_key)
    write_csv(output / "real_background_injection_power.csv", rows)

    grouped: list[dict[str, Any]] = []
    spaces = sorted({str(row["evidence_space"]) for row in rows})
    strengths = sorted({float(row["strength"]) for row in rows})
    delays = sorted({float(row["true_delay_ms"]) for row in rows})
    for space in spaces:
        for strength in strengths:
            for delay in delays:
                selected = [
                    row
                    for row in rows
                    if str(row["evidence_space"]) == space
                    and float(row["strength"]) == strength
                    and float(row["true_delay_ms"]) == delay
                ]
                grouped.append(
                    {
                        "evidence_space": space,
                        "strength": strength,
                        "true_delay_ms": delay,
                        **_summary_row(
                            f"space={space};strength={strength};delay_ms={delay}", selected
                        ),
                    }
                )
    write_csv(output / "power_by_space_strength_delay.csv", grouped)

    primary = [
        row for row in rows if float(row["strength"]) >= 0.25 and float(row["true_delay_ms"]) >= 4.0
    ]
    primary_rows = [_summary_row("primary_all", primary)]
    for space in spaces:
        primary_rows.append(
            _summary_row(
                f"primary_space={space}",
                [row for row in primary if str(row["evidence_space"]) == space],
            )
        )
    write_csv(output / "primary_power_uncertainty.csv", primary_rows)

    protocol = load_yaml(ROOT / args.protocol_config)
    power = protocol["mechanism_power"]
    overall = primary_rows[0]
    space_summaries = {
        row["scope"].split("=", maxsplit=1)[1]: row
        for row in primary_rows[1:]
        if row["scope"].startswith("primary_space=")
    }
    passing_spaces = sorted(
        space
        for space, row in space_summaries.items()
        if float(row["direction_accuracy"]) >= float(power["minimum_direction_accuracy"])
        and float(row["median_delay_mae_ms_among_detected"])
        <= float(power["maximum_median_delay_mae_ms"])
    )
    confidence_qualified_spaces = sorted(
        space
        for space, row in space_summaries.items()
        if float(row["direction_wilson_95_lower"]) >= float(power["minimum_direction_accuracy"])
        and float(row["median_delay_mae_ms_among_detected"])
        <= float(power["maximum_median_delay_mae_ms"])
    )
    point_passed = bool(passing_spaces)
    decision = {
        "status": "passed" if point_passed else "failed",
        "pre_registered_point_gate_passed": point_passed,
        "confidence_qualified_gate_passed": bool(confidence_qualified_spaces),
        "passing_evidence_spaces": passing_spaces,
        "confidence_qualified_evidence_spaces": confidence_qualified_spaces,
        "classifier_training": False,
        "heldout_session_S2_accessed": False,
        "source_fingerprint": next(iter(fingerprints)),
        "shards": [str(path) for path in shard_dirs],
        "primary": overall,
        "primary_by_evidence_space": space_summaries,
        "thresholds": {
            "minimum_direction_accuracy": power["minimum_direction_accuracy"],
            "maximum_median_delay_mae_ms": power["maximum_median_delay_mae_ms"],
        },
        "note": (
            "Joint direction-and-delay accuracy and Wilson intervals are robustness diagnostics; "
            "they do not replace the preregistered point-estimate gate."
        ),
    }
    write_json(output / "stage_decision.json", decision)
    write_json(
        output / "merged_manifest.json",
        {
            "experiment_id": next(iter(experiments)),
            "source_fingerprint": next(iter(fingerprints)),
            "shard_count": len(shard_dirs),
            "row_count": len(rows),
            "unique_condition_count": len(seen),
            "classifier_training": False,
            "heldout_session_S2_accessed": False,
        },
    )


if __name__ == "__main__":
    main()
