#!/usr/bin/env python3
"""Aggregate independently executed V7 E3 subject shards."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import file_sha256  # noqa: E402
from dpc_snn.experiments.v7_delay import paired_delay_gate  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    integer_fields = {
        "subject",
        "seed",
        "folds",
        "first_only_correct",
        "second_only_correct",
        "discordant_predictions",
    }
    float_fields = {
        "full_accuracy",
        "locked_zero_accuracy",
        "delta_accuracy",
        "amplitude_relative_difference",
        "exact_mcnemar_p",
    }
    boolean_fields = {"complete_oof", "session_e_accessed"}
    converted = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        for field in integer_fields & item.keys():
            item[field] = int(item[field])
        for field in float_fields & item.keys():
            item[field] = float(item[field])
        for field in boolean_fields & item.keys():
            item[field] = _as_bool(item[field])
        converted.append(item)
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config",
        default="configs/experiments/v7_e3_static_slow_within.yaml",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = ensure_dir(Path(args.output).resolve())
    rows: list[dict[str, Any]] = []
    shards = []
    for raw_root in args.inputs:
        root = Path(raw_root).resolve()
        summary = root / "paired_summary.csv"
        status_path = root / "campaign_status.json"
        if not summary.is_file() or not status_path.is_file():
            raise FileNotFoundError(f"incomplete E3 shard: {root}")
        status = read_json(status_path)
        if bool(status.get("session_e_accessed", True)):
            raise RuntimeError(f"E3 shard accessed Session E: {root}")
        shard_rows = _read_rows(summary)
        if not shard_rows:
            raise RuntimeError(f"E3 shard contains no paired rows: {root}")
        for row in shard_rows:
            if not bool(row.get("complete_oof", False)):
                raise RuntimeError(f"E3 shard has incomplete OOF predictions: {root}")
            if bool(row.get("session_e_accessed", True)):
                raise RuntimeError(f"E3 row accessed Session E: {root}")
        rows.extend(shard_rows)
        shards.append(
            {
                "root": str(root),
                "paired_summary_sha256": file_sha256(summary),
                "campaign_status_sha256": file_sha256(status_path),
                "rows": len(shard_rows),
            }
        )

    keys = [(int(row["subject"]), int(row["seed"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate subject-seed pair across E3 shards")
    rows.sort(key=lambda row: (int(row["subject"]), int(row["seed"])))
    expected = {
        (int(subject), int(seed))
        for subject in config["subjects"]
        for seed in config["seeds"]
    }
    observed = set(keys)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise RuntimeError(
            f"E3 aggregate pair mismatch: missing={missing}, unexpected={unexpected}"
        )

    gate = paired_delay_gate(rows, config)
    write_csv(output / "paired_summary.csv", rows)
    write_json(output / "gate_report.json", gate)
    write_json(
        output / "aggregation_manifest.json",
        {
            "stage": config["stage"],
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "aggregator_sha256": file_sha256(Path(__file__)),
            "shards": shards,
            "subject_seed_pairs": [list(key) for key in sorted(observed)],
        },
    )
    write_json(
        output / "campaign_status.json",
        {
            "status": "completed",
            "stage": config["stage"],
            "session_e_accessed": False,
            "gate": gate,
        },
    )
    print(gate)


if __name__ == "__main__":
    main()
