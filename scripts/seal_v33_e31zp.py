#!/usr/bin/env python3
"""Seal the complete E31-ZP checkpoint set before evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import file_sha256  # noqa: E402
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    source_tree_digest,
)
from dpc_snn.utils.io import read_json, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--barrier", required=True)
    parser.add_argument(
        "--config", default="configs/experiments/v33_e31_zero_penalty.yaml"
    )
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = Path(args.output).resolve()
    checkpoints: dict[str, str] = {}
    for dataset, dataset_config in config["datasets"].items():
        budgets = config["budgets"][dataset]
        for subject in dataset_config["subjects"]:
            status = read_json(
                output / dataset / f"subject_{int(subject):02d}" / "training_status.json"
            )
            if status.get("status") != "completed":
                raise RuntimeError(f"incomplete E31-ZP subject: {dataset} {subject}")
            for seed in config["seeds"]:
                for budget in budgets:
                    path = (
                        output
                        / dataset
                        / f"subject_{int(subject):02d}"
                        / f"seed_{int(seed)}"
                        / f"budget_{budget}"
                        / "sew_clif_ce_fr0"
                        / "checkpoint.pt"
                    ).resolve()
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    checkpoints[str(path)] = file_sha256(path)
    expected = sum(
        len(value["subjects"]) * len(config["seeds"]) * len(config["budgets"][name])
        for name, value in config["datasets"].items()
    )
    if len(checkpoints) != expected:
        raise RuntimeError(f"expected {expected} E31-ZP checkpoints, found {len(checkpoints)}")
    write_json(
        Path(args.barrier).resolve(),
        {
            "schema": "dpc-snn-e31zp-checkpoint-barrier/v1",
            "status": "sealed",
            "config_sha256": file_sha256(config_path),
            "source_tree_sha256": source_tree_digest(collect_source_tree_manifest(ROOT)),
            "checkpoint_count": len(checkpoints),
            "checkpoints": checkpoints,
        },
    )


if __name__ == "__main__":
    main()
