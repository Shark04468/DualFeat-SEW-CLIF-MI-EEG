#!/usr/bin/env python3
"""Seal E29-ZP or E30-ZP checkpoints before evaluation."""

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


SETTINGS = {
    "openbmi": {
        "config": "v33_e29_zero_penalty.yaml",
        "schema": "dpc-snn-e29zp-checkpoint-barrier/v1",
        "subject_key": "subjects",
        "path": "{root}/subject_{subject:02d}/seed_{seed}/sew_clif_ce_fr0/checkpoint.pt",
    },
    "bnci2014_004": {
        "config": "v33_e30_zero_penalty.yaml",
        "schema": "dpc-snn-e30zp-checkpoint-barrier/v1",
        "subject_key": "dataset.subjects",
        "path": (
            "{root}/subject_{subject:02d}/seed_{seed}/students/"
            "sew_clif_ce_fr0/checkpoint.pt"
        ),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(SETTINGS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--barrier", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    setting = SETTINGS[args.dataset]
    config_path = (
        (ROOT / args.config).resolve()
        if args.config
        else ROOT / "configs" / "experiments" / str(setting["config"])
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    subjects = (
        config["subjects"]
        if args.dataset == "openbmi"
        else config["dataset"]["subjects"]
    )
    root = Path(args.output).resolve()
    checkpoints: dict[str, str] = {}
    for subject_value in subjects:
        subject = int(subject_value)
        status = read_json(root / f"subject_{subject:02d}" / "training_status.json")
        if status.get("status") != "completed":
            raise RuntimeError(f"incomplete {args.dataset} subject {subject}")
        for seed_value in config["seeds"]:
            path = Path(
                str(setting["path"]).format(
                    root=root, subject=subject, seed=int(seed_value)
                )
            ).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            checkpoints[str(path)] = file_sha256(path)
    expected = len(subjects) * len(config["seeds"])
    if len(checkpoints) != expected:
        raise RuntimeError(f"expected {expected} checkpoints, found {len(checkpoints)}")
    write_json(
        Path(args.barrier).resolve(),
        {
            "schema": setting["schema"],
            "status": "sealed",
            "dataset": args.dataset,
            "config_sha256": file_sha256(config_path),
            "source_tree_sha256": source_tree_digest(collect_source_tree_manifest(ROOT)),
            "checkpoint_count": len(checkpoints),
            "checkpoints": checkpoints,
        },
    )


if __name__ == "__main__":
    main()
