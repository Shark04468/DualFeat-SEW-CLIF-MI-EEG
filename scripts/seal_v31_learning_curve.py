#!/usr/bin/env python3
"""Seal every V31 student checkpoint before any evaluation pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import file_sha256, sha256_fingerprint  # noqa: E402
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    source_tree_digest,
)
from dpc_snn.utils.io import read_json, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--barrier", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    barrier_path = Path(args.barrier).resolve()
    config_path = ROOT / "configs" / "experiments" / "v31_decoder_learning_curve.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_sha256 = file_sha256(config_path)
    source_sha256 = source_tree_digest(collect_source_tree_manifest(ROOT))
    statuses: list[dict[str, object]] = []
    checkpoints: dict[str, str] = {}
    for dataset, dataset_config in config["datasets"].items():
        for subject in dataset_config["subjects"]:
            path = output / dataset / f"subject_{int(subject):02d}" / "training_status.json"
            if not path.is_file():
                raise RuntimeError(f"V31 training is incomplete: {path}")
            status = read_json(path)
            if (
                status.get("status") != "completed"
                or status.get("config_sha256") != config_sha256
                or status.get("source_tree_sha256") != source_sha256
                or status.get("evaluation_data_accessed_during_training") is not False
            ):
                raise RuntimeError(f"invalid V31 training status: {path}")
            statuses.append(status)
            for checkpoint_value in status["checkpoints"]:
                checkpoint = Path(str(checkpoint_value)).resolve()
                if not checkpoint.is_file():
                    raise RuntimeError(f"missing V31 checkpoint: {checkpoint}")
                checkpoints[str(checkpoint)] = file_sha256(checkpoint)
    payload = {
        "schema": "dpc-snn-v31-global-checkpoint-barrier/v1",
        "status": "sealed",
        "config_sha256": config_sha256,
        "source_tree_sha256": source_sha256,
        "subjects": len(statuses),
        "checkpoint_count": len(checkpoints),
        "checkpoints": dict(sorted(checkpoints.items())),
        "evaluation_access_before_barrier": False,
    }
    payload["combined_sha256"] = sha256_fingerprint(payload)
    if barrier_path.exists():
        if read_json(barrier_path) != payload:
            raise RuntimeError(f"existing V31 barrier does not match: {barrier_path}")
    else:
        barrier_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(barrier_path, payload)
    print(
        json.dumps(
            {
                "status": "sealed",
                "subjects": len(statuses),
                "checkpoints": len(checkpoints),
                "combined_sha256": payload["combined_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
