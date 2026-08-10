#!/usr/bin/env python3
"""Verify and seal all V29 S1-trained students before opening S2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import file_sha256, sha256_fingerprint  # noqa: E402
from dpc_snn.utils.io import read_json, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    freeze = read_json(Path(args.freeze).resolve())
    config = yaml.safe_load(
        (ROOT / "configs" / "experiments" / "v29_openbmi_replication.yaml").read_text(
            encoding="utf-8"
        )
    )
    checkpoints: dict[str, str] = {}
    parameter_counts: dict[str, set[int]] = {variant: set() for variant in config["variants"]}
    for subject in config["subjects"]:
        status = read_json(root / f"subject_{subject:02d}" / "training_status.json")
        if status.get("status") != "completed" or status.get("openbmi_s2_accessed") is not False:
            raise RuntimeError(f"V29 Subject {subject} S1 training is incomplete or contaminated")
        for seed in config["seeds"]:
            paired_counts: list[int] = []
            for variant in config["variants"]:
                directory = root / f"subject_{subject:02d}" / f"seed_{seed}" / variant
                metrics = read_json(directory / "training_metrics.json")
                if metrics.get("status") != "training_completed_checkpoint_sealed":
                    raise RuntimeError(f"V29 training metrics are incomplete: {directory}")
                if metrics.get("openbmi_s2_accessed") is not False:
                    raise RuntimeError(f"V29 training accessed OpenBMI S2: {directory}")
                checkpoint_path = directory / "checkpoint.pt"
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if checkpoint.get("freeze_sha256") != freeze["combined_sha256"]:
                    raise RuntimeError(f"V29 checkpoint belongs to another freeze: {checkpoint_path}")
                count = int(sum(value.numel() for value in checkpoint["state_dict"].values()))
                paired_counts.append(count)
                parameter_counts[variant].add(count)
                checkpoints[str(checkpoint_path)] = file_sha256(checkpoint_path)
            if len(set(paired_counts)) != 1:
                raise RuntimeError(f"V29 ANN/SNN capacity mismatch for Subject {subject} seed {seed}")
    expected = len(config["subjects"]) * len(config["seeds"]) * len(config["variants"])
    if len(checkpoints) != expected:
        raise RuntimeError(f"V29 checkpoint count mismatch: {len(checkpoints)} != {expected}")
    payload = {
        "schema": "dpc-snn-v29-student-checkpoint-barrier/v1",
        "status": "sealed",
        "freeze_sha256": freeze["combined_sha256"],
        "expected_checkpoints": expected,
        "capacity_matched": True,
        "parameter_counts": {key: sorted(value) for key, value in parameter_counts.items()},
        "checkpoints": dict(sorted(checkpoints.items())),
        "openbmi_s2_accessed_before_barrier": False,
    }
    payload["combined_sha256"] = sha256_fingerprint(payload)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"V29 checkpoint barrier must be new: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "checkpoints"}, indent=2))


if __name__ == "__main__":
    main()
