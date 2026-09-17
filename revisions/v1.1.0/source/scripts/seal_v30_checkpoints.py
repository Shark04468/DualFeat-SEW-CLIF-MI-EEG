#!/usr/bin/env python3
"""Seal every V30 teacher and matched-student checkpoint before held-out access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import file_sha256, sha256_fingerprint
from dpc_snn.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "v30_bnci2014_004_blind.yaml"),
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    freeze = read_json(Path(args.freeze).resolve())
    config = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    if list(root.glob("subject_*/evaluation_status.json")) or list(
        root.glob("subject_*/seed_*/evaluation")
    ):
        raise RuntimeError("V30 held-out artifacts exist before the checkpoint barrier")

    checkpoints: dict[str, str] = {}
    teacher_counts: dict[str, set[int]] = {model: set() for model in config["teacher_models"]}
    student_counts: dict[str, set[int]] = {variant: set() for variant in config["variants"]}
    for subject in config["dataset"]["subjects"]:
        status = read_json(root / f"subject_{subject:02d}" / "training_status.json")
        if (
            status.get("status") != "completed"
            or status.get("evaluation_sessions_accessed") is not False
        ):
            raise RuntimeError(f"V30 Subject {subject} training is incomplete or contaminated")
        for seed in config["seeds"]:
            for model in config["teacher_models"]:
                path = (
                    root
                    / f"subject_{subject:02d}"
                    / f"seed_{seed}"
                    / "teachers"
                    / model
                    / "checkpoint.pt"
                )
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                if checkpoint.get("freeze_sha256") != freeze["combined_sha256"]:
                    raise RuntimeError(f"V30 teacher belongs to another freeze: {path}")
                teacher_counts[model].add(
                    int(sum(value.numel() for value in checkpoint["state_dict"].values()))
                )
                checkpoints[str(path)] = file_sha256(path)

            paired_counts: list[int] = []
            for variant in config["variants"]:
                path = (
                    root
                    / f"subject_{subject:02d}"
                    / f"seed_{seed}"
                    / "students"
                    / variant
                    / "checkpoint.pt"
                )
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                if checkpoint.get("freeze_sha256") != freeze["combined_sha256"]:
                    raise RuntimeError(f"V30 student belongs to another freeze: {path}")
                count = int(sum(value.numel() for value in checkpoint["state_dict"].values()))
                paired_counts.append(count)
                student_counts[variant].add(count)
                checkpoints[str(path)] = file_sha256(path)
            if len(set(paired_counts)) != 1:
                raise RuntimeError(
                    f"V30 ANN/SNN capacity mismatch for Subject {subject} seed {seed}"
                )

    expected = (
        len(config["dataset"]["subjects"])
        * len(config["seeds"])
        * (len(config["teacher_models"]) + len(config["variants"]))
    )
    if len(checkpoints) != expected:
        raise RuntimeError(f"V30 checkpoint count mismatch: {len(checkpoints)} != {expected}")
    payload = {
        "schema": "dpc-snn-v30-global-checkpoint-barrier/v1",
        "status": "sealed",
        "freeze_sha256": freeze["combined_sha256"],
        "expected_checkpoints": expected,
        "capacity_matched_students": True,
        "teacher_parameter_counts": {key: sorted(value) for key, value in teacher_counts.items()},
        "student_parameter_counts": {key: sorted(value) for key, value in student_counts.items()},
        "checkpoints": dict(sorted(checkpoints.items())),
        "evaluation_sessions_accessed_before_barrier": False,
    }
    payload["combined_sha256"] = sha256_fingerprint(payload)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"V30 checkpoint barrier must be new: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, payload)
    print(
        json.dumps({key: value for key, value in payload.items() if key != "checkpoints"}, indent=2)
    )


if __name__ == "__main__":
    main()
