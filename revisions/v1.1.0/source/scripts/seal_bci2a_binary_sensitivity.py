"""Seal every REV-E5 BCI2a binary checkpoint before evaluation access."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v8_protocol import collect_source_tree_manifest, source_tree_digest
from dpc_snn.experiments.v62_protocol import file_sha256
from dpc_snn.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "bci2a_binary_sensitivity.yaml"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--barrier", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = Path(args.output).resolve()
    subjects = (
        [int(config["dataset"]["subjects"][0])]
        if args.smoke
        else [int(value) for value in config["dataset"]["subjects"]]
    )
    seeds = [int(config["seeds"][0])] if args.smoke else [int(value) for value in config["seeds"]]
    checkpoints: dict[str, str] = {}
    for subject in subjects:
        status = read_json(
            output / "bci2a_binary" / f"subject_{subject:02d}" / "training_status.json"
        )
        if (
            status.get("status") != "completed"
            or status.get("evaluation_data_accessed") is not False
        ):
            raise RuntimeError(f"REV-E5 subject training is incomplete: {subject}")
        for seed in seeds:
            seed_root = output / "bci2a_binary" / f"subject_{subject:02d}" / f"seed_{seed}"
            budget_dirs = sorted(seed_root.glob("budget_*"))
            if args.smoke:
                budget_dirs = [budget_dirs[0], budget_dirs[-1]]
            for budget_dir in budget_dirs:
                for variant in config["variants"]:
                    checkpoint = budget_dir / variant / "checkpoint.pt"
                    if not checkpoint.is_file():
                        raise FileNotFoundError(checkpoint)
                    checkpoints[str(checkpoint.resolve())] = file_sha256(checkpoint)
    write_json(
        Path(args.barrier).resolve(),
        {
            "schema": "dpc-snn-rev-e5-checkpoint-barrier/v1",
            "status": "sealed",
            "lineage": config["lineage"],
            "config_sha256": file_sha256(config_path),
            "source_tree_sha256": source_tree_digest(collect_source_tree_manifest(ROOT)),
            "smoke": bool(args.smoke),
            "checkpoint_count": len(checkpoints),
            "checkpoints": dict(sorted(checkpoints.items())),
        },
    )
    print(f"sealed {len(checkpoints)} REV-E5 checkpoints")


if __name__ == "__main__":
    main()
