"""Create the immutable checkpoint barrier for reviewer-control evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v8_protocol import (
    collect_source_tree_manifest,
    source_tree_digest,
)
from dpc_snn.experiments.v62_protocol import file_sha256
from dpc_snn.utils.io import read_json, write_json


def budgets(config: dict, dataset: str, variant: str) -> list[str]:
    spec = config["variants"][variant]
    if spec["budget_group"] == "fixed_budget_controls":
        return [str(value) for value in config["fixed_budget_controls"]]
    return [str(value) for value in config["datasets"][dataset]["equal_update_budgets"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "experiments" / "reviewer_controls.yaml")
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--barrier", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = Path(args.output).resolve()
    checkpoints: dict[str, str] = {}
    seeds = [int(config["seeds"][0])] if args.smoke else [int(value) for value in config["seeds"]]
    for dataset, dataset_config in config["datasets"].items():
        subjects = (
            [int(dataset_config["subjects"][0])]
            if args.smoke
            else [int(value) for value in dataset_config["subjects"]]
        )
        for subject in subjects:
            status_path = output / dataset / f"subject_{subject:02d}" / "training_status.json"
            if read_json(status_path).get("status") != "completed":
                raise RuntimeError(f"subject training is incomplete: {status_path}")
            for seed in seeds:
                for variant in config["variants"]:
                    selected = budgets(config, dataset, variant)
                    if args.smoke:
                        selected = sorted({selected[0], selected[-1]})
                    for budget in selected:
                        path = (
                            output
                            / dataset
                            / f"subject_{subject:02d}"
                            / f"seed_{seed}"
                            / f"budget_{budget}"
                            / variant
                            / "checkpoint.pt"
                        )
                        if not path.is_file():
                            raise FileNotFoundError(path)
                        checkpoints[str(path.resolve())] = file_sha256(path)
    write_json(
        Path(args.barrier).resolve(),
        {
            "schema": "dpc-snn-reviewer-checkpoint-barrier/v1",
            "status": "sealed",
            "lineage": config["lineage"],
            "config_sha256": file_sha256(config_path),
            "source_tree_sha256": source_tree_digest(collect_source_tree_manifest(ROOT)),
            "smoke": bool(args.smoke),
            "checkpoint_count": len(checkpoints),
            "checkpoints": dict(sorted(checkpoints.items())),
        },
    )
    print(f"sealed {len(checkpoints)} reviewer-control checkpoints")


if __name__ == "__main__":
    main()
