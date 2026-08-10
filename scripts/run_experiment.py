#!/usr/bin/env python
"""Run a registered DPC-SNN experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.utils.storage import configure_cache_env

configure_cache_env()

from dpc_snn.config import load_experiment_config
from dpc_snn.experiments.runners import run_registered_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="Experiment id, e.g. E1")
    parser.add_argument("--config", default="configs/experiments/all_experiments.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.device is not None:
        overrides["device"] = args.device
    training = {}
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    if training:
        overrides["training"] = training
    cfg = load_experiment_config(args.config, args.experiment, overrides=overrides)
    exp_name = cfg["experiment"]["name"]
    output = Path(args.output) if args.output else Path(cfg.get("output_root", "runs")) / f"{args.experiment}_{exp_name}"
    result = run_registered_experiment(cfg, output)
    print(f"completed {args.experiment} -> {output}")
    print(result)


if __name__ == "__main__":
    main()
