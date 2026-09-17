#!/usr/bin/env python
"""Run E0 environment/data audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.config import load_experiment_config
from dpc_snn.experiments.runners import run_environment_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/all_experiments.yaml")
    parser.add_argument("--output", default="runs/E0_environment_data_reproducibility")
    args = parser.parse_args()
    cfg = load_experiment_config(args.config, "E0")
    run_environment_audit(cfg, Path(args.output))
    print(f"environment audit written to {args.output}")


if __name__ == "__main__":
    main()

