#!/usr/bin/env python
"""Run all experiments marked priority=must in registry order."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.utils.storage import configure_cache_env

configure_cache_env()

from dpc_snn.config import load_experiment_config, load_yaml
from dpc_snn.experiments.runners import run_registered_experiment
from dpc_snn.utils.io import ensure_dir, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/all_experiments.yaml")
    parser.add_argument("--output", default="runs/must_package")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    registry = load_yaml(args.config)
    output_root = ensure_dir(args.output)
    rows = []
    for exp_id, exp in registry.get("experiments", {}).items():
        if exp.get("priority") != "must":
            continue
        cfg = load_experiment_config(args.config, exp_id)
        out = output_root / f"{exp_id}_{exp['name']}"
        try:
            run_registered_experiment(cfg, out)
            rows.append({"experiment_id": exp_id, "name": exp["name"], "status": "completed_or_recorded", "output": str(out)})
        except Exception as exc:
            rows.append({"experiment_id": exp_id, "name": exp["name"], "status": "failed", "output": str(out), "error": repr(exc)})
            traceback.print_exc()
            if not args.continue_on_error:
                break
    write_csv(output_root / "must_package_summary.csv", rows)
    print(f"wrote {output_root / 'must_package_summary.csv'}")


if __name__ == "__main__":
    main()
