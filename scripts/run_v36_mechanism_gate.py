#!/usr/bin/env python
"""Run the complete three-seed V3.6 synthetic mechanism gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default="configs/experiments/v36_mechanism_gate.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    for seed in (0, 1, 2):
        jobs = (
            ("E1", root / f"seed_{seed}"),
            ("E2", root / "controls" / f"seed_{seed}" / "E2"),
            ("E3", root / "controls" / f"seed_{seed}" / "E3"),
        )
        for experiment, output in jobs:
            print(f"__DS_PROGRESS__ seed={seed} experiment={experiment} status=started", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_experiment.py",
                    "--experiment",
                    experiment,
                    "--config",
                    args.config,
                    "--output",
                    str(output),
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                ],
                check=True,
            )
            print(f"__DS_PROGRESS__ seed={seed} experiment={experiment} status=completed", flush=True)

    subprocess.run(
        [sys.executable, "scripts/evaluate_v35_mechanism_gate.py", "--root", str(root)],
        check=True,
    )


if __name__ == "__main__":
    main()
