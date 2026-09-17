#!/usr/bin/env python
"""Prepare real continuous PhysioNet EEGMMI windows for E13."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.data.physionet import prepare_physionet_async_npz  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--task", default="imagined_left_right", choices=["imagined_left_right", "imagined_hands_feet"])
    parser.add_argument("--root", default="data/raw/physionet_eegmmi")
    parser.add_argument("--output", default="data/processed/async_physionet")
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--step-sec", type=float, default=0.25)
    parser.add_argument("--interval-sec", type=float, default=4.0)
    args = parser.parse_args()

    configure_cache_env()
    for path in prepare_physionet_async_npz(
        args.subjects,
        args.task,
        args.root,
        args.output,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        interval_sec=args.interval_sec,
    ):
        print(path)


if __name__ == "__main__":
    main()
