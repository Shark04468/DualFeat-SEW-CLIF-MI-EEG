#!/usr/bin/env python
"""Prepare PhysioNet EEGMMI runs into processed NPZ files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.utils.storage import configure_cache_env

configure_cache_env()

from dpc_snn.data.physionet import prepare_physionet_npz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--task", default="imagined_left_right", choices=["imagined_left_right", "imagined_hands_feet", "rest_vs_mi"])
    parser.add_argument("--root", default="data/raw/physionet_eegmmi")
    parser.add_argument("--output", default="data/processed/physionet_eegmmi")
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    args = parser.parse_args()
    written = prepare_physionet_npz(args.subjects, args.task, args.root, args.output, tmin=args.tmin, tmax=args.tmax)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
