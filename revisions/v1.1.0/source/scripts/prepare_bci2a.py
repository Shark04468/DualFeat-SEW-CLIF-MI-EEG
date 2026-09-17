#!/usr/bin/env python
"""Prepare BCI IV-2a GDF files into processed NPZ files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.data.bci2a import prepare_from_mne_gdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", nargs="+", required=True, help="GDF files, e.g. A01T.gdf A01E.gdf")
    parser.add_argument("--output", default="data/processed/bci2a")
    parser.add_argument("--tmin", type=float, default=-1.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    args = parser.parse_args()
    written = prepare_from_mne_gdf([Path(p) for p in args.raw], args.output, tmin=args.tmin, tmax=args.tmax)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
