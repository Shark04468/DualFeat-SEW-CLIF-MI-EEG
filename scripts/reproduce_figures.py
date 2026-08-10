#!/usr/bin/env python
"""Reproduce source-data based figures.

The script is deliberately conservative: if matplotlib is missing, it still
writes a manifest of available source data so figure provenance is auditable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.plots.figures import reproduce_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--figures", default="figures")
    args = parser.parse_args()
    manifest = reproduce_figures(Path(args.results), Path(args.figures))
    print(f"figure manifest: {manifest}")


if __name__ == "__main__":
    main()

