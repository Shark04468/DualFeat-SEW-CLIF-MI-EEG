#!/usr/bin/env python
"""Run the V3.9 sub-sample delay recovery gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.analysis.fractional_delay import (  # noqa: E402
    recover_fractional_delays,
    summarize_fractional_recovery,
)
from dpc_snn.utils.io import write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/v39_fractional_delay/metrics.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=39)
    args = parser.parse_args()
    summary = summarize_fractional_recovery(
        recover_fractional_delays(device=args.device, seed=args.seed)
    )
    write_json(args.output, summary)
    print(json.dumps(summary, indent=2))
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
