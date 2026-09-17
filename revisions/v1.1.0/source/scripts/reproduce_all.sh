#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiments/all_experiments.yaml}"
OUT="${2:-runs/reproduce_all}"
mkdir -p "$OUT"

for EXP in E0 E1 E2 E3 E4 E5 E6 E7 E8 E9 E10 E11 E12 E13 E14 E15 E16 E17 E18 E19 E20 E21 E22 E23 E24; do
  python scripts/run_experiment.py --config "$CONFIG" --experiment "$EXP" --output "$OUT/$EXP"
done

python scripts/reproduce_figures.py --results "$OUT" --figures "$OUT/figures"
