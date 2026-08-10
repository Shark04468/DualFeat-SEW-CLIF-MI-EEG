#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/DPC-SNN}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
DATA="${DATA:-/root/autodl-tmp/DPC-SNN_storage/project/data/processed/bci2a_v62}"
V9_ROOT="${V9_ROOT:-/root/autodl-tmp/DPC-SNN_storage/runs/v9_accuracy_first/E3_dual_feature_8fe5f147e7ce}"
OUTPUT="${OUTPUT:?OUTPUT must name a new immutable E16-A run directory}"
CONFIG="${CONFIG:-${ROOT}/configs/experiments/v16_a_continuous_fusion.yaml}"
mkdir -p "${OUTPUT}"

run_subject() {
  local subject="$1"
  for fold in 0 1 2 3 4 5; do
    fold_output="${OUTPUT}/subject_$(printf '%02d' "${subject}")/seed_0/fold_${fold}"
    "${PYTHON}" "${ROOT}/scripts/run_v16_a_continuous_fold.py" \
      --config "${CONFIG}" \
      --data "${DATA}" \
      --v9-root "${V9_ROOT}" \
      --output "${fold_output}" \
      --subject "${subject}" \
      --seed 0 \
      --fold "${fold}" \
      --device cuda
  done
}

pids=()
for subject in 1 3 8; do
  run_subject "${subject}" >"${OUTPUT}/subject_${subject}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

"${PYTHON}" "${ROOT}/scripts/aggregate_v16_a_continuous.py" --root "${OUTPUT}"
