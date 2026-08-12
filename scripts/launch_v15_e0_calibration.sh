#!/usr/bin/env bash
set -euo pipefail

SNAPSHOT="/root/autodl-tmp/DPC-SNN_storage/code_snapshots/v15_e0_b7fef085ba19"
DATA="/root/autodl-tmp/DPC-SNN_storage/project/data/processed/bci2a_v62"
V9_ROOT="/root/autodl-tmp/DPC-SNN_storage/runs/v9_accuracy_first/E3_dual_feature_8fe5f147e7ce"
V14_ROOT="/root/autodl-tmp/DPC-SNN_storage/runs/v14_accuracy_first/R1_shared_residual_46d2e270e3d7"
OUTPUT="/root/autodl-tmp/DPC-SNN_storage/runs/v15_accuracy_first/E0_calibration_b7fef085ba19"
SOURCE_DIGEST="b7fef085ba195049c4e6a7ac39651008e3322b57dc239eed04513ab7ecb88701"
PYTHON="/root/miniconda3/bin/python"

mkdir -p "${OUTPUT}"
export PYTHONPATH="${SNAPSHOT}/src"
export PYTHONUNBUFFERED=1

write_status() {
  printf '{"status":"%s","subject":%s,"seed":0,"fold":%s,"source_tree_sha256":"%s"}\n' \
    "$1" "${2:--1}" "${3:--1}" "${SOURCE_DIGEST}" > "${OUTPUT}/formal_status.json"
}

on_error() {
  local exit_code=$?
  write_status failed "${CURRENT_SUBJECT:--1}" "${CURRENT_FOLD:--1}"
  printf '__E15_CAMPAIGN_FAILED__ exit=%s subject=%s fold=%s\n' \
    "${exit_code}" "${CURRENT_SUBJECT:--1}" "${CURRENT_FOLD:--1}"
  exit "${exit_code}"
}
trap on_error ERR

write_status running
for subject in 1 3 8; do
  for fold in 0 1 2 3 4 5; do
    CURRENT_SUBJECT="${subject}"
    CURRENT_FOLD="${fold}"
    write_status running "${subject}" "${fold}"
    printf '__E15_FOLD_START__ subject=%s seed=0 fold=%s\n' "${subject}" "${fold}"
    "${PYTHON}" "${SNAPSHOT}/scripts/run_v15_e0_calibration_fold.py" \
      --data "${DATA}" \
      --v9-anchor-root "${V9_ROOT}" \
      --v14-root "${V14_ROOT}" \
      --output "${OUTPUT}/subject_$(printf '%02d' "${subject}")/seed_0/fold_${fold}" \
      --subject "${subject}" \
      --seed 0 \
      --fold "${fold}" \
      --alphas 0,0.05,0.1,0.2,0.4,0.6,0.8,1.0,1.5,2.0 \
      --batch-size 48 \
      --device cuda
    printf '__E15_FOLD_DONE__ subject=%s seed=0 fold=%s\n' "${subject}" "${fold}"
  done
done

"${PYTHON}" "${SNAPSHOT}/scripts/aggregate_v15_e0_calibration.py" \
  --root "${OUTPUT}" \
  --output "${OUTPUT}/aggregate_seed0" \
  --subjects 1,3,8 \
  --seeds 0 \
  --folds 0,1,2,3,4,5 \
  --expected-source-digest "${SOURCE_DIGEST}" \
  --minimum-gain-pp 0.5 \
  --minimum-positive-pairs 2 \
  --maximum-regression-pp 1.0 \
  --minimum-capacity-adjusted-gain-pp 0.3

write_status completed
printf '__E15_CAMPAIGN_DONE__\n'
