#!/usr/bin/env bash
set -euo pipefail

SNAPSHOT="${SNAPSHOT:-/root/autodl-tmp/DPC-SNN_storage/code_snapshots/v14_r1_46d2e270e3d7}"
DATA="${DATA:-/root/autodl-tmp/DPC-SNN_storage/project/data/processed/bci2a_v62}"
ANCHOR="${ANCHOR:-/root/autodl-tmp/DPC-SNN_storage/runs/v9_accuracy_first/E3_dual_feature_8fe5f147e7ce}"
OUTPUT="${OUTPUT:-/root/autodl-tmp/DPC-SNN_storage/runs/v14_accuracy_first/R1_shared_residual_46d2e270e3d7}"
SOURCE_DIGEST="46d2e270e3d7f74da8705ff60fe06388267a41202efe181f32e7af59383a0f11"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
VARIANTS="r0_shared_replay,r1_atc_residual,r2_fbc_residual,r3_dual_residual,r4_generic_residual"

mkdir -p "${OUTPUT}"
export PYTHONPATH="${SNAPSHOT}/src"
export PYTHONUNBUFFERED=1

write_status() {
  local status="$1"
  local subject="${2:--1}"
  local fold="${3:--1}"
  printf '{"status":"%s","subject":%s,"seed":0,"fold":%s,"source_tree_sha256":"%s"}\n' \
    "${status}" "${subject}" "${fold}" "${SOURCE_DIGEST}" > "${OUTPUT}/formal_status.json"
}

on_error() {
  local exit_code=$?
  write_status failed "${CURRENT_SUBJECT:--1}" "${CURRENT_FOLD:--1}"
  printf '__V14_CAMPAIGN_FAILED__ exit=%s subject=%s fold=%s\n' \
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
    printf '__V14_FOLD_START__ subject=%s seed=0 fold=%s\n' "${subject}" "${fold}"
    "${PYTHON}" "${SNAPSHOT}/scripts/run_v14_r1_residual_fold.py" \
      --data "${DATA}" \
      --anchor-root "${ANCHOR}" \
      --output "${OUTPUT}/subject_$(printf '%02d' "${subject}")/seed_0/fold_${fold}" \
      --subject "${subject}" \
      --seed 0 \
      --fold "${fold}" \
      --variants "${VARIANTS}" \
      --learning-rates 0.0003,0.001 \
      --weight-decay 0.001 \
      --epochs 100 \
      --patience 20 \
      --minimum-outer-epochs 20 \
      --pretrain-epochs 10 \
      --batch-size 48 \
      --device cuda
    printf '__V14_FOLD_DONE__ subject=%s seed=0 fold=%s\n' "${subject}" "${fold}"
  done
done

"${PYTHON}" "${SNAPSHOT}/scripts/aggregate_v14_r1_residual.py" \
  --root "${OUTPUT}" \
  --output "${OUTPUT}/aggregate_seed0" \
  --subjects 1,3,8 \
  --seeds 0 \
  --folds 0,1,2,3,4,5 \
  --variants "${VARIANTS}" \
  --expected-source-digest "${SOURCE_DIGEST}" \
  --minimum-gain-pp 0.5 \
  --minimum-positive-pairs 2 \
  --maximum-regression-pp 1.0 \
  --minimum-gap-recovery 0.4 \
  --minimum-gain-over-capacity-control-pp 0.3 \
  --maximum-class-regression-pp 2.0

write_status completed
printf '__V14_CAMPAIGN_DONE__\n'
