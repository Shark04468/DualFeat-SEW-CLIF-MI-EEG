#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/DPC-SNN}"
STORAGE_ROOT="${DPC_SNN_STORAGE_ROOT:-/root/autodl-tmp/DPC-SNN_storage}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
DATA="${DATA:-$PROJECT_ROOT/data/processed/bci2a_v8_session_t_4d04d3337fe3}"
OUTPUT="${OUTPUT:-$STORAGE_ROOT/runs/v32_purity}"
MAX_JOBS="${MAX_JOBS:-3}"
AUTO_CONTINUE="${AUTO_CONTINUE:-1}"

E28="$STORAGE_ROOT/runs/v28_controls/E28_full_9x3_8fe5f147e7ce"
E9="$STORAGE_ROOT/runs/v9_accuracy_first/E3_dual_feature_8fe5f147e7ce"

cd "$PROJECT_ROOT"
if [ -f .env.storage ]; then
  # shellcheck disable=SC1091
  source .env.storage
fi
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT/logs"

resolve_cache() {
  local subject="$1" seed="$2" fold="$3" candidate
  candidate="$E28/subject_$(printf '%02d' "$subject")/seed_${seed}/fold_${fold}"
  if [ -f "$candidate/frozen_dual_feature_cache.npz" ]; then
    printf '%s\n' "$candidate"
    return
  fi
  candidate="$E9/subject_$(printf '%02d' "$subject")/seed_${seed}/fold_${fold}"
  if [ ! -f "$candidate/frozen_dual_feature_cache.npz" ]; then
    echo "missing sealed cache for subject=$subject seed=$seed fold=$fold" >&2
    return 1
  fi
  printf '%s\n' "$candidate"
}
export -f resolve_cache
export PROJECT_ROOT STORAGE_ROOT PYTHON DATA OUTPUT E28 E9 PYTHONPATH CUDA_VISIBLE_DEVICES PYTHONUNBUFFERED

run_fold() {
  local subject="$1" seed="$2" fold="$3" cache output log
  cache="$(resolve_cache "$subject" "$seed" "$fold")"
  output="$OUTPUT/subject_$(printf '%02d' "$subject")/seed_${seed}/fold_${fold}"
  log="$OUTPUT/logs/subject_$(printf '%02d' "$subject")_seed_${seed}_fold_${fold}.log"
  "$PYTHON" scripts/run_v32_purity_fold.py \
    --data "$DATA" --cache-fold "$cache" --output "$output" \
    --subject "$subject" --seed "$seed" --fold "$fold" --device cuda \
    >"$log" 2>&1
}
export -f run_fold

task_stream_canary() {
  local subject fold
  for subject in 1 3 8; do
    for fold in $(seq 0 5); do
      printf '%s 0 %s\n' "$subject" "$fold"
    done
  done
}

task_stream_full() {
  local subject seed fold
  for subject in $(seq 1 9); do
    for seed in $(seq 0 2); do
      for fold in $(seq 0 5); do
        printf '%s %s %s\n' "$subject" "$seed" "$fold"
      done
    done
  done
}

echo '{"status":"canary_training"}' >"$OUTPUT/pipeline_status.json"
task_stream_canary | xargs -P "$MAX_JOBS" -n 3 bash -c 'run_fold "$0" "$1" "$2"'
"$PYTHON" scripts/aggregate_v32_purity.py \
  --input "$OUTPUT" --output "$OUTPUT/aggregate_canary" --scope canary \
  >"$OUTPUT/logs/aggregate_canary.log" 2>&1

CANARY_STATUS="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$OUTPUT/aggregate_canary/decision.json")"
if [ "$CANARY_STATUS" != "pass" ]; then
  echo '{"status":"stopped_canary_gate_failed"}' >"$OUTPUT/pipeline_status.json"
  exit 0
fi
if [ "$AUTO_CONTINUE" != "1" ]; then
  echo '{"status":"canary_passed_waiting_for_full"}' >"$OUTPUT/pipeline_status.json"
  exit 0
fi

echo '{"status":"full_training"}' >"$OUTPUT/pipeline_status.json"
task_stream_full | xargs -P "$MAX_JOBS" -n 3 bash -c 'run_fold "$0" "$1" "$2"'
"$PYTHON" scripts/aggregate_v32_purity.py \
  --input "$OUTPUT" --output "$OUTPUT/aggregate_full" --scope full \
  >"$OUTPUT/logs/aggregate_full.log" 2>&1
FULL_STATUS="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$OUTPUT/aggregate_full/decision.json")"
if [ "$FULL_STATUS" = "pass" ]; then
  echo '{"status":"full_purity_passed_fusion_authorized"}' >"$OUTPUT/pipeline_status.json"
else
  echo '{"status":"stopped_full_purity_gate_failed"}' >"$OUTPUT/pipeline_status.json"
fi
