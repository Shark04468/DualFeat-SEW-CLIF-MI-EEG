#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/DPC-SNN}"
STORAGE_ROOT="${DPC_SNN_STORAGE_ROOT:-/root/autodl-tmp/DPC-SNN_storage}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
DATA="${DATA:-$PROJECT_ROOT/data/processed/bci2a_v8_session_t_4d04d3337fe3}"
PURITY_ROOT="${PURITY_ROOT:-$STORAGE_ROOT/runs/v32_purity}"
OUTPUT="${OUTPUT:-$STORAGE_ROOT/runs/v32_fusion}"
MAX_JOBS="${MAX_JOBS:-3}"

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

if [ ! -f "$PURITY_ROOT/aggregate_full/decision.json" ] || \
   [ "$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$PURITY_ROOT/aggregate_full/decision.json")" != "pass" ]; then
  echo "V32 fusion is not authorized because the full purity gate has not passed" >&2
  exit 3
fi

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
export PROJECT_ROOT STORAGE_ROOT PYTHON DATA PURITY_ROOT OUTPUT E28 E9
export PYTHONPATH CUDA_VISIBLE_DEVICES PYTHONUNBUFFERED

run_fold() {
  local mode="$1" subject="$2" seed="$3" fold="$4" cache output log
  cache="$(resolve_cache "$subject" "$seed" "$fold")"
  output="$OUTPUT/$mode/subject_$(printf '%02d' "$subject")/seed_${seed}/fold_${fold}"
  log="$OUTPUT/logs/${mode}_subject_$(printf '%02d' "$subject")_seed_${seed}_fold_${fold}.log"
  "$PYTHON" scripts/run_v32_purity_fold.py \
    --data "$DATA" --cache-fold "$cache" --output "$output" \
    --subject "$subject" --seed "$seed" --fold "$fold" \
    --fusion-mode "$mode" --device cuda >"$log" 2>&1
}
export -f run_fold

task_stream() {
  local mode subject seed fold
  for mode in atc_only fbc_only simple; do
    for subject in $(seq 1 9); do
      for seed in $(seq 0 2); do
        for fold in $(seq 0 5); do
          printf '%s %s %s %s\n' "$mode" "$subject" "$seed" "$fold"
        done
      done
    done
  done
}

echo '{"status":"training"}' >"$OUTPUT/pipeline_status.json"
task_stream | xargs -P "$MAX_JOBS" -n 4 bash -c 'run_fold "$0" "$1" "$2" "$3"'
"$PYTHON" scripts/aggregate_v32_fusion.py \
  --fusion-root "$OUTPUT" --purity-root "$PURITY_ROOT" \
  --output "$OUTPUT/aggregate" >"$OUTPUT/logs/aggregate.log" 2>&1
echo '{"status":"completed"}' >"$OUTPUT/pipeline_status.json"
