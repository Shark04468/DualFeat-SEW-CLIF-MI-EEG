#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/DPC-SNN}"
STORAGE_ROOT="${DPC_SNN_STORAGE_ROOT:-/root/autodl-tmp/DPC-SNN_storage}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-$STORAGE_ROOT/baselines/source}"
PUBLICATION_ROOT="${PUBLICATION_ROOT:-$STORAGE_ROOT/runs/v8_accuracy_first/publication_baselines_d4061c11d7c4}"
V30_PARENT="${V30_PARENT:-$STORAGE_ROOT/runs/v30_bnci2014_004/E30_blind_f4e74252b318}"
E29_PARENT="${E29_PARENT:-$STORAGE_ROOT/runs/v29_openbmi/E29_s1_to_s2_4400c0dfbed7}"
E31_PARENT="${E31_PARENT:-$STORAGE_ROOT/runs/v31_learning_curve/E31_decoder_curve_b4458e9433dc}"
E28_ROOT="${E28_ROOT:-$STORAGE_ROOT/runs/v28_controls/E28_full_9x3_8fe5f147e7ce}"
E28_FALLBACK="${E28_FALLBACK:-$STORAGE_ROOT/runs/v9_accuracy_first/E3_dual_feature_8fe5f147e7ce}"
MAX_TRAIN_JOBS="${MAX_TRAIN_JOBS:-3}"
MAX_EVAL_JOBS="${MAX_EVAL_JOBS:-4}"

cd "$PROJECT_ROOT"
if [ -f .env.storage ]; then
  # shellcheck disable=SC1091
  source .env.storage
fi
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

SOURCE_SHA="$($PYTHON -c 'from pathlib import Path; from dpc_snn.experiments.v8_protocol import collect_source_tree_manifest, source_tree_digest; print(source_tree_digest(collect_source_tree_manifest(Path.cwd())))')"
RUN_ROOT="${RUN_ROOT:-$STORAGE_ROOT/runs/v33_objective_pure_supplement/${SOURCE_SHA:0:12}}"
E31_OUT="$RUN_ROOT/E31Z"
E29_OUT="$RUN_ROOT/E29Z"
E30_OUT="$RUN_ROOT/E30Z"
SMOKE_ROOT="$RUN_ROOT/smoke"
LOG_ROOT="$RUN_ROOT/logs"
mkdir -p "$LOG_ROOT" "$SMOKE_ROOT"

exec 9>"$RUN_ROOT/pipeline.lock"
if ! flock -n 9; then
  echo "V33 objective-pure supplement is already running" >&2
  exit 2
fi

status() {
  printf '{"status":"%s","source_tree_sha256":"%s"}\n' "$1" "$SOURCE_SHA" \
    >"$RUN_ROOT/pipeline_status.json"
}

e31_subject() {
  local phase="$1" dataset="$2" subject="$3" output="$4" config="$5" barrier="${6:-}"
  local extra=()
  if [ "$phase" = evaluate ]; then extra=(--checkpoint-barrier "$barrier"); fi
  "$PYTHON" scripts/run_v33_e31zp_subject.py \
    --phase "$phase" --dataset "$dataset" --subject "$subject" \
    --parent "$E31_PARENT" --output "$output" --config "$config" \
    --publication-root "$PUBLICATION_ROOT" --v30-root "$V30_PARENT" \
    --source-root "$SOURCE_ROOT" --device cuda --feature-batch-size 96 \
    "${extra[@]}"
}

e29_subject() {
  local phase="$1" subject="$2" output="$3" config="$4" barrier="${5:-}"
  local extra=()
  if [ "$phase" = evaluate ]; then extra=(--checkpoint-barrier "$barrier"); fi
  "$PYTHON" scripts/run_v33_e29zp_subject.py \
    --phase "$phase" --subject "$subject" --parent "$E29_PARENT" \
    --output "$output" --config "$config" --publication-root "$PUBLICATION_ROOT/openbmi" \
    --source-root "$SOURCE_ROOT" --device cuda --feature-batch-size 96 \
    "${extra[@]}"
}

e30_subject() {
  local phase="$1" subject="$2" output="$3" config="$4" barrier="${5:-}"
  local extra=()
  if [ "$phase" = evaluate ]; then extra=(--checkpoint-barrier "$barrier"); fi
  "$PYTHON" scripts/run_v33_e30zp_subject.py \
    --phase "$phase" --subject "$subject" --parent "$V30_PARENT" \
    --output "$output" --config "$config" --source-root "$SOURCE_ROOT" \
    --device cuda --feature-batch-size 96 "${extra[@]}"
}

export -f e31_subject e29_subject e30_subject
export PROJECT_ROOT STORAGE_ROOT PYTHON SOURCE_ROOT PUBLICATION_ROOT V30_PARENT
export E29_PARENT E31_PARENT PYTHONPATH CUDA_VISIBLE_DEVICES PYTHONUNBUFFERED

status smoke
E31_SMOKE="$SMOKE_ROOT/E31Z"
for dataset in bci2a openbmi bnci2014_004; do
  e31_subject train "$dataset" 1 "$E31_SMOKE" \
    configs/experiments/v33_e31_zero_penalty_smoke.yaml \
    >"$LOG_ROOT/smoke_e31_${dataset}_train.log" 2>&1
done
"$PYTHON" scripts/seal_v33_e31zp.py --output "$E31_SMOKE" \
  --barrier "$E31_SMOKE/checkpoint_barrier.json" \
  --config configs/experiments/v33_e31_zero_penalty_smoke.yaml
for dataset in bci2a openbmi bnci2014_004; do
  e31_subject evaluate "$dataset" 1 "$E31_SMOKE" \
    configs/experiments/v33_e31_zero_penalty_smoke.yaml \
    "$E31_SMOKE/checkpoint_barrier.json" \
    >"$LOG_ROOT/smoke_e31_${dataset}_evaluate.log" 2>&1
done

E29_SMOKE="$SMOKE_ROOT/E29Z"
e29_subject train 1 "$E29_SMOKE" configs/experiments/v33_e29_zero_penalty_smoke.yaml \
  >"$LOG_ROOT/smoke_e29_train.log" 2>&1
"$PYTHON" scripts/seal_v33_external_zp.py --dataset openbmi --output "$E29_SMOKE" \
  --barrier "$E29_SMOKE/checkpoint_barrier.json" \
  --config configs/experiments/v33_e29_zero_penalty_smoke.yaml
e29_subject evaluate 1 "$E29_SMOKE" configs/experiments/v33_e29_zero_penalty_smoke.yaml \
  "$E29_SMOKE/checkpoint_barrier.json" >"$LOG_ROOT/smoke_e29_evaluate.log" 2>&1

E30_SMOKE="$SMOKE_ROOT/E30Z"
e30_subject train 1 "$E30_SMOKE" configs/experiments/v33_e30_zero_penalty_smoke.yaml \
  >"$LOG_ROOT/smoke_e30_train.log" 2>&1
"$PYTHON" scripts/seal_v33_external_zp.py --dataset bnci2014_004 --output "$E30_SMOKE" \
  --barrier "$E30_SMOKE/checkpoint_barrier.json" \
  --config configs/experiments/v33_e30_zero_penalty_smoke.yaml
e30_subject evaluate 1 "$E30_SMOKE" configs/experiments/v33_e30_zero_penalty_smoke.yaml \
  "$E30_SMOKE/checkpoint_barrier.json" >"$LOG_ROOT/smoke_e30_evaluate.log" 2>&1

status primary_metrics
"$PYTHON" scripts/aggregate_v33_primary_metrics.py \
  --e28-root "$E28_ROOT" --e28-fallback "$E28_FALLBACK" \
  --e29-root "$E29_PARENT" --e30-root "$V30_PARENT" \
  --output "$RUN_ROOT/primary_metrics" >"$LOG_ROOT/primary_metrics.log" 2>&1

status e31zp_canary
pids=()
for dataset in bci2a openbmi bnci2014_004; do
  e31_subject train "$dataset" 1 "$E31_OUT" \
    configs/experiments/v33_e31_zero_penalty.yaml \
    >"$LOG_ROOT/e31_canary_${dataset}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

status e31zp_training
{
  for dataset in bci2a bnci2014_004; do
    for subject in $(seq 1 9); do printf '%s %s\n' "$dataset" "$subject"; done
  done
  for subject in $(seq 1 54); do printf 'openbmi %s\n' "$subject"; done
} | xargs -P "$MAX_TRAIN_JOBS" -n 2 bash -c \
  'e31_subject train "$0" "$1" "'"$E31_OUT"'" configs/experiments/v33_e31_zero_penalty.yaml >"'"$LOG_ROOT"'/e31_train_${0}_${1}.log" 2>&1'
"$PYTHON" scripts/seal_v33_e31zp.py --output "$E31_OUT" \
  --barrier "$E31_OUT/checkpoint_barrier.json"
status e31zp_evaluating
{
  for dataset in bci2a bnci2014_004; do
    for subject in $(seq 1 9); do printf '%s %s\n' "$dataset" "$subject"; done
  done
  for subject in $(seq 1 54); do printf 'openbmi %s\n' "$subject"; done
} | xargs -P "$MAX_EVAL_JOBS" -n 2 bash -c \
  'e31_subject evaluate "$0" "$1" "'"$E31_OUT"'" configs/experiments/v33_e31_zero_penalty.yaml "'"$E31_OUT"'/checkpoint_barrier.json" >"'"$LOG_ROOT"'/e31_eval_${0}_${1}.log" 2>&1'
"$PYTHON" scripts/aggregate_v33_e31zp.py --input "$E31_OUT" --parent "$E31_PARENT" \
  --output "$E31_OUT/aggregate" >"$LOG_ROOT/e31_aggregate.log" 2>&1

status e29zp_training
seq 1 54 | xargs -P "$MAX_TRAIN_JOBS" -n 1 bash -c \
  'e29_subject train "$0" "'"$E29_OUT"'" configs/experiments/v33_e29_zero_penalty.yaml >"'"$LOG_ROOT"'/e29_train_${0}.log" 2>&1'
"$PYTHON" scripts/seal_v33_external_zp.py --dataset openbmi --output "$E29_OUT" \
  --barrier "$E29_OUT/checkpoint_barrier.json"
status e29zp_evaluating
seq 1 54 | xargs -P "$MAX_EVAL_JOBS" -n 1 bash -c \
  'e29_subject evaluate "$0" "'"$E29_OUT"'" configs/experiments/v33_e29_zero_penalty.yaml "'"$E29_OUT"'/checkpoint_barrier.json" >"'"$LOG_ROOT"'/e29_eval_${0}.log" 2>&1'
"$PYTHON" scripts/aggregate_v33_external_zp.py --dataset openbmi \
  --root "$E29_OUT" --parent "$E29_PARENT" --output "$E29_OUT/aggregate" \
  >"$LOG_ROOT/e29_aggregate.log" 2>&1

status e30zp_training
seq 1 9 | xargs -P "$MAX_TRAIN_JOBS" -n 1 bash -c \
  'e30_subject train "$0" "'"$E30_OUT"'" configs/experiments/v33_e30_zero_penalty.yaml >"'"$LOG_ROOT"'/e30_train_${0}.log" 2>&1'
"$PYTHON" scripts/seal_v33_external_zp.py --dataset bnci2014_004 --output "$E30_OUT" \
  --barrier "$E30_OUT/checkpoint_barrier.json"
status e30zp_evaluating
seq 1 9 | xargs -P "$MAX_EVAL_JOBS" -n 1 bash -c \
  'e30_subject evaluate "$0" "'"$E30_OUT"'" configs/experiments/v33_e30_zero_penalty.yaml "'"$E30_OUT"'/checkpoint_barrier.json" >"'"$LOG_ROOT"'/e30_eval_${0}.log" 2>&1'
"$PYTHON" scripts/aggregate_v33_external_zp.py --dataset bnci2014_004 \
  --root "$E30_OUT" --parent "$V30_PARENT" --output "$E30_OUT/aggregate" \
  >"$LOG_ROOT/e30_aggregate.log" 2>&1

status completed
echo "$RUN_ROOT"
