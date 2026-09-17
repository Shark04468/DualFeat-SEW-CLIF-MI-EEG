#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/DPC-SNN}"
STORAGE_ROOT="${DPC_SNN_STORAGE_ROOT:-/root/autodl-tmp/DPC-SNN_storage}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
PUBLICATION_ROOT="${PUBLICATION_ROOT:-$STORAGE_ROOT/runs/v8_accuracy_first/publication_baselines_d4061c11d7c4}"
V30_ROOT="${V30_ROOT:-$STORAGE_ROOT/runs/v30_bnci2014_004/E30_blind_f4e74252b318}"
SOURCE_ROOT="${SOURCE_ROOT:-$STORAGE_ROOT/baselines/source}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/experiments/v31_decoder_learning_curve.yaml}"
RECOVERY_LABEL_ROOT="${RECOVERY_LABEL_ROOT:-$STORAGE_ROOT/recovery_labels}"
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
RUN_ID="E31_decoder_curve_${SOURCE_SHA:0:12}"
OUTPUT="${OUTPUT:-$STORAGE_ROOT/runs/v31_learning_curve/$RUN_ID}"
LOG_ROOT="$OUTPUT/logs"
BARRIER="$OUTPUT/checkpoint_barrier.json"
mkdir -p "$LOG_ROOT/train" "$LOG_ROOT/evaluate"

exec 9>"$OUTPUT/pipeline.lock"
if ! flock -n 9; then
  echo "V31 pipeline is already running for $OUTPUT" >&2
  exit 2
fi

run_subject() {
  local phase="$1"
  local dataset="$2"
  local subject="$3"
  local log="$LOG_ROOT/$phase/${dataset}_subject_$(printf '%02d' "$subject").log"
  local extra=()
  if [ "$phase" = "evaluate" ]; then
    extra=(--checkpoint-barrier "$BARRIER")
  fi
  "$PYTHON" scripts/run_v31_decoder_learning_curve_subject.py \
    --phase "$phase" \
    --config "$CONFIG" \
    --dataset "$dataset" \
    --subject "$subject" \
    --publication-root "$PUBLICATION_ROOT" \
    --v30-root "$V30_ROOT" \
    --source-root "$SOURCE_ROOT" \
    --output "$OUTPUT" \
    --bci2a-train-root data/processed/bci2a_v8_session_t_4d04d3337fe3 \
    --bci2a-eval-root data/processed/bci2a_v8_session_e_4d04d3337fe3 \
    --device cuda \
    --feature-batch-size 96 \
    --recovery-label-root "$RECOVERY_LABEL_ROOT" \
    "${extra[@]}" \
    >"$log" 2>&1
}
export -f run_subject
export PROJECT_ROOT STORAGE_ROOT PYTHON PUBLICATION_ROOT V30_ROOT SOURCE_ROOT CONFIG RECOVERY_LABEL_ROOT OUTPUT LOG_ROOT BARRIER PYTHONPATH CUDA_VISIBLE_DEVICES PYTHONUNBUFFERED

task_stream() {
  local dataset subject
  for dataset in bci2a bnci2014_004; do
    for subject in $(seq 1 9); do
      printf '%s %s\n' "$dataset" "$subject"
    done
  done
  for subject in $(seq 1 54); do
    printf '%s %s\n' openbmi "$subject"
  done
}

printf '{"status":"training","run_id":"%s","source_tree_sha256":"%s"}\n' "$RUN_ID" "$SOURCE_SHA" >"$OUTPUT/pipeline_status.json"
task_stream | xargs -P "$MAX_TRAIN_JOBS" -n 2 bash -c 'run_subject train "$0" "$1" || exit 255'

"$PYTHON" scripts/seal_v31_learning_curve.py --config "$CONFIG" --output "$OUTPUT" --barrier "$BARRIER" \
  >"$LOG_ROOT/seal.log" 2>&1
printf '{"status":"evaluating","run_id":"%s","source_tree_sha256":"%s"}\n' "$RUN_ID" "$SOURCE_SHA" >"$OUTPUT/pipeline_status.json"
task_stream | xargs -P "$MAX_EVAL_JOBS" -n 2 bash -c 'run_subject evaluate "$0" "$1" || exit 255'

"$PYTHON" scripts/aggregate_v31_learning_curve.py \
  --config "$CONFIG" \
  --input "$OUTPUT" \
  --output "$OUTPUT/aggregate" \
  >"$LOG_ROOT/aggregate.log" 2>&1
printf '{"status":"completed","run_id":"%s","source_tree_sha256":"%s"}\n' "$RUN_ID" "$SOURCE_SHA" >"$OUTPUT/pipeline_status.json"
echo "$OUTPUT"
