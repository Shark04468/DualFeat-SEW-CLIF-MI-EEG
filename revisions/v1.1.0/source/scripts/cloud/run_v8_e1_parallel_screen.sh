#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/bin/python
STORAGE=/root/autodl-tmp/DPC-SNN_storage
SNAPSHOT="$STORAGE/code_snapshots/v8_e1_nested_00fbfabdd545"
OUTPUT="$STORAGE/runs/v8_accuracy_first/E1_baselines_nested_00fbfabdd545"
DATA="$STORAGE/project/data/processed/bci2a_v62"
SOURCES="$STORAGE/baselines/source"
CONFIG=configs/experiments/v8_e1_baselines.yaml
WORKERS="$OUTPUT/parallel_workers"
MODELS=(eegnet fbcnet atcnet tcformer eeg_conformer mi_snn_plif bfatcnet)
SUBJECTS=(1 3 8)

mkdir -p "$WORKERS"
cd "$SNAPSHOT"
export DPC_SNN_STORAGE_ROOT="$STORAGE"
export PYTHONPATH=src
export PYTHONPYCACHEPREFIX="$STORAGE/pycache"

pids=()
labels=()
for model in "${MODELS[@]}"; do
  for subject in "${SUBJECTS[@]}"; do
    label="${model}_s${subject}"
    "$PYTHON" scripts/run_v8_e1_baselines.py \
      --data "$DATA" \
      --source-root "$SOURCES" \
      --output "$OUTPUT" \
      --config "$CONFIG" \
      --models "$model" \
      --subjects "$subject" \
      --confirmation-seeds 0 \
      --confirmation-top-k 1 \
      --device cuda \
      > "$WORKERS/$label.log" \
      2> "$WORKERS/$label.err" &
    pids+=("$!")
    labels+=("$label")
  done
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    printf 'worker failed: run=%s pid=%s\n' "${labels[$index]}" "${pids[$index]}" >&2
    failed=1
  fi
done
test "$failed" -eq 0

"$PYTHON" scripts/run_v8_e1_baselines.py \
  --data "$DATA" \
  --source-root "$SOURCES" \
  --output "$OUTPUT" \
  --config "$CONFIG" \
  --device cuda

echo '__V8_E1_PARALLEL_FINALIZER_DONE__'
