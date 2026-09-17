#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/dpc_recovery/repo}"
STORAGE_ROOT="${DPC_SNN_STORAGE_ROOT:-/root/autodl-tmp/DPC-SNN_storage}"
PYTHON="${PYTHON:-/root/autodl-tmp/dpc_recovery/venv/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
EXPECTED_GPU_SUBSTRING="${EXPECTED_GPU_SUBSTRING:-5090}"
RUN_ROOT="${RUN_ROOT:-$STORAGE_ROOT/runs/revision_20260901}"
SOURCE_ROOT="${SOURCE_ROOT:-$STORAGE_ROOT/baselines/source}"
DATA_ROOT="$STORAGE_ROOT/data/processed"
LABEL_ROOT="$STORAGE_ROOT/recovery_labels"
CONTROL_ROOT="$RUN_ROOT/control"
LOG_ROOT="$RUN_ROOT/logs"
PUBLICATION_ROOT="$RUN_ROOT/publication"
V30_ROOT="$RUN_ROOT/v30_recovered"
V31_ROOT="$RUN_ROOT/v31_recovered"
REVIEWER_ROOT="$RUN_ROOT/reviewer_controls"
BINARY_ROOT="$RUN_ROOT/bci2a_binary"
BNCI_EVAL_STAGE_ROOT="$STORAGE_ROOT/data/staged/bnci2014_004_eval"
BNCI_EVAL_SEAL_ROOT="${BNCI2014_004_EVAL_SEAL_ROOT:-$(dirname "$STORAGE_ROOT")/dpc_recovery/sealed_evaluation/bnci2014_004_eval}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export DPC_SNN_STORAGE_ROOT="$STORAGE_ROOT"
export PYTHONUNBUFFERED=1
mkdir -p "$CONTROL_ROOT" "$LOG_ROOT" "$DATA_ROOT" "$LABEL_ROOT"

gpu_count="$($PYTHON -c 'import torch; print(torch.cuda.device_count())')"
if [ "$gpu_count" -ne 4 ]; then
  echo "Expected exactly four visible GPUs, found $gpu_count" >&2
  exit 2
fi
export EXPECTED_GPU_SUBSTRING
$PYTHON - <<'PY'
import os

import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
expected = os.environ["EXPECTED_GPU_SUBSTRING"]
if any(expected not in name for name in names):
    raise SystemExit(f"Four-card campaign requires GPUs matching {expected!r}; found {names}")
print(names)
PY

stage() {
  local name="$1"
  shift
  local barrier="$CONTROL_ROOT/${name}.done"
  local log="$LOG_ROOT/${name}.log"
  if [ -f "$barrier" ]; then
    echo "resume: $name already complete"
    return
  fi
  echo "start: $name"
  "$@" >"$log" 2>&1
  touch "$barrier"
  echo "complete: $name"
}

seal_bnci2014_004_evaluation() {
  local storage_resolved seal_resolved
  storage_resolved="$(realpath -m "$STORAGE_ROOT")"
  seal_resolved="$(realpath -m "$BNCI_EVAL_SEAL_ROOT")"
  case "$seal_resolved/" in
    "$storage_resolved/"*)
      echo "BNCI evaluation seal must be outside STORAGE_ROOT: $BNCI_EVAL_SEAL_ROOT" >&2
      return 2
      ;;
  esac
  if [ -e "$BNCI_EVAL_STAGE_ROOT" ] && [ -e "$BNCI_EVAL_SEAL_ROOT" ]; then
    echo "Both staged and sealed BNCI evaluation roots exist; refusing an ambiguous move" >&2
    return 2
  fi
  if [ -e "$BNCI_EVAL_STAGE_ROOT" ]; then
    mkdir -p "$(dirname "$BNCI_EVAL_SEAL_ROOT")"
    mv -- "$BNCI_EVAL_STAGE_ROOT" "$BNCI_EVAL_SEAL_ROOT"
  fi
  for subject in {1..9}; do
    number="$(printf '%02d' "$subject")"
    path="$BNCI_EVAL_SEAL_ROOT/B${number}E.mat"
    if [ ! -f "$path" ] || [ "$(stat -c %s "$path")" -le 10485760 ]; then
      echo "Sealed BNCI evaluation file is absent or truncated: $path" >&2
      return 1
    fi
  done
  if find "$STORAGE_ROOT" -type f -name 'B??E.mat' -print -quit | grep -q .; then
    echo "BNCI evaluation data remain visible under STORAGE_ROOT before recovery freeze" >&2
    return 1
  fi
  echo "BNCI2014-004 evaluation cache sealed outside the recovery storage root"
}

restore_bnci2014_004_evaluation() {
  "$PYTHON" - "$V30_ROOT/checkpoint_barrier.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "sealed" or payload.get("evaluation_sessions_accessed_before_barrier") is not False:
    raise SystemExit("V30 checkpoint barrier is not a valid pre-evaluation seal")
PY
  if [ -e "$BNCI_EVAL_SEAL_ROOT" ] && [ -e "$BNCI_EVAL_STAGE_ROOT" ]; then
    echo "Both sealed and staged BNCI evaluation roots exist; refusing an ambiguous restore" >&2
    return 2
  fi
  if [ -e "$BNCI_EVAL_SEAL_ROOT" ]; then
    mkdir -p "$(dirname "$BNCI_EVAL_STAGE_ROOT")"
    mv -- "$BNCI_EVAL_SEAL_ROOT" "$BNCI_EVAL_STAGE_ROOT"
  fi
  env DPC_SNN_STORAGE_ROOT="$STORAGE_ROOT" \
    V30_CHECKPOINT_BARRIER="$V30_ROOT/checkpoint_barrier.json" \
    bash scripts/cloud/restore_bnci2014_004_eval_cache.sh
}

stage bci2a_compact_data bash scripts/cloud/prepare_bci2a_clone_cache.sh

if ! command -v aria2c >/dev/null 2>&1; then
  apt-get update >"$LOG_ROOT/aria2_install.log" 2>&1
  apt-get install -y aria2 >>"$LOG_ROOT/aria2_install.log" 2>&1
fi
stage openbmi_compact_data env OPENBMI_SUBJECTS=1-54 \
  DPC_SNN_STORAGE_ROOT="$STORAGE_ROOT" bash scripts/cloud/prepare_openbmi_clone_cache.sh

stage bnci2014_004_staged_data env DPC_SNN_STORAGE_ROOT="$STORAGE_ROOT" \
  bash scripts/cloud/prepare_bnci2014_004_clone_cache.sh
stage bnci2014_004_evaluation_seal seal_bnci2014_004_evaluation

stage v30_recovery_freeze "$PYTHON" scripts/freeze_v30_recovery.py \
  --config configs/experiments/v30_bnci2014_004_recovery.yaml \
  --storage-root "$STORAGE_ROOT" --output "$V30_ROOT/freeze"

stage publication_bci2a "$PYTHON" scripts/launch_publication_baselines.py \
  --dataset bci2a --phase all --output-root "$PUBLICATION_ROOT" \
  --source-root "$SOURCE_ROOT" --storage-root "$STORAGE_ROOT" \
  --bci2a-train-root "$DATA_ROOT/bci2a_train" --bci2a-eval-root "$DATA_ROOT/bci2a_eval" \
  --gpus "$GPU_IDS" --models atcnet,fbcnet --seeds 0,1,2
stage publication_openbmi "$PYTHON" scripts/launch_publication_baselines.py \
  --dataset openbmi --phase all --output-root "$PUBLICATION_ROOT" \
  --source-root "$SOURCE_ROOT" --storage-root "$STORAGE_ROOT" \
  --bci2a-train-root "$DATA_ROOT/bci2a_train" --bci2a-eval-root "$DATA_ROOT/bci2a_eval" \
  --gpus "$GPU_IDS" --models atcnet,fbcnet --seeds 0,1,2

stage v30_recovered_train "$PYTHON" scripts/launch_v30_bnci2014_004.py \
  --phase train --config configs/experiments/v30_bnci2014_004_recovery.yaml \
  --source-root "$SOURCE_ROOT" --freeze "$V30_ROOT/freeze/freeze_manifest.json" \
  --output "$V30_ROOT" --gpus "$GPU_IDS"
stage v30_evaluation_cache_restore restore_bnci2014_004_evaluation
stage v30_recovered_evaluate "$PYTHON" scripts/launch_v30_bnci2014_004.py \
  --phase evaluate --config configs/experiments/v30_bnci2014_004_recovery.yaml \
  --source-root "$SOURCE_ROOT" --freeze "$V30_ROOT/freeze/freeze_manifest.json" \
  --output "$V30_ROOT" --gpus "$GPU_IDS"

stage v31_recovered "$PYTHON" scripts/launch_v31_learning_curve.py \
  --phase all --config configs/experiments/v31_decoder_learning_curve.yaml \
  --publication-root "$PUBLICATION_ROOT" --v30-root "$V30_ROOT" \
  --source-root "$SOURCE_ROOT" --output "$V31_ROOT" \
  --bci2a-train-root "$DATA_ROOT/bci2a_train" --bci2a-eval-root "$DATA_ROOT/bci2a_eval" \
  --recovery-label-root "$LABEL_ROOT" --gpus "$GPU_IDS"

stage resolve_reviewer_configs "$PYTHON" scripts/resolve_reviewer_runtime_configs.py \
  --e31 "$V31_ROOT" --publication "$PUBLICATION_ROOT" --v30 "$V30_ROOT" \
  --baseline-source "$SOURCE_ROOT" --training-labels "$LABEL_ROOT" --output "$CONTROL_ROOT"
REVIEWER_CONFIG="$CONTROL_ROOT/reviewer_controls.resolved.yaml"
BINARY_CONFIG="$CONTROL_ROOT/bci2a_binary_sensitivity.resolved.yaml"

stage reviewer_train "$PYTHON" scripts/launch_reviewer_controls.py \
  --phase train --config "$REVIEWER_CONFIG" --output "$REVIEWER_ROOT" --gpus "$GPU_IDS"
stage reviewer_seal "$PYTHON" scripts/seal_reviewer_controls.py \
  --config "$REVIEWER_CONFIG" --output "$REVIEWER_ROOT" \
  --barrier "$REVIEWER_ROOT/CHECKPOINT_BARRIER.json"
stage reviewer_evaluate "$PYTHON" scripts/launch_reviewer_controls.py \
  --phase evaluate --config "$REVIEWER_CONFIG" --output "$REVIEWER_ROOT" \
  --gpus "$GPU_IDS" --checkpoint-barrier "$REVIEWER_ROOT/CHECKPOINT_BARRIER.json"
stage reviewer_aggregate "$PYTHON" scripts/aggregate_reviewer_controls.py \
  --config "$REVIEWER_CONFIG" --input "$REVIEWER_ROOT" --output "$REVIEWER_ROOT/aggregate"

stage binary_train "$PYTHON" scripts/launch_bci2a_binary_sensitivity.py \
  --phase train --config "$BINARY_CONFIG" --output "$BINARY_ROOT" --gpus "$GPU_IDS"
stage binary_seal "$PYTHON" scripts/seal_bci2a_binary_sensitivity.py \
  --config "$BINARY_CONFIG" --output "$BINARY_ROOT" \
  --barrier "$BINARY_ROOT/CHECKPOINT_BARRIER.json"
stage binary_evaluate "$PYTHON" scripts/launch_bci2a_binary_sensitivity.py \
  --phase evaluate --config "$BINARY_CONFIG" --output "$BINARY_ROOT" --gpus "$GPU_IDS" \
  --checkpoint-barrier "$BINARY_ROOT/CHECKPOINT_BARRIER.json"
stage binary_aggregate "$PYTHON" scripts/aggregate_bci2a_binary_sensitivity.py \
  --config "$BINARY_CONFIG" --input "$BINARY_ROOT" --output "$BINARY_ROOT/aggregate"

stage utility_profile "$PYTHON" scripts/profile_reviewer_utility.py \
  --config "$REVIEWER_CONFIG" --reviewer-root "$REVIEWER_ROOT" \
  --output "$REVIEWER_ROOT/utility" --datasets bci2a,openbmi \
  --budget n25 --seed 0 --warmup 100 --repetitions 1000 --device cuda

stage final_validation "$PYTHON" scripts/validate_revision_campaign.py \
  --root "$RUN_ROOT" --reviewer-config "$REVIEWER_CONFIG" \
  --binary-config "$BINARY_CONFIG" --output "$RUN_ROOT/READY_FOR_MANUSCRIPT_REVISION.json"

echo "$RUN_ROOT/READY_FOR_MANUSCRIPT_REVISION.json"
