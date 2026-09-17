#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/dpc_recovery/repo}"
STORAGE_ROOT="${DPC_SNN_STORAGE_ROOT:-/root/autodl-tmp/DPC-SNN_storage}"
PYTHON="${PYTHON:-/root/autodl-tmp/dpc_recovery/venv/bin/python}"
TRAIN_ROOT="$STORAGE_ROOT/data/processed/bci2a_train"
EVAL_ROOT="$STORAGE_ROOT/data/processed/bci2a_eval"
SUBJECTS=(1 2 3 4 5 6 7 8 9)

cd "$PROJECT_ROOT"
export DPC_SNN_STORAGE_ROOT="$STORAGE_ROOT"
mkdir -p "$TRAIN_ROOT" "$EVAL_ROOT"

prepare_or_validate() {
  local session="$1"
  local root="$2"
  if "$PYTHON" scripts/validate_bci2a_processed.py \
    --root "$root" --session "$session" --subjects "${SUBJECTS[@]}" >/dev/null 2>&1; then
    echo "resume: BCI2a session $session compact data already verified"
    return
  fi
  bash scripts/cloud/prefetch_bci2a_aria2.sh
  "$PYTHON" scripts/prepare_bci2a_moabb.py \
    --output "$root" --subjects "${SUBJECTS[@]}" \
    --sessions "$session" --tmin -1 --tmax 4 --resample 250
  "$PYTHON" scripts/validate_bci2a_processed.py \
    --root "$root" --session "$session" --subjects "${SUBJECTS[@]}"
}

prepare_or_validate T "$TRAIN_ROOT"
prepare_or_validate E "$EVAL_ROOT"
