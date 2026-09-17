#!/usr/bin/env bash
set -euo pipefail

STORAGE_ROOT="${DPC_SNN_STORAGE_ROOT:-/root/autodl-tmp/DPC-SNN_storage}"
PYTHON="${PYTHON:-/root/autodl-tmp/dpc_recovery/venv/bin/python}"
ACTIVE_ROOT="$STORAGE_ROOT/cache/mne_data/MNE-bnci-data/~bci/database/004-2014"
EVAL_STAGE_ROOT="${BNCI2014_004_EVAL_STAGE_ROOT:-$STORAGE_ROOT/data/staged/bnci2014_004_eval}"
BARRIER="${V30_CHECKPOINT_BARRIER:?V30_CHECKPOINT_BARRIER is required}"

"$PYTHON" - "$BARRIER" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "sealed" or payload.get("evaluation_sessions_accessed_before_barrier") is not False:
    raise SystemExit("V30 checkpoint barrier is not a valid pre-evaluation seal")
PY

mkdir -p "$ACTIVE_ROOT"
for subject in {1..9}; do
  number="$(printf '%02d' "$subject")"
  source="$EVAL_STAGE_ROOT/B${number}E.mat"
  target="$ACTIVE_ROOT/B${number}E.mat"
  test -f "$source"
  if [ -f "$target" ]; then
    test "$(sha256sum "$source" | awk '{print $1}')" = "$(sha256sum "$target" | awk '{print $1}')"
  else
    cp --reflink=auto -- "$source" "$target"
  fi
  test "$(sha256sum "$source" | awk '{print $1}')" = "$(sha256sum "$target" | awk '{print $1}')"
done
echo "BNCI2014-004 evaluation cache restored only after the sealed checkpoint barrier"
