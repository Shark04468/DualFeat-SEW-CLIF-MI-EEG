#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/dpc_recovery/repo}"
STORAGE_ROOT="${DPC_SNN_STORAGE_ROOT:-/root/autodl-tmp/DPC-SNN_storage}"
PYTHON="${PYTHON:-/root/autodl-tmp/dpc_recovery/venv/bin/python}"
SUBJECTS="${OPENBMI_SUBJECTS:-1-54}"
OUTPUT_ROOT="${OPENBMI_V8_ROOT:-$STORAGE_ROOT/data/processed/openbmi_v8}"
LOG_ROOT="${OPENBMI_PREP_LOG_ROOT:-/root/autodl-tmp/dpc_recovery/logs/openbmi_compact}"

cd "$PROJECT_ROOT"
export DPC_SNN_STORAGE_ROOT="$STORAGE_ROOT"
export DPC_SNN_OPENBMI_V8_ROOT="$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

parse_subjects() {
  local token first last subject
  IFS=',' read -ra tokens <<<"$SUBJECTS"
  for token in "${tokens[@]}"; do
    if [[ "$token" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      first="${BASH_REMATCH[1]}"
      last="${BASH_REMATCH[2]}"
      for ((subject=first; subject<=last; subject++)); do
        printf '%d\n' "$subject"
      done
    elif [[ "$token" =~ ^[0-9]+$ ]]; then
      printf '%d\n' "$token"
    else
      echo "Invalid OPENBMI_SUBJECTS token: $token" >&2
      return 2
    fi
  done
}

while read -r subject; do
  ready="$OUTPUT_ROOT/subject_$(printf '%02d' "$subject")/READY.json"
  if [ -f "$ready" ]; then
    "$PYTHON" scripts/prepare_openbmi_v8_cache.py \
      --storage-root "$STORAGE_ROOT" --output "$OUTPUT_ROOT" \
      --subjects "$subject" --validate-only \
      >"$LOG_ROOT/subject_$(printf '%02d' "$subject").log" 2>&1
    echo "resume: OpenBMI subject $subject compact cache already verified"
    continue
  fi
  echo "start: OpenBMI subject $subject raw prefetch"
  OPENBMI_SUBJECTS="$subject" DPC_SNN_STORAGE_ROOT="$STORAGE_ROOT" \
    bash scripts/cloud/prefetch_openbmi_aria2.sh \
    >"$LOG_ROOT/subject_$(printf '%02d' "$subject")_download.log" 2>&1
  echo "start: OpenBMI subject $subject compact conversion"
  "$PYTHON" scripts/prepare_openbmi_v8_cache.py \
    --storage-root "$STORAGE_ROOT" --output "$OUTPUT_ROOT" \
    --subjects "$subject" --cleanup-raw \
    >"$LOG_ROOT/subject_$(printf '%02d' "$subject").log" 2>&1
  echo "complete: OpenBMI subject $subject compact cache"
done < <(parse_subjects)

"$PYTHON" scripts/prepare_openbmi_v8_cache.py \
  --storage-root "$STORAGE_ROOT" --output "$OUTPUT_ROOT" \
  --subjects "$SUBJECTS" --validate-only
