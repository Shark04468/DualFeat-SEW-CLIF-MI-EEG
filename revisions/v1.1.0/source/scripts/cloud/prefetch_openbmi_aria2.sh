#!/usr/bin/env bash
set -euo pipefail

STORAGE_ROOT="${DPC_SNN_STORAGE_ROOT:-/root/autodl-tmp/DPC-SNN_storage}"
SUBJECTS="${OPENBMI_SUBJECTS:-1-54}"
PARALLEL_FILES="${OPENBMI_PARALLEL_FILES:-4}"
CONNECTIONS_PER_FILE="${OPENBMI_CONNECTIONS_PER_FILE:-8}"
CACHE_ROOT="$STORAGE_ROOT/cache/mne_data/MNE-lee2019-mi-data"
BASE_URL="https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542"

if ! command -v aria2c >/dev/null 2>&1; then
  echo "aria2c is required; install the aria2 package before running this script" >&2
  exit 2
fi

mkdir -p "$CACHE_ROOT"
input_file="$(mktemp)"
trap 'rm -f "$input_file"' EXIT

parse_subjects() {
  local token start end value
  IFS=',' read -ra tokens <<<"$SUBJECTS"
  for token in "${tokens[@]}"; do
    if [[ "$token" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      start="${BASH_REMATCH[1]}"
      end="${BASH_REMATCH[2]}"
      for ((value=start; value<=end; value++)); do
        printf '%d\n' "$value"
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
  if ((subject < 1 || subject > 54)); then
    echo "OpenBMI subject must be in 1..54: $subject" >&2
    exit 2
  fi
  for session in 1 2; do
    if ((session == 1)); then
      prefix="sess01"
    else
      prefix="sess02"
    fi
    target_dir="$CACHE_ROOT/gigadb-datasets/live/pub/10.5524/100001_101000/100542/session${session}/s${subject}"
    target_name="${prefix}_subj$(printf '%02d' "$subject")_EEG_MI.mat"
    mkdir -p "$target_dir"
    printf '%s/session%d/s%d/%s\n' "$BASE_URL" "$session" "$subject" "$target_name" >>"$input_file"
    printf '  dir=%s\n' "$target_dir" >>"$input_file"
    printf '  out=%s\n' "$target_name" >>"$input_file"
  done
done < <(parse_subjects)

aria2c \
  --input-file="$input_file" \
  --continue=true \
  --allow-overwrite=false \
  --auto-file-renaming=false \
  --check-integrity=true \
  --max-concurrent-downloads="$PARALLEL_FILES" \
  --split="$CONNECTIONS_PER_FILE" \
  --max-connection-per-server="$CONNECTIONS_PER_FILE" \
  --min-split-size=16M \
  --file-allocation=none \
  --retry-wait=5 \
  --max-tries=0 \
  --timeout=60 \
  --connect-timeout=30 \
  --summary-interval=60

verified=0
while read -r subject; do
  for session in 1 2; do
    target="$CACHE_ROOT/gigadb-datasets/live/pub/10.5524/100001_101000/100542/session${session}/s${subject}/sess0${session}_subj$(printf '%02d' "$subject")_EEG_MI.mat"
    if [ ! -f "$target" ] || [ "$(stat -c %s "$target")" -le 104857600 ]; then
      echo "OpenBMI prefetch target is absent or truncated: $target" >&2
      exit 1
    fi
    verified=$((verified + 1))
  done
done < <(parse_subjects)
echo "OpenBMI prefetch complete: $verified requested files verified under $CACHE_ROOT"
