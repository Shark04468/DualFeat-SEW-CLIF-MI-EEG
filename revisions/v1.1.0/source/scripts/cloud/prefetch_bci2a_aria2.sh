#!/usr/bin/env bash
set -euo pipefail

STORAGE_ROOT="${DPC_SNN_STORAGE_ROOT:-/root/autodl-tmp/DPC-SNN_storage}"
CACHE_ROOT="$STORAGE_ROOT/cache/mne_data/MNE-bnci-data/~bci/database/001-2014"
BASE_URL="https://lampx.tugraz.at/~bci/database/001-2014"

if ! command -v aria2c >/dev/null 2>&1; then
  echo "aria2c is required; install the aria2 package before running this script" >&2
  exit 2
fi
mkdir -p "$CACHE_ROOT"

input_file="$(mktemp)"
trap 'rm -f "$input_file"' EXIT
for subject in {1..9}; do
  number="$(printf '%02d' "$subject")"
  for role in T E; do
    printf '%s/A%s%s.mat\n  dir=%s\n  out=A%s%s.mat\n' \
      "$BASE_URL" "$number" "$role" "$CACHE_ROOT" "$number" "$role" >>"$input_file"
  done
done

aria2c \
  --input-file="$input_file" \
  --continue=true \
  --allow-overwrite=false \
  --auto-file-renaming=false \
  --check-integrity=true \
  --max-concurrent-downloads=8 \
  --split=8 \
  --max-connection-per-server=8 \
  --min-split-size=4M \
  --file-allocation=none \
  --retry-wait=5 \
  --max-tries=0 \
  --timeout=60 \
  --connect-timeout=30 \
  --summary-interval=60

for subject in {1..9}; do
  number="$(printf '%02d' "$subject")"
  for role in T E; do
    path="$CACHE_ROOT/A${number}${role}.mat"
    if [ ! -f "$path" ] || [ "$(stat -c %s "$path")" -le 10485760 ]; then
      echo "BCI2a cache target is absent or truncated: $path" >&2
      exit 1
    fi
  done
done
sha256sum "$CACHE_ROOT"/A??T.mat "$CACHE_ROOT"/A??E.mat \
  >"$CACHE_ROOT/DATASET_SHA256SUMS.txt"
echo "BCI2a raw cache is complete and hash-manifested"
