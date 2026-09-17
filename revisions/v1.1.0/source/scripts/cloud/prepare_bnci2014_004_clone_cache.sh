#!/usr/bin/env bash
set -euo pipefail

STORAGE_ROOT="${DPC_SNN_STORAGE_ROOT:-/root/autodl-tmp/DPC-SNN_storage}"
ACTIVE_ROOT="$STORAGE_ROOT/cache/mne_data/MNE-bnci-data/~bci/database/004-2014"
EVAL_STAGE_ROOT="${BNCI2014_004_EVAL_STAGE_ROOT:-$STORAGE_ROOT/data/staged/bnci2014_004_eval}"
BASE_URL="https://lampx.tugraz.at/~bci/database/004-2014"

if ! command -v aria2c >/dev/null 2>&1; then
  echo "aria2c is required; install the aria2 package before running this script" >&2
  exit 2
fi
mkdir -p "$ACTIVE_ROOT" "$EVAL_STAGE_ROOT"

for subject in {1..9}; do
  active_eval="$ACTIVE_ROOT/B$(printf '%02d' "$subject")E.mat"
  staged_eval="$EVAL_STAGE_ROOT/B$(printf '%02d' "$subject")E.mat"
  if [ -f "$active_eval" ]; then
    if [ -f "$staged_eval" ]; then
      if [ "$(sha256sum "$active_eval" | awk '{print $1}')" != "$(sha256sum "$staged_eval" | awk '{print $1}')" ]; then
        echo "Active and staged BNCI2014-004 evaluation files differ for subject $subject" >&2
        exit 1
      fi
      rm -- "$active_eval"
    else
      mv -- "$active_eval" "$staged_eval"
    fi
  fi
done

input_file="$(mktemp)"
trap 'rm -f "$input_file"' EXIT
for subject in {1..9}; do
  number="$(printf '%02d' "$subject")"
  printf '%s/B%sT.mat\n  dir=%s\n  out=B%sT.mat\n' \
    "$BASE_URL" "$number" "$ACTIVE_ROOT" "$number" >>"$input_file"
  printf '%s/B%sE.mat\n  dir=%s\n  out=B%sE.mat\n' \
    "$BASE_URL" "$number" "$EVAL_STAGE_ROOT" "$number" >>"$input_file"
done

aria2c \
  --input-file="$input_file" \
  --continue=true \
  --allow-overwrite=false \
  --auto-file-renaming=false \
  --check-certificate=false \
  --check-integrity=true \
  --max-concurrent-downloads=4 \
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
  for path in "$ACTIVE_ROOT/B${number}T.mat" "$EVAL_STAGE_ROOT/B${number}E.mat"; do
    if [ ! -f "$path" ] || [ "$(stat -c %s "$path")" -le 10485760 ]; then
      echo "BNCI2014-004 cache target is absent or truncated: $path" >&2
      exit 1
    fi
  done
done
if find "$ACTIVE_ROOT" -maxdepth 1 -type f -name 'B??E.mat' -print -quit | grep -q .; then
  echo "Evaluation cache leaked into the active BNCI directory" >&2
  exit 1
fi
sha256sum "$ACTIVE_ROOT"/B??T.mat "$EVAL_STAGE_ROOT"/B??E.mat \
  >"$EVAL_STAGE_ROOT/DATASET_SHA256SUMS.txt"
echo "BNCI2014-004 training cache and sealed evaluation staging are ready"
