#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/dpc_recovery/repo}"
STORAGE_ROOT="${DPC_SNN_STORAGE_ROOT:-/root/autodl-tmp/DPC-SNN_storage}"
PYTHON="${PYTHON:-/root/autodl-tmp/dpc_recovery/venv/bin/python}"
PAYLOAD_ROOT="${CLONE_PAYLOAD_ROOT:-/root/DPC-SNN_clone_payload}"
OPENBMI_ROOT="$STORAGE_ROOT/data/processed/openbmi_v8"
BNCI_ACTIVE="$STORAGE_ROOT/cache/mne_data/MNE-bnci-data/~bci/database/004-2014"
BNCI_EVAL_STAGE="$STORAGE_ROOT/data/staged/bnci2014_004_eval"

cd "$PROJECT_ROOT"
export DPC_SNN_STORAGE_ROOT="$STORAGE_ROOT"

"$PYTHON" scripts/validate_bci2a_processed.py \
  --root "$STORAGE_ROOT/data/processed/bci2a_train" --session T \
  --subjects 1 2 3 4 5 6 7 8 9 >/dev/null
"$PYTHON" scripts/validate_bci2a_processed.py \
  --root "$STORAGE_ROOT/data/processed/bci2a_eval" --session E \
  --subjects 1 2 3 4 5 6 7 8 9 >/dev/null
"$PYTHON" scripts/prepare_openbmi_v8_cache.py \
  --storage-root "$STORAGE_ROOT" --output "$OPENBMI_ROOT" \
  --subjects 1-54 --validate-only >/dev/null

for subject in {1..9}; do
  number="$(printf '%02d' "$subject")"
  test -f "$BNCI_ACTIVE/B${number}T.mat"
  test ! -f "$BNCI_ACTIVE/B${number}E.mat"
  test -f "$BNCI_EVAL_STAGE/B${number}E.mat"
done
test "$(wc -l < "$BNCI_EVAL_STAGE/DATASET_SHA256SUMS.txt")" -eq 18
sha256sum -c "$BNCI_EVAL_STAGE/DATASET_SHA256SUMS.txt" >/dev/null

if [ -e "$PAYLOAD_ROOT" ]; then
  echo "Clone payload target already exists: $PAYLOAD_ROOT" >&2
  exit 2
fi
mkdir -p "$PAYLOAD_ROOT"

find . -type d \( -name .git -o -name .pytest_cache -o -name .ruff_cache -o -name __pycache__ -o -name '*.egg-info' \) -prune -o \
  -type f ! -name '*.pyc' ! -name 'SOURCE_SHA256SUMS.txt' -print0 | sort -z | xargs -0 sha256sum \
  >SOURCE_SHA256SUMS.txt
tar --exclude=.git --exclude=.pytest_cache --exclude=.ruff_cache \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='*.egg-info' \
  -czf "$PAYLOAD_ROOT/DPC-SNN_revision_source_clone.tar.gz" .

storage_paths=(
  data/processed/bci2a_train
  data/processed/bci2a_eval
  data/processed/openbmi_v8
  data/staged/bnci2014_004_eval
  cache/mne_data/MNE-bnci-data/~bci/database/004-2014
  baselines/source
)
if [ -d "$STORAGE_ROOT/recovery_labels" ]; then
  storage_paths+=(recovery_labels)
fi
tar -czf "$PAYLOAD_ROOT/DPC-SNN_storage_seed.tar.gz" \
  -C "$STORAGE_ROOT" "${storage_paths[@]}"
cp -- scripts/cloud/bootstrap_cloned_instance.sh "$PAYLOAD_ROOT/bootstrap_cloned_instance.sh"
chmod 755 "$PAYLOAD_ROOT/bootstrap_cloned_instance.sh"

(
  cd "$PAYLOAD_ROOT"
  sha256sum \
    DPC-SNN_revision_source_clone.tar.gz \
    DPC-SNN_storage_seed.tar.gz \
    bootstrap_cloned_instance.sh \
    >PAYLOAD_SHA256SUMS.txt
  sha256sum -c PAYLOAD_SHA256SUMS.txt
)
echo "$PAYLOAD_ROOT"
