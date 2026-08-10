#!/usr/bin/env bash
set -euo pipefail

# Relocate heavyweight DPC-SNN data and framework caches from the root disk to a
# larger data disk. The script preserves the existing project paths with symlinks.
STORAGE_ROOT="${1:-${DPC_SNN_STORAGE_ROOT:-/root/autodl-tmp/DPC-SNN_storage}}"
PROJECT_ROOT="${PROJECT_ROOT:-/root/DPC-SNN}"
CACHE_ROOT="$STORAGE_ROOT/cache"
PROJECT_STORAGE="$STORAGE_ROOT/project"

mkdir -p "$CACHE_ROOT" "$PROJECT_STORAGE"

move_link() {
  local src="$1"
  local dest="$2"
  if [ -L "$src" ]; then
    echo "skip symlink: $src -> $(readlink "$src")"
    return 0
  fi
  if [ ! -e "$src" ]; then
    echo "skip missing: $src"
    return 0
  fi
  if [ -e "$dest" ]; then
    echo "destination exists, not overwriting: $dest" >&2
    return 1
  fi
  mkdir -p "$(dirname "$dest")"
  mv "$src" "$dest"
  ln -s "$dest" "$src"
  echo "linked: $src -> $dest"
}

move_link /root/mne_data "$CACHE_ROOT/mne_data"
move_link "$PROJECT_ROOT/data" "$PROJECT_STORAGE/data"
move_link "$PROJECT_ROOT/runs" "$PROJECT_STORAGE/runs"
move_link "$PROJECT_ROOT/remote_logs" "$PROJECT_STORAGE/remote_logs"

mkdir -p "$CACHE_ROOT"/{mne_data,moabb,xdg,hf,huggingface/datasets,torch,matplotlib,pip,tmp}
mkdir -p "$STORAGE_ROOT"/{pycache,logs/wandb,checkpoints,tmp,runs}
cat > "$PROJECT_ROOT/.env.storage" <<EOF
export DPC_SNN_STORAGE_ROOT=$STORAGE_ROOT
export MNE_DATA=$CACHE_ROOT/mne_data
export MNE_DATASETS_EEGBCI_PATH=$CACHE_ROOT/mne_data/MNE-eegbci-data
export MOABB_DATA=$CACHE_ROOT/moabb
export XDG_CACHE_HOME=$CACHE_ROOT/xdg
export HF_HOME=$CACHE_ROOT/hf
export HF_DATASETS_CACHE=$CACHE_ROOT/huggingface/datasets
export TORCH_HOME=$CACHE_ROOT/torch
export MPLCONFIGDIR=$CACHE_ROOT/matplotlib
export PIP_CACHE_DIR=$CACHE_ROOT/pip
export TMPDIR=$CACHE_ROOT/tmp
export TMP=$CACHE_ROOT/tmp
export TEMP=$CACHE_ROOT/tmp
export PYTHONPYCACHEPREFIX=$STORAGE_ROOT/pycache
export WANDB_DIR=$STORAGE_ROOT/logs/wandb
EOF

echo "storage configured at $STORAGE_ROOT"
df -hT / "$STORAGE_ROOT" || true
