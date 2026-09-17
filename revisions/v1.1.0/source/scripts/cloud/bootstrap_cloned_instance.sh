#!/usr/bin/env bash
set -euo pipefail

PAYLOAD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${CLONE_DATA_ROOT:-/root/autodl-tmp}"
PROJECT_ROOT="$DATA_ROOT/dpc_recovery/repo"
STORAGE_ROOT="$DATA_ROOT/DPC-SNN_storage"

cd "$PAYLOAD_ROOT"
sha256sum -c PAYLOAD_SHA256SUMS.txt
mkdir -p "$PROJECT_ROOT" "$STORAGE_ROOT"
tar -xzf DPC-SNN_revision_source_clone.tar.gz -C "$PROJECT_ROOT"
tar -xzf DPC-SNN_storage_seed.tar.gz -C "$STORAGE_ROOT"

export PROJECT_ROOT
export DPC_SNN_STORAGE_ROOT="$STORAGE_ROOT"
bash "$PROJECT_ROOT/scripts/cloud/bootstrap_revision_4gpu.sh"
echo "Clone payload restored and four-GPU bootstrap passed"
