#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/dpc_recovery/repo}"
STORAGE_ROOT="${DPC_SNN_STORAGE_ROOT:-/root/autodl-tmp/DPC-SNN_storage}"
VENV_ROOT="${VENV_ROOT:-/root/autodl-tmp/dpc_recovery/venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONTROL_ROOT="${CONTROL_ROOT:-/root/autodl-tmp/dpc_recovery/bootstrap}"
MIN_DATA_DISK_GIB="${MIN_DATA_DISK_GIB:-35}"
EXPECTED_GPU_SUBSTRING="${EXPECTED_GPU_SUBSTRING:-5090}"

mkdir -p "$CONTROL_ROOT" "$STORAGE_ROOT"

available_kib="$(df -Pk "$STORAGE_ROOT" | awk 'NR == 2 {print $4}')"
minimum_kib="$((MIN_DATA_DISK_GIB * 1024 * 1024))"
if ((available_kib < minimum_kib)); then
  echo "At least ${MIN_DATA_DISK_GIB} GiB free is required; found $((available_kib / 1024 / 1024)) GiB" >&2
  exit 2
fi

if ! command -v aria2c >/dev/null 2>&1; then
  apt-get update >"$CONTROL_ROOT/aria2_install.log" 2>&1
  apt-get install -y aria2 >>"$CONTROL_ROOT/aria2_install.log" 2>&1
fi

if [ ! -x "$VENV_ROOT/bin/python" ]; then
  "$PYTHON_BIN" -m venv --system-site-packages "$VENV_ROOT"
fi
source "$VENV_ROOT/bin/activate"
cd "$PROJECT_ROOT"

python -m pip install --disable-pip-version-check -e '.[all,dev]' \
  >"$CONTROL_ROOT/pip_install.log" 2>&1

if [ -f SOURCE_SHA256SUMS.txt ]; then
  sha256sum -c SOURCE_SHA256SUMS.txt >"$CONTROL_ROOT/source_manifest_check.log"
fi

python scripts/validate_bci2a_processed.py \
  --root "$STORAGE_ROOT/data/processed/bci2a_train" --session T \
  --subjects 1 2 3 4 5 6 7 8 9 \
  >"$CONTROL_ROOT/bci2a_train_validation.json"
python scripts/validate_bci2a_processed.py \
  --root "$STORAGE_ROOT/data/processed/bci2a_eval" --session E \
  --subjects 1 2 3 4 5 6 7 8 9 \
  >"$CONTROL_ROOT/bci2a_eval_validation.json"
python scripts/prepare_openbmi_v8_cache.py \
  --storage-root "$STORAGE_ROOT" \
  --output "$STORAGE_ROOT/data/processed/openbmi_v8" \
  --subjects 1-54 --validate-only \
  >"$CONTROL_ROOT/openbmi_validation.log"
test "$(wc -l < "$STORAGE_ROOT/data/staged/bnci2014_004_eval/DATASET_SHA256SUMS.txt")" -eq 18
sha256sum -c \
  "$STORAGE_ROOT/data/staged/bnci2014_004_eval/DATASET_SHA256SUMS.txt" \
  >"$CONTROL_ROOT/bnci2014_004_sha256_check.log"

bash -n \
  scripts/cloud/bootstrap_revision_4gpu.sh \
  scripts/cloud/prefetch_openbmi_aria2.sh \
  scripts/cloud/run_revision_recovery_4gpu.sh

export EXPECTED_GPU_SUBSTRING
python - <<'PY' >"$CONTROL_ROOT/hardware.json"
import json
import os
import platform

import torch

names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
expected = os.environ["EXPECTED_GPU_SUBSTRING"]
if len(names) != 4 or any(expected not in name for name in names):
    raise SystemExit(f"Expected exactly four GPUs matching {expected!r}; found {names}")
payload = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "expected_gpu_substring": expected,
    "gpu_names": names,
}
print(json.dumps(payload, indent=2))
PY

python -m pytest -q 2>&1 | tee "$CONTROL_ROOT/pytest.log"
python scripts/validate_revision_launch_plan.py \
  --output "$CONTROL_ROOT/four_gpu_launch_dry_run.json"
touch "$CONTROL_ROOT/READY_TO_LAUNCH"
echo "$CONTROL_ROOT/READY_TO_LAUNCH"
