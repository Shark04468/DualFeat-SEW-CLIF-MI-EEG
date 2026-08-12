#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/DPC-SNN}"
STORAGE_ROOT="${DPC_SNN_STORAGE_ROOT:-/root/autodl-tmp/DPC-SNN_storage}"
STAGING_ROOT="${STAGING_ROOT:-/root/autodl-tmp/v32_fusion_staging}"
PURITY_ROOT="${PURITY_ROOT:-$STORAGE_ROOT/runs/v32_purity}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
LOG_ROOT="$STORAGE_ROOT/runs/v32_fusion_activation"

mkdir -p "$LOG_ROOT"
while true; do
  if [ -f "$PURITY_ROOT/aggregate_full/decision.json" ]; then
    decision="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$PURITY_ROOT/aggregate_full/decision.json")"
    if [ "$decision" != "pass" ]; then
      printf '{"status":"blocked","reason":"purity_gate_%s"}\n' "$decision" \
        >"$LOG_ROOT/status.json"
      exit 3
    fi
    if ! pgrep -f '[s]cripts/cloud/run_v32_purity.sh' >/dev/null; then
      break
    fi
  fi
  sleep 60
done

install -m 0644 "$STAGING_ROOT/run_v32_purity_fold.py" \
  "$PROJECT_ROOT/scripts/run_v32_purity_fold.py"
install -m 0644 "$STAGING_ROOT/aggregate_v32_fusion.py" \
  "$PROJECT_ROOT/scripts/aggregate_v32_fusion.py"
install -m 0755 "$STAGING_ROOT/run_v32_fusion.sh" \
  "$PROJECT_ROOT/scripts/cloud/run_v32_fusion.sh"
install -m 0644 "$STAGING_ROOT/v32_matched_purity_and_fusion_smoke.yaml" \
  "$PROJECT_ROOT/configs/experiments/v32_matched_purity_and_fusion_smoke.yaml"
install -m 0644 "$STAGING_ROOT/test_v32_fusion_aggregation.py" \
  "$PROJECT_ROOT/tests/test_v32_fusion_aggregation.py"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
bash -n scripts/cloud/run_v32_fusion.sh
"$PYTHON" -m py_compile \
  scripts/run_v32_purity_fold.py scripts/aggregate_v32_fusion.py
"$PYTHON" -m pytest \
  tests/test_v32_fusion_aggregation.py \
  tests/models/test_v9_dual_feature_student.py \
  tests/models/test_v10_strong_controls.py \
  tests/test_v9_dual_feature_training.py -q \
  >"$LOG_ROOT/tests.log" 2>&1

cache="$STORAGE_ROOT/runs/v9_accuracy_first/E3_dual_feature_8fe5f147e7ce/subject_01/seed_0/fold_0"
if [ ! -f "$cache/frozen_dual_feature_cache.npz" ]; then
  printf '{"status":"blocked","reason":"missing_smoke_cache"}\n' >"$LOG_ROOT/status.json"
  exit 4
fi
smoke_root="$STORAGE_ROOT/tmp_v32_fusion_smoke_$(date +%s)"
for mode in atc_only fbc_only simple; do
  "$PYTHON" scripts/run_v32_purity_fold.py \
    --data "$PROJECT_ROOT/data/processed/bci2a_v8_session_t_4d04d3337fe3" \
    --cache-fold "$cache" --output "$smoke_root/$mode" \
    --subject 1 --seed 0 --fold 0 --fusion-mode "$mode" --device cuda \
    --config configs/experiments/v32_matched_purity_and_fusion_smoke.yaml \
    >"$LOG_ROOT/smoke_${mode}.log" 2>&1
done

printf '{"status":"fusion_training","smoke_root":"%s"}\n' "$smoke_root" \
  >"$LOG_ROOT/status.json"
exec bash scripts/cloud/run_v32_fusion.sh \
  >"$LOG_ROOT/fusion_pipeline.log" 2>&1
