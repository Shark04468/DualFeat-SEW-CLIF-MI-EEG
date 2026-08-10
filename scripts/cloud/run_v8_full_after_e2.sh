#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/bin/python
STORAGE=/root/autodl-tmp/DPC-SNN_storage
SNAPSHOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SOURCE_DIGEST=$(cat "$SNAPSHOT/SOURCE_TREE_SHA256")
TAG=${SOURCE_DIGEST:0:12}
RUNS="$STORAGE/runs/v8_accuracy_first"
DATA="$STORAGE/project/data/processed/bci2a_v62"
SOURCES="$STORAGE/baselines/source"
P0="$RUNS/P0_protocol_$TAG"
E0="$RUNS/E0_invariants_$TAG"
E1="$RUNS/E1_baselines_nested_00fbfabdd545"
BASE_E2=${BASE_E2_ROOT:-$RUNS/E2_zero_delay_6228318113df}
BASE_E2_SUPERVISOR=${BASE_E2_SUPERVISOR_PID_FILE:-$RUNS/r5_gain_e3_6228318113df/supervisor.pid}
BASE_E2_SNAPSHOT=${BASE_E2_SOURCE_SNAPSHOT:-$STORAGE/code_snapshots/v8_r5_gain_e3_6228318113df}
STATE="$RUNS/full_after_e2_$TAG"
BASE_E2_GATE="$RUNS/E2_base_gate_$TAG"
E5_CANARY="$RUNS/E5_hpo_canary_$TAG"
E5="$RUNS/E5_hpo_$TAG"
E5_AUDIT="$RUNS/E5_hpo_audit_$TAG"
E2_SELECTED_CANARY="$RUNS/E2_selected_canary_$TAG"
E2_SELECTED="$RUNS/E2_selected_$TAG"
E2_SELECTED_GATE="$RUNS/E2_selected_gate_$TAG"
E3_SEQUENCE="$RUNS/E3_sequence_$TAG"
E4_CANARY="$RUNS/E4_canary_$TAG"
E4="$RUNS/E4_decoder_controls_$TAG"
E4_GATE="$RUNS/E4_gate_$TAG"
FREEZE="$RUNS/freeze_$TAG"
E6_CANARY="$RUNS/E6_canary_$TAG"
E6_BASELINE_CANARY="$RUNS/E6_baseline_canary_$TAG"
E6="$RUNS/E6_bci2a_$TAG"
E6_BASELINES="$RUNS/E6_baselines_$TAG"
E6_AUDIT="$RUNS/E6_audit_$TAG"
E7="$RUNS/E7_utility_$TAG"
E8_UNLOCK="$RUNS/E8_unlock_$TAG"
E8_CANARY="$RUNS/E8_canary_$TAG"
E8="$RUNS/E8_openbmi_$TAG"
E8_AUDIT="$RUNS/E8_audit_$TAG"
E9_CANARY="$RUNS/E9_canary_$TAG"
E9="$RUNS/E9_ablations_$TAG"
E9_AUDIT="$RUNS/E9_audit_$TAG"

mkdir -p "$STATE"
if ! mkdir "$STATE/lock" 2>/dev/null; then
  echo "V8 continuation already owns $STATE" >&2
  exit 2
fi
trap 'rmdir "$STATE/lock" 2>/dev/null || true' EXIT

export DPC_SNN_STORAGE_ROOT="$STORAGE"
export PYTHONPYCACHEPREFIX="$STORAGE/pycache"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="$SNAPSHOT/src:$SNAPSHOT"
cd "$SNAPSHOT"

write_state() {
  local stage=$1
  local status=$2
  "$PY" - "$STATE/status.json" "$stage" "$status" "$TAG" <<'PY'
import json, pathlib, sys, time
path, stage, status, tag = sys.argv[1:]
payload = {"stage": stage, "status": status, "tag": tag, "updated_at": time.time()}
pathlib.Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

write_state waiting_for_base_e2 running
if [ -f "$BASE_E2_SUPERVISOR" ]; then
  E2_PID=$(cat "$BASE_E2_SUPERVISOR")
  while kill -0 "$E2_PID" 2>/dev/null; do
    printf '__V8_WAIT_BASE_E2__ pid=%s time=%s\n' "$E2_PID" "$(date -Iseconds)"
    sleep 60
  done
fi
"$PY" -c "import json; p=json.load(open('$BASE_E2/campaign_status.json')); assert p['status']=='completed' and p['stage']=='E2' and not p['session_e_accessed']"

write_state snapshot_validation running
"$PY" -m pytest -q
printf '{"status":"validated","source_digest":"%s"}\n' "$SOURCE_DIGEST" > "$SNAPSHOT/SNAPSHOT_STATUS.json"

write_state P0 running
"$PY" scripts/run_v8_p0.py \
  --config configs/experiments/v8_p0_protocol.yaml \
  --output "$P0" --storage-root "$STORAGE"

write_state E0 running
"$PY" scripts/run_v8_e0_invariants.py \
  --config configs/experiments/v8_e0_invariants.yaml \
  --output "$E0" --p0 "$P0" --storage-root "$STORAGE" --require-cuda

write_state base_e2_audit running
BEFORE=$(find "$BASE_E2" -path '*/fold_*/result.json' -type f -exec sha256sum {} + | sort | sha256sum | cut -d' ' -f1)
(
  cd "$BASE_E2_SNAPSHOT"
  PYTHONPATH="$BASE_E2_SNAPSHOT/src:$BASE_E2_SNAPSHOT" "$PY" scripts/run_v8_e2_zero_delay.py \
  --data "$DATA" --output "$BASE_E2" \
  --config configs/experiments/v8_e2_zero_delay.yaml --device cuda
)
AFTER=$(find "$BASE_E2" -path '*/fold_*/result.json' -type f -exec sha256sum {} + | sort | sha256sum | cut -d' ' -f1)
test "$BEFORE" = "$AFTER"
"$PY" scripts/evaluate_v8_e2_gate.py \
  --e1 "$E1" --e2 "$BASE_E2" --output "$BASE_E2_GATE" \
  --v8-variant full_ann --maximum-gap-pp 0.5

write_state E5 running
"$PY" scripts/run_v8_e5_bounded_hpo.py \
  --data "$DATA" --output "$E5_CANARY" --physical-cache-root "$BASE_E2" \
  --config configs/experiments/v8_e5_bounded_hpo.yaml \
  --subjects 1 --candidate-limit 1 --stages stage_1 --max-epochs 2 \
  --workers 3 --canary --device cuda
"$PY" scripts/run_v8_e5_bounded_hpo.py \
  --data "$DATA" --output "$E5" --physical-cache-root "$BASE_E2" \
  --config configs/experiments/v8_e5_bounded_hpo.yaml --workers 3 --device cuda
"$PY" scripts/audit_v8_e5_campaign.py \
  --campaign "$E5" --data "$DATA" --output "$E5_AUDIT" \
  --config configs/experiments/v8_e5_bounded_hpo.yaml

SELECTED_MODEL="$E5/selected_model.yaml"
SELECTED_E2_CONFIG="$E5/selected_e2_config.yaml"
SELECTED_E4_CONFIG="$E5/selected_e4_config.yaml"
SELECTED_VARIANT=hpo_selected_full_ann
"$PY" scripts/derive_v8_selected_e4_config.py \
  --selected-e2 "$SELECTED_E2_CONFIG" \
  --base-e4 configs/experiments/v8_e4_decoder_controls.yaml \
  --output "$SELECTED_E4_CONFIG"

write_state selected_E2 running
"$PY" scripts/run_v8_e2_zero_delay.py \
  --data "$DATA" --output "$E2_SELECTED_CANARY" \
  --config "$SELECTED_E2_CONFIG" --model-config "$SELECTED_MODEL" \
  --subjects 1 --variants "$SELECTED_VARIANT" --screening-seed 0 \
  --confirmation-seeds 0 --confirmation-top-k 1 --max-epochs 2 --patience 2 \
  --device cuda
"$PY" scripts/run_v8_e2_zero_delay.py \
  --data "$DATA" --output "$E2_SELECTED" \
  --config "$SELECTED_E2_CONFIG" --model-config "$SELECTED_MODEL" --device cuda
"$PY" scripts/evaluate_v8_e2_gate.py \
  --e1 "$E1" --e2 "$E2_SELECTED" --output "$E2_SELECTED_GATE" \
  --v8-variant "$SELECTED_VARIANT" --maximum-gap-pp 0.5
SELECTED_E2_PASSED=$("$PY" -c "import json; print(int(bool(json.load(open('$E2_SELECTED_GATE/gate_decision.json'))['passed'])))")
if [ "$SELECTED_E2_PASSED" != 1 ]; then
  write_state selected_E2_gate stopped
  echo '__V8_STOP__ selected E2 failed the registered accuracy gate'
  exit 0
fi

write_state E3 running
"$PY" scripts/run_v8_e3_sequence.py \
  --data "$DATA" --parent-e2 "$E2_SELECTED" \
  --parent-variant "$SELECTED_VARIANT" --model-config "$SELECTED_MODEL" \
  --output "$E3_SEQUENCE" --device cuda
test -f "$E3_SEQUENCE/sequence_status.json"

write_state E4 running
"$PY" scripts/run_v8_e4_decoder_controls.py \
  --data "$DATA" --output "$E4_CANARY" --physical-cache-root "$E2_SELECTED" \
  --model-config "$SELECTED_MODEL" --config "$SELECTED_E4_CONFIG" \
  --variants ann_residual,clif_plain --subjects 1 --seeds 0 \
  --max-epochs 2 --patience 2 --canary --device cuda
"$PY" scripts/run_v8_e4_decoder_controls.py \
  --data "$DATA" --output "$E4" --physical-cache-root "$E2_SELECTED" \
  --model-config "$SELECTED_MODEL" --config "$SELECTED_E4_CONFIG" --device cuda
"$PY" scripts/evaluate_v8_e4_gate.py \
  --e4 "$E4" --output "$E4_GATE" --config "$SELECTED_E4_CONFIG"

write_state freeze running
FREEZE_E3_ARGS=(--e3-sequence "$E3_SEQUENCE")
SELECTED_DELAY_STAGE=$("$PY" -c "import json; print(json.load(open('$E3_SEQUENCE/sequence_status.json')).get('selected_delay_stage') or '')")
if [ -n "$SELECTED_DELAY_STAGE" ]; then
  SELECTED_E3_CAMPAIGN=$("$PY" -c "import json; p=json.load(open('$E3_SEQUENCE/sequence_status.json')); print(next(x['campaign'] for x in p['results'] if x['stage']=='$SELECTED_DELAY_STAGE'))")
  SELECTED_E3_GATE=$("$PY" -c "import json; p=json.load(open('$E3_SEQUENCE/sequence_status.json')); print(next(x['gate'] for x in p['results'] if x['stage']=='$SELECTED_DELAY_STAGE'))")
  FREEZE_E3_ARGS+=(--e3 "$SELECTED_E3_CAMPAIGN" --e3-gate "$SELECTED_E3_GATE")
fi
"$PY" scripts/freeze_v8_architecture.py \
  --p0 "$P0" --e0 "$E0" \
  --e1 "$E1" --e1-audit "$E1/audit" \
  --e5 "$E5" --e5-audit "$E5_AUDIT" \
  --selected-e2 "$E2_SELECTED" --selected-e2-gate "$E2_SELECTED_GATE" \
  --e4 "$E4" --e4-gate "$E4_GATE" \
  "${FREEZE_E3_ARGS[@]}" --output "$FREEZE"
FREEZE_MANIFEST="$FREEZE/freeze_manifest.json"

write_state E6_canary running
"$PY" scripts/run_v8_e6_bci2a_frozen.py \
  --data "$DATA" --freeze "$FREEZE_MANIFEST" --output "$E6_CANARY" \
  --subjects 1 --seeds 0 --variants frozen_primary --workers 6 --canary --device cuda
"$PY" scripts/run_v8_e6_baselines_frozen.py \
  --data "$DATA" --source-root "$SOURCES" --freeze "$FREEZE_MANIFEST" \
  --output "$E6_BASELINE_CANARY" --models atcnet --subjects 1 --seeds 0 \
  --workers 6 --canary --device cuda

write_state E6 running
"$PY" scripts/run_v8_e6_bci2a_frozen.py \
  --data "$DATA" --freeze "$FREEZE_MANIFEST" --output "$E6" --workers 6 --device cuda
"$PY" scripts/run_v8_e6_baselines_frozen.py \
  --data "$DATA" --source-root "$SOURCES" --freeze "$FREEZE_MANIFEST" \
  --output "$E6_BASELINES" --workers 6 --device cuda
"$PY" scripts/audit_v8_e6_campaign.py \
  --e6 "$E6" --baselines "$E6_BASELINES" --freeze "$FREEZE_MANIFEST" \
  --output "$E6_AUDIT"

write_state E7 running
"$PY" scripts/run_v8_e7_utility.py \
  --data "$DATA" --e6 "$E6" --e6-audit "$E6_AUDIT" \
  --freeze "$FREEZE_MANIFEST" --output "$E7" --device cuda

write_state E8_unlock running
"$PY" scripts/create_v8_e8_unlock.py \
  --freeze "$FREEZE_MANIFEST" --e6-audit "$E6_AUDIT" --e7 "$E7" \
  --output "$E8_UNLOCK"
E8_UNLOCK_MANIFEST="$E8_UNLOCK/external_unlock_manifest.json"
"$PY" scripts/run_v8_e8_openbmi_confirmation.py \
  --freeze "$FREEZE_MANIFEST" --unlock "$E8_UNLOCK_MANIFEST" \
  --output "$E8_CANARY" --subjects 1 --seeds 0 --arms frozen_primary \
  --workers 6 --canary --device cuda

write_state E8 running
"$PY" scripts/run_v8_e8_openbmi_confirmation.py \
  --freeze "$FREEZE_MANIFEST" --unlock "$E8_UNLOCK_MANIFEST" \
  --output "$E8" --workers 6 --device cuda
"$PY" scripts/audit_v8_e8_openbmi.py \
  --e8 "$E8" --freeze "$FREEZE_MANIFEST" --unlock "$E8_UNLOCK_MANIFEST" \
  --output "$E8_AUDIT"

write_state E9_canary running
"$PY" scripts/run_v8_e9_frozen_ablations.py \
  --data "$DATA" --freeze "$FREEZE_MANIFEST" --e6 "$E6" \
  --e6-audit "$E6_AUDIT" --output "$E9_CANARY" \
  --subjects 1 --seeds 0 --variants no_covariance --workers 6 --canary --device cuda

write_state E9 running
"$PY" scripts/run_v8_e9_frozen_ablations.py \
  --data "$DATA" --freeze "$FREEZE_MANIFEST" --e6 "$E6" \
  --e6-audit "$E6_AUDIT" --output "$E9" --workers 6 --device cuda
"$PY" scripts/audit_v8_e9_ablations.py \
  --e9 "$E9" --e6 "$E6" --e6-audit "$E6_AUDIT" --e7 "$E7" \
  --e8-audit "$E8_AUDIT" --freeze "$FREEZE_MANIFEST" --output "$E9_AUDIT"

write_state E0_E9 completed
echo '__V8_E0_E9_COMPLETE__'
