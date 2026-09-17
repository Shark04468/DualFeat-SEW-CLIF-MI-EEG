#!/usr/bin/env bash
set -euo pipefail

STORAGE=${DPC_SNN_STORAGE_ROOT:-/root/autodl-tmp/DPC-SNN_storage}
PYTHON=${DPC_SNN_PYTHON:-/root/miniconda3/bin/python}
: "${DPC_SNN_SOURCE_SNAPSHOT:?Set DPC_SNN_SOURCE_SNAPSHOT to the validated immutable code snapshot}"

SNAPSHOT=$DPC_SNN_SOURCE_SNAPSHOT
DATA=${DPC_SNN_BCI2A_DATA:-$STORAGE/data/processed/bci2a}
CACHE_PARENT=${DPC_SNN_PHYSICAL_CACHE_PARENT:-$STORAGE/runs/v8_accuracy_first/E2_zero_delay_6228318113df}
BASELINE_ROOT=${DPC_SNN_E1_BASELINE_ROOT:-$STORAGE/runs/v8_accuracy_first/E1_baselines_nested_00fbfabdd545}
WORKERS=${DPC_SNN_E3_WORKERS:-2}

test -d "$SNAPSHOT"
test -d "$DATA"
test -d "$CACHE_PARENT/shared_physical_rates"
test -d "$BASELINE_ROOT/atcnet"
test -d "$BASELINE_ROOT/fbcnet"
test -f "$SNAPSHOT/SOURCE_TREE_SHA256"
test -f "$SNAPSHOT/SNAPSHOT_STATUS.json"

DIGEST=$(cat "$SNAPSHOT/SOURCE_TREE_SHA256")
TAG=${DIGEST:0:12}
"$PYTHON" -c "import json; p=json.load(open('$SNAPSHOT/SNAPSHOT_STATUS.json')); assert p['status']=='validated' and p['source_digest']=='$DIGEST'"

export DPC_SNN_STORAGE_ROOT=$STORAGE
export TORCH_HOME=$STORAGE/cache/torch
export XDG_CACHE_HOME=$STORAGE/cache/xdg
export HF_HOME=$STORAGE/cache/huggingface
export MPLCONFIGDIR=$STORAGE/cache/matplotlib
export TMPDIR=$STORAGE/tmp
mkdir -p "$TORCH_HOME" "$XDG_CACHE_HOME" "$HF_HOME" "$MPLCONFIGDIR" "$TMPDIR" "$STORAGE/logs"

PILOT=$STORAGE/runs/v8_accuracy_first/E3_matched_transport_canary_$TAG
HPO=$STORAGE/runs/v8_accuracy_first/E5_matched_transport_hpo_$TAG
HPO_AUDIT=$STORAGE/runs/v8_accuracy_first/E5_matched_transport_hpo_audit_$TAG
FORMAL=$STORAGE/runs/v8_accuracy_first/E3_matched_transport_formal_$TAG
GATE=$STORAGE/runs/v8_accuracy_first/E3_matched_transport_gate_$TAG
STATUS=$STORAGE/runs/v8_accuracy_first/E3_matched_transport_status_$TAG.json

cd "$SNAPSHOT"

"$PYTHON" scripts/run_v8_e3_delay_residual_campaign.py \
  --data "$DATA" \
  --cache-parent "$CACHE_PARENT" \
  --atc-root "$BASELINE_ROOT" \
  --fbc-root "$BASELINE_ROOT" \
  --output "$PILOT" \
  --subjects 1 --seeds 0 --folds 0 \
  --workers 1 --device cuda --canary

"$PYTHON" -c "import json; p=json.load(open('$PILOT/campaign_status.json')); assert p['status']=='completed' and p['canary'] and p['fold_runs']==1 and not p['session_e_accessed'] and not p['openbmi_s2_accessed']"

"$PYTHON" scripts/run_v8_e5_matched_transport_hpo.py \
  --data "$DATA" \
  --cache-parent "$CACHE_PARENT" \
  --atc-root "$BASELINE_ROOT" \
  --fbc-root "$BASELINE_ROOT" \
  --output "$HPO" \
  --workers "$WORKERS" --device cuda

"$PYTHON" scripts/audit_v8_e5_matched_transport_hpo.py \
  --campaign "$HPO" \
  --output "$HPO_AUDIT"

"$PYTHON" -c "import json; p=json.load(open('$HPO_AUDIT/audit_report.json')); assert p['status']=='passed' and p['rankings_recomputed'] and p['selection_uses_inner_validation_only'] and not p['full_delay_metrics_used'] and not p['outer_test_metrics_used'] and not p['session_e_accessed'] and not p['openbmi_s2_accessed']"

SELECTED_CONFIG=$HPO/selected_e3_config.yaml
test -f "$SELECTED_CONFIG"

"$PYTHON" scripts/run_v8_e3_delay_residual_campaign.py \
  --data "$DATA" \
  --cache-parent "$CACHE_PARENT" \
  --atc-root "$BASELINE_ROOT" \
  --fbc-root "$BASELINE_ROOT" \
  --output "$FORMAL" \
  --config "$SELECTED_CONFIG" \
  --workers "$WORKERS" --device cuda

"$PYTHON" scripts/evaluate_v8_e3_matched_transport_gate.py \
  --campaign "$FORMAL" \
  --data "$DATA" \
  --output "$GATE" \
  --config "$SELECTED_CONFIG"

"$PYTHON" -c "import json; p=json.load(open('$GATE/gate_decision.json')); a=json.load(open('$HPO_AUDIT/audit_report.json')); json.dump({'status':'completed','source_digest':'$DIGEST','pilot':'$PILOT','hpo':'$HPO','hpo_audit':'$HPO_AUDIT','selected_candidate':a['selected_candidate'],'selected_e3_config':'$SELECTED_CONFIG','formal':'$FORMAL','gate':'$GATE','delay_passed':bool(p['passed']),'decision':p['decision'],'session_e_accessed':False,'openbmi_s2_accessed':False},open('$STATUS','w'),indent=2); print(json.dumps(p,indent=2))"

echo "__V8_E3_MATCHED_TRANSPORT_COMPLETE__ status=$STATUS"
