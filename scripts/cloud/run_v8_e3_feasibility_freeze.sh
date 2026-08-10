#!/usr/bin/env bash
set -euo pipefail

STORAGE=${DPC_SNN_STORAGE_ROOT:-/root/autodl-tmp/DPC-SNN_storage}
PYTHON=${DPC_SNN_PYTHON:-/root/miniconda3/bin/python}
: "${DPC_SNN_SOURCE_SNAPSHOT:?Set DPC_SNN_SOURCE_SNAPSHOT to the validated evaluator snapshot}"
: "${DPC_SNN_E5_PARTIAL_ROOT:?Set DPC_SNN_E5_PARTIAL_ROOT to the stopped E5 campaign}"
: "${DPC_SNN_E4_CONFIRMATION_GATE:?Set DPC_SNN_E4_CONFIRMATION_GATE to the locked E4 gate.json}"

SNAPSHOT=$DPC_SNN_SOURCE_SNAPSHOT
PRIOR_ROOT=$DPC_SNN_E5_PARTIAL_ROOT/_priors
E4_GATE=$DPC_SNN_E4_CONFIRMATION_GATE

test -d "$SNAPSHOT"
test -d "$PRIOR_ROOT"
test -f "$E4_GATE"
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

FEASIBILITY=$STORAGE/runs/v8_accuracy_first/E3_prior_feasibility_gate_$TAG
FINAL_GATE=$STORAGE/runs/v8_accuracy_first/V8_final_architecture_gate_$TAG
STATUS=$STORAGE/runs/v8_accuracy_first/V8_final_architecture_status_$TAG.json

cd "$SNAPSHOT"

"$PYTHON" scripts/evaluate_v8_e3_prior_feasibility_gate.py \
  --prior-root "$PRIOR_ROOT" \
  --output "$FEASIBILITY"

"$PYTHON" -c "import json; p=json.load(open('$FEASIBILITY/feasibility_decision.json')); assert p['status']=='completed' and not p['passed'] and not p['classifier_training_authorized'] and not p['session_e_accessed'] and not p['openbmi_s2_accessed']"

"$PYTHON" scripts/evaluate_v8_e3_final_architecture_gate.py \
  --feasibility-gate "$FEASIBILITY" \
  --e4-confirmation-gate "$E4_GATE" \
  --output "$FINAL_GATE"

"$PYTHON" -c "import json; p=json.load(open('$FINAL_GATE/decision.json')); assert p['status']=='completed' and p['selected_branch']=='confirmed_e4_sequence_residual' and not p['e3_delay_gate_evaluated'] and not p['session_e_accessed'] and not p['openbmi_s2_accessed']; json.dump({'status':'completed','source_digest':'$DIGEST','feasibility_gate':'$FEASIBILITY','final_gate':'$FINAL_GATE','selected_branch':p['selected_branch'],'session_e_accessed':False,'openbmi_s2_accessed':False},open('$STATUS','w'),indent=2); print(json.dumps(p,indent=2))"

echo "__V8_FINAL_ARCHITECTURE_GATE_COMPLETE__ status=$STATUS"
