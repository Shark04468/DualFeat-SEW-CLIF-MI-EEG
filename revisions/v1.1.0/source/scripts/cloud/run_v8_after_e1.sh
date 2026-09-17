#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/bin/python
STORAGE=/root/autodl-tmp/DPC-SNN_storage
AUDIT_SNAPSHOT="$STORAGE/code_snapshots/v8_audit_f09c2ebe29bf"
E1_ROOT="$STORAGE/runs/v8_accuracy_first/E1_baselines_nested_00fbfabdd545"
E2_SNAPSHOT="$STORAGE/code_snapshots/v8_e2_zero_delay_f411cbda6503"
E2_CANARY="$STORAGE/runs/v8_accuracy_first/E2_zero_delay_canary_f411cbda6503"
E2_FORMAL="$STORAGE/runs/v8_accuracy_first/E2_zero_delay_f411cbda6503"
E2_GATE="$STORAGE/runs/v8_accuracy_first/E2_gate_f411cbda6503"
E5_SNAPSHOT="$STORAGE/code_snapshots/v8_e5_selection_active"
E5_CANARY="$STORAGE/runs/v8_accuracy_first/E5_hpo_canary_r1"
E5_FORMAL="$STORAGE/runs/v8_accuracy_first/E5_hpo_r1"
E5_AUDIT="$STORAGE/runs/v8_accuracy_first/E5_hpo_audit_r1"
E2_SELECTED_CANARY="$STORAGE/runs/v8_accuracy_first/E2_hpo_selected_canary_r1"
E2_SELECTED="$STORAGE/runs/v8_accuracy_first/E2_hpo_selected_r1"
E2_SELECTED_GATE="$STORAGE/runs/v8_accuracy_first/E2_hpo_selected_gate_r1"
E3_SNAPSHOT="$E5_SNAPSHOT"
E3_FORMAL="$STORAGE/runs/v8_accuracy_first/E3_selected_static_delay_r1"
E3_GATE="$STORAGE/runs/v8_accuracy_first/E3_selected_gate_r1"
E4_SNAPSHOT="$STORAGE/code_snapshots/v8_e4_decoder_controls_active"
E4_CANARY="$STORAGE/runs/v8_accuracy_first/E4_selected_decoder_canary_r3"
E4_FORMAL="$STORAGE/runs/v8_accuracy_first/E4_selected_decoder_controls_r3"
E4_GATE="$STORAGE/runs/v8_accuracy_first/E4_selected_gate_r3"
DATA="$STORAGE/project/data/processed/bci2a_v62"
STATE="$STORAGE/runs/v8_accuracy_first/after_e1_f411cbda6503"

mkdir -p "$STATE"
if ! mkdir "$STATE/lock" 2>/dev/null; then
  echo "after-E1 supervisor already started" >&2
  exit 2
fi
trap 'rmdir "$STATE/lock" 2>/dev/null || true' EXIT

export DPC_SNN_STORAGE_ROOT="$STORAGE"
export PYTHONPYCACHEPREFIX="$STORAGE/pycache"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

E1_PID=$(cat "$E1_ROOT/campaign.pid")
while kill -0 "$E1_PID" 2>/dev/null; do
  printf '__V8_SUPERVISOR_WAIT__ e1_pid=%s time=%s\n' "$E1_PID" "$(date -Iseconds)"
  sleep 60
done

"$PYTHON" -c "import json; p=json.load(open('$E1_ROOT/campaign_status.json')); assert p['status']=='completed' and p['protocol']=='bci2a_session_t_nested_six_fold_oof' and not p['session_e_accessed'] and p['screened_models']==['eegnet','fbcnet','atcnet','tcformer','eeg_conformer','mi_snn_plif','bfatcnet'] and p['subjects']==[1,3,8] and p['confirmation_seeds']==[0,1,2] and p['runs']==27"
"$PYTHON" -c "import json; p=json.load(open('$AUDIT_SNAPSHOT/SNAPSHOT_STATUS.json')); assert p['status']=='validated' and p['source_digest']=='f09c2ebe29bf5264e6f511466d34d714a22486a5a5f562badd1a377d5560243e'"

cd "$AUDIT_SNAPSHOT"
export PYTHONPATH=src
"$PYTHON" scripts/audit_v8_e1_campaign.py \
  --campaign "$E1_ROOT" \
  --config configs/experiments/v8_e1_baselines.yaml \
  --output "$E1_ROOT/audit"

cd "$E2_SNAPSHOT"
export PYTHONPATH=src
"$PYTHON" -m pytest -q
printf '{"status":"validated","source_digest":"f411cbda650390608f156d20be661a6d8c89aa9a7ebb17e1e5945b01c1c960bc"}\n' \
  > "$E2_SNAPSHOT/SNAPSHOT_STATUS.json"

"$PYTHON" scripts/run_v8_e2_zero_delay.py \
  --data "$DATA" \
  --output "$E2_CANARY" \
  --config configs/experiments/v8_e2_zero_delay.yaml \
  --subjects 1 \
  --variants full_ann \
  --screening-seed 0 \
  --confirmation-seeds 0 \
  --confirmation-top-k 1 \
  --max-epochs 2 \
  --patience 2 \
  --device cuda

BEFORE=$(find "$E2_CANARY" -path '*/fold_*/result.json' -type f -exec sha256sum {} + \
  | sort | sha256sum | cut -d' ' -f1)
"$PYTHON" scripts/run_v8_e2_zero_delay.py \
  --data "$DATA" \
  --output "$E2_CANARY" \
  --config configs/experiments/v8_e2_zero_delay.yaml \
  --subjects 1 \
  --variants full_ann \
  --screening-seed 0 \
  --confirmation-seeds 0 \
  --confirmation-top-k 1 \
  --max-epochs 2 \
  --patience 2 \
  --device cuda
AFTER=$(find "$E2_CANARY" -path '*/fold_*/result.json' -type f -exec sha256sum {} + \
  | sort | sha256sum | cut -d' ' -f1)
test "$BEFORE" = "$AFTER"

"$PYTHON" scripts/run_v8_e2_zero_delay.py \
  --data "$DATA" \
  --output "$E2_FORMAL" \
  --config configs/experiments/v8_e2_zero_delay.yaml \
  --device cuda

FORMAL_BEFORE=$(find "$E2_FORMAL" -path '*/fold_*/result.json' -type f -exec sha256sum {} + \
  | sort | sha256sum | cut -d' ' -f1)
"$PYTHON" scripts/run_v8_e2_zero_delay.py \
  --data "$DATA" \
  --output "$E2_FORMAL" \
  --config configs/experiments/v8_e2_zero_delay.yaml \
  --device cuda
FORMAL_AFTER=$(find "$E2_FORMAL" -path '*/fold_*/result.json' -type f -exec sha256sum {} + \
  | sort | sha256sum | cut -d' ' -f1)
test "$FORMAL_BEFORE" = "$FORMAL_AFTER"

"$PYTHON" scripts/evaluate_v8_e2_gate.py \
  --e1 "$E1_ROOT" \
  --e2 "$E2_FORMAL" \
  --output "$E2_GATE" \
  --v8-variant full_ann \
  --maximum-gap-pp 0.5

BASE_E2_DECISION=$("$PYTHON" -c "import json; print(json.load(open('$E2_GATE/gate_decision.json'))['decision'])")

E5_DIGEST=$(cat "$E5_SNAPSHOT/SOURCE_TREE_SHA256")
"$PYTHON" -c "import json; p=json.load(open('$E5_SNAPSHOT/SNAPSHOT_STATUS.json')); assert p['status']=='validated' and p['source_digest']=='$E5_DIGEST'"
cd "$E5_SNAPSHOT"
export PYTHONPATH=src

"$PYTHON" scripts/run_v8_e5_bounded_hpo.py \
  --data "$DATA" \
  --output "$E5_CANARY" \
  --physical-cache-root "$E2_FORMAL" \
  --config configs/experiments/v8_e5_bounded_hpo.yaml \
  --subjects 1 \
  --candidate-limit 1 \
  --stages stage_1 \
  --max-epochs 2 \
  --canary \
  --device cuda
E5_CANARY_BEFORE=$(find "$E5_CANARY" -path '*/fold_*/result.json' -type f -exec sha256sum {} + \
  | sort | sha256sum | cut -d' ' -f1)
"$PYTHON" scripts/run_v8_e5_bounded_hpo.py \
  --data "$DATA" \
  --output "$E5_CANARY" \
  --physical-cache-root "$E2_FORMAL" \
  --config configs/experiments/v8_e5_bounded_hpo.yaml \
  --subjects 1 \
  --candidate-limit 1 \
  --stages stage_1 \
  --max-epochs 2 \
  --canary \
  --device cuda
E5_CANARY_AFTER=$(find "$E5_CANARY" -path '*/fold_*/result.json' -type f -exec sha256sum {} + \
  | sort | sha256sum | cut -d' ' -f1)
test "$E5_CANARY_BEFORE" = "$E5_CANARY_AFTER"

"$PYTHON" scripts/run_v8_e5_bounded_hpo.py \
  --data "$DATA" \
  --output "$E5_FORMAL" \
  --physical-cache-root "$E2_FORMAL" \
  --config configs/experiments/v8_e5_bounded_hpo.yaml \
  --device cuda
E5_FORMAL_BEFORE=$(find "$E5_FORMAL" -path '*/fold_*/result.json' -type f -exec sha256sum {} + \
  | sort | sha256sum | cut -d' ' -f1)
"$PYTHON" scripts/run_v8_e5_bounded_hpo.py \
  --data "$DATA" \
  --output "$E5_FORMAL" \
  --physical-cache-root "$E2_FORMAL" \
  --config configs/experiments/v8_e5_bounded_hpo.yaml \
  --device cuda
E5_FORMAL_AFTER=$(find "$E5_FORMAL" -path '*/fold_*/result.json' -type f -exec sha256sum {} + \
  | sort | sha256sum | cut -d' ' -f1)
test "$E5_FORMAL_BEFORE" = "$E5_FORMAL_AFTER"
"$PYTHON" scripts/audit_v8_e5_campaign.py \
  --campaign "$E5_FORMAL" \
  --data "$DATA" \
  --output "$E5_AUDIT" \
  --config configs/experiments/v8_e5_bounded_hpo.yaml

SELECTED_MODEL="$E5_FORMAL/selected_model.yaml"
SELECTED_E2_CONFIG="$E5_FORMAL/selected_e2_config.yaml"
SELECTED_E4_CONFIG="$E5_FORMAL/selected_e4_config.yaml"
SELECTED_VARIANT="hpo_selected_full_ann"

"$PYTHON" -c "import yaml; e2=yaml.safe_load(open('$SELECTED_E2_CONFIG')); e4=yaml.safe_load(open('$E4_SNAPSHOT/configs/experiments/v8_e4_decoder_controls.yaml')); e4['architecture_version']=e2['architecture_version']; e4['training']=e2['training']; e4['augmentation']=e2['augmentation']; e4['preprocessing']=e2['preprocessing']; e4['selection'].update({k:e2['selection'][k] for k in ('max_epochs','patience','minimum_epochs','minimum_outer_retrain_epochs')}); open('$SELECTED_E4_CONFIG','w').write(yaml.safe_dump(e4,sort_keys=False))"

"$PYTHON" scripts/run_v8_e2_zero_delay.py \
  --data "$DATA" \
  --output "$E2_SELECTED_CANARY" \
  --config "$SELECTED_E2_CONFIG" \
  --model-config "$SELECTED_MODEL" \
  --subjects 1 \
  --variants "$SELECTED_VARIANT" \
  --screening-seed 0 \
  --confirmation-seeds 0 \
  --confirmation-top-k 1 \
  --max-epochs 2 \
  --patience 2 \
  --device cuda
E2_SELECTED_CANARY_BEFORE=$(find "$E2_SELECTED_CANARY" -path '*/fold_*/result.json' -type f -exec sha256sum {} + \
  | sort | sha256sum | cut -d' ' -f1)
"$PYTHON" scripts/run_v8_e2_zero_delay.py \
  --data "$DATA" \
  --output "$E2_SELECTED_CANARY" \
  --config "$SELECTED_E2_CONFIG" \
  --model-config "$SELECTED_MODEL" \
  --subjects 1 \
  --variants "$SELECTED_VARIANT" \
  --screening-seed 0 \
  --confirmation-seeds 0 \
  --confirmation-top-k 1 \
  --max-epochs 2 \
  --patience 2 \
  --device cuda
E2_SELECTED_CANARY_AFTER=$(find "$E2_SELECTED_CANARY" -path '*/fold_*/result.json' -type f -exec sha256sum {} + \
  | sort | sha256sum | cut -d' ' -f1)
test "$E2_SELECTED_CANARY_BEFORE" = "$E2_SELECTED_CANARY_AFTER"

"$PYTHON" scripts/run_v8_e2_zero_delay.py \
  --data "$DATA" \
  --output "$E2_SELECTED" \
  --config "$SELECTED_E2_CONFIG" \
  --model-config "$SELECTED_MODEL" \
  --device cuda
E2_SELECTED_BEFORE=$(find "$E2_SELECTED" -path '*/fold_*/result.json' -type f -exec sha256sum {} + \
  | sort | sha256sum | cut -d' ' -f1)
"$PYTHON" scripts/run_v8_e2_zero_delay.py \
  --data "$DATA" \
  --output "$E2_SELECTED" \
  --config "$SELECTED_E2_CONFIG" \
  --model-config "$SELECTED_MODEL" \
  --device cuda
E2_SELECTED_AFTER=$(find "$E2_SELECTED" -path '*/fold_*/result.json' -type f -exec sha256sum {} + \
  | sort | sha256sum | cut -d' ' -f1)
test "$E2_SELECTED_BEFORE" = "$E2_SELECTED_AFTER"
"$PYTHON" scripts/evaluate_v8_e2_gate.py \
  --e1 "$E1_ROOT" \
  --e2 "$E2_SELECTED" \
  --output "$E2_SELECTED_GATE" \
  --v8-variant "$SELECTED_VARIANT" \
  --maximum-gap-pp 0.5
SELECTED_E2_DECISION=$("$PYTHON" -c "import json; print(json.load(open('$E2_SELECTED_GATE/gate_decision.json'))['decision'])")

E3_DECISION="not_run_selected_e2_gate_failed"
E3_RC=0
E4_DECISION="not_run_selected_e2_gate_failed"
if [ "$SELECTED_E2_DECISION" = "advance_to_E3" ]; then
  set +e
  (
    set -e
    cd "$E3_SNAPSHOT"
    export PYTHONPATH=src
    "$PYTHON" scripts/run_v8_e3_static_delay.py \
      --data "$DATA" \
      --parent-e2 "$E2_SELECTED" \
      --parent-variant "$SELECTED_VARIANT" \
      --model-config "$SELECTED_MODEL" \
      --output "$E3_FORMAL" \
      --config configs/experiments/v8_e3_static_delay.yaml \
      --subjects 1 \
      --seeds 0 \
      --device cuda
    E3_PILOT_BEFORE=$(find "$E3_FORMAL/subject_01/seed_0" -path '*/fold_*/result.json' -type f -exec sha256sum {} + \
      | sort | sha256sum | cut -d' ' -f1)
    "$PYTHON" scripts/run_v8_e3_static_delay.py \
      --data "$DATA" \
      --parent-e2 "$E2_SELECTED" \
      --parent-variant "$SELECTED_VARIANT" \
      --model-config "$SELECTED_MODEL" \
      --output "$E3_FORMAL" \
      --config configs/experiments/v8_e3_static_delay.yaml \
      --subjects 1 \
      --seeds 0 \
      --device cuda
    E3_PILOT_AFTER=$(find "$E3_FORMAL/subject_01/seed_0" -path '*/fold_*/result.json' -type f -exec sha256sum {} + \
      | sort | sha256sum | cut -d' ' -f1)
    test "$E3_PILOT_BEFORE" = "$E3_PILOT_AFTER"
    "$PYTHON" scripts/run_v8_e3_static_delay.py \
      --data "$DATA" \
      --parent-e2 "$E2_SELECTED" \
      --parent-variant "$SELECTED_VARIANT" \
      --model-config "$SELECTED_MODEL" \
      --output "$E3_FORMAL" \
      --config configs/experiments/v8_e3_static_delay.yaml \
      --device cuda
    E3_FORMAL_BEFORE=$(find "$E3_FORMAL" -path '*/fold_*/result.json' -type f -exec sha256sum {} + \
      | sort | sha256sum | cut -d' ' -f1)
    "$PYTHON" scripts/run_v8_e3_static_delay.py \
      --data "$DATA" \
      --parent-e2 "$E2_SELECTED" \
      --parent-variant "$SELECTED_VARIANT" \
      --model-config "$SELECTED_MODEL" \
      --output "$E3_FORMAL" \
      --config configs/experiments/v8_e3_static_delay.yaml \
      --device cuda
    E3_FORMAL_AFTER=$(find "$E3_FORMAL" -path '*/fold_*/result.json' -type f -exec sha256sum {} + \
      | sort | sha256sum | cut -d' ' -f1)
    test "$E3_FORMAL_BEFORE" = "$E3_FORMAL_AFTER"
    "$PYTHON" scripts/evaluate_v8_e3_gate.py \
      --e3 "$E3_FORMAL" \
      --output "$E3_GATE" \
      --config configs/experiments/v8_e3_static_delay.yaml \
      --parent-variant "$SELECTED_VARIANT" \
      --minimum-median-gain-pp 0.5 \
      --minimum-positive-pairs 6
  )
  E3_RC=$?
  set -e
  if [ -f "$E3_GATE/gate_decision.json" ]; then
    E3_DECISION=$("$PYTHON" -c "import json; print(json.load(open('$E3_GATE/gate_decision.json'))['decision'])")
  else
    E3_DECISION="stopped_before_gate_rc_${E3_RC}"
  fi

  E4_DIGEST=$(cat "$E4_SNAPSHOT/SOURCE_TREE_SHA256")
  "$PYTHON" -c "import json; p=json.load(open('$E4_SNAPSHOT/SNAPSHOT_STATUS.json')); assert p['status']=='validated' and p['source_digest']=='$E4_DIGEST'"
  cd "$E4_SNAPSHOT"
  export PYTHONPATH=src
  "$PYTHON" scripts/run_v8_e4_decoder_controls.py \
    --data "$DATA" \
    --output "$E4_CANARY" \
    --physical-cache-root "$E2_SELECTED" \
    --model-config "$SELECTED_MODEL" \
    --config "$SELECTED_E4_CONFIG" \
    --variants ann_residual,clif_plain \
    --subjects 1 \
    --seeds 0 \
    --max-epochs 2 \
    --patience 2 \
    --canary \
    --device cuda
  E4_CANARY_BEFORE=$(find "$E4_CANARY" -path '*/fold_*/result.json' -type f -exec sha256sum {} + \
    | sort | sha256sum | cut -d' ' -f1)
  "$PYTHON" scripts/run_v8_e4_decoder_controls.py \
    --data "$DATA" \
    --output "$E4_CANARY" \
    --physical-cache-root "$E2_SELECTED" \
    --model-config "$SELECTED_MODEL" \
    --config "$SELECTED_E4_CONFIG" \
    --variants ann_residual,clif_plain \
    --subjects 1 \
    --seeds 0 \
    --max-epochs 2 \
    --patience 2 \
    --canary \
    --device cuda
  E4_CANARY_AFTER=$(find "$E4_CANARY" -path '*/fold_*/result.json' -type f -exec sha256sum {} + \
    | sort | sha256sum | cut -d' ' -f1)
  test "$E4_CANARY_BEFORE" = "$E4_CANARY_AFTER"
  "$PYTHON" scripts/run_v8_e4_decoder_controls.py \
    --data "$DATA" \
    --output "$E4_FORMAL" \
    --physical-cache-root "$E2_SELECTED" \
    --model-config "$SELECTED_MODEL" \
    --config "$SELECTED_E4_CONFIG" \
    --device cuda
  E4_FORMAL_BEFORE=$(find "$E4_FORMAL" -path '*/fold_*/result.json' -type f -exec sha256sum {} + \
    | sort | sha256sum | cut -d' ' -f1)
  "$PYTHON" scripts/run_v8_e4_decoder_controls.py \
    --data "$DATA" \
    --output "$E4_FORMAL" \
    --physical-cache-root "$E2_SELECTED" \
    --model-config "$SELECTED_MODEL" \
    --config "$SELECTED_E4_CONFIG" \
    --device cuda
  E4_FORMAL_AFTER=$(find "$E4_FORMAL" -path '*/fold_*/result.json' -type f -exec sha256sum {} + \
    | sort | sha256sum | cut -d' ' -f1)
  test "$E4_FORMAL_BEFORE" = "$E4_FORMAL_AFTER"
  "$PYTHON" scripts/evaluate_v8_e4_gate.py \
    --e4 "$E4_FORMAL" \
    --output "$E4_GATE" \
    --config "$SELECTED_E4_CONFIG"
  E4_DECISION=$("$PYTHON" -c "import json; print(json.load(open('$E4_GATE/gate_decision.json'))['decision'])")
fi

printf '{"status":"completed","e1_audit":"passed","base_e2_gate_decision":"%s","e5_audit":"passed","selected_e2_gate_decision":"%s","e3_exit_code":%s,"e3_gate_decision":"%s","e4_gate_decision":"%s","session_e_accessed":false,"openbmi_s2_accessed":false}\n' \
  "$BASE_E2_DECISION" "$SELECTED_E2_DECISION" "$E3_RC" "$E3_DECISION" "$E4_DECISION" > "$STATE/status.json"
echo '__V8_SUPERVISOR_DONE__'
