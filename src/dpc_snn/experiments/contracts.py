"""Post-run contract checks for experiment outputs.

These checks are intentionally stricter than smoke tests. They guard against
CSV-producing runner shortcuts that do not satisfy the research protocol.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


class ExperimentContractError(RuntimeError):
    """Raised when an experiment output violates its declared protocol."""


def _status_is_needs_input(output_dir: Path, result: dict[str, Any]) -> bool:
    if str(result.get("status", "")).lower() == "needs_input":
        return True
    status_path = output_dir / "runner_status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        if str(status.get("status", "")).lower() == "needs_input":
            return True
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        return str(manifest.get("status", "")).lower() == "needs_input"
    return False


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise ExperimentContractError(f"Missing required CSV: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _require_nonempty(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path)
    if not rows:
        raise ExperimentContractError(f"CSV must not be empty for a completed run: {path}")
    return rows


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _require_columns(rows: list[dict[str, str]], columns: set[str], path: Path) -> None:
    present = set(rows[0]) if rows else set()
    missing = columns.difference(present)
    if missing:
        raise ExperimentContractError(f"{path} missing required columns: {sorted(missing)}")


def validate_experiment_contract(cfg: dict[str, Any], output_dir: str | Path, result: dict[str, Any]) -> None:
    """Validate key semantic contracts after a runner finishes.

    Missing-input runs are allowed to exit without producing full result tables;
    completed runs must satisfy the checks for their experiment id.
    """

    output_dir = Path(output_dir)
    if _status_is_needs_input(output_dir, result):
        return

    exp_id = str(cfg.get("experiment_id", ""))

    if exp_id == "E6":
        rows = _require_nonempty(output_dir / "cross_session_results.csv")
        _require_columns(rows, {"subject", "direction", "accuracy", "kappa", "macro_f1"}, output_dir / "cross_session_results.csv")
        directions = {row.get("direction", "") for row in rows}
        if not {"T_to_E", "E_to_T"}.issubset(directions):
            raise ExperimentContractError("E6 must contain both T_to_E and E_to_T directions.")
        gap_rows = _require_nonempty(output_dir / "session_gap.csv")
        _require_columns(gap_rows, {"subject", "accuracy_gap_E_to_T_minus_T_to_E"}, output_dir / "session_gap.csv")

    elif exp_id == "E5":
        rows = _require_nonempty(output_dir / "main_bci2a_subject_dependent.csv")
        _require_columns(rows, {"subject", "accuracy", "kappa", "macro_f1", "evaluated_on_heldout_test"}, output_dir / "main_bci2a_subject_dependent.csv")
        if not (output_dir / "confusion_matrices.npy").exists():
            raise ExperimentContractError("E5 must export one aggregated confusion_matrices.npy artifact.")

    elif exp_id == "E8":
        rows = _require_nonempty(output_dir / "few_shot_results.csv")
        _require_columns(
            rows,
            {
                "subject",
                "model",
                "calibration_k",
                "calibration_draw_seed",
                "calibration_mode",
                "accuracy",
                "evaluated_on_target",
                "evaluated_on_heldout_test",
                "target_calibration_session",
                "target_evaluation_session",
                "calibration_evaluation_overlap",
                "physical_eeg_space",
            },
            output_dir / "few_shot_results.csv",
        )
        bad = [row for row in rows if str(row.get("evaluated_on_target", "")).lower() != "true"]
        if bad:
            raise ExperimentContractError("E8 rows must all be evaluated on target eval data.")
        if any(
            row.get("target_calibration_session") != "T"
            or row.get("target_evaluation_session") != "E"
            or str(row.get("calibration_evaluation_overlap", "")).lower() != "false"
            or str(row.get("evaluated_on_heldout_test", "")).lower() != "true"
            for row in rows
        ):
            raise ExperimentContractError("E8 requires target-session T calibration and untouched target-session E evaluation.")
        models = {row.get("model", "") for row in rows}
        if not {"dpc_snn", "eegnet"}.issubset(models):
            raise ExperimentContractError("E8 confirmatory results must include DPC-SNN and EEGNet.")
        for model in models:
            if not any(row.get("model") == model and row.get("calibration_mode") == "source_only" for row in rows):
                raise ExperimentContractError(f"E8 model {model} is missing source-only target-session E evaluation rows.")

    elif exp_id == "E9":
        rows = _require_nonempty(output_dir / "unlabeled_adaptation_results.csv")
        _require_columns(
            rows,
            {"subject", "adaptation_method", "accuracy", "target_unlabeled_session", "target_evaluation_session", "target_labels_used_for_split", "physical_eeg_space"},
            output_dir / "unlabeled_adaptation_results.csv",
        )
        if any(
            row.get("target_unlabeled_session") != "T"
            or row.get("target_evaluation_session") != "E"
            or str(row.get("target_labels_used_for_split", "")).lower() != "false"
            for row in rows
        ):
            raise ExperimentContractError("E9 requires target T unlabeled adaptation, target E evaluation, and no target-label split.")

    elif exp_id == "E11":
        rows = _require_nonempty(output_dir / "moabb_mini_results.csv")
        _require_columns(
            rows,
            {"dataset", "protocol", "subject", "session", "model", "status", "evaluated_on_heldout_test"},
            output_dir / "moabb_mini_results.csv",
        )
        failed = [row for row in rows if row.get("status") == "failed"]
        if failed:
            raise ExperimentContractError("E11 benchmark contains failed model evaluations.")
        completed = [row for row in rows if row.get("status", "completed") == "completed"]
        models = {row.get("model", "") for row in completed}
        if len(models) < 2:
            raise ExperimentContractError("E11 MOABB benchmark must contain at least two completed evaluated models.")
        if not all(_finite(row.get("accuracy")) and _finite(row.get("kappa")) and _finite(row.get("macro_f1")) for row in completed):
            raise ExperimentContractError("E11 completed model rows must contain finite accuracy, kappa, and macro_f1.")
        if any(row.get("protocol") != "moabb_within_session" for row in completed):
            raise ExperimentContractError("E11 completed rows must use the explicit moabb_within_session protocol.")
        if any(not row.get("session") or row.get("session") == "unknown" for row in completed):
            raise ExperimentContractError("E11 completed rows must retain a non-empty MOABB session identifier.")
        if any(str(row.get("evaluated_on_heldout_test", "")).lower() != "true" for row in completed):
            raise ExperimentContractError("E11 completed rows must be evaluated on held-out trials.")

    elif exp_id == "E13":
        rows = _require_nonempty(output_dir / "async_detection_results.csv")
        _require_columns(
            rows,
            {"protocol", "accuracy", "balanced_accuracy", "false_activation_per_min", "source", "evaluated_on_heldout_test"},
            output_dir / "async_detection_results.csv",
        )
        detector_rows = [row for row in rows if row.get("protocol") == "asynchronous_rest_vs_mi"]
        if not detector_rows:
            raise ExperimentContractError("E13 must include a rest-vs-MI detector row.")
        if any(row.get("source") != "real_async" for row in detector_rows):
            raise ExperimentContractError("E13 completed results require real continuous asynchronous data.")
        if any(str(row.get("evaluated_on_heldout_test", "")).lower() != "true" for row in detector_rows):
            raise ExperimentContractError("E13 detector metrics must use held-out recordings.")
        prediction_rows = _require_nonempty(output_dir / "sliding_window_predictions.csv")
        _require_columns(prediction_rows, {"event_uid", "prediction_split"}, output_dir / "sliding_window_predictions.csv")

    elif exp_id == "E14":
        rows = _require_nonempty(output_dir / "baseline_audit.csv")
        _require_columns(rows, {"model", "subject", "seed", "accuracy", "kappa", "macro_f1", "status", "hparam_trials_executed", "selected_hparams"}, output_dir / "baseline_audit.csv")
        completed = [row for row in rows if row.get("status") == "completed"]
        if len({row.get("model", "") for row in completed}) < 2:
            raise ExperimentContractError("E14 must evaluate at least two completed baseline/model rows.")
        if not all(_finite(row.get("accuracy")) and _finite(row.get("kappa")) and _finite(row.get("macro_f1")) for row in completed):
            raise ExperimentContractError("E14 completed baseline rows must contain finite accuracy, kappa, and macro_f1.")
        if any(str(row.get("evaluated_on_heldout_test", "")).lower() != "true" for row in completed):
            raise ExperimentContractError("E14 completed baseline rows must use held-out session-E evaluation.")
        trials = _require_nonempty(output_dir / "hparam_trials.csv")
        _require_columns(trials, {"subject", "model", "trial_id", "candidate", "selection_split", "evaluated_on_heldout_test", "status"}, output_dir / "hparam_trials.csv")
        if any(str(row.get("evaluated_on_heldout_test", "")).lower() != "false" for row in trials):
            raise ExperimentContractError("E14 hyperparameter trials must not evaluate held-out session-E data.")
        if any(str(row.get("heldout_test_accessed", "")).lower() not in {"", "false"} for row in trials):
            raise ExperimentContractError("E14 hyperparameter trials must not access or preprocess held-out session-E data.")
        if any(str(row.get("evaluation_split") or "") not in {"", "session_T_inner_validation"} for row in trials):
            raise ExperimentContractError("E14 hyperparameter trials must select only on session-T inner validation.")
        dpc_trials = [row for row in trials if row.get("model") == "dpc_snn" and row.get("status") == "completed"]
        if dpc_trials:
            first_subject = dpc_trials[0].get("subject")
            candidates = [json.loads(row["candidate"]) for row in dpc_trials if row.get("subject") == first_subject]
            if len(candidates) >= 9:
                learning_rates = {candidate.get("training", {}).get("lr") for candidate in candidates}
                weight_decays = {candidate.get("training", {}).get("weight_decay") for candidate in candidates}
                if len(learning_rates) < 3 or len(weight_decays) < 3:
                    raise ExperimentContractError("E14 DPC-SNN search must cover all learning-rate and weight-decay levels.")
            if len(candidates) >= 20:
                architectures = {(candidate.get("hidden_channels"), candidate.get("d_max")) for candidate in candidates}
                if len(architectures) < 20:
                    raise ExperimentContractError("E14 DPC-SNN budget-20 search must cover all 20 architecture combinations.")

    elif exp_id == "E17":
        _require_nonempty(output_dir / "band_ablation.csv")

    elif exp_id == "E18":
        _require_nonempty(output_dir / "erd_ers_topomap.csv")
        if not (output_dir / "edge_importance.npy").exists() or not (output_dir / "delay_matrix.npy").exists():
            raise ExperimentContractError("E18 must export learned edge_importance.npy and delay_matrix.npy artifacts.")
        if not (output_dir / "model_interpretation_manifest.json").exists():
            raise ExperimentContractError("E18 must record the reproducible source checkpoints and baseline interval.")

    elif exp_id == "E19":
        rows = _require_nonempty(output_dir / "occlusion_results.csv")
        _require_columns(rows, {"dataset", "subject", "seed", "occlusion", "occlusion_drop", "evaluated_on_heldout_test", "occlusion_protocol"}, output_dir / "occlusion_results.csv")
        if any(row.get("dataset") != "bci2a" or str(row.get("evaluated_on_heldout_test", "")).lower() != "true" for row in rows):
            raise ExperimentContractError("E19 requires real BCI2a held-out-test occlusion rows.")
        stability = _require_nonempty(output_dir / "edge_stability.csv")
        _require_columns(stability, {"subject", "seed_i", "seed_j", "topk_jaccard", "stability_scope"}, output_dir / "edge_stability.csv")

    elif exp_id == "E20":
        rows = _require_nonempty(output_dir / "reference_sensitivity.csv")
        _require_columns(rows, {"dataset", "subject", "reference", "accuracy", "evaluated_on_heldout_test"}, output_dir / "reference_sensitivity.csv")
        if any(row.get("dataset") != "bci2a" or str(row.get("evaluated_on_heldout_test", "")).lower() != "true" for row in rows):
            raise ExperimentContractError("E20 requires subject-wise real BCI2a held-out-test rows.")
        connectivity = _require_nonempty(output_dir / "volume_conduction_checks.csv")
        _require_columns(connectivity, {"subject", "reference", "dwpli_corr", "connectivity_data"}, output_dir / "volume_conduction_checks.csv")

    elif exp_id == "E21":
        rows = _require_nonempty(output_dir / "decision_auc.csv")
        _require_columns(rows, {"checkpoint", "subject", "source_experiment", "evaluation_split", "feature_protocol", "decision_auc"}, output_dir / "decision_auc.csv")
        if any(row.get("source_experiment") != "E14" or row.get("feature_protocol") != "prefix_only_recomputed_fft_hilbert" for row in rows):
            raise ExperimentContractError("E21 must use E14 DPC-SNN checkpoints and prefix-only reconstructed features.")

    elif exp_id == "E22":
        rows = _require_nonempty(output_dir / "artifact_robustness.csv")
        _require_columns(rows, {"artifact", "accuracy", "kappa", "macro_f1"}, output_dir / "artifact_robustness.csv")
        if not all(_finite(row.get("accuracy")) for row in rows):
            raise ExperimentContractError("E22 artifact robustness rows must contain finite accuracy.")

    elif exp_id == "E23":
        rows = _require_nonempty(output_dir / "low_channel_results.csv")
        _require_columns(rows, {"dataset", "subject", "channel_subset", "channels", "accuracy", "evaluated_on_heldout_test"}, output_dir / "low_channel_results.csv")
        if any(row.get("dataset") != "bci2a" or str(row.get("evaluated_on_heldout_test", "")).lower() != "true" for row in rows):
            raise ExperimentContractError("E23 requires real BCI2a subject-wise held-out-test rows.")

    elif exp_id == "E24":
        rows = _require_nonempty(output_dir / "stat_tests.csv")
        _require_columns(
            rows,
            {"experiment", "dataset", "protocol", "comparison_type", "model_a", "model_b", "metric", "n", "p_value", "p_fdr"},
            output_dir / "stat_tests.csv",
        )
        if any(not row.get("experiment") or not row.get("dataset") or not row.get("protocol") for row in rows):
            raise ExperimentContractError("E24 statistical comparisons must retain experiment, dataset, and protocol strata.")
