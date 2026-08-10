from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.contracts import ExperimentContractError, validate_experiment_contract
from dpc_snn.utils.io import write_csv, write_json


def test_e6_contract_requires_bidirectional_cross_session(tmp_path: Path) -> None:
    write_csv(
        tmp_path / "cross_session_results.csv",
        [{"subject": "A01", "direction": "T_to_E", "accuracy": 0.5, "kappa": 0.0, "macro_f1": 0.4}],
    )
    write_csv(tmp_path / "session_gap.csv", [{"subject": "A01", "accuracy_gap_E_to_T_minus_T_to_E": 0.0}])

    with pytest.raises(ExperimentContractError, match="both T_to_E and E_to_T"):
        validate_experiment_contract({"experiment_id": "E6"}, tmp_path, {})


def test_e5_contract_requires_aggregated_confusion_matrices(tmp_path: Path) -> None:
    write_csv(
        tmp_path / "main_bci2a_subject_dependent.csv",
        [{"subject": "A01", "accuracy": 0.5, "kappa": 0.0, "macro_f1": 0.4, "evaluated_on_heldout_test": True}],
    )
    with pytest.raises(ExperimentContractError, match="confusion_matrices"):
        validate_experiment_contract({"experiment_id": "E5"}, tmp_path, {})
    np.save(tmp_path / "confusion_matrices.npy", np.zeros((1, 2, 2), dtype=np.int64))
    validate_experiment_contract({"experiment_id": "E5"}, tmp_path, {})


def test_e8_contract_rejects_source_validation_metrics(tmp_path: Path) -> None:
    write_csv(
        tmp_path / "few_shot_results.csv",
        [
            {
                "subject": "A01",
                "model": model,
                "calibration_k": 1,
                "calibration_draw_seed": 0,
                "calibration_mode": "source_only",
                "accuracy": 0.5,
                "evaluated_on_target": "false",
                "evaluated_on_heldout_test": True,
                "target_calibration_session": "T",
                "target_evaluation_session": "E",
                "calibration_evaluation_overlap": False,
                "physical_eeg_space": "spherical_spline_csd_before_physical_projection",
            }
            for model in ["dpc_snn", "eegnet"]
        ],
    )

    with pytest.raises(ExperimentContractError, match="evaluated on target"):
        validate_experiment_contract({"experiment_id": "E8"}, tmp_path, {})


def test_e8_contract_rejects_pooled_target_sessions(tmp_path: Path) -> None:
    write_csv(
        tmp_path / "few_shot_results.csv",
        [
            {
                "subject": "A01",
                "model": model,
                "calibration_k": 1,
                "calibration_draw_seed": 0,
                "calibration_mode": "source_only",
                "accuracy": 0.5,
                "evaluated_on_target": True,
                "evaluated_on_heldout_test": True,
                "target_calibration_session": "T+E",
                "target_evaluation_session": "T+E",
                "calibration_evaluation_overlap": True,
                "physical_eeg_space": "spherical_spline_csd_before_physical_projection",
            }
            for model in ["dpc_snn", "eegnet"]
        ],
    )

    with pytest.raises(ExperimentContractError, match="target-session T calibration"):
        validate_experiment_contract({"experiment_id": "E8"}, tmp_path, {})


def test_e14_contract_requires_completed_evaluated_baselines(tmp_path: Path) -> None:
    write_csv(
        tmp_path / "baseline_audit.csv",
        [
            {"model": "dpc_snn", "subject": "A01", "seed": 0, "accuracy": 0.5, "kappa": 0.0, "macro_f1": 0.4, "status": "completed", "hparam_trials_executed": 1, "selected_hparams": "{}", "evaluated_on_heldout_test": True},
            {"model": "eegnet", "subject": "A01", "seed": 0, "accuracy": 0.45, "kappa": 0.0, "macro_f1": 0.3, "status": "completed", "hparam_trials_executed": 1, "selected_hparams": "{}", "evaluated_on_heldout_test": True},
        ],
    )
    write_csv(
        tmp_path / "hparam_trials.csv",
        [{"subject": "A01", "model": "dpc_snn", "trial_id": 0, "candidate": "{}", "selection_split": "inner_validation_on_session_T", "evaluated_on_heldout_test": False, "status": "completed"}],
    )

    validate_experiment_contract({"experiment_id": "E14"}, tmp_path, {})


def test_e14_contract_rejects_hyperparameter_search_that_touched_session_e(
    tmp_path: Path,
) -> None:
    write_csv(
        tmp_path / "baseline_audit.csv",
        [
            {"model": model, "subject": "A01", "seed": 0, "accuracy": 0.5, "kappa": 0.0, "macro_f1": 0.4, "status": "completed", "hparam_trials_executed": 1, "selected_hparams": "{}", "evaluated_on_heldout_test": True}
            for model in ("dpc_snn", "eegnet")
        ],
    )
    write_csv(
        tmp_path / "hparam_trials.csv",
        [{"subject": "A01", "model": "dpc_snn", "trial_id": 0, "candidate": "{}", "selection_split": "inner_validation_on_session_T", "evaluation_split": "heldout_session_E", "heldout_test_accessed": True, "evaluated_on_heldout_test": False, "status": "completed"}],
    )

    with pytest.raises(ExperimentContractError, match="must not access or preprocess"):
        validate_experiment_contract({"experiment_id": "E14"}, tmp_path, {})


def test_needs_input_skips_strict_contract(tmp_path: Path) -> None:
    write_json(tmp_path / "runner_status.json", {"status": "needs_input", "message": "missing real async data"})

    validate_experiment_contract({"experiment_id": "E13"}, tmp_path, {"status": "needs_input"})


def test_run_manifest_needs_input_skips_strict_contract(tmp_path: Path) -> None:
    write_json(tmp_path / "run_manifest.json", {"status": "needs_input", "message": "no valid rows"})
    write_csv(tmp_path / "moabb_mini_results.csv", [])

    validate_experiment_contract({"experiment_id": "E11"}, tmp_path, {})
