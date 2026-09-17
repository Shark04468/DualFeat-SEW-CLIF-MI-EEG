from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dpc_snn.data.async_windows import AsyncWindowConfig, make_windows_from_labeled_intervals  # noqa: E402
from dpc_snn.experiments.contracts import ExperimentContractError, validate_experiment_contract  # noqa: E402
from dpc_snn.experiments.common import (  # noqa: E402
    fold_prior_cache_fingerprint,
    fold_prior_scientific_gate_failures,
)
from dpc_snn.experiments.runners import (
    _average_rank_rows,
    _benchmark_model_names,
    _common_channel_alignment,
    _crop_to_common_task_window,
    _load_async_window_npz,
    _split_protocol_train_validation,
    _split_protocol_train_validation_test,
    _target_session_unlabeled_eval_indices,
    run_final_statistics,
)  # noqa: E402
from dpc_snn.plots.figures import reproduce_figures  # noqa: E402
from dpc_snn.preprocessing.filters import fft_bandpass  # noqa: E402
from dpc_snn.utils.io import write_csv, write_json  # noqa: E402
from audit_v42_fold_evidence import (  # noqa: E402
    _audit_space,
    _inject_task_delay,
    _inject_task_delay_latent,
    _known_route_metrics,
)
from run_v40_delay_gate import _load_completed  # noqa: E402


def test_filter_rejects_invalid_sampling_rate_instead_of_dividing_by_zero() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        fft_bandpass(np.ones((2, 16), dtype=np.float32), 0.0, 8.0, 13.0)


def test_moabb_pipeline_selection_does_not_expand_to_unconfigured_baselines() -> None:
    models = _benchmark_model_names(
        {"model": {"baselines_config": "configs/models/baselines.yaml"}},
        configured_models=["dpc_snn", "eegnet", "csp_lda", "riemann_lr"],
    )

    assert models == ["dpc_snn", "eegnet", "csp_lda", "riemann_lr"]


def test_channel_alignment_normalizes_standard_eeg_display_variants() -> None:
    source = {"X": np.zeros((2, 3, 8), dtype=np.float32), "ch_names": ["EEG-Fz", "FCz", "C3"]}
    target = {"X": np.zeros((2, 3, 8), dtype=np.float32), "ch_names": ["FZ.", "FCZ", "C3"]}

    aligned_source, aligned_target, audit = _common_channel_alignment(source, target)

    assert audit["status"] == "aligned_by_name"
    assert audit["common_channels"] == 3
    assert aligned_source["X"].shape[1] == aligned_target["X"].shape[1] == 3


def test_real_async_windows_exclude_transition_crossing_windows() -> None:
    x = np.zeros((2, 40), dtype=np.float32)
    intervals = np.asarray([[0, 10, -1], [10, 20, 0], [20, 30, -1], [30, 40, 1]], dtype=np.int64)

    windows = make_windows_from_labeled_intervals(
        x,
        intervals,
        AsyncWindowConfig(sfreq=10.0, window_sec=1.0, step_sec=0.5),
    )

    assert windows["X"].shape[0] == 4
    assert windows["y_binary"].tolist() == [0, 1, 0, 1]
    assert np.isnan(windows["onset_latency_sec"][[0, 2]]).all()


def test_async_loader_namespaces_repeated_local_event_ids_by_recording(tmp_path: Path) -> None:
    for recording in ["S001_R04", "S001_R08"]:
        np.savez_compressed(
            tmp_path / f"{recording}.npz",
            X=np.zeros((2, 2, 10), dtype=np.float32),
            y_binary=np.asarray([1, 1], dtype=np.int64),
            y_mi=np.asarray([0, 0], dtype=np.int64),
            window_start=np.asarray([0, 5], dtype=np.int64),
            event_id=np.asarray([0, 0], dtype=np.int64),
            event_start=np.asarray([0, 0], dtype=np.int64),
            subject=np.asarray(["S001", "S001"]),
            recording_id=np.asarray([recording, recording]),
        )

    loaded = _load_async_window_npz(tmp_path)

    assert set(loaded["event_uid"]) == {"S001_R04:0", "S001_R08:0"}


def test_e9_target_partition_uses_session_not_target_labels() -> None:
    first = {"session": np.asarray(["E", "T", "E", "T"]), "y": np.asarray([3, 2, 1, 0])}
    second = {"session": np.asarray(["E", "T", "E", "T"]), "y": np.asarray([0, 1, 2, 3])}

    first_unlabeled, first_evaluation = _target_session_unlabeled_eval_indices(first)
    second_unlabeled, second_evaluation = _target_session_unlabeled_eval_indices(second)

    np.testing.assert_array_equal(first_unlabeled, second_unlabeled)
    np.testing.assert_array_equal(first_evaluation, second_evaluation)
    np.testing.assert_array_equal(first_unlabeled, np.asarray([1, 3]))
    np.testing.assert_array_equal(first_evaluation, np.asarray([0, 2]))


def test_protocol_split_retains_training_standardization_statistics() -> None:
    raw_train = {
        "X": np.arange(8 * 2 * 16, dtype=np.float32).reshape(8, 2, 16),
        "y": np.asarray([0, 1] * 4),
        "sfreq": 250.0,
    }
    raw_test = {
        "X": np.ones((4, 2, 16), dtype=np.float32),
        "y": np.asarray([0, 1, 0, 1]),
        "sfreq": 250.0,
    }

    train, _, _, _ = _split_protocol_train_validation_test(raw_train, raw_test, {"seed": 0})

    assert train["standardize_mean"].shape == (1, 2, 1)
    assert train["standardize_std"].shape == (1, 2, 1)


def test_shared_protocol_split_is_session_t_only() -> None:
    raw_train = {
        "X": np.arange(8 * 2 * 16, dtype=np.float32).reshape(8, 2, 16),
        "y": np.asarray([0, 1] * 4),
        "session": np.asarray(["T"] * 8),
        "sfreq": 250.0,
    }

    train, validation, audit = _split_protocol_train_validation(raw_train, {"seed": 0})

    assert set(np.asarray(train["session"]).tolist()) == {"T"}
    assert set(np.asarray(validation["session"]).tolist()) == {"T"}
    assert audit["heldout_test_accessed"] is False
    assert audit["evaluation_split"] == "session_T_inner_validation"


def test_common_task_window_preserves_declared_pre_cue_baseline() -> None:
    data = {
        "X": np.zeros((4, 2, 1250), dtype=np.float32),
        "y": np.asarray([0, 1, 2, 3]),
        "sfreq": 250.0,
        "epoch_tmin": -1.0,
        "epoch_tmax": 4.0,
    }
    cfg = {"model": {"dpc_snn_config": "configs/models/dpc_snn.yaml"}}

    cropped = _crop_to_common_task_window(data, cfg)

    assert cropped["X"].shape[-1] == 1250
    assert cropped["epoch_tmin"] == -1.0
    assert cropped["epoch_tmax"] == 4.0


def test_fold_prior_cache_fingerprint_changes_with_data_and_evidence_config() -> None:
    train = {
        "X": np.zeros((4, 2, 16), dtype=np.float32),
        "y": np.asarray([0, 1, 0, 1]),
        "sfreq": 250.0,
    }
    model_cfg = {
        "band_edges_hz": [[8.0, 13.0]],
        "delay_evidence_band_edges_hz": [[8.0, 10.0], [10.0, 13.0]],
        "latent_nodes": 2,
        "graph_timesteps": 16,
        "d_max": 2,
    }
    original = fold_prior_cache_fingerprint(train, model_cfg, seed=0)
    changed_data = {**train, "X": train["X"].copy()}
    changed_data["X"][0, 0, 0] = 1.0
    changed_config = {**model_cfg, "d_max": 3}
    changed_algorithm = {**model_cfg, "fold_prior_algorithm_fingerprint": "new-code"}

    assert fold_prior_cache_fingerprint(changed_data, model_cfg, seed=0) != original
    assert fold_prior_cache_fingerprint(train, changed_config, seed=0) != original
    assert fold_prior_cache_fingerprint(train, model_cfg, seed=1) != original
    assert fold_prior_cache_fingerprint(train, changed_algorithm, seed=0) != original
    changed_threshold = {
        **model_cfg,
        "continuous_delay_min_bayes_factor": 5.0,
    }
    assert fold_prior_cache_fingerprint(train, changed_threshold, seed=0) != original


def test_fold_prior_fingerprint_uses_csd_evidence_not_carrier_when_supplied() -> None:
    first = {
        "X": np.zeros((4, 2, 8), dtype=np.float32),
        "delay_evidence_X": np.ones((4, 2, 8), dtype=np.float32),
        "y": np.asarray([0, 1, 0, 1]),
        "sfreq": 32.0,
    }
    second = {**first, "X": np.full((4, 2, 8), 9.0, dtype=np.float32)}
    model_cfg = {
        "band_edges_hz": [[2.0, 4.0]],
        "delay_evidence_band_edges_hz": [[2.0, 4.0]],
        "classification_band_edges_hz": [[2.0, 4.0]],
        "delay_evidence_reference": "csd",
    }
    assert fold_prior_cache_fingerprint(first, model_cfg, 0) == fold_prior_cache_fingerprint(
        second, model_cfg, 0
    )
    changed_evidence = {
        **second,
        "delay_evidence_X": np.full((4, 2, 8), 2.0, dtype=np.float32),
    }
    assert fold_prior_cache_fingerprint(first, model_cfg, 0) != fold_prior_cache_fingerprint(
        changed_evidence, model_cfg, 0
    )


def test_atomic_json_write_preserves_previous_file_on_serialization_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "status.json"
    write_json(path, {"status": "valid"})

    with pytest.raises(TypeError):
        write_json(path, {"invalid": object()})

    assert path.read_text(encoding="utf-8").strip() == '{\n  "status": "valid"\n}'


def test_v42_fold_prior_gate_stops_null_or_surrogate_invariant_evidence() -> None:
    summary = {
        "accepted_evidence_edges": 0,
        "normal_node_edges": 0,
        "phase_surrogate_node_edges": 0,
        "phase_surrogate_support_drop": -0.007,
        "time_reversal_transpose_support_mae": 1e-7,
    }
    failures = fold_prior_scientific_gate_failures(summary, {})

    assert "no_stable_delay_edges" in failures
    assert "phase_surrogate_did_not_reduce_edges" in failures
    assert "phase_surrogate_did_not_reduce_support" in failures
    assert "time_reversal_did_not_transpose_direction" not in failures


def test_v42_evidence_audit_injection_is_causal_and_leaves_baseline_untouched() -> None:
    x = np.zeros((1, 2, 50), dtype=np.float32)
    x[0, 0, 10:] = np.arange(40, dtype=np.float32)
    data = {"X": x, "sfreq": 10.0, "epoch_tmin": -1.0}

    injected = _inject_task_delay(
        data,
        source=0,
        target=1,
        delay_graph_steps=0.5,
        strength=1.0,
        graph_steps=40,
        task_tmin=0.0,
        task_tmax=4.0,
    )

    np.testing.assert_array_equal(injected["X"][0, 1, :10], 0.0)
    np.testing.assert_array_equal(injected["X"][0, 0], data["X"][0, 0])
    assert injected["X"][0, 1, 10] == pytest.approx(0.0)
    assert injected["X"][0, 1, 11] == pytest.approx(0.5)
    np.testing.assert_array_equal(data["X"][0, 1], 0.0)


def test_v42_evidence_audit_scores_the_known_direction_and_fractional_delay() -> None:
    route = np.zeros((2, 2, 2, 2), dtype=np.float32)
    route[0, 0, 1, 0] = 0.8
    positive = np.zeros((2, 2, 2, 2, 4), dtype=np.float32)
    positive[..., 0] = 1.0
    positive[0, 0, 1, 0] = np.asarray([0.0, 0.0, 1.0, 0.0])
    fraction = np.zeros_like(route)
    fraction[0, 0, 1, 0] = 0.5

    metrics = _known_route_metrics(
        {
            "route_probability": route,
            "positive_delay_probability": positive,
            "fractional_delay_target": fraction,
        },
        source_node=0,
        target_node=1,
        true_delay=2.5,
    )

    assert metrics["known_direction_detected"] is True
    assert metrics["known_forward_routes"] == 1
    assert metrics["known_reverse_routes"] == 0
    assert metrics["known_delay_recovered"] == pytest.approx(2.5)
    assert metrics["known_delay_absolute_error"] == pytest.approx(0.0)


def test_v42_known_delay_uses_declared_node_map_not_transport_mean() -> None:
    route = np.zeros((2, 2, 2, 2), dtype=np.float32)
    route[0, 0, 1, 0] = 0.8
    positive = np.zeros((2, 2, 2, 2, 5), dtype=np.float32)
    positive[..., 0] = 1.0
    positive[0, 0, 1, 0] = np.asarray([0.0, 0.0, 0.5, 0.0, 0.5])
    fraction = np.zeros_like(route)
    metrics = _known_route_metrics(
        {
            "route_probability": route,
            "positive_delay_probability": positive,
            "fractional_delay_target": fraction,
            "signed_node_delay_map": np.asarray([[0.0, 0.0], [2.0, 0.0]]),
            "signed_node_delay_mean": np.asarray([[0.0, 0.0], [3.0, 0.0]]),
            "signed_node_accepted": np.asarray([[0, 0], [1, 0]], dtype=np.uint8),
        },
        source_node=0,
        target_node=1,
        true_delay=2.0,
    )

    assert metrics["known_delay_estimator"] == "signed_node_posterior_map"
    assert metrics["known_delay_recovered"] == pytest.approx(2.0)
    assert metrics["known_delay_absolute_error"] == pytest.approx(0.0)
    assert metrics["known_transport_delay_posterior_mean"] == pytest.approx(3.0)
    assert metrics["known_node_delay_posterior_mean"] == pytest.approx(3.0)


def test_v42_split_half_delay_uses_node_graph_not_exact_band_pair_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import audit_v42_fold_evidence as audit_module

    route_first = np.zeros((2, 2, 3, 3), dtype=np.float32)
    route_second = np.zeros_like(route_first)
    node_active = np.zeros((3, 3), dtype=np.uint8)
    node_delay = np.zeros((3, 3), dtype=np.float32)
    for target, source, delay in ((1, 0, 1.0), (2, 0, 2.0), (2, 1, 3.0)):
        route_first[0, 0, target, source] = 1.0
        route_second[1, 1, target, source] = 1.0
        node_active[target, source] = 1
        node_delay[target, source] = delay
    positive = np.zeros((2, 2, 3, 3, 4), dtype=np.float32)
    positive[..., 0] = 1.0
    fraction = np.zeros((2, 2, 3, 3), dtype=np.float32)

    def arrays(route: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "route_probability": route,
            "positive_delay_probability": positive,
            "fractional_delay_target": fraction,
            "signed_node_accepted": node_active,
            "signed_node_delay_map": node_delay,
        }

    summary = {
        "accepted_evidence_edges": 3,
        "time_reversal_transpose_support_mae": 0.0,
        "phase_surrogate_support_drop": 0.1,
        "normal_node_edges": 3,
        "phase_surrogate_node_edges": 0,
    }
    calls = iter(
        [
            (arrays(route_first), summary),
            (arrays(route_first), summary),
            (arrays(route_second), summary),
        ]
    )
    monkeypatch.setattr(audit_module, "_fit_prior", lambda *args, **kwargs: next(calls))

    report, _ = _audit_space(
        "car",
        {"X": np.zeros((6, 2, 8), dtype=np.float32), "y": np.arange(6) % 2},
        {},
        seed=1,
        device="cpu",
        bootstraps=8,
    )

    assert report["split_half_delay_scope"] == "signed_node_posterior_map"
    assert report["split_half_common_node_edges"] == 3
    assert report["split_half_common_transport_routes"] == 0
    assert report["split_half_delay_correlation"] == pytest.approx(1.0)


def test_v42_latent_injection_produces_the_exact_projected_route() -> None:
    x = np.zeros((1, 2, 50), dtype=np.float32)
    x[0, 0, 10:] = np.arange(40, dtype=np.float32)
    data = {"X": x, "sfreq": 10.0, "epoch_tmin": -1.0}
    anchors = np.eye(2, dtype=np.float32)

    injected = _inject_task_delay_latent(
        data,
        anchors=anchors,
        source_node=0,
        target_node=1,
        delay_graph_steps=0.5,
        strength=1.0,
        graph_steps=40,
        task_tmin=0.0,
        task_tmax=4.0,
    )
    projected = np.einsum("kc,nct->nkt", anchors, injected["X"])

    np.testing.assert_array_equal(projected[0, 1, :10], 0.0)
    assert projected[0, 1, 10] == pytest.approx(0.0)
    assert projected[0, 1, 11] == pytest.approx(0.5)
    np.testing.assert_array_equal(projected[0, 0], data["X"][0, 0])


def test_v42_evidence_audit_is_session_t_only_and_classifier_free() -> None:
    source = (ROOT / "scripts" / "audit_v42_fold_evidence.py").read_text(encoding="utf-8")

    assert 'session="T"' in source
    assert 'session="E"' not in source
    assert "train_model" not in source


def test_v42_resume_rejects_any_stale_scientific_contract_input(tmp_path: Path) -> None:
    contract = {
        "source_fingerprint": "source-a",
        "model_config_sha256": "config-a",
        "resolved_run_config_sha256": "run-a",
        "initial_checkpoint_sha256": "checkpoint-a",
        "fold_prior_sha256": "prior-a",
        "environment": {"torch": "2.x", "cuda_device_name": "gpu-a"},
        "environment_manifest_sha256": "environment-a",
        "split": "repeat-0-fold-0",
    }
    write_json(tmp_path / "resume_contract.json", contract)
    write_json(tmp_path / "training_status.json", {"status": "completed"})
    write_json(tmp_path / "metrics.json", {"accuracy": 0.5})
    np.save(tmp_path / "selection_y_true.npy", np.asarray([0, 1]))
    np.save(tmp_path / "selection_y_pred.npy", np.asarray([0, 1]))
    np.save(tmp_path / "selection_logits.npy", np.zeros((2, 2), dtype=np.float32))

    assert _load_completed(tmp_path, contract) is not None
    assert _load_completed(tmp_path, {**contract, "source_fingerprint": "source-b"}) is None
    assert _load_completed(tmp_path, {**contract, "model_config_sha256": "config-b"}) is None
    for key in (
        "resolved_run_config_sha256",
        "initial_checkpoint_sha256",
        "fold_prior_sha256",
        "environment_manifest_sha256",
    ):
        assert _load_completed(tmp_path, {**contract, key: f"stale-{key}"}) is None
    assert (
        _load_completed(
            tmp_path,
            {**contract, "environment": {"torch": "2.x", "cuda_device_name": "gpu-b"}},
        )
        is None
    )


def test_figure_reproduction_uses_only_declared_results_root(tmp_path: Path) -> None:
    results = tmp_path / "results"
    external = tmp_path / "external_runs"
    write_csv(results / "E1" / "metrics.csv", [{"accuracy": 0.5}])
    write_csv(external / "E99" / "metrics.csv", [{"accuracy": 1.0}])

    manifest = reproduce_figures(results, tmp_path / "figures")

    text = manifest.read_text(encoding="utf-8")
    assert str((results / "E1" / "metrics.csv").resolve()) in text
    assert str((external / "E99" / "metrics.csv").resolve()) not in text


def test_e11_contract_rejects_failed_model_rows(tmp_path: Path) -> None:
    write_csv(
        tmp_path / "moabb_mini_results.csv",
        [
            {
                "dataset": "d",
                "protocol": "moabb_within_session",
                "subject": "s",
                "session": "run_1",
                "model": "dpc_snn",
                "status": "completed",
                "accuracy": 0.5,
                "kappa": 0.0,
                "macro_f1": 0.5,
                "evaluated_on_heldout_test": True,
            },
            {
                "dataset": "d",
                "protocol": "moabb_within_session",
                "subject": "s",
                "session": "run_1",
                "model": "eegnet",
                "status": "failed",
                "error": "runtime",
                "evaluated_on_heldout_test": True,
            },
        ],
    )

    with pytest.raises(ExperimentContractError, match="failed model"):
        validate_experiment_contract({"experiment_id": "E11"}, tmp_path, {})


def test_e11_average_rank_is_computed_within_subject_session_units() -> None:
    rows = [
        {
            "dataset": "d",
            "protocol": "moabb_within_session",
            "subject": "s1",
            "session": "a",
            "model": "dpc_snn",
            "status": "completed",
            "accuracy": 0.9,
        },
        {
            "dataset": "d",
            "protocol": "moabb_within_session",
            "subject": "s1",
            "session": "a",
            "model": "eegnet",
            "status": "completed",
            "accuracy": 0.8,
        },
        {
            "dataset": "d",
            "protocol": "moabb_within_session",
            "subject": "s1",
            "session": "b",
            "model": "dpc_snn",
            "status": "completed",
            "accuracy": 0.1,
        },
        {
            "dataset": "d",
            "protocol": "moabb_within_session",
            "subject": "s1",
            "session": "b",
            "model": "eegnet",
            "status": "completed",
            "accuracy": 0.2,
        },
    ]

    ranks = {row["model"]: row for row in _average_rank_rows(rows)}

    assert ranks["dpc_snn"]["average_rank"] == 1.5
    assert ranks["eegnet"]["average_rank"] == 1.5
    assert ranks["dpc_snn"]["n_evaluation_units"] == 2


def test_e24_contract_requires_stratified_statistical_fields(tmp_path: Path) -> None:
    write_csv(
        tmp_path / "stat_tests.csv",
        [
            {
                "experiment": "E14",
                "dataset": "bci2a",
                "protocol": "baseline_fairness",
                "comparison_type": "primary_model",
                "model_a": "dpc_snn",
                "model_b": "eegnet",
                "metric": "accuracy",
                "n": 9,
                "p_value": 0.1,
                "p_fdr": 0.1,
            }
        ],
    )

    validate_experiment_contract({"experiment_id": "E24"}, tmp_path, {})


def test_e21_contract_rejects_full_trial_feature_slicing(tmp_path: Path) -> None:
    write_csv(
        tmp_path / "decision_auc.csv",
        [
            {
                "checkpoint": "x.pt",
                "subject": "A01",
                "source_experiment": "E14",
                "evaluation_split": "heldout_session_E",
                "feature_protocol": "full_trial_features_then_sliced",
                "decision_auc": 0.5,
            }
        ],
    )

    with pytest.raises(ExperimentContractError, match="prefix-only"):
        validate_experiment_contract({"experiment_id": "E21"}, tmp_path, {})


def test_final_statistics_keeps_experiments_and_protocols_separate(tmp_path: Path) -> None:
    e11 = tmp_path / "E11"
    e14 = tmp_path / "E14"
    write_csv(
        e11 / "moabb_mini_results.csv",
        [
            {
                "dataset": "moabb_a",
                "protocol": "moabb_within_session",
                "subject": "s1",
                "session": "a",
                "model": "dpc_snn",
                "status": "completed",
                "accuracy": 0.4,
            },
            {
                "dataset": "moabb_a",
                "protocol": "moabb_within_session",
                "subject": "s1",
                "session": "a",
                "model": "eegnet",
                "status": "completed",
                "accuracy": 0.6,
            },
            {
                "dataset": "moabb_a",
                "protocol": "moabb_within_session",
                "subject": "s2",
                "session": "a",
                "model": "dpc_snn",
                "status": "completed",
                "accuracy": 0.5,
            },
            {
                "dataset": "moabb_a",
                "protocol": "moabb_within_session",
                "subject": "s2",
                "session": "a",
                "model": "eegnet",
                "status": "completed",
                "accuracy": 0.7,
            },
        ],
    )
    write_csv(
        e14 / "baseline_audit.csv",
        [
            {
                "dataset": "bci2a",
                "protocol": "baseline_fairness",
                "subject": "s1",
                "model": "dpc_snn",
                "status": "completed",
                "accuracy": 0.3,
            },
            {
                "dataset": "bci2a",
                "protocol": "baseline_fairness",
                "subject": "s1",
                "model": "eegnet",
                "status": "completed",
                "accuracy": 0.5,
            },
            {
                "dataset": "bci2a",
                "protocol": "baseline_fairness",
                "subject": "s2",
                "model": "dpc_snn",
                "status": "completed",
                "accuracy": 0.4,
            },
            {
                "dataset": "bci2a",
                "protocol": "baseline_fairness",
                "subject": "s2",
                "model": "eegnet",
                "status": "completed",
                "accuracy": 0.6,
            },
        ],
    )

    output = tmp_path / "E24"
    run_final_statistics({"statistics_results_root": str(tmp_path)}, output)

    rows = (output / "stat_tests.csv").read_text(encoding="utf-8").splitlines()
    assert any("moabb_a" in row and "E11" in row for row in rows)
    assert any("bci2a" in row and "E14" in row for row in rows)


def test_final_statistics_keeps_calibration_k_separate(tmp_path: Path) -> None:
    e8 = tmp_path / "E8"
    write_csv(
        e8 / "few_shot_results.csv",
        [
            {
                "dataset": "bci2a",
                "protocol": "few_shot",
                "subject": subject,
                "status": "completed",
                "calibration_k": k,
                "calibration_mode": mode,
                "accuracy": accuracy,
            }
            for subject in ["s1", "s2"]
            for k, source_accuracy, full_accuracy in [("1", 0.2, 0.8), ("5", 0.7, 0.4)]
            for mode, accuracy in [("source_only", source_accuracy), ("full", full_accuracy)]
        ],
    )

    output = tmp_path / "E24"
    run_final_statistics({"statistics_results_root": str(tmp_path)}, output)

    rows = [
        row
        for row in csv.DictReader((output / "stat_tests.csv").open(encoding="utf-8"))
        if row["experiment"] == "E8"
    ]
    assert {row["calibration_k"] for row in rows} == {"1", "5"}
    assert len(rows) == 2
