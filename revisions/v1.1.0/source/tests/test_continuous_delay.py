from __future__ import annotations

import numpy as np
import pytest

from dpc_snn.analysis.continuous_delay import (
    ContinuousDelayConfig,
    fit_continuous_delay_evidence,
    trial_continuous_phase_scores,
)
from dpc_snn.analysis.evidence_space import (
    complex_phase_surrogate,
    continuous_phase_surrogate_bank,
    fold_local_continuous_delay_prior,
    stratified_split_half_indices,
)


def test_stratified_split_half_preserves_every_run_class_cell() -> None:
    strata = ["r0-c0"] * 4 + ["r0-c1"] * 4 + ["r1-c0"] * 4
    first, second = stratified_split_half_indices(len(strata), strata)
    assert set(first).isdisjoint(second)
    assert sorted(np.concatenate((first, second)).tolist()) == list(range(len(strata)))
    for value in sorted(set(strata)):
        assert sum(strata[index] == value for index in first) == 2
        assert sum(strata[index] == value for index in second) == 2


def test_stratified_split_half_rejects_singleton_cells() -> None:
    with pytest.raises(ValueError, match="at least two"):
        stratified_split_half_indices(4, ["a", "a", "b", "c"])


def test_continuous_surrogate_bank_is_reproducible_and_retains_requested_draws() -> None:
    analytic, frequencies, timestep = _analytic_delay_trials(0.5, trials=4)
    config = ContinuousDelayConfig(
        max_delay_steps=2,
        grid_oversample=2,
        bootstrap_samples=4,
        random_seed=5,
    )
    first = continuous_phase_surrogate_bank(
        analytic,
        frequencies,
        timestep_seconds=timestep,
        config=config,
        seed=101,
        replicates=3,
        retain_analytic=True,
    )
    second = continuous_phase_surrogate_bank(
        analytic,
        frequencies,
        timestep_seconds=timestep,
        config=config,
        seed=101,
        replicates=3,
        retain_analytic=True,
    )
    assert len(first[0]) == 3
    for first_value, second_value in zip(first[1:], second[1:], strict=True):
        np.testing.assert_allclose(first_value, second_value)
    with pytest.raises(ValueError, match="must be positive"):
        continuous_phase_surrogate_bank(
            analytic,
            frequencies,
            timestep_seconds=timestep,
            config=config,
            seed=101,
            replicates=0,
        )


def _analytic_delay_trials(
    delay_steps: float,
    *,
    seed: int = 7,
    trials: int = 64,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    frequencies = np.asarray([7.0, 9.0, 11.0, 14.0, 18.0, 23.0, 29.0, 36.0], dtype=np.float32)
    timestep_seconds = 0.008
    time = np.arange(512, dtype=np.float32) * timestep_seconds
    analytic = np.empty((trials, frequencies.size, 2, time.size), dtype=np.complex64)
    for trial in range(trials):
        intercept = rng.uniform(-1.0, 1.0)
        source_offset = rng.uniform(-np.pi, np.pi, size=(frequencies.size, 1))
        source_phase = 2.0 * np.pi * frequencies[:, None] * time + source_offset
        target_phase = (
            source_phase
            - 2.0 * np.pi * frequencies[:, None] * delay_steps * timestep_seconds
            + intercept
        )
        source = np.exp(1j * source_phase)
        target = np.exp(1j * target_phase)
        noise = 0.04 * (rng.normal(size=source.shape) + 1j * rng.normal(size=source.shape))
        analytic[trial, :, 0] = source + noise
        analytic[trial, :, 1] = target + 0.8 * noise
    return analytic, frequencies, timestep_seconds


def test_continuous_signed_delay_recovers_subsample_direction_and_fraction() -> None:
    analytic, frequencies, timestep = _analytic_delay_trials(0.5)
    config = ContinuousDelayConfig(
        max_delay_steps=3,
        grid_oversample=4,
        bootstrap_samples=64,
        min_bayes_factor=2.0,
        min_bootstrap_frequency=0.60,
        min_direction_probability=0.75,
        random_seed=11,
    )
    score, grid, _ = trial_continuous_phase_scores(
        analytic,
        frequencies,
        timestep_seconds=timestep,
        config=config,
    )
    null_score, null_grid, _ = trial_continuous_phase_scores(
        complex_phase_surrogate(analytic, seed=23),
        frequencies,
        timestep_seconds=timestep,
        config=config,
    )
    fit = fit_continuous_delay_evidence(score, null_score, grid, config)

    np.testing.assert_array_equal(grid, null_grid)
    assert abs(float(fit["signed_delay_map"][1, 0]) - 0.5) <= 0.25
    assert abs(float(fit["signed_delay_map"][0, 1]) + 0.5) <= 0.25
    assert bool(fit["accepted"][1, 0])
    assert not bool(fit["accepted"][0, 1])
    assert float(fit["direction_probability"][1, 0]) > 0.75
    assert float(fit["direction_probability"][0, 1]) < 0.25
    assert float(fit["positive_delay_bayes_factor"][1, 0]) > 2.0
    assert float(fit["positive_delay_bootstrap_frequency"][1, 0]) >= 0.60
    assert float(fit["zero_delay_probability"][1, 0]) < 0.25
    assert abs(float(fit["fractional_delay_target"][1, 0]) - 0.5) <= 0.25
    np.testing.assert_allclose(fit["positive_delay_probability"][1, 0].sum(), 1.0, atol=1e-6)
    assert float(fit["signed_delay_map"][0, 0]) == 0.0
    assert float(fit["null_probability"][0, 0]) == 1.0
    assert not bool(fit["accepted"][0, 0])


def test_continuous_delay_time_reversal_transposes_direction() -> None:
    analytic, frequencies, timestep = _analytic_delay_trials(1.25, seed=17)
    config = ContinuousDelayConfig(
        max_delay_steps=3,
        grid_oversample=4,
        bootstrap_samples=16,
        random_seed=19,
    )
    normal, grid, _ = trial_continuous_phase_scores(
        analytic, frequencies, timestep_seconds=timestep, config=config
    )
    reversed_score, reversed_grid, _ = trial_continuous_phase_scores(
        analytic[..., ::-1].conj().copy(),
        frequencies,
        timestep_seconds=timestep,
        config=config,
    )

    normal_delay = float(grid[normal[:, 1, 0].mean(axis=0).argmax()])
    reversed_delay = float(reversed_grid[reversed_score[:, 0, 1].mean(axis=0).argmax()])
    assert abs(normal_delay - 1.25) <= 0.25
    assert abs(reversed_delay - 1.25) <= 0.25


def test_route_null_is_not_the_zero_delay_member() -> None:
    analytic, frequencies, timestep = _analytic_delay_trials(0.0, seed=29)
    config = ContinuousDelayConfig(
        max_delay_steps=2,
        grid_oversample=4,
        bootstrap_samples=32,
        min_bayes_factor=2.0,
        min_direction_probability=0.80,
        random_seed=31,
    )
    score, grid, _ = trial_continuous_phase_scores(
        analytic, frequencies, timestep_seconds=timestep, config=config
    )
    null_score, _, _ = trial_continuous_phase_scores(
        complex_phase_surrogate(analytic, seed=37),
        frequencies,
        timestep_seconds=timestep,
        config=config,
    )
    fit = fit_continuous_delay_evidence(score, null_score, grid, config)

    assert float(fit["bayes_factor"][1, 0]) > 1.0
    assert abs(float(fit["signed_delay_map"][1, 0])) <= 0.25
    assert float(fit["positive_delay_bayes_factor"][1, 0]) < 2.0
    assert not bool(fit["accepted"][1, 0])


def test_psi_imaginary_support_is_an_independent_route_gate() -> None:
    analytic, frequencies, timestep = _analytic_delay_trials(0.75, seed=41)
    config = ContinuousDelayConfig(
        max_delay_steps=3,
        grid_oversample=4,
        bootstrap_samples=32,
        min_bayes_factor=2.0,
        min_bootstrap_frequency=0.60,
        min_direction_probability=0.70,
        random_seed=43,
    )
    score, grid, _ = trial_continuous_phase_scores(
        analytic, frequencies, timestep_seconds=timestep, config=config
    )
    surrogate = complex_phase_surrogate(analytic, seed=47)
    null_score, _, _ = trial_continuous_phase_scores(
        surrogate, frequencies, timestep_seconds=timestep, config=config
    )
    support = np.full(score.shape[:-1], 0.9, dtype=np.float32)
    null_support = np.full(score.shape[:-1], 0.1, dtype=np.float32)
    fit = fit_continuous_delay_evidence(
        score,
        null_score,
        grid,
        config,
        direction_support=support,
        null_direction_support=null_support,
    )
    rejected = fit_continuous_delay_evidence(
        score,
        null_score,
        grid,
        config,
        direction_support=null_support,
        null_direction_support=support,
    )

    assert bool(fit["accepted"][1, 0])
    assert float(fit["psi_imaginary_bayes_factor"][1, 0]) > 2.0
    assert not bool(rejected["accepted"][1, 0])


def test_fold_local_continuous_prior_exports_model_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(17)
    trials, bands, nodes, steps = 12, 3, 2, 48
    phase = rng.uniform(-np.pi, np.pi, size=(trials, bands, nodes, steps))
    analytic = np.exp(1j * phase).astype(np.complex64)

    monkeypatch.setattr(
        "dpc_snn.analysis.evidence_space.model_evidence_analytic_features",
        lambda *args, **kwargs: analytic,
    )
    arrays, summary = fold_local_continuous_delay_prior(
        {"X": np.zeros((trials, 2, 64), dtype=np.float32), "sfreq": 64.0},
        evidence_band_edges_hz=[[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]],
        output_band_edges_hz=[[2.0, 5.0], [5.0, 8.0]],
        n_nodes=nodes,
        graph_steps=steps,
        max_delay_steps=2,
        bootstrap_samples=8,
        grid_oversample=2,
        task_tmin=0.0,
        task_tmax=1.0,
    )

    assert arrays["route_probability"].shape == (2, 2, nodes, nodes)
    assert arrays["positive_delay_probability"].shape == (
        2,
        2,
        nodes,
        nodes,
        3,
    )
    assert arrays["fractional_delay_target"].shape == (2, 2, nodes, nodes)
    assert arrays["connectivity_prior"].shape == (2, 2, nodes, nodes)
    assert arrays["time_reversed_signed_node_delay_mean"].shape == (nodes, nodes)
    assert arrays["time_reversed_signed_node_accepted"].shape == (nodes, nodes)
    assert arrays["phase_surrogate_signed_node_bayes_factor"].shape == (nodes, nodes)
    assert summary["heldout_data_accessed"] is False
    assert summary["zero_delay_is_route_null"] is False
    assert summary["source_target_band_pairs_preserved"] is True
    assert set(summary["evidence_checks"]) == {
        "natural_nonzero_edges",
        "split_half_delay",
        "bootstrap_frequency",
        "time_reversal",
        "phase_surrogate",
    }
    assert isinstance(summary["evidence_pipeline_passed"], bool)
    assert (
        summary["split_half_delay_scope"]
        == "union_of_independently_selected_node_edges"
    )
    assert summary["minimum_split_half_edge_jaccard"] == 0.50
    assert summary["minimum_time_reversal_edge_jaccard"] == 0.50


def test_fold_local_continuous_prior_accepts_precomputed_online_representation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(23)
    trials, bands, nodes, steps = 10, 3, 2, 32
    phase = rng.uniform(-np.pi, np.pi, size=(trials, bands, nodes, steps))
    analytic = np.exp(1j * phase).astype(np.complex64)

    def _unexpected_legacy_projection(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("the divergent legacy evidence projector was called")

    monkeypatch.setattr(
        "dpc_snn.analysis.evidence_space.model_evidence_analytic_features",
        _unexpected_legacy_projection,
    )
    _, summary = fold_local_continuous_delay_prior(
        {"X": np.zeros((trials, 2, 64), dtype=np.float32), "sfreq": 64.0},
        evidence_band_edges_hz=[[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]],
        output_band_edges_hz=[[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]],
        n_nodes=nodes,
        graph_steps=steps,
        max_delay_steps=2,
        bootstrap_samples=4,
        grid_oversample=2,
        task_tmin=0.0,
        task_tmax=1.0,
        analytic_features=analytic,
        analytic_representation="unit_test_online_frontend",
    )

    assert summary["precomputed_analytic_features"] is True
    assert summary["analytic_representation"] == "unit_test_online_frontend"
