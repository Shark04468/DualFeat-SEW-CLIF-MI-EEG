from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dpc_snn.analysis.evidence_space import (
    HurdleEvidenceConfig,
    evaluate_evidence_space,
    fit_hurdle_evidence,
    fit_var_innovations,
    fold_local_model_evidence_prior,
    complex_phase_surrogate,
    directed_phase_delay_scores,
    model_evidence_features,
    model_evidence_analytic_features,
    trial_band_lag_scores,
    trial_phase_slope_scores,
    trial_psi_imaginary_support,
    trial_lag_scores,
)
from dpc_snn.models.eeg_frontend import (
    AnchoredSpatialProjection,
    LearnableAnalyticFilterBank,
)


def _delayed_trials(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(96, 3, 192)).astype(np.float32)
    x[:, 1, 4:] += 1.5 * x[:, 0, :-4]
    return x


def test_hurdle_evidence_separates_edge_existence_from_positive_delay() -> None:
    scores = trial_lag_scores(_delayed_trials(), 192.0, 96.0, max_delay_steps=5)
    fit = fit_hurdle_evidence(
        scores,
        HurdleEvidenceConfig(max_delay_steps=5, bootstrap_samples=128, random_seed=7),
    )

    assert bool(fit["accepted"][1, 0])
    assert int(fit["delay_map"][1, 0]) == 2
    assert float(fit["null_probability"][1, 0]) < 0.30
    assert not bool(fit["accepted"][0, 0])
    assert np.allclose(fit["positive_delay_probability"][1, 0].sum(), 1.0)


def test_hurdle_does_not_force_edges_when_only_zero_lag_exists() -> None:
    rng = np.random.default_rng(11)
    source = rng.normal(size=(96, 1, 192)).astype(np.float32)
    x = np.concatenate((source, source + 0.01 * rng.normal(size=source.shape)), axis=1)
    scores = trial_lag_scores(x, 192.0, 96.0, max_delay_steps=5)
    fit = fit_hurdle_evidence(
        scores,
        HurdleEvidenceConfig(max_delay_steps=5, bootstrap_samples=128, random_seed=13),
    )

    assert not bool(fit["accepted"][0, 1])
    assert not bool(fit["accepted"][1, 0])
    assert float(fit["null_probability"][1, 0]) > 0.70


def test_model_evidence_audit_uses_band_node_entities() -> None:
    rng = np.random.default_rng(17)
    x = rng.normal(size=(4, 3, 64)).astype(np.float32)
    carrier, envelope = model_evidence_features(
        x,
        sfreq=128.0,
        band_edges_hz=[[6.0, 10.0], [10.0, 20.0]],
        n_nodes=2,
        graph_steps=32,
        batch_size=2,
    )
    scores = trial_band_lag_scores(carrier, envelope, max_delay_steps=2)

    assert carrier.shape == (4, 2, 2, 32)
    assert envelope.shape == carrier.shape
    assert scores.shape == (4, 4, 4, 3)


def test_phase_slope_recovers_delay_despite_static_phase_intercept() -> None:
    frequencies = np.asarray([8.0, 12.0, 18.0, 24.0], dtype=np.float32)
    timestep = 0.01
    true_delay = 3
    time = np.arange(256, dtype=np.float32) * timestep
    analytic = np.empty((12, frequencies.size, 2, time.size), dtype=np.complex64)
    for trial in range(analytic.shape[0]):
        intercept = 0.4 + 0.03 * trial
        for band, frequency in enumerate(frequencies):
            source_phase = 2.0 * np.pi * frequency * time
            analytic[trial, band, 0] = np.exp(1j * source_phase)
            analytic[trial, band, 1] = np.exp(
                1j
                * (
                    source_phase
                    - 2.0 * np.pi * frequency * true_delay * timestep
                    + intercept
                )
            )

    score = trial_phase_slope_scores(
        analytic,
        frequencies,
        max_delay_steps=6,
        timestep_seconds=timestep,
    )

    assert int(score[:, 1, 0].mean(axis=0).argmax()) == true_delay


def test_fold_local_prior_has_model_geometry_and_no_heldout_access() -> None:
    rng = np.random.default_rng(23)
    data = {
        "X": rng.normal(size=(12, 3, 320)).astype(np.float32),
        "y": np.tile(np.arange(3), 4),
        "sfreq": 64.0,
        "epoch_tmin": -1.0,
        "epoch_tmax": 4.0,
    }

    arrays, summary = fold_local_model_evidence_prior(
        data,
        evidence_band_edges_hz=[[6.0, 10.0], [10.0, 14.0], [14.0, 20.0]],
        output_band_edges_hz=[[6.0, 10.0], [14.0, 20.0]],
        n_nodes=2,
        graph_steps=192,
        max_delay_steps=2,
        bootstrap_samples=4,
        seed=4,
    )

    assert arrays["route_probability"].shape == (2, 2, 2, 2)
    assert arrays["positive_delay_probability"].shape == (2, 2, 2, 2, 3)
    assert arrays["fractional_delay_target"].shape == (2, 2, 2, 2)
    assert arrays["connectivity_prior"].shape == (2, 2, 2, 2)
    assert summary["fit_scope"] == "inner_training_fold_only"
    assert summary["heldout_data_accessed"] is False
    assert summary["route_direction_evidence"] == (
        "phase_slope_plus_PSI_and_imaginary_coherency"
    )
    assert "time_reversal_transpose_support_mae" in summary
    assert "phase_surrogate_support_drop" in summary


def test_psi_support_tracks_direction_and_weakens_after_phase_surrogate() -> None:
    rng = np.random.default_rng(41)
    trials, bands, nodes, steps = 48, 8, 2, 256
    frequencies = np.linspace(6.0, 34.0, bands, dtype=np.float32)
    analytic = np.empty((trials, bands, nodes, steps), dtype=np.complex64)
    sample_rate = 128.0
    time = np.arange(steps, dtype=np.float32) / sample_rate
    for trial in range(trials):
        phase_offset = rng.uniform(-np.pi, np.pi, (bands, 1))
        source = np.exp(
            1j * (2.0 * np.pi * frequencies[:, None] * time + phase_offset)
        )
        source += 0.03 * (
            rng.normal(size=source.shape) + 1j * rng.normal(size=source.shape)
        )
        analytic[trial, :, 0] = source
        analytic[trial, :, 1, 3:] = source[:, :-3]
        analytic[trial, :, 1, :3] = 0.0

    support, _, _ = trial_psi_imaginary_support(analytic)
    reversed_support, _, _ = trial_psi_imaginary_support(
        analytic[..., ::-1].conj().copy()
    )
    _, surrogate_support = directed_phase_delay_scores(
        complex_phase_surrogate(analytic, seed=9),
        frequencies,
        max_delay_steps=6,
        timestep_seconds=1.0 / sample_rate,
    )

    assert support[:, 1, 0].mean() > support[:, 0, 1].mean()
    assert reversed_support[:, 0, 1].mean() > reversed_support[:, 1, 0].mean()
    assert surrogate_support[:, 1, 0].mean() < support[:, 1, 0].mean()


def test_offline_prior_preprocessing_matches_online_evidence_sequence() -> None:
    rng = np.random.default_rng(53)
    x = rng.normal(size=(3, 3, 320)).astype(np.float32)
    bands = [[6.0, 10.0], [14.0, 20.0]]
    offline = model_evidence_analytic_features(
        x,
        sfreq=64.0,
        band_edges_hz=bands,
        n_nodes=2,
        graph_steps=192,
        batch_size=3,
        epoch_tmin=-1.0,
        task_tmin=0.0,
        task_tmax=4.0,
        baseline_normalize=True,
    )

    filterbank = LearnableAnalyticFilterBank(
        64.0,
        bands,
        max_center_shift_hz=0.5,
        min_bandwidth_hz=1.0,
        transition_hz=0.5,
        max_high_hz=23.0,
    ).eval()
    projector = AnchoredSpatialProjection(3, 2, max_deviation=0.0).eval()
    with torch.no_grad():
        latent = projector(filterbank(torch.from_numpy(x)))
        baseline = latent[..., :64].abs().mean(dim=-1, keepdim=True)
        floor = (0.05 * baseline.amax(dim=-2, keepdim=True)).clamp_min(1e-4)
        baseline = torch.maximum(baseline, floor)
        latent = latent[..., 64:320] / baseline
        shape = latent.shape
        real = F.interpolate(
            latent.real.reshape(-1, 1, shape[-1]),
            size=192,
            mode="linear",
            align_corners=False,
        ).reshape(*shape[:-1], 192)
        imag = F.interpolate(
            latent.imag.reshape(-1, 1, shape[-1]),
            size=192,
            mode="linear",
            align_corners=False,
        ).reshape(*shape[:-1], 192)
        online_sequence = torch.complex(real, imag).numpy()

    np.testing.assert_allclose(offline, online_sequence, rtol=1e-5, atol=1e-6)


def test_var_innovations_preserve_trial_boundaries_and_whiten() -> None:
    innovations = fit_var_innovations(_delayed_trials(), order=4)
    flat = innovations.transpose(0, 2, 1).reshape(-1, innovations.shape[1])
    covariance = np.cov(flat, rowvar=False)

    assert innovations.shape == (96, 3, 188)
    np.testing.assert_allclose(np.diag(covariance), np.ones(3), atol=0.05)


def test_unwhitened_var_innovations_preserve_target_channel_axes() -> None:
    x = _delayed_trials()
    residual = fit_var_innovations(x, order=4, whiten=False)
    whitened = fit_var_innovations(x, order=4, whiten=True)
    assert residual.shape == whitened.shape == (96, 3, 188)
    assert not np.allclose(residual, whitened)


def test_evidence_audit_script_does_not_import_classifier_training() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "audit_v37_evidence_spaces.py").read_text()

    assert "train_model" not in script
    assert "build_model" not in script
    assert '"classifier_training": False' in script


def test_evidence_summary_is_json_serializable() -> None:
    trials = _delayed_trials()[:12]
    normal_scores = trial_lag_scores(trials, 192.0, 96.0, max_delay_steps=3)
    reversed_scores = trial_lag_scores(
        trials[..., ::-1].copy(), 192.0, 96.0, max_delay_steps=3
    )
    summary, _ = evaluate_evidence_space(
        normal_scores,
        reversed_scores,
        normal_scores.copy(),
        HurdleEvidenceConfig(
            max_delay_steps=3,
            bootstrap_samples=8,
            random_seed=13,
        ),
    )

    json.dumps(summary, allow_nan=True)
