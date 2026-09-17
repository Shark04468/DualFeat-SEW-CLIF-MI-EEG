"""Classifier-free real-EEG delay evidence audits."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
from scipy import signal, stats
from scipy.special import logsumexp
import torch
import torch.nn.functional as F

from dpc_snn.models.eeg_frontend import AnchoredSpatialProjection, LearnableAnalyticFilterBank
from dpc_snn.preprocessing.standardize import euclidean_alignment_matrix
from dpc_snn.analysis.continuous_delay import (
    ContinuousDelayConfig,
    fit_band_pair_route_evidence,
    fit_continuous_delay_evidence,
    trial_band_pair_scores_at_node_delay,
    trial_continuous_phase_scores,
)


@dataclass(frozen=True)
class HurdleEvidenceConfig:
    max_delay_steps: int = 8
    bootstrap_samples: int = 256
    min_bayes_factor: float = 3.0
    min_bootstrap_frequency: float = 0.70
    random_seed: int = 0


def stratified_split_half_indices(
    n_trials: int,
    split_strata: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic halves while preserving every declared stratum."""

    if int(n_trials) < 4:
        raise ValueError("split-half evidence requires at least four trials")
    if split_strata is None:
        indices = np.arange(int(n_trials), dtype=np.int64)
        return indices[::2], indices[1::2]
    strata = [str(value) for value in split_strata]
    if len(strata) != int(n_trials):
        raise ValueError("split-half strata must match the number of evidence trials")
    groups: dict[str, list[int]] = {}
    for index, value in enumerate(strata):
        groups.setdefault(value, []).append(index)
    if any(len(indices) < 2 for indices in groups.values()):
        raise ValueError("every split-half stratum must contain at least two trials")
    first = sorted(index for indices in groups.values() for index in indices[::2])
    second = sorted(index for indices in groups.values() for index in indices[1::2])
    if not first or not second or set(first).intersection(second):
        raise RuntimeError("invalid stratified split-half partition")
    if sorted((*first, *second)) != list(range(int(n_trials))):
        raise RuntimeError("stratified split-half partition is incomplete")
    return np.asarray(first, dtype=np.int64), np.asarray(second, dtype=np.int64)


def task_window(data: dict[str, Any], tmin: float = 0.0, tmax: float = 4.0) -> np.ndarray:
    x = np.asarray(data["X"], dtype=np.float32)
    sfreq = float(data.get("sfreq", 250.0))
    epoch_tmin = float(data.get("epoch_tmin", 0.0))
    start = max(0, int(round((tmin - epoch_tmin) * sfreq)))
    stop = min(x.shape[-1], int(round((tmax - epoch_tmin) * sfreq)))
    if stop <= start:
        raise ValueError(f"Empty task window [{tmin}, {tmax}] for epoch starting at {epoch_tmin}")
    return x[..., start:stop]


def apply_euclidean_alignment(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = euclidean_alignment_matrix(x)
    return np.einsum("ij,njt->nit", matrix, x).astype(np.float32), matrix


def apply_alignment_matrix(x: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.einsum("ij,njt->nit", matrix, x).astype(np.float32)


def current_source_density(x: np.ndarray, ch_names: list[str], sfreq: float) -> np.ndarray:
    """Apply spherical-spline CSD with the canonical high-density montage."""

    import mne

    info = mne.create_info(ch_names=list(ch_names), sfreq=float(sfreq), ch_types="eeg")
    # standard_1005 contains the extended FTT/TPP sites used by OpenBMI and is
    # also a strict superset of the BCI2a 10-20 channels.
    info.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="raise")
    epochs = mne.EpochsArray(np.asarray(x, dtype=np.float64), info, verbose="ERROR")
    csd = mne.preprocessing.compute_current_source_density(
        epochs, lambda2=1e-5, stiffness=4, n_legendre_terms=50, copy=True
    )
    return csd.get_data(copy=True).astype(np.float32)


def fit_var_innovations(
    x: np.ndarray,
    order: int = 6,
    ridge: float = 1e-2,
    *,
    whiten: bool = True,
) -> np.ndarray:
    """Fit one ridge VAR without cross-trial transitions.

    Unwhitened residuals preserve the original target-channel axes and are
    therefore suitable for physical-node evidence. Whitening is retained as
    the backwards-compatible default for representation diagnostics.
    """

    x64 = np.asarray(x, dtype=np.float64)
    n_trials, n_channels, n_time = x64.shape
    if n_time <= order:
        raise ValueError("VAR order must be shorter than every trial")
    width = order * n_channels
    xtx = np.zeros((width, width), dtype=np.float64)
    xty = np.zeros((width, n_channels), dtype=np.float64)
    for trial in x64:
        target = trial[:, order:].T
        design = np.concatenate(
            [trial[:, order - lag : n_time - lag].T for lag in range(1, order + 1)],
            axis=1,
        )
        xtx += design.T @ design
        xty += design.T @ target
    scale = np.trace(xtx) / max(1, width)
    coefficients = np.linalg.solve(xtx + ridge * max(scale, 1e-8) * np.eye(width), xty)
    residual = np.zeros((n_trials, n_channels, n_time - order), dtype=np.float64)
    for index, trial in enumerate(x64):
        target = trial[:, order:].T
        design = np.concatenate(
            [trial[:, order - lag : n_time - lag].T for lag in range(1, order + 1)],
            axis=1,
        )
        residual[index] = (target - design @ coefficients).T
    if not whiten:
        return residual.astype(np.float32)
    flat = residual.transpose(0, 2, 1).reshape(-1, n_channels)
    covariance = flat.T @ flat / max(1, flat.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    whitening = (eigenvectors * np.maximum(eigenvalues, 1e-8) ** -0.5) @ eigenvectors.T
    return np.einsum("ij,njt->nit", whitening, residual).astype(np.float32)


def phase_surrogate(x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(x, axis=-1)
    random_phase = rng.uniform(-np.pi, np.pi, spectrum.shape)
    random_phase[..., 0] = np.angle(spectrum[..., 0])
    if x.shape[-1] % 2 == 0:
        random_phase[..., -1] = np.angle(spectrum[..., -1])
    return np.fft.irfft(
        np.abs(spectrum) * np.exp(1j * random_phase), n=x.shape[-1], axis=-1
    ).astype(np.float32)


def trial_lag_scores(
    x: np.ndarray, original_sfreq: float, evidence_rate_hz: float, max_delay_steps: int
) -> np.ndarray:
    """Absolute directed trial correlations [trial,target,source,delay]."""

    ratio = float(original_sfreq) / float(evidence_rate_hz)
    rounded = int(round(ratio))
    if not np.isclose(ratio, rounded) or rounded < 1:
        raise ValueError("Evidence rate must be an integer divisor of the original sampling rate")
    reduced = signal.resample_poly(np.asarray(x, dtype=np.float64), 1, rounded, axis=-1)
    outputs = []
    for delay in range(max_delay_steps + 1):
        target = reduced if delay == 0 else reduced[..., delay:]
        source = reduced if delay == 0 else reduced[..., :-delay]
        target = target - target.mean(axis=-1, keepdims=True)
        source = source - source.mean(axis=-1, keepdims=True)
        target /= np.sqrt(np.mean(target**2, axis=-1, keepdims=True)).clip(1e-8)
        source /= np.sqrt(np.mean(source**2, axis=-1, keepdims=True)).clip(1e-8)
        outputs.append(np.abs(np.einsum("nct,ndt->ncd", target, source) / target.shape[-1]))
    return np.stack(outputs, axis=-1).astype(np.float32)


def model_evidence_features(
    x: np.ndarray,
    sfreq: float,
    band_edges_hz: list[list[float]],
    n_nodes: int = 16,
    graph_steps: int = 500,
    batch_size: int = 8,
    device: str = "cpu",
    electrode_coordinates: np.ndarray | list[list[float]] | None = None,
    spatial_anchor_indices: np.ndarray | list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the fixed model evidence projector: 12 bands by 16 latent nodes."""

    analytic = model_evidence_analytic_features(
        x,
        sfreq,
        band_edges_hz,
        n_nodes=n_nodes,
        graph_steps=graph_steps,
        batch_size=batch_size,
        device=device,
        electrode_coordinates=electrode_coordinates,
        spatial_anchor_indices=spatial_anchor_indices,
    )
    return analytic.real.astype(np.float32), np.log1p(np.abs(analytic)).astype(np.float32)


def model_evidence_analytic_features(
    x: np.ndarray,
    sfreq: float,
    band_edges_hz: list[list[float]],
    n_nodes: int = 16,
    graph_steps: int = 500,
    batch_size: int = 8,
    device: str = "cpu",
    epoch_tmin: float | None = None,
    task_tmin: float | None = None,
    task_tmax: float | None = None,
    baseline_normalize: bool = False,
    electrode_coordinates: np.ndarray | list[list[float]] | None = None,
    spatial_anchor_indices: np.ndarray | list[int] | None = None,
) -> np.ndarray:
    """Return the fixed complex 12-band/16-node evidence representation."""

    x = np.asarray(x, dtype=np.float32)
    if task_tmin is None or task_tmax is None:
        duration_seconds = x.shape[-1] / float(sfreq)
    else:
        duration_seconds = float(task_tmax) - float(task_tmin)
        if duration_seconds <= 0:
            raise ValueError("task_tmax must be greater than task_tmin")
    graph_rate_hz = int(graph_steps) / max(duration_seconds, 1e-8)
    filterbank = (
        LearnableAnalyticFilterBank(
            sfreq,
            band_edges_hz,
            max_center_shift_hz=0.5,
            min_bandwidth_hz=1.0,
            transition_hz=0.5,
            max_high_hz=graph_rate_hz / 2.0 - 1.0,
        )
        .to(device)
        .eval()
    )
    projector = (
        AnchoredSpatialProjection(
            x.shape[1],
            min(int(n_nodes), x.shape[1]),
            max_deviation=0.0,
            electrode_coordinates=electrode_coordinates,
            anchor_indices=spatial_anchor_indices,
        )
        .to(device)
        .eval()
    )
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], max(1, int(batch_size))):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            analytic = projector(filterbank(batch))
            if task_tmin is not None and task_tmax is not None:
                start_index = int(round((float(task_tmin) - float(epoch_tmin or 0.0)) * sfreq))
                stop_index = int(round((float(task_tmax) - float(epoch_tmin or 0.0)) * sfreq))
                start_index = max(0, min(analytic.shape[-1] - 1, start_index))
                stop_index = max(start_index + 1, min(analytic.shape[-1], stop_index))
                if baseline_normalize:
                    baseline_end = start_index
                    if baseline_end <= 0:
                        raise ValueError("baseline_normalize requires a pre-task baseline")
                    baseline = analytic[..., :baseline_end].abs().mean(dim=-1, keepdim=True)
                    node_reference = baseline.amax(dim=-2, keepdim=True)
                    floor = (0.05 * node_reference).clamp_min(1e-4)
                    baseline = torch.maximum(baseline, floor)
                else:
                    baseline = None
                analytic = analytic[..., start_index:stop_index]
                if baseline is not None:
                    analytic = analytic / baseline
            shape = analytic.shape
            real = F.interpolate(
                analytic.real.reshape(-1, 1, shape[-1]),
                size=int(graph_steps),
                mode="linear",
                align_corners=False,
            ).reshape(*shape[:-1], int(graph_steps))
            imag = F.interpolate(
                analytic.imag.reshape(-1, 1, shape[-1]),
                size=int(graph_steps),
                mode="linear",
                align_corners=False,
            ).reshape(*shape[:-1], int(graph_steps))
            outputs.append(torch.complex(real, imag).cpu().numpy().astype(np.complex64))
    return np.concatenate(outputs)


def trial_phase_slope_scores(
    analytic: np.ndarray,
    band_frequencies_hz: np.ndarray,
    max_delay_steps: int,
    timestep_seconds: float,
) -> np.ndarray:
    """Trial-wise group-delay evidence with a nuisance phase intercept.

    A single delay is fitted from the phase slope across all evidence bands.
    Taking the complex magnitude after frequency alignment analytically removes
    a static phase intercept; the trainable residual phase preference remains
    fixed at zero until this delay target has been fitted.
    """

    analytic = np.asarray(analytic, dtype=np.complex64)
    frequencies = np.asarray(band_frequencies_hz, dtype=np.float32)
    if analytic.ndim != 4 or analytic.shape[1] != frequencies.size:
        raise ValueError("analytic evidence must have shape [N, B, K, T]")
    amplitude = np.abs(analytic)
    confidence = amplitude / (amplitude + np.median(amplitude, axis=-1, keepdims=True).clip(1e-6))
    phasor = confidence * analytic / amplitude.clip(1e-6)
    cross = np.einsum("nbkt,nblt->nbkl", phasor, phasor.conj()) / analytic.shape[-1]
    weight = np.abs(cross).clip(1e-6)
    unit_cross = cross / weight
    delays = np.arange(int(max_delay_steps) + 1, dtype=np.float32) * float(timestep_seconds)
    rotation = np.exp(1j * 2.0 * np.pi * frequencies[:, None] * delays[None]).astype(np.complex64)
    aligned = unit_cross[..., None] * rotation[None, :, None, None, :]
    score = np.abs(np.sum(weight[..., None] * aligned, axis=1))
    score /= np.sum(weight, axis=1)[..., None].clip(1e-6)
    return score.astype(np.float32)


def trial_psi_imaginary_support(
    analytic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return directed PSI support and imaginary coherency per trial/node pair."""

    analytic = np.asarray(analytic, dtype=np.complex64)
    if analytic.ndim != 4 or analytic.shape[1] < 2:
        raise ValueError("PSI evidence requires [N, B>=2, K, T]")
    cross = np.einsum("nbkt,nblt->nbkl", analytic, analytic.conj())
    power = np.mean(np.abs(analytic) ** 2, axis=-1).clip(1e-8)
    denominator = np.sqrt(power[:, :, :, None] * power[:, :, None, :])
    coherency = cross / (analytic.shape[-1] * denominator).clip(1e-8)
    # For target_i(t)=source_j(t-tau), C_ij(f)=exp(-j2*pi*f*tau).
    # The minus sign therefore makes positive PSI denote j -> i.
    psi = -np.imag(np.sum(coherency[:, :-1].conj() * coherency[:, 1:], axis=1))
    imaginary = np.mean(np.abs(np.imag(coherency)), axis=1)
    n_nodes = analytic.shape[2]
    off_diagonal = ~np.eye(n_nodes, dtype=bool)
    psi_scale = np.median(np.abs(psi[:, off_diagonal]), axis=1).clip(1e-4)
    imag_scale = np.median(imaginary[:, off_diagonal], axis=1).clip(1e-4)
    standardized_psi = np.clip(psi / psi_scale[:, None, None], -30.0, 30.0)
    direction = 1.0 / (1.0 + np.exp(-standardized_psi))
    noninstantaneous = imaginary / (imaginary + imag_scale[:, None, None])
    support = direction * noninstantaneous
    support[:, np.arange(n_nodes), np.arange(n_nodes)] = 0.0
    return (
        support.astype(np.float32),
        psi.astype(np.float32),
        imaginary.astype(np.float32),
    )


def complex_phase_surrogate(analytic: np.ndarray, seed: int) -> np.ndarray:
    """Destroy cross-signal phase relations while preserving each spectrum."""

    rng = np.random.default_rng(seed)
    spectrum = np.fft.fft(np.asarray(analytic, dtype=np.complex64), axis=-1)
    rotation = np.exp(1j * rng.uniform(-np.pi, np.pi, spectrum.shape))
    return np.fft.ifft(spectrum * rotation, axis=-1).astype(np.complex64)


def continuous_phase_surrogate_bank(
    analytic: np.ndarray,
    frequencies: np.ndarray,
    *,
    timestep_seconds: float,
    config: ContinuousDelayConfig,
    seed: int,
    replicates: int,
    retain_analytic: bool = False,
) -> tuple[
    tuple[np.ndarray, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Estimate the phase-randomized null with fixed Monte Carlo replicates."""

    if int(replicates) < 1:
        raise ValueError("phase-surrogate replicates must be positive")
    retained: list[np.ndarray] = []
    score_sum: np.ndarray | None = None
    support_sum: np.ndarray | None = None
    strength_sum: np.ndarray | None = None
    reference_grid: np.ndarray | None = None
    for index in range(int(replicates)):
        surrogate = complex_phase_surrogate(
            analytic,
            int(seed) + index * 104_729,
        )
        scores, grid, strength = trial_continuous_phase_scores(
            surrogate,
            frequencies,
            timestep_seconds=timestep_seconds,
            config=config,
        )
        support, _, _ = trial_psi_imaginary_support(surrogate)
        if reference_grid is None:
            reference_grid = grid
            score_sum = np.zeros_like(scores, dtype=np.float64)
            support_sum = np.zeros_like(support, dtype=np.float64)
            strength_sum = np.zeros_like(strength, dtype=np.float64)
        elif not np.array_equal(reference_grid, grid):
            raise RuntimeError("phase-surrogate delay grids diverged")
        score_sum += scores
        support_sum += support
        strength_sum += strength
        if retain_analytic:
            retained.append(surrogate)
    if (
        reference_grid is None
        or score_sum is None
        or support_sum is None
        or strength_sum is None
    ):
        raise RuntimeError("phase-surrogate bank did not produce an estimate")
    denominator = float(replicates)
    return (
        tuple(retained),
        (score_sum / denominator).astype(np.float32),
        (support_sum / denominator).astype(np.float32),
        (strength_sum / denominator).astype(np.float32),
        reference_grid,
    )


def directed_phase_delay_scores(
    analytic: np.ndarray,
    band_frequencies_hz: np.ndarray,
    max_delay_steps: int,
    timestep_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine group-delay likelihood with PSI/imaginary route support."""

    phase_score = trial_phase_slope_scores(
        analytic,
        band_frequencies_hz,
        max_delay_steps=max_delay_steps,
        timestep_seconds=timestep_seconds,
    )
    support, _, _ = trial_psi_imaginary_support(analytic)
    directed = phase_score.copy()
    directed[..., 1:] *= 0.5 + 0.5 * support[..., None]
    return directed.astype(np.float32), support


def trial_band_lag_scores(
    carrier: np.ndarray,
    envelope: np.ndarray,
    max_delay_steps: int,
    device: str = "cpu",
    batch_size: int = 16,
) -> np.ndarray:
    """Model-matched route evidence [trial, target entity, source entity, lag].

    Same-band routes use carrier lag evidence; cross-band routes use envelope
    lag evidence. An entity is one source/target band-node pair.
    """

    carrier = np.asarray(carrier, dtype=np.float32)
    envelope = np.asarray(envelope, dtype=np.float32)
    if carrier.shape != envelope.shape or carrier.ndim != 4:
        raise ValueError("carrier and envelope must share [N, B, K, T]")
    n_trials, n_bands, n_nodes, n_time = carrier.shape
    carrier = carrier.reshape(n_trials, n_bands * n_nodes, n_time)
    envelope = envelope.reshape(n_trials, n_bands * n_nodes, n_time)
    band_index = np.repeat(np.arange(n_bands), n_nodes)
    same_band = band_index[:, None] == band_index[None, :]

    output = np.empty(
        (n_trials, n_bands * n_nodes, n_bands * n_nodes, int(max_delay_steps) + 1),
        dtype=np.float32,
    )
    same_band_torch = torch.from_numpy(same_band).to(device)
    with torch.no_grad():
        for start in range(0, n_trials, max(1, int(batch_size))):
            stop = min(n_trials, start + max(1, int(batch_size)))
            carrier_batch = torch.from_numpy(carrier[start:stop]).to(device)
            envelope_batch = torch.from_numpy(envelope[start:stop]).to(device)
            for delay in range(int(max_delay_steps) + 1):
                correlations = []
                for values in (carrier_batch, envelope_batch):
                    target = values if delay == 0 else values[..., delay:]
                    source = values if delay == 0 else values[..., :-delay]
                    target = target - target.mean(dim=-1, keepdim=True)
                    source = source - source.mean(dim=-1, keepdim=True)
                    target = target / target.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(
                        1e-8
                    )
                    source = source / source.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(
                        1e-8
                    )
                    correlations.append(
                        torch.bmm(target, source.transpose(1, 2)).abs() / target.shape[-1]
                    )
                selected = torch.where(same_band_torch[None], correlations[0], correlations[1])
                output[start:stop, ..., delay] = selected.cpu().numpy()
    return output


def fit_hurdle_evidence(scores: np.ndarray, config: HurdleEvidenceConfig) -> dict[str, np.ndarray]:
    """Fit a null-versus-positive-lag hurdle using trial-level evidence.

    The first hurdle uses a direction-constrained BIC approximation to BF10 for
    each positive lag versus lag zero, then averages those alternatives under a
    uniform lag prior. Bootstrap frequency refits that complete hurdle, so lag
    selection is not fixed from the full sample.
    """

    if scores.shape[-1] < 2:
        raise ValueError("Hurdle evidence requires a zero-lag bin and at least one positive lag")

    def log_bf(scores_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        difference = np.asarray(scores_, dtype=np.float64)[..., 1:] - scores_[..., [0]]
        n_trials = difference.shape[0]
        mean_difference = difference.mean(axis=0)
        centered = difference - mean_difference[None]
        rss_null = np.sum(difference**2, axis=0).clip(1e-12)
        rss_alternative = np.sum(centered**2, axis=0).clip(1e-12)
        per_lag = 0.5 * (n_trials * np.log(rss_null / rss_alternative) - np.log(max(2, n_trials)))
        # The alternative is explicitly positive lag evidence, not merely any
        # departure from the instantaneous null.
        per_lag = np.where(mean_difference > 0.0, per_lag, -50.0)
        mixture = logsumexp(per_lag, axis=-1) - math.log(per_lag.shape[-1])
        return per_lag, mixture

    rng = np.random.default_rng(config.random_seed)
    route_shape = scores.shape[1:-1]
    flat_scores = np.asarray(scores).reshape(scores.shape[0], -1, scores.shape[-1])
    flat_per_lag_log_bf = np.empty(
        (flat_scores.shape[1], flat_scores.shape[-1] - 1), dtype=np.float64
    )
    flat_mixture_log_bf = np.empty(flat_scores.shape[1], dtype=np.float64)
    route_chunk = 512
    for start in range(0, flat_scores.shape[1], route_chunk):
        stop = min(flat_scores.shape[1], start + route_chunk)
        per_lag_chunk, mixture_chunk = log_bf(flat_scores[:, start:stop])
        flat_per_lag_log_bf[start:stop] = per_lag_chunk
        flat_mixture_log_bf[start:stop] = mixture_chunk
    candidate = flat_mixture_log_bf >= math.log(config.min_bayes_factor)
    wins = np.zeros(flat_scores.shape[1], dtype=np.int64)
    candidate_indices = np.flatnonzero(candidate)
    for _ in range(config.bootstrap_samples):
        if candidate_indices.size == 0:
            break
        indices = rng.integers(0, scores.shape[0], size=scores.shape[0])
        for start in range(0, candidate_indices.size, route_chunk):
            selected = candidate_indices[start : start + route_chunk]
            _, bootstrap_log_bf = log_bf(flat_scores[indices][:, selected])
            wins[selected] += bootstrap_log_bf >= math.log(config.min_bayes_factor)
    per_lag_log_bf = flat_per_lag_log_bf.reshape(*route_shape, scores.shape[-1] - 1)
    mixture_log_bf = flat_mixture_log_bf.reshape(route_shape)
    best_positive = per_lag_log_bf.argmax(axis=-1) + 1
    frequency = (wins / max(1, config.bootstrap_samples)).reshape(route_shape)
    bayes_factor = np.exp(np.clip(mixture_log_bf, -50.0, 50.0))
    null_probability = 1.0 / (1.0 + bayes_factor)
    accepted = (bayes_factor >= config.min_bayes_factor) & (
        frequency >= config.min_bootstrap_frequency
    )
    np.fill_diagonal(accepted, False)
    positive = np.exp(per_lag_log_bf - per_lag_log_bf.max(axis=-1, keepdims=True))
    positive /= positive.sum(axis=-1, keepdims=True).clip(1e-8)
    delay_grid = np.arange(1, positive.shape[-1] + 1, dtype=np.float32)
    posterior_mean = np.sum(positive * delay_grid, axis=-1)
    mean_scores = flat_scores.mean(axis=0).reshape(*route_shape, scores.shape[-1])
    peak = best_positive.astype(np.int64)
    left_index = np.maximum(0, peak - 1)
    right_index = np.minimum(scores.shape[-1] - 1, peak + 1)
    peak_score = np.take_along_axis(mean_scores, peak[..., None], axis=-1)[..., 0]
    left_score = np.take_along_axis(mean_scores, left_index[..., None], axis=-1)[..., 0]
    right_score = np.take_along_axis(mean_scores, right_index[..., None], axis=-1)[..., 0]
    denominator = left_score - 2.0 * peak_score + right_score
    offset = np.divide(
        0.5 * (left_score - right_score),
        denominator,
        out=np.zeros_like(denominator),
        where=np.abs(denominator) > 1e-8,
    )
    offset = np.clip(offset, -0.5, 0.5)
    continuous_peak = np.clip(peak + offset, 0.0, scores.shape[-1] - 1)
    negative_offset = offset < 0.0
    fractional_target = np.where(negative_offset, 1.0 + offset, offset)
    fractional_target = np.clip(fractional_target, 0.0, 1.0)
    transport_probability = np.zeros((*route_shape, scores.shape[-1]), dtype=np.float32)
    transport_probability[..., 1:] = positive.astype(np.float32)
    shifted_probability = np.zeros_like(transport_probability)
    shifted_probability[..., :-1] = positive.astype(np.float32)
    transport_probability = np.where(
        negative_offset[..., None], shifted_probability, transport_probability
    )
    return {
        "bayes_factor": bayes_factor.astype(np.float32),
        "log_bayes_factor": mixture_log_bf.astype(np.float32),
        "bootstrap_frequency": frequency.astype(np.float32),
        "null_probability": null_probability.astype(np.float32),
        "accepted": accepted,
        "delay_map": best_positive.astype(np.float32),
        "delay_mean": posterior_mean.astype(np.float32),
        "positive_delay_probability": positive.astype(np.float32),
        "positive_transport_probability": transport_probability,
        "fractional_delay_target": fractional_target.astype(np.float32),
        "continuous_delay_map": continuous_peak.astype(np.float32),
    }


def fold_local_model_evidence_prior(
    data: dict[str, Any],
    evidence_band_edges_hz: list[list[float]],
    output_band_edges_hz: list[list[float]],
    n_nodes: int,
    graph_steps: int,
    max_delay_steps: int,
    bootstrap_samples: int = 128,
    seed: int = 0,
    device: str = "cpu",
    task_tmin: float = 0.0,
    task_tmax: float = 4.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit and project a classifier-free prior using one inner-training fold."""

    x = np.asarray(data["X"], dtype=np.float32)
    analytic = model_evidence_analytic_features(
        x,
        float(data.get("sfreq", 250.0)),
        evidence_band_edges_hz,
        n_nodes=n_nodes,
        graph_steps=graph_steps,
        device=device,
        epoch_tmin=float(data.get("epoch_tmin", 0.0)),
        task_tmin=float(task_tmin),
        task_tmax=float(task_tmax),
        baseline_normalize=True,
        electrode_coordinates=data.get("electrode_coordinates"),
        spatial_anchor_indices=data.get("spatial_anchor_indices"),
    )
    carrier = analytic.real.astype(np.float32)
    envelope = np.log(np.abs(analytic).clip(1e-4)).astype(np.float32)
    scores = trial_band_lag_scores(
        carrier,
        envelope,
        max_delay_steps=max_delay_steps,
        device=device,
    )
    evidence_centres = np.asarray(evidence_band_edges_hz, dtype=np.float32).mean(axis=1)
    timestep_seconds = (float(task_tmax) - float(task_tmin)) / float(graph_steps)
    phase_slope, node_direction_support = directed_phase_delay_scores(
        analytic,
        evidence_centres,
        max_delay_steps=max_delay_steps,
        timestep_seconds=timestep_seconds,
    )
    evidence_bands = len(evidence_band_edges_hz)
    nodes = carrier.shape[2]
    target_band = np.repeat(np.arange(evidence_bands), nodes)
    same_band = target_band[:, None] == target_band[None, :]
    target_node = np.tile(np.arange(nodes), evidence_bands)
    source_node = target_node.copy()
    shared_group_delay = phase_slope[:, target_node[:, None], source_node[None, :], :]
    scores = np.where(same_band[None, ..., None], shared_group_delay, scores)
    entity_direction_support = node_direction_support[:, target_node[:, None], source_node[None, :]]
    scores[..., 1:] = np.where(
        same_band[None, ..., None],
        scores[..., 1:],
        scores[..., 1:] * (0.5 + 0.5 * entity_direction_support[..., None]),
    )
    fitted = fit_hurdle_evidence(
        scores,
        HurdleEvidenceConfig(
            max_delay_steps=max_delay_steps,
            bootstrap_samples=bootstrap_samples,
            random_seed=seed,
        ),
    )
    control_bootstraps = max(8, min(32, int(bootstrap_samples)))
    control_config = HurdleEvidenceConfig(
        max_delay_steps=max_delay_steps,
        bootstrap_samples=control_bootstraps,
        random_seed=seed + 1009,
    )
    reversed_analytic = analytic[..., ::-1].conj().copy()
    reversed_scores, reversed_support = directed_phase_delay_scores(
        reversed_analytic,
        evidence_centres,
        max_delay_steps=max_delay_steps,
        timestep_seconds=timestep_seconds,
    )
    surrogate_scores, surrogate_support = directed_phase_delay_scores(
        complex_phase_surrogate(analytic, seed + 2017),
        evidence_centres,
        max_delay_steps=max_delay_steps,
        timestep_seconds=timestep_seconds,
    )
    normal_node_fit = fit_hurdle_evidence(phase_slope, control_config)
    reversed_node_fit = fit_hurdle_evidence(reversed_scores, control_config)
    surrogate_node_fit = fit_hurdle_evidence(surrogate_scores, control_config)
    route = (fitted["accepted"].astype(np.float32) * (1.0 - fitted["null_probability"])).reshape(
        evidence_bands, nodes, evidence_bands, nodes
    )
    positive = fitted["positive_transport_probability"].reshape(
        evidence_bands, nodes, evidence_bands, nodes, max_delay_steps + 1
    )
    fraction = fitted["fractional_delay_target"].reshape(
        evidence_bands, nodes, evidence_bands, nodes
    )
    evidence_centres = np.asarray(evidence_band_edges_hz, dtype=np.float64).mean(axis=1)
    output_centres = np.asarray(output_band_edges_hz, dtype=np.float64).mean(axis=1)
    # A one-hot physiological band assignment preserves accepted-route
    # calibration. Gaussian interpolation made every output route weakly
    # non-zero and destroyed the hurdle target.
    projection = np.zeros((output_centres.size, evidence_centres.size), dtype=np.float64)
    nearest = np.abs(output_centres[:, None] - evidence_centres[None]).argmin(axis=1)
    projection[np.arange(output_centres.size), nearest] = 1.0
    route_out = np.einsum("ae,eifj,bf->abij", projection, route, projection)
    positive_out = np.einsum("ae,eifjd,bf->abijd", projection, positive, projection)
    positive_out /= positive_out.sum(axis=-1, keepdims=True).clip(1e-8)
    fraction_out = np.einsum("ae,eifj,bf->abij", projection, fraction * route, projection)
    fraction_out /= route_out.clip(1e-8)
    for band in range(len(output_band_edges_hz)):
        np.fill_diagonal(route_out[band, band], 0.0)
    arrays = {
        "route_probability": route_out.astype(np.float32),
        "positive_delay_probability": positive_out.astype(np.float32),
        "fractional_delay_target": np.clip(fraction_out, 0.0, 1.0).astype(np.float32),
        "connectivity_prior": np.broadcast_to(
            node_direction_support.mean(axis=0)[None, None],
            (len(output_band_edges_hz), len(output_band_edges_hz), nodes, nodes),
        )
        .copy()
        .astype(np.float32),
    }
    summary = {
        "fit_scope": "inner_training_fold_only",
        "n_trials": int(x.shape[0]),
        "evidence_bands": evidence_bands,
        "output_bands": len(output_band_edges_hz),
        "nodes": nodes,
        "accepted_evidence_edges": int(fitted["accepted"].sum()),
        "projected_route_mean": float(route_out.mean()),
        "heldout_data_accessed": False,
        "task_tmin": float(task_tmin),
        "task_tmax": float(task_tmax),
        "delay_estimator": "12_band_group_phase_slope_with_nuisance_intercept",
        "phase_residual_during_delay_fit": 0.0,
        "route_direction_evidence": "phase_slope_plus_PSI_and_imaginary_coherency",
        "mean_psi_imaginary_support": float(node_direction_support.mean()),
        "time_reversal_transpose_support_mae": float(
            np.mean(np.abs(node_direction_support.transpose(0, 2, 1) - reversed_support))
        ),
        "phase_surrogate_support_drop": float(
            node_direction_support.mean() - surrogate_support.mean()
        ),
        "normal_node_edges": int(normal_node_fit["accepted"].sum()),
        "time_reversed_node_edges": int(reversed_node_fit["accepted"].sum()),
        "phase_surrogate_node_edges": int(surrogate_node_fit["accepted"].sum()),
        "negative_control_bootstraps": control_bootstraps,
    }
    return arrays, summary


def fold_local_continuous_delay_prior(
    data: dict[str, Any],
    evidence_band_edges_hz: list[list[float]],
    output_band_edges_hz: list[list[float]],
    n_nodes: int,
    graph_steps: int,
    max_delay_steps: int,
    bootstrap_samples: int = 128,
    grid_oversample: int = 4,
    min_bayes_factor: float = 3.0,
    min_bootstrap_frequency: float = 0.70,
    min_direction_probability: float = 0.80,
    seed: int = 0,
    device: str = "cpu",
    task_tmin: float = 0.0,
    task_tmax: float = 4.0,
    analytic_features: np.ndarray | None = None,
    analytic_representation: str = "legacy_fixed_evidence_projector",
    split_strata: Sequence[str] | None = None,
    surrogate_replicates: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit the V5 continuous signed-delay prior on one inner-training fold."""

    x = np.asarray(data["X"], dtype=np.float32)
    if analytic_features is None:
        analytic = model_evidence_analytic_features(
            x,
            float(data.get("sfreq", 250.0)),
            evidence_band_edges_hz,
            n_nodes=n_nodes,
            graph_steps=graph_steps,
            device=device,
            epoch_tmin=float(data.get("epoch_tmin", 0.0)),
            task_tmin=float(task_tmin),
            task_tmax=float(task_tmax),
            baseline_normalize=True,
            electrode_coordinates=data.get("electrode_coordinates"),
            spatial_anchor_indices=data.get("spatial_anchor_indices"),
        )
    else:
        analytic = np.asarray(analytic_features, dtype=np.complex64)
        expected_shape = (
            x.shape[0],
            len(evidence_band_edges_hz),
            int(n_nodes),
            int(graph_steps),
        )
        if analytic.shape != expected_shape:
            raise ValueError(
                "precomputed analytic evidence has shape "
                f"{analytic.shape}, expected {expected_shape}"
            )
        if not np.isfinite(analytic.real).all() or not np.isfinite(analytic.imag).all():
            raise ValueError("precomputed analytic evidence must be finite")
    frequencies = np.asarray(evidence_band_edges_hz, dtype=np.float32).mean(axis=1)
    timestep_seconds = (float(task_tmax) - float(task_tmin)) / float(graph_steps)
    estimator_config = ContinuousDelayConfig(
        max_delay_steps=max_delay_steps,
        grid_oversample=grid_oversample,
        bootstrap_samples=bootstrap_samples,
        min_bayes_factor=min_bayes_factor,
        min_bootstrap_frequency=min_bootstrap_frequency,
        min_direction_probability=min_direction_probability,
        random_seed=seed,
    )

    direction_support, psi, imaginary = trial_psi_imaginary_support(analytic)
    normal_scores, delay_grid, coherence_strength = trial_continuous_phase_scores(
        analytic,
        frequencies,
        timestep_seconds=timestep_seconds,
        config=estimator_config,
    )
    (
        surrogate_analytics,
        surrogate_scores,
        surrogate_direction_support,
        surrogate_strength,
        surrogate_grid,
    ) = continuous_phase_surrogate_bank(
        analytic,
        frequencies,
        timestep_seconds=timestep_seconds,
        config=estimator_config,
        seed=seed + 2017,
        replicates=int(surrogate_replicates),
        retain_analytic=True,
    )
    if not np.array_equal(delay_grid, surrogate_grid):
        raise RuntimeError("normal and surrogate delay grids diverged")
    node_fit = fit_continuous_delay_evidence(
        normal_scores,
        surrogate_scores,
        delay_grid,
        estimator_config,
        direction_support=direction_support,
        null_direction_support=surrogate_direction_support,
    )

    (
        _,
        second_surrogate_scores,
        second_surrogate_direction_support,
        _,
        second_surrogate_grid,
    ) = continuous_phase_surrogate_bank(
        analytic,
        frequencies,
        timestep_seconds=timestep_seconds,
        config=estimator_config,
        seed=seed + 3011,
        replicates=int(surrogate_replicates),
    )
    if not np.array_equal(delay_grid, second_surrogate_grid):
        raise RuntimeError("secondary surrogate delay grid diverged")
    surrogate_fit = fit_continuous_delay_evidence(
        surrogate_scores,
        second_surrogate_scores,
        delay_grid,
        ContinuousDelayConfig(**{**estimator_config.__dict__, "random_seed": seed + 1}),
        direction_support=surrogate_direction_support,
        null_direction_support=second_surrogate_direction_support,
    )

    reversed_analytic = analytic[..., ::-1].conj().copy()
    reversed_direction_support, _, _ = trial_psi_imaginary_support(reversed_analytic)
    reversed_scores, reversed_grid, _ = trial_continuous_phase_scores(
        reversed_analytic,
        frequencies,
        timestep_seconds=timestep_seconds,
        config=estimator_config,
    )
    (
        _,
        reversed_null_scores,
        reversed_null_direction_support,
        _,
        reversed_null_grid,
    ) = continuous_phase_surrogate_bank(
        reversed_analytic,
        frequencies,
        timestep_seconds=timestep_seconds,
        config=estimator_config,
        seed=seed + 4013,
        replicates=int(surrogate_replicates),
    )
    if not np.array_equal(reversed_grid, reversed_null_grid):
        raise RuntimeError("time-reversal surrogate delay grid diverged")
    reversed_fit = fit_continuous_delay_evidence(
        reversed_scores,
        reversed_null_scores,
        reversed_grid,
        ContinuousDelayConfig(**{**estimator_config.__dict__, "random_seed": seed + 2}),
        direction_support=reversed_direction_support,
        null_direction_support=reversed_null_direction_support,
    )

    first_indices, second_indices = stratified_split_half_indices(
        analytic.shape[0], split_strata
    )
    first_fit = fit_continuous_delay_evidence(
        normal_scores[first_indices],
        surrogate_scores[first_indices],
        delay_grid,
        ContinuousDelayConfig(**{**estimator_config.__dict__, "random_seed": seed + 3}),
        direction_support=direction_support[first_indices],
        null_direction_support=surrogate_direction_support[first_indices],
    )
    second_fit = fit_continuous_delay_evidence(
        normal_scores[second_indices],
        surrogate_scores[second_indices],
        delay_grid,
        ContinuousDelayConfig(**{**estimator_config.__dict__, "random_seed": seed + 4}),
        direction_support=direction_support[second_indices],
        null_direction_support=surrogate_direction_support[second_indices],
    )
    split_mask = first_fit["accepted"] & second_fit["accepted"]
    split_common_correlation = _masked_correlation(
        first_fit["signed_delay_mean"],
        second_fit["signed_delay_mean"],
        split_mask,
    )
    split_union = first_fit["accepted"] | second_fit["accepted"]
    split_edge_jaccard = float(split_mask.sum() / max(1, split_union.sum()))
    off_diagonal_nodes = ~np.eye(analytic.shape[2], dtype=bool)
    split_all_correlation = _masked_correlation(
        first_fit["signed_delay_mean"],
        second_fit["signed_delay_mean"],
        off_diagonal_nodes,
    )
    split_union_correlation = _masked_correlation(
        first_fit["signed_delay_mean"],
        second_fit["signed_delay_mean"],
        split_union,
    )
    split_full_selected_correlation = _masked_correlation(
        first_fit["signed_delay_mean"],
        second_fit["signed_delay_mean"],
        node_fit["accepted"],
    )
    reversal_mask = node_fit["accepted"].T & reversed_fit["accepted"]
    reversal_union = node_fit["accepted"].T | reversed_fit["accepted"]
    reversal_edge_jaccard = float(reversal_mask.sum() / max(1, reversal_union.sum()))
    reversal_correlation = _masked_correlation(
        node_fit["signed_delay_mean"].T,
        reversed_fit["signed_delay_mean"],
        reversal_mask,
    )

    carrier = analytic.real.astype(np.float32)
    envelope = np.log(np.abs(analytic).clip(1e-4)).astype(np.float32)
    node_selection = node_fit["accepted"]
    band_scores = trial_band_pair_scores_at_node_delay(
        carrier,
        envelope,
        node_fit["signed_delay_map"],
        node_selection,
    )
    null_band_scores = np.zeros_like(band_scores, dtype=np.float64)
    for surrogate_analytic in surrogate_analytics:
        null_band_scores += trial_band_pair_scores_at_node_delay(
            surrogate_analytic.real.astype(np.float32),
            np.log(np.abs(surrogate_analytic).clip(1e-4)).astype(np.float32),
            node_fit["signed_delay_map"],
            node_selection,
        )
    null_band_scores = (null_band_scores / float(surrogate_replicates)).astype(
        np.float32
    )
    band_fit = fit_band_pair_route_evidence(band_scores, null_band_scores, estimator_config)
    evidence_route = band_fit["route_probability"] * node_fit["route_probability"][None, None]
    evidence_bands = len(evidence_band_edges_hz)
    nodes = analytic.shape[2]
    evidence_positive = np.broadcast_to(
        node_fit["positive_delay_probability"][None, None],
        (evidence_bands, evidence_bands, nodes, nodes, max_delay_steps + 1),
    ).copy()
    evidence_fraction = np.broadcast_to(
        node_fit["fractional_delay_target"][None, None],
        (evidence_bands, evidence_bands, nodes, nodes),
    ).copy()

    evidence_centres = np.asarray(evidence_band_edges_hz, dtype=np.float64).mean(axis=1)
    output_centres = np.asarray(output_band_edges_hz, dtype=np.float64).mean(axis=1)
    projection = np.zeros((output_centres.size, evidence_centres.size), dtype=np.float64)
    nearest = np.abs(output_centres[:, None] - evidence_centres[None]).argmin(axis=1)
    projection[np.arange(output_centres.size), nearest] = 1.0
    route_out = np.einsum("ae,efij,bf->abij", projection, evidence_route, projection)
    positive_out = np.einsum("ae,efijd,bf->abijd", projection, evidence_positive, projection)
    positive_out /= positive_out.sum(axis=-1, keepdims=True).clip(1e-8)
    fraction_out = np.einsum("ae,efij,bf->abij", projection, evidence_fraction, projection)
    direction_out = np.broadcast_to(
        node_fit["psi_imaginary_support"][None, None], route_out.shape
    ).copy()
    for band in range(len(output_band_edges_hz)):
        np.fill_diagonal(route_out[band, band], 0.0)

    normal_route = node_fit["route_probability"]
    reversed_route = reversed_fit["route_probability"]
    route_reversal_mae = float(np.mean(np.abs(normal_route.T - reversed_route)))
    reversal_mae = float(
        np.mean(np.abs(direction_support.mean(axis=0).T - reversed_direction_support.mean(axis=0)))
    )
    selected = node_fit["accepted"]
    if np.any(selected):
        normal_support = normal_scores.max(axis=-1).mean(axis=0)[selected]
        null_support = surrogate_scores.max(axis=-1).mean(axis=0)[selected]
        surrogate_drop = float(np.mean(normal_support - null_support))
        psi_surrogate_drop = float(
            np.mean(
                direction_support.mean(axis=0)[selected]
                - surrogate_direction_support.mean(axis=0)[selected]
            )
        )
        psi_surrogate_scope = "accepted_node_edges"
    else:
        surrogate_drop = float(
            normal_scores.max(axis=-1).mean() - surrogate_scores.max(axis=-1).mean()
        )
        fallback_mask = ~np.eye(nodes, dtype=bool)
        psi_surrogate_drop = float(
            direction_support[:, fallback_mask].mean()
            - surrogate_direction_support[:, fallback_mask].mean()
        )
        psi_surrogate_scope = "all_off_diagonal_edges_no_route_accepted"

    selected_bootstrap_frequency = np.minimum.reduce(
        (
            node_fit["bootstrap_frequency"],
            node_fit["positive_delay_bootstrap_frequency"],
            node_fit["direction_bootstrap_frequency"],
            node_fit["psi_imaginary_bootstrap_frequency"],
        )
    )[selected]
    selected_normal_log_bf = np.log(node_fit["bayes_factor"][selected].clip(1e-8))
    selected_surrogate_log_bf = np.log(surrogate_fit["bayes_factor"][selected].clip(1e-8))
    if selected_normal_log_bf.size >= 3:
        try:
            phase_surrogate_p = float(
                stats.wilcoxon(
                    selected_normal_log_bf,
                    selected_surrogate_log_bf,
                    alternative="greater",
                ).pvalue
            )
        except ValueError:
            phase_surrogate_p = 1.0
        phase_surrogate_log_bf_drop = float(
            np.median(selected_normal_log_bf - selected_surrogate_log_bf)
        )
    else:
        phase_surrogate_p = 1.0
        phase_surrogate_log_bf_drop = float("nan")
    off_diagonal = ~np.eye(nodes, dtype=bool)
    evidence_checks = {
        "natural_nonzero_edges": bool(0 < int(selected.sum()) < int(off_diagonal.sum())),
        "split_half_delay": bool(
            int(split_union.sum()) >= 3
            and np.isfinite(split_union_correlation)
            and split_union_correlation >= 0.50
            and np.isfinite(split_all_correlation)
            and split_all_correlation >= 0.50
            and split_edge_jaccard >= 0.50
        ),
        "bootstrap_frequency": bool(selected_bootstrap_frequency.size)
        and float(selected_bootstrap_frequency.min()) >= min_bootstrap_frequency,
        "time_reversal": bool(
            int(reversal_union.sum()) >= 3
            and np.isfinite(reversal_correlation)
            and reversal_correlation >= 0.50
            and reversal_edge_jaccard >= 0.50
        ),
        "phase_surrogate": bool(
            phase_surrogate_p < 0.05
            and phase_surrogate_log_bf_drop >= math.log(1.5)
            and int(surrogate_fit["accepted"].sum()) < int(selected.sum())
        ),
    }

    arrays = {
        "route_probability": route_out.astype(np.float32),
        "positive_delay_probability": positive_out.astype(np.float32),
        "fractional_delay_target": np.clip(fraction_out, 0.0, 1.0).astype(np.float32),
        "connectivity_prior": direction_out.astype(np.float32),
        "signed_node_delay_map": node_fit["signed_delay_map"].astype(np.float32),
        "signed_node_delay_mean": node_fit["signed_delay_mean"].astype(np.float32),
        "signed_delay_grid_steps": delay_grid.astype(np.float32),
        "signed_node_delay_probability": node_fit["signed_delay_probability"].astype(np.float32),
        "signed_node_bayes_factor": node_fit["bayes_factor"].astype(np.float32),
        "signed_node_null_probability": node_fit["null_probability"].astype(np.float32),
        "signed_node_bootstrap_frequency": node_fit["bootstrap_frequency"].astype(np.float32),
        "signed_node_positive_delay_bayes_factor": node_fit["positive_delay_bayes_factor"].astype(
            np.float32
        ),
        "signed_node_positive_delay_bootstrap_frequency": node_fit[
            "positive_delay_bootstrap_frequency"
        ].astype(np.float32),
        "signed_node_zero_delay_probability": node_fit["zero_delay_probability"].astype(np.float32),
        "signed_node_direction_probability": node_fit["direction_probability"].astype(np.float32),
        "signed_node_direction_bootstrap_frequency": node_fit[
            "direction_bootstrap_frequency"
        ].astype(np.float32),
        "signed_node_psi_imaginary_support": node_fit["psi_imaginary_support"].astype(np.float32),
        "signed_node_psi_imaginary_bayes_factor": node_fit["psi_imaginary_bayes_factor"].astype(
            np.float32
        ),
        "signed_node_psi_imaginary_bootstrap_frequency": node_fit[
            "psi_imaginary_bootstrap_frequency"
        ].astype(np.float32),
        "signed_node_accepted": node_fit["accepted"].astype(np.uint8),
        "band_pair_bayes_factor": band_fit["bayes_factor"].astype(np.float32),
        "band_pair_bootstrap_frequency": band_fit["bootstrap_frequency"].astype(np.float32),
        "band_pair_accepted": band_fit["accepted"].astype(np.uint8),
        "split_first_signed_node_delay_mean": first_fit["signed_delay_mean"].astype(
            np.float32
        ),
        "split_second_signed_node_delay_mean": second_fit["signed_delay_mean"].astype(
            np.float32
        ),
        "split_first_signed_node_accepted": first_fit["accepted"].astype(np.uint8),
        "split_second_signed_node_accepted": second_fit["accepted"].astype(np.uint8),
        "time_reversed_signed_node_delay_map": reversed_fit["signed_delay_map"].astype(
            np.float32
        ),
        "time_reversed_signed_node_delay_mean": reversed_fit["signed_delay_mean"].astype(
            np.float32
        ),
        "time_reversed_signed_node_accepted": reversed_fit["accepted"].astype(np.uint8),
        "time_reversed_signed_node_route_probability": reversed_fit[
            "route_probability"
        ].astype(np.float32),
        "phase_surrogate_signed_node_delay_map": surrogate_fit["signed_delay_map"].astype(
            np.float32
        ),
        "phase_surrogate_signed_node_delay_mean": surrogate_fit["signed_delay_mean"].astype(
            np.float32
        ),
        "phase_surrogate_signed_node_accepted": surrogate_fit["accepted"].astype(np.uint8),
        "phase_surrogate_signed_node_bayes_factor": surrogate_fit["bayes_factor"].astype(
            np.float32
        ),
    }
    summary = {
        "fit_scope": "inner_training_fold_only",
        "n_trials": int(x.shape[0]),
        "analytic_representation": str(analytic_representation),
        "precomputed_analytic_features": analytic_features is not None,
        "evidence_bands": evidence_bands,
        "output_bands": len(output_band_edges_hz),
        "nodes": nodes,
        "accepted_evidence_edges": int(band_fit["accepted"].sum()),
        "accepted_node_edges": int(node_fit["accepted"].sum()),
        "projected_route_mean": float(route_out.mean()),
        "heldout_data_accessed": False,
        "task_tmin": float(task_tmin),
        "task_tmax": float(task_tmax),
        "delay_estimator": "continuous_signed_phase_profile_independence_null",
        "delay_grid_oversample": int(grid_oversample),
        "minimum_bayes_factor": float(min_bayes_factor),
        "minimum_bootstrap_frequency": float(min_bootstrap_frequency),
        "minimum_direction_probability": float(min_direction_probability),
        "phase_surrogate_replicates": int(surrogate_replicates),
        "phase_surrogate_aggregation": "per_trial_monte_carlo_mean",
        "minimum_selected_bootstrap_frequency": float(selected_bootstrap_frequency.min())
        if selected_bootstrap_frequency.size
        else float("nan"),
        "split_half_common_node_edges": int(split_mask.sum()),
        "split_half_delay_correlation": split_union_correlation,
        "split_half_delay_scope": "union_of_independently_selected_node_edges",
        "split_half_scheme": (
            "run_class_stratified_deterministic_alternation"
            if split_strata is not None
            else "legacy_trial_order_alternation"
        ),
        "split_half_strata_count": len(set(str(value) for value in split_strata))
        if split_strata is not None
        else 0,
        "split_half_first_trials": int(first_indices.size),
        "split_half_second_trials": int(second_indices.size),
        "split_half_common_accepted_delay_correlation": split_common_correlation,
        "split_half_first_node_edges": int(first_fit["accepted"].sum()),
        "split_half_second_node_edges": int(second_fit["accepted"].sum()),
        "split_half_union_node_edges": int(split_union.sum()),
        "split_half_edge_jaccard": split_edge_jaccard,
        "minimum_split_half_edge_jaccard": 0.50,
        "split_half_all_off_diagonal_delay_correlation": split_all_correlation,
        "split_half_union_delay_correlation": split_union_correlation,
        "split_half_full_selected_delay_correlation": split_full_selected_correlation,
        "time_reversal_common_node_edges": int(reversal_mask.sum()),
        "time_reversal_union_node_edges": int(reversal_union.sum()),
        "time_reversal_edge_jaccard": reversal_edge_jaccard,
        "minimum_time_reversal_edge_jaccard": 0.50,
        "time_reversal_transpose_delay_correlation": reversal_correlation,
        "phase_surrogate_log_bf_drop": phase_surrogate_log_bf_drop,
        "phase_surrogate_wilcoxon_p": phase_surrogate_p,
        "phase_surrogate_tested_node_edges": int(selected.sum()),
        "evidence_checks": evidence_checks,
        "evidence_pipeline_passed": bool(all(evidence_checks.values())),
        "zero_delay_is_route_null": False,
        "phase_residual_during_delay_fit": 0.0,
        "route_direction_evidence": (
            "independence_route_BF_plus_conditional_positive_delay_BF_plus_PSI_imaginary"
        ),
        "mean_coherence_strength": float(coherence_strength.mean()),
        "mean_surrogate_coherence_strength": float(surrogate_strength.mean()),
        "mean_zero_delay_probability": float(
            node_fit["zero_delay_probability"][off_diagonal].mean()
        ),
        "median_zero_delay_probability": float(
            np.median(node_fit["zero_delay_probability"][off_diagonal])
        ),
        "positive_delay_bf_nodes": int(
            (
                node_fit["positive_delay_bayes_factor"][off_diagonal]
                >= estimator_config.min_bayes_factor
            ).sum()
        ),
        "mean_psi_imaginary_support": float(direction_support[:, off_diagonal].mean()),
        "mean_surrogate_psi_imaginary_support": float(
            surrogate_direction_support[:, off_diagonal].mean()
        ),
        "mean_absolute_psi": float(np.abs(psi[:, off_diagonal]).mean()),
        "mean_imaginary_coherency": float(imaginary[:, off_diagonal].mean()),
        "time_reversal_route_transpose_mae": route_reversal_mae,
        "time_reversal_transpose_support_mae": reversal_mae,
        "phase_profile_surrogate_drop": surrogate_drop,
        "global_phase_surrogate_support_drop": float(
            direction_support[:, off_diagonal].mean()
            - surrogate_direction_support[:, off_diagonal].mean()
        ),
        "phase_surrogate_support_drop": psi_surrogate_drop,
        "phase_surrogate_support_scope": psi_surrogate_scope,
        "normal_node_edges": int(node_fit["accepted"].sum()),
        "time_reversed_node_edges": int(reversed_fit["accepted"].sum()),
        "phase_surrogate_node_edges": int(surrogate_fit["accepted"].sum()),
        "negative_control_bootstraps": int(bootstrap_samples),
        "source_target_band_pairs_preserved": True,
    }
    return arrays, summary


def _masked_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) < 3:
        return float("nan")
    x, y = np.asarray(a)[mask], np.asarray(b)[mask]
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def evaluate_evidence_space(
    normal_scores: np.ndarray,
    reversed_scores: np.ndarray,
    surrogate_scores: np.ndarray,
    config: HurdleEvidenceConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    normal = fit_hurdle_evidence(normal_scores, config)
    first = fit_hurdle_evidence(normal_scores[::2], config)
    second = fit_hurdle_evidence(
        normal_scores[1::2],
        HurdleEvidenceConfig(**{**config.__dict__, "random_seed": config.random_seed + 1}),
    )
    reversed_fit = fit_hurdle_evidence(
        reversed_scores,
        HurdleEvidenceConfig(**{**config.__dict__, "random_seed": config.random_seed + 2}),
    )
    surrogate = fit_hurdle_evidence(
        surrogate_scores,
        HurdleEvidenceConfig(**{**config.__dict__, "random_seed": config.random_seed + 3}),
    )
    split_mask = first["accepted"] & second["accepted"]
    reversal_mask = normal["accepted"].T & reversed_fit["accepted"]
    split_corr = _masked_correlation(first["delay_mean"], second["delay_mean"], split_mask)
    reversal_corr = _masked_correlation(
        normal["delay_mean"].T, reversed_fit["delay_mean"], reversal_mask
    )
    off_diagonal = ~np.eye(normal["accepted"].shape[0], dtype=bool)
    # Evaluate the negative control on routes selected before looking at the
    # surrogate. Using all routes makes the median identically zero because
    # the overwhelming majority are null under both conditions.
    surrogate_mask = normal["accepted"] & off_diagonal
    normal_log_bf = np.log(normal["bayes_factor"][surrogate_mask].clip(1e-8))
    surrogate_log_bf = np.log(surrogate["bayes_factor"][surrogate_mask].clip(1e-8))
    if normal_log_bf.size >= 3:
        try:
            surrogate_p = float(
                stats.wilcoxon(normal_log_bf, surrogate_log_bf, alternative="greater").pvalue
            )
        except ValueError:
            surrogate_p = 1.0
        log_bf_drop = float(np.median(normal_log_bf - surrogate_log_bf))
    else:
        surrogate_p = 1.0
        log_bf_drop = float("nan")
    selected_frequency = normal["bootstrap_frequency"][normal["accepted"]]
    selected_null = normal["null_probability"][normal["accepted"]]
    selected_bf = normal["bayes_factor"][normal["accepted"]]
    off_diagonal_null = normal["null_probability"][off_diagonal]
    checks = {
        "natural_nonzero_edges": bool(0 < int(normal["accepted"].sum()) < int(off_diagonal.sum())),
        "split_half_delay": bool(np.isfinite(split_corr) and split_corr >= 0.50),
        "bootstrap_frequency": bool(selected_frequency.size)
        and float(selected_frequency.min()) >= 0.70,
        "time_reversal": bool(np.isfinite(reversal_corr) and reversal_corr >= 0.50),
        "phase_surrogate": bool(surrogate_p < 0.05 and log_bf_drop >= math.log(1.5)),
    }
    summary = {
        "accepted_edges": int(normal["accepted"].sum()),
        "accepted_density": float(normal["accepted"].sum() / off_diagonal.sum()),
        "mean_off_diagonal_null_probability": float(off_diagonal_null.mean()),
        "mean_selected_null_probability": float(selected_null.mean())
        if selected_null.size
        else float("nan"),
        "median_selected_nonzero_bayes_factor": float(np.median(selected_bf))
        if selected_bf.size
        else float("nan"),
        "minimum_selected_nonzero_bayes_factor": float(selected_bf.min())
        if selected_bf.size
        else float("nan"),
        "minimum_selected_bootstrap_frequency": float(selected_frequency.min())
        if selected_frequency.size
        else float("nan"),
        "split_half_common_edges": int(split_mask.sum()),
        "split_half_delay_correlation": split_corr,
        "time_reversal_common_edges": int(reversal_mask.sum()),
        "time_reversal_transpose_correlation": reversal_corr,
        "phase_surrogate_log_bf_drop": log_bf_drop,
        "phase_surrogate_wilcoxon_p": surrogate_p,
        "phase_surrogate_tested_edges": int(surrogate_mask.sum()),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
    arrays = {
        **{f"normal_{key}": value for key, value in normal.items()},
        **{f"reversed_{key}": value for key, value in reversed_fit.items()},
        **{f"surrogate_{key}": value for key, value in surrogate.items()},
    }
    return summary, arrays
