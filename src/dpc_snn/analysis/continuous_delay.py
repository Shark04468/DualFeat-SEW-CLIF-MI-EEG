"""Continuous signed delay evidence with an explicit independence null.

The estimator profiles a nuisance phase intercept while fitting one signed
group delay across frequency.  Route existence is tested against phase-
surrogated data, not against the zero-delay member of the coupled model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ContinuousDelayConfig:
    """Scientific contract for classifier-free signed-delay estimation."""

    max_delay_steps: int = 8
    grid_oversample: int = 4
    bootstrap_samples: int = 256
    min_bayes_factor: float = 3.0
    min_bootstrap_frequency: float = 0.70
    min_direction_probability: float = 0.80
    random_seed: int = 0

    def delay_grid_steps(self) -> np.ndarray:
        if self.max_delay_steps < 1:
            raise ValueError("max_delay_steps must be positive")
        if self.grid_oversample < 1:
            raise ValueError("grid_oversample must be positive")
        extent = self.max_delay_steps * self.grid_oversample
        return np.arange(-extent, extent + 1, dtype=np.float64) / float(self.grid_oversample)


def trial_continuous_phase_scores(
    analytic: np.ndarray,
    band_frequencies_hz: np.ndarray,
    *,
    timestep_seconds: float,
    config: ContinuousDelayConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Profile trial-wise phase concentration over a signed delay grid.

    Args:
        analytic: Complex analytic evidence ``[trial, band, node, time]``.
        band_frequencies_hz: Centre frequency for each evidence band.
        timestep_seconds: Duration represented by one graph step.
        config: Signed-delay grid configuration.

    Returns:
        Concentration scores ``[trial, target, source, delay]``, the signed
        delay grid in graph steps, and mean cross-frequency coherence strength
        ``[trial, target, source]``.
    """

    values = np.asarray(analytic, dtype=np.complex64)
    frequencies = np.asarray(band_frequencies_hz, dtype=np.float64)
    if values.ndim != 4 or values.shape[1] != frequencies.size:
        raise ValueError("analytic evidence must have shape [N, B, K, T]")
    if frequencies.size < 3:
        raise ValueError("continuous phase-slope delay requires at least three bands")
    if not np.all(np.diff(frequencies) > 0):
        raise ValueError("band frequencies must be strictly increasing")
    if timestep_seconds <= 0:
        raise ValueError("timestep_seconds must be positive")

    amplitude = np.abs(values)
    amplitude_floor = np.median(amplitude, axis=-1, keepdims=True).clip(1e-6)
    confidence = amplitude / (amplitude + amplitude_floor)
    phasor = confidence * values / amplitude.clip(1e-6)
    cross = np.einsum("nbkt,nblt->nbkl", phasor, phasor.conj(), optimize=True)
    cross /= float(values.shape[-1])
    weight = np.abs(cross).astype(np.float64).clip(1e-8)
    unit_cross = cross / np.abs(cross).clip(1e-8)

    delay_grid_steps = config.delay_grid_steps()
    delay_seconds = delay_grid_steps * float(timestep_seconds)
    rotation = np.exp(1j * 2.0 * np.pi * frequencies[:, None] * delay_seconds[None]).astype(
        np.complex64
    )
    aligned = unit_cross[..., None] * rotation[None, :, None, None, :]
    numerator = np.abs(np.sum(weight[..., None] * aligned, axis=1))
    denominator = np.sum(weight, axis=1)[..., None].clip(1e-8)
    score = numerator / denominator
    strength = np.mean(weight, axis=1)

    diagonal = np.arange(values.shape[2])
    score[:, diagonal, diagonal] = 0.0
    strength[:, diagonal, diagonal] = 0.0
    return (
        score.astype(np.float32),
        delay_grid_steps.astype(np.float32),
        strength.astype(np.float32),
    )


def _one_sample_bic_log_bf(difference: np.ndarray) -> np.ndarray:
    """BIC approximation for a positive paired mean against a zero null."""

    values = np.asarray(difference, dtype=np.float64)
    if values.shape[0] < 3:
        raise ValueError("at least three trials are required for route evidence")
    mean = values.mean(axis=0)
    centered = values - mean[None]
    rss_null = np.sum(values**2, axis=0).clip(1e-12)
    rss_alternative = np.sum(centered**2, axis=0).clip(1e-12)
    log_bf = 0.5 * (
        values.shape[0] * np.log(rss_null / rss_alternative) - math.log(values.shape[0])
    )
    return np.where(mean > 0.0, log_bf, -50.0)


def _delay_posterior(scores: np.ndarray) -> np.ndarray:
    """Convert trial-level profile scores into a calibrated categorical posterior."""

    values = np.asarray(scores, dtype=np.float64)
    mean = values.mean(axis=0)
    standard_error = values.std(axis=0, ddof=1) / math.sqrt(max(1, values.shape[0]))
    route_scale = np.median(standard_error, axis=-1, keepdims=True).clip(1e-3)
    logits = (mean - mean.max(axis=-1, keepdims=True)) / route_scale
    logits = np.clip(logits, -60.0, 0.0)
    probability = np.exp(logits)
    return probability / probability.sum(axis=-1, keepdims=True).clip(1e-12)


def _positive_base_distribution(
    posterior: np.ndarray,
    delay_grid_steps: np.ndarray,
    max_delay_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project positive continuous mass to integer bases plus one fraction target."""

    grid = np.asarray(delay_grid_steps, dtype=np.float64)
    positive_mask = grid > 0.0
    positive_mass = posterior[..., positive_mask]
    positive_grid = grid[positive_mask]
    normalizer = positive_mass.sum(axis=-1, keepdims=True)
    conditional = np.divide(
        positive_mass,
        normalizer,
        out=np.zeros_like(positive_mass),
        where=normalizer > 1e-12,
    )
    no_positive = normalizer[..., 0] <= 1e-12
    if np.any(no_positive):
        conditional[no_positive] = 1.0 / max(1, positive_grid.size)

    bases = np.floor(positive_grid).astype(np.int64).clip(0, max_delay_steps)
    fractions = positive_grid - np.floor(positive_grid)
    base_probability = np.zeros((*posterior.shape[:-1], max_delay_steps + 1), dtype=np.float64)
    for index, base in enumerate(bases):
        base_probability[..., base] += conditional[..., index]
    base_probability /= base_probability.sum(axis=-1, keepdims=True).clip(1e-12)
    fraction_target = np.sum(conditional * fractions, axis=-1)
    return base_probability.astype(np.float32), fraction_target.astype(np.float32)


def fit_continuous_delay_evidence(
    scores: np.ndarray,
    null_scores: np.ndarray,
    delay_grid_steps: np.ndarray,
    config: ContinuousDelayConfig,
    *,
    direction_support: np.ndarray | None = None,
    null_direction_support: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Fit route existence and a signed continuous delay posterior.

    The coupled model contains negative, zero, and positive delays.  The null
    model is supplied independently (normally from phase-surrogated analytic
    signals), so zero delay never competes with route existence.
    """

    observed = np.asarray(scores, dtype=np.float64)
    null = np.asarray(null_scores, dtype=np.float64)
    grid = np.asarray(delay_grid_steps, dtype=np.float64)
    if observed.shape != null.shape or observed.ndim != 4:
        raise ValueError("scores and null_scores must share [N, target, source, delay]")
    if observed.shape[-1] != grid.size:
        raise ValueError("delay grid does not match score profile")
    if not np.any(grid < 0.0) or not np.any(grid > 0.0) or not np.any(grid == 0.0):
        raise ValueError("continuous delay grid must contain negative, zero, and positive values")

    observed_best = observed.max(axis=-1)
    null_best = null.max(axis=-1)
    paired_difference = observed_best - null_best
    log_bf = _one_sample_bic_log_bf(paired_difference)
    posterior = _delay_posterior(observed)
    positive_direction_probability = posterior[..., grid > 0.0].sum(axis=-1)
    zero_index = int(np.flatnonzero(grid == 0.0)[0])
    zero_delay_probability = posterior[..., zero_index]

    # Route existence and conditional direction are different hypotheses.  The
    # former compares a coupled phase profile with an independent surrogate;
    # the latter asks whether a positive delay is better than every zero or
    # negative-delay explanation inside the coupled model.
    positive_best = observed[..., grid > 0.0].max(axis=-1)
    nonpositive_best = observed[..., grid <= 0.0].max(axis=-1)
    positive_difference = positive_best - nonpositive_best
    positive_log_bf = _one_sample_bic_log_bf(positive_difference)

    support_supplied = direction_support is not None or null_direction_support is not None
    if support_supplied:
        if direction_support is None or null_direction_support is None:
            raise ValueError(
                "direction_support and null_direction_support must be supplied together"
            )
        observed_support = np.asarray(direction_support, dtype=np.float64)
        null_support = np.asarray(null_direction_support, dtype=np.float64)
        expected_shape = observed.shape[:-1]
        if observed_support.shape != expected_shape or null_support.shape != expected_shape:
            raise ValueError("direction support must have shape [N, target, source]")
        support_difference = observed_support - null_support
        support_log_bf = _one_sample_bic_log_bf(support_difference)
        mean_direction_support = observed_support.mean(axis=0)
    else:
        support_difference = np.zeros(observed.shape[:-1], dtype=np.float64)
        support_log_bf = np.zeros(observed.shape[1:-1], dtype=np.float64)
        mean_direction_support = np.ones(observed.shape[1:-1], dtype=np.float64)

    diagonal = np.arange(posterior.shape[-2])
    posterior[diagonal, diagonal] = 0.0
    posterior[diagonal, diagonal, zero_index] = 1.0
    positive_direction_probability[diagonal, diagonal] = 0.0
    zero_delay_probability[diagonal, diagonal] = 1.0
    mean_direction_support[diagonal, diagonal] = 0.0

    rng = np.random.default_rng(config.random_seed)
    threshold = math.log(config.min_bayes_factor)
    candidate = log_bf >= threshold
    positive_candidate = positive_log_bf >= threshold
    support_candidate = (
        support_log_bf >= threshold if support_supplied else np.ones(candidate.shape, dtype=bool)
    )
    wins = np.zeros(candidate.shape, dtype=np.int64)
    positive_wins = np.zeros(candidate.shape, dtype=np.int64)
    direction_wins = np.zeros(candidate.shape, dtype=np.int64)
    support_wins = np.zeros(candidate.shape, dtype=np.int64)
    for _ in range(config.bootstrap_samples):
        indices = rng.integers(0, observed.shape[0], size=observed.shape[0])
        bootstrap_log_bf = _one_sample_bic_log_bf(paired_difference[indices])
        bootstrap_positive_log_bf = _one_sample_bic_log_bf(positive_difference[indices])
        bootstrap_posterior = _delay_posterior(observed[indices])
        bootstrap_direction = bootstrap_posterior[..., grid > 0.0].sum(axis=-1)
        wins += bootstrap_log_bf >= threshold
        positive_wins += bootstrap_positive_log_bf >= threshold
        direction_wins += bootstrap_direction >= config.min_direction_probability
        if support_supplied:
            bootstrap_support_log_bf = _one_sample_bic_log_bf(support_difference[indices])
            support_wins += bootstrap_support_log_bf >= threshold

    bootstrap_frequency = wins / max(1, config.bootstrap_samples)
    positive_bootstrap_frequency = positive_wins / max(1, config.bootstrap_samples)
    direction_frequency = direction_wins / max(1, config.bootstrap_samples)
    support_bootstrap_frequency = (
        support_wins / max(1, config.bootstrap_samples)
        if support_supplied
        else np.ones(candidate.shape, dtype=np.float64)
    )
    bayes_factor = np.exp(np.clip(log_bf, -50.0, 50.0))
    positive_bayes_factor = np.exp(np.clip(positive_log_bf, -50.0, 50.0))
    support_bayes_factor = np.exp(np.clip(support_log_bf, -50.0, 50.0))
    null_probability = 1.0 / (1.0 + bayes_factor)
    accepted = (
        candidate
        & (bootstrap_frequency >= config.min_bootstrap_frequency)
        & positive_candidate
        & (positive_bootstrap_frequency >= config.min_bootstrap_frequency)
        & (positive_direction_probability >= config.min_direction_probability)
        & (direction_frequency >= config.min_bootstrap_frequency)
        & support_candidate
        & (support_bootstrap_frequency >= config.min_bootstrap_frequency)
    )
    accepted[diagonal, diagonal] = False

    map_index = posterior.argmax(axis=-1)
    signed_delay_map = grid[map_index]
    signed_delay_mean = np.sum(posterior * grid, axis=-1)
    base_probability, fraction_target = _positive_base_distribution(
        posterior, grid, config.max_delay_steps
    )
    route_probability = (
        accepted.astype(np.float64) * (1.0 - null_probability) * positive_direction_probability
    )
    signed_delay_map[diagonal, diagonal] = 0.0
    signed_delay_mean[diagonal, diagonal] = 0.0
    base_probability[diagonal, diagonal] = 0.0
    base_probability[diagonal, diagonal, 0] = 1.0
    fraction_target[diagonal, diagonal] = 0.0
    bayes_factor[diagonal, diagonal] = 1.0
    log_bf[diagonal, diagonal] = 0.0
    positive_bayes_factor[diagonal, diagonal] = 1.0
    positive_log_bf[diagonal, diagonal] = 0.0
    positive_bootstrap_frequency[diagonal, diagonal] = 0.0
    support_bayes_factor[diagonal, diagonal] = 1.0
    support_log_bf[diagonal, diagonal] = 0.0
    support_bootstrap_frequency[diagonal, diagonal] = 0.0
    null_probability[diagonal, diagonal] = 1.0
    bootstrap_frequency[diagonal, diagonal] = 0.0
    direction_frequency[diagonal, diagonal] = 0.0

    return {
        "bayes_factor": bayes_factor.astype(np.float32),
        "log_bayes_factor": log_bf.astype(np.float32),
        "null_probability": null_probability.astype(np.float32),
        "bootstrap_frequency": bootstrap_frequency.astype(np.float32),
        "positive_delay_bayes_factor": positive_bayes_factor.astype(np.float32),
        "positive_delay_log_bayes_factor": positive_log_bf.astype(np.float32),
        "positive_delay_bootstrap_frequency": positive_bootstrap_frequency.astype(np.float32),
        "direction_bootstrap_frequency": direction_frequency.astype(np.float32),
        "direction_probability": positive_direction_probability.astype(np.float32),
        "zero_delay_probability": zero_delay_probability.astype(np.float32),
        "psi_imaginary_support": mean_direction_support.astype(np.float32),
        "psi_imaginary_bayes_factor": support_bayes_factor.astype(np.float32),
        "psi_imaginary_log_bayes_factor": support_log_bf.astype(np.float32),
        "psi_imaginary_bootstrap_frequency": support_bootstrap_frequency.astype(np.float32),
        "accepted": accepted,
        "route_probability": route_probability.astype(np.float32),
        "signed_delay_map": signed_delay_map.astype(np.float32),
        "signed_delay_mean": signed_delay_mean.astype(np.float32),
        "signed_delay_probability": posterior.astype(np.float32),
        "positive_delay_probability": base_probability,
        "fractional_delay_target": fraction_target,
        "delay_grid_steps": grid.astype(np.float32),
    }


def trial_band_pair_scores_at_node_delay(
    carrier: np.ndarray,
    envelope: np.ndarray,
    node_delay_steps: np.ndarray,
    node_selection: np.ndarray,
) -> np.ndarray:
    """Score true source-band to target-band routes at fitted node delays.

    Delay is identified across frequencies at node level.  This second stage
    tests which band pairs carry reproducible signal at that delay without
    refitting or discretely re-selecting a lag for every band pair.
    """

    carrier_values = np.asarray(carrier, dtype=np.float64)
    envelope_values = np.asarray(envelope, dtype=np.float64)
    delays = np.asarray(node_delay_steps, dtype=np.float64)
    selected = np.asarray(node_selection, dtype=bool)
    if carrier_values.shape != envelope_values.shape or carrier_values.ndim != 4:
        raise ValueError("carrier and envelope must share [N, B, K, T]")
    if delays.shape != (carrier_values.shape[2], carrier_values.shape[2]):
        raise ValueError("node_delay_steps must match [target_node, source_node]")
    if selected.shape != delays.shape:
        raise ValueError("node_selection must match node_delay_steps")

    n_trials, n_bands, n_nodes, _ = carrier_values.shape
    output = np.zeros((n_trials, n_bands, n_bands, n_nodes, n_nodes), dtype=np.float32)
    for target_node, source_node in np.argwhere(selected):
        delay = float(delays[target_node, source_node])
        if delay <= 0.0:
            continue
        base = int(math.floor(delay))
        fraction = delay - base
        start = base + (1 if fraction > 1e-8 else 0)
        if start >= carrier_values.shape[-1] - 2:
            continue

        correlations: list[np.ndarray] = []
        for values in (carrier_values, envelope_values):
            target = values[:, :, target_node, start:]
            source_base = values[:, :, source_node, start - base : values.shape[-1] - base]
            if fraction > 1e-8:
                source_previous = values[
                    :,
                    :,
                    source_node,
                    start - base - 1 : values.shape[-1] - base - 1,
                ]
                source = (1.0 - fraction) * source_base + fraction * source_previous
            else:
                source = source_base
            common = min(target.shape[-1], source.shape[-1])
            target = target[..., :common]
            source = source[..., :common]
            target = target - target.mean(axis=-1, keepdims=True)
            source = source - source.mean(axis=-1, keepdims=True)
            target /= np.sqrt(np.mean(target**2, axis=-1, keepdims=True)).clip(1e-8)
            source /= np.sqrt(np.mean(source**2, axis=-1, keepdims=True)).clip(1e-8)
            correlations.append(
                np.abs(np.einsum("nat,nbt->nab", target, source, optimize=True) / float(common))
            )
        same_band = np.eye(n_bands, dtype=bool)[None]
        route_score = np.where(same_band, correlations[0], correlations[1])
        output[..., target_node, source_node] = route_score.astype(np.float32)
    return output


def fit_band_pair_route_evidence(
    scores: np.ndarray,
    null_scores: np.ndarray,
    config: ContinuousDelayConfig,
) -> dict[str, np.ndarray]:
    """Test band-pair existence against an independently generated null."""

    observed = np.asarray(scores, dtype=np.float64)
    null = np.asarray(null_scores, dtype=np.float64)
    if observed.shape != null.shape or observed.ndim != 5:
        raise ValueError(
            "scores and null_scores must share [N, target_band, source_band, target_node, source_node]"
        )
    difference = observed - null
    log_bf = _one_sample_bic_log_bf(difference)
    candidate = log_bf >= math.log(config.min_bayes_factor)
    rng = np.random.default_rng(config.random_seed + 7919)
    wins = np.zeros(candidate.shape, dtype=np.int64)
    for _ in range(config.bootstrap_samples):
        indices = rng.integers(0, observed.shape[0], size=observed.shape[0])
        wins += _one_sample_bic_log_bf(difference[indices]) >= math.log(config.min_bayes_factor)
    frequency = wins / max(1, config.bootstrap_samples)
    accepted = candidate & (frequency >= config.min_bootstrap_frequency)
    for target_band in range(accepted.shape[0]):
        if target_band < accepted.shape[1]:
            np.fill_diagonal(accepted[target_band, target_band], False)
    bayes_factor = np.exp(np.clip(log_bf, -50.0, 50.0))
    route_probability = accepted * (bayes_factor / (1.0 + bayes_factor))
    return {
        "bayes_factor": bayes_factor.astype(np.float32),
        "log_bayes_factor": log_bf.astype(np.float32),
        "bootstrap_frequency": frequency.astype(np.float32),
        "accepted": accepted,
        "route_probability": route_probability.astype(np.float32),
    }
