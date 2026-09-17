"""Synthetic delay-phase oscillatory data for E1-E3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SyntheticDelayPhaseConfig:
    n_trials: int = 512
    n_channels: int = 22
    n_time: int = 256
    n_classes: int = 4
    sfreq: float = 250.0
    # Raw EEG samples.  Runners convert this to compressed SNN delay bins.
    max_delay: int = 16
    noise_std: float = 0.35
    style_std: float = 0.15
    phase_jitter: float = 0.25
    seed: int = 0
    # Non-harmonic carriers are required to identify delays longer than one
    # carrier cycle from cross-band phase slope.
    bands: tuple[float, ...] = (10.0, 18.0)


def make_sensor_positions(n_channels: int) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
    radius = 0.45 + 0.15 * (np.arange(n_channels) % 3)
    return np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1).astype(np.float32)


def global_delay_graph(n_channels: int, max_delay: int) -> tuple[np.ndarray, np.ndarray]:
    """Build one sparse directed graph with raw-sample edge delays.

    A DPC-SNN exports a single graph, so the synthetic target must also be a
    single graph.  Every active edge below directly contributes a delayed
    source waveform to its target; no row average is used in generation.
    """

    center = n_channels // 2
    c3 = max(0, center - 2)
    cz = center
    c4 = min(n_channels - 1, center + 2)
    delay = np.zeros((n_channels, n_channels), dtype=np.float32)
    edge = np.zeros_like(delay)
    base = max(1, max_delay // 4)
    path = [c3, cz, c4]
    for step, (src, dst) in enumerate(zip(path[:-1], path[1:], strict=False), start=1):
        edge[dst, src] = 1.0
        delay[dst, src] = min(max_delay, step * base)
    # Add a forward sparse graph. Reciprocal edges with different positive
    # delays are not identifiable from one pairwise cross-spectrum because
    # C_ij(f) is the conjugate of C_ji(f).
    for src in range(n_channels - 1):
        hop = 1 + (src % 2)
        dst = src + hop
        if dst < n_channels and edge[src, dst] == 0.0:
            edge[dst, src] = 1.0
            delay[dst, src] = min(max_delay, base + hop * max(1, base // 2))
    np.fill_diagonal(edge, 0.0)
    np.fill_diagonal(delay, 0.0)
    return edge.astype(np.float32), delay.astype(np.float32)


def _zero_padded_delay(signal: np.ndarray, delay: int) -> np.ndarray:
    if delay <= 0:
        return signal
    out = np.zeros_like(signal)
    out[delay:] = signal[:-delay]
    return out


def _class_amplitude_pattern(label: int, n_channels: int) -> np.ndarray:
    center = n_channels // 2
    anchors = [max(0, center - 2), min(n_channels - 1, center + 2), center, max(0, center - 4)]
    gain = np.ones(n_channels, dtype=np.float32)
    gain[anchors[label % len(anchors)]] = 1.45
    gain[anchors[(label + 1) % len(anchors)]] = 0.75
    return gain


def _smooth_random_envelope(rng: np.random.Generator, n_time: int) -> np.ndarray:
    """Create source-specific nonstationarity that makes long delays identifiable."""
    n_knots = max(6, min(16, n_time // 16))
    knot_time = np.linspace(0.0, n_time - 1, n_knots, dtype=np.float32)
    knot_value = rng.normal(0.0, 1.0, size=n_knots).astype(np.float32)
    smooth = np.interp(
        np.arange(n_time, dtype=np.float32), knot_time, knot_value
    ).astype(np.float32)
    smooth = (smooth - smooth.mean()) / max(float(smooth.std()), 1e-6)
    return np.exp(0.3 * smooth).astype(np.float32)


def generate_delay_phase_dataset(cfg: SyntheticDelayPhaseConfig | dict[str, Any]) -> dict[str, np.ndarray]:
    if isinstance(cfg, dict):
        cfg = SyntheticDelayPhaseConfig(**{k: v for k, v in cfg.items() if k in SyntheticDelayPhaseConfig.__annotations__})

    rng = np.random.default_rng(cfg.seed)
    t = np.arange(cfg.n_time, dtype=np.float32) / cfg.sfreq
    x = np.zeros((cfg.n_trials, cfg.n_channels, cfg.n_time), dtype=np.float32)
    y = np.arange(cfg.n_trials, dtype=np.int64) % cfg.n_classes
    rng.shuffle(y)
    edge_gt, delay_gt = global_delay_graph(cfg.n_channels, cfg.max_delay)
    phase_gt = 2 * np.pi * delay_gt / max(1, cfg.max_delay + 1)

    for n, label in enumerate(y):
        subject_style = rng.normal(0.0, cfg.style_std, size=(cfg.n_channels, 1)).astype(np.float32)
        amp_style = rng.lognormal(mean=0.0, sigma=cfg.style_std, size=(cfg.n_channels, 1)).astype(np.float32)
        sources = np.zeros((cfg.n_channels, cfg.n_time), dtype=np.float32)
        class_gain = _class_amplitude_pattern(int(label), cfg.n_channels)
        for c in range(cfg.n_channels):
            signal = np.zeros(cfg.n_time, dtype=np.float32)
            source_phase = rng.uniform(-np.pi, np.pi)
            envelope = _smooth_random_envelope(rng, cfg.n_time)
            for freq in cfg.bands:
                phase = (
                    2 * np.pi * freq * t
                    + source_phase
                    + rng.normal(0.0, cfg.phase_jitter)
                )
                signal += envelope.astype(np.float32) * np.sin(phase).astype(np.float32)
            signal /= max(1, len(cfg.bands))
            sources[c] = class_gain[c] * signal
        trial = sources.copy()
        for target in range(cfg.n_channels):
            active_sources = np.flatnonzero(edge_gt[target] != 0.0)
            for source in active_sources:
                delayed = _zero_padded_delay(sources[source], int(delay_gt[target, source]))
                trial[target] += edge_gt[target, source] * delayed
        trial = amp_style * trial + subject_style
        trial += rng.normal(0.0, cfg.noise_std, size=trial.shape).astype(np.float32)
        x[n] = trial

    subjects = np.asarray([f"S{(i % 9) + 1:02d}" for i in range(cfg.n_trials)])
    sessions = np.asarray(["T" if i % 2 == 0 else "E" for i in range(cfg.n_trials)])
    return {
        "X": x,
        "y": y,
        "subject": subjects,
        "session": sessions,
        "sfreq": np.asarray(cfg.sfreq, dtype=np.float32),
        "edge_gt": edge_gt,
        "delay_gt": delay_gt,
        "phase_gt": phase_gt,
        "sensor_xy": make_sensor_positions(cfg.n_channels),
    }


def source_to_scalp_mix(
    data: dict[str, np.ndarray],
    n_sources: int | None = None,
    seed: int = 0,
    common_source_strength: float = 0.25,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = data["X"]
    n_trials, n_channels, n_time = x.shape
    n_sources = n_sources or max(4, n_channels // 2)
    source_idx = rng.choice(n_channels, size=n_sources, replace=False)
    sources = x[:, source_idx, :]
    mixing = rng.normal(0.0, 1.0, size=(n_channels, n_sources)).astype(np.float32)
    mixing /= np.maximum(np.linalg.norm(mixing, axis=1, keepdims=True), 1e-6)
    scalp = np.einsum("cs,nst->nct", mixing, sources).astype(np.float32)
    common = rng.normal(0.0, 1.0, size=(n_trials, 1, n_time)).astype(np.float32)
    scalp = scalp + common_source_strength * common
    out = dict(data)
    out["X"] = scalp
    out["mixing_matrix"] = mixing
    out["source_indices"] = source_idx
    return out


def phase_surrogate(x: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(x, axis=-1)
    amp = np.abs(spectrum)
    random_phase = rng.uniform(-np.pi, np.pi, size=spectrum.shape)
    random_phase[..., 0] = np.angle(spectrum[..., 0])
    if spectrum.shape[-1] > 1:
        random_phase[..., -1] = np.angle(spectrum[..., -1])
    surrogate = amp * np.exp(1j * random_phase)
    return np.fft.irfft(surrogate, n=x.shape[-1], axis=-1).astype(np.float32)


def time_reverse(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reverse time and the directed ground-truth graph consistently."""

    out = dict(data)
    out["X"] = np.asarray(data["X"])[..., ::-1].copy()
    out["edge_gt"] = np.asarray(data["edge_gt"]).T.copy()
    out["delay_gt"] = np.asarray(data["delay_gt"]).T.copy()
    if "phase_gt" in data:
        out["phase_gt"] = np.asarray(data["phase_gt"]).T.copy()
    return out


def label_shuffle(y: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(y).copy()
    rng.shuffle(out)
    return out
