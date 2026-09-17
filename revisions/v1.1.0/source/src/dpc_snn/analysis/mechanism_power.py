"""Controlled analytic-signal generators for delay mechanism power studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter


@dataclass(frozen=True)
class SyntheticMechanismSpec:
    """Sampling contract for one classifier-free mechanism replicate."""

    frequencies_hz: tuple[float, ...]
    trials: int = 40
    nodes: int = 4
    time_steps: int = 384
    timestep_seconds: float = 0.004
    ar_coefficient: float = 0.97
    burn_in_steps: int = 64

    def validate(self) -> None:
        if len(self.frequencies_hz) < 3:
            raise ValueError("At least three ordered frequencies are required")
        if not np.all(np.diff(self.frequencies_hz) > 0.0):
            raise ValueError("Synthetic frequencies must be strictly increasing")
        if self.trials < 3 or self.nodes < 2 or self.time_steps < 32:
            raise ValueError("Synthetic shape is too small for delay evidence")
        if self.timestep_seconds <= 0.0:
            raise ValueError("Synthetic timestep_seconds must be positive")
        if not 0.0 < self.ar_coefficient < 1.0:
            raise ValueError("Synthetic ar_coefficient must be in (0, 1)")


def fractional_causal_delay(values: np.ndarray, delay_steps: float) -> np.ndarray:
    """Apply the model's adjacent-sample causal fractional-delay equation."""

    source = np.asarray(values)
    delay = float(delay_steps)
    if delay < 0.0 or not np.isfinite(delay):
        raise ValueError("delay_steps must be finite and non-negative")
    base = int(np.floor(delay))
    fraction = delay - base
    start = base + int(fraction > 1e-12)
    output = np.zeros_like(source)
    if start >= source.shape[-1]:
        return output
    stop = source.shape[-1] - base
    current = source[..., start - base : stop]
    if fraction > 1e-12:
        previous = source[..., start - base - 1 : stop - 1]
        current = (1.0 - fraction) * current + fraction * previous
    output[..., start : start + current.shape[-1]] = current
    return output


def _analytic_background(spec: SyntheticMechanismSpec, seed: int) -> np.ndarray:
    spec.validate()
    rng = np.random.default_rng(seed)
    total_steps = spec.time_steps + spec.burn_in_steps
    shape = (spec.trials, len(spec.frequencies_hz), spec.nodes, total_steps)
    innovation = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    baseband = lfilter([1.0], [1.0, -spec.ar_coefficient], innovation, axis=-1)
    baseband = baseband[..., spec.burn_in_steps :]
    scale = np.sqrt(np.mean(np.abs(baseband) ** 2, axis=-1, keepdims=True)).clip(1e-8)
    baseband /= scale
    frequency = np.asarray(spec.frequencies_hz, dtype=np.float64)
    time = np.arange(spec.time_steps, dtype=np.float64) * spec.timestep_seconds
    carrier = np.exp(2j * np.pi * frequency[:, None] * time[None])
    return (baseband * carrier[None, :, None]).astype(np.complex64)


def generate_coupled_analytic(
    spec: SyntheticMechanismSpec,
    *,
    delay_steps: float,
    coupling_strength: float,
    seed: int,
    source_node: int = 0,
    target_node: int = 1,
) -> np.ndarray:
    """Add one known delayed source to an independent narrow-band target."""

    if not 0.0 <= float(coupling_strength) <= 1.0:
        raise ValueError("coupling_strength must be in [0, 1]")
    if source_node == target_node or not (
        0 <= source_node < spec.nodes and 0 <= target_node < spec.nodes
    ):
        raise ValueError("source_node and target_node must be distinct valid nodes")
    analytic = _analytic_background(spec, seed)
    delayed = fractional_causal_delay(analytic[:, :, source_node], delay_steps)
    rng = np.random.default_rng(seed + 104729)
    nuisance_intercept = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(spec.trials, 1, 1)))
    analytic[:, :, target_node] += (float(coupling_strength) * nuisance_intercept * delayed).astype(
        np.complex64
    )
    return analytic


def generate_common_source_analytic(
    spec: SyntheticMechanismSpec,
    *,
    mixing_strength: float,
    seed: int,
    first_node: int = 0,
    second_node: int = 1,
) -> np.ndarray:
    """Generate instantaneous real-valued common-source mixing with no delay."""

    if not 0.0 <= float(mixing_strength) <= 1.0:
        raise ValueError("mixing_strength must be in [0, 1]")
    analytic = _analytic_background(spec, seed)
    latent_spec = SyntheticMechanismSpec(
        frequencies_hz=spec.frequencies_hz,
        trials=spec.trials,
        nodes=2,
        time_steps=spec.time_steps,
        timestep_seconds=spec.timestep_seconds,
        ar_coefficient=spec.ar_coefficient,
        burn_in_steps=spec.burn_in_steps,
    )
    latent = _analytic_background(latent_spec, seed + 130363)[:, :, 0]
    analytic[:, :, first_node] += float(mixing_strength) * latent
    analytic[:, :, second_node] += 0.8 * float(mixing_strength) * latent
    return analytic
