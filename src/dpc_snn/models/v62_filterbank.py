"""Causal fixed analytic front end for DASP-SNN V6.2-R1.

The legacy project filter bank uses a whole-trial FFT and is intentionally not
reused here.  This module exposes the latency of every FIR stage and keeps all
resampling operations causal.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


DEFAULT_V62_BANDS: tuple[tuple[float, float], ...] = (
    (6.0, 8.0),
    (8.0, 10.0),
    (10.0, 12.0),
    (12.0, 14.0),
    (14.0, 17.0),
    (17.0, 20.0),
    (20.0, 23.0),
    (23.0, 26.0),
    (26.0, 30.0),
    (30.0, 34.0),
    (34.0, 37.0),
    (37.0, 40.0),
)


@dataclass(frozen=True)
class DualRateFeatures:
    """Timestamped task features produced by the causal front end."""

    fast: torch.Tensor
    slow: torch.Tensor
    fast_timestamps: torch.Tensor
    slow_timestamps: torch.Tensor
    fast_availability: torch.Tensor
    slow_availability: torch.Tensor
    analytic_group_delay_seconds: float
    alignment_group_delay_seconds: float


def _validate_odd_taps(taps: int, name: str) -> int:
    taps = int(taps)
    if taps < 3 or taps % 2 == 0:
        raise ValueError(f"{name} must be an odd integer of at least three")
    return taps


def _complex_bandpass_taps(
    low_hz: float,
    high_hz: float,
    sfreq: float,
    taps: int,
) -> torch.Tensor:
    """Windowed one-sided complex band-pass impulse response.

    Taps are indexed by causal lag.  The centre-frequency response is
    normalized to two so a unit real sinusoid has approximately unit analytic
    amplitude after its negative-frequency component is rejected.
    """

    group_delay = (taps - 1) // 2
    lag = torch.arange(taps, dtype=torch.float64)
    centred = lag - group_delay
    width = float(high_hz - low_hz)
    centre = 0.5 * float(high_hz + low_hz)
    prototype = 2.0 * width / sfreq * torch.sinc(width * centred / sfreq)
    carrier = torch.exp(2j * math.pi * centre * centred / sfreq)
    window = torch.kaiser_window(taps, periodic=False, beta=8.0, dtype=torch.float64)
    impulse = prototype.to(torch.complex128) * carrier * window
    omega = 2.0 * math.pi * centre / sfreq
    response = torch.sum(impulse * torch.exp(-1j * omega * lag))
    impulse = impulse * (2.0 / response.abs().clamp_min(1e-12))
    return impulse.to(torch.complex64)


def _lowpass_taps(cutoff_hz: float, sfreq: float, taps: int) -> torch.Tensor:
    group_delay = (taps - 1) // 2
    lag = torch.arange(taps, dtype=torch.float64)
    centred = lag - group_delay
    impulse = 2.0 * cutoff_hz / sfreq * torch.sinc(2.0 * cutoff_hz * centred / sfreq)
    impulse = impulse * torch.kaiser_window(
        taps, periodic=False, beta=8.0, dtype=torch.float64
    )
    return (impulse / impulse.sum().clamp_min(1e-12)).to(torch.float32)


def _causal_fir_real(x: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    flat = x.reshape(-1, 1, shape[-1])
    weight = taps.to(device=x.device, dtype=x.dtype).flip(0).view(1, 1, -1)
    filtered = F.conv1d(F.pad(flat, (taps.numel() - 1, 0)), weight)
    return filtered.reshape(shape)


def _causal_delay(x: torch.Tensor, samples: int) -> torch.Tensor:
    samples = int(samples)
    if samples <= 0:
        return x
    return F.pad(x[..., :-samples], (samples, 0))


class CausalAnalyticFilterBank(nn.Module):
    """Fixed 6-40 Hz analytic FIR bank with an explicit group delay."""

    def __init__(
        self,
        band_edges_hz: Sequence[Sequence[float]] = DEFAULT_V62_BANDS,
        sfreq: float = 250.0,
        taps: int = 129,
    ) -> None:
        super().__init__()
        self.sfreq = float(sfreq)
        self.taps = _validate_odd_taps(taps, "analytic taps")
        edges = torch.as_tensor(band_edges_hz, dtype=torch.float32)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("band_edges_hz must have shape [bands, 2]")
        if bool((edges[:, 0] <= 0).any()) or bool((edges[:, 1] <= edges[:, 0]).any()):
            raise ValueError("analytic bands must have strictly positive ordered edges")
        if float(edges[:, 1].max()) >= self.sfreq / 2.0:
            raise ValueError("analytic bands must lie below Nyquist")
        kernels = torch.stack(
            [
                _complex_bandpass_taps(float(low), float(high), self.sfreq, self.taps)
                for low, high in edges.tolist()
            ]
        )
        self.register_buffer("band_edges_hz", edges)
        self.register_buffer("kernel_real", kernels.real[:, None, :].contiguous())
        self.register_buffer("kernel_imag", kernels.imag[:, None, :].contiguous())

    @property
    def n_bands(self) -> int:
        return int(self.band_edges_hz.shape[0])

    @property
    def group_delay_samples(self) -> int:
        return (self.taps - 1) // 2

    @property
    def group_delay_seconds(self) -> float:
        return self.group_delay_samples / self.sfreq

    @property
    def support_seconds(self) -> float:
        return self.taps / self.sfreq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("causal analytic filter bank expects [N, C, T]")
        n_trials, n_channels, n_time = x.shape
        flat = x.reshape(n_trials * n_channels, 1, n_time)
        padded = F.pad(flat, (self.taps - 1, 0))
        real = F.conv1d(padded, self.kernel_real.to(dtype=x.dtype))
        imag = F.conv1d(padded, self.kernel_imag.to(dtype=x.dtype))
        real = real.reshape(n_trials, n_channels, self.n_bands, n_time).permute(0, 2, 1, 3)
        imag = imag.reshape(n_trials, n_channels, self.n_bands, n_time).permute(0, 2, 1, 3)
        return torch.complex(real, imag)

    def frequency_response(self, frequencies_hz: torch.Tensor) -> torch.Tensor:
        frequencies_hz = torch.as_tensor(
            frequencies_hz, device=self.kernel_real.device, dtype=self.kernel_real.dtype
        )
        lag = torch.arange(self.taps, device=frequencies_hz.device, dtype=frequencies_hz.dtype)
        kernels = torch.complex(self.kernel_real[:, 0], self.kernel_imag[:, 0])
        phase = torch.exp(
            -2j * math.pi * frequencies_hz[:, None] * lag[None, :] / self.sfreq
        )
        return torch.einsum("bl,fl->bf", kernels, phase)


class DualRateCausalResampler(nn.Module):
    """Create aligned 125 Hz carrier and 62.5 Hz log-envelope features.

    The envelope low-pass adds a causal group delay.  The carrier is delayed by
    the same number of 250 Hz samples before decimation, making the two paths
    share an effective timestamp without reading future samples.
    """

    def __init__(
        self,
        n_bands: int,
        n_nodes: int,
        sfreq: float = 250.0,
        analytic_group_delay_samples: int = 64,
        envelope_cutoff_hz: float = 20.0,
        envelope_taps: int = 33,
        fast_decimation: int = 2,
        gain_invariant_envelope: bool = False,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.n_bands = int(n_bands)
        self.n_nodes = int(n_nodes)
        self.sfreq = float(sfreq)
        self.analytic_group_delay_samples = int(analytic_group_delay_samples)
        self.envelope_taps = _validate_odd_taps(envelope_taps, "envelope taps")
        self.fast_decimation = int(fast_decimation)
        self.slow_decimation = 2 * self.fast_decimation
        self.gain_invariant_envelope = bool(gain_invariant_envelope)
        if self.fast_decimation < 1:
            raise ValueError("fast decimation must be positive")
        self.eps = float(eps)
        self.register_buffer(
            "envelope_kernel",
            _lowpass_taps(float(envelope_cutoff_hz), self.sfreq, self.envelope_taps),
        )
        self.register_buffer("training_gain", torch.ones(self.n_bands, self.n_nodes))
        self.register_buffer("training_gain_ready", torch.tensor(False))

    @property
    def alignment_group_delay_samples(self) -> int:
        return (self.envelope_taps - 1) // 2

    @property
    def total_group_delay_seconds(self) -> float:
        return (
            self.analytic_group_delay_samples + self.alignment_group_delay_samples
        ) / self.sfreq

    def set_training_gain(self, gain: torch.Tensor) -> None:
        gain = torch.as_tensor(gain, dtype=self.training_gain.dtype)
        if gain.shape != self.training_gain.shape or not bool(torch.isfinite(gain).all()):
            raise ValueError(
                f"training gain must be finite with shape {tuple(self.training_gain.shape)}"
            )
        if bool((gain <= 0).any()):
            raise ValueError("training gain must be strictly positive")
        self.training_gain.copy_(gain)
        self.training_gain_ready.fill_(True)

    def forward(
        self,
        projected: torch.Tensor,
        *,
        epoch_tmin: float = -1.0,
        task_tmin: float = 0.0,
        task_tmax: float = 4.0,
    ) -> DualRateFeatures:
        if projected.ndim != 4 or not projected.is_complex():
            raise ValueError("dual-rate resampler expects complex [N, B, K, T]")
        if projected.shape[1:3] != (self.n_bands, self.n_nodes):
            raise ValueError("dual-rate resampler received incompatible band/node axes")
        if task_tmax <= task_tmin:
            raise ValueError("task_tmax must exceed task_tmin")
        gain = self.training_gain.to(projected.real)[None, :, :, None]
        envelope_source = projected if self.gain_invariant_envelope else projected * gain
        projected = projected * gain

        raw_start = int(round((float(task_tmin) - float(epoch_tmin)) * self.sfreq))
        raw_stop = int(round((float(task_tmax) - float(epoch_tmin)) * self.sfreq))
        if raw_start <= 0:
            raise ValueError("V6.2 requires a pre-task baseline before task_tmin")
        if raw_stop > projected.shape[-1]:
            raise ValueError("task interval exceeds the available epoch")
        if (raw_stop - raw_start) % self.slow_decimation:
            raise ValueError("task duration must align with the configured dual rates")

        # V8 defines the baseline-relative envelope in physical sensor units.
        # Applying the fold gain after this guarded logarithm makes the cached
        # and direct envelope paths exactly identical, including near zero.
        log_envelope = envelope_source.abs().clamp_min(self.eps).log()
        baseline = log_envelope[..., :raw_start].mean(dim=-1, keepdim=True)
        relative_envelope = log_envelope - baseline
        slow_full = _causal_fir_real(relative_envelope, self.envelope_kernel)
        fast_aligned = _causal_delay(projected, self.alignment_group_delay_samples)

        fast = fast_aligned[..., raw_start:raw_stop : self.fast_decimation]
        slow = slow_full[..., raw_start:raw_stop : self.slow_decimation]
        availability_fast = torch.arange(
            raw_start,
            raw_stop,
            self.fast_decimation,
            device=projected.device,
            dtype=projected.real.dtype,
        ) / self.sfreq + float(epoch_tmin)
        availability_slow = torch.arange(
            raw_start,
            raw_stop,
            self.slow_decimation,
            device=projected.device,
            dtype=projected.real.dtype,
        ) / self.sfreq + float(epoch_tmin)
        delay = self.total_group_delay_seconds
        return DualRateFeatures(
            fast=fast,
            slow=slow,
            fast_timestamps=availability_fast - delay,
            slow_timestamps=availability_slow - delay,
            fast_availability=availability_fast,
            slow_availability=availability_slow,
            analytic_group_delay_seconds=self.analytic_group_delay_samples / self.sfreq,
            alignment_group_delay_seconds=self.alignment_group_delay_samples / self.sfreq,
        )


def causal_linear_upsample_2x(slow: torch.Tensor) -> torch.Tensor:
    """Causal first-order hold from 62.5 Hz to 125 Hz.

    Even output samples equal the current slow sample.  Odd samples extrapolate
    half a step from the two most recently available samples.  This exactly
    preserves constants and, after the initial sample, linear ramps without
    consulting the next slow sample.
    """

    if slow.shape[-1] < 1:
        raise ValueError("cannot upsample an empty sequence")
    previous = torch.cat((slow[..., :1], slow[..., :-1]), dim=-1)
    half_step = slow + 0.5 * (slow - previous)
    return torch.stack((slow, half_step), dim=-1).flatten(-2)
