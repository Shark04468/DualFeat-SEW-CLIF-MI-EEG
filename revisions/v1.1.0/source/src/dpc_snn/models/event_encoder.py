"""Event-native EEG encoding and causal spike-state readout.

The encoder converts analytic EEG features into two sparse signed streams:
phase zero-crossing events and envelope ON/OFF delta events.  Thresholds are
global per band and therefore never depend on validation/test trials.
"""

from __future__ import annotations

import torch
from torch import nn

from .surrogate import spike_fn


def _inverse_softplus(value: float) -> float:
    value_tensor = torch.tensor(max(float(value), 1e-6))
    return float(torch.log(torch.expm1(value_tensor)))


def fractional_delay_events(
    events: torch.Tensor, delay_steps: float | torch.Tensor
) -> torch.Tensor:
    """Apply a causal two-tap fractional delay to an event stream.

    ``delay_steps`` may be scalar or broadcastable to ``events.shape[:-1]``.
    A delay ``d + f`` returns ``(1-f)x[t-d] + f*x[t-d-1]`` with causal zero
    padding.  This is the exact clocked-synapse interpretation used by the
    delay expert bank; it does not interpolate the EEG waveform itself.
    """

    if events.ndim < 1:
        raise ValueError("events must contain a time axis")
    delay = torch.as_tensor(delay_steps, device=events.device, dtype=events.dtype)
    if torch.any(delay < 0):
        raise ValueError("fractional event delays must be non-negative")
    delay = torch.broadcast_to(delay, events.shape[:-1])
    base = torch.floor(delay).to(torch.long)
    fraction = delay - base.to(delay.dtype)
    time = torch.arange(events.shape[-1], device=events.device)
    source_low = time.view(*([1] * base.ndim), -1) - base[..., None]
    source_high = source_low - 1

    low_valid = source_low >= 0
    high_valid = source_high >= 0
    low = torch.gather(events, -1, source_low.clamp_min(0))
    high = torch.gather(events, -1, source_high.clamp_min(0))
    low = low * low_valid.to(events.dtype)
    high = high * high_valid.to(events.dtype)
    return (1.0 - fraction[..., None]) * low + fraction[..., None] * high


class PhaseDeltaEventEncoder(nn.Module):
    """Encode phase crossings and asynchronous envelope changes.

    Carrier phase is represented by signed upward/downward zero crossings.
    Slow ERD/ERS information is represented by signed ON/OFF events whenever
    the log-envelope delta exceeds a fixed per-band threshold.  An optional
    deterministic timing intervention preserves event counts while destroying
    the original temporal order for mechanism ablations.
    """

    def __init__(
        self,
        n_bands: int,
        n_nodes: int,
        timesteps: int,
        envelope_threshold: float = 0.05,
        phase_confidence_threshold: float = 0.10,
        learnable_envelope_threshold: bool = False,
        timing_intervention: str = "none",
        timing_seed: int = 0,
    ):
        super().__init__()
        self.n_bands = int(n_bands)
        self.n_nodes = int(n_nodes)
        self.timesteps = int(timesteps)
        self.phase_confidence_threshold = float(phase_confidence_threshold)
        self.timing_intervention = str(timing_intervention).lower()
        if self.timing_intervention not in {"none", "shuffle", "reverse"}:
            raise ValueError(
                "timing_intervention must be 'none', 'shuffle', or 'reverse'"
            )
        raw = torch.full(
            (self.n_bands,), _inverse_softplus(envelope_threshold)
        )
        if learnable_envelope_threshold:
            self.envelope_threshold_raw = nn.Parameter(raw)
        else:
            self.register_buffer("envelope_threshold_raw", raw)
        self.register_buffer("envelope_threshold_ready", torch.tensor(False))

        generator = torch.Generator().manual_seed(int(timing_seed))
        permutations = torch.stack(
            [
                torch.randperm(self.timesteps, generator=generator)
                for _ in range(self.n_bands * self.n_nodes)
            ]
        ).reshape(self.n_bands, self.n_nodes, self.timesteps)
        self.register_buffer("timing_permutation", permutations)

    @property
    def envelope_threshold(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.envelope_threshold_raw).clamp_min(
            1e-5
        )

    @torch.no_grad()
    def set_envelope_thresholds(self, thresholds: torch.Tensor) -> None:
        values = torch.as_tensor(
            thresholds,
            device=self.envelope_threshold_raw.device,
            dtype=self.envelope_threshold_raw.dtype,
        )
        if values.shape != (self.n_bands,):
            raise ValueError("event thresholds must contain one value per band")
        if not torch.isfinite(values).all() or torch.any(values <= 0):
            raise ValueError("event thresholds must be finite and positive")
        self.envelope_threshold_raw.copy_(
            torch.log(torch.expm1(values.clamp_min(1e-5)))
        )
        self.envelope_threshold_ready.fill_(True)

    def _intervene(self, events: torch.Tensor) -> torch.Tensor:
        if self.timing_intervention == "none":
            return events
        if self.timing_intervention == "reverse":
            return events.flip(-1)
        if events.shape[1:3] != (self.n_bands, self.n_nodes):
            raise ValueError("event geometry changed after encoder construction")
        if events.shape[-1] != self.timesteps:
            raise ValueError("event time axis changed after encoder construction")
        indices = self.timing_permutation[None].expand(events.shape[0], -1, -1, -1)
        return torch.gather(events, -1, indices)

    def forward(
        self,
        carrier: torch.Tensor,
        envelope: torch.Tensor,
        confidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if carrier.shape != envelope.shape or carrier.shape != confidence.shape:
            raise ValueError(
                "carrier, envelope, and confidence must share [N, B, K, T]"
            )
        if carrier.shape[1:3] != (self.n_bands, self.n_nodes):
            raise ValueError("event encoder received incompatible band/node geometry")
        if carrier.shape[-1] != self.timesteps:
            raise ValueError("event encoder received an incompatible time axis")

        previous = carrier[..., :-1]
        current = carrier[..., 1:]
        confidence_gate = spike_fn(
            torch.minimum(confidence[..., :-1], confidence[..., 1:])
            - self.phase_confidence_threshold
        )
        phase_up = spike_fn(current) * spike_fn(-previous) * confidence_gate
        phase_down = spike_fn(-current) * spike_fn(previous) * confidence_gate
        phase_up = torch.nn.functional.pad(phase_up, (1, 0))
        phase_down = torch.nn.functional.pad(phase_down, (1, 0))

        delta = envelope[..., 1:] - envelope[..., :-1]
        threshold = self.envelope_threshold.to(envelope)[None, :, None, None]
        envelope_on = torch.nn.functional.pad(
            spike_fn(delta - threshold), (1, 0)
        )
        envelope_off = torch.nn.functional.pad(
            spike_fn(-delta - threshold), (1, 0)
        )

        phase_events = self._intervene(phase_up - phase_down)
        envelope_events = self._intervene(envelope_on - envelope_off)
        return {
            "phase_events": phase_events,
            "envelope_events": envelope_events,
            "phase_up": self._intervene(phase_up),
            "phase_down": self._intervene(phase_down),
            "envelope_on": self._intervene(envelope_on),
            "envelope_off": self._intervene(envelope_off),
            "phase_event_density": (phase_up + phase_down).mean(),
            "envelope_event_density": (envelope_on + envelope_off).mean(),
            "envelope_threshold": self.envelope_threshold,
            "envelope_threshold_ready": self.envelope_threshold_ready,
        }


class CausalSpikeAccumulator(nn.Module):
    """Read binary spike history through causal synaptic traces only."""

    def __init__(
        self,
        channels: int,
        output_channels: int,
        decays: tuple[float, ...] = (0.5, 0.9, 0.99),
    ):
        super().__init__()
        if not decays or any(not 0.0 <= float(value) < 1.0 for value in decays):
            raise ValueError("accumulator decays must lie in [0, 1)")
        self.projection = nn.Conv1d(
            int(channels), int(output_channels), kernel_size=1, bias=False
        )
        self.register_buffer("decays", torch.tensor(decays, dtype=torch.float32))

    @property
    def out_features(self) -> int:
        return self.projection.out_channels * self.decays.numel()

    def forward(
        self, spikes: torch.Tensor, capture_steps: tuple[int, ...] = ()
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if spikes.ndim != 3:
            raise ValueError("causal spike accumulator expects [N, C, T]")
        projected = self.projection(spikes)
        decay = self.decays.to(projected)[None, :, None]
        state = projected.new_zeros(
            projected.shape[0], self.decays.numel(), projected.shape[1]
        )
        capture = {int(step) for step in capture_steps}
        snapshots: list[torch.Tensor] = []
        for step in range(projected.shape[-1]):
            step_current = projected[:, :, step][:, None, :]
            state = decay * state + (1.0 - decay) * step_current
            if step + 1 in capture:
                snapshots.append(state.flatten(1))
        final = state.flatten(1)
        if snapshots:
            captured = torch.stack(snapshots, dim=1)
        else:
            captured = final.new_zeros(final.shape[0], 0, final.shape[1])
        return final, captured
