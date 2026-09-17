"""Sliding-window data construction for asynchronous BCI experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AsyncWindowConfig:
    sfreq: float
    window_sec: float = 1.0
    step_sec: float = 0.125
    mi_duration_sec: float = 4.0
    rest_label: int = -1


def make_windows_from_continuous(
    x: np.ndarray,
    events: np.ndarray,
    cfg: AsyncWindowConfig,
) -> dict[str, np.ndarray]:
    """Build rest-vs-MI sliding windows from continuous EEG.

    Args:
        x: continuous EEG, shape [channels, time].
        events: array with columns [sample, class_label]. Each event starts an
            MI interval of ``mi_duration_sec``.
    """

    x = np.asarray(x, dtype=np.float32)
    events = np.asarray(events)
    win = int(round(cfg.window_sec * cfg.sfreq))
    step = int(round(cfg.step_sec * cfg.sfreq))
    mi_len = int(round(cfg.mi_duration_sec * cfg.sfreq))
    intervals = []
    for sample, label in events[:, :2].astype(int):
        intervals.append((sample, sample + mi_len, label))

    windows = []
    binary = []
    multiclass = []
    onset_latency = []
    starts = []
    for start in range(0, max(0, x.shape[-1] - win + 1), max(1, step)):
        stop = start + win
        overlap_labels = []
        latencies = []
        for lo, hi, label in intervals:
            overlap = max(0, min(stop, hi) - max(start, lo))
            if overlap > 0:
                overlap_labels.append((overlap, label))
                latencies.append(max(0, start - lo) / cfg.sfreq)
        if overlap_labels:
            label = max(overlap_labels, key=lambda item: item[0])[1]
            binary.append(1)
            multiclass.append(label)
            onset_latency.append(min(latencies))
        else:
            binary.append(0)
            multiclass.append(cfg.rest_label)
            onset_latency.append(np.nan)
        windows.append(x[:, start:stop])
        starts.append(start)

    return {
        "X": np.stack(windows).astype(np.float32),
        "y_binary": np.asarray(binary, dtype=np.int64),
        "y_mi": np.asarray(multiclass, dtype=np.int64),
        "window_start": np.asarray(starts, dtype=np.int64),
        "onset_latency_sec": np.asarray(onset_latency, dtype=np.float32),
        "sfreq": np.asarray(cfg.sfreq, dtype=np.float32),
    }


def make_windows_from_labeled_intervals(
    x: np.ndarray,
    intervals: np.ndarray,
    cfg: AsyncWindowConfig,
) -> dict[str, np.ndarray]:
    """Construct windows wholly contained in known rest or MI intervals.

    Unlike pseudo-continuous smoke data, this preserves the recording timeline
    and excludes transition-crossing windows. ``intervals`` has columns
    ``[start_sample, stop_sample, label]``; ``cfg.rest_label`` denotes rest.
    """

    x = np.asarray(x, dtype=np.float32)
    intervals = np.asarray(intervals, dtype=np.int64)
    if x.ndim != 2:
        raise ValueError(f"Continuous EEG must be [channels, time], got {x.shape}")
    if intervals.ndim != 2 or intervals.shape[1] < 3:
        raise ValueError("Intervals must have shape [n_intervals, 3] with start, stop, label columns")
    if not np.isfinite(cfg.sfreq) or cfg.sfreq <= 0.0:
        raise ValueError(f"Sampling rate must be positive, got {cfg.sfreq!r}")
    win = int(round(cfg.window_sec * cfg.sfreq))
    step = int(round(cfg.step_sec * cfg.sfreq))
    if win < 2 or step < 1:
        raise ValueError("window_sec and step_sec produce an invalid window configuration")

    windows = []
    binary = []
    multiclass = []
    starts = []
    event_ids = []
    event_starts = []
    onset_latency = []
    for start in range(0, max(0, x.shape[-1] - win + 1), step):
        stop = start + win
        containing = np.where((intervals[:, 0] <= start) & (stop <= intervals[:, 1]))[0]
        if containing.size != 1:
            continue
        event_id = int(containing[0])
        event_start, _, label = intervals[event_id, :3]
        windows.append(x[:, start:stop])
        binary.append(0 if int(label) == cfg.rest_label else 1)
        multiclass.append(cfg.rest_label if int(label) == cfg.rest_label else int(label))
        starts.append(start)
        event_ids.append(event_id)
        event_starts.append(event_start)
        onset_latency.append(np.nan if int(label) == cfg.rest_label else (start - event_start) / cfg.sfreq)
    if not windows:
        raise ValueError("No valid windows were contained in the supplied labelled intervals")
    return {
        "X": np.stack(windows).astype(np.float32),
        "y_binary": np.asarray(binary, dtype=np.int64),
        "y_mi": np.asarray(multiclass, dtype=np.int64),
        "window_start": np.asarray(starts, dtype=np.int64),
        "event_id": np.asarray(event_ids, dtype=np.int64),
        "event_start": np.asarray(event_starts, dtype=np.int64),
        "onset_latency_sec": np.asarray(onset_latency, dtype=np.float32),
        "sfreq": np.asarray(cfg.sfreq, dtype=np.float32),
    }


def make_pseudo_async_from_trials(
    x: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    rest_sec: float = 1.0,
    window_sec: float = 1.0,
    step_sec: float = 0.25,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Create a deterministic pseudo-continuous stream from epoched trials.

    This is a smoke-test path for code validation. It should be reported as a
    pseudo-async simulation, not as a real asynchronous BCI result.
    """

    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    rest_n = int(round(rest_sec * sfreq))
    parts = []
    events = []
    cursor = 0
    for trial, label in zip(x, y, strict=False):
        noise_scale = float(np.std(trial) * 0.05)
        rest = rng.normal(0.0, noise_scale, size=(trial.shape[0], rest_n)).astype(np.float32)
        parts.extend([rest, trial])
        cursor += rest.shape[-1]
        events.append((cursor, int(label)))
        cursor += trial.shape[-1]
    continuous = np.concatenate(parts, axis=-1)
    return make_windows_from_continuous(
        continuous,
        np.asarray(events, dtype=np.int64),
        AsyncWindowConfig(
            sfreq=sfreq,
            window_sec=window_sec,
            step_sec=step_sec,
            mi_duration_sec=x.shape[-1] / sfreq,
        ),
    )
