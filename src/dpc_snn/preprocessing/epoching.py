"""Simple epoching utilities for already loaded continuous EEG arrays."""

from __future__ import annotations

import numpy as np


def epoch_continuous(
    x: np.ndarray,
    events: list[tuple[int, int]],
    sfreq: float,
    tmin: float,
    tmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut epochs from continuous data.

    Args:
        x: continuous EEG, shape [channels, time].
        events: list of ``(sample_index, label)`` pairs.
        sfreq: sampling rate.
        tmin: seconds relative to event.
        tmax: seconds relative to event.
    """

    start_offset = int(round(tmin * sfreq))
    stop_offset = int(round(tmax * sfreq))
    epochs = []
    labels = []
    for sample, label in events:
        lo = sample + start_offset
        hi = sample + stop_offset
        if lo < 0 or hi > x.shape[-1] or hi <= lo:
            continue
        epochs.append(x[:, lo:hi])
        labels.append(label)
    if not epochs:
        raise ValueError("No valid epochs could be extracted")
    return np.stack(epochs).astype(np.float32), np.asarray(labels, dtype=np.int64)

