"""Strict subject loader for the Cho2017 motor-imagery dataset."""

from __future__ import annotations

from typing import Any

import numpy as np

from dpc_snn.data.electrodes import electrode_coordinates, motor_anchor_indices
from dpc_snn.utils.imports import optional_import
from dpc_snn.utils.storage import configure_cache_env


def load_cho2017_subject(
    subject: int,
    *,
    tmin: float = 0.0,
    tmax: float = 3.0,
    resample: float = 512.0,
) -> dict[str, Any]:
    """Load one Cho2017 subject with a verified single-session protocol."""

    subject = int(subject)
    if not 1 <= subject <= 52:
        raise ValueError("Cho2017 subject must be in [1, 52]")
    if not np.isfinite(resample) or float(resample) <= 2.0 * 40.0:
        raise ValueError("Cho2017 delay evidence requires a finite sampling rate above 80 Hz")
    if float(tmax) <= float(tmin):
        raise ValueError("Cho2017 tmax must be greater than tmin")

    configure_cache_env()
    datasets = optional_import("moabb.datasets", "Cho2017 loading")
    paradigms = optional_import("moabb.paradigms", "Cho2017 loading")
    dataset = datasets.Cho2017(subjects=[subject])
    # MOABB 1.5 forwards the constructor's one-based session identifiers to
    # data keyed by "0". Explicitly selecting the actual key avoids an empty
    # paradigm result and is verified against metadata below.
    dataset._selected_sessions = ["0"]
    paradigm = paradigms.MotorImagery(
        n_classes=2,
        events=["left_hand", "right_hand"],
        tmin=float(tmin),
        tmax=float(tmax),
        resample=float(resample),
    )
    epochs, labels, metadata = paradigm.get_data(
        dataset=dataset,
        subjects=[subject],
        return_epochs=True,
    )
    observed_sessions = sorted(str(value) for value in metadata["session"].unique())
    observed_runs = sorted(str(value) for value in metadata["run"].unique())
    if observed_sessions != ["0"] or observed_runs != ["0"]:
        raise RuntimeError(
            f"Cho2017 returned sessions/runs {observed_sessions}/{observed_runs}, expected 0/0"
        )

    label_map = {"left_hand": 0, "right_hand": 1}
    observed_labels = {str(value) for value in labels}
    if observed_labels != set(label_map):
        raise RuntimeError(f"Cho2017 returned unexpected labels: {sorted(observed_labels)}")
    y = np.asarray([label_map[str(value)] for value in labels], dtype=np.int64)
    if len(y) != 200 or np.bincount(y, minlength=2).tolist() != [100, 100]:
        raise RuntimeError("Cho2017 must expose exactly 100 trials per motor-imagery class")

    x = epochs.get_data(copy=False).astype(np.float32, copy=False)
    channel_names = [str(name) for name in epochs.ch_names]
    if len(channel_names) != 64:
        raise RuntimeError(f"Cho2017 must expose 64 EEG channels, got {len(channel_names)}")
    coordinates = electrode_coordinates(epochs)
    anchors = motor_anchor_indices(channel_names)
    sfreq = float(epochs.info["sfreq"])
    expected_samples = int(round((float(tmax) - float(tmin)) * sfreq))
    if x.shape[-1] not in {expected_samples, expected_samples + 1}:
        raise RuntimeError(
            f"Cho2017 epoch has {x.shape[-1]} samples, expected {expected_samples} or "
            f"{expected_samples + 1}"
        )
    if not np.isfinite(x).all():
        raise FloatingPointError("Cho2017 data contain NaN or Inf")

    return {
        "X": x,
        "y": y,
        "subject": np.asarray([str(subject)] * len(y)),
        "session": np.asarray(["S1"] * len(y), dtype="U2"),
        "run": np.asarray(["R1"] * len(y), dtype="U2"),
        "sfreq": sfreq,
        "ch_names": channel_names,
        "electrode_coordinates": coordinates,
        "spatial_anchor_indices": anchors,
        "epoch_tmin": float(tmin),
        "epoch_tmax": float(tmin) + x.shape[-1] / sfreq,
        "dataset_name": "Cho2017_GigaDB_MI",
        "task_events": ["left_hand", "right_hand"],
        "label_map": label_map,
        "n_classes": 2,
        "single_session_verified": True,
    }
