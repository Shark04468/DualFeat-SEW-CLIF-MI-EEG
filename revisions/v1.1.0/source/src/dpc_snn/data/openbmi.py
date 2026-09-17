"""Strict subject/session loader for the OpenBMI motor-imagery dataset."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from dpc_snn.data.electrodes import electrode_coordinates, motor_anchor_indices
from dpc_snn.utils.imports import optional_import
from dpc_snn.utils.storage import configure_cache_env


def openbmi_session_keys(sessions: Iterable[str | int]) -> tuple[list[str], list[str]]:
    """Return MOABB's zero-based keys and stable public session labels.

    MOABB 1.5 passes OpenBMI's one-based file-session identifiers through its
    generic selector even though the dataset exposes keys ``0`` and ``1``.
    Selecting ``sessions=[1]`` therefore reads only the second recording. This
    explicit mapping keeps Session 2 physically unopened during development.
    """

    keys: list[str] = []
    labels: list[str] = []
    for value in sessions:
        normalized = str(value).strip().upper()
        if normalized in {"1", "S1", "SESSION1", "SESSION_1"}:
            key, label = "0", "S1"
        elif normalized in {"2", "S2", "SESSION2", "SESSION_2"}:
            key, label = "1", "S2"
        else:
            raise ValueError(f"Unknown OpenBMI session {value!r}; expected S1 or S2")
        if key not in keys:
            keys.append(key)
            labels.append(label)
    if not keys:
        raise ValueError("At least one OpenBMI session is required")
    return keys, labels


def load_openbmi_subject(
    subject: int,
    *,
    sessions: Iterable[str | int] = ("S1",),
    tmin: float = -1.0,
    tmax: float = 4.0,
    resample: float = 1000.0,
) -> dict[str, Any]:
    """Load one subject and only the explicitly requested labelled sessions."""

    subject = int(subject)
    if not 1 <= subject <= 54:
        raise ValueError("OpenBMI subject must be in [1, 54]")
    if not np.isfinite(resample) or float(resample) <= 2.0 * 40.0:
        raise ValueError("OpenBMI delay evidence requires a finite sampling rate above 80 Hz")
    if float(tmax) <= float(tmin):
        raise ValueError("OpenBMI tmax must be greater than tmin")

    configure_cache_env()
    datasets = optional_import("moabb.datasets", "OpenBMI loading")
    paradigms = optional_import("moabb.paradigms", "OpenBMI loading")
    dataset = datasets.Lee2019_MI(train_run=True, test_run=False)
    session_keys, session_labels = openbmi_session_keys(sessions)

    # See openbmi_session_keys: this private selector is required for MOABB
    # 1.5.0 and is immediately verified against returned metadata below.
    dataset._selected_sessions = session_keys
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
    observed_keys = [str(value) for value in metadata["session"].astype(str).unique()]
    if sorted(observed_keys) != sorted(session_keys):
        raise RuntimeError(
            f"OpenBMI returned sessions {observed_keys}, requested exactly {session_keys}"
        )

    key_to_label = dict(zip(session_keys, session_labels, strict=True))
    stable_sessions = np.asarray(
        [key_to_label[str(value)] for value in metadata["session"].astype(str)],
        dtype="U2",
    )
    label_map = {"left_hand": 0, "right_hand": 1}
    observed_labels = {str(value) for value in labels}
    if observed_labels != set(label_map):
        raise RuntimeError(f"OpenBMI returned unexpected labels: {sorted(observed_labels)}")
    y = np.asarray([label_map[str(value)] for value in labels], dtype=np.int64)
    x = epochs.get_data(copy=False).astype(np.float32, copy=False)
    channel_names = [str(name) for name in epochs.ch_names]
    coordinates = electrode_coordinates(epochs)
    anchors = motor_anchor_indices(channel_names)
    sfreq = float(epochs.info["sfreq"])
    runs = metadata["run"].astype(str).to_numpy()
    counters: dict[tuple[str, str], int] = {}
    trial_ids: list[str] = []
    for session, run in zip(stable_sessions.tolist(), runs.tolist(), strict=True):
        key = (str(session), str(run))
        index = counters.get(key, 0)
        counters[key] = index + 1
        trial_ids.append(f"OpenBMI-S{subject:02d}-{session}-{run}-trial{index:03d}")
    expected_samples = int(round((float(tmax) - float(tmin)) * sfreq))
    if x.shape[-1] not in {expected_samples, expected_samples + 1}:
        raise RuntimeError(
            f"OpenBMI epoch has {x.shape[-1]} samples, expected {expected_samples} or "
            f"{expected_samples + 1}"
        )
    if not np.isfinite(x).all():
        raise FloatingPointError("OpenBMI data contain NaN or Inf")

    return {
        "X": x,
        "y": y,
        "subject": np.asarray([str(subject)] * len(y)),
        "session": stable_sessions,
        "run": runs,
        "trial_id": np.asarray(trial_ids),
        "sfreq": sfreq,
        "ch_names": channel_names,
        "electrode_coordinates": coordinates,
        "spatial_anchor_indices": anchors,
        "epoch_tmin": float(tmin),
        "epoch_tmax": float(tmin) + x.shape[-1] / sfreq,
        "dataset_name": "OpenBMI_Lee2019_MI",
        "task_events": ["left_hand", "right_hand"],
        "label_map": label_map,
        "n_classes": 2,
        "moabb_session_keys": session_keys,
        "heldout_session_accessed": "S2" in set(stable_sessions.tolist()),
    }
