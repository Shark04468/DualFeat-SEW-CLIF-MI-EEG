"""Strict blind-protocol loader for BCI Competition IV Dataset 2b."""

from __future__ import annotations

from types import MethodType
from typing import Any, Iterable

import numpy as np

from dpc_snn.data.electrodes import electrode_coordinates
from dpc_snn.utils.imports import optional_import
from dpc_snn.utils.storage import configure_cache_env


SESSION_TO_MOABB_KEY = {
    "01T": "0train",
    "02T": "1train",
    "03T": "2train",
    "04E": "3test",
    "05E": "4test",
}
def bnci2014_004_session_keys(
    sessions: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Return exact MOABB keys and stable public session labels."""

    labels = [str(value).upper() for value in sessions]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("BNCI2014-004 sessions must be non-empty and unique")
    unknown = sorted(set(labels).difference(SESSION_TO_MOABB_KEY))
    if unknown:
        raise ValueError(f"Unknown BNCI2014-004 sessions: {unknown}")
    return [SESSION_TO_MOABB_KEY[label] for label in labels], labels


def bnci2014_004_physical_files(sessions: Iterable[str]) -> list[str]:
    """Return the only MAT file roles authorized by a stable session selection."""

    _, labels = bnci2014_004_session_keys(sessions)
    roles = []
    if any(label.endswith("T") for label in labels):
        roles.append("T")
    if any(label.endswith("E") for label in labels):
        roles.append("E")
    if len(roles) != 1:
        raise ValueError("BNCI2014-004 may not mix training and evaluation physical files")
    return roles


def _physically_separated_dataset(
    datasets: Any,
    *,
    subject: int,
    session_keys: list[str],
    physical_role: str,
) -> Any:
    """Build a MOABB dataset that never opens the unauthorized MAT file."""

    def get_single_subject_data(
        self: Any, requested_subject: int
    ) -> dict[str, Any]:
        module = optional_import(
            "moabb.datasets.bnci.bnci_2014",
            "physically separated BNCI2014-004 loading",
        )
        url = (
            f"{module.BNCI_URL}004-2014/"
            f"B{int(requested_subject):02d}{physical_role}.mat"
        )
        filename = module.data_path(
            url,
            path=None,
            force_update=False,
            update_path=None,
        )[0]
        raws, _ = module._convert_mi(
            filename,
            ["C3", "Cz", "C4", "EOG1", "EOG2", "EOG3"],
            ["eeg", "eeg", "eeg", "eog", "eog", "eog"],
            dataset_code="BNCI2014-004",
            subject_id=int(requested_subject),
        )
        all_keys = (
            ["0train", "1train", "2train"]
            if physical_role == "T"
            else ["3test", "4test"]
        )
        if len(raws) != len(all_keys):
            raise RuntimeError(
                f"BNCI2014-004 {physical_role} file returned {len(raws)} sessions, "
                f"expected {len(all_keys)}"
            )
        sessions_by_key = {
            key: {"0": raw} for key, raw in zip(all_keys, raws, strict=True)
        }
        return {key: sessions_by_key[key] for key in session_keys}

    dataset = datasets.BNCI2014_004(subjects=[subject])
    dataset._get_single_subject_data = MethodType(get_single_subject_data, dataset)
    dataset._selected_sessions = session_keys
    return dataset


def load_bnci2014_004_subject(
    subject: int,
    *,
    sessions: Iterable[str],
    tmin: float = -1.0,
    tmax: float = 4.0,
    resample: float = 250.0,
) -> dict[str, Any]:
    """Load one subject and only explicitly authorized sessions."""

    subject = int(subject)
    if not 1 <= subject <= 9:
        raise ValueError("BNCI2014-004 subject must be in [1, 9]")
    if not np.isfinite(resample) or float(resample) < 80.0:
        raise ValueError("BNCI2014-004 requires a finite sampling rate of at least 80 Hz")
    if float(tmax) <= float(tmin):
        raise ValueError("BNCI2014-004 tmax must be greater than tmin")
    session_keys, session_labels = bnci2014_004_session_keys(sessions)
    physical_role = bnci2014_004_physical_files(session_labels)[0]

    configure_cache_env()
    datasets = optional_import("moabb.datasets", "BNCI2014-004 loading")
    paradigms = optional_import("moabb.paradigms", "BNCI2014-004 loading")
    dataset = _physically_separated_dataset(
        datasets,
        subject=subject,
        session_keys=session_keys,
        physical_role=physical_role,
    )
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
            f"BNCI2014-004 returned sessions {observed_keys}, requested {session_keys}"
        )
    key_to_label = dict(zip(session_keys, session_labels, strict=True))
    stable_sessions = np.asarray(
        [key_to_label[str(value)] for value in metadata["session"].astype(str)],
        dtype="U3",
    )

    label_map = {"left_hand": 0, "right_hand": 1}
    observed_labels = {str(value) for value in labels}
    if observed_labels != set(label_map):
        raise RuntimeError(
            f"BNCI2014-004 returned unexpected labels: {sorted(observed_labels)}"
        )
    y = np.asarray([label_map[str(value)] for value in labels], dtype=np.int64)
    counts = np.bincount(y, minlength=2)
    if counts[0] != counts[1]:
        raise RuntimeError(f"BNCI2014-004 class counts are not balanced: {counts.tolist()}")
    session_counts = {
        label: int(np.count_nonzero(stable_sessions == label))
        for label in session_labels
    }
    if any(value < 100 or value > 200 for value in session_counts.values()):
        raise RuntimeError(
            f"BNCI2014-004 session trial counts are implausible: {session_counts}"
        )

    x = epochs.get_data(copy=False).astype(np.float32, copy=False)
    channel_names = [str(name) for name in epochs.ch_names]
    if [name.upper() for name in channel_names] != ["C3", "CZ", "C4"]:
        raise RuntimeError(f"Unexpected BNCI2014-004 EEG channels: {channel_names}")
    sfreq = float(epochs.info["sfreq"])
    expected_samples = int(round((float(tmax) - float(tmin)) * sfreq))
    if x.shape[-1] not in {expected_samples, expected_samples + 1}:
        raise RuntimeError(
            f"BNCI2014-004 epoch has {x.shape[-1]} samples, expected "
            f"{expected_samples} or {expected_samples + 1}"
        )
    if not np.isfinite(x).all():
        raise FloatingPointError("BNCI2014-004 data contain NaN or Inf")

    runs = metadata["run"].astype(str).to_numpy()
    counters: dict[tuple[str, str], int] = {}
    trial_ids: list[str] = []
    for session, run in zip(stable_sessions.tolist(), runs.tolist(), strict=True):
        key = (session, run)
        index = counters.get(key, 0)
        counters[key] = index + 1
        trial_ids.append(
            f"BNCI2014-004-S{subject:02d}-{session}-{run}-trial{index:03d}"
        )
    if len(set(trial_ids)) != y.size:
        raise RuntimeError("BNCI2014-004 trial identifiers are not unique")

    return {
        "X": np.ascontiguousarray(x[..., :expected_samples], dtype=np.float32),
        "y": y,
        "subject": np.asarray([str(subject)] * y.size),
        "session": stable_sessions,
        "run": runs,
        "trial_id": np.asarray(trial_ids),
        "sfreq": sfreq,
        "ch_names": channel_names,
        "electrode_coordinates": electrode_coordinates(epochs),
        "spatial_anchor_indices": np.arange(len(channel_names), dtype=np.int64),
        "epoch_tmin": float(tmin),
        "epoch_tmax": float(tmin) + expected_samples / sfreq,
        "dataset_name": "BNCI2014_004",
        "task_events": ["left_hand", "right_hand"],
        "label_map": label_map,
        "n_classes": 2,
        "moabb_session_keys": session_keys,
        "session_trial_counts": session_counts,
        "physical_mat_role": physical_role,
        "heldout_session_accessed": bool({"04E", "05E"}.intersection(session_labels)),
    }
