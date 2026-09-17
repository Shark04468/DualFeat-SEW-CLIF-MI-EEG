"""BCI Competition IV-2a / BNCI 001-2014 loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_processed_npz(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files found under {p}")
        parts = [load_processed_npz(f) for f in files]
        merged: dict[str, Any] = {}
        for key in ["X", "y", "subject", "session"]:
            merged[key] = np.concatenate([np.asarray(part[key]) for part in parts], axis=0)
        for key in ["run", "trial_id", "moabb_session", "moabb_run"]:
            present = [key in part for part in parts]
            if any(present) and not all(present):
                raise ValueError(f"Processed EEG files under {p} mix missing and present {key} metadata.")
            if all(present):
                merged[key] = np.concatenate(
                    [np.asarray(part[key]).astype(str) for part in parts], axis=0
                )
        sfreqs = [float(part.get("sfreq", 250.0)) for part in parts]
        if any(not np.isfinite(rate) or rate <= 0.0 for rate in sfreqs):
            raise ValueError(f"Processed EEG files under {p} contain an invalid sampling rate: {sfreqs}")
        if not np.allclose(sfreqs, sfreqs[0]):
            raise ValueError(f"Processed EEG files under {p} have inconsistent sampling rates: {sfreqs}")
        channel_lists = [part.get("ch_names") for part in parts]
        if any(channels is None for channels in channel_lists):
            if not all(channels is None for channels in channel_lists):
                raise ValueError(f"Processed EEG files under {p} mix missing and present channel metadata.")
            merged["ch_names"] = None
        else:
            reference = [str(name) for name in channel_lists[0]]
            if any([str(name) for name in channels] != reference for channels in channel_lists[1:]):
                raise ValueError(f"Processed EEG files under {p} have inconsistent channel names or order.")
            merged["ch_names"] = reference
        merged["sfreq"] = sfreqs[0]
        for key in ["epoch_tmin", "epoch_tmax", "dataset_name", "label_names"]:
            values = [part.get(key) for part in parts]
            if all(value is not None for value in values):
                reference = values[0]
                if isinstance(reference, (list, tuple, np.ndarray)):
                    equal = all(
                        np.array_equal(np.asarray(value), np.asarray(reference))
                        for value in values[1:]
                    )
                else:
                    equal = all(value == reference for value in values[1:])
                if not equal:
                    raise ValueError(f"Processed EEG files under {p} have inconsistent {key} metadata.")
                merged[key] = reference
        return merged

    with np.load(p, allow_pickle=True) as npz:
        required = {"X", "y"}
        missing = required.difference(npz.files)
        if missing:
            raise ValueError(f"{p} is missing required arrays: {sorted(missing)}")
        x = np.asarray(npz["X"], dtype=np.float32)
        y = np.asarray(npz["y"], dtype=np.int64)
        if x.ndim != 3 or x.shape[0] != y.shape[0]:
            raise ValueError(f"{p} must contain X=[trials, channels, time] and matching y; got {x.shape}, {y.shape}")
        n = x.shape[0]
        subject = np.asarray(
            npz["subject"] if "subject" in npz.files else ["unknown"] * n
        ).astype(str)
        session = np.asarray(
            npz["session"] if "session" in npz.files else ["unknown"] * n
        ).astype(str)
        run = np.asarray(npz["run"]).astype(str) if "run" in npz.files else None
        trial_id = (
            np.asarray(npz["trial_id"]).astype(str) if "trial_id" in npz.files else None
        )
        moabb_session = (
            np.asarray(npz["moabb_session"]).astype(str)
            if "moabb_session" in npz.files
            else None
        )
        moabb_run = (
            np.asarray(npz["moabb_run"]).astype(str)
            if "moabb_run" in npz.files
            else None
        )
        sfreq = float(np.asarray(npz["sfreq"]).item()) if "sfreq" in npz.files else 250.0
        ch_names = np.asarray(npz["ch_names"]).tolist() if "ch_names" in npz.files else None
        epoch_tmin = float(np.asarray(npz["epoch_tmin"]).item()) if "epoch_tmin" in npz.files else None
        epoch_tmax = float(np.asarray(npz["epoch_tmax"]).item()) if "epoch_tmax" in npz.files else None
        dataset_name = str(np.asarray(npz["dataset_name"]).item()) if "dataset_name" in npz.files else None
        label_names = (
            np.asarray(npz["label_names"]).astype(str).tolist()
            if "label_names" in npz.files
            else None
        )
    if not np.isfinite(sfreq) or sfreq <= 0.0:
        raise ValueError(f"{p} has invalid sampling rate {sfreq!r}")
    if ch_names is not None and len(ch_names) != x.shape[1]:
        raise ValueError(f"{p} has {len(ch_names)} channel names for {x.shape[1]} EEG channels")
    out = {"X": x, "y": y, "subject": subject, "session": session, "sfreq": sfreq, "ch_names": ch_names}
    for key, value in {
        "run": run,
        "trial_id": trial_id,
        "moabb_session": moabb_session,
        "moabb_run": moabb_run,
    }.items():
        if value is not None:
            if value.shape != (n,):
                raise ValueError(f"{p} has invalid {key} shape {value.shape}; expected {(n,)}")
            out[key] = value
    if epoch_tmin is not None:
        out["epoch_tmin"] = epoch_tmin
    if epoch_tmax is not None:
        out["epoch_tmax"] = epoch_tmax
    if dataset_name is not None:
        out["dataset_name"] = dataset_name
    if label_names is not None:
        out["label_names"] = label_names
    return out


def subject_session_split(
    data: dict[str, Any],
    subject: str | int,
    train_session: str = "T",
    test_session: str = "E",
) -> tuple[dict[str, Any], dict[str, Any]]:
    subjects = np.asarray(data["subject"]).astype(str)
    sessions = np.asarray(data["session"]).astype(str)
    subject = str(subject)
    train_idx = np.where((subjects == subject) & (sessions == train_session))[0]
    test_idx = np.where((subjects == subject) & (sessions == test_session))[0]
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError(f"Missing train/test data for subject={subject}, sessions={train_session}->{test_session}")
    return subset(data, train_idx), subset(data, test_idx)


def subject_session_data(
    data: dict[str, Any],
    subject: str | int,
    session: str,
) -> dict[str, Any]:
    """Load one subject/session without requiring or touching another session."""

    subjects = np.asarray(data["subject"]).astype(str)
    sessions = np.asarray(data["session"]).astype(str)
    indices = np.where((subjects == str(subject)) & (sessions == str(session)))[0]
    if indices.size == 0:
        raise ValueError(f"Missing data for subject={subject}, session={session}")
    return subset(data, indices)


def subset(data: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    out = {}
    for key, value in data.items():
        arr = np.asarray(value)
        if arr.shape[:1] == (len(data["y"]),):
            out[key] = arr[indices]
        else:
            out[key] = value
    return out


def prepare_from_mne_gdf(
    raw_paths: list[str | Path],
    output_dir: str | Path,
    tmin: float = -1.0,
    tmax: float = 4.0,
) -> list[Path]:
    """Prepare GDF files with MNE if it is installed.

    This helper is intentionally conservative. Final BCI2a event-id handling
    should be checked against the downloaded data version and recorded in E0.
    """

    from dpc_snn.utils.imports import optional_import
    from dpc_snn.preprocessing.epoching import epoch_continuous
    from dpc_snn.utils.io import ensure_dir, save_npz

    mne = optional_import("mne", "raw BCI2a GDF preparation")
    output_dir = ensure_dir(output_dir)
    written = []
    for raw_path in raw_paths:
        raw_path = Path(raw_path)
        raw = mne.io.read_raw_gdf(raw_path, preload=True, verbose="ERROR")
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        mi_codes = {k: v for k, v in event_id.items() if k in {"769", "770", "771", "772"}}
        if not mi_codes:
            raise ValueError(f"No MI cue annotations found in {raw_path}")
        inv = {v: i for i, v in enumerate(sorted(mi_codes.values()))}
        event_pairs = [(int(e[0]), inv[int(e[2])]) for e in events if int(e[2]) in inv]
        picks = mne.pick_types(raw.info, eeg=True, eog=False)
        x = raw.get_data(picks=picks)
        epochs, y = epoch_continuous(x, event_pairs, raw.info["sfreq"], tmin, tmax)
        subject = raw_path.stem[:3]
        session = "T" if raw_path.stem.endswith("T") else "E"
        out = output_dir / f"{raw_path.stem}.npz"
        save_npz(
            out,
            X=epochs,
            y=y,
            subject=np.asarray([subject] * len(y)),
            session=np.asarray([session] * len(y)),
            sfreq=np.asarray(raw.info["sfreq"], dtype=np.float32),
            ch_names=np.asarray([raw.ch_names[i] for i in picks]),
            epoch_tmin=np.asarray(tmin, dtype=np.float32),
            epoch_tmax=np.asarray(tmax, dtype=np.float32),
        )
        written.append(out)
    return written
