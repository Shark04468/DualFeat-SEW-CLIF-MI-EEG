"""PhysioNet EEGMMI optional loader and preparation hooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dpc_snn.data.async_windows import AsyncWindowConfig, make_windows_from_labeled_intervals
from dpc_snn.preprocessing.epoching import epoch_continuous
from dpc_snn.utils.io import ensure_dir, save_npz
from dpc_snn.utils.storage import configure_cache_env


def load_physionet_with_mne(subjects: list[int], runs: list[int], root: str | Path) -> list[Any]:
    configure_cache_env()
    from dpc_snn.utils.imports import optional_import

    mne = optional_import("mne", "PhysioNet EEGMMI loading")
    raw_paths = mne.datasets.eegbci.load_data(subjects, runs, path=str(root), update_path=True)
    raws = []
    for p in raw_paths:
        raws.append(mne.io.read_raw_edf(p, preload=True, verbose="ERROR"))
    return raws


def physionet_task_mapping() -> dict[str, Any]:
    return {
        "imagined_left_right": {
            "runs": [4, 8, 12],
            "event_map": {"T1": 0, "T2": 1},
            "classes": ["left_fist", "right_fist"],
        },
        "imagined_hands_feet": {
            "runs": [6, 10, 14],
            "event_map": {"T1": 0, "T2": 1},
            "classes": ["both_fists", "both_feet"],
        },
        "rest_vs_mi": {
            "runs": [4, 6, 8, 10, 12, 14],
            "event_map": {"T0": 0, "T1": 1, "T2": 1},
            "classes": ["rest", "motor_imagery"],
        },
    }


def prepare_physionet_npz(
    subjects: list[int],
    task: str,
    root: str | Path,
    output_dir: str | Path,
    tmin: float = 0.0,
    tmax: float = 4.0,
) -> list[Path]:
    """Download/load PhysioNet EEGMMI with MNE and export processed NPZ files."""

    configure_cache_env()
    from dpc_snn.utils.imports import optional_import

    mne = optional_import("mne", "PhysioNet EEGMMI preparation")
    mapping = physionet_task_mapping()
    if task not in mapping:
        raise KeyError(f"Unknown PhysioNet task {task!r}. Known: {sorted(mapping)}")
    task_cfg = mapping[task]
    output_dir = ensure_dir(output_dir)
    raw_paths = mne.datasets.eegbci.load_data(
        subjects, task_cfg["runs"], path=str(root), update_path=True
    )
    by_subject: dict[int, list[dict[str, Any]]] = {int(s): [] for s in subjects}
    for path in raw_paths:
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        raw.rename_channels(lambda s: s.strip("."))
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        event_pairs = []
        for ann, label in task_cfg["event_map"].items():
            if ann in event_id:
                for event in events[events[:, 2] == event_id[ann]]:
                    event_pairs.append((int(event[0]), int(label)))
        picks = mne.pick_types(raw.info, eeg=True, eog=False)
        if not event_pairs:
            continue
        x = raw.get_data(picks=picks)
        epochs, y = epoch_continuous(x, event_pairs, float(raw.info["sfreq"]), tmin, tmax)
        subject = int(Path(path).stem.split("S")[-1][:3]) if "S" in Path(path).stem else int(subjects[0])
        by_subject.setdefault(subject, []).append(
            {
                "X": epochs,
                "y": y,
                "sfreq": float(raw.info["sfreq"]),
                "ch_names": np.asarray([raw.ch_names[i] for i in picks]),
            }
        )
    written = []
    for subject, parts in by_subject.items():
        if not parts:
            continue
        x = np.concatenate([p["X"] for p in parts], axis=0).astype(np.float32)
        y = np.concatenate([p["y"] for p in parts], axis=0).astype(np.int64)
        session = np.asarray([task] * len(y))
        out = output_dir / f"physionet_S{subject:03d}_{task}.npz"
        save_npz(
            out,
            X=x,
            y=y,
            subject=np.asarray([f"S{subject:03d}"] * len(y)),
            session=session,
            sfreq=np.asarray(parts[0]["sfreq"], dtype=np.float32),
            ch_names=parts[0]["ch_names"],
        )
        written.append(out)
    return written


def prepare_physionet_async_npz(
    subjects: list[int],
    task: str,
    root: str | Path,
    output_dir: str | Path,
    window_sec: float = 1.0,
    step_sec: float = 0.25,
    interval_sec: float = 4.0,
) -> list[Path]:
    """Export real continuous PhysioNet recordings as labelled sliding windows.

    Windows are accepted only when wholly inside an annotated T0 (rest), T1, or
    T2 interval.  This keeps transition windows out of both the detector and
    the held-out run-level evaluation.
    """

    configure_cache_env()
    from dpc_snn.utils.imports import optional_import

    mne = optional_import("mne", "PhysioNet asynchronous window preparation")
    mapping = physionet_task_mapping()
    if task not in mapping:
        raise KeyError(f"Unknown PhysioNet task {task!r}. Known: {sorted(mapping)}")
    task_cfg = mapping[task]
    output_dir = ensure_dir(output_dir)
    raw_paths = mne.datasets.eegbci.load_data(subjects, task_cfg["runs"], path=str(root), update_path=True)
    written = []
    for path in raw_paths:
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        raw.rename_channels(lambda s: s.strip("."))
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        picks = mne.pick_types(raw.info, eeg=True, eog=False)
        x = raw.get_data(picks=picks)
        sfreq = float(raw.info["sfreq"])
        intervals = []
        for event in events:
            label = None
            for annotation, event_code in event_id.items():
                if int(event[2]) != int(event_code):
                    continue
                if annotation == "T0":
                    label = -1
                elif annotation in task_cfg["event_map"]:
                    label = int(task_cfg["event_map"][annotation])
                break
            if label is None:
                continue
            start = max(0, int(event[0]) - int(raw.first_samp))
            stop = min(x.shape[-1], start + int(round(interval_sec * sfreq)))
            if stop - start >= int(round(window_sec * sfreq)):
                intervals.append((start, stop, label))
        windows = make_windows_from_labeled_intervals(
            x,
            np.asarray(intervals, dtype=np.int64),
            AsyncWindowConfig(sfreq=sfreq, window_sec=window_sec, step_sec=step_sec),
        )
        stem = Path(path).stem
        subject = int(stem.split("S")[-1][:3]) if "S" in stem else int(subjects[0])
        run = stem.split("R")[-1] if "R" in stem else stem
        recording_id = f"S{subject:03d}_R{run}"
        event_uid = np.asarray(
            [f"{recording_id}:{event_id}" for event_id in np.asarray(windows["event_id"], dtype=int)],
            dtype=str,
        )
        output = output_dir / f"{recording_id}_{task}.npz"
        save_npz(
            output,
            **windows,
            subject=np.asarray([f"S{subject:03d}"] * len(windows["y_binary"])),
            session=np.asarray([f"R{run}"] * len(windows["y_binary"])),
            recording_id=np.asarray([recording_id] * len(windows["y_binary"])),
            event_uid=event_uid,
            ch_names=np.asarray([raw.ch_names[index] for index in picks]),
            dataset_name=np.asarray("physionet_eegmmi_async"),
        )
        written.append(output)
    return written
