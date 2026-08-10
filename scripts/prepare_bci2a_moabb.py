#!/usr/bin/env python
"""Prepare BCI Competition IV-2a / BNCI2014_001 through MOABB."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from dpc_snn.utils.imports import optional_import  # noqa: E402
from dpc_snn.utils.io import ensure_dir, save_npz  # noqa: E402

configure_cache_env()


def _session_dataset(base_dataset, sessions: Sequence[str]):
    """Return a BNCI2014-001 view that loads only the requested recordings.

    MOABB's public BNCI loader always materializes both T and E.  The V8
    protocol physically keeps Session E unread until the pre-E6 freeze, so a
    session-restricted dataset is required instead of filtering after loading.
    """

    requested = tuple(str(value).upper() for value in sessions)
    if not requested or any(value not in {"T", "E"} for value in requested):
        raise ValueError("sessions must contain only T and/or E")
    if len(set(requested)) != len(requested):
        raise ValueError("sessions must not contain duplicates")
    if set(requested) == {"T", "E"}:
        return base_dataset

    from moabb.datasets.bnci.bnci_2014 import _convert_mi

    channel_names = [
        "Fz",
        "FC3",
        "FC1",
        "FCz",
        "FC2",
        "FC4",
        "C5",
        "C3",
        "C1",
        "Cz",
        "C2",
        "C4",
        "C6",
        "CP3",
        "CP1",
        "CPz",
        "CP2",
        "CP4",
        "P1",
        "Pz",
        "P2",
        "POz",
        "EOG1",
        "EOG2",
        "EOG3",
    ]
    channel_types = ["eeg"] * 22 + ["eog"] * 3

    class SessionRestrictedBNCI2014_001(type(base_dataset)):
        def _get_single_subject_data(self, subject):
            filenames = [Path(value) for value in base_dataset.data_path(subject)]
            by_session = {
                suffix: [path for path in filenames if path.stem.upper().endswith(suffix)]
                for suffix in requested
            }
            missing = [suffix for suffix, paths in by_session.items() if len(paths) != 1]
            if missing:
                raise RuntimeError(
                    f"expected one cached BNCI2014-001 file for {missing}, got {filenames}"
                )
            output = {}
            for session_index, suffix in enumerate(requested):
                runs, _ = _convert_mi(
                    str(by_session[suffix][0]),
                    channel_names,
                    channel_types,
                    dataset_code="BNCI2014-001",
                    subject_id=int(subject),
                )
                session_name = "train" if suffix == "T" else "test"
                output[f"{session_index}{session_name}"] = {
                    str(index): run for index, run in enumerate(runs)
                }
            return output

    return SessionRestrictedBNCI2014_001()


def _first_eeg_channel_names(dataset, subject: int) -> list[str]:
    """Read the EEG channel order from the same MOABB raw recording used by the paradigm."""

    mne = optional_import("mne", "BCI2a MOABB channel metadata")
    records = dataset.get_data(subjects=[subject])
    subject_records = records[int(subject)]
    for runs in subject_records.values():
        for raw in runs.values():
            picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False)
            names = [raw.ch_names[int(index)] for index in picks]
            if names:
                return names
    raise ValueError(f"Could not recover EEG channel names for BCI2a subject {subject}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/bci2a_v62")
    parser.add_argument("--subjects", nargs="*", type=int, default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--tmin", type=float, default=-1.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument("--resample", type=float, default=250.0)
    parser.add_argument(
        "--sessions",
        nargs="+",
        choices=("T", "E"),
        default=("T", "E"),
        help="Physically load only these sessions; use T before the pre-E6 freeze.",
    )
    args = parser.parse_args()

    moabb_datasets = optional_import("moabb.datasets", "BCI2a MOABB preparation")
    moabb_paradigms = optional_import("moabb.paradigms", "BCI2a MOABB preparation")
    dataset = _session_dataset(moabb_datasets.BNCI2014_001(), args.sessions)
    subjects = args.subjects or list(dataset.subject_list)
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]
    if args.resample <= 0.0:
        raise ValueError("--resample must be positive")
    paradigm = moabb_paradigms.MotorImagery(tmin=args.tmin, tmax=args.tmax, resample=args.resample)
    x, y, meta = paradigm.get_data(dataset=dataset, subjects=subjects)
    ch_names = _first_eeg_channel_names(dataset, int(subjects[0]))
    if len(ch_names) != x.shape[1]:
        raise ValueError(f"MOABB returned {x.shape[1]} channels but raw metadata contains {len(ch_names)} EEG channels")
    labels = {label: idx for idx, label in enumerate(sorted(set(y)))}
    y_num = np.asarray([labels[label] for label in y], dtype=np.int64)
    if not {"subject", "session", "run"}.issubset(meta.columns):
        raise ValueError(
            "MOABB BCI2a export requires subject/session/run metadata; "
            f"received columns {sorted(meta.columns)}"
        )
    subject_arr = np.asarray(meta["subject"].astype(str).tolist(), dtype=np.str_)
    session_raw = np.asarray(meta["session"].astype(str).tolist(), dtype=np.str_)
    run_raw = np.asarray(meta["run"].astype(str).tolist(), dtype=np.str_)
    session_arr = np.empty(session_raw.shape, dtype="<U8")
    for subject in sorted(set(subject_arr)):
        mask = subject_arr == subject
        sessions = list(dict.fromkeys(session_raw[mask].tolist()))
        expected_sessions = {str(value).upper() for value in args.sessions}
        if len(sessions) != len(expected_sessions):
            raise ValueError(
                f"BCI2a subject {subject} must expose {len(expected_sessions)} "
                f"requested session(s), got {sessions}"
            )
        session_map: dict[str, str] = {}
        for session in sessions:
            normalized = session.lower()
            if "train" in normalized:
                session_map[session] = "T"
            elif "test" in normalized:
                session_map[session] = "E"
        if set(session_map.values()) != expected_sessions:
            raise ValueError(
                "BCI2a session names must identify the requested train/test sessions "
                f"explicitly; expected {sorted(expected_sessions)}, got {sessions}"
            )
        session_arr[mask] = [session_map[s] for s in session_raw[mask]]

    trial_ids = np.empty(len(y_num), dtype="<U64")
    counters: dict[tuple[str, str, str], int] = {}
    for index, (subject, session, run) in enumerate(
        zip(subject_arr, session_arr, run_raw, strict=True)
    ):
        key = (str(subject), str(session), str(run))
        ordinal = counters.get(key, 0)
        counters[key] = ordinal + 1
        trial_ids[index] = f"bci2a:A{int(subject):02d}:{session}:{run}:{ordinal:03d}"

    output = ensure_dir(args.output)
    written = []
    for subject in sorted(set(subject_arr)):
        idx = np.where(subject_arr == subject)[0]
        out = output / f"A{int(subject):02d}.npz" if str(subject).isdigit() else output / f"{subject}.npz"
        save_npz(
            out,
            X=np.asarray(x[idx], dtype=np.float32),
            y=y_num[idx],
            subject=subject_arr[idx],
            session=session_arr[idx],
            run=run_raw[idx],
            trial_id=trial_ids[idx],
            sfreq=np.asarray(args.resample, dtype=np.float32),
            ch_names=np.asarray(ch_names),
            label_names=np.asarray([k for k, _ in sorted(labels.items(), key=lambda item: item[1])]),
            moabb_session=session_raw[idx],
            moabb_run=run_raw[idx],
            epoch_tmin=np.asarray(args.tmin, dtype=np.float32),
            epoch_tmax=np.asarray(args.tmax, dtype=np.float32),
            dataset_name=np.asarray("bci2a"),
        )
        written.append(out)
    print({"subjects": subjects, "label_map": labels, "written": [str(p) for p in written]})


if __name__ == "__main__":
    main()
