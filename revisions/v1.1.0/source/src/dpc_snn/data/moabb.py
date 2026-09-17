"""MOABB optional integration."""

from __future__ import annotations

from typing import Any

import numpy as np

from dpc_snn.utils.storage import configure_cache_env


def get_moabb_dataset(name: str) -> Any:
    configure_cache_env()
    from dpc_snn.utils.imports import optional_import

    moabb_datasets = optional_import("moabb.datasets", "MOABB mini benchmark")
    if not hasattr(moabb_datasets, name):
        raise KeyError(f"MOABB dataset {name!r} not found")
    return getattr(moabb_datasets, name)()


def load_moabb_motor_imagery(
    dataset_names: list[str],
    max_subjects: int | None = None,
    tmin: float = 0.0,
    tmax: float = 4.0,
    include_rest: bool = False,
    resample: float | None = None,
    n_classes: int | None = None,
    events: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load MOABB motor-imagery arrays as NPZ-like dictionaries.

    This keeps MOABB optional and avoids forcing MOABB objects into the rest of
    the codebase. Each returned item represents one dataset.
    """

    configure_cache_env()
    from dpc_snn.utils.imports import optional_import

    paradigms = optional_import("moabb.paradigms", "MOABB motor imagery loading")
    MotorImagery = getattr(paradigms, "MotorImagery")
    if resample is None:
        raise ValueError(
            "MOABB benchmark loading requires an explicit positive resample rate. "
            "Dataset classes do not reliably expose their native sampling rates."
        )
    try:
        target_sfreq = float(resample)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid MOABB resample rate: {resample!r}") from exc
    if not np.isfinite(target_sfreq) or target_sfreq <= 0.0:
        raise ValueError(f"Invalid MOABB resample rate: {resample!r}")

    if n_classes is None or n_classes < 2:
        raise ValueError("MOABB benchmark loading requires an explicit n_classes >= 2.")
    if not events or len(events) != n_classes:
        raise ValueError("MOABB benchmark loading requires an explicit event list matching n_classes.")
    requested_events = [str(event) for event in events]
    if len(set(requested_events)) != len(requested_events):
        raise ValueError("MOABB benchmark event names must be unique.")

    loaded = []
    for name in dataset_names:
        dataset = get_moabb_dataset(name)
        subjects = getattr(dataset, "subject_list", None)
        if subjects is not None and max_subjects is not None:
            subjects = list(subjects)[:max_subjects]
        epoch_tmin = tmin
        epoch_tmax = tmax
        try:
            paradigm = MotorImagery(
                n_classes=int(n_classes),
                events=requested_events,
                tmin=epoch_tmin,
                tmax=epoch_tmax,
                resample=target_sfreq,
            )
        except TypeError as exc:
            raise RuntimeError(
                "Installed MOABB does not support explicit tmin/tmax/resample arguments; "
                "refuse to infer a sampling rate from array length."
            ) from exc
        skipped_subjects = []
        try:
            x, y, meta = paradigm.get_data(dataset=dataset, subjects=subjects)
        except ValueError as exc:
            if "input array dimensions" not in str(exc) or subjects is None:
                raise
            chunks = []
            labels = []
            metas = []
            for subject in subjects:
                try:
                    sx, sy, smeta = paradigm.get_data(dataset=dataset, subjects=[subject])
                except ValueError as subject_exc:
                    skipped_subjects.append({"subject": subject, "error": repr(subject_exc)})
                    continue
                chunks.append(np.asarray(sx))
                labels.extend(list(sy))
                metas.append(smeta)
            if not chunks:
                raise ValueError(f"No MOABB subjects could be loaded for {name}; skipped={skipped_subjects}") from exc
            min_time = min(chunk.shape[-1] for chunk in chunks)
            x = np.concatenate([chunk[..., :min_time] for chunk in chunks], axis=0)
            y = np.asarray(labels)
            try:
                import pandas as pd

                meta = pd.concat(metas, ignore_index=True)
            except Exception:
                meta = {}
        excluded_rest_trials = 0
        if not include_rest:
            y_arr = np.asarray(y)
            keep = np.asarray(["rest" not in str(label).lower() for label in y_arr], dtype=bool)
            excluded_rest_trials = int((~keep).sum())
            if keep.any():
                x = np.asarray(x)[keep]
                y = y_arr[keep]
                try:
                    meta = meta.iloc[keep].reset_index(drop=True)
                except Exception:
                    pass
            else:
                raise ValueError(f"All MOABB trials for {name} are rest; cannot build a motor-imagery benchmark.")
        observed_events = sorted({str(label) for label in y})
        if observed_events != sorted(requested_events):
            raise ValueError(
                f"MOABB dataset {name} returned events {observed_events}, expected exactly {sorted(requested_events)}."
            )
        labels = {label: idx for idx, label in enumerate(requested_events)}
        y_num = np.asarray([labels[label] for label in y], dtype=np.int64)
        subject = meta["subject"].astype(str).to_numpy() if "subject" in meta else np.asarray(["unknown"] * len(y_num))
        session = meta["session"].astype(str).to_numpy() if "session" in meta else np.asarray(["unknown"] * len(y_num))
        loaded.append(
            {
                "name": name,
                "X": np.asarray(x, dtype=np.float32),
                "y": y_num,
                "subject": subject,
                "session": session,
                "label_map": labels,
                "task_events": requested_events,
                "n_classes": int(n_classes),
                "sfreq": target_sfreq,
                "resample_sfreq": target_sfreq,
                "epoch_tmin": epoch_tmin,
                "epoch_tmax": epoch_tmax,
                "epoch_time_samples": int(np.asarray(x).shape[-1]),
                "skipped_subjects": skipped_subjects,
                "excluded_rest_trials": excluded_rest_trials,
            }
        )
    return loaded
