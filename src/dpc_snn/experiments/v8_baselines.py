"""Session-T-only helpers for the V8 strong-baseline campaign."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dpc_snn.experiments.v62_protocol import validate_trial_metadata
from dpc_snn.experiments.v8_protocol import assert_v8_data_access


class V8BaselineProtocolError(ValueError):
    """Raised when a baseline run violates the locked development protocol."""


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else value


def session_t_development_view(
    data: Mapping[str, Any],
    *,
    expected_trials: int = 288,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Return the Session-T tensors before any feature or fitting operation.

    The processed BCI2a container also carries Session E.  This function first
    resolves the session mask, then exposes only T-indexed signals, labels, and
    metadata to development code.  The returned access manifest makes that
    boundary explicit and is included in every run fingerprint.
    """

    required = {"X", "y", "subject", "session", "run", "trial_id"}
    missing = sorted(required.difference(data))
    if missing:
        raise V8BaselineProtocolError(f"processed data is missing fields: {missing}")

    sessions = np.asarray(data["session"]).astype(str)
    indices = np.flatnonzero(sessions == "T")
    if indices.size != int(expected_trials):
        raise V8BaselineProtocolError(
            f"expected {expected_trials} Session-T trials, got {indices.size}"
        )

    x = np.asarray(data["X"], dtype=np.float32)[indices]
    y = np.asarray(data["y"], dtype=np.int64)[indices]
    subjects = np.asarray(data["subject"]).astype(str)[indices]
    runs = np.asarray(data["run"]).astype(str)[indices]
    trial_ids = np.asarray(data["trial_id"]).astype(str)[indices]
    dataset = str(_scalar(data.get("dataset_name", "bci2a")))
    sfreq = float(_scalar(data["sfreq"]))
    epoch_tmin = float(_scalar(data["epoch_tmin"]))
    epoch_tmax = float(_scalar(data["epoch_tmax"]))
    channel_names = [str(name) for name in data["ch_names"]]

    rows = validate_trial_metadata(
        [
            {
                "dataset": dataset,
                "subject": subjects[index],
                "session": "T",
                "run": runs[index],
                "trial_id": trial_ids[index],
                "class": int(y[index]),
                "sfreq": sfreq,
                "ch_names": channel_names,
                "epoch_tmin": epoch_tmin,
                "epoch_tmax": epoch_tmax,
            }
            for index in range(indices.size)
        ],
        allowed_sessions=("T",),
    )
    assert_v8_data_access(rows, stage="development", role="training")
    assert_v8_data_access(rows, stage="development", role="validation")
    if len(set(trial_ids.tolist())) != indices.size:
        raise V8BaselineProtocolError("Session-T trial identifiers are not unique")

    labels, counts = np.unique(y, return_counts=True)
    if labels.tolist() != [0, 1, 2, 3] or len(set(counts.tolist())) != 1:
        raise V8BaselineProtocolError(
            f"Session-T labels are not balanced four-class targets: {dict(zip(labels, counts))}"
        )
    excluded_counts = {
        session: int(np.sum(sessions == session))
        for session in sorted(set(sessions.tolist()) - {"T"})
    }
    access_manifest = {
        "stage": "development",
        "selected_session": "T",
        "selected_trial_count": int(indices.size),
        "selected_container_indices_sha256_input": indices.tolist(),
        "excluded_session_trial_counts": excluded_counts,
        "heldout_signals_used_by_development": False,
        "heldout_labels_used_by_development": False,
        "allowed_roles": ["training", "validation", "normalization", "augmentation"],
    }
    return x, y, rows, access_manifest


def merge_oof_predictions(
    parts: Sequence[Mapping[str, Any]],
    *,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge fold predictions while requiring exact one-pass OOF coverage."""

    if not parts:
        raise V8BaselineProtocolError("OOF prediction parts are empty")
    indices = np.concatenate([np.asarray(part["indices"], dtype=np.int64) for part in parts])
    logits = np.concatenate([np.asarray(part["logits"], dtype=np.float32) for part in parts])
    fold_labels = np.concatenate(
        [np.asarray(part["labels"], dtype=np.int64) for part in parts]
    )
    count = int(np.asarray(labels).shape[0])
    if indices.shape != (count,) or sorted(indices.tolist()) != list(range(count)):
        raise V8BaselineProtocolError(
            "OOF folds must cover every Session-T trial exactly once"
        )
    if logits.ndim != 2 or logits.shape != (count, 4):
        raise V8BaselineProtocolError(f"expected OOF logits [{count}, 4], got {logits.shape}")
    order = np.argsort(indices)
    indices = indices[order]
    logits = logits[order]
    fold_labels = fold_labels[order]
    expected = np.asarray(labels, dtype=np.int64)
    if not np.array_equal(fold_labels, expected):
        raise V8BaselineProtocolError("fold labels do not match Session-T labels")
    if not np.isfinite(logits).all():
        raise V8BaselineProtocolError("OOF logits contain non-finite values")
    return indices, logits, fold_labels


def nested_run_grouped_indices(
    metadata: Sequence[Mapping[str, Any]],
    outer_train_indices: np.ndarray,
    outer_test_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Choose one inner validation run without touching the outer test run.

    The inner run is the cyclic successor of the outer test run in canonical
    run order.  Across six BCI2a outer folds this rotates every run through both
    roles and is independent of labels, model identity, and observed metrics.
    """

    rows = list(metadata)
    outer_train = np.asarray(outer_train_indices, dtype=np.int64)
    outer_test = np.asarray(outer_test_indices, dtype=np.int64)
    if outer_train.ndim != 1 or outer_test.ndim != 1:
        raise V8BaselineProtocolError("nested fold indices must be one-dimensional")
    if set(outer_train.tolist()) & set(outer_test.tolist()):
        raise V8BaselineProtocolError("outer train and test indices overlap")
    if sorted(np.concatenate((outer_train, outer_test)).tolist()) != list(range(len(rows))):
        raise V8BaselineProtocolError("outer fold does not partition Session T")
    all_runs = sorted({str(row["run"]) for row in rows})
    test_runs = {str(rows[int(index)]["run"]) for index in outer_test}
    if len(test_runs) != 1:
        raise V8BaselineProtocolError("each outer test fold must contain exactly one run")
    outer_test_run = next(iter(test_runs))
    if outer_test_run not in all_runs or len(all_runs) < 3:
        raise V8BaselineProtocolError("nested run selection requires at least three runs")
    test_position = all_runs.index(outer_test_run)
    inner_run = all_runs[(test_position + 1) % len(all_runs)]
    inner_validation = np.asarray(
        [index for index in outer_train if str(rows[int(index)]["run"]) == inner_run],
        dtype=np.int64,
    )
    inner_train = np.asarray(
        [index for index in outer_train if str(rows[int(index)]["run"]) != inner_run],
        dtype=np.int64,
    )
    if inner_train.size == 0 or inner_validation.size == 0:
        raise V8BaselineProtocolError("nested fold produced an empty inner split")
    if set(inner_train.tolist()) & set(inner_validation.tolist()):
        raise V8BaselineProtocolError("inner train and validation indices overlap")
    if sorted(np.concatenate((inner_train, inner_validation)).tolist()) != sorted(
        outer_train.tolist()
    ):
        raise V8BaselineProtocolError("inner split does not partition the outer train fold")
    return inner_train, inner_validation, inner_run


def rank_screening_models(
    rows: Sequence[Mapping[str, Any]],
    *,
    subjects: Sequence[int],
    screening_seed: int,
) -> list[dict[str, Any]]:
    """Rank models by macro subject-level Session-T OOF accuracy."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["seed"]) == int(screening_seed):
            grouped[str(row["model"])].append(row)
    expected_subjects = sorted(int(subject) for subject in subjects)
    ranking: list[dict[str, Any]] = []
    for model, model_rows in grouped.items():
        observed = sorted(int(row["subject"]) for row in model_rows)
        if observed != expected_subjects:
            raise V8BaselineProtocolError(
                f"screening model {model!r} has subjects {observed}, expected {expected_subjects}"
            )
        ranking.append(
            {
                "model": model,
                "screening_seed": int(screening_seed),
                "subjects": len(model_rows),
                "mean_subject_accuracy": float(
                    np.mean([float(row["accuracy"]) for row in model_rows])
                ),
                "mean_subject_kappa": float(
                    np.mean([float(row["kappa"]) for row in model_rows])
                ),
                "mean_subject_macro_f1": float(
                    np.mean([float(row["macro_f1"]) for row in model_rows])
                ),
            }
        )
    return sorted(
        ranking,
        key=lambda row: (
            -row["mean_subject_accuracy"],
            -row["mean_subject_kappa"],
            row["model"],
        ),
    )
