"""Split helpers for subject/session protocols."""

from __future__ import annotations

import numpy as np


def stratified_split_indices(
    y: np.ndarray,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(idx.size * val_fraction))) if idx.size > 1 else 0
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)


def k_shot_indices(y: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    selected = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        selected.extend(idx[: min(k, idx.size)].tolist())
    return np.asarray(sorted(selected), dtype=int)


def leave_one_subject_out(subjects: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, str]]:
    subjects = np.asarray(subjects)
    splits = []
    for subject in np.unique(subjects):
        test = np.where(subjects == subject)[0]
        train = np.where(subjects != subject)[0]
        splits.append((train, test, str(subject)))
    return splits


def run_grouped_folds(
    y: np.ndarray,
    runs: np.ndarray,
    n_splits: int | None = None,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray, tuple[str, ...]]]:
    """Create deterministic folds without placing one run in both partitions.

    BCI2a has balanced recording runs, but the greedy class-count objective also
    handles small deviations without requiring scikit-learn at data-load time.
    """

    y = np.asarray(y)
    runs = np.asarray(runs).astype(str)
    if y.ndim != 1 or runs.shape != y.shape:
        raise ValueError("run-grouped folds require matching one-dimensional y and runs")
    unique_runs = np.unique(runs)
    if unique_runs.size < 2:
        raise ValueError("run-grouped validation requires at least two distinct runs")
    n_splits = unique_runs.size if n_splits is None else int(n_splits)
    if not 2 <= n_splits <= unique_runs.size:
        raise ValueError("n_splits must lie between 2 and the number of runs")

    classes = np.unique(y)
    rng = np.random.default_rng(seed)
    tie_order = {run: rank for rank, run in enumerate(rng.permutation(unique_runs))}
    group_counts: dict[str, np.ndarray] = {}
    for run in unique_runs:
        mask = runs == run
        group_counts[str(run)] = np.asarray(
            [np.sum(y[mask] == cls) for cls in classes], dtype=np.float64
        )
    ordered = sorted(
        (str(run) for run in unique_runs),
        key=lambda run: (-float(group_counts[run].sum()), tie_order[run]),
    )
    fold_groups: list[list[str]] = [[] for _ in range(n_splits)]
    fold_counts = np.zeros((n_splits, classes.size), dtype=np.float64)
    target = np.asarray([np.sum(y == cls) for cls in classes], dtype=np.float64) / n_splits
    for index, run in enumerate(ordered):
        if index < n_splits:
            selected = index
        else:
            objectives = []
            for fold in range(n_splits):
                candidate = fold_counts.copy()
                candidate[fold] += group_counts[run]
                imbalance = float(np.square(candidate - target).sum())
                objectives.append((imbalance, len(fold_groups[fold]), fold))
            selected = min(objectives)[-1]
        fold_groups[selected].append(run)
        fold_counts[selected] += group_counts[run]

    folds = []
    for groups in fold_groups:
        validation = np.where(np.isin(runs, groups))[0]
        training = np.where(~np.isin(runs, groups))[0]
        if validation.size == 0 or training.size == 0:
            raise RuntimeError("run-grouped fold construction produced an empty partition")
        folds.append((training, validation, tuple(sorted(groups))))
    return folds
