"""Sampling and integrity helpers for the V31 decoder learning curve."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LearningCurveSubset:
    label: str
    requested_per_class: int | None
    indices: np.ndarray
    class_counts: dict[int, int]

    @property
    def examples_per_class(self) -> float:
        return float(np.mean(list(self.class_counts.values())))

    @property
    def index_sha256(self) -> str:
        value = np.ascontiguousarray(self.indices, dtype=np.int64)
        digest = hashlib.sha256()
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes())
        return digest.hexdigest()


def nested_stratified_subsets(
    labels: np.ndarray,
    requested_per_class: Iterable[int],
    *,
    seed: int,
) -> list[LearningCurveSubset]:
    """Return deterministic nested class-balanced subsets plus the full set.

    Numeric budgets that exceed the least populated class are omitted. A
    numeric budget identical to an exactly balanced full set is also omitted,
    because it would duplicate the ``all`` arm without adding evidence.
    """

    y = np.asarray(labels, dtype=np.int64)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("learning-curve labels must be a non-empty vector")
    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError("learning-curve sampling requires at least two classes")

    generator = np.random.default_rng(int(seed))
    shuffled: dict[int, np.ndarray] = {}
    available: dict[int, int] = {}
    for value in classes:
        indices = np.flatnonzero(y == value).astype(np.int64)
        generator.shuffle(indices)
        shuffled[int(value)] = indices
        available[int(value)] = int(indices.size)

    minimum = min(available.values())
    exactly_balanced = len(set(available.values())) == 1
    budgets = sorted({int(value) for value in requested_per_class})
    if not budgets or any(value < 1 for value in budgets):
        raise ValueError("numeric learning-curve budgets must be positive")

    subsets: list[LearningCurveSubset] = []
    previous: set[int] = set()
    for budget in budgets:
        if budget > minimum or (exactly_balanced and budget == minimum):
            continue
        indices = np.sort(
            np.concatenate([shuffled[int(value)][:budget] for value in classes])
        )
        current = set(indices.tolist())
        if not previous.issubset(current):
            raise RuntimeError("learning-curve subsets are not nested")
        previous = current
        subsets.append(
            LearningCurveSubset(
                label=f"n{budget}",
                requested_per_class=budget,
                indices=indices,
                class_counts={int(value): budget for value in classes},
            )
        )

    full_indices = np.arange(y.size, dtype=np.int64)
    if not previous.issubset(set(full_indices.tolist())):
        raise RuntimeError("numeric subsets are not contained in the full set")
    subsets.append(
        LearningCurveSubset(
            label="all",
            requested_per_class=None,
            indices=full_indices,
            class_counts=available,
        )
    )
    return subsets


def paired_run_seed(dataset_index: int, subject: int, seed: int) -> int:
    """Use one initialisation across budgets and matched ANN/SNN variants."""

    if dataset_index < 0 or subject < 1 or seed < 0:
        raise ValueError("invalid V31 seed components")
    return 3_100_000 + int(dataset_index) * 1_000_000 + int(subject) * 10_000 + int(seed) * 101
