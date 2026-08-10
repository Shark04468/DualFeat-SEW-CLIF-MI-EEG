"""Deterministic, balanced candidate generation and ranking for V8 E5."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import product
import random
from typing import Any

import numpy as np

from dpc_snn.experiments.v62_protocol import sha256_fingerprint


class V8HPOContractError(ValueError):
    """Raised when the bounded E5 search contract is incomplete or biased."""


_TRAINING_KEYS = frozenset({"learning_rate"})


def generate_balanced_v8_candidates(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand and deterministically shuffle the complete registered factorial."""

    factors = config.get("candidate_factors")
    if not isinstance(factors, Mapping) or not factors:
        raise V8HPOContractError("E5 candidate_factors must be a non-empty mapping")
    factor_names = list(factors)
    factor_levels: list[list[tuple[str, Mapping[str, Any]]]] = []
    for factor in factor_names:
        levels = factors[factor]
        if not isinstance(levels, Mapping) or len(levels) < 2:
            raise V8HPOContractError(f"E5 factor {factor!r} requires at least two levels")
        normalized: list[tuple[str, Mapping[str, Any]]] = []
        for level, overrides in levels.items():
            if not isinstance(overrides, Mapping) or not overrides:
                raise V8HPOContractError(
                    f"E5 factor level {factor}.{level} has no overrides"
                )
            normalized.append((str(level), overrides))
        factor_levels.append(normalized)
    fixed_model = dict(config.get("fixed_model", {}))
    if not fixed_model:
        raise V8HPOContractError("E5 fixed_model is empty")

    candidates: list[dict[str, Any]] = []
    for combination in product(*factor_levels):
        selected_levels: dict[str, str] = {}
        model_overrides = dict(fixed_model)
        training_overrides: dict[str, Any] = {}
        for factor, (level, overrides) in zip(
            factor_names, combination, strict=True
        ):
            selected_levels[factor] = level
            for key, value in overrides.items():
                target = training_overrides if key in _TRAINING_KEYS else model_overrides
                if key in target and target[key] != value:
                    raise V8HPOContractError(
                        f"conflicting E5 override for {key!r} in {selected_levels}"
                    )
                target[key] = value
        payload = {
            "factor_levels": selected_levels,
            "model_overrides": model_overrides,
            "training_overrides": training_overrides,
        }
        candidates.append(
            {
                "candidate_id": "hpo_" + sha256_fingerprint(payload)[:12],
                **payload,
            }
        )
    maximum = int(config.get("maximum_unique_configurations", 0))
    if len(candidates) > maximum:
        raise V8HPOContractError(
            f"registered factorial has {len(candidates)} candidates above limit {maximum}"
        )
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise V8HPOContractError("E5 candidate IDs are not unique")
    for factor, levels in zip(factor_names, factor_levels, strict=True):
        observed = Counter(row["factor_levels"][factor] for row in candidates)
        expected = len(candidates) // len(levels)
        if set(observed.values()) != {expected}:
            raise V8HPOContractError(
                f"E5 factor {factor!r} is not marginally balanced: {dict(observed)}"
            )
    rng = random.Random(int(config.get("candidate_order_seed", 0)))
    rng.shuffle(candidates)
    return [dict(row, execution_order=index) for index, row in enumerate(candidates)]


def rank_v8_hpo_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_ids: Sequence[str],
    subjects: Sequence[int],
    folds: Sequence[int],
) -> list[dict[str, Any]]:
    """Rank candidates using subject-macro inner-validation metrics only."""

    expected = {
        (str(candidate), int(subject), int(fold))
        for candidate in candidate_ids
        for subject in subjects
        for fold in folds
    }
    indexed: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("evaluation_role") != "inner_validation":
            raise V8HPOContractError("E5 ranking received a non-inner-validation row")
        key = (str(row["candidate_id"]), int(row["subject"]), int(row["fold"]))
        if key in indexed:
            raise V8HPOContractError(f"duplicate E5 result row {key}")
        indexed[key] = row
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise V8HPOContractError(
            f"E5 ranking coverage mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )

    ranking: list[dict[str, Any]] = []
    for candidate in candidate_ids:
        subject_kappa = []
        subject_accuracy = []
        parameters: set[int] = set()
        selected_epochs = []
        for subject in subjects:
            subset = [indexed[(str(candidate), int(subject), int(fold))] for fold in folds]
            kappa = np.asarray([float(row["validation_kappa"]) for row in subset])
            accuracy = np.asarray([float(row["validation_accuracy"]) for row in subset])
            if not np.isfinite(kappa).all() or not np.isfinite(accuracy).all():
                raise V8HPOContractError(
                    f"non-finite E5 metric for candidate {candidate}, subject {subject}"
                )
            subject_kappa.append(float(kappa.mean()))
            subject_accuracy.append(float(accuracy.mean()))
            parameters.update(int(row["parameters"]) for row in subset)
            selected_epochs.extend(int(row["best_epoch"]) for row in subset)
        if len(parameters) != 1:
            raise V8HPOContractError(
                f"E5 candidate {candidate} changed parameter count across folds"
            )
        ranking.append(
            {
                "candidate_id": str(candidate),
                "subject_macro_mean_validation_kappa": float(np.mean(subject_kappa)),
                "subject_macro_mean_validation_accuracy": float(
                    np.mean(subject_accuracy)
                ),
                "subject_validation_kappa": subject_kappa,
                "subject_validation_accuracy": subject_accuracy,
                "parameters": next(iter(parameters)),
                "mean_best_epoch": float(np.mean(selected_epochs)),
            }
        )
    return sorted(
        ranking,
        key=lambda row: (
            -row["subject_macro_mean_validation_kappa"],
            -row["subject_macro_mean_validation_accuracy"],
            row["parameters"],
            row["candidate_id"],
        ),
    )
