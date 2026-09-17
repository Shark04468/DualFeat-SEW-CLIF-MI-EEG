"""Balanced candidate generation and leakage-safe ranking for V8 E5."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
import math
import random
from typing import Any, Mapping, Sequence

import numpy as np


def enumerate_delay_hpo_candidates(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the complete shuffled factorial without front-of-list truncation."""

    factors = config["candidate_factors"]
    required = (
        "decoder_channels",
        "decoder_layers",
        "temporal_decimation",
        "learning_rate",
    )
    if set(factors) != set(required):
        raise ValueError("E5 delay candidate factors changed")
    levels = []
    for factor in required:
        mapping = factors[factor]
        if not isinstance(mapping, Mapping) or not mapping:
            raise ValueError(f"E5 factor {factor} is empty")
        levels.append([(str(label), value) for label, value in mapping.items()])
    candidates = []
    for values in product(*levels):
        labels = {factor: value[0] for factor, value in zip(required, values, strict=True)}
        resolved = {factor: value[1] for factor, value in zip(required, values, strict=True)}
        candidates.append(
            {
                "candidate_id": "__".join(labels[factor] for factor in required),
                "factor_labels": labels,
                **resolved,
            }
        )
    expected = int(config["maximum_unique_configurations"])
    if len(candidates) != expected or len({row["candidate_id"] for row in candidates}) != expected:
        raise ValueError(
            f"E5 factorial produced {len(candidates)} unique candidates; expected {expected}"
        )
    random.Random(int(config["candidate_order_seed"])).shuffle(candidates)
    return candidates


def apply_delay_hpo_candidate(
    base_config: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply only the four registered E5 factors to the matched E3 config."""

    output = deepcopy(dict(base_config))
    output["expert"]["decoder_channels"] = int(candidate["decoder_channels"])
    output["expert"]["decoder_layers"] = int(candidate["decoder_layers"])
    output["expert"]["temporal_decimation"] = int(candidate["temporal_decimation"])
    output["training"]["learning_rate"] = float(candidate["learning_rate"])
    return output


def rank_delay_hpo_candidates(
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
        key = (str(row["candidate_id"]), int(row["subject"]), int(row["fold"]))
        if key in indexed:
            raise ValueError(f"duplicate E5 candidate result: {key}")
        indexed[key] = row
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(f"E5 candidate coverage differs; missing={missing}, extra={extra}")

    ranking = []
    for candidate_id in candidate_ids:
        subject_rows = []
        parameters = set()
        best_epochs = []
        for subject in subjects:
            selected = [
                indexed[(str(candidate_id), int(subject), int(fold))] for fold in folds
            ]
            values = np.asarray(
                [
                    [
                        float(row["validation_kappa"]),
                        float(row["validation_accuracy"]),
                        float(row["validation_nll"]),
                    ]
                    for row in selected
                ],
                dtype=np.float64,
            )
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite E5 metric for candidate {candidate_id}")
            subject_rows.append(values.mean(axis=0))
            parameters.update(int(row["parameters"]) for row in selected)
            best_epochs.extend(int(row["best_epoch"]) for row in selected)
        if len(parameters) != 1:
            raise ValueError(f"candidate {candidate_id} changed parameter count across folds")
        subject_values = np.asarray(subject_rows)
        ranking.append(
            {
                "candidate_id": str(candidate_id),
                "subject_macro_mean_validation_kappa": float(subject_values[:, 0].mean()),
                "subject_macro_mean_validation_accuracy": float(subject_values[:, 1].mean()),
                "subject_macro_mean_validation_nll": float(subject_values[:, 2].mean()),
                "parameters": int(next(iter(parameters))),
                "median_best_epoch": float(np.median(best_epochs)),
                "subjects": len(subjects),
                "folds_per_subject": len(folds),
                "validation_rows": len(subjects) * len(folds),
            }
        )
    ranking.sort(
        key=lambda row: (
            -float(row["subject_macro_mean_validation_kappa"]),
            -float(row["subject_macro_mean_validation_accuracy"]),
            float(row["subject_macro_mean_validation_nll"]),
            int(row["parameters"]),
            str(row["candidate_id"]),
        )
    )
    for index, row in enumerate(ranking, start=1):
        row["rank"] = index
    if not ranking or not math.isfinite(
        float(ranking[0]["subject_macro_mean_validation_kappa"])
    ):
        raise ValueError("E5 ranking is empty or non-finite")
    return ranking
