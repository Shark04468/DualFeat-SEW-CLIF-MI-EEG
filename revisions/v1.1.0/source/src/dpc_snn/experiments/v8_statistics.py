"""Paired statistical summaries for the V8 staged gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np


class V8PairingError(ValueError):
    """Raised when two experiment arms cannot be paired exactly."""


def pair_subject_seed_rows(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
    *,
    value: str = "accuracy",
) -> list[dict[str, Any]]:
    """Pair two arms by subject and seed with no silent row dropping."""

    def index(rows: Sequence[Mapping[str, Any]], name: str) -> dict[tuple[int, int], Mapping[str, Any]]:
        output: dict[tuple[int, int], Mapping[str, Any]] = {}
        for row in rows:
            key = (int(row["subject"]), int(row["seed"]))
            if key in output:
                raise V8PairingError(f"{name} contains duplicate subject-seed row {key}")
            output[key] = row
        return output

    first_index = index(first, "first arm")
    second_index = index(second, "second arm")
    if set(first_index) != set(second_index):
        missing_first = sorted(set(second_index) - set(first_index))
        missing_second = sorted(set(first_index) - set(second_index))
        raise V8PairingError(
            f"subject-seed coverage differs; missing_first={missing_first}, "
            f"missing_second={missing_second}"
        )
    paired = []
    for subject, seed in sorted(first_index):
        first_value = float(first_index[(subject, seed)][value])
        second_value = float(second_index[(subject, seed)][value])
        if not math.isfinite(first_value) or not math.isfinite(second_value):
            raise V8PairingError(f"non-finite paired metric at subject={subject}, seed={seed}")
        paired.append(
            {
                "subject": subject,
                "seed": seed,
                "first": first_value,
                "second": second_value,
                "delta_second_minus_first": second_value - first_value,
            }
        )
    return paired


def paired_delta_summary(
    paired: Sequence[Mapping[str, Any]],
    *,
    seed: int = 0,
    bootstrap_samples: int = 20_000,
) -> dict[str, Any]:
    """Return subject-blocked inference plus seed-pair diagnostics."""

    delta = np.asarray([float(row["delta_second_minus_first"]) for row in paired])
    if delta.ndim != 1 or delta.size < 1 or not np.isfinite(delta).all():
        raise V8PairingError("paired deltas must be a non-empty finite vector")
    subject_values: dict[int, list[float]] = {}
    for row in paired:
        subject_values.setdefault(int(row["subject"]), []).append(
            float(row["delta_second_minus_first"])
        )
    if not subject_values:
        raise V8PairingError("paired rows do not contain any subjects")
    seed_counts = {subject: len(values) for subject, values in subject_values.items()}
    if len(set(seed_counts.values())) != 1:
        raise V8PairingError(f"subjects have unequal seed coverage: {seed_counts}")
    subject_delta = np.asarray(
        [np.mean(subject_values[subject]) for subject in sorted(subject_values)], dtype=np.float64
    )
    rng = np.random.default_rng(int(seed))
    subject_indices = rng.integers(
        0, subject_delta.size, size=(int(bootstrap_samples), subject_delta.size)
    )
    subject_bootstrap = subject_delta[subject_indices].mean(axis=1)
    pair_indices = rng.integers(0, delta.size, size=(int(bootstrap_samples), delta.size))
    pair_bootstrap = delta[pair_indices].mean(axis=1)
    positives = int(np.sum(delta > 0.0))
    negatives = int(np.sum(delta < 0.0))
    nonzero = positives + negatives
    if nonzero:
        pair_one_sided_sign_p = sum(
            math.comb(nonzero, count) for count in range(positives, nonzero + 1)
        ) / (2.0**nonzero)
    else:
        pair_one_sided_sign_p = 1.0
    positive_subjects = int(np.sum(subject_delta > 0.0))
    negative_subjects = int(np.sum(subject_delta < 0.0))
    nonzero_subjects = positive_subjects + negative_subjects
    if nonzero_subjects:
        subject_one_sided_sign_p = sum(
            math.comb(nonzero_subjects, count)
            for count in range(positive_subjects, nonzero_subjects + 1)
        ) / (2.0**nonzero_subjects)
    else:
        subject_one_sided_sign_p = 1.0
    return {
        "pairs": int(delta.size),
        "subjects": int(subject_delta.size),
        "seeds_per_subject": next(iter(seed_counts.values())),
        "subject_macro_mean_delta": float(subject_delta.mean()),
        "subject_macro_median_delta": float(np.median(subject_delta)),
        "subject_standard_deviation": (
            float(subject_delta.std(ddof=1)) if subject_delta.size > 1 else 0.0
        ),
        "subject_bootstrap_mean_ci95_low": float(np.quantile(subject_bootstrap, 0.025)),
        "subject_bootstrap_mean_ci95_high": float(np.quantile(subject_bootstrap, 0.975)),
        "positive_subjects": positive_subjects,
        "negative_subjects": negative_subjects,
        "tied_subjects": int(subject_delta.size - nonzero_subjects),
        "subject_one_sided_sign_p": float(subject_one_sided_sign_p),
        "pair_diagnostic_mean_delta": float(delta.mean()),
        "pair_diagnostic_median_delta": float(np.median(delta)),
        "pair_diagnostic_standard_deviation": (
            float(delta.std(ddof=1)) if delta.size > 1 else 0.0
        ),
        "pair_diagnostic_bootstrap_mean_ci95_low": float(
            np.quantile(pair_bootstrap, 0.025)
        ),
        "pair_diagnostic_bootstrap_mean_ci95_high": float(
            np.quantile(pair_bootstrap, 0.975)
        ),
        "positive_pairs": positives,
        "negative_pairs": negatives,
        "tied_pairs": int(delta.size - nonzero),
        "pair_diagnostic_one_sided_sign_p": float(pair_one_sided_sign_p),
    }


def static_delay_gate_decision(
    paired: Sequence[Mapping[str, Any]],
    *,
    expected_pairs: int = 9,
    minimum_median_gain_pp: float = 0.5,
    minimum_positive_pairs: int = 6,
    seed: int = 0,
    bootstrap_samples: int = 20_000,
) -> dict[str, Any]:
    """Apply the registered E3 pair gate without treating seeds as subjects."""

    if len(paired) != int(expected_pairs):
        raise V8PairingError(
            f"static delay gate requires {expected_pairs} pairs, got {len(paired)}"
        )
    summary = paired_delta_summary(
        paired,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
    )
    threshold = float(minimum_median_gain_pp) / 100.0
    passed = bool(
        summary["pair_diagnostic_median_delta"] >= threshold
        and summary["positive_pairs"] >= int(minimum_positive_pairs)
    )
    return {
        "passed": passed,
        "minimum_median_gain_pp": float(minimum_median_gain_pp),
        "minimum_positive_pairs": int(minimum_positive_pairs),
        "observed_median_gain_pp": 100.0 * summary["pair_diagnostic_median_delta"],
        "observed_positive_pairs": int(summary["positive_pairs"]),
        "paired_summary": summary,
    }


def matched_snn_gate_decision(
    candidates: Sequence[Mapping[str, Any]],
    *,
    maximum_accuracy_gap_pp: float = 0.3,
    minimum_early_gain_pp: float = 0.5,
    minimum_activity_density_reduction: float = 0.5,
    minimum_nondegenerate_activity_rate: float = 0.005,
    maximum_sparse_activity_rate: float = 0.25,
) -> dict[str, Any]:
    """Select the strongest development SNN and apply the registered E4 gate."""

    if not candidates:
        raise V8PairingError("matched SNN gate requires at least one candidate")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for order, candidate in enumerate(candidates):
        variant = str(candidate["variant"])
        if variant in seen:
            raise V8PairingError(f"duplicate matched SNN candidate {variant!r}")
        seen.add(variant)
        final_gain = float(candidate["final_gain_pp"])
        early_gain = float(candidate["early_gain_pp"])
        ann_activity = float(candidate["ann_final_activity_nonzero_rate"])
        snn_activity = float(candidate["snn_final_activity_nonzero_rate"])
        binary_rate = float(candidate["snn_binary_spike_rate"])
        values = (final_gain, early_gain, ann_activity, snn_activity, binary_rate)
        if not all(math.isfinite(value) for value in values):
            raise V8PairingError(f"non-finite E4 utility metric for {variant!r}")
        if ann_activity <= 0.0 or not 0.0 <= snn_activity <= 1.0:
            raise V8PairingError(f"invalid activity density for {variant!r}")
        if not 0.0 <= binary_rate <= 1.0:
            raise V8PairingError(f"invalid binary spike rate for {variant!r}")
        reduction = 1.0 - snn_activity / ann_activity
        final_win = final_gain > 0.0
        early_win = early_gain >= float(minimum_early_gain_pp)
        sparsity_win = bool(
            reduction >= float(minimum_activity_density_reduction)
            and snn_activity >= float(minimum_nondegenerate_activity_rate)
            and snn_activity <= float(maximum_sparse_activity_rate)
        )
        rows.append(
            {
                **dict(candidate),
                "variant": variant,
                "registered_order": order,
                "final_gain_pp": final_gain,
                "early_gain_pp": early_gain,
                "activity_density_reduction": reduction,
                "final_accuracy_win": final_win,
                "early_accuracy_win": early_win,
                "activity_sparsity_proxy_win": sparsity_win,
                "utility_win": bool(final_win or early_win or sparsity_win),
            }
        )
    selected = max(rows, key=lambda row: (row["final_gain_pp"], -row["registered_order"]))
    accuracy_passed = selected["final_gain_pp"] >= -float(maximum_accuracy_gap_pp)
    passed = bool(accuracy_passed and selected["utility_win"])
    return {
        "passed": passed,
        "selected_variant": selected["variant"],
        "selection_rule": "highest final subject-macro accuracy, then registered order",
        "maximum_accuracy_gap_pp": float(maximum_accuracy_gap_pp),
        "minimum_early_gain_pp": float(minimum_early_gain_pp),
        "minimum_activity_density_reduction": float(
            minimum_activity_density_reduction
        ),
        "minimum_nondegenerate_activity_rate": float(
            minimum_nondegenerate_activity_rate
        ),
        "maximum_sparse_activity_rate": float(maximum_sparse_activity_rate),
        "accuracy_passed": accuracy_passed,
        "utility_passed": bool(selected["utility_win"]),
        "selected_candidate": selected,
        "candidates": rows,
        "activity_metric_scope": "software activation sparsity proxy, not hardware energy",
    }
