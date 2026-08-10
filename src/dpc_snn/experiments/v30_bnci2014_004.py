"""Validation and promotion gate for the blind BNCI2014-004 confirmation."""

from __future__ import annotations

from typing import Any

import numpy as np


def v30_blind_gate(
    subject_gain_pp: np.ndarray,
    *,
    one_sided_wilcoxon_p: float,
) -> dict[str, Any]:
    values = np.asarray(subject_gain_pp, dtype=float)
    if values.shape != (9,) or not np.isfinite(values).all():
        raise ValueError("V30 requires one finite paired gain for each of 9 subjects")
    mean_gain = float(values.mean())
    positive = int(np.count_nonzero(values > 0.0))
    checks = {
        "mean_subject_gain_at_least_0_5pp": mean_gain >= 0.5,
        "at_least_6_subjects_positive": positive >= 6,
        "one_sided_wilcoxon_below_0_05": float(one_sided_wilcoxon_p) < 0.05,
        "snn_not_inferior_on_mean": mean_gain >= 0.0,
    }
    passed = all(checks.values())
    return {
        "status": "pass" if passed else "fail",
        "mean_subject_gain_pp": mean_gain,
        "median_subject_gain_pp": float(np.median(values)),
        "positive_subjects": positive,
        "negative_subjects": int(np.count_nonzero(values < 0.0)),
        "ties": int(np.count_nonzero(values == 0.0)),
        "one_sided_wilcoxon_p": float(one_sided_wilcoxon_p),
        "checks": checks,
        "authorized_next_stage": "E31_publication_baselines" if passed else None,
    }
