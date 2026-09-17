"""Pre-registered gate for the full matched ANN/SNN control campaign."""

from __future__ import annotations

from typing import Any, Mapping


def v28_control_gate(
    comparison: Mapping[str, Any],
    *,
    worst_subject_mean_delta_pp: float,
) -> dict[str, Any]:
    """Apply the frozen E28 expansion criteria without inspecting later stages."""
    pairs = int(comparison["pairs"])
    required_positive = (3 * pairs + 3) // 4
    checks = {
        "mean_gain_at_least_0_5pp": float(comparison["mean_delta_pp"]) >= 0.5,
        "at_least_75pct_subject_seed_pairs_positive": int(
            comparison["positive_pairs"]
        )
        >= required_positive,
        "one_sided_wilcoxon_below_0_05": float(
            comparison["wilcoxon_p_greater"]
        )
        < 0.05,
        "no_subject_mean_below_minus_1pp": float(worst_subject_mean_delta_pp)
        >= -1.0,
    }
    passed = all(checks.values())
    return {
        "status": "pass" if passed else "fail",
        "comparison": "sew_clif_ce_minus_ann_sew_ce",
        "pairs": pairs,
        "required_positive_pairs": required_positive,
        "worst_subject_mean_delta_pp": float(worst_subject_mean_delta_pp),
        "checks": checks,
        "authorized_next_stage": "E29_openbmi_replication" if passed else None,
    }
