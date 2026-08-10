from __future__ import annotations

import numpy as np

from scripts.aggregate_v32_fusion import _holm, _one_sided_wilcoxon


def test_one_sided_wilcoxon_treats_all_ties_as_no_evidence() -> None:
    assert _one_sided_wilcoxon(np.zeros(9, dtype=float)) == 1.0


def test_holm_adjustment_is_monotone_in_sorted_pvalues() -> None:
    adjusted = _holm({"atc_only": 0.01, "fbc_only": 0.04, "simple": 0.03})

    assert adjusted["atc_only"] == 0.03
    assert adjusted["simple"] == 0.06
    assert adjusted["fbc_only"] == 0.06
