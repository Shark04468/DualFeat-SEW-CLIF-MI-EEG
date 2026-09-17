from __future__ import annotations

import pytest

from dpc_snn.experiments.v25_confirmation import (
    select_aggregate_residual_scale,
    upper_median_epoch,
    validate_v25_freeze,
)


def test_upper_median_epoch_uses_conservative_integer() -> None:
    assert upper_median_epoch([40, 50, 60, 61, 70, 80]) == 61
    assert upper_median_epoch([20, 20, 20]) == 20


def test_aggregate_scale_uses_fold_means_and_smaller_tie() -> None:
    rows = []
    for fold in range(6):
        for scale in (0.0, 0.25, 0.5, 1.0):
            rows.append(
                {
                    "fold": fold,
                    "scale": scale,
                    "validation_kappa": 0.7 if scale in (0.0, 0.25) else 0.6,
                    "validation_accuracy": 0.8,
                }
            )
    selected, summary = select_aggregate_residual_scale(rows)
    assert selected == 0.0
    assert len(summary) == 4


def test_aggregate_scale_rejects_incomplete_fold_coverage() -> None:
    with pytest.raises(ValueError, match="fold coverage"):
        select_aggregate_residual_scale(
            [
                {"scale": scale, "validation_kappa": 0.5, "validation_accuracy": 0.5}
                for scale in (0.0, 0.25, 0.5, 1.0)
            ]
        )


def test_freeze_requires_historical_exposure_disclosure() -> None:
    entries = {}
    for seed in range(5):
        for subject in range(1, 10):
            entries[f"subject_{subject:02d}_seed_{seed}"] = {
                "fixed_epochs": {
                    "atcnet": 40,
                    "fbcnet": 40,
                    "ann_plain_ce": 20,
                    "sew_clif_ce": 20,
                },
                "residual_scales": {"ann_plain_ce": 0.0, "sew_clif_ce": 0.25},
            }
    payload = {
        "schema_version": 1,
        "architecture_id": "v25_equal_probability_dual_feature_sew_clif_residual",
        "subjects": list(range(1, 10)),
        "seeds": list(range(5)),
        "variants": ["ann_plain_ce", "sew_clif_ce"],
        "historical_data_exposure": {"bci2a_session_e": True},
        "entries": entries,
    }
    assert validate_v25_freeze(payload)["entries"] == entries
    payload["historical_data_exposure"]["bci2a_session_e"] = False
    with pytest.raises(ValueError, match="disclose"):
        validate_v25_freeze(payload)
