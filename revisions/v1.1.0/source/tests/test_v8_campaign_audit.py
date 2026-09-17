from __future__ import annotations

import pytest

from dpc_snn.experiments.v8_campaign_audit import (
    V8CampaignAuditError,
    expected_e1_run_keys,
    index_e1_summary_rows,
    validate_e1_campaign_contract,
    validate_nested_trial_sets,
)


def test_e1_expected_coverage_is_exact() -> None:
    status = {
        "screened_models": ["a", "b"],
        "subjects": [1, 3],
        "screening_seed": 0,
        "confirmed_models": ["b"],
        "confirmation_seeds": [0, 1, 2],
        "runs": 8,
    }
    selection = {"models": ["b"], "top_k": 1}
    config = {
        "models": ["a", "b"],
        "subjects": [1, 3],
        "screening_seed": 0,
        "confirmation_seeds": [0, 1, 2],
        "confirmation_top_k": 1,
    }
    validate_e1_campaign_contract(status, selection, config)
    expected = expected_e1_run_keys(status, selection)
    assert len(expected) == 8
    rows = [
        {"model": model, "subject": subject, "seed": seed}
        for model, subject, seed in sorted(expected)
    ]
    assert set(index_e1_summary_rows(rows, expected)) == expected
    with pytest.raises(V8CampaignAuditError, match="coverage mismatch"):
        index_e1_summary_rows(rows[:-1], expected)
    partial = {**status, "screened_models": ["b"], "runs": 2}
    with pytest.raises(V8CampaignAuditError, match="model set differs"):
        validate_e1_campaign_contract(partial, selection, config)


def test_nested_trial_sets_reject_outer_leakage() -> None:
    fold = {
        "outer_train_trial_ids": ["a", "b", "c"],
        "outer_test_trial_ids": ["d"],
        "inner_train_trial_ids": ["a", "b"],
        "inner_validation_trial_ids": ["c"],
        "outer_test_run": "6",
        "inner_validation_run": "5",
    }
    validate_nested_trial_sets(fold)
    fold["outer_test_trial_ids"] = ["c"]
    with pytest.raises(V8CampaignAuditError, match="overlap"):
        validate_nested_trial_sets(fold)
