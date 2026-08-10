"""Exact campaign-coverage helpers for V8 staged experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class V8CampaignAuditError(RuntimeError):
    """Raised when a completed V8 campaign is incomplete or internally inconsistent."""


def validate_e1_campaign_contract(
    status: Mapping[str, Any],
    selection: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """Bind completion metadata to the preregistered E1 campaign configuration."""

    models = [str(value) for value in config["models"]]
    subjects = [int(value) for value in config["subjects"]]
    screening_seed = int(config["screening_seed"])
    confirmation_seeds = [int(value) for value in config["confirmation_seeds"]]
    top_k = int(config["confirmation_top_k"])
    selected = [str(value) for value in selection["models"]]
    if [str(value) for value in status["screened_models"]] != models:
        raise V8CampaignAuditError("completed E1 model set differs from preregistration")
    if [int(value) for value in status["subjects"]] != subjects:
        raise V8CampaignAuditError("completed E1 subject set differs from preregistration")
    if int(status["screening_seed"]) != screening_seed:
        raise V8CampaignAuditError("completed E1 screening seed differs from preregistration")
    if [int(value) for value in status["confirmation_seeds"]] != confirmation_seeds:
        raise V8CampaignAuditError("completed E1 confirmation seeds differ from preregistration")
    if len(selected) != top_k or int(selection["top_k"]) != top_k:
        raise V8CampaignAuditError("completed E1 confirmation count differs from preregistration")
    expected_runs = len(models) * len(subjects) + len(selected) * len(subjects) * (
        len(confirmation_seeds) - 1
    )
    if int(status["runs"]) != expected_runs:
        raise V8CampaignAuditError(
            f"completed E1 run count differs from preregistration: {status['runs']} != {expected_runs}"
        )


def expected_e1_run_keys(
    status: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> set[tuple[str, int, int]]:
    """Return the exact screening plus confirmation model/subject/seed coverage."""

    models = [str(value) for value in status["screened_models"]]
    subjects = [int(value) for value in status["subjects"]]
    screening_seed = int(status["screening_seed"])
    selected = [str(value) for value in selection["models"]]
    confirmation_seeds = [int(value) for value in status["confirmation_seeds"]]
    if selected != [str(value) for value in status["confirmed_models"]]:
        raise V8CampaignAuditError("campaign status and confirmation selection disagree")
    if screening_seed not in confirmation_seeds:
        raise V8CampaignAuditError("confirmation seeds omit the screening seed")
    if not set(selected).issubset(models):
        raise V8CampaignAuditError("confirmation selected a model outside the screening set")
    keys = {
        (model, subject, screening_seed)
        for model in models
        for subject in subjects
    }
    keys.update(
        (model, subject, seed)
        for model in selected
        for subject in subjects
        for seed in confirmation_seeds
    )
    return keys


def index_e1_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    expected: set[tuple[str, int, int]],
) -> dict[tuple[str, int, int], Mapping[str, Any]]:
    """Index summary rows while rejecting duplicates, omissions and extras."""

    indexed: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["model"]), int(row["subject"]), int(row["seed"]))
        if key in indexed:
            raise V8CampaignAuditError(f"duplicate E1 summary row: {key}")
        indexed[key] = row
    actual = set(indexed)
    if actual != expected:
        raise V8CampaignAuditError(
            "E1 summary coverage mismatch; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return indexed


def validate_nested_trial_sets(fold: Mapping[str, Any]) -> None:
    """Require an exact nested split with no outer-test leakage."""

    outer_train = set(map(str, fold["outer_train_trial_ids"]))
    outer_test = set(map(str, fold["outer_test_trial_ids"]))
    inner_train = set(map(str, fold["inner_train_trial_ids"]))
    inner_validation = set(map(str, fold["inner_validation_trial_ids"]))
    if not outer_train or not outer_test or not inner_train or not inner_validation:
        raise V8CampaignAuditError("nested fold contains an empty partition")
    if outer_train & outer_test:
        raise V8CampaignAuditError("outer training and test trials overlap")
    if inner_train & inner_validation:
        raise V8CampaignAuditError("inner training and validation trials overlap")
    if inner_train | inner_validation != outer_train:
        raise V8CampaignAuditError("inner partitions do not exactly cover outer training")
    if str(fold["outer_test_run"]) == str(fold["inner_validation_run"]):
        raise V8CampaignAuditError("inner validation reused the outer test run")
