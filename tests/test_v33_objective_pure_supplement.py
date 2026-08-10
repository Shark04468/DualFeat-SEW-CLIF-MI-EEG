from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from scripts.aggregate_v33_e31zp import _common_slope
from scripts.aggregate_v33_external_zp import _metric_summary


ROOT = Path(__file__).resolve().parents[1]


def _config(name: str) -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / "experiments" / name).read_text(encoding="utf-8")
    )


def test_e31zp_registered_checkpoint_count_is_513() -> None:
    config = _config("v33_e31_zero_penalty.yaml")
    count = sum(
        len(dataset["subjects"])
        * len(config["seeds"])
        * len(config["budgets"][name])
        for name, dataset in config["datasets"].items()
    )

    assert count == 513


def test_all_objective_pure_configs_disable_firing_rate_penalty() -> None:
    e31 = _config("v33_e31_zero_penalty.yaml")
    e29 = _config("v33_e29_zero_penalty.yaml")
    e30 = _config("v33_e30_zero_penalty.yaml")

    assert e31["training"]["firing_rate_weight"] == 0.0
    assert e29["training"]["firing_rate_weight"] == 0.0
    assert e30["training"]["firing_rate_weight"] == 0.0


def test_common_slope_uses_log2_budget_with_dataset_intercepts() -> None:
    rows = []
    for dataset, intercept in (("a", 0.1), ("b", -0.2)):
        for subject in (1, 2):
            for examples in (25.0, 50.0, 100.0):
                rows.append(
                    {
                        "dataset": dataset,
                        "subject": subject,
                        "examples_per_class": examples,
                        "gain_accuracy": intercept - 0.02 * np.log2(examples),
                    }
                )

    assert np.isclose(_common_slope(rows, "gain_accuracy"), -0.02)


def test_external_metric_summary_handles_all_ties() -> None:
    result = _metric_summary(np.zeros(9), bootstrap_samples=100, seed=1)

    assert result["mean_delta_pp"] == 0.0
    assert result["one_sided_wilcoxon_p"] == 1.0
    assert result["ties"] == 9
