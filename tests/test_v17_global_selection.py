from __future__ import annotations

from dpc_snn.experiments.v17_global_selection import select_global_configuration


def test_global_selection_uses_mean_inner_metrics_and_median_epoch() -> None:
    rows = []
    for fold in range(3):
        rows.extend(
            [
                {
                    "config_index": 0,
                    "hidden_channels": 64,
                    "dropout": 0.1,
                    "learning_rate": 0.001,
                    "validation_kappa": 0.6,
                    "validation_accuracy": 0.7,
                    "best_epoch": [3, 7, 11][fold],
                },
                {
                    "config_index": 1,
                    "hidden_channels": 96,
                    "dropout": 0.25,
                    "learning_rate": 0.0003,
                    "validation_kappa": 0.7,
                    "validation_accuracy": 0.75,
                    "best_epoch": [5, 9, 13][fold],
                },
            ]
        )
    selected = select_global_configuration(rows)
    assert selected.config_index == 1
    assert selected.fixed_epoch == 9
    assert selected.folds == 3
