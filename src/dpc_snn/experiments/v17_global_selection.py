"""Inner-only global configuration selection for E17-1."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class GlobalSelection:
    config_index: int
    config: dict[str, float | int]
    fixed_epoch: int
    mean_validation_kappa: float
    mean_validation_accuracy: float
    folds: int


def select_global_configuration(rows: Sequence[dict[str, Any]]) -> GlobalSelection:
    if not rows:
        raise ValueError("global selection requires HPO rows")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["config_index"])].append(row)
    fold_counts = {len(group) for group in grouped.values()}
    if len(fold_counts) != 1:
        raise ValueError("every HPO configuration must occur in every development fold")
    signatures: dict[int, dict[str, float | int]] = {}
    best_key: tuple[float, float, int] | None = None
    selected_index = min(grouped)
    for config_index, group in sorted(grouped.items()):
        signature = {
            "hidden_channels": int(group[0]["hidden_channels"]),
            "dropout": float(group[0]["dropout"]),
            "learning_rate": float(group[0]["learning_rate"]),
        }
        if any(
            int(row["hidden_channels"]) != signature["hidden_channels"]
            or float(row["dropout"]) != signature["dropout"]
            or float(row["learning_rate"]) != signature["learning_rate"]
            for row in group
        ):
            raise ValueError(f"configuration {config_index} changes across folds")
        signatures[config_index] = signature
        mean_kappa = float(np.mean([float(row["validation_kappa"]) for row in group]))
        mean_accuracy = float(
            np.mean([float(row["validation_accuracy"]) for row in group])
        )
        key = (mean_kappa, mean_accuracy, -config_index)
        if best_key is None or key > best_key:
            best_key = key
            selected_index = config_index
    selected_rows = grouped[selected_index]
    fixed_epoch = max(
        1, int(np.floor(np.median([int(row["best_epoch"]) for row in selected_rows])))
    )
    return GlobalSelection(
        config_index=selected_index,
        config=signatures[selected_index],
        fixed_epoch=fixed_epoch,
        mean_validation_kappa=float(
            np.mean([float(row["validation_kappa"]) for row in selected_rows])
        ),
        mean_validation_accuracy=float(
            np.mean([float(row["validation_accuracy"]) for row in selected_rows])
        ),
        folds=len(selected_rows),
    )
