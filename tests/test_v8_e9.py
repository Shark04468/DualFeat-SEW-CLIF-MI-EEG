from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from scripts.audit_v8_e9_ablations import _exact_sign_flip_p, _holm
from scripts.run_v8_e9_frozen_ablations import _ablation_contract


ROOT = Path(__file__).resolve().parents[1]


def _freeze() -> dict:
    return {
        "architecture": {
            "model_config": {
                "use_statistical_branch": True,
                "use_covariance_branch": True,
                "use_temporal_branch": True,
                "delay_auxiliary_enabled": False,
            }
        },
        "augmentation": {"enabled": True, "probability": 0.5},
        "checkpoint_rule": {"final_epoch": 41},
    }


def _config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / "experiments" / "v8_e9_frozen_ablation.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_e9_ablation_registry_is_fixed_and_each_entry_changes_one_factor() -> None:
    config = _config()
    assert list(config["retrained_single_variable_ablations"]) == [
        "no_covariance",
        "no_statistics",
        "no_temporal",
        "no_augmentation",
    ]
    frozen = _freeze()
    for variant in config["retrained_single_variable_ablations"]:
        model, augmentation, epoch = _ablation_contract(frozen, config, variant)
        assert epoch == 41
        assert model["delay_auxiliary_enabled"] is False
        changed_model = {
            name
            for name in (
                "use_statistical_branch",
                "use_covariance_branch",
                "use_temporal_branch",
            )
            if model[name] is not frozen["architecture"]["model_config"][name]
        }
        changed_augmentation = augmentation != frozen["augmentation"]
        assert len(changed_model) + int(changed_augmentation) == 1


def test_exact_sign_flip_and_holm_are_deterministic_and_conservative() -> None:
    assert _exact_sign_flip_p(np.zeros(9)) == 1.0
    strong = _exact_sign_flip_p(np.ones(9))
    assert strong == 2.0 / 512.0
    raw = {"a": 0.01, "b": 0.04, "c": 0.20, "d": 0.80}
    adjusted = _holm(raw)
    assert set(adjusted) == set(raw)
    assert all(adjusted[name] >= raw[name] for name in raw)
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"] <= adjusted["d"]
