from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.evaluate_v8_e4_gate import _comparison_config
from scripts.run_v8_e4_decoder_controls import (
    EXPECTED_VARIANTS,
    _build_e4_variant,
    _capacity_audit,
)


ROOT = Path(__file__).resolve().parents[1]


def _model_config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/models/v8_accuracy_first.yaml").read_text(encoding="utf-8")
    )


def test_e4_registered_variants_are_exactly_capacity_matched() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/experiments/v8_e4_decoder_controls.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["variants"] == EXPECTED_VARIANTS
    audit = _capacity_audit(_model_config(), EXPECTED_VARIANTS)
    assert audit["status"] == "passed"
    assert len({row["parameters"] for row in audit["variants"].values()}) == 1
    assert len(
        {
            str(row["trainable_parameter_shape_multiset"])
            for row in audit["variants"].values()
        }
    ) == 1


def test_e4_builder_rejects_decoder_contract_drift() -> None:
    wrong = {**EXPECTED_VARIANTS["ann_residual"], "decoder_residual_mode": "plain"}
    with pytest.raises(RuntimeError, match="contract mismatch"):
        _build_e4_variant(_model_config(), "ann_residual", wrong, seed=0)


def test_e4_comparison_config_removes_only_registered_decoder_differences(
    tmp_path: Path,
) -> None:
    base = {
        "active_variant": "ann_residual",
        "active_subject": 1,
        "active_seed": 0,
        "training": {"learning_rate": 7e-4},
        "resolved_model": {
            "decoder_kind": "ann",
            "decoder_residual_mode": "plain",
            "decoder_channels": 64,
        },
    }
    ann = tmp_path / "ann.yaml"
    snn = tmp_path / "snn.yaml"
    ann.write_text(yaml.safe_dump(base), encoding="utf-8")
    snn.write_text(
        yaml.safe_dump(
            {
                **base,
                "active_variant": "clif_plain",
                "resolved_model": {
                    **base["resolved_model"],
                    "decoder_kind": "clif",
                },
            }
        ),
        encoding="utf-8",
    )
    assert _comparison_config(ann) == _comparison_config(snn)

    changed = yaml.safe_load(snn.read_text(encoding="utf-8"))
    changed["training"]["learning_rate"] = 1e-3
    snn.write_text(yaml.safe_dump(changed), encoding="utf-8")
    assert _comparison_config(ann) != _comparison_config(snn)
