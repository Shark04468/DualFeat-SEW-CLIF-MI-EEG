from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dpc_snn.baselines.neural import build_v62_neural_baseline
from dpc_snn.models.v8_atc_backbone import V8ATCAccuracyBackbone


def test_v8_atc_wrapper_is_logit_exact_and_exposes_sequence() -> None:
    source_root = Path(__file__).resolve().parents[2] / ".baseline_extract"
    if not (source_root / "TCFormer" / "SOURCE_LOCK.txt").is_file():
        pytest.skip("pinned TCFormer source is not available")
    adapter = build_v62_neural_baseline(
        "atcnet",
        source_root=source_root,
        n_channels=22,
        n_classes=4,
        samples=1000,
    )
    core = adapter.module
    wrapper = V8ATCAccuracyBackbone(core)
    wrapper.eval()
    carrier = torch.randn(2, 22, 1000)
    with torch.no_grad():
        direct = core(carrier)
        output = wrapper(carrier)
    torch.testing.assert_close(output["logits"], direct, rtol=0.0, atol=0.0)
    assert output["aux"]["continuous_sequence"].shape[:2] == (2, 18)
    assert wrapper.parameter_count == sum(p.numel() for p in core.parameters())
