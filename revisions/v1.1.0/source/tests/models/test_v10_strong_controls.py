from __future__ import annotations

import torch

from dpc_snn.models.v10_strong_controls import (
    V10_STRONG_CONTROL_MODELS,
    V10DualFeatureFusion,
    V10StrongANNControl,
    build_v10_strong_control,
)
from dpc_snn.models.v9_dual_feature_student import V9DualFeatureStudent


def test_v10_fusion_is_exactly_the_v9_front_end() -> None:
    torch.manual_seed(13)
    v9 = V9DualFeatureStudent(dropout=0.0).eval()
    fusion = V10DualFeatureFusion(dropout=0.0).eval()
    state = {
        name: value
        for name, value in v9.state_dict().items()
        if not name.startswith("decoder.")
    }
    fusion.load_state_dict(state, strict=True)
    atc = torch.randn(4, 18, 32)
    fbc = torch.randn(4, 4, 288)
    torch.testing.assert_close(fusion(atc, fbc), v9.fuse(atc, fbc), rtol=0.0, atol=0.0)


def test_v10_strong_controls_stay_within_five_percent_capacity() -> None:
    counts = {
        name: sum(parameter.numel() for parameter in build_v10_strong_control(name).parameters())
        for name in V10_STRONG_CONTROL_MODELS
    }
    assert max(counts.values()) / min(counts.values()) <= 1.05
    assert counts["ann_leaky_sew"] == counts["sew_clif"]


def test_v10_ann_controls_have_finite_outputs_and_gradients() -> None:
    for kind in ("tcn", "gru", "lstm"):
        model = V10StrongANNControl(kind, dropout=0.0)
        atc = torch.randn(3, 18, 32)
        fbc = torch.randn(3, 4, 288)
        output = model(atc, fbc)
        assert output.logits.shape == (3, 4)
        assert output.fused_sequence.shape == (3, 18, 64)
        assert output.final_activity.shape[-1] == 18
        assert output.binary_spikes == ()
        assert output.firing_rate_loss.item() == 0.0
        assert torch.isfinite(output.logits).all()
        output.logits.square().mean().backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )


def test_v10_ann_decoders_are_causal() -> None:
    torch.manual_seed(5)
    prefix = 9
    sequence = torch.randn(2, 18, 64)
    changed = sequence.clone()
    changed[:, prefix:] += 100.0
    for kind in ("tcn", "gru", "lstm"):
        model = V10StrongANNControl(kind, dropout=0.0).eval()
        _, first = model.decoder(sequence)
        _, second = model.decoder(changed)
        torch.testing.assert_close(first[..., :prefix], second[..., :prefix])
