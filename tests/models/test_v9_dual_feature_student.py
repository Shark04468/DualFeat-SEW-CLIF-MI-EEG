from __future__ import annotations

import torch
from torch import nn

from dpc_snn.models.v9_dual_feature_student import (
    V9_DUAL_FEATURE_MODEL_VARIANTS,
    V9DualFeatureStudent,
    V9FBCFeatureBackbone,
    build_v9_dual_feature_student,
)


class _FakeFBC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.strideFactor = 4
        self.nBands = 2
        self.m = 2
        self.scb = nn.Sequential(nn.Conv2d(2, 4, (3, 1), groups=2), nn.ELU())
        self.temporalLayer = lambda value: value.var(dim=3, keepdim=True, unbiased=False).log()
        self.lastLayer = nn.Linear(16, 3)

    def forward(self, carrier: torch.Tensor) -> torch.Tensor:
        value = torch.squeeze(carrier.permute(0, 4, 2, 3, 1), dim=4)
        value = self.scb(value)
        value = value.reshape(value.shape[0], value.shape[1], 4, value.shape[-1] // 4)
        value = self.temporalLayer(value)
        return self.lastLayer(torch.flatten(value, start_dim=1))


def test_fbc_feature_wrapper_reconstructs_exact_logits() -> None:
    torch.manual_seed(7)
    core = _FakeFBC().eval()
    wrapper = V9FBCFeatureBackbone(core).eval()
    carrier = torch.rand(5, 1, 3, 20, 2) + 0.1
    expected = core(carrier)
    output = wrapper(carrier)
    assert output["continuous_sequence"].shape == (5, 4, 4)
    torch.testing.assert_close(output["logits"], expected, rtol=0.0, atol=0.0)


def test_dual_feature_variants_are_exactly_capacity_matched() -> None:
    rows = [build_v9_dual_feature_student(name) for name in V9_DUAL_FEATURE_MODEL_VARIANTS]
    assert len({model.parameter_count for model in rows}) == 1
    shapes = {
        tuple(sorted(tuple(parameter.shape) for parameter in model.parameters()))
        for model in rows
    }
    assert len(shapes) == 1


def test_dual_feature_student_shapes_binary_spikes_and_gradients() -> None:
    for kind in ("ann", "plif", "clif"):
        model = V9DualFeatureStudent(
            decoder_kind=kind,
            residual_mode="sew_add",
            hidden_channels=16,
            readout_features=12,
            dropout=0.0,
        )
        atc = torch.randn(3, 18, 32)
        fbc = torch.randn(3, 4, 288)
        output = model(atc, fbc)
        assert output.logits.shape == (3, 4)
        assert output.fused_sequence.shape == (3, 18, 16)
        assert output.final_activity.shape == (3, 16, 18)
        assert torch.isfinite(output.logits).all()
        output.logits.square().mean().backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        if kind == "ann":
            assert output.binary_spikes == ()
            assert output.firing_rate_loss.item() == 0.0
        else:
            assert len(output.binary_spikes) == 4
            for spikes in output.binary_spikes:
                assert bool(((spikes == 0.0) | (spikes == 1.0)).all())


def test_dual_feature_student_rejects_misaligned_features() -> None:
    model = build_v9_dual_feature_student("ann_plain")
    try:
        model(torch.randn(2, 18, 32), torch.randn(3, 4, 288))
    except ValueError as exc:
        assert "not aligned" in str(exc)
    else:
        raise AssertionError("misaligned feature batches were accepted")


def test_fusion_ablation_modes_keep_capacity_and_isolate_branches() -> None:
    torch.manual_seed(19)
    models = {
        mode: V9DualFeatureStudent(
            hidden_channels=16,
            readout_features=12,
            dropout=0.0,
            fusion_mode=mode,
        ).eval()
        for mode in ("interaction", "simple", "atc_only", "fbc_only")
    }
    assert len({model.parameter_count for model in models.values()}) == 1

    atc = torch.randn(2, 18, 32)
    fbc = torch.randn(2, 4, 288)
    changed_atc = atc + torch.randn_like(atc)
    changed_fbc = fbc + torch.randn_like(fbc)
    torch.testing.assert_close(
        models["atc_only"].fuse(atc, fbc),
        models["atc_only"].fuse(atc, changed_fbc),
    )
    torch.testing.assert_close(
        models["fbc_only"].fuse(atc, fbc),
        models["fbc_only"].fuse(changed_atc, fbc),
    )
    assert not torch.allclose(
        models["simple"].fuse(atc, fbc),
        models["simple"].fuse(changed_atc, changed_fbc),
    )


def test_dual_feature_student_rejects_unknown_fusion_mode() -> None:
    try:
        V9DualFeatureStudent(fusion_mode="unknown")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "fusion mode" in str(exc)
    else:
        raise AssertionError("unknown fusion mode was accepted")
