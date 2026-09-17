from __future__ import annotations

import pytest
import torch

from dpc_snn.models.v12_multirate_student import build_v12_student


@pytest.mark.parametrize(
    "variant",
    [
        "interpolated_branch_kd",
        "interpolated_static_gate_kd",
        "dual_rate_branch_kd",
        "dual_rate_branch_temporal",
    ],
)
def test_v12_variants_have_finite_outputs_and_gradients(variant: str) -> None:
    torch.manual_seed(4)
    model = build_v12_student(variant)
    atc = torch.randn(2, 18, 32)
    fbc = torch.randn(2, 4, 288)
    output = model(atc, fbc)
    assert output.logits.shape == (2, 4)
    assert output.atc_logits.shape == (2, 4)
    assert output.fbc_logits.shape == (2, 4)
    assert torch.isfinite(output.logits).all()
    (output.logits.square().mean() + output.firing_rate_loss).backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(value).all() for value in gradients)


def test_v12_static_gate_is_initialized_balanced_and_receives_gradient() -> None:
    torch.manual_seed(7)
    model = build_v12_student("interpolated_static_gate_kd")
    gate = model.backbone.branch_gate_logit
    torch.testing.assert_close(torch.sigmoid(gate), torch.full_like(gate, 0.5))
    output = model(torch.randn(2, 18, 32), torch.randn(2, 4, 288))
    (output.logits.square().mean() + output.firing_rate_loss).backward()
    assert gate.grad is not None
    assert torch.isfinite(gate.grad).all()
    assert float(gate.grad.abs().sum()) > 0.0


def test_v12_dual_rate_endpoint_is_causal() -> None:
    torch.manual_seed(5)
    model = build_v12_student("dual_rate_branch_temporal").eval()
    atc = torch.randn(2, 18, 32)
    fbc = torch.randn(2, 4, 288)
    changed_atc = atc.clone()
    changed_fbc = fbc.clone()
    changed_atc[:, 4:] += 20.0 * torch.randn_like(changed_atc[:, 4:])
    changed_fbc[:, 1:] += 20.0 * torch.randn_like(changed_fbc[:, 1:])
    with torch.no_grad():
        first = model(atc, fbc).endpoint_logits[:, 0]
        second = model(changed_atc, changed_fbc).endpoint_logits[:, 0]
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)


def test_v12_unknown_variant_is_rejected() -> None:
    with pytest.raises(KeyError):
        build_v12_student("unknown")
