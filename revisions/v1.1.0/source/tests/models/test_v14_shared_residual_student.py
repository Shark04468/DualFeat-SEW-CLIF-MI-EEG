from __future__ import annotations

import pytest
import torch

from dpc_snn.models.v14_shared_residual_student import (
    V14_VARIANTS,
    build_v14_student,
)


EXPECTED_PARAMETERS = {
    "r0_shared_replay": 73_348,
    "r1_atc_residual": 78_332,
    "r2_fbc_residual": 81_404,
    "r3_dual_residual": 86_388,
    "r4_generic_residual": 86_384,
}


@pytest.mark.parametrize("variant", V14_VARIANTS)
def test_v14_parameters_and_frozen_shared_backbone(variant: str) -> None:
    model = build_v14_student(variant)
    assert model.parameter_count == EXPECTED_PARAMETERS[variant]
    assert not any(parameter.requires_grad for parameter in model.shared.parameters())
    model.train()
    assert not model.shared.training
    assert model.trainable_parameter_count == model.parameter_count - 73_348


def test_v14_parameter_matched_control_differs_by_four_parameters() -> None:
    dual = build_v14_student("r3_dual_residual")
    generic = build_v14_student("r4_generic_residual")
    assert abs(dual.parameter_count - generic.parameter_count) == 4


def test_v14_r0_is_exact_shared_replay() -> None:
    torch.manual_seed(14)
    model = build_v14_student("r0_shared_replay").eval()
    atc = torch.randn(2, 18, 32)
    fbc = torch.randn(2, 4, 288)
    with torch.no_grad():
        expected = model.shared(atc, fbc).logits
        output = model(atc, fbc)
    torch.testing.assert_close(output.logits, expected, atol=0.0, rtol=0.0)
    assert output.binary_spikes == ()


@pytest.mark.parametrize(
    ("variant", "expert_name", "gate_name"),
    [
        ("r1_atc_residual", "atc_expert", "atc_gate_logit"),
        ("r2_fbc_residual", "fbc_expert", "fbc_gate_logit"),
        ("r4_generic_residual", "generic_expert", "generic_gate_logit"),
    ],
)
def test_v14_residual_expert_and_gate_receive_gradients(
    variant: str, expert_name: str, gate_name: str
) -> None:
    torch.manual_seed(15)
    model = build_v14_student(variant)
    output = model(torch.randn(2, 18, 32), torch.randn(2, 4, 288))
    output.logits.square().mean().backward()
    expert = getattr(model, expert_name)
    gate = getattr(model, gate_name)
    assert expert.current_encoder.weight.grad is not None
    assert float(expert.current_encoder.weight.grad.abs().sum()) > 0.0
    assert gate.grad is not None
    assert float(gate.grad.abs().sum()) > 0.0
    active_gate = model.gate_values()[variant.split("_")[1]]
    torch.testing.assert_close(active_gate, torch.full((4,), 0.05))


def test_v14_unknown_variant_is_rejected() -> None:
    with pytest.raises(KeyError):
        build_v14_student("unknown")
