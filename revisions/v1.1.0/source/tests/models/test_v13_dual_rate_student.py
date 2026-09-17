from __future__ import annotations

import pytest
import torch

from dpc_snn.models.v13_dual_rate_student import (
    V13_MODEL_VARIANTS,
    build_v13_student,
)


EXPECTED_PARAMETERS = {
    "dual_snn_logit_mean": 74_312,
    "dual_snn_feature_late": 74_668,
}


@pytest.mark.parametrize("variant", list(V13_MODEL_VARIANTS))
def test_v13_variants_are_parameter_matched_and_train_both_branches(
    variant: str,
) -> None:
    torch.manual_seed(13)
    model = build_v13_student(variant)
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        EXPECTED_PARAMETERS[variant]
    )
    assert abs(model.parameter_count - 74_124) / 74_124 < 0.01
    output = model(torch.randn(2, 18, 32), torch.randn(2, 4, 288))
    assert output.logits.shape == (2, 4)
    assert output.endpoint_logits.shape == (2, 4, 4)
    assert output.atc_logits.shape == (2, 4)
    assert output.fbc_logits.shape == (2, 4)
    assert len(output.binary_spikes) == 8
    assert torch.isfinite(output.logits).all()
    output.logits.square().mean().backward()
    for decoder in (model.atc_decoder, model.fbc_decoder):
        gradient = decoder.current_encoder.weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert float(gradient.abs().sum()) > 0.0


def test_v13_logit_mean_is_exact() -> None:
    torch.manual_seed(14)
    model = build_v13_student("dual_snn_logit_mean").eval()
    with torch.no_grad():
        output = model(torch.randn(2, 18, 32), torch.randn(2, 4, 288))
    torch.testing.assert_close(
        output.logits,
        0.5 * (output.atc_logits + output.fbc_logits),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("variant", list(V13_MODEL_VARIANTS))
def test_v13_first_endpoint_is_causal(variant: str) -> None:
    torch.manual_seed(15)
    model = build_v13_student(variant).eval()
    atc = torch.randn(2, 18, 32)
    fbc = torch.randn(2, 4, 288)
    changed_atc = atc.clone()
    changed_fbc = fbc.clone()
    changed_atc[:, 4:] += 20.0 * torch.randn_like(changed_atc[:, 4:])
    changed_fbc[:, 1:] += 20.0 * torch.randn_like(changed_fbc[:, 1:])
    with torch.no_grad():
        first = model(atc, fbc).endpoint_logits[:, 0]
        changed = model(changed_atc, changed_fbc).endpoint_logits[:, 0]
    torch.testing.assert_close(first, changed, atol=1e-6, rtol=1e-6)


def test_v13_unknown_variant_is_rejected() -> None:
    with pytest.raises(KeyError):
        build_v13_student("unknown")
