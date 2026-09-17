from __future__ import annotations

import pytest
import torch

from dpc_snn.models.v8_sequence_decoder import (
    V8_SEQUENCE_DECODER_VARIANTS,
    build_v8_sequence_decoder,
)


def test_v8_sequence_decoder_variants_are_exactly_parameter_matched() -> None:
    models = {name: build_v8_sequence_decoder(name) for name in V8_SEQUENCE_DECODER_VARIANTS}
    counts = {model.parameter_count for model in models.values()}
    trainable = {model.trainable_parameter_count for model in models.values()}
    shapes = {
        tuple(sorted(tuple(parameter.shape) for parameter in model.parameters()))
        for model in models.values()
    }
    assert len(counts) == 1
    assert len(trainable) == 1
    assert len(shapes) == 1


@pytest.mark.parametrize("variant", tuple(V8_SEQUENCE_DECODER_VARIANTS))
def test_v8_sequence_decoder_outputs_finite_logits(variant: str) -> None:
    torch.manual_seed(0)
    model = build_v8_sequence_decoder(variant, dropout=0.0)
    output = model(torch.randn(3, 18, 32))
    assert output.logits.shape == (3, 4)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.final_membrane).all()
    if model.is_spiking:
        assert output.binary_spikes
        for spikes in output.binary_spikes:
            assert torch.all((spikes == 0) | (spikes == 1))
    else:
        assert not output.binary_spikes


def test_v8_sequence_decoder_rejects_wrong_sequence_shape() -> None:
    model = build_v8_sequence_decoder("clif_plain")
    with pytest.raises(ValueError, match="shape mismatch"):
        model(torch.randn(2, 17, 32))
