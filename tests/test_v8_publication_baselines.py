from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from dpc_snn.experiments.v8_publication_baselines import (
    REGION_CHANNELS,
    component_error_profile,
    fft_band_occlusion,
    input_gradient_channel_saliency,
    validate_region_partition,
    zero_reference_region,
)


CHANNELS = [channel for region in REGION_CHANNELS.values() for channel in region]


def test_region_partition_is_exact_and_occlusion_is_local() -> None:
    partition = validate_region_partition(CHANNELS)
    assert sum(len(indices) for indices in partition.values()) == len(CHANNELS) == 22
    carrier = np.ones((2, 22, 8), dtype=np.float32)
    occluded = zero_reference_region(
        carrier, channel_names=CHANNELS, region="midline_motor"
    )
    assert np.all(occluded[:, partition["midline_motor"], :] == 0.0)
    assert np.count_nonzero(occluded == 0.0) == 2 * 3 * 8

    with pytest.raises(ValueError, match="cover"):
        validate_region_partition(CHANNELS[:-1])


def test_fft_band_occlusion_removes_only_selected_tone() -> None:
    sfreq = 250.0
    time = np.arange(1000) / sfreq
    carrier = (np.sin(2 * np.pi * 10 * time) + np.sin(2 * np.pi * 25 * time))[None, None]
    occluded = fft_band_occlusion(
        carrier.astype(np.float32), sfreq=sfreq, low_hz=8.0, high_hz=13.0
    )
    spectrum = np.abs(np.fft.rfft(occluded[0, 0]))
    frequencies = np.fft.rfftfreq(1000, d=1.0 / sfreq)
    ten = spectrum[np.argmin(np.abs(frequencies - 10.0))]
    twenty_five = spectrum[np.argmin(np.abs(frequencies - 25.0))]
    assert ten < 1e-3
    assert twenty_five > 400.0


def test_gradient_times_input_saliency_respects_channel_axis() -> None:
    class WeightedModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            score = (x[:, 0] * 4.0).sum(dim=-1) + x[:, 1].sum(dim=-1)
            return torch.stack((score, -score), dim=1)

    x = np.ones((3, 2, 5), dtype=np.float32)
    labels = np.zeros(3, dtype=np.int64)
    raw, normalized = input_gradient_channel_saliency(
        WeightedModel(), x, labels, device="cpu", batch_size=2, channel_axis=1
    )
    assert raw.shape == normalized.shape == (3, 2)
    np.testing.assert_allclose(raw[:, 0] / raw[:, 1], 4.0)
    np.testing.assert_allclose(normalized.sum(axis=1), 1.0)


def test_error_profile_exposes_component_rescues() -> None:
    labels = np.asarray([0, 0, 1, 1])
    first = np.asarray([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.7, 0.3]])
    second = np.asarray([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.1, 0.9]])
    profile = component_error_profile(labels, first, second)
    assert profile["first_only_correct"] == 1
    assert profile["second_only_correct"] == 2
    assert profile["double_fault_rate"] == 0.0
    assert profile["oracle_accuracy"] == 1.0

