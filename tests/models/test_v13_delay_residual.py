from __future__ import annotations

import torch

from dpc_snn.models.v13_delay_residual import SparseLowRankDelayResidual


def test_v13_controls_only_replace_lag_posterior() -> None:
    torch.manual_seed(13)
    module = SparseLowRankDelayResidual(channels=8, max_lag=3, mixing_rank=4)
    sequence = torch.randn(2, 12, 8)
    full, full_stats = module(sequence, control="full")
    zero, zero_stats = module(sequence, control="zero")
    shuffled, shuffled_stats = module(sequence, control="shuffled")
    assert full.shape == zero.shape == shuffled.shape == sequence.shape
    torch.testing.assert_close(
        full_stats["route_probability"], zero_stats["route_probability"]
    )
    torch.testing.assert_close(
        full_stats["route_probability"], shuffled_stats["route_probability"]
    )
    assert torch.all(zero_stats["lag_posterior"][:, 0] == 1.0)
    assert torch.all(zero_stats["lag_posterior"][:, 1:] == 0.0)
    torch.testing.assert_close(
        shuffled_stats["lag_posterior"],
        full_stats["lag_posterior"][module.shuffle_permutation],
    )


def test_v13_delay_residual_is_causal() -> None:
    torch.manual_seed(14)
    module = SparseLowRankDelayResidual(channels=8, max_lag=3, mixing_rank=4).eval()
    sequence = torch.randn(2, 12, 8)
    changed = sequence.clone()
    changed[:, 6:] += 50.0 * torch.randn_like(changed[:, 6:])
    with torch.no_grad():
        first, _ = module(sequence)
        second, _ = module(changed)
    torch.testing.assert_close(first[:, :6], second[:, :6], atol=1e-6, rtol=1e-6)
