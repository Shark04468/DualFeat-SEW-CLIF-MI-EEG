"""Surrogate spike functions."""

from __future__ import annotations

import torch


class FastSigmoidSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        ctx.slope = slope
        return (input_tensor >= 0).to(input_tensor.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (input_tensor,) = ctx.saved_tensors
        slope = ctx.slope
        grad = grad_output / (slope * input_tensor.abs() + 1.0).pow(2)
        return grad, None


def spike_fn(x: torch.Tensor, slope: float = 10.0) -> torch.Tensor:
    return FastSigmoidSpike.apply(x, slope)

