"""Accuracy anchor around the pinned official ATCNet core.

The wrapper deliberately does not reimplement third-party layers.  It exposes
the continuous convolutional sequence needed by later matched ANN/SNN heads,
while the anchor forward remains exactly the official ATCNet computation.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class V8ATCAccuracyBackbone(nn.Module):
    """Expose an official ATCNet core without changing its logits."""

    architecture_version = "dpc_snn_v8_atc_accuracy_anchor_r1"
    implementation_id = "pinned_official_atcnet_core_exact_wrapper"

    def __init__(self, official_core: nn.Module) -> None:
        super().__init__()
        required = ("conv_block", "rearrange", "atc_blocks", "n_windows", "n_classes")
        missing = [name for name in required if not hasattr(official_core, name)]
        if missing:
            raise TypeError(f"official ATC core is missing required attributes: {missing}")
        self.official_core = official_core

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def continuous_sequence(self, carrier: torch.Tensor) -> torch.Tensor:
        """Return the official post-convolution sequence before ATC windows."""

        if carrier.ndim != 3:
            raise ValueError("ATC carrier must have shape [batch, channels, time]")
        convolved = self.official_core.conv_block(carrier)
        sequence = self.official_core.rearrange(convolved)
        if sequence.ndim != 3:
            raise RuntimeError("official ATC stem did not return [batch, time, features]")
        return sequence

    def anchor_logits_from_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """Run the untouched official ATC windows on an extracted sequence."""

        if sequence.ndim != 3:
            raise ValueError("ATC sequence must have shape [batch, time, features]")
        batch, steps, _ = sequence.shape
        windows = int(self.official_core.n_windows)
        if steps < windows:
            raise ValueError("ATC sequence is shorter than its registered window count")
        logits = sequence.new_zeros(batch, int(self.official_core.n_classes))
        for index, block in enumerate(self.official_core.atc_blocks):
            logits = logits + block(sequence[:, index : steps - windows + index + 1])
        return logits / float(windows)

    def forward(self, carrier: torch.Tensor, **_: Any) -> dict[str, Any]:
        sequence = self.continuous_sequence(carrier)
        logits = self.anchor_logits_from_sequence(sequence)
        return {
            "logits": logits,
            "aux": {
                "architecture_version": self.architecture_version,
                "implementation_id": self.implementation_id,
                "continuous_sequence": sequence,
                "parameter_count": self.parameter_count,
                "trainable_parameter_count": self.trainable_parameter_count,
            },
        }
