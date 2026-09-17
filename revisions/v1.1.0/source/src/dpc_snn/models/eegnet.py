"""Compact EEGNet baseline with the original architectural constraints."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _max_norm_weight(weight: torch.Tensor, maximum: float) -> torch.Tensor:
    flat = weight.flatten(1)
    scale = (float(maximum) / flat.norm(dim=1, keepdim=True).clamp_min(1e-8)).clamp_max(1.0)
    return (flat * scale).reshape_as(weight)


class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        samples: int = 256,
        sfreq: float = 128.0,
        f1: int = 8,
        d: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.implementation_id = "eegnet_v1_sfreq_scaled_maxnorm"
        f2 = f1 * d
        temporal_kernel = max(3, int(round(float(sfreq) / 2.0)))
        separable_kernel = max(3, int(round(float(sfreq) / 8.0)))
        self.temporal = nn.Conv2d(
            1, f1, kernel_size=(1, temporal_kernel), padding="same", bias=False
        )
        self.bn1 = nn.BatchNorm2d(f1)
        self.depthwise = nn.Conv2d(f1, f2, kernel_size=(n_channels, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f2)
        self.sep_depth = nn.Conv2d(
            f2,
            f2,
            kernel_size=(1, separable_kernel),
            padding="same",
            groups=f2,
            bias=False,
        )
        self.sep_point = nn.Conv2d(f2, f2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.dropout = nn.Dropout(dropout)
        with torch.no_grad():
            self.eval()
            dummy = torch.zeros(1, 1, n_channels, samples)
            features = self._features(dummy).flatten(1).shape[1]
            self.train()
        self.classifier = nn.Linear(features, n_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.bn1(x)
        x = F.conv2d(
            x,
            _max_norm_weight(self.depthwise.weight, 1.0),
            bias=None,
            stride=self.depthwise.stride,
            padding=self.depthwise.padding,
            dilation=self.depthwise.dilation,
            groups=self.depthwise.groups,
        )
        x = F.elu(self.bn2(x))
        x = F.avg_pool2d(x, kernel_size=(1, 4))
        x = self.dropout(x)
        x = self.sep_depth(x)
        x = self.sep_point(x)
        x = F.elu(self.bn3(x))
        x = F.avg_pool2d(x, kernel_size=(1, 8))
        x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor, **_: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: [batch, channels, time]
        features = self._features(x.unsqueeze(1)).flatten(1)
        logits = F.linear(
            features,
            _max_norm_weight(self.classifier.weight, 0.25),
            self.classifier.bias,
        )
        return {"logits": logits, "aux": {}}
