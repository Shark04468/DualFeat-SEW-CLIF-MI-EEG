"""Strong neural baselines for the V6.2-R1 comparison protocol.

Official source trees are loaded at runtime from a separately locked directory.
This keeps source provenance explicit and avoids copying third-party licensed
implementations into the project package.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType
from typing import Any, Iterator

import torch
from torch import nn
import torch.nn.functional as F

from dpc_snn.models.surrogate import spike_fn


SOURCE_LOCKS = {
    "TCFormer": (
        "74c89b7ab8c64e4eb51e0f748dd87dd4c94e68c5",
        "da612a53187e1de7ace7b3698ca568174ab7828f73dc4e7677d42e1da8e4fa58",
    ),
    "FBCNet": (
        "de1bbdd8a54cb1e466830e3d47070e0e56761a37",
        "47ced8b2425321ca4cd59b8dbf932bda2e4fc679a783964488ceb95d0c5fab8a",
    ),
    "EEG-ATCNet": (
        "f01c80c1ca86ae7708c5075eb4d9663284d323b4",
        "3691b14fbed00bef5d97c928e966cbe00f269673ac06480e56029bc6506e0c73",
    ),
    "EEG-Conformer": (
        "9ae149ba62487ceae723277d13adac27837113d2",
        "9b0c2b7878d8a8829c9c49390105412926730e64b6e02a95aa9b293ffeacbaa6",
    ),
    "SnnForMI": (
        "57f47431ef0b22814093a3eff3d5e4a370761a70",
        "6e48caf008abc9bb6584137fc4b8247bfb746720ffc6a790d7fa123ae0f3b294",
    ),
}

_IMPORT_LOCK = threading.Lock()
_MODULE_CACHE: dict[tuple[str, str], ModuleType] = {}


def verify_official_source_locks(source_root: str | Path) -> dict[str, dict[str, str]]:
    """Require every official source directory to carry the preregistered lock."""

    root = Path(source_root)
    verified: dict[str, dict[str, str]] = {}
    for name, (commit, archive_sha256) in SOURCE_LOCKS.items():
        lock_path = root / name / "SOURCE_LOCK.txt"
        if not lock_path.is_file():
            raise FileNotFoundError(f"missing official baseline source lock: {lock_path}")
        fields = lock_path.read_text(encoding="utf-8").strip().split()
        expected = [name, commit, archive_sha256]
        if fields != expected:
            raise RuntimeError(
                f"official source lock mismatch for {name}: expected {expected}, got {fields}"
            )
        verified[name] = {"commit": commit, "archive_sha256": archive_sha256}
    return verified


class _ClassificationStub(nn.Module):
    """Import-only parent for official pure modules; never instantiated."""


@contextmanager
def _official_utils(source_dir: Path) -> Iterator[None]:
    previous = {key: value for key, value in sys.modules.items() if key == "utils" or key.startswith("utils.")}
    for key in list(previous):
        del sys.modules[key]
    package = ModuleType("utils")
    package.__path__ = [str(source_dir / "utils")]  # type: ignore[attr-defined]
    package.__package__ = "utils"
    sys.modules["utils"] = package
    try:
        yield
    finally:
        for key in [key for key in sys.modules if key == "utils" or key.startswith("utils.")]:
            del sys.modules[key]
        sys.modules.update(previous)


def _load_tcformer_file(source_root: Path, stem: str) -> ModuleType:
    source_dir = source_root / "TCFormer"
    key = (str(source_dir.resolve()), stem)
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    with _IMPORT_LOCK:
        if key in _MODULE_CACHE:
            return _MODULE_CACHE[key]
        digest = hashlib.sha256(str(source_dir.resolve()).encode()).hexdigest()[:12]
        namespace = f"_dpc_official_tcformer_{digest}"
        if namespace not in sys.modules:
            package = ModuleType(namespace)
            package.__path__ = [str(source_dir)]  # type: ignore[attr-defined]
            package.__package__ = namespace
            sys.modules[namespace] = package
        models_name = f"{namespace}.models"
        if models_name not in sys.modules:
            models = ModuleType(models_name)
            models.__path__ = [str(source_dir / "models")]  # type: ignore[attr-defined]
            models.__package__ = models_name
            sys.modules[models_name] = models
        stub_name = f"{models_name}.classification_module"
        if stub_name not in sys.modules:
            stub = ModuleType(stub_name)
            stub.ClassificationModule = _ClassificationStub  # type: ignore[attr-defined]
            sys.modules[stub_name] = stub
        module_name = f"{models_name}.{stem}"
        spec = importlib.util.spec_from_file_location(module_name, source_dir / "models" / f"{stem}.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load official TCFormer module {stem}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            with _official_utils(source_dir):
                spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        _MODULE_CACHE[key] = module
        return module


def _load_fbcnet_file(source_root: Path) -> ModuleType:
    source_dir = source_root / "FBCNet"
    path = source_dir / "codes" / "centralRepo" / "networks.py"
    key = (str(source_dir.resolve()), "networks")
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    with _IMPORT_LOCK:
        if key in _MODULE_CACHE:
            return _MODULE_CACHE[key]
        name = f"_dpc_official_fbcnet_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load official FBCNet module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MODULE_CACHE[key] = module
        return module


class TensorLogitsAdapter(nn.Module):
    """Convert an official tensor-output module to the project model contract."""

    def __init__(self, module: nn.Module, implementation_id: str):
        super().__init__()
        self.module = module
        self.implementation_id = implementation_id

    def forward(self, x: torch.Tensor, **_: torch.Tensor) -> dict[str, Any]:
        return {"logits": self.module(x), "aux": {}}


class TorchPLIFMI(nn.Module):
    """Torch backend of the published SnnForMI Conv1d-PLIF classifier."""

    implementation_id = "snnformi_conv1d_plif_torch_backend_v1"

    def __init__(
        self,
        n_channels: int = 22,
        n_classes: int = 4,
        samples: int = 1000,
        width_multiplier: float = 2.0,
        initial_w: float = 0.5,
    ) -> None:
        super().__init__()
        width = int(round(width_multiplier * n_channels))
        kernel = max(3, samples // 32)
        if kernel % 2 == 0:
            kernel += 1
        self.channel_encoder = nn.Conv1d(n_channels, width, 1, bias=False)
        self.temporal_encoder = nn.Conv1d(
            width, width, kernel, padding=kernel // 2, groups=width, bias=False
        )
        self.normalization = nn.BatchNorm1d(width)
        self.w = nn.Parameter(torch.full((width,), float(initial_w)))
        self.classifier = nn.Linear(width, n_classes)

    def forward(self, x: torch.Tensor, **_: torch.Tensor) -> dict[str, Any]:
        current = self.normalization(self.temporal_encoder(self.channel_encoder(x)))
        membrane = torch.zeros_like(current[..., 0])
        spike_sum = torch.zeros_like(membrane)
        tau = torch.exp(-self.w).add(1.0).view(1, -1)
        for index in range(current.shape[-1]):
            membrane = membrane + (current[..., index] - membrane) / tau
            spike = spike_fn(membrane - 1.0)
            membrane = membrane * (1.0 - spike.detach())
            spike_sum = spike_sum + spike
        rate = spike_sum / current.shape[-1]
        return {
            "logits": self.classifier(rate),
            "aux": {"spike_rate": rate.mean(), "binary_spikes_only": True},
        }


class _PositiveScaleFusion(nn.Module):
    def __init__(self, scales: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(scales, scales))

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        weights = F.softplus(self.logits)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return [sum(weights[i, j] * value for j, value in enumerate(features)) for i in range(len(features))]


class _CBAM1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.channel = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, channels)
        )
        self.temporal = nn.Conv1d(2, 1, 7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=-1) + x.amax(dim=-1)
        x = x * torch.sigmoid(self.channel(pooled)).unsqueeze(-1)
        temporal = torch.cat((x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)), dim=1)
        return x * torch.sigmoid(self.temporal(temporal))


class _ResidualTCN(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.padding = 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def _causal(self, layer: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        return layer(F.pad(x, (self.padding, 0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout(F.elu(self.norm1(self._causal(self.conv1, x))))
        x = self.dropout(F.elu(self.norm2(self._causal(self.conv2, x))))
        return F.elu(x + residual)


class BFATCNetReconstruction(nn.Module):
    """Paper-faithful BFATCNet reconstruction; no official implementation exists."""

    implementation_id = "bfatcnet_paper_reconstruction_v1"

    def __init__(self, n_channels: int = 22, n_classes: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        width = 16
        kernels = (31, 63, 125)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, 8, (1, kernel), padding=(0, kernel // 2), bias=False),
                    nn.BatchNorm2d(8),
                    nn.Conv2d(8, width, (n_channels, 1), groups=8, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ELU(),
                    nn.AvgPool2d((1, 8)),
                    nn.Dropout(dropout),
                )
                for kernel in kernels
            ]
        )
        self.bifpn = _PositiveScaleFusion(len(kernels))
        channels = width * len(kernels)
        self.cbam = _CBAM1D(channels)
        self.attention = nn.MultiheadAttention(channels, num_heads=4, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(channels)
        self.tcn = nn.Sequential(
            _ResidualTCN(channels, dilation=1, dropout=dropout),
            _ResidualTCN(channels, dilation=2, dropout=dropout),
        )
        self.classifier = nn.Linear(channels * 2, n_classes)

    def forward(self, x: torch.Tensor, **_: torch.Tensor) -> dict[str, Any]:
        image = x.unsqueeze(1)
        scales = [branch(image).squeeze(2) for branch in self.branches]
        features = self.cbam(torch.cat(self.bifpn(scales), dim=1))
        tokens = features.transpose(1, 2)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        features = self.attention_norm(tokens + attended).transpose(1, 2)
        features = self.tcn(features)
        summary = torch.cat((features.mean(dim=-1), features[..., -1]), dim=1)
        return {"logits": self.classifier(summary), "aux": {}}


def _conformer_input_size(samples: int, embedding_size: int = 40) -> int:
    tokens = (samples - 25 - 75 + 1) // 15 + 1
    if tokens <= 0:
        raise ValueError("EEG Conformer input is too short")
    return tokens * embedding_size


def build_v62_neural_baseline(
    name: str,
    *,
    source_root: str | Path,
    n_channels: int = 22,
    n_classes: int = 4,
    samples: int = 1000,
) -> nn.Module:
    """Build one baseline under the common V6.2 tensor contract."""

    root = Path(source_root)
    normalized = name.lower().replace("-", "_")
    if normalized == "eegnet":
        module = _load_tcformer_file(root, "eegnet")
        core = module.EEGNetModule(
            n_channels=n_channels,
            n_classes=n_classes,
            input_window_samples=samples,
            F1=8,
            D=2,
            F2=16,
            kernel_length=32,
            drop_prob=0.5,
            pool_time_length=4,
            pool_time_stride=4,
            kernel_length_dw_sep=16,
        )
        return TensorLogitsAdapter(core, "tcformer_official_eegnet")
    if normalized == "atcnet":
        module = _load_tcformer_file(root, "atcnet")
        core = module.ATCNetModule(n_channels=n_channels, n_classes=n_classes)
        return TensorLogitsAdapter(core, "tcformer_official_atcnet")
    if normalized == "tcformer":
        module = _load_tcformer_file(root, "tcformer")
        core = module.TCFormerModule(
            n_channels=n_channels,
            n_classes=n_classes,
            F1=32,
            temp_kernel_lengths=(20, 32, 64),
            d_group=16,
            D=2,
            pool_length_1=8,
            pool_length_2=7,
            dropout_conv=0.4,
            q_heads=4,
            kv_heads=2,
            trans_depth=5,
            trans_dropout=0.4,
            tcn_depth=2,
            kernel_length_tcn=4,
            dropout_tcn=0.3,
        )
        return TensorLogitsAdapter(core, "tcformer_official_tcformer")
    if normalized in {"eeg_conformer", "eegconformer"}:
        module = _load_tcformer_file(root, "eegconformer")
        core = module.EEGConformerModule(
            n_channels=n_channels,
            n_classes=n_classes,
            embedding_size=40,
            depth=6,
            input_size_cls=_conformer_input_size(samples),
        )
        return TensorLogitsAdapter(core, "tcformer_official_eegconformer")
    if normalized == "fbcnet":
        module = _load_fbcnet_file(root)
        core = module.FBCNet(
            nChan=n_channels,
            nTime=samples,
            nClass=n_classes,
            nBands=9,
            m=32,
            temporalLayer="LogVarLayer",
            strideFactor=4,
            doWeightNorm=True,
        )
        return TensorLogitsAdapter(core, "fbcnet_official_network")
    if normalized in {"mi_snn", "mi_snn_plif", "snnformi"}:
        return TorchPLIFMI(n_channels, n_classes, samples)
    if normalized == "bfatcnet":
        return BFATCNetReconstruction(n_channels, n_classes)
    raise KeyError(f"unknown V6.2 neural baseline: {name}")


V62_NEURAL_BASELINES = (
    "eegnet",
    "fbcnet",
    "atcnet",
    "tcformer",
    "eeg_conformer",
    "mi_snn_plif",
    "bfatcnet",
)
