"""Fold-local feature caching and training for the V6.2 zero-delay scaffold."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import random
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dpc_snn.experiments.v62_baselines import (
    FixedGain,
    apply_fixed_gain,
    fbcnet_filterbank,
    fit_baseline,
    fit_fixed_gain,
    task_carrier,
)
from dpc_snn.baselines.neural import build_v62_neural_baseline
from dpc_snn.models.dasp_snn_v62 import DASPSNNV62
from dpc_snn.models.delayed_full_window_atc_readout import (
    DelayedFullWindowATCReadout,
)
from dpc_snn.utils.metrics import classification_metrics


@dataclass(frozen=True)
class CachedRates:
    fast: torch.Tensor
    slow: torch.Tensor
    broadband: torch.Tensor
    context: torch.Tensor | None
    gain: torch.Tensor
    frontend_fingerprint: str


@dataclass
class ScaffoldFitResult:
    model: DASPSNNV62
    history: list[dict[str, float | int]]
    best_epoch: int
    best_metric: float
    best_state: dict[str, torch.Tensor]
    last_state: dict[str, torch.Tensor]
    optimizer_steps: int
    elapsed_seconds: float


def seed_scaffold(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def project_registered_max_norm_constraints_(module: nn.Module) -> int:
    """Canonicalize third-party max-norm weights outside inference forward.

    The locked TCFormer implementation assigns ``weight.data`` in ``forward``.
    Applying the identical projection after every optimizer update keeps its
    training semantics while ensuring saved checkpoints and evaluation calls
    are state-invariant.
    """

    projected = 0
    constrained_types = (nn.Linear, nn.Conv1d, nn.Conv2d)
    for layer in module.modules():
        maximum = getattr(layer, "max_norm", None)
        weight = getattr(layer, "weight", None)
        if maximum is None or weight is None or not isinstance(layer, constrained_types):
            continue
        weight.copy_(
            torch.renorm(
                weight,
                p=2,
                dim=0,
                maxnorm=float(maximum),
            )
        )
        projected += 1
    return projected


def build_zero_scaffold(
    *,
    seed: int,
    model_config: dict[str, Any],
    official_source_root: str | None = None,
) -> DASPSNNV62:
    seed_scaffold(seed)
    resolved_model_config = dict(model_config)
    atc_variant = str(
        resolved_model_config.pop("atc_readout_variant", "causal")
    ).lower()
    atc_delayed_residual_bound = float(
        resolved_model_config.pop("atc_delayed_residual_bound", 0.0)
    )
    atc_delayed_fusion_mode = str(
        resolved_model_config.pop("atc_delayed_fusion_mode", "additive")
    )
    atc_delayed_statistics_bins = int(
        resolved_model_config.pop("atc_delayed_statistics_bins", 0)
    )
    atc_delayed_statistics_interaction_gain = float(
        resolved_model_config.pop("atc_delayed_statistics_interaction_gain", 0.0)
    )
    atc_delayed_statistics_uncertainty_gate = bool(
        resolved_model_config.pop("atc_delayed_statistics_uncertainty_gate", False)
    )
    atc_delayed_statistics_cp_rank = int(
        resolved_model_config.pop("atc_delayed_statistics_cp_rank", 0)
    )
    atc_delayed_statistics_directional_moments = bool(
        resolved_model_config.pop(
            "atc_delayed_statistics_directional_moments",
            False,
        )
    )
    atc_delayed_route_statistics_bins = int(
        resolved_model_config.pop("atc_delayed_route_statistics_bins", 0)
    )
    atc_delayed_route_statistics_cp_rank = int(
        resolved_model_config.pop("atc_delayed_route_statistics_cp_rank", 0)
    )
    model = DASPSNNV62(
        **resolved_model_config,
        decoder_kind="ann",
        force_zero_delay=True,
    )
    if atc_variant not in {"causal", "official_full_window"}:
        raise ValueError(f"unknown ATC readout variant: {atc_variant}")
    if atc_variant == "official_full_window":
        if official_source_root is None:
            raise ValueError("official_full_window ATC requires an official source root")
        if model.atc_readout is None:
            raise ValueError("official_full_window ATC requires use_atc_readout=true")
        if not model.spatial.frozen:
            raise ValueError("official ATC sensor reconstruction requires a frozen basis")
        # Match the official baseline constructor exactly. Building the frozen
        # delay shell above must not advance the ATC core's initialization RNG.
        seed_scaffold(seed)
        core = build_v62_neural_baseline(
            "atcnet",
            source_root=official_source_root,
            n_channels=model.n_channels,
            n_classes=model.n_classes,
            samples=model.decoder.endpoint_samples[-1],
        )
        spatial_weight = model.spatial.shared_weight().detach()
        gram = spatial_weight @ spatial_weight.transpose(0, 1)
        exact_orthonormal_basis = (
            spatial_weight.shape[0] == spatial_weight.shape[1]
            and torch.equal(
                gram,
                torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device),
            )
        )
        node_reconstruction = (
            spatial_weight.transpose(0, 1)
            if exact_orthonormal_basis
            else torch.linalg.pinv(spatial_weight)
        )
        model.atc_readout = DelayedFullWindowATCReadout(
            core,
            model.n_bands,
            model.n_nodes,
            model.n_classes,
            endpoint_samples=model.decoder.endpoint_samples,
            node_reconstruction=node_reconstruction,
            delayed_residual_bound=atc_delayed_residual_bound,
            delayed_fusion_mode=atc_delayed_fusion_mode,
            delayed_statistics_bins=atc_delayed_statistics_bins,
            delayed_statistics_interaction_gain=(
                atc_delayed_statistics_interaction_gain
            ),
            delayed_statistics_uncertainty_gate=(
                atc_delayed_statistics_uncertainty_gate
            ),
            delayed_statistics_cp_rank=atc_delayed_statistics_cp_rank,
            delayed_statistics_directional_moments=(
                atc_delayed_statistics_directional_moments
            ),
            delayed_route_statistics_bins=atc_delayed_route_statistics_bins,
            delayed_route_statistics_cp_rank=(
                atc_delayed_route_statistics_cp_rank
            ),
        )
        model.delay.retain_slow_route_delay_contrast = bool(
            atc_delayed_route_statistics_bins
        )
        if model.parameter_count > model.parameter_ceiling:
            raise ValueError(
                f"official ATC scaffold has {model.parameter_count:,} parameters, above "
                f"the registered {model.parameter_ceiling:,} ceiling"
            )
    for name, parameter in model.delay.named_parameters():
        if name.startswith(
            (
                "slow_delay_field.",
                "fast_delay_field.",
                "slow_context.",
                "fast_context.",
            )
        ) or name in {"slow_delay_bias", "fast_delay_bias"}:
            parameter.requires_grad_(False)
    model.delay.phase_preference.requires_grad_(False)
    if (
        model.delay.slow_residual_route_scale == 0.0
        and model.delay.fast_residual_route_scale == 0.0
    ):
        for module in (
            model.delay.slow_amplitude,
            model.delay.fast_amplitude,
            model.delay.slow_gate_field,
            model.delay.fast_gate_field,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        model.delay.slow_gate_bias.requires_grad_(False)
        model.delay.fast_gate_bias.requires_grad_(False)
    if model.delay.target_interaction_bound == 0.0:
        for parameter in model.delay.target_interaction.parameters():
            parameter.requires_grad_(False)
    if model.delay.phase_bound == 0.0:
        model.delay.phase_strength_raw.requires_grad_(False)
    if model.statistical_fusion == "statistics" and model.statistical_readout is not None:
        for module in (model.temporal_pyramid, model.geometry, model.decoder):
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        if model.statistical_classifier is None:
            raise RuntimeError("statistics-only scaffold requires its registered classifier")
        for parameter in model.statistical_classifier.parameters():
            parameter.requires_grad_(True)
        model.delay.fast_contribution_raw.requires_grad_(False)
    elif model.atc_readout is not None and model.statistical_readout is None:
        for module in (model.temporal_pyramid, model.geometry, model.decoder):
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        if hasattr(model.atc_readout, "delayed_residual_mix_raw"):
            model.atc_readout.delayed_residual_mix_raw.requires_grad_(False)
    return model


def build_frontend_only_scaffold(
    *,
    seed: int,
    model_config: dict[str, Any],
) -> DASPSNNV62:
    """Build the registered V6.2/V7 frontend without an unrelated readout dependency."""

    frontend_config = dict(model_config)
    frontend_config.update(
        {
            "use_atc_readout": False,
            "use_statistical_readout": False,
            "use_geometry": False,
            "atc_readout_variant": "causal",
        }
    )
    return build_zero_scaffold(seed=seed, model_config=frontend_config)


def _raw_loader(x: np.ndarray, batch_size: int) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.as_tensor(np.asarray(x), dtype=torch.float32)),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def fit_projected_training_gain(
    model: DASPSNNV62,
    x_inner_train: np.ndarray,
    *,
    device: str,
    batch_size: int = 16,
) -> torch.Tensor:
    """Fit inverse carrier RMS using inner-train trials and the frozen front end."""

    if not model.spatial.frozen:
        raise ValueError("fold-local caching requires a frozen physical spatial basis")
    model.eval().to(device)
    model.set_training_gain(torch.ones(model.n_bands, model.n_nodes))
    energy = torch.zeros(model.n_bands, model.n_nodes, dtype=torch.float64)
    observations = 0
    for (batch_x,) in _raw_loader(x_inner_train, batch_size):
        prepared, _ = model._prepare_raw(batch_x.to(device, non_blocking=True))
        analytic = model.filterbank(prepared)
        projected = model.spatial(analytic)
        rates = model.resampler(
            projected,
            epoch_tmin=model.epoch_tmin,
            task_tmin=model.task_tmin,
            task_tmax=model.task_tmax,
        )
        energy += rates.fast.abs().square().sum(dim=(0, 3)).double().cpu()
        observations += rates.fast.shape[0] * rates.fast.shape[-1]
    if observations < 1:
        raise ValueError("projected gain fitting requires at least one training trial")
    rms = (energy / observations).sqrt()
    finite = rms[torch.isfinite(rms) & (rms > 0)]
    if finite.numel() == 0:
        raise ValueError("projected training features have no finite non-zero RMS")
    floor = max(float(finite.median()) * 1e-3, torch.finfo(torch.float32).tiny)
    gain = rms.clamp_min(floor).reciprocal().float()
    if not bool(torch.isfinite(gain).all()):
        raise FloatingPointError("fold-local projected gain is non-finite")
    model.set_training_gain(gain)
    return gain.cpu()


@torch.no_grad()
def cache_rate_features(
    model: DASPSNNV62,
    x: np.ndarray,
    *,
    device: str,
    batch_size: int = 16,
) -> CachedRates:
    """Cache only frozen-front-end outputs, never post-delay or readout features."""

    if not model.spatial.frozen or not bool(model.resampler.training_gain_ready):
        raise ValueError("feature caching requires a frozen basis and fitted training gain")
    model.eval().to(device)
    fast_parts: list[torch.Tensor] = []
    slow_parts: list[torch.Tensor] = []
    broadband_parts: list[torch.Tensor] = []
    context_parts: list[torch.Tensor] = []
    for (batch_x,) in _raw_loader(x, batch_size):
        prepared, context = model._prepare_raw(batch_x.to(device, non_blocking=True))
        if context is not None:
            context_parts.append(context.float().cpu())
        analytic = model.filterbank(prepared)
        projected = model.spatial(analytic)
        rates = model.resampler(
            projected,
            epoch_tmin=model.epoch_tmin,
            task_tmin=model.task_tmin,
            task_tmax=model.task_tmax,
        )
        if model.resampler.fast_decimation == 1:
            task_start = int(round((model.task_tmin - model.epoch_tmin) * model.sfreq))
            task_stop = int(round((model.task_tmax - model.epoch_tmin) * model.sfreq))
            broadband = model.spatial.project_real(
                prepared[..., task_start:task_stop]
            )
        else:
            broadband = rates.fast.real.sum(dim=1) / (model.n_bands**0.5)
        broadband_parts.append(broadband.float().cpu())
        fast_parts.append(rates.fast.to(torch.complex64).cpu())
        slow_parts.append(rates.slow.float().cpu())
    return CachedRates(
        fast=torch.cat(fast_parts),
        slow=torch.cat(slow_parts),
        broadband=torch.cat(broadband_parts),
        context=torch.cat(context_parts) if context_parts else None,
        gain=model.resampler.training_gain.detach().cpu().clone(),
        frontend_fingerprint=model.frontend_fingerprint(),
    )


def fit_official_fbc_channel_gain(
    x_inner_train: np.ndarray,
    *,
    sfreq: float = 250.0,
    epoch_tmin: float = -1.0,
    clip: float = 12.0,
) -> FixedGain:
    carrier = task_carrier(
        x_inner_train,
        sfreq=float(sfreq),
        epoch_tmin=float(epoch_tmin),
    )
    return fit_fixed_gain(carrier, clip=float(clip))


@torch.no_grad()
def cache_official_fbc_rate_features(
    model: DASPSNNV62,
    x: np.ndarray,
    channel_gain: FixedGain,
    *,
    device: str,
    batch_size: int = 16,
) -> CachedRates:
    """Cache official causal FBC signals immediately before mandatory delay."""

    if model.n_bands != 9:
        raise ValueError("official FBC cache requires the registered nine 4-Hz bands")
    if model.resampler.fast_decimation != 1:
        raise ValueError("official FBC cache preserves the full 250 Hz carrier rate")
    if not model.spatial.frozen or model.context_encoder is not None:
        raise ValueError("official FBC E2 cache requires a frozen basis and no context encoder")
    model.eval().to(device)
    model.set_training_gain(torch.ones(model.n_bands, model.n_nodes))
    fast_parts: list[torch.Tensor] = []
    broadband_parts: list[torch.Tensor] = []
    array = np.asarray(x, dtype=np.float32)
    for start in range(0, array.shape[0], int(batch_size)):
        raw_batch = array[start : start + int(batch_size)]
        carrier = task_carrier(
            raw_batch,
            sfreq=model.sfreq,
            epoch_tmin=model.epoch_tmin,
        )
        normalized = apply_fixed_gain(carrier, channel_gain)
        filtered = fbcnet_filterbank(normalized, sfreq=model.sfreq)
        real = torch.from_numpy(filtered[:, 0]).permute(0, 3, 1, 2).to(device)
        analytic = torch.complex(real, torch.zeros_like(real))
        projected = model.spatial(analytic)
        fast_parts.append(projected.to(torch.complex64).cpu())
        broadband_parts.append(
            model.spatial.project_real(torch.from_numpy(normalized).to(device)).float().cpu()
        )
    fast = torch.cat(fast_parts)
    slow = torch.zeros(
        fast.shape[0],
        fast.shape[1],
        fast.shape[2],
        fast.shape[-1] // 2,
        dtype=torch.float32,
    )
    return CachedRates(
        fast=fast,
        slow=slow,
        broadband=torch.cat(broadband_parts),
        context=None,
        gain=model.resampler.training_gain.detach().cpu().clone(),
        frontend_fingerprint=model.frontend_fingerprint(),
    )


@torch.no_grad()
def cache_analytic_fixed_channel_gain_rate_features(
    model: DASPSNNV62,
    x: np.ndarray,
    channel_gain: FixedGain,
    *,
    device: str,
    batch_size: int = 16,
) -> CachedRates:
    """Cache the full analytic delay bank while preserving ATC input parity.

    The fold-fitted channel gain is applied to the complete baseline-corrected
    epoch. The mandatory broadband path is therefore exactly the normalized
    task carrier used by the official ATC baseline, while the 12-band fast and
    slow tensors remain available for later delay stages.
    """

    if not model.spatial.frozen:
        raise ValueError("fixed-channel analytic caching requires a frozen basis")
    values = torch.as_tensor(channel_gain.values, dtype=torch.float32)
    if values.shape != (1, model.n_channels, 1):
        raise ValueError("fixed channel gain must have shape [1, channels, 1]")
    model.eval().to(device)
    model.set_training_gain(torch.ones(model.n_bands, model.n_nodes))
    values = values.to(device)
    fast_parts: list[torch.Tensor] = []
    slow_parts: list[torch.Tensor] = []
    broadband_parts: list[torch.Tensor] = []
    context_parts: list[torch.Tensor] = []
    array = np.asarray(x, dtype=np.float32)
    for (batch_x,) in _raw_loader(array, batch_size):
        raw_numpy = batch_x.numpy()
        prepared, context = model._prepare_raw(batch_x.to(device, non_blocking=True))
        if context is not None:
            context_parts.append(context.float().cpu())
        normalized = (prepared / values).clamp(-channel_gain.clip, channel_gain.clip)
        analytic = model.filterbank(normalized)
        projected = model.spatial(analytic)
        rates = model.resampler(
            projected,
            epoch_tmin=model.epoch_tmin,
            task_tmin=model.task_tmin,
            task_tmax=model.task_tmax,
        )
        fast_parts.append(rates.fast.to(torch.complex64).cpu())
        slow_parts.append(rates.slow.float().cpu())
        exact_task = apply_fixed_gain(
            task_carrier(
                raw_numpy,
                sfreq=model.sfreq,
                epoch_tmin=model.epoch_tmin,
                task_start=model.task_tmin,
                task_stop=model.task_tmax,
            ),
            channel_gain,
        )
        broadband_parts.append(
            model.spatial.project_real(torch.from_numpy(exact_task).to(device))
            .float()
            .cpu()
        )
    return CachedRates(
        fast=torch.cat(fast_parts),
        slow=torch.cat(slow_parts),
        broadband=torch.cat(broadband_parts),
        context=torch.cat(context_parts) if context_parts else None,
        gain=model.resampler.training_gain.detach().cpu().clone(),
        frontend_fingerprint=model.frontend_fingerprint(),
    )


def aligned_slow_rate_analytic_evidence(rates: CachedRates) -> torch.Tensor:
    """Return online fast-path analytic samples at every slow-path timestamp."""

    if not rates.fast.is_complex() or rates.slow.is_complex():
        raise ValueError("aligned evidence requires complex fast and real slow features")
    if rates.fast.shape[:3] != rates.slow.shape[:3]:
        raise ValueError("fast and slow feature axes do not match")
    if rates.fast.shape[-1] != 2 * rates.slow.shape[-1]:
        raise ValueError("fast and slow rates must have an exact 2:1 sample ratio")
    aligned = rates.fast[..., ::2]
    if aligned.shape != rates.slow.shape:
        raise RuntimeError("aligned analytic evidence does not match the slow path")
    return aligned


def save_cached_rates(path: str | Any, rates: CachedRates) -> None:
    torch.save(
        {
            "fast": rates.fast,
            "slow": rates.slow,
            "broadband": rates.broadband,
            "context": rates.context,
            "gain": rates.gain,
            "frontend_fingerprint": rates.frontend_fingerprint,
        },
        path,
    )


def load_cached_rates(path: str | Any, model: DASPSNNV62) -> CachedRates:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = model.frontend_fingerprint()
    if payload["frontend_fingerprint"] != expected:
        raise RuntimeError("cached front-end fingerprint does not match the active model")
    gain = payload["gain"].float()
    model.set_training_gain(gain)
    return CachedRates(
        fast=payload["fast"].to(torch.complex64),
        slow=payload["slow"].float(),
        broadband=payload["broadband"].float(),
        context=(payload["context"].float() if payload["context"] is not None else None),
        gain=gain,
        frontend_fingerprint=expected,
    )


def _rates_loader(
    rates: CachedRates,
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    if (
        rates.fast.shape[0] != len(labels)
        or rates.slow.shape[0] != len(labels)
        or rates.broadband.shape[0] != len(labels)
    ):
        raise ValueError("cached rates and labels have different trial counts")
    generator = torch.Generator().manual_seed(int(seed))
    context = (
        rates.context
        if rates.context is not None
        else torch.empty(rates.fast.shape[0], 0, dtype=torch.float32)
    )
    return DataLoader(
        TensorDataset(
            rates.fast,
            rates.slow,
            rates.broadband,
            context,
            torch.as_tensor(labels, dtype=torch.long),
        ),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def paired_rate_reconstruction(
    fast: torch.Tensor,
    slow: torch.Tensor,
    broadband: torch.Tensor,
    labels: torch.Tensor,
    *,
    segments: int = 8,
    probability: float = 0.5,
    carrier_scale_range: Sequence[float] = (0.9, 1.1),
    noise_std: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recombine aligned fast/slow segments using same-class current-batch donors."""

    if probability <= 0.0 or torch.rand((), device=fast.device) >= probability:
        return fast, slow, broadband
    if fast.shape[-1] != 2 * slow.shape[-1]:
        raise ValueError("paired reconstruction requires exactly aligned 2:1 rates")
    mixed_fast = fast.clone()
    mixed_slow = slow.clone()
    mixed_broadband = broadband.clone()
    slow_boundaries = torch.linspace(0, slow.shape[-1], segments + 1, device=slow.device)
    slow_boundaries = slow_boundaries.round().long()
    for class_value in torch.unique(labels):
        indices = torch.where(labels == class_value)[0]
        if indices.numel() < 2:
            continue
        for segment in range(int(segments)):
            donors = indices[torch.randperm(indices.numel(), device=indices.device)]
            slow_start = int(slow_boundaries[segment])
            slow_stop = int(slow_boundaries[segment + 1])
            fast_start, fast_stop = 2 * slow_start, 2 * slow_stop
            mixed_fast[indices, ..., fast_start:fast_stop] = fast[
                donors, ..., fast_start:fast_stop
            ]
            mixed_slow[indices, ..., slow_start:slow_stop] = slow[
                donors, ..., slow_start:slow_stop
            ]
            mixed_broadband[indices, ..., fast_start:fast_stop] = broadband[
                donors, ..., fast_start:fast_stop
            ]
    low, high = (float(value) for value in carrier_scale_range)
    scale = torch.empty(
        (fast.shape[0], 1, 1, 1), device=fast.device, dtype=fast.real.dtype
    ).uniform_(low, high)
    if float(noise_std) > 0.0:
        mixed_broadband = mixed_broadband + torch.randn_like(mixed_broadband) * float(
            noise_std
        )
    return mixed_fast * scale, mixed_slow, mixed_broadband * scale.squeeze(1)


def _scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_updates: int,
    warmup_updates: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def scale(update: int) -> float:
        current = update + 1
        if current <= warmup_updates:
            return current / max(1, warmup_updates)
        progress = (current - warmup_updates) / max(1, total_updates - warmup_updates)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


@torch.no_grad()
def predict_scaffold(
    model: DASPSNNV62,
    rates: CachedRates,
    labels: np.ndarray,
    *,
    device: str,
    batch_size: int,
    delay_override: str | None = None,
    retain_diagnostic_tensors: bool = False,
) -> dict[str, Any]:
    model.eval()
    loader = _rates_loader(rates, labels, batch_size=batch_size, shuffle=False, seed=0)
    logits: list[np.ndarray] = []
    endpoint_logits: list[np.ndarray] = []
    observed_labels: list[np.ndarray] = []
    fused_square_sum = 0.0
    fused_elements = 0
    synthesized_square_sum = 0.0
    synthesized_elements = 0
    contrast_square_sum = 0.0
    contrast_elements = 0
    route_contrast_square_sum = 0.0
    route_contrast_elements = 0
    phase_pair_square_sum = 0.0
    phase_pair_elements = 0
    statistics_logit_square_sum = 0.0
    statistics_logit_elements = 0
    route_statistics_logit_square_sum = 0.0
    route_statistics_logit_elements = 0
    statistics_gate_sum = 0.0
    statistics_gate_elements = 0
    statistics_gate_min = math.inf
    statistics_gate_max = -math.inf
    broadband_delay_sources: set[str] = set()
    diagnostic_synthesized: list[np.ndarray] = []
    diagnostic_broadband: list[np.ndarray] = []
    for fast, slow, broadband, context, batch_y in loader:
        batch_context = context.to(device, non_blocking=True) if context.shape[1] else None
        output = model.forward_rate_features(
            fast.to(device, non_blocking=True),
            slow.to(device, non_blocking=True),
            broadband=broadband.to(device, non_blocking=True),
            context=batch_context,
            delay_override=delay_override,
        )
        if not bool(torch.isfinite(output["logits"]).all()):
            raise FloatingPointError("scaffold evaluation produced non-finite logits")
        logits.append(output["logits"].float().cpu().numpy())
        endpoint_logits.append(output["prefix_logits"].float().cpu().numpy())
        observed_labels.append(batch_y.numpy())
        transport = output["aux"]["transport"]
        broadband_delay_sources.add(str(transport.broadband_delay_source))
        fused_square_sum += float(transport.fused_current.float().square().sum().cpu())
        fused_elements += int(transport.fused_current.numel())
        synthesized = output["aux"].get("delayed_atc_synthesized_carrier")
        if synthesized is not None:
            synthesized_square_sum += float(synthesized.float().square().sum().cpu())
            synthesized_elements += int(synthesized.numel())
            if retain_diagnostic_tensors:
                diagnostic_synthesized.append(synthesized.float().cpu().numpy())
        if retain_diagnostic_tensors:
            broadband_current = transport.broadband_current
            if broadband_current is None:
                raise RuntimeError("paired E3 diagnostics require a broadband current")
            diagnostic_broadband.append(broadband_current.float().cpu().numpy())
        contrast = transport.slow_delay_contrast_current
        contrast_square_sum += float(contrast.float().square().sum().cpu())
        contrast_elements += int(contrast.numel())
        route_contrast = transport.slow_route_delay_contrast_current
        if route_contrast is not None:
            route_contrast_square_sum += float(
                route_contrast.float().square().sum().cpu()
            )
            route_contrast_elements += int(route_contrast.numel())
        phase_pair_current = output["aux"].get("delayed_phase_pair_current")
        if phase_pair_current is not None:
            phase_pair_square_sum += float(
                phase_pair_current.float().square().sum().cpu()
            )
            phase_pair_elements += int(phase_pair_current.numel())
        statistics_logits = output["aux"].get("delayed_atc_statistics_logits")
        if statistics_logits is not None:
            statistics_logit_square_sum += float(
                statistics_logits.float().square().sum().cpu()
            )
            statistics_logit_elements += int(statistics_logits.numel())
        route_statistics_logits = output["aux"].get(
            "delayed_atc_route_statistics_logits"
        )
        if route_statistics_logits is not None:
            route_statistics_logit_square_sum += float(
                route_statistics_logits.float().square().sum().cpu()
            )
            route_statistics_logit_elements += int(route_statistics_logits.numel())
        statistics_gate = output["aux"].get("delayed_atc_statistics_gate")
        if statistics_gate is not None:
            gate = statistics_gate.float()
            statistics_gate_sum += float(gate.sum().cpu())
            statistics_gate_elements += int(gate.numel())
            statistics_gate_min = min(statistics_gate_min, float(gate.min().cpu()))
            statistics_gate_max = max(statistics_gate_max, float(gate.max().cpu()))
    all_logits = np.concatenate(logits)
    all_endpoint_logits = np.concatenate(endpoint_logits)
    all_labels = np.concatenate(observed_labels)
    prediction = all_logits.argmax(axis=1)
    if len(broadband_delay_sources) != 1:
        raise RuntimeError("scaffold batches used inconsistent broadband delay routes")
    return {
        "logits": all_logits,
        "endpoint_logits": all_endpoint_logits,
        "labels": all_labels,
        "pred": prediction,
        "delay_override": delay_override,
        "broadband_delay_source": next(iter(broadband_delay_sources)),
        "diagnostic_synthesized_carrier": (
            np.concatenate(diagnostic_synthesized)
            if retain_diagnostic_tensors and diagnostic_synthesized
            else None
        ),
        "diagnostic_broadband_current": (
            np.concatenate(diagnostic_broadband)
            if retain_diagnostic_tensors and diagnostic_broadband
            else None
        ),
        "fused_current_rms": float(
            math.sqrt(fused_square_sum / max(1, fused_elements))
        ),
        "synthesized_carrier_rms": float(
            math.sqrt(synthesized_square_sum / max(1, synthesized_elements))
        ),
        "delay_contrast_rms": float(
            math.sqrt(contrast_square_sum / max(1, contrast_elements))
        ),
        "route_delay_contrast_rms": float(
            math.sqrt(route_contrast_square_sum / max(1, route_contrast_elements))
        ),
        "phase_pair_current_rms": float(
            math.sqrt(phase_pair_square_sum / max(1, phase_pair_elements))
        ),
        "delayed_statistics_logits_rms": float(
            math.sqrt(
                statistics_logit_square_sum / max(1, statistics_logit_elements)
            )
        ),
        "delayed_route_statistics_logits_rms": float(
            math.sqrt(
                route_statistics_logit_square_sum
                / max(1, route_statistics_logit_elements)
            )
        ),
        "delayed_statistics_gate_mean": (
            float(statistics_gate_sum / statistics_gate_elements)
            if statistics_gate_elements
            else None
        ),
        "delayed_statistics_gate_min": (
            float(statistics_gate_min) if statistics_gate_elements else None
        ),
        "delayed_statistics_gate_max": (
            float(statistics_gate_max) if statistics_gate_elements else None
        ),
        **classification_metrics(all_labels, prediction, n_classes=all_logits.shape[1]),
    }


def scaffold_optimizer_groups(
    model: DASPSNNV62,
    *,
    learning_rate: float,
    delay_learning_rate_multiplier: float = 1.0,
    atc_core_learning_rate_multiplier: float = 1.0,
) -> list[dict[str, Any]]:
    """Partition trainable parameters into heads, ATC core, and delay groups."""

    base_lr = float(learning_rate)
    delay_multiplier = float(delay_learning_rate_multiplier)
    atc_core_multiplier = float(atc_core_learning_rate_multiplier)
    if base_lr <= 0.0:
        raise ValueError("learning rate must be positive")
    if delay_multiplier <= 0.0:
        raise ValueError("delay learning-rate multiplier must be positive")
    if atc_core_multiplier <= 0.0:
        raise ValueError("ATC-core learning-rate multiplier must be positive")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    delay_parameter_ids = {
        id(parameter) for parameter in model.delay.parameters() if parameter.requires_grad
    }
    atc_core = (
        getattr(model.atc_readout, "core", None)
        if model.atc_readout is not None
        else None
    )
    atc_core_parameter_ids = {
        id(parameter)
        for parameter in (() if atc_core is None else atc_core.parameters())
        if parameter.requires_grad
    }
    if delay_parameter_ids & atc_core_parameter_ids:
        raise RuntimeError("delay and ATC-core optimizer groups overlap")

    groups: list[dict[str, Any]] = []
    partitions = (
        (
            "base",
            [
                parameter
                for parameter in trainable
                if id(parameter) not in delay_parameter_ids
                and id(parameter) not in atc_core_parameter_ids
            ],
            base_lr,
        ),
        (
            "atc_core",
            [
                parameter
                for parameter in trainable
                if id(parameter) in atc_core_parameter_ids
            ],
            base_lr * atc_core_multiplier,
        ),
        (
            "delay",
            [
                parameter
                for parameter in trainable
                if id(parameter) in delay_parameter_ids
            ],
            base_lr * delay_multiplier,
        ),
    )
    assigned: set[int] = set()
    for name, parameters, group_lr in partitions:
        if not parameters:
            continue
        parameter_ids = {id(parameter) for parameter in parameters}
        if assigned & parameter_ids:
            raise RuntimeError("optimizer parameter groups overlap")
        assigned.update(parameter_ids)
        groups.append({"params": parameters, "lr": group_lr, "group_name": name})
    if assigned != {id(parameter) for parameter in trainable}:
        raise RuntimeError("optimizer parameter partition is incomplete")
    if not groups:
        raise RuntimeError("scaffold has no trainable parameters")
    return groups


def fit_zero_scaffold(
    model: DASPSNNV62,
    *,
    train_rates: CachedRates,
    y_train: np.ndarray,
    validation_rates: CachedRates | None,
    y_validation: np.ndarray | None,
    device: str,
    seed: int,
    epochs: int,
    patience: int,
    minimum_epochs: int,
    batch_size: int,
    accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    auxiliary_weights: Sequence[float],
    delay_learning_rate_multiplier: float = 1.0,
    atc_core_learning_rate_multiplier: float = 1.0,
    matched_zero_kl_weight: float = 0.0,
    matched_zero_kl_teacher_correct_weight: float | None = None,
    matched_zero_kl_teacher_incorrect_weight: float | None = None,
    matched_zero_kl_temperature: float = 1.0,
    beta1: float = 0.9,
    scheduler_warmup_epochs: int = 10,
    use_scheduler: bool = True,
    scheduler_step_unit: str = "update",
    reseed_before_training: bool = True,
    validation_interval: int = 1,
    augmentation: dict[str, Any] | None = None,
    fixed_epoch: int | None = None,
    run_label: str = "",
) -> ScaffoldFitResult:
    """Train a scaffold using only cached pre-delay rates and Session-T selection."""

    if reseed_before_training:
        seed_scaffold(seed)
    model.to(device)
    project_registered_max_norm_constraints_(model)
    actual_epochs = int(fixed_epoch if fixed_epoch is not None else epochs)
    scheduler_epochs = int(epochs)
    if scheduler_epochs < actual_epochs:
        raise ValueError("scheduler horizon must not precede the training cutoff")
    if train_rates.frontend_fingerprint != model.frontend_fingerprint():
        raise RuntimeError("training cache does not belong to the active frozen front end")
    model.set_training_gain(train_rates.gain)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    zero_kl_weight = float(matched_zero_kl_weight)
    adaptive_zero_kl = (
        matched_zero_kl_teacher_correct_weight is not None
        or matched_zero_kl_teacher_incorrect_weight is not None
    )
    if adaptive_zero_kl and (
        matched_zero_kl_teacher_correct_weight is None
        or matched_zero_kl_teacher_incorrect_weight is None
    ):
        raise ValueError("both matched-zero teacher-conditional weights are required")
    teacher_correct_weight = float(
        zero_kl_weight
        if matched_zero_kl_teacher_correct_weight is None
        else matched_zero_kl_teacher_correct_weight
    )
    teacher_incorrect_weight = float(
        zero_kl_weight
        if matched_zero_kl_teacher_incorrect_weight is None
        else matched_zero_kl_teacher_incorrect_weight
    )
    zero_kl_temperature = float(matched_zero_kl_temperature)
    if zero_kl_weight < 0.0:
        raise ValueError("matched-zero KL weight must be non-negative")
    if teacher_correct_weight < 0.0 or teacher_incorrect_weight < 0.0:
        raise ValueError("matched-zero teacher-conditional weights must be non-negative")
    if zero_kl_temperature <= 0.0:
        raise ValueError("matched-zero KL temperature must be positive")
    optimizer_groups = scaffold_optimizer_groups(
        model,
        learning_rate=learning_rate,
        delay_learning_rate_multiplier=delay_learning_rate_multiplier,
        atc_core_learning_rate_multiplier=atc_core_learning_rate_multiplier,
    )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=float(learning_rate),
        betas=(float(beta1), 0.999),
        weight_decay=float(weight_decay),
    )
    loader = _rates_loader(
        train_rates,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    updates_per_epoch = math.ceil(len(loader) / int(accumulation_steps))
    if scheduler_step_unit not in {"update", "epoch"}:
        raise ValueError("scheduler_step_unit must be 'update' or 'epoch'")
    if int(validation_interval) < 1:
        raise ValueError("validation_interval must be positive")
    if use_scheduler:
        if scheduler_step_unit == "update":
            scheduler = _scheduler(
                optimizer,
                total_updates=scheduler_epochs * updates_per_epoch,
                warmup_updates=min(
                    int(scheduler_warmup_epochs) * updates_per_epoch,
                    max(1, scheduler_epochs * updates_per_epoch // 5),
                ),
            )
        else:
            scheduler = _scheduler(
                optimizer,
                total_updates=scheduler_epochs,
                warmup_updates=min(
                    int(scheduler_warmup_epochs),
                    max(1, scheduler_epochs // 5),
                ),
            )
    else:
        scheduler = None
    criterion = nn.CrossEntropyLoss()
    auxiliary = tuple(float(value) for value in auxiliary_weights)
    if len(auxiliary) != len(model.decoder.endpoint_samples) - 1:
        raise ValueError("auxiliary weights must cover every non-final causal endpoint")
    history: list[dict[str, float | int]] = []
    best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    best_metric = -float("inf")
    best_epoch = 0
    stale = 0
    optimizer_steps = 0
    started = time.perf_counter()
    augmentation = augmentation or {}
    if validation_rates is not None and y_validation is not None:
        initial_evaluation = predict_scaffold(
            model,
            validation_rates,
            y_validation,
            device=device,
            batch_size=max(batch_size, 16),
        )
        base_lr = float(
            next(
                (
                    group["lr"]
                    for group in optimizer.param_groups
                    if group.get("group_name") == "base"
                ),
                optimizer.param_groups[0]["lr"],
            )
        )
        delay_lr = float(
            next(
                (
                    group["lr"]
                    for group in optimizer.param_groups
                    if group.get("group_name") == "delay"
                ),
                base_lr,
            )
        )
        atc_core_lr = float(
            next(
                (
                    group["lr"]
                    for group in optimizer.param_groups
                    if group.get("group_name") == "atc_core"
                ),
                base_lr,
            )
        )
        history.append(
            {
                "epoch": 0,
                "loss": float("nan"),
                "lr": base_lr,
                "atc_core_lr": atc_core_lr,
                "delay_lr": delay_lr,
                "optimizer_steps": 0,
                "val_accuracy": float(initial_evaluation["accuracy"]),
                "val_balanced_accuracy": float(
                    initial_evaluation["balanced_accuracy"]
                ),
                "val_kappa": float(initial_evaluation["kappa"]),
                "val_macro_f1": float(initial_evaluation["macro_f1"]),
            }
        )
        best_metric = float(initial_evaluation["kappa"])
    for epoch in range(1, actual_epochs + 1):
        model.train()
        for frozen_module in getattr(model, "_frozen_eval_modules", ()):
            frozen_module.eval()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        ce_losses: list[float] = []
        zero_kl_losses: list[float] = []
        zero_kl_penalties: list[float] = []
        for batch_index, (fast, slow, broadband, context, batch_y) in enumerate(loader):
            fast = fast.to(device, non_blocking=True)
            slow = slow.to(device, non_blocking=True)
            broadband = broadband.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_context = context.to(device, non_blocking=True) if context.shape[1] else None
            if augmentation.get("enabled", False):
                fast, slow, broadband = paired_rate_reconstruction(
                    fast,
                    slow,
                    broadband,
                    batch_y,
                    segments=int(augmentation.get("segments", 8)),
                    probability=float(augmentation.get("probability", 0.5)),
                    carrier_scale_range=augmentation.get(
                        "carrier_scale_range", (0.9, 1.1)
                    ),
                    noise_std=float(augmentation.get("noise_std", 0.0)),
                )
            output = model.forward_rate_features(
                fast,
                slow,
                broadband=broadband,
                context=batch_context,
            )
            classification_loss = criterion(output["logits"], batch_y)
            loss = classification_loss
            matched_zero_kl = output["logits"].new_zeros(())
            matched_zero_penalty = output["logits"].new_zeros(())
            if max(teacher_correct_weight, teacher_incorrect_weight) > 0.0:
                with torch.no_grad():
                    zero_logits = model.forward_rate_features(
                        fast,
                        slow,
                        broadband=broadband,
                        context=batch_context,
                        delay_override="zero",
                    )["logits"]
                per_trial_zero_kl = F.kl_div(
                    F.log_softmax(output["logits"] / zero_kl_temperature, dim=1),
                    F.softmax(zero_logits / zero_kl_temperature, dim=1),
                    reduction="none",
                ).sum(dim=1) * (zero_kl_temperature**2)
                matched_zero_kl = per_trial_zero_kl.mean()
                teacher_correct = zero_logits.argmax(dim=1).eq(batch_y)
                per_trial_weight = torch.where(
                    teacher_correct,
                    per_trial_zero_kl.new_full((), teacher_correct_weight),
                    per_trial_zero_kl.new_full((), teacher_incorrect_weight),
                )
                matched_zero_penalty = (per_trial_weight * per_trial_zero_kl).mean()
                loss = loss + matched_zero_penalty
            for endpoint, weight in enumerate(auxiliary):
                loss = loss + weight * criterion(
                    output["prefix_logits"][:, endpoint], batch_y
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite scaffold loss at epoch {epoch}")
            chunk_start = (batch_index // accumulation_steps) * accumulation_steps
            chunk_size = min(accumulation_steps, len(loader) - chunk_start)
            (loss / chunk_size).backward()
            losses.append(float(loss.detach().cpu()))
            ce_losses.append(float(classification_loss.detach().cpu()))
            zero_kl_losses.append(float(matched_zero_kl.detach().cpu()))
            zero_kl_penalties.append(float(matched_zero_penalty.detach().cpu()))
            end_of_chunk = (batch_index + 1) % accumulation_steps == 0
            if end_of_chunk or batch_index + 1 == len(loader):
                nn.utils.clip_grad_norm_(trainable, max_norm=5.0, error_if_nonfinite=True)
                optimizer.step()
                project_registered_max_norm_constraints_(model)
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and scheduler_step_unit == "update":
                    scheduler.step()
                optimizer_steps += 1
        if scheduler is not None and scheduler_step_unit == "epoch":
            scheduler.step()
        row: dict[str, float | int] = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "classification_loss": float(np.mean(ce_losses)),
            "matched_zero_kl": float(np.mean(zero_kl_losses)),
            "matched_zero_kl_penalty": float(np.mean(zero_kl_penalties)),
            "matched_zero_kl_weight": zero_kl_weight,
            "matched_zero_kl_teacher_correct_weight": teacher_correct_weight,
            "matched_zero_kl_teacher_incorrect_weight": teacher_incorrect_weight,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "optimizer_steps": optimizer_steps,
        }
        delay_group = next(
            (
                group
                for group in optimizer.param_groups
                if group.get("group_name") == "delay"
            ),
            None,
        )
        row["delay_lr"] = float(
            row["lr"] if delay_group is None else delay_group["lr"]
        )
        atc_core_group = next(
            (
                group
                for group in optimizer.param_groups
                if group.get("group_name") == "atc_core"
            ),
            None,
        )
        row["atc_core_lr"] = float(
            row["lr"] if atc_core_group is None else atc_core_group["lr"]
        )
        should_validate = validation_rates is not None and y_validation is not None and (
            epoch % int(validation_interval) == 0 or epoch == actual_epochs
        )
        if should_validate:
            evaluation = predict_scaffold(
                model,
                validation_rates,
                y_validation,
                device=device,
                batch_size=max(batch_size, 16),
            )
            row.update(
                val_accuracy=float(evaluation["accuracy"]),
                val_balanced_accuracy=float(evaluation["balanced_accuracy"]),
                val_kappa=float(evaluation["kappa"]),
                val_macro_f1=float(evaluation["macro_f1"]),
            )
            metric = float(evaluation["kappa"])
        elif validation_rates is None or y_validation is None:
            metric = -float(row["loss"])
        else:
            metric = None
        history.append(row)
        print(
            "__V62_SCAFFOLD_PROGRESS__ "
            f"run={run_label or 'zero_ann'} epoch={epoch}/{actual_epochs} "
            f"loss={float(row['loss']):.6f} "
            f"val_kappa={float(row.get('val_kappa', float('nan'))):.6f}",
            flush=True,
        )
        if metric is not None and metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
            stale = 0
        elif metric is not None:
            stale += int(validation_interval) if should_validate else 1
        if validation_rates is not None and epoch >= minimum_epochs and stale >= patience:
            break
    last_state = deepcopy(
        {key: value.detach().cpu() for key, value in model.state_dict().items()}
    )
    model.load_state_dict(best_state if validation_rates is not None else last_state)
    return ScaffoldFitResult(
        model=model,
        history=history,
        best_epoch=best_epoch if validation_rates is not None else actual_epochs,
        best_metric=best_metric,
        best_state=best_state if validation_rates is not None else last_state,
        last_state=last_state,
        optimizer_steps=optimizer_steps,
        elapsed_seconds=time.perf_counter() - started,
    )


def fit_official_atc_scaffold_parity(
    model: DASPSNNV62,
    *,
    source_root: str,
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    channel_gain: FixedGain,
    x_validation_raw: np.ndarray | None,
    y_validation: np.ndarray | None,
    sfreq: float,
    epoch_tmin: float,
    device: str,
    seed: int,
    epochs: int,
    patience: int,
    augmentation: dict[str, Any],
    fixed_epoch: int | None = None,
    run_label: str = "",
) -> ScaffoldFitResult:
    """Train the exact E1 ATC core, then embed it in the delay scaffold.

    Gate A is a representation-parity test. Reimplementing the optimizer loop
    inside the scaffold would confound it with RNG ordering and tiny sensor
    reconstruction differences, so this path deliberately calls the canonical
    E1 fitter and transfers its byte-identical core state.
    """

    if model.atc_readout is None or not hasattr(model.atc_readout, "core"):
        raise ValueError("official ATC parity requires the locked full-window adapter")
    if not model.force_zero_delay:
        raise ValueError("official ATC parity is only valid for the zero-delay stage")
    if abs(float(model.atc_readout.delayed_residual_scale.detach())) > 1e-12:
        raise ValueError("official ATC parity requires a zero multiband residual")
    x_train = apply_fixed_gain(
        task_carrier(x_train_raw, sfreq=sfreq, epoch_tmin=epoch_tmin),
        channel_gain,
    )
    x_validation = (
        None
        if x_validation_raw is None
        else apply_fixed_gain(
            task_carrier(x_validation_raw, sfreq=sfreq, epoch_tmin=epoch_tmin),
            channel_gain,
        )
    )
    baseline_augmentation = {
        "enabled": bool(augmentation.get("enabled", False)),
        "segments": int(augmentation.get("segments", 8)),
        "probability": float(augmentation.get("probability", 0.5)),
        "noise_std": float(augmentation.get("noise_std", 0.01)),
        "scale_range": tuple(
            float(value)
            for value in augmentation.get("carrier_scale_range", (0.9, 1.1))
        ),
    }
    fitted = fit_baseline(
        "atcnet",
        source_root=source_root,
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        device=device,
        seed=seed,
        epochs=epochs,
        patience=patience,
        augmentation=baseline_augmentation,
        fixed_epoch=fixed_epoch,
        run_label=run_label,
    )
    model.cpu()
    core = model.atc_readout.core
    core.load_state_dict(fitted.best_state, strict=True)
    best_state = deepcopy(
        {key: value.detach().cpu() for key, value in model.state_dict().items()}
    )
    core.load_state_dict(fitted.last_state, strict=True)
    last_state = deepcopy(
        {key: value.detach().cpu() for key, value in model.state_dict().items()}
    )
    core.load_state_dict(fitted.best_state, strict=True)
    model.to(device)
    return ScaffoldFitResult(
        model=model,
        history=fitted.history,
        best_epoch=fitted.best_epoch,
        best_metric=fitted.best_metric,
        best_state=best_state,
        last_state=last_state,
        optimizer_steps=fitted.optimizer_steps,
        elapsed_seconds=fitted.elapsed_seconds,
    )
