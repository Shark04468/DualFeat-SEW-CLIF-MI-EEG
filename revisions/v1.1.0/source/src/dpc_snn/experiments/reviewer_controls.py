"""Reviewer-requested decoder controls on the frozen V31 feature contract.

This module contains only model, optimisation, and audit primitives. Dataset
access and sealed train/evaluation barriers remain the responsibility of the
campaign runner.
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dpc_snn.models.lif import CLIFLayer
from dpc_snn.models.v9_dual_feature_student import (
    V9DualFeatureStudent,
    build_v9_dual_feature_student,
)
from dpc_snn.models.v62_snn_decoder import HeterogeneousCLIF
from dpc_snn.utils.metrics import classification_metrics

ReviewerVariant = Literal[
    "soft_clif_exact",
    "gru_stat",
    "tcn_stat",
    "ann_sew",
    "sew_clif",
]
ControlKind = Literal["gru_stat", "tcn_stat"]
FIXED_SOFT_GATE_SLOPE = 10.0


class ExactGradientSoftCLIFLayer(CLIFLayer):
    """CLIF with a continuous sigmoid gate and its exact autograd derivative."""

    def __init__(self, *args: Any, soft_gate_slope: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.soft_gate_slope = float(soft_gate_slope)
        if self.soft_gate_slope <= 0.0:
            raise ValueError("Soft-CLIF gate slope must be positive")

    def forward(self, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if current.ndim != 3 and self.current_norm is not None:
            raise ValueError("Causal channel normalization expects current [N, C, T]")
        membrane = torch.zeros_like(current[..., 0])
        complement = torch.zeros_like(membrane)
        gates: list[torch.Tensor] = []
        membranes: list[torch.Tensor] = []
        for step in range(current.shape[-1]):
            decay = self._decay_for(membrane)
            step_current = current[..., step]
            if self.current_norm is not None:
                step_current = self.current_norm(step_current)
            membrane = decay * membrane + step_current
            # Do not wrap this sigmoid in a custom autograd Function. The
            # analytic derivative k*sigma(kx)*(1-sigma(kx)) is the control.
            gate = torch.sigmoid(self.soft_gate_slope * (membrane - self.threshold))
            complement = complement * torch.sigmoid((1.0 - decay) * membrane) + gate
            membrane = membrane - gate * (self.threshold + torch.sigmoid(complement))
            gates.append(gate)
            membranes.append(membrane)
        return torch.stack(gates, dim=-1), torch.stack(membranes, dim=-1)


def build_exact_gradient_soft_clif(
    *,
    n_classes: int,
    dropout: float = 0.25,
    slope: float = FIXED_SOFT_GATE_SLOPE,
) -> V9DualFeatureStudent:
    """Build the frozen hard architecture and replace only its gate semantics."""

    model = build_v9_dual_feature_student(
        "sew_clif",
        n_classes=int(n_classes),
        hidden_channels=64,
        decoder_layers=2,
        readout_features=96,
        dropout=float(dropout),
    )
    wrappers = [module for module in model.modules() if isinstance(module, HeterogeneousCLIF)]
    if len(wrappers) != 4:
        raise RuntimeError(f"expected four CLIF state sites, found {len(wrappers)}")
    for wrapper in wrappers:
        hard = wrapper.layer
        soft = ExactGradientSoftCLIFLayer(
            decay=0.9,
            threshold=hard.threshold,
            channels=hard.channels,
            learnable_decay=isinstance(hard.decay_raw, nn.Parameter),
            membrane_norm_groups=0,
            causal_channel_norm=hard.causal_channel_norm,
            soft_gate_slope=float(slope),
        )
        soft.load_state_dict(hard.state_dict(), strict=True)
        wrapper.layer = soft
    return model


def temporal_statistics(activity: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Return the exact C5 mean/population-std statistics for two state streams."""

    if activity.ndim != 3 or state.shape != activity.shape:
        raise ValueError("activity and state must share shape [N, C, T]")
    return torch.cat(
        (
            activity.mean(dim=-1),
            activity.std(dim=-1, unbiased=False),
            state.mean(dim=-1),
            state.std(dim=-1, unbiased=False),
        ),
        dim=1,
    )


class SharedStatisticalHead(nn.Module):
    """The registered 256 -> 96 -> K statistical readout used by C5."""

    def __init__(self, channels: int, n_classes: int, dropout: float) -> None:
        super().__init__()
        self.channels = int(channels)
        self.readout = nn.Sequential(
            nn.Linear(4 * self.channels, 96, bias=False),
            nn.LayerNorm(96, elementwise_affine=False),
            nn.ELU(),
            nn.Dropout(float(dropout)),
        )
        self.classifier = nn.Linear(96, int(n_classes))

    def forward(self, activity: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.readout(temporal_statistics(activity, state)))


class CausalResidualConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = int(kernel_size) - 1
        self.conv = nn.Conv1d(int(channels), int(channels), int(kernel_size), bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        update = self.conv(F.pad(sequence, (self.left_padding, 0)))
        return sequence + self.dropout(self.activation(update))


@dataclass(frozen=True)
class ReviewerOutput:
    logits: torch.Tensor
    fused_sequence: torch.Tensor
    final_activity: torch.Tensor
    final_state: torch.Tensor
    binary_spikes: tuple[torch.Tensor, ...] = ()


class ReadoutMatchedContinuousControl(nn.Module):
    """GRU/TCN comparator with the exact fusion and C5 statistical head."""

    architecture_version = "dpc_snn_reviewer_stat_control_r1"

    def __init__(
        self,
        kind: ControlKind,
        *,
        n_classes: int,
        temporal_width: int,
        hidden_channels: int = 64,
        dropout: float = 0.25,
        tcn_blocks: int = 3,
        tcn_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if kind not in ("gru_stat", "tcn_stat"):
            raise ValueError(f"unknown statistical comparator: {kind}")
        if temporal_width < 2:
            raise ValueError("temporal width must be at least two")
        self.kind: ControlKind = kind
        self.n_classes = int(n_classes)
        self.hidden_channels = int(hidden_channels)
        self.temporal_width = int(temporal_width)
        self.atc_projection = nn.Linear(32, self.hidden_channels, bias=False)
        self.fbc_projection = nn.Linear(288, self.hidden_channels, bias=False)
        self.interaction = nn.Sequential(
            nn.LayerNorm(4 * self.hidden_channels),
            nn.Linear(4 * self.hidden_channels, self.hidden_channels, bias=False),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.fusion_norm = nn.LayerNorm(self.hidden_channels)
        if self.kind == "gru_stat":
            self.input_projection: nn.Module = nn.Identity()
            self.temporal: nn.Module = nn.GRU(
                self.hidden_channels, self.temporal_width, num_layers=1, batch_first=True
            )
        else:
            self.input_projection = nn.Linear(self.hidden_channels, self.temporal_width, bias=False)
            self.temporal = nn.Sequential(
                *[
                    CausalResidualConv(self.temporal_width, tcn_kernel_size, dropout)
                    for _ in range(int(tcn_blocks))
                ]
            )
        self.output_projection = nn.Linear(self.temporal_width, self.hidden_channels, bias=False)
        self.head = SharedStatisticalHead(self.hidden_channels, self.n_classes, dropout)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def fuse(self, atc_sequence: torch.Tensor, fbc_sequence: torch.Tensor) -> torch.Tensor:
        if atc_sequence.ndim != 3 or atc_sequence.shape[1:] != (18, 32):
            raise ValueError("ATC feature shape must be [N, 18, 32]")
        if fbc_sequence.ndim != 3 or fbc_sequence.shape[1:] != (4, 288):
            raise ValueError("FBC feature shape must be [N, 4, 288]")
        if atc_sequence.shape[0] != fbc_sequence.shape[0]:
            raise ValueError("ATC and FBC batches are not aligned")
        atc = self.atc_projection(atc_sequence)
        fbc = self.fbc_projection(fbc_sequence)
        fbc = F.interpolate(
            fbc.transpose(1, 2), size=18, mode="linear", align_corners=False
        ).transpose(1, 2)
        interaction = torch.cat((atc, fbc, atc * fbc, (atc - fbc).abs()), dim=-1)
        return self.fusion_norm(0.5 * (atc + fbc) + self.interaction(interaction))

    def forward(self, atc_sequence: torch.Tensor, fbc_sequence: torch.Tensor) -> ReviewerOutput:
        fused = self.fuse(atc_sequence, fbc_sequence)
        if self.kind == "gru_stat":
            sequence, _ = self.temporal(fused)
        else:
            encoded = self.input_projection(fused).transpose(1, 2)
            sequence = self.temporal(encoded).transpose(1, 2)
        state = self.output_projection(sequence).transpose(1, 2)
        activity = torch.tanh(state)
        return ReviewerOutput(
            logits=self.head(activity, state),
            fused_sequence=fused,
            final_activity=activity,
            final_state=state,
        )


@dataclass(frozen=True)
class WidthSelection:
    kind: ControlKind
    temporal_width: int
    parameters: int
    target_parameters: int
    relative_error: float


def solve_temporal_width(
    kind: ControlKind,
    *,
    n_classes: int,
    target_parameters: int,
    minimum: int = 8,
    maximum: int = 192,
) -> WidthSelection:
    """Deterministically select the closest integer width before evaluation access."""

    candidates: list[WidthSelection] = []
    for width in range(int(minimum), int(maximum) + 1):
        model = ReadoutMatchedContinuousControl(
            kind, n_classes=int(n_classes), temporal_width=width
        )
        count = model.parameter_count
        candidates.append(
            WidthSelection(
                kind=kind,
                temporal_width=width,
                parameters=count,
                target_parameters=int(target_parameters),
                relative_error=abs(count - int(target_parameters)) / int(target_parameters),
            )
        )
    return min(candidates, key=lambda value: (value.relative_error, value.temporal_width))


def build_reviewer_model(
    variant: ReviewerVariant,
    *,
    n_classes: int,
    dropout: float = 0.25,
    soft_gate_slope: float = FIXED_SOFT_GATE_SLOPE,
    temporal_width: int | None = None,
) -> nn.Module:
    if variant == "soft_clif_exact":
        return build_exact_gradient_soft_clif(
            n_classes=n_classes, dropout=dropout, slope=soft_gate_slope
        )
    if variant in ("ann_sew", "sew_clif"):
        return build_v9_dual_feature_student(
            variant,
            n_classes=int(n_classes),
            hidden_channels=64,
            decoder_layers=2,
            readout_features=96,
            dropout=float(dropout),
        )
    if variant in ("gru_stat", "tcn_stat"):
        if temporal_width is None:
            raise ValueError(f"{variant} requires a frozen temporal_width")
        return ReadoutMatchedContinuousControl(
            variant,
            n_classes=int(n_classes),
            temporal_width=int(temporal_width),
            dropout=float(dropout),
        )
    raise KeyError(f"unknown reviewer variant: {variant}")


@dataclass(frozen=True)
class EqualUpdatePlan:
    full_samples: int
    subset_samples: int
    batch_size: int
    epochs: int
    batch_sizes_per_epoch: tuple[int, ...]

    @property
    def optimizer_steps(self) -> int:
        return len(self.batch_sizes_per_epoch) * self.epochs

    @property
    def sample_exposures(self) -> int:
        return sum(self.batch_sizes_per_epoch) * self.epochs


def make_equal_update_plan(
    *, full_samples: int, subset_samples: int, batch_size: int, epochs: int
) -> EqualUpdatePlan:
    if min(full_samples, subset_samples, batch_size, epochs) < 1:
        raise ValueError("equal-update plan values must be positive")
    full_batches = [batch_size] * (full_samples // batch_size)
    if full_samples % batch_size:
        full_batches.append(full_samples % batch_size)
    return EqualUpdatePlan(
        full_samples=int(full_samples),
        subset_samples=int(subset_samples),
        batch_size=int(batch_size),
        epochs=int(epochs),
        batch_sizes_per_epoch=tuple(int(value) for value in full_batches),
    )


def equal_update_epoch_indices(plan: EqualUpdatePlan, *, seed: int, epoch: int) -> list[np.ndarray]:
    """Draw deterministic cyclic permutations with full-label exposure per epoch."""

    generator = np.random.default_rng(int(seed) + int(epoch) * 1_000_003)
    pool = np.empty(0, dtype=np.int64)
    batches: list[np.ndarray] = []
    for size in plan.batch_sizes_per_epoch:
        while pool.size < size:
            pool = np.concatenate((pool, generator.permutation(plan.subset_samples)))
        batches.append(np.ascontiguousarray(pool[:size], dtype=np.int64))
        pool = pool[size:]
    return batches


@dataclass
class ReviewerFit:
    model: nn.Module
    history: list[dict[str, float | int]]
    last_state: dict[str, torch.Tensor]
    optimizer_steps: int
    sample_exposures: int
    elapsed_seconds: float


def _seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _validate_arrays(
    atc: np.ndarray, fbc: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.ascontiguousarray(atc, dtype=np.float32)
    f = np.ascontiguousarray(fbc, dtype=np.float32)
    y = np.ascontiguousarray(labels, dtype=np.int64)
    if a.ndim != 3 or a.shape[1:] != (18, 32):
        raise ValueError("ATC features must have shape [N, 18, 32]")
    if f.ndim != 3 or f.shape[1:] != (4, 288) or f.shape[0] != a.shape[0]:
        raise ValueError("FBC features must have shape [N, 4, 288] and align with ATC")
    if y.shape != (a.shape[0],) or np.any(y < 0):
        raise ValueError("labels do not align with frozen features")
    if not np.isfinite(a).all() or not np.isfinite(f).all():
        raise FloatingPointError("frozen features contain non-finite values")
    return a, f, y


def fit_reviewer_model(
    variant: ReviewerVariant,
    *,
    atc_train: np.ndarray,
    fbc_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    device: str,
    seed: int,
    fixed_epochs: int,
    batch_size: int = 48,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    gradient_clip_norm: float = 5.0,
    dropout: float = 0.25,
    soft_gate_slope: float = FIXED_SOFT_GATE_SLOPE,
    temporal_width: int | None = None,
    equal_update_full_samples: int | None = None,
    firing_rate_weight: float = 0.0,
) -> ReviewerFit:
    """Fit a reviewer control with CE only and an auditable exposure ledger."""

    atc, fbc, labels = _validate_arrays(atc_train, fbc_train, y_train)
    if firing_rate_weight < 0.0:
        raise ValueError("firing_rate_weight must be non-negative")
    _seed(seed)
    model = build_reviewer_model(
        variant,
        n_classes=n_classes,
        dropout=dropout,
        soft_gate_slope=soft_gate_slope,
        temporal_width=temporal_width,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(fixed_epochs), eta_min=float(learning_rate) * 0.05
    )
    if equal_update_full_samples is None:
        plan = make_equal_update_plan(
            full_samples=atc.shape[0],
            subset_samples=atc.shape[0],
            batch_size=batch_size,
            epochs=fixed_epochs,
        )
    else:
        plan = make_equal_update_plan(
            full_samples=int(equal_update_full_samples),
            subset_samples=atc.shape[0],
            batch_size=batch_size,
            epochs=fixed_epochs,
        )
    history: list[dict[str, float | int]] = []
    optimizer_steps = 0
    sample_exposures = 0
    started = time.perf_counter()
    for epoch in range(1, int(fixed_epochs) + 1):
        model.train()
        total_loss = 0.0
        total_cross_entropy = 0.0
        total_firing_rate_loss = 0.0
        total_examples = 0
        last_gradient_norm = float("nan")
        for indices in equal_update_epoch_indices(plan, seed=seed, epoch=epoch):
            batch_atc = torch.from_numpy(atc[indices]).to(device, non_blocking=True)
            batch_fbc = torch.from_numpy(fbc[indices]).to(device, non_blocking=True)
            batch_y = torch.from_numpy(labels[indices]).to(device, non_blocking=True)
            output = model(batch_atc, batch_fbc)
            cross_entropy = F.cross_entropy(output.logits, batch_y)
            firing_rate_loss = getattr(output, "firing_rate_loss", cross_entropy.new_zeros(()))
            loss = cross_entropy + float(firing_rate_weight) * firing_rate_loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"{variant} CE loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(gradient_clip_norm)
            )
            last_gradient_norm = float(gradient_norm)
            optimizer.step()
            examples = int(indices.size)
            total_loss += float(loss.detach()) * examples
            total_cross_entropy += float(cross_entropy.detach()) * examples
            total_firing_rate_loss += float(firing_rate_loss.detach()) * examples
            total_examples += examples
            optimizer_steps += 1
            sample_exposures += examples
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / max(total_examples, 1),
                "cross_entropy": total_cross_entropy / max(total_examples, 1),
                "mean_firing_rate_penalty": total_firing_rate_loss / max(total_examples, 1),
                "firing_rate_weight": float(firing_rate_weight),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "last_batch_preclip_gradient_norm": last_gradient_norm,
                "optimizer_steps_cumulative": optimizer_steps,
                "sample_exposures_cumulative": sample_exposures,
            }
        )
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if optimizer_steps != plan.optimizer_steps or sample_exposures != plan.sample_exposures:
        raise RuntimeError("equal-update ledger differs from the frozen plan")
    return ReviewerFit(
        model=model,
        history=history,
        last_state=state,
        optimizer_steps=optimizer_steps,
        sample_exposures=sample_exposures,
        elapsed_seconds=time.perf_counter() - started,
    )


@torch.no_grad()
def predict_reviewer_model(
    model: nn.Module,
    atc: np.ndarray,
    fbc: np.ndarray,
    labels: np.ndarray,
    *,
    n_classes: int,
    device: str,
    batch_size: int = 48,
) -> dict[str, Any]:
    a, f, y = _validate_arrays(atc, fbc, labels)
    model.eval().to(device)
    logits: list[np.ndarray] = []
    binary: list[tuple[torch.Tensor, ...]] = []
    continuous_activities: list[float] = []
    exact_soft = any(isinstance(module, ExactGradientSoftCLIFLayer) for module in model.modules())
    for start in range(0, a.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), a.shape[0])
        output = model(
            torch.from_numpy(a[start:stop]).to(device),
            torch.from_numpy(f[start:stop]).to(device),
        )
        logits.append(output.logits.float().cpu().numpy())
        state_sites = getattr(output, "binary_spikes", ())
        if state_sites and exact_soft:
            continuous_activities.extend(
                float(value.detach().mean().cpu()) for value in state_sites
            )
        elif state_sites:
            binary.append(tuple(value.detach().cpu() for value in state_sites))
    all_logits = np.concatenate(logits)
    metrics = classification_metrics(y, all_logits.argmax(axis=1), n_classes=n_classes)
    return {
        **metrics,
        "logits": all_logits,
        "labels": y,
        "spike_summary": summarize_binary_spikes(binary),
        "mean_continuous_activity": (
            float(np.mean(continuous_activities)) if continuous_activities else None
        ),
    }


def summarize_binary_spikes(
    batches: Sequence[Sequence[torch.Tensor]],
) -> dict[str, Any]:
    if not batches:
        return {
            "binary": False,
            "layer_rates": [],
            "layer_sparsities": [],
            "layer_events": [],
            "layer_elements": [],
            "events": 0,
            "elements": 0,
        }
    layer_count = len(batches[0])
    if any(len(batch) != layer_count for batch in batches):
        raise ValueError("binary spike batches have inconsistent layer counts")
    rates: list[float] = []
    layer_events: list[int] = []
    layer_elements: list[int] = []
    for layer in range(layer_count):
        values = torch.cat([batch[layer].reshape(-1) for batch in batches])
        if not bool(torch.logical_or(values == 0, values == 1).all()):
            raise ValueError("hard-CLIF spike tensor contains non-binary values")
        count = int(values.numel())
        event_count = int(values.sum())
        rates.append(float(event_count / count))
        layer_events.append(event_count)
        layer_elements.append(count)
    return {
        "binary": True,
        "layer_rates": rates,
        "layer_sparsities": [1.0 - value for value in rates],
        "layer_events": layer_events,
        "layer_elements": layer_elements,
        "events": int(sum(layer_events)),
        "elements": int(sum(layer_elements)),
    }


def event_accumulation_proxy(
    spike_summary: dict[str, Any],
    *,
    eligible_fanouts: Sequence[int] = (3, 1, 1, 2),
) -> dict[str, Any]:
    """Return a conservative, explicitly scoped event-triggered addition proxy.

    The four default fan-outs correspond to stem-to-first-depthwise temporal
    taps, two branch-to-residual additions, and two final binary-statistic
    accumulators. Analog pointwise convolutions, residual activities, state
    updates, normalization, and memory traffic remain excluded.
    """

    if spike_summary.get("binary") is not True:
        return {
            "applicable": False,
            "eligible_event_accumulations": 0,
            "energy_claim_supported": False,
        }
    events = [int(value) for value in spike_summary.get("layer_events", [])]
    fanouts = [int(value) for value in eligible_fanouts]
    if len(events) != len(fanouts) or any(value < 0 for value in fanouts):
        raise ValueError("event proxy requires one non-negative fan-out per binary site")
    per_layer = [event * fanout for event, fanout in zip(events, fanouts, strict=True)]
    return {
        "applicable": True,
        "scope": "optimistic_directly_event_triggerable_additions_only",
        "layer_events": events,
        "eligible_fanouts": fanouts,
        "per_layer_event_accumulations": per_layer,
        "eligible_event_accumulations": int(sum(per_layer)),
        "excluded": [
            "analog_pointwise_convolutions",
            "multivalued_residual_activities",
            "state_update_scalar_operations",
            "normalization",
            "memory_traffic",
        ],
        "actual_gpu_sparse_execution_measured": False,
        "energy_claim_supported": False,
    }


def bci2a_lh_rh_subset(
    labels: np.ndarray, *, left_hand_label: int = 0, right_hand_label: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64)
    keep = np.flatnonzero((y == left_hand_label) | (y == right_hand_label)).astype(np.int64)
    if keep.size == 0 or set(np.unique(y[keep]).tolist()) != {left_hand_label, right_hand_label}:
        raise ValueError("BCI2a LH/RH subset does not contain both registered classes")
    remapped = np.where(y[keep] == left_hand_label, 0, 1).astype(np.int64)
    return keep, remapped


def benchmark_latency(
    model: nn.Module,
    atc: torch.Tensor,
    fbc: torch.Tensor,
    *,
    warmup: int = 20,
    repetitions: int = 100,
) -> dict[str, Any]:
    if warmup < 1 or repetitions < 2:
        raise ValueError("latency benchmark requires warmup and at least two repetitions")
    device = atc.device
    if fbc.device != device:
        raise ValueError("latency inputs must share a device")
    model.eval().to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(atc, fbc)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples: list[float] = []
        for _ in range(repetitions):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(atc, fbc)
                end.record()
                torch.cuda.synchronize(device)
                samples.append(float(start.elapsed_time(end)))
            else:
                started = time.perf_counter()
                model(atc, fbc)
                samples.append((time.perf_counter() - started) * 1_000.0)
    values = np.asarray(samples, dtype=np.float64)
    return {
        "repetitions": repetitions,
        "batch_size": int(atc.shape[0]),
        "mean_batch_ms": float(np.mean(values)),
        "median_batch_ms": float(np.median(values)),
        "p95_batch_ms": float(np.quantile(values, 0.95)),
        "median_per_trial_ms": float(np.median(values) / atc.shape[0]),
        "samples_ms": values.tolist(),
    }


@torch.no_grad()
def dense_mac_proxy(
    model: nn.Module,
    atc: torch.Tensor,
    fbc: torch.Tensor,
) -> dict[str, Any]:
    """Count dense Linear/Conv1d multiply-accumulates for one forward pass.

    This deliberately excludes state-update scalar operations, nonlinearities,
    normalization, interpolation, and memory traffic. It is an algorithmic
    operation proxy, not an energy estimate.
    """

    counts: dict[str, int] = {"linear": 0, "conv1d": 0}

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("MAC proxy hook expected a tensor output")
        if isinstance(module, nn.Linear):
            counts["linear"] += int(output.numel()) * int(module.in_features)
        elif isinstance(module, nn.Conv1d):
            kernel = int(module.kernel_size[0])
            fan_in = int(module.in_channels // module.groups) * kernel
            counts["conv1d"] += int(output.numel()) * fan_in

    handles = [
        module.register_forward_hook(hook)
        for module in model.modules()
        if isinstance(module, (nn.Linear, nn.Conv1d))
    ]
    try:
        model.eval().to(atc.device)
        model(atc, fbc)
    finally:
        for handle in handles:
            handle.remove()
    total = int(sum(counts.values()))
    return {
        "scope": "dense_nn_Linear_and_Conv1d_MACs_only",
        "batch_size": int(atc.shape[0]),
        "batch_macs": total,
        "macs_per_trial": float(total / atc.shape[0]),
        "breakdown": counts,
        "excluded": [
            "state_update_scalar_operations",
            "normalization",
            "nonlinearities",
            "interpolation",
            "memory_traffic",
        ],
        "energy_claim_supported": False,
    }
