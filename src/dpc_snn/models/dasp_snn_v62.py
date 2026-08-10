"""DASP-SNN V6.2-R1 end-to-end model.

This model is deliberately composed from new causal modules and does not
inherit the legacy DPCSNN implementation.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .coupled_dual_delay import CoupledDualDelay, DelayOverride, DualDelayOutput
from .delayed_causal_atc_readout import DelayedCausalATCReadout
from .delayed_geometry_sketch import CausalDelayedGeometryFiLM
from .delayed_statistical_readout import DelayedStatisticalReadout
from .delayed_temporal_pyramid import DelayedTemporalPyramid
from .v62_filterbank import (
    CausalAnalyticFilterBank,
    DEFAULT_V62_BANDS,
    DualRateCausalResampler,
    causal_linear_upsample_2x,
)
from .v62_snn_decoder import DecoderOutput, V62SNNDecoder
from .v62_spatial import BCI2A_CHANNEL_NAMES, CoupledSpatialBasis


class _MaxNormLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, max_norm: float) -> None:
        super().__init__(in_features, out_features)
        self.max_norm = float(max_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.renorm(self.weight, p=2, dim=0, maxnorm=self.max_norm)
        return F.linear(x, weight, self.bias)


class _BoundedPhasePairAdapter(nn.Module):
    """Bias-free band/node gain for a zero-safe delayed phase current."""

    def __init__(
        self,
        n_bands: int,
        n_nodes: int,
        *,
        bound: float,
        initial_scale: float,
    ) -> None:
        super().__init__()
        self.bound = float(bound)
        if self.bound <= 0.0:
            raise ValueError("phase-pair adapter bound must be positive")
        if abs(float(initial_scale)) >= self.bound:
            raise ValueError("initial phase-pair scale must lie inside its bound")
        initial = torch.full(
            (int(n_bands), int(n_nodes)),
            math.atanh(float(initial_scale) / self.bound),
        )
        self.raw_scale = nn.Parameter(initial)

    @property
    def scale(self) -> torch.Tensor:
        return self.bound * torch.tanh(self.raw_scale)

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        if current.ndim != 4 or current.shape[1:3] != self.raw_scale.shape:
            raise ValueError("phase-pair current must have shape [N, B, K, T]")
        return current * self.scale.to(current)[None, :, :, None]


class DASPSNNV62(nn.Module):
    """Delay-Aligned Spectro-Spatial Pyramid Spiking Neural Network."""

    architecture_version = "dasp_snn_v7_degree_normalized_route_r7"
    frontend_architecture_version = "dasp_snn_v6_2_r1"

    def __init__(
        self,
        n_channels: int = 22,
        n_classes: int = 4,
        channel_names: Sequence[str] = BCI2A_CHANNEL_NAMES,
        electrode_coordinates: Sequence[Sequence[float]] | None = None,
        n_bands: int = 12,
        n_nodes: int = 16,
        band_edges_hz: Sequence[Sequence[float]] = DEFAULT_V62_BANDS,
        sfreq: float = 250.0,
        epoch_tmin: float = -1.0,
        task_tmin: float = 0.0,
        task_tmax: float = 4.0,
        analytic_taps: int = 129,
        envelope_taps: int = 33,
        fast_decimation: int = 2,
        spatial_rank: int = 4,
        spatial_anchor_sigma: float = 0.32,
        spatial_exact_sensor_basis: bool = False,
        freeze_spatial: bool = True,
        route_rank: int = 4,
        delay_rank: int = 4,
        fast_max_delay: int = 4,
        slow_max_delay: int = 16,
        fast_initial_contribution: float = 0.10,
        phase_coupling_bound: float = 0.15,
        target_interaction_bound: float = 0.50,
        dynamic_context_features: int = 0,
        temporal_dilations: Sequence[int] = (1, 2, 4, 8),
        temporal_depth: int = 2,
        temporal_scale_attention: bool = True,
        use_geometry: bool = True,
        geometry_components: int = 6,
        geometry_windows_seconds: Sequence[float] = (0.25, 0.5, 1.0),
        snn_channels: int = 64,
        decoder_kind: str = "clif",
        decoder_layers: int = 2,
        decoder_decays: Sequence[float] = (0.65, 0.90, 0.975),
        dropout: float = 0.25,
        force_zero_delay: bool = False,
        slow_delay_enabled: bool = True,
        cross_band_enabled: bool = True,
        fast_delay_enabled: bool = True,
        phase_residual_enabled: bool = False,
        phase_pair_current_enabled: bool = False,
        phase_pair_current_bound: float = 0.0,
        phase_pair_initial_scale: float = 0.0,
        identity_delay_backbone: bool = True,
        residual_route_scale: float = 0.10,
        slow_residual_route_scale: float | None = None,
        fast_residual_route_scale: float | None = None,
        initial_delay_fraction: float = 0.05,
        rejected_route_prior_floor: float = 0.02,
        slow_carrier_residual_scale: float = 0.05,
        use_statistical_readout: bool = True,
        statistical_spatial_filters: int = 8,
        statistical_features: int = 96,
        statistical_compression_scale: float = 0.10,
        statistical_fusion: str = "residual",
        statistical_components: Sequence[str] = (
            "carrier_log_variance",
            "envelope_mean",
            "envelope_log_variance",
        ),
        statistical_encoder: str = "linear",
        statistical_temporal_segments: int = 1,
        statistical_variance_transform: str = "log1p",
        statistical_post_spatial_activation: str = "none",
        statistical_classifier_max_norm: float | None = None,
        use_atc_readout: bool = False,
        atc_reconstruct_sensors: bool = False,
        atc_temporal_filters: int = 16,
        atc_depth_multiplier: int = 2,
        atc_temporal_kernel: int = 64,
        atc_first_pool: int = 8,
        atc_second_pool: int = 7,
        atc_convolution_dropout: float = 0.3,
        atc_key_features: int = 8,
        atc_attention_heads: int = 2,
        atc_attention_dropout: float = 0.5,
        atc_tcn_depth: int = 2,
        atc_tcn_kernel: int = 4,
        atc_tcn_dropout: float = 0.3,
        atc_windows: int = 5,
        atc_fusion_initial: float = 0.5,
        parameter_ceiling: int = 220_000,
    ) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        self.n_classes = int(n_classes)
        self.n_bands = int(n_bands)
        self.n_nodes = int(n_nodes)
        self.sfreq = float(sfreq)
        self.epoch_tmin = float(epoch_tmin)
        self.task_tmin = float(task_tmin)
        self.task_tmax = float(task_tmax)
        self.force_zero_delay = bool(force_zero_delay)
        self.slow_delay_enabled = bool(slow_delay_enabled)
        self.cross_band_enabled = bool(cross_band_enabled)
        self.fast_delay_enabled = bool(fast_delay_enabled)
        self.phase_residual_enabled = bool(phase_residual_enabled)
        self.phase_pair_current_enabled = bool(phase_pair_current_enabled)
        if self.phase_pair_current_enabled:
            self.architecture_version = "dasp_snn_v7_phase_pair_current_r9"
        self.use_statistical_readout = bool(use_statistical_readout)
        self.use_atc_readout = bool(use_atc_readout)
        self.statistical_fusion = str(statistical_fusion).lower()
        if self.statistical_fusion not in {"statistics", "residual"}:
            raise ValueError("statistical_fusion must be 'statistics' or 'residual'")
        self.use_geometry = bool(use_geometry)
        self.channel_names = tuple(str(name) for name in channel_names)
        if len(self.channel_names) != self.n_channels:
            raise ValueError("n_channels does not match channel_names")
        if self.n_bands != len(tuple(band_edges_hz)):
            raise ValueError("n_bands does not match band_edges_hz")
        if self.task_tmin <= self.epoch_tmin:
            raise ValueError("DASP-SNN requires a pre-task baseline")
        if self.task_tmax <= self.task_tmin:
            raise ValueError("task_tmax must exceed task_tmin")

        self.filterbank = CausalAnalyticFilterBank(
            band_edges_hz=band_edges_hz,
            sfreq=self.sfreq,
            taps=int(analytic_taps),
        )
        self.spatial = CoupledSpatialBasis(
            self.channel_names,
            n_bands=self.n_bands,
            n_nodes=self.n_nodes,
            electrode_coordinates=electrode_coordinates,
            residual_rank=int(spatial_rank),
            anchor_sigma=float(spatial_anchor_sigma),
            exact_sensor_basis=bool(spatial_exact_sensor_basis),
            trainable=not bool(freeze_spatial),
        )
        self.resampler = DualRateCausalResampler(
            self.n_bands,
            self.n_nodes,
            sfreq=self.sfreq,
            analytic_group_delay_samples=self.filterbank.group_delay_samples,
            envelope_taps=int(envelope_taps),
            fast_decimation=int(fast_decimation),
        )
        self.delay = CoupledDualDelay(
            self.n_bands,
            self.n_nodes,
            route_rank=int(route_rank),
            delay_rank=int(delay_rank),
            fast_max_delay=int(fast_max_delay),
            slow_max_delay=int(slow_max_delay),
            fast_initial_contribution=float(fast_initial_contribution),
            phase_bound=float(phase_coupling_bound),
            target_interaction_bound=float(target_interaction_bound),
            context_features=int(dynamic_context_features),
            cross_band_enabled=self.cross_band_enabled,
            identity_backbone=bool(identity_delay_backbone),
            residual_route_scale=float(residual_route_scale),
            slow_residual_route_scale=slow_residual_route_scale,
            fast_residual_route_scale=fast_residual_route_scale,
            initial_delay_fraction=float(initial_delay_fraction),
            rejected_route_prior_floor=float(rejected_route_prior_floor),
            slow_carrier_residual_scale=float(slow_carrier_residual_scale),
            phase_pair_current_enabled=self.phase_pair_current_enabled,
        )
        self.phase_pair_adapter = (
            _BoundedPhasePairAdapter(
                self.n_bands,
                self.n_nodes,
                bound=float(phase_pair_current_bound),
                initial_scale=float(phase_pair_initial_scale),
            )
            if self.phase_pair_current_enabled
            else None
        )
        if not self.phase_residual_enabled:
            self.delay.phase_preference.requires_grad_(False)
        self.temporal_pyramid = DelayedTemporalPyramid(
            self.n_bands,
            self.n_nodes,
            dilations=temporal_dilations,
            depth=int(temporal_depth),
            scale_attention=bool(temporal_scale_attention),
        )
        signed_channels = int(snn_channels) // 2
        self.geometry = (
            CausalDelayedGeometryFiLM(
                self.n_bands,
                self.n_nodes,
                output_channels=signed_channels,
                sfreq=self.sfreq / float(fast_decimation),
                windows_seconds=geometry_windows_seconds,
                covariance_components=int(geometry_components),
            )
            if self.use_geometry
            else None
        )
        self.decoder = V62SNNDecoder(
            self.n_bands,
            self.n_nodes,
            self.n_classes,
            snn_channels=int(snn_channels),
            decoder_kind=str(decoder_kind).lower(),
            decoder_layers=int(decoder_layers),
            decays=decoder_decays,
            sfreq=self.sfreq / float(fast_decimation),
            endpoint_seconds=(1.0, 2.0, 3.0, self.task_tmax - self.task_tmin),
            dropout=float(dropout),
        )
        if self.use_atc_readout:
            if bool(atc_reconstruct_sensors):
                if not self.spatial.frozen:
                    raise ValueError(
                        "ATC sensor reconstruction requires a frozen spatial basis"
                    )
                atc_node_reconstruction = torch.linalg.pinv(
                    self.spatial.shared_weight().detach()
                )
            else:
                atc_node_reconstruction = None
            self.atc_readout = DelayedCausalATCReadout(
                self.n_bands,
                self.n_nodes,
                self.n_classes,
                endpoint_samples=self.decoder.endpoint_samples,
                node_reconstruction=atc_node_reconstruction,
                temporal_filters=int(atc_temporal_filters),
                depth_multiplier=int(atc_depth_multiplier),
                temporal_kernel=int(atc_temporal_kernel),
                first_pool=int(atc_first_pool),
                second_pool=int(atc_second_pool),
                convolution_dropout=float(atc_convolution_dropout),
                key_features=int(atc_key_features),
                attention_heads=int(atc_attention_heads),
                attention_dropout=float(atc_attention_dropout),
                tcn_depth=int(atc_tcn_depth),
                tcn_kernel=int(atc_tcn_kernel),
                tcn_dropout=float(atc_tcn_dropout),
                windows=int(atc_windows),
            )
        else:
            self.atc_readout = None
        if self.use_statistical_readout:
            self.statistical_readout = DelayedStatisticalReadout(
                self.n_bands,
                self.n_nodes,
                spatial_filters=int(statistical_spatial_filters),
                endpoint_samples=self.decoder.endpoint_samples,
                output_features=int(statistical_features),
                compression_scale=float(statistical_compression_scale),
                dropout=float(dropout),
                components=statistical_components,
                encoder_kind=str(statistical_encoder),
                temporal_segments=int(statistical_temporal_segments),
                variance_transform=str(statistical_variance_transform),
                post_spatial_activation=str(statistical_post_spatial_activation),
            )
            if self.statistical_fusion == "residual":
                if self.statistical_readout.output_features != 96:
                    raise ValueError(
                        "residual statistical fusion requires a 96-D statistical encoder"
                    )
                self.endpoint_fusion_gate = nn.Parameter(torch.tensor(-2.0))
                self.statistical_classifier = None
            else:
                self.register_parameter("endpoint_fusion_gate", None)
                if statistical_classifier_max_norm is None:
                    self.statistical_classifier = nn.Linear(
                        self.statistical_readout.output_features, self.n_classes
                    )
                else:
                    self.statistical_classifier = _MaxNormLinear(
                        self.statistical_readout.output_features,
                        self.n_classes,
                        float(statistical_classifier_max_norm),
                    )
        else:
            self.statistical_readout = None
            self.register_parameter("endpoint_fusion_gate", None)
            self.statistical_classifier = None
        if self.atc_readout is not None and self.statistical_readout is not None:
            if not 0.0 < float(atc_fusion_initial) < 1.0:
                raise ValueError("initial ATC fusion weight must lie strictly in (0, 1)")
            self.atc_fusion_logit = nn.Parameter(
                torch.logit(torch.tensor(float(atc_fusion_initial)))
            )
        else:
            self.register_parameter("atc_fusion_logit", None)
        if int(dynamic_context_features) > 0:
            self.context_encoder = nn.Sequential(
                nn.Linear(2 * self.n_channels, int(dynamic_context_features)),
                nn.Tanh(),
            )
        else:
            self.context_encoder = None

        self.parameter_ceiling = int(parameter_ceiling)
        count = self.parameter_count
        if count > self.parameter_ceiling:
            raise ValueError(
                f"DASP-SNN V6.2-R1 has {count:,} parameters, above the "
                f"pre-registered {self.parameter_ceiling:,} ceiling"
            )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def validate_channel_names(self, channel_names: Sequence[str]) -> None:
        self.spatial.validate_channel_names(channel_names)

    def set_training_gain(self, gain: torch.Tensor) -> None:
        self.resampler.set_training_gain(gain)

    def frontend_fingerprint(self) -> str:
        digest = hashlib.sha256()
        payload = {
            # Post-delay transport/readout revisions must not invalidate a
            # physically identical fold-local filterbank/spatial prior.
            "architecture": self.frontend_architecture_version,
            "channel_names": self.channel_names,
            "sfreq": self.sfreq,
            "epoch_tmin": self.epoch_tmin,
            "task_tmin": self.task_tmin,
            "task_tmax": self.task_tmax,
            "spatial": self.spatial.state_fingerprint(),
        }
        digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
        for name in ("kernel_real", "kernel_imag"):
            digest.update(getattr(self.filterbank, name).detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def _baseline_context(self, x_car: torch.Tensor, baseline_stop: int) -> torch.Tensor | None:
        if self.context_encoder is None:
            return None
        baseline = x_car[..., :baseline_stop]
        return torch.cat(
            (
                baseline.mean(dim=-1),
                baseline.std(dim=-1, unbiased=False),
            ),
            dim=1,
        )

    def _encode_context(self, context: torch.Tensor | None) -> torch.Tensor | None:
        if self.context_encoder is None:
            if context is not None:
                raise ValueError("context was supplied to a model with no context encoder")
            return None
        if context is None or context.ndim != 2 or context.shape[1] != 2 * self.n_channels:
            raise ValueError("trial context must contain baseline mean/std for every channel")
        return self.context_encoder(context)

    def load_fold_local_slow_prior(
        self,
        route_probability: torch.Tensor,
        positive_delay_probability: torch.Tensor,
        fractional_delay_target: torch.Tensor,
    ) -> None:
        self.delay.load_fold_local_slow_prior(
            route_probability,
            positive_delay_probability,
            fractional_delay_target,
        )

    def load_fold_local_phase_amplitude_scale(self, scale: torch.Tensor) -> None:
        self.delay.load_fold_local_phase_amplitude_scale(scale)

    def _prepare_raw(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if x.ndim != 3 or x.shape[1] != self.n_channels:
            raise ValueError("DASP-SNN expects raw EEG [N, configured channels, T]")
        baseline_stop = int(round((self.task_tmin - self.epoch_tmin) * self.sfreq))
        task_stop = int(round((self.task_tmax - self.epoch_tmin) * self.sfreq))
        if baseline_stop <= 0 or task_stop > x.shape[-1]:
            raise ValueError("raw EEG does not cover the configured baseline/task interval")
        x_car = x - x.mean(dim=1, keepdim=True)
        context = self._baseline_context(x_car, baseline_stop)
        baseline_mean = x_car[..., :baseline_stop].mean(dim=-1, keepdim=True)
        return x_car - baseline_mean, context

    def decode_delayed_current(
        self,
        delayed_current: torch.Tensor,
        *,
        fast_current: torch.Tensor | None = None,
        slow_current: torch.Tensor | None = None,
        broadband_current: torch.Tensor | None = None,
        delayed_interaction_current: torch.Tensor | None = None,
        delayed_route_contrast_current: torch.Tensor | None = None,
        delayed_phase_pair_current: torch.Tensor | None = None,
    ) -> tuple[DecoderOutput, dict[str, Any]]:
        temporal, scale_weights = self.temporal_pyramid(delayed_current)
        if self.geometry is None:
            geometry_scale = None
            geometry_features = None
        else:
            geometry_scale, geometry_features = self.geometry(delayed_current)
        decoded = self.decoder(temporal, geometry_scale)
        if self.statistical_readout is None:
            statistical = None
        else:
            statistical = self.statistical_readout(
                delayed_current,
                fast_current=fast_current,
                slow_current=slow_current,
            )
            if self.statistical_fusion == "statistics":
                fused_features = statistical.encoded
                if self.statistical_classifier is None:
                    raise RuntimeError("statistics-only fusion has no classifier")
                fused_logits = self.statistical_classifier(fused_features)
            else:
                gate = torch.sigmoid(self.endpoint_fusion_gate)
                fused_features = statistical.encoded + gate * decoded.endpoint_features
                fused_logits = self.decoder.classifier(fused_features)
            decoded = replace(
                decoded,
                logits=fused_logits,
                endpoint_features=fused_features,
            )
        if self.atc_readout is None:
            atc = None
            atc_fusion_weight = None
        else:
            if bool(getattr(self.atc_readout, "supports_broadband_fusion", False)):
                atc = self.atc_readout(
                    delayed_current,
                    broadband_carrier=broadband_current,
                    statistics_carrier=delayed_interaction_current,
                    route_statistics_carrier=delayed_route_contrast_current,
                    phase_pair_carrier=delayed_phase_pair_current,
                )
            else:
                if broadband_current is not None:
                    atc_source = broadband_current
                else:
                    atc_source = delayed_current if fast_current is None else fast_current
                atc = self.atc_readout(atc_source)
            if statistical is None:
                atc_fusion_weight = None
                fused_logits = atc.logits
            else:
                if self.atc_fusion_logit is None:
                    raise RuntimeError("joint statistical/ATC readout has no fusion gate")
                atc_fusion_weight = torch.sigmoid(self.atc_fusion_logit)
                fused_logits = (
                    (1.0 - atc_fusion_weight) * decoded.logits
                    + atc_fusion_weight * atc.logits
                )
            decoded = replace(decoded, logits=fused_logits)
        return decoded, {
            "temporal_scale_weights": scale_weights,
            "geometry_scale": geometry_scale,
            "geometry_features": geometry_features,
            "delayed_statistical_features": (
                None if statistical is None else statistical.raw_statistics
            ),
            "delayed_statistical_carrier": (
                None if statistical is None else statistical.carrier_spatial_signals
            ),
            "delayed_statistical_envelope": (
                None if statistical is None else statistical.envelope_spatial_signals
            ),
            "delayed_atc_logits": None if atc is None else atc.logits,
            "delayed_atc_synthesized_carrier": (
                None if atc is None else atc.synthesized_carrier
            ),
            "delayed_atc_statistics_logits": (
                None
                if atc is None
                else getattr(atc, "delayed_statistics_logits", None)
            ),
            "delayed_atc_route_statistics_logits": (
                None
                if atc is None
                else getattr(atc, "delayed_route_statistics_logits", None)
            ),
            "delayed_phase_pair_current": delayed_phase_pair_current,
            "delayed_atc_statistics_gate": (
                None
                if atc is None
                else getattr(atc, "delayed_statistics_gate", None)
            ),
            "delayed_atc_fusion_weight": atc_fusion_weight,
        }

    def forward_rate_features(
        self,
        fast: torch.Tensor,
        slow: torch.Tensor,
        *,
        broadband: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        delay_override: DelayOverride | None = None,
    ) -> dict[str, Any]:
        """Run cached fixed-front-end features through the mandatory delay bottleneck."""

        if delay_override is not None:
            slow_override = delay_override
            fast_override = delay_override
        elif self.force_zero_delay:
            slow_override = "zero"
            fast_override = "zero"
        else:
            slow_override = "learned" if self.slow_delay_enabled else "zero"
            fast_override = "learned" if self.fast_delay_enabled else "zero"
        encoded_context = self._encode_context(context)
        broadband_delay_source = (
            "slow_route_residual"
            if self.slow_delay_enabled and not self.fast_delay_enabled
            else "fast"
        )
        transport: DualDelayOutput = self.delay(
            fast,
            slow,
            broadband_source=broadband,
            context=encoded_context,
            slow_delay_override=slow_override,
            fast_delay_override=fast_override,
            broadband_delay_source=broadband_delay_source,
        )
        phase_pair_current = None
        decoder_current = transport.fused_current
        if self.phase_pair_current_enabled:
            if self.phase_pair_adapter is None:
                raise RuntimeError("phase-pair current has no registered adapter")
            slow_phase_pair = transport.slow_phase_pair_delay_contrast_current
            if slow_phase_pair is None:
                raise RuntimeError("delay transport did not return phase-pair current")
            phase_pair_current = self.phase_pair_adapter(
                causal_linear_upsample_2x(slow_phase_pair)
            )
            decoder_current = decoder_current + phase_pair_current
        decoded, post_delay = self.decode_delayed_current(
            decoder_current,
            fast_current=transport.fast_current,
            slow_current=transport.slow_current,
            broadband_current=transport.broadband_current,
            delayed_interaction_current=causal_linear_upsample_2x(
                transport.slow_delay_contrast_current
            ),
            delayed_route_contrast_current=(
                transport.slow_route_delay_contrast_current
            ),
            delayed_phase_pair_current=phase_pair_current,
        )
        return {
            "logits": decoded.logits[:, -1],
            "prefix_logits": decoded.logits,
            "spikes": decoded.final_spikes,
            "membrane": decoded.final_membrane,
            "firing_rate_loss": decoded.firing_rate_loss,
            "aux": {
                "architecture_version": self.architecture_version,
                "parameter_count": self.parameter_count,
                "frontend_fingerprint": self.frontend_fingerprint(),
                "transport": transport,
                "phase_pair_adapter_scale": (
                    None
                    if self.phase_pair_adapter is None
                    else self.phase_pair_adapter.scale
                ),
                "binary_spikes": decoded.binary_spikes,
                "residual_activities": decoded.residual_activities,
                **post_delay,
            },
        }

    def forward(
        self,
        x: torch.Tensor,
        *,
        delay_override: DelayOverride | None = None,
    ) -> dict[str, Any]:
        x, context = self._prepare_raw(x)
        analytic = self.filterbank(x)
        projected = self.spatial(analytic)
        rates = self.resampler(
            projected,
            epoch_tmin=self.epoch_tmin,
            task_tmin=self.task_tmin,
            task_tmax=self.task_tmax,
        )
        if self.resampler.fast_decimation == 1:
            task_start = int(round((self.task_tmin - self.epoch_tmin) * self.sfreq))
            task_stop = int(round((self.task_tmax - self.epoch_tmin) * self.sfreq))
            broadband = self.spatial.project_real(x[..., task_start:task_stop])
        else:
            broadband = rates.fast.real.sum(dim=1) / (self.n_bands**0.5)
        output = self.forward_rate_features(
            rates.fast,
            rates.slow,
            broadband=broadband,
            context=context,
            delay_override=delay_override,
        )
        output["aux"].update(
            {
                "fast_timestamps": rates.fast_timestamps,
                "slow_timestamps": rates.slow_timestamps,
                "fast_availability": rates.fast_availability,
                "slow_availability": rates.slow_availability,
            }
        )
        return output
