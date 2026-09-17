"""Accuracy-first causal EEG model for the V8 experiment programme.

The classification trunk is intentionally independent of the optional delay
auxiliary.  A frozen physical sensor basis remains available for fold-local
delay evidence, while the classifier learns separate band-specific spatial
filters.  This prevents learned latent channels from being interpreted as
physical delay nodes.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence

import torch
from torch import nn

from .delayed_statistical_readout import DelayedStatisticalReadout
from .delayed_temporal_pyramid import DelayedTemporalPyramid
from .eeg_frontend import BandLogCovarianceBranch
from .v62_filterbank import (
    CausalAnalyticFilterBank,
    DEFAULT_V62_BANDS,
    DualRateCausalResampler,
    causal_linear_upsample_2x,
)
from .v62_snn_decoder import DecoderKind, ResidualMode, V62SNNDecoder
from .v62_spatial import BCI2A_CHANNEL_NAMES, CoupledSpatialBasis
from .v8_delay_auxiliary import (
    SparsePhysicalDelayAuxiliary,
    V8DelayAuxiliaryOutput,
    V8DelayOverride,
)


class V8AccuracyFirstModel(nn.Module):
    """Causal filter-bank/covariance trunk with matched ANN or SNN decoding."""

    architecture_version = "dpc_snn_v8_accuracy_first_r4_gain_invariant_envelope"
    physical_frontend_version = "dpc_snn_v8_accuracy_first_r2_gain_invariant_envelope"

    def __init__(
        self,
        n_channels: int = 22,
        n_classes: int = 4,
        channel_names: Sequence[str] = BCI2A_CHANNEL_NAMES,
        electrode_coordinates: Sequence[Sequence[float]] | None = None,
        n_bands: int = 12,
        n_latent_nodes: int = 16,
        band_edges_hz: Sequence[Sequence[float]] = DEFAULT_V62_BANDS,
        sfreq: float = 250.0,
        epoch_tmin: float = -1.0,
        task_tmin: float = 0.0,
        task_tmax: float = 4.0,
        analytic_taps: int = 129,
        envelope_taps: int = 33,
        fast_decimation: int = 2,
        spatial_rank: int = 4,
        temporal_dilations: Sequence[int] = (1, 2, 4, 8),
        temporal_depth: int = 2,
        temporal_scale_attention: bool = True,
        decoder_kind: DecoderKind = "ann",
        decoder_residual_mode: ResidualMode = "sew_add",
        decoder_layers: int = 2,
        decoder_channels: int = 64,
        decoder_decays: Sequence[float] = (0.65, 0.90, 0.975),
        endpoint_seconds: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
        statistical_spatial_filters: int = 8,
        statistical_features: int = 96,
        statistical_segments: int = 4,
        statistical_components: Sequence[str] = (
            "carrier_log_variance",
            "envelope_mean",
            "envelope_log_variance",
        ),
        statistical_encoder_kind: str = "linear",
        statistical_variance_transform: str = "log1p",
        statistical_post_spatial_activation: str = "none",
        statistical_spatial_initialization: str = "random",
        statistical_spatial_trainable: bool = True,
        covariance_features_per_band: int = 4,
        use_statistical_branch: bool = True,
        use_covariance_branch: bool = True,
        use_temporal_branch: bool = True,
        delay_auxiliary_enabled: bool = False,
        delay_maximum_routes: int = 256,
        delay_maximum_samples: int = 8,
        delay_allow_cross_band: bool = False,
        delay_signal_mode: str = "fast_phase",
        delay_contextual_residual_enabled: bool = False,
        delay_contextual_residual_bound: float = 0.5,
        delay_phase_residual_enabled: bool = False,
        delay_phase_residual_bound: float = 0.25,
        delay_fusion_bound: float = 0.25,
        delay_fusion_initial: float = 0.05,
        fusion_features: int = 128,
        fusion_kind: str = "mlp",
        dropout: float = 0.25,
        parameter_ceiling: int = 300_000,
    ) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        self.n_classes = int(n_classes)
        self.n_bands = int(n_bands)
        self.n_latent_nodes = int(n_latent_nodes)
        self.sfreq = float(sfreq)
        self.epoch_tmin = float(epoch_tmin)
        self.task_tmin = float(task_tmin)
        self.task_tmax = float(task_tmax)
        self.channel_names = tuple(str(name) for name in channel_names)
        self.use_statistical_branch = bool(use_statistical_branch)
        self.use_covariance_branch = bool(use_covariance_branch)
        self.use_temporal_branch = bool(use_temporal_branch)
        self.delay_auxiliary_enabled = bool(delay_auxiliary_enabled)
        if len(self.channel_names) != self.n_channels:
            raise ValueError("n_channels does not match channel_names")
        if self.n_bands != len(tuple(band_edges_hz)):
            raise ValueError("n_bands does not match band_edges_hz")
        if self.task_tmin <= self.epoch_tmin or self.task_tmax <= self.task_tmin:
            raise ValueError("V8 requires a pre-task baseline and a positive task interval")
        if not (
            self.use_statistical_branch
            or self.use_covariance_branch
            or self.use_temporal_branch
        ):
            raise ValueError("V8 requires at least one active classification branch")
        if self.delay_auxiliary_enabled and not self.use_temporal_branch:
            raise ValueError("the V8 delay auxiliary requires the temporal branch")

        self.filterbank = CausalAnalyticFilterBank(
            band_edges_hz=band_edges_hz,
            sfreq=self.sfreq,
            taps=int(analytic_taps),
        )
        # The delay evidence/online delay branch always sees exact ordered
        # sensors.  It never reuses learned classifier nodes.
        self.physical_basis = CoupledSpatialBasis(
            self.channel_names,
            n_bands=self.n_bands,
            n_nodes=self.n_channels,
            electrode_coordinates=electrode_coordinates,
            anchor_indices=tuple(range(self.n_channels)),
            residual_rank=1,
            exact_sensor_basis=True,
            trainable=False,
        )
        self.classifier_spatial = CoupledSpatialBasis(
            self.channel_names,
            n_bands=self.n_bands,
            n_nodes=self.n_latent_nodes,
            electrode_coordinates=electrode_coordinates,
            residual_rank=int(spatial_rank),
            exact_sensor_basis=False,
            trainable=True,
        )
        self.resampler = DualRateCausalResampler(
            self.n_bands,
            self.n_channels,
            sfreq=self.sfreq,
            analytic_group_delay_samples=self.filterbank.group_delay_samples,
            envelope_taps=int(envelope_taps),
            fast_decimation=int(fast_decimation),
            gain_invariant_envelope=True,
        )
        self.endpoint_seconds = tuple(float(value) for value in endpoint_seconds)
        if not self.endpoint_seconds or tuple(sorted(self.endpoint_seconds)) != (
            self.endpoint_seconds
        ):
            raise ValueError("endpoint_seconds must be non-empty and increasing")
        fast_sfreq = self.sfreq / float(fast_decimation)
        self.endpoint_samples = tuple(
            int(round(value * fast_sfreq)) for value in self.endpoint_seconds
        )
        expected_steps = int(round((self.task_tmax - self.task_tmin) * fast_sfreq))
        if self.endpoint_samples[-1] != expected_steps:
            raise ValueError("the final V8 endpoint must equal the configured task duration")

        if self.use_statistical_branch:
            self.statistical_branch = DelayedStatisticalReadout(
                self.n_bands,
                self.n_channels,
                spatial_filters=int(statistical_spatial_filters),
                endpoint_samples=self.endpoint_samples,
                output_features=int(statistical_features),
                compression_scale=0.10,
                dropout=float(dropout),
                components=statistical_components,
                encoder_kind=str(statistical_encoder_kind),
                temporal_segments=int(statistical_segments),
                variance_transform=str(statistical_variance_transform),
                post_spatial_activation=str(statistical_post_spatial_activation),
                spatial_initialization=str(statistical_spatial_initialization),
                trainable_spatial=bool(statistical_spatial_trainable),
            )
            statistical_output_features = self.statistical_branch.output_features
        else:
            self.statistical_branch = None
            statistical_output_features = 0

        if self.use_covariance_branch:
            self.covariance_branch = BandLogCovarianceBranch(
                self.n_bands,
                self.n_latent_nodes,
                int(covariance_features_per_band),
            )
            covariance_output_features = self.n_bands * int(covariance_features_per_band)
        else:
            self.covariance_branch = None
            covariance_output_features = 0

        if self.use_temporal_branch:
            stream_bands = 2 * self.n_bands
            self.temporal_pyramid = DelayedTemporalPyramid(
                stream_bands,
                self.n_latent_nodes,
                dilations=temporal_dilations,
                depth=int(temporal_depth),
                scale_attention=bool(temporal_scale_attention),
            )
            self.decoder = V62SNNDecoder(
                stream_bands,
                self.n_latent_nodes,
                self.n_classes,
                snn_channels=int(decoder_channels),
                decoder_kind=decoder_kind,
                decoder_residual_mode=decoder_residual_mode,
                decoder_layers=int(decoder_layers),
                decays=decoder_decays,
                sfreq=fast_sfreq,
                endpoint_seconds=self.endpoint_seconds,
                readout_features=96,
                dropout=float(dropout),
            )
            temporal_output_features = 96 + self.n_classes
        else:
            self.temporal_pyramid = None
            self.decoder = None
            temporal_output_features = 0

        if self.delay_auxiliary_enabled:
            self.delay_auxiliary = SparsePhysicalDelayAuxiliary(
                self.n_bands,
                self.n_channels,
                maximum_routes=int(delay_maximum_routes),
                maximum_delay=int(delay_maximum_samples),
                allow_cross_band=bool(delay_allow_cross_band),
                signal_mode=delay_signal_mode,
                contextual_residual_enabled=bool(delay_contextual_residual_enabled),
                contextual_residual_bound=float(delay_contextual_residual_bound),
                phase_residual_enabled=bool(delay_phase_residual_enabled),
                phase_residual_bound=float(delay_phase_residual_bound),
            )
            self.delay_fusion_bound = float(delay_fusion_bound)
            if self.delay_fusion_bound <= 0.0:
                raise ValueError("delay fusion bound must be positive")
            if abs(float(delay_fusion_initial)) >= self.delay_fusion_bound:
                raise ValueError("initial delay fusion must lie strictly inside its bound")
            self.delay_fusion_raw = nn.Parameter(
                torch.tensor(math.atanh(float(delay_fusion_initial) / self.delay_fusion_bound))
            )
        else:
            self.delay_auxiliary = None
            self.delay_fusion_bound = 0.0
            self.register_parameter("delay_fusion_raw", None)

        combined_features = (
            statistical_output_features
            + covariance_output_features
            + temporal_output_features
        )
        self.fusion_kind = str(fusion_kind).lower()
        if self.fusion_kind == "mlp":
            self.fusion = nn.Sequential(
                nn.LayerNorm(combined_features, elementwise_affine=False),
                nn.Linear(combined_features, int(fusion_features), bias=False),
                nn.ELU(),
                nn.Dropout(float(dropout)),
            )
            classifier_features = int(fusion_features)
        elif self.fusion_kind == "linear":
            self.fusion = nn.Identity()
            classifier_features = combined_features
        else:
            raise ValueError("V8 fusion kind must be 'mlp' or 'linear'")
        self.classifier = nn.Linear(classifier_features, self.n_classes)
        self.parameter_ceiling = int(parameter_ceiling)
        if self.parameter_count > self.parameter_ceiling:
            raise ValueError(
                f"V8 model has {self.parameter_count:,} parameters, above the "
                f"registered {self.parameter_ceiling:,} ceiling"
            )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def validate_channel_names(self, channel_names: Sequence[str]) -> None:
        self.physical_basis.validate_channel_names(channel_names)
        self.classifier_spatial.validate_channel_names(channel_names)

    def set_training_gain(self, gain: torch.Tensor) -> None:
        self.resampler.set_training_gain(gain)

    @property
    def delay_fusion_scale(self) -> torch.Tensor:
        if self.delay_fusion_raw is None:
            return self.classifier.weight.new_zeros(())
        return self.delay_fusion_bound * torch.tanh(self.delay_fusion_raw)

    def load_fold_delay_prior(self, **prior: torch.Tensor) -> None:
        if self.delay_auxiliary is None:
            raise RuntimeError("cannot load a delay prior when the auxiliary is disabled")
        self.delay_auxiliary.load_fold_prior(**prior)

    def physical_frontend_fingerprint(self) -> str:
        """Fingerprint the exact node space used by offline/online delay."""

        digest = hashlib.sha256()
        payload = {
            # Retain the r1 payload key/value so E2 physical caches remain
            # reusable when only a downstream delay integration point changes.
            "architecture": self.physical_frontend_version,
            "channel_names": self.channel_names,
            "sfreq": self.sfreq,
            "epoch_tmin": self.epoch_tmin,
            "task_tmin": self.task_tmin,
            "task_tmax": self.task_tmax,
            "physical_basis": self.physical_basis.state_fingerprint(),
        }
        digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
        for name in ("kernel_real", "kernel_imag"):
            digest.update(getattr(self.filterbank, name).detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def _prepare_raw(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.n_channels:
            raise ValueError("V8 expects raw EEG [N, configured channels, T]")
        baseline_stop = int(round((self.task_tmin - self.epoch_tmin) * self.sfreq))
        task_stop = int(round((self.task_tmax - self.epoch_tmin) * self.sfreq))
        if baseline_stop <= 0 or task_stop > x.shape[-1]:
            raise ValueError("raw EEG does not cover the V8 baseline/task interval")
        car = x - x.mean(dim=1, keepdim=True)
        baseline = car[..., :baseline_stop].mean(dim=-1, keepdim=True)
        return car - baseline

    def _covariance_features(self, latent_carrier: torch.Tensor) -> torch.Tensor:
        if self.covariance_branch is None:
            raise RuntimeError("covariance branch is disabled")
        return torch.stack(
            [
                self.covariance_branch(latent_carrier[..., :stop])
                for stop in self.endpoint_samples
            ],
            dim=1,
        )

    def forward_rate_features(
        self,
        physical_fast: torch.Tensor,
        physical_slow: torch.Tensor,
        *,
        delay_override: V8DelayOverride = "off",
    ) -> dict[str, Any]:
        """Classify cached, fold-scaled physical analytic rate features."""

        if physical_fast.ndim != 4 or not physical_fast.is_complex():
            raise ValueError("V8 fast features must be complex [N, B, C, T]")
        if physical_fast.shape[1:3] != (self.n_bands, self.n_channels):
            raise ValueError("V8 fast band/channel axes are incompatible")
        if physical_slow.shape[:-1] != physical_fast.shape[:-1]:
            raise ValueError("V8 slow band/channel axes are incompatible")
        if physical_slow.shape[-1] * 2 != physical_fast.shape[-1]:
            raise ValueError("V8 slow features must be at exactly half the fast rate")

        latent_fast = self.classifier_spatial(physical_fast)
        latent_slow = self.classifier_spatial(physical_slow)
        feature_parts: list[torch.Tensor] = []
        statistical_output = None
        covariance_output = None
        decoded = None
        scale_weights = None
        delay_output: V8DelayAuxiliaryOutput | None = None

        if self.statistical_branch is not None:
            statistical_output = self.statistical_branch(
                physical_fast.real,
                fast_current=physical_fast.real,
                slow_current=physical_slow,
            )
            feature_parts.append(statistical_output.encoded)

        if self.covariance_branch is not None:
            covariance_output = self._covariance_features(latent_fast.real)
            feature_parts.append(covariance_output)

        if self.temporal_pyramid is not None and self.decoder is not None:
            envelope_fast = causal_linear_upsample_2x(latent_slow)
            continuous = torch.cat((latent_fast.real, envelope_fast), dim=1)
            if delay_override != "off":
                if self.delay_auxiliary is None:
                    raise RuntimeError(
                        "a non-off delay override requires an enabled delay auxiliary"
                    )
                delay_output = self.delay_auxiliary(
                    physical_fast,
                    physical_slow,
                    override=delay_override,
                )
                latent_delay = self.classifier_spatial(delay_output.physical_current)
                delay_stream = torch.cat((latent_delay, torch.zeros_like(latent_delay)), dim=1)
                continuous = continuous + self.delay_fusion_scale.to(continuous) * delay_stream
            temporal, scale_weights = self.temporal_pyramid(continuous)
            decoded = self.decoder(temporal)
            feature_parts.extend((decoded.endpoint_features, decoded.logits))

        fused_features = self.fusion(torch.cat(feature_parts, dim=-1))
        prefix_logits = self.classifier(fused_features)
        zero = physical_fast.real.new_zeros(())
        spatial_orthogonality = self.classifier_spatial.orthogonality_loss()
        statistical_orthogonality = (
            zero
            if self.statistical_branch is None
            else self.statistical_branch.orthogonality_loss()
        )
        firing_rate_loss = zero if decoded is None else decoded.firing_rate_loss
        return {
            "logits": prefix_logits[:, -1],
            "prefix_logits": prefix_logits,
            "firing_rate_loss": firing_rate_loss,
            "spatial_orthogonality_loss": spatial_orthogonality,
            "statistical_orthogonality_loss": statistical_orthogonality,
            "aux": {
                "architecture_version": self.architecture_version,
                "decoder_kind": None if self.decoder is None else self.decoder.decoder_kind,
                "decoder_residual_mode": (
                    None if self.decoder is None else self.decoder.decoder_residual_mode
                ),
                "parameter_count": self.parameter_count,
                "trainable_parameter_count": self.trainable_parameter_count,
                "fusion_kind": self.fusion_kind,
                "physical_frontend_fingerprint": self.physical_frontend_fingerprint(),
                "physical_fast": physical_fast,
                "physical_slow": physical_slow,
                "latent_fast": latent_fast,
                "latent_slow": latent_slow,
                "statistical_output": statistical_output,
                "covariance_features": covariance_output,
                "temporal_scale_weights": scale_weights,
                "delay_auxiliary": delay_output,
                "delay_fusion_scale": self.delay_fusion_scale,
                "binary_spikes": () if decoded is None else decoded.binary_spikes,
                "residual_activities": () if decoded is None else decoded.residual_activities,
                "final_spikes": None if decoded is None else decoded.final_spikes,
                "final_membrane": None if decoded is None else decoded.final_membrane,
            },
        }

    def forward(
        self,
        x: torch.Tensor,
        *,
        delay_override: V8DelayOverride = "off",
    ) -> dict[str, Any]:
        prepared = self._prepare_raw(x)
        analytic = self.filterbank(prepared)
        physical = self.physical_basis(analytic)
        rates = self.resampler(
            physical,
            epoch_tmin=self.epoch_tmin,
            task_tmin=self.task_tmin,
            task_tmax=self.task_tmax,
        )
        output = self.forward_rate_features(
            rates.fast,
            rates.slow,
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
