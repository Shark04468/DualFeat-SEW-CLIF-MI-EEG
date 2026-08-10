"""DPC-SNN V3: continuous EEG geometry before mandatory delay-phase spiking."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .delay_phase_synapse import DelayPhaseGraphSynapse
from .eeg_frontend import (
    AnchoredSpatialProjection,
    BandLogCovarianceBranch,
    BandSpecificAnchoredSpatialProjection,
    DelayedBandPairStatistics,
    DelayedDirectionalMoments,
    LearnableAnalyticFilterBank,
)
from .event_encoder import CausalSpikeAccumulator, PhaseDeltaEventEncoder
from .lif import (
    CLIFLayer,
    PLIFLayer,
    MultiTimescaleSEWBlock,
    MultiTimescaleStateLayer,
    SEWCLIFBlock,
)


DEFAULT_BAND_EDGES = [
    [6.0, 8.0],
    [8.0, 10.0],
    [10.0, 13.0],
    [13.0, 18.0],
    [18.0, 24.0],
    [24.0, 30.0],
    [30.0, 40.0],
]
DEFAULT_DELAY_EVIDENCE_EDGES = [
    [6.0, 8.0],
    [8.0, 10.0],
    [10.0, 12.0],
    [12.0, 14.0],
    [14.0, 17.0],
    [17.0, 20.0],
    [20.0, 23.0],
    [23.0, 26.0],
    [26.0, 30.0],
    [30.0, 34.0],
    [34.0, 37.0],
    [37.0, 40.0],
]


class DPCSNN(nn.Module):
    def __init__(
        self,
        n_classes: int,
        n_channels: int,
        n_bands: int = 7,
        hidden_channels: int = 32,
        timesteps: int = 500,
        d_max: int = 8,
        slow_d_max: int = 8,
        slow_downsample: int = 4,
        graph_sparsity: float = 0.35,
        encoder_threshold: float = 0.5,
        encoder_deterministic: bool = True,
        tau_mem: float = 0.9,
        lif_threshold: float = 1.0,
        current_gain: float = 0.5,
        use_membrane_readout: bool = True,
        spike_rate_reg: float = 0.001,
        graph_l1_reg: float = 0.0001,
        delay_smooth_reg: float = 0.0001,
        delay_entropy_reg: float | None = None,
        delay_gate_l1_reg: float = 0.0001,
        delay_gate_structure_reg: float | None = None,
        spatial_orthogonal_reg: float = 0.001,
        graph_prior_reg: float = 0.0001,
        delay_alignment_reg: float = 0.01,
        delay_anchor_kl_reg: float = 0.1,
        delay_temperature: float = 0.75,
        min_delay_temperature: float = 0.2,
        delay_gate_init: float = 0.0,
        max_phase_residual: float = 0.25,
        temporal_pool_bins: int = 64,
        timestep_seconds: float = 1.0,
        sfreq: float = 250.0,
        graph_timesteps: int = 500,
        latent_nodes: int = 16,
        spatial_max_deviation: float = 0.75,
        snn_channels: int = 64,
        band_edges_hz: list[list[float]] | None = None,
        classification_band_edges_hz: list[list[float]] | None = None,
        alignment_matrix: list[list[float]] | torch.Tensor | None = None,
        reference_augmentation_prob: float = 0.5,
        epoch_tmin: float = 0.0,
        task_tmin: float = 0.0,
        task_tmax: float | None = None,
        spike_rate_target: float = 0.12,
        use_covariance: bool = True,
        use_phase_confidence: bool = True,
        delay_lr_multiplier: float = 0.25,
        use_cross_band_routes: bool = True,
        n_delay_experts: int = 4,
        delay_expert_topk: int = 2,
        preserve_graph_rate: bool = True,
        temporal_pool_channels: int = 16,
        expert_balance_reg: float = 0.001,
        expert_diversity_reg: float = 0.01,
        force_zero_delay: bool = False,
        route_rank: int = 4,
        decoder_layers: int = 3,
        tet_scales: tuple[int, ...] = (8, 16, 32, 64),
        identity_spatial_projection: bool = False,
        delay_evidence_band_edges_hz: list[list[float]] | None = None,
        router_warmup_fraction: float = 0.35,
        router_noise_scale: float = 1.0,
        route_odds_threshold: float | None = None,
        hurdle_min_bayes_factor: float | None = 3.0,
        hurdle_route_temperature: float = 0.5,
        spatial_band_rank: int = 4,
        spatial_band_deviation: float = 0.25,
        snn_membrane_norm_groups: int = 8,
        cumulative_readout_seconds: tuple[float, ...] = (),
        causal_channel_norm: bool = False,
        decoder_neuron: str = "clif",
        sew_connect_function: str = "ADD",
        fixed_delay_steps: float | None = None,
        freeze_delay_posterior: bool = False,
        architecture_version: str = "dpc_snn_v3_9_scientific_repair",
        freeze_shared_physical_basis: bool = False,
        preserve_transport_band_pairs: bool = False,
        delayed_stat_channels: int = 4,
        rejected_route_prior_floor: float = 0.02,
        matched_transport_control: bool = False,
        freeze_delay_after_pretrain: bool = False,
        freeze_shared_filterbank: bool = False,
        delayed_directional_moments: bool = False,
        event_native_transport: bool = False,
        event_envelope_threshold: float = 0.05,
        event_phase_confidence_threshold: float = 0.10,
        learnable_event_threshold: bool = False,
        event_timing_intervention: str = "none",
        event_timing_seed: int = 0,
        multiscale_snn: bool = False,
        decoder_timescales: tuple[float, ...] = (0.65, 0.90, 0.975),
        snn_native_readout: bool = False,
        snn_readout_decays: tuple[float, ...] = (0.5, 0.9, 0.99),
        decoder_mode: str = "snn",
        checkpoint_event_routes: bool = False,
        electrode_coordinates: list[list[float]] | torch.Tensor | None = None,
        spatial_anchor_indices: list[int] | torch.Tensor | None = None,
    ):
        super().__init__()
        del encoder_threshold, encoder_deterministic, timestep_seconds
        band_edges_hz = band_edges_hz or DEFAULT_BAND_EDGES[:n_bands]
        n_bands = len(band_edges_hz)
        classification_band_edges_hz = classification_band_edges_hz or band_edges_hz
        n_classification_bands = len(classification_band_edges_hz)
        latent_nodes = min(int(latent_nodes), int(n_channels))
        self.n_classes = int(n_classes)
        self.architecture_version = str(architecture_version)
        self.implementation_id = self.architecture_version
        self.n_channels = int(n_channels)
        self.n_bands = int(n_bands)
        self.n_classification_bands = int(n_classification_bands)
        self.graph_timesteps = int(graph_timesteps)
        self.preserve_graph_rate = bool(preserve_graph_rate)
        self.timesteps = self.graph_timesteps if self.preserve_graph_rate else int(timesteps)
        self.sfreq = float(sfreq)
        self.epoch_tmin = float(epoch_tmin)
        self.task_tmin = float(task_tmin)
        self.task_tmax = None if task_tmax is None else float(task_tmax)
        self.current_gain = float(current_gain)
        self.use_membrane_readout = bool(use_membrane_readout)
        self.reference_augmentation_prob = float(reference_augmentation_prob)
        self.spike_rate_target = float(spike_rate_target)
        self.use_covariance = bool(use_covariance)
        self.use_phase_confidence = bool(use_phase_confidence)
        self.delay_lr_multiplier = float(delay_lr_multiplier)
        self.freeze_delay_posterior = bool(freeze_delay_posterior)
        self.freeze_shared_physical_basis = bool(freeze_shared_physical_basis)
        self.preserve_transport_band_pairs = bool(preserve_transport_band_pairs)
        self.delayed_stat_channels = max(0, int(delayed_stat_channels))
        self.matched_transport_control = bool(matched_transport_control)
        self.freeze_delay_after_pretrain = bool(freeze_delay_after_pretrain)
        self.freeze_shared_filterbank = bool(freeze_shared_filterbank)
        self.use_delayed_directional_moments = bool(delayed_directional_moments)
        self.event_native_transport = bool(event_native_transport)
        self.multiscale_snn = bool(multiscale_snn)
        self.snn_native_readout = bool(snn_native_readout)
        self.decoder_mode = str(decoder_mode).lower()
        if self.decoder_mode not in {"snn", "matched_ann"}:
            raise ValueError("decoder_mode must be 'snn' or 'matched_ann'")
        self.decoder_is_spiking = self.decoder_mode == "snn"
        if self.architecture_version == "dpc_snn_v4_2_identifiable_decoder":
            required_contract = {
                "freeze_shared_physical_basis": self.freeze_shared_physical_basis,
                "freeze_shared_filterbank": self.freeze_shared_filterbank,
                "freeze_delay_after_pretrain": self.freeze_delay_after_pretrain,
                "preserve_transport_band_pairs": self.preserve_transport_band_pairs,
                "delayed_directional_moments": self.use_delayed_directional_moments,
            }
            violated = sorted(name for name, enabled in required_contract.items() if not enabled)
            if violated:
                raise ValueError("V4.2 scientific contract requires: " + ", ".join(violated))
        if self.architecture_version.startswith("dpc_snn_v5"):
            required_contract = {
                "freeze_shared_physical_basis": self.freeze_shared_physical_basis,
                "freeze_shared_filterbank": self.freeze_shared_filterbank,
                "freeze_delay_after_pretrain": self.freeze_delay_after_pretrain,
                "preserve_transport_band_pairs": self.preserve_transport_band_pairs,
                "preserve_graph_rate": self.preserve_graph_rate,
                "event_native_transport": self.event_native_transport,
                "multiscale_snn": self.multiscale_snn,
                "snn_native_readout": self.snn_native_readout,
            }
            violated = sorted(name for name, enabled in required_contract.items() if not enabled)
            if violated:
                raise ValueError("V5 scientific contract requires: " + ", ".join(violated))
            if self.delayed_stat_channels or self.use_delayed_directional_moments:
                raise ValueError("V5 event-native decoding forbids broadcast delayed statistics")
        self.spike_rate_reg = float(spike_rate_reg)
        self.graph_l1_reg = float(graph_l1_reg)
        self.delay_entropy_reg = float(
            delay_smooth_reg if delay_entropy_reg is None else delay_entropy_reg
        )
        self.delay_gate_structure_reg = float(
            delay_gate_l1_reg if delay_gate_structure_reg is None else delay_gate_structure_reg
        )
        self.spatial_orthogonal_reg = float(spatial_orthogonal_reg)
        self.graph_prior_reg = float(graph_prior_reg)
        self.delay_alignment_reg = float(delay_alignment_reg)
        self.delay_anchor_kl_reg = float(delay_anchor_kl_reg)
        self.delay_anchor_kl_scale = 1.0
        self.expert_balance_reg = float(expert_balance_reg)
        self.expert_diversity_reg = float(expert_diversity_reg)
        pool_limit = max(1, min(int(temporal_pool_bins), self.timesteps))
        self.temporal_pool_bins = tuple(
            bins for bins in (1, 2, 4, 8, 16, 32, 64, 128) if bins <= pool_limit
        )
        self.tet_scales = tuple(
            sorted({int(scale) for scale in tet_scales if 0 < int(scale) <= self.timesteps})
        )
        duration = self._task_duration_seconds()
        self.cumulative_readout_seconds = tuple(
            sorted(
                {
                    float(seconds)
                    for seconds in cumulative_readout_seconds
                    if 0 < float(seconds) <= duration + 1e-6
                }
            )
        )

        if alignment_matrix is None:
            alignment = torch.eye(n_channels, dtype=torch.float32)
        else:
            alignment = torch.as_tensor(alignment_matrix, dtype=torch.float32)
            if alignment.shape != (n_channels, n_channels):
                raise ValueError("alignment_matrix must match [n_channels, n_channels]")
        self.register_buffer("alignment_matrix", alignment)

        graph_rate_hz = self.graph_timesteps / max(self._task_duration_seconds(), 1e-6)
        upper_hz = max(float(edges[1]) for edges in band_edges_hz)
        evidence_edges = delay_evidence_band_edges_hz or DEFAULT_DELAY_EVIDENCE_EDGES
        evidence_upper_hz = max(float(edges[1]) for edges in evidence_edges)
        required_upper_hz = max(upper_hz, evidence_upper_hz)
        if graph_rate_hz / 2.0 <= required_upper_hz + 1.0:
            raise ValueError(
                f"graph rate {graph_rate_hz:.3f} Hz has no anti-alias transition "
                f"above {required_upper_hz:.3f} Hz"
            )
        anti_alias_high_hz = graph_rate_hz / 2.0 - 1.0
        if self.freeze_shared_filterbank:
            if torch.as_tensor(band_edges_hz).shape != torch.as_tensor(
                evidence_edges
            ).shape or not torch.allclose(
                torch.as_tensor(band_edges_hz, dtype=torch.float32),
                torch.as_tensor(evidence_edges, dtype=torch.float32),
            ):
                raise ValueError(
                    "A shared frozen filterbank requires identical carrier and evidence bands"
                )
            self.filterbank = LearnableAnalyticFilterBank(
                self.sfreq,
                band_edges_hz,
                max_center_shift_hz=0.5,
                min_bandwidth_hz=1.0,
                transition_hz=0.5,
                max_high_hz=anti_alias_high_hz,
            )
        else:
            self.filterbank = LearnableAnalyticFilterBank(
                self.sfreq,
                band_edges_hz,
                max_high_hz=anti_alias_high_hz,
            )
        self.delay_evidence_filterbank = LearnableAnalyticFilterBank(
            self.sfreq,
            evidence_edges,
            max_center_shift_hz=0.5,
            min_bandwidth_hz=1.0,
            transition_hz=0.5,
            max_high_hz=anti_alias_high_hz,
        )
        if self.freeze_shared_filterbank:
            self.delay_evidence_filterbank.load_state_dict(
                self.filterbank.state_dict(), strict=True
            )
            for parameter in self.filterbank.parameters():
                parameter.requires_grad = False
        for parameter in self.delay_evidence_filterbank.parameters():
            parameter.requires_grad = False
        if self.freeze_shared_physical_basis:
            self.spatial = AnchoredSpatialProjection(
                n_channels,
                latent_nodes,
                max_deviation=0.0,
                identity_init=identity_spatial_projection,
                electrode_coordinates=electrode_coordinates,
                anchor_indices=spatial_anchor_indices,
            )
        else:
            self.spatial = BandSpecificAnchoredSpatialProjection(
                n_channels,
                latent_nodes,
                n_bands,
                rank=spatial_band_rank,
                max_deviation=spatial_max_deviation,
                max_band_deviation=spatial_band_deviation,
                identity_init=identity_spatial_projection,
                electrode_coordinates=electrode_coordinates,
                anchor_indices=spatial_anchor_indices,
            )
        self.evidence_spatial = AnchoredSpatialProjection(
            n_channels,
            latent_nodes,
            max_deviation=0.0,
            identity_init=identity_spatial_projection,
            electrode_coordinates=electrode_coordinates,
            anchor_indices=spatial_anchor_indices,
        )
        if self.freeze_shared_physical_basis:
            self.evidence_spatial.load_state_dict(self.spatial.state_dict(), strict=True)
        for parameter in self.evidence_spatial.parameters():
            parameter.requires_grad = False
        if self.freeze_shared_physical_basis:
            for parameter in self.spatial.parameters():
                parameter.requires_grad = False
        self.phase_confidence_raw = nn.Parameter(torch.zeros(n_bands))
        self.envelope_gain_raw = nn.Parameter(torch.full((n_bands,), -0.43275213))
        self.event_encoder = (
            PhaseDeltaEventEncoder(
                n_bands,
                latent_nodes,
                self.graph_timesteps,
                envelope_threshold=event_envelope_threshold,
                phase_confidence_threshold=event_phase_confidence_threshold,
                learnable_envelope_threshold=learnable_event_threshold,
                timing_intervention=event_timing_intervention,
                timing_seed=event_timing_seed,
            )
            if self.event_native_transport
            else None
        )
        self.synapse = DelayPhaseGraphSynapse(
            n_bands=n_bands,
            n_channels=latent_nodes,
            d_max=d_max,
            slow_d_max=slow_d_max,
            slow_downsample=slow_downsample,
            graph_sparsity=graph_sparsity,
            delay_temperature=delay_temperature,
            min_delay_temperature=min_delay_temperature,
            delay_gate_init=delay_gate_init,
            max_phase_residual=max_phase_residual,
            timestep_seconds=1.0
            / (self.graph_timesteps / max(self._task_duration_seconds(), 1e-6)),
            use_cross_band_routes=use_cross_band_routes,
            n_delay_experts=n_delay_experts,
            expert_topk=delay_expert_topk,
            context_dim=n_bands * latent_nodes,
            node_coordinates=self.spatial.node_coordinates,
            force_zero_delay=force_zero_delay,
            route_rank=route_rank,
            router_warmup_fraction=router_warmup_fraction,
            router_noise_scale=router_noise_scale,
            route_odds_threshold=route_odds_threshold,
            hurdle_min_bayes_factor=hurdle_min_bayes_factor,
            hurdle_route_temperature=hurdle_route_temperature,
            fixed_delay_steps=fixed_delay_steps,
            use_fixed_fold_posterior=freeze_delay_posterior,
            rejected_route_prior_floor=rejected_route_prior_floor,
            matched_transport_control=matched_transport_control,
            event_native_transport=self.event_native_transport,
            checkpoint_event_routes=checkpoint_event_routes,
        )
        # Every transport band is delayed before this non-negative, row-normalized
        # physiological aggregation. No classification feature can bypass delay.
        band_map = self._initial_band_map(
            classification_band_edges_hz,
            band_edges_hz,
        )
        self.classification_band_map_logits = nn.Parameter(
            torch.log(torch.expm1(band_map.clamp_min(1e-4)))
        )
        if self.preserve_transport_band_pairs:
            graph_features = (
                n_bands
                * n_bands
                * (
                    latent_nodes
                    + self.delayed_stat_channels
                    + (
                        DelayedDirectionalMoments.out_features
                        if self.use_delayed_directional_moments
                        else 0
                    )
                )
            )
        else:
            graph_features = n_classification_bands * n_classification_bands * latent_nodes
        self.covariance = BandLogCovarianceBranch(n_bands, latent_nodes, latent_nodes)
        self.delayed_statistics = (
            DelayedBandPairStatistics(latent_nodes, self.delayed_stat_channels)
            if self.preserve_transport_band_pairs and self.delayed_stat_channels > 0
            else None
        )
        self.delayed_directional_moments = (
            DelayedDirectionalMoments()
            if self.preserve_transport_band_pairs and self.use_delayed_directional_moments
            else None
        )
        self.covariance_scale_raw = nn.Parameter(torch.tensor(-1.5))
        self.register_buffer("train_route_rms", torch.tensor(1.0))
        self.register_buffer("train_route_gain_ready", torch.tensor(False))
        if snn_channels % 2:
            raise ValueError("snn_channels must be even for signed dual-population coding")
        self.current_encoder = nn.Sequential(
            nn.Conv1d(graph_features, snn_channels // 2, kernel_size=1, bias=False),
        )
        neuron_type = str(decoder_neuron).lower()
        neuron_class = (
            CLIFLayer if neuron_type == "clif" else PLIFLayer if neuron_type == "plif" else None
        )
        if neuron_class is None:
            raise ValueError("decoder_neuron must be 'clif' or 'plif'")
        if self.multiscale_snn:
            dynamics = neuron_type if self.decoder_is_spiking else "ann"
            self.lif = MultiTimescaleStateLayer(
                snn_channels,
                decays=decoder_timescales,
                threshold=lif_threshold,
                neuron_type=dynamics,
                causal_channel_norm=causal_channel_norm,
            )
            self.spiking_blocks = nn.ModuleList(
                MultiTimescaleSEWBlock(
                    snn_channels,
                    decays=decoder_timescales,
                    threshold=lif_threshold,
                    causal_channel_norm=causal_channel_norm,
                    connect_function=sew_connect_function,
                    neuron_type=dynamics,
                )
                for _ in range(max(0, int(decoder_layers) - 1))
            )
            self.output_lif = (
                MultiTimescaleStateLayer(
                    snn_channels,
                    decays=decoder_timescales,
                    threshold=lif_threshold,
                    neuron_type=dynamics,
                    causal_channel_norm=causal_channel_norm,
                )
                if len(self.spiking_blocks) > 0
                else None
            )
        else:
            if not self.decoder_is_spiking:
                raise ValueError("matched_ann requires multiscale_snn=True")
            neuron_kwargs = {
                "decay": tau_mem,
                "threshold": lif_threshold,
                "channels": snn_channels,
                "learnable_decay": True,
                "causal_channel_norm": causal_channel_norm,
            }
            if neuron_class is CLIFLayer:
                neuron_kwargs["membrane_norm_groups"] = snn_membrane_norm_groups
            self.lif = neuron_class(**neuron_kwargs)
            self.spiking_blocks = nn.ModuleList(
                SEWCLIFBlock(
                    snn_channels,
                    decay=tau_mem,
                    threshold=lif_threshold,
                    membrane_norm_groups=snn_membrane_norm_groups,
                    causal_channel_norm=causal_channel_norm,
                    connect_function=sew_connect_function,
                    neuron_type=neuron_type,
                )
                for _ in range(max(0, int(decoder_layers) - 1))
            )
            # SEW-ADD residual states are not binary spikes. Re-spike the
            # accumulated state so exported spikes and membrane agree.
            self.output_lif = (
                neuron_class(**neuron_kwargs) if len(self.spiking_blocks) > 0 else None
            )

        if self.snn_native_readout:
            self.temporal_projection = None
            self.spike_accumulator = CausalSpikeAccumulator(
                snn_channels,
                temporal_pool_channels,
                decays=snn_readout_decays,
            )
            self.feature = nn.Identity()
            self.readout = nn.Linear(self.spike_accumulator.out_features, n_classes)
            self.tet_readout = None
        else:
            self.spike_accumulator = None
            self.temporal_projection = nn.Conv1d(
                snn_channels, temporal_pool_channels, kernel_size=1, bias=False
            )
            pooled_features = temporal_pool_channels * sum(self.temporal_pool_bins)
            pooled_features *= 4 if self.use_membrane_readout else 2
            pooled_features += 1
            self.feature = nn.Sequential(
                nn.Linear(pooled_features, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ELU(),
                nn.Dropout(0.25),
            )
            self.readout = nn.Linear(hidden_channels, n_classes)
            tet_features = temporal_pool_channels * (4 if self.use_membrane_readout else 2)
            self.tet_readout = (
                nn.Linear(tet_features, n_classes) if self.cumulative_readout_seconds else None
            )

    @staticmethod
    def _initial_band_map(
        classification_edges: list[list[float]],
        transport_edges: list[list[float]],
    ) -> torch.Tensor:
        """Build an overlap-initialized map from transport to classification bands."""

        classification = torch.as_tensor(classification_edges, dtype=torch.float32)
        transport = torch.as_tensor(transport_edges, dtype=torch.float32)
        if classification.ndim != 2 or classification.shape[1] != 2:
            raise ValueError("classification_band_edges_hz must have shape [bands, 2]")
        if transport.ndim != 2 or transport.shape[1] != 2:
            raise ValueError("band_edges_hz must have shape [bands, 2]")
        lower = torch.maximum(classification[:, None, 0], transport[None, :, 0])
        upper = torch.minimum(classification[:, None, 1], transport[None, :, 1])
        overlap = (upper - lower).clamp_min(0.0)
        empty = overlap.sum(dim=1) <= 0
        if empty.any():
            classification_centres = classification.mean(dim=1)
            transport_centres = transport.mean(dim=1)
            nearest = (
                (classification_centres[:, None] - transport_centres[None, :]).abs().argmin(dim=1)
            )
            overlap[empty] = 0.0
            overlap[empty, nearest[empty]] = 1.0
        return overlap / overlap.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def classification_band_map(self) -> torch.Tensor:
        weights = F.softplus(self.classification_band_map_logits)
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def set_training_progress(self, progress: float, anneal_lag: bool = True) -> None:
        self.synapse.set_training_progress(progress, anneal_lag=anneal_lag)
        if anneal_lag:
            self.delay_anchor_kl_scale = max(0.0, 1.0 - float(progress))

    def set_router_training_progress(self, progress: float) -> None:
        self.synapse.set_router_training_progress(progress)

    def begin_task_training(self) -> None:
        self.synapse.begin_task_training()
        self.delay_anchor_kl_scale = 1.0

    def load_representation_state(self, state: dict[str, torch.Tensor]) -> None:
        """Load common weights without replacing fold-local scientific inputs."""

        preserved_names = (
            "route_mask",
            "edge_prior",
            "fold_route_prior",
            "fold_positive_delay_prior",
            "fold_fraction_target",
            "fold_evidence_ready",
            "fast_evidence_ema",
            "slow_evidence_ema",
            "coarse_reliability_ema",
            "route_reliability_ema",
            "connectivity_prior_ema",
            "phase_residual_enabled",
        )
        preserved = {name: getattr(self.synapse, name).detach().clone() for name in preserved_names}
        preserved_event = None
        if self.event_encoder is not None:
            preserved_event = {
                "envelope_threshold_raw": self.event_encoder.envelope_threshold_raw.detach().clone(),
                "envelope_threshold_ready": self.event_encoder.envelope_threshold_ready.detach().clone(),
            }
        self.load_state_dict(state, strict=True)
        for name, value in preserved.items():
            getattr(self.synapse, name).copy_(value)
        if preserved_event is not None and self.event_encoder is not None:
            for name, value in preserved_event.items():
                getattr(self.event_encoder, name).copy_(value)

    def set_training_stage(self, stage: str) -> None:
        posterior_ids = {id(parameter) for parameter in self.synapse.posterior_parameters()}
        residual_ids = {id(parameter) for parameter in self.synapse.residual_delay_parameters()}
        matched_transport_ids = (
            {id(parameter) for parameter in self.synapse.parameters()}
            if self.matched_transport_control
            else set()
        )
        evidence_ids = {
            id(parameter)
            for module in (self.delay_evidence_filterbank, self.evidence_spatial)
            for parameter in module.parameters()
        }
        if self.freeze_shared_physical_basis:
            evidence_ids.update(id(parameter) for parameter in self.spatial.parameters())
        if self.freeze_shared_filterbank:
            evidence_ids.update(id(parameter) for parameter in self.filterbank.parameters())
        for parameter in self.parameters():
            if stage == "representation":
                parameter.requires_grad = (
                    id(parameter) not in residual_ids | evidence_ids | matched_transport_ids
                )
            elif stage == "joint":
                parameter.requires_grad = id(
                    parameter
                ) not in evidence_ids | matched_transport_ids and not (
                    (self.freeze_delay_posterior or self.freeze_delay_after_pretrain)
                    and id(parameter) in posterior_ids
                )
            else:
                raise ValueError(f"Unknown DPC-SNN training stage: {stage}")

    def set_delay_evidence_updates(self, enabled: bool) -> None:
        self.synapse.set_evidence_updates(enabled)

    def capture_delay_anchor(self) -> None:
        self.synapse.capture_delay_anchor()

    def load_fold_local_evidence_prior(
        self,
        route_probability: torch.Tensor,
        positive_delay_probability: torch.Tensor,
        fractional_delay_target: torch.Tensor | None = None,
        connectivity_prior: torch.Tensor | None = None,
    ) -> None:
        self.synapse.load_fold_local_evidence_prior(
            route_probability,
            positive_delay_probability,
            fractional_delay_target,
            connectivity_prior,
        )

    @torch.no_grad()
    def set_train_fitted_route_rms(self, value: float | torch.Tensor) -> None:
        fitted = torch.as_tensor(
            value, device=self.train_route_rms.device, dtype=self.train_route_rms.dtype
        )
        if fitted.numel() != 1 or not torch.isfinite(fitted) or float(fitted) <= 0.0:
            raise ValueError("Training-fitted delayed-current RMS must be finite and positive")
        self.train_route_rms.copy_(fitted.reshape(()).clamp_min(1e-8))
        self.train_route_gain_ready.fill_(True)

    def delay_pretraining_parameters(self) -> list[nn.Parameter]:
        # Fold-local evidence supervises route, conditional lag, and continuous
        # fractional offset before task labels train the expert router.
        return self.synapse.delay_pretraining_parameters()

    def delay_parameters(self) -> list[nn.Parameter]:
        return self.synapse.delay_parameters()

    def residual_delay_parameters(self) -> list[nn.Parameter]:
        return self.synapse.residual_delay_parameters()

    def set_calibration_mode(self, mode: str) -> int:
        evidence_ids = {
            id(parameter)
            for module in (self.delay_evidence_filterbank, self.evidence_spatial)
            for parameter in module.parameters()
        }
        mode = mode.lower()
        for parameter in self.parameters():
            parameter.requires_grad = mode == "full" and id(parameter) not in evidence_ids
        if mode == "full":
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        if mode == "readout":
            modules = [self.feature, self.readout]
            if self.spike_accumulator is not None:
                modules.append(self.spike_accumulator)
            selected = [p for module in modules for p in module.parameters()]
        elif mode in {"delay", "phase_delay"}:
            selected = self.synapse.delay_parameters()
            if mode == "phase_delay":
                selected.append(self.phase_confidence_raw)
            else:
                phase_ids = {
                    id(self.synapse.phase_pref),
                    id(self.synapse.phase_pref_target),
                    id(self.synapse.phase_pref_source),
                    id(self.synapse.phase_pref_rank),
                }
                selected = [parameter for parameter in selected if id(parameter) not in phase_ids]
        else:
            raise ValueError(f"Unknown calibration mode: {mode}")
        for parameter in selected:
            parameter.requires_grad = True
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def delay_pretraining_loss(self, aux: dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            aux["route_prior_loss"]
            + aux["positive_delay_kl_loss"]
            + aux["fractional_delay_loss"]
            + 0.1 * aux["edge_prior_loss"]
            + 0.1 * aux["delay_confidence_prior_loss"]
            + 0.1 * aux["phase_evidence_loss"]
            + 0.01 * aux["delay_gate_structure_loss"]
        )

    def optimizer_parameter_groups(self, lr: float, weight_decay: float) -> list[dict[str, object]]:
        posterior_parameters = self.synapse.residual_delay_parameters()
        posterior_ids = {id(parameter) for parameter in posterior_parameters}
        router_parameters = list(self.synapse.expert_router.parameters())
        router_ids = {id(parameter) for parameter in router_parameters}
        other_parameters = [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in posterior_ids | router_ids
        ]
        return [
            {"params": other_parameters, "lr": lr, "weight_decay": weight_decay},
            {"params": router_parameters, "lr": lr, "weight_decay": weight_decay},
            {
                "params": posterior_parameters,
                "lr": lr * self.delay_lr_multiplier,
                "weight_decay": weight_decay,
            },
        ]

    def _task_duration_seconds(self) -> float:
        if self.task_tmax is not None:
            return max(1.0 / self.sfreq, self.task_tmax - self.task_tmin)
        return max(1.0 / self.sfreq, self.timesteps / self.sfreq)

    def _apply_alignment_and_reference(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("ij,njt->nit", self.alignment_matrix.to(x.dtype), x)
        if self.training and self.reference_augmentation_prob > 0:
            use_car = (
                torch.rand(x.shape[0], 1, 1, device=x.device) < self.reference_augmentation_prob
            ).to(x.dtype)
            x = x - use_car * x.mean(dim=1, keepdim=True)
        return x

    @staticmethod
    def _resize_complex(x: torch.Tensor, steps: int) -> torch.Tensor:
        shape = x.shape
        real = F.interpolate(
            x.real.reshape(-1, 1, shape[-1]), size=steps, mode="linear", align_corners=False
        )
        imag = F.interpolate(
            x.imag.reshape(-1, 1, shape[-1]), size=steps, mode="linear", align_corners=False
        )
        return torch.complex(real, imag).reshape(*shape[:-1], steps)

    @staticmethod
    def _baseline_scale(latent: torch.Tensor, baseline_end: int) -> torch.Tensor:
        baseline = latent[..., :baseline_end].abs().mean(dim=-1, keepdim=True)
        # A spatial filter can nearly cancel one baseline node. A relative floor
        # prevents that node from amplifying task traffic while preserving
        # between-trial and between-band ERD/ERS amplitude.
        node_reference = baseline.detach().amax(dim=-2, keepdim=True)
        floor = (0.05 * node_reference).clamp_min(1e-4)
        return torch.maximum(baseline, floor)

    def _task_bounds(self, n_time: int) -> tuple[int, int, int]:
        baseline_end = int(round((self.task_tmin - self.epoch_tmin) * self.sfreq))
        start = max(0, min(n_time - 1, baseline_end))
        if self.task_tmax is None:
            end = n_time
        else:
            end = int(round((self.task_tmax - self.epoch_tmin) * self.sfreq))
            end = max(start + 1, min(n_time, end))
        return max(0, baseline_end), start, end

    def _carrier_features_from_analytic(
        self, analytic: torch.Tensor
    ) -> dict[str, torch.Tensor | int | None]:
        """Apply the shared physical basis and model-matched task baseline."""

        latent = self.spatial(analytic)
        baseline_end, start, end = self._task_bounds(latent.shape[-1])
        baseline = self._baseline_scale(latent, baseline_end) if baseline_end > 0 else None
        latent = latent[..., start:end]
        covariance_input = latent.real if baseline is None else latent.real / baseline
        covariance_context = self.covariance(covariance_input)
        latent = self._resize_complex(latent, self.graph_timesteps)
        amplitude_graph = latent.abs()
        if baseline is None:
            carrier = latent.real
            normalized_amplitude = amplitude_graph
            raw_envelope = torch.log1p(amplitude_graph)
        else:
            normalized_amplitude = amplitude_graph / baseline
            carrier = latent.real / baseline
            raw_envelope = torch.log(normalized_amplitude.clamp_min(1e-4))
        envelope_gain = F.softplus(self.envelope_gain_raw)[None, :, None, None]
        # Event thresholds are fitted in raw log-envelope units.  Letting a
        # trainable gain alter event count would invalidate that fold-local
        # threshold; analog V4 paths retain their original gain semantics.
        envelope = raw_envelope if self.event_native_transport else envelope_gain * raw_envelope
        threshold = F.softplus(self.phase_confidence_raw)[None, :, None, None]
        threshold = threshold.clamp_min(1e-3)
        confidence = normalized_amplitude / (normalized_amplitude + threshold)
        if not self.use_phase_confidence:
            confidence = torch.ones_like(confidence)
        return {
            "carrier": carrier,
            "envelope": envelope,
            "confidence": confidence,
            "phase": torch.angle(latent),
            "covariance_context": covariance_context,
            "baseline_end": baseline_end,
            "start": start,
            "end": end,
            "envelope_gain": envelope_gain,
        }

    @torch.no_grad()
    def event_envelope_deltas(self, x: torch.Tensor) -> torch.Tensor:
        """Return threshold-independent envelope deltas for train-fold fitting."""

        if self.event_encoder is None:
            raise RuntimeError("event threshold fitting requires event-native transport")
        aligned = self._apply_alignment_and_reference(x)
        features = self._carrier_features_from_analytic(self.filterbank(aligned))
        envelope = features["envelope"]
        if not isinstance(envelope, torch.Tensor):
            raise RuntimeError("carrier feature extraction did not return an envelope")
        return (envelope[..., 1:] - envelope[..., :-1]).abs()

    @torch.no_grad()
    def set_train_fitted_event_thresholds(self, thresholds: torch.Tensor) -> None:
        if self.event_encoder is None:
            raise RuntimeError("this model has no event encoder")
        self.event_encoder.set_envelope_thresholds(thresholds)

    def _continuous_features(
        self,
        x: torch.Tensor | None,
        amplitude: torch.Tensor | None,
        phase: torch.Tensor | None,
        delay_evidence_x: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if x is not None:
            aligned_x = self._apply_alignment_and_reference(x)
            analytic = self.filterbank(aligned_x)
            if delay_evidence_x is None:
                aligned_evidence = aligned_x
            else:
                aligned_evidence = torch.einsum(
                    "ij,njt->nit",
                    self.alignment_matrix.to(delay_evidence_x.dtype),
                    delay_evidence_x,
                )
            evidence_analytic = self.delay_evidence_filterbank(aligned_evidence)
        elif amplitude is not None and phase is not None:
            analytic = torch.polar(amplitude, phase)
            analytic = torch.einsum(
                "ij,nbjt->nbit", self.alignment_matrix.to(analytic.dtype), analytic
            )
            evidence_analytic = None
            aligned_x = analytic.real.mean(dim=1)
        else:
            raise KeyError("DPC-SNN V3 requires raw x or amplitude/phase")

        carrier_features = self._carrier_features_from_analytic(analytic)
        carrier = carrier_features["carrier"]
        envelope = carrier_features["envelope"]
        confidence = carrier_features["confidence"]
        phase_graph = carrier_features["phase"]
        covariance_context = carrier_features["covariance_context"]
        baseline_end = int(carrier_features["baseline_end"])
        start = int(carrier_features["start"])
        end = int(carrier_features["end"])
        envelope_gain = carrier_features["envelope_gain"]
        if not all(
            isinstance(value, torch.Tensor)
            for value in (
                carrier,
                envelope,
                confidence,
                phase_graph,
                covariance_context,
                envelope_gain,
            )
        ):
            raise RuntimeError("carrier feature extraction returned invalid tensors")

        evidence_phase = None
        evidence_confidence = None
        evidence_frequencies = None
        evidence_carrier = None
        evidence_envelope = None
        if evidence_analytic is not None:
            # Preserve evidence-node identity independently of the trainable
            # classification projection.
            evidence_latent = self.evidence_spatial(evidence_analytic)
            if baseline_end > 0:
                evidence_baseline = self._baseline_scale(evidence_latent, baseline_end)
            else:
                evidence_baseline = None
            evidence_latent = evidence_latent[..., start:end]
            evidence_latent = self._resize_complex(evidence_latent, self.graph_timesteps)
            evidence_amplitude = evidence_latent.abs()
            if evidence_baseline is not None:
                evidence_amplitude = evidence_amplitude / evidence_baseline
                evidence_carrier = evidence_latent.real / evidence_baseline
                evidence_envelope = torch.log(evidence_amplitude.clamp_min(1e-4))
            else:
                evidence_carrier = evidence_latent.real
                evidence_envelope = torch.log1p(evidence_amplitude)
            evidence_threshold = (
                evidence_amplitude.detach().median(dim=-1, keepdim=True).values.clamp_min(1e-3)
            )
            evidence_confidence = evidence_amplitude / (evidence_amplitude + evidence_threshold)
            evidence_phase = torch.angle(evidence_latent)
            evidence_frequencies, _, _ = self.delay_evidence_filterbank.band_parameters()

        if not self.use_covariance:
            covariance_context = torch.zeros_like(covariance_context)
        band_center_hz, _, _ = self.filterbank.band_parameters()
        event_features: dict[str, torch.Tensor] | None = None
        transport_carrier = carrier
        transport_envelope = envelope
        if self.event_encoder is not None:
            event_features = self.event_encoder(carrier, envelope, confidence)
            transport_carrier = event_features["phase_events"]
            transport_envelope = event_features["envelope_events"]
        graph_current, syn_aux = self.synapse(
            transport_carrier,
            phase_graph,
            confidence,
            envelope=transport_envelope,
            band_frequencies_hz=band_center_hz,
            evidence_signal=evidence_carrier,
            evidence_envelope=evidence_envelope,
            context=covariance_context,
            evidence_phase=evidence_phase,
            evidence_confidence=evidence_confidence,
            evidence_frequencies_hz=evidence_frequencies,
        )
        syn_aux["phase_confidence_mean"] = confidence.mean()
        syn_aux["envelope_gain_mean"] = envelope_gain.mean()
        syn_aux["baseline_samples"] = carrier.new_tensor(float(baseline_end))
        syn_aux["signed_erd_ers_mean"] = envelope.mean()
        if event_features is not None:
            syn_aux["phase_event_density"] = event_features["phase_event_density"]
            syn_aux["envelope_event_density"] = event_features["envelope_event_density"]
            syn_aux["event_envelope_threshold"] = event_features["envelope_threshold"]
            syn_aux["event_envelope_threshold_ready"] = event_features["envelope_threshold_ready"]
        else:
            syn_aux["phase_event_density"] = carrier.new_zeros(())
            syn_aux["envelope_event_density"] = carrier.new_zeros(())
            syn_aux["event_envelope_threshold"] = carrier.new_zeros(0)
            syn_aux["event_envelope_threshold_ready"] = carrier.new_tensor(False)
        syn_aux["delay_evidence_geometry"] = carrier.new_tensor(
            [self.delay_evidence_filterbank.n_bands, self.spatial.n_nodes]
        )
        return graph_current, covariance_context, syn_aux

    def _temporal_pyramid(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_projection(x)
        pooled = []
        for bins in self.temporal_pool_bins:
            mean = F.adaptive_avg_pool1d(x, bins)
            second = F.adaptive_avg_pool1d(x.square(), bins)
            std = (second - mean.square()).clamp_min(1e-6).sqrt()
            pooled.extend((mean.flatten(1), std.flatten(1)))
        return torch.cat(pooled, dim=1)

    def _cumulative_logits(self, spikes: torch.Tensor, membrane: torch.Tensor) -> torch.Tensor:
        if self.temporal_projection is None:
            raise RuntimeError("temporal-statistics readout is disabled")
        spike_features = self.temporal_projection(spikes)
        membrane_features = self.temporal_projection(torch.tanh(membrane))
        outputs = []
        graph_rate_hz = spikes.shape[-1] / max(self._task_duration_seconds(), 1e-6)
        for seconds in self.cumulative_readout_seconds:
            stop = min(spikes.shape[-1], max(1, int(round(seconds * graph_rate_hz))))
            spike_prefix = spike_features[..., :stop]
            parts = [
                spike_prefix.mean(dim=-1),
                spike_prefix.var(dim=-1, unbiased=False).clamp_min(1e-6).sqrt(),
            ]
            if self.use_membrane_readout:
                membrane_prefix = membrane_features[..., :stop]
                parts.extend(
                    (
                        membrane_prefix.mean(dim=-1),
                        membrane_prefix.var(dim=-1, unbiased=False).clamp_min(1e-6).sqrt(),
                    )
                )
            if self.tet_readout is not None:
                outputs.append(self.tet_readout(torch.cat(parts, dim=1)))
        if not outputs:
            return spikes.new_zeros(spikes.shape[0], 0, self.n_classes)
        return torch.stack(outputs, dim=1)

    def _cumulative_capture_steps(self, n_time: int) -> tuple[int, ...]:
        graph_rate_hz = n_time / max(self._task_duration_seconds(), 1e-6)
        return tuple(
            sorted(
                {
                    min(n_time, max(1, int(round(seconds * graph_rate_hz))))
                    for seconds in self.cumulative_readout_seconds
                }
            )
        )

    def forward(
        self,
        x: torch.Tensor | None = None,
        amplitude: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
        delay_evidence_x: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        graph_current, covariance_context, syn_aux = self._continuous_features(
            x,
            amplitude,
            phase,
            delay_evidence_x=delay_evidence_x,
        )
        covariance_scale = F.softplus(self.covariance_scale_raw)
        # Covariance can condition delayed traffic but cannot bypass it.
        covariance_context_nodes = covariance_context.reshape(
            covariance_context.shape[0], self.n_bands, -1
        )
        covariance_modulation = 1.0 + covariance_scale * torch.tanh(
            covariance_context_nodes[:, :, None, :, None]
        )
        delayed_transport = graph_current * covariance_modulation
        band_map = self.classification_band_map()
        if self.preserve_transport_band_pairs:
            pair_features = [delayed_transport]
            delayed_pair_statistics = None
            if self.delayed_statistics is not None:
                delayed_pair_statistics = self.delayed_statistics(delayed_transport)
                broadcast_statistics = delayed_pair_statistics[..., None].expand(
                    *delayed_pair_statistics.shape, delayed_transport.shape[-1]
                )
                pair_features.append(broadcast_statistics)
            delayed_directional_moments = None
            if self.delayed_directional_moments is not None:
                delayed_directional_moments = self.delayed_directional_moments(delayed_transport)
                broadcast_moments = delayed_directional_moments[..., None].expand(
                    *delayed_directional_moments.shape, delayed_transport.shape[-1]
                )
                pair_features.append(broadcast_moments)
            fused = torch.cat(pair_features, dim=3).flatten(1, 3)
        else:
            classified_transport = torch.einsum(
                "ca,nabkt,db->ncdkt",
                band_map,
                delayed_transport,
                band_map,
            )
            fused = classified_transport.flatten(1, 3)
            delayed_pair_statistics = None
            delayed_directional_moments = None
        if fused.shape[-1] != self.timesteps:
            if self.event_native_transport:
                raise RuntimeError("V5 event traffic cannot be resampled after delay transport")
            fused = F.interpolate(fused, size=self.timesteps, mode="linear", align_corners=False)
        # The gain is fitted once on the inner-training fold. Trial-to-trial
        # route magnitude therefore remains observable and validation/test can
        # never fit their own scale.
        route_rms = fused.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
        normalized_fused = fused / self.train_route_rms.to(fused).clamp_min(1e-8)
        signed_current = self.current_encoder(normalized_fused)
        current = (
            torch.cat((F.relu(signed_current), F.relu(-signed_current)), dim=1) * self.current_gain
        )
        hidden_spikes, membrane = self.lif(current)
        binary_spike_layers = [hidden_spikes]
        residual_activation = hidden_spikes
        for block in self.spiking_blocks:
            residual_activation, branch_spikes, _ = block.forward_with_state(residual_activation)
            binary_spike_layers.append(branch_spikes)
        if self.output_lif is not None:
            hidden_spikes, membrane = self.output_lif(residual_activation)
            binary_spike_layers.append(hidden_spikes)
        else:
            hidden_spikes = residual_activation
        binary_spike_layers_tensor = torch.stack(binary_spike_layers, dim=1)
        if self.spike_accumulator is not None:
            capture_steps = self._cumulative_capture_steps(hidden_spikes.shape[-1])
            readout_features, cumulative_features = self.spike_accumulator(
                hidden_spikes, capture_steps=capture_steps
            )
            logits = self.readout(readout_features)
            cumulative_logits = self.readout(cumulative_features)
            temporal_bins = current.new_zeros(0)
        else:
            pooled_parts = [self._temporal_pyramid(hidden_spikes)]
            if self.use_membrane_readout:
                pooled_parts.append(self._temporal_pyramid(torch.tanh(membrane)))
            delayed_energy = route_rms.flatten(1).log()
            readout_features = torch.cat((*pooled_parts, delayed_energy), dim=1)
            logits = self.readout(self.feature(readout_features))
            cumulative_logits = self._cumulative_logits(hidden_spikes, membrane)
            temporal_bins = current.new_tensor(self.temporal_pool_bins)
        aux = {
            "input_spikes": hidden_spikes,
            "spike_prob": torch.sigmoid(current),
            "current": current,
            "route_rms": route_rms,
            "route_power_sum": fused.square().sum(),
            "route_element_count": fused.new_tensor(fused.numel()),
            "train_route_rms": self.train_route_rms,
            "train_route_gain_ready": self.train_route_gain_ready,
            "hidden_spikes": hidden_spikes,
            "binary_spike_layers": binary_spike_layers_tensor,
            "decoder_is_spiking": current.new_tensor(self.decoder_is_spiking),
            "snn_native_readout": current.new_tensor(self.snn_native_readout),
            "snn_readout_features": readout_features,
            "sew_residual_activation": residual_activation,
            "membrane": membrane,
            "cumulative_logits": cumulative_logits,
            # Retain the old key for result readers while changing the loss
            # semantics from local windows to cumulative prefixes.
            "tet_logits": cumulative_logits,
            "cumulative_readout_seconds": current.new_tensor(self.cumulative_readout_seconds),
            "temporal_pyramid_bins": temporal_bins,
            "covariance_context": covariance_context,
            "covariance_scale": covariance_scale,
            "classification_band_map": band_map,
            "delayed_pair_statistics": (
                delayed_pair_statistics
                if delayed_pair_statistics is not None
                else current.new_zeros(current.shape[0], 0)
            ),
            "delayed_directional_moments": (
                delayed_directional_moments
                if delayed_directional_moments is not None
                else current.new_zeros(current.shape[0], 0)
            ),
            "transport_pairs_preserved_to_snn": current.new_tensor(
                bool(self.preserve_transport_band_pairs)
            ),
            "transport_band_count": current.new_tensor(self.n_bands),
            "classification_band_count": current.new_tensor(self.n_classification_bands),
            "spatial_orthogonality": self.spatial.orthogonality_loss(),
            **syn_aux,
        }
        return {"logits": logits, "aux": aux}

    def regularization_loss(self, aux: dict[str, torch.Tensor]) -> torch.Tensor:
        binary_layers = aux.get("binary_spike_layers", aux["hidden_spikes"][:, None])
        if self.decoder_is_spiking:
            layer_spike_rate = binary_layers.mean(dim=(0, 2, 3))
            reg = self.spike_rate_reg * (layer_spike_rate - self.spike_rate_target).square().mean()
        else:
            reg = binary_layers.new_zeros(())
        reg = reg + self.graph_l1_reg * aux["edge_weight"].abs().mean()
        # Posterior uncertainty is controlled by forward KL to evidence and the
        # lag-temperature schedule, not by an entropy-collapse reward.
        reg = reg + self.delay_gate_structure_reg * aux["delay_gate_structure_loss"]
        reg = reg + self.spatial_orthogonal_reg * aux["spatial_orthogonality"]
        reg = reg + self.graph_prior_reg * aux["edge_prior_loss"]
        reg = reg + self.delay_alignment_reg * aux["delay_alignment_loss"]
        reg = reg + self.delay_anchor_kl_reg * self.delay_anchor_kl_scale * aux["delay_anchor_kl"]
        reg = reg + self.expert_balance_reg * aux["expert_balance_loss"]
        reg = reg + self.expert_diversity_reg * aux["expert_diversity_loss"]
        return reg

    def _spatial_weight(self) -> torch.Tensor:
        weight = self.spatial.weight().abs()
        if weight.ndim == 2:
            weight = weight[None].expand(self.n_bands, -1, -1)
        return weight

    def _sensor_matrix(self, latent: torch.Tensor) -> torch.Tensor:
        spatial = self._spatial_weight()
        return torch.einsum("aki,abkl,blj->abij", spatial, latent, spatial)

    def _sensor_delay(self, latent_delay: torch.Tensor) -> torch.Tensor:
        spatial = self._spatial_weight()
        numerator = torch.einsum("aki,abkl,blj->abij", spatial, latent_delay, spatial)
        mass = spatial.sum(dim=1)
        denominator = (mass[:, None, :, None] * mass[None, :, None, :]).clamp_min(1e-6)
        return numerator / denominator

    def _sensor_weighted_delay(
        self, latent_delay: torch.Tensor, route_strength: torch.Tensor
    ) -> torch.Tensor:
        spatial = self._spatial_weight()
        strength = route_strength.clamp_min(0.0)
        numerator = torch.einsum("aki,abkl,blj->abij", spatial, latent_delay * strength, spatial)
        denominator = torch.einsum("aki,abkl,blj->abij", spatial, strength, spatial).clamp_min(1e-6)
        return numerator / denominator

    def export_learned_parameters(self) -> dict[str, torch.Tensor]:
        edge_latent = self.synapse.edge_weight() * self.synapse.delay_gate()
        delay_latent = self.synapse.learned_delay()
        delay_map_latent = self.synapse.learned_delay_map()
        delay_seconds_latent = self.synapse.learned_delay_seconds()
        slow_delay_latent = self.synapse.learned_slow_delay()
        route_strength = self.synapse.delay_gate()
        center, width, gain = self.filterbank.band_parameters()
        evidence_center, evidence_width, _ = self.delay_evidence_filterbank.band_parameters()
        return {
            "edge_weight": self._sensor_matrix(edge_latent).detach().cpu(),
            "edge_weight_latent": edge_latent.detach().cpu(),
            "delay": self._sensor_weighted_delay(delay_latent, route_strength).detach().cpu(),
            "delay_unweighted": self._sensor_delay(delay_latent).detach().cpu(),
            "delay_latent": delay_latent.detach().cpu(),
            "delay_map": self._sensor_weighted_delay(delay_map_latent, route_strength)
            .detach()
            .cpu(),
            "delay_map_unweighted": self._sensor_delay(delay_map_latent).detach().cpu(),
            "delay_map_latent": delay_map_latent.detach().cpu(),
            "delay_seconds": self._sensor_weighted_delay(delay_seconds_latent, route_strength)
            .detach()
            .cpu(),
            "delay_seconds_latent": delay_seconds_latent.detach().cpu(),
            "slow_delay": self._sensor_weighted_delay(slow_delay_latent, route_strength)
            .detach()
            .cpu(),
            "slow_delay_latent": slow_delay_latent.detach().cpu(),
            "phase_pref": self.synapse.effective_phase_pref().detach().cpu(),
            "delay_prob": self.synapse.delay_prob().detach().cpu(),
            "delay_null_probability": self.synapse.delay_null_probability().detach().cpu(),
            "online_route_log_odds": self.synapse.online_route_log_odds().detach().cpu(),
            "fast_carrier_prob": self.synapse.fast_delay_prob().detach().cpu(),
            "slow_delay_prob": self.synapse.slow_delay_prob().detach().cpu(),
            "delay_gate": self.synapse.delay_gate().detach().cpu(),
            "delay_confidence": self.synapse.delay_confidence(self.synapse.delay_prob())
            .detach()
            .cpu(),
            "edge_existence": self.synapse.edge_existence().detach().cpu(),
            "edge_selection": self.synapse.effective_edge_selection_expert().mean(0).detach().cpu(),
            "fast_fraction": self.synapse.fast_fraction().detach().cpu(),
            "slow_fraction": self.synapse.slow_fraction().detach().cpu(),
            "coarse_reliability": self.synapse.coarse_reliability_ema.detach().cpu(),
            "spatial_projection": self._spatial_weight().detach().cpu(),
            "evidence_spatial_projection": self.evidence_spatial.weight().detach().cpu(),
            "band_center_hz": center.detach().cpu(),
            "band_width_hz": width.detach().cpu(),
            "band_gain": gain.detach().cpu(),
            "evidence_band_center_hz": evidence_center.detach().cpu(),
            "evidence_band_width_hz": evidence_width.detach().cpu(),
            "envelope_gain": F.softplus(self.envelope_gain_raw).detach().cpu(),
            "architecture_version": self.architecture_version,
            "event_native_transport": self.event_native_transport,
            "decoder_mode": self.decoder_mode,
            "snn_native_readout": self.snn_native_readout,
        }
