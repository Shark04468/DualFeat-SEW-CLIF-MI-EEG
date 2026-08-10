"""Model factory."""

from __future__ import annotations

from typing import Any

from .dasp_snn_v62 import DASPSNNV62
from .dpc_snn import DPCSNN
from .eegnet import EEGNet
from .graph_snn import GraphSNNNoDelay
from .mechanism_controls import ComplexValuedCNN, DelayGraphANN, DilatedTCN, PhaseGatedGNN
from .v8_accuracy_first import V8AccuracyFirstModel
from .vanilla_snn import VanillaSNN


def build_model(name: str, cfg: dict[str, Any]) -> Any:
    name = name.lower()
    n_channels = int(cfg.get("n_channels", 22))
    n_classes = int(cfg.get("n_classes", 4))
    n_bands = int(cfg.get("n_bands", 2))
    timesteps = int(cfg.get("timesteps", 16))
    if name in {"v8_accuracy_first", "dpc_snn_v8_accuracy_first"}:
        return V8AccuracyFirstModel(
            n_channels=n_channels,
            n_classes=n_classes,
            channel_names=cfg.get(
                "channel_names",
                [
                    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3",
                    "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz",
                    "CP2", "CP4", "P1", "Pz", "P2", "POz",
                ],
            ),
            electrode_coordinates=cfg.get("electrode_coordinates"),
            n_bands=int(cfg.get("n_bands", 12)),
            n_latent_nodes=int(cfg.get("n_latent_nodes", 16)),
            band_edges_hz=cfg.get(
                "band_edges_hz",
                [
                    [6, 8], [8, 10], [10, 12], [12, 14], [14, 17], [17, 20],
                    [20, 23], [23, 26], [26, 30], [30, 34], [34, 37], [37, 40],
                ],
            ),
            sfreq=float(cfg.get("sfreq", 250.0)),
            epoch_tmin=float(cfg.get("epoch_tmin", -1.0)),
            task_tmin=float(cfg.get("task_tmin", 0.0)),
            task_tmax=float(cfg.get("task_tmax", 4.0)),
            analytic_taps=int(cfg.get("analytic_taps", 129)),
            envelope_taps=int(cfg.get("envelope_taps", 33)),
            fast_decimation=int(cfg.get("fast_decimation", 2)),
            spatial_rank=int(cfg.get("spatial_rank", 4)),
            temporal_dilations=tuple(cfg.get("temporal_dilations", [1, 2, 4, 8])),
            temporal_depth=int(cfg.get("temporal_depth", 2)),
            temporal_scale_attention=bool(cfg.get("temporal_scale_attention", True)),
            decoder_kind=str(cfg.get("decoder_kind", "ann")).lower(),
            decoder_residual_mode=str(
                cfg.get("decoder_residual_mode", "sew_add")
            ).lower(),
            decoder_layers=int(cfg.get("decoder_layers", 2)),
            decoder_channels=int(cfg.get("decoder_channels", 64)),
            decoder_decays=tuple(cfg.get("decoder_decays", [0.65, 0.90, 0.975])),
            endpoint_seconds=tuple(cfg.get("endpoint_seconds", [0.5, 1.0, 2.0, 4.0])),
            statistical_spatial_filters=int(cfg.get("statistical_spatial_filters", 8)),
            statistical_features=int(cfg.get("statistical_features", 96)),
            statistical_segments=int(cfg.get("statistical_segments", 4)),
            statistical_components=tuple(
                cfg.get(
                    "statistical_components",
                    [
                        "carrier_log_variance",
                        "envelope_mean",
                        "envelope_log_variance",
                    ],
                )
            ),
            statistical_encoder_kind=str(
                cfg.get("statistical_encoder_kind", "linear")
            ),
            statistical_variance_transform=str(
                cfg.get("statistical_variance_transform", "log1p")
            ),
            statistical_post_spatial_activation=str(
                cfg.get("statistical_post_spatial_activation", "none")
            ),
            statistical_spatial_initialization=str(
                cfg.get("statistical_spatial_initialization", "random")
            ),
            statistical_spatial_trainable=bool(
                cfg.get("statistical_spatial_trainable", True)
            ),
            covariance_features_per_band=int(cfg.get("covariance_features_per_band", 4)),
            use_statistical_branch=bool(cfg.get("use_statistical_branch", True)),
            use_covariance_branch=bool(cfg.get("use_covariance_branch", True)),
            use_temporal_branch=bool(cfg.get("use_temporal_branch", True)),
            delay_auxiliary_enabled=bool(cfg.get("delay_auxiliary_enabled", False)),
            delay_maximum_routes=int(cfg.get("delay_maximum_routes", 256)),
            delay_maximum_samples=int(cfg.get("delay_maximum_samples", 8)),
            delay_allow_cross_band=bool(cfg.get("delay_allow_cross_band", False)),
            delay_signal_mode=str(cfg.get("delay_signal_mode", "fast_phase")),
            delay_contextual_residual_enabled=bool(
                cfg.get("delay_contextual_residual_enabled", False)
            ),
            delay_contextual_residual_bound=float(
                cfg.get("delay_contextual_residual_bound", 0.5)
            ),
            delay_phase_residual_enabled=bool(
                cfg.get("delay_phase_residual_enabled", False)
            ),
            delay_phase_residual_bound=float(
                cfg.get("delay_phase_residual_bound", 0.25)
            ),
            delay_fusion_bound=float(cfg.get("delay_fusion_bound", 0.25)),
            delay_fusion_initial=float(cfg.get("delay_fusion_initial", 0.05)),
            fusion_features=int(cfg.get("fusion_features", 128)),
            fusion_kind=str(cfg.get("fusion_kind", "mlp")),
            dropout=float(cfg.get("dropout", 0.25)),
            parameter_ceiling=int(cfg.get("parameter_ceiling", 300_000)),
        )
    if name in {"dasp_snn_v62", "dasp_snn_v6_2_r1"}:
        return DASPSNNV62(
            n_channels=n_channels,
            n_classes=n_classes,
            channel_names=cfg.get(
                "channel_names",
                [
                    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3",
                    "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz",
                    "CP2", "CP4", "P1", "Pz", "P2", "POz",
                ],
            ),
            electrode_coordinates=cfg.get("electrode_coordinates"),
            n_bands=int(cfg.get("n_bands", 12)),
            n_nodes=int(cfg.get("n_nodes", 16)),
            band_edges_hz=cfg.get(
                "band_edges_hz",
                [
                    [6, 8], [8, 10], [10, 12], [12, 14], [14, 17], [17, 20],
                    [20, 23], [23, 26], [26, 30], [30, 34], [34, 37], [37, 40],
                ],
            ),
            sfreq=float(cfg.get("sfreq", 250.0)),
            epoch_tmin=float(cfg.get("epoch_tmin", -1.0)),
            task_tmin=float(cfg.get("task_tmin", 0.0)),
            task_tmax=float(cfg.get("task_tmax", 4.0)),
            analytic_taps=int(cfg.get("analytic_taps", 129)),
            envelope_taps=int(cfg.get("envelope_taps", 33)),
            spatial_rank=int(cfg.get("spatial_rank", 4)),
            freeze_spatial=bool(cfg.get("freeze_spatial", True)),
            route_rank=int(cfg.get("route_rank", 4)),
            delay_rank=int(cfg.get("delay_rank", 4)),
            fast_max_delay=int(cfg.get("fast_max_delay", 4)),
            slow_max_delay=int(cfg.get("slow_max_delay", 16)),
            fast_initial_contribution=float(cfg.get("fast_initial_contribution", 0.10)),
            dynamic_context_features=int(cfg.get("dynamic_context_features", 0)),
            temporal_dilations=tuple(cfg.get("temporal_dilations", [1, 2, 4, 8])),
            temporal_depth=int(cfg.get("temporal_depth", 2)),
            temporal_scale_attention=bool(cfg.get("temporal_scale_attention", True)),
            use_geometry=bool(cfg.get("use_geometry", True)),
            geometry_components=int(cfg.get("geometry_components", 6)),
            geometry_windows_seconds=tuple(
                cfg.get("geometry_windows_seconds", [0.25, 0.5, 1.0])
            ),
            snn_channels=int(cfg.get("snn_channels", 64)),
            decoder_kind=str(cfg.get("decoder_kind", "clif")),
            decoder_layers=int(cfg.get("decoder_layers", 2)),
            decoder_decays=tuple(cfg.get("decoder_decays", [0.65, 0.90, 0.975])),
            dropout=float(cfg.get("dropout", 0.25)),
            force_zero_delay=bool(cfg.get("force_zero_delay", False)),
            parameter_ceiling=int(cfg.get("parameter_ceiling", 220_000)),
        )
    if name == "dpc_snn":
        encoder = cfg.get("encoder", {})
        lif = cfg.get("lif", {})
        reg = cfg.get("regularization", {})
        return DPCSNN(
            n_classes=n_classes,
            n_channels=n_channels,
            n_bands=n_bands,
            hidden_channels=int(cfg.get("hidden_channels", 32)),
            timesteps=timesteps,
            d_max=int(cfg.get("d_max", 16)),
            slow_d_max=int(cfg.get("slow_d_max", 8)),
            slow_downsample=int(cfg.get("slow_downsample", 4)),
            graph_sparsity=float(cfg.get("graph_sparsity", 0.35)),
            encoder_threshold=float(encoder.get("threshold", 0.5)),
            encoder_deterministic=bool(encoder.get("deterministic", True)),
            tau_mem=float(lif.get("tau_mem", 0.9)),
            lif_threshold=float(lif.get("threshold", 1.0)),
            current_gain=float(cfg.get("current_gain", 0.5)),
            use_membrane_readout=bool(cfg.get("use_membrane_readout", True)),
            spike_rate_reg=float(reg.get("spike_rate", 0.001)),
            graph_l1_reg=float(reg.get("graph_l1", 0.0001)),
            delay_smooth_reg=float(reg.get("delay_smooth", 0.0001)),
            delay_entropy_reg=float(reg.get("delay_entropy", reg.get("delay_smooth", 0.0001))),
            delay_gate_l1_reg=float(reg.get("delay_gate_l1", 0.0001)),
            delay_gate_structure_reg=float(
                reg.get("delay_gate_structure", reg.get("delay_gate_l1", 0.0001))
            ),
            spatial_orthogonal_reg=float(reg.get("spatial_orthogonal", 0.001)),
            graph_prior_reg=float(reg.get("graph_prior", 0.0001)),
            delay_alignment_reg=float(reg.get("delay_alignment", 0.01)),
            delay_anchor_kl_reg=float(reg.get("delay_anchor_kl", 0.1)),
            delay_temperature=float(cfg.get("delay_temperature", 0.75)),
            min_delay_temperature=float(cfg.get("min_delay_temperature", 0.2)),
            delay_gate_init=float(cfg.get("delay_gate_init", 0.0)),
            max_phase_residual=float(cfg.get("max_phase_residual", 0.25)),
            temporal_pool_bins=int(cfg.get("temporal_pool_bins", 8)),
            timestep_seconds=float(
                cfg.get(
                    "timestep_seconds",
                    float(cfg.get("input_duration_seconds", timesteps)) / timesteps,
                )
            ),
            sfreq=float(cfg.get("sfreq", 250.0)),
            graph_timesteps=int(cfg.get("graph_timesteps", max(timesteps, 500))),
            latent_nodes=int(cfg.get("latent_nodes", 8)),
            spatial_max_deviation=float(cfg.get("spatial_max_deviation", 0.35)),
            snn_channels=int(cfg.get("snn_channels", 64)),
            band_edges_hz=cfg.get("band_edges_hz"),
            classification_band_edges_hz=cfg.get("classification_band_edges_hz"),
            alignment_matrix=cfg.get("alignment_matrix"),
            reference_augmentation_prob=float(cfg.get("reference_augmentation_prob", 0.5)),
            epoch_tmin=float(cfg.get("epoch_tmin", 0.0)),
            task_tmin=float(cfg.get("task_tmin", 0.0)),
            task_tmax=cfg.get("task_tmax"),
            spike_rate_target=float(cfg.get("spike_rate_target", 0.12)),
            use_covariance=bool(cfg.get("use_covariance", True)),
            use_phase_confidence=bool(cfg.get("use_phase_confidence", True)),
            delay_lr_multiplier=float(cfg.get("delay_lr_multiplier", 0.25)),
            use_cross_band_routes=bool(cfg.get("use_cross_band_routes", True)),
            n_delay_experts=int(cfg.get("n_delay_experts", 4)),
            delay_expert_topk=int(cfg.get("delay_expert_topk", 2)),
            preserve_graph_rate=bool(cfg.get("preserve_graph_rate", True)),
            temporal_pool_channels=int(cfg.get("temporal_pool_channels", 16)),
            expert_balance_reg=float(reg.get("expert_balance", 0.001)),
            expert_diversity_reg=float(reg.get("expert_diversity", 0.01)),
            force_zero_delay=bool(cfg.get("force_zero_delay", False)),
            route_rank=int(cfg.get("route_rank", 4)),
            decoder_layers=int(cfg.get("decoder_layers", 3)),
            tet_scales=tuple(int(scale) for scale in cfg.get("tet_scales", [8, 16, 32, 64])),
            identity_spatial_projection=bool(cfg.get("identity_spatial_projection", False)),
            delay_evidence_band_edges_hz=cfg.get("delay_evidence_band_edges_hz"),
            router_warmup_fraction=float(cfg.get("router_warmup_fraction", 0.35)),
            router_noise_scale=float(cfg.get("router_noise_scale", 1.0)),
            route_odds_threshold=(
                None
                if cfg.get("route_odds_threshold") is None
                else float(cfg["route_odds_threshold"])
            ),
            hurdle_min_bayes_factor=(
                None
                if cfg.get("hurdle_min_bayes_factor") is None
                else float(cfg.get("hurdle_min_bayes_factor", 3.0))
            ),
            hurdle_route_temperature=float(cfg.get("hurdle_route_temperature", 0.5)),
            spatial_band_rank=int(cfg.get("spatial_band_rank", 4)),
            spatial_band_deviation=float(cfg.get("spatial_band_deviation", 0.25)),
            snn_membrane_norm_groups=int(cfg.get("snn_membrane_norm_groups", 8)),
            cumulative_readout_seconds=tuple(
                float(seconds) for seconds in cfg.get("cumulative_readout_seconds", [])
            ),
            causal_channel_norm=bool(cfg.get("causal_channel_norm", False)),
            decoder_neuron=str(cfg.get("decoder_neuron", "clif")),
            sew_connect_function=str(cfg.get("sew_connect_function", "ADD")),
            fixed_delay_steps=(
                None if cfg.get("fixed_delay_steps") is None else float(cfg["fixed_delay_steps"])
            ),
            freeze_delay_posterior=bool(cfg.get("freeze_delay_posterior", False)),
            architecture_version=str(
                cfg.get("architecture_version", "dpc_snn_v3_9_scientific_repair")
            ),
            freeze_shared_physical_basis=bool(cfg.get("freeze_shared_physical_basis", False)),
            preserve_transport_band_pairs=bool(cfg.get("preserve_transport_band_pairs", False)),
            delayed_stat_channels=int(cfg.get("delayed_stat_channels", 4)),
            rejected_route_prior_floor=float(cfg.get("rejected_route_prior_floor", 0.02)),
            matched_transport_control=bool(cfg.get("matched_transport_control", False)),
            freeze_delay_after_pretrain=bool(cfg.get("freeze_delay_after_pretrain", False)),
            freeze_shared_filterbank=bool(cfg.get("freeze_shared_filterbank", False)),
            delayed_directional_moments=bool(cfg.get("delayed_directional_moments", False)),
            event_native_transport=bool(cfg.get("event_native_transport", False)),
            event_envelope_threshold=float(cfg.get("event_envelope_threshold", 0.05)),
            event_phase_confidence_threshold=float(
                cfg.get("event_phase_confidence_threshold", 0.10)
            ),
            learnable_event_threshold=bool(cfg.get("learnable_event_threshold", False)),
            event_timing_intervention=str(cfg.get("event_timing_intervention", "none")),
            event_timing_seed=int(cfg.get("event_timing_seed", 0)),
            multiscale_snn=bool(cfg.get("multiscale_snn", False)),
            decoder_timescales=tuple(
                float(value) for value in cfg.get("decoder_timescales", [0.65, 0.90, 0.975])
            ),
            snn_native_readout=bool(cfg.get("snn_native_readout", False)),
            snn_readout_decays=tuple(
                float(value) for value in cfg.get("snn_readout_decays", [0.5, 0.9, 0.99])
            ),
            decoder_mode=str(cfg.get("decoder_mode", "snn")),
            checkpoint_event_routes=bool(cfg.get("checkpoint_event_routes", False)),
            electrode_coordinates=cfg.get("electrode_coordinates"),
            spatial_anchor_indices=cfg.get("spatial_anchor_indices"),
        )
    if name == "eegnet":
        return EEGNet(
            n_channels=n_channels,
            n_classes=n_classes,
            samples=int(cfg.get("samples", 256)),
            sfreq=float(cfg.get("sfreq", 128.0)),
            f1=int(cfg.get("f1", 8)),
            d=int(cfg.get("depth_multiplier", cfg.get("d", 2))),
            dropout=float(cfg.get("dropout", 0.25)),
        )
    if name == "vanilla_snn":
        return VanillaSNN(
            n_channels=n_channels,
            n_classes=n_classes,
            hidden=int(cfg.get("hidden_channels", cfg.get("hidden", 64))),
            timesteps=timesteps,
        )
    if name in {"graph_snn", "graph_snn_no_delay"}:
        return GraphSNNNoDelay(
            n_channels=n_channels,
            n_classes=n_classes,
            hidden_channels=int(cfg.get("hidden_channels", 32)),
            timesteps=timesteps,
        )
    if name == "delay_graph_ann":
        return DelayGraphANN(
            n_channels=n_channels,
            n_classes=n_classes,
            timesteps=timesteps,
            d_max=int(cfg.get("d_max", 16)),
            hidden=int(cfg.get("hidden_channels", cfg.get("hidden", 64))),
        )
    if name == "phase_gated_gnn":
        return PhaseGatedGNN(
            n_channels=n_channels,
            n_classes=n_classes,
            n_bands=n_bands,
            timesteps=timesteps,
            hidden=int(cfg.get("hidden_channels", cfg.get("hidden", 64))),
        )
    if name == "dilated_tcn":
        return DilatedTCN(
            n_channels=n_channels,
            n_classes=n_classes,
            hidden=int(cfg.get("hidden_channels", cfg.get("hidden", 32))),
        )
    if name in {"complex_cnn", "complex_valued_cnn"}:
        return ComplexValuedCNN(
            n_channels=n_channels,
            n_classes=n_classes,
            n_bands=n_bands,
            hidden=int(cfg.get("hidden_channels", cfg.get("hidden", 32))),
        )
    raise KeyError(f"Unknown model: {name}")
