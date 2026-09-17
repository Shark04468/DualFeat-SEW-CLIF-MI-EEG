"""Mandatory low-rank context-conditioned cross-band delay-phase synapse."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .hurdle_delay import HurdleDelayOutput, hurdle_delay_posterior


class DelayPhaseGraphSynapse(nn.Module):
    """Route all classifier traffic through sparse learnable delay experts."""

    def __init__(
        self,
        n_bands: int,
        n_channels: int,
        d_max: int = 8,
        slow_d_max: int = 8,
        slow_downsample: int = 4,
        graph_sparsity: float = 0.15,
        phase_gain: float = 2.0,
        delay_temperature: float = 0.75,
        min_delay_temperature: float = 0.2,
        delay_gate_init: float = 0.0,
        max_phase_residual: float = 0.25,
        timestep_seconds: float = 1.0,
        evidence_momentum: float = 0.9,
        evidence_strength: float = 1.5,
        use_cross_band_routes: bool = True,
        n_delay_experts: int = 4,
        expert_topk: int = 2,
        context_dim: int | None = None,
        node_coordinates: torch.Tensor | None = None,
        force_zero_delay: bool = False,
        route_rank: int = 4,
        router_warmup_fraction: float = 0.35,
        router_noise_scale: float = 1.0,
        route_odds_threshold: float | None = None,
        hurdle_min_bayes_factor: float | None = 3.0,
        hurdle_route_temperature: float = 0.5,
        fixed_delay_steps: float | None = None,
        use_fixed_fold_posterior: bool = False,
        rejected_route_prior_floor: float = 0.02,
        matched_transport_control: bool = False,
        event_native_transport: bool = False,
        checkpoint_event_routes: bool = False,
    ):
        super().__init__()
        self.n_bands = int(n_bands)
        self.n_channels = int(n_channels)
        self.d_max = int(d_max)
        self.slow_downsample = max(1, int(slow_downsample))
        self.slow_d_max = int(slow_d_max)
        self.graph_sparsity = float(graph_sparsity)
        self.phase_gain = float(phase_gain)
        self.initial_delay_temperature = float(delay_temperature)
        self.min_delay_temperature = float(min_delay_temperature)
        self.register_buffer("delay_temperature_state", torch.tensor(float(delay_temperature)))
        self.max_phase_residual = float(max_phase_residual)
        self.timestep_seconds = float(timestep_seconds)
        self.evidence_momentum = float(evidence_momentum)
        self.evidence_strength = float(evidence_strength)
        self.use_cross_band_routes = bool(use_cross_band_routes)
        self.evidence_updates_enabled = True
        self.n_delay_experts = max(1, int(n_delay_experts))
        self.expert_topk = max(1, min(int(expert_topk), self.n_delay_experts))
        self.force_zero_delay = bool(force_zero_delay)
        self.route_rank = max(1, int(route_rank))
        self.router_warmup_fraction = max(0.0, float(router_warmup_fraction))
        self.router_noise_scale = max(0.0, float(router_noise_scale))
        if route_odds_threshold is None:
            legacy_bf = 3.0 if hurdle_min_bayes_factor is None else float(hurdle_min_bayes_factor)
            route_odds_threshold = math.log(max(legacy_bf, 1e-6))
        self.route_odds_threshold = float(route_odds_threshold)
        self.hurdle_route_temperature = max(1e-4, float(hurdle_route_temperature))
        self.fixed_delay_steps = None if fixed_delay_steps is None else float(fixed_delay_steps)
        self.use_fixed_fold_posterior = bool(use_fixed_fold_posterior)
        self.rejected_route_prior_floor = float(rejected_route_prior_floor)
        if not 0.0 < self.rejected_route_prior_floor < 0.5:
            raise ValueError("rejected_route_prior_floor must lie in (0, 0.5)")
        self.matched_transport_control = bool(matched_transport_control)
        self.event_native_transport = bool(event_native_transport)
        self.checkpoint_event_routes = bool(checkpoint_event_routes)
        if self.fixed_delay_steps is not None and not 0.0 <= self.fixed_delay_steps <= self.d_max:
            raise ValueError("fixed_delay_steps must lie within [0, d_max]")
        self.register_buffer("training_progress_state", torch.tensor(0.0))
        self.register_buffer("phase_residual_enabled", torch.tensor(False))

        e, b, c, r = self.n_delay_experts, self.n_bands, self.n_channels, self.route_rank
        fast_bins = self.d_max + 1
        slow_bins = self.slow_d_max + 1

        # CP interactions capture route-specific target/source structure without
        # materialising a trainable B^2*C^2 tensor.
        self.edge_logits = nn.Parameter(torch.zeros(e, b, b))
        self.edge_target_logits = nn.Parameter(torch.empty(e, b, c, r))
        self.edge_source_logits = nn.Parameter(torch.empty(e, b, c, r))
        self.edge_rank_logits = nn.Parameter(torch.empty(e, b, b, r))
        self.edge_existence_logits = nn.Parameter(torch.full((e, b, b), float(delay_gate_init)))
        self.edge_existence_target_logits = nn.Parameter(torch.empty(e, b, c, r))
        self.edge_existence_source_logits = nn.Parameter(torch.empty(e, b, c, r))
        self.edge_existence_rank_logits = nn.Parameter(torch.empty(e, b, b, r))
        self.delay_confidence_logits = nn.Parameter(torch.zeros(e, b, b))
        self.delay_confidence_target_logits = nn.Parameter(torch.empty(e, b, c, r))
        self.delay_confidence_source_logits = nn.Parameter(torch.empty(e, b, c, r))
        self.delay_confidence_rank_logits = nn.Parameter(torch.empty(e, b, b, r))

        self.fast_delay_logits = nn.Parameter(torch.zeros(e, b, b, fast_bins))
        self.fast_delay_target_logits = nn.Parameter(torch.empty(e, b, c, r))
        self.fast_delay_source_logits = nn.Parameter(torch.empty(e, b, c, r))
        self.fast_delay_rank_logits = nn.Parameter(torch.empty(e, b, b, r, fast_bins))
        self.slow_delay_logits = nn.Parameter(torch.zeros(e, b, b, slow_bins))
        self.slow_delay_target_logits = nn.Parameter(torch.empty(e, b, c, r))
        self.slow_delay_source_logits = nn.Parameter(torch.empty(e, b, c, r))
        self.slow_delay_rank_logits = nn.Parameter(torch.empty(e, b, b, r, slow_bins))

        self.fast_fraction_raw = nn.Parameter(torch.full((e, b, b), -6.0))
        self.fast_fraction_target_raw = nn.Parameter(torch.empty(e, b, c, r))
        self.fast_fraction_source_raw = nn.Parameter(torch.empty(e, b, c, r))
        self.fast_fraction_rank_raw = nn.Parameter(torch.empty(e, b, b, r))
        self.slow_fraction_raw = nn.Parameter(torch.full((e, b, b), -6.0))
        self.slow_fraction_target_raw = nn.Parameter(torch.empty(e, b, c, r))
        self.slow_fraction_source_raw = nn.Parameter(torch.empty(e, b, c, r))
        self.slow_fraction_rank_raw = nn.Parameter(torch.empty(e, b, b, r))
        self.phase_pref = nn.Parameter(torch.zeros(e, b, b))
        self.phase_pref_target = nn.Parameter(torch.empty(e, b, c, r))
        self.phase_pref_source = nn.Parameter(torch.empty(e, b, c, r))
        self.phase_pref_rank = nn.Parameter(torch.empty(e, b, b, r))
        self.slow_modulation_raw = nn.Parameter(torch.full((e,), -0.43275213))

        context_dim = int(context_dim or b * c)
        self.expert_router = nn.Linear(context_dim, e)
        nn.init.normal_(self.expert_router.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.expert_router.bias)

        route_shape = (b, b, c, c)
        mask = torch.ones(route_shape)
        for band in range(b):
            mask[band, band].fill_diagonal_(0.0)
        if not self.use_cross_band_routes:
            mask = mask * torch.eye(b)[:, :, None, None]
        self.register_buffer("route_mask", mask)

        if node_coordinates is None:
            coordinates = torch.stack((torch.linspace(-1.0, 1.0, c), torch.zeros(c)), dim=1)
        else:
            coordinates = torch.as_tensor(node_coordinates, dtype=torch.float32)
            if coordinates.shape != (c, 2):
                raise ValueError("node_coordinates must have shape [n_channels, 2]")
        distance = torch.cdist(coordinates, coordinates)
        spatial = torch.exp(-distance / 0.55)
        band_index = torch.arange(b, dtype=torch.float32)
        cross_band = torch.exp(-(band_index[:, None] - band_index[None, :]).abs() / 1.5)
        prior = 0.35 * cross_band[:, :, None, None] * spatial[None, None, :, :]
        self.register_buffer("node_coordinates", coordinates)
        self.register_buffer("edge_prior", prior * mask)
        self.register_buffer("fold_route_prior", torch.full(route_shape, 0.5) * mask)
        self.register_buffer(
            "fold_positive_delay_prior",
            torch.full((*route_shape, self.d_max + 1), 1.0 / (self.d_max + 1)) * mask[..., None],
        )
        self.register_buffer("fold_fraction_target", torch.zeros(route_shape))
        self.register_buffer("fold_evidence_ready", torch.tensor(False))
        self.register_buffer("connectivity_prior_ema", torch.full(route_shape, 0.5) * mask)

        self.register_buffer("fast_evidence_ema", torch.zeros(*route_shape, fast_bins))
        self.register_buffer("slow_evidence_ema", torch.zeros(*route_shape, slow_bins))
        self.register_buffer("coarse_reliability_ema", torch.zeros(route_shape))
        self.register_buffer("route_reliability_ema", torch.zeros(route_shape))
        self.register_buffer("fast_evidence_accum", torch.zeros(*route_shape, fast_bins), persistent=False)
        self.register_buffer("slow_evidence_accum", torch.zeros(*route_shape, slow_bins), persistent=False)
        self.register_buffer("reliability_accum", torch.zeros(route_shape), persistent=False)
        self.register_buffer("route_reliability_accum", torch.zeros(route_shape), persistent=False)
        self.register_buffer("connectivity_prior_accum", torch.zeros(route_shape), persistent=False)
        self.register_buffer("evidence_accum_weight", torch.tensor(0.0), persistent=False)
        self.register_buffer(
            "fast_anchor_prob",
            torch.full((e, *route_shape, fast_bins), 1.0 / fast_bins),
        )
        self.register_buffer(
            "slow_anchor_prob",
            torch.full((e, *route_shape, slow_bins), 1.0 / slow_bins),
        )
        self.register_buffer("anchor_ready", torch.tensor(False))
        self.register_buffer("edge_weight_override", torch.empty(0), persistent=False)
        self.reset_parameters()

    @property
    def delay_logits(self) -> nn.Parameter:
        return self.fast_delay_logits

    @property
    def delay_temperature(self) -> float:
        return float(self.delay_temperature_state)

    @delay_temperature.setter
    def delay_temperature(self, value: float) -> None:
        self.delay_temperature_state.fill_(float(value))

    @property
    def delay_gate_logits(self) -> nn.Parameter:
        return self.edge_existence_logits

    def delay_parameters(self) -> list[nn.Parameter]:
        names = (
            "fast_delay_logits", "fast_delay_target_logits", "fast_delay_source_logits", "fast_delay_rank_logits",
            "slow_delay_logits", "slow_delay_target_logits", "slow_delay_source_logits", "slow_delay_rank_logits",
            "fast_fraction_raw", "fast_fraction_target_raw", "fast_fraction_source_raw", "fast_fraction_rank_raw",
            "slow_fraction_raw", "slow_fraction_target_raw", "slow_fraction_source_raw", "slow_fraction_rank_raw",
            "edge_logits", "edge_target_logits", "edge_source_logits", "edge_rank_logits",
            "edge_existence_logits", "edge_existence_target_logits", "edge_existence_source_logits", "edge_existence_rank_logits",
            "delay_confidence_logits", "delay_confidence_target_logits", "delay_confidence_source_logits", "delay_confidence_rank_logits",
            "phase_pref", "phase_pref_target", "phase_pref_source", "phase_pref_rank", "slow_modulation_raw",
        )
        return [getattr(self, name) for name in names] + list(self.expert_router.parameters())

    def delay_pretraining_parameters(self) -> list[nn.Parameter]:
        """Mechanism parameters learned before task labels train the router.

        The expert router and post-delay classifier are deliberately excluded.
        Edge existence, conditional delay, and the low-rank evidence prior are
        fitted while phase preference is fixed at zero. Residual phase is only
        enabled after the delay target has been identified.
        """

        names = (
            "fast_delay_logits", "fast_delay_target_logits", "fast_delay_source_logits", "fast_delay_rank_logits",
            "slow_delay_logits", "slow_delay_target_logits", "slow_delay_source_logits", "slow_delay_rank_logits",
            "fast_fraction_raw", "fast_fraction_target_raw", "fast_fraction_source_raw", "fast_fraction_rank_raw",
            "slow_fraction_raw", "slow_fraction_target_raw", "slow_fraction_source_raw", "slow_fraction_rank_raw",
            "edge_logits", "edge_target_logits", "edge_source_logits", "edge_rank_logits",
            "edge_existence_logits", "edge_existence_target_logits", "edge_existence_source_logits", "edge_existence_rank_logits",
            "delay_confidence_logits", "delay_confidence_target_logits", "delay_confidence_source_logits", "delay_confidence_rank_logits",
        )
        return [getattr(self, name) for name in names]

    def posterior_parameters(self) -> list[nn.Parameter]:
        """Parameters defining route existence and conditional transport lag."""

        names = (
            "fast_delay_logits", "fast_delay_target_logits", "fast_delay_source_logits", "fast_delay_rank_logits",
            "slow_delay_logits", "slow_delay_target_logits", "slow_delay_source_logits", "slow_delay_rank_logits",
            "fast_fraction_raw", "fast_fraction_target_raw", "fast_fraction_source_raw", "fast_fraction_rank_raw",
            "slow_fraction_raw", "slow_fraction_target_raw", "slow_fraction_source_raw", "slow_fraction_rank_raw",
            "edge_existence_logits", "edge_existence_target_logits", "edge_existence_source_logits", "edge_existence_rank_logits",
        )
        return [getattr(self, name) for name in names]

    def residual_delay_parameters(self) -> list[nn.Parameter]:
        """All non-router delay-mechanism parameters for low-LR fine-tuning."""

        return [*self.delay_pretraining_parameters(), self.slow_modulation_raw]

    def reset_parameters(self) -> None:
        with torch.no_grad():
            factor_names = (
                "edge_target_logits", "edge_source_logits", "edge_existence_target_logits",
                "edge_existence_source_logits", "delay_confidence_target_logits",
                "delay_confidence_source_logits", "fast_delay_target_logits",
                "fast_delay_source_logits", "slow_delay_target_logits", "slow_delay_source_logits",
                "fast_fraction_target_raw", "fast_fraction_source_raw", "slow_fraction_target_raw",
                "slow_fraction_source_raw", "phase_pref_target", "phase_pref_source",
            )
            rank_names = (
                "edge_rank_logits", "edge_existence_rank_logits", "delay_confidence_rank_logits",
                "fast_delay_rank_logits", "slow_delay_rank_logits", "fast_fraction_rank_raw",
                "slow_fraction_rank_raw", "phase_pref_rank",
            )
            for name in factor_names:
                nn.init.normal_(getattr(self, name), mean=0.0, std=0.15)
            for name in rank_names:
                nn.init.normal_(getattr(self, name), mean=0.0, std=0.03)
            self.phase_pref.zero_()
            self.phase_pref_target.zero_()
            self.phase_pref_source.zero_()
            self.phase_pref_rank.zero_()
            # Experts begin close but not identical, avoiding permanent symmetric collapse.
            fast_pattern = torch.linspace(-1.0, 1.0, self.d_max + 1)
            slow_pattern = torch.linspace(-1.0, 1.0, self.slow_d_max + 1)
            for expert in range(self.n_delay_experts):
                offset = (expert - (self.n_delay_experts - 1) / 2.0) * 0.01
                self.fast_delay_logits[expert].add_(offset * fast_pattern)
                self.slow_delay_logits[expert].sub_(offset * slow_pattern)

    @staticmethod
    def _factor_route(
        base: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
        rank: torch.Tensor,
    ) -> torch.Tensor:
        if base.ndim == 3:
            interaction = torch.einsum("eabr,eair,ebjr->eabij", rank, target, source)
            return base[:, :, :, None, None] + interaction / math.sqrt(target.shape[-1])
        if base.ndim == 4:
            interaction = torch.einsum("eabrd,eair,ebjr->eabijd", rank, target, source)
            return base[:, :, :, None, None, :] + interaction / math.sqrt(target.shape[-1])
        raise ValueError("Low-rank route factors must have three or four dimensions")

    def _edge_logits_full(self) -> torch.Tensor:
        return self._factor_route(
            self.edge_logits, self.edge_target_logits, self.edge_source_logits, self.edge_rank_logits
        )

    def _edge_existence_logits_full(self) -> torch.Tensor:
        return self._factor_route(
            self.edge_existence_logits,
            self.edge_existence_target_logits,
            self.edge_existence_source_logits,
            self.edge_existence_rank_logits,
        )

    def _delay_confidence_logits_full(self) -> torch.Tensor:
        return self._factor_route(
            self.delay_confidence_logits,
            self.delay_confidence_target_logits,
            self.delay_confidence_source_logits,
            self.delay_confidence_rank_logits,
        )

    def _fast_logits_full(self) -> torch.Tensor:
        return self._factor_route(
            self.fast_delay_logits, self.fast_delay_target_logits, self.fast_delay_source_logits,
            self.fast_delay_rank_logits,
        )

    def _slow_logits_full(self) -> torch.Tensor:
        return self._factor_route(
            self.slow_delay_logits, self.slow_delay_target_logits, self.slow_delay_source_logits,
            self.slow_delay_rank_logits,
        )

    def _fraction_full(
        self, base: torch.Tensor, target: torch.Tensor, source: torch.Tensor, rank: torch.Tensor
    ) -> torch.Tensor:
        return torch.sigmoid(self._factor_route(base, target, source, rank)) * self.route_mask[None]

    def _phase_full(self) -> torch.Tensor:
        raw = self._factor_route(
            self.phase_pref, self.phase_pref_target, self.phase_pref_source, self.phase_pref_rank
        )
        return self.max_phase_residual * torch.tanh(raw) * self.route_mask[None]

    def set_edge_weight_override(self, multiplier: torch.Tensor | None) -> None:
        if multiplier is None:
            self.edge_weight_override = torch.empty(0, device=self.edge_logits.device)
            return
        if multiplier.shape != self.route_mask.shape:
            raise ValueError(f"Expected edge multiplier {tuple(self.route_mask.shape)}, got {tuple(multiplier.shape)}")
        self.edge_weight_override = multiplier.detach().to(self.edge_logits)

    def edge_weight_expert(self) -> torch.Tensor:
        prior_logit = torch.atanh(self.effective_edge_prior().clamp(-0.95, 0.95))[None]
        weight = torch.tanh(self._edge_logits_full() + prior_logit) * self.route_mask[None]
        if self.edge_weight_override.numel():
            weight = weight * self.edge_weight_override[None]
        return weight

    def edge_weight(self) -> torch.Tensor:
        return self.edge_weight_expert().mean(0)

    def edge_selection_expert(self) -> torch.Tensor:
        # Sparsity is enforced as an upper-bound penalty on the Bernoulli route
        # posterior. There is no fixed quota and no edge is created merely to
        # fill a top-k allocation.
        return self.route_mask[None].expand(self.n_delay_experts, -1, -1, -1, -1)

    def edge_existence_expert(self) -> torch.Tensor:
        return self.delay_route_probability_expert()

    def edge_existence(self) -> torch.Tensor:
        return self.edge_existence_expert().mean(0)

    @staticmethod
    def _entropy_confidence(probability: torch.Tensor) -> torch.Tensor:
        if probability.shape[-1] <= 1:
            return torch.ones_like(probability[..., 0])
        entropy = -(probability.clamp_min(1e-8) * probability.clamp_min(1e-8).log()).sum(-1)
        return (1.0 - entropy / math.log(probability.shape[-1])).clamp(0.0, 1.0)

    def delay_confidence_expert(self, probability: torch.Tensor) -> torch.Tensor:
        del probability
        learned = torch.sigmoid(self._delay_confidence_logits_full()) * self.route_mask[None]
        return learned

    def delay_confidence(self, probability: torch.Tensor) -> torch.Tensor:
        if probability.ndim == 5:
            probability = probability[None].expand(self.n_delay_experts, *probability.shape)
        return self.delay_confidence_expert(probability).mean(0)

    def _route_logits_full(self) -> torch.Tensor:
        logits = self._edge_existence_logits_full()
        if bool(self.fold_evidence_ready):
            prior = self.fold_route_prior.clamp(
                self.rejected_route_prior_floor, 1.0 - self.rejected_route_prior_floor
            )
            # Calibrate the prior in the coordinates used by
            # sigmoid((route_logit - threshold) / temperature). Otherwise a
            # nominal 0.02 floor becomes ~1e-5 after thresholding.
            calibrated_prior_logit = self.route_odds_threshold + (
                self.hurdle_route_temperature * torch.logit(prior)
            )
            logits = logits + calibrated_prior_logit[None]
        return logits * self.route_mask[None]

    def _posterior_components(
        self, logits: torch.Tensor, evidence: torch.Tensor
    ) -> HurdleDelayOutput:
        # Evidence is a categorical likelihood, not a logit. Using its raw
        # probability made small zero-lag advantages dominate after annealing.
        log_evidence = evidence.clamp_min(1e-6).log()
        log_evidence = log_evidence - log_evidence.mean(dim=-1, keepdim=True)
        score = (
            logits + self.evidence_strength * log_evidence[None]
        ) / max(self.delay_temperature, 1e-4)
        return hurdle_delay_posterior(
            score,
            route_odds_threshold=self.route_odds_threshold,
            route_temperature=self.hurdle_route_temperature,
            force_zero_delay=self.force_zero_delay,
            route_logit=self._route_logits_full(),
        )

    def fast_delay_prob_expert(self) -> torch.Tensor:
        if self.fixed_delay_steps is not None or self.force_zero_delay:
            value = 0.0 if self.force_zero_delay else float(self.fixed_delay_steps)
            base = min(self.d_max, int(math.floor(value)))
            probability = self.fast_evidence_ema.new_zeros(
                self.n_delay_experts, *self.route_mask.shape, self.d_max + 1
            )
            probability[..., base] = 1.0
            return probability * self.route_mask[None, ..., None]
        if self.use_fixed_fold_posterior and bool(self.fold_evidence_ready):
            return self.fold_positive_delay_prior[None].expand(
                self.n_delay_experts, *self.fold_positive_delay_prior.shape
            )
        return self._posterior_components(
            self._fast_logits_full(), self.fast_evidence_ema
        ).positive_probability

    def fast_delay_route_probability_expert(self) -> torch.Tensor:
        return self._posterior_components(
            self._fast_logits_full(), self.fast_evidence_ema
        ).route_probability

    def slow_delay_prob_expert(self) -> torch.Tensor:
        if self.fixed_delay_steps is not None or self.force_zero_delay:
            value = 0.0 if self.force_zero_delay else float(self.fixed_delay_steps)
            coarse_value = value / self.slow_downsample
            base = min(self.slow_d_max, int(math.floor(coarse_value)))
            probability = self.slow_evidence_ema.new_zeros(
                self.n_delay_experts, *self.route_mask.shape, self.slow_d_max + 1
            )
            probability[..., base] = 1.0
            return probability * self.route_mask[None, ..., None]
        if self.use_fixed_fold_posterior and bool(self.fold_evidence_ready):
            coarse = self._coarse_evidence(self.fold_positive_delay_prior)
            return coarse[None].expand(self.n_delay_experts, *coarse.shape)
        return self._posterior_components(
            self._slow_logits_full(), self.slow_evidence_ema
        ).positive_probability

    def slow_delay_route_probability_expert(self) -> torch.Tensor:
        return self._posterior_components(
            self._slow_logits_full(), self.slow_evidence_ema
        ).route_probability

    def fast_delay_prob(self) -> torch.Tensor:
        return self.fast_delay_prob_expert().mean(0)

    def slow_delay_prob(self) -> torch.Tensor:
        return self.slow_delay_prob_expert().mean(0)

    def _total_delay_prob(self, carrier_probability: torch.Tensor, coarse_probability: torch.Tensor) -> torch.Tensor:
        coarse_weight = torch.stack(
            [coarse_probability[..., min(delay // self.slow_downsample, coarse_probability.shape[-1] - 1)] for delay in range(self.d_max + 1)],
            -1,
        )
        reliability = self.coarse_reliability_ema[None, ..., None]
        total = carrier_probability * ((1.0 - reliability) + reliability * coarse_weight)
        return total / total.sum(-1, keepdim=True).clamp_min(1e-8)

    def delay_prob_expert(self) -> torch.Tensor:
        return self._total_delay_prob(self.fast_delay_prob_expert(), self.slow_delay_prob_expert())

    def delay_route_probability_expert(self) -> torch.Tensor:
        if self.use_fixed_fold_posterior and bool(self.fold_evidence_ready):
            return self.fold_route_prior[None].expand(
                self.n_delay_experts, *self.fold_route_prior.shape
            )
        return self.fast_delay_route_probability_expert()

    def delay_null_probability(self) -> torch.Tensor:
        return 1.0 - self.delay_route_probability_expert().mean(0)

    def online_route_log_odds(self) -> torch.Tensor:
        route = self.delay_route_probability_expert().clamp(1e-6, 1.0 - 1e-6)
        return torch.logit(route).mean(0)

    def delay_prob(self) -> torch.Tensor:
        return self.delay_prob_expert().mean(0)

    def delay_gate(self) -> torch.Tensor:
        probability = self.delay_prob_expert()
        return (
            self.delay_confidence_expert(probability)
            * self.delay_route_probability_expert()
        ).mean(0)

    def effective_edge_selection_expert(self) -> torch.Tensor:
        route = self.delay_route_probability_expert()
        hard = (route > 0.5).to(route.dtype)
        accepted = hard + route - route.detach()
        return self.route_mask[None] * accepted

    def set_training_progress(self, progress: float, anneal_lag: bool = True) -> None:
        progress = min(1.0, max(0.0, float(progress)))
        if anneal_lag:
            self.delay_temperature = (
                (1.0 - progress) * self.initial_delay_temperature
                + progress * self.min_delay_temperature
            )
        else:
            self.training_progress_state.fill_(progress)

    def set_router_training_progress(self, progress: float) -> None:
        """Advance router warmup without coupling it to lag annealing."""

        progress = min(1.0, max(0.0, float(progress)))
        self.training_progress_state.fill_(progress)

    def set_temperature_progress(self, progress: float) -> None:
        """Backward-compatible alias for lag-temperature annealing."""
        self.set_training_progress(progress, anneal_lag=True)

    def begin_task_training(self) -> None:
        self.delay_temperature = self.initial_delay_temperature
        self.set_evidence_updates(False)
        if not self.matched_transport_control:
            self.phase_residual_enabled.fill_(True)

    @torch.no_grad()
    def load_fold_local_evidence_prior(
        self,
        route_probability: torch.Tensor,
        positive_delay_probability: torch.Tensor,
        fractional_delay_target: torch.Tensor | None = None,
        connectivity_prior: torch.Tensor | None = None,
    ) -> None:
        if route_probability.shape != self.route_mask.shape:
            raise ValueError("Fold-local route prior has incompatible shape")
        expected = (*self.route_mask.shape, self.d_max + 1)
        if positive_delay_probability.shape != expected:
            raise ValueError(f"Fold-local positive-delay prior must have shape {expected}")
        route = route_probability.to(self.route_mask).clamp(0.0, 1.0)
        route = torch.where(
            self.route_mask > 0,
            route.clamp_min(self.rejected_route_prior_floor),
            torch.zeros_like(route),
        )
        positive = positive_delay_probability.to(self.route_mask).clamp_min(0.0)
        positive = positive / positive.sum(-1, keepdim=True).clamp_min(1e-8)
        self.fold_route_prior.copy_(route)
        self.fold_positive_delay_prior.copy_(positive * self.route_mask[..., None])
        self.fast_evidence_ema.zero_()
        self.fast_evidence_ema.copy_(self.fold_positive_delay_prior)
        self.slow_evidence_ema.zero_()
        self.slow_evidence_ema.copy_(self._coarse_evidence(self.fold_positive_delay_prior))
        if fractional_delay_target is not None:
            if fractional_delay_target.shape != self.route_mask.shape:
                raise ValueError("Fold-local fractional-delay target has incompatible shape")
            self.fold_fraction_target.copy_(fractional_delay_target.to(self.route_mask).clamp(0.0, 1.0))
        if connectivity_prior is not None:
            if connectivity_prior.shape != self.route_mask.shape:
                raise ValueError("Fold-local connectivity prior has incompatible shape")
            self.connectivity_prior_ema.copy_(
                connectivity_prior.to(self.route_mask).clamp(0.0, 1.0)
                * self.route_mask
            )
        self.fold_evidence_ready.fill_(True)

    def set_evidence_updates(self, enabled: bool) -> None:
        # Fold-local priors are immutable scientific inputs. They must never be
        # replaced by batch-order-dependent online EMA evidence.
        self.evidence_updates_enabled = bool(enabled) and not bool(self.fold_evidence_ready)

    def begin_evidence_epoch(self) -> None:
        self.fast_evidence_accum.zero_()
        self.slow_evidence_accum.zero_()
        self.reliability_accum.zero_()
        self.route_reliability_accum.zero_()
        self.connectivity_prior_accum.zero_()
        self.evidence_accum_weight.zero_()

    def end_evidence_epoch(self) -> None:
        if float(self.evidence_accum_weight) <= 0:
            return
        weight = self.evidence_accum_weight.clamp_min(1.0)
        fast = self.fast_evidence_accum / weight
        slow = self.slow_evidence_accum / weight
        reliability = self.reliability_accum / weight
        route_reliability = self.route_reliability_accum / weight
        connectivity = self.connectivity_prior_accum / weight
        self.fast_evidence_ema.mul_(self.evidence_momentum).add_(fast, alpha=1.0 - self.evidence_momentum)
        self.slow_evidence_ema.mul_(self.evidence_momentum).add_(slow, alpha=1.0 - self.evidence_momentum)
        self.coarse_reliability_ema.mul_(self.evidence_momentum).add_(reliability, alpha=1.0 - self.evidence_momentum)
        self.route_reliability_ema.mul_(self.evidence_momentum).add_(
            route_reliability, alpha=1.0 - self.evidence_momentum
        )
        self.connectivity_prior_ema.mul_(self.evidence_momentum).add_(
            connectivity, alpha=1.0 - self.evidence_momentum
        )

    def effective_edge_prior(self) -> torch.Tensor:
        """Coordinate prior conditioned by lagged, directed training connectivity."""

        return self.edge_prior * (0.5 + self.connectivity_prior_ema)

    def capture_delay_anchor(self) -> None:
        with torch.no_grad():
            self.fast_anchor_prob.copy_(self.fast_delay_prob_expert())
            self.slow_anchor_prob.copy_(self.slow_delay_prob_expert())
            self.anchor_ready.fill_(True)

    def effective_phase_pref_expert(self) -> torch.Tensor:
        if not bool(self.phase_residual_enabled):
            return self.route_mask.new_zeros(
                self.n_delay_experts, *self.route_mask.shape
            )
        return self._phase_full()

    def effective_phase_pref(self) -> torch.Tensor:
        return self.effective_phase_pref_expert().mean(0)

    def fast_fraction_expert(self) -> torch.Tensor:
        if self.force_zero_delay:
            return self.route_mask.new_zeros(
                self.n_delay_experts, self.n_bands, self.n_bands, self.n_channels, self.n_channels
            )
        if self.use_fixed_fold_posterior and bool(self.fold_evidence_ready):
            return self.fold_fraction_target[None].expand(
                self.n_delay_experts, *self.fold_fraction_target.shape
            )
        return self._fraction_full(
            self.fast_fraction_raw, self.fast_fraction_target_raw, self.fast_fraction_source_raw,
            self.fast_fraction_rank_raw,
        )

    def slow_fraction_expert(self) -> torch.Tensor:
        if self.force_zero_delay:
            return self.route_mask.new_zeros(
                self.n_delay_experts, self.n_bands, self.n_bands, self.n_channels, self.n_channels
            )
        if self.fixed_delay_steps is not None:
            coarse_value = float(self.fixed_delay_steps) / self.slow_downsample
            fraction = coarse_value - math.floor(coarse_value)
            return torch.full_like(self.fold_fraction_target, fraction)[None].expand(
                self.n_delay_experts, *self.fold_fraction_target.shape
            ) * self.route_mask[None]
        if self.use_fixed_fold_posterior and bool(self.fold_evidence_ready):
            probability = self.fold_positive_delay_prior
            bins = torch.arange(
                probability.shape[-1], device=probability.device, dtype=probability.dtype
            )
            expected = (probability * bins).sum(-1)
            expected = expected + self.fold_fraction_target * (1.0 - probability[..., -1])
            coarse_fraction = torch.remainder(expected / self.slow_downsample, 1.0)
            return coarse_fraction[None].expand(
                self.n_delay_experts, *coarse_fraction.shape
            ) * self.route_mask[None]
        return self._fraction_full(
            self.slow_fraction_raw, self.slow_fraction_target_raw, self.slow_fraction_source_raw,
            self.slow_fraction_rank_raw,
        )

    def combined_fraction_expert(self) -> torch.Tensor:
        if self.force_zero_delay:
            return torch.zeros_like(self.fast_fraction_expert())
        if self.fixed_delay_steps is not None:
            fraction = self.fixed_delay_steps - math.floor(self.fixed_delay_steps)
            return torch.full_like(self.fast_fraction_expert(), fraction) * self.route_mask[None]
        if self.use_fixed_fold_posterior and bool(self.fold_evidence_ready):
            return self.fold_fraction_target[None].expand(
                self.n_delay_experts, *self.fold_fraction_target.shape
            )
        return self.fast_fraction_expert()

    def fast_fraction(self) -> torch.Tensor:
        return self.fast_fraction_expert().mean(0)

    def slow_fraction(self) -> torch.Tensor:
        return self.slow_fraction_expert().mean(0)

    @staticmethod
    def _expected_delay(probability: torch.Tensor, fraction: torch.Tensor) -> torch.Tensor:
        bins = torch.arange(probability.shape[-1], device=probability.device, dtype=probability.dtype)
        return (probability * bins).sum(-1) + fraction * (1.0 - probability[..., -1])

    def learned_delay_expert(self) -> torch.Tensor:
        return self._expected_delay(self.delay_prob_expert(), self.combined_fraction_expert())

    def learned_delay(self) -> torch.Tensor:
        return self.learned_delay_expert().mean(0)

    def learned_fast_delay(self) -> torch.Tensor:
        return self.learned_delay()

    def learned_delay_map(self) -> torch.Tensor:
        if self.force_zero_delay:
            return torch.zeros_like(self.learned_delay())
        probability = self.delay_prob()
        delay = probability.argmax(-1).to(probability.dtype)
        return delay + self.fast_fraction() * (delay < self.d_max).to(delay.dtype)

    def learned_slow_delay(self) -> torch.Tensor:
        return self.slow_downsample * self._expected_delay(self.slow_delay_prob(), self.slow_fraction())

    def learned_delay_seconds(self) -> torch.Tensor:
        return self.learned_delay() * self.timestep_seconds

    def _expert_probability(self, context: torch.Tensor | None, batch_size: int) -> torch.Tensor:
        if context is None:
            logits = self.expert_router.bias[None].expand(batch_size, -1)
        else:
            logits = self.expert_router(context)
        return torch.softmax(logits, dim=-1)

    def expert_mixture(
        self, context: torch.Tensor | None, batch_size: int, explore: bool = False
    ) -> torch.Tensor:
        probability = self._expert_probability(context, batch_size)
        routing_probability = probability
        if (
            explore
            and self.router_noise_scale > 0.0
            and float(self.training_progress_state) < self.router_warmup_fraction
        ):
            uniform = torch.rand_like(probability).clamp_(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(uniform))
            routing_probability = torch.softmax(
                probability.clamp_min(1e-8).log() + self.router_noise_scale * gumbel,
                dim=-1,
            )
        if self.expert_topk < self.n_delay_experts:
            _, indices = routing_probability.topk(self.expert_topk, dim=-1)
            mask = torch.zeros_like(probability).scatter_(1, indices, 1.0)
            sparse = routing_probability * mask
            sparse = sparse / sparse.sum(-1, keepdim=True).clamp_min(1e-8)
            # Sparse forward pass, dense router gradients so inactive experts can recover.
            probability = sparse + probability - probability.detach()
        return probability

    @staticmethod
    def _integer_stack(signal: torch.Tensor, max_delay: int) -> torch.Tensor:
        outputs = []
        n_time = signal.shape[-1]
        for delay in range(max_delay + 2):
            if delay == 0:
                outputs.append(signal)
            elif delay >= n_time:
                outputs.append(torch.zeros_like(signal))
            else:
                outputs.append(F.pad(signal[..., :-delay], (delay, 0)))
        return torch.stack(outputs, dim=-2)

    @staticmethod
    def _validity(n_time: int, max_delay: int, device, dtype) -> torch.Tensor:
        time = torch.arange(n_time, device=device)
        delay = torch.arange(max_delay + 2, device=device)[:, None]
        return (time[None, :] >= delay).to(dtype)

    def _fractional_pair_candidates(
        self, signal: torch.Tensor, max_delay: int, fraction: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stack = self._integer_stack(signal, max_delay)
        low = stack[:, None, :, : max_delay + 1, :]
        high_stack = torch.cat((stack[..., 1 : max_delay + 1, :], stack[..., max_delay : max_delay + 1, :]), dim=-2)
        high = high_stack[:, None]
        frac = fraction[None, :, :, None, None]
        candidates = (1.0 - frac) * low + frac * high
        validity = self._validity(signal.shape[-1], max_delay, signal.device, signal.real.dtype)
        valid_high = torch.cat((validity[1 : max_delay + 1], validity[max_delay : max_delay + 1]), 0)
        frac_mask = fraction[:, :, None, None]
        valid = (1.0 - frac_mask) * validity[: max_delay + 1][None, None]
        valid = valid + frac_mask * valid_high[None, None]
        return candidates, valid

    @staticmethod
    def _gcc_phat_evidence(signal: torch.Tensor, n_delays: int) -> torch.Tensor:
        n_time = signal.shape[-1]
        n_fft = max(2 * n_time, int(n_delays))
        spectrum = torch.fft.rfft(signal, n=n_fft, dim=-1)
        target = spectrum[:, :, None, :, None, :]
        source = spectrum[:, None, :, None, :, :]
        cross = target * source.conj()
        phat = cross / cross.abs().clamp_min(1e-6)
        corr = torch.fft.irfft(phat, n=n_fft, dim=-1)[..., :n_delays]
        return torch.softmax(8.0 * corr.mean(0), -1)

    @staticmethod
    def _lagged_correlation_evidence(
        signal: torch.Tensor, n_delays: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return lag likelihood and edge-existence reliability separately."""

        centered = signal - signal.mean(dim=-1, keepdim=True)
        normalized = centered / centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-4)
        correlations = []
        for delay in range(n_delays):
            if delay == 0:
                target = normalized
                source = normalized
            elif delay >= normalized.shape[-1]:
                correlations.append(
                    normalized.new_zeros(
                        normalized.shape[1], normalized.shape[1], normalized.shape[2], normalized.shape[2]
                    )
                )
                continue
            else:
                target = normalized[..., delay:]
                source = normalized[..., :-delay]
            cross = (
                target[:, :, None, :, None, :] * source[:, None, :, None, :, :]
            ).mean(dim=(0, -1))
            correlations.append(cross.abs())
        correlation = torch.stack(correlations, dim=-1).clamp(0.0, 1.0)
        peak = correlation.max(dim=-1).values
        background = correlation.median(dim=-1).values
        reliability = ((peak - background) / (1.0 - background).clamp_min(1e-4)).clamp(0.0, 1.0)
        return torch.softmax(8.0 * correlation, dim=-1), reliability

    @staticmethod
    def _evidence_band_projection(
        output_frequencies_hz: torch.Tensor,
        evidence_frequencies_hz: torch.Tensor,
    ) -> torch.Tensor:
        if evidence_frequencies_hz.numel() == 1:
            return output_frequencies_hz.new_ones(output_frequencies_hz.numel(), 1)
        spacing = torch.diff(evidence_frequencies_hz.sort().values).median().clamp_min(0.5)
        distance = (
            output_frequencies_hz[:, None] - evidence_frequencies_hz[None, :]
        ) / spacing
        return torch.softmax(-0.5 * distance.square(), dim=-1)

    def _aggregate_evidence_bands(
        self,
        evidence: torch.Tensor,
        output_frequencies_hz: torch.Tensor,
        evidence_frequencies_hz: torch.Tensor,
    ) -> torch.Tensor:
        projection = self._evidence_band_projection(
            output_frequencies_hz, evidence_frequencies_hz
        ).to(evidence)
        return torch.einsum("ae,efijd,bf->abijd", projection, evidence, projection)

    def _phase_slope_evidence(
        self,
        phase: torch.Tensor,
        confidence: torch.Tensor,
        evidence_frequencies_hz: torch.Tensor,
        n_delays: int,
        output_frequencies_hz: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_frequencies_hz is None:
            output_frequencies_hz = evidence_frequencies_hz[: self.n_bands]
        phasor = torch.polar(confidence, phase)
        delay_seconds = torch.arange(n_delays, device=phase.device, dtype=phase.dtype) * self.timestep_seconds
        cross_spectra = []
        for band in range(phase.shape[1]):
            cross = (phasor[:, band, :, None, :] * phasor[:, band, None, :, :].conj()).mean(dim=(0, -1))
            cross_spectra.append(cross)
        cross = torch.stack(cross_spectra, 0)
        weight = cross.abs().clamp_min(1e-6)
        rotation = torch.exp(
            1j
            * 2.0
            * math.pi
            * evidence_frequencies_hz[:, None]
            * delay_seconds[None, :]
        )
        # The complex magnitude analytically maximises over an unknown phase
        # intercept. Taking only the real part incorrectly assumes zero intercept.
        unit_cross = cross / cross.abs().clamp_min(1e-6)
        aligned = unit_cross[..., None] * rotation[:, None, None, :]
        projection = self._evidence_band_projection(
            output_frequencies_hz, evidence_frequencies_hz
        ).to(weight)
        weighted_alignment = weight[..., None] * aligned
        group_score = torch.einsum(
            "be,eijd->bijd", projection.to(weighted_alignment.dtype), weighted_alignment
        ).abs()
        denominator = torch.einsum("be,eij->bij", projection, weight)[..., None]
        shared_delay = torch.softmax(6.0 * group_score / denominator.clamp_min(1e-6), -1)
        uniform = phase.new_full((self.n_channels, self.n_channels, n_delays), 1.0 / n_delays)
        return torch.stack(
            [
                torch.stack(
                    [shared_delay[target_band] if target_band == source_band else uniform for source_band in range(self.n_bands)],
                    0,
                )
                for target_band in range(self.n_bands)
            ],
            0,
        )

    @staticmethod
    def _directed_connectivity_prior(
        phase: torch.Tensor, confidence: torch.Tensor, n_output_bands: int
    ) -> torch.Tensor:
        """PSI/imaginary-coherence prior robust to instantaneous common sources."""

        phasor = torch.polar(confidence, phase)
        cross = (
            phasor[:, :, :, None, :] * phasor[:, :, None, :, :].conj()
        ).mean(dim=(0, -1))
        imaginary = cross.imag.abs().mean(0)
        if cross.shape[0] > 1:
            # C_ij(f)=exp(-j*2*pi*f*tau) for source j -> delayed target i.
            # Match the offline convention: positive support denotes j -> i.
            psi = -(cross[:-1].conj() * cross[1:]).imag.sum(0)
        else:
            psi = torch.zeros_like(imaginary)
        off_diagonal = ~torch.eye(psi.shape[0], device=psi.device, dtype=torch.bool)
        scale = psi[off_diagonal].abs().median().clamp_min(1e-4)
        direction = torch.sigmoid(psi / scale)
        lag_scale = imaginary[off_diagonal].median().clamp_min(1e-4)
        lag_strength = (imaginary / (imaginary + lag_scale)).clamp(0.0, 1.0)
        node_prior = direction * lag_strength
        node_prior.fill_diagonal_(0.0)
        return node_prior[None, None].expand(
            n_output_bands, n_output_bands, -1, -1
        )

    def _coarse_evidence(self, evidence: torch.Tensor) -> torch.Tensor:
        bins = []
        for coarse in range(self.slow_d_max + 1):
            start = coarse * self.slow_downsample
            stop = min(start + self.slow_downsample, self.d_max + 1)
            if start >= evidence.shape[-1]:
                bins.append(torch.zeros_like(evidence[..., 0]))
            else:
                bins.append(evidence[..., start:stop].amax(-1))
        coarse = torch.stack(bins, -1)
        return coarse / coarse.sum(-1, keepdim=True).clamp_min(1e-8)

    def _evidence_route_selection(
        self, evidence: torch.Tensor, route_reliability: torch.Tensor
    ) -> torch.Tensor:
        """Select reliable routes before fitting shared low-rank delay factors."""

        if bool(self.fold_evidence_ready):
            return (self.fold_route_prior > 0.5).to(evidence.dtype) * self.route_mask

        confidence = self._entropy_confidence(
            evidence / evidence.sum(-1, keepdim=True).clamp_min(1e-8)
        )
        score = (
            confidence
            * route_reliability
            * (0.5 + self.connectivity_prior_ema)
            * self.route_mask
        )
        flat_score = score.permute(0, 2, 1, 3).flatten(2)
        flat_mask = self.route_mask.permute(0, 2, 1, 3).flatten(2)
        available = int(flat_mask[0, 0].sum().item())
        keep = max(1, min(available, round(self.graph_sparsity * available)))
        flat_score = flat_score.masked_fill(
            flat_mask == 0, torch.finfo(flat_score.dtype).min
        )
        candidate = (flat_score > 0.5).to(flat_score.dtype) * flat_mask
        indices = flat_score.topk(keep, dim=-1).indices
        capped = torch.zeros_like(flat_score).scatter_(-1, indices, 1.0) * flat_mask
        exceeds_cap = candidate.sum(-1, keepdim=True) > keep
        selected = torch.where(exceeds_cap, capped * candidate, candidate)
        return selected.reshape(
            self.n_bands, self.n_channels, self.n_bands, self.n_channels
        ).permute(0, 2, 1, 3)

    @staticmethod
    def _posterior_kl(probability: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        return (
            probability.clamp_min(1e-8)
            * (probability.clamp_min(1e-8).log() - anchor.clamp_min(1e-8).log())
        ).sum(-1).mean()

    def _event_pair_contribution(
        self,
        source_carrier: torch.Tensor,
        source_envelope: torch.Tensor,
        source_phase: torch.Tensor,
        source_confidence: torch.Tensor,
        target_phase: torch.Tensor,
        target_confidence: torch.Tensor,
        target_envelope: torch.Tensor,
        fraction: torch.Tensor,
        delay_probability: torch.Tensor,
        phase_offset: torch.Tensor,
        route_weight: torch.Tensor,
        modulation_gain: torch.Tensor,
        target_band: int,
        source_band: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transport one event band-pair with optional rematerialisation."""

        carrier_candidates, valid = self._fractional_pair_candidates(
            source_carrier, self.d_max, fraction
        )
        envelope_candidates, _ = self._fractional_pair_candidates(
            source_envelope, self.d_max, fraction
        )
        phase_candidates, _ = self._fractional_pair_candidates(
            torch.polar(torch.ones_like(source_phase), source_phase),
            self.d_max,
            fraction,
        )
        confidence_candidates, _ = self._fractional_pair_candidates(
            source_confidence, self.d_max, fraction
        )
        target_phase_view = target_phase[:, :, None, None, :]
        target_confidence_view = target_confidence[:, :, None, None, :]
        phase_offset_view = phase_offset[None, :, :, None, None]
        if target_band == source_band:
            coupling = torch.cos(
                target_phase_view - torch.angle(phase_candidates) - phase_offset_view
            )
            phase_weight = target_confidence_view * confidence_candidates
            gate = torch.sigmoid(self.phase_gain * coupling)
        elif target_band > source_band:
            coupling = torch.cos(torch.angle(phase_candidates) - phase_offset_view)
            phase_weight = confidence_candidates * target_confidence_view
            gate = torch.sigmoid(self.phase_gain * coupling)
        else:
            phase_weight = confidence_candidates * target_confidence_view
            gate = torch.ones_like(envelope_candidates)
            coupling = torch.zeros_like(gate)

        probability = delay_probability[None, :, :, :, None] * valid[None]
        relative_weight = gate * probability
        normalization = relative_weight.sum(3).clamp_min(1e-6)
        carrier_mixed = (
            carrier_candidates * relative_weight
        ).sum(3) / normalization
        envelope_mixed = (
            envelope_candidates * relative_weight
        ).sum(3) / normalization
        transport_confidence = (
            phase_weight * relative_weight
        ).sum(3) / normalization
        coupling_strength = relative_weight.sum(3) / probability.sum(3).clamp_min(
            1e-6
        )
        if target_band == source_band:
            routed = (
                carrier_mixed + modulation_gain * envelope_mixed
            ) * coupling_strength
        else:
            target_event = target_envelope[:, :, None, :].abs()
            routed = (
                envelope_mixed
                * (1.0 + modulation_gain * target_event)
                * coupling_strength
            )
        routed = routed * transport_confidence
        contribution = torch.einsum("ij,nijt->nit", route_weight, routed)

        phase_supervision_weight = (
            phase_weight
            if target_band >= source_band
            else torch.zeros_like(phase_weight)
        )
        phase_supervision_weight = phase_supervision_weight * probability.detach()
        phase_numerator = (
            (1.0 - coupling) * phase_supervision_weight
        ).sum()
        phase_denominator = phase_supervision_weight.sum()
        return contribution, gate.sum(), phase_numerator, phase_denominator

    def forward(
        self,
        carrier: torch.Tensor,
        phase: torch.Tensor,
        phase_confidence: torch.Tensor,
        envelope: torch.Tensor | None = None,
        band_frequencies_hz: torch.Tensor | None = None,
        evidence_signal: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        evidence_phase: torch.Tensor | None = None,
        evidence_confidence: torch.Tensor | None = None,
        evidence_frequencies_hz: torch.Tensor | None = None,
        evidence_envelope: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if carrier.shape != phase.shape or carrier.shape != phase_confidence.shape:
            raise ValueError("carrier, phase, and confidence must share [N,B,C,T]")
        envelope = carrier.abs() if envelope is None else envelope
        evidence_signal = carrier if evidence_signal is None else evidence_signal
        if band_frequencies_hz is None:
            band_frequencies_hz = torch.arange(1, self.n_bands + 1, device=carrier.device, dtype=carrier.dtype)

        slow_steps = max(2, math.ceil(carrier.shape[-1] / self.slow_downsample))
        slow_envelope = F.adaptive_avg_pool1d(
            envelope.flatten(0, 2).unsqueeze(1), slow_steps
        ).squeeze(1).reshape(*envelope.shape[:-1], slow_steps)
        if self.training and self.evidence_updates_enabled:
            with torch.no_grad():
                evidence_envelope = envelope if evidence_envelope is None else evidence_envelope
                carrier_gcc_raw, carrier_reliability_raw = self._lagged_correlation_evidence(
                    evidence_signal.detach(), self.d_max + 1
                )
                envelope_gcc_raw, envelope_fast_reliability_raw = self._lagged_correlation_evidence(
                    evidence_envelope.detach(), self.d_max + 1
                )
                slope_phase = phase if evidence_phase is None else evidence_phase
                slope_confidence = phase_confidence if evidence_confidence is None else evidence_confidence
                slope_frequencies = (
                    band_frequencies_hz
                    if evidence_frequencies_hz is None
                    else evidence_frequencies_hz
                )
                carrier_gcc = self._aggregate_evidence_bands(
                    carrier_gcc_raw, band_frequencies_hz, slope_frequencies
                )
                envelope_gcc = self._aggregate_evidence_bands(
                    envelope_gcc_raw, band_frequencies_hz, slope_frequencies
                )
                carrier_reliability = self._aggregate_evidence_bands(
                    carrier_reliability_raw[..., None], band_frequencies_hz, slope_frequencies
                ).squeeze(-1)
                envelope_fast_reliability = self._aggregate_evidence_bands(
                    envelope_fast_reliability_raw[..., None], band_frequencies_hz, slope_frequencies
                ).squeeze(-1)
                slope = self._phase_slope_evidence(
                    slope_phase.detach(),
                    slope_confidence.detach(),
                    slope_frequencies,
                    self.d_max + 1,
                    output_frequencies_hz=band_frequencies_hz,
                )
                same_band = torch.eye(self.n_bands, device=carrier.device, dtype=carrier.dtype)[:, :, None, None, None]
                # Lag correlation is the robust finite-window estimator; the
                # phase-slope term refines it only when the group-delay model fits.
                fast_evidence = same_band * (0.75 * carrier_gcc + 0.25 * slope) + (1.0 - same_band) * envelope_gcc
                slope_reliability = self._entropy_confidence(slope)
                same_route = same_band.squeeze(-1)
                fast_route_reliability = same_route * torch.maximum(
                    carrier_reliability, slope_reliability
                ) + (1.0 - same_route) * envelope_fast_reliability
                envelope_coarse = self._gcc_phat_evidence(slow_envelope.detach(), self.slow_d_max + 1)
                carrier_coarse = self._coarse_evidence(fast_evidence)
                if envelope_coarse.shape[-1] > 1:
                    nonzero = envelope_coarse[..., 1:].amax(-1)
                    envelope_reliability = (5.0 * (nonzero - envelope_coarse[..., 0])).clamp(0, 1)
                else:
                    envelope_reliability = envelope_coarse[..., 0].new_zeros(envelope_coarse.shape[:-1])
                slow_evidence = (
                    (1.0 - envelope_reliability[..., None]) * carrier_coarse
                    + envelope_reliability[..., None] * envelope_coarse
                )
                weight = float(carrier.shape[0])
                self.fast_evidence_accum.add_(fast_evidence, alpha=weight)
                self.slow_evidence_accum.add_(slow_evidence, alpha=weight)
                self.reliability_accum.add_(envelope_reliability, alpha=weight)
                self.route_reliability_accum.add_(fast_route_reliability, alpha=weight)
                connectivity = self._directed_connectivity_prior(
                    slope_phase.detach(), slope_confidence.detach(), self.n_bands
                ) * self.route_mask
                self.connectivity_prior_accum.add_(connectivity, alpha=weight)
                self.evidence_accum_weight.add_(weight)
        else:
            fast_evidence = self.fast_evidence_ema
            slow_evidence = self.slow_evidence_ema
            envelope_reliability = self.coarse_reliability_ema
            fast_route_reliability = self.route_reliability_ema

        fast_prob = self.fast_delay_prob_expert()
        slow_prob = self.slow_delay_prob_expert()
        total_prob = self._total_delay_prob(fast_prob, slow_prob)
        hurdle_route = self.delay_route_probability_expert()
        total_fraction = self.combined_fraction_expert()
        phase_pref = self.effective_phase_pref_expert()
        route_weight = (
            self.edge_weight_expert()
            * self.delay_confidence_expert(total_prob)
            * hurdle_route
        )
        soft_mixture = self._expert_probability(context, carrier.shape[0])
        mixture = self.expert_mixture(
            context,
            carrier.shape[0],
            explore=self.training and not self.matched_transport_control,
        )
        expert_current = carrier.new_zeros(
            carrier.shape[0], self.n_delay_experts, self.n_bands, self.n_bands,
            self.n_channels, carrier.shape[-1]
        )
        gate_sum = carrier.new_zeros(())
        gate_count = 0
        phase_evidence_numerator = carrier.new_zeros(())
        phase_evidence_denominator = carrier.new_zeros(())
        for expert in range(self.n_delay_experts):
            active = torch.nonzero(mixture[:, expert].detach() != 0, as_tuple=False).flatten()
            if active.numel() == 0:
                continue
            carrier_active = carrier.index_select(0, active)
            phase_active = phase.index_select(0, active)
            confidence_active = phase_confidence.index_select(0, active)
            envelope_active = envelope.index_select(0, active)
            slow_envelope_active = slow_envelope.index_select(0, active)
            modulation_gain = F.softplus(self.slow_modulation_raw[expert])
            for target_band in range(self.n_bands):
                for source_band in range(self.n_bands):
                    fraction = total_fraction[expert, target_band, source_band]
                    if self.event_native_transport:
                        pair_inputs = (
                            carrier_active[:, source_band],
                            envelope_active[:, source_band],
                            phase_active[:, source_band],
                            confidence_active[:, source_band],
                            phase_active[:, target_band],
                            confidence_active[:, target_band],
                            envelope_active[:, target_band],
                            fraction,
                            total_prob[expert, target_band, source_band],
                            phase_pref[expert, target_band, source_band],
                            route_weight[expert, target_band, source_band],
                            modulation_gain,
                        )

                        def event_pair(
                            *values,
                            target_band_index=target_band,
                            source_band_index=source_band,
                        ):
                            return self._event_pair_contribution(
                                *values,
                                target_band=target_band_index,
                                source_band=source_band_index,
                            )

                        if self.training and self.checkpoint_event_routes:
                            contribution, pair_gate, pair_phase_num, pair_phase_den = checkpoint(
                                event_pair,
                                *pair_inputs,
                                use_reentrant=False,
                            )
                        else:
                            contribution, pair_gate, pair_phase_num, pair_phase_den = event_pair(
                                *pair_inputs
                            )
                        expert_current[
                            active, expert, target_band, source_band
                        ] = (
                            expert_current[
                                active, expert, target_band, source_band
                            ]
                            + contribution
                        )
                        gate_sum = gate_sum + pair_gate
                        gate_count += (
                            active.numel()
                            * self.n_channels
                            * self.n_channels
                            * (self.d_max + 1)
                            * carrier.shape[-1]
                        )
                        phase_evidence_numerator = (
                            phase_evidence_numerator + pair_phase_num
                        )
                        phase_evidence_denominator = (
                            phase_evidence_denominator + pair_phase_den
                        )
                        continue
                    carrier_candidates, valid = self._fractional_pair_candidates(
                        carrier_active[:, source_band], self.d_max, fraction
                    )
                    envelope_candidates, _ = self._fractional_pair_candidates(
                        envelope_active[:, source_band], self.d_max, fraction
                    )
                    phase_candidates, _ = self._fractional_pair_candidates(
                        torch.polar(
                            torch.ones_like(phase_active[:, source_band]),
                            phase_active[:, source_band],
                        ),
                        self.d_max,
                        fraction,
                    )
                    confidence_candidates, _ = self._fractional_pair_candidates(
                        confidence_active[:, source_band], self.d_max, fraction
                    )
                    target_phase = phase_active[:, target_band, :, None, None, :]
                    target_confidence = confidence_active[:, target_band, :, None, None, :]
                    phase_offset = phase_pref[expert, target_band, source_band][None, :, :, None, None]
                    if target_band == source_band:
                        coupling = torch.cos(
                            target_phase - torch.angle(phase_candidates) - phase_offset
                        )
                        phase_weight = target_confidence * confidence_candidates
                        gate = torch.sigmoid(self.phase_gain * coupling)
                    elif target_band > source_band:
                        # Cross-frequency routes use delayed source phase to gate
                        # higher-band target amplitude (low-to-high PAC).
                        coupling = torch.cos(torch.angle(phase_candidates) - phase_offset)
                        phase_weight = confidence_candidates * target_confidence
                        gate = torch.sigmoid(self.phase_gain * coupling)
                    else:
                        # High-to-low routes are delayed envelope interactions,
                        # not invalid unequal-frequency phase subtraction.
                        phase_weight = confidence_candidates * target_confidence
                        gate = torch.ones_like(envelope_candidates)
                        coupling = torch.zeros_like(gate)
                    probability = total_prob[expert, target_band, source_band][None, :, :, :, None] * valid[None]
                    relative_weight = gate * probability
                    normalization = relative_weight.sum(3).clamp_min(1e-6)
                    carrier_mixed = (carrier_candidates * relative_weight).sum(3) / normalization
                    envelope_mixed = (envelope_candidates * relative_weight).sum(3) / normalization
                    transport_confidence = (phase_weight * relative_weight).sum(3) / normalization
                    coupling_strength = relative_weight.sum(3) / probability.sum(3).clamp_min(1e-6)
                    phase_supervision_weight = (
                        phase_weight
                        if target_band >= source_band
                        else torch.zeros_like(phase_weight)
                    )
                    phase_supervision_weight = phase_supervision_weight * probability.detach()
                    phase_evidence_numerator = phase_evidence_numerator + (
                        (1.0 - coupling) * phase_supervision_weight
                    ).sum()
                    phase_evidence_denominator = phase_evidence_denominator + phase_supervision_weight.sum()
                    slow_candidates, slow_valid = self._fractional_pair_candidates(
                        slow_envelope_active[:, source_band],
                        self.slow_d_max,
                        self.slow_fraction_expert()[expert, target_band, source_band],
                    )
                    slow_weight = slow_prob[expert, target_band, source_band][None, :, :, :, None] * slow_valid[None]
                    slow_mixed = (slow_candidates * slow_weight).sum(3) / slow_weight.sum(3).clamp_min(1e-6)
                    shape = slow_mixed.shape
                    slow_mixed = F.interpolate(
                        slow_mixed.reshape(-1, 1, shape[-1]), size=carrier.shape[-1], mode="linear", align_corners=False
                    ).reshape(*shape[:-1], carrier.shape[-1])
                    if target_band == source_band:
                        routed = (
                            carrier_mixed
                            * (1.0 + modulation_gain * torch.tanh(slow_mixed))
                            + modulation_gain * torch.tanh(envelope_mixed)
                        ) * coupling_strength
                    else:
                        # Cross-band traffic is an explicitly pair-labelled,
                        # delayed envelope/PAC interaction. A source carrier is
                        # never relabelled as a target-band carrier.
                        target_envelope = torch.tanh(
                            envelope_active[:, target_band, :, None, :]
                        )
                        routed = (
                            torch.tanh(envelope_mixed)
                            * target_envelope
                            * coupling_strength
                        )
                    # Confidence controls transported magnitude after relative
                    # lag weights are normalised, so low-amplitude phase cannot
                    # cancel between numerator and denominator.
                    routed = routed * transport_confidence
                    contribution = torch.einsum(
                        "ij,nijt->nit", route_weight[expert, target_band, source_band], routed
                    )
                    expert_current[active, expert, target_band, source_band] = (
                        expert_current[active, expert, target_band, source_band] + contribution
                    )
                    gate_sum += gate.sum()
                    gate_count += gate.numel()
        expert_current = expert_current / math.sqrt(max(1, self.n_bands * self.n_channels))
        current = torch.einsum("ne,neabct->nabct", mixture, expert_current)
        phase_evidence_loss = phase_evidence_numerator / phase_evidence_denominator.clamp_min(1.0)

        fast_entropy = -(fast_prob.clamp_min(1e-8) * fast_prob.clamp_min(1e-8).log()).sum(-1)
        slow_entropy = -(slow_prob.clamp_min(1e-8) * slow_prob.clamp_min(1e-8).log()).sum(-1)
        route_mask = self.route_mask[None]
        evidence_selection = self._evidence_route_selection(
            fast_evidence, fast_route_reliability
        )[None]
        entropy = ((fast_entropy + slow_entropy) * evidence_selection).sum() / (
            2.0 * evidence_selection.sum().clamp_min(1) * self.n_delay_experts
        )
        route_target = (
            self.fold_route_prior
            if bool(self.fold_evidence_ready)
            else (fast_route_reliability * evidence_selection.squeeze(0)).clamp(0.0, 1.0)
        ) * self.route_mask
        route_prediction = hurdle_route.clamp(1e-6, 1.0 - 1e-6)
        route_target_expert = route_target[None].expand_as(route_prediction)
        route_bce = F.binary_cross_entropy(
            route_prediction,
            route_target_expert,
            reduction="none",
        ) * route_mask
        positive_routes = (route_target_expert > 0.5).to(route_bce.dtype) * route_mask
        negative_routes = (route_target_expert <= 0.5).to(route_bce.dtype) * route_mask
        route_loss = 0.5 * (
            (route_bce * positive_routes).sum() / positive_routes.sum().clamp_min(1.0)
            + (route_bce * negative_routes).sum() / negative_routes.sum().clamp_min(1.0)
        )
        edge_target = self.effective_edge_prior()[None].expand_as(
            self.edge_weight_expert()
        )
        edge_prior_error = F.smooth_l1_loss(
            self.edge_weight_expert().abs(), edge_target, reduction="none"
        )
        edge_prior_loss = (
            edge_prior_error * route_target_expert
        ).sum() / route_target_expert.sum().clamp_min(1.0)
        confidence_prediction = torch.sigmoid(
            self._delay_confidence_logits_full()
        ).clamp(1e-6, 1.0 - 1e-6)
        confidence_bce = F.binary_cross_entropy(
            confidence_prediction, route_target_expert, reduction="none"
        ) * route_mask
        confidence_prior_loss = confidence_bce.sum() / (
            route_mask.sum() * self.n_delay_experts
        ).clamp_min(1.0)

        fast_target = (
            self.fold_positive_delay_prior
            if bool(self.fold_evidence_ready)
            else fast_evidence
        )
        fast_target = fast_target / fast_target.sum(-1, keepdim=True).clamp_min(1e-8)
        slow_target = slow_evidence
        slow_target = slow_target / slow_target.sum(-1, keepdim=True).clamp_min(1e-8)
        fast_conditional = fast_prob.clamp_min(1e-8)
        slow_conditional = slow_prob.clamp_min(1e-8)
        fast_kl = (
            fast_target[None]
            * (fast_target[None].clamp_min(1e-8).log() - fast_conditional.log())
        ).sum(-1)
        slow_kl = (
            slow_target[None]
            * (slow_target[None].clamp_min(1e-8).log() - slow_conditional.log())
        ).sum(-1)
        lag_weight = route_target[None] * route_mask
        positive_delay_kl = (
            ((fast_kl + slow_kl) * lag_weight).sum()
            / (2.0 * lag_weight.sum().clamp_min(1.0))
        )
        alignment = positive_delay_kl
        fraction_target = self.fold_fraction_target[None]
        fraction_loss = F.smooth_l1_loss(
            self.fast_fraction_expert(),
            fraction_target.expand_as(self.fast_fraction_expert()),
            reduction="none",
        )
        fraction_loss = (fraction_loss * lag_weight).sum() / lag_weight.sum().clamp_min(1.0)
        anchor_kl = current.new_zeros(())
        if bool(self.anchor_ready) and self.fast_anchor_prob.numel():
            anchor_kl = self._posterior_kl(fast_prob, self.fast_anchor_prob)
            anchor_kl += self._posterior_kl(slow_prob, self.slow_anchor_prob)
        selection = self.effective_edge_selection_expert()
        effective_gate_expert = (
            self.delay_confidence_expert(total_prob)
            * hurdle_route
        )
        available = route_mask.sum(dim=(2, 4)).clamp_min(1.0)
        incoming_density = (hurdle_route * route_mask).sum(dim=(2, 4)) / available
        target_density = self.graph_sparsity if self.graph_sparsity > 0 else 1.0
        gate_structure = F.relu(incoming_density - target_density).square().mean()
        target_usage = 1.0 / self.n_delay_experts
        soft_balance = (soft_mixture.mean(0) - target_usage).square().mean()
        hard_balance = (mixture.mean(0) - target_usage).square().mean()
        expert_balance_loss = soft_balance + 5.0 * hard_balance
        expert_entropy = -(mixture.clamp_min(1e-8) * mixture.clamp_min(1e-8).log()).sum(-1).mean()
        flattened_delay = total_prob.reshape(self.n_delay_experts, -1)
        flattened_delay = F.normalize(flattened_delay, p=2, dim=-1)
        similarity = flattened_delay @ flattened_delay.transpose(0, 1)
        diversity_mask = ~torch.eye(self.n_delay_experts, device=similarity.device, dtype=torch.bool)
        expert_diversity_loss = similarity[diversity_mask].mean() if self.n_delay_experts > 1 else current.new_zeros(())

        return current, {
            "edge_weight": self.edge_weight(),
            "edge_existence": self.edge_existence(),
            "edge_selection": selection.mean(0),
            "delay_confidence": self.delay_confidence(total_prob.mean(0)),
            "delay": self.learned_delay(),
            "slow_delay": self.learned_slow_delay(),
            "delay_seconds": self.learned_delay_seconds(),
            "delay_prob": total_prob.mean(0),
            "delay_null_probability": 1.0 - hurdle_route.mean(0),
            "online_route_log_odds": torch.logit(
                hurdle_route.clamp(1e-6, 1.0 - 1e-6)
            ),
            "fast_carrier_prob": fast_prob.mean(0),
            "slow_delay_prob": slow_prob.mean(0),
            "phase_pref": phase_pref.mean(0),
            "phase_gate_mean": gate_sum / max(1, gate_count),
            "delay_entropy": entropy,
            "delay_gate": effective_gate_expert.mean(0),
            "delay_gate_structure_loss": gate_structure,
            "delayed_current": current,
            "edge_prior_loss": edge_prior_loss,
            "route_prior_loss": route_loss,
            "delay_confidence_prior_loss": confidence_prior_loss,
            "positive_delay_kl_loss": positive_delay_kl,
            "fractional_delay_loss": fraction_loss,
            "delay_alignment_loss": alignment,
            "phase_evidence_loss": phase_evidence_loss,
            "delay_anchor_kl": anchor_kl,
            "fast_delay_evidence": fast_evidence,
            "evidence_route_selection": evidence_selection.squeeze(0),
            "slow_delay_evidence": slow_evidence,
            "envelope_delay_reliability": envelope_reliability.mean(),
            "route_reliability": fast_route_reliability,
            "coarse_reliability": self.coarse_reliability_ema.mean(),
            "fast_fraction": self.fast_fraction(),
            "slow_fraction": self.slow_fraction(),
            "slow_modulation_gain": F.softplus(self.slow_modulation_raw).mean(),
            "expert_mixture": mixture,
            "expert_balance_loss": expert_balance_loss,
            "expert_diversity_loss": expert_diversity_loss,
            "expert_entropy": expert_entropy,
            "expert_soft_mixture": soft_mixture,
            "connectivity_prior": self.connectivity_prior_ema,
            "expert_current": expert_current,
            "event_native_transport": current.new_tensor(
                self.event_native_transport
            ),
        }
