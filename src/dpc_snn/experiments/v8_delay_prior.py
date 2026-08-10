"""Fold-local audited delay priors for the V8 accuracy-first programme."""

from __future__ import annotations

import hashlib
from typing import Any, Literal, Sequence

import numpy as np
import torch

from dpc_snn.analysis.evidence_space import fold_local_continuous_delay_prior
from dpc_snn.experiments.v8_evidence_stability import (
    evaluate_v8_evidence_seed_stability,
)
from dpc_snn.models.coupled_dual_delay import posterior_fractional_causal_shift

from .v8_training import V8CachedRates


V8_PRIOR_FIELDS = (
    "source_band",
    "source_node",
    "target_band",
    "target_node",
    "delay_probability",
    "fractional_target",
    "route_weight",
    "route_confidence",
    "phase_preference",
    "amplitude_scale",
)

V8RouteScope = Literal["within_band", "cross_band", "all"]
V8PriorScope = Literal["inner", "outer"]


def v8_delay_prior_seed(
    subject: int,
    fold: int,
    replicate: int = 0,
    *,
    scope: V8PriorScope = "inner",
) -> int:
    """Return the one canonical seed schedule shared by audit and training."""

    if (
        int(subject) < 1
        or int(fold) not in range(6)
        or int(replicate) < 0
        or scope not in ("inner", "outer")
    ):
        raise ValueError("invalid subject, fold, or replicate for a V8 delay-prior seed")
    return (
        8_300_003
        + int(subject) * 10_007
        + int(fold) * 1_009
        + int(replicate) * 1_000_003
        + (500_009 if scope == "outer" else 0)
    )


class V8DelayEvidenceRejected(RuntimeError):
    """Carry rejected fold-local evidence to the artifact boundary."""

    def __init__(
        self,
        message: str,
        *,
        arrays: dict[str, np.ndarray],
        summary: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.arrays = {name: np.asarray(value) for name, value in arrays.items()}
        self.summary = dict(summary)


def v8_delay_prior_digest(prior: dict[str, torch.Tensor]) -> str:
    """Hash the exact sparse route tensors in a stable field order."""

    if set(prior) != set(V8_PRIOR_FIELDS):
        raise ValueError("V8 delay prior has missing or unexpected fields")
    digest = hashlib.sha256()
    for name in V8_PRIOR_FIELDS:
        value = torch.as_tensor(prior[name]).detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _validate_evidence_shapes(
    rates: V8CachedRates,
    arrays: dict[str, np.ndarray],
    *,
    maximum_delay: int,
) -> None:
    bands, nodes = rates.fast.shape[1:3]
    route_shape = (bands, bands, nodes, nodes)
    required = {
        "route_probability": route_shape,
        "positive_delay_probability": (*route_shape, int(maximum_delay) + 1),
        "fractional_delay_target": route_shape,
        "connectivity_prior": route_shape,
        "band_pair_accepted": route_shape,
    }
    for name, shape in required.items():
        if name not in arrays or np.asarray(arrays[name]).shape != shape:
            raise ValueError(f"V8 evidence field {name!r} must have shape {shape}")


def sparse_v8_prior_from_evidence(
    rates: V8CachedRates,
    arrays: dict[str, np.ndarray],
    summary: dict[str, Any],
    *,
    maximum_routes: int,
    maximum_delay: int,
    route_scope: V8RouteScope = "within_band",
    require_evidence_pipeline_passed: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Convert accepted target/source evidence axes into immutable sparse routes."""

    if require_evidence_pipeline_passed and not bool(summary.get("evidence_pipeline_passed")):
        raise V8DelayEvidenceRejected(
            "fold-local delay evidence failed its negative-control gate",
            arrays=arrays,
            summary=summary,
        )
    _validate_evidence_shapes(rates, arrays, maximum_delay=maximum_delay)
    if maximum_routes < 1:
        raise ValueError("maximum_routes must be positive")

    accepted = np.asarray(arrays["band_pair_accepted"], dtype=bool)
    route_probability = np.asarray(arrays["route_probability"], dtype=np.float64)
    bands, nodes = rates.fast.shape[1:3]
    within_band = np.eye(bands, dtype=bool)[:, :, None, None]
    if route_scope == "within_band":
        band_scope = within_band
    elif route_scope == "cross_band":
        band_scope = ~within_band
    elif route_scope == "all":
        band_scope = np.ones_like(within_band)
    else:
        raise ValueError("route_scope must be 'within_band', 'cross_band', or 'all'")
    non_identity = ~np.eye(nodes, dtype=bool)[None, None]
    eligible = accepted & band_scope & non_identity & (route_probability > 0.0)
    coordinates = np.argwhere(eligible)
    if coordinates.size == 0:
        raise V8DelayEvidenceRejected(
            f"audited V8 evidence contains no accepted {route_scope.replace('_', '-')} route",
            arrays=arrays,
            summary=summary,
        )
    scores = route_probability[tuple(coordinates.T)]
    order = np.lexsort(
        (
            coordinates[:, 3],
            coordinates[:, 2],
            coordinates[:, 1],
            coordinates[:, 0],
            -scores,
        )
    )
    coordinates = coordinates[order[: int(maximum_routes)]]
    target_band, source_band, target_node, source_node = coordinates.T

    probability = torch.from_numpy(
        np.asarray(arrays["positive_delay_probability"], dtype=np.float32)[
            target_band, source_band, target_node, source_node
        ]
    )
    fraction = torch.from_numpy(
        np.asarray(arrays["fractional_delay_target"], dtype=np.float32)[
            target_band, source_band, target_node, source_node
        ]
    )
    source = rates.fast[:, source_band, source_node]
    target = rates.fast[:, target_band, target_node]
    delayed = posterior_fractional_causal_shift(
        source,
        probability,
        fraction,
        max_delay=int(maximum_delay),
    )
    trim = min(int(maximum_delay) + 1, delayed.shape[-1] - 1)
    source_amplitude = source[..., trim:].abs()
    target_amplitude = target[..., trim:].abs()
    amplitude_scale = torch.cat((source_amplitude, target_amplitude), dim=0).median(dim=0).values
    amplitude_scale = amplitude_scale.median(dim=-1).values.clamp_min(1e-6)
    delayed_confidence = delayed[..., trim:].abs() / (
        delayed[..., trim:].abs() + amplitude_scale[None, :, None]
    )
    target_confidence = target[..., trim:].abs() / (
        target[..., trim:].abs() + amplitude_scale[None, :, None]
    )
    cross = (
        delayed_confidence
        * target_confidence
        * target[..., trim:]
        * delayed[..., trim:].conj()
    ).sum(dim=(0, 2))
    phase_preference = torch.angle(cross).float()

    selected_weight = torch.from_numpy(
        route_probability[target_band, source_band, target_node, source_node].astype(
            np.float32
        )
    )
    confidence = np.asarray(arrays["connectivity_prior"], dtype=np.float32)[
        target_band, source_band, target_node, source_node
    ]
    selected_confidence = torch.from_numpy(np.clip(confidence, 1e-3, 1.0))
    prior = {
        "source_band": torch.from_numpy(source_band.astype(np.int64)),
        "source_node": torch.from_numpy(source_node.astype(np.int64)),
        "target_band": torch.from_numpy(target_band.astype(np.int64)),
        "target_node": torch.from_numpy(target_node.astype(np.int64)),
        "delay_probability": probability.float(),
        "fractional_target": fraction.float(),
        "route_weight": selected_weight.float(),
        "route_confidence": selected_confidence.float(),
        "phase_preference": phase_preference,
        "amplitude_scale": amplitude_scale.float(),
    }
    digest = v8_delay_prior_digest(prior)
    enriched = {
        **summary,
        "sparse_route_policy": f"accepted_{route_scope}_top_probability_ceiling",
        "route_scope": route_scope,
        "accepted_routes_before_ceiling": int(eligible.sum()),
        "selected_sparse_routes": int(coordinates.shape[0]),
        "maximum_sparse_routes": int(maximum_routes),
        "maximum_delay_samples": int(maximum_delay),
        "phase_preference": "fold_train_confidence_weighted_circular_intercept",
        "amplitude_scale": "fold_train_route_median_analytic_amplitude",
        "prior_sha256": digest,
    }
    return prior, enriched


def fit_v8_fold_delay_prior(
    rates: V8CachedRates,
    *,
    evidence_rates: V8CachedRates | None = None,
    analytic_representation: str = "v8_exact_online_physical_fast_after_fold_train_fixed_gain",
    band_edges_hz: Sequence[Sequence[float]],
    task_seconds: float,
    maximum_routes: int,
    maximum_delay: int,
    route_scope: V8RouteScope = "within_band",
    bootstrap_samples: int,
    grid_oversample: int,
    minimum_bayes_factor: float,
    minimum_bootstrap_frequency: float,
    minimum_direction_probability: float,
    seed: int,
    split_strata: Sequence[str] | None = None,
    surrogate_replicates: int = 1,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, np.ndarray]]:
    """Fit and sparsify one audited prior using only one active training fold."""

    audit_rates = rates if evidence_rates is None else evidence_rates
    if audit_rates.fast.shape[1:] != rates.fast.shape[1:]:
        raise ValueError("V8 evidence and transport rates must share band/node/time axes")
    edges = [[float(low), float(high)] for low, high in band_edges_hz]
    if len(edges) != audit_rates.fast.shape[1]:
        raise ValueError("band_edges_hz does not match the V8 rate cache")
    arrays, summary = fold_local_continuous_delay_prior(
        {"X": np.empty((audit_rates.fast.shape[0], 1, 1), dtype=np.float32)},
        edges,
        edges,
        n_nodes=audit_rates.fast.shape[2],
        graph_steps=audit_rates.fast.shape[-1],
        max_delay_steps=int(maximum_delay),
        bootstrap_samples=int(bootstrap_samples),
        grid_oversample=int(grid_oversample),
        min_bayes_factor=float(minimum_bayes_factor),
        min_bootstrap_frequency=float(minimum_bootstrap_frequency),
        min_direction_probability=float(minimum_direction_probability),
        seed=int(seed),
        device="cpu",
        task_tmin=0.0,
        task_tmax=float(task_seconds),
        analytic_features=audit_rates.fast.detach().cpu().numpy(),
        analytic_representation=str(analytic_representation),
        split_strata=split_strata,
        surrogate_replicates=int(surrogate_replicates),
    )
    prior, enriched = sparse_v8_prior_from_evidence(
        rates,
        arrays,
        summary,
        maximum_routes=int(maximum_routes),
        maximum_delay=int(maximum_delay),
        route_scope=route_scope,
    )
    enriched = {
        **enriched,
        "evidence_frontend_fingerprint": audit_rates.physical_frontend_fingerprint,
        "transport_frontend_fingerprint": rates.physical_frontend_fingerprint,
        "evidence_transport_axes_matched": True,
    }
    return prior, enriched, arrays


def _aggregate_v8_evidence_replicates(
    evidence: Sequence[dict[str, np.ndarray]],
    *,
    consensus_frequency: float,
) -> dict[str, np.ndarray]:
    if not evidence:
        raise ValueError("V8 evidence aggregation requires at least one replicate")
    names = set(evidence[0])
    if any(set(item) != names for item in evidence[1:]):
        raise ValueError("V8 evidence replicates expose different fields")
    aggregate: dict[str, np.ndarray] = {}
    for name in sorted(names):
        values = [np.asarray(item[name]) for item in evidence]
        if any(value.shape != values[0].shape for value in values[1:]):
            raise ValueError(f"V8 evidence replicate shape differs for {name!r}")
        if name == "signed_delay_grid_steps":
            if any(not np.array_equal(value, values[0]) for value in values[1:]):
                raise ValueError("V8 evidence replicates use different delay grids")
            aggregate[name] = values[0].copy()
        elif name.endswith("_accepted"):
            aggregate[name] = (
                np.mean(np.stack(values).astype(np.float64), axis=0)
                >= float(consensus_frequency)
            ).astype(np.uint8)
        else:
            aggregate[name] = np.mean(
                np.stack(values).astype(np.float64), axis=0
            ).astype(np.float32)
    probability = np.asarray(aggregate["positive_delay_probability"], dtype=np.float64)
    normalizer = probability.sum(axis=-1, keepdims=True)
    aggregate["positive_delay_probability"] = (
        probability / np.maximum(normalizer, 1e-12)
    ).astype(np.float32)
    return aggregate


def fit_v8_fold_delay_prior_ensemble(
    rates: V8CachedRates,
    *,
    seeds: Sequence[int],
    evidence_rates: V8CachedRates | None = None,
    analytic_representation: str = "v8_exact_online_physical_fast_after_fold_train_fixed_gain",
    band_edges_hz: Sequence[Sequence[float]],
    task_seconds: float,
    maximum_routes: int,
    maximum_delay: int,
    route_scope: V8RouteScope = "within_band",
    bootstrap_samples: int,
    grid_oversample: int,
    minimum_bayes_factor: float,
    minimum_bootstrap_frequency: float,
    minimum_direction_probability: float,
    minimum_pass_fraction: float = 0.80,
    consensus_frequency: float = 0.80,
    minimum_consensus_edges: int = 3,
    minimum_median_edge_jaccard: float = 0.50,
    minimum_edge_jaccard_floor: float = 0.25,
    minimum_median_delay_correlation: float = 0.50,
    split_strata: Sequence[str] | None = None,
    surrogate_replicates: int = 1,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, np.ndarray]]:
    """Bag five fold-local audits and sparsify only their consensus routes."""

    fixed_seeds = [int(value) for value in seeds]
    if len(fixed_seeds) != 5 or len(set(fixed_seeds)) != 5:
        raise ValueError("V8 delay-prior ensemble requires exactly five unique seeds")
    audit_rates = rates if evidence_rates is None else evidence_rates
    if audit_rates.fast.shape[1:] != rates.fast.shape[1:]:
        raise ValueError("V8 evidence and transport rates must share band/node/time axes")
    edges = [[float(low), float(high)] for low, high in band_edges_hz]
    if len(edges) != audit_rates.fast.shape[1]:
        raise ValueError("band_edges_hz does not match the V8 rate cache")
    replicate_arrays: list[dict[str, np.ndarray]] = []
    replicate_summaries: list[dict[str, Any]] = []
    for seed in fixed_seeds:
        arrays, summary = fold_local_continuous_delay_prior(
            {"X": np.empty((audit_rates.fast.shape[0], 1, 1), dtype=np.float32)},
            edges,
            edges,
            n_nodes=audit_rates.fast.shape[2],
            graph_steps=audit_rates.fast.shape[-1],
            max_delay_steps=int(maximum_delay),
            bootstrap_samples=int(bootstrap_samples),
            grid_oversample=int(grid_oversample),
            min_bayes_factor=float(minimum_bayes_factor),
            min_bootstrap_frequency=float(minimum_bootstrap_frequency),
            min_direction_probability=float(minimum_direction_probability),
            seed=seed,
            device="cpu",
            task_tmin=0.0,
            task_tmax=float(task_seconds),
            analytic_features=audit_rates.fast.detach().cpu().numpy(),
            analytic_representation=str(analytic_representation),
            split_strata=split_strata,
            surrogate_replicates=int(surrogate_replicates),
        )
        replicate_arrays.append(arrays)
        replicate_summaries.append(summary)
    stability, pairwise = evaluate_v8_evidence_seed_stability(
        replicate_summaries,
        replicate_arrays,
        minimum_replicates=5,
        minimum_pass_fraction=float(minimum_pass_fraction),
        minimum_consensus_frequency=float(consensus_frequency),
        minimum_consensus_edges=int(minimum_consensus_edges),
        minimum_median_edge_jaccard=float(minimum_median_edge_jaccard),
        minimum_edge_jaccard_floor=float(minimum_edge_jaccard_floor),
        minimum_median_delay_correlation=float(minimum_median_delay_correlation),
    )
    aggregate = _aggregate_v8_evidence_replicates(
        replicate_arrays,
        consensus_frequency=float(consensus_frequency),
    )
    archive = dict(aggregate)
    for index, arrays in enumerate(replicate_arrays):
        for name, value in arrays.items():
            archive[f"replicate_{index}__{name}"] = np.asarray(value)
    summary = {
        "schema": "dpc-snn-v8-fold-delay-prior-ensemble/v1",
        "evidence_pipeline_passed": bool(stability["passed"]),
        "analytic_representation": str(analytic_representation),
        "ensemble_seeds": fixed_seeds,
        "ensemble_replicates": len(fixed_seeds),
        "consensus_frequency": float(consensus_frequency),
        "replicate_summaries": replicate_summaries,
        "stability_gate": stability,
        "pairwise_stability": pairwise,
        "classifier_training": False,
        "heldout_session_e_accessed": False,
    }
    if not bool(stability["passed"]):
        raise V8DelayEvidenceRejected(
            "fold-local delay evidence failed its across-seed stability gate",
            arrays=archive,
            summary=summary,
        )
    prior, enriched = sparse_v8_prior_from_evidence(
        rates,
        archive,
        summary,
        maximum_routes=int(maximum_routes),
        maximum_delay=int(maximum_delay),
        route_scope=route_scope,
    )
    enriched = {
        **enriched,
        "evidence_frontend_fingerprint": audit_rates.physical_frontend_fingerprint,
        "transport_frontend_fingerprint": rates.physical_frontend_fingerprint,
        "evidence_transport_axes_matched": True,
    }
    return prior, enriched, archive
