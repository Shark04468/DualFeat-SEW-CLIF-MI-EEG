"""Across-seed stability gates for fold-local V8 delay evidence."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np


def _correlation(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) < 3:
        return float("nan")
    x = np.asarray(first)[mask]
    y = np.asarray(second)[mask]
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def evaluate_v8_evidence_seed_stability(
    summaries: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, np.ndarray]],
    *,
    minimum_replicates: int = 5,
    minimum_pass_fraction: float = 0.80,
    minimum_consensus_frequency: float = 0.80,
    minimum_consensus_edges: int = 3,
    minimum_median_edge_jaccard: float = 0.50,
    minimum_edge_jaccard_floor: float = 0.25,
    minimum_median_delay_correlation: float = 0.50,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Require one evidence pipeline to be stable across fixed audit seeds."""

    if len(summaries) != len(evidence) or len(evidence) < int(minimum_replicates):
        raise ValueError("V8 evidence stability requires matched summaries and replicates")
    masks = [np.asarray(item["signed_node_accepted"], dtype=bool) for item in evidence]
    delays = [np.asarray(item["signed_node_delay_mean"], dtype=np.float64) for item in evidence]
    shape = masks[0].shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("V8 seed stability requires square node evidence")
    if any(mask.shape != shape for mask in masks) or any(delay.shape != shape for delay in delays):
        raise ValueError("V8 seed-stability evidence shapes differ")
    off_diagonal = ~np.eye(shape[0], dtype=bool)
    masks = [mask & off_diagonal for mask in masks]
    pair_rows: list[dict[str, Any]] = []
    for first, second in combinations(range(len(masks)), 2):
        intersection = masks[first] & masks[second]
        union = masks[first] | masks[second]
        pair_rows.append(
            {
                "first_replicate": first,
                "second_replicate": second,
                "first_edges": int(masks[first].sum()),
                "second_edges": int(masks[second].sum()),
                "common_edges": int(intersection.sum()),
                "union_edges": int(union.sum()),
                "edge_jaccard": float(intersection.sum() / max(1, union.sum())),
                "delay_correlation_on_union": _correlation(
                    delays[first], delays[second], union
                ),
                "delay_correlation_all_off_diagonal": _correlation(
                    delays[first], delays[second], off_diagonal
                ),
            }
        )
    jaccard = np.asarray([row["edge_jaccard"] for row in pair_rows], dtype=np.float64)
    delay_correlation = np.asarray(
        [row["delay_correlation_on_union"] for row in pair_rows], dtype=np.float64
    )
    finite_delay = delay_correlation[np.isfinite(delay_correlation)]
    edge_frequency = np.mean(np.stack(masks), axis=0)
    consensus = (edge_frequency >= float(minimum_consensus_frequency)) & off_diagonal
    control_names = sorted(
        set.intersection(
            *(set(dict(row.get("evidence_checks", {}))) for row in summaries)
        )
    )
    control_pass_fraction = {
        name: float(
            np.mean([bool(dict(row.get("evidence_checks", {})).get(name)) for row in summaries])
        )
        for name in control_names
    }
    pass_fraction = float(
        np.mean([bool(row.get("evidence_pipeline_passed")) for row in summaries])
    )
    criteria = {
        "replicate_pass_fraction": pass_fraction >= float(minimum_pass_fraction),
        "each_control_pass_fraction": bool(control_pass_fraction)
        and min(control_pass_fraction.values()) >= float(minimum_pass_fraction),
        "consensus_edge_count": int(consensus.sum()) >= int(minimum_consensus_edges),
        "median_pairwise_edge_jaccard": float(np.median(jaccard))
        >= float(minimum_median_edge_jaccard),
        "pairwise_edge_jaccard_floor": float(np.min(jaccard))
        >= float(minimum_edge_jaccard_floor),
        "median_pairwise_delay_correlation": bool(finite_delay.size)
        and float(np.median(finite_delay)) >= float(minimum_median_delay_correlation),
    }
    report = {
        "schema": "dpc-snn-v8-evidence-seed-stability/v1",
        "passed": bool(all(criteria.values())),
        "replicates": len(evidence),
        "replicate_pass_fraction": pass_fraction,
        "control_pass_fraction": control_pass_fraction,
        "consensus_frequency": float(minimum_consensus_frequency),
        "consensus_edge_count": int(consensus.sum()),
        "consensus_edge_indices": np.argwhere(consensus).tolist(),
        "median_pairwise_edge_jaccard": float(np.median(jaccard)),
        "minimum_pairwise_edge_jaccard": float(np.min(jaccard)),
        "median_pairwise_delay_correlation": float(np.median(finite_delay))
        if finite_delay.size
        else float("nan"),
        "criteria": criteria,
        "thresholds": {
            "minimum_replicates": int(minimum_replicates),
            "minimum_pass_fraction": float(minimum_pass_fraction),
            "minimum_consensus_frequency": float(minimum_consensus_frequency),
            "minimum_consensus_edges": int(minimum_consensus_edges),
            "minimum_median_edge_jaccard": float(minimum_median_edge_jaccard),
            "minimum_edge_jaccard_floor": float(minimum_edge_jaccard_floor),
            "minimum_median_delay_correlation": float(minimum_median_delay_correlation),
        },
        "heldout_session_e_accessed": False,
        "classifier_training": False,
    }
    return report, pair_rows
