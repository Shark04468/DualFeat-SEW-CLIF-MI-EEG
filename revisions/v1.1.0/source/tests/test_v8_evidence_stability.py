from __future__ import annotations

import numpy as np

from dpc_snn.experiments.v8_evidence_stability import (
    evaluate_v8_evidence_seed_stability,
)


def _replicate(edges: list[tuple[int, int]], offset: float = 0.0) -> dict[str, np.ndarray]:
    accepted = np.zeros((4, 4), dtype=np.uint8)
    delay = np.zeros((4, 4), dtype=np.float32)
    for index, (target, source) in enumerate(edges):
        accepted[target, source] = 1
        delay[target, source] = 0.5 + index + offset
    return {"signed_node_accepted": accepted, "signed_node_delay_mean": delay}


def _summary(passed: bool = True) -> dict[str, object]:
    return {
        "evidence_pipeline_passed": passed,
        "evidence_checks": {
            "bootstrap_frequency": passed,
            "natural_nonzero_edges": True,
            "phase_surrogate": True,
            "split_half_delay": passed,
            "time_reversal": True,
        },
    }


def test_seed_stability_accepts_four_of_five_with_stable_consensus() -> None:
    stable = [(1, 0), (2, 0), (3, 1), (3, 2)]
    evidence = [_replicate(stable, 0.01 * index) for index in range(4)]
    evidence.append(_replicate(stable[:3] + [(0, 3)], 0.04))
    report, pairs = evaluate_v8_evidence_seed_stability(
        [_summary(), _summary(), _summary(), _summary(), _summary(False)],
        evidence,
    )
    assert report["passed"] is True
    assert report["consensus_edge_count"] == 4
    assert len(pairs) == 10


def test_seed_stability_rejects_seed_specific_edge_sets() -> None:
    evidence = [
        _replicate([(1, 0), (2, 0), (3, 0)]),
        _replicate([(0, 1), (2, 1), (3, 1)]),
        _replicate([(0, 2), (1, 2), (3, 2)]),
        _replicate([(0, 3), (1, 3), (2, 3)]),
        _replicate([(1, 0), (2, 1), (3, 2)]),
    ]
    report, _ = evaluate_v8_evidence_seed_stability([_summary()] * 5, evidence)
    assert report["passed"] is False
    assert report["criteria"]["median_pairwise_edge_jaccard"] is False
