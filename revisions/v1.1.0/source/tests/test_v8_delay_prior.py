from __future__ import annotations

import numpy as np
import pytest
import torch

from dpc_snn.experiments import v8_delay_prior
from dpc_snn.experiments.v8_delay_prior import (
    V8DelayEvidenceRejected,
    fit_v8_fold_delay_prior,
    fit_v8_fold_delay_prior_ensemble,
    sparse_v8_prior_from_evidence,
    v8_delay_prior_seed,
)
from dpc_snn.experiments.v8_training import V8CachedRates


def _rates(phase_offset: float = 0.4) -> V8CachedRates:
    time = torch.arange(40, dtype=torch.float32)
    source = torch.exp(1j * 0.3 * time)
    target = torch.zeros_like(source)
    rotation = torch.exp(torch.tensor(1j * phase_offset, dtype=torch.complex64))
    target[1:] = source[:-1] * rotation
    fast = torch.ones(4, 2, 3, 40, dtype=torch.complex64)
    fast[:, 0, 0] = source
    fast[:, 0, 1] = target
    return V8CachedRates(
        fast=fast,
        slow=torch.zeros(4, 2, 3, 20),
        gain=torch.ones(2, 3),
        physical_frontend_fingerprint="test",
    )


def _evidence() -> dict[str, np.ndarray]:
    shape = (2, 2, 3, 3)
    route = np.zeros(shape, dtype=np.float32)
    accepted = np.zeros(shape, dtype=np.uint8)
    route[0, 0, 1, 0] = 0.9
    accepted[0, 0, 1, 0] = 1
    route[1, 0, 2, 0] = 0.99
    accepted[1, 0, 2, 0] = 1
    route[0, 0, 0, 0] = 1.0
    accepted[0, 0, 0, 0] = 1
    probability = np.zeros((*shape, 3), dtype=np.float32)
    probability[..., 0] = 1.0
    probability[0, 0, 1, 0] = np.asarray([0.0, 1.0, 0.0])
    fraction = np.zeros(shape, dtype=np.float32)
    confidence = np.full(shape, 0.8, dtype=np.float32)
    return {
        "route_probability": route,
        "positive_delay_probability": probability,
        "fractional_delay_target": fraction,
        "connectivity_prior": confidence,
        "band_pair_accepted": accepted,
    }


def _ensemble_evidence() -> dict[str, np.ndarray]:
    evidence = _evidence()
    accepted = np.zeros((3, 3), dtype=np.uint8)
    accepted[1, 0] = 1
    accepted[2, 0] = 1
    accepted[2, 1] = 1
    delay = np.zeros((3, 3), dtype=np.float32)
    delay[1, 0] = 0.5
    delay[2, 0] = 1.5
    delay[2, 1] = 2.5
    evidence.update(
        {
            "signed_node_accepted": accepted,
            "signed_node_delay_mean": delay,
            "signed_delay_grid_steps": np.asarray([-2.0, 0.0, 2.0]),
        }
    )
    return evidence


def _accepted_summary(passed: bool = True) -> dict[str, object]:
    return {
        "evidence_pipeline_passed": passed,
        "evidence_checks": {
            "bootstrap_frequency": True,
            "natural_nonzero_edges": True,
            "phase_surrogate": True,
            "split_half_delay": passed,
            "time_reversal": True,
        },
    }


def test_sparse_prior_keeps_only_nonself_within_band_routes_and_axes() -> None:
    prior, summary = sparse_v8_prior_from_evidence(
        _rates(),
        _evidence(),
        {"evidence_pipeline_passed": True},
        maximum_routes=4,
        maximum_delay=2,
    )
    assert prior["source_band"].tolist() == [0]
    assert prior["target_band"].tolist() == [0]
    assert prior["source_node"].tolist() == [0]
    assert prior["target_node"].tolist() == [1]
    assert prior["delay_probability"].tolist() == [[0.0, 1.0, 0.0]]
    assert summary["selected_sparse_routes"] == 1
    assert len(summary["prior_sha256"]) == 64


def test_delay_prior_seed_schedule_is_shared_and_replicate_distinct() -> None:
    assert v8_delay_prior_seed(1, 0, 0) == 8_310_010
    assert v8_delay_prior_seed(1, 0, 1) == 9_310_013
    assert v8_delay_prior_seed(3, 5, 0) != v8_delay_prior_seed(1, 0, 0)
    assert v8_delay_prior_seed(1, 0, 0, scope="outer") != v8_delay_prior_seed(
        1, 0, 0, scope="inner"
    )
    with pytest.raises(ValueError):
        v8_delay_prior_seed(1, 6, 0)


def test_sparse_prior_recovers_fold_train_phase_intercept() -> None:
    phase_offset = 0.4
    prior, _ = sparse_v8_prior_from_evidence(
        _rates(phase_offset),
        _evidence(),
        {"evidence_pipeline_passed": True},
        maximum_routes=1,
        maximum_delay=2,
    )
    assert float(prior["phase_preference"][0]) == pytest.approx(phase_offset, abs=1e-4)
    assert float(prior["amplitude_scale"][0]) > 0.0


def test_sparse_prior_cross_band_scope_selects_true_source_target_band_edges() -> None:
    prior, summary = sparse_v8_prior_from_evidence(
        _rates(),
        _evidence(),
        {"evidence_pipeline_passed": True},
        maximum_routes=4,
        maximum_delay=2,
        route_scope="cross_band",
    )
    assert prior["source_band"].tolist() == [0]
    assert prior["target_band"].tolist() == [1]
    assert prior["source_node"].tolist() == [0]
    assert prior["target_node"].tolist() == [2]
    assert summary["route_scope"] == "cross_band"
    assert summary["sparse_route_policy"] == (
        "accepted_cross_band_top_probability_ceiling"
    )


def test_sparse_prior_never_forces_failed_or_empty_evidence() -> None:
    with pytest.raises(V8DelayEvidenceRejected, match="negative-control gate") as failed:
        sparse_v8_prior_from_evidence(
            _rates(),
            _evidence(),
            {"evidence_pipeline_passed": False},
            maximum_routes=1,
            maximum_delay=2,
        )
    assert failed.value.summary["evidence_pipeline_passed"] is False
    assert set(failed.value.arrays) == set(_evidence())
    empty = _evidence()
    empty["band_pair_accepted"].fill(0)
    with pytest.raises(V8DelayEvidenceRejected, match="no accepted within-band route"):
        sparse_v8_prior_from_evidence(
            _rates(),
            empty,
            {"evidence_pipeline_passed": True},
            maximum_routes=1,
            maximum_delay=2,
        )


def test_fit_prior_preserves_rejected_fold_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    evidence = _evidence()
    summary = {
        "evidence_pipeline_passed": False,
        "time_reversal_passed": False,
        "phase_surrogate_passed": True,
    }

    def rejected_audit(*args: object, **kwargs: object) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        return evidence, summary

    monkeypatch.setattr(
        v8_delay_prior,
        "fold_local_continuous_delay_prior",
        rejected_audit,
    )
    with pytest.raises(V8DelayEvidenceRejected, match="negative-control gate") as rejected:
        fit_v8_fold_delay_prior(
            _rates(),
            band_edges_hz=((6.0, 8.0), (8.0, 10.0)),
            task_seconds=4.0,
            maximum_routes=4,
            maximum_delay=2,
            bootstrap_samples=4,
            grid_oversample=2,
            minimum_bayes_factor=3.0,
            minimum_bootstrap_frequency=0.7,
            minimum_direction_probability=0.8,
            seed=0,
        )
    assert rejected.value.summary == summary
    assert np.array_equal(
        rejected.value.arrays["route_probability"],
        evidence["route_probability"],
    )


def test_fit_prior_uses_separate_evidence_rates_but_transport_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _rates()
    evidence_rates = V8CachedRates(
        fast=transport.fast * (2.0 + 0.0j),
        slow=transport.slow,
        gain=transport.gain,
        physical_frontend_fingerprint="evidence",
    )
    captured: dict[str, np.ndarray] = {}

    def accepted_audit(*args: object, **kwargs: object) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        captured["analytic"] = np.asarray(kwargs["analytic_features"])
        return _evidence(), {"evidence_pipeline_passed": True}

    monkeypatch.setattr(v8_delay_prior, "fold_local_continuous_delay_prior", accepted_audit)
    _, summary, _ = fit_v8_fold_delay_prior(
        transport,
        evidence_rates=evidence_rates,
        analytic_representation="separate_test_evidence",
        band_edges_hz=((6.0, 8.0), (8.0, 10.0)),
        task_seconds=4.0,
        maximum_routes=4,
        maximum_delay=2,
        bootstrap_samples=4,
        grid_oversample=2,
        minimum_bayes_factor=3.0,
        minimum_bootstrap_frequency=0.7,
        minimum_direction_probability=0.8,
        seed=0,
    )
    np.testing.assert_allclose(captured["analytic"], evidence_rates.fast.numpy())
    assert summary["evidence_frontend_fingerprint"] == "evidence"
    assert summary["transport_frontend_fingerprint"] == "test"


def test_fit_prior_ensemble_uses_all_five_replicates_without_seed_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def accepted_audit(
        *args: object, **kwargs: object
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        calls.append(int(kwargs["seed"]))
        return _ensemble_evidence(), _accepted_summary()

    monkeypatch.setattr(v8_delay_prior, "fold_local_continuous_delay_prior", accepted_audit)
    seeds = [10, 20, 30, 40, 50]
    prior, summary, archive = fit_v8_fold_delay_prior_ensemble(
        _rates(),
        seeds=seeds,
        band_edges_hz=((6.0, 8.0), (8.0, 10.0)),
        task_seconds=4.0,
        maximum_routes=4,
        maximum_delay=2,
        bootstrap_samples=4,
        grid_oversample=2,
        minimum_bayes_factor=3.0,
        minimum_bootstrap_frequency=0.7,
        minimum_direction_probability=0.8,
    )
    assert calls == seeds
    assert summary["stability_gate"]["passed"] is True
    assert summary["ensemble_seeds"] == seeds
    assert prior["target_node"].tolist() == [1]
    assert "replicate_4__route_probability" in archive


def test_fit_prior_ensemble_persists_all_replicates_when_gate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rejected_audit(
        *args: object, **kwargs: object
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        return _ensemble_evidence(), _accepted_summary(False)

    monkeypatch.setattr(v8_delay_prior, "fold_local_continuous_delay_prior", rejected_audit)
    with pytest.raises(V8DelayEvidenceRejected, match="across-seed") as rejected:
        fit_v8_fold_delay_prior_ensemble(
            _rates(),
            seeds=[10, 20, 30, 40, 50],
            band_edges_hz=((6.0, 8.0), (8.0, 10.0)),
            task_seconds=4.0,
            maximum_routes=4,
            maximum_delay=2,
            bootstrap_samples=4,
            grid_oversample=2,
            minimum_bayes_factor=3.0,
            minimum_bootstrap_frequency=0.7,
            minimum_direction_probability=0.8,
        )
    assert rejected.value.summary["stability_gate"]["passed"] is False
    assert "replicate_4__route_probability" in rejected.value.arrays
