from __future__ import annotations

import numpy as np
import torch

from dpc_snn.experiments.v8_delay_prior import v8_delay_prior_digest
from dpc_snn.experiments.v8_training import V8CachedRates
from scripts import run_v8_e3_static_delay


def _rates() -> V8CachedRates:
    return V8CachedRates(
        fast=torch.ones(4, 2, 3, 20, dtype=torch.complex64),
        slow=torch.ones(4, 2, 3, 10),
        gain=torch.ones(2, 3),
        physical_frontend_fingerprint="transport",
    )


def _prior() -> dict[str, torch.Tensor]:
    return {
        "source_band": torch.tensor([0]),
        "source_node": torch.tensor([0]),
        "target_band": torch.tensor([0]),
        "target_node": torch.tensor([1]),
        "delay_probability": torch.tensor([[0.0, 1.0, 0.0]]),
        "fractional_target": torch.tensor([0.25]),
        "route_weight": torch.tensor([0.9]),
        "route_confidence": torch.tensor([0.8]),
        "phase_preference": torch.tensor([0.1]),
        "amplitude_scale": torch.tensor([1.0]),
    }


def test_load_or_fit_prior_persists_and_reloads_ensemble(
    tmp_path, monkeypatch
) -> None:
    calls: list[list[int]] = []

    def fitted(*args, **kwargs):
        calls.append([int(value) for value in kwargs["seeds"]])
        prior = _prior()
        summary = {
            "evidence_pipeline_passed": True,
            "prior_sha256": v8_delay_prior_digest(prior),
            "stability_gate": {"passed": True},
        }
        evidence = {
            "route_probability": np.asarray([0.9], dtype=np.float32),
            "replicate_0__route_probability": np.asarray([0.8], dtype=np.float32),
        }
        return prior, summary, evidence

    monkeypatch.setattr(
        run_v8_e3_static_delay,
        "fit_v8_fold_delay_prior_ensemble",
        fitted,
    )
    seeds = [10, 20, 30, 40, 50]
    kwargs = {
        "scope": "inner_training_fold_only",
        "trial_ids": ["trial-0", "trial-1", "trial-2", "trial-3"],
        "rates": _rates(),
        "gain": torch.ones(2, 3),
        "model_config": {
            "band_edges_hz": ((6.0, 8.0), (8.0, 10.0)),
            "task_tmin": 0.0,
            "task_tmax": 4.0,
        },
        "delay_config": {
            "maximum_routes": 4,
            "maximum_delay_samples": 2,
            "route_scope": "within_band",
            "prior": {
                "bootstrap_samples": 4,
                "grid_oversample": 2,
                "minimum_bayes_factor": 3.0,
                "minimum_bootstrap_frequency": 0.7,
                "minimum_direction_probability": 0.8,
            },
        },
        "source_sha256": "a" * 64,
        "seed": seeds[0],
        "ensemble_seeds": seeds,
        "split_strata": ["run=0|class=0", "run=0|class=0", "run=0|class=1", "run=0|class=1"],
    }
    first, summary = run_v8_e3_static_delay._load_or_fit_prior(tmp_path, **kwargs)
    assert calls == [seeds]
    assert summary["input_payload"]["ensemble_seeds"] == seeds
    assert summary["input_payload"]["split_half_scheme"] == (
        "run_class_stratified_deterministic_alternation"
    )
    assert torch.equal(first["delay_probability"], _prior()["delay_probability"])
    assert (tmp_path / "manifest.json").is_file()

    def should_not_refit(*args, **kwargs):
        raise AssertionError("valid ensemble prior should load from its manifest")

    monkeypatch.setattr(
        run_v8_e3_static_delay,
        "fit_v8_fold_delay_prior_ensemble",
        should_not_refit,
    )
    second, _ = run_v8_e3_static_delay._load_or_fit_prior(tmp_path, **kwargs)
    assert torch.equal(second["route_weight"], first["route_weight"])
