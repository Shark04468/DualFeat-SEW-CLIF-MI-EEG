from __future__ import annotations

import numpy as np
import pytest
import torch

from dpc_snn.experiments.reviewer_controls import (
    ExactGradientSoftCLIFLayer,
    bci2a_lh_rh_subset,
    benchmark_latency,
    build_exact_gradient_soft_clif,
    build_reviewer_model,
    dense_mac_proxy,
    equal_update_epoch_indices,
    event_accumulation_proxy,
    fit_reviewer_model,
    make_equal_update_plan,
    solve_temporal_width,
    summarize_binary_spikes,
    temporal_statistics,
)


def _arrays(n: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5)
    return (
        rng.normal(size=(n, 18, 32)).astype(np.float32),
        rng.normal(size=(n, 4, 288)).astype(np.float32),
        (np.arange(n) % 2).astype(np.int64),
    )


def test_exact_gradient_gate_uses_analytic_sigmoid_derivative() -> None:
    x = torch.tensor([[-0.2, 0.0, 0.3]], dtype=torch.float64, requires_grad=True)
    slope = 10.0
    gate = torch.sigmoid(slope * x)
    gate.sum().backward()
    expected = slope * gate.detach() * (1.0 - gate.detach())
    legacy = 1.0 / (slope * x.detach().abs() + 1.0).square()
    torch.testing.assert_close(x.grad, expected)
    assert not torch.allclose(x.grad, legacy)


def test_exact_gradient_model_replaces_all_four_state_sites() -> None:
    model = build_exact_gradient_soft_clif(n_classes=2)
    sites = [module for module in model.modules() if isinstance(module, ExactGradientSoftCLIFLayer)]
    assert len(sites) == 4
    atc, fbc, _ = _arrays(2)
    output = model(torch.from_numpy(atc), torch.from_numpy(fbc))
    assert output.logits.shape == (2, 2)
    assert len(output.binary_spikes) == 4
    assert any(
        not bool(torch.logical_or(value == 0, value == 1).all()) for value in output.binary_spikes
    )


def test_population_statistics_are_mean_and_unbiased_false_std() -> None:
    activity = torch.tensor([[[1.0, 3.0], [2.0, 6.0]]])
    state = activity + 1.0
    result = temporal_statistics(activity, state)
    expected = torch.cat(
        (
            activity.mean(-1),
            activity.std(-1, unbiased=False),
            state.mean(-1),
            state.std(-1, unbiased=False),
        ),
        dim=1,
    )
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("kind", ["gru_stat", "tcn_stat"])
def test_stat_control_width_solver_and_forward(kind: str) -> None:
    target = build_reviewer_model("sew_clif", n_classes=2).trainable_parameter_count
    selected = solve_temporal_width(kind, n_classes=2, target_parameters=target)
    assert selected.relative_error < 0.05
    model = build_reviewer_model(kind, n_classes=2, temporal_width=selected.temporal_width)
    atc, fbc, _ = _arrays(3)
    output = model(torch.from_numpy(atc), torch.from_numpy(fbc))
    assert output.logits.shape == (3, 2)
    assert output.final_activity.shape == (3, 64, 18)
    assert output.final_state.shape == (3, 64, 18)


def test_equal_update_plan_matches_full_steps_and_exposure() -> None:
    plan = make_equal_update_plan(full_samples=121, subset_samples=50, batch_size=48, epochs=80)
    assert plan.batch_sizes_per_epoch == (48, 48, 25)
    assert plan.optimizer_steps == 240
    assert plan.sample_exposures == 9680
    one = equal_update_epoch_indices(plan, seed=3, epoch=1)
    two = equal_update_epoch_indices(plan, seed=3, epoch=1)
    assert [len(value) for value in one] == [48, 48, 25]
    assert all(np.array_equal(a, b) for a, b in zip(one, two, strict=True))
    assert all(np.all((value >= 0) & (value < 50)) for value in one)


def test_fit_equal_update_writes_exact_ledger() -> None:
    atc, fbc, labels = _arrays(6)
    fit = fit_reviewer_model(
        "ann_sew",
        atc_train=atc,
        fbc_train=fbc,
        y_train=labels,
        n_classes=2,
        device="cpu",
        seed=7,
        fixed_epochs=2,
        batch_size=4,
        equal_update_full_samples=10,
    )
    assert fit.optimizer_steps == 6
    assert fit.sample_exposures == 20
    assert fit.history[-1]["optimizer_steps_cumulative"] == 6


def test_binary_spike_audit_and_bci2a_subset() -> None:
    summary = summarize_binary_spikes(
        [
            (torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0])),
            (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0])),
        ]
    )
    assert summary["binary"] is True
    assert summary["events"] == 4
    assert summary["elements"] == 8
    assert summary["layer_events"] == [2, 2]
    assert summary["layer_elements"] == [4, 4]
    proxy = event_accumulation_proxy(summary, eligible_fanouts=(3, 2))
    assert proxy["eligible_event_accumulations"] == 10
    assert proxy["energy_claim_supported"] is False
    with pytest.raises(ValueError):
        summarize_binary_spikes([(torch.tensor([0.2]),)])
    indices, labels = bci2a_lh_rh_subset(np.array([0, 2, 1, 3, 0, 1]))
    np.testing.assert_array_equal(indices, np.array([0, 2, 4, 5]))
    np.testing.assert_array_equal(labels, np.array([0, 1, 0, 1]))


def test_operation_and_latency_proxies_are_explicit_and_finite() -> None:
    model = build_reviewer_model("ann_sew", n_classes=2)
    atc, fbc, _ = _arrays(1)
    tensor_atc = torch.from_numpy(atc)
    tensor_fbc = torch.from_numpy(fbc)
    operations = dense_mac_proxy(model, tensor_atc, tensor_fbc)
    assert operations["macs_per_trial"] > 0
    assert operations["energy_claim_supported"] is False
    timing = benchmark_latency(model, tensor_atc, tensor_fbc, warmup=1, repetitions=2)
    assert len(timing["samples_ms"]) == 2
    assert timing["mean_batch_ms"] > 0
