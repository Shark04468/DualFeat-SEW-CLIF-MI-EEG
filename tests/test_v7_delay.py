from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_scaffold import build_zero_scaffold  # noqa: E402
from dpc_snn.experiments.v7_delay import (  # noqa: E402
    _configure_atc_training_scope,
    configure_delay_stage,
    delay_diagnostics,
    paired_delay_gate,
)


def _config() -> dict:
    return {
        "n_bands": 2,
        "n_nodes": 3,
        "band_edges_hz": ((8.0, 12.0), (18.0, 24.0)),
        "analytic_taps": 33,
        "envelope_taps": 9,
        "spatial_rank": 2,
        "route_rank": 2,
        "delay_rank": 2,
        "temporal_dilations": (1, 2),
        "temporal_depth": 1,
        "temporal_scale_attention": False,
        "use_geometry": False,
        "snn_channels": 8,
        "decoder_layers": 0,
        "statistical_spatial_filters": 3,
        "dropout": 0.0,
    }


def test_paired_delay_gate_requires_all_pairs_and_registered_thresholds() -> None:
    config = {
        "gate": {
            "total_subject_seed_pairs": 9,
            "median_delta_pp": 0.5,
            "minimum_positive_pairs": 6,
        },
        "control": {"amplitude_rms_tolerance_fraction": 0.05},
    }
    rows = [
        {
            "delta_accuracy": delta,
            "amplitude_relative_difference": 0.001,
        }
        for delta in np.asarray([0.01] * 6 + [0.0] * 3)
    ]

    gate = paired_delay_gate(rows, config)

    assert gate["passed"] is True
    assert gate["positive_pairs"] == 6
    assert gate["median_delta_pp"] == pytest.approx(1.0)
    assert paired_delay_gate(rows[:-1], config)["passed"] is False


def test_static_slow_stage_keeps_fast_backbone_and_trains_tau() -> None:
    model = build_zero_scaffold(seed=1, model_config=_config())
    configure_delay_stage(model, "static_slow_within")
    diagnostics = delay_diagnostics(model)
    assert model.delay.slow_residual_route_scale == 1.0
    assert model.delay.fast_residual_route_scale == 0.0
    assert model.delay.slow_carrier_residual_scale == 0.0
    assert model.delay.target_interaction_bound == 0.50
    assert diagnostics["tau_trainable"] is True
    assert diagnostics["phase_trainable"] is False
    assert diagnostics["tau_theta_simultaneously_trainable"] is False
    assert diagnostics["slow_active_delay_mean_samples"] == pytest.approx(4.0, abs=0.2)
    assert diagnostics["slow_active_delay_std_samples"] == pytest.approx(0.75, abs=1e-4)
    parameters = model.delay._route_parameters()
    diagonal = torch.arange(model.n_bands)
    eta = parameters["eta"][diagonal, diagonal]
    torch.testing.assert_close(eta, torch.full_like(eta, 0.10), atol=1e-6, rtol=0)
    eta.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.delay.target_interaction.parameters()
    )


def test_static_slow_stage_full_forward_does_not_enable_fast_delay() -> None:
    model = build_zero_scaffold(seed=11, model_config=_config()).eval()
    configure_delay_stage(model, "static_slow_within", train_readout=False)
    fast = torch.complex(
        torch.randn(1, 2, 3, 500), torch.randn(1, 2, 3, 500)
    )
    slow = torch.randn(1, 2, 3, 250)
    broadband = torch.randn(1, 3, 500)

    with torch.no_grad():
        full = model.forward_rate_features(fast, slow, broadband=broadband)
        locked_zero = model.forward_rate_features(
            fast,
            slow,
            broadband=broadband,
            delay_override="zero",
        )

    full_transport = full["aux"]["transport"]
    zero_transport = locked_zero["aux"]["transport"]
    assert full_transport.slow_delay_override == "learned"
    assert full_transport.fast_delay_override == "zero"
    assert full_transport.broadband_delay_source == "slow_route_residual"
    assert zero_transport.slow_delay_override == "zero"
    assert zero_transport.fast_delay_override == "zero"
    assert zero_transport.broadband_delay_source == "slow_route_residual"
    assert full_transport.broadband_current is not None
    assert zero_transport.broadband_current is not None
    assert full_transport.broadband_current.shape == broadband.shape
    assert zero_transport.broadband_current.shape == broadband.shape
    torch.testing.assert_close(zero_transport.broadband_current, broadband)
    torch.testing.assert_close(
        full_transport.broadband_current,
        zero_transport.broadband_current,
    )
    assert model.delay.slow_carrier_residual_scale == 0.0
    assert torch.count_nonzero(full_transport.slow_delay_contrast_current) > 0


def test_phase_stage_freezes_tau_before_theta() -> None:
    model = build_zero_scaffold(seed=2, model_config=_config())
    configure_delay_stage(model, "phase_residual")
    diagnostics = delay_diagnostics(model)
    assert diagnostics["phase_trainable"] is True
    assert diagnostics["tau_trainable"] is False
    assert diagnostics["tau_theta_simultaneously_trainable"] is False
    assert model.delay.phase_bound == 0.15


def test_delay_stage_bounded_initialization_overrides_are_explicit() -> None:
    model = build_zero_scaffold(seed=12, model_config=_config())
    configure_delay_stage(
        model,
        "static_slow_within",
        train_readout=False,
        initial_slow_delay_fraction=0.25,
        slow_route_scale=1.0,
        slow_carrier_residual_scale=0.20,
    )
    parameters = model.delay._route_parameters()
    diagonal = torch.arange(model.n_bands)
    diagonal_delay = parameters["slow_delay"][diagonal, diagonal]
    assert float(diagonal_delay.detach().mean()) == pytest.approx(
        0.25 * model.delay.slow_max_delay,
        abs=0.2,
    )
    assert model.delay.slow_residual_route_scale == 1.0
    assert model.delay.slow_carrier_residual_scale == 0.20


def test_dynamic_stage_requires_eeg_context_topology() -> None:
    model = build_zero_scaffold(seed=3, model_config=_config())
    try:
        configure_delay_stage(model, "dynamic_residual")
    except ValueError as error:
        assert "context encoder" in str(error)
    else:
        raise AssertionError("dynamic delay was enabled without an EEG context encoder")


def test_delay_only_stage_freezes_readout_and_batchnorm_state() -> None:
    model = build_zero_scaffold(seed=4, model_config=_config())
    configure_delay_stage(model, "static_slow_within", train_readout=False)
    assert getattr(model, "_frozen_eval_modules")
    assert all(
        not parameter.requires_grad
        for parameter in model.decoder.parameters()
    )


def test_fixed_delay_head_scope_freezes_transport_parameters() -> None:
    class Readout(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = torch.nn.Linear(4, 4)
            self.delayed_statistics_classifier = torch.nn.Linear(5, 4)
            self.delayed_route_statistics_classifier = torch.nn.Linear(6, 4)
            self.delayed_residual_mix_raw = torch.nn.Parameter(torch.zeros(()))
            self.delayed_residual_bound = 0.25

    model = build_zero_scaffold(seed=40, model_config=_config())
    model.atc_readout = Readout()
    configure_delay_stage(
        model,
        "static_slow_within",
        readout_training_scope="fixed_delay_head",
    )

    assert all(not parameter.requires_grad for parameter in model.delay.parameters())
    assert all(
        parameter.requires_grad
        for name, parameter in model.atc_readout.delayed_statistics_classifier.named_parameters()
        if name != "bias"
    )
    assert all(
        parameter.requires_grad
        for name, parameter in model.atc_readout.delayed_route_statistics_classifier.named_parameters()
        if name != "bias"
    )
    assert not model.atc_readout.delayed_statistics_classifier.bias.requires_grad
    assert not model.atc_readout.delayed_route_statistics_classifier.bias.requires_grad
    assert not model.atc_readout.delayed_residual_mix_raw.requires_grad


def test_fixed_full_scope_trains_atc_but_freezes_audited_transport() -> None:
    class Readout(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.BatchNorm1d(4),
            )
            self.delayed_statistics_classifier = torch.nn.Linear(5, 4)
            self.delayed_residual_mix_raw = torch.nn.Parameter(torch.zeros(()))
            self.delayed_residual_bound = 0.25

    model = build_zero_scaffold(seed=41, model_config=_config())
    model.atc_readout = Readout()
    configure_delay_stage(
        model,
        "static_slow_within",
        readout_training_scope="fixed_full",
    )

    assert all(not parameter.requires_grad for parameter in model.delay.parameters())
    assert all(parameter.requires_grad for parameter in model.atc_readout.core.parameters())
    assert model.atc_readout.delayed_residual_mix_raw.requires_grad
    assert getattr(model, "_frozen_eval_modules") == ()


def test_fixed_head_scope_trains_only_registered_and_delay_heads() -> None:
    class Window(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature = torch.nn.Linear(5, 5)
            self.linear = torch.nn.Linear(5, 4)

    class Official(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.atc_blocks = torch.nn.ModuleList([Window(), Window()])

    class Adapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.module = Official()

    class Readout(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = Adapter()
            self.delayed_statistics_classifier = torch.nn.Linear(5, 4)
            self.delayed_residual_mix_raw = torch.nn.Parameter(torch.zeros(()))
            self.delayed_residual_bound = 0.25

    model = build_zero_scaffold(seed=42, model_config=_config())
    model.atc_readout = Readout()
    configure_delay_stage(
        model,
        "static_slow_within",
        readout_training_scope="fixed_head",
    )

    assert all(not parameter.requires_grad for parameter in model.delay.parameters())
    assert getattr(model, "_frozen_eval_modules") == (model.atc_readout.core,)
    for block in model.atc_readout.core.module.atc_blocks:
        assert all(not parameter.requires_grad for parameter in block.feature.parameters())
        assert all(parameter.requires_grad for parameter in block.linear.parameters())
    assert all(
        parameter.requires_grad
        for name, parameter in model.atc_readout.delayed_statistics_classifier.named_parameters()
        if name != "bias"
    )
    assert not model.atc_readout.delayed_statistics_classifier.bias.requires_grad
    assert model.atc_readout.delayed_residual_mix_raw.requires_grad


def test_fixed_head_scope_does_not_require_a_delayed_statistics_head() -> None:
    class Window(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature = torch.nn.Linear(5, 5)
            self.linear = torch.nn.Linear(5, 4)

    class Official(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.atc_blocks = torch.nn.ModuleList([Window(), Window()])

    class Adapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.module = Official()

    class Readout(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = Adapter()
            self.delayed_statistics_classifier = None
            self.delayed_route_statistics_classifier = None
            self.delayed_residual_mix_raw = torch.nn.Parameter(torch.zeros(()))
            self.delayed_residual_bound = 0.25

    model = build_zero_scaffold(seed=43, model_config=_config())
    model.atc_readout = Readout()
    configure_delay_stage(
        model,
        "static_slow_within",
        readout_training_scope="fixed_head",
    )

    assert all(not parameter.requires_grad for parameter in model.delay.parameters())
    for block in model.atc_readout.core.module.atc_blocks:
        assert all(not parameter.requires_grad for parameter in block.feature.parameters())
        assert all(parameter.requires_grad for parameter in block.linear.parameters())
    assert model.atc_readout.delayed_residual_mix_raw.requires_grad


def test_head_only_scope_unfreezes_only_registered_window_classifiers() -> None:
    class Window(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature = torch.nn.Linear(5, 5)
            self.linear = torch.nn.Linear(5, 4)

    class Official(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.atc_blocks = torch.nn.ModuleList([Window(), Window()])

    class Adapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.module = Official()

    core = Adapter()
    frozen_eval = _configure_atc_training_scope(core, "head")
    assert frozen_eval == (core,)
    for block in core.module.atc_blocks:
        assert all(not parameter.requires_grad for parameter in block.feature.parameters())
        assert all(parameter.requires_grad for parameter in block.linear.parameters())
