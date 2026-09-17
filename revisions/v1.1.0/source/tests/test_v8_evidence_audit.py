from __future__ import annotations

import numpy as np

from dpc_snn.experiments import v8_evidence_audit
from dpc_snn.experiments.v8_evidence_audit import (
    prepare_v8_evidence_space,
    pool_v8_anatomical_regions,
    select_physical_evidence_space,
    within_band_route_count,
)


def _signals(trials: int = 8, channels: int = 3, samples: int = 101) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.normal(size=(trials, channels, samples)).astype(np.float32)


def test_innovations_adjust_timing_and_are_not_physical_nodes() -> None:
    transformed, epoch_tmin, metadata = prepare_v8_evidence_space(
        _signals(),
        "car_innovations",
        channel_names=("Fz", "C3", "C4"),
        sfreq=20.0,
        epoch_tmin=-1.0,
        task_tmin=0.0,
        innovation_order=4,
    )
    assert transformed.shape == (8, 3, 97)
    assert epoch_tmin == -0.8
    assert metadata["eligible_for_physical_delay_routes"] is False
    assert metadata["innovation_order"] == 4


def test_unwhitened_innovations_keep_physical_target_nodes() -> None:
    transformed, epoch_tmin, metadata = prepare_v8_evidence_space(
        _signals(),
        "car_innovations_unwhitened",
        channel_names=("Fz", "C3", "C4"),
        sfreq=20.0,
        epoch_tmin=-1.0,
        task_tmin=0.0,
        innovation_order=4,
    )
    assert transformed.shape == (8, 3, 97)
    assert epoch_tmin == -0.8
    assert metadata["eligible_for_physical_delay_routes"] is True
    assert metadata["innovation_whitened"] is False


def test_fixed_anatomical_regions_cover_each_bci2a_electrode_once() -> None:
    names = (
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
        "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
    )
    fast = np.arange(2 * 3 * 22 * 5).reshape(2, 3, 22, 5)
    rates = v8_evidence_audit.V8CachedRates(
        fast=v8_evidence_audit.torch.as_tensor(fast, dtype=v8_evidence_audit.torch.complex64),
        slow=v8_evidence_audit.torch.as_tensor(fast[..., :2], dtype=v8_evidence_audit.torch.float32),
        gain=v8_evidence_audit.torch.ones(3, 22),
        physical_frontend_fingerprint="test",
    )
    pooled, metadata = pool_v8_anatomical_regions(rates, names)
    assert pooled.fast.shape == (2, 3, 9, 5)
    assert pooled.slow.shape == (2, 3, 9, 2)
    flattened = [name for members in metadata["region_members"] for name in members]
    assert sorted(flattened) == sorted(names)


def test_regional_unwhitened_space_is_interpretable_but_not_sensor_level() -> None:
    _, _, metadata = prepare_v8_evidence_space(
        _signals(),
        "car_innovations_unwhitened_regions",
        channel_names=("Fz", "C3", "C4"),
        sfreq=20.0,
        epoch_tmin=-1.0,
        task_tmin=0.0,
        innovation_order=4,
    )
    assert metadata["eligible_for_physical_delay_routes"] is False
    assert metadata["eligible_for_region_delay_routes"] is True
    assert metadata["eligible_for_interpretable_delay_routes"] is True


def test_csd_keeps_physical_node_identity(monkeypatch: object) -> None:
    values = _signals()
    monkeypatch.setattr(
        v8_evidence_audit,
        "current_source_density",
        lambda x, channel_names, sfreq: np.asarray(x) * 2.0,
    )
    transformed, epoch_tmin, metadata = prepare_v8_evidence_space(
        values,
        "csd",
        channel_names=("Fz", "C3", "C4"),
        sfreq=20.0,
        epoch_tmin=-1.0,
        task_tmin=0.0,
    )
    assert transformed.shape == values.shape
    assert epoch_tmin == -1.0
    assert metadata["eligible_for_physical_delay_routes"] is True
    np.testing.assert_allclose(transformed.mean(axis=1), 0.0, atol=1e-6)


def test_within_band_route_count_excludes_self_and_cross_band() -> None:
    accepted = np.zeros((2, 2, 3, 3), dtype=np.uint8)
    accepted[0, 0, 1, 0] = 1
    accepted[0, 0, 0, 0] = 1
    accepted[1, 0, 2, 0] = 1
    assert within_band_route_count({"band_pair_accepted": accepted}) == 1


def test_selection_never_promotes_mixed_or_failed_spaces() -> None:
    rows = [
        {
            "evidence_space": "csd_innovations",
            "evidence_pipeline_passed": True,
            "eligible_for_physical_delay_routes": False,
            "accepted_within_band_routes": 12,
        },
        {
            "evidence_space": "csd",
            "evidence_pipeline_passed": False,
            "eligible_for_physical_delay_routes": True,
            "accepted_within_band_routes": 12,
        },
        {
            "evidence_space": "car",
            "evidence_pipeline_passed": True,
            "eligible_for_physical_delay_routes": True,
            "accepted_within_band_routes": 2,
        },
    ]
    assert select_physical_evidence_space(rows) == "car"


def test_selection_prefers_predeclared_unwhitened_physical_innovations() -> None:
    rows = [
        {
            "evidence_space": "car",
            "evidence_pipeline_passed": True,
            "eligible_for_physical_delay_routes": True,
            "accepted_within_band_routes": 2,
        },
        {
            "evidence_space": "csd_innovations_unwhitened",
            "evidence_pipeline_passed": True,
            "eligible_for_physical_delay_routes": True,
            "accepted_within_band_routes": 3,
        },
    ]
    assert select_physical_evidence_space(rows) == "csd_innovations_unwhitened"
