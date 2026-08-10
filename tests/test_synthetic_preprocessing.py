from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.data.synthetic import SyntheticDelayPhaseConfig, generate_delay_phase_dataset, time_reverse
from dpc_snn.experiments.runners import _synthetic_delay_model_cfg
from dpc_snn.models.delay_phase_synapse import DelayPhaseGraphSynapse
from dpc_snn.preprocessing.hilbert import band_amplitude_phase


def test_synthetic_shapes_and_features():
    cfg = SyntheticDelayPhaseConfig(n_trials=16, n_channels=6, n_time=64, max_delay=4, seed=1)
    data = generate_delay_phase_dataset(cfg)
    assert data["X"].shape == (16, 6, 64)
    assert data["delay_gt"].shape == (6, 6)
    assert data["edge_gt"].shape == (6, 6)
    assert np.count_nonzero(data["edge_gt"]) > 0
    amp, phase, bands = band_amplitude_phase(data["X"], 250.0, {"mu": [8, 13], "beta": [13, 30]})
    assert amp.shape == (16, 2, 6, 64)
    assert phase.shape == (16, 2, 6, 64)
    assert bands == ["mu", "beta"]


def test_synthetic_delay_graph_has_no_unidentifiable_reciprocal_edges():
    data = generate_delay_phase_dataset(
        SyntheticDelayPhaseConfig(n_trials=8, n_channels=8, n_time=64, max_delay=12, seed=3)
    )
    edge = np.asarray(data["edge_gt"]) != 0

    assert not np.any(edge & edge.T)
    assert int(edge.sum()) >= 6


def test_synthetic_delay_grid_preserves_carrier_nyquist_rate():
    data = generate_delay_phase_dataset(
        SyntheticDelayPhaseConfig(n_trials=16, n_channels=8, n_time=256, max_delay=64)
    )

    model_cfg, raw_samples_per_step = _synthetic_delay_model_cfg(
        {"model": {"graph_rate_hz": 31.25}}, data
    )

    assert model_cfg["graph_rate_hz"] >= 80.0
    assert model_cfg["graph_rate_hz"] / 2.0 > 24.0
    assert raw_samples_per_step <= 250.0 / 80.0
    assert model_cfg["d_max"] * raw_samples_per_step >= float(data["delay_gt"].max())
    assert model_cfg["euclidean_alignment"] is False
    assert model_cfg["reference_augmentation_prob"] == 0.0


def test_nonharmonic_synthetic_carriers_make_long_delays_identifiable():
    data = generate_delay_phase_dataset(
        SyntheticDelayPhaseConfig(
            n_trials=64, n_channels=12, n_time=256, max_delay=64, seed=2
        )
    )
    carrier = torch.as_tensor(data["X"][:, None])
    carrier = torch.nn.functional.interpolate(
        carrier.flatten(0, 1), size=82, mode="linear", align_corners=False
    ).reshape(64, 1, 12, 82)

    evidence = DelayPhaseGraphSynapse._gcc_phat_evidence(carrier, n_delays=12)
    learned = evidence[0, 0].argmax(dim=-1).numpy() * (256.0 / 82.0)
    active = data["edge_gt"].astype(bool)

    assert np.corrcoef(data["delay_gt"][active], learned[active])[0, 1] > 0.95


def test_time_reversal_flips_directed_delay_ground_truth():
    data = generate_delay_phase_dataset(
        SyntheticDelayPhaseConfig(n_trials=8, n_channels=6, n_time=64, max_delay=8)
    )

    reversed_data = time_reverse(data)

    np.testing.assert_array_equal(reversed_data["X"], data["X"][..., ::-1])
    np.testing.assert_array_equal(reversed_data["edge_gt"], data["edge_gt"].T)
    np.testing.assert_array_equal(reversed_data["delay_gt"], data["delay_gt"].T)
