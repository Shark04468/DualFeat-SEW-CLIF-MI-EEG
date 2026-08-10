from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dpc_snn.config import load_yaml
from dpc_snn.data.electrodes import motor_anchor_indices
from dpc_snn.data.cho2017 import load_cho2017_subject
from dpc_snn.data.openbmi import openbmi_session_keys
from dpc_snn.data.v8_openbmi import prepare_openbmi_v8_view
from dpc_snn.experiments.v62_protocol import (
    validate_prediction_schema,
    write_trial_predictions,
)
from dpc_snn.models.eeg_frontend import AnchoredSpatialProjection


ROOT = Path(__file__).resolve().parents[1]


def test_openbmi_session_mapping_prevents_second_session_leakage() -> None:
    keys, labels = openbmi_session_keys(["S1"])

    assert keys == ["0"]
    assert labels == ["S1"]
    assert "1" not in keys


def test_openbmi_session_mapping_is_explicit_and_deduplicated() -> None:
    assert openbmi_session_keys([1, "S1", 2]) == (["0", "1"], ["S1", "S2"])
    with pytest.raises(ValueError, match="expected S1 or S2"):
        openbmi_session_keys(["T"])


def test_spatial_projection_uses_supplied_coordinates_and_motor_anchors() -> None:
    coordinates = torch.tensor(
        [
            [-2.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    projection = AnchoredSpatialProjection(
        5,
        2,
        max_deviation=0.0,
        electrode_coordinates=coordinates,
        anchor_indices=[3, 1, 4],
    )
    normalized = coordinates - coordinates.mean(dim=0, keepdim=True)
    normalized /= normalized.square().sum(dim=-1).sqrt().amax()

    torch.testing.assert_close(projection.electrode_coordinates, normalized)
    torch.testing.assert_close(projection.node_coordinates, normalized[[3, 1]])
    assert projection.anchors.shape == (2, 5)


def test_v51_split_is_disjoint_complete_and_locked() -> None:
    dataset = load_yaml(ROOT / "configs" / "datasets" / "openbmi.yaml")
    development = set(dataset["development_subjects"])
    confirmatory = set(dataset["confirmatory_subjects"])

    assert development.isdisjoint(confirmatory)
    assert development | confirmatory == set(range(1, 55))
    assert len(development) == 12
    assert dataset["development_session"] == "S1"
    assert dataset["confirmatory_test_session"] == "S2"


def test_v51_delay_grid_has_one_millisecond_resolution() -> None:
    protocol = load_yaml(ROOT / "configs" / "experiments" / "v51_hardware_free_plan.yaml")
    model = protocol["model_overrides"]
    task_seconds = 4.0
    graph_step_ms = 1000.0 * task_seconds / int(model["graph_timesteps"])
    grid_step_ms = graph_step_ms / int(model["continuous_delay_grid_oversample"])

    assert grid_step_ms == pytest.approx(1.0)
    assert protocol["real_evidence_gate"]["minimum_routes_per_subject"] == 3
    assert protocol["real_evidence_gate"]["minimum_passing_development_subjects"] == 8
    assert protocol["mechanism_power"]["semi_synthetic_seeds"] == [0, 1, 2]


def test_shared_motor_anchors_are_physical_channel_indices() -> None:
    channel_names = [
        "Fp1",
        "FC3",
        "C3",
        "Cz",
        "C4",
        "FC4",
        "CP3",
        "CP4",
        "FC1",
        "FC2",
        "C1",
        "C2",
        "CP1",
        "CP2",
        "FC5",
        "FC6",
        "C5",
        "C6",
        "CP5",
        "CP6",
    ]
    indices = motor_anchor_indices(channel_names)

    assert len(indices) == 19
    assert channel_names[int(indices[0])] == "FC5"
    assert channel_names[int(indices[-1])] == "CP6"


def test_cho2017_is_reserved_as_external_single_session_replication() -> None:
    dataset = load_yaml(ROOT / "configs" / "datasets" / "cho2017.yaml")

    assert dataset["subjects"] == list(range(1, 53))
    assert dataset["session"] == "S1"
    assert dataset["role"] == "external_mechanism_replication_after_architecture_freeze"
    assert dataset["evaluation"] == "repeated_stratified_within_session_cv"


def test_cho2017_loader_rejects_invalid_subjects_before_dataset_access() -> None:
    with pytest.raises(ValueError, match=r"\[1, 52\]"):
        load_cho2017_subject(0)
    with pytest.raises(ValueError, match=r"\[1, 52\]"):
        load_cho2017_subject(53)


def test_v8_openbmi_adapter_selects_channels_and_antialiases() -> None:
    channel_names = [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
        "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
    ]
    all_names = ["Fp1", *channel_names, "O1"]
    trials = 8
    data = {
        "X": np.random.default_rng(4).normal(size=(trials, len(all_names), 5000)).astype(np.float32),
        "y": np.tile(np.asarray([0, 1]), trials // 2),
        "subject": np.asarray(["1"] * trials),
        "session": np.asarray(["S1"] * trials),
        "run": np.asarray(["0"] * trials),
        "trial_id": np.asarray([f"trial-{index}" for index in range(trials)]),
        "ch_names": all_names,
        "sfreq": 1000.0,
        "epoch_tmin": -1.0,
        "epoch_tmax": 4.0,
    }

    x, y, metadata, manifest = prepare_openbmi_v8_view(
        data, channel_names=channel_names, role="training"
    )

    assert x.shape == (trials, 22, 1250)
    assert y.shape == (trials,)
    assert len(metadata) == trials
    assert manifest["resample_poly_down"] == 4
    assert manifest["sessions"] == ["S1"]


def test_v8_openbmi_adapter_uses_fixed_fcz_interpolation() -> None:
    from dpc_snn.data.v8_openbmi import OPENBMI_FIXED_INPUT_ADAPTER

    channel_names = [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
        "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
    ]
    source_names = [name for name in channel_names if name != "FCz"]
    trials = 4
    values = np.zeros((trials, len(source_names), 5000), dtype=np.float32)
    values[:, source_names.index("FC1"), :] = 2.0
    values[:, source_names.index("FC2"), :] = 4.0
    data = {
        "X": values,
        "y": np.asarray([0, 1, 0, 1]),
        "subject": np.asarray(["1"] * trials),
        "session": np.asarray(["S1"] * trials),
        "run": np.asarray(["0"] * trials),
        "trial_id": np.asarray([f"derived-{index}" for index in range(trials)]),
        "ch_names": source_names,
        "sfreq": 1000.0,
        "epoch_tmin": -1.0,
        "epoch_tmax": 4.0,
    }

    x, _, _, manifest = prepare_openbmi_v8_view(
        data,
        channel_names=channel_names,
        input_adapter=OPENBMI_FIXED_INPUT_ADAPTER,
        role="training",
    )

    assert x.shape == (trials, 22, 1250)
    expected_fcz = 0.5 * (
        x[:, channel_names.index("FC1"), :]
        + x[:, channel_names.index("FC2"), :]
    )
    assert np.allclose(x[:, channel_names.index("FCz"), :], expected_fcz)
    assert manifest["source_channel_indices"][channel_names.index("FCz")] is None
    assert manifest["input_adapter"]["applied_derived_channels"]["FCz"] == {
        "source_channels": ["FC1", "FC2"],
        "weights": [0.5, 0.5],
    }


def test_prediction_schema_accepts_locked_openbmi_session_labels(tmp_path: Path) -> None:
    logits = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    paths = write_trial_predictions(
        tmp_path,
        logits=logits,
        probabilities=probabilities,
        pred=np.asarray([0, 1]),
        label=np.asarray([0, 1]),
        subject="1",
        session="S2",
        run="0",
        trial_id=["s2-0", "s2-1"],
        seed=0,
        model="v8",
    )

    assert validate_prediction_schema(paths["npz"], paths["csv"])["n_trials"] == 2
