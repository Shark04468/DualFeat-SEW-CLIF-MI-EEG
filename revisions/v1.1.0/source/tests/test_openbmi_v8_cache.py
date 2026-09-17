from __future__ import annotations

import numpy as np
import pytest

from dpc_snn.data.openbmi_v8_cache import (
    default_openbmi_v8_cache_root,
    load_openbmi_v8_cache,
    openbmi_v8_cache_paths,
    write_openbmi_v8_cache,
)
from dpc_snn.experiments.v8_publication_baselines import array_sha256


def _fixture(trials: int = 4):
    x = np.arange(trials * 2 * 5, dtype=np.float32).reshape(trials, 2, 5)
    y = np.asarray([0, 1, 0, 1], dtype=np.int64)
    rows = [
        {
            "dataset": "OpenBMI_Lee2019_MI",
            "subject": "1",
            "session": "S1",
            "run": "0",
            "trial_id": f"trial-{index}",
            "class": int(y[index]),
            "sfreq": 250.0,
            "ch_names": ["C3", "C4"],
            "epoch_tmin": -1.0,
            "epoch_tmax": -0.98,
        }
        for index in range(trials)
    ]
    manifest = {
        "role": "training",
        "channel_names": ["C3", "C4"],
        "signal_sha256": array_sha256(x),
        "label_sha256": array_sha256(y),
    }
    identity = {"dataset": "OpenBMI_Lee2019_MI", "subject": 1, "session": "S1"}
    return x, y, rows, manifest, identity


def test_openbmi_v8_cache_roundtrip_and_hash_guard(tmp_path):
    x, y, rows, manifest, identity = _fixture()
    write_openbmi_v8_cache(
        tmp_path,
        subject=1,
        session="S1",
        role="training",
        x=x,
        y=y,
        rows=rows,
        manifest=manifest,
        identity=identity,
    )
    loaded = load_openbmi_v8_cache(
        tmp_path,
        subject=1,
        session="S1",
        role="training",
        expected_trials=4,
        channel_names=["C3", "C4"],
    )
    assert np.array_equal(loaded[0], x)
    assert np.array_equal(loaded[1], y)
    assert loaded[2:] == (rows, manifest, identity)

    npz_path, _ = openbmi_v8_cache_paths(tmp_path, 1, "S1")
    npz_path.write_bytes(npz_path.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="file hash changed"):
        load_openbmi_v8_cache(
            tmp_path,
            subject=1,
            session="S1",
            role="training",
            expected_trials=4,
            channel_names=["C3", "C4"],
        )


def test_openbmi_v8_cache_default_root(monkeypatch, tmp_path):
    monkeypatch.delenv("DPC_SNN_OPENBMI_V8_ROOT", raising=False)
    monkeypatch.setenv("DPC_SNN_STORAGE_ROOT", str(tmp_path))
    assert default_openbmi_v8_cache_root() == tmp_path / "data" / "processed" / "openbmi_v8"

    explicit = tmp_path / "explicit"
    monkeypatch.setenv("DPC_SNN_OPENBMI_V8_ROOT", str(explicit))
    assert default_openbmi_v8_cache_root() == explicit
