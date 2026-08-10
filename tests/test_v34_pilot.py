from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_pilot_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_v34_pilot.py"
    spec = importlib.util.spec_from_file_location("run_v34_pilot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_existing_complete_metrics_are_recovered(tmp_path: Path) -> None:
    pilot = _load_pilot_module()
    metrics_dir = tmp_path / "subject_1" / "seed_0" / "full" / "dpc_snn"
    metrics_dir.mkdir(parents=True)
    metrics = {
        "accuracy": 0.5,
        "balanced_accuracy": 0.5,
        "kappa": 1 / 3,
        "macro_f1": 0.49,
    }
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    row = pilot._load_existing_result(tmp_path, "1", 0, "full")

    assert row is not None
    assert row["status"] == "completed"
    assert row["variant"] == "full"
    assert row["protocol"] == "v35_paired_T_to_E_pilot"
    assert row["evaluation_split"] == "heldout_session_E"


def test_incomplete_metrics_are_not_recovered(tmp_path: Path) -> None:
    pilot = _load_pilot_module()
    metrics_dir = tmp_path / "subject_1" / "seed_0" / "full" / "dpc_snn"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.json").write_text('{"accuracy": 0.5}', encoding="utf-8")

    assert pilot._load_existing_result(tmp_path, "1", 0, "full") is None
