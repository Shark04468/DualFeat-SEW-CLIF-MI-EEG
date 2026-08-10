from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from scripts.run_v8_e6_bci2a_frozen import (
    _run_subject_worker,
    _subject_worker_command,
)
from scripts.run_v8_e6_baselines_frozen import (
    _baseline_subject_worker_command,
    _run_baseline_subject_worker,
)
from scripts.run_v8_e8_openbmi_confirmation import (
    _e8_subject_worker_command,
    _run_e8_subject_worker,
)
from scripts.run_v8_e9_frozen_ablations import (
    _e9_subject_worker_command,
    _run_e9_subject_worker,
)


def test_e6_subject_worker_command_preserves_registered_identity(tmp_path: Path) -> None:
    args = Namespace(
        data=str(tmp_path / "data"),
        freeze=str(tmp_path / "freeze.json"),
        config=str(tmp_path / "e6.yaml"),
        device="cuda",
    )
    command = _subject_worker_command(
        args=args,
        output=tmp_path / "campaign",
        subject=7,
        variants=["frozen_primary", "matched_ann"],
        seeds=[0, 1, 2, 3, 4],
    )
    joined = " ".join(command)
    assert "--worker-subject 7" in joined
    assert "--worker-variants frozen_primary,matched_ann" in joined
    assert "--worker-seeds 0,1,2,3,4" in joined
    assert "--canary" not in command


def test_e6_subject_worker_rejects_invalid_cpu_thread_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DPC_SNN_E6_WORKER_THREADS", "33")
    with pytest.raises(ValueError, match="between 1 and 32"):
        _run_subject_worker(["unused"], [tmp_path / "metrics.json"])


def test_e6_baseline_subject_worker_command_preserves_registered_identity(
    tmp_path: Path,
) -> None:
    args = Namespace(
        data=str(tmp_path / "data"),
        source_root=str(tmp_path / "sources"),
        freeze=str(tmp_path / "freeze.json"),
        config=str(tmp_path / "e6_baselines.yaml"),
        device="cuda",
    )
    command = _baseline_subject_worker_command(
        args=args,
        output=tmp_path / "campaign",
        subject=9,
        models=["atcnet", "fbcnet"],
        seeds=[0, 1],
    )
    joined = " ".join(command)
    assert "--worker-subject 9" in joined
    assert "--worker-models atcnet,fbcnet" in joined
    assert "--worker-seeds 0,1" in joined


def test_e6_baseline_worker_rejects_invalid_cpu_thread_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DPC_SNN_E6_BASELINE_WORKER_THREADS", "0")
    with pytest.raises(ValueError, match="between 1 and 32"):
        _run_baseline_subject_worker(["unused"], [tmp_path / "metrics.json"])


def test_e8_subject_worker_command_keeps_s1_s2_inside_one_subject(
    tmp_path: Path,
) -> None:
    args = Namespace(
        freeze=str(tmp_path / "freeze.json"),
        unlock=str(tmp_path / "unlock.json"),
        device="cuda",
    )
    command = _e8_subject_worker_command(
        args=args,
        output=tmp_path / "campaign",
        subject=42,
        arms=["frozen_primary", "matched_ann"],
        seeds=[0, 1, 2, 3, 4],
    )
    joined = " ".join(command)
    assert "--worker-subject 42" in joined
    assert "--worker-arms frozen_primary,matched_ann" in joined
    assert "--worker-seeds 0,1,2,3,4" in joined
    assert "--canary" not in command


def test_e8_worker_rejects_invalid_cpu_thread_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DPC_SNN_E8_WORKER_THREADS", "0")
    with pytest.raises(ValueError, match="between 1 and 32"):
        _run_e8_subject_worker(["unused"], [tmp_path / "metrics.json"])


def test_e9_subject_worker_command_preserves_frozen_ablation_identity(
    tmp_path: Path,
) -> None:
    args = Namespace(
        data=str(tmp_path / "data"),
        freeze=str(tmp_path / "freeze.json"),
        e6=str(tmp_path / "e6"),
        e6_audit=str(tmp_path / "e6_audit"),
        config=str(tmp_path / "e9.yaml"),
        device="cuda",
    )
    command = _e9_subject_worker_command(
        args=args,
        output=tmp_path / "campaign",
        subject=8,
        variants=["no_covariance", "no_temporal"],
        seeds=[0, 1, 2, 3, 4],
    )
    joined = " ".join(command)
    assert "--worker-subject 8" in joined
    assert "--worker-variants no_covariance,no_temporal" in joined
    assert "--worker-seeds 0,1,2,3,4" in joined


def test_e9_worker_rejects_invalid_cpu_thread_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DPC_SNN_E9_WORKER_THREADS", "33")
    with pytest.raises(ValueError, match="between 1 and 32"):
        _run_e9_subject_worker(["unused"], [tmp_path / "metrics.json"])
