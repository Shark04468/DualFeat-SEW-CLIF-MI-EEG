from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.evaluate_v8_fixed_fusion_gate import (
    fixed_fusion_gate,
    two_way_cluster_bootstrap_mean_ci,
)


def test_fixed_fusion_gate_enforces_all_preregistered_thresholds() -> None:
    passing = [
        {"delta_pp": value}
        for value in (1.2, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4, -0.1, -0.2)
    ]
    result = fixed_fusion_gate(passing)
    assert result["passed"] is True
    assert result["positive_pairs"] == 7

    insufficient_signs = [{"delta_pp": value} for value in (2, 2, 2, 2, 2, 2, 0, 0, 0)]
    result = fixed_fusion_gate(insufficient_signs)
    assert result["passed"] is False
    assert result["checks"]["positive_pairs"] is False


def test_two_way_cluster_bootstrap_is_deterministic_and_finite() -> None:
    values = np.asarray([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0], [-1.0, 0.0, 1.0]])
    first = two_way_cluster_bootstrap_mean_ci(values, samples=500, seed=7)
    second = two_way_cluster_bootstrap_mean_ci(values, samples=500, seed=7)
    assert first == second
    assert np.isfinite(first).all()
    assert first[0] <= values.mean() <= first[1]


def test_fixed_fusion_gate_script_is_directly_executable() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "evaluate_v8_fixed_fusion_gate.py"), "--help"],
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--e1-root" in completed.stdout
