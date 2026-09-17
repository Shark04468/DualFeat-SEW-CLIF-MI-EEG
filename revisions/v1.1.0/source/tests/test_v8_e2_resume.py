from __future__ import annotations

import numpy as np
import pytest

from dpc_snn.experiments.v62_protocol import (
    ArtifactManifestError,
    write_run_artifact_manifest,
)
from scripts.run_v8_e2_zero_delay import (
    FOLD_FILES,
    FOLD_REQUIRED_FILES,
    _load_fold,
)


def _complete_fold(directory) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray([0, 2], dtype=np.int64)
    labels = np.asarray([1, 3], dtype=np.int64)
    for name in FOLD_FILES:
        (directory / name).write_bytes(b"placeholder")
    (directory / "result.json").write_text(
        '{"fold": 0, "elapsed_seconds": 1.0, "optimizer_steps": 1, "parameters": 1}',
        encoding="utf-8",
    )
    np.savez_compressed(
        directory / "outer_test_predictions.npz",
        indices=indices,
        logits=np.zeros((2, 4), dtype=np.float32),
        prefix_logits=np.zeros((2, 2, 4), dtype=np.float32),
        labels=labels,
    )
    write_run_artifact_manifest(directory, required_files=FOLD_REQUIRED_FILES)
    return indices, labels


def test_e2_partial_resume_verifies_completed_fold_hashes(tmp_path) -> None:
    indices, labels = _complete_fold(tmp_path)
    loaded = _load_fold(
        tmp_path,
        expected_indices=indices,
        expected_labels=labels,
        endpoints=2,
    )
    assert loaded is not None

    (tmp_path / "history.csv").write_text("corrupted", encoding="utf-8")
    with pytest.raises(ArtifactManifestError, match="artifact size/hash mismatch"):
        _load_fold(
            tmp_path,
            expected_indices=indices,
            expected_labels=labels,
            endpoints=2,
        )
