"""Compact, hash-verified OpenBMI views for clone-safe experiment recovery."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from dpc_snn.experiments.v8_publication_baselines import array_sha256
from dpc_snn.experiments.v62_protocol import file_sha256
from dpc_snn.utils.io import ensure_dir, read_json, write_json

CACHE_SCHEMA = "dpc-snn-openbmi-v8-compact-cache/v1"


def default_openbmi_v8_cache_root() -> Path | None:
    explicit = os.environ.get("DPC_SNN_OPENBMI_V8_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    storage = os.environ.get("DPC_SNN_STORAGE_ROOT")
    if storage:
        return Path(storage).expanduser().resolve() / "data" / "processed" / "openbmi_v8"
    return None


def openbmi_v8_cache_paths(root: str | Path, subject: int, session: str) -> tuple[Path, Path]:
    normalized = str(session).strip().upper()
    if normalized not in {"S1", "S2"}:
        raise ValueError(f"OpenBMI cache session must be S1 or S2, got {session!r}")
    directory = Path(root).expanduser().resolve() / f"subject_{int(subject):02d}"
    return directory / f"{normalized}.npz", directory / f"{normalized}.json"


def write_openbmi_v8_cache(
    root: str | Path,
    *,
    subject: int,
    session: str,
    role: str,
    x: np.ndarray,
    y: np.ndarray,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    identity: dict[str, Any],
) -> dict[str, Any]:
    npz_path, json_path = openbmi_v8_cache_paths(root, subject, session)
    ensure_dir(npz_path.parent)
    x_value = np.ascontiguousarray(x, dtype=np.float32)
    y_value = np.ascontiguousarray(y, dtype=np.int64)
    if x_value.ndim != 3 or y_value.shape != (x_value.shape[0],):
        raise ValueError("OpenBMI compact cache requires X=[trial,channel,time] and y=[trial]")
    temporary = npz_path.with_suffix(".npz.partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, X=x_value, y=y_value)
    temporary.replace(npz_path)
    payload = {
        "schema": CACHE_SCHEMA,
        "subject": int(subject),
        "session": str(session).upper(),
        "role": str(role),
        "npz_file": npz_path.name,
        "npz_sha256": file_sha256(npz_path),
        "signal_sha256": array_sha256(x_value),
        "label_sha256": array_sha256(y_value),
        "shape": list(x_value.shape),
        "rows": rows,
        "manifest": manifest,
        "identity": identity,
    }
    write_json(json_path, payload)
    return payload


def load_openbmi_v8_cache(
    root: str | Path,
    *,
    subject: int,
    session: str,
    role: str,
    expected_trials: int,
    channel_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    npz_path, json_path = openbmi_v8_cache_paths(root, subject, session)
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"OpenBMI compact cache is absent: {npz_path}")
    payload = read_json(json_path)
    expected_header = {
        "schema": CACHE_SCHEMA,
        "subject": int(subject),
        "session": str(session).upper(),
        "role": str(role),
        "npz_file": npz_path.name,
    }
    for key, value in expected_header.items():
        if payload.get(key) != value:
            raise RuntimeError(f"OpenBMI compact cache metadata mismatch for {key}: {json_path}")
    if payload.get("npz_sha256") != file_sha256(npz_path):
        raise RuntimeError(f"OpenBMI compact cache file hash changed: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as archive:
        x = np.ascontiguousarray(archive["X"], dtype=np.float32)
        y = np.ascontiguousarray(archive["y"], dtype=np.int64)
    if list(x.shape) != payload.get("shape") or x.shape[0] != int(expected_trials):
        raise RuntimeError(f"OpenBMI compact cache shape/trial count mismatch: {npz_path}")
    if y.shape != (x.shape[0],):
        raise RuntimeError(f"OpenBMI compact cache labels are misaligned: {npz_path}")
    if payload.get("signal_sha256") != array_sha256(x):
        raise RuntimeError(f"OpenBMI compact signal hash changed: {npz_path}")
    if payload.get("label_sha256") != array_sha256(y):
        raise RuntimeError(f"OpenBMI compact label hash changed: {npz_path}")
    rows = list(payload["rows"])
    if len(rows) != x.shape[0] or len({str(row["trial_id"]) for row in rows}) != x.shape[0]:
        raise RuntimeError(f"OpenBMI compact cache trial metadata is invalid: {json_path}")
    requested_channels = [str(value) for value in channel_names]
    if payload["manifest"].get("channel_names") != requested_channels:
        raise RuntimeError(f"OpenBMI compact cache channel contract mismatch: {json_path}")
    return x, y, rows, dict(payload["manifest"]), dict(payload["identity"])
