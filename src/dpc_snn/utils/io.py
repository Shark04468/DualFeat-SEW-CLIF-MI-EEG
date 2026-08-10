"""Small IO helpers used by scripts and experiment runners."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _atomic_replace(path: Path, writer: Any) -> Path:
    """Write beside the destination and publish with one atomic replace."""

    ensure_dir(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    p = Path(path)

    def writer(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

    return _atomic_replace(p, writer)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
    rows = list(rows)
    p = Path(path)
    ensure_dir(p.parent)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    def write_rows(temporary: Path) -> None:
        with temporary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())

    return _atomic_replace(p, write_rows)


def save_npz(path: str | Path, **arrays: Any) -> Path:
    p = Path(path)

    def writer(temporary: Path) -> None:
        with temporary.open("wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())

    return _atomic_replace(p, writer)


def save_npy(path: str | Path, array: Any) -> Path:
    p = Path(path)

    def writer(temporary: Path) -> None:
        with temporary.open("wb") as f:
            np.save(f, array)
            f.flush()
            os.fsync(f.fileno())

    return _atomic_replace(p, writer)
