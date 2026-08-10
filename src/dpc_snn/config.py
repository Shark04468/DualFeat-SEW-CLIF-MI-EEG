"""Configuration loading and merging utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(data).__name__}")
    return data


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (Path(base_dir) if base_dir else Path.cwd()) / p


def load_experiment_config(
    config_path: str | Path,
    experiment_id: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = load_yaml(config_path)
    cfg = deepcopy(raw.get("defaults", {}))
    cfg["config_path"] = str(config_path)
    cfg["experiments"] = raw.get("experiments", {})

    if experiment_id is not None:
        experiments = raw.get("experiments", {})
        if experiment_id not in experiments:
            known = ", ".join(sorted(experiments))
            raise KeyError(f"Unknown experiment {experiment_id!r}. Known experiments: {known}")
        cfg = deep_update(cfg, {"experiment_id": experiment_id, "experiment": experiments[experiment_id]})

    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg

