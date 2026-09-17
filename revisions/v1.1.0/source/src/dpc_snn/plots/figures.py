"""Figure reproduction utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dpc_snn.utils.imports import has_module
from dpc_snn.utils.io import ensure_dir, write_csv


def reproduce_figures(results_root: Path, figures_root: Path) -> Path:
    results_root = results_root.resolve()
    if not results_root.exists():
        raise FileNotFoundError(f"Figure results root does not exist: {results_root}")
    ensure_dir(figures_root / "source_data")
    ensure_dir(figures_root / "main")
    source_files = sorted({path.resolve() for path in results_root.glob("**/*.csv") if figures_root.resolve() not in path.resolve().parents})
    rows = []
    for i, source in enumerate(source_files):
        manifest = _nearest_run_manifest(source, results_root)
        run_info = _load_json(manifest) if manifest else {}
        checkpoint = _nearest_checkpoint(source, results_root)
        config_value = str(run_info.get("config_path", "")).strip()
        config_path = Path(config_value) if config_value else None
        rows.append(
            {
                "figure_id": "auto",
                "panel_id": f"source_{i:03d}",
                "source_csv": str(source),
                "source_npz": "",
                "config_yaml": str(config_path.resolve()) if config_path and config_path.is_file() else "",
                "script_path": "scripts/reproduce_figures.py",
                "generated_time": "",
                "git_commit": "",
                "data_hash": _sha256(source),
                "model_checkpoint": str(checkpoint) if checkpoint else "",
                "checkpoint_hash": _sha256(checkpoint) if checkpoint else "",
                "run_manifest": str(manifest) if manifest else "",
                "run_manifest_hash": _sha256(manifest) if manifest else "",
                "config_hash": _sha256(config_path) if config_path and config_path.is_file() else "",
                "statistical_test_file": str(source) if source.name == "stat_tests.csv" else "",
            }
        )
    manifest = figures_root / "source_data" / "manifest.csv"
    write_csv(manifest, rows)
    if has_module("matplotlib") and source_files:
        _plot_first_metric(source_files[0], figures_root / "main" / "auto_metric.png")
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _nearest_run_manifest(source: Path, results_root: Path) -> Path | None:
    for parent in [source.parent, *source.parents]:
        candidate = parent / "run_manifest.json"
        if candidate.exists():
            return candidate
        if parent == results_root:
            break
    return None


def _nearest_checkpoint(source: Path, results_root: Path) -> Path | None:
    for parent in [source.parent, *source.parents]:
        candidate = parent / "model_checkpoint.pt"
        if candidate.exists():
            return candidate
        if parent == results_root:
            break
    return None


def _plot_first_metric(source: Path, output: Path) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(source)
    metric_cols = [c for c in ["accuracy", "kappa", "macro_f1", "delay_corr"] if c in df.columns]
    if not metric_cols:
        return
    ax = df[metric_cols].plot(kind="bar", figsize=(6, 3))
    ax.set_ylabel("metric")
    ax.figure.tight_layout()
    ensure_dir(output.parent)
    ax.figure.savefig(output, dpi=200)
    plt.close(ax.figure)
