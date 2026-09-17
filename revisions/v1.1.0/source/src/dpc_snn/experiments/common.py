"""Shared experiment-runner helpers."""

from __future__ import annotations

import copy
import hashlib
from importlib import metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dpc_snn.config import load_yaml
from dpc_snn.data.splits import stratified_split_indices
from dpc_snn.preprocessing.hilbert import band_amplitude_phase
from dpc_snn.preprocessing.standardize import euclidean_alignment_matrix, standardize_train_test
from dpc_snn.utils.imports import has_module
from dpc_snn.utils.io import ensure_dir, save_npz, write_json
from dpc_snn.utils.seed import seed_everything


DEFAULT_BANDS = {
    "theta_mu": [6.0, 8.0],
    "mu_low": [8.0, 10.0],
    "mu_high": [10.0, 13.0],
    "beta_low": [13.0, 18.0],
    "beta_mid": [18.0, 24.0],
    "beta_high": [24.0, 30.0],
    "gamma_low": [30.0, 40.0],
}


def load_referenced_yaml(path_or_dict: str | dict[str, Any] | None) -> dict[str, Any]:
    if path_or_dict is None:
        return {}
    if isinstance(path_or_dict, dict):
        return copy.deepcopy(path_or_dict)
    return load_yaml(path_or_dict)


def resolve_dataset_cfg(cfg: dict[str, Any], key: str = "synthetic_config") -> dict[str, Any]:
    direct = cfg.get("dataset", {})
    if direct:
        return copy.deepcopy(direct)
    data_cfg = cfg.get("data", {})
    path = data_cfg.get(key)
    return load_referenced_yaml(path)


def resolve_model_cfg(cfg: dict[str, Any], model_name: str = "dpc_snn") -> dict[str, Any]:
    model_cfg = cfg.get("model", {})
    path = model_cfg.get("dpc_snn_config") if isinstance(model_cfg, dict) else None
    base = load_referenced_yaml(path) if path else {}
    if isinstance(model_cfg, dict):
        overrides = {
            k: v
            for k, v in model_cfg.items()
            if not k.endswith("_config") and k != "baselines_config"
        }
        base.update(overrides)
    base["model"] = model_name
    return base


def add_features(
    data: dict[str, Any], bands: dict[str, list[float]] | None = None
) -> dict[str, Any]:
    bands = bands or DEFAULT_BANDS
    amp, phase, band_names = band_amplitude_phase(data["X"], float(data.get("sfreq", 250.0)), bands)
    out = dict(data)
    out["amplitude"] = amp.astype(np.float32)
    out["phase"] = phase.astype(np.float32)
    out["band_names"] = np.asarray(band_names)
    return out


def split_train_val(
    data: dict[str, Any], seed: int = 0, val_fraction: float = 0.2
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_idx, val_idx = stratified_split_indices(data["y"], val_fraction=val_fraction, seed=seed)
    x_train, x_val, stats = standardize_train_test(data["X"][train_idx], data["X"][val_idx])
    train = subset(data, train_idx)
    val = subset(data, val_idx)
    train["X"] = x_train
    val["X"] = x_val
    train["standardize_mean"] = stats["mean"]
    train["standardize_std"] = stats["std"]
    return train, val


def subset(data: dict[str, Any], idx: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n = len(data["y"])
    for key, value in data.items():
        arr = np.asarray(value)
        if arr.shape[:1] == (n,):
            out[key] = arr[idx]
        else:
            out[key] = value
    return out


def _single_metadata_value(data: dict[str, Any], key: str) -> str:
    if key not in data:
        return ""
    arr = np.asarray(data[key]).astype(str)
    if arr.shape[:1] != (len(data["y"]),):
        return ""
    values = np.unique(arr)
    return str(values[0]) if values.size == 1 else ""


def _feature_bands_from_cfg(cfg: dict[str, Any]) -> dict[str, list[float]]:
    bands = cfg.get("feature_bands")
    if isinstance(bands, dict) and bands:
        return {str(key): [float(value[0]), float(value[1])] for key, value in bands.items()}
    return DEFAULT_BANDS


def fold_prior_cache_fingerprint(
    train: dict[str, Any], model_cfg: dict[str, Any], seed: int
) -> str:
    """Bind a fold prior to exact inputs and every evidence-defining setting."""

    settings = {
        "schema": 5,
        "seed": int(seed),
        "sfreq": float(train.get("sfreq", 250.0)),
        "epoch_tmin": float(train.get("epoch_tmin", 0.0)),
        "epoch_tmax": train.get("epoch_tmax"),
        "task_tmin": float(model_cfg.get("task_tmin", 0.0)),
        "task_tmax": float(model_cfg.get("task_tmax", 4.0)),
        "evidence_band_edges_hz": model_cfg.get("delay_evidence_band_edges_hz", []),
        "output_band_edges_hz": model_cfg.get("band_edges_hz", []),
        "classification_band_edges_hz": model_cfg.get("classification_band_edges_hz", []),
        "latent_nodes": int(model_cfg.get("latent_nodes", np.asarray(train["X"]).shape[1])),
        "graph_timesteps": int(model_cfg.get("graph_timesteps", 0)),
        "d_max": int(model_cfg.get("d_max", 8)),
        "bootstrap_samples": int(model_cfg.get("fold_local_bootstrap_samples", 128)),
        "delay_evidence_estimator": str(
            model_cfg.get("delay_evidence_estimator", "legacy_positive_lag_hurdle")
        ),
        "continuous_delay_grid_oversample": int(
            model_cfg.get("continuous_delay_grid_oversample", 4)
        ),
        "continuous_delay_min_bayes_factor": float(
            model_cfg.get("continuous_delay_min_bayes_factor", 3.0)
        ),
        "continuous_delay_min_bootstrap_frequency": float(
            model_cfg.get("continuous_delay_min_bootstrap_frequency", 0.70)
        ),
        "continuous_delay_min_direction_probability": float(
            model_cfg.get("continuous_delay_min_direction_probability", 0.80)
        ),
        "delay_evidence_reference": str(model_cfg.get("delay_evidence_reference", "carrier")),
        "algorithm_fingerprint": str(
            model_cfg.get("fold_prior_algorithm_fingerprint", "unversioned")
        ),
    }
    digest = hashlib.sha256(
        json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for key in ("y",):
        array = np.ascontiguousarray(train[key])
        digest.update(key.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    evidence_key = "delay_evidence_X" if "delay_evidence_X" in train else "X"
    array = np.ascontiguousarray(train[evidence_key])
    digest.update(evidence_key.encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def fold_prior_scientific_gate_failures(
    summary: dict[str, Any], model_cfg: dict[str, Any]
) -> list[str]:
    """Return classifier-free reasons a fold prior is not scientifically usable."""

    failures: list[str] = []
    min_edges = int(model_cfg.get("fold_prior_min_accepted_edges", 1))
    if int(summary.get("accepted_evidence_edges", 0)) < min_edges:
        failures.append("no_stable_delay_edges")
    normal_edges = int(summary.get("normal_node_edges", 0))
    surrogate_edges = int(summary.get("phase_surrogate_node_edges", 0))
    if normal_edges <= surrogate_edges:
        failures.append("phase_surrogate_did_not_reduce_edges")
    min_drop = float(model_cfg.get("fold_prior_min_surrogate_support_drop", 0.0))
    if float(summary.get("phase_surrogate_support_drop", float("-inf"))) <= min_drop:
        failures.append("phase_surrogate_did_not_reduce_support")
    max_reversal_mae = float(model_cfg.get("fold_prior_max_reversal_support_mae", 0.05))
    if float(summary.get("time_reversal_transpose_support_mae", float("inf"))) > max_reversal_mae:
        failures.append("time_reversal_did_not_transpose_direction")
    return failures


def train_split(
    model_name: str,
    train: dict[str, Any],
    val: dict[str, Any],
    cfg: dict[str, Any],
    output_dir: str | Path,
    model_cfg: dict[str, Any] | None = None,
    test: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from dpc_snn.models import build_model
    from dpc_snn.training.train import train_model
    from dpc_snn.utils.torch import resolve_device

    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    bands = _feature_bands_from_cfg(cfg)
    model_cfg = model_cfg or resolve_model_cfg(cfg, model_name=model_name)
    if model_name == "dpc_snn":
        if not model_cfg.get("band_edges_hz"):
            model_cfg["band_edges_hz"] = [
                [float(edge[0]), float(edge[1])] for edge in bands.values()
            ]
        model_cfg["n_bands"] = len(model_cfg["band_edges_hz"])
        if bool(model_cfg.get("euclidean_alignment", False)):
            if bool(model_cfg.get("coordinate_spatial_projection", True)):
                raise ValueError(
                    "Sensor-space EA cannot precede a fixed electrode-coordinate prior. "
                    "Project to physical sources first or disable coordinate_spatial_projection."
                )
            model_cfg["alignment_matrix"] = euclidean_alignment_matrix(train["X"]).tolist()
    else:
        train = add_features(train, bands)
        val = add_features(val, bands)
        test = add_features(test, bands) if test is not None else None
    model_cfg["n_channels"] = int(train["X"].shape[1])
    model_cfg["n_classes"] = int(np.max(train["y"]) + 1)
    for metadata_key in ("electrode_coordinates", "spatial_anchor_indices"):
        if metadata_key in train:
            model_cfg[metadata_key] = np.asarray(train[metadata_key]).tolist()
    if "amplitude" in train:
        model_cfg["n_bands"] = int(train["amplitude"].shape[1])
    model_cfg["samples"] = int(train["X"].shape[-1])
    sfreq = float(train.get("sfreq", 250.0))
    model_cfg["sfreq"] = sfreq
    model_cfg["input_duration_seconds"] = model_cfg["samples"] / sfreq
    model_cfg["timestep_seconds"] = model_cfg["input_duration_seconds"] / int(
        model_cfg.get("timesteps", 16)
    )
    epoch_tmin = float(train.get("epoch_tmin", 0.0))
    epoch_tmax = float(train.get("epoch_tmax", epoch_tmin + model_cfg["input_duration_seconds"]))
    model_cfg["epoch_tmin"] = epoch_tmin
    task_tmin = max(epoch_tmin, float(model_cfg.get("task_tmin", 0.0)))
    task_tmax = min(epoch_tmax, float(model_cfg.get("task_tmax", epoch_tmax)))
    if task_tmax <= task_tmin:
        task_tmin, task_tmax = epoch_tmin, epoch_tmax
    model_cfg["task_tmin"] = task_tmin
    model_cfg["task_tmax"] = task_tmax
    graph_rate = float(model_cfg.get("graph_rate_hz", 80.0))
    model_cfg["graph_timesteps"] = max(1, int(round((task_tmax - task_tmin) * graph_rate)))
    # Delay steps live on the task-window graph, not on the baseline-inclusive
    # input epoch. Keep persisted timing metadata identical to the synapse's
    # actual physical step duration.
    model_cfg["timestep_seconds"] = (task_tmax - task_tmin) / model_cfg["graph_timesteps"]
    model = build_model(model_name, model_cfg)
    fold_prior_summary = None
    if model_name == "dpc_snn" and bool(model_cfg.get("fold_local_evidence_prior", False)):
        from dpc_snn.analysis.evidence_space import (
            fold_local_continuous_delay_prior,
            fold_local_model_evidence_prior,
        )

        cache_value = model_cfg.get("fold_local_evidence_prior_path")
        cache_path = Path(cache_value) if cache_value else None
        cache_summary_path = cache_path.with_suffix(".json") if cache_path is not None else None
        expected_fingerprint = fold_prior_cache_fingerprint(train, model_cfg, seed)
        prior_train = train
        if "delay_evidence_X" in train:
            prior_train = {**train, "X": train["delay_evidence_X"]}
        cache_invalid_reason = "cache_missing"
        cache_reused = False
        if cache_path is not None and cache_path.exists() and cache_summary_path.exists():
            try:
                candidate_summary = json.loads(cache_summary_path.read_text(encoding="utf-8"))
                if candidate_summary.get("cache_fingerprint") != expected_fingerprint:
                    cache_invalid_reason = "fingerprint_mismatch"
                else:
                    with np.load(cache_path, allow_pickle=False) as cached:
                        prior_arrays = {key: cached[key] for key in cached.files}
                    required = {
                        "route_probability",
                        "positive_delay_probability",
                        "fractional_delay_target",
                        "connectivity_prior",
                    }
                    if not required.issubset(prior_arrays):
                        raise KeyError(
                            f"missing arrays: {sorted(required.difference(prior_arrays))}"
                        )
                    fold_prior_summary = {**candidate_summary, "cache_reused": True}
                    cache_reused = True
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                cache_invalid_reason = f"cache_invalid:{type(exc).__name__}"
        if not cache_reused:
            prior_builder = (
                fold_local_continuous_delay_prior
                if str(model_cfg.get("delay_evidence_estimator", "")) == "continuous_signed_phase"
                else fold_local_model_evidence_prior
            )
            prior_kwargs = {}
            if prior_builder is fold_local_continuous_delay_prior:
                prior_kwargs.update(
                    grid_oversample=int(model_cfg.get("continuous_delay_grid_oversample", 4)),
                    min_bayes_factor=float(model_cfg.get("continuous_delay_min_bayes_factor", 3.0)),
                    min_bootstrap_frequency=float(
                        model_cfg.get("continuous_delay_min_bootstrap_frequency", 0.70)
                    ),
                    min_direction_probability=float(
                        model_cfg.get("continuous_delay_min_direction_probability", 0.80)
                    ),
                )
            prior_arrays, fold_prior_summary = prior_builder(
                prior_train,
                evidence_band_edges_hz=model_cfg.get("delay_evidence_band_edges_hz", []),
                output_band_edges_hz=model_cfg["band_edges_hz"],
                n_nodes=min(
                    int(model_cfg.get("latent_nodes", train["X"].shape[1])), train["X"].shape[1]
                ),
                graph_steps=int(model_cfg["graph_timesteps"]),
                max_delay_steps=int(model_cfg.get("d_max", 8)),
                bootstrap_samples=int(model_cfg.get("fold_local_bootstrap_samples", 128)),
                seed=seed,
                device=resolve_device(str(cfg.get("device", "cpu"))),
                task_tmin=float(model_cfg.get("task_tmin", 0.0)),
                task_tmax=float(model_cfg.get("task_tmax", 4.0)),
                **prior_kwargs,
            )
            fold_prior_summary = {
                **fold_prior_summary,
                "seed": seed,
                "cache_reused": False,
                "cache_fingerprint": expected_fingerprint,
                "cache_invalid_reason": cache_invalid_reason,
            }
            if cache_path is not None and cache_summary_path is not None:
                save_npz(cache_path, **prior_arrays)
                write_json(cache_summary_path, fold_prior_summary)
        gate_failures = fold_prior_scientific_gate_failures(fold_prior_summary, model_cfg)
        fold_prior_summary = {
            **fold_prior_summary,
            "scientific_gate_passed": not gate_failures,
            "scientific_gate_failures": gate_failures,
        }
        prior_dir = ensure_dir(Path(output_dir))
        save_npz(prior_dir / "fold_local_evidence_prior.npz", **prior_arrays)
        write_json(prior_dir / "fold_local_evidence_prior.json", fold_prior_summary)
        if cache_summary_path is not None:
            write_json(cache_summary_path, fold_prior_summary)
        if bool(model_cfg.get("require_valid_fold_delay_evidence", False)) and gate_failures:
            raise RuntimeError(
                "Fold-local delay evidence failed the classifier-free scientific gate: "
                + ", ".join(gate_failures)
            )
        model.load_fold_local_evidence_prior(
            torch.from_numpy(prior_arrays["route_probability"]),
            torch.from_numpy(prior_arrays["positive_delay_probability"]),
            torch.from_numpy(prior_arrays["fractional_delay_target"]),
            torch.from_numpy(prior_arrays["connectivity_prior"]),
        )
    run_cfg = copy.deepcopy(cfg)
    run_cfg["device"] = resolve_device(str(cfg.get("device", "cpu")))
    run_cfg["n_classes"] = model_cfg["n_classes"]
    run_cfg["model_name"] = model_name
    run_cfg["resolved_model_config"] = copy.deepcopy(model_cfg)
    if fold_prior_summary is not None:
        run_cfg["fold_local_evidence_prior"] = fold_prior_summary
    run_cfg["dataset_name"] = str(train.get("dataset_name", cfg.get("dataset_name", "")))
    run_cfg["subject"] = str(
        cfg.get("subject")
        or _single_metadata_value(val, "subject")
        or _single_metadata_value(train, "subject")
    )
    return train_model(model, train, val, run_cfg, output_dir, test_data=test)


def build_model_for_data(
    model_name: str,
    data: dict[str, Any],
    cfg: dict[str, Any],
    model_cfg: dict[str, Any] | None = None,
) -> Any:
    from dpc_snn.models import build_model

    model_cfg = model_cfg or resolve_model_cfg(cfg, model_name=model_name)
    model_cfg["n_channels"] = int(data["X"].shape[1])
    model_cfg["n_classes"] = (
        int(np.max(data["y"]) + 1) if "y" in data else int(model_cfg.get("n_classes", 2))
    )
    for metadata_key in ("electrode_coordinates", "spatial_anchor_indices"):
        if metadata_key in data:
            model_cfg[metadata_key] = np.asarray(data[metadata_key]).tolist()
    model_cfg["samples"] = int(data["X"].shape[-1])
    sfreq = float(data.get("sfreq", model_cfg.get("sfreq", 250.0)))
    model_cfg["sfreq"] = sfreq
    model_cfg["input_duration_seconds"] = model_cfg["samples"] / sfreq
    epoch_tmin = float(data.get("epoch_tmin", model_cfg.get("epoch_tmin", 0.0)))
    epoch_tmax = float(data.get("epoch_tmax", epoch_tmin + model_cfg["input_duration_seconds"]))
    model_cfg["epoch_tmin"] = epoch_tmin
    task_tmin = max(epoch_tmin, float(model_cfg.get("task_tmin", epoch_tmin)))
    task_tmax = min(epoch_tmax, float(model_cfg.get("task_tmax", epoch_tmax)))
    if task_tmax <= task_tmin:
        raise ValueError(
            f"Checkpoint evaluation task window {task_tmin}..{task_tmax} "
            f"is outside input epoch {epoch_tmin}..{epoch_tmax}"
        )
    model_cfg["task_tmin"] = task_tmin
    model_cfg["task_tmax"] = task_tmax
    if "graph_rate_hz" in model_cfg:
        graph_rate = float(model_cfg["graph_rate_hz"])
        model_cfg["graph_timesteps"] = max(1, int(round((task_tmax - task_tmin) * graph_rate)))
    if "amplitude" in data:
        model_cfg["n_bands"] = int(data["amplitude"].shape[1])
    return build_model(model_name, model_cfg)


def add_features_pair(
    first: dict[str, Any],
    second: dict[str, Any],
    bands: dict[str, list[float]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return add_features(first, bands), add_features(second, bands)


def write_run_manifest(
    output_dir: str | Path, cfg: dict[str, Any], status: str, extra: dict[str, Any] | None = None
) -> None:
    output_dir = ensure_dir(output_dir)
    payload = {
        "experiment_id": cfg.get("experiment_id"),
        "experiment": cfg.get("experiment", {}),
        "config_path": cfg.get("config_path", ""),
        "effective_config": cfg,
        "status": status,
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": dependency_snapshot(),
    }
    if extra:
        payload.update(extra)
    write_json(Path(output_dir) / "run_manifest.json", payload)


def dependency_snapshot() -> dict[str, Any]:
    mods = ["numpy", "pandas", "yaml", "torch", "scipy", "sklearn", "matplotlib", "mne", "moabb"]
    snap = {}
    for mod in mods:
        try:
            version = metadata.version(
                "pyyaml" if mod == "yaml" else "scikit-learn" if mod == "sklearn" else mod
            )
        except metadata.PackageNotFoundError:
            version = ""
        if not has_module(mod):
            snap[mod] = {"installed": False, "version": version}
            continue
        snap[mod] = {"installed": True, "version": version}
    return snap


def needs_input(output_dir: str | Path, cfg: dict[str, Any], message: str) -> dict[str, Any]:
    ensure_dir(output_dir)
    payload = {"status": "needs_input", "message": message}
    write_json(Path(output_dir) / "runner_status.json", payload)
    write_run_manifest(output_dir, cfg, status="needs_input", extra=payload)
    return payload
