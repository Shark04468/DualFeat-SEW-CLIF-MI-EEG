#!/usr/bin/env python3
"""Audit classifier-free V7 delay evidence on Session-T outer-train folds."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.analysis.evidence_space import fold_local_continuous_delay_prior  # noqa: E402
from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    session_t_run_grouped_folds,
)
from dpc_snn.experiments.v62_scaffold import (  # noqa: E402
    aligned_slow_rate_analytic_evidence,
    build_frontend_only_scaffold,
    cache_analytic_fixed_channel_gain_rate_features,
    fit_official_fbc_channel_gain,
)
from dpc_snn.utils.io import ensure_dir, save_npz, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402


def _subject_file(data_root: Path, subject: int) -> Path:
    candidates = sorted(data_root.glob(f"*A{subject:02d}*.npz"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one processed file for subject {subject}")
    return candidates[0]


def _fold_data(data: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    selected = {
        "X": np.asarray(data["X"])[indices],
        "y": np.asarray(data["y"])[indices],
        "sfreq": float(data["sfreq"]),
        "epoch_tmin": float(data["epoch_tmin"]),
    }
    for key in ("electrode_coordinates", "spatial_anchor_indices"):
        if key in data:
            selected[key] = data[key]
    return selected


def _audit_scopes(
    train_indices: np.ndarray,
    labels: np.ndarray,
    *,
    scope_mode: str,
    class_labels: list[int],
) -> list[tuple[str, np.ndarray]]:
    available = {int(value) for value in np.unique(labels[train_indices])}
    if scope_mode == "pooled":
        if class_labels:
            raise ValueError("--class-labels is only valid with --scope-mode classes")
        return [("pooled", train_indices)]
    if scope_mode == "all":
        if class_labels:
            raise ValueError("--class-labels is only valid with --scope-mode classes")
        selected_labels = sorted(available)
        scopes: list[tuple[str, np.ndarray]] = [("pooled", train_indices)]
    elif scope_mode == "classes":
        if not class_labels:
            raise ValueError("--scope-mode classes requires --class-labels")
        selected_labels = sorted(set(class_labels))
        missing = set(selected_labels) - available
        if missing:
            raise ValueError(f"requested class labels are absent from the fold: {sorted(missing)}")
        scopes = []
    else:
        raise ValueError(f"unsupported scope mode: {scope_mode!r}")
    for label in selected_labels:
        class_indices = train_indices[labels[train_indices] == label]
        scopes.append((f"class_{label}", class_indices))
    return scopes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--folds", default="0")
    parser.add_argument("--bootstrap-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--scope-mode", choices=("all", "pooled", "classes"), default="all"
    )
    parser.add_argument(
        "--class-labels",
        default="",
        help="comma-separated labels; required only when --scope-mode classes",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--config",
        default="configs/experiments/v7_e2_zero_scaffold_core_atc.yaml",
    )
    args = parser.parse_args()

    class_labels = [
        int(value.strip()) for value in args.class_labels.split(",") if value.strip()
    ]
    if args.scope_mode == "classes" and not class_labels:
        parser.error("--scope-mode classes requires --class-labels")
    if args.scope_mode != "classes" and class_labels:
        parser.error("--class-labels is only valid with --scope-mode classes")

    configure_cache_env()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = config["model"]
    data_root = Path(args.data).resolve()
    subject_path = _subject_file(data_root, args.subject)
    data = load_processed_npz(subject_path)
    sessions = np.asarray(data["session"]).astype(str)
    t_indices = np.flatnonzero(sessions == "T")
    x_t = np.asarray(data["X"])[t_indices]
    y_t = np.asarray(data["y"], dtype=np.int64)[t_indices]
    metadata_t = [
        {
            "dataset": str(data.get("dataset_name", "bci2a")),
            "trial_id": str(np.asarray(data["trial_id"])[index]),
            "subject": int(np.asarray(data["subject"])[index]),
            "session": str(np.asarray(data["session"])[index]),
            "run": int(np.asarray(data["run"])[index]),
            "class": int(np.asarray(data["y"])[index]),
            "sfreq": float(data["sfreq"]),
            "ch_names": data["ch_names"],
            "epoch_tmin": float(data["epoch_tmin"]),
            "epoch_tmax": float(data["epoch_tmax"]),
        }
        for index in t_indices
    ]
    folds = session_t_run_grouped_folds(
        metadata_t,
        n_splits=6,
        seed=0,
        shuffle=True,
    )
    selected_folds = [int(value) for value in args.folds.split(",") if value]
    output = ensure_dir(Path(args.output).resolve())
    rows = []
    for fold in selected_folds:
        train_indices, validation_indices = folds[fold]
        gain = fit_official_fbc_channel_gain(
            x_t[train_indices],
            sfreq=float(data["sfreq"]),
            epoch_tmin=float(data["epoch_tmin"]),
            clip=float(config["preprocessing"]["clip_after_gain"]),
        )
        frontend = build_frontend_only_scaffold(
            seed=int(config["selection_seed"]),
            model_config=dict(model),
        )
        cached = cache_analytic_fixed_channel_gain_rate_features(
            frontend,
            x_t[train_indices],
            gain,
            device=args.device,
            batch_size=max(1, int(args.batch_size)),
        )
        fold_analytic = (
            aligned_slow_rate_analytic_evidence(cached).cpu().numpy().astype(np.complex64)
        )
        if fold_analytic.shape[-1] != 500:
            raise RuntimeError("V7 evidence audit requires the registered 500-step slow grid")
        frontend_fingerprint = cached.frontend_fingerprint
        del cached, frontend
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        scopes = _audit_scopes(
            train_indices,
            y_t,
            scope_mode=args.scope_mode,
            class_labels=class_labels,
        )
        for scope, indices in scopes:
            scope_dir = ensure_dir(output / f"fold_{fold}" / scope)
            fold_positions = np.flatnonzero(np.isin(train_indices, indices))
            scope_analytic = fold_analytic[fold_positions]
            arrays, summary = fold_local_continuous_delay_prior(
                _fold_data(
                    {
                        **data,
                        "X": x_t,
                        "y": y_t,
                    },
                    indices,
                ),
                evidence_band_edges_hz=model["band_edges_hz"],
                output_band_edges_hz=model["band_edges_hz"],
                n_nodes=int(model["n_nodes"]),
                graph_steps=500,
                max_delay_steps=int(model["slow_max_delay"]),
                bootstrap_samples=int(args.bootstrap_samples),
                grid_oversample=4,
                min_bayes_factor=3.0,
                min_bootstrap_frequency=0.70,
                min_direction_probability=0.80,
                seed=(
                    fold * 1009
                    + (0 if scope == "pooled" else int(scope.split("_", 1)[1]) + 1)
                ),
                device=args.device,
                task_tmin=float(model["task_tmin"]),
                task_tmax=float(model["task_tmax"]),
                analytic_features=scope_analytic,
                analytic_representation=(
                    "v7_online_car_baseline_mean_fold_gain_causal_filterbank_"
                    "exact_sensor_fast_aligned_to_slow"
                ),
            )
            row = {
                "subject": int(args.subject),
                "fold": fold,
                "scope": scope,
                "train_trials": int(len(indices)),
                "outer_validation_trials": int(len(validation_indices)),
                "session_e_accessed": False,
                "frontend_fingerprint": frontend_fingerprint,
                "fold_gain_clip": float(gain.clip),
                **summary,
            }
            save_npz(scope_dir / "fold_local_evidence_prior.npz", **arrays)
            write_json(scope_dir / "fold_local_evidence_prior.json", row)
            rows.append(row)
            write_csv(output / "evidence_summary.csv", rows)
            print(
                "__V7_EVIDENCE_DONE__ "
                f"subject={args.subject} fold={fold} scope={scope} "
                f"edges={summary['accepted_evidence_edges']} "
                f"node_edges={summary['accepted_node_edges']}",
                flush=True,
            )
    write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "experiment_tier": (
                "main_evidence"
                if int(args.bootstrap_samples) >= 128
                else "engineering_smoke"
            ),
            "scientific_gate_eligible": int(args.bootstrap_samples) >= 128,
            "bootstrap_samples": int(args.bootstrap_samples),
            "scope_mode": args.scope_mode,
            "class_labels": class_labels,
            "subject": int(args.subject),
            "folds": selected_folds,
            "data": str(subject_path),
            "data_sha256": file_sha256(subject_path),
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "script_sha256": file_sha256(Path(__file__)),
            "frontend_builder": "build_frontend_only_scaffold",
            "session_e_accessed": False,
            "rows": len(rows),
        },
    )


if __name__ == "__main__":
    main()
