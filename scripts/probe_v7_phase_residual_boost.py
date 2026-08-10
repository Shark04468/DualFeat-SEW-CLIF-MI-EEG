#!/usr/bin/env python3
"""Test a zero-safe functional-gradient phase-delay expert on one outer fold."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dpc_snn.analysis.delay_pair_features import (  # noqa: E402
    accepted_within_band_routes,
    fixed_feature_scales,
)
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
    predict_scaffold,
)
from dpc_snn.experiments.v7_delay import build_delay_stage_model, state_sha256  # noqa: E402
from dpc_snn.utils.io import ensure_dir, save_npz, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from probe_v7_phase_pair_contrast import (  # noqa: E402
    _feature_bank,
    _metadata_rows,
    _scale_without_centering,
    _subject_file,
)
from run_v7_e3_delay import (  # noqa: E402
    _apply_registered_model_overrides,
    _load_audited_fold_prior,
    _parent_checkpoint,
    _validate_route_only_readout_config,
    _validate_route_only_readout_model,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def _cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    probability = _softmax(logits)
    return float(-np.log(probability[np.arange(len(labels)), labels].clip(1e-12)).mean())


def _functional_gradient(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    target = np.eye(logits.shape[1], dtype=np.float64)[labels]
    return target - _softmax(logits)


def _ridge_matrix(
    train_x: np.ndarray,
    train_target: np.ndarray,
    evaluate_x: np.ndarray,
    regularization: float,
) -> np.ndarray:
    train_scaled, evaluate_scaled, _ = _scale_without_centering(train_x, evaluate_x)
    kernel = train_scaled @ train_scaled.T
    alpha = np.linalg.solve(
        kernel + float(regularization) * np.eye(len(kernel)),
        train_target,
    )
    return evaluate_scaled @ train_scaled.T @ alpha


def _select_boost(
    features: np.ndarray,
    labels: np.ndarray,
    base_logits: np.ndarray,
    groups: np.ndarray,
    regularization_grid: list[float],
    step_grid: list[float],
) -> tuple[float, float, list[dict[str, float]]]:
    rows = []
    unique_groups = sorted(np.unique(groups).tolist())
    for regularization in regularization_grid:
        correction = np.zeros_like(base_logits, dtype=np.float64)
        for group in unique_groups:
            validation = groups == group
            training = ~validation
            correction[validation] = _ridge_matrix(
                features[training],
                _functional_gradient(base_logits[training], labels[training]),
                features[validation],
                regularization,
            )
        for step in step_grid:
            fused = base_logits + float(step) * correction
            rows.append(
                {
                    "regularization": float(regularization),
                    "step": float(step),
                    "inner_accuracy": float((fused.argmax(axis=1) == labels).mean()),
                    "inner_cross_entropy": _cross_entropy(fused, labels),
                }
            )
    selected = max(
        rows,
        key=lambda row: (
            row["inner_accuracy"],
            -row["inner_cross_entropy"],
            row["regularization"],
            -row["step"],
        ),
    )
    return selected["regularization"], selected["step"], rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-output", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--evidence-prior-root", required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--regularization-grid",
        default="0.001,0.01,0.1,1,10,100,1000",
    )
    parser.add_argument("--step-grid", default="0.125,0.25,0.5,1,2,4,8,16,32")
    parser.add_argument(
        "--config",
        default="configs/experiments/v7_e3_static_slow_within.yaml",
    )
    args = parser.parse_args()
    configure_cache_env()

    output = ensure_dir(Path(args.output).resolve())
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    parent_config_path = (ROOT / config["parent_config"]).resolve()
    parent_config = yaml.safe_load(parent_config_path.read_text(encoding="utf-8"))
    model_config = dict(parent_config["model"])
    _apply_registered_model_overrides(
        model_config,
        dict(config.get("model_overrides", {})),
    )
    _validate_route_only_readout_config(model_config)

    data_root = Path(args.data).resolve()
    subject_path = _subject_file(data_root, int(args.subject))
    data = load_processed_npz(subject_path)
    sessions = np.asarray(data["session"]).astype(str)
    t_indices = np.flatnonzero(sessions == "T")
    metadata_t = _metadata_rows(data, t_indices)
    folds = session_t_run_grouped_folds(metadata_t, n_splits=6, seed=0, shuffle=True)
    train_indices, validation_indices = folds[int(args.fold)]
    x_t = np.asarray(data["X"])[t_indices]
    y_t = np.asarray(data["y"], dtype=np.int64)[t_indices]
    run_t = np.asarray(data["run"], dtype=np.int64)[t_indices]

    gain = fit_official_fbc_channel_gain(
        x_t[train_indices],
        sfreq=float(data["sfreq"]),
        epoch_tmin=float(data["epoch_tmin"]),
        clip=float(parent_config["preprocessing"]["clip_after_gain"]),
    )
    frontend = build_frontend_only_scaffold(
        seed=int(config["selection_seed"]),
        model_config=model_config,
    )
    train_rates = cache_analytic_fixed_channel_gain_rate_features(
        frontend,
        x_t[train_indices],
        gain,
        device=args.device,
        batch_size=int(args.batch_size),
    )
    validation_rates = cache_analytic_fixed_channel_gain_rate_features(
        frontend,
        x_t[validation_indices],
        gain,
        device=args.device,
        batch_size=int(args.batch_size),
    )
    del frontend

    prior_arrays, prior_provenance = _load_audited_fold_prior(
        Path(args.evidence_prior_root).resolve(),
        subject=int(args.subject),
        fold=int(args.fold),
        scope="pooled",
    )
    if prior_provenance["frontend_fingerprint"] != train_rates.frontend_fingerprint:
        raise RuntimeError("prior and online feature frontends do not match")
    routes = accepted_within_band_routes(
        torch.from_numpy(prior_arrays["route_probability"]),
        torch.from_numpy(prior_arrays["positive_delay_probability"]),
        torch.from_numpy(prior_arrays["fractional_delay_target"]),
    )
    analytic_scale, envelope_scale = fixed_feature_scales(
        aligned_slow_rate_analytic_evidence(train_rates),
        train_rates.slow,
    )
    train_features = _feature_bank(
        train_rates,
        routes,
        analytic_scale=analytic_scale,
        envelope_scale=envelope_scale,
        bins=int(args.bins),
        batch_size=int(args.batch_size),
        device=args.device,
    )["phase_pair"]
    validation_features = _feature_bank(
        validation_rates,
        routes,
        analytic_scale=analytic_scale,
        envelope_scale=envelope_scale,
        bins=int(args.bins),
        batch_size=int(args.batch_size),
        device=args.device,
    )["phase_pair"]

    parent_checkpoint = _parent_checkpoint(
        Path(args.parent_output).resolve(),
        subject=int(args.subject),
        seed=int(args.seed),
        fold=int(args.fold),
    )
    model = build_delay_stage_model(
        seed=int(args.seed),
        model_config=model_config,
        parent_checkpoint=parent_checkpoint,
        stage=config["stage"],
        train_readout=False,
        readout_training_scope="fixed_delay_head",
        initial_atc_residual_scale=0.0,
        slow_carrier_residual_scale=0.0,
        delayed_statistics_interaction_gain=0.0,
        official_source_root=str(Path(args.source_root).resolve()),
    )
    _validate_route_only_readout_model(model)
    model.load_fold_local_slow_prior(
        torch.from_numpy(prior_arrays["route_probability"]),
        torch.from_numpy(prior_arrays["positive_delay_probability"]),
        torch.from_numpy(prior_arrays["fractional_delay_target"]),
    )
    model.set_training_gain(train_rates.gain)
    model.to(args.device)
    state_before = state_sha256(model)
    train_base = predict_scaffold(
        model,
        train_rates,
        y_t[train_indices],
        device=args.device,
        batch_size=int(args.batch_size),
        delay_override="zero",
    )
    validation_base = predict_scaffold(
        model,
        validation_rates,
        y_t[validation_indices],
        device=args.device,
        batch_size=int(args.batch_size),
        delay_override="zero",
    )
    if state_before != state_sha256(model):
        raise RuntimeError("base-logit extraction mutated the frozen checkpoint")

    regularization_grid = [
        float(value.strip())
        for value in args.regularization_grid.split(",")
        if value.strip()
    ]
    step_grid = [
        float(value.strip()) for value in args.step_grid.split(",") if value.strip()
    ]
    if not regularization_grid or not step_grid or min(regularization_grid + step_grid) <= 0:
        parser.error("ridge regularization and boost steps must be positive")
    selected_regularization, selected_step, tuning = _select_boost(
        train_features,
        y_t[train_indices],
        train_base["logits"],
        run_t[train_indices],
        regularization_grid,
        step_grid,
    )
    correction = _ridge_matrix(
        train_features,
        _functional_gradient(train_base["logits"], y_t[train_indices]),
        validation_features,
        selected_regularization,
    )
    zero_correction = _ridge_matrix(
        train_features,
        _functional_gradient(train_base["logits"], y_t[train_indices]),
        np.zeros_like(validation_features),
        selected_regularization,
    )
    if float(np.abs(zero_correction).max()) > 1e-12:
        raise RuntimeError("functional-gradient expert violates matched-zero semantics")
    full_logits = validation_base["logits"] + selected_step * correction
    full_prediction = full_logits.argmax(axis=1)
    zero_prediction = validation_base["logits"].argmax(axis=1)
    labels = y_t[validation_indices]
    corrections = int(((zero_prediction != labels) & (full_prediction == labels)).sum())
    regressions = int(((zero_prediction == labels) & (full_prediction != labels)).sum())
    full_accuracy = float((full_prediction == labels).mean())
    zero_accuracy = float((zero_prediction == labels).mean())
    result = {
        "status": "completed",
        "experiment": "v7_r8_zero_safe_phase_functional_gradient_probe",
        "subject": int(args.subject),
        "seed": int(args.seed),
        "fold": int(args.fold),
        "selection_session": "T",
        "session_e_accessed": False,
        "selected_regularization": selected_regularization,
        "selected_step": selected_step,
        "base_train_accuracy": float(
            (train_base["pred"] == y_t[train_indices]).mean()
        ),
        "full_accuracy": full_accuracy,
        "locked_zero_accuracy": zero_accuracy,
        "delta_accuracy": full_accuracy - zero_accuracy,
        "corrections": corrections,
        "regressions": regressions,
        "full_cross_entropy": _cross_entropy(full_logits, labels),
        "locked_zero_cross_entropy": _cross_entropy(validation_base["logits"], labels),
        "matched_zero_score_max_abs": float(np.abs(zero_correction).max()),
        "accepted_within_band_routes": routes.n_routes,
        "feature_count": int(train_features.shape[1]),
        "checkpoint_state_unchanged": True,
        "prior": prior_provenance,
        "frontend_fingerprint": train_rates.frontend_fingerprint,
        "parent_checkpoint": str(parent_checkpoint),
        "parent_checkpoint_sha256": file_sha256(parent_checkpoint),
        "gate": {
            "passed": bool(full_accuracy - zero_accuracy >= 0.02),
            "minimum_delta_accuracy": 0.02,
        },
    }
    write_csv(output / "boost_tuning.csv", tuning)
    write_json(output / "result.json", result)
    save_npz(
        output / "paired_validation_predictions.npz",
        indices=np.asarray(validation_indices, dtype=np.int64),
        labels=labels.astype(np.int64),
        full_logits=full_logits.astype(np.float32),
        locked_zero_logits=validation_base["logits"].astype(np.float32),
        phase_correction=correction.astype(np.float32),
    )
    print(
        "__V7_PHASE_BOOST__ "
        f"fold={args.fold} full={full_accuracy:.6f} zero={zero_accuracy:.6f} "
        f"corrections={corrections} regressions={regressions} "
        f"passed={str(result['gate']['passed']).lower()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
