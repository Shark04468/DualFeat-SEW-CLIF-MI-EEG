#!/usr/bin/env python3
"""Probe whether audited delay yields trial-level source-target evidence."""

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
sys.path.insert(0, str(ROOT / "scripts"))

from dpc_snn.analysis.delay_pair_features import (  # noqa: E402
    WithinBandRouteBank,
    accepted_within_band_routes,
    delay_pair_contrasts,
    fixed_feature_scales,
    pooled_route_features,
)
from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    session_t_run_grouped_folds,
)
from dpc_snn.experiments.v62_scaffold import (  # noqa: E402
    CachedRates,
    aligned_slow_rate_analytic_evidence,
    build_frontend_only_scaffold,
    cache_analytic_fixed_channel_gain_rate_features,
    fit_official_fbc_channel_gain,
)
from dpc_snn.utils.io import ensure_dir, save_npz, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from run_v7_e3_delay import _load_audited_fold_prior  # noqa: E402


def _subject_file(data_root: Path, subject: int) -> Path:
    candidates = sorted(data_root.glob(f"*A{subject:02d}*.npz"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one processed file for subject {subject}")
    return candidates[0]


def _metadata_rows(data: dict[str, Any], indices: np.ndarray) -> list[dict[str, Any]]:
    return [
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
        for index in indices
    ]


@torch.no_grad()
def _feature_bank(
    rates: CachedRates,
    routes: WithinBandRouteBank,
    *,
    analytic_scale: torch.Tensor,
    envelope_scale: torch.Tensor,
    bins: int,
    batch_size: int,
    device: str,
) -> dict[str, np.ndarray]:
    analytic = aligned_slow_rate_analytic_evidence(rates)
    envelope = rates.slow
    parts: dict[str, list[np.ndarray]] = {
        "source_only": [],
        "envelope_pair": [],
        "phase_pair": [],
    }
    for start in range(0, analytic.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), analytic.shape[0])
        contrasts = delay_pair_contrasts(
            analytic[start:stop].to(device),
            envelope[start:stop].to(device),
            routes,
            analytic_scale=analytic_scale.to(device),
            envelope_scale=envelope_scale.to(device),
        )
        for name, contrast in contrasts.items():
            pooled = pooled_route_features(contrast, bins=bins)
            parts[name].append(pooled.flatten(1).cpu().numpy().astype(np.float64))
    return {name: np.concatenate(values, axis=0) for name, values in parts.items()}


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("AUC requires both classes")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop + 1)
        start = stop
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _balanced_target(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("balanced target requires both classes")
    return np.where(labels, 1.0, -float(positives) / float(negatives))


def _scale_without_centering(
    train: np.ndarray,
    evaluate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    scale = np.sqrt(np.mean(np.square(train), axis=0))
    retained = scale > 1e-8
    if not bool(retained.any()):
        raise RuntimeError("the delay contrast has no non-constant training features")
    train_scaled = train[:, retained] / scale[retained]
    evaluate_scaled = evaluate[:, retained] / scale[retained]
    normalization = np.sqrt(float(retained.sum()))
    return train_scaled / normalization, evaluate_scaled / normalization, int(retained.sum())


def _ridge_scores(
    train_x: np.ndarray,
    train_y: np.ndarray,
    evaluate_x: np.ndarray,
    regularization: float,
) -> np.ndarray:
    train_scaled, evaluate_scaled, _ = _scale_without_centering(train_x, evaluate_x)
    kernel = train_scaled @ train_scaled.T
    alpha = np.linalg.solve(
        kernel + float(regularization) * np.eye(len(kernel)),
        _balanced_target(train_y),
    )
    return evaluate_scaled @ train_scaled.T @ alpha


def _multiclass_target(labels: np.ndarray, n_classes: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or labels.min() < 0 or labels.max() >= int(n_classes):
        raise ValueError("multiclass labels are outside the registered class range")
    target = np.eye(int(n_classes), dtype=np.float64)[labels]
    return target - 1.0 / float(n_classes)


def _ridge_multiclass_scores(
    train_x: np.ndarray,
    train_y: np.ndarray,
    evaluate_x: np.ndarray,
    regularization: float,
    *,
    n_classes: int,
) -> np.ndarray:
    train_scaled, evaluate_scaled, _ = _scale_without_centering(train_x, evaluate_x)
    kernel = train_scaled @ train_scaled.T
    alpha = np.linalg.solve(
        kernel + float(regularization) * np.eye(len(kernel)),
        _multiclass_target(train_y, n_classes),
    )
    return evaluate_scaled @ train_scaled.T @ alpha


def _select_regularization(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    candidates: list[float],
) -> tuple[float, float, list[dict[str, float]]]:
    rows = []
    unique_groups = sorted(np.unique(groups).tolist())
    for regularization in candidates:
        scores = np.zeros(len(labels), dtype=np.float64)
        covered = np.zeros(len(labels), dtype=bool)
        for group in unique_groups:
            validation = groups == group
            training = ~validation
            scores[validation] = _ridge_scores(
                features[training],
                labels[training],
                features[validation],
                regularization,
            )
            covered[validation] = True
        if not bool(covered.all()):
            raise RuntimeError("inner run-group probe did not cover every trial")
        rows.append(
            {
                "regularization": float(regularization),
                "inner_auc": _auc(labels, scores),
            }
        )
    selected = max(rows, key=lambda row: (row["inner_auc"], row["regularization"]))
    return selected["regularization"], selected["inner_auc"], rows


def _select_multiclass_regularization(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    candidates: list[float],
    *,
    n_classes: int,
) -> tuple[float, float, list[dict[str, float]]]:
    rows = []
    unique_groups = sorted(np.unique(groups).tolist())
    for regularization in candidates:
        scores = np.zeros((len(labels), int(n_classes)), dtype=np.float64)
        covered = np.zeros(len(labels), dtype=bool)
        for group in unique_groups:
            validation = groups == group
            training = ~validation
            scores[validation] = _ridge_multiclass_scores(
                features[training],
                labels[training],
                features[validation],
                regularization,
                n_classes=n_classes,
            )
            covered[validation] = True
        if not bool(covered.all()):
            raise RuntimeError("inner multiclass probe did not cover every trial")
        rows.append(
            {
                "regularization": float(regularization),
                "inner_accuracy": float((scores.argmax(axis=1) == labels).mean()),
            }
        )
    selected = max(
        rows,
        key=lambda row: (row["inner_accuracy"], row["regularization"]),
    )
    return selected["regularization"], selected["inner_accuracy"], rows


def _permutation_test(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    regularization: float,
    observed_auc: float,
    permutations: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    train_scaled, validation_scaled, _ = _scale_without_centering(
        train_x, validation_x
    )
    kernel = train_scaled @ train_scaled.T
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    rng = np.random.default_rng(seed)
    targets = np.column_stack(
        [_balanced_target(rng.permutation(train_y)) for _ in range(permutations)]
    )
    alpha = eigenvectors @ (
        (eigenvectors.T @ targets)
        / (eigenvalues[:, None] + float(regularization))
    )
    null_scores = validation_scaled @ train_scaled.T @ alpha
    null_auc = np.asarray(
        [_auc(validation_y, null_scores[:, index]) for index in range(permutations)]
    )
    p_value = float((1 + int((null_auc >= observed_auc).sum())) / (permutations + 1))
    return p_value, null_auc


def _multiclass_permutation_test(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    regularization: float,
    observed_accuracy: float,
    permutations: int,
    seed: int,
    n_classes: int,
) -> tuple[float, np.ndarray]:
    train_scaled, validation_scaled, _ = _scale_without_centering(
        train_x, validation_x
    )
    kernel = train_scaled @ train_scaled.T
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    rng = np.random.default_rng(seed)
    targets = np.concatenate(
        [
            _multiclass_target(rng.permutation(train_y), n_classes)
            for _ in range(permutations)
        ],
        axis=1,
    )
    alpha = eigenvectors @ (
        (eigenvectors.T @ targets)
        / (eigenvalues[:, None] + float(regularization))
    )
    scores = validation_scaled @ train_scaled.T @ alpha
    scores = scores.reshape(len(validation_y), permutations, int(n_classes))
    null_accuracy = (scores.argmax(axis=2) == validation_y[:, None]).mean(axis=0)
    p_value = float(
        (1 + int((null_accuracy >= observed_accuracy).sum())) / (permutations + 1)
    )
    return p_value, null_accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence-prior-root", required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--class-label", type=int, default=3)
    parser.add_argument(
        "--prior-scope",
        choices=("pooled", "class_0", "class_1", "class_2", "class_3"),
        default=None,
        help="audited prior scope; defaults to the target class",
    )
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--regularization-grid",
        default="0.0001,0.001,0.01,0.1,1,10,100,1000",
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/v7_e2_zero_scaffold_core_atc.yaml",
    )
    args = parser.parse_args()
    if args.permutations < 19:
        parser.error("--permutations must be at least 19")

    configure_cache_env()
    output = ensure_dir(Path(args.output).resolve())
    data_root = Path(args.data).resolve()
    subject_path = _subject_file(data_root, args.subject)
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = dict(config["model"])
    data = load_processed_npz(subject_path)
    sessions = np.asarray(data["session"]).astype(str)
    t_indices = np.flatnonzero(sessions == "T")
    metadata_t = _metadata_rows(data, t_indices)
    folds = session_t_run_grouped_folds(
        metadata_t,
        n_splits=6,
        seed=0,
        shuffle=True,
    )
    train_indices, validation_indices = folds[int(args.fold)]
    x_t = np.asarray(data["X"])[t_indices]
    y_t = np.asarray(data["y"], dtype=np.int64)[t_indices]
    run_t = np.asarray(data["run"], dtype=np.int64)[t_indices]

    gain = fit_official_fbc_channel_gain(
        x_t[train_indices],
        sfreq=float(data["sfreq"]),
        epoch_tmin=float(data["epoch_tmin"]),
        clip=float(config["preprocessing"]["clip_after_gain"]),
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
        batch_size=args.batch_size,
    )
    validation_rates = cache_analytic_fixed_channel_gain_rate_features(
        frontend,
        x_t[validation_indices],
        gain,
        device=args.device,
        batch_size=args.batch_size,
    )
    del frontend
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    scope = args.prior_scope or f"class_{int(args.class_label)}"
    prior_arrays, prior_provenance = _load_audited_fold_prior(
        Path(args.evidence_prior_root).resolve(),
        subject=int(args.subject),
        fold=int(args.fold),
        scope=scope,
    )
    if prior_provenance["frontend_fingerprint"] != train_rates.frontend_fingerprint:
        raise RuntimeError("prior and online feature frontends do not match")
    routes = accepted_within_band_routes(
        torch.from_numpy(prior_arrays["route_probability"]),
        torch.from_numpy(prior_arrays["positive_delay_probability"]),
        torch.from_numpy(prior_arrays["fractional_delay_target"]),
    )
    train_analytic = aligned_slow_rate_analytic_evidence(train_rates)
    analytic_scale, envelope_scale = fixed_feature_scales(
        train_analytic,
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
    )
    validation_features = _feature_bank(
        validation_rates,
        routes,
        analytic_scale=analytic_scale,
        envelope_scale=envelope_scale,
        bins=int(args.bins),
        batch_size=int(args.batch_size),
        device=args.device,
    )
    train_binary = y_t[train_indices] == int(args.class_label)
    validation_binary = y_t[validation_indices] == int(args.class_label)
    n_classes = int(np.max(y_t)) + 1
    candidates = [
        float(value.strip())
        for value in args.regularization_grid.split(",")
        if value.strip()
    ]
    if not candidates or any(value <= 0 for value in candidates):
        parser.error("regularization candidates must be positive")

    rows: list[dict[str, Any]] = []
    score_arrays: dict[str, np.ndarray] = {}
    tuning_rows: list[dict[str, Any]] = []
    for feature_index, name in enumerate(("source_only", "envelope_pair", "phase_pair")):
        selected, inner_auc, tuning = _select_regularization(
            train_features[name],
            train_binary,
            run_t[train_indices],
            candidates,
        )
        for row in tuning:
            tuning_rows.append({"feature": name, **row})
        scores = _ridge_scores(
            train_features[name],
            train_binary,
            validation_features[name],
            selected,
        )
        outer_auc = _auc(validation_binary, scores)
        prediction = scores > 0.0
        sensitivity = float(prediction[validation_binary].mean())
        specificity = float((~prediction[~validation_binary]).mean())
        permutation_p, null_auc = _permutation_test(
            train_features[name],
            train_binary,
            validation_features[name],
            validation_binary,
            regularization=selected,
            observed_auc=outer_auc,
            permutations=int(args.permutations),
            seed=int(args.seed) + feature_index * 1009,
        )
        multiclass_regularization, inner_multiclass_accuracy, multiclass_tuning = (
            _select_multiclass_regularization(
                train_features[name],
                y_t[train_indices],
                run_t[train_indices],
                candidates,
                n_classes=n_classes,
            )
        )
        for row in multiclass_tuning:
            tuning_rows.append(
                {"feature": name, "task": "multiclass", **row}
            )
        multiclass_scores = _ridge_multiclass_scores(
            train_features[name],
            y_t[train_indices],
            validation_features[name],
            multiclass_regularization,
            n_classes=n_classes,
        )
        outer_multiclass_accuracy = float(
            (multiclass_scores.argmax(axis=1) == y_t[validation_indices]).mean()
        )
        multiclass_permutation_p, null_multiclass_accuracy = (
            _multiclass_permutation_test(
                train_features[name],
                y_t[train_indices],
                validation_features[name],
                y_t[validation_indices],
                regularization=multiclass_regularization,
                observed_accuracy=outer_multiclass_accuracy,
                permutations=int(args.permutations),
                seed=int(args.seed) + 5003 + feature_index * 1009,
                n_classes=n_classes,
            )
        )
        _, _, retained = _scale_without_centering(
            train_features[name], validation_features[name]
        )
        matched_zero_score = _ridge_scores(
            train_features[name],
            train_binary,
            np.zeros_like(validation_features[name]),
            selected,
        )
        matched_zero_score_max_abs = float(np.abs(matched_zero_score).max())
        if matched_zero_score_max_abs > 1e-12:
            raise RuntimeError("zero-safe ridge probe produced a non-zero matched control")
        matched_zero_multiclass_score = _ridge_multiclass_scores(
            train_features[name],
            y_t[train_indices],
            np.zeros_like(validation_features[name]),
            multiclass_regularization,
            n_classes=n_classes,
        )
        matched_zero_multiclass_score_max_abs = float(
            np.abs(matched_zero_multiclass_score).max()
        )
        if matched_zero_multiclass_score_max_abs > 1e-12:
            raise RuntimeError(
                "zero-safe multiclass probe produced a non-zero matched control"
            )
        rows.append(
            {
                "feature": name,
                "selected_regularization": selected,
                "inner_run_group_auc": inner_auc,
                "outer_fold_auc": outer_auc,
                "outer_balanced_accuracy": 0.5 * (sensitivity + specificity),
                "outer_sensitivity": sensitivity,
                "outer_specificity": specificity,
                "permutation_p": permutation_p,
                "null_auc_95th_percentile": float(np.quantile(null_auc, 0.95)),
                "multiclass_selected_regularization": multiclass_regularization,
                "inner_run_group_multiclass_accuracy": inner_multiclass_accuracy,
                "outer_multiclass_accuracy": outer_multiclass_accuracy,
                "multiclass_permutation_p": multiclass_permutation_p,
                "null_multiclass_accuracy_95th_percentile": float(
                    np.quantile(null_multiclass_accuracy, 0.95)
                ),
                "raw_feature_count": int(train_features[name].shape[1]),
                "retained_feature_count": retained,
                "accepted_within_band_routes": routes.n_routes,
                "matched_zero_feature_max_abs": 0.0,
                "matched_zero_score_max_abs": matched_zero_score_max_abs,
                "matched_zero_multiclass_score_max_abs": (
                    matched_zero_multiclass_score_max_abs
                ),
            }
        )
        score_arrays[f"{name}_score"] = scores.astype(np.float32)
        score_arrays[f"{name}_null_auc"] = null_auc.astype(np.float32)
        score_arrays[f"{name}_multiclass_scores"] = multiclass_scores.astype(
            np.float32
        )
        score_arrays[f"{name}_null_multiclass_accuracy"] = (
            null_multiclass_accuracy.astype(np.float32)
        )
        print(
            "__V7_PHASE_PAIR_PROBE__ "
            f"feature={name} inner_auc={inner_auc:.6f} "
            f"outer_auc={outer_auc:.6f} p={permutation_p:.6f} "
            f"multiclass={outer_multiclass_accuracy:.6f} "
            f"multiclass_p={multiclass_permutation_p:.6f}",
            flush=True,
        )

    by_name = {row["feature"]: row for row in rows}
    phase = by_name["phase_pair"]
    source = by_name["source_only"]
    gate_passed = bool(
        phase["outer_fold_auc"] >= 0.65
        and phase["permutation_p"] < 0.05
        and phase["outer_fold_auc"] >= source["outer_fold_auc"] + 0.05
    )
    write_csv(output / "probe_metrics.csv", rows)
    write_csv(output / "regularization_tuning.csv", tuning_rows)
    save_npz(
        output / "validation_scores.npz",
        validation_indices=np.asarray(validation_indices, dtype=np.int64),
        validation_labels=y_t[validation_indices].astype(np.int64),
        **score_arrays,
    )
    write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "experiment": "v7_r8_delay_pair_feature_probe",
            "subject": int(args.subject),
            "fold": int(args.fold),
            "class_label": int(args.class_label),
            "prior_scope": scope,
            "selection_session": "T",
            "session_e_accessed": False,
            "train_trials": int(len(train_indices)),
            "outer_validation_trials": int(len(validation_indices)),
            "accepted_within_band_routes": routes.n_routes,
            "temporal_bins": int(args.bins),
            "feature_normalization": "outer_train_rms_scale_only_no_center_no_bias",
            "permutations": int(args.permutations),
            "gate": {
                "passed": gate_passed,
                "minimum_phase_pair_auc": 0.65,
                "maximum_permutation_p": 0.05,
                "minimum_auc_gain_over_source_only": 0.05,
            },
            "prior": prior_provenance,
            "frontend_fingerprint": train_rates.frontend_fingerprint,
            "data": str(subject_path),
            "data_sha256": file_sha256(subject_path),
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "script_sha256": file_sha256(Path(__file__)),
            "feature_module_sha256": file_sha256(
                ROOT / "src" / "dpc_snn" / "analysis" / "delay_pair_features.py"
            ),
            "metrics": rows,
        },
    )
    print(
        f"__V7_PHASE_PAIR_GATE__ passed={str(gate_passed).lower()} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
