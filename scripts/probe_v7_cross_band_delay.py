#!/usr/bin/env python3
"""Probe audited cross-band delayed envelope and PAC evidence."""

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
    CrossBandRouteBank,
    WithinBandRouteBank,
    accepted_cross_band_routes,
    accepted_within_band_routes,
    complex_phase_pair_contrasts,
    cross_band_delay_contrasts,
    fixed_feature_scales,
    pooled_route_features,
)
from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v62_protocol import session_t_run_grouped_folds  # noqa: E402
from dpc_snn.experiments.v62_scaffold import (  # noqa: E402
    CachedRates,
    aligned_slow_rate_analytic_evidence,
    build_frontend_only_scaffold,
    cache_analytic_fixed_channel_gain_rate_features,
    fit_official_fbc_channel_gain,
)
from dpc_snn.utils.io import ensure_dir, save_npz, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from probe_v7_phase_aggregation_loss import _target_aggregate  # noqa: E402
from probe_v7_phase_pair_contrast import (  # noqa: E402
    _auc,
    _metadata_rows,
    _ridge_multiclass_scores,
    _ridge_scores,
    _scale_without_centering,
    _select_multiclass_regularization,
    _select_regularization,
    _subject_file,
)
from run_v7_e3_delay import _load_audited_fold_prior  # noqa: E402


def _aggregate_cross_band(
    contrast: torch.Tensor,
    routes: CrossBandRouteBank,
    *,
    n_bands: int,
    n_nodes: int,
) -> torch.Tensor:
    target = routes.target_band * int(n_nodes) + routes.target_node
    aggregate = contrast.new_zeros(
        contrast.shape[0], int(n_bands) * int(n_nodes), contrast.shape[-1]
    )
    aggregate.index_add_(1, target.to(contrast.device), contrast)
    degree_square = contrast.new_zeros(int(n_bands) * int(n_nodes))
    degree_square.index_add_(
        0,
        target.to(contrast.device),
        routes.weight.to(contrast).square(),
    )
    return aggregate / degree_square.sqrt().clamp_min(1.0)[None, :, None]


@torch.no_grad()
def _feature_banks(
    rates: CachedRates,
    within_routes: WithinBandRouteBank,
    cross_routes: CrossBandRouteBank,
    *,
    analytic_scale: torch.Tensor,
    envelope_scale: torch.Tensor,
    bins: int,
    batch_size: int,
    device: str,
) -> dict[str, np.ndarray]:
    analytic = aligned_slow_rate_analytic_evidence(rates)
    parts: dict[str, list[np.ndarray]] = {
        "within_real_phase": [],
        "cross_envelope": [],
        "cross_pac": [],
        "cross_combined": [],
    }

    def pooled(value: torch.Tensor) -> np.ndarray:
        return (
            pooled_route_features(value, bins=int(bins))
            .flatten(1)
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    for start in range(0, len(analytic), int(batch_size)):
        stop = min(start + int(batch_size), len(analytic))
        batch_analytic = analytic[start:stop].to(device)
        batch_envelope = rates.slow[start:stop].to(device)
        within = complex_phase_pair_contrasts(
            batch_analytic,
            batch_envelope,
            within_routes,
            analytic_scale=analytic_scale.to(device),
        )["real_phase_pair"]
        within_aggregate = _target_aggregate(
            within,
            within_routes,
            n_bands=analytic.shape[1],
            n_nodes=analytic.shape[2],
        )
        cross = cross_band_delay_contrasts(
            batch_analytic,
            batch_envelope,
            cross_routes,
            analytic_scale=analytic_scale.to(device),
            envelope_scale=envelope_scale.to(device),
        )
        envelope_aggregate = _aggregate_cross_band(
            cross["cross_band_envelope_pair"],
            cross_routes,
            n_bands=analytic.shape[1],
            n_nodes=analytic.shape[2],
        )
        pac_real = _aggregate_cross_band(
            cross["cross_band_pac_real"],
            cross_routes,
            n_bands=analytic.shape[1],
            n_nodes=analytic.shape[2],
        )
        pac_imaginary = _aggregate_cross_band(
            cross["cross_band_pac_imaginary"],
            cross_routes,
            n_bands=analytic.shape[1],
            n_nodes=analytic.shape[2],
        )
        within_feature = pooled(within_aggregate)
        envelope_feature = pooled(envelope_aggregate)
        pac_feature = np.concatenate((pooled(pac_real), pooled(pac_imaginary)), axis=1)
        parts["within_real_phase"].append(within_feature)
        parts["cross_envelope"].append(envelope_feature)
        parts["cross_pac"].append(pac_feature)
        parts["cross_combined"].append(
            np.concatenate((envelope_feature, pac_feature), axis=1)
        )
    return {name: np.concatenate(values) for name, values in parts.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-output", required=True)
    parser.add_argument("--evidence-prior-root", required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--class-label", type=int, default=3)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
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
    configure_cache_env()

    subject_path = _subject_file(Path(args.data).resolve(), int(args.subject))
    data = load_processed_npz(subject_path)
    session = np.asarray(data["session"]).astype(str)
    t_indices = np.flatnonzero(session == "T")
    metadata = _metadata_rows(data, t_indices)
    folds = session_t_run_grouped_folds(metadata, n_splits=6, seed=0, shuffle=True)
    train_indices, validation_indices = folds[int(args.fold)]
    x_t = np.asarray(data["X"])[t_indices]
    y_t = np.asarray(data["y"], dtype=np.int64)[t_indices]
    run_t = np.asarray(data["run"], dtype=np.int64)[t_indices]

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gain = fit_official_fbc_channel_gain(
        x_t[train_indices],
        sfreq=float(data["sfreq"]),
        epoch_tmin=float(data["epoch_tmin"]),
        clip=float(config["preprocessing"]["clip_after_gain"]),
    )
    frontend = build_frontend_only_scaffold(
        seed=int(config["selection_seed"]),
        model_config=dict(config["model"]),
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

    prior, provenance = _load_audited_fold_prior(
        Path(args.evidence_prior_root).resolve(),
        subject=int(args.subject),
        fold=int(args.fold),
        scope="pooled",
    )
    if provenance["frontend_fingerprint"] != train_rates.frontend_fingerprint:
        raise RuntimeError("prior and online feature frontends do not match")
    route = torch.from_numpy(prior["route_probability"])
    posterior = torch.from_numpy(prior["positive_delay_probability"])
    fraction = torch.from_numpy(prior["fractional_delay_target"])
    within_routes = accepted_within_band_routes(route, posterior, fraction)
    cross_routes = accepted_cross_band_routes(route, posterior, fraction)
    analytic_scale, envelope_scale = fixed_feature_scales(
        aligned_slow_rate_analytic_evidence(train_rates),
        train_rates.slow,
    )
    train = _feature_banks(
        train_rates,
        within_routes,
        cross_routes,
        analytic_scale=analytic_scale,
        envelope_scale=envelope_scale,
        bins=int(args.bins),
        batch_size=int(args.batch_size),
        device=args.device,
    )
    validation = _feature_banks(
        validation_rates,
        within_routes,
        cross_routes,
        analytic_scale=analytic_scale,
        envelope_scale=envelope_scale,
        bins=int(args.bins),
        batch_size=int(args.batch_size),
        device=args.device,
    )

    prediction_path = (
        Path(args.parent_output).resolve()
        / f"subject_{int(args.subject):02d}"
        / "selection"
        / f"fold_{int(args.fold)}"
        / "validation_predictions.npz"
    )
    with np.load(prediction_path, allow_pickle=False) as archive:
        parent_indices = archive["indices"].astype(np.int64)
        parent_logits = archive["logits"].astype(np.float64)
        parent_labels = archive["labels"].astype(np.int64)
    if not np.array_equal(parent_indices, np.asarray(validation_indices)):
        raise RuntimeError("parent predictions do not match the registered fold")
    if not np.array_equal(parent_labels, y_t[validation_indices]):
        raise RuntimeError("parent labels do not match the registered fold")
    parent_correct = parent_logits.argmax(1) == parent_labels

    candidates = [float(value) for value in args.regularization_grid.split(",")]
    train_binary = y_t[train_indices] == int(args.class_label)
    validation_binary = y_t[validation_indices] == int(args.class_label)
    n_classes = int(y_t.max()) + 1
    rows: list[dict[str, Any]] = []
    score_arrays: dict[str, np.ndarray] = {}
    for name in (
        "within_real_phase",
        "cross_envelope",
        "cross_pac",
        "cross_combined",
    ):
        binary_regularization, inner_auc, _ = _select_regularization(
            train[name], train_binary, run_t[train_indices], candidates
        )
        binary_score = _ridge_scores(
            train[name], train_binary, validation[name], binary_regularization
        )
        multiclass_regularization, inner_accuracy, _ = (
            _select_multiclass_regularization(
                train[name],
                y_t[train_indices],
                run_t[train_indices],
                candidates,
                n_classes=n_classes,
            )
        )
        multiclass_score = _ridge_multiclass_scores(
            train[name],
            y_t[train_indices],
            validation[name],
            multiclass_regularization,
            n_classes=n_classes,
        )
        prediction = multiclass_score.argmax(1)
        correct = prediction == parent_labels
        _, _, retained = _scale_without_centering(train[name], validation[name])
        rows.append(
            {
                "feature": name,
                "raw_features": int(train[name].shape[1]),
                "retained_features": retained,
                "binary_regularization": binary_regularization,
                "inner_class3_auc": inner_auc,
                "outer_class3_auc": _auc(validation_binary, binary_score),
                "multiclass_regularization": multiclass_regularization,
                "inner_multiclass_accuracy": inner_accuracy,
                "outer_multiclass_accuracy": float(correct.mean()),
                "correct_on_parent_errors": int(
                    np.count_nonzero(correct & ~parent_correct)
                ),
                "wrong_on_parent_correct": int(
                    np.count_nonzero(~correct & parent_correct)
                ),
            }
        )
        score_arrays[f"{name}_multiclass_score"] = multiclass_score.astype(
            np.float32
        )

    by_name = {row["feature"]: row for row in rows}
    within = by_name["within_real_phase"]
    selected = max(
        (by_name["cross_envelope"], by_name["cross_pac"], by_name["cross_combined"]),
        key=lambda row: (
            row["outer_multiclass_accuracy"],
            row["correct_on_parent_errors"],
            row["outer_class3_auc"],
        ),
    )
    accuracy_gain = float(
        selected["outer_multiclass_accuracy"] - within["outer_multiclass_accuracy"]
    )
    correction_gain = int(
        selected["correct_on_parent_errors"] - within["correct_on_parent_errors"]
    )
    supports_cross_band = bool(
        accuracy_gain >= 0.05
        or (accuracy_gain >= -0.02 and correction_gain >= 2)
    )
    report = {
        "status": "completed",
        "experiment": "v7_r13_cross_band_delay_probe",
        "subject": int(args.subject),
        "fold": int(args.fold),
        "selection_session": "T",
        "session_e_accessed": False,
        "parent_accuracy": float(parent_correct.mean()),
        "accepted_within_band_routes": within_routes.n_routes,
        "accepted_cross_band_routes": cross_routes.n_routes,
        "rows": rows,
        "selected_cross_band_feature": selected["feature"],
        "selected_minus_within_multiclass_accuracy": accuracy_gain,
        "selected_minus_within_parent_error_corrections": correction_gain,
        "supports_cross_band_delay": supports_cross_band,
        "prior": provenance,
    }
    output = ensure_dir(Path(args.output).resolve())
    write_csv(output / "cross_band_metrics.csv", rows)
    save_npz(
        output / "validation_scores.npz",
        validation_indices=np.asarray(validation_indices, dtype=np.int64),
        validation_labels=parent_labels,
        parent_logits=parent_logits.astype(np.float32),
        **score_arrays,
    )
    write_json(output / "cross_band_probe.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
