#!/usr/bin/env python
"""Audit V4.2 delay evidence without training or accessing Session E."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.analysis.evidence_space import (  # noqa: E402
    fit_var_innovations,
    fold_local_continuous_delay_prior,
    fold_local_model_evidence_prior,
)
from dpc_snn.config import load_experiment_config, load_yaml  # noqa: E402
from dpc_snn.data.bci2a import load_processed_npz, subject_session_data  # noqa: E402
from dpc_snn.experiments.common import resolve_dataset_cfg, subset  # noqa: E402
from dpc_snn.experiments.runners import _crop_to_common_task_window  # noqa: E402
from dpc_snn.models.eeg_frontend import AnchoredSpatialProjection  # noqa: E402
from dpc_snn.preprocessing.standardize import (  # noqa: E402
    apply_channelwise_zscore,
    channelwise_train_stats,
)
from dpc_snn.utils.io import ensure_dir, write_csv, write_json  # noqa: E402
from run_v40_delay_gate import (  # noqa: E402
    _model_cfg,
    _reference_space,
    _repeated_stratified_folds,
    _source_fingerprint,
)


def _correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) < 3:
        return float("nan")
    x, y = a[mask], b[mask]
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _expected_delay(arrays: dict[str, np.ndarray]) -> np.ndarray:
    probability = arrays["positive_delay_probability"]
    grid = np.arange(probability.shape[-1], dtype=np.float32)
    return np.sum(probability * grid, axis=-1) + arrays["fractional_delay_target"]


def _fit_prior(
    data: dict[str, Any], model_cfg: dict[str, Any], seed: int, device: str, bootstraps: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    prior_builder = (
        fold_local_continuous_delay_prior
        if str(model_cfg.get("delay_evidence_estimator", "")) == "continuous_signed_phase"
        else fold_local_model_evidence_prior
    )
    extra = (
        {
            "grid_oversample": int(model_cfg.get("continuous_delay_grid_oversample", 4)),
            "min_bayes_factor": float(model_cfg.get("continuous_delay_min_bayes_factor", 3.0)),
            "min_bootstrap_frequency": float(
                model_cfg.get("continuous_delay_min_bootstrap_frequency", 0.70)
            ),
            "min_direction_probability": float(
                model_cfg.get("continuous_delay_min_direction_probability", 0.80)
            ),
        }
        if prior_builder is fold_local_continuous_delay_prior
        else {}
    )
    return prior_builder(
        data,
        evidence_band_edges_hz=model_cfg["delay_evidence_band_edges_hz"],
        output_band_edges_hz=model_cfg["band_edges_hz"],
        n_nodes=int(model_cfg["latent_nodes"]),
        graph_steps=int(model_cfg["graph_timesteps"]),
        max_delay_steps=int(model_cfg["d_max"]),
        bootstrap_samples=int(bootstraps),
        seed=int(seed),
        device=device,
        task_tmin=float(model_cfg["task_tmin"]),
        task_tmax=float(model_cfg["task_tmax"]),
        **extra,
    )


def _audit_space(
    name: str,
    data: dict[str, Any],
    model_cfg: dict[str, Any],
    seed: int,
    device: str,
    bootstraps: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays, summary = _fit_prior(data, model_cfg, seed, device, bootstraps)
    first = subset(data, np.arange(0, len(data["y"]), 2))
    second = subset(data, np.arange(1, len(data["y"]), 2))
    first_arrays, _ = _fit_prior(first, model_cfg, seed + 1, device, min(64, bootstraps))
    second_arrays, _ = _fit_prior(second, model_cfg, seed + 2, device, min(64, bootstraps))
    first_transport_active = first_arrays["route_probability"] > 0
    second_transport_active = second_arrays["route_probability"] > 0
    common_transport = first_transport_active & second_transport_active
    transport_split_correlation = _correlation(
        _expected_delay(first_arrays), _expected_delay(second_arrays), common_transport
    )
    if "signed_node_accepted" in first_arrays and "signed_node_accepted" in second_arrays:
        first_node_active = np.asarray(first_arrays["signed_node_accepted"], dtype=bool)
        second_node_active = np.asarray(second_arrays["signed_node_accepted"], dtype=bool)
        common_node = first_node_active & second_node_active
        split_correlation = _correlation(
            np.asarray(first_arrays["signed_node_delay_map"]),
            np.asarray(second_arrays["signed_node_delay_map"]),
            common_node,
        )
        split_common_edges = int(common_node.sum())
        split_scope = "signed_node_posterior_map"
    else:
        split_correlation = transport_split_correlation
        split_common_edges = int(common_transport.sum())
        split_scope = "legacy_transport_posterior_mean"
    checks = {
        "stable_nonzero_edges": int(summary["accepted_evidence_edges"]) > 0,
        "split_half_delay": bool(np.isfinite(split_correlation) and split_correlation >= 0.5),
        "time_reversal": float(summary["time_reversal_transpose_support_mae"]) <= 0.05,
        "phase_surrogate_support": float(summary["phase_surrogate_support_drop"]) > 0,
        "phase_surrogate_edges": int(summary["normal_node_edges"])
        > int(summary["phase_surrogate_node_edges"]),
    }
    report = {
        "evidence_space": name,
        "split_half_common_edges": split_common_edges,
        "split_half_delay_correlation": split_correlation,
        "split_half_delay_scope": split_scope,
        "split_half_common_node_edges": split_common_edges
        if split_scope == "signed_node_posterior_map"
        else 0,
        "split_half_common_transport_routes": int(common_transport.sum()),
        "split_half_transport_delay_correlation": transport_split_correlation,
        "checks": checks,
        "passed": bool(all(checks.values())),
        **summary,
    }
    return report, arrays


def _inject_task_delay(
    data: dict[str, Any],
    source: int,
    target: int,
    delay_graph_steps: float,
    strength: float,
    graph_steps: int,
    task_tmin: float,
    task_tmax: float,
) -> dict[str, Any]:
    injected = {**data, "X": np.asarray(data["X"], dtype=np.float32).copy()}
    sfreq = float(data["sfreq"])
    epoch_tmin = float(data["epoch_tmin"])
    start = int(round((task_tmin - epoch_tmin) * sfreq))
    stop = int(round((task_tmax - epoch_tmin) * sfreq))
    task = injected["X"][..., start:stop]
    delay_samples = delay_graph_steps * (task_tmax - task_tmin) * sfreq / graph_steps
    grid = np.arange(task.shape[-1], dtype=np.float32)
    shifted = np.stack(
        [
            np.interp(grid - delay_samples, grid, trial[source], left=0.0, right=0.0)
            for trial in task
        ]
    ).astype(np.float32)
    task[:, target] += float(strength) * shifted
    injected["X"][..., start:stop] = task
    return injected


def _inject_task_delay_latent(
    data: dict[str, Any],
    anchors: np.ndarray,
    source_node: int,
    target_node: int,
    delay_graph_steps: float,
    strength: float,
    graph_steps: int,
    task_tmin: float,
    task_tmax: float,
) -> dict[str, Any]:
    """Inject one exact fixed-projector node route into normalized EEG background."""

    injected = {**data, "X": np.asarray(data["X"], dtype=np.float32).copy()}
    sfreq = float(data["sfreq"])
    epoch_tmin = float(data["epoch_tmin"])
    start = int(round((task_tmin - epoch_tmin) * sfreq))
    stop = int(round((task_tmax - epoch_tmin) * sfreq))
    task = injected["X"][..., start:stop]
    weights = np.asarray(anchors, dtype=np.float64)
    source_trace = np.einsum("c,nct->nt", weights[source_node], task)
    target_steering = np.linalg.pinv(weights)[:, target_node]
    delay_samples = delay_graph_steps * (task_tmax - task_tmin) * sfreq / graph_steps
    grid = np.arange(task.shape[-1], dtype=np.float64)
    shifted = np.stack(
        [
            np.interp(grid - delay_samples, grid, trial, left=0.0, right=0.0)
            for trial in source_trace
        ]
    )
    task += (float(strength) * target_steering[None, :, None] * shifted[:, None, :]).astype(
        np.float32
    )
    injected["X"][..., start:stop] = task
    return injected


def _evidence_spaces(
    name: str,
    referenced_data: dict[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, dict[str, Any]]:
    standardized = apply_channelwise_zscore(referenced_data["X"], mean, std)
    innovations = fit_var_innovations(standardized)
    return {
        name: {**referenced_data, "X": standardized},
        f"{name}_innovations": {
            **referenced_data,
            "X": innovations,
            "epoch_tmin": float(referenced_data["epoch_tmin"])
            + 6.0 / float(referenced_data["sfreq"]),
        },
    }


def _known_route_metrics(
    arrays: dict[str, np.ndarray],
    source_node: int,
    target_node: int,
    true_delay: float,
) -> dict[str, Any]:
    route = arrays["route_probability"]
    recovered = _expected_delay(arrays)
    bands = np.arange(route.shape[0])
    forward = route[bands, bands, target_node, source_node]
    reverse = route[bands, bands, source_node, target_node]
    active = forward > 0
    transport_estimate = float("nan")
    transport_error = float("nan")
    if active.any():
        best_band = int(np.argmax(forward))
        transport_estimate = float(recovered[best_band, best_band, target_node, source_node])
        transport_error = abs(transport_estimate - float(true_delay))
    else:
        best_band = -1

    signed_map = arrays.get("signed_node_delay_map")
    signed_mean = arrays.get("signed_node_delay_mean")
    signed_accepted = arrays.get("signed_node_accepted")
    if signed_map is not None and signed_accepted is not None:
        node_accepted = bool(np.asarray(signed_accepted)[target_node, source_node])
        node_map = float(np.asarray(signed_map)[target_node, source_node])
        node_mean = (
            float(np.asarray(signed_mean)[target_node, source_node])
            if signed_mean is not None
            else float("nan")
        )
        estimator = "signed_node_posterior_map"
    else:
        node_accepted = bool(active.any())
        node_map = transport_estimate
        node_mean = transport_estimate
        estimator = "legacy_transport_posterior_mean"

    # A delay is usable by the classifier only when a transport band-pair is
    # active. Its value, however, is identified once at node level and must be
    # evaluated with the declared posterior-MAP estimator rather than a
    # different band-expanded posterior mean.
    estimate = node_map if active.any() and node_accepted else float("nan")
    error = abs(estimate - float(true_delay)) if np.isfinite(estimate) else float("nan")
    return {
        "known_forward_routes": int(active.sum()),
        "known_reverse_routes": int((reverse > 0).sum()),
        "known_forward_max_probability": float(forward.max(initial=0.0)),
        "known_reverse_max_probability": float(reverse.max(initial=0.0)),
        "known_direction_probability_margin": float(
            forward.max(initial=0.0) - reverse.max(initial=0.0)
        ),
        "known_best_band": best_band,
        "known_delay_estimator": estimator,
        "known_delay_recovered": estimate,
        "known_delay_absolute_error": error,
        "known_node_direction_detected": node_accepted,
        "known_node_delay_map": node_map,
        "known_node_delay_map_absolute_error": abs(node_map - float(true_delay)),
        "known_node_delay_posterior_mean": node_mean,
        "known_node_delay_posterior_mean_absolute_error": abs(node_mean - float(true_delay)),
        "known_transport_delay_posterior_mean": transport_estimate,
        "known_transport_delay_posterior_mean_absolute_error": transport_error,
        "known_transport_direction_detected": bool(active.any()),
        "known_direction_detected": bool(active.any()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/all_experiments.yaml")
    parser.add_argument(
        "--protocol-config", default="configs/experiments/v42_identifiable_decoder.yaml"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--subject", default="1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--injection-strengths", default="0.25,0.5")
    parser.add_argument("--injection-delays", default="0.25,0.5,1,2,4,8")
    parser.add_argument("--spaces", default="car,csd")
    parser.add_argument("--no-innovations", action="store_true")
    parser.add_argument("--skip-injection", action="store_true")
    args = parser.parse_args()

    output = ensure_dir(args.output)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    protocol_path = Path(args.protocol_config)
    if not protocol_path.is_absolute():
        protocol_path = (ROOT / protocol_path).resolve()
    source_fingerprint = _source_fingerprint(protocol_path, config_path)
    cfg = load_experiment_config(args.config, "E14", overrides={"device": args.device})
    protocol = load_yaml(args.protocol_config)
    model_cfg = _model_cfg(cfg, protocol)
    write_json(
        output / "audit_manifest.json",
        {
            "experiment_id": str(protocol.get("experiment_id", "V42_IDENTIFIABLE_DECODER")),
            "protocol": str(protocol.get("protocol", "")),
            "architecture_version": str(model_cfg.get("architecture_version", "")),
            "delay_estimator": str(model_cfg.get("delay_evidence_estimator", "legacy")),
            "subject": str(args.subject),
            "seed": int(args.seed),
            "bootstrap_samples": int(args.bootstrap_samples),
            "source_fingerprint": source_fingerprint,
            "classifier_training": False,
            "heldout_session_E_accessed": False,
        },
    )
    dataset_cfg = resolve_dataset_cfg(cfg, key="bci2a_config")
    data = load_processed_npz(dataset_cfg["root"])
    session_t = subject_session_data(data, args.subject, session="T")
    cropped = _crop_to_common_task_window(session_t, {"model": model_cfg})
    folds = _repeated_stratified_folds(np.asarray(cropped["y"]), 3, 2, args.seed)
    train_idx, _ = folds[0]

    requested_spaces = [value.strip().lower() for value in args.spaces.split(",") if value.strip()]
    invalid_spaces = sorted(set(requested_spaces).difference({"car", "csd"}))
    if invalid_spaces:
        raise ValueError(f"Unknown evidence spaces: {invalid_spaces}")
    referenced = {
        name: subset(_reference_space(cropped, model_cfg, name)[0], train_idx)
        for name in requested_spaces
    }
    reference_stats = {
        name: channelwise_train_stats(data["X"]) for name, data in referenced.items()
    }
    spaces: dict[str, dict[str, Any]] = {}
    for name, referenced_data in referenced.items():
        if args.no_innovations:
            spaces[name] = {
                **referenced_data,
                "X": apply_channelwise_zscore(referenced_data["X"], *reference_stats[name]),
            }
        else:
            spaces.update(_evidence_spaces(name, referenced_data, *reference_stats[name]))

    summaries: list[dict[str, Any]] = []
    for index, (name, space_data) in enumerate(spaces.items()):
        print(json.dumps({"event": "space_started", "space": name}), flush=True)
        summary, arrays = _audit_space(
            name,
            space_data,
            model_cfg,
            args.seed + 100 * index,
            args.device,
            args.bootstrap_samples,
        )
        summaries.append(summary)
        np.savez_compressed(output / f"{name}_prior.npz", **arrays)
        write_json(output / f"{name}_summary.json", summary)
        write_csv(output / "evidence_space_summary.csv", summaries)
        print(
            json.dumps({"event": "space_completed", "space": name, "passed": summary["passed"]}),
            flush=True,
        )

    if args.skip_injection:
        passing = [row["evidence_space"] for row in summaries if row["passed"]]
        write_json(
            output / "audit_decision.json",
            {
                "status": "passed" if passing else "failed",
                "passing_spaces": passing,
                "classifier_training": False,
                "heldout_session_E_accessed": False,
                "subject": str(args.subject),
                "session": "T",
                "inner_train_trials": int(train_idx.size),
                "delay_estimator": str(model_cfg.get("delay_evidence_estimator", "legacy")),
                "source_fingerprint": source_fingerprint,
                "injection_skipped": True,
            },
        )
        return

    source, target = 7, 11
    channel_names = [str(value) for value in cropped.get("ch_names", [])]
    if "C3" in channel_names and "C4" in channel_names:
        source, target = channel_names.index("C3"), channel_names.index("C4")
    anchors = (
        AnchoredSpatialProjection(
            len(channel_names) or int(cropped["X"].shape[1]),
            int(model_cfg["latent_nodes"]),
            max_deviation=0.0,
            electrode_coordinates=cropped.get("electrode_coordinates"),
            anchor_indices=cropped.get("spatial_anchor_indices"),
        )
        .anchors.detach()
        .cpu()
        .numpy()
    )
    source_node = int(np.argmax(anchors[:, source]))
    target_node = int(np.argmax(anchors[:, target]))
    projector_identity_error = float(
        np.max(np.abs(anchors @ np.linalg.pinv(anchors) - np.eye(anchors.shape[0])))
    )
    strengths = [float(value) for value in args.injection_strengths.split(",") if value]
    delays = [float(value) for value in args.injection_delays.split(",") if value]
    injection_rows: list[dict[str, Any]] = []
    injection_space_index = 0
    for reference_name, referenced_data in referenced.items():
        normalized_reference = {
            **referenced_data,
            "X": apply_channelwise_zscore(referenced_data["X"], *reference_stats[reference_name]),
        }
        for strength in strengths:
            for delay in delays:
                injected_reference = _inject_task_delay_latent(
                    normalized_reference,
                    anchors,
                    source_node,
                    target_node,
                    delay,
                    strength,
                    int(model_cfg["graph_timesteps"]),
                    float(model_cfg["task_tmin"]),
                    float(model_cfg["task_tmax"]),
                )
                injected_spaces = {
                    reference_name: injected_reference,
                }
                if not args.no_innovations:
                    innovations = fit_var_innovations(injected_reference["X"])
                    injected_spaces[f"{reference_name}_innovations"] = {
                        **injected_reference,
                        "X": innovations,
                        "epoch_tmin": float(injected_reference["epoch_tmin"])
                        + 6.0 / float(injected_reference["sfreq"]),
                    }
                for name, injected in injected_spaces.items():
                    arrays, summary = _fit_prior(
                        injected,
                        model_cfg,
                        args.seed
                        + 10000
                        + 1000 * injection_space_index
                        + int(100 * strength)
                        + int(10 * delay),
                        args.device,
                        min(64, args.bootstrap_samples),
                    )
                    active = arrays["route_probability"] > 0
                    row = {
                        "evidence_space": name,
                        "injection_stage": "post_reference_fixed_gain_pre_projector",
                        "strength": strength,
                        "true_delay_graph_steps": delay,
                        "source_node": source_node,
                        "target_node": target_node,
                        "accepted_edges": int(summary["accepted_evidence_edges"]),
                        "active_output_routes": int(active.sum()),
                        "phase_surrogate_support_drop": float(
                            summary["phase_surrogate_support_drop"]
                        ),
                        "normal_node_edges": int(summary["normal_node_edges"]),
                        "surrogate_node_edges": int(summary["phase_surrogate_node_edges"]),
                        **_known_route_metrics(arrays, source_node, target_node, delay),
                    }
                    injection_rows.append(row)
                    write_csv(output / "injection_power.csv", injection_rows)
                    print(
                        json.dumps({"event": "injection_completed", **row}),
                        flush=True,
                    )
                    injection_space_index += 1

    passing = [row["evidence_space"] for row in summaries if row["passed"]]
    write_json(
        output / "audit_decision.json",
        {
            "status": "passed" if passing else "failed",
            "passing_spaces": passing,
            "classifier_training": False,
            "heldout_session_E_accessed": False,
            "subject": str(args.subject),
            "session": "T",
            "inner_train_trials": int(train_idx.size),
            "source_channel": channel_names[source] if channel_names else source,
            "target_channel": channel_names[target] if channel_names else target,
            "source_node": source_node,
            "target_node": target_node,
            "projector_pseudoinverse_identity_error": projector_identity_error,
            "injection_stage": "post_reference_fixed_gain_pre_projector",
        },
    )


if __name__ == "__main__":
    main()
