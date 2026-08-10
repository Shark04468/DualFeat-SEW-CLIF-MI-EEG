#!/usr/bin/env python3
"""Audit V8 delay evidence spaces on one Session-T inner fold without training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.analysis.evidence_space import fold_local_continuous_delay_prior  # noqa: E402
from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    sha256_fingerprint,
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_baselines import session_t_development_view  # noqa: E402
from dpc_snn.experiments.v8_evidence_audit import (  # noqa: E402
    V8_EVIDENCE_SPACES,
    build_v8_evidence_rates,
    pool_v8_anatomical_regions,
    prepare_v8_evidence_space,
    select_physical_evidence_space,
    within_band_route_count,
)
from dpc_snn.experiments.v8_delay_prior import v8_delay_prior_seed  # noqa: E402
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    source_tree_digest,
    v8_heldout_lock_manifest,
)
from dpc_snn.experiments.v8_training import (  # noqa: E402
    apply_v8_fold_gain,
    fit_v8_gain_from_cached_rates,
)
from dpc_snn.utils.io import ensure_dir, read_json, save_npz, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v8_e1_baselines import _subject_file  # noqa: E402
from scripts.run_v8_e2_zero_delay import _nested_folds  # noqa: E402
from scripts.run_v8_e3_delay_residual_fold import _reference_model  # noqa: E402
from scripts.run_v8_e3_static_delay import _load_parent_rates  # noqa: E402


SPACE_FILES = ("manifest.json", "fingerprint.json", "evidence.npz", "summary.json")


def _csv(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _space_summary(
    *,
    name: str,
    rates: Any,
    model_config: dict[str, Any],
    delay_config: dict[str, Any],
    bootstrap_samples: int,
    seed: int,
    transform: dict[str, Any],
    split_strata: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    started = time.time()
    arrays, summary = fold_local_continuous_delay_prior(
        {"X": np.empty((rates.fast.shape[0], 1, 1), dtype=np.float32)},
        model_config["band_edges_hz"],
        model_config["band_edges_hz"],
        n_nodes=rates.fast.shape[2],
        graph_steps=rates.fast.shape[-1],
        max_delay_steps=int(delay_config["maximum_delay_samples"]),
        bootstrap_samples=int(bootstrap_samples),
        grid_oversample=int(delay_config["prior"]["grid_oversample"]),
        min_bayes_factor=float(delay_config["prior"]["minimum_bayes_factor"]),
        min_bootstrap_frequency=float(
            delay_config["prior"]["minimum_bootstrap_frequency"]
        ),
        min_direction_probability=float(
            delay_config["prior"]["minimum_direction_probability"]
        ),
        seed=int(seed),
        device="cpu",
        task_tmin=0.0,
        task_tmax=float(model_config["task_tmax"] - model_config["task_tmin"]),
        analytic_features=rates.fast.detach().cpu().numpy(),
        analytic_representation=f"v8_e3_{name}_fixed_gain_physical_rates",
        split_strata=split_strata,
        surrogate_replicates=int(delay_config["prior"].get("surrogate_replicates", 1)),
    )
    within = within_band_route_count(arrays)
    summary = {
        **summary,
        **transform,
        "evidence_space": name,
        "accepted_within_band_routes": within,
        "maximum_sparse_routes": int(delay_config["maximum_routes"]),
        "elapsed_seconds": time.time() - started,
        "classifier_training": False,
        "heldout_session_e_accessed": False,
    }
    return arrays, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache-parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default="configs/experiments/v8_e3_delay_residual_canary.yaml"
    )
    parser.add_argument("--model-config", default="configs/models/v8_accuracy_first.yaml")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=128)
    parser.add_argument(
        "--spaces",
        default=(
            "car,csd,ea,car_innovations,csd_innovations,"
            "car_innovations_unwhitened,csd_innovations_unwhitened,"
            "car_innovations_unwhitened_regions,"
            "csd_innovations_unwhitened_regions"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    configure_cache_env()
    output = ensure_dir(Path(args.output).resolve())
    spaces = _csv(args.spaces)
    unknown = sorted(set(spaces).difference(V8_EVIDENCE_SPACES))
    if unknown or len(spaces) != len(set(spaces)):
        raise ValueError(f"invalid or duplicate V8 evidence spaces: {unknown or spaces}")
    config_path = Path(args.config).resolve()
    model_path = Path(args.model_config).resolve()
    experiment_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    delay_config = dict(experiment_config["delay"])
    if int(args.bootstrap_samples) < 32:
        raise ValueError("strict V8 evidence audit requires at least 32 bootstraps")

    source_manifest = collect_source_tree_manifest(ROOT)
    source_sha = source_tree_digest(source_manifest)
    subject_path = _subject_file(Path(args.data).resolve(), int(args.subject))
    data = load_processed_npz(subject_path)
    x, labels, metadata, access = session_t_development_view(data)
    nested = _nested_folds(metadata, n_splits=6, split_seed=0)
    _, _, inner_train, _, inner_run = nested[int(args.fold)]
    trial_ids = [metadata[int(index)]["trial_id"] for index in inner_train]
    split_strata = [
        f"run={metadata[int(index)]['run']}|class={int(labels[int(index)])}"
        for index in inner_train
    ]
    payload = {
        "schema": "dpc-snn-v8-e3-evidence-space-audit/v1",
        "subject": int(args.subject),
        "fold": int(args.fold),
        "seed": int(args.seed),
        "inner_validation_run": str(inner_run),
        "inner_train_trial_ids": trial_ids,
        "split_half_scheme": "run_class_stratified_deterministic_alternation",
        "split_strata": split_strata,
        "spaces": spaces,
        "bootstrap_samples": int(args.bootstrap_samples),
        "data_sha256": file_sha256(subject_path),
        "cache_parent": str(Path(args.cache_parent).resolve()),
        "config_sha256": file_sha256(config_path),
        "model_config_sha256": file_sha256(model_path),
        "source_tree_sha256": source_sha,
        "data_access": access,
        "classifier_training": False,
        "heldout_session_e_accessed": False,
    }
    fingerprint = sha256_fingerprint(payload)
    if (output / "manifest.json").is_file():
        required = tuple(read_json(output / "manifest.json")["required_files"])
        validate_run_artifact_manifest(output, required_files=required, verify_hashes=True)
        if read_json(output / "fingerprint.json").get("combined_sha256") != fingerprint:
            raise RuntimeError("V8 evidence audit resume fingerprint mismatch")
        print(read_json(output / "audit_decision.json"), flush=True)
        return

    write_json(output / "heldout_lock_manifest.json", v8_heldout_lock_manifest())
    write_json(output / "source_tree_manifest.json", source_manifest)
    write_json(output / "fingerprint.json", {**payload, "combined_sha256": fingerprint})
    reference_model = _reference_model(model_config)
    base_rates, cache_manifest = _load_parent_rates(
        Path(args.cache_parent).resolve(),
        subject=int(args.subject),
        data_sha256=file_sha256(subject_path),
        model=reference_model,
    )
    car_gain = fit_v8_gain_from_cached_rates(base_rates, inner_train)
    car_rates = apply_v8_fold_gain(base_rates.subset(inner_train), car_gain)
    channel_names = [str(value) for value in data["ch_names"]]
    sfreq = float(np.asarray(data["sfreq"]).item())
    epoch_tmin = float(np.asarray(data["epoch_tmin"]).item())
    summaries: list[dict[str, Any]] = []
    audit_seed = v8_delay_prior_seed(args.subject, args.fold, args.seed)
    for name in spaces:
        directory = ensure_dir(output / name)
        space_fingerprint = sha256_fingerprint({**payload, "evidence_space": name})
        if all((directory / file).is_file() for file in SPACE_FILES):
            validate_run_artifact_manifest(directory, required_files=SPACE_FILES)
            saved = read_json(directory / "summary.json")
            if saved.get("input_fingerprint") != space_fingerprint:
                raise RuntimeError(f"cached evidence-space fingerprint mismatch: {name}")
            summaries.append(saved)
            continue
        if name == "car":
            rates = car_rates
            transform = {
                "space": "car",
                "reference": "exact_cached_v8_car_physical_frontend",
                "fit_scope": "active_inner_training_fold_only",
                "input_trials": int(inner_train.size),
                "input_shape": list(x[inner_train].shape),
                "output_shape": list(rates.fast.shape),
                "output_sha256": cache_manifest["cache_file_sha256"],
                "epoch_tmin": epoch_tmin,
                "exact_physical_sensor_nodes": True,
                "eligible_for_physical_delay_routes": True,
                "eligible_for_region_delay_routes": False,
                "eligible_for_interpretable_delay_routes": True,
                "region_pooling_requested": False,
            }
        else:
            transformed, adjusted_epoch_tmin, transform = prepare_v8_evidence_space(
                x[inner_train],
                name,
                channel_names=channel_names,
                sfreq=sfreq,
                epoch_tmin=epoch_tmin,
                task_tmin=float(model_config["task_tmin"]),
            )
            rates = build_v8_evidence_rates(
                transformed,
                model_config,
                epoch_tmin=adjusted_epoch_tmin,
                transform_sha256=str(transform["output_sha256"]),
                device=args.device,
                batch_size=int(args.batch_size),
            )
            if bool(transform.get("region_pooling_requested")):
                rates, region_metadata = pool_v8_anatomical_regions(rates, channel_names)
                transform = {**transform, **region_metadata}
        arrays, summary = _space_summary(
            name=name,
            rates=rates,
            model_config=model_config,
            delay_config=delay_config,
            bootstrap_samples=int(args.bootstrap_samples),
            seed=audit_seed,
            transform=transform,
            split_strata=split_strata,
        )
        summary["input_fingerprint"] = space_fingerprint
        save_npz(directory / "evidence.npz", **arrays)
        write_json(
            directory / "fingerprint.json",
            {**payload, "evidence_space": name, "combined_sha256": space_fingerprint},
        )
        write_json(directory / "summary.json", summary)
        write_run_artifact_manifest(directory, required_files=SPACE_FILES)
        summaries.append(summary)
        write_csv(output / "evidence_space_summary.csv", summaries)
        print(
            {
                "event": "evidence_space_completed",
                "space": name,
                "passed": summary["evidence_pipeline_passed"],
                "checks": summary["evidence_checks"],
            },
            flush=True,
        )

    selected = select_physical_evidence_space(summaries)
    selected_physical = next(
        (
            row["evidence_space"]
            for row in summaries
            if row["evidence_space"] == selected
            and row["eligible_for_physical_delay_routes"]
        ),
        None,
    )
    decision = {
        "schema": "dpc-snn-v8-e3-evidence-space-decision/v1",
        "status": "passed" if selected is not None else "failed",
        "selected_physical_evidence_space": selected_physical,
        "selected_interpretable_evidence_space": selected,
        "selection_priority_locked_before_audit": [
            "csd_innovations_unwhitened_regions",
            "car_innovations_unwhitened_regions",
            "csd_innovations_unwhitened",
            "car_innovations_unwhitened",
            "csd",
            "car",
        ],
        "strict_passing_spaces": [
            row["evidence_space"] for row in summaries if row["evidence_pipeline_passed"]
        ],
        "strict_passing_physical_spaces": [
            row["evidence_space"]
            for row in summaries
            if row["evidence_pipeline_passed"]
            and row["eligible_for_physical_delay_routes"]
            and int(row["accepted_within_band_routes"]) > 0
        ],
        "strict_passing_interpretable_spaces": [
            row["evidence_space"]
            for row in summaries
            if row["evidence_pipeline_passed"]
            and row["eligible_for_interpretable_delay_routes"]
            and int(row["accepted_within_band_routes"]) > 0
        ],
        "classifier_training": False,
        "heldout_session_e_accessed": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_tree_sha256": source_sha,
        "input_fingerprint": fingerprint,
    }
    write_json(output / "audit_decision.json", decision)
    write_csv(output / "evidence_space_summary.csv", summaries)
    required = (
        "manifest.json",
        "fingerprint.json",
        "heldout_lock_manifest.json",
        "source_tree_manifest.json",
        "evidence_space_summary.csv",
        "audit_decision.json",
        *(f"{name}/{file}" for name in spaces for file in SPACE_FILES),
    )
    write_run_artifact_manifest(output, required_files=required)
    print(decision, flush=True)


if __name__ == "__main__":
    main()
