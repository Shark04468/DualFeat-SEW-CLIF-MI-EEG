#!/usr/bin/env python3
"""Run V8 zero-delay branch construction under nested Session-T OOF."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    session_t_run_grouped_folds,
    sha256_fingerprint,
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
    write_trial_predictions,
)
from dpc_snn.experiments.v8_baselines import (  # noqa: E402
    merge_oof_predictions,
    nested_run_grouped_indices,
    rank_screening_models,
    session_t_development_view,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    build_v8_run_fingerprint,
    collect_source_tree_manifest,
    source_tree_digest,
    validate_v8_resume_fingerprint,
    v8_heldout_lock_manifest,
    write_v8_fingerprint,
)
from dpc_snn.experiments.v8_training import (  # noqa: E402
    V8CachedRates,
    apply_v8_fold_gain,
    cache_v8_physical_rates,
    fit_v8,
    fit_v8_gain_from_cached_rates,
    fit_v8_physical_gain,
    load_v8_rates,
    predict_v8,
    save_v8_rates,
    seed_v8,
)
from dpc_snn.models.build import build_model  # noqa: E402
from dpc_snn.models.v8_accuracy_first import V8AccuracyFirstModel  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402


RUN_FILES = (
    "manifest.json",
    "resolved_config.yaml",
    "source_fingerprint.json",
    "source_tree_manifest.json",
    "data_access_manifest.json",
    "split_manifest.json",
    "augmentation_manifest.json",
    "metrics.json",
    "predictions.npz",
    "predictions.csv",
    "prefix_predictions.npz",
    "runtime_status.json",
    "stdout.log",
    "stderr.log",
)
FOLD_FILES = (
    "best.pt",
    "last.pt",
    "history.csv",
    "result.json",
    "outer_test_predictions.npz",
    "selection_best.pt",
    "selection_last.pt",
    "selection_history.csv",
    "selection_result.json",
    "selection_predictions.npz",
)
FOLD_REQUIRED_FILES = ("manifest.json", *FOLD_FILES)


@dataclass(frozen=True)
class SubjectBundle:
    subject_path: Path
    data_sha256: str
    x: np.ndarray
    y: np.ndarray
    metadata: list[dict[str, Any]]
    access_manifest: dict[str, Any]
    nested_folds: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]
    split_manifest: dict[str, Any]
    base_rates: V8CachedRates
    cache_manifest: dict[str, Any]


def _csv(value: str, cast: Any = str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _subject_file(data_root: Path, subject: int) -> Path:
    matches = [
        path
        for path in (data_root / f"A{subject:02d}.npz", data_root / f"A{subject:02d}_all.npz")
        if path.is_file()
    ]
    if len(matches) == 1:
        return matches[0]
    fallback = sorted(data_root.glob(f"A{subject:02d}*.npz"))
    if len(fallback) != 1:
        raise FileNotFoundError(f"cannot uniquely resolve subject {subject} under {data_root}")
    return fallback[0]


def _environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "torch", "scipy", "einops", "mne"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda,
        "storage_root": os.environ.get("DPC_SNN_STORAGE_ROOT"),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def _required_files(n_folds: int) -> tuple[str, ...]:
    return RUN_FILES + tuple(
        f"fold_{fold}/{name}"
        for fold in range(n_folds)
        for name in FOLD_REQUIRED_FILES
    )


def _build_variant(
    model_config: dict[str, Any],
    variant_name: str,
    overrides: dict[str, Any],
    *,
    seed: int,
) -> V8AccuracyFirstModel:
    seed_v8(seed)
    resolved = {**model_config, **overrides}
    model = build_model("v8_accuracy_first", resolved)
    if not isinstance(model, V8AccuracyFirstModel):
        raise TypeError("V8 E2 model factory returned an unexpected model")
    if model.delay_auxiliary_enabled or model.delay_auxiliary is not None:
        raise RuntimeError(f"zero-delay E2 variant {variant_name!r} enabled a delay branch")
    if model.decoder is not None and model.decoder.decoder_kind != "ann":
        raise RuntimeError(f"E2 variant {variant_name!r} must use the matched ANN decoder")
    return model


def _probabilities(logits: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.from_numpy(logits).float(), dim=1).numpy()


def _nested_folds(
    metadata: list[dict[str, Any]],
    *,
    n_splits: int,
    split_seed: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    outer = session_t_run_grouped_folds(
        metadata,
        n_splits=int(n_splits),
        seed=int(split_seed),
        shuffle=True,
    )
    return [
        (
            outer_train,
            outer_test,
            *nested_run_grouped_indices(metadata, outer_train, outer_test),
        )
        for outer_train, outer_test in outer
    ]


def _split_manifest(
    metadata: list[dict[str, Any]],
    nested: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]],
    *,
    subject: int,
    split_seed: int,
) -> dict[str, Any]:
    return {
        "stage": "development",
        "subject": int(subject),
        "session": "T",
        "method": "nested_run_grouped_oof",
        "n_splits": len(nested),
        "split_seed": int(split_seed),
        "folds": [
            {
                "fold": fold,
                "outer_train_trial_ids": [
                    metadata[int(index)]["trial_id"] for index in outer_train
                ],
                "outer_test_trial_ids": [
                    metadata[int(index)]["trial_id"] for index in outer_test
                ],
                "inner_train_trial_ids": [
                    metadata[int(index)]["trial_id"] for index in inner_train
                ],
                "inner_validation_trial_ids": [
                    metadata[int(index)]["trial_id"] for index in inner_validation
                ],
                "outer_test_run": metadata[int(outer_test[0])]["run"],
                "inner_validation_run": inner_run,
            }
            for fold, (
                outer_train,
                outer_test,
                inner_train,
                inner_validation,
                inner_run,
            ) in enumerate(nested)
        ],
    }


def _base_cache_payload(
    *,
    subject_path: Path,
    data_sha256: str,
    metadata: list[dict[str, Any]],
    model: V8AccuracyFirstModel,
    preprocessing: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "dpc-snn-v8-unit-gain-physical-cache/v2-gain-invariant-envelope",
        "subject_file": subject_path.name,
        "data_sha256": data_sha256,
        "trial_ids": [row["trial_id"] for row in metadata],
        "session": "T",
        "physical_frontend_fingerprint": model.physical_frontend_fingerprint(),
        "preprocessing": preprocessing,
        "unit_gain": True,
        "trainable_representation_cached": False,
    }


def _load_or_build_base_rates(
    *,
    output: Path,
    subject: int,
    subject_path: Path,
    data_sha256: str,
    x: np.ndarray,
    metadata: list[dict[str, Any]],
    model: V8AccuracyFirstModel,
    preprocessing: dict[str, Any],
    canary_train_indices: np.ndarray,
    device: str,
) -> tuple[V8CachedRates, dict[str, Any]]:
    directory = ensure_dir(output / "shared_physical_rates" / f"subject_{subject:02d}")
    cache_path = directory / "unit_gain_rates.pt"
    manifest_path = directory / "manifest.json"
    payload = _base_cache_payload(
        subject_path=subject_path,
        data_sha256=data_sha256,
        metadata=metadata,
        model=model,
        preprocessing=preprocessing,
    )
    expected_fingerprint = sha256_fingerprint(payload)
    if cache_path.is_file() and manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest.get("cache_fingerprint") != expected_fingerprint:
            raise RuntimeError("V8 physical-rate cache fingerprint mismatch")
        if manifest.get("cache_file_sha256") != file_sha256(cache_path):
            raise RuntimeError("V8 physical-rate cache file hash mismatch")
        rates = load_v8_rates(cache_path, model)
        if not torch.equal(rates.gain, torch.ones_like(rates.gain)):
            raise RuntimeError("V8 shared physical cache is not unit-gain")
        return rates, manifest

    model.set_training_gain(torch.ones(model.n_bands, model.n_channels))
    rates = cache_v8_physical_rates(model, x, device=device, batch_size=16)
    if not torch.equal(rates.gain, torch.ones_like(rates.gain)):
        raise RuntimeError("new V8 shared physical cache is not unit-gain")
    cached_gain = fit_v8_gain_from_cached_rates(rates, canary_train_indices)
    direct_gain = fit_v8_physical_gain(
        model,
        x[canary_train_indices],
        device=device,
        batch_size=16,
    )
    gain_max_abs = float((cached_gain - direct_gain).abs().max())
    gain_max_relative = float(
        (
            (cached_gain - direct_gain).abs()
            / direct_gain.abs().clamp_min(torch.finfo(direct_gain.dtype).tiny)
        ).max()
    )
    probe = np.asarray(canary_train_indices[: min(4, len(canary_train_indices))])
    direct = cache_v8_physical_rates(model, x[probe], device=device, batch_size=4)
    transformed = apply_v8_fold_gain(rates.subset(probe), cached_gain)
    fast_max_abs = float((direct.fast - transformed.fast).abs().max())
    slow_max_abs = float((direct.slow - transformed.slow).abs().max())
    tolerance = float(preprocessing["cache_slow_gain_invariance_tolerance"])
    gain_tolerance = float(preprocessing["cache_gain_relative_tolerance"])
    if (
        gain_max_relative > gain_tolerance
        or fast_max_abs > 1e-5
        or slow_max_abs > tolerance
    ):
        raise RuntimeError(
            "V8 unit-gain cache canary failed: "
            f"gain_abs={gain_max_abs:.3e}, gain_rel={gain_max_relative:.3e}, "
            f"fast={fast_max_abs:.3e}, slow={slow_max_abs:.3e}"
        )
    save_v8_rates(cache_path, rates)
    manifest = {
        **payload,
        "cache_fingerprint": expected_fingerprint,
        "cache_file_sha256": file_sha256(cache_path),
        "shape_fast": list(rates.fast.shape),
        "shape_slow": list(rates.slow.shape),
        "gain_canary_max_abs": gain_max_abs,
        "gain_canary_max_relative": gain_max_relative,
        "fast_canary_max_abs": fast_max_abs,
        "slow_canary_max_abs": slow_max_abs,
        "canary_passed": True,
    }
    write_json(manifest_path, manifest)
    model.set_training_gain(torch.ones(model.n_bands, model.n_channels))
    return rates, manifest


def _load_fold(
    fold_dir: Path,
    *,
    expected_indices: np.ndarray,
    expected_labels: np.ndarray,
    endpoints: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    if not all((fold_dir / name).is_file() for name in FOLD_REQUIRED_FILES):
        return None
    validate_run_artifact_manifest(
        fold_dir,
        required_files=FOLD_REQUIRED_FILES,
        verify_hashes=True,
        verify_prediction_schema=False,
    )
    result = read_json(fold_dir / "result.json")
    with np.load(fold_dir / "outer_test_predictions.npz", allow_pickle=False) as archive:
        prediction = {key: archive[key] for key in archive.files}
    if set(prediction) != {"indices", "logits", "prefix_logits", "labels"}:
        return None
    if not np.array_equal(prediction["indices"], expected_indices):
        return None
    if not np.array_equal(prediction["labels"], expected_labels):
        return None
    if prediction["logits"].shape != (expected_indices.size, 4):
        return None
    if prediction["prefix_logits"].shape != (expected_indices.size, endpoints, 4):
        return None
    if not np.isfinite(prediction["logits"]).all() or not np.isfinite(
        prediction["prefix_logits"]
    ).all():
        return None
    return result, prediction


def _fit_kwargs(
    training: dict[str, Any], augmentation: dict[str, Any]
) -> dict[str, Any]:
    return {
        "batch_size": int(training["batch_size"]),
        "accumulation_steps": int(training["gradient_accumulation_steps"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "max_gradient_norm": float(training["max_gradient_norm"]),
        "warmup_fraction": float(training["warmup_fraction"]),
        "endpoint_weights": tuple(training["endpoint_weights"]),
        "endpoint_loss_weight": float(training["endpoint_loss_weight"]),
        "firing_rate_weight": float(training["firing_rate_weight"]),
        "spatial_orthogonality_weight": float(training["spatial_orthogonality_weight"]),
        "statistical_orthogonality_weight": float(
            training["statistical_orthogonality_weight"]
        ),
        "label_smoothing": float(training["label_smoothing"]),
        "augmentation": augmentation,
    }


def _merge_prefix_predictions(
    parts: list[dict[str, np.ndarray]],
    *,
    count: int,
) -> np.ndarray:
    indices = np.concatenate([part["indices"] for part in parts])
    prefixes = np.concatenate([part["prefix_logits"] for part in parts])
    if sorted(indices.tolist()) != list(range(count)):
        raise RuntimeError("V8 prefix OOF coverage is incomplete or duplicated")
    return prefixes[np.argsort(indices)]


def _run_one(
    *,
    output: Path,
    config: dict[str, Any],
    model_config: dict[str, Any],
    source_tree: dict[str, str],
    environment: dict[str, Any],
    bundle: SubjectBundle,
    variant_name: str,
    variant_overrides: dict[str, Any],
    seed: int,
    device: str,
    stage: str = "E2",
    build_variant_fn: Callable[..., V8AccuracyFirstModel] | None = None,
) -> dict[str, Any]:
    subject = int(bundle.metadata[0]["subject"])
    selection = dict(config["selection"])
    training = dict(config["training"])
    augmentation = dict(config["augmentation"])
    builder = _build_variant if build_variant_fn is None else build_variant_fn
    resolved_model = {**model_config, **variant_overrides}
    resolved = {
        **config,
        "active_variant": variant_name,
        "active_subject": subject,
        "active_seed": int(seed),
        "resolved_model": resolved_model,
    }
    fingerprint = build_v8_run_fingerprint(
        resolved_run_config=resolved,
        source_tree=source_tree,
        data={
            bundle.subject_path.name: bundle.data_sha256,
            "access": bundle.access_manifest,
            "physical_rate_cache": bundle.cache_manifest,
        },
        split=bundle.split_manifest,
        augmentation=augmentation,
        prior={"policy": "none", "delay_auxiliary_enabled": False},
        checkpoint={
            "policy": (
                "inner-run best validation kappa -> fixed-epoch outer-train retrain -> "
                "single outer-run test"
            ),
            "evaluation_scope": "nested Session-T outer-run OOF only",
            "session_e_accessed": False,
        },
        environment=environment,
    )
    run_dir = ensure_dir(
        output / variant_name / f"subject_{subject:02d}" / f"seed_{seed}"
    )
    fingerprint_path = run_dir / "source_fingerprint.json"
    n_folds = len(bundle.nested_folds)
    if (run_dir / "manifest.json").is_file():
        validate_v8_resume_fingerprint(fingerprint_path, fingerprint)
        validate_run_artifact_manifest(
            run_dir,
            required_files=_required_files(n_folds),
            verify_hashes=True,
            verify_prediction_schema=True,
        )
        return read_json(run_dir / "metrics.json")
    if fingerprint_path.is_file():
        validate_v8_resume_fingerprint(fingerprint_path, fingerprint)
    else:
        write_v8_fingerprint(fingerprint_path, fingerprint)
        write_json(run_dir / "source_tree_manifest.json", source_tree)
        write_json(run_dir / "data_access_manifest.json", bundle.access_manifest)
        write_json(run_dir / "split_manifest.json", bundle.split_manifest)
        write_json(run_dir / "augmentation_manifest.json", augmentation)
        (run_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )
    write_json(
        run_dir / "runtime_status.json",
        {"status": "running", "started_at": time.time(), "session_e_accessed": False},
    )

    fold_rows: list[dict[str, Any]] = []
    oof_parts: list[dict[str, np.ndarray]] = []
    total_seconds = 0.0
    total_steps = 0
    parameters = 0
    spike_weighted_sum = 0.0
    spike_trials = 0
    activity_nonzero_weighted_sum = 0.0
    activity_absolute_weighted_sum = 0.0
    activity_trials = 0
    endpoint_count = len(model_config["endpoint_seconds"])
    fit_kwargs = _fit_kwargs(training, augmentation)
    for fold_index, (
        outer_train_indices,
        outer_test_indices,
        inner_train_indices,
        inner_validation_indices,
        inner_validation_run,
    ) in enumerate(bundle.nested_folds):
        fold_dir = ensure_dir(run_dir / f"fold_{fold_index}")
        cached = _load_fold(
            fold_dir,
            expected_indices=outer_test_indices,
            expected_labels=bundle.y[outer_test_indices],
            endpoints=endpoint_count,
        )
        if cached is not None:
            result, prediction = cached
        else:
            fold_seed = int(seed) * 100_003 + fold_index * 1_009
            inner_model = builder(
                model_config, variant_name, variant_overrides, seed=fold_seed
            )
            inner_gain = fit_v8_gain_from_cached_rates(
                bundle.base_rates, inner_train_indices
            )
            inner_train_rates = apply_v8_fold_gain(
                bundle.base_rates.subset(inner_train_indices), inner_gain
            )
            inner_validation_rates = apply_v8_fold_gain(
                bundle.base_rates.subset(inner_validation_indices), inner_gain
            )
            selection_fit = fit_v8(
                inner_model,
                inner_train_rates,
                bundle.y[inner_train_indices],
                validation_rates=inner_validation_rates,
                validation_labels=bundle.y[inner_validation_indices],
                device=device,
                seed=fold_seed,
                epochs=int(selection["max_epochs"]),
                patience=int(selection["patience"]),
                minimum_epochs=int(selection["minimum_epochs"]),
                run_label=f"V8-{stage}-select:{variant_name}:S{subject}:seed{seed}:fold{fold_index}",
                **fit_kwargs,
            )
            selection_evaluation = predict_v8(
                selection_fit.model,
                inner_validation_rates,
                bundle.y[inner_validation_indices],
                device=device,
                batch_size=int(training["batch_size"]),
            )
            selected_epoch = min(
                int(selection["max_epochs"]),
                max(
                    int(selection["minimum_outer_retrain_epochs"]),
                    int(selection_fit.best_epoch),
                ),
            )
            selection_result = {
                "fold": fold_index,
                "fold_seed": fold_seed,
                "inner_validation_run": inner_validation_run,
                "best_epoch": selection_fit.best_epoch,
                "selected_outer_retrain_epoch": selected_epoch,
                "best_selection_metric": selection_fit.best_metric,
                "accuracy": selection_evaluation["accuracy"],
                "balanced_accuracy": selection_evaluation["balanced_accuracy"],
                "kappa": selection_evaluation["kappa"],
                "macro_f1": selection_evaluation["macro_f1"],
                "endpoint_metrics": selection_evaluation["endpoint_metrics"],
                "optimizer_steps": selection_fit.optimizer_steps,
                "elapsed_seconds": selection_fit.elapsed_seconds,
                "gain": inner_gain.tolist(),
            }
            torch.save(selection_fit.best_state, fold_dir / "selection_best.pt")
            torch.save(selection_fit.last_state, fold_dir / "selection_last.pt")
            write_csv(fold_dir / "selection_history.csv", selection_fit.history)
            write_json(fold_dir / "selection_result.json", selection_result)
            np.savez_compressed(
                fold_dir / "selection_predictions.npz",
                indices=inner_validation_indices.astype(np.int64),
                logits=selection_evaluation["logits"].astype(np.float32),
                prefix_logits=selection_evaluation["prefix_logits"].astype(np.float32),
                labels=selection_evaluation["labels"].astype(np.int64),
            )

            outer_seed = fold_seed + 1_000_003
            outer_model = builder(
                model_config, variant_name, variant_overrides, seed=outer_seed
            )
            outer_gain = fit_v8_gain_from_cached_rates(
                bundle.base_rates, outer_train_indices
            )
            outer_train_rates = apply_v8_fold_gain(
                bundle.base_rates.subset(outer_train_indices), outer_gain
            )
            outer_test_rates = apply_v8_fold_gain(
                bundle.base_rates.subset(outer_test_indices), outer_gain
            )
            outer_fit = fit_v8(
                outer_model,
                outer_train_rates,
                bundle.y[outer_train_indices],
                validation_rates=None,
                validation_labels=None,
                device=device,
                seed=outer_seed,
                epochs=selected_epoch,
                patience=selected_epoch,
                minimum_epochs=selected_epoch,
                fixed_epoch=selected_epoch,
                scheduler_epochs=int(selection["max_epochs"]),
                run_label=f"V8-{stage}-outer:{variant_name}:S{subject}:seed{seed}:fold{fold_index}",
                **fit_kwargs,
            )
            evaluation = predict_v8(
                outer_fit.model,
                outer_test_rates,
                bundle.y[outer_test_indices],
                device=device,
                batch_size=int(training["batch_size"]),
            )
            parameters = outer_fit.model.parameter_count
            result = {
                "fold": fold_index,
                "fold_seed": fold_seed,
                "outer_seed": outer_seed,
                "inner_validation_run": inner_validation_run,
                "inner_best_epoch": selection_fit.best_epoch,
                "selected_outer_retrain_epoch": selected_epoch,
                "inner_best_selection_metric": selection_fit.best_metric,
                "accuracy": evaluation["accuracy"],
                "balanced_accuracy": evaluation["balanced_accuracy"],
                "kappa": evaluation["kappa"],
                "macro_f1": evaluation["macro_f1"],
                "endpoint_metrics": evaluation["endpoint_metrics"],
                "binary_spike_rate": evaluation["binary_spike_rate"],
                "final_activity_nonzero_rate": evaluation[
                    "final_activity_nonzero_rate"
                ],
                "final_activity_absolute_mean": evaluation[
                    "final_activity_absolute_mean"
                ],
                "selection_optimizer_steps": selection_fit.optimizer_steps,
                "outer_optimizer_steps": outer_fit.optimizer_steps,
                "optimizer_steps": selection_fit.optimizer_steps + outer_fit.optimizer_steps,
                "selection_elapsed_seconds": selection_fit.elapsed_seconds,
                "outer_elapsed_seconds": outer_fit.elapsed_seconds,
                "elapsed_seconds": selection_fit.elapsed_seconds + outer_fit.elapsed_seconds,
                "parameters": parameters,
                "trainable_parameters": outer_fit.model.trainable_parameter_count,
                "outer_gain": outer_gain.tolist(),
            }
            prediction = {
                "indices": outer_test_indices.astype(np.int64),
                "logits": evaluation["logits"].astype(np.float32),
                "prefix_logits": evaluation["prefix_logits"].astype(np.float32),
                "labels": evaluation["labels"].astype(np.int64),
            }
            torch.save(outer_fit.last_state, fold_dir / "best.pt")
            torch.save(outer_fit.last_state, fold_dir / "last.pt")
            write_csv(fold_dir / "history.csv", outer_fit.history)
            write_json(fold_dir / "result.json", result)
            np.savez_compressed(fold_dir / "outer_test_predictions.npz", **prediction)
            write_run_artifact_manifest(
                fold_dir,
                required_files=FOLD_REQUIRED_FILES,
            )
            del (
                inner_model,
                outer_model,
                selection_fit,
                outer_fit,
                selection_evaluation,
                evaluation,
                inner_train_rates,
                inner_validation_rates,
                outer_train_rates,
                outer_test_rates,
            )
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        fold_rows.append(result)
        oof_parts.append(prediction)
        total_seconds += float(result["elapsed_seconds"])
        total_steps += int(result["optimizer_steps"])
        parameters = int(result["parameters"])
        fold_trials = int(prediction["labels"].size)
        if result.get("binary_spike_rate") is not None:
            spike_weighted_sum += float(result["binary_spike_rate"]) * fold_trials
            spike_trials += fold_trials
        if result.get("final_activity_nonzero_rate") is not None:
            activity_nonzero_weighted_sum += (
                float(result["final_activity_nonzero_rate"]) * fold_trials
            )
            activity_absolute_weighted_sum += (
                float(result["final_activity_absolute_mean"]) * fold_trials
            )
            activity_trials += fold_trials
        print(
            f"__V8_{stage}_FOLD_DONE__ "
            f"variant={variant_name} subject={subject} seed={seed} fold={fold_index} "
            f"kappa={float(result['kappa']):.6f}",
            flush=True,
        )

    indices, logits, labels = merge_oof_predictions(oof_parts, labels=bundle.y)
    prefixes = _merge_prefix_predictions(oof_parts, count=len(bundle.y))
    probabilities = _probabilities(logits)
    predictions = probabilities.argmax(axis=1)
    pooled = classification_metrics(labels, predictions, n_classes=4)
    endpoint_metrics = [
        classification_metrics(labels, prefixes[:, index].argmax(axis=1), n_classes=4)
        for index in range(prefixes.shape[1])
    ]
    write_trial_predictions(
        run_dir,
        logits=logits,
        probabilities=probabilities,
        pred=predictions,
        label=labels,
        subject=[bundle.metadata[int(index)]["subject"] for index in indices],
        session="T",
        run=[bundle.metadata[int(index)]["run"] for index in indices],
        trial_id=[bundle.metadata[int(index)]["trial_id"] for index in indices],
        seed=seed,
        model=f"v8_{variant_name}_{stage.lower()}_oof",
    )
    np.savez_compressed(
        run_dir / "prefix_predictions.npz",
        indices=indices,
        logits=prefixes,
        labels=labels,
        endpoint_seconds=np.asarray(model_config["endpoint_seconds"], dtype=np.float32),
    )
    metrics = {
        "status": "completed",
        "stage": stage,
        "protocol": config["protocol"],
        "model": "v8_accuracy_first",
        "variant": variant_name,
        "subject": subject,
        "seed": int(seed),
        "accuracy": pooled["accuracy"],
        "balanced_accuracy": pooled["balanced_accuracy"],
        "kappa": pooled["kappa"],
        "macro_f1": pooled["macro_f1"],
        "endpoint_metrics": endpoint_metrics,
        "binary_spike_rate": (
            spike_weighted_sum / spike_trials if spike_trials else None
        ),
        "final_activity_nonzero_rate": (
            activity_nonzero_weighted_sum / activity_trials if activity_trials else None
        ),
        "final_activity_absolute_mean": (
            activity_absolute_weighted_sum / activity_trials if activity_trials else None
        ),
        "fold_accuracy_mean": float(np.mean([row["accuracy"] for row in fold_rows])),
        "fold_kappa_mean": float(np.mean([row["kappa"] for row in fold_rows])),
        "parameters": parameters,
        "optimizer_steps": total_steps,
        "train_seconds": total_seconds,
        "selection_session": "T",
        "evaluation_scope": "Session-T nested outer-run OOF test",
        "delay_auxiliary_enabled": False,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
        "run_fingerprint": fingerprint["combined_sha256"],
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "runtime_status.json",
        {"status": "completed", "completed_at": time.time(), "session_e_accessed": False},
    )
    (run_dir / "stdout.log").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    write_run_artifact_manifest(run_dir, required_files=_required_files(n_folds))
    return metrics


def _variant_smoke(
    output: Path,
    model_config: dict[str, Any],
    variants: dict[str, dict[str, Any]],
    device: str,
    build_variant_fn: Callable[..., V8AccuracyFirstModel] | None = None,
    stage: str = "E2",
) -> None:
    report: dict[str, Any] = {"status": "passed", "stage": stage, "variants": {}}
    builder = _build_variant if build_variant_fn is None else build_variant_fn
    for index, (name, overrides) in enumerate(variants.items()):
        model = builder(model_config, name, overrides, seed=index).to(device)
        model.set_training_gain(torch.ones(model.n_bands, model.n_channels))
        fast_steps = model.endpoint_samples[-1]
        fast = torch.randn(
            2, model.n_bands, model.n_channels, fast_steps, device=device
        ).to(torch.complex64)
        slow = torch.randn(
            2, model.n_bands, model.n_channels, fast_steps // 2, device=device
        )
        output_value = model.forward_rate_features(fast, slow, delay_override="off")
        loss = output_value["logits"].square().mean()
        loss.backward()
        finite = bool(torch.isfinite(output_value["logits"]).all()) and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        if not finite:
            raise FloatingPointError(f"V8 {stage} variant smoke failed for {name}")
        report["variants"][name] = {
            "parameters": model.parameter_count,
            "trainable_parameters": model.trainable_parameter_count,
            "logit_shape": list(output_value["logits"].shape),
            "prefix_shape": list(output_value["prefix_logits"].shape),
            "finite": finite,
        }
        del model, fast, slow, output_value, loss
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    write_json(output / "variant_smoke.json", report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/experiments/v8_e2_zero_delay.yaml")
    parser.add_argument("--model-config", default="configs/models/v8_accuracy_first.yaml")
    parser.add_argument("--variants", default="")
    parser.add_argument("--subjects", default="")
    parser.add_argument("--screening-seed", type=int, default=None)
    parser.add_argument("--confirmation-seeds", default="")
    parser.add_argument("--confirmation-top-k", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-smoke-only", action="store_true")
    args = parser.parse_args()

    configure_cache_env()
    output = ensure_dir(Path(args.output).resolve())
    data_root = Path(args.data).resolve()
    config = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    model_config = yaml.safe_load(
        Path(args.model_config).resolve().read_text(encoding="utf-8")
    )
    if config.get("stage") != "development" or config["data_access"].get(
        "heldout_session_e_accessed"
    ):
        raise RuntimeError("V8 E2 must keep Session E locked")
    variants = dict(config["variants"])
    if args.variants:
        selected = _csv(args.variants)
        unknown = sorted(set(selected) - set(variants))
        if unknown:
            raise ValueError(f"unknown V8 E2 variants: {unknown}")
        variants = {name: variants[name] for name in selected}
    subjects = _csv(args.subjects, int) if args.subjects else list(config["subjects"])
    screening_seed = int(
        args.screening_seed if args.screening_seed is not None else config["screening_seed"]
    )
    confirmation_seeds = (
        _csv(args.confirmation_seeds, int)
        if args.confirmation_seeds
        else list(config["confirmation_seeds"])
    )
    top_k = int(
        args.confirmation_top_k
        if args.confirmation_top_k is not None
        else config["confirmation_top_k"]
    )
    if screening_seed not in confirmation_seeds:
        raise ValueError("V8 E2 confirmation seeds must include the screening seed")
    if not 1 <= top_k <= len(variants):
        raise ValueError("V8 E2 confirmation_top_k is outside the active variant set")
    if args.max_epochs is not None:
        config["selection"]["max_epochs"] = int(args.max_epochs)
        config["selection"]["minimum_epochs"] = min(
            int(config["selection"]["minimum_epochs"]), int(args.max_epochs)
        )
        config["selection"]["minimum_outer_retrain_epochs"] = min(
            int(config["selection"]["minimum_outer_retrain_epochs"]), int(args.max_epochs)
        )
    if args.patience is not None:
        config["selection"]["patience"] = int(args.patience)
    if int(config["training"]["effective_batch_size"]) != int(
        config["training"]["batch_size"]
    ) * int(config["training"]["gradient_accumulation_steps"]):
        raise RuntimeError("V8 E2 effective batch size metadata is inconsistent")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")

    source_tree = collect_source_tree_manifest(ROOT)
    environment = _environment()
    write_json(output / "source_tree_manifest.json", source_tree)
    write_json(
        output / "source_tree_summary.json",
        {"files": len(source_tree), "sha256": source_tree_digest(source_tree)},
    )
    write_json(output / "heldout_lock_manifest.json", v8_heldout_lock_manifest())
    _variant_smoke(output, model_config, variants, args.device)
    if args.source_smoke_only:
        print(json.dumps({"status": "passed", "source_smoke_only": True}, indent=2))
        return

    bundles: dict[int, SubjectBundle] = {}
    reference_name = next(iter(variants))
    reference_overrides = variants[reference_name]
    for subject in subjects:
        subject_path = _subject_file(data_root, int(subject))
        data_sha256 = file_sha256(subject_path)
        data = load_processed_npz(subject_path)
        x, y, metadata, access_manifest = session_t_development_view(data)
        nested = _nested_folds(
            metadata,
            n_splits=int(config["selection"]["n_splits"]),
            split_seed=int(config["selection"]["split_seed"]),
        )
        split_manifest = _split_manifest(
            metadata,
            nested,
            subject=int(subject),
            split_seed=int(config["selection"]["split_seed"]),
        )
        reference = _build_variant(
            model_config, reference_name, reference_overrides, seed=0
        )
        base_rates, cache_manifest = _load_or_build_base_rates(
            output=output,
            subject=int(subject),
            subject_path=subject_path,
            data_sha256=data_sha256,
            x=x,
            metadata=metadata,
            model=reference,
            preprocessing=dict(config["preprocessing"]),
            canary_train_indices=nested[0][0],
            device=args.device,
        )
        bundles[int(subject)] = SubjectBundle(
            subject_path=subject_path,
            data_sha256=data_sha256,
            x=x,
            y=y,
            metadata=metadata,
            access_manifest=access_manifest,
            nested_folds=nested,
            split_manifest=split_manifest,
            base_rates=base_rates,
            cache_manifest=cache_manifest,
        )
        del data, reference
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    for variant_name, overrides in variants.items():
        for subject in subjects:
            row = _run_one(
                output=output,
                config=config,
                model_config=model_config,
                source_tree=source_tree,
                environment=environment,
                bundle=bundles[int(subject)],
                variant_name=variant_name,
                variant_overrides=overrides,
                seed=screening_seed,
                device=args.device,
            )
            rows.append(row)
            write_csv(output / "summary.csv", rows)

    ranking_input = [{**row, "model": row["variant"]} for row in rows]
    ranking = rank_screening_models(
        ranking_input, subjects=subjects, screening_seed=screening_seed
    )
    write_csv(output / "screening_leaderboard.csv", ranking)
    ranked_variants = [row["model"] for row in ranking[:top_k]]
    required_variants = list(config.get("required_confirmation_variants", []))
    unknown_required = sorted(set(required_variants) - set(variants))
    if unknown_required:
        raise ValueError(
            f"required E2 confirmation variants are inactive: {unknown_required}"
        )
    selected_variants = list(dict.fromkeys(ranked_variants + required_variants))
    write_json(
        output / "confirmation_selection.json",
        {
            "selection_scope": "Session-T nested outer-run OOF only",
            "ranking_metric": config["selection"]["ranking_metric"],
            "top_k": top_k,
            "ranked_variants": ranked_variants,
            "required_variants": required_variants,
            "variants": selected_variants,
            "confirmation_seeds": confirmation_seeds,
            "session_e_accessed": False,
        },
    )
    for variant_name in selected_variants:
        for seed in confirmation_seeds:
            if int(seed) == screening_seed:
                continue
            for subject in subjects:
                row = _run_one(
                    output=output,
                    config=config,
                    model_config=model_config,
                    source_tree=source_tree,
                    environment=environment,
                    bundle=bundles[int(subject)],
                    variant_name=variant_name,
                    variant_overrides=variants[variant_name],
                    seed=int(seed),
                    device=args.device,
                )
                rows.append(row)
                write_csv(output / "summary.csv", rows)
    rows = sorted(
        rows, key=lambda row: (row["variant"], int(row["subject"]), int(row["seed"]))
    )
    write_csv(output / "summary.csv", rows)
    write_json(
        output / "campaign_status.json",
        {
            "status": "completed",
            "stage": "E2",
            "protocol": config["protocol"],
            "screened_variants": list(variants),
            "subjects": subjects,
            "screening_seed": screening_seed,
            "confirmed_variants": selected_variants,
            "confirmation_seeds": confirmation_seeds,
            "runs": len(rows),
            "session_e_accessed": False,
            "source_tree_sha256": source_tree_digest(source_tree),
        },
    )
    print(json.dumps({"status": "completed", "runs": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
