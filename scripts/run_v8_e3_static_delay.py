#!/usr/bin/env python3
"""Run the V8 static within-band delay intervention against matched zero."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    sha256_fingerprint,
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
    write_trial_predictions,
)
from dpc_snn.experiments.v8_baselines import (  # noqa: E402
    merge_oof_predictions,
    session_t_development_view,
)
from dpc_snn.experiments.v8_delay_prior import (  # noqa: E402
    V8_PRIOR_FIELDS,
    V8DelayEvidenceRejected,
    fit_v8_fold_delay_prior,
    fit_v8_fold_delay_prior_ensemble,
    v8_delay_prior_digest,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    build_v8_run_fingerprint,
    collect_source_tree_manifest,
    mapping_sha256,
    source_tree_digest,
    validate_v8_resume_fingerprint,
    v8_heldout_lock_manifest,
    write_v8_fingerprint,
)
from dpc_snn.experiments.v8_training import (  # noqa: E402
    V8CachedRates,
    apply_v8_fold_gain,
    fit_v8,
    fit_v8_gain_from_cached_rates,
    load_v8_rates,
    predict_v8,
    seed_v8,
)
from dpc_snn.models.build import build_model  # noqa: E402
from dpc_snn.models.v8_accuracy_first import V8AccuracyFirstModel  # noqa: E402
from dpc_snn.utils.io import (  # noqa: E402
    ensure_dir,
    read_json,
    save_npz,
    write_csv,
    write_json,
)
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from run_v8_e2_zero_delay import (  # noqa: E402
    SubjectBundle,
    _environment,
    _fit_kwargs,
    _merge_prefix_predictions,
    _nested_folds,
    _probabilities,
    _required_files as e2_required_files,
    _split_manifest,
    _subject_file,
)


PRIOR_FILES = (
    "manifest.json",
    "fingerprint.json",
    "prior.npz",
    "evidence.npz",
    "summary.json",
)
REJECTED_PRIOR_FILES = (
    "manifest.json",
    "fingerprint.json",
    "rejected_evidence.npz",
    "rejected_summary.json",
)
FOLD_FILES = (
    "manifest.json",
    "selection_best.pt",
    "selection_history.csv",
    "selection_result.json",
    "best.pt",
    "history.csv",
    "result.json",
    "paired_predictions.npz",
    *(f"inner_prior/{name}" for name in PRIOR_FILES),
    *(f"outer_prior/{name}" for name in PRIOR_FILES),
)
RUN_FILES = (
    "manifest.json",
    "resolved_config.yaml",
    "source_fingerprint.json",
    "source_tree_manifest.json",
    "data_access_manifest.json",
    "split_manifest.json",
    "metrics.json",
    "full_predictions.npz",
    "full_predictions.csv",
    "zero_predictions.npz",
    "zero_predictions.csv",
    "full_prefix_predictions.npz",
    "zero_prefix_predictions.npz",
    "runtime_status.json",
    "stdout.log",
    "stderr.log",
)


def _csv(value: str, cast: Any = str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _required_files(n_folds: int) -> tuple[str, ...]:
    return RUN_FILES + tuple(
        f"fold_{fold}/{name}" for fold in range(n_folds) for name in FOLD_FILES
    )


def _delay_model_config(
    model_config: dict[str, Any], delay: dict[str, Any]
) -> dict[str, Any]:
    return {
        **model_config,
        "delay_auxiliary_enabled": True,
        "delay_maximum_routes": int(delay["maximum_routes"]),
        "delay_maximum_samples": int(delay["maximum_delay_samples"]),
        "delay_allow_cross_band": bool(delay["allow_cross_band"]),
        "delay_signal_mode": str(delay["signal_mode"]),
        "delay_contextual_residual_enabled": bool(
            delay["contextual_residual_enabled"]
        ),
        "delay_contextual_residual_bound": float(delay["contextual_residual_bound"]),
        "delay_phase_residual_enabled": bool(delay["phase_residual_enabled"]),
        "delay_phase_residual_bound": float(delay["phase_residual_bound"]),
        "delay_fusion_initial": float(delay["fusion_initial"]),
        "delay_fusion_bound": float(delay["fusion_bound"]),
    }


def _build_delay_model(
    model_config: dict[str, Any], delay: dict[str, Any], *, seed: int
) -> V8AccuracyFirstModel:
    seed_v8(seed)
    model = build_model("v8_accuracy_first", _delay_model_config(model_config, delay))
    if not isinstance(model, V8AccuracyFirstModel) or model.delay_auxiliary is None:
        raise TypeError("V8 E3 did not construct a delay-enabled accuracy-first model")
    if model.decoder is None or model.decoder.decoder_kind != "ann":
        raise RuntimeError("V8 E3 delay stage must use the E2 matched ANN decoder")
    if float(model.delay_fusion_scale.detach()) != 0.0:
        raise RuntimeError("V8 E3 delay fusion must start at the exact E2 parent")
    return model


def _delay_trainable_names(delay: dict[str, Any]) -> set[str]:
    allowed = {"delay_fusion_raw"}
    if bool(delay["contextual_residual_enabled"]):
        allowed.update(
            {
                "delay_auxiliary.context_weight",
                "delay_auxiliary.context_bias",
                "delay_auxiliary.context_route_raw",
            }
        )
    if bool(delay["phase_residual_enabled"]):
        allowed.add("delay_auxiliary.phase_residual_raw")
    return allowed


def _freeze_delay_only(
    model: V8AccuracyFirstModel,
    delay: dict[str, Any] | None = None,
) -> None:
    if model.delay_auxiliary is None:
        raise RuntimeError("cannot configure E3 training without a delay auxiliary")
    if delay is None:
        delay = {
            "contextual_residual_enabled": (
                model.delay_auxiliary.contextual_residual_enabled
            ),
            "phase_residual_enabled": model.delay_auxiliary.phase_residual_enabled,
        }
    allowed = _delay_trainable_names(delay)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in allowed)
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    if set(trainable) != allowed:
        raise RuntimeError(f"unexpected E3 trainable parameters: {trainable}")


def _load_parent_state(model: V8AccuracyFirstModel, path: Path) -> str:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"parent checkpoint is not a state dictionary: {path}")
    model_keys = set(model.state_dict())
    state_keys = set(state)
    expected_missing = model_keys - state_keys
    if any(
        name != "delay_fusion_raw" and not name.startswith("delay_auxiliary.")
        for name in expected_missing
    ):
        raise RuntimeError(f"parent checkpoint misses shared model keys: {expected_missing}")
    unexpected = state_keys - model_keys
    if unexpected:
        raise RuntimeError(f"parent checkpoint has unexpected model keys: {unexpected}")
    incompatible = model.load_state_dict(state, strict=False)
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError("parent checkpoint compatibility changed during load")
    return sha256_fingerprint(mapping_sha256(state))


def _load_parent_rates(
    parent_root: Path,
    *,
    subject: int,
    data_sha256: str,
    model: V8AccuracyFirstModel,
) -> tuple[V8CachedRates, dict[str, Any]]:
    directory = parent_root / "shared_physical_rates" / f"subject_{subject:02d}"
    cache_path = directory / "unit_gain_rates.pt"
    manifest_path = directory / "manifest.json"
    if not cache_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"verified E2 parent cache is missing for subject {subject}")
    manifest = read_json(manifest_path)
    if manifest.get("data_sha256") != data_sha256:
        raise RuntimeError("E2 parent cache data hash differs from E3 data")
    if manifest.get("cache_file_sha256") != file_sha256(cache_path):
        raise RuntimeError("E2 parent cache file hash mismatch")
    if manifest.get("physical_frontend_fingerprint") != model.physical_frontend_fingerprint():
        raise RuntimeError("E2 parent cache physical front end differs from E3")
    rates = load_v8_rates(cache_path, model)
    if not torch.equal(rates.gain, torch.ones_like(rates.gain)):
        raise RuntimeError("E2 parent cache is not unit gain")
    return rates, manifest


def _parent_run(
    parent_root: Path,
    parent_variant: str,
    subject: int,
    seed: int,
    n_folds: int,
) -> Path:
    run = parent_root / parent_variant / f"subject_{subject:02d}" / f"seed_{seed}"
    validate_run_artifact_manifest(
        run,
        required_files=e2_required_files(n_folds),
        verify_hashes=True,
        verify_prediction_schema=True,
    )
    metrics = read_json(run / "metrics.json")
    if (
        metrics.get("status") != "completed"
        or metrics.get("variant") != parent_variant
        or bool(metrics.get("delay_auxiliary_enabled"))
        or bool(metrics.get("session_e_accessed"))
    ):
        raise RuntimeError(f"invalid E2 parent run: {run}")
    return run


def _parent_artifacts(run: Path, n_folds: int) -> dict[str, str]:
    paths = {
        "run_fingerprint": run / "source_fingerprint.json",
        "metrics": run / "metrics.json",
        "predictions": run / "predictions.npz",
    }
    for fold in range(n_folds):
        directory = run / f"fold_{fold}"
        for name in (
            "selection_best.pt",
            "selection_predictions.npz",
            "best.pt",
            "outer_test_predictions.npz",
            "result.json",
        ):
            paths[f"fold_{fold}/{name}"] = directory / name
    return {name: file_sha256(path) for name, path in sorted(paths.items())}


def _prior_payload(
    *,
    scope: str,
    trial_ids: list[str],
    gain: torch.Tensor,
    rates: V8CachedRates,
    evidence_rates: V8CachedRates | None = None,
    evidence_identity: dict[str, Any] | None = None,
    prior_config: dict[str, Any],
    source_sha256: str,
    seed: int,
    ensemble_seeds: Sequence[int] | None = None,
    split_strata: Sequence[str] | None = None,
) -> dict[str, Any]:
    gain_array = gain.detach().cpu().contiguous().numpy()
    audit_rates = rates if evidence_rates is None else evidence_rates
    return {
        "schema": "dpc-snn-v8-fold-delay-prior/v1",
        "scope": scope,
        "trial_ids": trial_ids,
        "gain_sha256": sha256_fingerprint(gain_array.tolist()),
        "physical_frontend_fingerprint": rates.physical_frontend_fingerprint,
        "rate_shape": list(rates.fast.shape),
        "transport_frontend_fingerprint": rates.physical_frontend_fingerprint,
        "transport_rate_shape": list(rates.fast.shape),
        "evidence_frontend_fingerprint": audit_rates.physical_frontend_fingerprint,
        "evidence_rate_shape": list(audit_rates.fast.shape),
        "evidence_identity": evidence_identity or {"space": "same_as_transport"},
        "prior_config": prior_config,
        "source_tree_sha256": source_sha256,
        "seed": int(seed),
        "ensemble_seeds": (
            [int(value) for value in ensemble_seeds]
            if ensemble_seeds is not None
            else None
        ),
        "split_half_scheme": (
            "run_class_stratified_deterministic_alternation"
            if split_strata is not None
            else "legacy_trial_order_alternation"
        ),
        "split_strata": (
            [str(value) for value in split_strata]
            if split_strata is not None
            else None
        ),
        "heldout_data_accessed": False,
    }


def _load_prior(directory: Path, expected_fingerprint: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    validate_run_artifact_manifest(
        directory,
        required_files=PRIOR_FILES,
        verify_hashes=True,
        verify_prediction_schema=False,
    )
    fingerprint = read_json(directory / "fingerprint.json")
    if fingerprint.get("combined_sha256") != expected_fingerprint:
        raise RuntimeError("fold-local delay prior fingerprint mismatch")
    with np.load(directory / "prior.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(V8_PRIOR_FIELDS):
            raise RuntimeError("fold-local sparse prior fields changed")
        prior = {name: torch.from_numpy(archive[name]) for name in archive.files}
    summary = read_json(directory / "summary.json")
    if summary.get("prior_sha256") != v8_delay_prior_digest(prior):
        raise RuntimeError("fold-local sparse prior digest mismatch")
    if not bool(summary.get("evidence_pipeline_passed")):
        raise RuntimeError("cached fold-local evidence did not pass its controls")
    return prior, summary


def _raise_cached_prior_rejection(directory: Path, expected_fingerprint: str) -> None:
    manifest = validate_run_artifact_manifest(
        directory,
        required_files=REJECTED_PRIOR_FILES,
        verify_hashes=True,
        verify_prediction_schema=False,
    )
    if manifest.get("status") != "rejected":
        raise RuntimeError("cached fold-local rejection manifest has invalid status")
    fingerprint = read_json(directory / "fingerprint.json")
    if fingerprint.get("combined_sha256") != expected_fingerprint:
        raise RuntimeError("cached fold-local rejection fingerprint mismatch")
    summary = read_json(directory / "rejected_summary.json")
    if summary.get("input_fingerprint") != expected_fingerprint:
        raise RuntimeError("cached fold-local rejection summary fingerprint mismatch")
    with np.load(directory / "rejected_evidence.npz", allow_pickle=False) as archive:
        evidence = {name: archive[name] for name in archive.files}
    raise V8DelayEvidenceRejected(
        str(summary.get("rejection_reason", "fold-local delay evidence was rejected")),
        arrays=evidence,
        summary=summary,
    )


def _load_or_fit_prior(
    directory: Path,
    *,
    scope: str,
    trial_ids: list[str],
    rates: V8CachedRates,
    evidence_rates: V8CachedRates | None = None,
    evidence_identity: dict[str, Any] | None = None,
    gain: torch.Tensor,
    model_config: dict[str, Any],
    delay_config: dict[str, Any],
    source_sha256: str,
    seed: int,
    ensemble_seeds: Sequence[int] | None = None,
    split_strata: Sequence[str] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    prior_config = dict(delay_config["prior"])
    payload = _prior_payload(
        scope=scope,
        trial_ids=trial_ids,
        gain=gain,
        rates=rates,
        evidence_rates=evidence_rates,
        evidence_identity=evidence_identity,
        prior_config=prior_config,
        source_sha256=source_sha256,
        seed=seed,
        ensemble_seeds=ensemble_seeds,
        split_strata=split_strata,
    )
    fingerprint = sha256_fingerprint(payload)
    if all((directory / name).is_file() for name in PRIOR_FILES):
        return _load_prior(directory, fingerprint)
    if all((directory / name).is_file() for name in REJECTED_PRIOR_FILES):
        _raise_cached_prior_rejection(directory, fingerprint)
    ensure_dir(directory)
    try:
        fit_kwargs = {
            "evidence_rates": evidence_rates,
            "analytic_representation": str(
                (evidence_identity or {}).get(
                    "analytic_representation",
                    "v8_exact_online_physical_fast_after_fold_train_fixed_gain",
                )
            ),
            "band_edges_hz": model_config["band_edges_hz"],
            "task_seconds": float(
                model_config["task_tmax"] - model_config["task_tmin"]
            ),
            "maximum_routes": int(delay_config["maximum_routes"]),
            "maximum_delay": int(delay_config["maximum_delay_samples"]),
            "route_scope": str(delay_config["route_scope"]),
            "bootstrap_samples": int(prior_config["bootstrap_samples"]),
            "grid_oversample": int(prior_config["grid_oversample"]),
            "minimum_bayes_factor": float(prior_config["minimum_bayes_factor"]),
            "minimum_bootstrap_frequency": float(
                prior_config["minimum_bootstrap_frequency"]
            ),
            "minimum_direction_probability": float(
                prior_config["minimum_direction_probability"]
            ),
            "split_strata": split_strata,
            "surrogate_replicates": int(
                prior_config.get("surrogate_replicates", 1)
            ),
        }
        if ensemble_seeds is None:
            prior, summary, evidence = fit_v8_fold_delay_prior(
                rates,
                seed=int(seed),
                **fit_kwargs,
            )
        else:
            prior, summary, evidence = fit_v8_fold_delay_prior_ensemble(
                rates,
                seeds=ensemble_seeds,
                minimum_pass_fraction=float(
                    prior_config.get("minimum_replicate_pass_fraction", 0.80)
                ),
                consensus_frequency=float(
                    prior_config.get("minimum_consensus_frequency", 0.80)
                ),
                minimum_consensus_edges=int(
                    prior_config.get("minimum_consensus_edges", 3)
                ),
                minimum_median_edge_jaccard=float(
                    prior_config.get("minimum_median_edge_jaccard", 0.50)
                ),
                minimum_edge_jaccard_floor=float(
                    prior_config.get("minimum_edge_jaccard_floor", 0.25)
                ),
                minimum_median_delay_correlation=float(
                    prior_config.get("minimum_median_delay_correlation", 0.50)
                ),
                **fit_kwargs,
            )
    except V8DelayEvidenceRejected as exc:
        rejected_summary = {
            **exc.summary,
            "status": "rejected",
            "rejection_reason": str(exc),
            "input_fingerprint": fingerprint,
            "input_payload": payload,
        }
        save_npz(directory / "rejected_evidence.npz", **exc.arrays)
        write_json(
            directory / "fingerprint.json",
            {**payload, "combined_sha256": fingerprint},
        )
        write_json(directory / "rejected_summary.json", rejected_summary)
        write_run_artifact_manifest(
            directory,
            required_files=REJECTED_PRIOR_FILES,
            status="rejected",
        )
        raise V8DelayEvidenceRejected(
            str(exc),
            arrays=exc.arrays,
            summary=rejected_summary,
        ) from exc
    summary = {
        **summary,
        "input_fingerprint": fingerprint,
        "input_payload": payload,
    }
    save_npz(
        directory / "prior.npz",
        **{name: value.detach().cpu().numpy() for name, value in prior.items()},
    )
    save_npz(directory / "evidence.npz", **evidence)
    write_json(directory / "fingerprint.json", {**payload, "combined_sha256": fingerprint})
    write_json(directory / "summary.json", summary)
    write_run_artifact_manifest(directory, required_files=PRIOR_FILES)
    return _load_prior(directory, fingerprint)


def _load_fold(
    directory: Path,
    *,
    indices: np.ndarray,
    labels: np.ndarray,
    endpoints: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    if not all((directory / name).is_file() for name in FOLD_FILES):
        return None
    validate_run_artifact_manifest(
        directory,
        required_files=FOLD_FILES,
        verify_hashes=True,
        verify_prediction_schema=False,
    )
    with np.load(directory / "paired_predictions.npz", allow_pickle=False) as archive:
        prediction = {name: archive[name] for name in archive.files}
    expected = {
        "indices",
        "labels",
        "full_logits",
        "zero_logits",
        "full_prefix_logits",
        "zero_prefix_logits",
    }
    if set(prediction) != expected:
        raise RuntimeError("cached E3 paired prediction schema changed")
    if not np.array_equal(prediction["indices"], indices) or not np.array_equal(
        prediction["labels"], labels
    ):
        raise RuntimeError("cached E3 fold identities differ from the active split")
    for name in ("full_logits", "zero_logits"):
        if prediction[name].shape != (indices.size, 4):
            raise RuntimeError("cached E3 fold logits have an invalid shape")
    for name in ("full_prefix_logits", "zero_prefix_logits"):
        if prediction[name].shape != (indices.size, endpoints, 4):
            raise RuntimeError("cached E3 prefix logits have an invalid shape")
    if not all(np.isfinite(value).all() for value in prediction.values()):
        raise RuntimeError("cached E3 fold prediction contains NaN or Inf")
    return read_json(directory / "result.json"), prediction


def _classification(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    return classification_metrics(labels, logits.argmax(axis=1), n_classes=4)


def _run_one(
    *,
    output: Path,
    parent_root: Path,
    parent_variant: str,
    config: dict[str, Any],
    model_config: dict[str, Any],
    source_tree: dict[str, str],
    environment: dict[str, Any],
    bundle: SubjectBundle,
    seed: int,
    device: str,
) -> dict[str, Any]:
    subject = int(bundle.metadata[0]["subject"])
    parent_run = _parent_run(
        parent_root, parent_variant, subject, seed, len(bundle.nested_folds)
    )
    delay = dict(config["delay"])
    resolved = {
        **config,
        "active_subject": subject,
        "active_seed": int(seed),
        "resolved_model": _delay_model_config(model_config, delay),
    }
    parent_hashes = _parent_artifacts(parent_run, len(bundle.nested_folds))
    fingerprint = build_v8_run_fingerprint(
        resolved_run_config=resolved,
        source_tree=source_tree,
        data={
            bundle.subject_path.name: bundle.data_sha256,
            "access": bundle.access_manifest,
            "physical_rate_cache": bundle.cache_manifest,
        },
        split=bundle.split_manifest,
        augmentation=config["augmentation"],
        prior={
            "policy": delay["prior"],
            "delay_stage": delay["stage"],
            "fit_scope": "active_inner_or_outer_training_partition_only",
        },
        checkpoint={
            "parent_stage": "E2",
            "parent_variant": parent_variant,
            "parent_artifacts": parent_hashes,
            "residual_epoch_policy": config["selection"],
        },
        environment=environment,
    )
    run_dir = ensure_dir(output / f"subject_{subject:02d}" / f"seed_{seed}")
    fingerprint_path = run_dir / "source_fingerprint.json"
    required = _required_files(len(bundle.nested_folds))
    if (run_dir / "manifest.json").is_file():
        validate_v8_resume_fingerprint(fingerprint_path, fingerprint)
        validate_run_artifact_manifest(
            run_dir,
            required_files=required,
            verify_hashes=True,
            verify_prediction_schema=False,
        )
        return read_json(run_dir / "metrics.json")
    if fingerprint_path.is_file():
        validate_v8_resume_fingerprint(fingerprint_path, fingerprint)
    else:
        write_v8_fingerprint(fingerprint_path, fingerprint)
        write_json(run_dir / "source_tree_manifest.json", source_tree)
        write_json(run_dir / "data_access_manifest.json", bundle.access_manifest)
        write_json(run_dir / "split_manifest.json", bundle.split_manifest)
        (run_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )
    write_json(
        run_dir / "runtime_status.json",
        {"status": "running", "started_at": time.time(), "session_e_accessed": False},
    )

    selection = dict(config["selection"])
    training = dict(config["training"])
    fit_kwargs = _fit_kwargs(training, dict(config["augmentation"]))
    source_sha256 = source_tree_digest(source_tree)
    endpoint_count = len(model_config["endpoint_seconds"])
    fold_rows: list[dict[str, Any]] = []
    full_parts: list[dict[str, np.ndarray]] = []
    zero_parts: list[dict[str, np.ndarray]] = []
    for fold_index, (
        outer_train,
        outer_test,
        inner_train,
        inner_validation,
        inner_validation_run,
    ) in enumerate(bundle.nested_folds):
        fold_dir = ensure_dir(run_dir / f"fold_{fold_index}")
        cached = _load_fold(
            fold_dir,
            indices=outer_test,
            labels=bundle.y[outer_test],
            endpoints=endpoint_count,
        )
        if cached is not None:
            result, prediction = cached
        else:
            parent_fold = parent_run / f"fold_{fold_index}"
            fold_seed = int(seed) * 100_003 + fold_index * 1_009 + 8_000_021
            inner_gain = fit_v8_gain_from_cached_rates(bundle.base_rates, inner_train)
            parent_inner_result = read_json(parent_fold / "selection_result.json")
            if np.max(np.abs(np.asarray(parent_inner_result["gain"]) - inner_gain.numpy())) > 1e-6:
                raise RuntimeError("E3 inner gain does not reproduce the E2 parent")
            inner_train_rates = apply_v8_fold_gain(
                bundle.base_rates.subset(inner_train), inner_gain
            )
            inner_validation_rates = apply_v8_fold_gain(
                bundle.base_rates.subset(inner_validation), inner_gain
            )
            inner_prior, inner_summary = _load_or_fit_prior(
                fold_dir / "inner_prior",
                scope="inner_training_fold_only",
                trial_ids=[bundle.metadata[int(index)]["trial_id"] for index in inner_train],
                rates=inner_train_rates,
                gain=inner_gain,
                model_config=model_config,
                delay_config=delay,
                source_sha256=source_sha256,
                seed=fold_seed + 101,
            )
            inner_model = _build_delay_model(model_config, delay, seed=fold_seed)
            inner_parent_sha = _load_parent_state(
                inner_model, parent_fold / "selection_best.pt"
            )
            inner_model.load_fold_delay_prior(**inner_prior)
            _freeze_delay_only(inner_model, delay)
            inner_fit = fit_v8(
                inner_model,
                inner_train_rates,
                bundle.y[inner_train],
                validation_rates=inner_validation_rates,
                validation_labels=bundle.y[inner_validation],
                device=device,
                seed=fold_seed,
                epochs=int(selection["max_residual_epochs"]),
                patience=int(selection["patience"]),
                minimum_epochs=int(selection["minimum_epochs_before_stopping"]),
                scheduler_epochs=int(selection["max_residual_epochs"]),
                delay_override="full",
                include_initial_validation=bool(selection["include_parent_epoch_zero"]),
                run_label=f"V8-E3-select:S{subject}:seed{seed}:fold{fold_index}",
                **fit_kwargs,
            )
            inner_full = predict_v8(
                inner_fit.model,
                inner_validation_rates,
                bundle.y[inner_validation],
                device=device,
                batch_size=int(training["batch_size"]),
                delay_override="full",
            )
            inner_zero = predict_v8(
                inner_fit.model,
                inner_validation_rates,
                bundle.y[inner_validation],
                device=device,
                batch_size=int(training["batch_size"]),
                delay_override="zero",
            )
            if float(inner_full["delay_current_rms"] or 0.0) <= 0.0 or float(
                inner_zero["delay_current_rms"] or 0.0
            ) <= 0.0:
                raise RuntimeError("E3 full/zero inner controls must both carry current")
            selected_epoch = int(inner_fit.best_epoch)
            selection_result = {
                "fold": fold_index,
                "fold_seed": fold_seed,
                "inner_validation_run": inner_validation_run,
                "selected_residual_epoch": selected_epoch,
                "best_validation_kappa": inner_fit.best_metric,
                "full": _classification(inner_full["logits"], inner_full["labels"]),
                "zero": _classification(inner_zero["logits"], inner_zero["labels"]),
                "full_current_rms": inner_full["delay_current_rms"],
                "zero_current_rms": inner_zero["delay_current_rms"],
                "control_type": "same_weight_point_zero_posterior_intervention",
                "delay_fusion_scale": float(inner_fit.model.delay_fusion_scale.detach().cpu()),
                "parent_checkpoint_sha256": inner_parent_sha,
                "prior_sha256": inner_summary["prior_sha256"],
            }
            torch.save(inner_fit.best_state, fold_dir / "selection_best.pt")
            write_csv(fold_dir / "selection_history.csv", inner_fit.history)
            write_json(fold_dir / "selection_result.json", selection_result)

            outer_gain = fit_v8_gain_from_cached_rates(bundle.base_rates, outer_train)
            parent_outer_result = read_json(parent_fold / "result.json")
            if np.max(np.abs(np.asarray(parent_outer_result["outer_gain"]) - outer_gain.numpy())) > 1e-6:
                raise RuntimeError("E3 outer gain does not reproduce the E2 parent")
            outer_train_rates = apply_v8_fold_gain(
                bundle.base_rates.subset(outer_train), outer_gain
            )
            outer_test_rates = apply_v8_fold_gain(
                bundle.base_rates.subset(outer_test), outer_gain
            )
            outer_prior, outer_summary = _load_or_fit_prior(
                fold_dir / "outer_prior",
                scope="outer_training_fold_only",
                trial_ids=[bundle.metadata[int(index)]["trial_id"] for index in outer_train],
                rates=outer_train_rates,
                gain=outer_gain,
                model_config=model_config,
                delay_config=delay,
                source_sha256=source_sha256,
                seed=fold_seed + 211,
            )
            outer_seed = fold_seed + 1_000_003
            outer_model = _build_delay_model(model_config, delay, seed=outer_seed)
            outer_parent_sha = _load_parent_state(outer_model, parent_fold / "best.pt")
            outer_model.load_fold_delay_prior(**outer_prior)
            _freeze_delay_only(outer_model, delay)
            if selected_epoch > 0:
                outer_fit = fit_v8(
                    outer_model,
                    outer_train_rates,
                    bundle.y[outer_train],
                    validation_rates=None,
                    validation_labels=None,
                    device=device,
                    seed=outer_seed,
                    epochs=selected_epoch,
                    patience=selected_epoch,
                    minimum_epochs=selected_epoch,
                    fixed_epoch=selected_epoch,
                    scheduler_epochs=int(selection["max_residual_epochs"]),
                    delay_override="full",
                    run_label=f"V8-E3-outer:S{subject}:seed{seed}:fold{fold_index}",
                    **fit_kwargs,
                )
                outer_model = outer_fit.model
                outer_history = outer_fit.history
                optimizer_steps = outer_fit.optimizer_steps
                elapsed_seconds = outer_fit.elapsed_seconds
            else:
                outer_model.to(device)
                outer_history = [{"epoch": 0, "status": "parent_epoch_zero"}]
                optimizer_steps = 0
                elapsed_seconds = 0.0
            state_before = mapping_sha256(outer_model.state_dict())
            full = predict_v8(
                outer_model,
                outer_test_rates,
                bundle.y[outer_test],
                device=device,
                batch_size=int(training["batch_size"]),
                delay_override="full",
            )
            state_between = mapping_sha256(outer_model.state_dict())
            zero = predict_v8(
                outer_model,
                outer_test_rates,
                bundle.y[outer_test],
                device=device,
                batch_size=int(training["batch_size"]),
                delay_override="zero",
            )
            state_after = mapping_sha256(outer_model.state_dict())
            if state_before != state_between or state_before != state_after:
                raise RuntimeError("E3 model state changed during paired evaluation")
            if float(full["delay_current_rms"] or 0.0) <= 0.0 or float(
                zero["delay_current_rms"] or 0.0
            ) <= 0.0:
                raise RuntimeError("E3 full/zero outer controls must both carry current")
            full_metrics = _classification(full["logits"], full["labels"])
            zero_metrics = _classification(zero["logits"], zero["labels"])
            result = {
                "fold": fold_index,
                "fold_seed": fold_seed,
                "outer_seed": outer_seed,
                "inner_validation_run": inner_validation_run,
                "selected_residual_epoch": selected_epoch,
                "full": full_metrics,
                "zero": zero_metrics,
                "accuracy_delta": float(full_metrics["accuracy"] - zero_metrics["accuracy"]),
                "delay_fusion_scale": float(outer_model.delay_fusion_scale.detach().cpu()),
                "full_current_rms": full["delay_current_rms"],
                "zero_current_rms": zero["delay_current_rms"],
                "control_type": "same_weight_point_zero_posterior_intervention",
                "state_sha256": sha256_fingerprint(state_after),
                "parent_checkpoint_sha256": outer_parent_sha,
                "inner_prior_sha256": inner_summary["prior_sha256"],
                "outer_prior_sha256": outer_summary["prior_sha256"],
                "inner_prior_routes": inner_summary["selected_sparse_routes"],
                "outer_prior_routes": outer_summary["selected_sparse_routes"],
                "optimizer_steps": optimizer_steps,
                "elapsed_seconds": elapsed_seconds,
            }
            prediction = {
                "indices": outer_test.astype(np.int64),
                "labels": bundle.y[outer_test].astype(np.int64),
                "full_logits": full["logits"].astype(np.float32),
                "zero_logits": zero["logits"].astype(np.float32),
                "full_prefix_logits": full["prefix_logits"].astype(np.float32),
                "zero_prefix_logits": zero["prefix_logits"].astype(np.float32),
            }
            torch.save(outer_model.state_dict(), fold_dir / "best.pt")
            write_csv(fold_dir / "history.csv", outer_history)
            write_json(fold_dir / "result.json", result)
            save_npz(fold_dir / "paired_predictions.npz", **prediction)
            write_run_artifact_manifest(fold_dir, required_files=FOLD_FILES)
            del inner_model, inner_fit, outer_model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        fold_rows.append(result)
        full_parts.append(
            {
                "indices": prediction["indices"],
                "labels": prediction["labels"],
                "logits": prediction["full_logits"],
                "prefix_logits": prediction["full_prefix_logits"],
            }
        )
        zero_parts.append(
            {
                "indices": prediction["indices"],
                "labels": prediction["labels"],
                "logits": prediction["zero_logits"],
                "prefix_logits": prediction["zero_prefix_logits"],
            }
        )
        print(
            "__V8_E3_FOLD_DONE__ "
            f"subject={subject} seed={seed} fold={fold_index} "
            f"delta={100.0 * float(result['accuracy_delta']):.3f}pp",
            flush=True,
        )

    full_indices, full_logits, labels = merge_oof_predictions(full_parts, labels=bundle.y)
    zero_indices, zero_logits, zero_labels = merge_oof_predictions(zero_parts, labels=bundle.y)
    if not np.array_equal(full_indices, zero_indices) or not np.array_equal(labels, zero_labels):
        raise RuntimeError("E3 full/zero OOF identities diverged")
    full_prefix = _merge_prefix_predictions(full_parts, count=len(bundle.y))
    zero_prefix = _merge_prefix_predictions(zero_parts, count=len(bundle.y))
    full_probability = _probabilities(full_logits)
    zero_probability = _probabilities(zero_logits)
    full_prediction = full_probability.argmax(axis=1)
    zero_prediction = zero_probability.argmax(axis=1)
    full_metrics = classification_metrics(labels, full_prediction, n_classes=4)
    zero_metrics = classification_metrics(labels, zero_prediction, n_classes=4)
    trial_subject = [bundle.metadata[int(index)]["subject"] for index in full_indices]
    trial_run = [bundle.metadata[int(index)]["run"] for index in full_indices]
    trial_id = [bundle.metadata[int(index)]["trial_id"] for index in full_indices]
    write_trial_predictions(
        run_dir,
        logits=full_logits,
        probabilities=full_probability,
        pred=full_prediction,
        label=labels,
        subject=trial_subject,
        session="T",
        run=trial_run,
        trial_id=trial_id,
        seed=seed,
        model=f"v8_e3_{delay['stage']}_full",
        basename="full_predictions",
    )
    write_trial_predictions(
        run_dir,
        logits=zero_logits,
        probabilities=zero_probability,
        pred=zero_prediction,
        label=labels,
        subject=trial_subject,
        session="T",
        run=trial_run,
        trial_id=trial_id,
        seed=seed,
        model=f"v8_e3_{delay['stage']}_matched_zero",
        basename="zero_predictions",
    )
    save_npz(
        run_dir / "full_prefix_predictions.npz",
        indices=full_indices,
        logits=full_prefix,
        labels=labels,
        endpoint_seconds=np.asarray(model_config["endpoint_seconds"], dtype=np.float32),
    )
    save_npz(
        run_dir / "zero_prefix_predictions.npz",
        indices=zero_indices,
        logits=zero_prefix,
        labels=zero_labels,
        endpoint_seconds=np.asarray(model_config["endpoint_seconds"], dtype=np.float32),
    )
    metrics = {
        "status": "completed",
        "stage": "E3",
        "protocol": config["protocol"],
        "variant": str(delay["stage"]),
        "parent_variant": parent_variant,
        "subject": subject,
        "seed": int(seed),
        "full_accuracy": full_metrics["accuracy"],
        "zero_accuracy": zero_metrics["accuracy"],
        "accuracy_delta": float(full_metrics["accuracy"] - zero_metrics["accuracy"]),
        "full_macro_f1": full_metrics["macro_f1"],
        "zero_macro_f1": zero_metrics["macro_f1"],
        "full_kappa": full_metrics["kappa"],
        "zero_kappa": zero_metrics["kappa"],
        "positive_fold_deltas": int(sum(row["accuracy_delta"] > 0 for row in fold_rows)),
        "minimum_full_current_rms": float(
            min(row["full_current_rms"] for row in fold_rows)
        ),
        "minimum_zero_current_rms": float(
            min(row["zero_current_rms"] for row in fold_rows)
        ),
        "control_type": "same_weight_point_zero_posterior_intervention",
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
    write_run_artifact_manifest(run_dir, required_files=required)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--parent-e2", required=True)
    parser.add_argument("--parent-variant", default="full_ann")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/experiments/v8_e3_static_delay.yaml")
    parser.add_argument("--model-config", default="configs/models/v8_accuracy_first.yaml")
    parser.add_argument("--subjects", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    configure_cache_env()
    output = ensure_dir(Path(args.output).resolve())
    parent_root = Path(args.parent_e2).resolve()
    parent_variant = str(args.parent_variant)
    data_root = Path(args.data).resolve()
    config = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    model_config = yaml.safe_load(
        Path(args.model_config).resolve().read_text(encoding="utf-8")
    )
    if config.get("stage") != "development" or bool(
        config["data_access"].get("heldout_session_e_accessed")
    ):
        raise RuntimeError("V8 E3 must keep Session E locked")
    delay = dict(config["delay"])
    stage = str(delay.get("stage", ""))
    route_scope = str(delay.get("route_scope", ""))
    signal_mode = str(delay.get("signal_mode", ""))
    allowed_stages = {
        "static_slow_within_band": ("within_band", "slow_envelope", False, False),
        "static_slow_cross_band": ("cross_band", "slow_envelope", False, False),
        "static_fast_within_band": ("within_band", "fast_phase", False, False),
        "contextual_slow_residual": ("within_band", "slow_envelope", True, False),
        "phase_fast_residual": ("within_band", "fast_phase", False, True),
    }
    if stage not in allowed_stages:
        raise RuntimeError(f"unregistered V8 E3 delay stage: {stage!r}")
    expected = allowed_stages[stage]
    observed = (
        route_scope,
        signal_mode,
        bool(delay.get("contextual_residual_enabled")),
        bool(delay.get("phase_residual_enabled")),
    )
    if observed != expected:
        raise RuntimeError(
            f"V8 E3 stage semantics changed: expected {expected}, observed {observed}"
        )
    if bool(delay["allow_cross_band"]) != (route_scope == "cross_band"):
        raise RuntimeError("V8 E3 cross-band model and prior scopes disagree")
    declared_trainable = set(config["training"]["trainable_parameters"])
    expected_trainable = _delay_trainable_names(delay)
    if declared_trainable != expected_trainable:
        raise RuntimeError(
            "V8 E3 declared trainable parameters differ from the implemented stage: "
            f"declared={sorted(declared_trainable)}, expected={sorted(expected_trainable)}"
        )
    if int(config["training"]["effective_batch_size"]) != int(
        config["training"]["batch_size"]
    ) * int(config["training"]["gradient_accumulation_steps"]):
        raise RuntimeError("V8 E3 effective batch size metadata is inconsistent")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    parent_status = read_json(parent_root / "campaign_status.json")
    if (
        parent_status.get("status") != "completed"
        or parent_status.get("stage") != "E2"
        or parent_variant not in parent_status.get("confirmed_variants", [])
        or bool(parent_status.get("session_e_accessed"))
    ):
        raise RuntimeError(
            f"E3 requires completed locked E2 parent variant {parent_variant!r}"
        )
    subjects = _csv(args.subjects, int) if args.subjects else list(config["subjects"])
    seeds = _csv(args.seeds, int) if args.seeds else list(config["seeds"])
    if sorted(subjects) != sorted(config["subjects"]) and not args.subjects:
        raise RuntimeError("V8 E3 subject contract changed unexpectedly")

    source_tree = collect_source_tree_manifest(ROOT)
    environment = _environment()
    write_json(output / "source_tree_manifest.json", source_tree)
    write_json(
        output / "source_tree_summary.json",
        {"files": len(source_tree), "sha256": source_tree_digest(source_tree)},
    )
    write_json(output / "heldout_lock_manifest.json", v8_heldout_lock_manifest())

    bundles: dict[int, SubjectBundle] = {}
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
        reference = _build_delay_model(model_config, config["delay"], seed=0)
        base_rates, cache_manifest = _load_parent_rates(
            parent_root,
            subject=int(subject),
            data_sha256=data_sha256,
            model=reference,
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

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for subject in subjects:
            rows.append(
                _run_one(
                    output=output,
                    parent_root=parent_root,
                    parent_variant=parent_variant,
                    config=config,
                    model_config=model_config,
                    source_tree=source_tree,
                    environment=environment,
                    bundle=bundles[int(subject)],
                    seed=int(seed),
                    device=args.device,
                )
            )
            write_csv(output / "summary.csv", rows)
    rows = sorted(rows, key=lambda row: (int(row["subject"]), int(row["seed"])))
    write_csv(output / "summary.csv", rows)
    write_json(
        output / "campaign_status.json",
        {
            "status": "completed",
            "stage": "E3",
            "protocol": config["protocol"],
            "variant": stage,
            "parent_variant": parent_variant,
            "subjects": subjects,
            "seeds": seeds,
            "runs": len(rows),
            "session_e_accessed": False,
            "source_tree_sha256": source_tree_digest(source_tree),
        },
    )
    print(json.dumps({"status": "completed", "runs": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
