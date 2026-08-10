#!/usr/bin/env python3
"""Run one leakage-safe V8 delay residual expert fold."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v8_anchor import (  # noqa: E402
    e1_selection_prediction_path,
    load_fused_e1_selection_anchor,
)
from dpc_snn.experiments.v8_delay_expert_training import (  # noqa: E402
    fit_delay_expert_input_gain,
    fit_v8_delay_expert,
    predict_v8_delay_expert,
    seed_v8_delay_expert,
)
from dpc_snn.experiments.v8_delay_prior import v8_delay_prior_seed  # noqa: E402
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
    write_trial_predictions,
)
from dpc_snn.experiments.v8_evidence_audit import (  # noqa: E402
    build_v8_evidence_rates,
    pool_v8_anatomical_regions,
    prepare_v8_evidence_space,
)
from dpc_snn.experiments.v8_fusion import entropy_residual_probability  # noqa: E402
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    file_sha256,
    mapping_sha256,
    sha256_fingerprint,
    source_tree_digest,
    v8_heldout_lock_manifest,
)
from dpc_snn.experiments.v8_training import (  # noqa: E402
    V8CachedRates,
    apply_v8_fold_gain,
    fit_v8_gain_from_cached_rates,
)
from dpc_snn.models.build import build_model  # noqa: E402
from dpc_snn.models.v8_accuracy_first import V8AccuracyFirstModel  # noqa: E402
from dpc_snn.models.v8_delay_residual_expert import (  # noqa: E402
    V8_DELAY_EXPERT_VARIANTS,
    V8DelayResidualExpert,
    build_v8_delay_residual_expert,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.analyze_v8_e4_atc_sequence_canaries import _load_e1  # noqa: E402
from scripts.run_v8_e1_baselines import _subject_file  # noqa: E402
from scripts.run_v8_e2_zero_delay import _nested_folds  # noqa: E402
from scripts.run_v8_e3_static_delay import (  # noqa: E402
    PRIOR_FILES,
    _load_or_fit_prior,
    _load_parent_rates,
)
from dpc_snn.experiments.v8_baselines import session_t_development_view  # noqa: E402


VARIANT_FILES = (
    "manifest.json",
    "selection_full_last.pt",
    "selection_zero_last.pt",
    "selection_full_history.csv",
    "selection_zero_history.csv",
    "full_last.pt",
    "matched_zero_last.pt",
    "full_history.csv",
    "matched_zero_history.csv",
    "metrics.json",
    "outer_predictions.npz",
    "full_predictions.npz",
    "full_predictions.csv",
    "matched_zero_predictions.npz",
    "matched_zero_predictions.csv",
    "same_weight_zero_predictions.npz",
    "same_weight_zero_predictions.csv",
)


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _expert_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    delay = config["delay"]
    expert = config["expert"]
    return {
        "n_bands": 12,
        "n_nodes": int(delay.get("transport_nodes", 22)),
        "n_classes": 4,
        "maximum_routes": int(delay["maximum_routes"]),
        "maximum_delay": int(delay["maximum_delay_samples"]),
        "allow_cross_band": bool(delay["allow_cross_band"]),
        "signal_mode": str(delay["signal_mode"]),
        "decoder_channels": int(expert["decoder_channels"]),
        "decoder_layers": int(expert["decoder_layers"]),
        "decays": tuple(float(value) for value in expert["decoder_decays"]),
        "sfreq": float(expert["fast_rate_hz"]),
        "temporal_decimation": int(expert["temporal_decimation"]),
        "endpoint_seconds": tuple(float(value) for value in expert["endpoint_seconds"]),
        "readout_features": int(expert["readout_features"]),
        "dropout": float(expert["dropout"]),
        "firing_rate_target": float(expert["firing_rate_target"]),
    }


def _build_expert(variant: str, config: Mapping[str, Any], *, seed: int) -> V8DelayResidualExpert:
    seed_v8_delay_expert(seed)
    return build_v8_delay_residual_expert(variant, **_expert_kwargs(config))


def capacity_audit(variants: Sequence[str], config: Mapping[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for variant in variants:
        model = _build_expert(variant, config, seed=0)
        rows[variant] = {
            "parameters": model.parameter_count,
            "trainable_parameters": model.trainable_parameter_count,
            "parameter_shapes": sorted([list(parameter.shape) for parameter in model.parameters()]),
            "decoder_kind": model.decoder_kind,
            "residual_mode": model.residual_mode,
        }
    if len({row["parameters"] for row in rows.values()}) != 1:
        raise RuntimeError("delay expert parameter counts are not matched")
    shapes = {json.dumps(row["parameter_shapes"], separators=(",", ":")) for row in rows.values()}
    if len(shapes) != 1:
        raise RuntimeError("delay expert parameter shapes are not matched")
    return {"status": "passed", "variants": rows}


def _reference_model(model_config: Mapping[str, Any]) -> V8AccuracyFirstModel:
    model = build_model("v8_accuracy_first", dict(model_config))
    if not isinstance(model, V8AccuracyFirstModel):
        raise TypeError("V8 physical cache reference model has the wrong type")
    if model.delay_auxiliary is not None:
        raise RuntimeError("physical cache reference must not enable delay")
    return model


def _subset(rates: V8CachedRates, indices: np.ndarray, gain: torch.Tensor) -> V8CachedRates:
    return apply_v8_fold_gain(rates.subset(indices), gain)


def _probability(logits: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.as_tensor(logits, dtype=torch.float32), dim=1).numpy()


def _metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return classification_metrics(labels, probability.argmax(axis=1), n_classes=4)


def _initial_parameter_match(first: V8DelayResidualExpert, second: V8DelayResidualExpert) -> None:
    first_parameters = dict(first.named_parameters())
    second_parameters = dict(second.named_parameters())
    if first_parameters.keys() != second_parameters.keys():
        raise RuntimeError("matched full/zero experts have different parameter names")
    for name in first_parameters:
        if not torch.equal(first_parameters[name].detach().cpu(), second_parameters[name].detach().cpu()):
            raise RuntimeError(f"matched full/zero initialization differs at {name}")


def _matched_transport_invariants(
    full: V8DelayResidualExpert,
    zero: V8DelayResidualExpert,
) -> dict[str, str]:
    if full.delay.routing_fingerprint() != zero.delay.routing_fingerprint():
        raise RuntimeError("matched full/zero routing invariants differ")
    if full.delay.prior_fingerprint() != zero.delay.prior_fingerprint():
        raise RuntimeError("matched full/zero fold priors differ")
    if not torch.equal(full.input_gain.detach().cpu(), zero.input_gain.detach().cpu()):
        raise RuntimeError("matched full/zero fold-fixed input gains differ")
    if not bool(full.input_gain_ready) or not bool(zero.input_gain_ready):
        raise RuntimeError("matched full/zero input gain is not ready")
    return {
        "routing_sha256": full.delay.routing_fingerprint(),
        "prior_sha256": full.delay.prior_fingerprint(),
        "input_gain_sha256": mapping_sha256({"input_gain": full.input_gain})[
            "input_gain"
        ],
    }


def _select_paired_epoch(
    full_history: Sequence[Mapping[str, float | int]],
    zero_history: Sequence[Mapping[str, float | int]],
    *,
    minimum_epoch: int,
) -> tuple[int, dict[str, float]]:
    if len(full_history) != len(zero_history) or not full_history:
        raise RuntimeError("paired selection histories are empty or misaligned")
    candidates: list[tuple[tuple[float, float, int], int, dict[str, float]]] = []
    for full_row, zero_row in zip(full_history, zero_history, strict=True):
        epoch = int(full_row["epoch"])
        if epoch != int(zero_row["epoch"]):
            raise RuntimeError("paired selection histories use different epochs")
        if epoch < int(minimum_epoch):
            continue
        mean_kappa = 0.5 * (
            float(full_row["validation_kappa"])
            + float(zero_row["validation_kappa"])
        )
        mean_accuracy = 0.5 * (
            float(full_row["validation_accuracy"])
            + float(zero_row["validation_accuracy"])
        )
        candidates.append(
            (
                (mean_kappa, mean_accuracy, -epoch),
                epoch,
                {
                    "mean_validation_kappa": mean_kappa,
                    "mean_validation_accuracy": mean_accuracy,
                    "full_validation_kappa": float(full_row["validation_kappa"]),
                    "zero_validation_kappa": float(zero_row["validation_kappa"]),
                },
            )
        )
    if not candidates:
        raise RuntimeError("paired checkpoint rule has no eligible epoch")
    _, epoch, metrics = max(candidates, key=lambda item: item[0])
    return epoch, metrics


def _build_partial_run_contract(
    *,
    source_digest: str,
    config_path: Path,
    model_config_path: Path,
    subject_path: Path,
    cache_path: Path,
    cache_manifest_path: Path,
    atc_path: Path,
    fbc_path: Path,
    atc_selection_path: Path,
    fbc_selection_path: Path,
    subject: int,
    seed: int,
    fold: int,
    variants: Sequence[str],
    device: str,
    prior_import: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "schema": "dpc-snn-v8-e3-delay-residual-partial-run/v1",
        "source_tree_sha256": str(source_digest),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "model_config_path": str(model_config_path),
        "model_config_sha256": file_sha256(model_config_path),
        "subject_file": str(subject_path),
        "subject_file_sha256": file_sha256(subject_path),
        "cache_path": str(cache_path),
        "cache_sha256": file_sha256(cache_path),
        "cache_manifest_path": str(cache_manifest_path),
        "cache_manifest_sha256": file_sha256(cache_manifest_path),
        "atc_predictions": str(atc_path),
        "atc_predictions_sha256": file_sha256(atc_path),
        "fbc_predictions": str(fbc_path),
        "fbc_predictions_sha256": file_sha256(fbc_path),
        "atc_fold_selection_predictions": str(atc_selection_path),
        "atc_fold_selection_predictions_sha256": file_sha256(atc_selection_path),
        "fbc_fold_selection_predictions": str(fbc_selection_path),
        "fbc_fold_selection_predictions_sha256": file_sha256(fbc_selection_path),
        "subject": int(subject),
        "seed": int(seed),
        "fold": int(fold),
        "variants": [str(value) for value in variants],
        "device": str(device),
        "prior_import": dict(prior_import) if prior_import is not None else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    return {**payload, "combined_sha256": sha256_fingerprint(payload)}


def _validated_prior_import_contract(
    source: Path,
    *,
    source_digest: str,
    config_path: Path,
    model_config_path: Path,
    subject: int,
    fold: int,
) -> dict[str, Any]:
    """Validate a completed canonical fold before reusing its seed-invariant prior."""

    if not (source / "manifest.json").is_file():
        raise RuntimeError("fold-prior import source is not a completed fold run")
    root_manifest = read_json(source / "manifest.json")
    validate_run_artifact_manifest(
        source,
        required_files=tuple(root_manifest["required_files"]),
        verify_hashes=True,
        verify_prediction_schema=False,
    )
    run = read_json(source / "run_manifest.json")
    expected = {
        "source_tree_sha256": str(source_digest),
        "config_sha256": file_sha256(config_path),
        "model_config_sha256": file_sha256(model_config_path),
        "subject": int(subject),
        "fold": int(fold),
    }
    for field, value in expected.items():
        if run.get(field) != value:
            raise RuntimeError(
                f"fold-prior import source differs at {field}: "
                f"expected {value!r}, got {run.get(field)!r}"
            )
    prior_hashes: dict[str, dict[str, str]] = {}
    for scope in ("inner", "outer"):
        directory = source / "fold_priors" / scope
        validate_run_artifact_manifest(
            directory,
            required_files=PRIOR_FILES,
            verify_hashes=True,
            verify_prediction_schema=False,
        )
        summary = read_json(directory / "summary.json")
        if not bool(summary.get("evidence_pipeline_passed")):
            raise RuntimeError(f"imported {scope} prior did not pass its evidence gate")
        prior_hashes[scope] = {
            name: file_sha256(directory / name) for name in PRIOR_FILES
        }
    payload = {
        "schema": "dpc-snn-v8-fold-prior-import/v1",
        "source": str(source),
        "source_run_manifest_sha256": file_sha256(source / "run_manifest.json"),
        "source_root_manifest_sha256": file_sha256(source / "manifest.json"),
        "source_classifier_seed": int(run["seed"]),
        "subject": int(subject),
        "fold": int(fold),
        "prior_hashes": prior_hashes,
    }
    return {**payload, "combined_sha256": sha256_fingerprint(payload)}


def _import_prior_bundle(source: Path, destination: Path) -> None:
    """Copy one hash-validated immutable prior, publishing its manifest last."""

    if all((destination / name).is_file() for name in PRIOR_FILES):
        validate_run_artifact_manifest(
            destination,
            required_files=PRIOR_FILES,
            verify_hashes=True,
            verify_prediction_schema=False,
        )
        return
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"partial prior import destination is not resumable: {destination}")
    ensure_dir(destination)
    for name in PRIOR_FILES:
        if name != "manifest.json":
            shutil.copy2(source / name, destination / name)
    shutil.copy2(source / "manifest.json", destination / "manifest.json")
    validate_run_artifact_manifest(
        destination,
        required_files=PRIOR_FILES,
        verify_hashes=True,
        verify_prediction_schema=False,
    )


def _prepare_partial_output(
    output: Path,
    contract: Mapping[str, Any],
) -> bool:
    """Create a new output or validate an exact partial-run resume contract."""

    contract_path = output / "partial_run_contract.json"
    if output.exists() and any(output.iterdir()):
        if not contract_path.is_file():
            raise RuntimeError("non-empty delay output has no partial-run contract")
        saved = read_json(contract_path)
        if saved != dict(contract):
            raise RuntimeError("delay partial-run contract mismatch")
        return True
    ensure_dir(output)
    write_json(contract_path, dict(contract))
    return False


def _load_completed_variant(
    directory: Path,
    *,
    expected_indices: np.ndarray,
    expected_labels: np.ndarray,
    expected_trial_ids: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    if not all((directory / name).is_file() for name in VARIANT_FILES):
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(f"incomplete delay variant cannot be resumed: {directory}")
        return None
    validate_run_artifact_manifest(
        directory,
        required_files=VARIANT_FILES,
        verify_hashes=True,
        verify_prediction_schema=False,
    )
    row = read_json(directory / "metrics.json")
    with np.load(directory / "outer_predictions.npz", allow_pickle=False) as archive:
        prediction = {name: archive[name] for name in archive.files}
    required_prediction_fields = {
        "indices",
        "labels",
        "subject",
        "session",
        "run",
        "trial_id",
        "anchor_probability",
        "full_expert_logits",
        "matched_zero_expert_logits",
        "same_weight_zero_expert_logits",
        "full_probability",
        "matched_zero_probability",
        "same_weight_zero_probability",
    }
    if set(prediction) != required_prediction_fields:
        raise RuntimeError("resumed delay variant prediction schema changed")
    if not np.array_equal(prediction.get("indices"), expected_indices):
        raise RuntimeError("resumed delay variant indices differ from the active fold")
    if not np.array_equal(prediction.get("labels"), expected_labels):
        raise RuntimeError("resumed delay variant labels differ from the active fold")
    if not np.array_equal(prediction.get("trial_id").astype(str), expected_trial_ids.astype(str)):
        raise RuntimeError("resumed delay variant trial identities differ from the active fold")
    return row, prediction


def main() -> None:
    started_at = datetime.now(timezone.utc)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache-parent", required=True)
    parser.add_argument("--atc-root", required=True)
    parser.add_argument("--fbc-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/experiments/v8_e3_delay_residual_canary.yaml")
    parser.add_argument("--model-config", default="configs/models/v8_accuracy_first.yaml")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--variants", default="ann_sew,sew_clif")
    parser.add_argument("--import-fold-priors", default="")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    configure_cache_env()
    config_path = Path(args.config).resolve()
    model_config_path = Path(args.model_config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    if config.get("schema") not in {
        "dpc-snn-v8-e3-delay-residual-canary/v2",
        "dpc-snn-v8-e3-delay-residual-campaign/v1",
    }:
        raise RuntimeError("delay residual canary schema changed")
    if config["data_access"].get("heldout_session_e_accessed") is not False:
        raise RuntimeError("Session E must remain locked")
    if config["delay"].get("stage") != "static_slow_within_band":
        raise RuntimeError("this canary only permits static slow within-band delay")
    if config["delay"].get("signal_mode") != "slow_envelope":
        raise RuntimeError("static slow delay must use the slow-envelope current")
    if int(config["training"]["effective_batch_size"]) != int(
        config["training"]["batch_size"]
    ) * int(config["training"]["gradient_accumulation_steps"]):
        raise RuntimeError("effective batch metadata is inconsistent")
    variants = _csv(args.variants)
    if any(value not in V8_DELAY_EXPERT_VARIANTS for value in variants):
        raise ValueError("unknown delay expert variant")
    for field, value in (
        ("subjects", int(args.subject)),
        ("seeds", int(args.seed)),
        ("folds", int(args.fold)),
    ):
        registered = [int(item) for item in config.get(field, [])]
        if registered and value not in registered:
            raise ValueError(f"{field} value {value} is outside the registered config")
    registered_variants = [str(value) for value in config.get("variants", [])]
    if registered_variants and any(value not in registered_variants for value in variants):
        raise ValueError("requested variants are outside the registered config")
    if args.fold not in range(6):
        raise ValueError("fold must lie in [0, 5]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    source_tree = collect_source_tree_manifest(ROOT)
    source_digest = source_tree_digest(source_tree)
    subject_path = _subject_file(Path(args.data).resolve(), int(args.subject))
    cache_parent = Path(args.cache_parent).resolve()
    cache_directory = (
        cache_parent / "shared_physical_rates" / f"subject_{args.subject:02d}"
    )
    cache_path = cache_directory / "unit_gain_rates.pt"
    cache_manifest_path = cache_directory / "manifest.json"
    atc_path = (
        Path(args.atc_root).resolve()
        / "atcnet"
        / f"subject_{args.subject:02d}"
        / f"seed_{args.seed}"
        / "predictions.npz"
    )
    fbc_path = (
        Path(args.fbc_root).resolve()
        / "fbcnet"
        / f"subject_{args.subject:02d}"
        / f"seed_{args.seed}"
        / "predictions.npz"
    )
    atc_selection_path = e1_selection_prediction_path(
        Path(args.atc_root).resolve(),
        model="atcnet",
        subject=int(args.subject),
        seed=int(args.seed),
        fold=int(args.fold),
    )
    fbc_selection_path = e1_selection_prediction_path(
        Path(args.fbc_root).resolve(),
        model="fbcnet",
        subject=int(args.subject),
        seed=int(args.seed),
        fold=int(args.fold),
    )
    prior_import_root = (
        Path(args.import_fold_priors).resolve() if args.import_fold_priors else None
    )
    prior_import = (
        _validated_prior_import_contract(
            prior_import_root,
            source_digest=source_digest,
            config_path=config_path,
            model_config_path=model_config_path,
            subject=int(args.subject),
            fold=int(args.fold),
        )
        if prior_import_root is not None
        else None
    )
    contract = _build_partial_run_contract(
        source_digest=source_digest,
        config_path=config_path,
        model_config_path=model_config_path,
        subject_path=subject_path,
        cache_path=cache_path,
        cache_manifest_path=cache_manifest_path,
        atc_path=atc_path,
        fbc_path=fbc_path,
        atc_selection_path=atc_selection_path,
        fbc_selection_path=fbc_selection_path,
        subject=int(args.subject),
        seed=int(args.seed),
        fold=int(args.fold),
        variants=variants,
        device=str(args.device),
        prior_import=prior_import,
    )
    output = Path(args.output).resolve()
    resumed = _prepare_partial_output(output, contract)
    if resumed and (output / "manifest.json").is_file():
        completed_manifest = read_json(output / "manifest.json")
        validate_run_artifact_manifest(
            output,
            required_files=tuple(completed_manifest["required_files"]),
            verify_hashes=True,
            verify_prediction_schema=False,
        )
        print(json.dumps({"status": "already_complete", "output": str(output)}))
        return
    header_payloads = {
        "heldout_lock_manifest.json": v8_heldout_lock_manifest(),
        "source_tree_manifest.json": source_tree,
        "source_tree_summary.json": {
            "files": len(source_tree),
            "sha256": source_digest,
        },
        "capacity_audit.json": capacity_audit(variants, config),
    }
    for name, payload in header_payloads.items():
        path = output / name
        if path.is_file() and read_json(path) != payload:
            raise RuntimeError(f"partial-run header mismatch: {name}")
        if not path.is_file():
            write_json(path, payload)

    data_sha = file_sha256(subject_path)
    data = load_processed_npz(subject_path)
    x, labels, metadata, access = session_t_development_view(data)
    nested = _nested_folds(metadata, n_splits=6, split_seed=0)
    outer_train, outer_test, inner_train, inner_validation, inner_run = nested[int(args.fold)]
    reference = _reference_model(model_config)
    base_rates, cache_manifest = _load_parent_rates(
        cache_parent,
        subject=int(args.subject),
        data_sha256=data_sha,
        model=reference,
    )

    atc = _load_e1(atc_path)
    fbc = _load_e1(fbc_path)
    expected_trial_ids = np.asarray([row["trial_id"] for row in metadata]).astype(str)
    if not np.array_equal(atc["label"], labels) or not np.array_equal(fbc["label"], labels):
        raise RuntimeError("accuracy anchor labels differ from the delay fold")
    if not np.array_equal(atc["trial_id"].astype(str), expected_trial_ids):
        raise RuntimeError("ATC OOF trial identities differ from the delay fold")
    if not np.array_equal(fbc["trial_id"].astype(str), expected_trial_ids):
        raise RuntimeError("FBC OOF trial identities differ from the delay fold")
    anchor = 0.5 * atc["probabilities"][outer_test] + 0.5 * fbc["probabilities"][outer_test]
    selection_anchor = load_fused_e1_selection_anchor(
        atc_selection_path,
        fbc_selection_path,
        expected_indices=inner_validation,
        expected_labels=labels[inner_validation],
    )
    anchor_metrics = _metrics(labels[outer_test], anchor)

    delay_config = dict(config["delay"])
    inner_gain = fit_v8_gain_from_cached_rates(base_rates, inner_train)
    outer_gain = fit_v8_gain_from_cached_rates(base_rates, outer_train)
    inner_train_rates = _subset(base_rates, inner_train, inner_gain)
    inner_validation_rates = _subset(base_rates, inner_validation, inner_gain)
    outer_train_rates = _subset(base_rates, outer_train, outer_gain)
    outer_test_rates = _subset(base_rates, outer_test, outer_gain)
    regional_transport = delay_config.get("transport_space") == "car_anatomical_regions"
    channel_names = [str(value) for value in data["ch_names"]]
    if regional_transport:
        inner_train_rates, region_metadata = pool_v8_anatomical_regions(
            inner_train_rates, channel_names
        )
        inner_validation_rates, _ = pool_v8_anatomical_regions(
            inner_validation_rates, channel_names
        )
        outer_train_rates, _ = pool_v8_anatomical_regions(outer_train_rates, channel_names)
        outer_test_rates, _ = pool_v8_anatomical_regions(outer_test_rates, channel_names)
        if inner_train_rates.fast.shape[2] != int(delay_config["transport_nodes"]):
            raise RuntimeError("regional transport node count differs from the config")
    else:
        region_metadata = {}

    def evidence_rates(indices: np.ndarray, scope: str) -> tuple[V8CachedRates | None, dict[str, Any]]:
        if not regional_transport:
            return None, {"space": "same_as_transport"}
        transformed, adjusted_epoch_tmin, transform = prepare_v8_evidence_space(
            x[indices],
            str(delay_config["evidence_space"]),
            channel_names=channel_names,
            sfreq=float(np.asarray(data["sfreq"]).item()),
            epoch_tmin=float(np.asarray(data["epoch_tmin"]).item()),
            task_tmin=float(model_config["task_tmin"]),
        )
        fitted = build_v8_evidence_rates(
            transformed,
            model_config,
            epoch_tmin=adjusted_epoch_tmin,
            transform_sha256=str(transform["output_sha256"]),
            device=args.device,
            batch_size=int(config["training"]["batch_size"]),
        )
        fitted, pooled = pool_v8_anatomical_regions(fitted, channel_names)
        identity = {
            **transform,
            **pooled,
            "scope": scope,
            "analytic_representation": (
                "v8_csd_unwhitened_var_innovations_fixed_anatomical_regions"
            ),
        }
        return fitted, identity

    prior_replicates = int(delay_config["prior"].get("audit_replicates", 1))
    if prior_replicates != 5:
        raise RuntimeError("regional delay pilot requires five prior audit replicates")
    inner_prior_seeds = [
        v8_delay_prior_seed(args.subject, args.fold, replicate, scope="inner")
        for replicate in range(prior_replicates)
    ]
    outer_prior_seeds = [
        v8_delay_prior_seed(args.subject, args.fold, replicate, scope="outer")
        for replicate in range(prior_replicates)
    ]

    def split_strata(indices: np.ndarray) -> list[str]:
        return [
            f"run={metadata[int(index)]['run']}|class={int(labels[int(index)])}"
            for index in indices
        ]

    prior_root = output / "fold_priors"
    if prior_import_root is not None:
        for scope in ("inner", "outer"):
            _import_prior_bundle(
                prior_import_root / "fold_priors" / scope,
                prior_root / scope,
            )
        import_path = output / "fold_prior_import.json"
        if import_path.is_file() and read_json(import_path) != prior_import:
            raise RuntimeError("fold-prior import provenance changed during resume")
        if not import_path.is_file():
            write_json(import_path, prior_import)
    inner_evidence_rates, inner_evidence_identity = evidence_rates(
        inner_train, "inner_training_fold_only"
    )
    inner_prior, inner_summary = _load_or_fit_prior(
        prior_root / "inner",
        scope="inner_training_fold_only",
        trial_ids=[metadata[int(index)]["trial_id"] for index in inner_train],
        rates=inner_train_rates,
        evidence_rates=inner_evidence_rates,
        evidence_identity=inner_evidence_identity,
        gain=inner_gain,
        model_config=model_config,
        delay_config=delay_config,
        source_sha256=source_digest,
        seed=inner_prior_seeds[0],
        ensemble_seeds=inner_prior_seeds,
        split_strata=split_strata(inner_train),
    )
    outer_evidence_rates, outer_evidence_identity = evidence_rates(
        outer_train, "outer_training_fold_only"
    )
    outer_prior, outer_summary = _load_or_fit_prior(
        prior_root / "outer",
        scope="outer_training_fold_only",
        trial_ids=[metadata[int(index)]["trial_id"] for index in outer_train],
        rates=outer_train_rates,
        evidence_rates=outer_evidence_rates,
        evidence_identity=outer_evidence_identity,
        gain=outer_gain,
        model_config=model_config,
        delay_config=delay_config,
        source_sha256=source_digest,
        seed=outer_prior_seeds[0],
        ensemble_seeds=outer_prior_seeds,
        split_strata=split_strata(outer_train),
    )

    selection = config["selection"]
    training = config["training"]
    maximum_weight = float(config["residual_fusion"]["maximum_expert_weight"])
    rows: list[dict[str, Any]] = []
    started = time.time()
    for variant in variants:
        variant_dir = output / variant
        completed = _load_completed_variant(
            variant_dir,
            expected_indices=outer_test,
            expected_labels=labels[outer_test],
            expected_trial_ids=expected_trial_ids[outer_test],
        )
        if completed is not None:
            row, _ = completed
            if row.get("variant") != variant:
                raise RuntimeError("resumed delay variant name mismatch")
            rows.append(row)
            print(f"__V8_DELAY_VARIANT_RESUMED__ variant={variant}", flush=True)
            continue
        variant_dir = ensure_dir(variant_dir)
        fold_seed = int(args.seed) * 100_003 + int(args.fold) * 1_009 + 8_400_003
        selection_full_model = _build_expert(variant, config, seed=fold_seed)
        selection_zero_model = _build_expert(variant, config, seed=fold_seed)
        selection_full_model.load_fold_prior(**inner_prior)
        selection_zero_model.load_fold_prior(**inner_prior)
        selection_gain = fit_delay_expert_input_gain(
            selection_full_model,
            inner_train_rates,
            device=args.device,
            batch_size=int(training["batch_size"]),
        )
        selection_zero_model.set_input_gain(selection_gain)
        _initial_parameter_match(selection_full_model, selection_zero_model)
        selection_invariants = _matched_transport_invariants(
            selection_full_model, selection_zero_model
        )
        selection_fit_kwargs = {
            "validation_rates": inner_validation_rates,
            "validation_labels": labels[inner_validation],
            "validation_anchor_probability": selection_anchor,
            "maximum_residual_weight": maximum_weight,
            "device": args.device,
            "seed": fold_seed,
            "epochs": int(selection["max_epochs"]),
            "patience": int(selection["max_epochs"]),
            "minimum_epochs": int(selection["max_epochs"]),
            "batch_size": int(training["batch_size"]),
            "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": float(training["weight_decay"]),
            "firing_rate_weight": float(training["firing_rate_weight"]),
        }
        selection_full_fit = fit_v8_delay_expert(
            selection_full_model,
            inner_train_rates,
            labels[inner_train],
            delay_override="full",
            run_label=f"V8-E3-delay-select-full:{variant}:S{args.subject}:fold{args.fold}",
            **selection_fit_kwargs,
        )
        selection_zero_fit = fit_v8_delay_expert(
            selection_zero_model,
            inner_train_rates,
            labels[inner_train],
            delay_override="zero",
            run_label=f"V8-E3-delay-select-zero:{variant}:S{args.subject}:fold{args.fold}",
            **selection_fit_kwargs,
        )
        selected_epoch, selection_metrics = _select_paired_epoch(
            selection_full_fit.history,
            selection_zero_fit.history,
            minimum_epoch=max(
                int(selection["minimum_epochs"]),
                int(selection["minimum_outer_epochs"]),
            ),
        )

        outer_seed = fold_seed + 1_000_003
        full_model = _build_expert(variant, config, seed=outer_seed)
        zero_model = _build_expert(variant, config, seed=outer_seed)
        full_model.load_fold_prior(**outer_prior)
        zero_model.load_fold_prior(**outer_prior)
        outer_input_gain = fit_delay_expert_input_gain(
            full_model,
            outer_train_rates,
            device=args.device,
            batch_size=int(training["batch_size"]),
        )
        zero_model.set_input_gain(outer_input_gain)
        _initial_parameter_match(full_model, zero_model)
        outer_invariants = _matched_transport_invariants(full_model, zero_model)
        full_initial = deepcopy(full_model.state_dict())
        zero_initial = deepcopy(zero_model.state_dict())
        if full_initial.keys() != zero_initial.keys() or any(
            not torch.equal(full_initial[name].cpu(), zero_initial[name].cpu())
            for name in full_initial
        ):
            raise RuntimeError("matched full/zero model states differ before training")

        fit_kwargs = {
            "validation_rates": None,
            "validation_labels": None,
            "device": args.device,
            "seed": outer_seed,
            "epochs": selected_epoch,
            "fixed_epoch": selected_epoch,
            "scheduler_epochs": int(selection["max_epochs"]),
            "batch_size": int(training["batch_size"]),
            "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": float(training["weight_decay"]),
            "firing_rate_weight": float(training["firing_rate_weight"]),
        }
        full_fit = fit_v8_delay_expert(
            full_model,
            outer_train_rates,
            labels[outer_train],
            delay_override="full",
            run_label=f"V8-E3-delay-full:{variant}:S{args.subject}:fold{args.fold}",
            **fit_kwargs,
        )
        zero_fit = fit_v8_delay_expert(
            zero_model,
            outer_train_rates,
            labels[outer_train],
            delay_override="zero",
            run_label=f"V8-E3-delay-zero:{variant}:S{args.subject}:fold{args.fold}",
            **fit_kwargs,
        )
        full = predict_v8_delay_expert(
            full_fit.model,
            outer_test_rates,
            labels[outer_test],
            delay_override="full",
            device=args.device,
        )
        same_weight_zero = predict_v8_delay_expert(
            full_fit.model,
            outer_test_rates,
            labels[outer_test],
            delay_override="zero",
            device=args.device,
        )
        matched_zero = predict_v8_delay_expert(
            zero_fit.model,
            outer_test_rates,
            labels[outer_test],
            delay_override="zero",
            device=args.device,
        )
        post_training_invariants = _matched_transport_invariants(
            full_fit.model, zero_fit.model
        )
        if post_training_invariants != outer_invariants:
            raise RuntimeError("matched transport invariants changed during training")
        final_full, gate = entropy_residual_probability(
            anchor, _probability(full["logits"]), maximum_weight=maximum_weight
        )
        final_same_zero, _ = entropy_residual_probability(
            anchor, _probability(same_weight_zero["logits"]), maximum_weight=maximum_weight
        )
        final_matched_zero, _ = entropy_residual_probability(
            anchor, _probability(matched_zero["logits"]), maximum_weight=maximum_weight
        )
        full_metrics = _metrics(labels[outer_test], final_full)
        same_zero_metrics = _metrics(labels[outer_test], final_same_zero)
        matched_zero_metrics = _metrics(labels[outer_test], final_matched_zero)
        row = {
            "variant": variant,
            "subject": int(args.subject),
            "seed": int(args.seed),
            "fold": int(args.fold),
            "selected_epoch": selected_epoch,
            "inner_checkpoint_rule": str(selection["shared_epoch_rule"]),
            **selection_metrics,
            "anchor_accuracy": anchor_metrics["accuracy"],
            "full_accuracy": full_metrics["accuracy"],
            "matched_zero_accuracy": matched_zero_metrics["accuracy"],
            "same_weight_zero_accuracy": same_zero_metrics["accuracy"],
            "full_minus_matched_zero_pp": 100.0
            * (float(full_metrics["accuracy"]) - float(matched_zero_metrics["accuracy"])),
            "full_minus_anchor_pp": 100.0
            * (float(full_metrics["accuracy"]) - float(anchor_metrics["accuracy"])),
            "full_expert_accuracy": full["accuracy"],
            "matched_zero_expert_accuracy": matched_zero["accuracy"],
            "full_current_rms": full["current_rms"],
            "zero_current_rms": matched_zero["current_rms"],
            "full_firing_rate": full["mean_firing_rate"],
            "mean_residual_gate": float(np.mean(gate)),
            "inner_prior_routes": inner_summary["selected_sparse_routes"],
            "outer_prior_routes": outer_summary["selected_sparse_routes"],
            "prior_audit_replicates": prior_replicates,
            "inner_prior_stability_passed": bool(
                inner_summary["stability_gate"]["passed"]
            ),
            "outer_prior_stability_passed": bool(
                outer_summary["stability_gate"]["passed"]
            ),
            "parameters": full_fit.model.parameter_count,
            "full_optimizer_steps": full_fit.optimizer_steps,
            "zero_optimizer_steps": zero_fit.optimizer_steps,
            **outer_invariants,
            "selection_routing_sha256": selection_invariants["routing_sha256"],
            "selection_prior_sha256": selection_invariants["prior_sha256"],
            "selection_input_gain_sha256": selection_invariants["input_gain_sha256"],
            "session_e_accessed": False,
            "delay_evidence_space": str(
                delay_config.get("evidence_space", "same_as_transport")
            ),
            "delay_transport_space": str(
                delay_config.get("transport_space", "physical_sensors")
            ),
            "delay_transport_nodes": int(inner_train_rates.fast.shape[2]),
        }
        if row["full_current_rms"] <= 0.0 or row["zero_current_rms"] <= 0.0:
            raise RuntimeError("matched full/zero routed currents must both be non-zero")
        if row["full_optimizer_steps"] != row["zero_optimizer_steps"]:
            raise RuntimeError("matched full/zero optimizer budgets differ")
        rows.append(row)
        torch.save(selection_full_fit.last_state, variant_dir / "selection_full_last.pt")
        torch.save(selection_zero_fit.last_state, variant_dir / "selection_zero_last.pt")
        torch.save(full_fit.last_state, variant_dir / "full_last.pt")
        torch.save(zero_fit.last_state, variant_dir / "matched_zero_last.pt")
        write_csv(variant_dir / "selection_full_history.csv", selection_full_fit.history)
        write_csv(variant_dir / "selection_zero_history.csv", selection_zero_fit.history)
        write_csv(variant_dir / "full_history.csv", full_fit.history)
        write_csv(variant_dir / "matched_zero_history.csv", zero_fit.history)
        write_json(variant_dir / "metrics.json", row)
        trial_ids = np.asarray(
            [metadata[int(index)]["trial_id"] for index in outer_test], dtype=np.str_
        )
        trial_runs = np.asarray(
            [metadata[int(index)]["run"] for index in outer_test], dtype=np.str_
        )
        np.savez_compressed(
            variant_dir / "outer_predictions.npz",
            indices=outer_test,
            labels=labels[outer_test],
            subject=np.full(outer_test.size, int(args.subject), dtype=np.int64),
            session=np.full(outer_test.size, "T", dtype=np.str_),
            run=trial_runs,
            trial_id=trial_ids,
            anchor_probability=anchor.astype(np.float32),
            full_expert_logits=np.asarray(full["logits"], dtype=np.float32),
            matched_zero_expert_logits=np.asarray(matched_zero["logits"], dtype=np.float32),
            same_weight_zero_expert_logits=np.asarray(same_weight_zero["logits"], dtype=np.float32),
            full_probability=final_full,
            matched_zero_probability=final_matched_zero,
            same_weight_zero_probability=final_same_zero,
        )
        for basename, probability, model_name in (
            ("full_predictions", final_full, f"v8_delay_{variant}_full"),
            (
                "matched_zero_predictions",
                final_matched_zero,
                f"v8_delay_{variant}_matched_zero",
            ),
            (
                "same_weight_zero_predictions",
                final_same_zero,
                f"v8_delay_{variant}_same_weight_zero",
            ),
        ):
            probability = np.asarray(probability, dtype=np.float32)
            write_trial_predictions(
                variant_dir,
                logits=np.log(np.clip(probability, 1e-7, 1.0)),
                probabilities=probability,
                pred=probability.argmax(axis=1).astype(np.int64),
                label=labels[outer_test].astype(np.int64),
                subject=int(args.subject),
                session="T",
                run=trial_runs,
                trial_id=trial_ids,
                seed=int(args.seed),
                model=model_name,
                basename=basename,
            )
        write_run_artifact_manifest(variant_dir, required_files=VARIANT_FILES)

    write_csv(output / "summary.csv", rows)
    write_json(
        output / "metrics.json",
        {
            "schema": "dpc-snn-v8-e3-delay-residual-fold-metrics/v2",
            "rows": rows,
            "primary_metric": "full_minus_matched_zero_pp",
            "session_e_accessed": False,
        },
    )
    metric_lines = [
        "# V8 E3 Delay Residual Fold Metrics",
        "",
        "| Variant | Anchor | Full | Matched zero | Full-zero (pp) | Full-anchor (pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metric_lines.append(
            "| {variant} | {anchor_accuracy:.4f} | {full_accuracy:.4f} | "
            "{matched_zero_accuracy:.4f} | {full_minus_matched_zero_pp:+.3f} | "
            "{full_minus_anchor_pp:+.3f} |".format(**row)
        )
    (output / "metrics.md").write_text("\n".join(metric_lines) + "\n", encoding="utf-8")
    status = {
        "status": "completed",
        "stage": "E3_delay_residual_fold_canary",
        "subject": int(args.subject),
        "seed": int(args.seed),
        "fold": int(args.fold),
        "inner_validation_run": inner_run,
        "variants": variants,
        "cache_manifest": cache_manifest,
        "delay_evidence_space": str(
            delay_config.get("evidence_space", "same_as_transport")
        ),
        "delay_transport_space": str(
            delay_config.get("transport_space", "physical_sensors")
        ),
        "delay_region_metadata": region_metadata,
        "data_access": access,
        "source_tree_sha256": source_digest,
        "elapsed_seconds": time.time() - started,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    write_json(output / "campaign_status.json", status)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    completed_at = datetime.now(timezone.utc)
    run_manifest = {
        "schema": "dpc-snn-v8-e3-delay-residual-run-manifest/v2",
        "run_id": f"v8_e3_delay_residual_s{args.subject}_seed{args.seed}_fold{args.fold}",
        "tier": "auxiliary/dev",
        "command": [sys.executable, *sys.argv],
        "working_directory": str(ROOT),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "model_config_path": str(model_config_path),
        "model_config_sha256": file_sha256(model_config_path),
        "subject_file": str(subject_path),
        "subject_file_sha256": data_sha,
        "atc_predictions": str(atc_path),
        "atc_predictions_sha256": file_sha256(atc_path),
        "fbc_predictions": str(fbc_path),
        "fbc_predictions_sha256": file_sha256(fbc_path),
        "atc_fold_selection_predictions": str(atc_selection_path),
        "atc_fold_selection_predictions_sha256": file_sha256(atc_selection_path),
        "fbc_fold_selection_predictions": str(fbc_selection_path),
        "fbc_fold_selection_predictions_sha256": file_sha256(fbc_selection_path),
        "cache_parent": str(cache_parent),
        "source_tree_sha256": source_digest,
        "subject": int(args.subject),
        "seed": int(args.seed),
        "fold": int(args.fold),
        "variants": variants,
        "matched_zero_semantics": (
            "same_routes_weights_confidence_amplitude_gain_with_delay_point_mass_at_zero"
        ),
        "checkpoint_rule": str(selection["shared_epoch_rule"]),
        "checkpoint_anchor_scope": "current_fold_inner_train_to_inner_validation",
        "partial_run_contract_sha256": str(contract["combined_sha256"]),
        "fold_prior_import": prior_import,
        "resumed_from_partial": bool(resumed),
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "storage_root": os.environ.get("DPC_SNN_STORAGE_ROOT"),
        },
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    write_json(output / "run_manifest.json", run_manifest)
    (output / "summary.md").write_text(
        "# V8 E3 Delay Residual Fold Summary\n\n"
        f"Status: completed for Subject {args.subject}, seed {args.seed}, fold {args.fold}.\n\n"
        "This auxiliary pilot compares the same non-zero sparse routed current under "
        "an audited delay posterior, a separately trained point-zero posterior, and a "
        "same-weight point-zero intervention. "
        "It is not a main-claim result until all locked subject-seed pairs complete.\n",
        encoding="utf-8",
    )
    (output / "runlog.summary.md").write_text(
        "# Run Log Summary\n\n"
        f"Started: {started_at.isoformat()}\n\n"
        f"Completed: {completed_at.isoformat()}\n\n"
        f"Source tree: `{source_digest}`\n",
        encoding="utf-8",
    )
    required_files = ["manifest.json"] + sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path != output / "manifest.json"
    )
    write_run_artifact_manifest(output, required_files=required_files)
    print(json.dumps({"status": "completed", "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
