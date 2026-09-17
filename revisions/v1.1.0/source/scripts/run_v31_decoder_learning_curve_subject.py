#!/usr/bin/env python3
"""Run one subject of the V31 frozen-representation decoder learning curve."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.baselines.neural import build_v62_neural_baseline
from dpc_snn.experiments.v8_protocol import (
    collect_source_tree_manifest,
    source_tree_digest,
)
from dpc_snn.experiments.v9_dual_feature_training import (
    FrozenFeatureStandardizer,
    equal_probability_teacher,
    fit_feature_standardizer,
    fit_v9_dual_feature,
    predict_v9_dual_feature,
)
from dpc_snn.experiments.v31_learning_curve import (
    LearningCurveSubset,
    nested_stratified_subsets,
    paired_run_seed,
)
from dpc_snn.experiments.v62_baselines import (
    apply_fixed_gain,
    prepare_model_input,
    task_carrier,
)
from dpc_snn.experiments.v62_protocol import file_sha256, sha256_fingerprint
from dpc_snn.experiments.v62_scaffold import (
    project_registered_max_norm_constraints_,
)
from dpc_snn.models.v8_atc_backbone import V8ATCAccuracyBackbone
from dpc_snn.models.v9_dual_feature_student import (
    V9FBCFeatureBackbone,
    build_v9_dual_feature_student,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json
from dpc_snn.utils.storage import configure_cache_env
from scripts.run_v8_publication_baselines import (
    _bci2a_view,
    _load_gain,
    _openbmi_view,
)
from scripts.run_v30_bnci2014_004_subject import (
    _data_view as _bnci_view,
)
from scripts.run_v30_bnci2014_004_subject import (
    _load_gain as _load_bnci_gain,
)
from scripts.run_v30_bnci2014_004_subject import (
    _prepared_inputs as _bnci_prepared_inputs,
)

DATASET_INDEX = {"bci2a": 0, "openbmi": 1, "bnci2014_004": 2}


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _write_recovery_training_labels(
    args: argparse.Namespace,
    labels: np.ndarray,
    trial_ids: list[str],
    data_identity: dict[str, Any],
) -> None:
    """Persist the exact training-label order used by each frozen feature cache."""

    if not args.recovery_label_root:
        return
    y = np.ascontiguousarray(labels, dtype=np.int64)
    ids = np.asarray([str(value) for value in trial_ids], dtype=np.str_)
    if y.shape != ids.shape:
        raise RuntimeError("training labels and trial identifiers are not aligned")
    directory = ensure_dir(
        Path(args.recovery_label_root).resolve() / args.dataset / f"subject_{args.subject:02d}"
    )
    path = directory / "training_labels.npz"
    metadata_path = path.with_suffix(".json")
    label_identity = sha256_fingerprint({"shape": list(y.shape), "values": y.tolist()})
    trial_identity = sha256_fingerprint(ids.tolist())
    metadata = {
        "schema": "dpc-snn-recovery-training-labels/v2",
        "dataset": args.dataset,
        "subject": int(args.subject),
        "trials": int(y.size),
        "label_identity_sha256": label_identity,
        "trial_ids_sha256": trial_identity,
        "data_identity": data_identity,
        "evaluation_data_accessed": False,
    }
    if path.is_file() or metadata_path.is_file():
        if not path.is_file() or not metadata_path.is_file():
            raise RuntimeError(f"partial recovery-label artifact: {directory}")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"label", "trial_id"}:
                raise RuntimeError(f"invalid recovery-label archive: {path}")
            saved_y = np.asarray(archive["label"], dtype=np.int64)
            saved_ids = np.asarray(archive["trial_id"], dtype=np.str_)
        if (
            not np.array_equal(saved_y, y)
            or not np.array_equal(saved_ids, ids)
            or read_json(metadata_path) != metadata
        ):
            raise RuntimeError(f"stale recovery-label artifact: {directory}")
        return
    _atomic_npz(path, label=y, trial_id=ids)
    write_json(metadata_path, metadata)


def _save_standardizer(path: Path, value: FrozenFeatureStandardizer) -> None:
    _atomic_npz(
        path,
        atc_mean=value.atc_mean,
        atc_scale=value.atc_scale,
        fbc_mean=value.fbc_mean,
        fbc_scale=value.fbc_scale,
    )


def _load_standardizer(path: Path) -> FrozenFeatureStandardizer:
    with np.load(path, allow_pickle=False) as archive:
        return FrozenFeatureStandardizer(
            atc_mean=np.asarray(archive["atc_mean"], dtype=np.float32),
            atc_scale=np.asarray(archive["atc_scale"], dtype=np.float32),
            fbc_mean=np.asarray(archive["fbc_mean"], dtype=np.float32),
            fbc_scale=np.asarray(archive["fbc_scale"], dtype=np.float32),
        )


def _publication_config() -> dict[str, Any]:
    return yaml.safe_load(
        (ROOT / "configs" / "experiments" / "v8_publication_baselines.yaml").read_text(
            encoding="utf-8"
        )
    )


def _publication_dataset_root(args: argparse.Namespace) -> Path:
    return Path(args.publication_root).resolve() / args.dataset


def _publication_checkpoint(args: argparse.Namespace, model: str, subject: int, seed: int) -> Path:
    return (
        _publication_dataset_root(args)
        / "runs"
        / model
        / f"subject_{subject:02d}"
        / f"seed_{seed}"
        / "training"
        / "checkpoint.pt"
    )


def _bnci_checkpoint(args: argparse.Namespace, model: str, subject: int, seed: int) -> Path:
    return (
        Path(args.v30_root).resolve()
        / f"subject_{subject:02d}"
        / f"seed_{seed}"
        / "teachers"
        / model
        / "checkpoint.pt"
    )


def _teacher_checkpoint(args: argparse.Namespace, model: str, subject: int, seed: int) -> Path:
    path = (
        _bnci_checkpoint(args, model, subject, seed)
        if args.dataset == "bnci2014_004"
        else _publication_checkpoint(args, model, subject, seed)
    )
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen V31 teacher checkpoint: {path}")
    return path


def _publication_view(
    args: argparse.Namespace, role: str
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str], dict[str, Any]]:
    publication = _publication_config()
    dataset = dict(publication["datasets"][args.dataset])
    channel_names = [str(value) for value in publication["channel_names"]]
    training = role == "training"
    session = str(dataset["train_session"] if training else dataset["evaluation_session"])
    expected = int(
        dataset["expected_train_trials"] if training else dataset["expected_evaluation_trials"]
    )
    if args.dataset == "bci2a":
        root = Path(args.bci2a_train_root if training else args.bci2a_eval_root).resolve()
        x, labels, rows, manifest, _ = _bci2a_view(
            root=root,
            subject=args.subject,
            session=session,
            role=role,
            channel_names=channel_names,
            expected_trials=expected,
        )
    else:
        x, labels, rows, manifest, _ = _openbmi_view(
            subject=args.subject,
            session=session,
            role=role,
            dataset_config=dataset,
            channel_names=channel_names,
            expected_trials=expected,
        )
    sfreq = float(rows[0]["sfreq"])
    carrier = task_carrier(x, sfreq=sfreq, epoch_tmin=float(rows[0]["epoch_tmin"]))
    gain = _load_gain(
        _publication_dataset_root(args)
        / "runs"
        / "atcnet"
        / f"subject_{args.subject:02d}"
        / "seed_0"
        / "training"
        / "gain.npz"
    )
    normalized = apply_fixed_gain(carrier, gain)
    prepared = {
        name: prepare_model_input(name, normalized, sfreq=sfreq) for name in ("atcnet", "fbcnet")
    }
    trial_ids = [str(row["trial_id"]) for row in rows]
    identity = {
        "dataset": args.dataset,
        "role": role,
        "session": session,
        "signal_sha256": _array_sha256(x),
        "label_sha256": _array_sha256(labels),
        "trial_ids_sha256": sha256_fingerprint(trial_ids),
        "source_manifest_sha256": sha256_fingerprint(manifest),
        "trials": int(np.asarray(labels).size),
    }
    return prepared, np.asarray(labels, dtype=np.int64), trial_ids, identity


def _load_view(
    args: argparse.Namespace, config: dict[str, Any], role: str
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str], dict[str, Any]]:
    if args.dataset != "bnci2014_004":
        return _publication_view(args, role)
    dataset = dict(config["datasets"]["bnci2014_004"])
    sessions = [str(value) for value in dataset[f"{role}_sessions"]]
    v30_config = yaml.safe_load(
        (ROOT / "configs" / "experiments" / "v30_bnci2014_004_blind.yaml").read_text(
            encoding="utf-8"
        )
    )
    data, labels, trial_ids, identity = _bnci_view(v30_config, args.subject, sessions)
    gain = _load_bnci_gain(
        Path(args.v30_root).resolve() / f"subject_{args.subject:02d}" / "preprocessing" / "gain.npz"
    )
    prepared = _bnci_prepared_inputs(data["carrier"], gain, sfreq=250.0)
    return prepared, labels, trial_ids, {**identity, "role": role}


@torch.no_grad()
def _extract_features(
    model_name: str,
    checkpoint_path: Path,
    carrier: np.ndarray,
    *,
    source_root: Path,
    device: str,
    batch_size: int,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    adapter = build_v62_neural_baseline(
        model_name,
        source_root=source_root,
        n_channels=int(checkpoint["n_channels"]),
        n_classes=int(checkpoint["n_classes"]),
        samples=int(checkpoint["samples"]),
    )
    adapter.load_state_dict(checkpoint["state_dict"], strict=True)
    project_registered_max_norm_constraints_(adapter)
    if model_name == "atcnet":
        wrapper = V8ATCAccuracyBackbone(adapter.module)
        expected_tail = (18, 32)
    elif model_name == "fbcnet":
        wrapper = V9FBCFeatureBackbone(adapter.module)
        expected_tail = (4, 288)
    else:
        raise ValueError(f"unsupported V31 teacher: {model_name}")
    wrapper.eval().to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.ascontiguousarray(carrier, dtype=np.float32))),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    sequences: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for (batch_x,) in loader:
        output = wrapper(batch_x.to(device, non_blocking=True))
        sequence = (
            output["aux"]["continuous_sequence"]
            if model_name == "atcnet"
            else output["continuous_sequence"]
        )
        sequences.append(sequence.float().cpu().numpy())
        logits.append(output["logits"].float().cpu().numpy())
    all_sequences = np.concatenate(sequences)
    all_logits = np.concatenate(logits)
    if all_sequences.shape != (carrier.shape[0], *expected_tail):
        raise RuntimeError(f"unexpected {model_name} feature shape: {all_sequences.shape}")
    if all_logits.shape != (carrier.shape[0], int(n_classes)):
        raise RuntimeError(f"unexpected {model_name} logit shape: {all_logits.shape}")
    del wrapper, adapter, checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_sequences, all_logits


def _feature_cache(
    args: argparse.Namespace,
    config: dict[str, Any],
    role: str,
    seed: int,
    prepared: dict[str, np.ndarray],
    data_identity: dict[str, Any],
    source_tree_sha256: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    directory = ensure_dir(
        Path(args.output).resolve()
        / args.dataset
        / f"subject_{args.subject:02d}"
        / f"seed_{seed}"
        / "feature_cache"
    )
    path = directory / f"{role}.npz"
    metadata_path = directory / f"{role}.json"
    teacher_paths = {
        name: _teacher_checkpoint(args, name, args.subject, seed) for name in ("atcnet", "fbcnet")
    }
    fingerprint = {
        "schema": "dpc-snn-v31-feature-cache/v1",
        "dataset": args.dataset,
        "subject": args.subject,
        "seed": int(seed),
        "role": role,
        "source_tree_sha256": source_tree_sha256,
        "data_identity": data_identity,
        "teacher_checkpoint_sha256": {
            name: file_sha256(value) for name, value in teacher_paths.items()
        },
    }
    fingerprint["combined_sha256"] = sha256_fingerprint(fingerprint)
    if path.is_file() and metadata_path.is_file():
        if read_json(metadata_path) != fingerprint:
            raise RuntimeError(f"stale V31 feature cache: {path}")
        with np.load(path, allow_pickle=False) as archive:
            return (
                np.asarray(archive["atc"], dtype=np.float32),
                np.asarray(archive["fbc"], dtype=np.float32),
                np.asarray(archive["teacher"], dtype=np.float32),
            )
    atc, atc_logits = _extract_features(
        "atcnet",
        teacher_paths["atcnet"],
        prepared["atcnet"],
        source_root=Path(args.source_root).resolve(),
        device=args.device,
        batch_size=args.feature_batch_size,
        n_classes=int(config["datasets"][args.dataset]["n_classes"]),
    )
    fbc, fbc_logits = _extract_features(
        "fbcnet",
        teacher_paths["fbcnet"],
        prepared["fbcnet"],
        source_root=Path(args.source_root).resolve(),
        device=args.device,
        batch_size=args.feature_batch_size,
        n_classes=int(config["datasets"][args.dataset]["n_classes"]),
    )
    teacher = equal_probability_teacher(atc_logits, fbc_logits)
    _atomic_npz(path, atc=atc, fbc=fbc, teacher=teacher)
    write_json(metadata_path, fingerprint)
    return atc, fbc, teacher


def _subsets(
    config: dict[str, Any], labels: np.ndarray, dataset: str, subject: int, seed: int
) -> list[LearningCurveSubset]:
    sampling_seed = 31_000_000 + DATASET_INDEX[dataset] * 1_000_000 + subject * 10_000 + seed
    return nested_stratified_subsets(
        labels,
        config["numeric_budgets_per_class"],
        seed=sampling_seed,
    )


def _budget_dir(output: Path, dataset: str, subject: int, seed: int, budget: str) -> Path:
    return output / dataset / f"subject_{subject:02d}" / f"seed_{seed}" / f"budget_{budget}"


def _model_kwargs(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    return {
        **dict(config["model"]),
        "n_classes": int(config["datasets"][dataset]["n_classes"]),
    }


def _train(
    args: argparse.Namespace,
    config: dict[str, Any],
    config_sha256: str,
    source_tree_sha256: str,
) -> None:
    output = Path(args.output).resolve()
    prepared, labels, trial_ids, data_identity = _load_view(args, config, "training")
    _write_recovery_training_labels(args, labels, trial_ids, data_identity)
    checkpoints: list[str] = []
    for seed in config["seeds"]:
        seed = int(seed)
        atc, fbc, teacher = _feature_cache(
            args, config, "training", seed, prepared, data_identity, source_tree_sha256
        )
        for subset in _subsets(config, labels, args.dataset, args.subject, seed):
            directory = ensure_dir(
                _budget_dir(output, args.dataset, args.subject, seed, subset.label)
            )
            standardizer_path = directory / "standardizer.npz"
            subset_path = directory / "subset.npz"
            subset_metadata = {
                "label": subset.label,
                "requested_per_class": subset.requested_per_class,
                "class_counts": {
                    str(key): int(value) for key, value in subset.class_counts.items()
                },
                "examples_per_class": subset.examples_per_class,
                "total_examples": int(subset.indices.size),
                "index_sha256": subset.index_sha256,
                "nested_sampling": True,
            }
            subset_metadata_path = directory / "subset.json"
            if (
                subset_metadata_path.is_file()
                and read_json(subset_metadata_path) != subset_metadata
            ):
                raise RuntimeError(f"stale V31 subset metadata: {directory}")
            if not subset_metadata_path.is_file():
                _atomic_npz(subset_path, indices=subset.indices)
                write_json(subset_metadata_path, subset_metadata)
            if standardizer_path.is_file():
                standardizer = _load_standardizer(standardizer_path)
            else:
                standardizer = fit_feature_standardizer(atc[subset.indices], fbc[subset.indices])
                _save_standardizer(standardizer_path, standardizer)
            train_atc, train_fbc = standardizer.transform(atc[subset.indices], fbc[subset.indices])
            run_seed = paired_run_seed(DATASET_INDEX[args.dataset], args.subject, seed)
            teacher_hashes = {
                name: file_sha256(_teacher_checkpoint(args, name, args.subject, seed))
                for name in ("atcnet", "fbcnet")
            }
            for variant in config["variants"]:
                variant = str(variant)
                variant_dir = ensure_dir(directory / variant)
                checkpoint_path = variant_dir / "checkpoint.pt"
                metrics_path = variant_dir / "training_metrics.json"
                fingerprint = {
                    "schema": "dpc-snn-v31-training-run/v1",
                    "config_sha256": config_sha256,
                    "source_tree_sha256": source_tree_sha256,
                    "dataset": args.dataset,
                    "subject": args.subject,
                    "seed": seed,
                    "run_seed": run_seed,
                    "variant": variant,
                    "subset": subset_metadata,
                    "training_data_identity": data_identity,
                    "teacher_checkpoint_sha256": teacher_hashes,
                    "representation_label_exposure": "all_training_session_labels",
                }
                fingerprint["combined_sha256"] = sha256_fingerprint(fingerprint)
                fingerprint_path = variant_dir / "training_fingerprint.json"
                if checkpoint_path.is_file() and metrics_path.is_file():
                    if read_json(fingerprint_path) != fingerprint:
                        raise RuntimeError(f"stale V31 training resume: {variant_dir}")
                    checkpoints.append(str(checkpoint_path.resolve()))
                    continue
                if fingerprint_path.is_file() and read_json(fingerprint_path) != fingerprint:
                    raise RuntimeError(f"partial stale V31 training resume: {variant_dir}")
                write_json(fingerprint_path, fingerprint)
                fit = fit_v9_dual_feature(
                    variant,
                    atc_train=train_atc,
                    fbc_train=train_fbc,
                    y_train=labels[subset.indices],
                    teacher_train=teacher[subset.indices],
                    atc_validation=None,
                    fbc_validation=None,
                    y_validation=None,
                    teacher_validation=None,
                    device=args.device,
                    seed=run_seed,
                    fixed_epoch=int(config["training"]["fixed_epochs"]),
                    scheduler_epochs=int(config["training"]["fixed_epochs"]),
                    batch_size=int(config["training"]["batch_size"]),
                    learning_rate=float(config["training"]["learning_rate"]),
                    weight_decay=float(config["training"]["weight_decay"]),
                    firing_rate_weight=float(config["training"]["firing_rate_weight"]),
                    model_kwargs=_model_kwargs(config, args.dataset),
                    run_label=(
                        f"V31:{args.dataset}:S{args.subject}:seed{seed}:{subset.label}:{variant}"
                    ),
                )
                _atomic_torch_save(
                    {
                        "state_dict": fit.last_state,
                        "dataset": args.dataset,
                        "subject": args.subject,
                        "seed": seed,
                        "run_seed": run_seed,
                        "budget": subset.label,
                        "variant": variant,
                        "n_classes": int(config["datasets"][args.dataset]["n_classes"]),
                        "training_fingerprint_sha256": fingerprint["combined_sha256"],
                    },
                    checkpoint_path,
                )
                write_csv(variant_dir / "history.csv", fit.history)
                write_json(
                    metrics_path,
                    {
                        "status": "training_completed_checkpoint_sealed",
                        "dataset": args.dataset,
                        "subject": args.subject,
                        "seed": seed,
                        "budget": subset.label,
                        "examples_per_class": subset.examples_per_class,
                        "variant": variant,
                        "fixed_epochs": int(config["training"]["fixed_epochs"]),
                        "optimizer_steps": fit.optimizer_steps,
                        "parameters": fit.model.parameter_count,
                        "elapsed_seconds": fit.elapsed_seconds,
                        "evaluation_data_accessed_during_training": False,
                    },
                )
                checkpoints.append(str(checkpoint_path.resolve()))
                del fit
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    status_dir = ensure_dir(output / args.dataset / f"subject_{args.subject:02d}")
    write_json(
        status_dir / "training_status.json",
        {
            "status": "completed",
            "schema": "dpc-snn-v31-subject-training-status/v1",
            "dataset": args.dataset,
            "subject": args.subject,
            "config_sha256": config_sha256,
            "source_tree_sha256": source_tree_sha256,
            "checkpoints": sorted(set(checkpoints)),
            "evaluation_data_accessed_during_training": False,
        },
    )


def _evaluate(
    args: argparse.Namespace,
    config: dict[str, Any],
    config_sha256: str,
    source_tree_sha256: str,
) -> None:
    output = Path(args.output).resolve()
    barrier = read_json(Path(args.checkpoint_barrier).resolve())
    if (
        barrier.get("schema") != "dpc-snn-v31-global-checkpoint-barrier/v1"
        or barrier.get("status") != "sealed"
        or barrier.get("config_sha256") != config_sha256
        or barrier.get("source_tree_sha256") != source_tree_sha256
    ):
        raise RuntimeError("V31 evaluation requires the matching global checkpoint barrier")
    prepared, labels, trial_ids, data_identity = _load_view(args, config, "evaluation")
    for seed in config["seeds"]:
        seed = int(seed)
        atc, fbc, teacher = _feature_cache(
            args, config, "evaluation", seed, prepared, data_identity, source_tree_sha256
        )
        # Recreate only the budget labels. Training indices are never applied to evaluation data.
        training_subset_root = (
            output / args.dataset / f"subject_{args.subject:02d}" / f"seed_{seed}"
        )
        budget_dirs = sorted(training_subset_root.glob("budget_*"))
        if not budget_dirs:
            raise RuntimeError(f"V31 evaluation found no trained budgets: {training_subset_root}")
        for directory in budget_dirs:
            subset_metadata = read_json(directory / "subset.json")
            standardizer = _load_standardizer(directory / "standardizer.npz")
            eval_atc, eval_fbc = standardizer.transform(atc, fbc)
            for variant in config["variants"]:
                variant = str(variant)
                variant_dir = directory / variant
                checkpoint_path = variant_dir / "checkpoint.pt"
                expected_sha = barrier["checkpoints"].get(str(checkpoint_path.resolve()))
                if expected_sha is None or file_sha256(checkpoint_path) != expected_sha:
                    raise RuntimeError(f"V31 checkpoint changed after barrier: {checkpoint_path}")
                evaluation_dir = ensure_dir(variant_dir / "evaluation")
                if (evaluation_dir / "metrics.json").is_file() and (
                    evaluation_dir / "predictions.npz"
                ).is_file():
                    continue
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                model = build_v9_dual_feature_student(
                    "ann_sew" if variant == "ann_sew_ce" else "sew_clif",
                    **_model_kwargs(config, args.dataset),
                )
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                prediction = predict_v9_dual_feature(
                    model,
                    eval_atc,
                    eval_fbc,
                    labels,
                    teacher,
                    device=args.device,
                    batch_size=int(config["training"]["batch_size"]),
                )
                _atomic_npz(
                    evaluation_dir / "predictions.npz",
                    logits=np.asarray(prediction["logits"], dtype=np.float32),
                    pred=np.asarray(prediction["logits"]).argmax(axis=1),
                    label=labels,
                    trial_id=np.asarray(trial_ids),
                )
                write_json(
                    evaluation_dir / "metrics.json",
                    {
                        "status": "completed",
                        "dataset": args.dataset,
                        "subject": args.subject,
                        "seed": seed,
                        "budget": subset_metadata["label"],
                        "examples_per_class": subset_metadata["examples_per_class"],
                        "variant": variant,
                        "accuracy": float(prediction["accuracy"]),
                        "balanced_accuracy": float(prediction["balanced_accuracy"]),
                        "kappa": float(prediction["kappa"]),
                        "macro_f1": float(prediction["macro_f1"]),
                        "mean_firing_rate": float(prediction["mean_firing_rate"]),
                        "training_checkpoint_sha256": expected_sha,
                        "evaluation_data_identity": data_identity,
                        "evaluation_gradient_updates": False,
                    },
                )
                del model, checkpoint
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    write_json(
        output / args.dataset / f"subject_{args.subject:02d}" / "evaluation_status.json",
        {
            "status": "completed",
            "schema": "dpc-snn-v31-subject-evaluation-status/v1",
            "dataset": args.dataset,
            "subject": args.subject,
            "evaluation_gradient_updates": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "v31_decoder_learning_curve.yaml"),
    )
    parser.add_argument("--phase", choices=("train", "evaluate"), required=True)
    parser.add_argument("--dataset", choices=tuple(DATASET_INDEX), required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--publication-root", required=True)
    parser.add_argument("--v30-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-barrier")
    parser.add_argument(
        "--bci2a-train-root",
        default="data/processed/bci2a_v8_session_t_4d04d3337fe3",
    )
    parser.add_argument(
        "--bci2a-eval-root",
        default="data/processed/bci2a_v8_session_e_4d04d3337fe3",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--recovery-label-root", default="")
    args = parser.parse_args()
    configure_cache_env()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.subject not in [int(value) for value in config["datasets"][args.dataset]["subjects"]]:
        raise ValueError(f"subject {args.subject} is outside the V31 {args.dataset} cohort")
    if args.phase == "evaluate" and not args.checkpoint_barrier:
        raise ValueError("V31 evaluation requires --checkpoint-barrier")
    config_sha256 = file_sha256(config_path)
    source_tree_sha256 = source_tree_digest(collect_source_tree_manifest(ROOT))
    started = time.time()
    if args.phase == "train":
        _train(args, config, config_sha256, source_tree_sha256)
    else:
        _evaluate(args, config, config_sha256, source_tree_sha256)
    print(
        json.dumps(
            {
                "status": "completed",
                "phase": args.phase,
                "dataset": args.dataset,
                "subject": args.subject,
                "elapsed_seconds": time.time() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
