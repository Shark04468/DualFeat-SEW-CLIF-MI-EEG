#!/usr/bin/env python3
"""Train or evaluate one BNCI2014-004 subject under the frozen V30 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.baselines.neural import build_v62_neural_baseline  # noqa: E402
from dpc_snn.data.bnci2014_004 import load_bnci2014_004_subject  # noqa: E402
from dpc_snn.experiments.v62_baselines import (  # noqa: E402
    FixedGain,
    apply_fixed_gain,
    fit_baseline,
    fit_fixed_gain,
    prepare_model_input,
    task_carrier,
)
from dpc_snn.experiments.v62_protocol import file_sha256, sha256_fingerprint  # noqa: E402
from dpc_snn.experiments.v62_scaffold import (  # noqa: E402
    project_registered_max_norm_constraints_,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    source_tree_digest,
)
from dpc_snn.experiments.v9_dual_feature_training import (  # noqa: E402
    FrozenFeatureStandardizer,
    equal_probability_teacher,
    fit_feature_standardizer,
    fit_v9_dual_feature,
    predict_v9_dual_feature,
)
from dpc_snn.models.v8_atc_backbone import V8ATCAccuracyBackbone  # noqa: E402
from dpc_snn.models.v9_dual_feature_student import (  # noqa: E402
    V9FBCFeatureBackbone,
    build_v9_dual_feature_student,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.numpy().tobytes())
    return digest.hexdigest()


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _save_gain(path: Path, gain: FixedGain) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            values=np.asarray(gain.values, dtype=np.float32),
            clip=np.asarray(gain.clip, dtype=np.float64),
        )
    temporary.replace(path)


def _load_gain(path: Path) -> FixedGain:
    with np.load(path, allow_pickle=False) as archive:
        return FixedGain(
            values=np.asarray(archive["values"], dtype=np.float32),
            clip=float(np.asarray(archive["clip"]).item()),
        )


def _save_standardizer(path: Path, value: FrozenFeatureStandardizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            atc_mean=value.atc_mean,
            atc_scale=value.atc_scale,
            fbc_mean=value.fbc_mean,
            fbc_scale=value.fbc_scale,
        )
    temporary.replace(path)


def _load_standardizer(path: Path) -> FrozenFeatureStandardizer:
    with np.load(path, allow_pickle=False) as archive:
        return FrozenFeatureStandardizer(
            atc_mean=np.asarray(archive["atc_mean"], dtype=np.float32),
            atc_scale=np.asarray(archive["atc_scale"], dtype=np.float32),
            fbc_mean=np.asarray(archive["fbc_mean"], dtype=np.float32),
            fbc_scale=np.asarray(archive["fbc_scale"], dtype=np.float32),
        )


def _teacher_path(output: Path, subject: int, seed: int, model: str) -> Path:
    return (
        output
        / f"subject_{subject:02d}"
        / f"seed_{seed}"
        / "teachers"
        / model
        / "checkpoint.pt"
    )


def _student_path(output: Path, subject: int, seed: int, variant: str) -> Path:
    return (
        output
        / f"subject_{subject:02d}"
        / f"seed_{seed}"
        / "students"
        / variant
        / "checkpoint.pt"
    )


def _data_view(
    config: dict[str, Any], subject: int, sessions: list[str]
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str], dict[str, Any]]:
    dataset = dict(config["dataset"])
    data = load_bnci2014_004_subject(
        subject,
        sessions=sessions,
        tmin=float(dataset["epoch_tmin"]),
        tmax=float(dataset["epoch_tmax"]),
        resample=float(dataset["source_sfreq"]),
    )
    x = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    trial_ids = [str(value) for value in data["trial_id"]]
    identity = {
        "dataset": "BNCI2014_004",
        "sessions": sessions,
        "signal_sha256": _array_sha256(x),
        "label_sha256": _array_sha256(y),
        "trial_ids_sha256": sha256_fingerprint(trial_ids),
        "trials": int(y.size),
        "sfreq": float(data["sfreq"]),
        "channels": list(data["ch_names"]),
        "session_trial_counts": dict(data["session_trial_counts"]),
        "epoch_tmin": float(data["epoch_tmin"]),
        "epoch_tmax": float(data["epoch_tmax"]),
    }
    carrier = task_carrier(
        x, sfreq=float(data["sfreq"]), epoch_tmin=float(data["epoch_tmin"])
    )
    return {"carrier": carrier}, y, trial_ids, identity


def _prepared_inputs(carrier: np.ndarray, gain: FixedGain, sfreq: float) -> dict[str, np.ndarray]:
    normalized = apply_fixed_gain(carrier, gain)
    return {
        model: prepare_model_input(model, normalized, sfreq=sfreq)
        for model in ("atcnet", "fbcnet")
    }


def _build_teacher(
    model_name: str,
    checkpoint: dict[str, Any],
    *,
    source_root: Path,
    device: str,
) -> torch.nn.Module:
    model = build_v62_neural_baseline(
        model_name,
        source_root=source_root,
        n_channels=int(checkpoint["n_channels"]),
        n_classes=int(checkpoint["n_classes"]),
        samples=int(checkpoint["samples"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    project_registered_max_norm_constraints_(model)
    return model.eval().to(device)


@torch.no_grad()
def _extract_features(
    model_name: str,
    model: torch.nn.Module,
    carrier: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "atcnet":
        wrapper = V8ATCAccuracyBackbone(model.module)
        expected_tail = (18, 32)
    elif model_name == "fbcnet":
        wrapper = V9FBCFeatureBackbone(model.module)
        expected_tail = (4, 288)
    else:
        raise ValueError(f"Unsupported V30 teacher model: {model_name}")
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
    if all_sequences.shape[1:] != expected_tail or all_logits.shape != (
        carrier.shape[0],
        2,
    ):
        raise RuntimeError(
            f"{model_name} returned V30 feature/logit shapes "
            f"{all_sequences.shape}/{all_logits.shape}"
        )
    return all_sequences, all_logits


def _training_fingerprint(
    freeze: dict[str, Any],
    config: dict[str, Any],
    data_identity: dict[str, Any],
    subject: int,
    seed: int,
) -> dict[str, Any]:
    payload = {
        "freeze_sha256": freeze["combined_sha256"],
        "source_tree_sha256": freeze["source_tree_sha256"],
        "config": config,
        "training_data": data_identity,
        "subject": int(subject),
        "seed": int(seed),
    }
    payload["combined_sha256"] = sha256_fingerprint(payload)
    return payload


def _train(
    args: argparse.Namespace, config: dict[str, Any], freeze: dict[str, Any]
) -> None:
    output = Path(args.output).resolve()
    subject_dir = ensure_dir(output / f"subject_{args.subject:02d}")
    sessions = [str(value) for value in config["dataset"]["training_sessions"]]
    data, labels, _, data_identity = _data_view(config, args.subject, sessions)
    if not (
        int(config["dataset"]["training_trials_minimum"])
        <= data_identity["trials"]
        <= int(config["dataset"]["training_trials_maximum"])
    ):
        raise RuntimeError("V30 training trial count lies outside the frozen range")
    gain_path = subject_dir / "preprocessing" / "gain.npz"
    if gain_path.is_file():
        gain = _load_gain(gain_path)
    else:
        gain = fit_fixed_gain(
            data["carrier"], clip=float(config["preprocessing"]["clip_after_gain"])
        )
        _save_gain(gain_path, gain)
        write_json(subject_dir / "preprocessing" / "training_data_identity.json", data_identity)
    prepared = _prepared_inputs(
        data["carrier"], gain, sfreq=float(config["dataset"]["source_sfreq"])
    )

    for seed in config["seeds"]:
        seed_dir = ensure_dir(subject_dir / f"seed_{seed}")
        fingerprint = _training_fingerprint(
            freeze, config, data_identity, args.subject, int(seed)
        )
        fingerprint_path = seed_dir / "training_fingerprint.json"
        if fingerprint_path.is_file() and read_json(fingerprint_path) != fingerprint:
            raise RuntimeError(f"Stale V30 training resume: {seed_dir}")
        if not fingerprint_path.is_file():
            write_json(fingerprint_path, fingerprint)

        teachers: dict[str, torch.nn.Module] = {}
        teacher_hashes: dict[str, str] = {}
        for model_index, model_name in enumerate(config["teacher_models"]):
            checkpoint_path = _teacher_path(
                output, args.subject, int(seed), str(model_name)
            )
            metrics_path = checkpoint_path.with_name("training_metrics.json")
            if checkpoint_path.is_file() and metrics_path.is_file():
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                if (
                    checkpoint.get("freeze_sha256") != freeze["combined_sha256"]
                    or checkpoint.get("training_fingerprint_sha256")
                    != fingerprint["combined_sha256"]
                ):
                    raise RuntimeError(f"Stale V30 teacher resume: {checkpoint_path}")
                model = _build_teacher(
                    str(model_name),
                    checkpoint,
                    source_root=Path(args.source_root).resolve(),
                    device=args.device,
                )
            else:
                run_seed = (
                    3_000_000
                    + args.subject * 10_000
                    + int(seed) * 101
                    + model_index * 1_000_000
                )
                fit = fit_baseline(
                    str(model_name),
                    source_root=Path(args.source_root).resolve(),
                    x_train=prepared[str(model_name)],
                    y_train=labels,
                    x_validation=None,
                    y_validation=None,
                    device=args.device,
                    seed=run_seed,
                    epochs=int(config["training"]["teacher_fixed_epochs"]),
                    patience=int(config["training"]["teacher_fixed_epochs"]),
                    augmentation=dict(config["training"]["augmentation"]),
                    fixed_epoch=int(config["training"]["teacher_fixed_epochs"]),
                    scheduler_epochs=int(config["training"]["teacher_fixed_epochs"]),
                    run_label=(
                        f"V30:S{args.subject}:seed{seed}:teacher:{model_name}"
                    ),
                    n_channels=len(config["dataset"]["channels"]),
                    n_classes=2,
                    physical_batch_size=int(
                        config["training"]["teacher_physical_batch_size"][
                            str(model_name)
                        ]
                    ),
                    effective_batch_size=int(
                        config["training"]["effective_batch_size"]
                    ),
                )
                model = fit.model
                checkpoint = {
                    "state_dict": fit.last_state,
                    "model": str(model_name),
                    "subject": args.subject,
                    "seed": int(seed),
                    "n_channels": len(config["dataset"]["channels"]),
                    "n_classes": 2,
                    "samples": int(
                        prepared[str(model_name)].shape[-2]
                        if prepared[str(model_name)].ndim == 5
                        else prepared[str(model_name)].shape[-1]
                    ),
                    "freeze_sha256": freeze["combined_sha256"],
                    "training_fingerprint_sha256": fingerprint["combined_sha256"],
                }
                _atomic_torch_save(checkpoint, checkpoint_path)
                write_csv(checkpoint_path.with_name("history.csv"), fit.history)
                write_json(
                    metrics_path,
                    {
                        "status": "training_completed_checkpoint_sealed",
                        "model": str(model_name),
                        "subject": args.subject,
                        "seed": int(seed),
                        "fixed_epochs": int(
                            config["training"]["teacher_fixed_epochs"]
                        ),
                        "optimizer_steps": fit.optimizer_steps,
                        "parameters": int(
                            sum(parameter.numel() for parameter in model.parameters())
                        ),
                        "elapsed_seconds": fit.elapsed_seconds,
                        "evaluation_sessions_accessed": False,
                    },
                )
                del fit
            teachers[str(model_name)] = model
            teacher_hashes[str(model_name)] = file_sha256(checkpoint_path)

        atc, atc_logits = _extract_features(
            "atcnet",
            teachers["atcnet"],
            prepared["atcnet"],
            device=args.device,
            batch_size=args.feature_batch_size,
        )
        fbc, fbc_logits = _extract_features(
            "fbcnet",
            teachers["fbcnet"],
            prepared["fbcnet"],
            device=args.device,
            batch_size=args.feature_batch_size,
        )
        standardizer_path = seed_dir / "feature_standardizer.npz"
        if standardizer_path.is_file():
            standardizer = _load_standardizer(standardizer_path)
        else:
            standardizer = fit_feature_standardizer(atc, fbc)
            _save_standardizer(standardizer_path, standardizer)
        train_atc, train_fbc = standardizer.transform(atc, fbc)
        teacher_logits = equal_probability_teacher(atc_logits, fbc_logits)

        for variant in config["variants"]:
            checkpoint_path = _student_path(
                output, args.subject, int(seed), str(variant)
            )
            metrics_path = checkpoint_path.with_name("training_metrics.json")
            if checkpoint_path.is_file() and metrics_path.is_file():
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                if (
                    checkpoint.get("freeze_sha256") != freeze["combined_sha256"]
                    or checkpoint.get("training_fingerprint_sha256")
                    != fingerprint["combined_sha256"]
                ):
                    raise RuntimeError(f"Stale V30 student resume: {checkpoint_path}")
                continue
            run_seed = 3_100_000 + args.subject * 10_000 + int(seed) * 101
            fit = fit_v9_dual_feature(
                str(variant),
                atc_train=train_atc,
                fbc_train=train_fbc,
                y_train=labels,
                teacher_train=teacher_logits,
                atc_validation=None,
                fbc_validation=None,
                y_validation=None,
                teacher_validation=None,
                device=args.device,
                seed=run_seed,
                fixed_epoch=int(config["training"]["student_fixed_epochs"]),
                scheduler_epochs=int(config["training"]["student_fixed_epochs"]),
                batch_size=int(config["training"]["student_batch_size"]),
                learning_rate=float(config["training"]["student_learning_rate"]),
                weight_decay=float(config["training"]["student_weight_decay"]),
                firing_rate_weight=float(config["training"]["firing_rate_weight"]),
                model_kwargs=dict(config["model"]),
                run_label=f"V30:S{args.subject}:seed{seed}:{variant}",
            )
            _atomic_torch_save(
                {
                    "state_dict": fit.last_state,
                    "variant": str(variant),
                    "subject": args.subject,
                    "seed": int(seed),
                    "n_classes": 2,
                    "freeze_sha256": freeze["combined_sha256"],
                    "training_fingerprint_sha256": fingerprint["combined_sha256"],
                    "teacher_checkpoint_sha256": teacher_hashes,
                },
                checkpoint_path,
            )
            write_csv(checkpoint_path.with_name("history.csv"), fit.history)
            write_json(
                metrics_path,
                {
                    "status": "training_completed_checkpoint_sealed",
                    "subject": args.subject,
                    "seed": int(seed),
                    "variant": str(variant),
                    "fixed_epochs": fit.best_epoch,
                    "optimizer_steps": fit.optimizer_steps,
                    "parameters": fit.model.parameter_count,
                    "elapsed_seconds": fit.elapsed_seconds,
                    "evaluation_sessions_accessed": False,
                },
            )
            del fit
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del teachers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(
        subject_dir / "training_status.json",
        {
            "status": "completed",
            "subject": args.subject,
            "training_sessions": sessions,
            "evaluation_sessions_accessed": False,
        },
    )


def _metric_payload(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    pred = np.asarray(logits).argmax(axis=1)
    metrics = classification_metrics(labels, pred, n_classes=2)
    return {
        key: float(metrics[key])
        for key in ("accuracy", "balanced_accuracy", "kappa", "macro_f1")
    }


def _evaluate(
    args: argparse.Namespace, config: dict[str, Any], freeze: dict[str, Any]
) -> None:
    output = Path(args.output).resolve()
    barrier = read_json(Path(args.checkpoint_barrier).resolve())
    if (
        barrier.get("schema") != "dpc-snn-v30-global-checkpoint-barrier/v1"
        or barrier.get("status") != "sealed"
        or barrier.get("freeze_sha256") != freeze["combined_sha256"]
    ):
        raise RuntimeError("V30 evaluation requires its complete global checkpoint barrier")

    sessions = [str(value) for value in config["dataset"]["evaluation_sessions"]]
    data, labels, trial_ids, data_identity = _data_view(config, args.subject, sessions)
    if not (
        int(config["dataset"]["evaluation_trials_minimum"])
        <= data_identity["trials"]
        <= int(config["dataset"]["evaluation_trials_maximum"])
    ):
        raise RuntimeError("V30 evaluation trial count lies outside the frozen range")
    subject_dir = output / f"subject_{args.subject:02d}"
    gain = _load_gain(subject_dir / "preprocessing" / "gain.npz")
    prepared = _prepared_inputs(
        data["carrier"], gain, sfreq=float(config["dataset"]["source_sfreq"])
    )

    for seed in config["seeds"]:
        seed_dir = subject_dir / f"seed_{seed}"
        features: dict[str, np.ndarray] = {}
        teacher_logits: dict[str, np.ndarray] = {}
        teacher_states: dict[str, dict[str, Any]] = {}
        for model_name in config["teacher_models"]:
            path = _teacher_path(output, args.subject, int(seed), str(model_name))
            expected_sha = barrier["checkpoints"].get(str(path))
            if expected_sha is None or file_sha256(path) != expected_sha:
                raise RuntimeError(f"V30 teacher changed after barrier: {path}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = _build_teacher(
                str(model_name),
                checkpoint,
                source_root=Path(args.source_root).resolve(),
                device=args.device,
            )
            before = _state_sha256(model)
            sequence, logits = _extract_features(
                str(model_name),
                model,
                prepared[str(model_name)],
                device=args.device,
                batch_size=args.feature_batch_size,
            )
            after = _state_sha256(model)
            if before != after or file_sha256(path) != expected_sha:
                raise RuntimeError(f"V30 teacher state changed during evaluation: {path}")
            features[str(model_name)] = sequence
            teacher_logits[str(model_name)] = logits
            teacher_states[str(model_name)] = {
                "checkpoint_sha256": expected_sha,
                "state_before": before,
                "state_after": after,
                "state_unchanged": True,
            }
            del model, checkpoint
        equal_teacher = equal_probability_teacher(
            teacher_logits["atcnet"], teacher_logits["fbcnet"]
        )
        evaluation_dir = ensure_dir(seed_dir / "evaluation")
        with (evaluation_dir / "reference_predictions.npz").open("wb") as handle:
            np.savez_compressed(
                handle,
                atcnet_logits=teacher_logits["atcnet"],
                fbcnet_logits=teacher_logits["fbcnet"],
                equal_teacher_logits=equal_teacher,
                label=labels,
                trial_id=np.asarray(trial_ids),
            )
        write_json(
            evaluation_dir / "reference_metrics.json",
            {
                "atcnet": _metric_payload(labels, teacher_logits["atcnet"]),
                "fbcnet": _metric_payload(labels, teacher_logits["fbcnet"]),
                "equal_teacher": _metric_payload(labels, equal_teacher),
                "teacher_state_audit": teacher_states,
                "evaluation_gradient_updates": False,
                "data_identity": data_identity,
            },
        )

        standardizer = _load_standardizer(seed_dir / "feature_standardizer.npz")
        eval_atc, eval_fbc = standardizer.transform(
            features["atcnet"], features["fbcnet"]
        )
        for variant in config["variants"]:
            path = _student_path(output, args.subject, int(seed), str(variant))
            expected_sha = barrier["checkpoints"].get(str(path))
            if expected_sha is None or file_sha256(path) != expected_sha:
                raise RuntimeError(f"V30 student changed after barrier: {path}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = build_v9_dual_feature_student(
                "ann_sew" if variant == "ann_sew_ce" else "sew_clif",
                **dict(config["model"]),
            )
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            before = _state_sha256(model)
            prediction = predict_v9_dual_feature(
                model,
                eval_atc,
                eval_fbc,
                labels,
                equal_teacher,
                device=args.device,
                batch_size=int(config["training"]["student_batch_size"]),
            )
            after = _state_sha256(model)
            if before != after or file_sha256(path) != expected_sha:
                raise RuntimeError(f"V30 student state changed during evaluation: {path}")
            student_dir = ensure_dir(
                seed_dir / "students" / str(variant) / "evaluation"
            )
            with (student_dir / "predictions.npz").open("wb") as handle:
                np.savez_compressed(
                    handle,
                    logits=prediction["logits"],
                    pred=prediction["logits"].argmax(axis=1),
                    label=labels,
                    trial_id=np.asarray(trial_ids),
                )
            metrics = {
                key: float(prediction[key])
                for key in (
                    "accuracy",
                    "balanced_accuracy",
                    "kappa",
                    "macro_f1",
                    "mean_firing_rate",
                )
            }
            metrics.update(
                {
                    "status": "completed",
                    "subject": args.subject,
                    "seed": int(seed),
                    "variant": str(variant),
                    "training_checkpoint_sha256": expected_sha,
                    "state_before": before,
                    "state_after": after,
                    "state_unchanged_during_evaluation": True,
                    "data_identity": data_identity,
                    "evaluation_gradient_updates": False,
                }
            )
            write_json(student_dir / "metrics.json", metrics)
            del model, checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_json(
        subject_dir / "evaluation_status.json",
        {
            "status": "completed",
            "subject": args.subject,
            "evaluation_sessions": sessions,
            "evaluation_sessions_accessed": True,
            "evaluation_gradient_updates": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "evaluate"), required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-barrier")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.subject not in range(1, 10):
        raise ValueError("BNCI2014-004 subject must lie in [1, 9]")
    if args.phase == "evaluate" and not args.checkpoint_barrier:
        raise ValueError("V30 evaluation requires --checkpoint-barrier")

    configure_cache_env()
    freeze = read_json(Path(args.freeze).resolve())
    if freeze.get("status") != "frozen_before_any_bnci2014_004_evaluation_access":
        raise RuntimeError("Invalid V30 freeze manifest")
    current_source = source_tree_digest(collect_source_tree_manifest(ROOT))
    if current_source != freeze["source_tree_sha256"]:
        raise RuntimeError("V30 source tree changed after protocol freeze")
    config = yaml.safe_load(
        (ROOT / "configs" / "experiments" / "v30_bnci2014_004_blind.yaml").read_text(
            encoding="utf-8"
        )
    )
    started = time.time()
    if args.phase == "train":
        _train(args, config, freeze)
    else:
        _evaluate(args, config, freeze)
    print(
        json.dumps(
            {
                "status": "completed",
                "phase": args.phase,
                "subject": args.subject,
                "elapsed_seconds": time.time() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
