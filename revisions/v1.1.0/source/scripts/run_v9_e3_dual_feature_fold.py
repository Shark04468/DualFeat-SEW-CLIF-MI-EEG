#!/usr/bin/env python3
"""Run one leakage-safe V9 dual-feature ANN/SNN development fold."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.baselines.neural import (  # noqa: E402
    build_v62_neural_baseline,
    verify_official_source_locks,
)
from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v62_baselines import (  # noqa: E402
    FixedGain,
    apply_fixed_gain,
    prepare_model_input,
    task_carrier,
)
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    session_t_run_grouped_folds,
)
from dpc_snn.experiments.v8_baselines import (  # noqa: E402
    nested_run_grouped_indices,
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
from dpc_snn.experiments.v9_dual_feature_training import (  # noqa: E402
    V9_DUAL_FEATURE_EXPERIMENT_VARIANTS,
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
from scripts.run_v8_e1_baselines import _subject_file  # noqa: E402


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _environment_manifest(device: str) -> dict[str, Any]:
    cuda_device = None
    if device.startswith("cuda") and torch.cuda.is_available():
        cuda_device = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_device": cuda_device,
        "deterministic_cudnn": True,
    }


def fixed_gain_from_result(result: Mapping[str, Any], key: str, *, clip: float) -> FixedGain:
    values = np.asarray(result[key], dtype=np.float32)
    if values.shape != (22,) or not np.isfinite(values).all() or np.any(values <= 0.0):
        raise RuntimeError(f"saved {key} is not a finite positive 22-channel gain")
    return FixedGain(values=values.reshape(1, 22, 1), clip=float(clip))


def _load_prediction(
    path: Path, expected_indices: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"indices", "logits", "labels"}:
            raise RuntimeError(f"invalid fold prediction archive: {path}")
        indices = archive["indices"]
        logits = archive["logits"]
        saved_labels = archive["labels"]
    if not np.array_equal(indices, expected_indices) or not np.array_equal(saved_labels, labels):
        raise RuntimeError(f"fold predictions are not aligned: {path}")
    return np.asarray(logits, dtype=np.float32)


@torch.no_grad()
def _extract_atc(
    *,
    source_root: Path,
    checkpoint: Path,
    carrier: np.ndarray,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    adapter = build_v62_neural_baseline(
        "atcnet", source_root=source_root, n_channels=22, n_classes=4, samples=1000
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    adapter.load_state_dict(state, strict=True)
    wrapper = V8ATCAccuracyBackbone(adapter.module).eval().to(device)
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
        sequences.append(output["aux"]["continuous_sequence"].float().cpu().numpy())
        logits.append(output["logits"].float().cpu().numpy())
    sequence = np.concatenate(sequences)
    all_logits = np.concatenate(logits)
    if sequence.shape != (carrier.shape[0], 18, 32) or all_logits.shape != (
        carrier.shape[0],
        4,
    ):
        raise RuntimeError("official ATCNet extraction returned an unexpected shape")
    del wrapper, adapter, state
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return sequence, all_logits


@torch.no_grad()
def _extract_fbc(
    *,
    source_root: Path,
    checkpoint: Path,
    carrier: np.ndarray,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    adapter = build_v62_neural_baseline(
        "fbcnet", source_root=source_root, n_channels=22, n_classes=4, samples=1000
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    adapter.load_state_dict(state, strict=True)
    wrapper = V9FBCFeatureBackbone(adapter.module).eval().to(device)
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
        sequences.append(output["continuous_sequence"].float().cpu().numpy())
        logits.append(output["logits"].float().cpu().numpy())
    sequence = np.concatenate(sequences)
    all_logits = np.concatenate(logits)
    if sequence.shape != (carrier.shape[0], 4, 288) or all_logits.shape != (
        carrier.shape[0],
        4,
    ):
        raise RuntimeError("official FBCNet extraction returned an unexpected shape")
    del wrapper, adapter, state
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return sequence, all_logits


def _capacity_audit(variants: Sequence[str]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for variant in variants:
        specification = V9_DUAL_FEATURE_EXPERIMENT_VARIANTS[variant]
        model = build_v9_dual_feature_student(specification.model_variant)
        rows[variant] = {
            "model_variant": specification.model_variant,
            "objective": specification.objective,
            "parameters": model.parameter_count,
            "trainable_parameters": model.trainable_parameter_count,
            "parameter_shapes": sorted([list(parameter.shape) for parameter in model.parameters()]),
            "decoder_kind": model.decoder_kind,
            "residual_mode": model.residual_mode,
        }
    counts = {row["parameters"] for row in rows.values()}
    trainable = {row["trainable_parameters"] for row in rows.values()}
    shapes = {json.dumps(row["parameter_shapes"], separators=(",", ":")) for row in rows.values()}
    if len(counts) != 1 or len(trainable) != 1 or len(shapes) != 1:
        raise RuntimeError("V9 dual-feature variants are not exactly capacity matched")
    return {
        "status": "passed",
        "permitted_differences": ["state equation", "residual merge", "training objective"],
        "variants": rows,
    }


def _metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    return classification_metrics(labels, logits.argmax(axis=1), n_classes=4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--e1-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--variants", default=",".join(V9_DUAL_FEATURE_EXPERIMENT_VARIANTS)
    )
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--minimum-outer-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    configure_cache_env()
    data_root = Path(args.data).resolve()
    source_root = Path(args.source_root).resolve()
    e1_root = Path(args.e1_root).resolve()
    output = Path(args.output).resolve()
    variants = _csv(args.variants)
    unknown = sorted(set(variants) - set(V9_DUAL_FEATURE_EXPERIMENT_VARIANTS))
    if unknown or len(variants) != len(set(variants)) or not variants:
        raise ValueError(f"invalid V9 dual-feature variants: {unknown or variants}")
    if args.subject not in range(1, 10) or args.fold not in range(6):
        raise ValueError("subject must lie in [1, 9] and fold in [0, 5]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    subject_path = _subject_file(data_root, int(args.subject))
    data = load_processed_npz(subject_path)
    x_raw, labels, metadata, access = session_t_development_view(data)
    raw_carrier = task_carrier(
        x_raw, sfreq=float(data["sfreq"]), epoch_tmin=float(data["epoch_tmin"])
    )
    folds = session_t_run_grouped_folds(metadata, n_splits=6, seed=0, shuffle=True)
    outer_train, outer_test = folds[int(args.fold)]
    inner_train, inner_validation, inner_run = nested_run_grouped_indices(
        metadata, outer_train, outer_test
    )

    branch_dirs: dict[str, Path] = {}
    branch_results: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    branch_gains: dict[str, tuple[FixedGain, FixedGain]] = {}
    checkpoint_paths: dict[str, Path] = {}
    clip = 12.0
    for branch in ("atcnet", "fbcnet"):
        fold_dir = (
            e1_root
            / branch
            / f"subject_{args.subject:02d}"
            / f"seed_{args.seed}"
            / f"fold_{args.fold}"
        )
        selection_result = read_json(fold_dir / "selection_result.json")
        outer_result = read_json(fold_dir / "result.json")
        if str(selection_result["inner_validation_run"]) != str(inner_run):
            raise RuntimeError(f"{branch} inner validation run differs from recomputed split")
        branch_dirs[branch] = fold_dir
        branch_results[branch] = (selection_result, outer_result)
        branch_gains[branch] = (
            fixed_gain_from_result(selection_result, "gain", clip=clip),
            fixed_gain_from_result(outer_result, "outer_gain", clip=clip),
        )
        checkpoint_paths[f"{branch}_selection"] = fold_dir / "selection_best.pt"
        checkpoint_paths[f"{branch}_outer"] = fold_dir / "best.pt"

    source_tree = collect_source_tree_manifest(ROOT)
    resolved_config = {
        **vars(args),
        "data": str(data_root),
        "source_root": str(source_root),
        "e1_root": str(e1_root),
        "output": str(output),
        "variants": variants,
        "protocol_stage": "Session-T development only",
        "teacher_target": "equal ATCNet/FBCNet probability",
    }
    fingerprint = build_v8_run_fingerprint(
        resolved_run_config=resolved_config,
        source_tree=source_tree,
        data={"path": str(subject_path), "sha256": file_sha256(subject_path)},
        split={
            "outer_train": _array_sha256(outer_train),
            "outer_test": _array_sha256(outer_test),
            "inner_train": _array_sha256(inner_train),
            "inner_validation": _array_sha256(inner_validation),
            "inner_validation_run": str(inner_run),
        },
        augmentation={"enabled": False},
        prior={"enabled": False, "feature_standardization": "train-fold-only"},
        checkpoint={name: file_sha256(path) for name, path in checkpoint_paths.items()},
        environment=_environment_manifest(args.device),
    )
    if (output / "campaign_status.json").is_file():
        validate_v8_resume_fingerprint(output / "run_fingerprint.json", fingerprint)
        status = read_json(output / "campaign_status.json")
        if status.get("status") == "completed":
            print(json.dumps({"status": "skipped_completed", "output": str(output)}))
            return

    ensure_dir(output)
    write_json(output / "heldout_lock_manifest.json", v8_heldout_lock_manifest())
    write_json(output / "official_source_locks.json", verify_official_source_locks(source_root))
    write_json(output / "source_tree_manifest.json", source_tree)
    write_json(
        output / "source_tree_summary.json",
        {"files": len(source_tree), "sha256": source_tree_digest(source_tree)},
    )
    write_v8_fingerprint(output / "run_fingerprint.json", fingerprint)
    (output / "resolved_run.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
    )
    capacity = _capacity_audit(variants)
    write_json(output / "capacity_audit.json", capacity)

    def prepared(branch: str, indices: np.ndarray, gain: FixedGain) -> np.ndarray:
        return prepare_model_input(
            branch,
            apply_fixed_gain(raw_carrier[indices], gain),
            sfreq=float(data["sfreq"]),
        )

    partitions = {
        "selection_train": inner_train,
        "selection_validation": inner_validation,
        "outer_train": outer_train,
        "outer_test": outer_test,
    }
    branch_features: dict[str, dict[str, np.ndarray]] = {"atcnet": {}, "fbcnet": {}}
    branch_logits: dict[str, dict[str, np.ndarray]] = {"atcnet": {}, "fbcnet": {}}
    for branch, extractor in (("atcnet", _extract_atc), ("fbcnet", _extract_fbc)):
        selection_gain, outer_gain = branch_gains[branch]
        for partition, indices in partitions.items():
            selection = partition.startswith("selection")
            checkpoint = checkpoint_paths[f"{branch}_{'selection' if selection else 'outer'}"]
            gain = selection_gain if selection else outer_gain
            sequence, logits = extractor(
                source_root=source_root,
                checkpoint=checkpoint,
                carrier=prepared(branch, indices, gain),
                device=args.device,
                batch_size=int(args.batch_size),
            )
            branch_features[branch][partition] = sequence
            branch_logits[branch][partition] = logits

    replay: dict[str, Any] = {"status": "passed", "branches": {}}
    for branch in ("atcnet", "fbcnet"):
        fold_dir = branch_dirs[branch]
        saved_selection = _load_prediction(
            fold_dir / "selection_predictions.npz",
            inner_validation,
            labels[inner_validation],
        )
        saved_outer = _load_prediction(
            fold_dir / "outer_test_predictions.npz", outer_test, labels[outer_test]
        )
        selection_error = float(
            np.max(
                np.abs(saved_selection - branch_logits[branch]["selection_validation"])
            )
        )
        outer_error = float(
            np.max(np.abs(saved_outer - branch_logits[branch]["outer_test"]))
        )
        if selection_error > 1e-5 or outer_error > 1e-5:
            raise RuntimeError(
                f"{branch} checkpoint replay failed: selection={selection_error}, "
                f"outer={outer_error}"
            )
        replay["branches"][branch] = {
            "selection_max_abs_logit_error": selection_error,
            "outer_max_abs_logit_error": outer_error,
            "selection_checkpoint_sha256": file_sha256(
                checkpoint_paths[f"{branch}_selection"]
            ),
            "outer_checkpoint_sha256": file_sha256(checkpoint_paths[f"{branch}_outer"]),
        }
    write_json(output / "teacher_replay.json", replay)

    teacher_targets = {
        partition: equal_probability_teacher(
            branch_logits["atcnet"][partition], branch_logits["fbcnet"][partition]
        )
        for partition in partitions
    }
    selection_standardizer = fit_feature_standardizer(
        branch_features["atcnet"]["selection_train"],
        branch_features["fbcnet"]["selection_train"],
    )
    outer_standardizer = fit_feature_standardizer(
        branch_features["atcnet"]["outer_train"],
        branch_features["fbcnet"]["outer_train"],
    )
    write_json(output / "selection_feature_standardizer.json", selection_standardizer.as_dict())
    write_json(output / "outer_feature_standardizer.json", outer_standardizer.as_dict())

    standardized: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for partition in partitions:
        standardizer = (
            selection_standardizer if partition.startswith("selection") else outer_standardizer
        )
        standardized[partition] = standardizer.transform(
            branch_features["atcnet"][partition],
            branch_features["fbcnet"][partition],
        )

    np.savez_compressed(
        output / "frozen_dual_feature_cache.npz",
        inner_train_indices=inner_train,
        inner_validation_indices=inner_validation,
        outer_train_indices=outer_train,
        outer_test_indices=outer_test,
        **{
            f"{branch}_{partition}_{kind}": value
            for branch in ("atcnet", "fbcnet")
            for partition in partitions
            for kind, value in (
                ("sequence", branch_features[branch][partition]),
                ("logits", branch_logits[branch][partition]),
            )
        },
        **{f"teacher_{partition}": value for partition, value in teacher_targets.items()},
    )

    outer_labels = labels[outer_test]
    atc_metrics = _metrics(outer_labels, branch_logits["atcnet"]["outer_test"])
    fbc_metrics = _metrics(outer_labels, branch_logits["fbcnet"]["outer_test"])
    teacher_metrics = _metrics(outer_labels, teacher_targets["outer_test"])
    rows: list[dict[str, Any]] = []
    started = time.time()
    fold_seed = int(args.seed) * 100_003 + int(args.fold) * 1_009 + 9_300_007
    for variant in variants:
        variant_dir = ensure_dir(output / variant)
        selection_atc, selection_fbc = standardized["selection_train"]
        validation_atc, validation_fbc = standardized["selection_validation"]
        selection_fit = fit_v9_dual_feature(
            variant,
            atc_train=selection_atc,
            fbc_train=selection_fbc,
            y_train=labels[inner_train],
            teacher_train=teacher_targets["selection_train"],
            atc_validation=validation_atc,
            fbc_validation=validation_fbc,
            y_validation=labels[inner_validation],
            teacher_validation=teacher_targets["selection_validation"],
            device=args.device,
            seed=fold_seed,
            epochs=int(args.epochs),
            patience=int(args.patience),
            batch_size=int(args.batch_size),
            run_label=f"V9-E3-select:{variant}:S{args.subject}:seed{args.seed}:fold{args.fold}",
        )
        selected_epoch = min(
            int(args.epochs),
            max(int(args.minimum_outer_epochs), int(selection_fit.best_epoch)),
        )
        outer_train_atc, outer_train_fbc = standardized["outer_train"]
        outer_test_atc, outer_test_fbc = standardized["outer_test"]
        outer_fit = fit_v9_dual_feature(
            variant,
            atc_train=outer_train_atc,
            fbc_train=outer_train_fbc,
            y_train=labels[outer_train],
            teacher_train=teacher_targets["outer_train"],
            atc_validation=None,
            fbc_validation=None,
            y_validation=None,
            teacher_validation=None,
            device=args.device,
            seed=fold_seed + 1_000_003,
            epochs=int(args.epochs),
            fixed_epoch=selected_epoch,
            scheduler_epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            run_label=f"V9-E3-outer:{variant}:S{args.subject}:seed{args.seed}:fold{args.fold}",
        )
        evaluation = predict_v9_dual_feature(
            outer_fit.model,
            outer_test_atc,
            outer_test_fbc,
            outer_labels,
            teacher_targets["outer_test"],
            device=args.device,
            batch_size=int(args.batch_size),
        )
        specification = V9_DUAL_FEATURE_EXPERIMENT_VARIANTS[variant]
        row = {
            "variant": variant,
            "model_variant": specification.model_variant,
            "objective": specification.objective,
            "subject": int(args.subject),
            "seed": int(args.seed),
            "fold": int(args.fold),
            "selected_epoch": selected_epoch,
            "inner_best_epoch": selection_fit.best_epoch,
            "inner_best_kappa": selection_fit.best_metric,
            "inner_best_accuracy": selection_fit.best_accuracy,
            "student_accuracy": evaluation["accuracy"],
            "student_kappa": evaluation["kappa"],
            "student_mean_firing_rate": evaluation["mean_firing_rate"],
            "atc_accuracy": atc_metrics["accuracy"],
            "fbc_accuracy": fbc_metrics["accuracy"],
            "equal_teacher_accuracy": teacher_metrics["accuracy"],
            "delta_vs_atc_pp": 100.0
            * (float(evaluation["accuracy"]) - float(atc_metrics["accuracy"])),
            "delta_vs_fbc_pp": 100.0
            * (float(evaluation["accuracy"]) - float(fbc_metrics["accuracy"])),
            "delta_vs_equal_teacher_pp": 100.0
            * (float(evaluation["accuracy"]) - float(teacher_metrics["accuracy"])),
            "parameters": capacity["variants"][variant]["parameters"],
            "optimizer_steps": selection_fit.optimizer_steps + outer_fit.optimizer_steps,
            "train_seconds": selection_fit.elapsed_seconds + outer_fit.elapsed_seconds,
            "session_e_accessed": False,
        }
        rows.append(row)
        torch.save(selection_fit.best_state, variant_dir / "selection_best.pt")
        torch.save(outer_fit.last_state, variant_dir / "outer_last.pt")
        write_csv(variant_dir / "selection_history.csv", selection_fit.history)
        write_csv(variant_dir / "outer_history.csv", outer_fit.history)
        write_json(variant_dir / "metrics.json", row)
        np.savez_compressed(
            variant_dir / "outer_predictions.npz",
            indices=outer_test,
            logits=np.asarray(evaluation["logits"], dtype=np.float32),
            labels=outer_labels,
        )
        del selection_fit, outer_fit, evaluation
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    write_csv(output / "summary.csv", rows)
    status = {
        "status": "completed",
        "stage": "V9-E3-dual-feature-development-fold",
        "scope": "one BCI2a Session-T nested outer fold",
        "subject": int(args.subject),
        "seed": int(args.seed),
        "fold": int(args.fold),
        "variants": variants,
        "teacher_replay": replay,
        "atc_accuracy": atc_metrics["accuracy"],
        "fbc_accuracy": fbc_metrics["accuracy"],
        "equal_teacher_accuracy": teacher_metrics["accuracy"],
        "elapsed_seconds": time.time() - started,
        "data_access": access,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
        "run_fingerprint": fingerprint["combined_sha256"],
    }
    write_json(output / "campaign_status.json", status)
    print(json.dumps({"status": "completed", "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
