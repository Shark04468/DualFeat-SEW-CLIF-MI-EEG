#!/usr/bin/env python3
"""Leakage-safe matched ANN/SNN canary on one frozen official ATCNet fold."""

from __future__ import annotations

import argparse
import json
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
    collect_source_tree_manifest,
    source_tree_digest,
    v8_heldout_lock_manifest,
)
from dpc_snn.experiments.v8_sequence_decoder_training import (  # noqa: E402
    fit_v8_sequence_decoder,
    predict_v8_sequence_decoder,
)
from dpc_snn.models.v8_atc_backbone import V8ATCAccuracyBackbone  # noqa: E402
from dpc_snn.models.v8_sequence_decoder import (  # noqa: E402
    V8_SEQUENCE_DECODER_VARIANTS,
    build_v8_sequence_decoder,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v8_e1_baselines import _subject_file  # noqa: E402


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def fixed_gain_from_result(result: Mapping[str, Any], key: str, *, clip: float) -> FixedGain:
    values = np.asarray(result[key], dtype=np.float32)
    if values.shape != (22,) or not np.isfinite(values).all() or np.any(values <= 0.0):
        raise RuntimeError(f"saved {key} is not a finite positive 22-channel gain")
    return FixedGain(values=values.reshape(1, 22, 1), clip=float(clip))


def probability_fusion(
    first_logits: np.ndarray,
    second_logits: np.ndarray,
    *,
    first_weight: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= first_weight <= 1.0:
        raise ValueError("fusion weight must lie in [0, 1]")
    first = torch.softmax(torch.as_tensor(first_logits, dtype=torch.float32), dim=1).numpy()
    second = torch.softmax(torch.as_tensor(second_logits, dtype=torch.float32), dim=1).numpy()
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("fusion logits are not aligned")
    probability = float(first_weight) * first + (1.0 - float(first_weight)) * second
    return probability, probability.argmax(axis=1)


@torch.no_grad()
def _extract_atc_sequence(
    *,
    source_root: Path,
    checkpoint: Path,
    x: np.ndarray,
    device: str,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    adapter = build_v62_neural_baseline(
        "atcnet", source_root=source_root, n_channels=22, n_classes=4, samples=1000
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    adapter.load_state_dict(state, strict=True)
    wrapper = V8ATCAccuracyBackbone(adapter.module).eval().to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    sequence: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for (batch_x,) in loader:
        output = wrapper(batch_x.to(device, non_blocking=True))
        sequence.append(output["aux"]["continuous_sequence"].float().cpu().numpy())
        logits.append(output["logits"].float().cpu().numpy())
    all_sequence = np.concatenate(sequence)
    all_logits = np.concatenate(logits)
    if all_sequence.shape != (x.shape[0], 18, 32) or all_logits.shape != (x.shape[0], 4):
        raise RuntimeError("official ATC feature extraction returned an unexpected shape")
    if not np.isfinite(all_sequence).all() or not np.isfinite(all_logits).all():
        raise FloatingPointError("official ATC feature extraction returned non-finite values")
    del wrapper, adapter, state
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return all_sequence, all_logits


def _load_prediction(path: Path, expected_indices: np.ndarray, labels: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"indices", "logits", "labels"}:
            raise RuntimeError(f"invalid fold prediction archive: {path}")
        indices = archive["indices"]
        logits = archive["logits"]
        saved_labels = archive["labels"]
    if not np.array_equal(indices, expected_indices) or not np.array_equal(saved_labels, labels):
        raise RuntimeError(f"fold predictions are not aligned: {path}")
    return np.asarray(logits, dtype=np.float32)


def _capacity_audit(variants: Sequence[str]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for variant in variants:
        model = build_v8_sequence_decoder(variant)
        rows[variant] = {
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
        raise RuntimeError("ATC sequence decoder variants are not exactly capacity matched")
    return {
        "status": "passed",
        "permitted_differences": ["state equation", "residual merge"],
        "variants": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--e1-root", required=True)
    parser.add_argument("--fbc-root", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--variants", default=",".join(V8_SEQUENCE_DECODER_VARIANTS))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--minimum-outer-epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    configure_cache_env()
    data_root = Path(args.data).resolve()
    source_root = Path(args.source_root).resolve()
    e1_root = Path(args.e1_root).resolve()
    fbc_root = Path(args.fbc_root).resolve() if args.fbc_root else e1_root
    output = ensure_dir(Path(args.output).resolve())
    variants = _csv(args.variants)
    unknown = sorted(set(variants) - set(V8_SEQUENCE_DECODER_VARIANTS))
    if unknown or len(variants) != len(set(variants)):
        raise ValueError(f"invalid decoder variants: {unknown or variants}")
    if args.fold not in range(6):
        raise ValueError("canary fold must lie in [0, 5]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    official_locks = verify_official_source_locks(source_root)
    source_tree = collect_source_tree_manifest(ROOT)
    write_json(output / "heldout_lock_manifest.json", v8_heldout_lock_manifest())
    write_json(output / "official_source_locks.json", official_locks)
    write_json(output / "source_tree_manifest.json", source_tree)
    write_json(
        output / "source_tree_summary.json",
        {"files": len(source_tree), "sha256": source_tree_digest(source_tree)},
    )
    capacity = _capacity_audit(variants)
    write_json(output / "capacity_audit.json", capacity)

    subject_path = _subject_file(data_root, int(args.subject))
    data = load_processed_npz(subject_path)
    x_raw, labels, metadata, access = session_t_development_view(data)
    carrier = task_carrier(
        x_raw, sfreq=float(data["sfreq"]), epoch_tmin=float(data["epoch_tmin"])
    )
    folds = session_t_run_grouped_folds(metadata, n_splits=6, seed=0, shuffle=True)
    outer_train, outer_test = folds[int(args.fold)]
    inner_train, inner_validation, inner_run = nested_run_grouped_indices(
        metadata, outer_train, outer_test
    )
    atc_run = e1_root / "atcnet" / f"subject_{args.subject:02d}" / f"seed_{args.seed}"
    fbc_run = fbc_root / "fbcnet" / f"subject_{args.subject:02d}" / f"seed_{args.seed}"
    fold_dir = atc_run / f"fold_{args.fold}"
    fbc_fold_dir = fbc_run / f"fold_{args.fold}"
    selection_result = read_json(fold_dir / "selection_result.json")
    outer_result = read_json(fold_dir / "result.json")
    if str(selection_result["inner_validation_run"]) != str(inner_run):
        raise RuntimeError("recomputed inner validation run differs from the E1 artifact")
    clip = 12.0
    selection_gain = fixed_gain_from_result(selection_result, "gain", clip=clip)
    outer_gain = fixed_gain_from_result(outer_result, "outer_gain", clip=clip)

    def prepared(indices: np.ndarray, gain: FixedGain) -> np.ndarray:
        return prepare_model_input(
            "atcnet",
            apply_fixed_gain(carrier[indices], gain),
            sfreq=float(data["sfreq"]),
        )

    selection_checkpoint = fold_dir / "selection_best.pt"
    outer_checkpoint = fold_dir / "best.pt"
    selection_train_sequence, selection_train_teacher = _extract_atc_sequence(
        source_root=source_root,
        checkpoint=selection_checkpoint,
        x=prepared(inner_train, selection_gain),
        device=args.device,
    )
    selection_validation_sequence, selection_validation_teacher = _extract_atc_sequence(
        source_root=source_root,
        checkpoint=selection_checkpoint,
        x=prepared(inner_validation, selection_gain),
        device=args.device,
    )
    outer_train_sequence, outer_train_teacher = _extract_atc_sequence(
        source_root=source_root,
        checkpoint=outer_checkpoint,
        x=prepared(outer_train, outer_gain),
        device=args.device,
    )
    outer_test_sequence, outer_test_teacher = _extract_atc_sequence(
        source_root=source_root,
        checkpoint=outer_checkpoint,
        x=prepared(outer_test, outer_gain),
        device=args.device,
    )

    saved_selection = _load_prediction(
        fold_dir / "selection_predictions.npz",
        inner_validation,
        labels[inner_validation],
    )
    saved_outer = _load_prediction(
        fold_dir / "outer_test_predictions.npz", outer_test, labels[outer_test]
    )
    selection_max_abs = float(np.max(np.abs(saved_selection - selection_validation_teacher)))
    outer_max_abs = float(np.max(np.abs(saved_outer - outer_test_teacher)))
    if selection_max_abs > 1e-5 or outer_max_abs > 1e-5:
        raise RuntimeError(
            "ATC checkpoint replay is not logit-equivalent: "
            f"selection={selection_max_abs}, outer={outer_max_abs}"
        )
    fbc_outer_logits = _load_prediction(
        fbc_fold_dir / "outer_test_predictions.npz", outer_test, labels[outer_test]
    )
    _, anchor_fused_prediction = probability_fusion(outer_test_teacher, fbc_outer_logits)
    teacher_metrics = classification_metrics(
        labels[outer_test], outer_test_teacher.argmax(axis=1), n_classes=4
    )
    fbc_metrics = classification_metrics(
        labels[outer_test], fbc_outer_logits.argmax(axis=1), n_classes=4
    )
    anchor_fused_metrics = classification_metrics(
        labels[outer_test], anchor_fused_prediction, n_classes=4
    )
    replay = {
        "status": "passed",
        "selection_max_abs_logit_error": selection_max_abs,
        "outer_max_abs_logit_error": outer_max_abs,
        "selection_checkpoint_sha256": file_sha256(selection_checkpoint),
        "outer_checkpoint_sha256": file_sha256(outer_checkpoint),
        "fbc_outer_predictions_sha256": file_sha256(
            fbc_fold_dir / "outer_test_predictions.npz"
        ),
    }
    write_json(output / "teacher_replay.json", replay)
    np.savez_compressed(
        output / "frozen_sequence_cache.npz",
        inner_train_indices=inner_train,
        inner_validation_indices=inner_validation,
        outer_train_indices=outer_train,
        outer_test_indices=outer_test,
        selection_train_sequence=selection_train_sequence,
        selection_validation_sequence=selection_validation_sequence,
        outer_train_sequence=outer_train_sequence,
        outer_test_sequence=outer_test_sequence,
        selection_train_teacher=selection_train_teacher,
        selection_validation_teacher=selection_validation_teacher,
        outer_train_teacher=outer_train_teacher,
        outer_test_teacher=outer_test_teacher,
    )

    rows: list[dict[str, Any]] = []
    started = time.time()
    for variant in variants:
        variant_dir = ensure_dir(output / variant)
        fold_seed = int(args.seed) * 100_003 + int(args.fold) * 1_009 + 8_000_003
        selection_fit = fit_v8_sequence_decoder(
            variant,
            x_train=selection_train_sequence,
            y_train=labels[inner_train],
            teacher_train=selection_train_teacher,
            x_validation=selection_validation_sequence,
            y_validation=labels[inner_validation],
            teacher_validation=selection_validation_teacher,
            device=args.device,
            seed=fold_seed,
            epochs=int(args.epochs),
            patience=int(args.patience),
            run_label=f"V8-E4-canary-select:{variant}:S{args.subject}:fold{args.fold}",
        )
        selected_epoch = min(
            int(args.epochs),
            max(int(args.minimum_outer_epochs), int(selection_fit.best_epoch)),
        )
        outer_fit = fit_v8_sequence_decoder(
            variant,
            x_train=outer_train_sequence,
            y_train=labels[outer_train],
            teacher_train=outer_train_teacher,
            x_validation=None,
            y_validation=None,
            teacher_validation=None,
            device=args.device,
            seed=fold_seed + 1_000_003,
            epochs=int(args.epochs),
            fixed_epoch=selected_epoch,
            scheduler_epochs=int(args.epochs),
            run_label=f"V8-E4-canary-outer:{variant}:S{args.subject}:fold{args.fold}",
        )
        evaluation = predict_v8_sequence_decoder(
            outer_fit.model,
            outer_test_sequence,
            labels[outer_test],
            outer_test_teacher,
            device=args.device,
        )
        _, fused_prediction = probability_fusion(evaluation["logits"], fbc_outer_logits)
        fused_metrics = classification_metrics(labels[outer_test], fused_prediction, n_classes=4)
        row = {
            "variant": variant,
            "subject": int(args.subject),
            "seed": int(args.seed),
            "fold": int(args.fold),
            "selected_epoch": selected_epoch,
            "inner_best_epoch": selection_fit.best_epoch,
            "inner_best_kappa": selection_fit.best_metric,
            "decoder_accuracy": evaluation["accuracy"],
            "decoder_kappa": evaluation["kappa"],
            "decoder_mean_firing_rate": evaluation["mean_firing_rate"],
            "decoder_plus_fbc_accuracy": fused_metrics["accuracy"],
            "decoder_plus_fbc_kappa": fused_metrics["kappa"],
            "teacher_atc_accuracy": teacher_metrics["accuracy"],
            "fbc_accuracy": fbc_metrics["accuracy"],
            "anchor_atc_plus_fbc_accuracy": anchor_fused_metrics["accuracy"],
            "decoder_delta_vs_atc_pp": 100.0
            * (float(evaluation["accuracy"]) - float(teacher_metrics["accuracy"])),
            "fused_delta_vs_anchor_pp": 100.0
            * (float(fused_metrics["accuracy"]) - float(anchor_fused_metrics["accuracy"])),
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
            labels=labels[outer_test],
        )
        del selection_fit, outer_fit, evaluation
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    write_csv(output / "summary.csv", rows)
    status = {
        "status": "completed",
        "stage": "E4_canary",
        "scope": "single Session-T nested outer fold",
        "subject": int(args.subject),
        "seed": int(args.seed),
        "fold": int(args.fold),
        "variants": variants,
        "teacher_replay": replay,
        "teacher_atc_accuracy": teacher_metrics["accuracy"],
        "fbc_accuracy": fbc_metrics["accuracy"],
        "anchor_atc_plus_fbc_accuracy": anchor_fused_metrics["accuracy"],
        "elapsed_seconds": time.time() - started,
        "data_access": access,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    write_json(output / "campaign_status.json", status)
    (output / "resolved_canary.yaml").write_text(
        yaml.safe_dump(vars(args), sort_keys=False), encoding="utf-8"
    )
    print(json.dumps({"status": "completed", "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
