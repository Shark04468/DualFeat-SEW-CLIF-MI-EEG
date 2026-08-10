#!/usr/bin/env python3
"""Run one equal-budget V10 strong-control development fold."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v62_protocol import file_sha256  # noqa: E402
from dpc_snn.experiments.v8_baselines import session_t_development_view  # noqa: E402
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    build_v8_run_fingerprint,
    collect_source_tree_manifest,
    source_tree_digest,
    validate_v8_resume_fingerprint,
    v8_heldout_lock_manifest,
    write_v8_fingerprint,
)
from dpc_snn.experiments.v9_dual_feature_training import (  # noqa: E402
    fit_feature_standardizer,
)
from dpc_snn.experiments.v10_control_training import (  # noqa: E402
    fit_v10_control,
    predict_v10_control,
)
from dpc_snn.models.v10_strong_controls import (  # noqa: E402
    V10_STRONG_CONTROL_MODELS,
    build_v10_strong_control,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v8_e1_baselines import _subject_file  # noqa: E402


ANCHOR_SOURCE_DIGEST = "8fe5f147e7ce5686fb6dcf85c75c7935c3f0df917d33be8cfa63966706ff6d00"


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_float(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)) or any(item <= 0.0 for item in values):
        raise ValueError("floating-point grid must be positive and unique")
    return values


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


def _capacity_audit(models: Sequence[str]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name in models:
        model = build_v10_strong_control(name)
        count = sum(parameter.numel() for parameter in model.parameters())
        rows[name] = {
            "parameters": count,
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "parameter_shapes": sorted([list(parameter.shape) for parameter in model.parameters()]),
        }
    counts = [int(row["parameters"]) for row in rows.values()]
    ratio = max(counts) / min(counts)
    if ratio > 1.05:
        raise RuntimeError(f"strong controls exceed the 5% capacity tolerance: ratio={ratio}")
    return {
        "status": "passed",
        "maximum_to_minimum_ratio": ratio,
        "tolerance": 1.05,
        "models": rows,
        "permitted_differences": ["sequence state equation", "recurrent topology"],
    }


def _load_anchor_predictions(
    path: Path, expected_indices: np.ndarray, expected_labels: np.ndarray
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"indices", "logits", "labels"}:
            raise RuntimeError(f"invalid anchor predictions: {path}")
        indices = archive["indices"]
        logits = archive["logits"]
        labels = archive["labels"]
    if not np.array_equal(indices, expected_indices) or not np.array_equal(
        labels, expected_labels
    ):
        raise RuntimeError("V9 anchor predictions are not aligned with V10 labels")
    return np.asarray(logits, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--anchor-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--models", default=",".join(V10_STRONG_CONTROL_MODELS))
    parser.add_argument("--learning-rates", default="0.0003,0.001,0.003")
    parser.add_argument("--weight-decays", default="0.0001,0.001")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--minimum-outer-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    configure_cache_env()
    data_root = Path(args.data).resolve()
    anchor_root = Path(args.anchor_root).resolve()
    output = Path(args.output).resolve()
    models = _csv(args.models)
    learning_rates = _csv_float(args.learning_rates)
    weight_decays = _csv_float(args.weight_decays)
    if set(models) - set(V10_STRONG_CONTROL_MODELS) or len(models) != len(set(models)):
        raise ValueError("unknown or duplicate V10 strong controls")
    if not models or args.subject not in range(1, 10) or args.fold not in range(6):
        raise ValueError("V10 subject, fold, or model list is invalid")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    anchor_fold = (
        anchor_root
        / f"subject_{args.subject:02d}"
        / f"seed_{args.seed}"
        / f"fold_{args.fold}"
    )
    anchor_status = read_json(anchor_fold / "campaign_status.json")
    anchor_source = read_json(anchor_fold / "source_tree_summary.json")
    anchor_replay = read_json(anchor_fold / "teacher_replay.json")
    if anchor_status.get("status") != "completed" or anchor_status.get(
        "session_e_accessed"
    ) is not False:
        raise RuntimeError("V9 anchor fold is incomplete or violated the Session E lock")
    if anchor_source.get("sha256") != ANCHOR_SOURCE_DIGEST:
        raise RuntimeError("V9 anchor source digest is not the frozen E3 snapshot")
    if anchor_replay.get("status") != "passed":
        raise RuntimeError("V9 teacher replay did not pass")
    for branch in ("atcnet", "fbcnet"):
        for key in ("selection_max_abs_logit_error", "outer_max_abs_logit_error"):
            if float(anchor_replay["branches"][branch][key]) > 1e-5:
                raise RuntimeError("V9 teacher replay exceeds tolerance")

    cache_path = anchor_fold / "frozen_dual_feature_cache.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        arrays = {name: np.asarray(cache[name]) for name in cache.files}
    required = {
        "inner_train_indices",
        "inner_validation_indices",
        "outer_train_indices",
        "outer_test_indices",
        "atcnet_selection_train_sequence",
        "atcnet_selection_validation_sequence",
        "atcnet_outer_train_sequence",
        "atcnet_outer_test_sequence",
        "fbcnet_selection_train_sequence",
        "fbcnet_selection_validation_sequence",
        "fbcnet_outer_train_sequence",
        "fbcnet_outer_test_sequence",
        "teacher_selection_train",
        "teacher_selection_validation",
        "teacher_outer_train",
        "teacher_outer_test",
    }
    if not required.issubset(arrays):
        raise RuntimeError("V9 frozen feature cache is incomplete")

    subject_path = _subject_file(data_root, int(args.subject))
    data = load_processed_npz(subject_path)
    _, labels, _, access = session_t_development_view(data)
    inner_train = arrays["inner_train_indices"].astype(np.int64)
    inner_validation = arrays["inner_validation_indices"].astype(np.int64)
    outer_train = arrays["outer_train_indices"].astype(np.int64)
    outer_test = arrays["outer_test_indices"].astype(np.int64)
    if any(
        np.unique(indices).size != indices.size
        for indices in (inner_train, inner_validation, outer_train, outer_test)
    ):
        raise RuntimeError("V9 cached split contains duplicate trial indices")

    anchor_snn_logits = _load_anchor_predictions(
        anchor_fold / "sew_clif_kd" / "outer_predictions.npz",
        outer_test,
        labels[outer_test],
    )
    anchor_ann_logits = _load_anchor_predictions(
        anchor_fold / "ann_sew_kd" / "outer_predictions.npz",
        outer_test,
        labels[outer_test],
    )
    source_tree = collect_source_tree_manifest(ROOT)
    capacity = _capacity_audit(models)
    grid = [
        {"learning_rate": learning_rate, "weight_decay": weight_decay}
        for learning_rate, weight_decay in itertools.product(learning_rates, weight_decays)
    ]
    resolved_config = {
        **vars(args),
        "data": str(data_root),
        "anchor_root": str(anchor_root),
        "output": str(output),
        "models": models,
        "learning_rates": learning_rates,
        "weight_decays": weight_decays,
        "hpo_grid": grid,
        "objective": "0.65 CE + 0.35 KD(T=2) + 0.01 firing-rate loss",
        "selection": "inner validation kappa, accuracy, earliest epoch, grid order",
    }
    fingerprint = build_v8_run_fingerprint(
        resolved_run_config=resolved_config,
        source_tree=source_tree,
        data={"path": str(subject_path), "sha256": file_sha256(subject_path)},
        split={
            "inner_train": _array_sha256(inner_train),
            "inner_validation": _array_sha256(inner_validation),
            "outer_train": _array_sha256(outer_train),
            "outer_test": _array_sha256(outer_test),
        },
        augmentation={"enabled": False},
        prior={"enabled": False, "feature_standardization": "train-fold-only"},
        checkpoint={
            "anchor_cache_sha256": file_sha256(cache_path),
            "anchor_run_fingerprint_sha256": file_sha256(
                anchor_fold / "run_fingerprint.json"
            ),
            "anchor_snn_predictions_sha256": file_sha256(
                anchor_fold / "sew_clif_kd" / "outer_predictions.npz"
            ),
            "anchor_ann_predictions_sha256": file_sha256(
                anchor_fold / "ann_sew_kd" / "outer_predictions.npz"
            ),
        },
        environment=_environment_manifest(args.device),
    )
    if (output / "campaign_status.json").is_file():
        validate_v8_resume_fingerprint(output / "run_fingerprint.json", fingerprint)
        if read_json(output / "campaign_status.json").get("status") == "completed":
            print(json.dumps({"status": "skipped_completed", "output": str(output)}))
            return

    ensure_dir(output)
    write_json(output / "heldout_lock_manifest.json", v8_heldout_lock_manifest())
    write_json(output / "source_tree_manifest.json", source_tree)
    write_json(
        output / "source_tree_summary.json",
        {"files": len(source_tree), "sha256": source_tree_digest(source_tree)},
    )
    write_json(output / "capacity_audit.json", capacity)
    write_json(
        output / "anchor_manifest.json",
        {
            "anchor_fold": str(anchor_fold),
            "anchor_source_sha256": ANCHOR_SOURCE_DIGEST,
            "cache_sha256": file_sha256(cache_path),
            "teacher_replay": anchor_replay,
        },
    )
    write_v8_fingerprint(output / "run_fingerprint.json", fingerprint)
    (output / "resolved_run.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
    )

    selection_standardizer = fit_feature_standardizer(
        arrays["atcnet_selection_train_sequence"],
        arrays["fbcnet_selection_train_sequence"],
    )
    outer_standardizer = fit_feature_standardizer(
        arrays["atcnet_outer_train_sequence"], arrays["fbcnet_outer_train_sequence"]
    )
    standardized = {
        "selection_train": selection_standardizer.transform(
            arrays["atcnet_selection_train_sequence"],
            arrays["fbcnet_selection_train_sequence"],
        ),
        "selection_validation": selection_standardizer.transform(
            arrays["atcnet_selection_validation_sequence"],
            arrays["fbcnet_selection_validation_sequence"],
        ),
        "outer_train": outer_standardizer.transform(
            arrays["atcnet_outer_train_sequence"], arrays["fbcnet_outer_train_sequence"]
        ),
        "outer_test": outer_standardizer.transform(
            arrays["atcnet_outer_test_sequence"], arrays["fbcnet_outer_test_sequence"]
        ),
    }

    hpo_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    started = time.time()
    fold_seed = int(args.seed) * 100_003 + int(args.fold) * 1_009 + 10_100_019
    for model_name in models:
        candidates: list[tuple[tuple[float, float, int, int], dict[str, Any], Any]] = []
        for grid_index, configuration in enumerate(grid):
            selection_fit = fit_v10_control(
                model_name,
                atc_train=standardized["selection_train"][0],
                fbc_train=standardized["selection_train"][1],
                y_train=labels[inner_train],
                teacher_train=arrays["teacher_selection_train"],
                atc_validation=standardized["selection_validation"][0],
                fbc_validation=standardized["selection_validation"][1],
                y_validation=labels[inner_validation],
                teacher_validation=arrays["teacher_selection_validation"],
                device=args.device,
                seed=fold_seed,
                epochs=int(args.epochs),
                patience=int(args.patience),
                batch_size=int(args.batch_size),
                learning_rate=float(configuration["learning_rate"]),
                weight_decay=float(configuration["weight_decay"]),
                run_label=(
                    f"V10-E1-select:{model_name}:grid{grid_index}:"
                    f"S{args.subject}:seed{args.seed}:fold{args.fold}"
                ),
            )
            row = {
                "model": model_name,
                "grid_index": grid_index,
                **configuration,
                "best_epoch": selection_fit.best_epoch,
                "best_kappa": selection_fit.best_metric,
                "best_accuracy": selection_fit.best_accuracy,
                "optimizer_steps": selection_fit.optimizer_steps,
                "train_seconds": selection_fit.elapsed_seconds,
            }
            hpo_rows.append(row)
            score = (
                float(selection_fit.best_metric),
                float(selection_fit.best_accuracy),
                -int(selection_fit.best_epoch),
                -grid_index,
            )
            candidates.append((score, configuration, selection_fit))
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, selected_configuration, selected_fit = candidates[0]
        selected_grid_index = grid.index(selected_configuration)
        selected_epoch = min(
            int(args.epochs),
            max(int(args.minimum_outer_epochs), int(selected_fit.best_epoch)),
        )
        for _, _, fit in candidates[1:]:
            del fit
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        outer_fit = fit_v10_control(
            model_name,
            atc_train=standardized["outer_train"][0],
            fbc_train=standardized["outer_train"][1],
            y_train=labels[outer_train],
            teacher_train=arrays["teacher_outer_train"],
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
            learning_rate=float(selected_configuration["learning_rate"]),
            weight_decay=float(selected_configuration["weight_decay"]),
            run_label=f"V10-E1-outer:{model_name}:S{args.subject}:seed{args.seed}:fold{args.fold}",
        )
        evaluation = predict_v10_control(
            outer_fit.model,
            standardized["outer_test"][0],
            standardized["outer_test"][1],
            labels[outer_test],
            arrays["teacher_outer_test"],
            device=args.device,
            batch_size=int(args.batch_size),
        )
        model_dir = ensure_dir(output / model_name)
        row = {
            "model": model_name,
            "subject": int(args.subject),
            "seed": int(args.seed),
            "fold": int(args.fold),
            "selected_grid_index": selected_grid_index,
            "selected_learning_rate": float(selected_configuration["learning_rate"]),
            "selected_weight_decay": float(selected_configuration["weight_decay"]),
            "selected_epoch": selected_epoch,
            "inner_best_epoch": selected_fit.best_epoch,
            "inner_best_kappa": selected_fit.best_metric,
            "inner_best_accuracy": selected_fit.best_accuracy,
            "accuracy": evaluation["accuracy"],
            "kappa": evaluation["kappa"],
            "mean_firing_rate": evaluation["mean_firing_rate"],
            "parameters": capacity["models"][model_name]["parameters"],
            "anchor_snn_accuracy": float(
                np.mean(anchor_snn_logits.argmax(axis=1) == labels[outer_test])
            ),
            "anchor_ann_accuracy": float(
                np.mean(anchor_ann_logits.argmax(axis=1) == labels[outer_test])
            ),
            "optimizer_steps": selected_fit.optimizer_steps + outer_fit.optimizer_steps,
            "train_seconds": selected_fit.elapsed_seconds + outer_fit.elapsed_seconds,
            "session_e_accessed": False,
        }
        summary_rows.append(row)
        torch.save(selected_fit.best_state, model_dir / "selection_best.pt")
        torch.save(outer_fit.last_state, model_dir / "outer_last.pt")
        write_csv(model_dir / "selection_history.csv", selected_fit.history)
        write_csv(model_dir / "outer_history.csv", outer_fit.history)
        write_json(model_dir / "metrics.json", row)
        np.savez_compressed(
            model_dir / "outer_predictions.npz",
            indices=outer_test,
            logits=np.asarray(evaluation["logits"], dtype=np.float32),
            labels=labels[outer_test],
        )
        del selected_fit, outer_fit, evaluation, candidates
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    write_csv(output / "hpo_results.csv", hpo_rows)
    write_csv(output / "summary.csv", summary_rows)
    status = {
        "status": "completed",
        "stage": "V10-E1-strong-control-development-fold",
        "subject": int(args.subject),
        "seed": int(args.seed),
        "fold": int(args.fold),
        "models": models,
        "hpo_configurations_per_model": len(grid),
        "capacity_audit": capacity,
        "anchor_source_sha256": ANCHOR_SOURCE_DIGEST,
        "elapsed_seconds": time.time() - started,
        "data_access": access,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
        "run_fingerprint": fingerprint["combined_sha256"],
    }
    write_json(output / "campaign_status.json", status)
    print(json.dumps({"status": "completed", "rows": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
