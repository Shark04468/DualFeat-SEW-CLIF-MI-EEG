#!/usr/bin/env python3
"""Run one bounded native-rate dual-SNN E13 fold."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any

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
from dpc_snn.experiments.v12_multirate_training import (  # noqa: E402
    V12_E12_OBJECTIVES,
    fit_v12,
    predict_v12,
)
from dpc_snn.models.v13_dual_rate_student import (  # noqa: E402
    V13_MODEL_VARIANTS,
    build_v13_student,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v10_e1_strong_control_fold import (  # noqa: E402
    ANCHOR_SOURCE_DIGEST,
    _array_sha256,
    _csv,
    _csv_float,
    _environment_manifest,
    _subject_file,
)


TRAINING_VARIANT = "interpolated_branch_kd"
REFERENCE_PARAMETERS = 74_124


def _prediction(path: Path, indices: np.ndarray, labels: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        saved_indices = np.asarray(archive["indices"], dtype=np.int64)
        saved_labels = np.asarray(archive["labels"], dtype=np.int64)
        logits = np.asarray(archive["logits"], dtype=np.float32)
    if not np.array_equal(saved_indices, indices) or not np.array_equal(
        saved_labels, labels
    ):
        raise RuntimeError(f"anchor prediction mismatch: {path}")
    return logits


def _softmax(logits: np.ndarray, temperature: float = 2.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    probability = np.exp(scaled)
    return probability / probability.sum(axis=1, keepdims=True)


def _diagnostics(
    labels: np.ndarray,
    fused: np.ndarray,
    atc: np.ndarray,
    fbc: np.ndarray,
    teacher: np.ndarray,
) -> dict[str, float]:
    fused_prediction = fused.argmax(axis=1)
    atc_prediction = atc.argmax(axis=1)
    fbc_prediction = fbc.argmax(axis=1)
    teacher_prediction = teacher.argmax(axis=1)
    teacher_probability = _softmax(teacher)
    fused_probability = _softmax(fused)
    kl = np.sum(
        teacher_probability
        * (
            np.log(np.clip(teacher_probability, 1e-12, 1.0))
            - np.log(np.clip(fused_probability, 1e-12, 1.0))
        ),
        axis=1,
    )
    atc_accuracy = float(np.mean(atc_prediction == labels))
    fbc_accuracy = float(np.mean(fbc_prediction == labels))
    fused_accuracy = float(np.mean(fused_prediction == labels))
    return {
        "atc_branch_accuracy": atc_accuracy,
        "fbc_branch_accuracy": fbc_accuracy,
        "best_branch_accuracy": max(atc_accuracy, fbc_accuracy),
        "fusion_gain_over_best_branch_pp": 100.0
        * (fused_accuracy - max(atc_accuracy, fbc_accuracy)),
        "teacher_accuracy": float(np.mean(teacher_prediction == labels)),
        "teacher_student_kl": float(np.mean(kl)),
        "teacher_student_disagreement": float(
            np.mean(teacher_prediction != fused_prediction)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--anchor-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--variants", default=",".join(V13_MODEL_VARIANTS))
    parser.add_argument("--learning-rates", default="0.0003,0.001")
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
    variants = _csv(args.variants)
    learning_rates = _csv_float(args.learning_rates)
    weight_decays = _csv_float(args.weight_decays)
    if set(variants) - set(V13_MODEL_VARIANTS) or len(variants) != len(set(variants)):
        raise ValueError("unknown or duplicate V13 variants")
    if not variants or args.subject not in range(1, 10) or args.fold not in range(6):
        raise ValueError("invalid V13 subject, fold, or variant list")
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
    if (
        anchor_status.get("status") != "completed"
        or anchor_status.get("session_e_accessed") is not False
        or anchor_source.get("sha256") != ANCHOR_SOURCE_DIGEST
        or anchor_replay.get("status") != "passed"
    ):
        raise RuntimeError("V9 anchor fold failed validation")

    cache_path = anchor_fold / "frozen_dual_feature_cache.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        arrays = {name: np.asarray(cache[name]) for name in cache.files}
    partitions = ("selection_train", "selection_validation", "outer_train", "outer_test")
    required = {
        "inner_train_indices",
        "inner_validation_indices",
        "outer_train_indices",
        "outer_test_indices",
        *{
            f"{branch}_{partition}_{kind}"
            for branch in ("atcnet", "fbcnet")
            for partition in partitions
            for kind in ("sequence", "logits")
        },
        *{f"teacher_{partition}" for partition in partitions},
    }
    if not required.issubset(arrays):
        raise RuntimeError("V9 cache lacks E13 branch features or targets")

    subject_path = _subject_file(data_root, int(args.subject))
    data = load_processed_npz(subject_path)
    _, labels, _, access = session_t_development_view(data)
    inner_train = arrays["inner_train_indices"].astype(np.int64)
    inner_validation = arrays["inner_validation_indices"].astype(np.int64)
    outer_train = arrays["outer_train_indices"].astype(np.int64)
    outer_test = arrays["outer_test_indices"].astype(np.int64)
    anchor_logits = _prediction(
        anchor_fold / "sew_clif_kd" / "outer_predictions.npz",
        outer_test,
        labels[outer_test],
    )

    capacities = {
        variant: int(
            sum(value.numel() for value in build_v13_student(variant).parameters())
        )
        for variant in variants
    }
    relative_capacity_difference = {
        variant: abs(parameters - REFERENCE_PARAMETERS) / REFERENCE_PARAMETERS
        for variant, parameters in capacities.items()
    }
    if any(value > 0.01 for value in relative_capacity_difference.values()):
        raise RuntimeError("V13 parameter matching exceeds one percent")
    grid = [
        {"learning_rate": learning_rate, "weight_decay": weight_decay}
        for learning_rate, weight_decay in itertools.product(learning_rates, weight_decays)
    ]
    if len(grid) * len(variants) > 12:
        raise RuntimeError("V13 bounded search exceeds 12 configurations")

    source_tree = collect_source_tree_manifest(ROOT)
    objective = V12_E12_OBJECTIVES["o0_current"]
    resolved_config = {
        **vars(args),
        "data": str(data_root),
        "anchor_root": str(anchor_root),
        "output": str(output),
        "variants": variants,
        "learning_rates": learning_rates,
        "weight_decays": weight_decays,
        "hpo_grid": grid,
        "objective": {
            **objective.__dict__,
            "hard_weight": objective.hard_weight,
        },
        "capacities": capacities,
        "reference_parameters": REFERENCE_PARAMETERS,
        "selection": "inner validation kappa, accuracy, earliest epoch, grid order",
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
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
    write_json(
        output / "capacity_audit.json",
        {
            "status": "passed",
            "reference_parameters": REFERENCE_PARAMETERS,
            "variant_parameters": capacities,
            "relative_difference": relative_capacity_difference,
            "maximum_allowed_relative_difference": 0.01,
            "raw_feature_bypass": False,
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

    def teachers(partition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            arrays[f"teacher_{partition}"],
            arrays[f"atcnet_{partition}_logits"],
            arrays[f"fbcnet_{partition}_logits"],
        )

    hpo_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    started = time.time()
    fold_seed = int(args.seed) * 100_003 + int(args.fold) * 1_009 + 13_100_019
    for variant in variants:
        def builder(kind: str, name: str = variant) -> torch.nn.Module:
            return build_v13_student(name, decoder_kind=kind)

        candidates: list[tuple[tuple[float, float, int, int], dict[str, float], Any]] = []
        for grid_index, configuration in enumerate(grid):
            fit = fit_v12(
                TRAINING_VARIANT,
                atc_train=standardized["selection_train"][0],
                fbc_train=standardized["selection_train"][1],
                y_train=labels[inner_train],
                equal_teacher_train=teachers("selection_train")[0],
                atc_teacher_train=teachers("selection_train")[1],
                fbc_teacher_train=teachers("selection_train")[2],
                atc_validation=standardized["selection_validation"][0],
                fbc_validation=standardized["selection_validation"][1],
                y_validation=labels[inner_validation],
                equal_teacher_validation=teachers("selection_validation")[0],
                atc_teacher_validation=teachers("selection_validation")[1],
                fbc_teacher_validation=teachers("selection_validation")[2],
                device=args.device,
                seed=fold_seed,
                epochs=int(args.epochs),
                patience=int(args.patience),
                batch_size=int(args.batch_size),
                learning_rate=float(configuration["learning_rate"]),
                weight_decay=float(configuration["weight_decay"]),
                decoder_kind="clif",
                objective_override=objective,
                model_builder=builder,
                run_label=(
                    f"V13-E1-select:{variant}:grid{grid_index}:"
                    f"S{args.subject}:seed{args.seed}:fold{args.fold}"
                ),
            )
            hpo_rows.append(
                {
                    "variant": variant,
                    "grid_index": grid_index,
                    **configuration,
                    "best_epoch": fit.best_epoch,
                    "best_kappa": fit.best_metric,
                    "best_accuracy": fit.best_accuracy,
                    "optimizer_steps": fit.optimizer_steps,
                    "train_seconds": fit.elapsed_seconds,
                }
            )
            candidates.append(
                (
                    (
                        float(fit.best_metric),
                        float(fit.best_accuracy),
                        -int(fit.best_epoch),
                        -grid_index,
                    ),
                    configuration,
                    fit,
                )
            )
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, selected_configuration, selected_fit = candidates[0]
        selected_epoch = min(
            int(args.epochs),
            max(int(args.minimum_outer_epochs), int(selected_fit.best_epoch)),
        )
        selected_grid_index = grid.index(selected_configuration)
        for _, _, candidate_fit in candidates[1:]:
            del candidate_fit
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        outer_fit = fit_v12(
            TRAINING_VARIANT,
            atc_train=standardized["outer_train"][0],
            fbc_train=standardized["outer_train"][1],
            y_train=labels[outer_train],
            equal_teacher_train=teachers("outer_train")[0],
            atc_teacher_train=teachers("outer_train")[1],
            fbc_teacher_train=teachers("outer_train")[2],
            atc_validation=None,
            fbc_validation=None,
            y_validation=None,
            equal_teacher_validation=None,
            atc_teacher_validation=None,
            fbc_teacher_validation=None,
            device=args.device,
            seed=fold_seed + 1_000_003,
            epochs=int(args.epochs),
            fixed_epoch=selected_epoch,
            scheduler_epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            learning_rate=float(selected_configuration["learning_rate"]),
            weight_decay=float(selected_configuration["weight_decay"]),
            decoder_kind="clif",
            objective_override=objective,
            model_builder=builder,
            run_label=(
                f"V13-E1-outer:{variant}:S{args.subject}:"
                f"seed{args.seed}:fold{args.fold}"
            ),
        )
        evaluation = predict_v12(
            outer_fit.model,
            standardized["outer_test"][0],
            standardized["outer_test"][1],
            labels[outer_test],
            teachers("outer_test")[0],
            teachers("outer_test")[1],
            teachers("outer_test")[2],
            device=args.device,
            batch_size=int(args.batch_size),
        )
        diagnostics = _diagnostics(
            labels[outer_test],
            np.asarray(evaluation["logits"]),
            np.asarray(evaluation["atc_logits"]),
            np.asarray(evaluation["fbc_logits"]),
            teachers("outer_test")[0],
        )
        variant_dir = ensure_dir(output / variant)
        row = {
            "variant": variant,
            "subject": int(args.subject),
            "seed": int(args.seed),
            "fold": int(args.fold),
            "selected_grid_index": selected_grid_index,
            "selected_learning_rate": float(selected_configuration["learning_rate"]),
            "selected_weight_decay": float(selected_configuration["weight_decay"]),
            "selected_epoch": selected_epoch,
            "inner_best_kappa": selected_fit.best_metric,
            "inner_best_accuracy": selected_fit.best_accuracy,
            "accuracy": float(evaluation["accuracy"]),
            "kappa": float(evaluation["kappa"]),
            "mean_firing_rate": float(evaluation["mean_firing_rate"]),
            "parameters": capacities[variant],
            "anchor_snn_accuracy": float(
                np.mean(anchor_logits.argmax(axis=1) == labels[outer_test])
            ),
            **diagnostics,
            "session_e_accessed": False,
        }
        summary_rows.append(row)
        endpoint_seconds = (1.0, 2.0, 3.0, 4.0)
        for endpoint_index, seconds in enumerate(endpoint_seconds):
            endpoint_rows.append(
                {
                    "variant": variant,
                    "subject": int(args.subject),
                    "seed": int(args.seed),
                    "fold": int(args.fold),
                    "endpoint_seconds": seconds,
                    "accuracy": float(
                        np.mean(
                            evaluation["endpoint_logits"][:, endpoint_index].argmax(
                                axis=1
                            )
                            == labels[outer_test]
                        )
                    ),
                }
            )
        torch.save(selected_fit.best_state, variant_dir / "selection_best.pt")
        torch.save(outer_fit.last_state, variant_dir / "outer_last.pt")
        write_csv(variant_dir / "selection_history.csv", selected_fit.history)
        write_csv(variant_dir / "outer_history.csv", outer_fit.history)
        write_json(variant_dir / "metrics.json", row)
        np.savez_compressed(
            variant_dir / "outer_predictions.npz",
            indices=outer_test,
            logits=np.asarray(evaluation["logits"], dtype=np.float32),
            endpoint_logits=np.asarray(evaluation["endpoint_logits"], dtype=np.float32),
            atc_logits=np.asarray(evaluation["atc_logits"], dtype=np.float32),
            fbc_logits=np.asarray(evaluation["fbc_logits"], dtype=np.float32),
            equal_teacher_logits=np.asarray(teachers("outer_test")[0], dtype=np.float32),
            labels=labels[outer_test],
        )
        del selected_fit, outer_fit, evaluation, candidates
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    write_csv(output / "hpo_results.csv", hpo_rows)
    write_csv(output / "summary.csv", summary_rows)
    write_csv(output / "endpoint_metrics.csv", endpoint_rows)
    status = {
        "status": "completed",
        "stage": "V13-E1-native-rate-dual-SNN-fold",
        "subject": int(args.subject),
        "seed": int(args.seed),
        "fold": int(args.fold),
        "variants": variants,
        "total_hpo_configurations": len(grid) * len(variants),
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
