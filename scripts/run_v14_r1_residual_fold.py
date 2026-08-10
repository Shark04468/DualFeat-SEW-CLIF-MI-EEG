#!/usr/bin/env python3
"""Run one V14 frozen-shared residual-expert development fold."""

from __future__ import annotations

import argparse
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
from dpc_snn.experiments.v14_residual_training import (  # noqa: E402
    fit_v14,
    predict_v14,
)
from dpc_snn.models.v14_shared_residual_student import (  # noqa: E402
    V14_VARIANTS,
    build_v14_student,
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


ANCHOR_VARIANT = "sew_clif_kd"


def _anchor_prediction(
    path: Path, indices: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        saved_indices = np.asarray(archive["indices"], dtype=np.int64)
        saved_labels = np.asarray(archive["labels"], dtype=np.int64)
        logits = np.asarray(archive["logits"], dtype=np.float32)
    if not np.array_equal(saved_indices, indices) or not np.array_equal(
        saved_labels, labels
    ):
        raise RuntimeError(f"anchor prediction mismatch: {path}")
    return logits


def _center(array: np.ndarray) -> np.ndarray:
    return array - array.mean(axis=1, keepdims=True)


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    numerator = np.sum(first * second, axis=1)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    valid = denominator > 1e-8
    if not np.any(valid):
        return 0.0
    return float(np.mean(numerator[valid] / denominator[valid]))


def _diagnostics(
    variant: str,
    evaluation: dict[str, Any],
    labels: np.ndarray,
    equal_teacher: np.ndarray,
    atc_teacher: np.ndarray,
    fbc_teacher: np.ndarray,
) -> dict[str, float]:
    final_prediction = evaluation["logits"].argmax(axis=1)
    shared_prediction = evaluation["shared_logits"].argmax(axis=1)
    final_correct = final_prediction == labels
    shared_correct = shared_prediction == labels
    residual_cosines: list[float] = []
    if variant in ("r1_atc_residual", "r3_dual_residual"):
        residual_cosines.append(
            _cosine(
                evaluation["atc_residual_logits"],
                _center(atc_teacher - evaluation["shared_logits"]),
            )
        )
    if variant in ("r2_fbc_residual", "r3_dual_residual"):
        residual_cosines.append(
            _cosine(
                evaluation["fbc_residual_logits"],
                _center(fbc_teacher - evaluation["shared_logits"]),
            )
        )
    if variant == "r4_generic_residual":
        residual_cosines.append(
            _cosine(
                evaluation["generic_residual_logits"],
                _center(equal_teacher - evaluation["shared_logits"]),
            )
        )
    active_gates = [
        value
        for name, value in evaluation["gates"].items()
        if name in variant
        or (variant == "r3_dual_residual" and name in ("atc", "fbc"))
    ]
    gate_values = np.concatenate(active_gates) if active_gates else np.zeros(1)
    return {
        "shared_accuracy": float(np.mean(shared_correct)),
        "teacher_accuracy": float(np.mean(equal_teacher.argmax(axis=1) == labels)),
        "rescue_rate": float(np.mean(final_correct & ~shared_correct)),
        "damage_rate": float(np.mean(~final_correct & shared_correct)),
        "net_rescue_pp": 100.0
        * float(np.mean(final_correct & ~shared_correct) - np.mean(~final_correct & shared_correct)),
        "residual_cosine": float(np.mean(residual_cosines)) if residual_cosines else 0.0,
        "active_gate_mean": float(np.mean(gate_values)),
        "active_gate_max": float(np.max(gate_values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--anchor-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--variants", default=",".join(V14_VARIANTS))
    parser.add_argument("--learning-rates", default="0.0003,0.001")
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--minimum-outer-epochs", type=int, default=20)
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    configure_cache_env()
    data_root = Path(args.data).resolve()
    anchor_root = Path(args.anchor_root).resolve()
    output = Path(args.output).resolve()
    variants = _csv(args.variants)
    learning_rates = _csv_float(args.learning_rates)
    if (
        set(variants) - set(V14_VARIANTS)
        or len(variants) != len(set(variants))
        or "r0_shared_replay" not in variants
    ):
        raise ValueError("V14 variants must be unique, known, and include R0")
    trainable_variants = [value for value in variants if value != "r0_shared_replay"]
    if len(trainable_variants) * len(learning_rates) > 10:
        raise RuntimeError("V14 bounded search exceeds ten expert configurations")
    if args.subject not in range(1, 10) or args.fold not in range(6):
        raise ValueError("invalid V14 subject or fold")
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
        raise RuntimeError("V9 cache lacks V14 features or teacher targets")

    subject_path = _subject_file(data_root, int(args.subject))
    data = load_processed_npz(subject_path)
    _, labels, _, access = session_t_development_view(data)
    inner_train = arrays["inner_train_indices"].astype(np.int64)
    inner_validation = arrays["inner_validation_indices"].astype(np.int64)
    outer_train = arrays["outer_train_indices"].astype(np.int64)
    outer_test = arrays["outer_test_indices"].astype(np.int64)
    anchor_logits = _anchor_prediction(
        anchor_fold / ANCHOR_VARIANT / "outer_predictions.npz",
        outer_test,
        labels[outer_test],
    )
    selection_shared_state = torch.load(
        anchor_fold / ANCHOR_VARIANT / "selection_best.pt",
        map_location="cpu",
        weights_only=True,
    )
    outer_shared_state = torch.load(
        anchor_fold / ANCHOR_VARIANT / "outer_last.pt",
        map_location="cpu",
        weights_only=True,
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

    capacities = {
        variant: {
            "total": build_v14_student(variant).parameter_count,
            "trainable": build_v14_student(variant).trainable_parameter_count,
        }
        for variant in variants
    }
    if abs(capacities["r3_dual_residual"]["total"] - capacities["r4_generic_residual"]["total"]) > 4:
        raise RuntimeError("V14 R3/R4 parameter matching failed")
    source_tree = collect_source_tree_manifest(ROOT)
    resolved_config = {
        **vars(args),
        "data": str(data_root),
        "anchor_root": str(anchor_root),
        "output": str(output),
        "variants": variants,
        "learning_rates": learning_rates,
        "capacities": capacities,
        "shared_backbone": "frozen exact V9 sew_clif_kd replay",
        "objective": {
            "hard": 0.50,
            "equal_teacher_kd": 0.35,
            "centered_residual": 0.15,
            "temporal_kd": 0.0,
        },
        "selection": "inner validation kappa, accuracy, earliest epoch, LR order",
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
            "selection_shared_sha256": file_sha256(
                anchor_fold / ANCHOR_VARIANT / "selection_best.pt"
            ),
            "outer_shared_sha256": file_sha256(
                anchor_fold / ANCHOR_VARIANT / "outer_last.pt"
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
            "shared_parameters": 73_348,
            "variants": capacities,
            "r3_r4_total_parameter_difference": abs(
                capacities["r3_dual_residual"]["total"]
                - capacities["r4_generic_residual"]["total"]
            ),
            "raw_feature_classifier_bypass": False,
        },
    )
    write_v8_fingerprint(output / "run_fingerprint.json", fingerprint)
    (output / "resolved_run.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
    )

    hpo_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    started = time.time()
    fold_seed = int(args.seed) * 100_003 + int(args.fold) * 1_009 + 14_100_019

    r0 = build_v14_student("r0_shared_replay")
    r0.load_shared_state(outer_shared_state)
    r0_evaluation = predict_v14(
        r0,
        standardized["outer_test"][0],
        standardized["outer_test"][1],
        labels[outer_test],
        teachers("outer_test")[0],
        teachers("outer_test")[1],
        teachers("outer_test")[2],
        device=args.device,
        batch_size=int(args.batch_size),
    )
    replay_difference = float(
        np.max(np.abs(r0_evaluation["logits"] - anchor_logits))
    )
    replay_mismatches = int(
        np.sum(r0_evaluation["logits"].argmax(axis=1) != anchor_logits.argmax(axis=1))
    )
    replay = {
        "status": (
            "passed" if replay_difference <= 1e-6 and replay_mismatches == 0 else "failed"
        ),
        "max_absolute_logit_difference": replay_difference,
        "prediction_mismatches": replay_mismatches,
    }
    write_json(output / "shared_replay.json", replay)
    if replay["status"] != "passed":
        raise RuntimeError("V14 R0 failed exact V9 replay")

    evaluations: dict[str, dict[str, Any]] = {"r0_shared_replay": r0_evaluation}
    selected_rows: dict[str, dict[str, Any]] = {}
    for variant in trainable_variants:
        candidates: list[tuple[tuple[float, float, int, int], float, Any]] = []
        for lr_index, learning_rate in enumerate(learning_rates):
            fit = fit_v14(
                variant,
                selection_shared_state,
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
                pretrain_epochs=int(args.pretrain_epochs),
                batch_size=int(args.batch_size),
                learning_rate=float(learning_rate),
                weight_decay=float(args.weight_decay),
                run_label=(
                    f"V14-R1-select:{variant}:lr{lr_index}:"
                    f"S{args.subject}:seed{args.seed}:fold{args.fold}"
                ),
            )
            hpo_rows.append(
                {
                    "variant": variant,
                    "lr_index": lr_index,
                    "learning_rate": learning_rate,
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
                        -lr_index,
                    ),
                    learning_rate,
                    fit,
                )
            )
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, selected_lr, selected_fit = candidates[0]
        selected_epoch = min(
            int(args.epochs),
            max(int(args.minimum_outer_epochs), int(selected_fit.best_epoch)),
        )
        for _, _, candidate in candidates[1:]:
            del candidate
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        outer_fit = fit_v14(
            variant,
            outer_shared_state,
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
            pretrain_epochs=int(args.pretrain_epochs),
            batch_size=int(args.batch_size),
            learning_rate=float(selected_lr),
            weight_decay=float(args.weight_decay),
            run_label=(
                f"V14-R1-outer:{variant}:S{args.subject}:"
                f"seed{args.seed}:fold{args.fold}"
            ),
        )
        evaluation = predict_v14(
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
        evaluations[variant] = evaluation
        selected_rows[variant] = {
            "selected_learning_rate": float(selected_lr),
            "selected_epoch": selected_epoch,
            "inner_best_kappa": float(selected_fit.best_metric),
            "inner_best_accuracy": float(selected_fit.best_accuracy),
        }
        variant_dir = ensure_dir(output / variant)
        torch.save(selected_fit.best_state, variant_dir / "selection_best.pt")
        torch.save(outer_fit.last_state, variant_dir / "outer_last.pt")
        write_csv(variant_dir / "selection_history.csv", selected_fit.history)
        write_csv(variant_dir / "outer_history.csv", outer_fit.history)
        del selected_fit, outer_fit, candidates
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    for variant in variants:
        evaluation = evaluations[variant]
        diagnostics = _diagnostics(
            variant,
            evaluation,
            labels[outer_test],
            teachers("outer_test")[0],
            teachers("outer_test")[1],
            teachers("outer_test")[2],
        )
        row = {
            "variant": variant,
            "subject": int(args.subject),
            "seed": int(args.seed),
            "fold": int(args.fold),
            **selected_rows.get(
                variant,
                {
                    "selected_learning_rate": float("nan"),
                    "selected_epoch": 0,
                    "inner_best_kappa": float("nan"),
                    "inner_best_accuracy": float("nan"),
                },
            ),
            "accuracy": float(evaluation["accuracy"]),
            "kappa": float(evaluation["kappa"]),
            "mean_firing_rate": float(evaluation["mean_firing_rate"]),
            "parameters": capacities[variant]["total"],
            "trainable_parameters": capacities[variant]["trainable"],
            **diagnostics,
            "session_e_accessed": False,
        }
        summary_rows.append(row)
        variant_dir = ensure_dir(output / variant)
        write_json(variant_dir / "metrics.json", row)
        np.savez_compressed(
            variant_dir / "outer_predictions.npz",
            indices=outer_test,
            labels=labels[outer_test],
            logits=np.asarray(evaluation["logits"], dtype=np.float32),
            shared_logits=np.asarray(evaluation["shared_logits"], dtype=np.float32),
            atc_residual_logits=np.asarray(
                evaluation["atc_residual_logits"], dtype=np.float32
            ),
            fbc_residual_logits=np.asarray(
                evaluation["fbc_residual_logits"], dtype=np.float32
            ),
            generic_residual_logits=np.asarray(
                evaluation["generic_residual_logits"], dtype=np.float32
            ),
            equal_teacher_logits=np.asarray(teachers("outer_test")[0], dtype=np.float32),
            atc_teacher_logits=np.asarray(teachers("outer_test")[1], dtype=np.float32),
            fbc_teacher_logits=np.asarray(teachers("outer_test")[2], dtype=np.float32),
            atc_gate=np.asarray(evaluation["gates"]["atc"], dtype=np.float32),
            fbc_gate=np.asarray(evaluation["gates"]["fbc"], dtype=np.float32),
            generic_gate=np.asarray(evaluation["gates"]["generic"], dtype=np.float32),
        )

    write_csv(output / "hpo_results.csv", hpo_rows)
    write_csv(output / "summary.csv", summary_rows)
    status = {
        "status": "completed",
        "stage": "V14-R1-frozen-shared-residual-fold",
        "subject": int(args.subject),
        "seed": int(args.seed),
        "fold": int(args.fold),
        "variants": variants,
        "total_hpo_configurations": len(trainable_variants) * len(learning_rates),
        "shared_replay": replay["status"],
        "elapsed_seconds": time.time() - started,
        "data_access": access,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
        "run_fingerprint": fingerprint["combined_sha256"],
    }
    write_json(output / "campaign_status.json", status)
    print(json.dumps({"status": "completed", "replay": replay, "rows": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
