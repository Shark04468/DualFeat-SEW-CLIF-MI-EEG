#!/usr/bin/env python3
"""Run one leakage-safe E12 distillation-objective fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
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
from dpc_snn.models.v12_multirate_student import build_v12_student  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v10_e1_strong_control_fold import (  # noqa: E402
    ANCHOR_SOURCE_DIGEST,
    _array_sha256,
    _csv,
    _environment_manifest,
    _subject_file,
)


REFERENCE_VARIANT = "interpolated_branch_kd"
REFERENCE_SOURCE_DIGEST = (
    "c26a6f7e8f3bf4714e70613cd443989ea92938292fe1be6efe9336312ba05296"
)


def _prediction(
    path: Path, indices: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        saved_indices = np.asarray(archive["indices"], dtype=np.int64)
        saved_labels = np.asarray(archive["labels"], dtype=np.int64)
        logits = np.asarray(archive["logits"], dtype=np.float32)
    if not np.array_equal(saved_indices, indices) or not np.array_equal(
        saved_labels, labels
    ):
        raise RuntimeError(f"prediction alignment mismatch: {path}")
    return logits, saved_labels


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    probability = np.exp(scaled)
    return probability / probability.sum(axis=1, keepdims=True)


def _teacher_diagnostics(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    labels: np.ndarray,
    *,
    temperature: float,
) -> dict[str, float]:
    student_prediction = student_logits.argmax(axis=1)
    teacher_prediction = teacher_logits.argmax(axis=1)
    student_correct = student_prediction == labels
    teacher_correct = teacher_prediction == labels
    teacher_probability = _softmax(teacher_logits, temperature)
    student_probability = _softmax(student_logits, temperature)
    kl = np.sum(
        teacher_probability
        * (
            np.log(np.clip(teacher_probability, 1e-12, 1.0))
            - np.log(np.clip(student_probability, 1e-12, 1.0))
        ),
        axis=1,
    )
    return {
        "teacher_accuracy": float(np.mean(teacher_correct)),
        "teacher_student_kl": float(np.mean(kl)),
        "teacher_student_disagreement": float(
            np.mean(teacher_prediction != student_prediction)
        ),
        "teacher_rescue_rate": float(np.mean(teacher_correct & ~student_correct)),
        "student_rescue_rate": float(np.mean(student_correct & ~teacher_correct)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--anchor-root", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--objectives", default=",".join(V12_E12_OBJECTIVES))
    parser.add_argument("--epoch-cap", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--firing-rate-weight", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    configure_cache_env()
    data_root = Path(args.data).resolve()
    anchor_root = Path(args.anchor_root).resolve()
    reference_root = Path(args.reference_root).resolve()
    output = Path(args.output).resolve()
    objectives = _csv(args.objectives)
    if (
        set(objectives) - set(V12_E12_OBJECTIVES)
        or len(objectives) != len(set(objectives))
        or "o0_current" not in objectives
    ):
        raise ValueError("E12 objectives must be unique, known, and include o0_current")
    if args.subject not in range(1, 10) or args.fold not in range(6):
        raise ValueError("invalid E12 subject or fold")
    if args.epoch_cap < 0:
        raise ValueError("epoch cap cannot be negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    anchor_fold = (
        anchor_root
        / f"subject_{args.subject:02d}"
        / f"seed_{args.seed}"
        / f"fold_{args.fold}"
    )
    reference_fold = (
        reference_root
        / f"subject_{args.subject:02d}"
        / f"seed_{args.seed}"
        / f"fold_{args.fold}"
    )
    anchor_status = read_json(anchor_fold / "campaign_status.json")
    anchor_source = read_json(anchor_fold / "source_tree_summary.json")
    anchor_replay = read_json(anchor_fold / "teacher_replay.json")
    reference_status = read_json(reference_fold / "campaign_status.json")
    reference_source = read_json(reference_fold / "source_tree_summary.json")
    if (
        anchor_status.get("status") != "completed"
        or anchor_status.get("session_e_accessed") is not False
        or anchor_source.get("sha256") != ANCHOR_SOURCE_DIGEST
        or anchor_replay.get("status") != "passed"
    ):
        raise RuntimeError("V9 anchor fold failed validation")
    if (
        reference_status.get("status") != "completed"
        or reference_status.get("session_e_accessed") is not False
        or reference_source.get("sha256") != REFERENCE_SOURCE_DIGEST
    ):
        raise RuntimeError("V12.1 reference fold failed validation")

    reference_summary = pd.read_csv(reference_fold / "summary.csv")
    reference_row = reference_summary.loc[
        reference_summary["variant"] == REFERENCE_VARIANT
    ]
    if len(reference_row) != 1:
        raise RuntimeError("V12.1 reference configuration is missing or duplicated")
    selected = reference_row.iloc[0]
    learning_rate = float(selected["selected_learning_rate"])
    weight_decay = float(selected["selected_weight_decay"])
    reference_epoch = int(selected["selected_epoch"])
    fixed_epoch = (
        min(reference_epoch, int(args.epoch_cap)) if args.epoch_cap else reference_epoch
    )

    cache_path = anchor_fold / "frozen_dual_feature_cache.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        arrays = {name: np.asarray(cache[name]) for name in cache.files}
    required = {
        "outer_train_indices",
        "outer_test_indices",
        *{
            f"{branch}_{partition}_{kind}"
            for branch in ("atcnet", "fbcnet")
            for partition in ("outer_train", "outer_test")
            for kind in ("sequence", "logits")
        },
        "teacher_outer_train",
        "teacher_outer_test",
    }
    if not required.issubset(arrays):
        raise RuntimeError("V9 cache lacks E12 outer-fold arrays")

    subject_path = _subject_file(data_root, int(args.subject))
    data = load_processed_npz(subject_path)
    _, labels, _, access = session_t_development_view(data)
    outer_train = arrays["outer_train_indices"].astype(np.int64)
    outer_test = arrays["outer_test_indices"].astype(np.int64)
    reference_logits, reference_labels = _prediction(
        reference_fold / REFERENCE_VARIANT / "outer_predictions.npz",
        outer_test,
        labels[outer_test],
    )

    standardizer = fit_feature_standardizer(
        arrays["atcnet_outer_train_sequence"], arrays["fbcnet_outer_train_sequence"]
    )
    train_features = standardizer.transform(
        arrays["atcnet_outer_train_sequence"], arrays["fbcnet_outer_train_sequence"]
    )
    test_features = standardizer.transform(
        arrays["atcnet_outer_test_sequence"], arrays["fbcnet_outer_test_sequence"]
    )

    def teachers(partition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            arrays[f"teacher_{partition}"],
            arrays[f"atcnet_{partition}_logits"],
            arrays[f"fbcnet_{partition}_logits"],
        )

    source_tree = collect_source_tree_manifest(ROOT)
    parameter_count = sum(
        value.numel()
        for value in build_v12_student(REFERENCE_VARIANT, decoder_kind="clif").parameters()
    )
    resolved_config = {
        **vars(args),
        "data": str(data_root),
        "anchor_root": str(anchor_root),
        "reference_root": str(reference_root),
        "output": str(output),
        "objectives": objectives,
        "objective_weights": {
            name: {
                **V12_E12_OBJECTIVES[name].__dict__,
                "hard_weight": V12_E12_OBJECTIVES[name].hard_weight,
            }
            for name in objectives
        },
        "model_variant": REFERENCE_VARIANT,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "reference_epoch": reference_epoch,
        "fixed_epoch": fixed_epoch,
        "selection": "none; exact V12.1 fold-local optimizer and epoch replay",
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    fingerprint = build_v8_run_fingerprint(
        resolved_run_config=resolved_config,
        source_tree=source_tree,
        data={"path": str(subject_path), "sha256": file_sha256(subject_path)},
        split={
            "outer_train": _array_sha256(outer_train),
            "outer_test": _array_sha256(outer_test),
        },
        augmentation={"enabled": False},
        prior={"enabled": False, "feature_standardization": "outer-train-only"},
        checkpoint={
            "anchor_cache_sha256": file_sha256(cache_path),
            "anchor_run_fingerprint_sha256": file_sha256(
                anchor_fold / "run_fingerprint.json"
            ),
            "reference_summary_sha256": file_sha256(reference_fold / "summary.csv"),
            "reference_run_fingerprint_sha256": file_sha256(
                reference_fold / "run_fingerprint.json"
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
            "status": "matched",
            "model_variant": REFERENCE_VARIANT,
            "parameters_per_objective": parameter_count,
            "objective_is_only_changed_factor": True,
        },
    )
    write_v8_fingerprint(output / "run_fingerprint.json", fingerprint)
    (output / "resolved_run.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    started = time.time()
    fold_seed = int(args.seed) * 100_003 + int(args.fold) * 1_009 + 12_100_019
    o0_logits: np.ndarray | None = None
    for objective_name in objectives:
        objective = V12_E12_OBJECTIVES[objective_name]
        fit = fit_v12(
            REFERENCE_VARIANT,
            atc_train=train_features[0],
            fbc_train=train_features[1],
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
            epochs=160,
            fixed_epoch=fixed_epoch,
            scheduler_epochs=160,
            batch_size=int(args.batch_size),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            distillation_temperature=float(args.distillation_temperature),
            firing_rate_weight=float(args.firing_rate_weight),
            decoder_kind="clif",
            objective_override=objective,
            run_label=(
                f"V12-E12:{objective_name}:S{args.subject}:"
                f"seed{args.seed}:fold{args.fold}"
            ),
        )
        evaluation = predict_v12(
            fit.model,
            test_features[0],
            test_features[1],
            labels[outer_test],
            teachers("outer_test")[0],
            teachers("outer_test")[1],
            teachers("outer_test")[2],
            device=args.device,
            batch_size=int(args.batch_size),
        )
        diagnostics = _teacher_diagnostics(
            np.asarray(evaluation["logits"]),
            teachers("outer_test")[0],
            labels[outer_test],
            temperature=float(args.distillation_temperature),
        )
        row = {
            "objective": objective_name,
            "subject": int(args.subject),
            "seed": int(args.seed),
            "fold": int(args.fold),
            "hard_weight": objective.hard_weight,
            **objective.__dict__,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "fixed_epoch": fixed_epoch,
            "accuracy": float(evaluation["accuracy"]),
            "kappa": float(evaluation["kappa"]),
            "mean_firing_rate": float(evaluation["mean_firing_rate"]),
            **diagnostics,
            "session_e_accessed": False,
        }
        rows.append(row)
        objective_dir = ensure_dir(output / objective_name)
        torch.save(fit.last_state, objective_dir / "outer_last.pt")
        write_csv(objective_dir / "outer_history.csv", fit.history)
        write_json(objective_dir / "metrics.json", row)
        np.savez_compressed(
            objective_dir / "outer_predictions.npz",
            indices=outer_test,
            logits=np.asarray(evaluation["logits"], dtype=np.float32),
            labels=labels[outer_test],
            equal_teacher_logits=np.asarray(teachers("outer_test")[0], dtype=np.float32),
            atc_teacher_logits=np.asarray(teachers("outer_test")[1], dtype=np.float32),
            fbc_teacher_logits=np.asarray(teachers("outer_test")[2], dtype=np.float32),
        )
        if objective_name == "o0_current":
            o0_logits = np.asarray(evaluation["logits"], dtype=np.float32)
        del fit, evaluation
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    assert o0_logits is not None
    max_logit_difference = float(np.max(np.abs(o0_logits - reference_logits)))
    prediction_mismatches = int(
        np.sum(o0_logits.argmax(axis=1) != reference_logits.argmax(axis=1))
    )
    replay_status = (
        "passed"
        if max_logit_difference <= 1e-6 and prediction_mismatches == 0
        else "failed"
    )
    replay = {
        "status": replay_status,
        "reference_variant": REFERENCE_VARIANT,
        "max_absolute_logit_difference": max_logit_difference,
        "prediction_mismatches": prediction_mismatches,
        "labels_aligned": bool(np.array_equal(reference_labels, labels[outer_test])),
        "epoch_cap_applied": bool(args.epoch_cap),
    }
    write_json(output / "reference_replay.json", replay)
    if replay_status != "passed":
        raise RuntimeError("E12 O0 failed to replay the V12.1 reference")

    write_csv(output / "summary.csv", rows)
    status = {
        "status": "completed",
        "stage": "V12-E12-distillation-objective-fold",
        "subject": int(args.subject),
        "seed": int(args.seed),
        "fold": int(args.fold),
        "objectives": objectives,
        "total_hpo_configurations": 0,
        "matched_objective_runs": len(objectives),
        "elapsed_seconds": time.time() - started,
        "data_access": access,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
        "reference_replay": replay_status,
        "run_fingerprint": fingerprint["combined_sha256"],
    }
    write_json(output / "campaign_status.json", status)
    print(json.dumps({"status": "completed", "replay": replay, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
