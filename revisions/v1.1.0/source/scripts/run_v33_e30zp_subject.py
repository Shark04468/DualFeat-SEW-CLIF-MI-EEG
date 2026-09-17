#!/usr/bin/env python3
"""Run one BNCI2014-004 subject for the E30 objective-pure replication."""

from __future__ import annotations

import argparse
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

from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    sha256_fingerprint,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    source_tree_digest,
)
from dpc_snn.experiments.v9_dual_feature_training import (  # noqa: E402
    equal_probability_teacher,
    fit_v9_dual_feature,
    predict_v9_dual_feature,
)
from dpc_snn.models.v9_dual_feature_student import (  # noqa: E402
    build_v9_dual_feature_student,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v30_bnci2014_004_subject import (  # noqa: E402
    _build_teacher,
    _data_view,
    _extract_features,
    _load_gain,
    _load_standardizer,
    _prepared_inputs,
    _teacher_path,
)


DEFAULT_CONFIG = "configs/experiments/v33_e30_zero_penalty.yaml"
CHILD_VARIANT = "sew_clif_ce_fr0"


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


def _config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _parent_seed(parent: Path, subject: int, seed: int) -> Path:
    return parent / f"subject_{subject:02d}" / f"seed_{seed}"


def _child_variant(output: Path, subject: int, seed: int) -> Path:
    return (
        output
        / f"subject_{subject:02d}"
        / f"seed_{seed}"
        / "students"
        / CHILD_VARIANT
    )


def _validate_parent(parent: Path) -> None:
    summary = read_json(parent / "aggregate" / "aggregate_summary.json")
    statuses = sorted(parent.glob("subject_*/evaluation_status.json"))
    complete = len(statuses) == 9 and all(
        read_json(path).get("status") == "completed" for path in statuses
    )
    if not complete or summary.get("status") != "completed":
        raise RuntimeError("E30-ZP requires the completed E30 parent")


def _train(
    args: argparse.Namespace,
    config: dict[str, Any],
    config_sha: str,
    source_sha: str,
) -> None:
    parent = Path(args.parent).resolve()
    output = Path(args.output).resolve()
    sessions = [str(value) for value in config["dataset"]["training_sessions"]]
    data, labels, _, data_identity = _data_view(config, args.subject, sessions)
    minimum = int(config["dataset"]["training_trials_minimum"])
    maximum = int(config["dataset"]["training_trials_maximum"])
    if not minimum <= int(data_identity["trials"]) <= maximum:
        raise RuntimeError("E30-ZP training trial count is outside the parent contract")
    gain_path = parent / f"subject_{args.subject:02d}" / "preprocessing" / "gain.npz"
    gain = _load_gain(gain_path)
    prepared = _prepared_inputs(
        data["carrier"], gain, sfreq=float(config["dataset"]["source_sfreq"])
    )
    checkpoints = []
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        parent_seed = _parent_seed(parent, args.subject, seed)
        teachers: dict[str, torch.nn.Module] = {}
        teacher_hashes: dict[str, str] = {}
        for model_name in config["teacher_models"]:
            checkpoint_path = _teacher_path(
                parent, args.subject, seed, str(model_name)
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            teachers[str(model_name)] = _build_teacher(
                str(model_name),
                checkpoint,
                source_root=Path(args.source_root).resolve(),
                device=args.device,
            )
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
        standardizer_path = parent_seed / "feature_standardizer.npz"
        standardizer = _load_standardizer(standardizer_path)
        train_atc, train_fbc = standardizer.transform(atc, fbc)
        teacher = equal_probability_teacher(atc_logits, fbc_logits)
        run_seed = 3_100_000 + args.subject * 10_000 + seed * 101
        fingerprint = {
            "schema": "dpc-snn-e30zp-training-run/v1",
            "config_sha256": config_sha,
            "source_tree_sha256": source_sha,
            "parent_run": parent.name,
            "parent_gain_sha256": file_sha256(gain_path),
            "parent_standardizer_sha256": file_sha256(standardizer_path),
            "parent_teacher_checkpoint_sha256": teacher_hashes,
            "training_data_identity": data_identity,
            "subject": args.subject,
            "seed": seed,
            "run_seed": run_seed,
            "firing_rate_weight": 0.0,
        }
        fingerprint["combined_sha256"] = sha256_fingerprint(fingerprint)
        variant_dir = _child_variant(output, args.subject, seed)
        checkpoint_path = variant_dir / "checkpoint.pt"
        metrics_path = variant_dir / "training_metrics.json"
        fingerprint_path = variant_dir / "training_fingerprint.json"
        if checkpoint_path.is_file() and metrics_path.is_file():
            if read_json(fingerprint_path) != fingerprint:
                raise RuntimeError(f"stale E30-ZP resume rejected: {variant_dir}")
            checkpoints.append(str(checkpoint_path.resolve()))
            continue
        ensure_dir(variant_dir)
        write_json(fingerprint_path, fingerprint)
        training = config["training"]
        fit = fit_v9_dual_feature(
            "sew_clif_ce",
            atc_train=train_atc,
            fbc_train=train_fbc,
            y_train=labels,
            teacher_train=teacher,
            atc_validation=None,
            fbc_validation=None,
            y_validation=None,
            teacher_validation=None,
            device=args.device,
            seed=run_seed,
            fixed_epoch=int(training["student_fixed_epochs"]),
            scheduler_epochs=int(training["student_fixed_epochs"]),
            batch_size=int(training["student_batch_size"]),
            learning_rate=float(training["student_learning_rate"]),
            weight_decay=float(training["student_weight_decay"]),
            firing_rate_weight=0.0,
            model_kwargs=dict(config["model"]),
            run_label=f"E30ZP:S{args.subject}:seed{seed}",
        )
        _atomic_torch_save(
            {
                "state_dict": fit.last_state,
                "subject": args.subject,
                "seed": seed,
                "variant": CHILD_VARIANT,
                "n_classes": 2,
                "training_fingerprint_sha256": fingerprint["combined_sha256"],
            },
            checkpoint_path,
        )
        write_csv(variant_dir / "history.csv", fit.history)
        write_json(
            metrics_path,
            {
                "status": "training_completed_checkpoint_sealed",
                "subject": args.subject,
                "seed": seed,
                "variant": CHILD_VARIANT,
                "fixed_epochs": int(training["student_fixed_epochs"]),
                "optimizer_steps": fit.optimizer_steps,
                "parameters": fit.model.parameter_count,
                "elapsed_seconds": fit.elapsed_seconds,
                "firing_rate_weight": 0.0,
                "evaluation_sessions_accessed": False,
            },
        )
        checkpoints.append(str(checkpoint_path.resolve()))
        del fit, teachers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(
        ensure_dir(output / f"subject_{args.subject:02d}") / "training_status.json",
        {
            "status": "completed",
            "subject": args.subject,
            "checkpoints": sorted(checkpoints),
            "evaluation_sessions_accessed": False,
        },
    )


def _evaluate(
    args: argparse.Namespace,
    config: dict[str, Any],
    config_sha: str,
    source_sha: str,
) -> None:
    barrier = read_json(Path(args.checkpoint_barrier).resolve())
    if (
        barrier.get("schema") != "dpc-snn-e30zp-checkpoint-barrier/v1"
        or barrier.get("status") != "sealed"
        or barrier.get("config_sha256") != config_sha
        or barrier.get("source_tree_sha256") != source_sha
    ):
        raise RuntimeError("E30-ZP evaluation requires its checkpoint barrier")
    parent = Path(args.parent).resolve()
    output = Path(args.output).resolve()
    sessions = [str(value) for value in config["dataset"]["evaluation_sessions"]]
    data, labels, trial_ids, data_identity = _data_view(config, args.subject, sessions)
    minimum = int(config["dataset"]["evaluation_trials_minimum"])
    maximum = int(config["dataset"]["evaluation_trials_maximum"])
    if not minimum <= int(data_identity["trials"]) <= maximum:
        raise RuntimeError("E30-ZP evaluation trial count is outside the contract")
    gain = _load_gain(
        parent / f"subject_{args.subject:02d}" / "preprocessing" / "gain.npz"
    )
    prepared = _prepared_inputs(
        data["carrier"], gain, sfreq=float(config["dataset"]["source_sfreq"])
    )
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        parent_seed = _parent_seed(parent, args.subject, seed)
        features: dict[str, np.ndarray] = {}
        teacher_logits: dict[str, np.ndarray] = {}
        for model_name in config["teacher_models"]:
            path = _teacher_path(parent, args.subject, seed, str(model_name))
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = _build_teacher(
                str(model_name),
                checkpoint,
                source_root=Path(args.source_root).resolve(),
                device=args.device,
            )
            sequence, logits = _extract_features(
                str(model_name),
                model,
                prepared[str(model_name)],
                device=args.device,
                batch_size=args.feature_batch_size,
            )
            features[str(model_name)] = sequence
            teacher_logits[str(model_name)] = logits
            del model, checkpoint
        teacher = equal_probability_teacher(
            teacher_logits["atcnet"], teacher_logits["fbcnet"]
        )
        standardizer = _load_standardizer(parent_seed / "feature_standardizer.npz")
        eval_atc, eval_fbc = standardizer.transform(
            features["atcnet"], features["fbcnet"]
        )
        ann_path = (
            parent_seed
            / "students"
            / "ann_sew_ce"
            / "evaluation"
            / "predictions.npz"
        )
        with np.load(ann_path, allow_pickle=False) as archive:
            if not np.array_equal(labels, archive["label"]) or not np.array_equal(
                np.asarray(trial_ids), archive["trial_id"]
            ):
                raise RuntimeError("E30 parent ANN prediction identity drifted")
        variant_dir = _child_variant(output, args.subject, seed)
        checkpoint_path = variant_dir / "checkpoint.pt"
        expected_sha = barrier["checkpoints"].get(str(checkpoint_path.resolve()))
        if expected_sha is None or file_sha256(checkpoint_path) != expected_sha:
            raise RuntimeError(f"E30-ZP checkpoint changed: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = build_v9_dual_feature_student("sew_clif", **dict(config["model"]))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        prediction = predict_v9_dual_feature(
            model,
            eval_atc,
            eval_fbc,
            labels,
            teacher,
            device=args.device,
            batch_size=int(config["training"]["student_batch_size"]),
        )
        evaluation_dir = ensure_dir(variant_dir / "evaluation")
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
                "subject": args.subject,
                "seed": seed,
                "variant": CHILD_VARIANT,
                **{
                    key: float(prediction[key])
                    for key in (
                        "accuracy",
                        "balanced_accuracy",
                        "kappa",
                        "macro_f1",
                        "mean_firing_rate",
                    )
                },
                "training_checkpoint_sha256": expected_sha,
                "parent_ann_predictions_sha256": file_sha256(ann_path),
                "evaluation_gradient_updates": False,
                "data_identity": data_identity,
            },
        )
        del model, checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(
        ensure_dir(output / f"subject_{args.subject:02d}") / "evaluation_status.json",
        {
            "status": "completed",
            "subject": args.subject,
            "evaluation_gradient_updates": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "evaluate"), required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-barrier")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=96)
    args = parser.parse_args()
    if args.subject not in range(1, 10):
        raise ValueError("BNCI2014-004 subject must lie in [1, 9]")
    if args.phase == "evaluate" and not args.checkpoint_barrier:
        raise ValueError("evaluation requires --checkpoint-barrier")
    configure_cache_env()
    config_path = (ROOT / args.config).resolve()
    config = _config(config_path)
    _validate_parent(Path(args.parent).resolve())
    config_sha = file_sha256(config_path)
    source_sha = source_tree_digest(collect_source_tree_manifest(ROOT))
    started = time.time()
    if args.phase == "train":
        _train(args, config, config_sha, source_sha)
    else:
        _evaluate(args, config, config_sha, source_sha)
    print(
        {
            "status": "completed",
            "phase": args.phase,
            "subject": args.subject,
            "elapsed_seconds": time.time() - started,
        }
    )


if __name__ == "__main__":
    main()
