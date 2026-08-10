#!/usr/bin/env python3
"""Train and evaluate one E31-ZP subject from sealed E31 parent assets."""

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

from dpc_snn.experiments.v31_learning_curve import paired_run_seed  # noqa: E402
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    sha256_fingerprint,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    source_tree_digest,
)
from dpc_snn.experiments.v9_dual_feature_training import (  # noqa: E402
    fit_v9_dual_feature,
    predict_v9_dual_feature,
)
from dpc_snn.models.v9_dual_feature_student import (  # noqa: E402
    build_v9_dual_feature_student,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v31_decoder_learning_curve_subject import (  # noqa: E402
    DATASET_INDEX,
    _load_standardizer,
    _load_view,
)


DEFAULT_CONFIG = "configs/experiments/v33_e31_zero_penalty.yaml"
CHILD_VARIANT = "sew_clif_ce_fr0"


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"atc", "fbc", "teacher"}:
            raise RuntimeError(f"invalid parent feature cache: {path}")
        return (
            np.asarray(archive["atc"], dtype=np.float32),
            np.asarray(archive["fbc"], dtype=np.float32),
            np.asarray(archive["teacher"], dtype=np.float32),
        )


def _parent_budget(
    parent: Path, dataset: str, subject: int, seed: int, budget: str
) -> Path:
    return (
        parent
        / dataset
        / f"subject_{subject:02d}"
        / f"seed_{seed}"
        / f"budget_{budget}"
    )


def _child_budget(
    output: Path, dataset: str, subject: int, seed: int, budget: str
) -> Path:
    return (
        output
        / dataset
        / f"subject_{subject:02d}"
        / f"seed_{seed}"
        / f"budget_{budget}"
    )


def _training_parent_assets(parent_budget: Path, feature_root: Path) -> dict[str, str]:
    paths = {
        "training_features": feature_root / "training.npz",
        "training_feature_metadata": feature_root / "training.json",
        "subset": parent_budget / "subset.npz",
        "subset_metadata": parent_budget / "subset.json",
        "standardizer": parent_budget / "standardizer.npz",
        "ann_checkpoint": parent_budget / "ann_sew_ce" / "checkpoint.pt",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"sealed E31 parent assets are missing: {missing}")
    return {name: file_sha256(path) for name, path in paths.items()}


def _validate_parent(parent: Path) -> None:
    status = read_json(parent / "pipeline_status.json")
    decision = read_json(parent / "aggregate" / "decision.json")
    if status.get("status") != "completed" or decision.get("status") != "pass":
        raise RuntimeError("E31-ZP requires the completed passing E31 parent")


def _config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _model_kwargs(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    return {
        **dict(config["model"]),
        "n_classes": int(config["datasets"][dataset]["n_classes"]),
    }


def _train(
    args: argparse.Namespace,
    config: dict[str, Any],
    config_sha: str,
    source_sha: str,
) -> None:
    parent = Path(args.parent).resolve()
    output = Path(args.output).resolve()
    _, labels, _, _ = _load_view(args, config, "training")
    checkpoints: list[str] = []
    started = time.perf_counter()
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        feature_root = (
            parent
            / args.dataset
            / f"subject_{args.subject:02d}"
            / f"seed_{seed}"
            / "feature_cache"
        )
        atc, fbc, teacher = _load_cache(feature_root / "training.npz")
        if atc.shape[0] != labels.size or fbc.shape[0] != labels.size:
            raise RuntimeError("parent training features do not align with labels")
        for budget in config["budgets"][args.dataset]:
            parent_budget = _parent_budget(
                parent, args.dataset, args.subject, seed, str(budget)
            )
            child_budget = _child_budget(
                output, args.dataset, args.subject, seed, str(budget)
            )
            subset_meta = read_json(parent_budget / "subset.json")
            with np.load(parent_budget / "subset.npz", allow_pickle=False) as archive:
                indices = np.asarray(archive["indices"], dtype=np.int64)
            if subset_meta.get("label") != budget or indices.size == 0:
                raise RuntimeError(f"invalid parent subset: {parent_budget}")
            standardizer = _load_standardizer(parent_budget / "standardizer.npz")
            train_atc, train_fbc = standardizer.transform(atc[indices], fbc[indices])
            parent_hashes = _training_parent_assets(parent_budget, feature_root)
            run_seed = paired_run_seed(
                DATASET_INDEX[args.dataset], args.subject, seed
            )
            fingerprint = {
                "schema": "dpc-snn-e31zp-training-run/v1",
                "config_sha256": config_sha,
                "source_tree_sha256": source_sha,
                "parent_run": parent.name,
                "parent_assets_sha256": parent_hashes,
                "dataset": args.dataset,
                "subject": args.subject,
                "seed": seed,
                "budget": str(budget),
                "run_seed": run_seed,
                "firing_rate_weight": 0.0,
            }
            fingerprint["combined_sha256"] = sha256_fingerprint(fingerprint)
            variant_dir = child_budget / CHILD_VARIANT
            checkpoint_path = variant_dir / "checkpoint.pt"
            fingerprint_path = variant_dir / "training_fingerprint.json"
            metrics_path = variant_dir / "training_metrics.json"
            if checkpoint_path.is_file() and metrics_path.is_file():
                if read_json(fingerprint_path) != fingerprint:
                    raise RuntimeError(f"stale E31-ZP resume rejected: {variant_dir}")
                checkpoints.append(str(checkpoint_path.resolve()))
                continue
            if fingerprint_path.is_file() and read_json(fingerprint_path) != fingerprint:
                raise RuntimeError(f"partial stale E31-ZP resume: {variant_dir}")
            ensure_dir(variant_dir)
            write_json(fingerprint_path, fingerprint)
            training = config["training"]
            fit = fit_v9_dual_feature(
                "sew_clif_ce",
                atc_train=train_atc,
                fbc_train=train_fbc,
                y_train=labels[indices],
                teacher_train=teacher[indices],
                atc_validation=None,
                fbc_validation=None,
                y_validation=None,
                teacher_validation=None,
                device=args.device,
                seed=run_seed,
                fixed_epoch=int(training["fixed_epochs"]),
                scheduler_epochs=int(training["fixed_epochs"]),
                batch_size=int(training["batch_size"]),
                learning_rate=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
                firing_rate_weight=0.0,
                model_kwargs=_model_kwargs(config, args.dataset),
                run_label=(
                    f"E31ZP:{args.dataset}:S{args.subject}:seed{seed}:{budget}"
                ),
            )
            _atomic_torch_save(
                {
                    "state_dict": fit.last_state,
                    "dataset": args.dataset,
                    "subject": args.subject,
                    "seed": seed,
                    "budget": str(budget),
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
                    "budget": str(budget),
                    "examples_per_class": float(subset_meta["examples_per_class"]),
                    "variant": CHILD_VARIANT,
                    "fixed_epochs": int(training["fixed_epochs"]),
                    "optimizer_steps": fit.optimizer_steps,
                    "parameters": fit.model.parameter_count,
                    "elapsed_seconds": fit.elapsed_seconds,
                    "firing_rate_weight": 0.0,
                    "evaluation_data_accessed": False,
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
            "dataset": args.dataset,
            "subject": args.subject,
            "checkpoints": sorted(checkpoints),
            "elapsed_seconds": time.perf_counter() - started,
            "evaluation_data_accessed": False,
        },
    )


def _evaluate(
    args: argparse.Namespace,
    config: dict[str, Any],
    config_sha: str,
    source_sha: str,
) -> None:
    parent = Path(args.parent).resolve()
    output = Path(args.output).resolve()
    barrier = read_json(Path(args.checkpoint_barrier).resolve())
    if (
        barrier.get("schema") != "dpc-snn-e31zp-checkpoint-barrier/v1"
        or barrier.get("status") != "sealed"
        or barrier.get("config_sha256") != config_sha
        or barrier.get("source_tree_sha256") != source_sha
    ):
        raise RuntimeError("E31-ZP evaluation requires its sealed checkpoint barrier")
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        feature_root = (
            parent
            / args.dataset
            / f"subject_{args.subject:02d}"
            / f"seed_{seed}"
            / "feature_cache"
        )
        atc, fbc, teacher = _load_cache(feature_root / "evaluation.npz")
        for budget in config["budgets"][args.dataset]:
            parent_budget = _parent_budget(
                parent, args.dataset, args.subject, seed, str(budget)
            )
            child_budget = _child_budget(
                output, args.dataset, args.subject, seed, str(budget)
            )
            ann_path = parent_budget / "ann_sew_ce" / "evaluation" / "predictions.npz"
            with np.load(ann_path, allow_pickle=False) as archive:
                labels = np.asarray(archive["label"], dtype=np.int64)
                trial_ids = np.asarray(archive["trial_id"])
            if labels.size != atc.shape[0] or labels.size != fbc.shape[0]:
                raise RuntimeError("parent ANN predictions do not align with features")
            standardizer = _load_standardizer(parent_budget / "standardizer.npz")
            eval_atc, eval_fbc = standardizer.transform(atc, fbc)
            variant_dir = child_budget / CHILD_VARIANT
            checkpoint_path = variant_dir / "checkpoint.pt"
            expected_sha = barrier["checkpoints"].get(str(checkpoint_path.resolve()))
            if expected_sha is None or file_sha256(checkpoint_path) != expected_sha:
                raise RuntimeError(f"E31-ZP checkpoint changed after barrier: {checkpoint_path}")
            evaluation_dir = ensure_dir(variant_dir / "evaluation")
            if (evaluation_dir / "metrics.json").is_file():
                continue
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = build_v9_dual_feature_student(
                "sew_clif", **_model_kwargs(config, args.dataset)
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
                trial_id=trial_ids,
            )
            write_json(
                evaluation_dir / "metrics.json",
                {
                    "status": "completed",
                    "dataset": args.dataset,
                    "subject": args.subject,
                    "seed": seed,
                    "budget": str(budget),
                    "variant": CHILD_VARIANT,
                    "examples_per_class": float(
                        read_json(parent_budget / "subset.json")["examples_per_class"]
                    ),
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
                },
            )
            del model, checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_json(
        ensure_dir(output / args.dataset / f"subject_{args.subject:02d}")
        / "evaluation_status.json",
        {
            "status": "completed",
            "dataset": args.dataset,
            "subject": args.subject,
            "evaluation_gradient_updates": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "evaluate"), required=True)
    parser.add_argument("--dataset", choices=tuple(DATASET_INDEX), required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-barrier")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--publication-root", required=True)
    parser.add_argument("--v30-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument(
        "--bci2a-train-root",
        default="data/processed/bci2a_v8_session_t_4d04d3337fe3",
    )
    parser.add_argument(
        "--bci2a-eval-root",
        default="data/processed/bci2a_v8_session_e_4d04d3337fe3",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=96)
    args = parser.parse_args()
    configure_cache_env()
    config_path = (ROOT / args.config).resolve()
    config = _config(config_path)
    subjects = [int(value) for value in config["datasets"][args.dataset]["subjects"]]
    if args.subject not in subjects:
        raise ValueError("subject is outside the registered E31-ZP cohort")
    if args.phase == "evaluate" and not args.checkpoint_barrier:
        raise ValueError("evaluation requires --checkpoint-barrier")
    parent = Path(args.parent).resolve()
    _validate_parent(parent)
    config_sha = file_sha256(config_path)
    source_sha = source_tree_digest(collect_source_tree_manifest(ROOT))
    if args.phase == "train":
        _train(args, config, config_sha, source_sha)
    else:
        _evaluate(args, config, config_sha, source_sha)


if __name__ == "__main__":
    main()
