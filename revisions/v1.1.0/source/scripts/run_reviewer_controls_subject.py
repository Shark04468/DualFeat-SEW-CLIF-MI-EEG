"""Train or evaluate one subject of the reviewer-control campaign."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.reviewer_controls import (
    build_reviewer_model,
    fit_reviewer_model,
    predict_reviewer_model,
)
from dpc_snn.experiments.v8_protocol import (
    collect_source_tree_manifest,
    source_tree_digest,
)
from dpc_snn.experiments.v31_learning_curve import paired_run_seed
from dpc_snn.experiments.v62_protocol import file_sha256, sha256_fingerprint
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json
from dpc_snn.utils.storage import configure_cache_env
from scripts.run_v31_decoder_learning_curve_subject import (
    DATASET_INDEX,
    _load_standardizer,
)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if not {"atc", "fbc"}.issubset(archive.files):
            raise RuntimeError(f"invalid V31 feature cache: {path}")
        return (
            np.asarray(archive["atc"], dtype=np.float32),
            np.asarray(archive["fbc"], dtype=np.float32),
        )


def _labels(path: Path) -> tuple[np.ndarray, str]:
    metadata = read_json(path.with_suffix(".json"))
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) not in ({"label"}, {"label", "trial_id"}):
            raise RuntimeError(f"invalid recovery label artifact: {path}")
        labels = np.asarray(archive["label"], dtype=np.int64)
        trial_ids = (
            np.asarray(archive["trial_id"], dtype=np.str_) if "trial_id" in archive.files else None
        )
    identity = sha256_fingerprint({"shape": list(labels.shape), "values": labels.tolist()})
    if metadata.get("label_identity_sha256") != identity:
        raise RuntimeError(f"recovery label identity mismatch: {path}")
    if trial_ids is not None:
        if trial_ids.shape != labels.shape:
            raise RuntimeError(f"recovery trial identifiers are not aligned: {path}")
        if metadata.get("trial_ids_sha256") != sha256_fingerprint(trial_ids.tolist()):
            raise RuntimeError(f"recovery trial-identifier identity mismatch: {path}")
    return labels, identity


def _budget_dir(root: Path, dataset: str, subject: int, seed: int, budget: str) -> Path:
    return root / dataset / f"subject_{subject:02d}" / f"seed_{seed}" / f"budget_{budget}"


def _variant_budgets(config: dict[str, Any], dataset: str, variant: str) -> list[str]:
    spec = config["variants"][variant]
    if spec["budget_group"] == "fixed_budget_controls":
        return [str(value) for value in config["fixed_budget_controls"]]
    return [str(value) for value in config["datasets"][dataset]["equal_update_budgets"]]


def _selected(config: dict[str, Any], dataset: str, smoke: bool) -> tuple[list[int], list[str]]:
    seeds = [int(value) for value in config["seeds"]]
    variants = list(config["variants"])
    return ([seeds[0]], variants) if smoke else (seeds, variants)


def _build(config: dict[str, Any], dataset: str, variant: str) -> torch.nn.Module:
    spec = config["variants"][variant]
    return build_reviewer_model(
        str(spec["model"]),
        n_classes=int(config["datasets"][dataset]["n_classes"]),
        dropout=float(config["training"]["dropout"]),
        soft_gate_slope=float(config["training"]["soft_gate_slope"]),
        temporal_width=(int(spec["temporal_width"]) if "temporal_width" in spec else None),
    )


def _train(
    args: argparse.Namespace,
    config: dict[str, Any],
    config_sha: str,
    source_sha: str,
) -> None:
    parent = Path(config["parents"]["e31"]).resolve()
    output = Path(args.output).resolve()
    label_path = (
        Path(config["parents"]["training_labels"]).resolve()
        / args.dataset
        / f"subject_{args.subject:02d}"
        / "training_labels.npz"
    )
    labels, label_identity = _labels(label_path)
    seeds, variants = _selected(config, args.dataset, args.smoke)
    checkpoints: list[str] = []
    epochs = int(
        config["training"]["smoke_epochs"] if args.smoke else config["training"]["fixed_epochs"]
    )
    for seed in seeds:
        feature_root = (
            parent / args.dataset / f"subject_{args.subject:02d}" / f"seed_{seed}" / "feature_cache"
        )
        training_cache = feature_root / "training.npz"
        atc, fbc = _cache(training_cache)
        if labels.shape != (atc.shape[0],):
            raise RuntimeError("training labels do not align with the V31 feature cache")
        full_meta = read_json(
            _budget_dir(parent, args.dataset, args.subject, seed, "all") / "subset.json"
        )
        full_samples = int(full_meta["total_examples"])
        for variant in variants:
            spec = config["variants"][variant]
            budgets = _variant_budgets(config, args.dataset, variant)
            if args.smoke:
                budgets = sorted({budgets[0], budgets[-1]})
            for budget in budgets:
                parent_budget = _budget_dir(parent, args.dataset, args.subject, seed, budget)
                subset_path = parent_budget / "subset.npz"
                subset_meta_path = parent_budget / "subset.json"
                standardizer_path = parent_budget / "standardizer.npz"
                with np.load(subset_path, allow_pickle=False) as archive:
                    indices = np.asarray(archive["indices"], dtype=np.int64)
                subset_meta = read_json(subset_meta_path)
                standardizer = _load_standardizer(standardizer_path)
                train_atc, train_fbc = standardizer.transform(atc[indices], fbc[indices])
                directory = ensure_dir(
                    _budget_dir(output, args.dataset, args.subject, seed, budget) / variant
                )
                checkpoint_path = directory / "checkpoint.pt"
                metrics_path = directory / "training_metrics.json"
                run_seed = paired_run_seed(DATASET_INDEX[args.dataset], args.subject, seed)
                model = _build(config, args.dataset, variant)
                fingerprint = {
                    "schema": "dpc-snn-reviewer-training-run/v1",
                    "lineage": config["lineage"],
                    "config_sha256": config_sha,
                    "source_tree_sha256": source_sha,
                    "dataset": args.dataset,
                    "subject": args.subject,
                    "seed": seed,
                    "budget": budget,
                    "variant": variant,
                    "variant_spec": spec,
                    "run_seed": run_seed,
                    "smoke": bool(args.smoke),
                    "fixed_epochs": epochs,
                    "subset": subset_meta,
                    "label_identity_sha256": label_identity,
                    "parent_assets_sha256": {
                        "training_cache": file_sha256(training_cache),
                        "subset": file_sha256(subset_path),
                        "standardizer": file_sha256(standardizer_path),
                    },
                    "trainable_parameters": sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ),
                    "evaluation_data_accessed": False,
                }
                fingerprint["combined_sha256"] = sha256_fingerprint(fingerprint)
                fingerprint_path = directory / "training_fingerprint.json"
                if checkpoint_path.is_file() and metrics_path.is_file():
                    if read_json(fingerprint_path) != fingerprint:
                        raise RuntimeError(f"stale reviewer-control resume rejected: {directory}")
                    checkpoints.append(str(checkpoint_path.resolve()))
                    continue
                write_json(fingerprint_path, fingerprint)
                fit = fit_reviewer_model(
                    str(spec["model"]),
                    atc_train=train_atc,
                    fbc_train=train_fbc,
                    y_train=labels[indices],
                    n_classes=int(config["datasets"][args.dataset]["n_classes"]),
                    device=args.device,
                    seed=run_seed,
                    fixed_epochs=epochs,
                    batch_size=int(config["training"]["batch_size"]),
                    learning_rate=float(config["training"]["learning_rate"]),
                    weight_decay=float(config["training"]["weight_decay"]),
                    gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
                    dropout=float(config["training"]["dropout"]),
                    soft_gate_slope=float(config["training"]["soft_gate_slope"]),
                    temporal_width=(
                        int(spec["temporal_width"]) if "temporal_width" in spec else None
                    ),
                    equal_update_full_samples=(full_samples if spec["equal_update"] else None),
                    firing_rate_weight=float(spec["firing_rate_weight"]),
                )
                _atomic_torch(
                    checkpoint_path,
                    {
                        "state_dict": fit.last_state,
                        "fingerprint_sha256": fingerprint["combined_sha256"],
                        "dataset": args.dataset,
                        "subject": args.subject,
                        "seed": seed,
                        "budget": budget,
                        "variant": variant,
                        "n_classes": int(config["datasets"][args.dataset]["n_classes"]),
                    },
                )
                write_csv(directory / "history.csv", fit.history)
                write_json(
                    metrics_path,
                    {
                        "status": "training_completed_checkpoint_sealed",
                        "lineage": config["lineage"],
                        "dataset": args.dataset,
                        "subject": args.subject,
                        "seed": seed,
                        "budget": budget,
                        "variant": variant,
                        "fixed_epochs": epochs,
                        "optimizer_steps": fit.optimizer_steps,
                        "sample_exposures": fit.sample_exposures,
                        "equal_update_full_samples": full_samples if spec["equal_update"] else None,
                        "parameters": fingerprint["trainable_parameters"],
                        "elapsed_seconds": fit.elapsed_seconds,
                        "evaluation_data_accessed": False,
                    },
                )
                checkpoints.append(str(checkpoint_path.resolve()))
                del fit, model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    write_json(
        output / args.dataset / f"subject_{args.subject:02d}" / "training_status.json",
        {
            "status": "completed",
            "dataset": args.dataset,
            "subject": args.subject,
            "smoke": bool(args.smoke),
            "checkpoints": sorted(checkpoints),
            "evaluation_data_accessed": False,
        },
    )


def _evaluate(
    args: argparse.Namespace,
    config: dict[str, Any],
    config_sha: str,
    source_sha: str,
) -> None:
    if not args.checkpoint_barrier:
        raise ValueError("evaluation requires --checkpoint-barrier")
    parent = Path(config["parents"]["e31"]).resolve()
    output = Path(args.output).resolve()
    barrier_path = Path(args.checkpoint_barrier).resolve()
    barrier = read_json(barrier_path)
    if (
        barrier.get("status") != "sealed"
        or barrier.get("config_sha256") != config_sha
        or barrier.get("source_tree_sha256") != source_sha
        or barrier.get("smoke") != bool(args.smoke)
    ):
        raise RuntimeError("evaluation requires the matching reviewer checkpoint barrier")
    seeds, variants = _selected(config, args.dataset, args.smoke)
    for seed in seeds:
        feature_root = (
            parent / args.dataset / f"subject_{args.subject:02d}" / f"seed_{seed}" / "feature_cache"
        )
        eval_cache = feature_root / "evaluation.npz"
        atc, fbc = _cache(eval_cache)
        for variant in variants:
            budgets = _variant_budgets(config, args.dataset, variant)
            if args.smoke:
                budgets = sorted({budgets[0], budgets[-1]})
            for budget in budgets:
                parent_budget = _budget_dir(parent, args.dataset, args.subject, seed, budget)
                prediction_source = parent_budget / "ann_sew_ce" / "evaluation" / "predictions.npz"
                with np.load(prediction_source, allow_pickle=False) as archive:
                    labels = np.asarray(archive["label"], dtype=np.int64)
                    trial_ids = np.asarray(archive["trial_id"])
                standardizer_path = parent_budget / "standardizer.npz"
                standardizer = _load_standardizer(standardizer_path)
                eval_atc, eval_fbc = standardizer.transform(atc, fbc)
                directory = _budget_dir(output, args.dataset, args.subject, seed, budget) / variant
                checkpoint_path = directory / "checkpoint.pt"
                expected = barrier["checkpoints"].get(str(checkpoint_path.resolve()))
                if expected is None or file_sha256(checkpoint_path) != expected:
                    raise RuntimeError(
                        f"reviewer checkpoint changed after sealing: {checkpoint_path}"
                    )
                evaluation = ensure_dir(directory / "evaluation")
                prediction_path = evaluation / "predictions.npz"
                metrics_path = evaluation / "metrics.json"
                if prediction_path.is_file() and metrics_path.is_file():
                    continue
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                model = _build(config, args.dataset, variant)
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                prediction = predict_reviewer_model(
                    model,
                    eval_atc,
                    eval_fbc,
                    labels,
                    n_classes=int(config["datasets"][args.dataset]["n_classes"]),
                    device=args.device,
                    batch_size=int(config["training"]["batch_size"]),
                )
                _atomic_npz(
                    prediction_path,
                    logits=np.asarray(prediction["logits"], dtype=np.float32),
                    pred=np.asarray(prediction["logits"]).argmax(axis=1),
                    label=labels,
                    trial_id=trial_ids,
                )
                write_json(
                    metrics_path,
                    {
                        "status": "completed",
                        "lineage": config["lineage"],
                        "dataset": args.dataset,
                        "subject": args.subject,
                        "seed": seed,
                        "budget": budget,
                        "variant": variant,
                        **{
                            key: float(prediction[key])
                            for key in ("accuracy", "balanced_accuracy", "kappa", "macro_f1")
                        },
                        "spike_summary": prediction["spike_summary"],
                        "mean_continuous_activity": prediction["mean_continuous_activity"],
                        "checkpoint_sha256": expected,
                        "prediction_sha256": file_sha256(prediction_path),
                        "evaluation_cache_sha256": file_sha256(eval_cache),
                        "standardizer_sha256": file_sha256(standardizer_path),
                        "parent_prediction_sha256": file_sha256(prediction_source),
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
            "dataset": args.dataset,
            "subject": args.subject,
            "smoke": bool(args.smoke),
            "evaluation_gradient_updates": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "evaluate"), required=True)
    parser.add_argument("--dataset", choices=tuple(DATASET_INDEX), required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "experiments" / "reviewer_controls.yaml")
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-barrier")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    configure_cache_env()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.subject not in [int(value) for value in config["datasets"][args.dataset]["subjects"]]:
        raise ValueError("subject is outside the registered reviewer-control cohort")
    config_sha = file_sha256(config_path)
    source_sha = source_tree_digest(collect_source_tree_manifest(ROOT))
    started = time.perf_counter()
    if args.phase == "train":
        _train(args, config, config_sha, source_sha)
    else:
        _evaluate(args, config, config_sha, source_sha)
    print(
        json.dumps(
            {
                "status": "completed",
                "phase": args.phase,
                "dataset": args.dataset,
                "subject": args.subject,
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
