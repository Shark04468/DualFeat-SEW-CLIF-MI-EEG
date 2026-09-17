"""Train or evaluate one BCI2a LH/RH sensitivity subject on frozen V31 features."""

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
    bci2a_lh_rh_subset,
    build_reviewer_model,
    fit_reviewer_model,
    predict_reviewer_model,
)
from dpc_snn.experiments.v8_protocol import (
    collect_source_tree_manifest,
    source_tree_digest,
)
from dpc_snn.experiments.v9_dual_feature_training import fit_feature_standardizer
from dpc_snn.experiments.v31_learning_curve import (
    nested_stratified_subsets,
    paired_run_seed,
)
from dpc_snn.experiments.v62_protocol import file_sha256, sha256_fingerprint
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json
from dpc_snn.utils.storage import configure_cache_env
from scripts.run_v31_decoder_learning_curve_subject import (
    _load_standardizer,
    _save_standardizer,
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


def _training_labels(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    metadata = read_json(path.with_suffix(".json"))
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"label", "trial_id"}:
            raise RuntimeError(f"REV-E5 requires v2 recovery labels with trial IDs: {path}")
        labels = np.asarray(archive["label"], dtype=np.int64)
        trial_ids = np.asarray(archive["trial_id"], dtype=np.str_)
    identity = sha256_fingerprint({"shape": list(labels.shape), "values": labels.tolist()})
    if labels.shape != trial_ids.shape:
        raise RuntimeError("training labels and trial IDs are not aligned")
    if metadata.get("label_identity_sha256") != identity:
        raise RuntimeError("training-label identity mismatch")
    if metadata.get("trial_ids_sha256") != sha256_fingerprint(trial_ids.tolist()):
        raise RuntimeError("training trial-ID identity mismatch")
    return labels, trial_ids, identity


def _binary_view(
    labels: np.ndarray,
    trial_ids: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = config["dataset"]["source_labels"]
    keep, remapped = bci2a_lh_rh_subset(
        labels,
        left_hand_label=int(source["left_hand"]),
        right_hand_label=int(source["right_hand"]),
    )
    if trial_ids.shape != labels.shape:
        raise RuntimeError("trial IDs do not align with BCI2a labels")
    return keep, remapped, trial_ids[keep]


def _budget_dir(root: Path, subject: int, seed: int, budget: str) -> Path:
    return root / "bci2a_binary" / f"subject_{subject:02d}" / f"seed_{seed}" / f"budget_{budget}"


def _write_retained_manifest(
    path: Path,
    *,
    role: str,
    keep: np.ndarray,
    labels: np.ndarray,
    trial_ids: np.ndarray,
    config: dict[str, Any],
) -> None:
    payload = {
        "schema": "dpc-snn-rev-e5-retained-trials/v1",
        "role": role,
        "source_labels": config["dataset"]["source_labels"],
        "remapped_labels": config["dataset"]["remapped_labels"],
        "retained_trials": int(keep.size),
        "class_counts": {
            str(value): int(np.count_nonzero(labels == value)) for value in np.unique(labels)
        },
        "original_indices_sha256": sha256_fingerprint(keep.tolist()),
        "trial_ids_sha256": sha256_fingerprint(trial_ids.tolist()),
    }
    if path.is_file() and read_json(path) != payload:
        raise RuntimeError(f"stale retained-trial manifest: {path}")
    if not path.is_file():
        write_json(path, payload)


def _build(config: dict[str, Any], variant: str) -> torch.nn.Module:
    return build_reviewer_model(
        str(config["variants"][variant]["model"]),
        n_classes=2,
        dropout=float(config["training"]["dropout"]),
        soft_gate_slope=float(config["training"]["soft_gate_slope"]),
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
        / "bci2a"
        / f"subject_{args.subject:02d}"
        / "training_labels.npz"
    )
    labels, trial_ids, label_identity = _training_labels(label_path)
    keep, binary_labels, binary_ids = _binary_view(labels, trial_ids, config)
    subject_root = ensure_dir(output / "bci2a_binary" / f"subject_{args.subject:02d}")
    _write_retained_manifest(
        subject_root / "retained_training_trials.json",
        role="training",
        keep=keep,
        labels=binary_labels,
        trial_ids=binary_ids,
        config=config,
    )
    seeds = [int(config["seeds"][0])] if args.smoke else [int(value) for value in config["seeds"]]
    epochs = int(
        config["training"]["smoke_epochs"] if args.smoke else config["training"]["fixed_epochs"]
    )
    checkpoints: list[str] = []
    for seed in seeds:
        feature_root = (
            parent / "bci2a" / f"subject_{args.subject:02d}" / f"seed_{seed}" / "feature_cache"
        )
        training_cache = feature_root / "training.npz"
        atc, fbc = _cache(training_cache)
        if atc.shape[0] != labels.size:
            raise RuntimeError("training features and labels are not aligned")
        filtered_atc, filtered_fbc = atc[keep], fbc[keep]
        subsets = nested_stratified_subsets(
            binary_labels,
            config["numeric_budgets_per_class"],
            seed=paired_run_seed(3, args.subject, seed),
        )
        if args.smoke:
            subsets = [subsets[0], subsets[-1]]
        for subset in subsets:
            directory = ensure_dir(_budget_dir(output, args.subject, seed, subset.label))
            original_indices = keep[subset.indices]
            retained_ids = trial_ids[original_indices]
            subset_path = directory / "subset.npz"
            subset_meta_path = directory / "subset.json"
            subset_meta = {
                "schema": "dpc-snn-rev-e5-subset/v1",
                "label": subset.label,
                "requested_per_class": subset.requested_per_class,
                "class_counts": {
                    str(key): int(value) for key, value in subset.class_counts.items()
                },
                "examples_per_class": subset.examples_per_class,
                "total_examples": int(subset.indices.size),
                "filtered_index_sha256": subset.index_sha256,
                "original_index_sha256": sha256_fingerprint(original_indices.tolist()),
                "trial_ids_sha256": sha256_fingerprint(retained_ids.tolist()),
                "nested_sampling": True,
            }
            if subset_path.is_file() or subset_meta_path.is_file():
                if not subset_path.is_file() or read_json(subset_meta_path) != subset_meta:
                    raise RuntimeError(f"stale REV-E5 subset: {directory}")
            else:
                _atomic_npz(
                    subset_path,
                    filtered_indices=subset.indices,
                    original_indices=original_indices,
                    trial_id=retained_ids,
                )
                write_json(subset_meta_path, subset_meta)
            standardizer_path = directory / "standardizer.npz"
            if standardizer_path.is_file():
                standardizer = _load_standardizer(standardizer_path)
            else:
                standardizer = fit_feature_standardizer(
                    filtered_atc[subset.indices], filtered_fbc[subset.indices]
                )
                _save_standardizer(standardizer_path, standardizer)
            train_atc, train_fbc = standardizer.transform(
                filtered_atc[subset.indices], filtered_fbc[subset.indices]
            )
            run_seed = paired_run_seed(3, args.subject, seed)
            for variant, spec in config["variants"].items():
                variant_dir = ensure_dir(directory / variant)
                checkpoint_path = variant_dir / "checkpoint.pt"
                metrics_path = variant_dir / "training_metrics.json"
                model = _build(config, variant)
                fingerprint = {
                    "schema": "dpc-snn-rev-e5-training-run/v1",
                    "lineage": config["lineage"],
                    "config_sha256": config_sha,
                    "source_tree_sha256": source_sha,
                    "subject": args.subject,
                    "seed": seed,
                    "budget": subset.label,
                    "variant": variant,
                    "variant_spec": spec,
                    "run_seed": run_seed,
                    "smoke": bool(args.smoke),
                    "fixed_epochs": epochs,
                    "label_identity_sha256": label_identity,
                    "training_cache_sha256": file_sha256(training_cache),
                    "subset_sha256": file_sha256(subset_path),
                    "standardizer_sha256": file_sha256(standardizer_path),
                    "trainable_parameters": sum(
                        value.numel() for value in model.parameters() if value.requires_grad
                    ),
                    "evaluation_data_accessed": False,
                }
                fingerprint["combined_sha256"] = sha256_fingerprint(fingerprint)
                fingerprint_path = variant_dir / "training_fingerprint.json"
                if checkpoint_path.is_file() and metrics_path.is_file():
                    if read_json(fingerprint_path) != fingerprint:
                        raise RuntimeError(f"stale REV-E5 resume: {variant_dir}")
                    checkpoints.append(str(checkpoint_path.resolve()))
                    continue
                write_json(fingerprint_path, fingerprint)
                fit = fit_reviewer_model(
                    str(spec["model"]),
                    atc_train=train_atc,
                    fbc_train=train_fbc,
                    y_train=binary_labels[subset.indices],
                    n_classes=2,
                    device=args.device,
                    seed=run_seed,
                    fixed_epochs=epochs,
                    batch_size=int(config["training"]["batch_size"]),
                    learning_rate=float(config["training"]["learning_rate"]),
                    weight_decay=float(config["training"]["weight_decay"]),
                    gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
                    dropout=float(config["training"]["dropout"]),
                    soft_gate_slope=float(config["training"]["soft_gate_slope"]),
                    firing_rate_weight=float(spec["firing_rate_weight"]),
                )
                _atomic_torch(
                    checkpoint_path,
                    {
                        "state_dict": fit.last_state,
                        "fingerprint_sha256": fingerprint["combined_sha256"],
                        "subject": args.subject,
                        "seed": seed,
                        "budget": subset.label,
                        "variant": variant,
                        "n_classes": 2,
                    },
                )
                write_csv(variant_dir / "history.csv", fit.history)
                write_json(
                    metrics_path,
                    {
                        "status": "training_completed_checkpoint_sealed",
                        "lineage": config["lineage"],
                        "subject": args.subject,
                        "seed": seed,
                        "budget": subset.label,
                        "variant": variant,
                        "fixed_epochs": epochs,
                        "optimizer_steps": fit.optimizer_steps,
                        "sample_exposures": fit.sample_exposures,
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
        subject_root / "training_status.json",
        {
            "status": "completed",
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
        raise ValueError("REV-E5 evaluation requires --checkpoint-barrier")
    parent = Path(config["parents"]["e31"]).resolve()
    output = Path(args.output).resolve()
    barrier = read_json(Path(args.checkpoint_barrier).resolve())
    if (
        barrier.get("status") != "sealed"
        or barrier.get("config_sha256") != config_sha
        or barrier.get("source_tree_sha256") != source_sha
        or barrier.get("smoke") != bool(args.smoke)
    ):
        raise RuntimeError("REV-E5 evaluation requires the matching global barrier")
    seeds = [int(config["seeds"][0])] if args.smoke else [int(value) for value in config["seeds"]]
    for seed in seeds:
        feature_root = (
            parent / "bci2a" / f"subject_{args.subject:02d}" / f"seed_{seed}" / "feature_cache"
        )
        eval_cache = feature_root / "evaluation.npz"
        atc, fbc = _cache(eval_cache)
        parent_prediction = (
            parent
            / "bci2a"
            / f"subject_{args.subject:02d}"
            / f"seed_{seed}"
            / "budget_all"
            / "ann_sew_ce"
            / "evaluation"
            / "predictions.npz"
        )
        with np.load(parent_prediction, allow_pickle=False) as archive:
            source_labels = np.asarray(archive["label"], dtype=np.int64)
            source_ids = np.asarray(archive["trial_id"], dtype=np.str_)
        keep, labels, trial_ids = _binary_view(source_labels, source_ids, config)
        if atc.shape[0] != source_labels.size:
            raise RuntimeError("evaluation features and labels are not aligned")
        subject_root = ensure_dir(output / "bci2a_binary" / f"subject_{args.subject:02d}")
        _write_retained_manifest(
            subject_root / "retained_evaluation_trials.json",
            role="evaluation",
            keep=keep,
            labels=labels,
            trial_ids=trial_ids,
            config=config,
        )
        filtered_atc, filtered_fbc = atc[keep], fbc[keep]
        seed_root = output / "bci2a_binary" / f"subject_{args.subject:02d}" / f"seed_{seed}"
        budget_dirs = sorted(seed_root.glob("budget_*"))
        for directory in budget_dirs:
            standardizer_path = directory / "standardizer.npz"
            standardizer = _load_standardizer(standardizer_path)
            eval_atc, eval_fbc = standardizer.transform(filtered_atc, filtered_fbc)
            for variant in config["variants"]:
                variant_dir = directory / variant
                checkpoint_path = variant_dir / "checkpoint.pt"
                expected = barrier["checkpoints"].get(str(checkpoint_path.resolve()))
                if expected is None or file_sha256(checkpoint_path) != expected:
                    raise RuntimeError(
                        f"REV-E5 checkpoint changed after sealing: {checkpoint_path}"
                    )
                evaluation = ensure_dir(variant_dir / "evaluation")
                prediction_path = evaluation / "predictions.npz"
                metrics_path = evaluation / "metrics.json"
                if prediction_path.is_file() and metrics_path.is_file():
                    continue
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                model = _build(config, variant)
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                prediction = predict_reviewer_model(
                    model,
                    eval_atc,
                    eval_fbc,
                    labels,
                    n_classes=2,
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
                        "dataset": "bci2a_binary",
                        "subject": args.subject,
                        "seed": seed,
                        "budget": directory.name.removeprefix("budget_"),
                        "variant": variant,
                        **{
                            key: float(prediction[key])
                            for key in ("accuracy", "balanced_accuracy", "kappa", "macro_f1")
                        },
                        "spike_summary": prediction["spike_summary"],
                        "checkpoint_sha256": expected,
                        "prediction_sha256": file_sha256(prediction_path),
                        "evaluation_cache_sha256": file_sha256(eval_cache),
                        "parent_prediction_sha256": file_sha256(parent_prediction),
                        "evaluation_gradient_updates": False,
                    },
                )
                del checkpoint, model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    write_json(
        output / "bci2a_binary" / f"subject_{args.subject:02d}" / "evaluation_status.json",
        {
            "status": "completed",
            "subject": args.subject,
            "smoke": bool(args.smoke),
            "evaluation_gradient_updates": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "evaluate"), required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "bci2a_binary_sensitivity.yaml"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-barrier")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    configure_cache_env()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.subject not in [int(value) for value in config["dataset"]["subjects"]]:
        raise ValueError("subject is outside the registered REV-E5 cohort")
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
                "subject": args.subject,
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
