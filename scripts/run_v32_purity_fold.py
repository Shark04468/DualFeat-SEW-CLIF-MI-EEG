"""Train one CE-only SEW-CLIF fold from a sealed E28 feature cache."""

from __future__ import annotations

import argparse
import hashlib
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
from dpc_snn.experiments.v8_baselines import session_t_development_view  # noqa: E402
from dpc_snn.experiments.v62_protocol import file_sha256  # noqa: E402
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    source_tree_digest,
)
from dpc_snn.experiments.v9_dual_feature_training import (  # noqa: E402
    fit_feature_standardizer,
    fit_v9_dual_feature,
    predict_v9_dual_feature,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v8_e1_baselines import _subject_file  # noqa: E402


PARTITIONS = (
    "selection_train",
    "selection_validation",
    "outer_train",
    "outer_test",
)
INDEX_KEYS = {
    "selection_train": "inner_train_indices",
    "selection_validation": "inner_validation_indices",
    "outer_train": "outer_train_indices",
    "outer_test": "outer_test_indices",
}


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_reference(path: Path, indices: np.ndarray, labels: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"indices", "logits", "labels"}:
            raise RuntimeError(f"invalid reference prediction archive: {path}")
        if not np.array_equal(archive["indices"], indices):
            raise RuntimeError(f"reference indices differ from sealed cache: {path}")
        if not np.array_equal(archive["labels"], labels):
            raise RuntimeError(f"reference labels differ from Session T data: {path}")
        return np.asarray(archive["logits"], dtype=np.float32)


def _validate_standardizer(cache_dir: Path, name: str, fitted: Any) -> None:
    saved = read_json(cache_dir / f"{name}_feature_standardizer.json")
    current = fitted.as_dict()
    for key in ("atc_mean", "atc_scale", "fbc_mean", "fbc_scale"):
        if not np.allclose(
            np.asarray(saved[key], dtype=np.float32),
            np.asarray(current[key], dtype=np.float32),
            rtol=1e-6,
            atol=1e-7,
        ):
            raise RuntimeError(f"{name} feature standardizer does not replay: {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache-fold", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--fusion-mode",
        choices=("interaction", "simple", "atc_only", "fbc_only"),
    )
    parser.add_argument("--config", default="configs/experiments/v32_matched_purity_and_fusion.yaml")
    args = parser.parse_args()

    configure_cache_env()
    if args.subject not in range(1, 10) or args.seed not in range(3) or args.fold not in range(6):
        raise ValueError("V32 BCI2a fold identity is outside the registered range")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    config_path = (ROOT / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    training = dict(config["training"])
    model_kwargs = dict(config["model"])
    if args.fusion_mode is not None:
        model_kwargs["fusion_mode"] = args.fusion_mode
    if float(training["firing_rate_weight"]) != 0.0:
        raise RuntimeError("V32 purity runner requires firing_rate_weight=0")

    data_root = Path(args.data).resolve()
    subject_path = _subject_file(data_root, args.subject)
    data = load_processed_npz(subject_path)
    _, labels, _, access = session_t_development_view(data)
    cache_dir = Path(args.cache_fold).resolve()
    cache_path = cache_dir / "frozen_dual_feature_cache.npz"
    ann_reference_path = cache_dir / "ann_sew_ce" / "outer_predictions.npz"
    regularized_reference_path = cache_dir / "sew_clif_ce" / "outer_predictions.npz"
    required = (
        cache_path,
        ann_reference_path,
        regularized_reference_path,
        cache_dir / "selection_feature_standardizer.json",
        cache_dir / "outer_feature_standardizer.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V32 sealed inputs are missing: {missing}")

    output = Path(args.output).resolve()
    source_sha = source_tree_digest(collect_source_tree_manifest(ROOT))
    fingerprint_payload = {
        "schema": config["schema"],
        "subject": args.subject,
        "seed": args.seed,
        "fold": args.fold,
        "training": training,
        "model": model_kwargs,
        "source_tree_sha256": source_sha,
        "data_sha256": file_sha256(subject_path),
        "cache_sha256": file_sha256(cache_path),
        "ann_reference_sha256": file_sha256(ann_reference_path),
        "regularized_reference_sha256": file_sha256(regularized_reference_path),
    }
    fingerprint = _fingerprint(fingerprint_payload)
    status_path = output / "status.json"
    if status_path.is_file():
        status = read_json(status_path)
        if status.get("fingerprint_sha256") != fingerprint:
            raise RuntimeError(f"stale V32 resume rejected: {output}")
        if status.get("status") == "completed":
            print(json.dumps({"status": "skipped_completed", "output": str(output)}))
            return

    ensure_dir(output)
    write_json(
        output / "fingerprint.json",
        {**fingerprint_payload, "combined_sha256": fingerprint},
    )
    with np.load(cache_path, allow_pickle=False) as archive:
        cache = {name: np.asarray(archive[name]) for name in archive.files}
    indices = {
        name: np.asarray(cache[INDEX_KEYS[name]], dtype=np.int64)
        for name in PARTITIONS
    }
    if len(np.unique(np.concatenate((indices["outer_train"], indices["outer_test"])))) != len(labels):
        raise RuntimeError("sealed outer folds do not cover Session T exactly once")

    features: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for partition in PARTITIONS:
        features[partition] = (
            np.asarray(cache[f"atcnet_{partition}_sequence"], dtype=np.float32),
            np.asarray(cache[f"fbcnet_{partition}_sequence"], dtype=np.float32),
        )
    selection_standardizer = fit_feature_standardizer(*features["selection_train"])
    outer_standardizer = fit_feature_standardizer(*features["outer_train"])
    _validate_standardizer(cache_dir, "selection", selection_standardizer)
    _validate_standardizer(cache_dir, "outer", outer_standardizer)
    standardized = {
        partition: (
            selection_standardizer if partition.startswith("selection") else outer_standardizer
        ).transform(*features[partition])
        for partition in PARTITIONS
    }

    outer_indices = indices["outer_test"]
    outer_labels = labels[outer_indices]
    ann_logits = _load_reference(ann_reference_path, outer_indices, outer_labels)
    regularized_logits = _load_reference(
        regularized_reference_path, outer_indices, outer_labels
    )
    ann_metrics = classification_metrics(outer_labels, ann_logits.argmax(1), n_classes=4)
    regularized_metrics = classification_metrics(
        outer_labels, regularized_logits.argmax(1), n_classes=4
    )

    fold_seed = args.seed * 100_003 + args.fold * 1_009 + 9_300_007
    started = time.perf_counter()
    selection_fit = fit_v9_dual_feature(
        str(training["variant"]),
        atc_train=standardized["selection_train"][0],
        fbc_train=standardized["selection_train"][1],
        y_train=labels[indices["selection_train"]],
        teacher_train=np.asarray(cache["teacher_selection_train"], dtype=np.float32),
        atc_validation=standardized["selection_validation"][0],
        fbc_validation=standardized["selection_validation"][1],
        y_validation=labels[indices["selection_validation"]],
        teacher_validation=np.asarray(
            cache["teacher_selection_validation"], dtype=np.float32
        ),
        device=args.device,
        seed=fold_seed,
        epochs=int(training["epochs"]),
        patience=int(training["patience"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        firing_rate_weight=0.0,
        model_kwargs=model_kwargs,
        run_label=f"V32-select:S{args.subject}:seed{args.seed}:fold{args.fold}",
    )
    selected_epoch = min(
        int(training["epochs"]),
        max(int(training["minimum_outer_epochs"]), int(selection_fit.best_epoch)),
    )
    outer_fit = fit_v9_dual_feature(
        str(training["variant"]),
        atc_train=standardized["outer_train"][0],
        fbc_train=standardized["outer_train"][1],
        y_train=labels[indices["outer_train"]],
        teacher_train=np.asarray(cache["teacher_outer_train"], dtype=np.float32),
        atc_validation=None,
        fbc_validation=None,
        y_validation=None,
        teacher_validation=None,
        device=args.device,
        seed=fold_seed + 1_000_003,
        epochs=int(training["epochs"]),
        fixed_epoch=selected_epoch,
        scheduler_epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        firing_rate_weight=0.0,
        model_kwargs=model_kwargs,
        run_label=f"V32-outer:S{args.subject}:seed{args.seed}:fold{args.fold}",
    )
    evaluation = predict_v9_dual_feature(
        outer_fit.model,
        standardized["outer_test"][0],
        standardized["outer_test"][1],
        outer_labels,
        np.asarray(cache["teacher_outer_test"], dtype=np.float32),
        device=args.device,
        batch_size=int(training["batch_size"]),
    )
    row = {
        "status": "completed",
        "subject": args.subject,
        "seed": args.seed,
        "fold": args.fold,
        "fusion_mode": model_kwargs["fusion_mode"],
        "firing_rate_weight": 0.0,
        "selected_epoch": selected_epoch,
        "fr0_accuracy": float(evaluation["accuracy"]),
        "ann_accuracy": float(ann_metrics["accuracy"]),
        "fr001_accuracy": float(regularized_metrics["accuracy"]),
        "fr0_minus_ann_pp": 100.0 * (float(evaluation["accuracy"]) - float(ann_metrics["accuracy"])),
        "fr0_minus_fr001_pp": 100.0 * (float(evaluation["accuracy"]) - float(regularized_metrics["accuracy"])),
        "mean_firing_rate": float(evaluation["mean_firing_rate"]),
        "parameters": int(outer_fit.model.parameter_count),
        "optimizer_steps": selection_fit.optimizer_steps + outer_fit.optimizer_steps,
        "train_seconds": selection_fit.elapsed_seconds + outer_fit.elapsed_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "fingerprint_sha256": fingerprint,
        "session_e_accessed": False,
        "data_access": access,
    }
    torch.save(selection_fit.best_state, output / "selection_best.pt")
    torch.save(outer_fit.last_state, output / "outer_last.pt")
    write_csv(output / "selection_history.csv", selection_fit.history)
    write_csv(output / "outer_history.csv", outer_fit.history)
    np.savez_compressed(
        output / "outer_predictions.npz",
        indices=outer_indices,
        labels=outer_labels,
        fr0_logits=np.asarray(evaluation["logits"], dtype=np.float32),
        ann_logits=ann_logits,
        fr001_logits=regularized_logits,
    )
    write_json(output / "metrics.json", row)
    write_json(output / "status.json", row)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
