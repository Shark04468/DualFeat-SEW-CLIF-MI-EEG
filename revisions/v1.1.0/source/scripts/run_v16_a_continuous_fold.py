#!/usr/bin/env python3
"""Run one E16-A continuous-fusion nested Session-T fold."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
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
from dpc_snn.experiments.v16_training import (  # noqa: E402
    fit_v16_continuous,
    predict_v16_continuous,
)
from dpc_snn.experiments.v62_protocol import file_sha256  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v8_e1_baselines import _subject_file  # noqa: E402


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _environment(device: str) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0)
        if device.startswith("cuda") and torch.cuda.is_available()
        else None,
        "deterministic_cudnn": True,
    }


def _metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    return classification_metrics(labels, logits.argmax(axis=1), n_classes=4)


def _load_v9_fold(
    fold_dir: Path, labels: np.ndarray, reference_variant: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    status = read_json(fold_dir / "campaign_status.json")
    if status.get("status") != "completed" or status.get("session_e_accessed") is not False:
        raise RuntimeError(f"V9 anchor fold is incomplete or held-out contaminated: {fold_dir}")
    cache_path = fold_dir / "frozen_dual_feature_cache.npz"
    with np.load(cache_path, allow_pickle=False) as archive:
        cache = {name: archive[name] for name in archive.files}
    required = {
        "inner_train_indices",
        "inner_validation_indices",
        "outer_train_indices",
        "outer_test_indices",
        *{
            f"{branch}_{partition}_sequence"
            for branch in ("atcnet", "fbcnet")
            for partition in (
                "selection_train",
                "selection_validation",
                "outer_train",
                "outer_test",
            )
        },
        *{
            f"teacher_{partition}"
            for partition in (
                "selection_train",
                "selection_validation",
                "outer_train",
                "outer_test",
            )
        },
    }
    if not required.issubset(cache):
        raise RuntimeError(f"V9 cache lacks required arrays: {sorted(required - set(cache))}")
    prediction_path = fold_dir / reference_variant / "outer_predictions.npz"
    with np.load(prediction_path, allow_pickle=False) as archive:
        reference = {name: archive[name] for name in archive.files}
    outer_test = cache["outer_test_indices"].astype(np.int64)
    if (
        set(reference) != {"indices", "logits", "labels"}
        or not np.array_equal(reference["indices"], outer_test)
        or not np.array_equal(reference["labels"], labels[outer_test])
    ):
        raise RuntimeError("V9 reference predictions are not aligned with the frozen cache")
    return cache, reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--v9-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-epochs", type=int, default=0)
    parser.add_argument("--max-configs", type=int, default=0)
    args = parser.parse_args()

    configure_cache_env()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or len(config.get("hpo", [])) > 12:
        raise ValueError("E16-A config must define at most 12 HPO configurations")
    if args.subject not in range(1, 10) or args.fold not in range(6):
        raise ValueError("subject must lie in [1, 9] and fold in [0, 5]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    data_root = Path(args.data).resolve()
    v9_root = Path(args.v9_root).resolve()
    output = Path(args.output).resolve()
    fold_dir = (
        v9_root
        / f"subject_{args.subject:02d}"
        / f"seed_{args.seed}"
        / f"fold_{args.fold}"
    )
    subject_path = _subject_file(data_root, int(args.subject))
    data = load_processed_npz(subject_path)
    _, labels, _, access = session_t_development_view(data)
    labels = np.asarray(labels, dtype=np.int64)
    reference_variant = str(config["gate"]["v9_reference_variant"])
    cache, reference = _load_v9_fold(fold_dir, labels, reference_variant)

    hpo = list(config["hpo"])
    if int(args.max_configs) > 0:
        hpo = hpo[: int(args.max_configs)]
    training = dict(config["training"])
    epochs = int(args.smoke_epochs or training["epochs"])
    patience = min(int(training["patience"]), max(epochs, 1))
    resolved = {
        **vars(args),
        "config": str(config_path),
        "data": str(data_root),
        "v9_root": str(v9_root),
        "output": str(output),
        "resolved_experiment_config": config,
        "active_hpo": hpo,
        "effective_epochs": epochs,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    source_tree = collect_source_tree_manifest(ROOT)
    cache_path = fold_dir / "frozen_dual_feature_cache.npz"
    reference_path = fold_dir / reference_variant / "outer_predictions.npz"
    fingerprint = build_v8_run_fingerprint(
        resolved_run_config=resolved,
        source_tree=source_tree,
        data={"path": str(subject_path), "sha256": file_sha256(subject_path)},
        split={
            name: _array_sha256(cache[name])
            for name in (
                "inner_train_indices",
                "inner_validation_indices",
                "outer_train_indices",
                "outer_test_indices",
            )
        },
        augmentation={"enabled": False},
        prior={"enabled": False, "teacher_logits_are_model_inputs": False},
        checkpoint={
            "v9_cache": file_sha256(cache_path),
            "v9_reference_predictions": file_sha256(reference_path),
            "v9_run_fingerprint": read_json(fold_dir / "run_fingerprint.json").get(
                "combined_sha256"
            ),
        },
        environment=_environment(args.device),
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
    write_v8_fingerprint(output / "run_fingerprint.json", fingerprint)
    (output / "resolved_run.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )

    partitions = {
        "selection_train": cache["inner_train_indices"].astype(np.int64),
        "selection_validation": cache["inner_validation_indices"].astype(np.int64),
        "outer_train": cache["outer_train_indices"].astype(np.int64),
        "outer_test": cache["outer_test_indices"].astype(np.int64),
    }
    selection_standardizer = fit_feature_standardizer(
        cache["atcnet_selection_train_sequence"],
        cache["fbcnet_selection_train_sequence"],
    )
    outer_standardizer = fit_feature_standardizer(
        cache["atcnet_outer_train_sequence"], cache["fbcnet_outer_train_sequence"]
    )
    standardized: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for partition in partitions:
        standardizer = (
            selection_standardizer if partition.startswith("selection") else outer_standardizer
        )
        standardized[partition] = standardizer.transform(
            cache[f"atcnet_{partition}_sequence"],
            cache[f"fbcnet_{partition}_sequence"],
        )
    write_json(output / "selection_standardizer.json", selection_standardizer.as_dict())
    write_json(output / "outer_standardizer.json", outer_standardizer.as_dict())

    fold_seed = int(args.seed) * 100_003 + int(args.fold) * 1_009 + 16_000_007
    hpo_rows: list[dict[str, Any]] = []
    best_key: tuple[float, float, int, int] | None = None
    selected: tuple[int, dict[str, Any], int] | None = None
    started = time.time()
    for config_index, candidate in enumerate(hpo):
        model_kwargs = {
            "hidden_channels": int(candidate["hidden_channels"]),
            "dropout": float(candidate["dropout"]),
            "temporal_layers": int(training["temporal_layers"]),
            "band_channels": int(training["band_channels"]),
        }
        fit = fit_v16_continuous(
            atc_train=standardized["selection_train"][0],
            fbc_train=standardized["selection_train"][1],
            y_train=labels[partitions["selection_train"]],
            atc_validation=standardized["selection_validation"][0],
            fbc_validation=standardized["selection_validation"][1],
            y_validation=labels[partitions["selection_validation"]],
            device=args.device,
            seed=fold_seed + config_index,
            epochs=epochs,
            patience=patience,
            batch_size=int(training["batch_size"]),
            learning_rate=float(candidate["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            model_kwargs=model_kwargs,
            run_label=(
                f"E16-A-select:c{config_index}:S{args.subject}:seed{args.seed}:fold{args.fold}"
            ),
        )
        row = {
            "config_index": config_index,
            **candidate,
            "best_epoch": fit.best_epoch,
            "validation_kappa": fit.best_metric,
            "validation_accuracy": fit.best_accuracy,
            "parameters": fit.model.parameter_count,
            "optimizer_steps": fit.optimizer_steps,
            "elapsed_seconds": fit.elapsed_seconds,
        }
        hpo_rows.append(row)
        key = (float(fit.best_metric), float(fit.best_accuracy), -fit.best_epoch, -config_index)
        if best_key is None or key > best_key:
            best_key = key
            selected = (config_index, candidate, fit.best_epoch)
        del fit
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    if selected is None:
        raise RuntimeError("E16-A HPO did not produce a selected configuration")
    selected_index, selected_config, inner_best_epoch = selected
    selected_epoch = min(
        epochs,
        max(
            min(int(training["minimum_outer_epochs"]), epochs),
            int(inner_best_epoch),
        ),
    )
    selected_kwargs = {
        "hidden_channels": int(selected_config["hidden_channels"]),
        "dropout": float(selected_config["dropout"]),
        "temporal_layers": int(training["temporal_layers"]),
        "band_channels": int(training["band_channels"]),
    }
    outer_fit = fit_v16_continuous(
        atc_train=standardized["outer_train"][0],
        fbc_train=standardized["outer_train"][1],
        y_train=labels[partitions["outer_train"]],
        atc_validation=None,
        fbc_validation=None,
        y_validation=None,
        device=args.device,
        seed=fold_seed + 1_000_003,
        epochs=epochs,
        fixed_epoch=selected_epoch,
        scheduler_epochs=epochs,
        batch_size=int(training["batch_size"]),
        learning_rate=float(selected_config["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        model_kwargs=selected_kwargs,
        run_label=f"E16-A-outer:S{args.subject}:seed{args.seed}:fold{args.fold}",
    )
    evaluation = predict_v16_continuous(
        outer_fit.model,
        standardized["outer_test"][0],
        standardized["outer_test"][1],
        labels[partitions["outer_test"]],
        device=args.device,
        batch_size=int(training["batch_size"]),
    )
    teacher_logits = np.asarray(cache["teacher_outer_test"], dtype=np.float32)
    teacher_metrics = _metrics(labels[partitions["outer_test"]], teacher_logits)
    v9_metrics = _metrics(labels[partitions["outer_test"]], reference["logits"])
    result = {
        "subject": int(args.subject),
        "seed": int(args.seed),
        "fold": int(args.fold),
        "selected_config_index": int(selected_index),
        "selected_config": selected_config,
        "inner_best_epoch": int(inner_best_epoch),
        "selected_epoch": selected_epoch,
        "student_accuracy": evaluation["accuracy"],
        "student_kappa": evaluation["kappa"],
        "prefix_accuracy": evaluation["prefix_accuracy"],
        "v9_accuracy": v9_metrics["accuracy"],
        "equal_teacher_accuracy": teacher_metrics["accuracy"],
        "delta_vs_v9_pp": 100.0
        * (float(evaluation["accuracy"]) - float(v9_metrics["accuracy"])),
        "delta_vs_equal_teacher_pp": 100.0
        * (float(evaluation["accuracy"]) - float(teacher_metrics["accuracy"])),
        "parameters": outer_fit.model.parameter_count,
        "optimizer_steps": outer_fit.optimizer_steps + sum(
            int(row["optimizer_steps"]) for row in hpo_rows
        ),
        "elapsed_seconds": time.time() - started,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
        "run_fingerprint": fingerprint["combined_sha256"],
    }
    write_csv(output / "hpo_results.csv", hpo_rows)
    write_csv(output / "outer_history.csv", outer_fit.history)
    write_json(output / "result.json", result)
    torch.save(outer_fit.last_state, output / "outer_last.pt")
    np.savez_compressed(
        output / "outer_predictions.npz",
        indices=partitions["outer_test"],
        labels=labels[partitions["outer_test"]],
        logits=np.asarray(evaluation["logits"], dtype=np.float32),
        prefix_logits=np.asarray(evaluation["prefix_logits"], dtype=np.float32),
        teacher_logits=teacher_logits,
        v9_logits=np.asarray(reference["logits"], dtype=np.float32),
    )
    status = {
        "status": "completed",
        "stage": "E16-A-continuous-information-gate",
        "scope": "one BCI2a Session-T nested outer fold",
        "subject": int(args.subject),
        "seed": int(args.seed),
        "fold": int(args.fold),
        "hpo_configurations": len(hpo),
        "data_access": access,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
        "run_fingerprint": fingerprint["combined_sha256"],
    }
    write_json(output / "campaign_status.json", status)
    print(json.dumps({"status": "completed", "result": result}, indent=2))


if __name__ == "__main__":
    main()
