#!/usr/bin/env python3
"""Run one V12 bounded multi-rate architecture-search fold."""

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
    V12_OBJECTIVES,
    fit_v12,
    predict_v12,
)
from dpc_snn.models.v12_multirate_student import (  # noqa: E402
    V12_MODEL_ARCHITECTURES,
    build_v12_student,
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


def _perturbations(
    atc: np.ndarray, fbc: np.ndarray, *, seed: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(int(seed))
    values: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "nominal": (atc, fbc),
        "gaussian_0.10": (
            np.ascontiguousarray(atc + 0.10 * rng.normal(size=atc.shape), dtype=np.float32),
            np.ascontiguousarray(fbc + 0.10 * rng.normal(size=fbc.shape), dtype=np.float32),
        ),
        "gaussian_0.20": (
            np.ascontiguousarray(atc + 0.20 * rng.normal(size=atc.shape), dtype=np.float32),
            np.ascontiguousarray(fbc + 0.20 * rng.normal(size=fbc.shape), dtype=np.float32),
        ),
    }
    atc_drop = atc.copy()
    fbc_drop = fbc.copy()
    atc_drop[..., np.arange(0, atc.shape[-1], 4)] = 0.0
    fbc_drop[..., np.arange(0, fbc.shape[-1], 4)] = 0.0
    values["atc_feature_drop_25pct"] = (atc_drop, fbc)
    values["fbc_feature_drop_25pct"] = (atc, fbc_drop)
    atc_shift = np.concatenate((np.zeros_like(atc[:, :1]), atc[:, :-1]), axis=1)
    fbc_shift = np.concatenate((np.zeros_like(fbc[:, :1]), fbc[:, :-1]), axis=1)
    values["atc_shift_one_step"] = (atc_shift, fbc)
    values["fbc_shift_one_window"] = (atc, fbc_shift)
    return values


def _latency_ms(
    model: torch.nn.Module,
    atc: np.ndarray,
    fbc: np.ndarray,
    *,
    device: str,
    repeats: int = 20,
) -> float:
    model.eval().to(device)
    atc_tensor = torch.from_numpy(atc).to(device)
    fbc_tensor = torch.from_numpy(fbc).to(device)
    with torch.no_grad():
        for _ in range(5):
            model(atc_tensor, fbc_tensor)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        samples: list[float] = []
        for _ in range(int(repeats)):
            tick = time.perf_counter()
            model(atc_tensor, fbc_tensor)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            samples.append(1000.0 * (time.perf_counter() - tick))
    return float(np.median(samples))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--anchor-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--variants", default=",".join(V12_MODEL_ARCHITECTURES))
    parser.add_argument("--decoder-kinds", default="clif")
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
    decoder_kinds = _csv(args.decoder_kinds)
    learning_rates = _csv_float(args.learning_rates)
    weight_decays = _csv_float(args.weight_decays)
    if set(variants) - set(V12_MODEL_ARCHITECTURES) or len(variants) != len(
        set(variants)
    ):
        raise ValueError("unknown or duplicate V12 variants")
    if set(decoder_kinds) - {"ann", "plif", "clif"} or len(decoder_kinds) != len(
        set(decoder_kinds)
    ):
        raise ValueError("unknown or duplicate V12 decoder kinds")
    if not variants or args.subject not in range(1, 10) or args.fold not in range(6):
        raise ValueError("invalid V12 subject, fold, or variant list")
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
        raise RuntimeError("V9 anchor fold is incomplete or unlocked")
    if anchor_source.get("sha256") != ANCHOR_SOURCE_DIGEST:
        raise RuntimeError("V9 anchor source digest mismatch")
    if anchor_replay.get("status") != "passed":
        raise RuntimeError("V9 teacher replay did not pass")

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
        raise RuntimeError("V9 cache lacks V12 branch targets")

    subject_path = _subject_file(data_root, int(args.subject))
    data = load_processed_npz(subject_path)
    _, labels, _, access = session_t_development_view(data)
    inner_train = arrays["inner_train_indices"].astype(np.int64)
    inner_validation = arrays["inner_validation_indices"].astype(np.int64)
    outer_train = arrays["outer_train_indices"].astype(np.int64)
    outer_test = arrays["outer_test_indices"].astype(np.int64)
    anchor_snn_logits = _prediction(
        anchor_fold / "sew_clif_kd" / "outer_predictions.npz",
        outer_test,
        labels[outer_test],
    )
    source_tree = collect_source_tree_manifest(ROOT)
    model_specs = [
        {
            "name": (
                variant
                if decoder_kinds == ["clif"]
                else f"{variant}__{decoder_kind}"
            ),
            "variant": variant,
            "decoder_kind": decoder_kind,
        }
        for variant in variants
        for decoder_kind in decoder_kinds
    ]
    capacities = {
        str(spec["name"]): int(
            sum(
                value.numel()
                for value in build_v12_student(
                    str(spec["variant"]), decoder_kind=str(spec["decoder_kind"])
                ).parameters()
            )
        )
        for spec in model_specs
    }
    grid = [
        {"learning_rate": learning_rate, "weight_decay": weight_decay}
        for learning_rate, weight_decay in itertools.product(learning_rates, weight_decays)
    ]
    if len(grid) * len(model_specs) > 12:
        raise RuntimeError("V12 bounded search exceeds 12 configurations")

    resolved_config = {
        **vars(args),
        "data": str(data_root),
        "anchor_root": str(anchor_root),
        "output": str(output),
        "variants": variants,
        "decoder_kinds": decoder_kinds,
        "model_specs": model_specs,
        "learning_rates": learning_rates,
        "weight_decays": weight_decays,
        "hpo_grid": grid,
        "objectives": {
            name: V12_OBJECTIVES[name].__dict__ for name in variants
        },
        "selection": "inner validation kappa, accuracy, earliest epoch, grid order",
        "session_e_accessed": False,
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
            "status": "recorded",
            "decoder_parameters": capacities,
            "frozen_frontend_parameters": 125_544,
            "comparison_scope": "architecture search; matched controls deferred to E14",
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
    robustness_rows: list[dict[str, Any]] = []
    started = time.time()
    fold_seed = int(args.seed) * 100_003 + int(args.fold) * 1_009 + 12_100_019
    perturbations = _perturbations(
        standardized["outer_test"][0],
        standardized["outer_test"][1],
        seed=fold_seed + 55_003,
    )
    for specification in model_specs:
        model_name = str(specification["name"])
        variant = str(specification["variant"])
        decoder_kind = str(specification["decoder_kind"])
        candidates: list[tuple[tuple[float, float, int, int], dict[str, float], Any]] = []
        for grid_index, configuration in enumerate(grid):
            fit = fit_v12(
                variant,
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
                decoder_kind=decoder_kind,
                run_label=(
                    f"V12-E1-select:{model_name}:grid{grid_index}:"
                    f"S{args.subject}:seed{args.seed}:fold{args.fold}"
                ),
            )
            hpo_rows.append(
                {
                    "model": model_name,
                    "variant": variant,
                    "decoder_kind": decoder_kind,
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
            variant,
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
            decoder_kind=decoder_kind,
            run_label=(
                f"V12-E1-outer:{model_name}:S{args.subject}:"
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
        variant_dir = ensure_dir(output / model_name)
        row = {
            "model": model_name,
            "variant": variant,
            "decoder_kind": decoder_kind,
            "subject": int(args.subject),
            "seed": int(args.seed),
            "fold": int(args.fold),
            "selected_grid_index": selected_grid_index,
            "selected_learning_rate": float(selected_configuration["learning_rate"]),
            "selected_weight_decay": float(selected_configuration["weight_decay"]),
            "selected_epoch": selected_epoch,
            "inner_best_kappa": selected_fit.best_metric,
            "inner_best_accuracy": selected_fit.best_accuracy,
            "accuracy": evaluation["accuracy"],
            "kappa": evaluation["kappa"],
            "mean_firing_rate": evaluation["mean_firing_rate"],
            "decoder_batch_latency_ms": _latency_ms(
                outer_fit.model,
                standardized["outer_test"][0],
                standardized["outer_test"][1],
                device=args.device,
            ),
            "decoder_parameters": capacities[model_name],
            "end_to_end_parameters": capacities[model_name] + 125_544,
            "anchor_snn_accuracy": float(
                np.mean(anchor_snn_logits.argmax(axis=1) == labels[outer_test])
            ),
            "session_e_accessed": False,
        }
        summary_rows.append(row)
        endpoint_seconds = (
            [4.0]
            if evaluation["endpoint_logits"].shape[1] == 1
            else [1.0, 2.0, 3.0, 4.0]
        )
        for endpoint_index, seconds in enumerate(endpoint_seconds):
            endpoint_prediction = evaluation["endpoint_logits"][:, endpoint_index].argmax(
                axis=1
            )
            endpoint_rows.append(
                {
                    "model": model_name,
                    "variant": variant,
                    "decoder_kind": decoder_kind,
                    "subject": int(args.subject),
                    "seed": int(args.seed),
                    "fold": int(args.fold),
                    "endpoint_seconds": seconds,
                    "accuracy": float(
                        np.mean(endpoint_prediction == labels[outer_test])
                    ),
                }
            )
        for perturbation, (perturbed_atc, perturbed_fbc) in perturbations.items():
            perturbed = predict_v12(
                outer_fit.model,
                perturbed_atc,
                perturbed_fbc,
                labels[outer_test],
                teachers("outer_test")[0],
                teachers("outer_test")[1],
                teachers("outer_test")[2],
                device=args.device,
                batch_size=int(args.batch_size),
            )
            robustness_rows.append(
                {
                    "model": model_name,
                    "variant": variant,
                    "decoder_kind": decoder_kind,
                    "subject": int(args.subject),
                    "seed": int(args.seed),
                    "fold": int(args.fold),
                    "perturbation": perturbation,
                    "accuracy": float(perturbed["accuracy"]),
                    "kappa": float(perturbed["kappa"]),
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
            labels=labels[outer_test],
        )
        del selected_fit, outer_fit, evaluation, candidates
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    write_csv(output / "hpo_results.csv", hpo_rows)
    write_csv(output / "summary.csv", summary_rows)
    write_csv(output / "endpoint_metrics.csv", endpoint_rows)
    write_csv(output / "robustness_metrics.csv", robustness_rows)
    status = {
        "status": "completed",
        "stage": "V12-E1-bounded-multirate-development-fold",
        "subject": int(args.subject),
        "seed": int(args.seed),
        "fold": int(args.fold),
        "variants": variants,
        "decoder_kinds": decoder_kinds,
        "model_specs": model_specs,
        "total_hpo_configurations": len(grid) * len(model_specs),
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
