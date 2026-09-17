#!/usr/bin/env python3
"""Run a frozen-base V13 full/zero/shuffled delay counterfactual fold."""

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
from dpc_snn.experiments.v13_delay_training import fit_v13, predict_v13  # noqa: E402
from dpc_snn.models.v13_delay_residual import V13DelayResidualStudent  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v10_e1_strong_control_fold import (  # noqa: E402
    _array_sha256,
    _csv_float,
    _environment_manifest,
    _subject_file,
)


CONTROLS = ("full", "zero", "shuffled")


def _csv_nonnegative(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or len(parsed) != len(set(parsed)) or any(item < 0.0 for item in parsed):
        raise ValueError("non-negative grid must be non-empty and unique")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--anchor-root", required=True)
    parser.add_argument("--v12-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-variant", required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--learning-rates", default="0.0003,0.001")
    parser.add_argument("--route-l0-weights", default="0.0,0.001")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--minimum-outer-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    configure_cache_env()
    data_root = Path(args.data).resolve()
    anchor_root = Path(args.anchor_root).resolve()
    v12_root = Path(args.v12_root).resolve()
    output = Path(args.output).resolve()
    learning_rates = _csv_float(args.learning_rates)
    l0_weights = _csv_nonnegative(args.route_l0_weights)
    if args.subject not in range(1, 10) or args.fold not in range(6):
        raise ValueError("invalid V13 subject or fold")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    grid = [
        {"learning_rate": learning_rate, "route_l0_weight": l0_weight}
        for learning_rate, l0_weight in itertools.product(learning_rates, l0_weights)
    ]
    if len(grid) > 4:
        raise RuntimeError("V13 delay search exceeds four configurations")

    anchor_fold = (
        anchor_root
        / f"subject_{args.subject:02d}"
        / f"seed_{args.seed}"
        / f"fold_{args.fold}"
    )
    v12_fold = (
        v12_root
        / f"subject_{args.subject:02d}"
        / f"seed_{args.seed}"
        / f"fold_{args.fold}"
    )
    v12_status = read_json(v12_fold / "campaign_status.json")
    if v12_status.get("status") != "completed" or v12_status.get(
        "session_e_accessed"
    ) is not False:
        raise RuntimeError("selected V12 fold is incomplete or unlocked")
    if args.base_variant not in set(v12_status["variants"]):
        raise RuntimeError("selected V12 variant is absent from the fold")

    cache_path = anchor_fold / "frozen_dual_feature_cache.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        arrays = {name: np.asarray(cache[name]) for name in cache.files}
    subject_path = _subject_file(data_root, int(args.subject))
    data = load_processed_npz(subject_path)
    _, labels, _, access = session_t_development_view(data)
    inner_train = arrays["inner_train_indices"].astype(np.int64)
    inner_validation = arrays["inner_validation_indices"].astype(np.int64)
    outer_train = arrays["outer_train_indices"].astype(np.int64)
    outer_test = arrays["outer_test_indices"].astype(np.int64)
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
    selection_state_path = v12_fold / args.base_variant / "selection_best.pt"
    outer_state_path = v12_fold / args.base_variant / "outer_last.pt"
    selection_state = torch.load(selection_state_path, map_location="cpu", weights_only=True)
    outer_state = torch.load(outer_state_path, map_location="cpu", weights_only=True)

    source_tree = collect_source_tree_manifest(ROOT)
    prototype = V13DelayResidualStudent(base_variant=args.base_variant)
    resolved_config = {
        **vars(args),
        "data": str(data_root),
        "anchor_root": str(anchor_root),
        "v12_root": str(v12_root),
        "output": str(output),
        "learning_rates": learning_rates,
        "route_l0_weights": l0_weights,
        "hpo_grid": grid,
        "controls": list(CONTROLS),
        "control_semantics": "one checkpoint; only lag posterior is intervened",
        "base_frozen": True,
        "delay_parameters": prototype.delay_parameter_count,
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
        prior={"enabled": False, "route_l0_is_upper_penalty": True},
        checkpoint={
            "v12_selection_state_sha256": file_sha256(selection_state_path),
            "v12_outer_state_sha256": file_sha256(outer_state_path),
            "anchor_cache_sha256": file_sha256(cache_path),
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
    write_v8_fingerprint(output / "run_fingerprint.json", fingerprint)
    (output / "resolved_run.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
    )

    started = time.time()
    hpo_rows: list[dict[str, Any]] = []
    candidates: list[tuple[tuple[float, float, int, int], dict[str, float], Any]] = []
    fold_seed = int(args.seed) * 100_003 + int(args.fold) * 1_009 + 13_100_019
    for grid_index, configuration in enumerate(grid):
        fit = fit_v13(
            base_variant=args.base_variant,
            base_state=selection_state,
            atc_train=standardized["selection_train"][0],
            fbc_train=standardized["selection_train"][1],
            y_train=labels[inner_train],
            teacher_train=arrays["teacher_selection_train"],
            atc_validation=standardized["selection_validation"][0],
            fbc_validation=standardized["selection_validation"][1],
            y_validation=labels[inner_validation],
            teacher_validation=arrays["teacher_selection_validation"],
            device=args.device,
            seed=fold_seed,
            epochs=int(args.epochs),
            patience=int(args.patience),
            batch_size=int(args.batch_size),
            learning_rate=float(configuration["learning_rate"]),
            route_l0_weight=float(configuration["route_l0_weight"]),
            run_label=(
                f"V13-E1-select:grid{grid_index}:S{args.subject}:"
                f"seed{args.seed}:fold{args.fold}"
            ),
        )
        hpo_rows.append(
            {
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
    selected_grid = grid.index(selected_configuration)
    for _, _, candidate_fit in candidates[1:]:
        del candidate_fit
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    outer_fit = fit_v13(
        base_variant=args.base_variant,
        base_state=outer_state,
        atc_train=standardized["outer_train"][0],
        fbc_train=standardized["outer_train"][1],
        y_train=labels[outer_train],
        teacher_train=arrays["teacher_outer_train"],
        atc_validation=None,
        fbc_validation=None,
        y_validation=None,
        teacher_validation=None,
        device=args.device,
        seed=fold_seed + 1_000_003,
        epochs=int(args.epochs),
        fixed_epoch=selected_epoch,
        scheduler_epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(selected_configuration["learning_rate"]),
        route_l0_weight=float(selected_configuration["route_l0_weight"]),
        run_label=(
            f"V13-E1-outer:S{args.subject}:seed{args.seed}:fold{args.fold}"
        ),
    )
    summary_rows: list[dict[str, Any]] = []
    for control in CONTROLS:
        evaluation = predict_v13(
            outer_fit.model,
            standardized["outer_test"][0],
            standardized["outer_test"][1],
            labels[outer_test],
            arrays["teacher_outer_test"],
            control=control,
            device=args.device,
            batch_size=int(args.batch_size),
        )
        control_dir = ensure_dir(output / control)
        row = {
            "control": control,
            "subject": int(args.subject),
            "seed": int(args.seed),
            "fold": int(args.fold),
            "accuracy": evaluation["accuracy"],
            "kappa": evaluation["kappa"],
            "mean_firing_rate": evaluation["mean_firing_rate"],
            "mean_nonzero_delay_mass": evaluation["mean_nonzero_delay_mass"],
            "mean_delay_residual_rms": evaluation["mean_delay_residual_rms"],
            "selected_grid_index": selected_grid,
            "selected_learning_rate": float(selected_configuration["learning_rate"]),
            "selected_route_l0_weight": float(
                selected_configuration["route_l0_weight"]
            ),
            "selected_epoch": selected_epoch,
            "session_e_accessed": False,
        }
        summary_rows.append(row)
        write_json(control_dir / "metrics.json", row)
        np.savez_compressed(
            control_dir / "outer_predictions.npz",
            indices=outer_test,
            logits=np.asarray(evaluation["logits"], dtype=np.float32),
            endpoint_logits=np.asarray(evaluation["endpoint_logits"], dtype=np.float32),
            labels=labels[outer_test],
        )
        if control == "full":
            np.savez_compressed(
                output / "delay_diagnostics.npz",
                route_probability=evaluation["route_probability"],
                lag_posterior=evaluation["lag_posterior"],
            )

    torch.save(selected_fit.best_state, output / "selection_best.pt")
    torch.save(outer_fit.last_state, output / "outer_last.pt")
    write_csv(output / "selection_history.csv", selected_fit.history)
    write_csv(output / "outer_history.csv", outer_fit.history)
    write_csv(output / "hpo_results.csv", hpo_rows)
    write_csv(output / "summary.csv", summary_rows)
    status = {
        "status": "completed",
        "stage": "V13-E1-frozen-base-delay-counterfactual-fold",
        "subject": int(args.subject),
        "seed": int(args.seed),
        "fold": int(args.fold),
        "base_variant": args.base_variant,
        "controls": list(CONTROLS),
        "same_checkpoint_for_all_controls": True,
        "only_lag_posterior_intervened": True,
        "hpo_configurations": len(grid),
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
