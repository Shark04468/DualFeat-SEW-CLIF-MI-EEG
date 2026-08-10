#!/usr/bin/env python3
"""Screen matched causal latent delays across three development seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v8_baselines import session_t_development_view  # noqa: E402
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    source_tree_digest,
    v8_heldout_lock_manifest,
)
from dpc_snn.experiments.v9_dual_feature_training import (  # noqa: E402
    fit_feature_standardizer,
    predict_v9_dual_feature,
)
from dpc_snn.experiments.v17_information_replay import residual_fusion_logits  # noqa: E402
from dpc_snn.experiments.v21_delay_screen import (  # noqa: E402
    causal_shift,
    select_lag_scale,
)
from dpc_snn.experiments.v62_protocol import file_sha256  # noqa: E402
from dpc_snn.models.v9_dual_feature_student import build_v9_dual_feature_student  # noqa: E402
from dpc_snn.utils.io import ensure_dir, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from scripts.run_v8_e1_baselines import _subject_file  # noqa: E402


def _metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    return classification_metrics(labels, logits.argmax(axis=1), n_classes=4)


def _load_model(checkpoint: Path, device: str) -> torch.nn.Module:
    model = build_v9_dual_feature_student("sew_clif")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return model.eval().to(device)


def _correction(
    model: torch.nn.Module,
    atc: np.ndarray,
    fbc: np.ndarray,
    labels: np.ndarray,
    teacher: np.ndarray,
    *,
    lag: int,
    device: str,
) -> np.ndarray:
    prediction = predict_v9_dual_feature(
        model,
        causal_shift(atc, lag),
        fbc,
        labels,
        teacher,
        device=device,
        batch_size=64,
    )
    return np.asarray(prediction["logits"], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--v9-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subjects", default="1,3,8")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    data_root = Path(args.data).resolve()
    v9_root = Path(args.v9_root).resolve()
    output = Path(args.output).resolve()
    subjects = [int(item) for item in args.subjects.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if output.exists():
        raise FileExistsError(f"E21-0 output must be immutable and new: {output}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    ensure_dir(output)
    write_json(output / "heldout_lock_manifest.json", v8_heldout_lock_manifest())
    source_tree = collect_source_tree_manifest(ROOT)
    write_json(output / "source_tree_manifest.json", source_tree)
    write_json(
        output / "source_tree_summary.json",
        {"files": len(source_tree), "sha256": source_tree_digest(source_tree)},
    )

    lags = (0, 1, 2, 4)
    scales = (0.0, 0.5, 1.0)
    fold_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    input_manifest: dict[str, str] = {}
    pair_parts: dict[tuple[int, int, str], list[np.ndarray]] = {}
    pair_labels: dict[tuple[int, int], list[np.ndarray]] = {}
    pair_indices: dict[tuple[int, int], list[np.ndarray]] = {}
    for subject in subjects:
        data_path = _subject_file(data_root, subject)
        data = load_processed_npz(data_path)
        _, labels, _, access = session_t_development_view(data)
        labels = np.asarray(labels, dtype=np.int64)
        if (
            access.get("selected_session") != "T"
            or access.get("heldout_signals_used_by_development") is not False
            or access.get("heldout_labels_used_by_development") is not False
        ):
            raise RuntimeError("E21-0 must not access Session E")
        input_manifest[f"subject_{subject:02d}_data"] = file_sha256(data_path)
        for seed in seeds:
            pair = (subject, seed)
            pair_labels[pair] = []
            pair_indices[pair] = []
            pair_parts[(subject, seed, "full")] = []
            pair_parts[(subject, seed, "zero")] = []
            for fold in range(6):
                fold_dir = v9_root / f"subject_{subject:02d}" / f"seed_{seed}" / f"fold_{fold}"
                cache_path = fold_dir / "frozen_dual_feature_cache.npz"
                with np.load(cache_path, allow_pickle=False) as archive:
                    cache = {name: archive[name] for name in archive.files}
                input_manifest[f"subject_{subject:02d}_seed_{seed}_fold_{fold}_cache"] = (
                    file_sha256(cache_path)
                )
                validation_indices = cache["inner_validation_indices"].astype(np.int64)
                test_indices = cache["outer_test_indices"].astype(np.int64)
                selection_standardizer = fit_feature_standardizer(
                    cache["atcnet_selection_train_sequence"],
                    cache["fbcnet_selection_train_sequence"],
                )
                outer_standardizer = fit_feature_standardizer(
                    cache["atcnet_outer_train_sequence"],
                    cache["fbcnet_outer_train_sequence"],
                )
                validation_atc, validation_fbc = selection_standardizer.transform(
                    cache["atcnet_selection_validation_sequence"],
                    cache["fbcnet_selection_validation_sequence"],
                )
                test_atc, test_fbc = outer_standardizer.transform(
                    cache["atcnet_outer_test_sequence"], cache["fbcnet_outer_test_sequence"]
                )
                validation_y = labels[validation_indices]
                test_y = labels[test_indices]
                validation_base = np.asarray(
                    cache["teacher_selection_validation"], dtype=np.float32
                )
                test_base = np.asarray(cache["teacher_outer_test"], dtype=np.float32)
                variant_dir = fold_dir / "sew_clif_ce"
                selection_checkpoint = variant_dir / "selection_best.pt"
                outer_checkpoint = variant_dir / "outer_last.pt"
                input_manifest[
                    f"subject_{subject:02d}_seed_{seed}_fold_{fold}_selection"
                ] = file_sha256(selection_checkpoint)
                input_manifest[
                    f"subject_{subject:02d}_seed_{seed}_fold_{fold}_outer"
                ] = file_sha256(outer_checkpoint)

                selection_model = _load_model(selection_checkpoint, args.device)
                validation_corrections = {
                    lag: _correction(
                        selection_model,
                        validation_atc,
                        validation_fbc,
                        validation_y,
                        validation_base,
                        lag=lag,
                        device=args.device,
                    )
                    for lag in lags
                }
                selected_lag, selected_scale, rows = select_lag_scale(
                    validation_base,
                    validation_corrections,
                    validation_y,
                    lags=lags,
                    scales=scales,
                )
                for row in rows:
                    search_rows.append(
                        {
                            "subject": subject,
                            "seed": seed,
                            "fold": fold,
                            **row,
                            "selected": row["lag"] == selected_lag
                            and row["scale"] == selected_scale,
                        }
                    )
                outer_model = _load_model(outer_checkpoint, args.device)
                full_correction = _correction(
                    outer_model,
                    test_atc,
                    test_fbc,
                    test_y,
                    test_base,
                    lag=selected_lag,
                    device=args.device,
                )
                zero_correction = _correction(
                    outer_model,
                    test_atc,
                    test_fbc,
                    test_y,
                    test_base,
                    lag=0,
                    device=args.device,
                )
                full_logits = residual_fusion_logits(
                    test_base, full_correction, selected_scale
                )
                zero_logits = residual_fusion_logits(
                    test_base, zero_correction, selected_scale
                )
                full_metrics = _metrics(test_y, full_logits)
                zero_metrics = _metrics(test_y, zero_logits)
                fold_rows.append(
                    {
                        "subject": subject,
                        "seed": seed,
                        "fold": fold,
                        "selected_lag": selected_lag,
                        "selected_scale": selected_scale,
                        "full_accuracy": full_metrics["accuracy"],
                        "zero_accuracy": zero_metrics["accuracy"],
                        "full_minus_zero_pp": 100.0
                        * (full_metrics["accuracy"] - zero_metrics["accuracy"]),
                        "session_e_accessed": False,
                    }
                )
                pair_parts[(subject, seed, "full")].append(full_logits)
                pair_parts[(subject, seed, "zero")].append(zero_logits)
                pair_labels[pair].append(test_y)
                pair_indices[pair].append(test_indices)
                np.savez_compressed(
                    output
                    / f"subject_{subject:02d}_seed_{seed}_fold_{fold}_predictions.npz",
                    indices=test_indices,
                    labels=test_y,
                    full_logits=full_logits,
                    zero_logits=zero_logits,
                )
                del selection_model, outer_model
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

    pair_rows: list[dict[str, Any]] = []
    for subject in subjects:
        for seed in seeds:
            pair = (subject, seed)
            all_indices = np.concatenate(pair_indices[pair])
            all_labels = np.concatenate(pair_labels[pair])
            if len(all_indices) != 288 or len(np.unique(all_indices)) != 288:
                raise RuntimeError(f"pair {pair} lacks exact six-fold coverage")
            full = _metrics(
                all_labels, np.concatenate(pair_parts[(subject, seed, "full")])
            )
            zero = _metrics(
                all_labels, np.concatenate(pair_parts[(subject, seed, "zero")])
            )
            rows = [
                row
                for row in fold_rows
                if row["subject"] == subject and row["seed"] == seed
            ]
            pair_rows.append(
                {
                    "subject": subject,
                    "seed": seed,
                    "full_accuracy": full["accuracy"],
                    "zero_accuracy": zero["accuracy"],
                    "full_minus_zero_pp": 100.0
                    * (full["accuracy"] - zero["accuracy"]),
                    "nonzero_lag_folds": sum(row["selected_lag"] > 0 for row in rows),
                    "active_residual_folds": sum(row["selected_scale"] > 0 for row in rows),
                }
            )
    deltas = np.asarray([row["full_minus_zero_pp"] for row in pair_rows])
    median_delta = float(np.median(deltas))
    positive_pairs = int(np.sum(deltas > 0.0))
    decision = {
        "status": "completed",
        "stage": "E21-0-causal-latent-delay-screen",
        "pairs": len(pair_rows),
        "mean_full_accuracy": float(np.mean([row["full_accuracy"] for row in pair_rows])),
        "mean_zero_accuracy": float(np.mean([row["zero_accuracy"] for row in pair_rows])),
        "median_full_minus_zero_pp": median_delta,
        "positive_pairs": positive_pairs,
        "median_gate_pass": median_delta >= 0.5,
        "positive_pair_gate_pass": positive_pairs >= 6,
        "gate_passed": median_delta >= 0.5 and positive_pairs >= 6,
        "screen_semantics": "global causal ATC latent lag; matched checkpoint and scale",
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    decision["next_stage"] = (
        "E21-sparse-low-rank-delay-training" if decision["gate_passed"] else "stop-delay-branch"
    )
    write_csv(output / "search_results.csv", search_rows)
    write_csv(output / "fold_summary.csv", fold_rows)
    write_csv(output / "pair_summary.csv", pair_rows)
    write_json(output / "input_manifest.json", input_manifest)
    write_json(output / "decision.json", decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
