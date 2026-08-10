#!/usr/bin/env python3
"""Evaluate the E19 CE to equal-teacher-KD objective transition."""

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
    V9_DUAL_FEATURE_EXPERIMENT_VARIANTS,
    fit_feature_standardizer,
    predict_v9_dual_feature,
)
from dpc_snn.experiments.v17_information_replay import (  # noqa: E402
    residual_fusion_logits,
    select_residual_scale,
)
from dpc_snn.experiments.v62_protocol import file_sha256  # noqa: E402
from dpc_snn.models.v9_dual_feature_student import build_v9_dual_feature_student  # noqa: E402
from dpc_snn.utils.io import ensure_dir, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from scripts.run_v8_e1_baselines import _subject_file  # noqa: E402


VARIANTS = ("sew_clif_ce", "sew_clif_kd")


def _metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    return classification_metrics(labels, logits.argmax(axis=1), n_classes=4)


def _load_model(variant: str, checkpoint: Path, device: str) -> torch.nn.Module:
    specification = V9_DUAL_FEATURE_EXPERIMENT_VARIANTS[variant]
    model = build_v9_dual_feature_student(specification.model_variant)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return model.eval().to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--v9-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subjects", default="1,3,8")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    data_root = Path(args.data).resolve()
    v9_root = Path(args.v9_root).resolve()
    output = Path(args.output).resolve()
    subjects = [int(item) for item in args.subjects.split(",") if item.strip()]
    if output.exists():
        raise FileExistsError(f"E19 output must be immutable and new: {output}")
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

    fold_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    input_manifest: dict[str, str] = {}
    parts: dict[tuple[int, str], list[np.ndarray]] = {}
    labels_by_subject: dict[int, list[np.ndarray]] = {}
    indices_by_subject: dict[int, list[np.ndarray]] = {}
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
            raise RuntimeError("E19 must not access Session E")
        input_manifest[f"subject_{subject:02d}_data"] = file_sha256(data_path)
        labels_by_subject[subject] = []
        indices_by_subject[subject] = []
        for variant in VARIANTS:
            parts[(subject, variant)] = []
        for fold in range(6):
            fold_dir = v9_root / f"subject_{subject:02d}" / f"seed_{args.seed}" / f"fold_{fold}"
            cache_path = fold_dir / "frozen_dual_feature_cache.npz"
            with np.load(cache_path, allow_pickle=False) as archive:
                cache = {name: archive[name] for name in archive.files}
            input_manifest[f"subject_{subject:02d}_fold_{fold}_cache"] = file_sha256(cache_path)
            selection_validation = cache["inner_validation_indices"].astype(np.int64)
            outer_test = cache["outer_test_indices"].astype(np.int64)
            selection_standardizer = fit_feature_standardizer(
                cache["atcnet_selection_train_sequence"],
                cache["fbcnet_selection_train_sequence"],
            )
            outer_standardizer = fit_feature_standardizer(
                cache["atcnet_outer_train_sequence"], cache["fbcnet_outer_train_sequence"]
            )
            validation_atc, validation_fbc = selection_standardizer.transform(
                cache["atcnet_selection_validation_sequence"],
                cache["fbcnet_selection_validation_sequence"],
            )
            test_atc, test_fbc = outer_standardizer.transform(
                cache["atcnet_outer_test_sequence"], cache["fbcnet_outer_test_sequence"]
            )
            validation_y = labels[selection_validation]
            test_y = labels[outer_test]
            validation_base = np.asarray(cache["teacher_selection_validation"], dtype=np.float32)
            test_base = np.asarray(cache["teacher_outer_test"], dtype=np.float32)
            labels_by_subject[subject].append(test_y)
            indices_by_subject[subject].append(outer_test)
            prediction_archive: dict[str, np.ndarray] = {}
            for variant in VARIANTS:
                variant_dir = fold_dir / variant
                selection_checkpoint = variant_dir / "selection_best.pt"
                outer_checkpoint = variant_dir / "outer_last.pt"
                input_manifest[
                    f"subject_{subject:02d}_fold_{fold}_{variant}_selection"
                ] = file_sha256(selection_checkpoint)
                input_manifest[f"subject_{subject:02d}_fold_{fold}_{variant}_outer"] = (
                    file_sha256(outer_checkpoint)
                )
                selection_model = _load_model(variant, selection_checkpoint, args.device)
                validation = predict_v9_dual_feature(
                    selection_model,
                    validation_atc,
                    validation_fbc,
                    validation_y,
                    validation_base,
                    device=args.device,
                    batch_size=64,
                )
                scale, rows = select_residual_scale(
                    validation_base,
                    validation["logits"],
                    validation_y,
                    scales=(0.0, 0.25, 0.5, 1.0),
                )
                selected_row = next(row for row in rows if row["scale"] == scale)
                for row in rows:
                    search_rows.append(
                        {
                            "subject": subject,
                            "fold": fold,
                            "variant": variant,
                            **row,
                            "selected": row["scale"] == scale,
                        }
                    )
                with np.load(variant_dir / "outer_predictions.npz", allow_pickle=False) as archive:
                    if (
                        not np.array_equal(archive["indices"], outer_test)
                        or not np.array_equal(archive["labels"], test_y)
                    ):
                        raise RuntimeError("V9 objective predictions are not aligned")
                    saved_correction = np.asarray(archive["logits"], dtype=np.float32)
                outer_model = _load_model(variant, outer_checkpoint, args.device)
                replay = predict_v9_dual_feature(
                    outer_model,
                    test_atc,
                    test_fbc,
                    test_y,
                    test_base,
                    device=args.device,
                    batch_size=64,
                )
                replay_error = float(np.max(np.abs(replay["logits"] - saved_correction)))
                if replay_error >= 1e-5:
                    raise RuntimeError(f"outer objective checkpoint replay failed for {variant}")
                logits = residual_fusion_logits(test_base, saved_correction, scale)
                metrics = _metrics(test_y, logits)
                fold_rows.append(
                    {
                        "subject": subject,
                        "seed": int(args.seed),
                        "fold": fold,
                        "variant": variant,
                        "selected_scale": scale,
                        "inner_validation_accuracy": selected_row["validation_accuracy"],
                        "inner_validation_kappa": selected_row["validation_kappa"],
                        "outer_accuracy": metrics["accuracy"],
                        "outer_kappa": metrics["kappa"],
                        "mean_firing_rate": replay["mean_firing_rate"],
                        "checkpoint_replay_error": replay_error,
                        "session_e_accessed": False,
                    }
                )
                parts[(subject, variant)].append(logits)
                prediction_archive[f"{variant}_logits"] = logits
                del selection_model, outer_model
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            np.savez_compressed(
                output / f"subject_{subject:02d}_fold_{fold}_predictions.npz",
                indices=outer_test,
                labels=test_y,
                **prediction_archive,
            )

    subject_rows: list[dict[str, Any]] = []
    for subject in subjects:
        all_indices = np.concatenate(indices_by_subject[subject])
        all_labels = np.concatenate(labels_by_subject[subject])
        if len(all_indices) != 288 or len(np.unique(all_indices)) != 288:
            raise RuntimeError(f"subject {subject} lacks exact six-fold coverage")
        for variant in VARIANTS:
            metrics = _metrics(all_labels, np.concatenate(parts[(subject, variant)]))
            rows = [
                row
                for row in fold_rows
                if row["subject"] == subject and row["variant"] == variant
            ]
            subject_rows.append(
                {
                    "subject": subject,
                    "seed": int(args.seed),
                    "variant": variant,
                    "accuracy": metrics["accuracy"],
                    "kappa": metrics["kappa"],
                    "nonzero_scale_folds": sum(row["selected_scale"] > 0.0 for row in rows),
                    "mean_firing_rate": float(
                        np.mean([row["mean_firing_rate"] for row in rows])
                    ),
                }
            )
    summary_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        rows = [row for row in subject_rows if row["variant"] == variant]
        inner = [row for row in fold_rows if row["variant"] == variant]
        summary_rows.append(
            {
                "variant": variant,
                "mean_inner_validation_kappa": float(
                    np.mean([row["inner_validation_kappa"] for row in inner])
                ),
                "mean_accuracy": float(np.mean([row["accuracy"] for row in rows])),
                "mean_kappa": float(np.mean([row["kappa"] for row in rows])),
                "nonzero_scale_folds": int(
                    sum(row["nonzero_scale_folds"] for row in rows)
                ),
                "mean_firing_rate": float(np.mean([row["mean_firing_rate"] for row in rows])),
            }
        )
    ce = next(row for row in summary_rows if row["variant"] == "sew_clif_ce")
    kd = next(row for row in summary_rows if row["variant"] == "sew_clif_kd")
    kd_inner_pass = kd["mean_inner_validation_kappa"] >= ce["mean_inner_validation_kappa"]
    decision = {
        "status": "completed",
        "stage": "E19-CE-to-KD",
        "models": {row["variant"]: row for row in summary_rows},
        "kd_minus_ce_pp": 100.0 * (kd["mean_accuracy"] - ce["mean_accuracy"]),
        "kd_inner_promotion_pass": kd_inner_pass,
        "next_stage": "E19-onset-aware-TET" if kd_inner_pass else "retain-CE-and-stop-KD-branch",
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    write_csv(output / "scale_search.csv", search_rows)
    write_csv(output / "fold_summary.csv", fold_rows)
    write_csv(output / "subject_summary.csv", subject_rows)
    write_csv(output / "model_summary.csv", summary_rows)
    write_json(output / "input_manifest.json", input_manifest)
    write_json(output / "decision.json", decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
