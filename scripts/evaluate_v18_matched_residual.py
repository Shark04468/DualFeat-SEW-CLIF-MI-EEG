#!/usr/bin/env python3
"""Evaluate matched ANN/SNN corrections on the E17 logit-preserving base."""

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
from dpc_snn.models.v9_dual_feature_student import (  # noqa: E402
    build_v9_dual_feature_student,
)
from dpc_snn.utils.io import ensure_dir, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from scripts.run_v8_e1_baselines import _subject_file  # noqa: E402


VARIANTS = ("ann_plain_ce", "plif_plain_ce", "clif_plain_ce", "sew_clif_ce")


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
    parser.add_argument("--noise-std", type=float, default=0.10)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--reference-accuracy", type=float, default=0.8622685185185185)
    args = parser.parse_args()

    data_root = Path(args.data).resolve()
    v9_root = Path(args.v9_root).resolve()
    output = Path(args.output).resolve()
    subjects = [int(item) for item in args.subjects.split(",") if item.strip()]
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    unknown_variants = sorted(set(variants) - set(V9_DUAL_FEATURE_EXPERIMENT_VARIANTS))
    snn_variants = tuple(variant for variant in variants if variant != "ann_plain_ce")
    if (
        unknown_variants
        or len(variants) != len(set(variants))
        or "ann_plain_ce" not in variants
        or not snn_variants
    ):
        raise ValueError(
            "variants must contain ann_plain_ce and at least one unique SNN variant; "
            f"unknown={unknown_variants}, received={variants}"
        )
    if output.exists():
        raise FileExistsError(f"E18 output must be immutable and new: {output}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.noise_std <= 0.0:
        raise ValueError("noise_std must be positive")
    ensure_dir(output)
    write_json(output / "heldout_lock_manifest.json", v8_heldout_lock_manifest())
    source_tree = collect_source_tree_manifest(ROOT)
    write_json(output / "source_tree_manifest.json", source_tree)
    write_json(
        output / "source_tree_summary.json",
        {"files": len(source_tree), "sha256": source_tree_digest(source_tree)},
    )

    capacity: dict[str, int] = {}
    for variant in variants:
        specification = V9_DUAL_FEATURE_EXPERIMENT_VARIANTS[variant]
        capacity[variant] = build_v9_dual_feature_student(
            specification.model_variant
        ).parameter_count
    if len(set(capacity.values())) != 1:
        raise RuntimeError(f"E18 variants are not parameter matched: {capacity}")

    fold_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    input_manifest: dict[str, str] = {}
    parts: dict[tuple[int, str, str], list[np.ndarray]] = {}
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
            raise RuntimeError("E18 must not access Session E")
        input_manifest[f"subject_{subject:02d}_data"] = file_sha256(data_path)
        labels_by_subject[subject] = []
        indices_by_subject[subject] = []
        for variant in variants:
            parts[(subject, variant, "clean")] = []
            parts[(subject, variant, "noise")] = []

        for fold in range(6):
            fold_dir = v9_root / f"subject_{subject:02d}" / f"seed_{args.seed}" / f"fold_{fold}"
            cache_path = fold_dir / "frozen_dual_feature_cache.npz"
            with np.load(cache_path, allow_pickle=False) as archive:
                cache = {name: archive[name] for name in archive.files}
            input_manifest[f"subject_{subject:02d}_fold_{fold}_cache"] = file_sha256(cache_path)
            split = {
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
            validation_atc, validation_fbc = selection_standardizer.transform(
                cache["atcnet_selection_validation_sequence"],
                cache["fbcnet_selection_validation_sequence"],
            )
            test_atc, test_fbc = outer_standardizer.transform(
                cache["atcnet_outer_test_sequence"], cache["fbcnet_outer_test_sequence"]
            )
            validation_y = labels[split["selection_validation"]]
            test_y = labels[split["outer_test"]]
            validation_base = np.asarray(cache["teacher_selection_validation"], dtype=np.float32)
            test_base = np.asarray(cache["teacher_outer_test"], dtype=np.float32)
            labels_by_subject[subject].append(test_y)
            indices_by_subject[subject].append(split["outer_test"])
            fold_predictions: dict[str, np.ndarray] = {}
            for variant_index, variant in enumerate(variants):
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
                selected_scale, rows = select_residual_scale(
                    validation_base,
                    validation["logits"],
                    validation_y,
                    scales=(0.0, 0.25, 0.5, 1.0),
                )
                selected_row = next(row for row in rows if row["scale"] == selected_scale)
                for row in rows:
                    scale_rows.append(
                        {
                            "subject": subject,
                            "fold": fold,
                            "variant": variant,
                            **row,
                            "selected": row["scale"] == selected_scale,
                        }
                    )
                with np.load(variant_dir / "outer_predictions.npz", allow_pickle=False) as archive:
                    if (
                        not np.array_equal(archive["indices"], split["outer_test"])
                        or not np.array_equal(archive["labels"], test_y)
                    ):
                        raise RuntimeError("saved V9 outer predictions are not aligned")
                    saved_correction = np.asarray(archive["logits"], dtype=np.float32)
                outer_model = _load_model(variant, outer_checkpoint, args.device)
                clean_replay = predict_v9_dual_feature(
                    outer_model,
                    test_atc,
                    test_fbc,
                    test_y,
                    test_base,
                    device=args.device,
                    batch_size=64,
                )
                replay_error = float(
                    np.max(np.abs(clean_replay["logits"] - saved_correction))
                )
                if replay_error >= 1e-5:
                    raise RuntimeError(f"V9 outer checkpoint replay failed for {variant}")
                clean_logits = residual_fusion_logits(
                    test_base, saved_correction, selected_scale
                )
                rng = np.random.default_rng(
                    int(args.seed) * 100_003 + subject * 10_007 + fold * 101 + variant_index
                )
                noisy_atc = test_atc + rng.normal(
                    0.0, float(args.noise_std), size=test_atc.shape
                ).astype(np.float32)
                noisy_fbc = test_fbc + rng.normal(
                    0.0, float(args.noise_std), size=test_fbc.shape
                ).astype(np.float32)
                noisy = predict_v9_dual_feature(
                    outer_model,
                    noisy_atc,
                    noisy_fbc,
                    test_y,
                    test_base,
                    device=args.device,
                    batch_size=64,
                )
                noisy_logits = residual_fusion_logits(
                    test_base, noisy["logits"], selected_scale
                )
                clean_metrics = _metrics(test_y, clean_logits)
                noisy_metrics = _metrics(test_y, noisy_logits)
                event_density = (
                    1.0
                    if variant == "ann_plain_ce"
                    else float(clean_replay["mean_firing_rate"])
                )
                fold_rows.append(
                    {
                        "subject": subject,
                        "seed": int(args.seed),
                        "fold": fold,
                        "variant": variant,
                        "selected_scale": selected_scale,
                        "inner_validation_accuracy": selected_row["validation_accuracy"],
                        "inner_validation_kappa": selected_row["validation_kappa"],
                        "clean_accuracy": clean_metrics["accuracy"],
                        "clean_kappa": clean_metrics["kappa"],
                        "noise_accuracy": noisy_metrics["accuracy"],
                        "noise_kappa": noisy_metrics["kappa"],
                        "estimated_event_density": event_density,
                        "checkpoint_replay_error": replay_error,
                        "parameters": capacity[variant],
                        "session_e_accessed": False,
                    }
                )
                parts[(subject, variant, "clean")].append(clean_logits)
                parts[(subject, variant, "noise")].append(noisy_logits)
                fold_predictions[f"{variant}_clean"] = clean_logits
                fold_predictions[f"{variant}_noise"] = noisy_logits
                del selection_model, outer_model
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            np.savez_compressed(
                output / f"subject_{subject:02d}_fold_{fold}_predictions.npz",
                indices=split["outer_test"],
                labels=test_y,
                **{f"{name}_logits": value for name, value in fold_predictions.items()},
            )

    subject_rows: list[dict[str, Any]] = []
    for subject in subjects:
        all_indices = np.concatenate(indices_by_subject[subject])
        all_labels = np.concatenate(labels_by_subject[subject])
        if len(all_indices) != 288 or len(np.unique(all_indices)) != 288:
            raise RuntimeError(f"subject {subject} lacks exact six-fold coverage")
        for variant in variants:
            clean = _metrics(all_labels, np.concatenate(parts[(subject, variant, "clean")]))
            noise = _metrics(all_labels, np.concatenate(parts[(subject, variant, "noise")]))
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
                    "clean_accuracy": clean["accuracy"],
                    "clean_kappa": clean["kappa"],
                    "noise_accuracy": noise["accuracy"],
                    "noise_kappa": noise["kappa"],
                    "mean_event_density": float(
                        np.mean([row["estimated_event_density"] for row in rows])
                    ),
                    "nonzero_scale_folds": sum(row["selected_scale"] > 0.0 for row in rows),
                }
            )
    summary_rows: list[dict[str, Any]] = []
    for variant in variants:
        rows = [row for row in subject_rows if row["variant"] == variant]
        inner_rows = [row for row in fold_rows if row["variant"] == variant]
        summary_rows.append(
            {
                "variant": variant,
                "mean_inner_validation_kappa": float(
                    np.mean([row["inner_validation_kappa"] for row in inner_rows])
                ),
                "mean_clean_accuracy": float(np.mean([row["clean_accuracy"] for row in rows])),
                "mean_noise_accuracy": float(np.mean([row["noise_accuracy"] for row in rows])),
                "mean_event_density": float(
                    np.mean([row["mean_event_density"] for row in rows])
                ),
                "nonzero_scale_folds": int(
                    sum(row["nonzero_scale_folds"] for row in rows)
                ),
                "parameters": capacity[variant],
            }
        )
    ann = next(row for row in summary_rows if row["variant"] == "ann_plain_ce")
    selected_snn = max(
        (row for row in summary_rows if row["variant"] in snn_variants),
        key=lambda row: (row["mean_inner_validation_kappa"], -variants.index(row["variant"])),
    )
    v9_reference = float(args.reference_accuracy)
    accuracy_match = selected_snn["mean_clean_accuracy"] - ann["mean_clean_accuracy"] >= -0.003
    v9_pass = selected_snn["mean_clean_accuracy"] - v9_reference >= 0.005
    sparsity_pass = selected_snn["mean_event_density"] <= 0.5
    robustness_pass = selected_snn["mean_noise_accuracy"] > ann["mean_noise_accuracy"]
    residual_active = selected_snn["nonzero_scale_folds"] >= 6
    decision = {
        "status": "completed",
        "stage": "E18-matched-logit-preserving-ANN-SNN",
        "selected_snn_by_inner_validation": selected_snn["variant"],
        "models": {row["variant"]: row for row in summary_rows},
        "snn_minus_ann_pp": 100.0
        * (selected_snn["mean_clean_accuracy"] - ann["mean_clean_accuracy"]),
        "snn_minus_v9_pp": 100.0
        * (selected_snn["mean_clean_accuracy"] - v9_reference),
        "accuracy_match_pass": accuracy_match,
        "v9_pass": v9_pass,
        "sparsity_pass": sparsity_pass,
        "robustness_pass": robustness_pass,
        "residual_active_advisory_pass": residual_active,
        "gate_passed": accuracy_match
        and v9_pass
        and (sparsity_pass or robustness_pass),
        "noise_definition": f"Gaussian noise std={args.noise_std} in train-fold standardized feature space",
        "event_density_definition": "ANN dense operations=1; SNN mean binary firing rate",
        "advisory_warning": (
            "Residual correction was selected in fewer than 6/18 folds; accuracy is "
            "therefore dominated by the logit-preserving base. This is not a preregistered "
            "E18 hard gate but must be improved before claiming SNN-specific accuracy gain."
            if not residual_active
            else None
        ),
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    decision["next_stage"] = "E19-objectives" if decision["gate_passed"] else "stop"
    write_csv(output / "scale_search.csv", scale_rows)
    write_csv(output / "fold_summary.csv", fold_rows)
    write_csv(output / "subject_summary.csv", subject_rows)
    write_csv(output / "model_summary.csv", summary_rows)
    write_json(output / "capacity_audit.json", capacity)
    write_json(output / "input_manifest.json", input_manifest)
    write_json(output / "decision.json", decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
