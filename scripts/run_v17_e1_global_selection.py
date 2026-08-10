#!/usr/bin/env python3
"""Run the E17-1 global inner-only configuration control."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
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
    collect_source_tree_manifest,
    source_tree_digest,
    v8_heldout_lock_manifest,
)
from dpc_snn.experiments.v9_dual_feature_training import fit_feature_standardizer  # noqa: E402
from dpc_snn.experiments.v16_training import (  # noqa: E402
    fit_v16_continuous,
    predict_v16_continuous,
)
from dpc_snn.experiments.v17_global_selection import (  # noqa: E402
    select_global_configuration,
)
from dpc_snn.experiments.v62_protocol import file_sha256  # noqa: E402
from dpc_snn.utils.io import ensure_dir, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v8_e1_baselines import _subject_file  # noqa: E402


def _read_hpo(path: Path, subject: int, fold: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 8:
        raise RuntimeError(f"expected eight E16 HPO rows: {path}")
    return [{**row, "subject": subject, "fold": fold} for row in rows]


def _metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    return classification_metrics(labels, logits.argmax(axis=1), n_classes=4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--v9-root", required=True)
    parser.add_argument("--e16-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subjects", default="1,3,8")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    configure_cache_env()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_root = Path(args.data).resolve()
    v9_root = Path(args.v9_root).resolve()
    e16_root = Path(args.e16_root).resolve()
    output = Path(args.output).resolve()
    subjects = [int(item) for item in args.subjects.split(",") if item.strip()]
    if output.exists():
        raise FileExistsError(f"E17-1 output must be immutable and new: {output}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    all_hpo_rows: list[dict[str, Any]] = []
    input_manifest: dict[str, str] = {}
    for subject in subjects:
        for fold in range(6):
            path = (
                e16_root
                / f"subject_{subject:02d}"
                / f"seed_{args.seed}"
                / f"fold_{fold}"
                / "hpo_results.csv"
            )
            all_hpo_rows.extend(_read_hpo(path, subject, fold))
            input_manifest[f"e16_subject_{subject:02d}_fold_{fold}_hpo"] = file_sha256(path)
    selection = select_global_configuration(all_hpo_rows)
    if selection.folds != len(subjects) * 6:
        raise RuntimeError("global configuration was not evaluated in every fold")

    ensure_dir(output)
    write_json(output / "heldout_lock_manifest.json", v8_heldout_lock_manifest())
    source_tree = collect_source_tree_manifest(ROOT)
    write_json(output / "source_tree_manifest.json", source_tree)
    write_json(
        output / "source_tree_summary.json",
        {"files": len(source_tree), "sha256": source_tree_digest(source_tree)},
    )
    write_json(
        output / "global_selection.json",
        {
            "config_index": selection.config_index,
            "config": selection.config,
            "fixed_epoch": selection.fixed_epoch,
            "mean_validation_kappa": selection.mean_validation_kappa,
            "mean_validation_accuracy": selection.mean_validation_accuracy,
            "folds": selection.folds,
            "selection_inputs": "inner validation metrics only",
        },
    )

    training = config["training"]
    fold_rows: list[dict[str, Any]] = []
    subject_parts: dict[int, dict[str, list[np.ndarray]]] = {}
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
            raise RuntimeError("E17-1 must not access Session E")
        input_manifest[f"subject_{subject:02d}_data"] = file_sha256(data_path)
        parts = {name: [] for name in ("indices", "labels", "student", "teacher", "v9")}
        subject_parts[subject] = parts
        for fold in range(6):
            v9_fold = v9_root / f"subject_{subject:02d}" / f"seed_{args.seed}" / f"fold_{fold}"
            cache_path = v9_fold / "frozen_dual_feature_cache.npz"
            with np.load(cache_path, allow_pickle=False) as archive:
                cache = {name: archive[name] for name in archive.files}
            input_manifest[f"v9_subject_{subject:02d}_fold_{fold}_cache"] = file_sha256(
                cache_path
            )
            outer_train = cache["outer_train_indices"].astype(np.int64)
            outer_test = cache["outer_test_indices"].astype(np.int64)
            standardizer = fit_feature_standardizer(
                cache["atcnet_outer_train_sequence"], cache["fbcnet_outer_train_sequence"]
            )
            train_atc, train_fbc = standardizer.transform(
                cache["atcnet_outer_train_sequence"], cache["fbcnet_outer_train_sequence"]
            )
            test_atc, test_fbc = standardizer.transform(
                cache["atcnet_outer_test_sequence"], cache["fbcnet_outer_test_sequence"]
            )
            model_kwargs = {
                "hidden_channels": int(selection.config["hidden_channels"]),
                "dropout": float(selection.config["dropout"]),
                "temporal_layers": int(training["temporal_layers"]),
                "band_channels": int(training["band_channels"]),
            }
            fold_seed = int(args.seed) * 100_003 + fold * 1_009 + 17_100_007
            fit = fit_v16_continuous(
                atc_train=train_atc,
                fbc_train=train_fbc,
                y_train=labels[outer_train],
                atc_validation=None,
                fbc_validation=None,
                y_validation=None,
                device=args.device,
                seed=fold_seed,
                epochs=int(training["epochs"]),
                fixed_epoch=selection.fixed_epoch,
                scheduler_epochs=int(training["epochs"]),
                batch_size=int(training["batch_size"]),
                learning_rate=float(selection.config["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
                model_kwargs=model_kwargs,
                run_label=f"E17-1:S{subject}:fold{fold}",
            )
            evaluation = predict_v16_continuous(
                fit.model,
                test_atc,
                test_fbc,
                labels[outer_test],
                device=args.device,
                batch_size=int(training["batch_size"]),
            )
            teacher = np.asarray(cache["teacher_outer_test"], dtype=np.float32)
            reference_path = v9_fold / "sew_clif_kd" / "outer_predictions.npz"
            with np.load(reference_path, allow_pickle=False) as archive:
                if not np.array_equal(archive["indices"], outer_test):
                    raise RuntimeError("V9 reference predictions are not aligned")
                v9_logits = np.asarray(archive["logits"], dtype=np.float32)
            outer_labels = labels[outer_test]
            teacher_metrics = _metrics(outer_labels, teacher)
            v9_metrics = _metrics(outer_labels, v9_logits)
            fold_rows.append(
                {
                    "subject": subject,
                    "seed": int(args.seed),
                    "fold": fold,
                    "student_accuracy": evaluation["accuracy"],
                    "student_kappa": evaluation["kappa"],
                    "teacher_accuracy": teacher_metrics["accuracy"],
                    "v9_accuracy": v9_metrics["accuracy"],
                    "fixed_epoch": selection.fixed_epoch,
                    "config_index": selection.config_index,
                    "session_e_accessed": False,
                }
            )
            parts["indices"].append(outer_test)
            parts["labels"].append(outer_labels)
            parts["student"].append(np.asarray(evaluation["logits"], dtype=np.float32))
            parts["teacher"].append(teacher)
            parts["v9"].append(v9_logits)
            fold_output = ensure_dir(
                output / f"subject_{subject:02d}" / f"seed_{args.seed}" / f"fold_{fold}"
            )
            write_csv(fold_output / "history.csv", fit.history)
            np.savez_compressed(
                fold_output / "predictions.npz",
                indices=outer_test,
                labels=outer_labels,
                student_logits=evaluation["logits"],
                teacher_logits=teacher,
                v9_logits=v9_logits,
            )
            del fit, evaluation
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    subject_rows: list[dict[str, Any]] = []
    for subject, parts in subject_parts.items():
        all_indices = np.concatenate(parts["indices"])
        all_labels = np.concatenate(parts["labels"])
        if len(all_indices) != 288 or len(np.unique(all_indices)) != 288:
            raise RuntimeError(f"subject {subject} does not have exact outer-fold coverage")
        metrics_by_name = {
            name: _metrics(all_labels, np.concatenate(parts[name]))
            for name in ("student", "teacher", "v9")
        }
        subject_rows.append(
            {
                "subject": subject,
                "seed": int(args.seed),
                "trials": len(all_labels),
                "student_accuracy": metrics_by_name["student"]["accuracy"],
                "student_kappa": metrics_by_name["student"]["kappa"],
                "teacher_accuracy": metrics_by_name["teacher"]["accuracy"],
                "v9_accuracy": metrics_by_name["v9"]["accuracy"],
            }
        )
    mean_student = float(np.mean([row["student_accuracy"] for row in subject_rows]))
    mean_teacher = float(np.mean([row["teacher_accuracy"] for row in subject_rows]))
    mean_v9 = float(np.mean([row["v9_accuracy"] for row in subject_rows]))
    decision = {
        "status": "completed",
        "stage": "E17-1-global-inner-selection",
        "global_config_index": selection.config_index,
        "global_config": selection.config,
        "fixed_epoch": selection.fixed_epoch,
        "mean_student_accuracy": mean_student,
        "mean_teacher_accuracy": mean_teacher,
        "mean_v9_accuracy": mean_v9,
        "delta_vs_teacher_pp": 100.0 * (mean_student - mean_teacher),
        "delta_vs_v9_pp": 100.0 * (mean_student - mean_v9),
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
        "next_stage": "E17-2-logit-preserving-residual",
    }
    write_csv(output / "fold_summary.csv", fold_rows)
    write_csv(output / "subject_summary.csv", subject_rows)
    write_json(output / "input_manifest.json", input_manifest)
    write_json(output / "decision.json", decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
