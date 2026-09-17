#!/usr/bin/env python3
"""Run E17-2 lossless logit-preserving residual fusion diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

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
from dpc_snn.experiments.v17_information_replay import (  # noqa: E402
    fit_ridge_probe,
    residual_fusion_logits,
    select_residual_fusion,
    select_ridge_alpha,
)
from dpc_snn.experiments.v62_protocol import file_sha256  # noqa: E402
from dpc_snn.utils.io import ensure_dir, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from scripts.run_v8_e1_baselines import _subject_file  # noqa: E402


VARIANTS = ("teacher", "branch_ridge", "branch_residual", "full_residual")


def _metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    return classification_metrics(labels, logits.argmax(axis=1), n_classes=4)


def _features(cache: dict[str, np.ndarray], partition: str) -> dict[str, np.ndarray]:
    atc = np.asarray(cache[f"atcnet_{partition}_sequence"], dtype=np.float32)
    fbc = np.asarray(cache[f"fbcnet_{partition}_sequence"], dtype=np.float32)
    atc_logits = np.asarray(cache[f"atcnet_{partition}_logits"], dtype=np.float32)
    fbc_logits = np.asarray(cache[f"fbcnet_{partition}_logits"], dtype=np.float32)
    branch = np.concatenate((atc_logits, fbc_logits), axis=1)
    full = np.concatenate(
        (atc.reshape(atc.shape[0], -1), fbc.reshape(fbc.shape[0], -1), branch), axis=1
    )
    return {"branch": branch, "full": full}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--v9-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subjects", default="1,3,8")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_root = Path(args.data).resolve()
    v9_root = Path(args.v9_root).resolve()
    output = Path(args.output).resolve()
    subjects = [int(item) for item in args.subjects.split(",") if item.strip()]
    if output.exists():
        raise FileExistsError(f"E17-2 output must be immutable and new: {output}")
    ensure_dir(output)
    write_json(output / "heldout_lock_manifest.json", v8_heldout_lock_manifest())
    source_tree = collect_source_tree_manifest(ROOT)
    write_json(output / "source_tree_manifest.json", source_tree)
    write_json(
        output / "source_tree_summary.json",
        {"files": len(source_tree), "sha256": source_tree_digest(source_tree)},
    )

    ridge_alphas = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    residual_alphas = (1.0, 10.0, 100.0)
    residual_scales = (0.0, 0.25, 0.5, 1.0)
    fold_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    parts: dict[tuple[int, str], list[np.ndarray]] = {}
    labels_by_subject: dict[int, list[np.ndarray]] = {}
    indices_by_subject: dict[int, list[np.ndarray]] = {}
    input_manifest: dict[str, str] = {}
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
            raise RuntimeError("E17-2 must not access Session E")
        input_manifest[f"subject_{subject:02d}_data"] = file_sha256(data_path)
        labels_by_subject[subject] = []
        indices_by_subject[subject] = []
        for variant in VARIANTS:
            parts[(subject, variant)] = []
        for fold in range(6):
            v9_fold = v9_root / f"subject_{subject:02d}" / f"seed_{args.seed}" / f"fold_{fold}"
            cache_path = v9_fold / "frozen_dual_feature_cache.npz"
            with np.load(cache_path, allow_pickle=False) as archive:
                cache = {name: archive[name] for name in archive.files}
            input_manifest[f"subject_{subject:02d}_fold_{fold}_cache"] = file_sha256(cache_path)
            split = {
                "selection_train": cache["inner_train_indices"].astype(np.int64),
                "selection_validation": cache["inner_validation_indices"].astype(np.int64),
                "outer_train": cache["outer_train_indices"].astype(np.int64),
                "outer_test": cache["outer_test_indices"].astype(np.int64),
            }
            feature = {
                name: _features(cache, name)
                for name in (
                    "selection_train",
                    "selection_validation",
                    "outer_train",
                    "outer_test",
                )
            }
            inner_y = labels[split["selection_train"]]
            validation_y = labels[split["selection_validation"]]
            outer_y = labels[split["outer_train"]]
            test_y = labels[split["outer_test"]]
            validation_base = np.asarray(cache["teacher_selection_validation"], dtype=np.float32)
            test_base = np.asarray(cache["teacher_outer_test"], dtype=np.float32)
            predictions = {"teacher": test_base}

            selected_alpha, rows = select_ridge_alpha(
                feature["selection_train"]["branch"],
                inner_y,
                feature["selection_validation"]["branch"],
                validation_y,
                alphas=ridge_alphas,
            )
            branch_probe = fit_ridge_probe(
                feature["outer_train"]["branch"], outer_y, alpha=selected_alpha
            )
            predictions["branch_ridge"] = branch_probe.predict_logits(
                feature["outer_test"]["branch"]
            )
            for row in rows:
                search_rows.append(
                    {
                        "subject": subject,
                        "fold": fold,
                        "variant": "branch_ridge",
                        "alpha": row["alpha"],
                        "scale": 1.0,
                        **{key: value for key, value in row.items() if key != "alpha"},
                    }
                )

            for variant, feature_name in (
                ("branch_residual", "branch"),
                ("full_residual", "full"),
            ):
                alpha, scale, rows = select_residual_fusion(
                    feature["selection_train"][feature_name],
                    inner_y,
                    feature["selection_validation"][feature_name],
                    validation_y,
                    validation_base,
                    alphas=residual_alphas,
                    scales=residual_scales,
                )
                if scale == 0.0:
                    logits = test_base.copy()
                else:
                    probe = fit_ridge_probe(
                        feature["outer_train"][feature_name], outer_y, alpha=alpha
                    )
                    correction = probe.predict_logits(feature["outer_test"][feature_name])
                    logits = residual_fusion_logits(test_base, correction, scale)
                predictions[variant] = logits
                for row in rows:
                    search_rows.append(
                        {
                            "subject": subject,
                            "fold": fold,
                            "variant": variant,
                            **row,
                            "selected": row["alpha"] == alpha and row["scale"] == scale,
                        }
                    )

            for variant, logits in predictions.items():
                metrics = _metrics(test_y, logits)
                fold_rows.append(
                    {
                        "subject": subject,
                        "seed": int(args.seed),
                        "fold": fold,
                        "variant": variant,
                        "outer_accuracy": metrics["accuracy"],
                        "outer_kappa": metrics["kappa"],
                        "session_e_accessed": False,
                    }
                )
                parts[(subject, variant)].append(logits)
            labels_by_subject[subject].append(test_y)
            indices_by_subject[subject].append(split["outer_test"])
            np.savez_compressed(
                output / f"subject_{subject:02d}_fold_{fold}_predictions.npz",
                indices=split["outer_test"],
                labels=test_y,
                **{f"{name}_logits": value for name, value in predictions.items()},
            )

    subject_rows: list[dict[str, Any]] = []
    for subject in subjects:
        all_indices = np.concatenate(indices_by_subject[subject])
        all_labels = np.concatenate(labels_by_subject[subject])
        if len(all_indices) != 288 or len(np.unique(all_indices)) != 288:
            raise RuntimeError(f"subject {subject} lacks exact six-fold coverage")
        for variant in VARIANTS:
            metrics = _metrics(all_labels, np.concatenate(parts[(subject, variant)]))
            subject_rows.append(
                {
                    "subject": subject,
                    "seed": int(args.seed),
                    "variant": variant,
                    "accuracy": metrics["accuracy"],
                    "kappa": metrics["kappa"],
                    "trials": len(all_labels),
                }
            )
    summary_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        rows = [row for row in subject_rows if row["variant"] == variant]
        summary_rows.append(
            {
                "variant": variant,
                "mean_subject_accuracy": float(np.mean([row["accuracy"] for row in rows])),
                "mean_subject_kappa": float(np.mean([row["kappa"] for row in rows])),
            }
        )
    primary = next(row for row in summary_rows if row["variant"] == "branch_ridge")
    teacher = next(row for row in summary_rows if row["variant"] == "teacher")
    accuracy_pass = primary["mean_subject_accuracy"] >= 0.885
    teacher_gap_pass = (
        teacher["mean_subject_accuracy"] - primary["mean_subject_accuracy"] <= 0.005
    )
    decision = {
        "status": "completed",
        "stage": "E17-2-logit-preserving-residual",
        "primary_variant": "branch_ridge",
        "models": {row["variant"]: row for row in summary_rows},
        "accuracy_pass": accuracy_pass,
        "teacher_gap_pass": teacher_gap_pass,
        "gate_passed": accuracy_pass or teacher_gap_pass,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
        "next_stage": "E18-matched-ANN-SNN" if accuracy_pass or teacher_gap_pass else "stop",
    }
    write_csv(output / "search_results.csv", search_rows)
    write_csv(output / "fold_summary.csv", fold_rows)
    write_csv(output / "subject_summary.csv", subject_rows)
    write_csv(output / "model_summary.csv", summary_rows)
    write_json(output / "input_manifest.json", input_manifest)
    write_json(output / "decision.json", decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
