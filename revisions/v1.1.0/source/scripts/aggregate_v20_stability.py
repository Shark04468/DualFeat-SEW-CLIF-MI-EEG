#!/usr/bin/env python3
"""Aggregate the frozen SEW-CLIF E20 three-seed stability gate."""

from __future__ import annotations

import argparse
import csv
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
from dpc_snn.utils.io import ensure_dir, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from scripts.run_v8_e1_baselines import _subject_file  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _v9_accuracy(
    v9_root: Path, labels: np.ndarray, *, subject: int, seed: int
) -> float:
    fold_labels: list[np.ndarray] = []
    fold_logits: list[np.ndarray] = []
    fold_indices: list[np.ndarray] = []
    for fold in range(6):
        path = (
            v9_root
            / f"subject_{subject:02d}"
            / f"seed_{seed}"
            / f"fold_{fold}"
            / "sew_clif_kd"
            / "outer_predictions.npz"
        )
        with np.load(path, allow_pickle=False) as archive:
            indices = archive["indices"].astype(np.int64)
            saved_labels = archive["labels"].astype(np.int64)
            if not np.array_equal(saved_labels, labels[indices]):
                raise RuntimeError(f"V9 labels are not aligned: {path}")
            fold_indices.append(indices)
            fold_labels.append(saved_labels)
            fold_logits.append(np.asarray(archive["logits"], dtype=np.float32))
    all_indices = np.concatenate(fold_indices)
    if len(all_indices) != 288 or len(np.unique(all_indices)) != 288:
        raise RuntimeError("V9 reference lacks exact six-fold coverage")
    all_labels = np.concatenate(fold_labels)
    all_logits = np.concatenate(fold_logits)
    return float(
        classification_metrics(
            all_labels, all_logits.argmax(axis=1), n_classes=4
        )["accuracy"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--v9-root", required=True)
    parser.add_argument("--seed0-root", required=True)
    parser.add_argument("--seed1-root", required=True)
    parser.add_argument("--seed2-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subjects", default="1,3,8")
    args = parser.parse_args()

    data_root = Path(args.data).resolve()
    v9_root = Path(args.v9_root).resolve()
    roots = {
        0: Path(args.seed0_root).resolve(),
        1: Path(args.seed1_root).resolve(),
        2: Path(args.seed2_root).resolve(),
    }
    output = Path(args.output).resolve()
    subjects = [int(item) for item in args.subjects.split(",") if item.strip()]
    if output.exists():
        raise FileExistsError(f"E20 output must be immutable and new: {output}")
    ensure_dir(output)

    pair_rows: list[dict[str, Any]] = []
    for subject in subjects:
        data = load_processed_npz(_subject_file(data_root, subject))
        _, labels, _, access = session_t_development_view(data)
        labels = np.asarray(labels, dtype=np.int64)
        if (
            access.get("selected_session") != "T"
            or access.get("heldout_signals_used_by_development") is not False
            or access.get("heldout_labels_used_by_development") is not False
        ):
            raise RuntimeError("E20 must not access Session E")
        for seed, root in roots.items():
            rows = _read_csv(root / "subject_summary.csv")
            ann = next(
                row
                for row in rows
                if int(row["subject"]) == subject and row["variant"] == "ann_plain_ce"
            )
            snn = next(
                row
                for row in rows
                if int(row["subject"]) == subject and row["variant"] == "sew_clif_ce"
            )
            ann_accuracy = float(ann["clean_accuracy"])
            snn_accuracy = float(snn["clean_accuracy"])
            v9_accuracy = _v9_accuracy(
                v9_root, labels, subject=subject, seed=seed
            )
            pair_rows.append(
                {
                    "subject": subject,
                    "seed": seed,
                    "snn_accuracy": snn_accuracy,
                    "ann_accuracy": ann_accuracy,
                    "v9_accuracy": v9_accuracy,
                    "snn_minus_ann_pp": 100.0 * (snn_accuracy - ann_accuracy),
                    "snn_minus_v9_pp": 100.0 * (snn_accuracy - v9_accuracy),
                    "nonzero_scale_folds": int(snn["nonzero_scale_folds"]),
                    "mean_event_density": float(snn["mean_event_density"]),
                }
            )

    mean_snn = float(np.mean([row["snn_accuracy"] for row in pair_rows]))
    mean_ann = float(np.mean([row["ann_accuracy"] for row in pair_rows]))
    mean_v9 = float(np.mean([row["v9_accuracy"] for row in pair_rows]))
    deltas_v9 = np.asarray([row["snn_minus_v9_pp"] for row in pair_rows])
    positive_pairs = int(np.sum(deltas_v9 > 0.0))
    mean_v9_pass = mean_snn - mean_v9 >= 0.01
    positive_pair_pass = positive_pairs >= 7
    worst_pair_pass = float(deltas_v9.min()) >= -1.0
    ann_match_pass = mean_snn - mean_ann >= -0.003
    decision = {
        "status": "completed",
        "stage": "E20-three-seed-stability",
        "frozen_variant": "sew_clif_ce",
        "subject_seed_pairs": len(pair_rows),
        "mean_snn_accuracy": mean_snn,
        "mean_ann_accuracy": mean_ann,
        "mean_v9_accuracy": mean_v9,
        "snn_minus_ann_pp": 100.0 * (mean_snn - mean_ann),
        "snn_minus_v9_pp": 100.0 * (mean_snn - mean_v9),
        "positive_vs_v9_pairs": positive_pairs,
        "worst_vs_v9_pair_pp": float(deltas_v9.min()),
        "mean_v9_pass": mean_v9_pass,
        "positive_pair_pass": positive_pair_pass,
        "worst_pair_pass": worst_pair_pass,
        "ann_match_pass": ann_match_pass,
        "mean_event_density": float(np.mean([row["mean_event_density"] for row in pair_rows])),
        "active_correction_pairs": int(
            sum(row["nonzero_scale_folds"] > 0 for row in pair_rows)
        ),
        "gate_passed": mean_v9_pass
        and positive_pair_pass
        and worst_pair_pass
        and ann_match_pass,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    decision["next_stage"] = "E21-delay-residual" if decision["gate_passed"] else "stop"
    write_csv(output / "pair_summary.csv", pair_rows)
    write_json(output / "decision.json", decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
