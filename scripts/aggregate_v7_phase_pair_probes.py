#!/usr/bin/env python3
"""Aggregate complete Subject-1 phase-pair probe folds with strict provenance."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402


FEATURES = ("source_only", "envelope_pair", "phase_pair")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    manifests: dict[int, dict[str, Any]] = {}
    for value in args.inputs:
        root = Path(value).resolve()
        manifest = read_json(root / "manifest.json")
        if manifest.get("status") != "completed":
            raise RuntimeError(f"incomplete probe: {root}")
        if bool(manifest.get("session_e_accessed")):
            raise RuntimeError(f"probe accessed Session E: {root}")
        if int(manifest.get("subject", -1)) != 1:
            raise RuntimeError("Subject-1 aggregate received another subject")
        if int(manifest.get("class_label", -1)) != 3:
            raise RuntimeError("phase-pair aggregate requires the class-3 target")
        if manifest.get("prior_scope") != "pooled":
            raise RuntimeError("aggregate requires label-agnostic pooled priors")
        if (
            manifest.get("feature_normalization")
            != "outer_train_rms_scale_only_no_center_no_bias"
        ):
            raise RuntimeError("probe does not satisfy the zero-safe normalization contract")
        fold = int(manifest["fold"])
        if fold in manifests:
            raise RuntimeError(f"duplicate fold {fold}")
        manifests[fold] = {"root": str(root), **manifest}
    if set(manifests) != set(range(6)):
        raise RuntimeError(f"expected folds 0-5, found {sorted(manifests)}")

    rows: list[dict[str, Any]] = []
    for fold, manifest in sorted(manifests.items()):
        metrics = {row["feature"]: row for row in manifest["metrics"]}
        if set(metrics) != set(FEATURES):
            raise RuntimeError(f"fold {fold} has an incomplete feature comparison")
        for feature in FEATURES:
            row = metrics[feature]
            if float(row.get("matched_zero_score_max_abs", 1.0)) > 1e-12:
                raise RuntimeError(f"fold {fold} violates matched-zero score semantics")
            rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "outer_fold_auc": float(row["outer_fold_auc"]),
                    "inner_run_group_auc": float(row["inner_run_group_auc"]),
                    "permutation_p": float(row["permutation_p"]),
                    "selected_regularization": float(row["selected_regularization"]),
                    "outer_balanced_accuracy": float(row["outer_balanced_accuracy"]),
                    "accepted_within_band_routes": int(
                        row["accepted_within_band_routes"]
                    ),
                    "run_root": manifest["root"],
                }
            )

    summary: dict[str, dict[str, Any]] = {}
    for feature in FEATURES:
        selected = [row for row in rows if row["feature"] == feature]
        auc = np.asarray([row["outer_fold_auc"] for row in selected])
        summary[feature] = {
            "mean_outer_auc": float(auc.mean()),
            "median_outer_auc": float(np.median(auc)),
            "minimum_outer_auc": float(auc.min()),
            "maximum_permutation_p": float(
                max(row["permutation_p"] for row in selected)
            ),
            "fold_auc": auc.tolist(),
        }
    phase_auc = np.asarray(summary["phase_pair"]["fold_auc"])
    source_auc = np.asarray(summary["source_only"]["fold_auc"])
    positive_advantage = int(((phase_auc - source_auc) >= 0.05).sum())
    gate = {
        "passed": bool(
            float(np.median(phase_auc)) >= 0.85
            and float(phase_auc.min()) >= 0.80
            and summary["phase_pair"]["maximum_permutation_p"] < 0.05
            and positive_advantage >= 5
        ),
        "minimum_median_phase_auc": 0.85,
        "minimum_fold_phase_auc": 0.80,
        "maximum_fold_permutation_p": 0.05,
        "minimum_folds_phase_advantage_ge_0_05": 5,
        "observed_folds_phase_advantage_ge_0_05": positive_advantage,
        "development_rule_declared_after_fold_0_not_confirmatory": True,
    }
    output = ensure_dir(Path(args.output).resolve())
    write_csv(output / "fold_metrics.csv", rows)
    write_json(
        output / "aggregate.json",
        {
            "status": "completed",
            "experiment": "v7_r8_phase_pair_probe_s1_sixfold_aggregate",
            "subject": 1,
            "folds": sorted(manifests),
            "class_label": 3,
            "prior_scope": "pooled",
            "selection_session": "T",
            "session_e_accessed": False,
            "feature_normalization": (
                "outer_train_rms_scale_only_no_center_no_bias"
            ),
            "summary": summary,
            "gate": gate,
            "input_manifests": {
                str(fold): manifest["root"] for fold, manifest in manifests.items()
            },
        },
    )
    print(
        "__V7_PHASE_PAIR_AGGREGATE__ "
        f"passed={str(gate['passed']).lower()} "
        f"median_auc={np.median(phase_auc):.6f} "
        f"min_auc={phase_auc.min():.6f} positive_advantage={positive_advantage}/6",
        flush=True,
    )


if __name__ == "__main__":
    main()
