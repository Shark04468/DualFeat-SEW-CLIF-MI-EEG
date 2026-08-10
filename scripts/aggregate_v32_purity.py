"""Aggregate the registered V32 firing-rate-purity comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import wilcoxon
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scope", choices=("canary", "full"), required=True)
    parser.add_argument("--config", default="configs/experiments/v32_matched_purity_and_fusion.yaml")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    scope = config["canary"] if args.scope == "canary" else config["dataset"]
    subjects = [int(value) for value in scope["subjects"]]
    seeds = [int(value) for value in scope["seeds"]]
    folds = [int(value) for value in scope["folds"]]
    input_root = Path(args.input).resolve()
    output = ensure_dir(Path(args.output).resolve())

    fold_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for subject in subjects:
        for seed in seeds:
            predictions: list[dict[str, np.ndarray]] = []
            for fold in folds:
                fold_dir = input_root / f"subject_{subject:02d}" / f"seed_{seed}" / f"fold_{fold}"
                metrics = read_json(fold_dir / "metrics.json")
                if metrics.get("status") != "completed" or metrics.get("session_e_accessed") is not False:
                    raise RuntimeError(f"invalid V32 fold status: {fold_dir}")
                fold_rows.append(metrics)
                with np.load(fold_dir / "outer_predictions.npz", allow_pickle=False) as archive:
                    predictions.append({name: np.asarray(archive[name]) for name in archive.files})
            indices = np.concatenate([row["indices"] for row in predictions])
            order = np.argsort(indices)
            if len(np.unique(indices)) != len(indices):
                raise RuntimeError(f"duplicate V32 OOF trial index for subject {subject}, seed {seed}")
            labels = np.concatenate([row["labels"] for row in predictions])[order]
            result: dict[str, float] = {}
            for name in ("fr0", "ann", "fr001"):
                logits = np.concatenate([row[f"{name}_logits"] for row in predictions])[order]
                result[name] = float(
                    classification_metrics(labels, logits.argmax(1), n_classes=4)["accuracy"]
                )
            pair_rows.append(
                {
                    "subject": subject,
                    "seed": seed,
                    "fr0_accuracy": result["fr0"],
                    "ann_accuracy": result["ann"],
                    "fr001_accuracy": result["fr001"],
                    "fr0_minus_ann_pp": 100.0 * (result["fr0"] - result["ann"]),
                    "fr001_minus_ann_pp": 100.0 * (result["fr001"] - result["ann"]),
                    "fr0_minus_fr001_pp": 100.0 * (result["fr0"] - result["fr001"]),
                }
            )

    subject_rows: list[dict[str, Any]] = []
    for subject in subjects:
        rows = [row for row in pair_rows if int(row["subject"]) == subject]
        subject_rows.append(
            {
                "subject": subject,
                **{
                    key: float(np.mean([float(row[key]) for row in rows]))
                    for key in (
                        "fr0_accuracy",
                        "ann_accuracy",
                        "fr001_accuracy",
                        "fr0_minus_ann_pp",
                        "fr001_minus_ann_pp",
                        "fr0_minus_fr001_pp",
                    )
                },
            }
        )

    new_gain = np.asarray([row["fr0_minus_ann_pp"] for row in subject_rows], dtype=float)
    original_gain = np.asarray(
        [row["fr001_minus_ann_pp"] for row in subject_rows], dtype=float
    )
    fr0_vs_regularized = np.asarray(
        [row["fr0_minus_fr001_pp"] for row in subject_rows], dtype=float
    )
    mean_original = float(original_gain.mean())
    retained = float(new_gain.mean() / mean_original) if mean_original > 0.0 else float("nan")
    gate_cfg = config["canary"]["gate"] if args.scope == "canary" else config["full_gate"]
    checks = {
        "mean_fr0_minus_ann": float(new_gain.mean())
        >= float(gate_cfg["mean_fr0_minus_ann_pp_minimum"]),
        "positive_subjects": int(np.count_nonzero(new_gain > 0.0))
        >= int(gate_cfg["positive_subjects_minimum"]),
        "original_gain_retained": np.isfinite(retained)
        and retained >= float(gate_cfg["original_gain_retained_fraction_minimum"]),
        "fr0_not_materially_below_fr001": float(fr0_vs_regularized.mean())
        >= float(gate_cfg["fr0_minus_fr001_pp_minimum"]),
    }
    one_sided_p = None
    if args.scope == "full":
        one_sided_p = float(wilcoxon(new_gain, alternative="greater").pvalue)
        checks["one_sided_wilcoxon"] = one_sided_p < float(
            gate_cfg["one_sided_wilcoxon_maximum"]
        )
    decision = {
        "status": "pass" if all(checks.values()) else "fail",
        "scope": args.scope,
        "subjects": subjects,
        "seeds": seeds,
        "folds": folds,
        "mean_fr0_accuracy": float(np.mean([row["fr0_accuracy"] for row in subject_rows])),
        "mean_ann_accuracy": float(np.mean([row["ann_accuracy"] for row in subject_rows])),
        "mean_fr001_accuracy": float(
            np.mean([row["fr001_accuracy"] for row in subject_rows])
        ),
        "mean_fr0_minus_ann_pp": float(new_gain.mean()),
        "mean_fr001_minus_ann_pp": mean_original,
        "mean_fr0_minus_fr001_pp": float(fr0_vs_regularized.mean()),
        "original_gain_retained_fraction": retained,
        "positive_subjects": int(np.count_nonzero(new_gain > 0.0)),
        "negative_subjects": int(np.count_nonzero(new_gain < 0.0)),
        "ties": int(np.count_nonzero(new_gain == 0.0)),
        "one_sided_wilcoxon_p": one_sided_p,
        "checks": checks,
        "authorized_next_stage": (
            "v32_full_purity" if args.scope == "canary" else "v32_fusion_ablation"
        )
        if all(checks.values())
        else None,
    }
    write_csv(output / "fold_metrics.csv", fold_rows)
    write_csv(output / "subject_seed_metrics.csv", pair_rows)
    write_csv(output / "subject_metrics.csv", subject_rows)
    write_json(output / "decision.json", decision)
    print(yaml.safe_dump(decision, sort_keys=False))


if __name__ == "__main__":
    main()

