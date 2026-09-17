"""Aggregate capacity-controlled V32 feature-fusion ablations."""

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


MODES = ("atc_only", "fbc_only", "simple", "interaction")


def _holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        value = min(1.0, (count - rank) * pvalues[name])
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def _one_sided_wilcoxon(values: np.ndarray) -> float:
    if np.allclose(values, 0.0):
        return 1.0
    return float(wilcoxon(values, alternative="greater").pvalue)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion-root", required=True)
    parser.add_argument("--purity-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config",
        default="configs/experiments/v32_matched_purity_and_fusion.yaml",
    )
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    subjects = [int(value) for value in config["dataset"]["subjects"]]
    seeds = [int(value) for value in config["dataset"]["seeds"]]
    folds = [int(value) for value in config["dataset"]["folds"]]
    fusion_root = Path(args.fusion_root).resolve()
    purity_root = Path(args.purity_root).resolve()
    output = ensure_dir(Path(args.output).resolve())

    pair_rows: list[dict[str, Any]] = []
    for subject in subjects:
        for seed in seeds:
            fold_data: dict[str, list[dict[str, np.ndarray]]] = {
                mode: [] for mode in MODES
            }
            for fold in folds:
                for mode in MODES:
                    if mode == "interaction":
                        fold_dir = (
                            purity_root
                            / f"subject_{subject:02d}"
                            / f"seed_{seed}"
                            / f"fold_{fold}"
                        )
                    else:
                        fold_dir = (
                            fusion_root
                            / mode
                            / f"subject_{subject:02d}"
                            / f"seed_{seed}"
                            / f"fold_{fold}"
                        )
                    metrics = read_json(fold_dir / "metrics.json")
                    if (
                        metrics.get("status") != "completed"
                        or metrics.get("fusion_mode") != mode
                    ):
                        raise RuntimeError(f"invalid fusion result: {fold_dir}")
                    with np.load(
                        fold_dir / "outer_predictions.npz", allow_pickle=False
                    ) as archive:
                        fold_data[mode].append(
                            {
                                "indices": np.asarray(archive["indices"]),
                                "labels": np.asarray(archive["labels"]),
                                "logits": np.asarray(archive["fr0_logits"]),
                            }
                        )
            row: dict[str, Any] = {"subject": subject, "seed": seed}
            reference_indices = None
            reference_labels = None
            for mode in MODES:
                indices = np.concatenate(
                    [item["indices"] for item in fold_data[mode]]
                )
                order = np.argsort(indices)
                labels = np.concatenate(
                    [item["labels"] for item in fold_data[mode]]
                )[order]
                logits = np.concatenate(
                    [item["logits"] for item in fold_data[mode]]
                )[order]
                if reference_indices is None:
                    reference_indices = indices[order]
                    reference_labels = labels
                elif not np.array_equal(
                    reference_indices, indices[order]
                ) or not np.array_equal(reference_labels, labels):
                    raise RuntimeError("fusion modes do not share identical OOF trials")
                row[f"{mode}_accuracy"] = float(
                    classification_metrics(
                        labels, logits.argmax(1), n_classes=4
                    )["accuracy"]
                )
            for comparator in ("atc_only", "fbc_only", "simple"):
                row[f"interaction_minus_{comparator}_pp"] = 100.0 * (
                    row["interaction_accuracy"] - row[f"{comparator}_accuracy"]
                )
            pair_rows.append(row)

    metric_keys = [f"{mode}_accuracy" for mode in MODES] + [
        f"interaction_minus_{mode}_pp"
        for mode in ("atc_only", "fbc_only", "simple")
    ]
    subject_rows: list[dict[str, Any]] = []
    for subject in subjects:
        rows = [row for row in pair_rows if row["subject"] == subject]
        subject_rows.append(
            {
                "subject": subject,
                **{
                    key: float(np.mean([float(row[key]) for row in rows]))
                    for key in metric_keys
                },
            }
        )

    pvalues: dict[str, float] = {}
    comparisons: dict[str, Any] = {}
    for comparator in ("atc_only", "fbc_only", "simple"):
        key = f"interaction_minus_{comparator}_pp"
        values = np.asarray([row[key] for row in subject_rows], dtype=float)
        pvalues[comparator] = _one_sided_wilcoxon(values)
        comparisons[comparator] = {
            "mean_delta_pp": float(values.mean()),
            "median_delta_pp": float(np.median(values)),
            "positive_subjects": int(np.count_nonzero(values > 0.0)),
            "negative_subjects": int(np.count_nonzero(values < 0.0)),
            "ties": int(np.count_nonzero(values == 0.0)),
            "one_sided_wilcoxon_p": pvalues[comparator],
        }
    adjusted = _holm(pvalues)
    for comparator, value in adjusted.items():
        comparisons[comparator]["holm_p"] = value
    decision = {
        "status": "completed",
        "mean_accuracy": {
            mode: float(
                np.mean([row[f"{mode}_accuracy"] for row in subject_rows])
            )
            for mode in MODES
        },
        "comparisons": comparisons,
        "interaction_supported_over_capacity_matched_simple": bool(
            comparisons["simple"]["mean_delta_pp"] > 0.0
            and comparisons["simple"]["holm_p"] < 0.05
        ),
        "claim_rule": (
            "interaction is a main contribution only if it exceeds "
            "capacity-matched simple fusion after Holm correction"
        ),
    }
    write_csv(output / "subject_seed_metrics.csv", pair_rows)
    write_csv(output / "subject_metrics.csv", subject_rows)
    write_json(output / "decision.json", decision)
    print(yaml.safe_dump(decision, sort_keys=False))


if __name__ == "__main__":
    main()
