"""Aggregate REV-E5 BCI2a LH/RH predictions at the participant level."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json
from dpc_snn.utils.metrics import classification_metrics

METRICS = ("accuracy", "balanced_accuracy", "kappa", "macro_f1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "bci2a_binary_sensitivity.yaml"),
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    root = Path(args.input).resolve()
    output = ensure_dir(Path(args.output).resolve())
    subjects = (
        [int(config["dataset"]["subjects"][0])]
        if args.smoke
        else [int(value) for value in config["dataset"]["subjects"]]
    )
    seeds = [int(config["seeds"][0])] if args.smoke else [int(value) for value in config["seeds"]]
    rows: list[dict[str, Any]] = []
    for subject in subjects:
        status = read_json(
            root / "bci2a_binary" / f"subject_{subject:02d}" / "evaluation_status.json"
        )
        if status.get("status") != "completed":
            raise RuntimeError(f"REV-E5 evaluation incomplete: subject {subject}")
        for seed in seeds:
            seed_root = root / "bci2a_binary" / f"subject_{subject:02d}" / f"seed_{seed}"
            for budget_dir in sorted(seed_root.glob("budget_*")):
                budget = budget_dir.name.removeprefix("budget_")
                for variant in config["variants"]:
                    evaluation = budget_dir / variant / "evaluation"
                    metrics = read_json(evaluation / "metrics.json")
                    with np.load(evaluation / "predictions.npz", allow_pickle=False) as archive:
                        label = np.asarray(archive["label"], dtype=np.int64)
                        pred = np.asarray(archive["pred"], dtype=np.int64)
                    recomputed = classification_metrics(label, pred, n_classes=2)
                    for metric in METRICS:
                        if not np.isclose(
                            float(metrics[metric]), float(recomputed[metric]), atol=1e-12
                        ):
                            raise RuntimeError(f"REV-E5 metric cannot be recomputed: {evaluation}")
                    rows.append(
                        {
                            "subject": subject,
                            "seed": seed,
                            "budget": budget,
                            "variant": variant,
                            **{metric: float(metrics[metric]) for metric in METRICS},
                        }
                    )
    write_csv(output / "seed_metrics.csv", rows)
    participant: list[dict[str, Any]] = []
    keys = sorted({(row["subject"], row["budget"], row["variant"]) for row in rows})
    for subject, budget, variant in keys:
        selected = [
            row
            for row in rows
            if (row["subject"], row["budget"], row["variant"]) == (subject, budget, variant)
        ]
        if len(selected) != len(seeds):
            raise RuntimeError("REV-E5 seed completeness mismatch")
        participant.append(
            {
                "subject": subject,
                "budget": budget,
                "variant": variant,
                **{metric: float(np.mean([row[metric] for row in selected])) for metric in METRICS},
            }
        )
    write_csv(output / "participant_metrics.csv", participant)
    lookup = {(row["subject"], row["budget"], row["variant"]): row for row in participant}
    contrasts: list[dict[str, Any]] = []
    generator = np.random.default_rng(20260905)
    budgets = sorted({row["budget"] for row in participant})
    for budget in budgets:
        values_by_metric: dict[str, list[float]] = {metric: [] for metric in METRICS}
        for subject in subjects:
            snn = lookup[(subject, budget, "sew_clif_binary")]
            ann = lookup[(subject, budget, "ann_sew_binary")]
            for metric in METRICS:
                values_by_metric[metric].append(float(snn[metric]) - float(ann[metric]))
        for metric, raw in values_by_metric.items():
            values = np.asarray(raw, dtype=np.float64)
            samples = generator.choice(
                values, size=(int(args.bootstrap_repetitions), values.size), replace=True
            ).mean(1)
            lower, upper = np.quantile(samples, (0.025, 0.975))
            contrasts.append(
                {
                    "budget": budget,
                    "metric": metric,
                    "participants": int(values.size),
                    "mean_snn_minus_ann": float(values.mean()),
                    "bootstrap_95_lower": float(lower),
                    "bootstrap_95_upper": float(upper),
                }
            )
    write_csv(output / "paired_participant_contrasts.csv", contrasts)
    summary = {
        "schema": "dpc-snn-rev-e5-aggregate/v1",
        "status": "completed",
        "smoke": bool(args.smoke),
        "seed_rows": len(rows),
        "participant_rows": len(participant),
        "contrast_rows": len(contrasts),
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
