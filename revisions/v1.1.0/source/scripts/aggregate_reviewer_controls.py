"""Aggregate REV-E1--E4 with participant-first paired bootstrap summaries."""

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

from dpc_snn.experiments.reviewer_controls import event_accumulation_proxy
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json
from dpc_snn.utils.metrics import classification_metrics

METRICS = ("accuracy", "balanced_accuracy", "kappa", "macro_f1")


def _prediction_metrics(path: Path, n_classes: int) -> dict[str, float]:
    with np.load(path, allow_pickle=False) as archive:
        label = np.asarray(archive["label"], dtype=np.int64)
        pred = np.asarray(archive["pred"], dtype=np.int64)
    if label.ndim != 1 or pred.shape != label.shape:
        raise RuntimeError(f"invalid prediction archive: {path}")
    return classification_metrics(label, pred, n_classes=n_classes)


def _checked_metrics(directory: Path, n_classes: int) -> dict[str, Any]:
    metrics_path = directory / "metrics.json"
    prediction_path = directory / "predictions.npz"
    metrics = read_json(metrics_path)
    recomputed = _prediction_metrics(prediction_path, n_classes)
    for key in METRICS:
        if not np.isclose(float(metrics[key]), float(recomputed[key]), atol=1e-12):
            raise RuntimeError(f"metric cannot be recomputed for {directory}: {key}")
    if metrics.get("evaluation_gradient_updates") is not False:
        raise RuntimeError(f"evaluation update flag is invalid: {directory}")
    return metrics


def _parent_metrics(directory: Path, n_classes: int) -> dict[str, Any]:
    metrics_path = directory / "metrics.json"
    prediction_path = directory / "predictions.npz"
    if metrics_path.is_file():
        metrics = read_json(metrics_path)
        recomputed = _prediction_metrics(prediction_path, n_classes)
        for key in METRICS:
            if not np.isclose(float(metrics[key]), float(recomputed[key]), atol=1e-12):
                raise RuntimeError(f"parent metric cannot be recomputed: {directory}")
        return metrics
    return _prediction_metrics(prediction_path, n_classes)


def _bootstrap(values: np.ndarray, repetitions: int, seed: int) -> tuple[float, float, float]:
    if values.ndim != 1 or values.size < 1:
        raise ValueError("paired bootstrap requires at least one participant")
    generator = np.random.default_rng(seed)
    samples = generator.choice(values, size=(int(repetitions), values.size), replace=True).mean(1)
    lower, upper = np.quantile(samples, (0.025, 0.975))
    return float(values.mean()), float(lower), float(upper)


def _budgets(config: dict[str, Any], dataset: str, variant: str, smoke: bool) -> list[str]:
    spec = config["variants"][variant]
    values = (
        config["fixed_budget_controls"]
        if spec["budget_group"] == "fixed_budget_controls"
        else config["datasets"][dataset]["equal_update_budgets"]
    )
    selected = [str(value) for value in values]
    return sorted({selected[0], selected[-1]}) if smoke else selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "reviewer_controls.yaml"),
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    root = Path(args.input).resolve()
    parent = Path(config["parents"]["e31"]).resolve()
    output = ensure_dir(Path(args.output).resolve())
    seeds = [int(config["seeds"][0])] if args.smoke else [int(value) for value in config["seeds"]]
    rows: list[dict[str, Any]] = []
    spike_rows: list[dict[str, Any]] = []
    operation_rows: list[dict[str, Any]] = []
    for dataset, dataset_config in config["datasets"].items():
        subjects = (
            [int(dataset_config["subjects"][0])]
            if args.smoke
            else [int(value) for value in dataset_config["subjects"]]
        )
        n_classes = int(dataset_config["n_classes"])
        for subject in subjects:
            status = read_json(root / dataset / f"subject_{subject:02d}" / "evaluation_status.json")
            if status.get("status") != "completed":
                raise RuntimeError(f"reviewer evaluation incomplete: {dataset} subject {subject}")
            for seed in seeds:
                parent_cache: dict[tuple[str, str], dict[str, Any]] = {}
                for variant in config["variants"]:
                    spec = config["variants"][variant]
                    for budget in _budgets(config, dataset, variant, args.smoke):
                        evaluation = (
                            root
                            / dataset
                            / f"subject_{subject:02d}"
                            / f"seed_{seed}"
                            / f"budget_{budget}"
                            / variant
                            / "evaluation"
                        )
                        metrics = _checked_metrics(evaluation, n_classes)
                        rows.append(
                            {
                                "dataset": dataset,
                                "subject": subject,
                                "seed": seed,
                                "budget": budget,
                                "variant": variant,
                                "experiment": spec["experiment"],
                                **{key: float(metrics[key]) for key in METRICS},
                            }
                        )
                        spike = metrics.get("spike_summary", {})
                        if spike.get("binary") is True:
                            event_proxy = event_accumulation_proxy(spike)
                            operation_rows.append(
                                {
                                    "dataset": dataset,
                                    "subject": subject,
                                    "seed": seed,
                                    "budget": budget,
                                    "variant": variant,
                                    "eligible_event_accumulations": event_proxy[
                                        "eligible_event_accumulations"
                                    ],
                                    "scope": event_proxy["scope"],
                                    "actual_gpu_sparse_execution_measured": False,
                                    "energy_claim_supported": False,
                                }
                            )
                            for layer, (rate, sparsity, events, elements) in enumerate(
                                zip(
                                    spike["layer_rates"],
                                    spike["layer_sparsities"],
                                    spike["layer_events"],
                                    spike["layer_elements"],
                                    strict=True,
                                )
                            ):
                                spike_rows.append(
                                    {
                                        "dataset": dataset,
                                        "subject": subject,
                                        "seed": seed,
                                        "budget": budget,
                                        "variant": variant,
                                        "layer": layer,
                                        "firing_rate": float(rate),
                                        "event_sparsity": float(sparsity),
                                        "events": int(events),
                                        "denominator_elements": int(elements),
                                        "denominator_definition": "batch_x_channel_x_time",
                                        "total_events_all_layers": int(spike["events"]),
                                    }
                                )
                        for parent_variant in ("ann_sew_ce", "sew_clif_ce"):
                            key = (budget, parent_variant)
                            if key in parent_cache:
                                continue
                            parent_eval = (
                                parent
                                / dataset
                                / f"subject_{subject:02d}"
                                / f"seed_{seed}"
                                / f"budget_{budget}"
                                / parent_variant
                                / "evaluation"
                            )
                            parent_cache[key] = _parent_metrics(parent_eval, n_classes)
                for (budget, variant), metrics in parent_cache.items():
                    rows.append(
                        {
                            "dataset": dataset,
                            "subject": subject,
                            "seed": seed,
                            "budget": budget,
                            "variant": variant,
                            "experiment": "RECOVERED_PARENT",
                            **{key: float(metrics[key]) for key in METRICS},
                        }
                    )
    write_csv(output / "seed_metrics.csv", rows)
    write_csv(output / "spike_sparsity.csv", spike_rows)
    write_csv(output / "event_accumulation_proxy.csv", operation_rows)
    subject_rows: list[dict[str, Any]] = []
    keys = sorted({(row["dataset"], row["subject"], row["budget"], row["variant"]) for row in rows})
    for dataset, subject, budget, variant in keys:
        selected = [
            row
            for row in rows
            if (row["dataset"], row["subject"], row["budget"], row["variant"])
            == (dataset, subject, budget, variant)
        ]
        if len(selected) != len(seeds):
            raise RuntimeError(
                f"seed completeness mismatch: {dataset}/{subject}/{budget}/{variant}"
            )
        subject_rows.append(
            {
                "dataset": dataset,
                "subject": subject,
                "budget": budget,
                "variant": variant,
                **{key: float(np.mean([row[key] for row in selected])) for key in METRICS},
            }
        )
    write_csv(output / "participant_metrics.csv", subject_rows)
    contrasts = {
        "REV-E1_Exact_minus_ANN": ("soft_clif_exact", "ann_sew_ce"),
        "REV-E1_Hard_minus_Exact": ("sew_clif_ce", "soft_clif_exact"),
        "REV-E2_Hard_minus_GRUStat": ("sew_clif_ce", "gru_stat"),
        "REV-E2_Hard_minus_TCNStat": ("sew_clif_ce", "tcn_stat"),
        "REV-E3_HardEqual_minus_ANNEqual": ("sew_clif_equal", "ann_sew_equal"),
    }
    lookup = {
        (row["dataset"], row["subject"], row["budget"], row["variant"]): row for row in subject_rows
    }
    contrast_rows: list[dict[str, Any]] = []
    for contrast_index, (name, (left, right)) in enumerate(contrasts.items()):
        groups = sorted(
            {
                (row["dataset"], row["budget"])
                for row in subject_rows
                if row["variant"] == left
                and (row["dataset"], row["subject"], row["budget"], right) in lookup
            }
        )
        for dataset, budget in groups:
            subjects = sorted(
                row["subject"]
                for row in subject_rows
                if row["dataset"] == dataset and row["budget"] == budget and row["variant"] == left
            )
            for metric_index, metric in enumerate(METRICS):
                values = np.asarray(
                    [
                        lookup[(dataset, subject, budget, left)][metric]
                        - lookup[(dataset, subject, budget, right)][metric]
                        for subject in subjects
                    ],
                    dtype=np.float64,
                )
                mean, lower, upper = _bootstrap(
                    values,
                    args.bootstrap_repetitions,
                    seed=20260901 + contrast_index * 101 + metric_index,
                )
                contrast_rows.append(
                    {
                        "contrast": name,
                        "left": left,
                        "right": right,
                        "dataset": dataset,
                        "budget": budget,
                        "metric": metric,
                        "participants": int(values.size),
                        "mean_difference": mean,
                        "bootstrap_95_lower": lower,
                        "bootstrap_95_upper": upper,
                    }
                )
    write_csv(output / "paired_participant_contrasts.csv", contrast_rows)
    summary = {
        "schema": "dpc-snn-reviewer-controls-aggregate/v1",
        "status": "completed",
        "smoke": bool(args.smoke),
        "seed_rows": len(rows),
        "participant_rows": len(subject_rows),
        "contrast_rows": len(contrast_rows),
        "spike_rows": len(spike_rows),
        "operation_proxy_rows": len(operation_rows),
        "seed_aggregation": "mean_within_participant_before_inference",
        "single_budget_inference": "paired_participant_bootstrap",
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
