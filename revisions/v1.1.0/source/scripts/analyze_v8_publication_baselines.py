#!/usr/bin/env python3
"""Build fusion, subject-level statistics, figures, and a calibrated report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.analysis.v8_utility import multiclass_calibration_metrics  # noqa: E402
from dpc_snn.experiments.v62_protocol import write_trial_predictions  # noqa: E402
from dpc_snn.experiments.v8_publication_baselines import (  # noqa: E402
    FREQUENCY_OCCLUSIONS_HZ,
    REGION_CHANNELS,
    component_error_profile,
    equal_probability_fusion,
    saliency_profile_correlation,
)
from dpc_snn.utils.io import ensure_dir, read_json, save_npz, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import bootstrap_ci, classification_metrics  # noqa: E402


MODEL_LABELS = {
    "eegnet": "EEGNet",
    "fbcnet": "FBCNet",
    "atcnet": "ATCNet",
    "tcformer": "TCFormer",
    "eeg_conformer": "EEG Conformer",
    "bfatcnet": "BFATCNet",
    "atc_fbc_fusion": "ATCNet+FBCNet",
}
MODEL_COLORS = {
    "eegnet": "#2F6B9A",
    "fbcnet": "#D08C2F",
    "atcnet": "#238B57",
    "tcformer": "#8657A5",
    "eeg_conformer": "#B84A5A",
    "bfatcnet": "#4D8F9C",
    "atc_fbc_fusion": "#C7352C",
}
METRICS = ("accuracy", "balanced_accuracy", "kappa", "macro_f1")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run_base(output: Path, model: str, subject: int, seed: int) -> Path:
    return output / "runs" / model / f"subject_{subject:02d}" / f"seed_{seed}"


def _prediction_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _perturbation_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _saliency_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _cohorts(dataset: str, config: dict[str, Any], subjects: list[int]) -> dict[str, list[int]]:
    if dataset == "bci2a":
        return {"full_9": subjects}
    configured = dict(config["datasets"]["openbmi"].get("reporting_cohorts", {}))
    active = set(subjects)
    cohorts = {
        name: [int(value) for value in values if int(value) in active]
        for name, values in configured.items()
    }
    cohorts = {name: values for name, values in cohorts.items() if values}
    if set(subjects) != set(config["datasets"]["openbmi"]["subjects"]):
        cohorts = {"active_scope": subjects, **cohorts}
    return cohorts


def _profile_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if np.isclose(np.std(first), 0.0) or np.isclose(np.std(second), 0.0):
        return math.nan
    return float(np.corrcoef(first, second)[0, 1])


def _build_fusion(
    output: Path,
    scope: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fusion_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    perturbation_rows: list[dict[str, Any]] = []
    for subject in scope["subjects"]:
        for seed in scope["seeds"]:
            first_dir = _run_base(output, "atcnet", subject, seed) / "evaluation"
            second_dir = _run_base(output, "fbcnet", subject, seed) / "evaluation"
            first_metrics = read_json(first_dir / "metrics.json")
            second_metrics = read_json(second_dir / "metrics.json")
            first = _prediction_payload(first_dir / "predictions.npz")
            second = _prediction_payload(second_dir / "predictions.npz")
            for field in ("label", "subject", "session", "run", "trial_id", "seed"):
                if not np.array_equal(first[field], second[field]):
                    raise RuntimeError(f"fusion components differ in paired field {field!r}")
            probability, prediction = equal_probability_fusion(
                first["probabilities"], second["probabilities"]
            )
            logits = np.log(np.clip(probability, 1e-12, 1.0)).astype(np.float32)
            labels = first["label"].astype(np.int64)
            metrics = classification_metrics(labels, prediction, n_classes=probability.shape[1])
            calibration = multiclass_calibration_metrics(logits, labels)
            fusion_dir = ensure_dir(
                output / "analysis" / "fusion" / f"subject_{subject:02d}" / f"seed_{seed}"
            )
            write_trial_predictions(
                fusion_dir,
                logits=logits,
                probabilities=probability,
                pred=prediction,
                label=labels,
                subject=first["subject"],
                session=first["session"],
                run=first["run"],
                trial_id=first["trial_id"],
                seed=seed,
                model="atc_fbc_fusion",
            )
            profile = component_error_profile(
                labels, first["probabilities"], second["probabilities"]
            )
            first_perturb = _perturbation_payload(first_dir / "perturbations.npz")
            second_perturb = _perturbation_payload(second_dir / "perturbations.npz")
            fused_perturb_arrays: dict[str, np.ndarray] = {"label": labels}
            run_perturb_rows: list[dict[str, Any]] = []
            drop_profiles: dict[str, tuple[list[float], list[float]]] = {}
            for kind, locked_names in (
                ("frequency", list(FREQUENCY_OCCLUSIONS_HZ)),
                ("region", list(REGION_CHANNELS)),
            ):
                first_names = first_perturb[f"{kind}_names"].astype(str).tolist()
                second_names = second_perturb[f"{kind}_names"].astype(str).tolist()
                if first_names != locked_names or second_names != locked_names:
                    raise RuntimeError("fusion perturbation order differs between components")
                fused_probabilities = 0.5 * (
                    first_perturb[f"{kind}_probabilities"]
                    + second_perturb[f"{kind}_probabilities"]
                )
                fused_predictions = fused_probabilities.argmax(axis=-1).astype(np.int64)
                fused_perturb_arrays[f"{kind}_names"] = np.asarray(locked_names)
                fused_perturb_arrays[f"{kind}_probabilities"] = fused_probabilities.astype(np.float32)
                fused_perturb_arrays[f"{kind}_pred"] = fused_predictions
                first_drops: list[float] = []
                second_drops: list[float] = []
                first_nominal = float(np.mean(first["pred"] == labels))
                second_nominal = float(np.mean(second["pred"] == labels))
                for index, name in enumerate(locked_names):
                    perturbed_metrics = classification_metrics(
                        labels, fused_predictions[index], n_classes=probability.shape[1]
                    )
                    row = {
                        "dataset": output.name,
                        "model": "atc_fbc_fusion",
                        "subject": subject,
                        "seed": seed,
                        "kind": kind,
                        "name": name,
                        **perturbed_metrics,
                        "nominal_accuracy": float(metrics["accuracy"]),
                        "accuracy_drop": float(metrics["accuracy"] - perturbed_metrics["accuracy"]),
                    }
                    run_perturb_rows.append(row)
                    perturbation_rows.append(row)
                    first_accuracy = float(
                        np.mean(first_perturb[f"{kind}_pred"][index] == labels)
                    )
                    second_accuracy = float(
                        np.mean(second_perturb[f"{kind}_pred"][index] == labels)
                    )
                    first_drops.append(first_nominal - first_accuracy)
                    second_drops.append(second_nominal - second_accuracy)
                drop_profiles[kind] = (first_drops, second_drops)
            save_npz(fusion_dir / "perturbations.npz", **fused_perturb_arrays)
            write_csv(fusion_dir / "perturbation_metrics.csv", run_perturb_rows)
            first_saliency = _saliency_payload(first_dir / "saliency.npz")
            second_saliency = _saliency_payload(second_dir / "saliency.npz")
            if not np.array_equal(first_saliency["channel_names"], second_saliency["channel_names"]):
                raise RuntimeError("fusion component saliency channel bases differ")
            first_profile = first_saliency["normalized"].mean(axis=0)
            second_profile = second_saliency["normalized"].mean(axis=0)
            saliency_correlation = saliency_profile_correlation(first_profile, second_profile)
            flat_profile = {
                "dataset": output.name,
                "subject": subject,
                "seed": seed,
                "atcnet_accuracy": profile["first"]["accuracy"],
                "fbcnet_accuracy": profile["second"]["accuracy"],
                "fusion_accuracy": profile["fusion"]["accuracy"],
                "oracle_accuracy": profile["oracle_accuracy"],
                "disagreement_rate": profile["disagreement_rate"],
                "double_fault_rate": profile["double_fault_rate"],
                "atcnet_only_correct": profile["first_only_correct"],
                "fbcnet_only_correct": profile["second_only_correct"],
                "fusion_only_correct": profile["fusion_only_correct"],
                "fusion_lost_component_correct": profile["fusion_lost_component_correct"],
                "atcnet_entropy": profile["first_entropy"],
                "fbcnet_entropy": profile["second_entropy"],
                "fusion_entropy": profile["fusion_entropy"],
                "fusion_gain_over_best_component": profile["fusion_gain_over_best_component"],
                "saliency_profile_correlation": saliency_correlation,
                "saliency_complementarity": (
                    1.0 - saliency_correlation if np.isfinite(saliency_correlation) else math.nan
                ),
                "frequency_drop_profile_correlation": _profile_correlation(*drop_profiles["frequency"]),
                "region_drop_profile_correlation": _profile_correlation(*drop_profiles["region"]),
                "atcnet_class_recall": json.dumps(profile["first_class_recall"]),
                "fbcnet_class_recall": json.dumps(profile["second_class_recall"]),
                "fusion_class_recall": json.dumps(profile["fusion_class_recall"]),
            }
            write_json(fusion_dir / "error_profile.json", flat_profile)
            error_rows.append(flat_profile)
            fusion_row = {
                "dataset": output.name,
                "model": "atc_fbc_fusion",
                "subject": subject,
                "seed": seed,
                **metrics,
                "negative_log_likelihood": calibration["negative_log_likelihood"],
                "brier_score": calibration["brier_score"],
                "ece": calibration["ece"],
                "parameter_count": int(first_metrics["parameter_count"])
                + int(second_metrics["parameter_count"]),
            }
            write_json(
                fusion_dir / "metrics.json",
                {**fusion_row, "calibration": calibration, "profile": flat_profile},
            )
            fusion_rows.append(fusion_row)
    return fusion_rows, error_rows, perturbation_rows


def _load_baseline_rows(
    output: Path, scope: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    perturbation_rows: list[dict[str, Any]] = []
    saliency_rows: list[dict[str, Any]] = []
    for model in scope["models"]:
        for subject in scope["subjects"]:
            for seed in scope["seeds"]:
                evaluation = _run_base(output, model, subject, seed) / "evaluation"
                metrics = read_json(evaluation / "metrics.json")
                prediction_payload = _prediction_payload(evaluation / "predictions.npz")
                labels = prediction_payload["label"].astype(np.int64)
                prediction = prediction_payload["pred"].astype(np.int64)
                class_recalls = []
                for class_index in range(prediction_payload["probabilities"].shape[1]):
                    selected = labels == class_index
                    class_recalls.append(
                        float(np.mean(prediction[selected] == class_index))
                        if np.any(selected)
                        else math.nan
                    )
                metric_rows.append(
                    {
                        "dataset": output.name,
                        "model": model,
                        "subject": subject,
                        "seed": seed,
                        **{metric: float(metrics[metric]) for metric in METRICS},
                        "negative_log_likelihood": float(
                            metrics["calibration"]["negative_log_likelihood"]
                        ),
                        "brier_score": float(metrics["calibration"]["brier_score"]),
                        "ece": float(metrics["calibration"]["ece"]),
                        "parameter_count": int(metrics["parameter_count"]),
                        "class_recall": json.dumps(class_recalls),
                    }
                )
                for row in _read_csv(evaluation / "perturbation_metrics.csv"):
                    perturbation_rows.append(
                        {
                            "dataset": output.name,
                            "model": model,
                            "subject": subject,
                            "seed": seed,
                            "kind": row["kind"],
                            "name": row["name"],
                            **{metric: float(row[metric]) for metric in METRICS},
                            "nominal_accuracy": float(row["nominal_accuracy"]),
                            "accuracy_drop": float(row["accuracy_drop"]),
                        }
                    )
                saliency = _saliency_payload(evaluation / "saliency.npz")
                profile = saliency["normalized"].mean(axis=0)
                for channel, value in zip(
                    saliency["channel_names"].astype(str), profile, strict=True
                ):
                    saliency_rows.append(
                        {
                            "dataset": output.name,
                            "model": model,
                            "subject": subject,
                            "seed": seed,
                            "channel": channel,
                            "saliency": float(value),
                        }
                    )
    return metric_rows, perturbation_rows, saliency_rows


def _model_subject_summary(
    rows: list[dict[str, Any]], *, models: list[str], subjects: list[int]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for model in models:
        for subject in subjects:
            selected = [
                row
                for row in rows
                if row["model"] == model and int(row["subject"]) == int(subject)
            ]
            if not selected:
                continue
            recalls = np.asarray(
                [json.loads(str(row["class_recall"])) for row in selected], dtype=float
            )
            mean_recall = np.nanmean(recalls, axis=0)
            accuracy = np.asarray([float(row["accuracy"]) for row in selected])
            summaries.append(
                {
                    "model": model,
                    "subject": subject,
                    "seeds": len(selected),
                    "accuracy_mean": float(accuracy.mean()),
                    "accuracy_seed_std": float(accuracy.std(ddof=1))
                    if accuracy.size > 1
                    else 0.0,
                    "balanced_accuracy_mean": float(
                        np.mean([float(row["balanced_accuracy"]) for row in selected])
                    ),
                    "kappa_mean": float(np.mean([float(row["kappa"]) for row in selected])),
                    "macro_f1_mean": float(
                        np.mean([float(row["macro_f1"]) for row in selected])
                    ),
                    "error_rate": float(1.0 - accuracy.mean()),
                    "class_recall": json.dumps(mean_recall.tolist()),
                    "hardest_class": int(np.nanargmin(mean_recall)),
                    "hardest_class_recall": float(np.nanmin(mean_recall)),
                }
            )
    return summaries


def _subject_means(
    rows: list[dict[str, Any]], *, subjects: list[int], models: list[str], value: str
) -> dict[str, dict[int, float]]:
    result: dict[str, dict[int, float]] = {}
    for model in models:
        result[model] = {}
        for subject in subjects:
            selected = [
                float(row[value])
                for row in rows
                if row["model"] == model and int(row["subject"]) == int(subject)
            ]
            if selected:
                result[model][subject] = float(np.mean(selected))
    return result


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def _summarise_models(
    rows: list[dict[str, Any]],
    *,
    cohorts: dict[str, list[int]],
    models: list[str],
    bootstrap_repeats: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for cohort, subjects in cohorts.items():
        means = _subject_means(rows, subjects=subjects, models=models, value="accuracy")
        for model in models:
            values = np.asarray([means[model][subject] for subject in subjects if subject in means[model]])
            interval = bootstrap_ci(
                values,
                seed=7001,
                n_boot=bootstrap_repeats,
            )
            model_rows = [row for row in rows if row["model"] == model]
            summaries.append(
                {
                    "cohort": cohort,
                    "model": model,
                    "subject_macro_accuracy": interval["mean"],
                    "ci_low": interval["low"],
                    "ci_high": interval["high"],
                    "subject_std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "subjects": int(values.size),
                    "seeds_per_subject": len({int(row["seed"]) for row in model_rows}),
                    "parameter_count": int(model_rows[0]["parameter_count"]),
                }
            )
        raw_p: list[float] = []
        cohort_comparisons: list[dict[str, Any]] = []
        fusion = means["atc_fbc_fusion"]
        for model in models:
            if model == "atc_fbc_fusion":
                continue
            paired_subjects = [
                subject for subject in subjects if subject in fusion and subject in means[model]
            ]
            delta = np.asarray(
                [fusion[subject] - means[model][subject] for subject in paired_subjects],
                dtype=float,
            )
            interval = bootstrap_ci(delta, seed=8101, n_boot=bootstrap_repeats)
            if np.allclose(delta, 0.0):
                p_value = 1.0
            else:
                p_value = float(stats.wilcoxon(delta, alternative="two-sided").pvalue)
            raw_p.append(p_value)
            cohort_comparisons.append(
                {
                    "cohort": cohort,
                    "first": "atc_fbc_fusion",
                    "second": model,
                    "mean_delta_pp": 100.0 * interval["mean"],
                    "ci_low_pp": 100.0 * interval["low"],
                    "ci_high_pp": 100.0 * interval["high"],
                    "positive_subjects": int(np.count_nonzero(delta > 0.0)),
                    "negative_subjects": int(np.count_nonzero(delta < 0.0)),
                    "ties": int(np.count_nonzero(delta == 0.0)),
                    "subjects": int(delta.size),
                    "wilcoxon_p": p_value,
                }
            )
        for row, adjusted in zip(cohort_comparisons, _holm_adjust(raw_p), strict=True):
            row["holm_p"] = adjusted
        comparisons.extend(cohort_comparisons)
    return summaries, comparisons


def _summarise_perturbations(
    rows: list[dict[str, Any]],
    *,
    cohorts: dict[str, list[int]],
    models: list[str],
    bootstrap_repeats: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for cohort, subjects in cohorts.items():
        for model in models:
            for kind, names in (
                ("frequency", list(FREQUENCY_OCCLUSIONS_HZ)),
                ("region", list(REGION_CHANNELS)),
            ):
                for name in names:
                    subject_values = []
                    for subject in subjects:
                        values = [
                            float(row["accuracy_drop"])
                            for row in rows
                            if row["model"] == model
                            and int(row["subject"]) == int(subject)
                            and row["kind"] == kind
                            and row["name"] == name
                        ]
                        if values:
                            subject_values.append(float(np.mean(values)))
                    interval = bootstrap_ci(
                        np.asarray(subject_values), seed=9101, n_boot=bootstrap_repeats
                    )
                    summaries.append(
                        {
                            "cohort": cohort,
                            "model": model,
                            "kind": kind,
                            "name": name,
                            "mean_accuracy_drop_pp": 100.0 * interval["mean"],
                            "ci_low_pp": 100.0 * interval["low"],
                            "ci_high_pp": 100.0 * interval["high"],
                            "subjects": interval["n"],
                        }
                    )
    return summaries


def _summarise_saliency(
    rows: list[dict[str, Any]],
    *,
    cohorts: dict[str, list[int]],
    models: list[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    channels = list(dict.fromkeys(row["channel"] for row in rows))
    for cohort, subjects in cohorts.items():
        for model in models:
            if model == "atc_fbc_fusion":
                continue
            for channel in channels:
                subject_values = []
                for subject in subjects:
                    values = [
                        float(row["saliency"])
                        for row in rows
                        if row["model"] == model
                        and int(row["subject"]) == int(subject)
                        and row["channel"] == channel
                    ]
                    if values:
                        subject_values.append(float(np.mean(values)))
                output.append(
                    {
                        "cohort": cohort,
                        "model": model,
                        "channel": channel,
                        "mean_normalized_saliency": float(np.mean(subject_values)),
                        "subject_std": float(np.std(subject_values, ddof=1))
                        if len(subject_values) > 1
                        else 0.0,
                        "subjects": len(subject_values),
                    }
                )
    return output


def _subject_error_summary(
    error_rows: list[dict[str, Any]], cohorts: dict[str, list[int]]
) -> list[dict[str, Any]]:
    numeric = (
        "atcnet_accuracy",
        "fbcnet_accuracy",
        "fusion_accuracy",
        "oracle_accuracy",
        "disagreement_rate",
        "double_fault_rate",
        "atcnet_only_correct",
        "fbcnet_only_correct",
        "fusion_only_correct",
        "fusion_lost_component_correct",
        "atcnet_entropy",
        "fbcnet_entropy",
        "fusion_entropy",
        "fusion_gain_over_best_component",
        "saliency_profile_correlation",
        "saliency_complementarity",
        "frequency_drop_profile_correlation",
        "region_drop_profile_correlation",
    )
    summaries: list[dict[str, Any]] = []
    for cohort, subjects in cohorts.items():
        for subject in subjects:
            selected = [row for row in error_rows if int(row["subject"]) == subject]
            if not selected:
                continue
            summary = {"cohort": cohort, "subject": subject, "seeds": len(selected)}
            for name in numeric:
                values = np.asarray([float(row[name]) for row in selected], dtype=float)
                summary[name] = float(np.nanmean(values)) if np.isfinite(values).any() else math.nan
            summary["oracle_headroom_over_best"] = float(
                summary["oracle_accuracy"]
                - max(summary["atcnet_accuracy"], summary["fbcnet_accuracy"])
            )
            summaries.append(summary)
    return summaries


def _mechanism_correlations(
    subject_rows: list[dict[str, Any]], cohorts: dict[str, list[int]]
) -> list[dict[str, Any]]:
    predictors = (
        "disagreement_rate",
        "double_fault_rate",
        "oracle_headroom_over_best",
        "saliency_complementarity",
        "frequency_drop_profile_correlation",
        "region_drop_profile_correlation",
    )
    results: list[dict[str, Any]] = []
    for cohort, subjects in cohorts.items():
        selected = [row for row in subject_rows if row["cohort"] == cohort and row["subject"] in subjects]
        target = np.asarray([row["fusion_gain_over_best_component"] for row in selected])
        for predictor in predictors:
            value = np.asarray([row[predictor] for row in selected], dtype=float)
            finite = np.isfinite(value) & np.isfinite(target)
            if np.count_nonzero(finite) >= 3 and not np.isclose(np.std(value[finite]), 0.0):
                result = stats.spearmanr(value[finite], target[finite])
                rho, p_value = float(result.statistic), float(result.pvalue)
            else:
                rho, p_value = math.nan, math.nan
            results.append(
                {
                    "cohort": cohort,
                    "predictor": predictor,
                    "target": "fusion_gain_over_best_component",
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "subjects": int(np.count_nonzero(finite)),
                }
            )
    return results


def _figures(
    analysis_dir: Path,
    *,
    dataset: str,
    primary_cohort: str,
    model_summary: list[dict[str, Any]],
    perturbation_summary: list[dict[str, Any]],
    saliency_summary: list[dict[str, Any]],
    subject_errors: list[dict[str, Any]],
    models: list[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = ensure_dir(analysis_dir / "figures")
    selected = [row for row in model_summary if row["cohort"] == primary_cohort]
    selected.sort(key=lambda row: models.index(row["model"]))
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    positions = np.arange(len(selected))
    values = np.asarray([100.0 * row["subject_macro_accuracy"] for row in selected])
    low = values - np.asarray([100.0 * row["ci_low"] for row in selected])
    high = np.asarray([100.0 * row["ci_high"] for row in selected]) - values
    axis.errorbar(
        positions,
        values,
        yerr=np.vstack((low, high)),
        fmt="none",
        ecolor="#333333",
        capsize=4,
        linewidth=1.2,
        zorder=1,
    )
    axis.scatter(
        positions,
        values,
        s=70,
        c=[MODEL_COLORS[row["model"]] for row in selected],
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )
    axis.set_xticks(positions, [MODEL_LABELS[row["model"]] for row in selected], rotation=25, ha="right")
    axis.set_ylabel("Subject-macro accuracy (%)")
    axis.set_title(f"{dataset.upper()} equal-budget held-out comparison")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_dir / "baseline_accuracy.png", dpi=300)
    figure.savefig(figure_dir / "baseline_accuracy.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for axis, (kind, names, title) in zip(
        axes,
        (
            ("frequency", list(FREQUENCY_OCCLUSIONS_HZ), "Frequency-band occlusion"),
            ("region", list(REGION_CHANNELS), "Regional sensor occlusion"),
        ),
        strict=True,
    ):
        matrix = np.full((len(models), len(names)), np.nan)
        for model_index, model in enumerate(models):
            for name_index, name in enumerate(names):
                match = [
                    row
                    for row in perturbation_summary
                    if row["cohort"] == primary_cohort
                    and row["model"] == model
                    and row["kind"] == kind
                    and row["name"] == name
                ]
                if match:
                    matrix[model_index, name_index] = match[0]["mean_accuracy_drop_pp"]
        maximum = max(1.0, float(np.nanmax(np.abs(matrix))))
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-maximum, vmax=maximum, aspect="auto")
        axis.set_xticks(np.arange(len(names)), [name.replace("_", "\n") for name in names], fontsize=8)
        axis.set_yticks(np.arange(len(models)), [MODEL_LABELS[model] for model in models])
        axis.set_title(title)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(matrix[row_index, column_index]) > maximum * 0.55 else "black",
                )
        figure.colorbar(image, ax=axis, shrink=0.78, label="Accuracy drop (pp)")
    figure.savefig(figure_dir / "frequency_region_occlusion.png", dpi=300)
    figure.savefig(figure_dir / "frequency_region_occlusion.pdf")
    plt.close(figure)

    selected_errors = [row for row in subject_errors if row["cohort"] == primary_cohort]
    figure, axis = plt.subplots(figsize=(6.2, 4.8))
    x = 100.0 * np.asarray([row["disagreement_rate"] for row in selected_errors])
    y = 100.0 * np.asarray(
        [row["fusion_gain_over_best_component"] for row in selected_errors]
    )
    axis.scatter(x, y, c="#C7352C", alpha=0.78, edgecolor="white", linewidth=0.6)
    if x.size >= 2 and not np.isclose(np.std(x), 0.0):
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 100)
        axis.plot(grid, slope * grid + intercept, color="#333333", linewidth=1.2)
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.set_xlabel("ATCNet/FBCNet disagreement (%)")
    axis.set_ylabel("Fusion gain over best component (pp)")
    axis.set_title("Subject-level fusion complementarity")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(figure_dir / "fusion_subject_complementarity.png", dpi=300)
    figure.savefig(figure_dir / "fusion_subject_complementarity.pdf")
    plt.close(figure)

    try:
        import mne

        top_models = [model for model in ("atcnet", "fbcnet") if model in models]
        channel_names = list(dict.fromkeys(row["channel"] for row in saliency_summary))
        info = mne.create_info(channel_names, sfreq=250.0, ch_types="eeg")
        info.set_montage(mne.channels.make_standard_montage("standard_1020"))
        profiles = []
        for model in top_models:
            profiles.append(
                np.asarray(
                    [
                        next(
                            row["mean_normalized_saliency"]
                            for row in saliency_summary
                            if row["cohort"] == primary_cohort
                            and row["model"] == model
                            and row["channel"] == channel
                        )
                        for channel in channel_names
                    ]
                )
            )
        values = profiles + ([profiles[0] - profiles[1]] if len(profiles) == 2 else [])
        titles = [MODEL_LABELS[model] for model in top_models] + (
            ["ATCNet - FBCNet"] if len(profiles) == 2 else []
        )
        figure, axes = plt.subplots(1, len(values), figsize=(4.0 * len(values), 3.8))
        axes = np.atleast_1d(axes)
        for index, (axis, profile, title) in enumerate(zip(axes, values, titles, strict=True)):
            if index < len(profiles):
                mne.viz.plot_topomap(profile, info, axes=axis, show=False, cmap="viridis")
            else:
                limit = max(float(np.max(np.abs(profile))), 1e-6)
                mne.viz.plot_topomap(
                    profile, info, axes=axis, show=False, cmap="RdBu_r", vlim=(-limit, limit)
                )
            axis.set_title(title)
        figure.suptitle(f"{dataset.upper()} true-class gradient x input saliency")
        figure.tight_layout()
        figure.savefig(figure_dir / "channel_topography.png", dpi=300)
        figure.savefig(figure_dir / "channel_topography.pdf")
        plt.close(figure)
    except Exception as exc:
        write_json(
            figure_dir / "channel_topography_error.json",
            {"status": "failed", "error": str(exc)},
        )


def _report(
    analysis_dir: Path,
    *,
    dataset: str,
    primary_cohort: str,
    summaries: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    perturbations: list[dict[str, Any]],
    mechanism: list[dict[str, Any]],
) -> None:
    cohort_models = [row for row in summaries if row["cohort"] == primary_cohort]
    ranked = sorted(cohort_models, key=lambda row: row["subject_macro_accuracy"], reverse=True)
    fusion = next(row for row in cohort_models if row["model"] == "atc_fbc_fusion")
    frequency = [
        row
        for row in perturbations
        if row["cohort"] == primary_cohort
        and row["model"] == "atc_fbc_fusion"
        and row["kind"] == "frequency"
    ]
    region = [
        row
        for row in perturbations
        if row["cohort"] == primary_cohort
        and row["model"] == "atc_fbc_fusion"
        and row["kind"] == "region"
    ]
    strongest_frequency = max(frequency, key=lambda row: row["mean_accuracy_drop_pp"])
    strongest_region = max(region, key=lambda row: row["mean_accuracy_drop_pp"])
    mechanism_lines = [
        f"- `{row['predictor']}` vs fusion gain: Spearman rho={row['spearman_rho']:.3f}, "
        f"p={row['p_value']:.4g}, n={row['subjects']}."
        for row in mechanism
        if row["cohort"] == primary_cohort
    ]
    comparison_lines = [
        f"- Fusion minus {MODEL_LABELS[row['second']]}: {row['mean_delta_pp']:+.3f} pp "
        f"(95% CI {row['ci_low_pp']:+.3f} to {row['ci_high_pp']:+.3f}; "
        f"Holm p={row['holm_p']:.4g})."
        for row in comparisons
        if row["cohort"] == primary_cohort
    ]
    text = f"""# {dataset.upper()} Publication Baseline And Fusion Analysis

## Scope

This is a post-hoc explanatory analysis performed after the held-out session had already been opened. Seeds were averaged within subject before inference; subjects are the inferential unit.

## Equal-Budget Accuracy

Best model: **{MODEL_LABELS[ranked[0]['model']]}**, subject-macro accuracy {100.0 * ranked[0]['subject_macro_accuracy']:.3f}% (95% bootstrap CI {100.0 * ranked[0]['ci_low']:.3f}-{100.0 * ranked[0]['ci_high']:.3f}%).

Fixed ATCNet/FBCNet fusion: {100.0 * fusion['subject_macro_accuracy']:.3f}% (95% CI {100.0 * fusion['ci_low']:.3f}-{100.0 * fusion['ci_high']:.3f}%).

{chr(10).join(comparison_lines)}

## Physiological Perturbations

The largest fusion frequency-occlusion drop was `{strongest_frequency['name']}` at {strongest_frequency['mean_accuracy_drop_pp']:.3f} pp. The largest regional-occlusion drop was `{strongest_region['name']}` at {strongest_region['mean_accuracy_drop_pp']:.3f} pp. Negative drops mean the occlusion improved accuracy and must not be interpreted as positive evidence. FFT deletion is a global post-hoc perturbation and zero-reference sensor masking creates distribution shift, so these observables localise dependence rather than establish causal neurophysiology.

## Why Fusion Helps Or Fails

{chr(10).join(mechanism_lines)}

Fusion is mechanistically supported only where its gain co-occurs with component disagreement, oracle headroom, complementary saliency, or complementary perturbation profiles. Correlation is explanatory, not causal; all adverse and null relations remain in the tables.
"""
    (analysis_dir / "REPORT.md").write_text(text, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset", choices=("bci2a", "openbmi"), required=True)
    parser.add_argument("--canary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = Path(args.output_root) / ("canary" if args.canary else "") / args.dataset
    audit = read_json(output / "audit" / "audit_report.json")
    if audit.get("status") != "passed":
        raise RuntimeError("baseline analysis requires a passed independent audit")
    contract = read_json(output / "contract.json")
    scope = dict(contract["scope"])
    config = dict(contract["resolved_config"])
    if not {"atcnet", "fbcnet"}.issubset(scope["models"]):
        raise RuntimeError("fusion analysis requires both ATCNet and FBCNet")
    analysis_dir = ensure_dir(output / "analysis")
    baseline_rows, baseline_perturbations, saliency_rows = _load_baseline_rows(output, scope)
    fusion_rows, error_rows, fusion_perturbations = _build_fusion(output, scope)
    metric_rows = baseline_rows + fusion_rows
    perturbation_rows = baseline_perturbations + fusion_perturbations
    models = [*scope["models"], "atc_fbc_fusion"]
    cohorts = _cohorts(args.dataset, config, scope["subjects"])
    bootstrap_repeats = int(config["analysis"]["bootstrap_repeats"])
    model_summary, comparisons = _summarise_models(
        metric_rows,
        cohorts=cohorts,
        models=models,
        bootstrap_repeats=bootstrap_repeats,
    )
    perturbation_summary = _summarise_perturbations(
        perturbation_rows,
        cohorts=cohorts,
        models=models,
        bootstrap_repeats=bootstrap_repeats,
    )
    saliency_summary = _summarise_saliency(
        saliency_rows, cohorts=cohorts, models=models
    )
    model_subject_errors = _model_subject_summary(
        baseline_rows, models=scope["models"], subjects=scope["subjects"]
    )
    subject_errors = _subject_error_summary(error_rows, cohorts)
    mechanism = _mechanism_correlations(subject_errors, cohorts)
    write_csv(analysis_dir / "all_model_subject_seed_metrics.csv", metric_rows)
    write_csv(analysis_dir / "model_cohort_summary.csv", model_summary)
    write_csv(analysis_dir / "fusion_paired_comparisons.csv", comparisons)
    write_csv(analysis_dir / "perturbation_subject_seed.csv", perturbation_rows)
    write_csv(analysis_dir / "perturbation_cohort_summary.csv", perturbation_summary)
    write_csv(analysis_dir / "channel_saliency_subject_seed.csv", saliency_rows)
    write_csv(analysis_dir / "channel_saliency_cohort_summary.csv", saliency_summary)
    write_csv(analysis_dir / "baseline_subject_error_summary.csv", model_subject_errors)
    write_csv(analysis_dir / "fusion_error_subject_seed.csv", error_rows)
    write_csv(analysis_dir / "fusion_error_subject_summary.csv", subject_errors)
    write_csv(analysis_dir / "fusion_mechanism_correlations.csv", mechanism)
    primary_cohort = "full_9" if args.dataset == "bci2a" else "full_54"
    if primary_cohort not in cohorts:
        primary_cohort = next(iter(cohorts))
    _figures(
        analysis_dir,
        dataset=args.dataset,
        primary_cohort=primary_cohort,
        model_summary=model_summary,
        perturbation_summary=perturbation_summary,
        saliency_summary=saliency_summary,
        subject_errors=subject_errors,
        models=models,
    )
    _report(
        analysis_dir,
        dataset=args.dataset,
        primary_cohort=primary_cohort,
        summaries=model_summary,
        comparisons=comparisons,
        perturbations=perturbation_summary,
        mechanism=mechanism,
    )
    status = {
        "schema": "dpc-snn-v8-posthoc-publication-analysis/v1",
        "status": "completed",
        "dataset": args.dataset,
        "posthoc_explanatory": True,
        "cohorts": cohorts,
        "models": models,
        "subject_seed_rows": len(metric_rows),
        "fusion_runs": len(fusion_rows),
    }
    write_json(analysis_dir / "analysis_status.json", status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
