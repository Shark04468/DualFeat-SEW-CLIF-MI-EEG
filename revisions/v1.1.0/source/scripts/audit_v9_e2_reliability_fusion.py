#!/usr/bin/env python3
"""Independently audit V9 E0-E2 splits, fusion outputs, and reported metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    validate_fingerprint_manifest,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402


GATE_VARIANTS = ("global_static", "class_static", "dynamic_global", "class_dynamic")
OUTPUT_ARMS = (
    "atcnet_raw",
    "fbcnet_raw",
    "equal_raw",
    "atcnet_calibrated",
    "fbcnet_calibrated",
    "equal_calibrated",
    *GATE_VARIANTS,
)
EPSILON = 1e-9


def _archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float64) / float(temperature)
    value -= value.max(axis=1, keepdims=True)
    exponential = np.exp(value)
    return exponential / exponential.sum(axis=1, keepdims=True)


def _features(atc: np.ndarray, fbc: np.ndarray) -> np.ndarray:
    classes = atc.shape[1]
    normalizer = math.log(classes)
    atc_entropy = -(atc * np.log(np.clip(atc, EPSILON, 1.0))).sum(1)
    fbc_entropy = -(fbc * np.log(np.clip(fbc, EPSILON, 1.0))).sum(1)
    atc_sorted = np.sort(atc, axis=1)
    fbc_sorted = np.sort(fbc, axis=1)
    midpoint = 0.5 * (atc + fbc)
    atc_kl = (atc * (np.log(np.clip(atc, EPSILON, 1.0)) - np.log(midpoint))).sum(1)
    fbc_kl = (fbc * (np.log(np.clip(fbc, EPSILON, 1.0)) - np.log(midpoint))).sum(1)
    return np.column_stack(
        (
            (fbc_entropy - atc_entropy) / normalizer,
            (atc_sorted[:, -1] - atc_sorted[:, -2])
            - (fbc_sorted[:, -1] - fbc_sorted[:, -2]),
            atc.max(1) - fbc.max(1),
            0.5 * (atc_kl + fbc_kl) / normalizer,
            (atc.argmax(1) != fbc.argmax(1)).astype(np.float64),
        )
    )


def _gate_weight(
    variant: str,
    parameters: np.ndarray,
    features: np.ndarray,
    classes: int,
) -> np.ndarray:
    trials = features.shape[0]
    if variant == "global_static":
        eta = np.full((trials, classes), parameters[0])
    elif variant == "class_static":
        eta = np.broadcast_to(parameters[None, :], (trials, classes))
    elif variant == "dynamic_global":
        eta = np.broadcast_to(
            (parameters[0] + features @ parameters[1:])[:, None], (trials, classes)
        )
    elif variant == "class_dynamic":
        eta = parameters[:classes][None, :] + (features @ parameters[classes:])[:, None]
    else:
        raise RuntimeError(f"unknown gate variant in audit: {variant}")
    eta = np.clip(eta, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-eta))


def _mixture(atc: np.ndarray, fbc: np.ndarray, weight: np.ndarray) -> np.ndarray:
    score = np.clip(weight * atc + (1.0 - weight) * fbc, EPSILON, None)
    return score / score.sum(axis=1, keepdims=True)


def _temperature_nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probability = _softmax(logits, temperature)
    return float(-np.log(np.clip(probability[np.arange(labels.size), labels], EPSILON, 1.0)).mean())


def _metrics(probability: np.ndarray, labels: np.ndarray, n_bins: int) -> dict[str, float]:
    prediction = probability.argmax(1)
    classes = probability.shape[1]
    confusion = np.zeros((classes, classes), dtype=np.int64)
    for truth, pred in zip(labels, prediction, strict=True):
        confusion[int(truth), int(pred)] += 1
    accuracy = float(np.trace(confusion) / confusion.sum())
    recalls = [
        float(confusion[index, index] / confusion[index].sum())
        if confusion[index].sum()
        else 0.0
        for index in range(classes)
    ]
    f1_values = []
    for index in range(classes):
        true_positive = confusion[index, index]
        false_positive = confusion[:, index].sum() - true_positive
        false_negative = confusion[index, :].sum() - true_positive
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1_values.append(
            2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    row = confusion.sum(axis=1).astype(np.float64)
    column = confusion.sum(axis=0).astype(np.float64)
    expected = float((row * column).sum() / (confusion.sum() ** 2))
    kappa = float((accuracy - expected) / (1.0 - expected)) if expected < 1.0 else 0.0
    confidence = probability.max(1)
    correct = prediction == labels
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    maximum_gap = 0.0
    for index in range(int(n_bins)):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidence >= lower) & (
            confidence <= upper if index == int(n_bins) - 1 else confidence < upper
        )
        if selected.any():
            gap = abs(float(correct[selected].mean() - confidence[selected].mean()))
            ece += float(selected.mean()) * gap
            maximum_gap = max(maximum_gap, gap)
    selected_probability = np.clip(
        probability[np.arange(labels.size), labels], EPSILON, 1.0
    )
    one_hot = np.eye(classes)[labels]
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "kappa": kappa,
        "macro_f1": float(np.mean(f1_values)),
        "negative_log_likelihood": float(-np.log(selected_probability).mean()),
        "brier_score": float(np.square(probability - one_hot).sum(1).mean()),
        "ece": float(ece),
        "maximum_calibration_error": float(maximum_gap),
    }


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _assert_close(observed: Any, expected: Any, message: str, tolerance: float = 2e-8) -> None:
    if not np.allclose(np.asarray(observed), np.asarray(expected), atol=tolerance, rtol=tolerance):
        raise RuntimeError(message)


def audit(output: Path) -> dict[str, Any]:
    status = read_json(output / "campaign_status.json")
    config = read_json(output / "resolved_config.json")
    fingerprint = validate_fingerprint_manifest(read_json(output / "run_fingerprint.json"))
    if status.get("status") != "completed" or status.get("session_e_accessed") is not False:
        raise RuntimeError("V9 campaign is incomplete or reports held-out access")
    if status.get("fingerprint") != fingerprint["combined_sha256"]:
        raise RuntimeError("campaign status fingerprint differs from the locked run fingerprint")

    source_manifest = read_json(output / "source_manifest.json")
    for relative, digest in source_manifest.items():
        path = ROOT / relative
        if not path.is_file() or file_sha256(path) != digest:
            raise RuntimeError(f"source changed after V9 execution: {relative}")
    input_manifest = read_json(output / "input_manifest.json")["files"]
    e1_root = Path(config["e1_root"])
    for row in input_manifest:
        path = e1_root / row["path"]
        if (
            not path.is_file()
            or int(path.stat().st_size) != int(row["size_bytes"])
            or file_sha256(path) != row["sha256"]
        ):
            raise RuntimeError(f"teacher input changed after V9 execution: {path}")

    subjects = [int(value) for value in config["subjects"]]
    seeds = [int(value) for value in config["seeds"]]
    folds = [int(value) for value in config["folds"]]
    classes = int(config["n_classes"])
    trials = int(config["expected_trials_per_subject"])
    ece_bins = int(config["ece_bins"])
    metric_rows = _csv_rows(output / "per_subject_seed_metrics.csv")
    metric_lookup = {
        (int(row["subject"]), int(row["seed"]), row["arm"]): row for row in metric_rows
    }
    independently_recomputed: dict[tuple[int, int, str], dict[str, float]] = {}
    checked_folds = 0
    checked_trials = 0
    gate_convergence: dict[str, list[bool]] = {variant: [] for variant in GATE_VARIANTS}

    for subject in subjects:
        for seed in seeds:
            base = {
                model: e1_root / model / f"subject_{subject:02d}" / f"seed_{seed}"
                for model in ("atcnet", "fbcnet")
            }
            full = {model: _archive(path / "predictions.npz") for model, path in base.items()}
            for field in ("label", "subject", "session", "run", "trial_id", "seed"):
                if not np.array_equal(full["atcnet"][field], full["fbcnet"][field]):
                    raise RuntimeError(f"teacher metadata mismatch during audit: {field}")
            if set(full["atcnet"]["session"].astype(str).tolist()) != {"T"}:
                raise RuntimeError("audit encountered a held-out session")
            labels = np.asarray(full["atcnet"]["label"], dtype=np.int64)
            runs = full["atcnet"]["run"].astype(str)
            assembled = {
                arm: np.full((trials, classes), np.nan, dtype=np.float64)
                for arm in OUTPUT_ARMS
            }
            seen = np.zeros(trials, dtype=bool)

            for fold in folds:
                fit = read_json(output / f"subject_{subject:02d}" / f"seed_{seed}" / f"fold_{fold}" / "fit.json")
                saved = _archive(
                    output
                    / f"subject_{subject:02d}"
                    / f"seed_{seed}"
                    / f"fold_{fold}"
                    / "outer_predictions.npz"
                )
                source = {
                    model: {
                        role: base[model] / f"fold_{fold}" / f"{role}_predictions.npz"
                        for role in ("selection", "outer_test")
                    }
                    for model in ("atcnet", "fbcnet")
                }
                for model in source:
                    if file_sha256(source[model]["selection"]) != fit["source_sha256"][model]["selection"]:
                        raise RuntimeError("selection source hash differs from fit manifest")
                    if file_sha256(source[model]["outer_test"]) != fit["source_sha256"][model]["outer"]:
                        raise RuntimeError("outer source hash differs from fit manifest")
                selection = {model: _archive(paths["selection"]) for model, paths in source.items()}
                outer = {model: _archive(paths["outer_test"]) for model, paths in source.items()}
                for role_data in (selection, outer):
                    if not np.array_equal(role_data["atcnet"]["indices"], role_data["fbcnet"]["indices"]):
                        raise RuntimeError("teacher fold indices differ during audit")
                    if not np.array_equal(role_data["atcnet"]["labels"], role_data["fbcnet"]["labels"]):
                        raise RuntimeError("teacher fold labels differ during audit")
                selection_indices = np.asarray(selection["atcnet"]["indices"], dtype=np.int64)
                outer_indices = np.asarray(outer["atcnet"]["indices"], dtype=np.int64)
                if np.intersect1d(selection_indices, outer_indices).size or seen[outer_indices].any():
                    raise RuntimeError("audit found inner/outer overlap or duplicate outer coverage")
                if set(runs[selection_indices]) & set(runs[outer_indices]):
                    raise RuntimeError("audit found run leakage between fit and evaluation")
                if not np.array_equal(saved["indices"], outer_indices) or not np.array_equal(
                    saved["labels"], labels[outer_indices]
                ):
                    raise RuntimeError("saved V9 fold identities differ from teacher fold")

                temperatures: dict[str, float] = {}
                for model in ("atcnet", "fbcnet"):
                    temperature = float(fit["calibration"][model]["temperature"])
                    grid = [float(value) for value in config["temperature_grid"]]
                    losses = [
                        _temperature_nll(
                            selection[model]["logits"], selection[model]["labels"], value
                        )
                        for value in grid
                    ]
                    selected_index = min(
                        range(len(grid)),
                        key=lambda index: (losses[index], abs(math.log(grid[index])), grid[index]),
                    )
                    if temperature != grid[selected_index]:
                        raise RuntimeError("saved fold temperature is not the inner-validation optimum")
                    temperatures[model] = temperature
                selection_probability = {
                    model: _softmax(selection[model]["logits"], temperatures[model])
                    for model in ("atcnet", "fbcnet")
                }
                outer_raw = {
                    model: _softmax(outer[model]["logits"]) for model in ("atcnet", "fbcnet")
                }
                outer_probability = {
                    model: _softmax(outer[model]["logits"], temperatures[model])
                    for model in ("atcnet", "fbcnet")
                }
                expected = {
                    "atcnet_raw": outer_raw["atcnet"],
                    "fbcnet_raw": outer_raw["fbcnet"],
                    "equal_raw": 0.5 * (outer_raw["atcnet"] + outer_raw["fbcnet"]),
                    "atcnet_calibrated": outer_probability["atcnet"],
                    "fbcnet_calibrated": outer_probability["fbcnet"],
                    "equal_calibrated": 0.5
                    * (outer_probability["atcnet"] + outer_probability["fbcnet"]),
                }
                raw_selection_features = _features(
                    selection_probability["atcnet"], selection_probability["fbcnet"]
                )
                raw_outer_features = _features(
                    outer_probability["atcnet"], outer_probability["fbcnet"]
                )
                for variant in GATE_VARIANTS:
                    gate = fit["gates"][variant]
                    gate_convergence[variant].append(bool(gate["converged"]))
                    mean = np.asarray(gate["feature_scaler"]["mean"], dtype=np.float64)
                    scale = np.asarray(gate["feature_scaler"]["scale"], dtype=np.float64)
                    independently_fitted_mean = raw_selection_features.mean(0)
                    independently_fitted_scale = raw_selection_features.std(0)
                    independently_fitted_scale = np.where(
                        independently_fitted_scale < 1e-8, 1.0, independently_fitted_scale
                    )
                    _assert_close(mean, independently_fitted_mean, "gate feature mean is not fold-local")
                    _assert_close(scale, independently_fitted_scale, "gate feature scale is not fold-local")
                    features = (raw_outer_features - mean[None, :]) / scale[None, :]
                    weight = _gate_weight(
                        variant,
                        np.asarray(gate["parameters"], dtype=np.float64),
                        features,
                        classes,
                    )
                    expected[variant] = _mixture(
                        outer_probability["atcnet"], outer_probability["fbcnet"], weight
                    )
                    _assert_close(saved[f"weight_{variant}"], weight, f"{variant} weight mismatch")
                for arm in OUTPUT_ARMS:
                    _assert_close(
                        saved[f"probability_{arm}"],
                        expected[arm],
                        f"{arm} fold probability mismatch",
                    )
                    assembled[arm][outer_indices] = expected[arm]
                seen[outer_indices] = True
                checked_folds += 1
                checked_trials += int(outer_indices.size)

            if not seen.all() or any(not np.isfinite(value).all() for value in assembled.values()):
                raise RuntimeError("audit did not recover complete OOF coverage")
            saved_subject = _archive(output / f"subject_{subject:02d}" / f"seed_{seed}" / "predictions.npz")
            for field in ("label", "subject", "session", "run", "trial_id", "seed"):
                if not np.array_equal(saved_subject[field], full["atcnet"][field]):
                    raise RuntimeError(f"saved V9 subject metadata differs on {field}")
            for arm, probability in assembled.items():
                _assert_close(
                    saved_subject[f"probability_{arm}"],
                    probability,
                    f"assembled {arm} subject archive mismatch",
                )
                if not np.array_equal(saved_subject[f"pred_{arm}"], probability.argmax(1)):
                    raise RuntimeError(f"saved {arm} predictions are inconsistent")
                metrics = _metrics(probability, labels, ece_bins)
                independently_recomputed[(subject, seed, arm)] = metrics
                reported = metric_lookup[(subject, seed, arm)]
                for name, value in metrics.items():
                    _assert_close(
                        float(reported[name]),
                        value,
                        f"reported metric differs: S{subject} seed{seed} {arm} {name}",
                        tolerance=5e-8,
                    )

    aggregate_rows = _csv_rows(output / "aggregate_metrics.csv")
    aggregate_lookup = {row["arm"]: row for row in aggregate_rows}
    for arm in OUTPUT_ARMS:
        selected = [
            metrics for (subject, seed, name), metrics in independently_recomputed.items() if name == arm
        ]
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "kappa",
            "macro_f1",
            "negative_log_likelihood",
            "brier_score",
            "ece",
        ):
            _assert_close(
                float(aggregate_lookup[arm][f"mean_{metric}"]),
                np.mean([row[metric] for row in selected]),
                f"aggregate metric differs for {arm} {metric}",
            )

    selected_candidate = max(
        GATE_VARIANTS,
        key=lambda arm: (
            float(aggregate_lookup[arm]["mean_accuracy"]),
            -float(aggregate_lookup[arm]["mean_negative_log_likelihood"]),
            -int(aggregate_lookup[arm]["parameters"]),
        ),
    )
    gate = read_json(output / "e2_promotion_gate.json")
    if gate["selected_candidate"] != selected_candidate or gate.get("session_e_accessed") is not False:
        raise RuntimeError("promotion gate candidate or held-out status is invalid")
    deltas = []
    nll_deltas = []
    for subject in subjects:
        for seed in seeds:
            candidate = independently_recomputed[(subject, seed, selected_candidate)]
            reference = independently_recomputed[(subject, seed, "equal_calibrated")]
            deltas.append(100.0 * (candidate["accuracy"] - reference["accuracy"]))
            nll_deltas.append(candidate["negative_log_likelihood"] - reference["negative_log_likelihood"])
    positives = sum(value > 0.0 for value in deltas)
    criteria = gate["criteria"]
    independently_passed = (
        float(np.median(deltas)) >= float(criteria["minimum_median_accuracy_delta_pp"])
        and positives >= int(criteria["minimum_positive_pairs"])
        and float(np.mean(nll_deltas)) <= float(criteria["maximum_mean_nll_delta"])
        and (
            all(gate_convergence[selected_candidate])
            or not bool(criteria["require_all_selected_gate_fits_converged"])
        )
    )
    if bool(gate["all_selected_gate_fits_converged"]) != all(
        gate_convergence[selected_candidate]
    ):
        raise RuntimeError("promotion gate convergence status does not reproduce")
    if bool(gate["passed"]) != bool(independently_passed):
        raise RuntimeError("promotion gate decision does not reproduce")

    return {
        "status": "passed",
        "campaign_fingerprint": fingerprint["combined_sha256"],
        "source_files_verified": len(source_manifest),
        "teacher_input_files_verified": len(input_manifest),
        "subject_seed_pairs_verified": len(subjects) * len(seeds),
        "folds_verified": checked_folds,
        "outer_trial_predictions_verified": checked_trials,
        "all_trial_identities_paired": True,
        "all_inner_outer_runs_disjoint": True,
        "all_temperatures_selected_on_inner_validation": True,
        "all_gate_features_scaled_on_inner_validation": True,
        "all_fusion_probabilities_independently_recomputed": True,
        "all_reported_metrics_independently_recomputed": True,
        "promotion_decision_reproduced": True,
        "selected_candidate": selected_candidate,
        "promotion_passed": bool(independently_passed),
        "session_e_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    audit_dir = ensure_dir(output / "audit")
    try:
        report = audit(output)
    except Exception as exc:
        failure = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        write_json(audit_dir / "audit_report.json", failure)
        write_csv(audit_dir / "issues.csv", [{"severity": "error", **failure}])
        raise
    write_json(audit_dir / "audit_report.json", report)
    write_csv(audit_dir / "issues.csv", [])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
