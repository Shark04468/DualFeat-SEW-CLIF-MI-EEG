#!/usr/bin/env python3
"""Measure ATC/FBC complementarity and fold-local routing reliability."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _csv_int(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    return -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1
    ) / np.log(probabilities.shape[1])


def _margin(probabilities: np.ndarray) -> np.ndarray:
    ordered = np.sort(probabilities, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def _logsumexp(logits: np.ndarray) -> np.ndarray:
    maximum = logits.max(axis=1, keepdims=True)
    return maximum[:, 0] + np.log(np.exp(logits - maximum).sum(axis=1))


def _selector_features(
    atc_logits: np.ndarray, fbc_logits: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    atc_probability = _softmax(atc_logits)
    fbc_probability = _softmax(fbc_logits)
    atc_entropy = _entropy(atc_probability)
    fbc_entropy = _entropy(fbc_probability)
    atc_margin = _margin(atc_probability)
    fbc_margin = _margin(fbc_probability)
    midpoint = 0.5 * (atc_probability + fbc_probability)
    js_divergence = 0.5 * (
        np.sum(
            atc_probability
            * np.log(np.clip(atc_probability / midpoint, 1e-12, None)),
            axis=1,
        )
        + np.sum(
            fbc_probability
            * np.log(np.clip(fbc_probability / midpoint, 1e-12, None)),
            axis=1,
        )
    )
    features = np.column_stack(
        [
            atc_probability.max(axis=1) - fbc_probability.max(axis=1),
            fbc_entropy - atc_entropy,
            atc_margin - fbc_margin,
            _logsumexp(atc_logits) - _logsumexp(fbc_logits),
            np.linalg.norm(atc_logits, axis=1) - np.linalg.norm(fbc_logits, axis=1),
            js_divergence,
            np.abs(atc_probability - fbc_probability).sum(axis=1),
        ]
    )
    diagnostics = {
        "atc_confidence": atc_probability.max(axis=1),
        "fbc_confidence": fbc_probability.max(axis=1),
        "atc_entropy": atc_entropy,
        "fbc_entropy": fbc_entropy,
        "atc_margin": atc_margin,
        "fbc_margin": fbc_margin,
        "js_divergence": js_divergence,
    }
    return features, diagnostics


def _fit_selector(
    atc_logits: np.ndarray, fbc_logits: np.ndarray, labels: np.ndarray
) -> tuple[object | None, float, int]:
    atc_prediction = atc_logits.argmax(axis=1)
    fbc_prediction = fbc_logits.argmax(axis=1)
    atc_correct = atc_prediction == labels
    fbc_correct = fbc_prediction == labels
    resolvable = (atc_prediction != fbc_prediction) & (atc_correct | fbc_correct)
    targets = atc_correct[resolvable].astype(np.int64)
    prior = float(targets.mean()) if targets.size else 0.5
    if targets.size < 8 or np.unique(targets).size < 2:
        return None, prior, int(targets.size)
    features, _ = _selector_features(atc_logits, fbc_logits)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=0,
        ),
    )
    model.fit(features[resolvable], targets)
    return model, prior, int(targets.size)


def _selector_probability(
    model: object | None,
    prior: float,
    atc_logits: np.ndarray,
    fbc_logits: np.ndarray,
) -> np.ndarray:
    if model is None:
        return np.full(atc_logits.shape[0], prior, dtype=np.float64)
    features, _ = _selector_features(atc_logits, fbc_logits)
    return model.predict_proba(features)[:, 1]


def _mean_subject_seed(frame: pd.DataFrame, column: str) -> float:
    return float(
        frame.groupby(["subject", "seed"], sort=True)[column].mean().mean()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--anchor-root", required=True)
    parser.add_argument("--e10-aggregate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subjects", default="1,3,8")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--folds", default="0,1,2,3,4,5")
    args = parser.parse_args()

    data_root = Path(args.data).resolve()
    anchor_root = Path(args.anchor_root).resolve()
    aggregate = Path(args.e10_aggregate).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    subjects = _csv_int(args.subjects)
    seeds = _csv_int(args.seeds)
    folds = _csv_int(args.folds)

    audit = json.loads((aggregate / "campaign_audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("session_e_accessed") is not False:
        raise RuntimeError("E11B requires a passed Session-E-locked E10 aggregate")
    e10 = pd.read_csv(aggregate / "trial_predictions.csv").set_index(
        ["subject", "seed", "trial_index"]
    )

    trial_rows: list[dict[str, object]] = []
    selector_rows: list[dict[str, object]] = []
    teacher_prediction_mismatches = 0
    for subject in subjects:
        dataset = np.load(data_root / f"A{subject:02d}.npz", allow_pickle=False)
        training_mask = dataset["session"].astype(str) == "T"
        labels_t = dataset["y"][training_mask].astype(np.int64)
        if labels_t.shape != (288,):
            raise RuntimeError(f"unexpected Session-T trial count for subject {subject}")
        for seed in seeds:
            for fold in folds:
                fold_root = (
                    anchor_root
                    / f"subject_{subject:02d}"
                    / f"seed_{seed}"
                    / f"fold_{fold}"
                )
                cache_path = fold_root / "frozen_dual_feature_cache.npz"
                if not cache_path.is_file():
                    raise RuntimeError(f"missing cache: {cache_path}")
                cache = np.load(cache_path, allow_pickle=False)
                validation_indices = cache["inner_validation_indices"].astype(np.int64)
                test_indices = cache["outer_test_indices"].astype(np.int64)
                validation_labels = labels_t[validation_indices]
                test_labels = labels_t[test_indices]
                atc_validation = cache["atcnet_selection_validation_logits"]
                fbc_validation = cache["fbcnet_selection_validation_logits"]
                atc_test = cache["atcnet_outer_test_logits"]
                fbc_test = cache["fbcnet_outer_test_logits"]
                teacher_test = cache["teacher_outer_test"]
                model, prior, fit_cases = _fit_selector(
                    atc_validation, fbc_validation, validation_labels
                )
                selector_probability = _selector_probability(
                    model, prior, atc_test, fbc_test
                )

                atc_probability = _softmax(atc_test)
                fbc_probability = _softmax(fbc_test)
                teacher_probability = _softmax(teacher_test)
                atc_prediction = atc_probability.argmax(axis=1)
                fbc_prediction = fbc_probability.argmax(axis=1)
                teacher_prediction = teacher_probability.argmax(axis=1)
                static_choose_atc = prior >= 0.5
                static_prediction = (
                    atc_prediction if static_choose_atc else fbc_prediction
                )
                dynamic_prediction = np.where(
                    selector_probability >= 0.5, atc_prediction, fbc_prediction
                )
                atc_correct = atc_prediction == test_labels
                fbc_correct = fbc_prediction == test_labels
                resolvable = (
                    (atc_prediction != fbc_prediction) & (atc_correct | fbc_correct)
                )
                selector_target = atc_correct[resolvable].astype(np.int64)
                selector_score = selector_probability[resolvable]
                selector_auc = float("nan")
                if selector_target.size and np.unique(selector_target).size == 2:
                    selector_auc = float(roc_auc_score(selector_target, selector_score))
                selector_rows.append(
                    {
                        "subject": subject,
                        "seed": seed,
                        "fold": fold,
                        "fit_resolvable_cases": fit_cases,
                        "test_resolvable_cases": int(selector_target.size),
                        "selector_prior_atc": prior,
                        "selector_auc": selector_auc,
                        "selector_choice_accuracy": float(
                            np.mean((selector_score >= 0.5) == selector_target)
                        )
                        if selector_target.size
                        else float("nan"),
                    }
                )
                _, diagnostics = _selector_features(atc_test, fbc_test)
                for row_index, trial_index in enumerate(test_indices):
                    key = (subject, seed, int(trial_index))
                    if key not in e10.index:
                        raise RuntimeError(f"missing E10 prediction row: {key}")
                    e10_row = e10.loc[key]
                    if int(e10_row["label"]) != int(test_labels[row_index]):
                        raise RuntimeError(f"label mismatch for {key}")
                    teacher_prediction_mismatches += int(
                        teacher_prediction[row_index]
                        != int(e10_row["prediction_equal_teacher"])
                    )
                    trial_rows.append(
                        {
                            "subject": subject,
                            "seed": seed,
                            "fold": fold,
                            "trial_index": int(trial_index),
                            "label": int(test_labels[row_index]),
                            "prediction_atc": int(atc_prediction[row_index]),
                            "prediction_fbc": int(fbc_prediction[row_index]),
                            "prediction_equal_teacher": int(teacher_prediction[row_index]),
                            "prediction_static_selector": int(static_prediction[row_index]),
                            "prediction_dynamic_selector": int(dynamic_prediction[row_index]),
                            "prediction_v9_snn": int(
                                e10_row["prediction_v9_sew_clif_kd"]
                            ),
                            "atc_correct": bool(atc_correct[row_index]),
                            "fbc_correct": bool(fbc_correct[row_index]),
                            "equal_teacher_correct": bool(
                                teacher_prediction[row_index] == test_labels[row_index]
                            ),
                            "static_selector_correct": bool(
                                static_prediction[row_index] == test_labels[row_index]
                            ),
                            "dynamic_selector_correct": bool(
                                dynamic_prediction[row_index] == test_labels[row_index]
                            ),
                            "v9_snn_correct": bool(
                                int(e10_row["prediction_v9_sew_clif_kd"])
                                == test_labels[row_index]
                            ),
                            "branch_oracle_correct": bool(
                                atc_correct[row_index] or fbc_correct[row_index]
                            ),
                            "selector_probability_atc": float(
                                selector_probability[row_index]
                            ),
                            **{
                                name: float(values[row_index])
                                for name, values in diagnostics.items()
                            },
                        }
                    )

    trials = pd.DataFrame(trial_rows).sort_values(
        ["subject", "seed", "trial_index"]
    )
    selectors = pd.DataFrame(selector_rows).sort_values(["subject", "seed", "fold"])
    expected_rows = len(subjects) * len(seeds) * 288
    if len(trials) != expected_rows or trials.duplicated(
        ["subject", "seed", "trial_index"]
    ).any():
        raise RuntimeError("incomplete or duplicate E11B trial coverage")

    metric_columns = {
        "atc": "atc_correct",
        "fbc": "fbc_correct",
        "equal_teacher": "equal_teacher_correct",
        "static_selector": "static_selector_correct",
        "dynamic_selector": "dynamic_selector_correct",
        "v9_snn": "v9_snn_correct",
        "branch_oracle": "branch_oracle_correct",
    }
    subject_seed_rows: list[dict[str, object]] = []
    for (subject, seed), group in trials.groupby(["subject", "seed"], sort=True):
        row: dict[str, object] = {"subject": int(subject), "seed": int(seed)}
        for name, column in metric_columns.items():
            row[f"accuracy_{name}"] = float(group[column].mean())
        row["atc_only_correct_rate"] = float(
            (group["atc_correct"] & ~group["fbc_correct"]).mean()
        )
        row["fbc_only_correct_rate"] = float(
            (~group["atc_correct"] & group["fbc_correct"]).mean()
        )
        row["both_wrong_rate"] = float(
            (~group["atc_correct"] & ~group["fbc_correct"]).mean()
        )
        row["teacher_rescue_v9_rate"] = float(
            (group["equal_teacher_correct"] & ~group["v9_snn_correct"]).mean()
        )
        row["v9_rescue_teacher_rate"] = float(
            (~group["equal_teacher_correct"] & group["v9_snn_correct"]).mean()
        )
        subject_seed_rows.append(row)
    subject_seed = pd.DataFrame(subject_seed_rows)
    subject = subject_seed.groupby("subject", as_index=False).mean(numeric_only=True)

    summary_accuracies = {
        name: _mean_subject_seed(trials, column)
        for name, column in metric_columns.items()
    }
    finite_auc = selectors["selector_auc"].dropna()
    selector_auc_mean = float(finite_auc.mean()) if len(finite_auc) else float("nan")
    dynamic_gain_pp = 100.0 * (
        summary_accuracies["dynamic_selector"] - summary_accuracies["equal_teacher"]
    )
    recommended_gate = (
        "dynamic"
        if selector_auc_mean >= 0.60 and dynamic_gain_pp >= -0.30
        else "static_low_capacity"
    )

    trials.to_csv(output / "trial_branch_features.csv", index=False)
    selectors.to_csv(output / "fold_selector_metrics.csv", index=False)
    subject_seed.to_csv(output / "subject_seed_summary.csv", index=False)
    subject.to_csv(output / "subject_summary.csv", index=False)
    summary = {
        "status": "completed",
        "subjects": subjects,
        "seeds": seeds,
        "folds": folds,
        "trial_rows": len(trials),
        "accuracies": summary_accuracies,
        "atc_only_correct_rate": _mean_subject_seed(trials, "atc_correct")
        - _mean_subject_seed(trials, "atc_correct")
        * 0.0,
        "selector_auc_mean": selector_auc_mean,
        "selector_auc_valid_folds": int(len(finite_auc)),
        "dynamic_minus_equal_teacher_pp": dynamic_gain_pp,
        "teacher_prediction_mismatches_vs_e10": teacher_prediction_mismatches,
        "recommended_gate": recommended_gate,
        "dynamic_gate_threshold": {
            "selector_auc_minimum": 0.60,
            "noninferiority_pp": -0.30,
        },
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    summary["atc_only_correct_rate"] = float(
        subject_seed["atc_only_correct_rate"].mean()
    )
    summary["fbc_only_correct_rate"] = float(
        subject_seed["fbc_only_correct_rate"].mean()
    )
    summary["both_wrong_rate"] = float(subject_seed["both_wrong_rate"].mean())
    summary["teacher_rescue_v9_rate"] = float(
        subject_seed["teacher_rescue_v9_rate"].mean()
    )
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output / "PLAN.md").write_text(
        "# E11B Branch Reliability Campaign\n\n"
        "Fold-local ATC/FBC complementarity and router-identifiability analysis. "
        "Only Session-T inner validation labels fit the selector; outer folds are evaluation only.\n",
        encoding="utf-8",
    )
    (output / "CHECKLIST.md").write_text(
        "# E11B Checklist\n\n"
        "- [x] V9 cache coverage validated\n"
        "- [x] E10 prediction coverage validated\n"
        "- [x] Fold-local selector evaluated\n"
        "- [x] Complementarity and oracle bounds aggregated\n"
        "- [x] V12.1 gate type selected\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
