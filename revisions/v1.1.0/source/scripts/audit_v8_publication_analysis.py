#!/usr/bin/env python3
"""Independently verify fixed fusion and subject-level publication summaries."""

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

from dpc_snn.experiments.v8_publication_baselines import (  # noqa: E402
    FREQUENCY_OCCLUSIONS_HZ,
    REGION_CHANNELS,
    component_error_profile,
    equal_probability_fusion,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_json  # noqa: E402
from dpc_snn.utils.metrics import bootstrap_ci, classification_metrics  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _run_base(output: Path, model: str, subject: int, seed: int) -> Path:
    return output / "runs" / model / f"subject_{subject:02d}" / f"seed_{seed}" / "evaluation"


def _cohorts(dataset: str, config: dict[str, Any], subjects: list[int]) -> dict[str, list[int]]:
    if dataset == "bci2a":
        return {"full_9": subjects}
    active = set(subjects)
    configured = config["datasets"]["openbmi"].get("reporting_cohorts", {})
    cohorts = {
        name: [int(value) for value in values if int(value) in active]
        for name, values in configured.items()
    }
    cohorts = {name: values for name, values in cohorts.items() if values}
    if set(subjects) != set(config["datasets"]["openbmi"]["subjects"]):
        cohorts = {"active_scope": subjects, **cohorts}
    return cohorts


def _close(first: float, second: float, atol: float = 1e-9) -> bool:
    return bool(np.isclose(float(first), float(second), rtol=0.0, atol=atol))


def _audit_fusion_run(
    output: Path,
    *,
    subject: int,
    seed: int,
) -> dict[str, float]:
    atc = _payload(_run_base(output, "atcnet", subject, seed) / "predictions.npz")
    fbc = _payload(_run_base(output, "fbcnet", subject, seed) / "predictions.npz")
    fusion_dir = output / "analysis" / "fusion" / f"subject_{subject:02d}" / f"seed_{seed}"
    fusion = _payload(fusion_dir / "predictions.npz")
    for field in ("label", "subject", "session", "run", "trial_id", "seed"):
        if not np.array_equal(atc[field], fbc[field]) or not np.array_equal(atc[field], fusion[field]):
            raise RuntimeError(f"paired fusion field {field!r} differs")
    probability, prediction = equal_probability_fusion(atc["probabilities"], fbc["probabilities"])
    if not np.allclose(fusion["probabilities"], probability, rtol=0.0, atol=1e-7):
        raise RuntimeError("saved fusion is not the fixed 0.5/0.5 probability average")
    if not np.array_equal(fusion["pred"], prediction):
        raise RuntimeError("saved fusion prediction differs from fixed fusion argmax")
    labels = atc["label"].astype(np.int64)
    metrics = classification_metrics(labels, prediction, n_classes=probability.shape[1])
    saved_metrics = read_json(fusion_dir / "metrics.json")
    for metric, value in metrics.items():
        if not _close(saved_metrics[metric], value):
            raise RuntimeError(f"fusion metric {metric} was not reproducible")
    expected_profile = component_error_profile(labels, atc["probabilities"], fbc["probabilities"])
    saved_profile = read_json(fusion_dir / "error_profile.json")
    profile_mapping = {
        "atcnet_accuracy": expected_profile["first"]["accuracy"],
        "fbcnet_accuracy": expected_profile["second"]["accuracy"],
        "fusion_accuracy": expected_profile["fusion"]["accuracy"],
        "oracle_accuracy": expected_profile["oracle_accuracy"],
        "disagreement_rate": expected_profile["disagreement_rate"],
        "double_fault_rate": expected_profile["double_fault_rate"],
        "fusion_gain_over_best_component": expected_profile["fusion_gain_over_best_component"],
    }
    for name, value in profile_mapping.items():
        if not _close(saved_profile[name], value):
            raise RuntimeError(f"fusion error-profile value {name} differs")
    atc_perturb = _payload(_run_base(output, "atcnet", subject, seed) / "perturbations.npz")
    fbc_perturb = _payload(_run_base(output, "fbcnet", subject, seed) / "perturbations.npz")
    fusion_perturb = _payload(fusion_dir / "perturbations.npz")
    for kind, names in (
        ("frequency", list(FREQUENCY_OCCLUSIONS_HZ)),
        ("region", list(REGION_CHANNELS)),
    ):
        if fusion_perturb[f"{kind}_names"].astype(str).tolist() != names:
            raise RuntimeError(f"fusion {kind} perturbation names differ")
        expected = 0.5 * (
            atc_perturb[f"{kind}_probabilities"] + fbc_perturb[f"{kind}_probabilities"]
        )
        if not np.allclose(
            fusion_perturb[f"{kind}_probabilities"], expected, rtol=0.0, atol=1e-7
        ):
            raise RuntimeError(f"fusion {kind} perturbation is not fixed 0.5/0.5 averaging")
        if not np.array_equal(
            fusion_perturb[f"{kind}_pred"], expected.argmax(axis=-1)
        ):
            raise RuntimeError(f"fusion {kind} perturbation prediction differs")
    return {name: float(value) for name, value in metrics.items()}


def _audit_model_summary(
    analysis: Path,
    *,
    metric_rows: list[dict[str, str]],
    cohorts: dict[str, list[int]],
    models: list[str],
    bootstrap_repeats: int,
) -> None:
    summary_rows = _read_csv(analysis / "model_cohort_summary.csv")
    index = {(row["cohort"], row["model"]): row for row in summary_rows}
    expected_count = len(cohorts) * len(models)
    if len(index) != expected_count:
        raise RuntimeError(f"model summary has {len(index)} unique rows; expected {expected_count}")
    for cohort, subjects in cohorts.items():
        for model in models:
            subject_values = []
            for subject in subjects:
                values = [
                    float(row["accuracy"])
                    for row in metric_rows
                    if row["model"] == model and int(row["subject"]) == subject
                ]
                if values:
                    subject_values.append(float(np.mean(values)))
            interval = bootstrap_ci(
                np.asarray(subject_values), seed=7001, n_boot=bootstrap_repeats
            )
            saved = index[(cohort, model)]
            for name, value in (
                ("subject_macro_accuracy", interval["mean"]),
                ("ci_low", interval["low"]),
                ("ci_high", interval["high"]),
            ):
                if not _close(float(saved[name]), value):
                    raise RuntimeError(f"{cohort}/{model} {name} differs from subject bootstrap")
            if int(saved["subjects"]) != len(subject_values):
                raise RuntimeError(f"{cohort}/{model} subject count differs")


def _audit_perturbation_summary(
    analysis: Path,
    *,
    rows: list[dict[str, str]],
    cohorts: dict[str, list[int]],
    models: list[str],
) -> None:
    summary = _read_csv(analysis / "perturbation_cohort_summary.csv")
    index = {
        (row["cohort"], row["model"], row["kind"], row["name"]): row
        for row in summary
    }
    expected_per_model = len(FREQUENCY_OCCLUSIONS_HZ) + len(REGION_CHANNELS)
    if len(index) != len(cohorts) * len(models) * expected_per_model:
        raise RuntimeError("perturbation cohort summary row count differs")
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
                            and int(row["subject"]) == subject
                            and row["kind"] == kind
                            and row["name"] == name
                        ]
                        if values:
                            subject_values.append(float(np.mean(values)))
                    expected = 100.0 * float(np.mean(subject_values))
                    saved = float(index[(cohort, model, kind, name)]["mean_accuracy_drop_pp"])
                    if not _close(saved, expected):
                        raise RuntimeError(
                            f"{cohort}/{model}/{kind}/{name} perturbation mean differs"
                        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset", choices=("bci2a", "openbmi"), required=True)
    parser.add_argument("--canary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = Path(args.output_root) / ("canary" if args.canary else "") / args.dataset
    baseline_audit = read_json(output / "audit" / "audit_report.json")
    if baseline_audit.get("status") != "passed":
        raise RuntimeError("fusion audit requires a passed baseline audit")
    analysis = output / "analysis"
    status = read_json(analysis / "analysis_status.json")
    if status.get("status") != "completed":
        raise RuntimeError("publication analysis is incomplete")
    contract = read_json(output / "contract.json")
    scope = dict(contract["scope"])
    config = dict(contract["resolved_config"])
    issues: list[dict[str, Any]] = []
    valid_fusion = 0
    for subject in scope["subjects"]:
        for seed in scope["seeds"]:
            try:
                _audit_fusion_run(output, subject=int(subject), seed=int(seed))
                valid_fusion += 1
            except Exception as exc:
                issues.append(
                    {
                        "stage": "fusion",
                        "subject": int(subject),
                        "seed": int(seed),
                        "error": str(exc),
                    }
                )
    models = [*scope["models"], "atc_fbc_fusion"]
    cohorts = _cohorts(args.dataset, config, scope["subjects"])
    try:
        metric_rows = _read_csv(analysis / "all_model_subject_seed_metrics.csv")
        expected_rows = len(models) * len(scope["subjects"]) * len(scope["seeds"])
        if len(metric_rows) != expected_rows:
            raise RuntimeError(f"metric rows {len(metric_rows)} != {expected_rows}")
        _audit_model_summary(
            analysis,
            metric_rows=metric_rows,
            cohorts=cohorts,
            models=models,
            bootstrap_repeats=int(config["analysis"]["bootstrap_repeats"]),
        )
        perturbation_rows = _read_csv(analysis / "perturbation_subject_seed.csv")
        _audit_perturbation_summary(
            analysis,
            rows=perturbation_rows,
            cohorts=cohorts,
            models=models,
        )
        required_figures = (
            "baseline_accuracy.png",
            "frequency_region_occlusion.png",
            "fusion_subject_complementarity.png",
        )
        missing = [
            name for name in required_figures if not (analysis / "figures" / name).is_file()
        ]
        if missing:
            raise RuntimeError(f"analysis is missing required figures: {missing}")
    except Exception as exc:
        issues.append({"stage": "aggregate", "error": str(exc)})
    audit_dir = ensure_dir(analysis / "audit")
    report = {
        "schema": "dpc-snn-v8-posthoc-publication-analysis-audit/v1",
        "dataset": args.dataset,
        "status": "passed" if not issues else "failed",
        "posthoc_explanatory": True,
        "expected_fusion_runs": len(scope["subjects"]) * len(scope["seeds"]),
        "valid_fusion_runs": valid_fusion,
        "issues": issues,
    }
    write_json(audit_dir / "audit_report.json", report)
    print(json.dumps(report, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
