#!/usr/bin/env python3
"""Run V9 E0-E2 reliability-aware fusion on frozen Session-T OOF logits."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    build_run_fingerprint,
    file_sha256,
    sha256_fingerprint,
    validate_resume_fingerprint,
    write_fingerprint_manifest,
)
from dpc_snn.experiments.v9_reliability_fusion import (  # noqa: E402
    FEATURE_NAMES,
    GATE_VARIANTS,
    OUTPUT_ARMS,
    gate_parameter_count,
    probability_metrics,
    run_fusion_fold,
    softmax_probabilities,
)
from dpc_snn.utils.io import ensure_dir, read_json, save_npz, write_csv, write_json  # noqa: E402


REQUIRED_FULL_FIELDS = {
    "logits",
    "probabilities",
    "pred",
    "label",
    "subject",
    "session",
    "run",
    "trial_id",
    "seed",
    "model",
}
REQUIRED_FOLD_FIELDS = {"indices", "logits", "labels"}
SOURCE_FILES = (
    "src/dpc_snn/experiments/v9_reliability_fusion.py",
    "scripts/run_v9_e2_reliability_fusion.py",
    "scripts/audit_v9_e2_reliability_fusion.py",
    "configs/experiments/v9_e2_reliability_fusion.yaml",
)


def _csv(value: str, cast: Any = str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _load_npz(path: Path, required: set[str]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    if set(values) != required:
        raise RuntimeError(
            f"archive fields differ for {path}: expected {sorted(required)}, got {sorted(values)}"
        )
    return values


def _load_full(path: Path, *, expected_trials: int, n_classes: int) -> dict[str, np.ndarray]:
    values = _load_npz(path, REQUIRED_FULL_FIELDS)
    if values["logits"].shape != (expected_trials, n_classes):
        raise RuntimeError(f"unexpected full prediction shape: {path}")
    if values["probabilities"].shape != values["logits"].shape:
        raise RuntimeError(f"full probabilities and logits differ in shape: {path}")
    if values["label"].shape != (expected_trials,) or values["pred"].shape != (
        expected_trials,
    ):
        raise RuntimeError(f"full prediction vectors have invalid shape: {path}")
    if set(values["session"].astype(str).tolist()) != {"T"}:
        raise RuntimeError(f"V9 E2 received non-Session-T predictions: {path}")
    if len(set(values["trial_id"].astype(str).tolist())) != expected_trials:
        raise RuntimeError(f"trial identities are not unique: {path}")
    reconstructed = softmax_probabilities(values["logits"])
    if not np.allclose(reconstructed, values["probabilities"], atol=2e-6):
        raise RuntimeError(f"full probabilities are not softmax(logits): {path}")
    if not np.array_equal(values["pred"], values["probabilities"].argmax(axis=1)):
        raise RuntimeError(f"full prediction labels are inconsistent: {path}")
    return values


def _load_fold(path: Path, *, n_classes: int) -> dict[str, np.ndarray]:
    values = _load_npz(path, REQUIRED_FOLD_FIELDS)
    indices = np.asarray(values["indices"], dtype=np.int64)
    logits = np.asarray(values["logits"], dtype=np.float32)
    labels = np.asarray(values["labels"], dtype=np.int64)
    if indices.ndim != 1 or logits.shape != (indices.size, n_classes) or labels.shape != indices.shape:
        raise RuntimeError(f"fold archive has invalid shapes: {path}")
    if np.unique(indices).size != indices.size or not np.isfinite(logits).all():
        raise RuntimeError(f"fold archive has duplicate indices or invalid logits: {path}")
    return {"indices": indices, "logits": logits, "labels": labels}


def _assert_aligned(first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray]) -> None:
    for field in ("label", "subject", "session", "run", "trial_id", "seed"):
        if not np.array_equal(first[field], second[field]):
            raise RuntimeError(f"ATCNet and FBCNet OOF predictions differ on {field}")


def _source_manifest() -> dict[str, str]:
    manifest: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"V9 source contract is incomplete: {path}")
        manifest[relative] = file_sha256(path)
    return manifest


def _input_paths(
    e1_root: Path,
    subjects: Sequence[int],
    seeds: Sequence[int],
    folds: Sequence[int],
) -> list[Path]:
    paths: list[Path] = []
    for model in ("atcnet", "fbcnet"):
        for subject in subjects:
            for seed in seeds:
                run = e1_root / model / f"subject_{subject:02d}" / f"seed_{seed}"
                paths.append(run / "predictions.npz")
                for fold in folds:
                    paths.extend(
                        (
                            run / f"fold_{fold}" / "selection_predictions.npz",
                            run / f"fold_{fold}" / "outer_test_predictions.npz",
                        )
                    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("V9 teacher cache is incomplete:\n" + "\n".join(missing[:20]))
    return paths


def _input_manifest(paths: Sequence[Path], root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(paths):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )
    return rows


def _environment_manifest() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "executable": sys.executable,
        "pid": os.getpid(),
    }


def _bootstrap_mean_ci(values: Sequence[float], *, seed: int, n_boot: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_boot), dtype=np.float64)
    for index in range(int(n_boot)):
        means[index] = rng.choice(array, size=array.size, replace=True).mean()
    return {
        "mean": float(array.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
    }


def _sign_test_p(positive: int, negative: int) -> float:
    count = int(positive) + int(negative)
    if count == 0:
        return 1.0
    lower = min(int(positive), int(negative))
    tail = sum(math.comb(count, index) for index in range(lower + 1)) / (2.0**count)
    return float(min(1.0, 2.0 * tail))


def _parameter_count(arm: str, n_classes: int) -> int:
    if arm not in GATE_VARIANTS:
        return 0
    return gate_parameter_count(arm, n_classes, len(FEATURE_NAMES))


def _write_report(output: Path, gate: Mapping[str, Any], aggregate: Sequence[Mapping[str, Any]]) -> None:
    selected = str(gate["selected_candidate"])
    selected_row = next(row for row in aggregate if row["arm"] == selected)
    fixed_row = next(row for row in aggregate if row["arm"] == "equal_calibrated")
    lines = [
        "# V9 E0-E2 Reliability Fusion Report",
        "",
        "- Protocol: BCI Competition IV 2a Session T, nested six-run OOF only.",
        "- Subjects: 1, 3, 8; seeds: 0, 1, 2.",
        "- Teacher logits: frozen ATCNet and FBCNet V8 E1 caches.",
        "- Gate fitting: current outer fold's inner-validation run only.",
        "- Session E accessed: false.",
        "",
        f"Selected development candidate: `{selected}`.",
        f"Mean accuracy: {100.0 * float(selected_row['mean_accuracy']):.3f}%.",
        f"Calibrated equal fusion: {100.0 * float(fixed_row['mean_accuracy']):.3f}%.",
        f"Promotion gate passed: {bool(gate['passed'])}.",
        "",
        "This result is development evidence and cannot be reported as external confirmation.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/experiments/v9_e2_reliability_fusion.yaml"),
    )
    parser.add_argument("--e1-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subjects")
    parser.add_argument("--seeds")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    e1_root = Path(args.e1_root).resolve()
    output = Path(args.output).resolve()
    subjects = _csv(args.subjects, int) if args.subjects else [int(v) for v in config["subjects"]]
    seeds = _csv(args.seeds, int) if args.seeds else [int(v) for v in config["seeds"]]
    folds = [int(value) for value in config["folds"]]
    n_classes = int(config["n_classes"])
    expected_trials = int(config["expected_trials_per_subject"])
    temperature_grid = [float(value) for value in config["temperature_grid"]]
    variants = [str(value) for value in config["gate"]["variants"]]
    if variants != list(GATE_VARIANTS):
        raise RuntimeError("formal V9 E2 must run all prespecified gate variants in order")
    if folds != list(range(6)) or not subjects or not seeds:
        raise RuntimeError("formal V9 E2 requires all six folds and non-empty subject/seed sets")

    resolved_config = {
        **config,
        "config_path": str(config_path),
        "e1_root": str(e1_root),
        "subjects": subjects,
        "seeds": seeds,
        "folds": folds,
    }
    source_manifest = _source_manifest()
    paths = _input_paths(e1_root, subjects, seeds, folds)
    input_manifest = _input_manifest(paths, e1_root)
    environment = _environment_manifest()
    fingerprint = build_run_fingerprint(
        resolved_config=resolved_config,
        source=source_manifest,
        data={
            "dataset": "BCI Competition IV 2a",
            "session": "T",
            "input_manifest_sha256": sha256_fingerprint(input_manifest),
        },
        split={
            "protocol": config["protocol"],
            "subjects": subjects,
            "seeds": seeds,
            "folds": folds,
            "fit_scope": "current_outer_fold_inner_validation_run_only",
            "evaluation_scope": "current_outer_held_run_only",
        },
        augmentation={"used": False},
        prior={
            "gate": config["gate"],
            "temperature_grid": temperature_grid,
            "features": list(FEATURE_NAMES),
        },
        checkpoint={"teacher_prediction_manifest": input_manifest},
        environment=environment,
    )

    fingerprint_path = output / "run_fingerprint.json"
    if fingerprint_path.is_file():
        validate_resume_fingerprint(fingerprint_path, fingerprint)
        status_path = output / "campaign_status.json"
        if status_path.is_file() and read_json(status_path).get("status") == "completed" and not args.force:
            print(json.dumps({"status": "already_completed", "output": str(output)}, indent=2))
            return
    ensure_dir(output)
    write_fingerprint_manifest(fingerprint_path, fingerprint)
    write_json(output / "resolved_config.json", resolved_config)
    write_json(output / "source_manifest.json", source_manifest)
    write_json(output / "input_manifest.json", {"files": input_manifest})
    write_json(output / "environment.json", environment)
    write_json(
        output / "e0_protocol_lock.json",
        {
            "status": "passed",
            "protocol": config["protocol"],
            "selection_session": "T",
            "heldout_session_e_accessed": False,
            "outer_test_labels_used_for_fitting": False,
            "teacher_models_frozen": True,
            "candidate_set_prespecified": variants,
            "primary_reference": "equal_calibrated",
            "fingerprint": fingerprint["combined_sha256"],
        },
    )
    write_json(
        output / "campaign_status.json",
        {
            "status": "running",
            "stage": "E1_teacher_cache_validation",
            "session_e_accessed": False,
            "fingerprint": fingerprint["combined_sha256"],
        },
    )

    fold_metric_rows: list[dict[str, Any]] = []
    subject_seed_rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []
    teacher_audit_rows: list[dict[str, Any]] = []

    for subject in subjects:
        for seed in seeds:
            run_dirs = {
                model: e1_root / model / f"subject_{subject:02d}" / f"seed_{seed}"
                for model in ("atcnet", "fbcnet")
            }
            full_paths = {model: directory / "predictions.npz" for model, directory in run_dirs.items()}
            full = {
                model: _load_full(path, expected_trials=expected_trials, n_classes=n_classes)
                for model, path in full_paths.items()
            }
            _assert_aligned(full["atcnet"], full["fbcnet"])
            labels = np.asarray(full["atcnet"]["label"], dtype=np.int64)
            runs = full["atcnet"]["run"].astype(str)
            if np.unique(runs).size != len(folds):
                raise RuntimeError("Session T run count does not match the six-fold contract")
            subject_output = ensure_dir(output / f"subject_{subject:02d}" / f"seed_{seed}")
            probabilities = {
                arm: np.full((expected_trials, n_classes), np.nan, dtype=np.float64)
                for arm in OUTPUT_ARMS
            }
            fold_assignment = np.full(expected_trials, -1, dtype=np.int64)
            seen = np.zeros(expected_trials, dtype=bool)

            for fold in folds:
                fold_sources: dict[str, dict[str, Path]] = {}
                fold_data: dict[str, dict[str, dict[str, np.ndarray]]] = {}
                for model in ("atcnet", "fbcnet"):
                    directory = run_dirs[model] / f"fold_{fold}"
                    fold_sources[model] = {
                        "selection": directory / "selection_predictions.npz",
                        "outer": directory / "outer_test_predictions.npz",
                    }
                    fold_data[model] = {
                        role: _load_fold(path, n_classes=n_classes)
                        for role, path in fold_sources[model].items()
                    }
                for role in ("selection", "outer"):
                    for field in ("indices", "labels"):
                        if not np.array_equal(
                            fold_data["atcnet"][role][field], fold_data["fbcnet"][role][field]
                        ):
                            raise RuntimeError(
                                f"teacher fold mismatch: subject={subject}, seed={seed}, "
                                f"fold={fold}, role={role}, field={field}"
                            )
                selection = fold_data["atcnet"]["selection"]
                outer = fold_data["atcnet"]["outer"]
                selection_indices = selection["indices"]
                outer_indices = outer["indices"]
                if np.intersect1d(selection_indices, outer_indices).size:
                    raise RuntimeError("inner-validation and outer-test trials overlap")
                if np.any(selection_indices < 0) or np.any(outer_indices < 0) or np.any(
                    np.concatenate((selection_indices, outer_indices)) >= expected_trials
                ):
                    raise RuntimeError("fold indices fall outside the full OOF archive")
                if not np.array_equal(labels[selection_indices], selection["labels"]) or not np.array_equal(
                    labels[outer_indices], outer["labels"]
                ):
                    raise RuntimeError("fold labels do not match frozen full OOF metadata")
                selection_runs = set(runs[selection_indices].tolist())
                outer_runs = set(runs[outer_indices].tolist())
                if len(selection_runs) != 1 or len(outer_runs) != 1 or selection_runs & outer_runs:
                    raise RuntimeError("fold fit/evaluation runs violate nested run grouping")
                if seen[outer_indices].any():
                    raise RuntimeError("outer folds overlap")

                result = run_fusion_fold(
                    fold_data["atcnet"]["selection"]["logits"],
                    fold_data["fbcnet"]["selection"]["logits"],
                    selection["labels"],
                    fold_data["atcnet"]["outer"]["logits"],
                    fold_data["fbcnet"]["outer"]["logits"],
                    temperature_grid=temperature_grid,
                    gate_variants=variants,
                    l2=float(config["gate"]["l2"]),
                    parameter_bound=float(config["gate"]["parameter_bound"]),
                    max_iterations=int(config["gate"]["max_iterations"]),
                )
                for arm, value in result["probabilities"].items():
                    probabilities[arm][outer_indices] = value
                    metrics = probability_metrics(value, outer["labels"], n_bins=int(config["ece_bins"]))
                    fold_metric_rows.append(
                        {
                            "subject": subject,
                            "seed": seed,
                            "fold": fold,
                            "outer_run": next(iter(outer_runs)),
                            "inner_validation_run": next(iter(selection_runs)),
                            "arm": arm,
                            "n_trials": int(outer_indices.size),
                            "parameters": _parameter_count(arm, n_classes),
                            **metrics,
                            "session_e_accessed": False,
                        }
                    )
                seen[outer_indices] = True
                fold_assignment[outer_indices] = fold

                fit_payload = {
                    "subject": subject,
                    "seed": seed,
                    "fold": fold,
                    "inner_validation_indices": selection_indices.tolist(),
                    "outer_test_indices": outer_indices.tolist(),
                    "inner_validation_run": next(iter(selection_runs)),
                    "outer_test_run": next(iter(outer_runs)),
                    "calibration": result["calibration"],
                    "gates": result["fits"],
                    "source_sha256": {
                        model: {
                            role: file_sha256(path) for role, path in sources.items()
                        }
                        for model, sources in fold_sources.items()
                    },
                    "session_e_accessed": False,
                }
                fold_output = ensure_dir(subject_output / f"fold_{fold}")
                write_json(fold_output / "fit.json", fit_payload)
                save_npz(
                    fold_output / "outer_predictions.npz",
                    indices=outer_indices,
                    labels=outer["labels"],
                    **{
                        f"probability_{arm}": value
                        for arm, value in result["probabilities"].items()
                    },
                    **{
                        f"weight_{arm}": value for arm, value in result["weights"].items()
                    },
                )
                for variant, fit in result["fits"].items():
                    convergence_rows.append(
                        {
                            "subject": subject,
                            "seed": seed,
                            "fold": fold,
                            "variant": variant,
                            "converged": bool(fit["converged"]),
                            "iterations": int(fit["iterations"]),
                            "objective": float(fit["objective"]),
                            "initial_objective": float(fit["initial_objective"]),
                            "message": fit["message"],
                        }
                    )

            if not seen.all() or np.any(fold_assignment < 0):
                raise RuntimeError("outer folds do not form complete OOF coverage")
            if not np.allclose(
                probabilities["atcnet_raw"], full["atcnet"]["probabilities"], atol=2e-6
            ) or not np.allclose(
                probabilities["fbcnet_raw"], full["fbcnet"]["probabilities"], atol=2e-6
            ):
                raise RuntimeError("assembled outer logits differ from frozen teacher OOF probabilities")
            if any(not np.isfinite(value).all() for value in probabilities.values()):
                raise FloatingPointError("V9 OOF probability archive is incomplete")

            save_npz(
                subject_output / "predictions.npz",
                label=labels,
                subject=full["atcnet"]["subject"],
                session=full["atcnet"]["session"],
                run=full["atcnet"]["run"],
                trial_id=full["atcnet"]["trial_id"],
                seed=full["atcnet"]["seed"],
                fold=fold_assignment,
                **{f"probability_{arm}": value for arm, value in probabilities.items()},
                **{
                    f"pred_{arm}": value.argmax(axis=1).astype(np.int64)
                    for arm, value in probabilities.items()
                },
            )
            for arm, value in probabilities.items():
                metrics = probability_metrics(value, labels, n_bins=int(config["ece_bins"]))
                subject_seed_rows.append(
                    {
                        "subject": subject,
                        "seed": seed,
                        "arm": arm,
                        "n_trials": expected_trials,
                        "parameters": _parameter_count(arm, n_classes),
                        **metrics,
                        "session_e_accessed": False,
                    }
                )
            teacher_audit_rows.append(
                {
                    "subject": subject,
                    "seed": seed,
                    "trials": expected_trials,
                    "folds": len(folds),
                    "trial_ids_unique": True,
                    "teacher_metadata_aligned": True,
                    "fold_logits_reconstruct_full_probabilities": True,
                    "inner_outer_disjoint": True,
                    "outer_oof_complete": True,
                    "session_e_accessed": False,
                    "atcnet_predictions_sha256": file_sha256(full_paths["atcnet"]),
                    "fbcnet_predictions_sha256": file_sha256(full_paths["fbcnet"]),
                }
            )

    write_csv(output / "per_fold_metrics.csv", fold_metric_rows)
    write_csv(output / "per_subject_seed_metrics.csv", subject_seed_rows)
    write_csv(output / "gate_convergence.csv", convergence_rows)
    write_csv(output / "e1_teacher_cache_audit.csv", teacher_audit_rows)
    write_json(
        output / "e1_teacher_cache_audit.json",
        {
            "status": "passed",
            "subject_seed_pairs": len(teacher_audit_rows),
            "fold_pairs": len(teacher_audit_rows) * len(folds),
            "input_files": len(input_manifest),
            "all_metadata_aligned": True,
            "all_fold_logits_reconstruct_full_probabilities": True,
            "all_inner_outer_partitions_disjoint": True,
            "session_e_accessed": False,
        },
    )

    aggregate_rows: list[dict[str, Any]] = []
    for arm in OUTPUT_ARMS:
        selected = [row for row in subject_seed_rows if row["arm"] == arm]
        aggregate_rows.append(
            {
                "arm": arm,
                "subject_seed_pairs": len(selected),
                "parameters": _parameter_count(arm, n_classes),
                **{
                    f"mean_{metric}": float(np.mean([float(row[metric]) for row in selected]))
                    for metric in (
                        "accuracy",
                        "balanced_accuracy",
                        "kappa",
                        "macro_f1",
                        "negative_log_likelihood",
                        "brier_score",
                        "ece",
                    )
                },
                "median_accuracy": float(np.median([float(row["accuracy"]) for row in selected])),
                "std_accuracy": float(
                    np.std(
                        [float(row["accuracy"]) for row in selected],
                        ddof=1 if len(selected) > 1 else 0,
                    )
                ),
                "session_e_accessed": False,
            }
        )
    write_csv(output / "aggregate_metrics.csv", aggregate_rows)

    reference_by_pair = {
        (int(row["subject"]), int(row["seed"])): row
        for row in subject_seed_rows
        if row["arm"] == "equal_calibrated"
    }
    paired_rows: list[dict[str, Any]] = []
    for variant in variants:
        for row in subject_seed_rows:
            if row["arm"] != variant:
                continue
            key = (int(row["subject"]), int(row["seed"]))
            reference = reference_by_pair[key]
            paired_rows.append(
                {
                    "subject": key[0],
                    "seed": key[1],
                    "candidate": variant,
                    "accuracy_delta_pp": 100.0
                    * (float(row["accuracy"]) - float(reference["accuracy"])),
                    "nll_delta": float(row["negative_log_likelihood"])
                    - float(reference["negative_log_likelihood"]),
                    "ece_delta": float(row["ece"]) - float(reference["ece"]),
                    "candidate_accuracy": float(row["accuracy"]),
                    "reference_accuracy": float(reference["accuracy"]),
                }
            )
    write_csv(output / "paired_deltas.csv", paired_rows)

    aggregate_by_arm = {str(row["arm"]): row for row in aggregate_rows}
    selected_candidate = max(
        variants,
        key=lambda arm: (
            float(aggregate_by_arm[arm]["mean_accuracy"]),
            -float(aggregate_by_arm[arm]["mean_negative_log_likelihood"]),
            -int(aggregate_by_arm[arm]["parameters"]),
        ),
    )
    selected_pairs = [row for row in paired_rows if row["candidate"] == selected_candidate]
    deltas = [float(row["accuracy_delta_pp"]) for row in selected_pairs]
    nll_deltas = [float(row["nll_delta"]) for row in selected_pairs]
    positives = sum(delta > 0.0 for delta in deltas)
    negatives = sum(delta < 0.0 for delta in deltas)
    criteria = config["promotion_gate"]
    median_delta = float(np.median(deltas))
    mean_nll_delta = float(np.mean(nll_deltas))
    selected_convergence = [
        row for row in convergence_rows if row["variant"] == selected_candidate
    ]
    all_selected_fits_converged = bool(selected_convergence) and all(
        bool(row["converged"]) for row in selected_convergence
    )
    passed = (
        median_delta >= float(criteria["minimum_median_accuracy_delta_pp"])
        and positives >= int(criteria["minimum_positive_pairs"])
        and mean_nll_delta <= float(criteria["maximum_mean_nll_delta"])
        and (
            all_selected_fits_converged
            or not bool(criteria["require_all_selected_gate_fits_converged"])
        )
    )
    gate = {
        "passed": bool(passed),
        "selected_candidate": selected_candidate,
        "selection_role": "development_candidate_selection_only",
        "reference": "equal_calibrated",
        "candidate_mean_accuracy": float(aggregate_by_arm[selected_candidate]["mean_accuracy"]),
        "reference_mean_accuracy": float(aggregate_by_arm["equal_calibrated"]["mean_accuracy"]),
        "mean_accuracy_delta_pp": float(np.mean(deltas)),
        "median_accuracy_delta_pp": median_delta,
        "positive_pairs": int(positives),
        "negative_pairs": int(negatives),
        "ties": int(len(deltas) - positives - negatives),
        "mean_nll_delta": mean_nll_delta,
        "selected_gate_fits": len(selected_convergence),
        "all_selected_gate_fits_converged": all_selected_fits_converged,
        "accuracy_delta_bootstrap_95_ci_pp": _bootstrap_mean_ci(
            deltas,
            seed=int(config["statistics"]["bootstrap_seed"]),
            n_boot=int(config["statistics"]["bootstrap_samples"]),
        ),
        "two_sided_sign_test_p": _sign_test_p(positives, negatives),
        "criteria": criteria,
        "next_stage": "E3_full_bci2a_session_t" if passed else "stop_and_diagnose_E2",
        "session_e_accessed": False,
    }
    write_json(output / "e2_promotion_gate.json", gate)
    _write_report(output, gate, aggregate_rows)
    write_json(
        output / "campaign_status.json",
        {
            "status": "completed",
            "stage": "E2_reliability_fusion_pilot",
            "e0_passed": True,
            "e1_passed": True,
            "e2_promotion_passed": bool(passed),
            "selected_candidate": selected_candidate,
            "subject_seed_pairs": len(teacher_audit_rows),
            "folds": len(fold_metric_rows) // len(OUTPUT_ARMS),
            "session_e_accessed": False,
            "fingerprint": fingerprint["combined_sha256"],
        },
    )
    print(json.dumps({"status": "completed", "output": str(output), "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
