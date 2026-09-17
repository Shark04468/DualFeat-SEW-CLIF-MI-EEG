#!/usr/bin/env python3
"""Independently audit and summarize a completed V25 BCI2a campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v25_confirmation import validate_v25_freeze  # noqa: E402
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    validate_run_artifact_manifest,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import bootstrap_ci, classification_metrics  # noqa: E402
from scripts.run_v25_bci2a_confirmation import CHECKPOINT_FILES, TRAIN_FILES  # noqa: E402


ARMS = (
    "sew_clif_ce",
    "ann_plain_ce",
    "teacher",
    "atcnet",
    "fbcnet",
)
PREDICTION_BASENAMES = {
    "sew_clif_ce": "predictions",
    "ann_plain_ce": "ann_plain_ce_predictions",
    "teacher": "teacher_predictions",
    "atcnet": "atcnet_predictions",
    "fbcnet": "fbcnet_predictions",
}
EVALUATION_FILES = (
    "metrics.json",
    "data_access_manifest.json",
    "evaluation_manifest.json",
    "predictions.npz",
    "predictions.csv",
    "ann_plain_ce_predictions.npz",
    "ann_plain_ce_predictions.csv",
    "teacher_predictions.npz",
    "teacher_predictions.csv",
    "atcnet_predictions.npz",
    "atcnet_predictions.csv",
    "fbcnet_predictions.npz",
    "fbcnet_predictions.csv",
)
IDENTITY_FIELDS = ("label", "subject", "session", "run", "trial_id", "seed")


def _load_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"logits", "probabilities", "pred", *IDENTITY_FIELDS}
        missing = sorted(required - set(archive.files))
        if missing:
            raise RuntimeError(f"{path} misses prediction arrays: {missing}")
        return {name: np.asarray(archive[name]) for name in archive.files}


def _assert_close(actual: float, recorded: Any, *, name: str) -> None:
    if not np.isclose(float(actual), float(recorded), rtol=0.0, atol=1e-12):
        raise RuntimeError(f"metric mismatch for {name}: recomputed={actual}, recorded={recorded}")


def _paired_summary(subject_deltas: np.ndarray, *, seed: int) -> dict[str, Any]:
    values = np.asarray(subject_deltas, dtype=float)
    nonzero = values[~np.isclose(values, 0.0)]
    if nonzero.size:
        test = wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided", method="exact")
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
    else:
        statistic = 0.0
        p_value = 1.0
    ci = bootstrap_ci(values, seed=seed, n_boot=20_000)
    return {
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "subject_positive": int(np.count_nonzero(values > 0.0)),
        "subject_tie": int(np.count_nonzero(np.isclose(values, 0.0))),
        "subject_negative": int(np.count_nonzero(values < 0.0)),
        "subject_cluster_bootstrap_95ci": [float(ci["low"]), float(ci["high"])],
        "exact_wilcoxon_statistic": statistic,
        "exact_wilcoxon_p": p_value,
        "n_subjects": int(values.size),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    campaign = Path(args.campaign).resolve()
    output = ensure_dir(Path(args.output).resolve())
    if any(output.iterdir()):
        raise FileExistsError(f"audit output must be new and empty: {output}")

    freeze = validate_v25_freeze(read_json(args.freeze))
    barrier_path = campaign / "global_checkpoint_barrier.json"
    if not barrier_path.is_file():
        raise FileNotFoundError("global checkpoint barrier is missing")
    barrier = read_json(barrier_path)
    barrier_sha256 = file_sha256(barrier_path)
    barrier_text = json.dumps(barrier, sort_keys=True)
    if freeze["combined_sha256"] not in barrier_text:
        raise RuntimeError("global barrier does not bind the active freeze")

    required_files = tuple(dict.fromkeys(("manifest.json",) + TRAIN_FILES + EVALUATION_FILES))
    rows: list[dict[str, Any]] = []
    violations: list[str] = []
    prediction_cache: dict[tuple[int, int, str], dict[str, np.ndarray]] = {}

    for subject in range(1, 10):
        for seed in range(5):
            run_dir = campaign / f"subject_{subject:02d}" / f"seed_{seed}"
            try:
                validate_run_artifact_manifest(
                    run_dir,
                    required_files=required_files,
                    verify_hashes=True,
                    verify_prediction_schema=True,
                )
                metrics = read_json(run_dir / "metrics.json")
                access = read_json(run_dir / "data_access_manifest.json")
                train = read_json(run_dir / "train_manifest.json")
                hashes = {
                    name: file_sha256(run_dir / filename)
                    for name, filename in CHECKPOINT_FILES.items()
                }
                if hashes != train["checkpoint_hashes"] or hashes != metrics["checkpoint_hashes"]:
                    raise RuntimeError("checkpoint hashes differ across train, evaluation, or disk")
                if access["global_checkpoint_barrier"] != barrier_sha256:
                    raise RuntimeError("evaluation did not bind the audited global barrier")
                required_true = (
                    access["session_e_arrays_first_loaded_after_global_barrier"],
                    train["session_e_semantically_loaded"] is False,
                    metrics["session_e_checkpoint_selection"] is False,
                    metrics["session_e_gradient_updates"] is False,
                    metrics["current_v25_session_e_used_for_selection"] is False,
                    metrics["historical_bci2a_session_e_accessed"] is True,
                    all(metrics["student_state_unchanged"].values()),
                )
                if not all(required_true):
                    raise RuntimeError("data-access or immutable-state invariant failed")

                arm_predictions: dict[str, dict[str, np.ndarray]] = {}
                reference: dict[str, np.ndarray] | None = None
                row: dict[str, Any] = {
                    "subject": subject,
                    "seed": seed,
                    "n_trials": 0,
                    "sew_firing_rate": float(metrics["firing_rates"]["sew_clif_ce"]),
                    "ann_residual_scale": float(metrics["residual_scales"]["ann_plain_ce"]),
                    "sew_residual_scale": float(metrics["residual_scales"]["sew_clif_ce"]),
                }
                for arm in ARMS:
                    pred = _load_prediction(
                        run_dir / f"{PREDICTION_BASENAMES[arm]}.npz"
                    )
                    if reference is None:
                        reference = pred
                        row["n_trials"] = int(pred["label"].size)
                        if row["n_trials"] != 288:
                            raise RuntimeError(f"expected 288 Session-E trials, got {row['n_trials']}")
                        if set(np.asarray(pred["session"]).astype(str)) != {"E"}:
                            raise RuntimeError("prediction package is not Session E only")
                        trial_keys = {
                            (str(run), str(trial))
                            for run, trial in zip(pred["run"], pred["trial_id"], strict=True)
                        }
                        if len(trial_keys) != row["n_trials"]:
                            raise RuntimeError("duplicate Session-E trial identities")
                    else:
                        for field in IDENTITY_FIELDS:
                            if not np.array_equal(pred[field], reference[field]):
                                raise RuntimeError(f"paired arm identity mismatch: {arm}/{field}")
                    recomputed = classification_metrics(pred["label"], pred["pred"], n_classes=4)
                    for metric_name, value in recomputed.items():
                        _assert_close(
                            value,
                            metrics["arms"][arm][metric_name],
                            name=f"S{subject}/seed{seed}/{arm}/{metric_name}",
                        )
                        row[f"{arm}_{metric_name}"] = float(value)
                    arm_predictions[arm] = pred
                    prediction_cache[(subject, seed, arm)] = pred
                rows.append(row)
            except Exception as exc:  # retain every failure in the independent report
                violations.append(f"subject={subject},seed={seed}: {type(exc).__name__}: {exc}")

    if len(rows) != 45:
        violations.append(f"complete audited units={len(rows)}, expected=45")
    write_csv(output / "run_summary.csv", rows)

    subject_rows: list[dict[str, Any]] = []
    for subject in range(1, 10):
        group = [row for row in rows if row["subject"] == subject]
        if len(group) != 5:
            continue
        item: dict[str, Any] = {"subject": subject, "n_seeds": 5}
        for arm in ARMS:
            for metric_name in ("accuracy", "balanced_accuracy", "kappa", "macro_f1"):
                item[f"{arm}_{metric_name}"] = float(
                    np.mean([row[f"{arm}_{metric_name}"] for row in group])
                )
        for comparator in ("ann_plain_ce", "teacher", "atcnet", "fbcnet"):
            item[f"sew_minus_{comparator}_accuracy"] = (
                item["sew_clif_ce_accuracy"] - item[f"{comparator}_accuracy"]
            )
        subject_rows.append(item)
    write_csv(output / "subject_summary.csv", subject_rows)

    aggregate: dict[str, Any] = {}
    for arm in ARMS:
        aggregate[arm] = {
            metric_name: float(np.mean([row[f"{arm}_{metric_name}"] for row in rows]))
            for metric_name in ("accuracy", "balanced_accuracy", "kappa", "macro_f1")
        }
    comparisons: dict[str, Any] = {}
    for index, comparator in enumerate(("ann_plain_ce", "teacher", "atcnet", "fbcnet")):
        subject_deltas = np.asarray(
            [row[f"sew_minus_{comparator}_accuracy"] for row in subject_rows], dtype=float
        )
        unit_deltas = np.asarray(
            [row["sew_clif_ce_accuracy"] - row[f"{comparator}_accuracy"] for row in rows],
            dtype=float,
        )
        comparison = _paired_summary(subject_deltas, seed=25_100 + index)
        comparison.update(
            {
                "subject_seed_positive": int(np.count_nonzero(unit_deltas > 0.0)),
                "subject_seed_tie": int(np.count_nonzero(np.isclose(unit_deltas, 0.0))),
                "subject_seed_negative": int(np.count_nonzero(unit_deltas < 0.0)),
            }
        )
        comparisons[f"sew_clif_ce_vs_{comparator}"] = comparison

    strongest_name = max(
        ("teacher", "atcnet", "fbcnet"), key=lambda name: aggregate[name]["accuracy"]
    ) if rows else "unknown"
    vs_ann = comparisons.get("sew_clif_ce_vs_ann_plain_ce", {})
    vs_strongest = comparisons.get(f"sew_clif_ce_vs_{strongest_name}", {})
    gates = {
        "artifact_integrity": not violations,
        "sew_minus_ann_at_least_0_5pp": vs_ann.get("mean_delta", -1.0) >= 0.005,
        "at_least_7_of_9_subjects_positive_vs_ann": vs_ann.get("subject_positive", 0) >= 7,
        "within_0_3pp_of_strongest_nonstudent_baseline": vs_strongest.get(
            "mean_delta", -1.0
        ) >= -0.003,
    }
    decision = {
        "status": "pass" if all(gates.values()) else "fail",
        "gates": gates,
        "strongest_nonstudent_baseline": strongest_name,
        "historical_data_exposure": freeze["historical_data_exposure"],
        "claim_scope": "locked retrospective cross-session evaluation; not project-level blind confirmation",
        "next_step": (
            "freeze architecture and run missing same-protocol baselines plus external confirmation"
            if all(gates.values())
            else "stop expansion and diagnose the failed gate on Session-T only"
        ),
    }
    report = {
        "status": "completed" if not violations else "integrity_failed",
        "campaign": str(campaign),
        "freeze_sha256": freeze["combined_sha256"],
        "global_checkpoint_barrier_sha256": barrier_sha256,
        "audited_units": len(rows),
        "aggregate": aggregate,
        "comparisons": comparisons,
        "decision": decision,
        "violations": violations,
    }
    write_json(output / "audit_report.json", report)
    write_json(output / "decision.json", decision)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args), indent=2))


if __name__ == "__main__":
    main()
