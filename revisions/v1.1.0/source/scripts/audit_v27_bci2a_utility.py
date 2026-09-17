#!/usr/bin/env python3
"""Independently audit and gate the complete V27 utility campaign."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v27_utility import utility_gate  # noqa: E402
from dpc_snn.experiments.v62_protocol import validate_run_artifact_manifest  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import bootstrap_ci, classification_metrics  # noqa: E402
from scripts.run_v27_bci2a_utility import ARMS, RUN_FILES  # noqa: E402


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _close(actual: float, recorded: Any, name: str) -> None:
    if not np.isclose(float(actual), float(recorded), rtol=0.0, atol=2e-7):
        raise RuntimeError(f"metric mismatch for {name}: {actual} != {recorded}")


def _paired(values: np.ndarray, seed: int) -> dict[str, Any]:
    delta = np.asarray(values, dtype=float)
    nonzero = delta[~np.isclose(delta, 0.0)]
    test = wilcoxon(nonzero, method="exact") if nonzero.size else None
    ci = bootstrap_ci(delta, seed=seed, n_boot=20_000)
    return {
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "positive": int(np.count_nonzero(delta > 0.0)),
        "tie": int(np.count_nonzero(np.isclose(delta, 0.0))),
        "negative": int(np.count_nonzero(delta < 0.0)),
        "subject_bootstrap_95ci": [float(ci["low"]), float(ci["high"])],
        "exact_wilcoxon_p": float(test.pvalue) if test is not None else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    campaign = Path(args.campaign).resolve()
    output = ensure_dir(Path(args.output).resolve())
    if any(output.iterdir()):
        raise FileExistsError(f"V27 audit output must be empty: {output}")
    freeze = read_json(args.freeze)
    endpoints = np.asarray(
        freeze["utility_config"]["early_decision"]["endpoints_seconds"], dtype=float
    )
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    for subject in range(1, 10):
        for seed in range(5):
            run = campaign / f"subject_{subject:02d}" / f"seed_{seed}"
            try:
                validate_run_artifact_manifest(run, required_files=RUN_FILES, verify_hashes=True)
                metrics = read_json(run / "metrics.json")
                if metrics["freeze_sha256"] != freeze["combined_sha256"]:
                    raise RuntimeError("unit freeze hash mismatch")
                if metrics["model_updates"] is not False:
                    raise RuntimeError("utility unit allowed model updates")
                if not read_json(run / "clean_replay.json")["valid"]:
                    raise RuntimeError("clean replay was not exact")
                if not read_json(run / "state_audit.json")["identical"]:
                    raise RuntimeError("frozen state changed")

                with np.load(run / "early_predictions.npz", allow_pickle=False) as archive:
                    early_arms = archive["arms"].astype(str).tolist()
                    saved_endpoints = archive["endpoint_seconds"].astype(float)
                    early_logits = archive["logits"]
                    labels = archive["labels"]
                    trial_id = archive["trial_id"]
                if early_arms != list(ARMS) or not np.allclose(saved_endpoints, endpoints):
                    raise RuntimeError("early-decision axes differ from the freeze")
                if early_logits.shape != (len(endpoints), 2, 288, 4):
                    raise RuntimeError("early prediction shape is incomplete")
                if len(set(trial_id.astype(str).tolist())) != 288:
                    raise RuntimeError("trial identity is not unique")

                with np.load(run / "robustness_predictions.npz", allow_pickle=False) as archive:
                    robust_arms = archive["arms"].astype(str).tolist()
                    conditions = archive["conditions"].astype(str).tolist()
                    robust_logits = archive["logits"]
                    robust_labels = archive["labels"]
                    robust_trial_id = archive["trial_id"]
                if robust_arms != list(ARMS) or robust_logits.shape != (
                    len(conditions), 2, 288, 4
                ):
                    raise RuntimeError("robustness prediction shape is incomplete")
                if not np.array_equal(labels, robust_labels) or not np.array_equal(
                    trial_id, robust_trial_id
                ):
                    raise RuntimeError("early and robustness trials are not paired")
                expected_conditions = 1 + sum(
                    len(freeze["utility_config"]["robustness"][key])
                    for key in (
                        "gaussian_snr_db",
                        "channel_drop_counts",
                        "time_mask_seconds",
                        "amplitude_scales",
                    )
                )
                if len(conditions) != expected_conditions or conditions[0] != "clean":
                    raise RuntimeError("robustness grid differs from the freeze")

                early_accuracy: dict[str, list[float]] = {arm: [] for arm in ARMS}
                for endpoint_index, endpoint in enumerate(endpoints):
                    for arm_index, arm in enumerate(ARMS):
                        computed = classification_metrics(
                            labels, early_logits[endpoint_index, arm_index].argmax(axis=1), n_classes=4
                        )
                        early_accuracy[arm].append(computed["accuracy"])
                robust_accuracy: dict[str, list[float]] = {arm: [] for arm in ARMS}
                for condition_index in range(1, len(conditions)):
                    for arm_index, arm in enumerate(ARMS):
                        computed = classification_metrics(
                            labels, robust_logits[condition_index, arm_index].argmax(axis=1), n_classes=4
                        )
                        robust_accuracy[arm].append(computed["accuracy"])
                row: dict[str, Any] = {"subject": subject, "seed": seed}
                for arm in ARMS:
                    auc = float(
                        np.trapezoid(early_accuracy[arm], endpoints)
                        / (endpoints[-1] - endpoints[0])
                    )
                    robust_mean = float(np.mean(robust_accuracy[arm]))
                    final = float(early_accuracy[arm][-1])
                    _close(auc, metrics["arms"][arm]["early_accuracy_auc"], f"{arm}/auc")
                    _close(
                        robust_mean,
                        metrics["arms"][arm]["robustness_mean_accuracy"],
                        f"{arm}/robustness",
                    )
                    _close(final, metrics["arms"][arm]["final_accuracy"], f"{arm}/final")
                    row[f"{arm}_early_auc"] = auc
                    row[f"{arm}_robustness_mean"] = robust_mean
                    row[f"{arm}_final_accuracy"] = final
                operation = read_json(run / "operation_proxy.json")
                _close(
                    operation["decoder_reduction"],
                    metrics["decoder_operation_reduction"],
                    "decoder reduction",
                )
                _close(
                    operation["full_student_reduction"],
                    metrics["full_student_operation_reduction"],
                    "full-student reduction",
                )
                row["decoder_operation_reduction"] = float(operation["decoder_reduction"])
                row["full_student_operation_reduction"] = float(
                    operation["full_student_reduction"]
                )
                rows.append(row)
            except Exception as exc:
                issues.append(f"subject={subject},seed={seed}: {type(exc).__name__}: {exc}")

    write_csv(output / "run_summary.csv", rows)
    if len(rows) != 45:
        issues.append(f"audited units={len(rows)}, expected=45")
    subject_rows: list[dict[str, Any]] = []
    for subject in range(1, 10):
        group = [row for row in rows if row["subject"] == subject]
        if len(group) != 5:
            continue
        item: dict[str, Any] = {"subject": subject}
        for arm in ARMS:
            for field in ("early_auc", "robustness_mean", "final_accuracy"):
                item[f"{arm}_{field}"] = float(
                    np.mean([row[f"{arm}_{field}"] for row in group])
                )
        item["early_auc_delta"] = item["sew_clif_ce_early_auc"] - item[
            "ann_plain_ce_early_auc"
        ]
        item["robustness_delta"] = item["sew_clif_ce_robustness_mean"] - item[
            "ann_plain_ce_robustness_mean"
        ]
        subject_rows.append(item)
    write_csv(output / "subject_summary.csv", subject_rows)

    early = np.asarray([row["early_auc_delta"] for row in subject_rows], dtype=float)
    robust = np.asarray([row["robustness_delta"] for row in subject_rows], dtype=float)
    decoder_reduction = float(np.mean([row["decoder_operation_reduction"] for row in rows])) if rows else 0.0
    full_reduction = float(np.mean([row["full_student_operation_reduction"] for row in rows])) if rows else 0.0
    clean_valid = not issues and len(rows) == 45
    gate = utility_gate(
        clean_replay_valid=clean_valid,
        early_auc_gain_pp=100.0 * float(early.mean()) if early.size else -100.0,
        robustness_gain_pp=100.0 * float(robust.mean()) if robust.size else -100.0,
        decoder_reduction=decoder_reduction,
        full_student_reduction=full_reduction,
    )
    report = {
        "status": "completed" if clean_valid else "integrity_failed",
        "audited_units": len(rows),
        "early_auc_sew_minus_ann": _paired(early, 27_001) if early.size else {},
        "robustness_sew_minus_ann": _paired(robust, 27_002) if robust.size else {},
        "decoder_operation_reduction": decoder_reduction,
        "full_student_operation_reduction": full_reduction,
        "gate": gate,
        "issues": issues,
        "claim_scope": "posthoc frozen utility; operation counts are not hardware energy measurements",
    }
    write_json(output / "audit_report.json", report)
    write_json(output / "gate_decision.json", gate)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
