#!/usr/bin/env python3
"""Audit frozen V8 ablations and build the final claim-to-evidence map."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    validate_prediction_schema,
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_protocol import validate_v8_freeze_manifest  # noqa: E402
from dpc_snn.experiments.v8_statistics import (  # noqa: E402
    pair_subject_seed_rows,
    paired_delta_summary,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import (  # noqa: E402
    classification_metrics,
    paired_prediction_comparison,
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty table: {path}")
    return rows


def _prediction(path: Path) -> dict[str, np.ndarray]:
    validate_prediction_schema(path, path.with_suffix(".csv"))
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _identity(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> None:
    for field in ("subject", "session", "run", "trial_id", "label", "seed"):
        if not np.array_equal(first[field], second[field]):
            raise RuntimeError(f"E9 paired identity mismatch for {field}")


def _close(first: float, second: float, tolerance: float = 1.0e-10) -> bool:
    return bool(np.isclose(float(first), float(second), rtol=0.0, atol=tolerance))


def _exact_sign_flip_p(subject_delta: np.ndarray) -> float:
    delta = np.asarray(subject_delta, dtype=np.float64)
    delta = delta[np.abs(delta) > 0.0]
    if delta.size == 0:
        return 1.0
    observed = abs(float(delta.mean()))
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=int(delta.size)):
        values.append(abs(float(np.mean(delta * np.asarray(signs)))))
    return float(np.mean(np.asarray(values) >= observed - 1.0e-15))


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, (total - rank) * float(p_values[name]))
        adjusted[name] = min(1.0, running)
    return adjusted


def _reference_prediction(
    e6: Path,
    freeze: dict[str, Any],
    subject: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], str]:
    run = e6 / "frozen_primary" / f"subject_{subject:02d}" / f"seed_{seed}"
    manifest = read_json(run / "manifest.json")
    validate_run_artifact_manifest(
        run,
        required_files=tuple(manifest["required_files"]),
        verify_hashes=True,
        verify_prediction_schema=True,
    )
    if bool(freeze["architecture"]["delay"]["enabled"]):
        return _prediction(run / "matched_zero_predictions.npz"), "matched_zero"
    return _prediction(run / "predictions.npz"), "frozen_primary"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e9", required=True)
    parser.add_argument("--e6", required=True)
    parser.add_argument("--e6-audit", required=True)
    parser.add_argument("--e7", required=True)
    parser.add_argument("--e8-audit", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default="configs/experiments/v8_e9_frozen_ablation.yaml"
    )
    args = parser.parse_args()

    e9 = Path(args.e9).resolve()
    e6 = Path(args.e6).resolve()
    e7 = Path(args.e7).resolve()
    output = ensure_dir(Path(args.output).resolve())
    freeze = validate_v8_freeze_manifest(Path(args.freeze).resolve())
    config = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    status = read_json(e9 / "campaign_status.json")
    e6_audit = read_json(Path(args.e6_audit).resolve() / "audit_report.json")
    e7_status = read_json(e7 / "campaign_status.json")
    e7_gate = read_json(e7 / "gate_decision.json")
    e8_audit_root = Path(args.e8_audit).resolve()
    e8_audit = read_json(e8_audit_root / "audit_report.json")
    e8_model_rows = _rows(e8_audit_root / "model_summary.csv")
    variants = list(config["retrained_single_variable_ablations"])
    expected = {
        (variant, subject, seed)
        for variant in variants
        for subject in config["subjects"]
        for seed in config["seeds"]
    }
    if (
        status.get("status") != "completed"
        or not bool(status.get("full_registered_contract"))
        or status.get("freeze_sha256") != freeze["combined_sha256"]
        or status.get("variants") != variants
        or e6_audit.get("status") != "passed"
        or e6_audit.get("freeze_sha256") != freeze["combined_sha256"]
        or e7_status.get("status") != "completed"
        or e7_status.get("freeze_sha256") != freeze["combined_sha256"]
        or e8_audit.get("status") != "passed"
        or e8_audit.get("freeze_sha256") != freeze["combined_sha256"]
    ):
        raise RuntimeError("E9, E6 audit, or E7 is incomplete or bound to another freeze")
    rows = _rows(e9 / "summary.csv")
    observed = {
        (row["variant"], int(row["subject"]), int(row["seed"])) for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise RuntimeError("E9 coverage is incomplete or duplicated")

    reference: dict[tuple[int, int], tuple[dict[str, np.ndarray], str]] = {}
    ablation_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in variants}
    reference_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in variants}
    trial_rows: list[dict[str, Any]] = []
    for row in rows:
        variant = row["variant"]
        subject, seed = int(row["subject"]), int(row["seed"])
        key = (subject, seed)
        if key not in reference:
            reference[key] = _reference_prediction(e6, freeze, subject, seed)
        reference_prediction, reference_name = reference[key]
        run = e9 / variant / f"subject_{subject:02d}" / f"seed_{seed}"
        manifest = read_json(run / "manifest.json")
        validate_run_artifact_manifest(
            run,
            required_files=tuple(manifest["required_files"]),
            verify_hashes=True,
            verify_prediction_schema=True,
        )
        prediction = _prediction(run / "predictions.npz")
        _identity(reference_prediction, prediction)
        metrics = classification_metrics(
            prediction["label"], prediction["pred"], n_classes=4
        )
        for field in ("accuracy", "balanced_accuracy", "kappa", "macro_f1"):
            if not _close(row[field], metrics[field]):
                raise RuntimeError(f"E9 metric drift for {variant}, S{subject}, seed{seed}")
        state = read_json(run / "state_audit.json")
        access = read_json(run / "data_access_manifest.json")
        if (
            not bool(state.get("identical"))
            or access.get("session_e_checkpoint_selection") is not False
            or access.get("session_e_gradient_updates") is not False
        ):
            raise RuntimeError("E9 held-out state/access contract failed")
        reference_metric = classification_metrics(
            reference_prediction["label"], reference_prediction["pred"], n_classes=4
        )
        ablation_rows[variant].append(
            {"subject": subject, "seed": seed, "accuracy": metrics["accuracy"]}
        )
        reference_rows[variant].append(
            {
                "subject": subject,
                "seed": seed,
                "accuracy": reference_metric["accuracy"],
            }
        )
        trial_rows.append(
            {
                "comparison": f"{reference_name}_minus_{variant}",
                "subject": subject,
                "seed": seed,
                **paired_prediction_comparison(
                    prediction["label"], prediction["pred"], reference_prediction["pred"]
                ),
            }
        )

    summaries: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    raw_p: dict[str, float] = {}
    for index, variant in enumerate(variants):
        paired = pair_subject_seed_rows(
            ablation_rows[variant], reference_rows[variant]
        )
        summary = paired_delta_summary(paired, seed=20260730 + index)
        subject_delta = np.asarray(
            [
                np.mean(
                    [
                        float(row["delta_second_minus_first"])
                        for row in paired
                        if int(row["subject"]) == subject
                    ]
                )
                for subject in config["subjects"]
            ]
        )
        raw_p[variant] = _exact_sign_flip_p(subject_delta)
        paired_rows.extend(
            {"ablation": variant, **row} for row in paired
        )
        summaries.append(
            {
                "ablation": variant,
                "reference_minus_ablation_mean_pp": 100.0
                * summary["subject_macro_mean_delta"],
                "reference_minus_ablation_median_pp": 100.0
                * summary["subject_macro_median_delta"],
                "ci95_low_pp": 100.0 * summary["subject_bootstrap_mean_ci95_low"],
                "ci95_high_pp": 100.0 * summary["subject_bootstrap_mean_ci95_high"],
                "positive_subjects": summary["positive_subjects"],
                "exact_subject_sign_flip_p": raw_p[variant],
            }
        )
    adjusted = _holm(raw_p)
    for row in summaries:
        row["holm_adjusted_p"] = adjusted[str(row["ablation"])]

    e6_delay = e6_audit.get("delay_full_minus_matched_zero")
    e6_snn = e6_audit.get("primary_minus_matched_ann")
    cross_band_passed = bool(
        freeze["development_evidence"].get("e3_cross_band_gate_passed", False)
    )
    by_variant = {str(row["ablation"]): row for row in summaries}
    external_primary = next(
        row for row in e8_model_rows if row["arm"] == "frozen_primary"
    )
    claims = [
        {
            "claim": "covariance branch contributes to frozen accuracy",
            "evidence": "E9 no_covariance paired T-to-E ablation",
            "supported": by_variant["no_covariance"]["ci95_low_pp"] > 0.0,
            "effect_pp": by_variant["no_covariance"]["reference_minus_ablation_mean_pp"],
        },
        {
            "claim": "statistical branch contributes to frozen accuracy",
            "evidence": "E9 no_statistics paired T-to-E ablation",
            "supported": by_variant["no_statistics"]["ci95_low_pp"] > 0.0,
            "effect_pp": by_variant["no_statistics"]["reference_minus_ablation_mean_pp"],
        },
        {
            "claim": "temporal branch contributes to frozen accuracy",
            "evidence": "E9 no_temporal paired T-to-E ablation",
            "supported": by_variant["no_temporal"]["ci95_low_pp"] > 0.0,
            "effect_pp": by_variant["no_temporal"]["reference_minus_ablation_mean_pp"],
        },
        {
            "claim": "augmentation contributes to frozen accuracy",
            "evidence": "E9 no_augmentation paired T-to-E ablation",
            "supported": by_variant["no_augmentation"]["ci95_low_pp"] > 0.0,
            "effect_pp": by_variant["no_augmentation"]["reference_minus_ablation_mean_pp"],
        },
        {
            "claim": "spiking decoder is necessary or utility-positive",
            "evidence": "E6 matched ANN plus E7 frozen utility gate",
            "supported": bool(e6_snn is not None and e7_gate.get("passed")),
            "effect_pp": None if e6_snn is None else 100.0 * e6_snn["subject_macro_mean_delta"],
        },
        {
            "claim": "delay improves frozen held-out accuracy",
            "evidence": "E3 development gate plus E6 full versus matched-zero",
            "supported": bool(
                e6_delay is not None
                and e6_delay["subject_macro_median_delta"] > 0.0
            ),
            "effect_pp": None if e6_delay is None else 100.0 * e6_delay["subject_macro_mean_delta"],
        },
        {
            "claim": "cross-band delay is independently supported",
            "evidence": "E3 preregistered sequential cross-band gate",
            "supported": cross_band_passed,
            "effect_pp": None,
        },
        {
            "claim": "SNN utility or robustness advantage",
            "evidence": "E7 frozen early/robustness/calibration/operation package",
            "supported": bool(e7_gate.get("passed")),
            "effect_pp": None,
        },
        {
            "claim": "frozen architecture was externally confirmed without S2 tuning",
            "evidence": "E8 OpenBMI S1-to-S2 external audit",
            "supported": True,
            "effect_pp": 100.0 * (float(external_primary["subject_macro_accuracy"]) - 0.5),
        },
    ]
    report = {
        "status": "passed",
        "stage": "E9_AUDIT",
        "freeze_sha256": freeze["combined_sha256"],
        "runs_audited": len(rows),
        "variants": variants,
        "all_artifact_hashes_verified": True,
        "all_metrics_recomputed": True,
        "all_trial_identities_paired": True,
        "holm_family_size": len(variants),
        "reused_e6_delay_control": e6_delay is not None,
        "reused_e6_snn_control": e6_snn is not None,
        "reused_e7_utility": True,
        "reused_e8_external_confirmation": True,
        "openbmi_s2_accessed": True,
    }
    write_csv(output / "ablation_summary.csv", summaries)
    write_csv(output / "paired_subject_seed_results.csv", paired_rows)
    write_csv(output / "paired_trial_diagnostics.csv", trial_rows)
    write_csv(output / "claim_to_evidence.csv", claims)
    write_json(output / "audit_report.json", report)
    write_run_artifact_manifest(
        output,
        required_files=(
            "manifest.json",
            "audit_report.json",
            "ablation_summary.csv",
            "paired_subject_seed_results.csv",
            "paired_trial_diagnostics.csv",
            "claim_to_evidence.csv",
        ),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
