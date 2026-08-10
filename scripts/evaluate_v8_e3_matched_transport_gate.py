#!/usr/bin/env python3
"""Independently audit and evaluate the formal V8 E3 matched-transport gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_baselines import session_t_development_view  # noqa: E402
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    source_tree_digest,
)
from dpc_snn.experiments.v8_statistics import (  # noqa: E402
    pair_subject_seed_rows,
    paired_delta_summary,
    static_delay_gate_decision,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import (  # noqa: E402
    classification_metrics,
    paired_prediction_comparison,
)
from scripts.run_v8_e1_baselines import _subject_file  # noqa: E402


CONFIG_SCHEMA = "dpc-snn-v8-e3-delay-residual-campaign/v1"
PREDICTION_FIELDS = {
    "indices",
    "labels",
    "subject",
    "session",
    "run",
    "trial_id",
    "anchor_probability",
    "full_expert_logits",
    "matched_zero_expert_logits",
    "same_weight_zero_expert_logits",
    "full_probability",
    "matched_zero_probability",
    "same_weight_zero_probability",
}
GATE_FILES = (
    "manifest.json",
    "campaign_manifest.json",
    "gate_decision.json",
    "paired_trial_diagnostics.csv",
    "per_subject_seed.csv",
    "primary_trial_predictions.csv",
    "source_tree_manifest.json",
)


def _fold_dir(root: Path, subject: int, seed: int, fold: int) -> Path:
    return root / f"subject_{subject:02d}" / f"seed_{seed}" / f"fold_{fold}"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _validate_probability(name: str, value: np.ndarray, count: int) -> None:
    if value.shape != (count, 4):
        raise RuntimeError(f"{name} has shape {value.shape}; expected {(count, 4)}")
    if not np.isfinite(value).all() or np.any(value < -1e-7) or np.any(value > 1.0 + 1e-7):
        raise RuntimeError(f"{name} contains invalid probabilities")
    if not np.allclose(value.sum(axis=1), 1.0, atol=1e-5, rtol=0.0):
        raise RuntimeError(f"{name} probabilities do not sum to one")


def _accuracy(labels: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean(np.argmax(probability, axis=1) == labels))


def _assert_close(actual: float, expected: Any, name: str) -> None:
    expected_value = float(expected)
    if not math.isfinite(actual) or not math.isclose(
        actual, expected_value, rel_tol=0.0, abs_tol=1e-7
    ):
        raise RuntimeError(f"recomputed {name}={actual} differs from stored {expected_value}")


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_fold_variant(
    directory: Path,
    *,
    variant: str,
    subject: int,
    seed: int,
    fold: int,
    expected_labels: np.ndarray,
    expected_trial_ids: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    variant_dir = directory / variant
    manifest = read_json(variant_dir / "manifest.json")
    validate_run_artifact_manifest(
        variant_dir,
        required_files=tuple(manifest["required_files"]),
        verify_hashes=True,
        verify_prediction_schema=False,
    )
    row = read_json(variant_dir / "metrics.json")
    expected_identity = {
        "variant": variant,
        "subject": int(subject),
        "seed": int(seed),
        "fold": int(fold),
        "session_e_accessed": False,
    }
    for field, value in expected_identity.items():
        if row.get(field) != value:
            raise RuntimeError(f"E3 fold metric identity mismatch at {variant_dir}: {field}")
    prediction = _load_npz(variant_dir / "outer_predictions.npz")
    if set(prediction) != PREDICTION_FIELDS:
        raise RuntimeError(f"E3 prediction fields changed at {variant_dir}")
    count = int(prediction["labels"].size)
    if count < 1:
        raise RuntimeError(f"E3 fold prediction is empty at {variant_dir}")
    indices = np.asarray(prediction["indices"], dtype=np.int64)
    labels = np.asarray(prediction["labels"], dtype=np.int64)
    if indices.shape != (count,) or len(set(indices.tolist())) != count:
        raise RuntimeError(f"E3 fold indices are invalid at {variant_dir}")
    if np.any(indices < 0) or np.any(indices >= expected_labels.size):
        raise RuntimeError(f"E3 fold indices are outside Session T at {variant_dir}")
    if not np.array_equal(labels, expected_labels[indices]):
        raise RuntimeError(f"E3 fold labels differ from source data at {variant_dir}")
    if not np.array_equal(
        prediction["trial_id"].astype(str), expected_trial_ids[indices].astype(str)
    ):
        raise RuntimeError(f"E3 fold trial IDs differ from source data at {variant_dir}")
    if not np.all(prediction["subject"] == int(subject)) or not np.all(
        prediction["session"].astype(str) == "T"
    ):
        raise RuntimeError(f"E3 fold subject/session identity changed at {variant_dir}")
    for name in (
        "anchor_probability",
        "full_probability",
        "matched_zero_probability",
        "same_weight_zero_probability",
    ):
        _validate_probability(name, np.asarray(prediction[name]), count)
    for name in (
        "full_expert_logits",
        "matched_zero_expert_logits",
        "same_weight_zero_expert_logits",
    ):
        value = np.asarray(prediction[name])
        if value.shape != (count, 4) or not np.isfinite(value).all():
            raise RuntimeError(f"{name} is invalid at {variant_dir}")
    _assert_close(
        _accuracy(labels, prediction["anchor_probability"]),
        row["anchor_accuracy"],
        "anchor accuracy",
    )
    _assert_close(
        _accuracy(labels, prediction["full_probability"]),
        row["full_accuracy"],
        "full accuracy",
    )
    _assert_close(
        _accuracy(labels, prediction["matched_zero_probability"]),
        row["matched_zero_accuracy"],
        "matched-zero accuracy",
    )
    _assert_close(
        _accuracy(labels, prediction["same_weight_zero_probability"]),
        row["same_weight_zero_accuracy"],
        "same-weight-zero accuracy",
    )
    if float(row["full_current_rms"]) <= 0.0 or float(row["zero_current_rms"]) <= 0.0:
        raise RuntimeError("matched full/zero routed currents must both be non-zero")
    if int(row["full_optimizer_steps"]) != int(row["zero_optimizer_steps"]):
        raise RuntimeError("matched full/zero optimizer budgets differ")
    if int(row["prior_audit_replicates"]) != 5:
        raise RuntimeError("formal E3 requires five fold-local prior audits")
    if not bool(row["inner_prior_stability_passed"]) or not bool(
        row["outer_prior_stability_passed"]
    ):
        raise RuntimeError("formal E3 consumed an unstable fold-local prior")
    for field in ("routing_sha256", "prior_sha256", "input_gain_sha256"):
        if not _is_sha256(row.get(field)):
            raise RuntimeError(f"matched transport invariant {field} is missing")
    return row, prediction


def _pair_metrics(
    *,
    variant: str,
    subject: int,
    seed: int,
    prediction: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    labels = np.asarray(prediction["labels"], dtype=np.int64)
    metrics = {
        name: classification_metrics(
            labels,
            np.asarray(prediction[name]).argmax(axis=1),
            n_classes=4,
        )
        for name in (
            "anchor_probability",
            "full_probability",
            "matched_zero_probability",
            "same_weight_zero_probability",
        )
    }
    return {
        "variant": variant,
        "subject": int(subject),
        "seed": int(seed),
        "trials": int(labels.size),
        "anchor_accuracy": metrics["anchor_probability"]["accuracy"],
        "full_accuracy": metrics["full_probability"]["accuracy"],
        "matched_zero_accuracy": metrics["matched_zero_probability"]["accuracy"],
        "same_weight_zero_accuracy": metrics["same_weight_zero_probability"]["accuracy"],
        "full_balanced_accuracy": metrics["full_probability"]["balanced_accuracy"],
        "full_macro_f1": metrics["full_probability"]["macro_f1"],
        "full_kappa": metrics["full_probability"]["kappa"],
        "full_minus_matched_zero_pp": 100.0
        * (
            metrics["full_probability"]["accuracy"]
            - metrics["matched_zero_probability"]["accuracy"]
        ),
        "full_minus_same_weight_zero_pp": 100.0
        * (
            metrics["full_probability"]["accuracy"]
            - metrics["same_weight_zero_probability"]["accuracy"]
        ),
        "full_minus_anchor_pp": 100.0
        * (
            metrics["full_probability"]["accuracy"]
            - metrics["anchor_probability"]["accuracy"]
        ),
        "session_e_accessed": False,
    }


def evaluate_campaign(
    *,
    campaign: Path,
    data_root: Path,
    config: Mapping[str, Any],
    source_digest: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    subjects = [int(value) for value in config["subjects"]]
    seeds = [int(value) for value in config["seeds"]]
    folds = [int(value) for value in config["folds"]]
    variants = [str(value) for value in config["variants"]]
    primary = str(config["primary_variant"])
    matched_ann = str(config["matched_ann_variant"])
    if primary not in variants or matched_ann not in variants or primary == matched_ann:
        raise RuntimeError("formal E3 primary/matched-ANN variants are invalid")

    pair_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    combined: dict[tuple[int, int, str], dict[str, list[np.ndarray]]] = {}
    for subject in subjects:
        data = load_processed_npz(_subject_file(data_root, subject))
        _, labels, metadata, access = session_t_development_view(data)
        if bool(access.get("session_e_accessed")):
            raise RuntimeError("E3 gate unexpectedly accessed Session E")
        trial_ids = np.asarray([str(row["trial_id"]) for row in metadata])
        if labels.size != int(config["gate"]["expected_trials_per_pair"]):
            raise RuntimeError(f"Subject {subject} Session-T trial count changed")
        for seed in seeds:
            anchor_reference: dict[int, np.ndarray] = {}
            for fold in folds:
                directory = _fold_dir(campaign, subject, seed, fold)
                fold_manifest = read_json(directory / "manifest.json")
                validate_run_artifact_manifest(
                    directory,
                    required_files=tuple(fold_manifest["required_files"]),
                    verify_hashes=True,
                    verify_prediction_schema=False,
                )
                status = read_json(directory / "campaign_status.json")
                if (
                    status.get("source_tree_sha256") != source_digest
                    or status.get("session_e_accessed") is not False
                    or status.get("openbmi_s2_accessed") is not False
                ):
                    raise RuntimeError(f"E3 fold source/held-out status is invalid: {directory}")
                for variant in variants:
                    metric, prediction = _validate_fold_variant(
                        directory,
                        variant=variant,
                        subject=subject,
                        seed=seed,
                        fold=fold,
                        expected_labels=labels,
                        expected_trial_ids=trial_ids,
                    )
                    key = (subject, seed, variant)
                    bucket = combined.setdefault(
                        key,
                        {name: [] for name in PREDICTION_FIELDS},
                    )
                    for name in PREDICTION_FIELDS:
                        bucket[name].append(np.asarray(prediction[name]))
                    indices = np.asarray(prediction["indices"], dtype=np.int64)
                    anchor = np.asarray(prediction["anchor_probability"])
                    if variant == variants[0]:
                        anchor_reference[fold] = anchor
                    elif not np.array_equal(anchor_reference[fold], anchor):
                        raise RuntimeError("matched E3 variants use different anchor probabilities")
                    if metric["delay_evidence_space"] != config["delay"]["evidence_space"]:
                        raise RuntimeError("E3 evidence space differs from the registered campaign")
                    if metric["delay_transport_space"] != config["delay"]["transport_space"]:
                        raise RuntimeError("E3 transport space differs from the registered campaign")
                    if int(metric["delay_transport_nodes"]) != int(
                        config["delay"]["transport_nodes"]
                    ):
                        raise RuntimeError("E3 transport node count differs from the campaign")
                    if indices.size != len(set(indices.tolist())):
                        raise RuntimeError("duplicate indices within one E3 fold")

            for variant in variants:
                key = (subject, seed, variant)
                merged = {
                    name: np.concatenate(combined[key][name], axis=0)
                    for name in PREDICTION_FIELDS
                }
                order = np.argsort(merged["indices"])
                merged = {name: value[order] for name, value in merged.items()}
                if not np.array_equal(
                    merged["indices"], np.arange(labels.size, dtype=np.int64)
                ):
                    raise RuntimeError(
                        f"E3 folds do not form exact Session-T OOF coverage for S{subject}/seed{seed}"
                    )
                if not np.array_equal(merged["labels"], labels) or not np.array_equal(
                    merged["trial_id"].astype(str), trial_ids.astype(str)
                ):
                    raise RuntimeError("merged E3 OOF identities differ from source data")
                pair_rows.append(
                    _pair_metrics(
                        variant=variant,
                        subject=subject,
                        seed=seed,
                        prediction=merged,
                    )
                )
                combined[key] = {name: [value] for name, value in merged.items()}

            primary_prediction = {
                name: combined[(subject, seed, primary)][name][0]
                for name in PREDICTION_FIELDS
            }
            full_pred = primary_prediction["full_probability"].argmax(axis=1)
            zero_pred = primary_prediction["matched_zero_probability"].argmax(axis=1)
            same_pred = primary_prediction["same_weight_zero_probability"].argmax(axis=1)
            diagnostics.append(
                {
                    "subject": subject,
                    "seed": seed,
                    "comparison": "full_vs_matched_zero",
                    **paired_prediction_comparison(labels, zero_pred, full_pred),
                }
            )
            diagnostics.append(
                {
                    "subject": subject,
                    "seed": seed,
                    "comparison": "full_vs_same_weight_zero",
                    **paired_prediction_comparison(labels, same_pred, full_pred),
                }
            )
            for index in range(labels.size):
                row: dict[str, Any] = {
                    "subject": subject,
                    "seed": seed,
                    "index": index,
                    "trial_id": str(primary_prediction["trial_id"][index]),
                    "run": str(primary_prediction["run"][index]),
                    "label": int(labels[index]),
                    "anchor_pred": int(primary_prediction["anchor_probability"][index].argmax()),
                    "full_pred": int(full_pred[index]),
                    "matched_zero_pred": int(zero_pred[index]),
                    "same_weight_zero_pred": int(same_pred[index]),
                }
                for condition in (
                    "anchor_probability",
                    "full_probability",
                    "matched_zero_probability",
                    "same_weight_zero_probability",
                ):
                    short = condition.replace("_probability", "")
                    for class_index in range(4):
                        row[f"{short}_p{class_index}"] = float(
                            primary_prediction[condition][index, class_index]
                        )
                trial_rows.append(row)

    primary_rows = [row for row in pair_rows if row["variant"] == primary]
    matched_rows = [row for row in pair_rows if row["variant"] == matched_ann]
    expected_pairs = int(config["gate"]["expected_subject_seed_pairs"])
    if len(primary_rows) != expected_pairs or len(matched_rows) != expected_pairs:
        raise RuntimeError("formal E3 subject-seed coverage is incomplete")
    full_vs_zero = pair_subject_seed_rows(
        [
            {"subject": row["subject"], "seed": row["seed"], "accuracy": row["matched_zero_accuracy"]}
            for row in primary_rows
        ],
        [
            {"subject": row["subject"], "seed": row["seed"], "accuracy": row["full_accuracy"]}
            for row in primary_rows
        ],
    )
    gate = static_delay_gate_decision(
        full_vs_zero,
        expected_pairs=expected_pairs,
        minimum_median_gain_pp=float(config["gate"]["minimum_median_gain_pp"]),
        minimum_positive_pairs=int(config["gate"]["minimum_positive_pairs"]),
        seed=int(config["gate"]["bootstrap_seed"]),
        bootstrap_samples=int(config["gate"]["bootstrap_samples"]),
    )
    full_vs_same = pair_subject_seed_rows(
        [
            {"subject": row["subject"], "seed": row["seed"], "accuracy": row["same_weight_zero_accuracy"]}
            for row in primary_rows
        ],
        [
            {"subject": row["subject"], "seed": row["seed"], "accuracy": row["full_accuracy"]}
            for row in primary_rows
        ],
    )
    snn_vs_ann = pair_subject_seed_rows(
        [
            {"subject": row["subject"], "seed": row["seed"], "accuracy": row["full_accuracy"]}
            for row in matched_rows
        ],
        [
            {"subject": row["subject"], "seed": row["seed"], "accuracy": row["full_accuracy"]}
            for row in primary_rows
        ],
    )
    decision = {
        "status": "completed",
        "stage": "E3_MATCHED_TRANSPORT_GATE",
        "protocol": "bci2a_session_t_nested_six_fold_oof",
        "passed": bool(gate["passed"]),
        "decision": (
            "advance_to_static_slow_cross_band"
            if gate["passed"]
            else "stop_delay_title_promotion_and_retain_as_ablation"
        ),
        "primary_variant": primary,
        "matched_ann_variant": matched_ann,
        "comparison": "separately_trained_full_vs_point_zero_with_matched_route_edge_confidence_gain_and_update_budget",
        "gate": gate,
        "same_weight_zero_diagnostic": paired_delta_summary(
            full_vs_same,
            seed=int(config["gate"]["bootstrap_seed"]) + 1,
            bootstrap_samples=int(config["gate"]["bootstrap_samples"]),
        ),
        "primary_snn_vs_matched_ann_diagnostic": paired_delta_summary(
            snn_vs_ann,
            seed=int(config["gate"]["bootstrap_seed"]) + 2,
            bootstrap_samples=int(config["gate"]["bootstrap_samples"]),
        ),
        "fold_manifest_and_hash_validation": True,
        "trial_metrics_recomputed": True,
        "exact_oof_trials_per_pair": int(config["gate"]["expected_trials_per_pair"]),
        "source_tree_sha256": source_digest,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    return pair_rows, diagnostics, trial_rows, decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config",
        default="configs/experiments/v8_e3_matched_transport_campaign.yaml",
    )
    args = parser.parse_args()

    campaign = Path(args.campaign).resolve()
    data_root = Path(args.data).resolve()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA:
        raise RuntimeError("E3 matched-transport gate config schema changed")
    source_tree = collect_source_tree_manifest(ROOT)
    source_digest = source_tree_digest(source_tree)
    campaign_manifest = read_json(campaign / "manifest.json")
    validate_run_artifact_manifest(
        campaign,
        required_files=tuple(campaign_manifest["required_files"]),
        verify_hashes=True,
        verify_prediction_schema=False,
    )
    status = read_json(campaign / "campaign_status.json")
    expected_runs = len(config["subjects"]) * len(config["seeds"]) * len(config["folds"])
    if (
        status.get("status") != "completed"
        or status.get("stage") != "E3_DELAY_RESIDUAL_CAMPAIGN"
        or status.get("protocol") != config["protocol"]
        or status.get("source_tree_sha256") != source_digest
        or status.get("subjects") != config["subjects"]
        or status.get("seeds") != config["seeds"]
        or status.get("folds") != config["folds"]
        or status.get("variants") != config["variants"]
        or int(status.get("fold_runs", -1)) != expected_runs
        or not bool(status.get("full_registered_contract"))
        or bool(status.get("canary"))
        or status.get("session_e_accessed") is not False
        or status.get("openbmi_s2_accessed") is not False
    ):
        raise RuntimeError("E3 campaign is incomplete, unlocked, or not the formal contract")
    contract = read_json(campaign / "campaign_contract.json")
    if contract.get("config_sha256") != file_sha256(config_path):
        raise RuntimeError("E3 gate config differs from the campaign config")

    pair_rows, diagnostics, trial_rows, decision = evaluate_campaign(
        campaign=campaign,
        data_root=data_root,
        config=config,
        source_digest=source_digest,
    )
    output = ensure_dir(Path(args.output).resolve())
    write_csv(output / "per_subject_seed.csv", pair_rows)
    write_csv(output / "paired_trial_diagnostics.csv", diagnostics)
    write_csv(output / "primary_trial_predictions.csv", trial_rows)
    write_json(output / "gate_decision.json", decision)
    write_json(output / "source_tree_manifest.json", source_tree)
    write_json(
        output / "campaign_manifest.json",
        {
            "path": str(campaign),
            "manifest_sha256": file_sha256(campaign / "manifest.json"),
            "campaign_contract_sha256": contract["combined_sha256"],
            "config_sha256": file_sha256(config_path),
        },
    )
    write_run_artifact_manifest(output, required_files=GATE_FILES)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
