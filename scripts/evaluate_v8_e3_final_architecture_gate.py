#!/usr/bin/env python3
"""Select the frozen V8 branch using fail-closed Session-T evidence gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    source_tree_digest,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402


CONFIG_SCHEMA = "dpc-snn-v8-e3-final-architecture-gate/v1"
FEASIBILITY_STAGE = "E3_DELAY_PRIOR_FEASIBILITY_GATE"
E3_STAGE = "E3_MATCHED_TRANSPORT_GATE"
E3_CAMPAIGN_STAGE = "E3_DELAY_RESIDUAL_CAMPAIGN"
PROTOCOL = "bci2a_session_t_nested_six_fold_oof"
E4_SCHEMA = "dpc-snn-v8-e4-entropy-residual-result/v1"
OUTPUT_FILES = (
    "manifest.json",
    "decision.json",
    "pair_diagnostics.csv",
    "input_manifest.json",
    "resolved_config.yaml",
    "source_tree_manifest.json",
)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty final-gate input: {path}")
    return rows


def _mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("final architecture gate received empty/non-finite values")
    return float(array.mean())


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"final architecture gate received non-finite {name}")
    return result


def _close(first: Any, second: Any, name: str, *, atol: float = 1e-12) -> None:
    if not math.isclose(
        _finite_float(first, name),
        _finite_float(second, name),
        rel_tol=0.0,
        abs_tol=atol,
    ):
        raise RuntimeError(f"final architecture gate recomputation differs: {name}")


def _validate_root(path: Path) -> dict[str, Any]:
    manifest = read_json(path / "manifest.json")
    validate_run_artifact_manifest(
        path,
        required_files=tuple(manifest["required_files"]),
        verify_hashes=True,
        verify_prediction_schema=False,
    )
    if manifest.get("status") != "completed":
        raise RuntimeError(f"final-gate input manifest is not completed: {path}")
    return manifest


def _validate_e4_gate(path: Path, fallback: Mapping[str, Any]) -> dict[str, Any]:
    e4 = read_json(path)
    if (
        e4.get("schema") != E4_SCHEMA
        or e4.get("passed") is not True
        or e4.get("selected_snn_variant") != fallback["selected_variant"]
        or e4.get("matched_ann_control") != fallback["matched_ann_control"]
        or e4.get("session_e_accessed") is not False
        or e4.get("openbmi_s2_accessed") is not False
    ):
        raise RuntimeError("confirmed E4 fallback evidence is invalid")
    thresholds = dict(e4.get("thresholds", {}))
    expected_pairs = {
        (int(subject), int(seed))
        for subject in thresholds.get("subjects", [])
        for seed in thresholds.get("seeds", [])
    }
    rows = list(e4.get("pair_rows", []))
    keys = [(int(row["subject"]), int(row["seed"])) for row in rows]
    if (
        not expected_pairs
        or len(keys) != len(set(keys))
        or set(keys) != expected_pairs
        or int(e4.get("confirmation_pairs", -1)) != len(expected_pairs)
    ):
        raise RuntimeError("confirmed E4 pair coverage is invalid")
    firing_low, firing_high = [
        _finite_float(value, "E4 firing threshold")
        for value in thresholds["firing_rate_interval"]
    ]
    snn: list[float] = []
    ann: list[float] = []
    anchor: list[float] = []
    positive = 0
    pair_floor: list[bool] = []
    firing_valid: list[bool] = []
    for row in rows:
        snn_accuracy = _finite_float(row["snn_accuracy"], "E4 SNN accuracy")
        ann_accuracy = _finite_float(
            row["matched_ann_accuracy"], "E4 matched ANN accuracy"
        )
        anchor_accuracy = _finite_float(row["anchor_accuracy"], "E4 anchor accuracy")
        firing = _finite_float(row["snn_mean_firing_rate"], "E4 firing rate")
        delta_anchor = 100.0 * (snn_accuracy - anchor_accuracy)
        delta_ann = 100.0 * (snn_accuracy - ann_accuracy)
        _close(delta_anchor, row["snn_delta_vs_anchor_pp"], "E4 delta vs anchor")
        _close(delta_ann, row["snn_delta_vs_matched_ann_pp"], "E4 delta vs ANN")
        if bool(row["positive_vs_anchor"]) != (delta_anchor > 0.0):
            raise RuntimeError("confirmed E4 positive-pair flag differs")
        if bool(row["passes_anchor_floor"]) != (
            delta_anchor >= float(thresholds["per_pair_snn_vs_anchor_floor_pp"])
        ):
            raise RuntimeError("confirmed E4 pair-floor flag differs")
        if bool(row["passes_firing_interval"]) != (firing_low <= firing <= firing_high):
            raise RuntimeError("confirmed E4 firing flag differs")
        snn.append(snn_accuracy)
        ann.append(ann_accuracy)
        anchor.append(anchor_accuracy)
        positive += int(delta_anchor > 0.0)
        pair_floor.append(delta_anchor >= float(thresholds["per_pair_snn_vs_anchor_floor_pp"]))
        firing_valid.append(firing_low <= firing <= firing_high)
    macro_snn = _mean(snn)
    macro_ann = _mean(ann)
    macro_anchor = _mean(anchor)
    macro_ann_delta = 100.0 * (macro_snn - macro_ann)
    macro_anchor_delta = 100.0 * (macro_snn - macro_anchor)
    recomputed = {
        "macro_snn_vs_matched_ann": macro_ann_delta
        >= float(thresholds["macro_snn_vs_matched_ann_minimum_delta_pp"]),
        "macro_snn_vs_anchor": macro_anchor_delta
        >= float(thresholds["macro_snn_vs_anchor_minimum_delta_pp"]),
        "minimum_positive_pairs": positive
        >= int(thresholds["minimum_positive_subject_seed_pairs"]),
        "all_pairs_above_anchor_floor": all(pair_floor),
        "all_pairs_have_nondegenerate_firing": all(firing_valid),
    }
    if e4.get("criteria") != recomputed or not all(recomputed.values()):
        raise RuntimeError("confirmed E4 criteria do not independently recompute")
    _close(macro_snn, e4["macro_snn_accuracy"], "E4 macro SNN accuracy")
    _close(macro_ann, e4["macro_matched_ann_accuracy"], "E4 macro ANN accuracy")
    _close(macro_anchor, e4["macro_anchor_accuracy"], "E4 macro anchor accuracy")
    _close(
        macro_ann_delta,
        e4["macro_snn_delta_vs_matched_ann_pp"],
        "E4 macro SNN-vs-ANN delta",
    )
    _close(
        macro_anchor_delta,
        e4["macro_snn_delta_vs_anchor_pp"],
        "E4 macro SNN-vs-anchor delta",
    )
    if int(e4.get("positive_subject_seed_pairs", -1)) != positive:
        raise RuntimeError("confirmed E4 positive-pair count differs")
    return e4


def _validate_feasibility_gate(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    _validate_root(path)
    decision = read_json(path / "feasibility_decision.json")
    if (
        decision.get("status") != "completed"
        or decision.get("stage") != FEASIBILITY_STAGE
        or decision.get("session_e_accessed") is not False
        or decision.get("openbmi_s2_accessed") is not False
        or decision.get("threshold_relaxation_after_observation") is not False
        or decision.get("metrics_recomputed_from_replicate_arrays") is not True
    ):
        raise RuntimeError("E3 prior-feasibility gate is invalid or unlocked")
    rows = _csv_rows(path / "per_subject_fold.csv")
    keys = [(int(row["subject"]), int(row["fold"])) for row in rows]
    if len(keys) != len(set(keys)) or len(keys) != int(
        decision["registered_subject_folds"]
    ):
        raise RuntimeError("E3 prior-feasibility coverage is invalid")
    passing = sum(str(row["passed"]).lower() == "true" for row in rows)
    if passing != int(decision["passing_subject_folds"]):
        raise RuntimeError("E3 prior-feasibility pass count differs")
    passed = passing == len(rows)
    if bool(decision["passed"]) != passed or bool(
        decision["classifier_training_authorized"]
    ) != passed:
        raise RuntimeError("E3 prior-feasibility decision is inconsistent")
    return decision, rows


def _fallback_decision(
    feasibility: Mapping[str, Any], e4: Mapping[str, Any]
) -> dict[str, Any]:
    if bool(feasibility["passed"]):
        raise RuntimeError("cannot use the feasibility-failure branch after a pass")
    return {
        "status": "completed",
        "stage": "E3_FINAL_ARCHITECTURE_GATE",
        "protocol": PROTOCOL,
        "selected_branch": "confirmed_e4_sequence_residual",
        "selection_reason": "registered_delay_prior_feasibility_failed",
        "full_delay_branch_passed": False,
        "zero_delay_branch_passed": False,
        "full_delay_criteria": {"prior_feasibility": False},
        "zero_delay_criteria": {"matched_transport_prerequisite": False},
        "summaries_pp": {},
        "e3_prior_feasibility_passed": False,
        "e3_delay_gate_evaluated": False,
        "e3_delay_gate_passed": False,
        "e4_fallback_gate_passed": True,
        "e4_macro_accuracy": _finite_float(
            e4["macro_snn_accuracy"], "E4 macro accuracy"
        ),
        "post_session_e_selection": False,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }


def evaluate_final_gate(
    *,
    feasibility_gate: Path,
    e4_gate_path: Path,
    config: Mapping[str, Any],
    campaign: Path | None = None,
    e3_gate: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    feasibility, feasibility_rows = _validate_feasibility_gate(feasibility_gate)
    e4 = _validate_e4_gate(e4_gate_path, config["fallback_e4"])
    if not bool(feasibility["passed"]):
        diagnostics = [
            {"diagnostic_scope": "prior_feasibility", **row}
            for row in feasibility_rows
        ]
        return diagnostics, _fallback_decision(feasibility, e4)
    if campaign is None or e3_gate is None:
        raise RuntimeError("passed delay feasibility requires formal E3 campaign and gate")

    _validate_root(campaign)
    _validate_root(e3_gate)
    campaign_status = read_json(campaign / "campaign_status.json")
    campaign_contract = read_json(campaign / "campaign_contract.json")
    e3_decision = read_json(e3_gate / "gate_decision.json")
    e3_reference = read_json(e3_gate / "campaign_manifest.json")
    expected = config["expected"]
    expected_runs = (
        len(expected["subjects"]) * len(expected["seeds"]) * len(expected["folds"])
    )
    if (
        campaign_status.get("status") != "completed"
        or campaign_status.get("stage") != E3_CAMPAIGN_STAGE
        or campaign_status.get("protocol") != PROTOCOL
        or not bool(campaign_status.get("full_registered_contract"))
        or bool(campaign_status.get("canary"))
        or int(campaign_status.get("fold_runs", -1)) != expected_runs
        or campaign_status.get("session_e_accessed") is not False
        or campaign_status.get("openbmi_s2_accessed") is not False
        or e3_decision.get("status") != "completed"
        or e3_decision.get("stage") != E3_STAGE
        or e3_decision.get("protocol") != PROTOCOL
        or e3_decision.get("fold_manifest_and_hash_validation") is not True
        or e3_decision.get("trial_metrics_recomputed") is not True
        or e3_decision.get("session_e_accessed") is not False
        or e3_decision.get("openbmi_s2_accessed") is not False
        or e3_decision.get("source_tree_sha256")
        != campaign_status.get("source_tree_sha256")
        or e3_reference.get("manifest_sha256")
        != file_sha256(campaign / "manifest.json")
        or e3_reference.get("campaign_contract_sha256")
        != campaign_contract.get("combined_sha256")
        or e3_reference.get("config_sha256")
        != campaign_contract.get("config_sha256")
    ):
        raise RuntimeError("formal E3 campaign/gate is incomplete, mismatched, or unlocked")
    if (
        campaign_status.get("subjects") != expected["subjects"]
        or campaign_status.get("seeds") != expected["seeds"]
        or campaign_status.get("folds") != expected["folds"]
        or campaign_status.get("variants")
        != [expected["matched_ann_variant"], expected["primary_variant"]]
        or e3_decision.get("primary_variant") != expected["primary_variant"]
        or e3_decision.get("matched_ann_variant") != expected["matched_ann_variant"]
    ):
        raise RuntimeError("formal E3 coverage differs from the final gate")

    rows = _csv_rows(e3_gate / "per_subject_seed.csv")
    primary = str(expected["primary_variant"])
    matched = str(expected["matched_ann_variant"])
    expected_variants = {primary, matched}
    keys = [
        (int(row["subject"]), int(row["seed"]), str(row["variant"]))
        for row in rows
    ]
    expected_pairs = {
        (int(subject), int(seed))
        for subject in expected["subjects"]
        for seed in expected["seeds"]
    }
    if len(expected_pairs) != int(expected["subject_seed_pairs"]):
        raise RuntimeError("registered final-gate pair count is inconsistent")
    expected_keys = {
        (subject, seed, variant)
        for subject, seed in expected_pairs
        for variant in expected_variants
    }
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise RuntimeError("E3 rows contain duplicate, missing, or extra identities")
    by_key = {key: row for key, row in zip(keys, rows, strict=True)}

    pair_rows: list[dict[str, Any]] = []
    firing_by_pair: dict[tuple[int, int], list[float]] = {
        pair: [] for pair in expected_pairs
    }
    for subject, seed in expected_pairs:
        for fold in expected["folds"]:
            metrics = read_json(
                campaign
                / f"subject_{subject:02d}"
                / f"seed_{seed}"
                / f"fold_{fold}"
                / primary
                / "metrics.json"
            )
            if (
                metrics.get("variant") != primary
                or int(metrics.get("subject", -1)) != subject
                or int(metrics.get("seed", -1)) != seed
                or int(metrics.get("fold", -1)) != int(fold)
                or metrics.get("session_e_accessed") is not False
            ):
                raise RuntimeError("E3 firing metric identity differs")
            firing_by_pair[(subject, seed)].append(
                _finite_float(metrics["full_firing_rate"], "E3 firing rate")
            )

    expected_trials = int(e3_decision["exact_oof_trials_per_pair"])
    if expected_trials != 288:
        raise RuntimeError("formal E3 exact OOF trial count changed")
    for subject, seed in sorted(expected_pairs):
        snn = by_key[(subject, seed, primary)]
        ann = by_key[(subject, seed, matched)]
        if int(snn["trials"]) != expected_trials or int(ann["trials"]) != expected_trials:
            raise RuntimeError("formal E3 pair trial count changed")
        anchor = _finite_float(snn["anchor_accuracy"], "SNN anchor accuracy")
        _close(anchor, ann["anchor_accuracy"], "SNN/ANN anchor accuracy")
        full = _finite_float(snn["full_accuracy"], "SNN full accuracy")
        zero = _finite_float(snn["matched_zero_accuracy"], "SNN zero accuracy")
        ann_full = _finite_float(ann["full_accuracy"], "ANN full accuracy")
        ann_zero = _finite_float(ann["matched_zero_accuracy"], "ANN zero accuracy")
        pair_rows.append(
            {
                "diagnostic_scope": "architecture_pair",
                "subject": subject,
                "seed": seed,
                "anchor_accuracy": anchor,
                "snn_full_accuracy": full,
                "snn_zero_accuracy": zero,
                "ann_full_accuracy": ann_full,
                "ann_zero_accuracy": ann_zero,
                "full_vs_anchor_pp": 100.0 * (full - anchor),
                "zero_vs_anchor_pp": 100.0 * (zero - anchor),
                "full_snn_vs_ann_pp": 100.0 * (full - ann_full),
                "zero_snn_vs_ann_pp": 100.0 * (zero - ann_zero),
                "mean_full_firing_rate": _mean(firing_by_pair[(subject, seed)]),
            }
        )

    full_rule = config["full_delay_branch"]
    zero_rule = config["zero_delay_branch"]
    full_anchor = [float(row["full_vs_anchor_pp"]) for row in pair_rows]
    zero_anchor = [float(row["zero_vs_anchor_pp"]) for row in pair_rows]
    full_ann = [float(row["full_snn_vs_ann_pp"]) for row in pair_rows]
    zero_ann = [float(row["zero_snn_vs_ann_pp"]) for row in pair_rows]
    firing_low, firing_high = [
        float(value) for value in full_rule["firing_rate_interval"]
    ]
    full_criteria = {
        "prior_feasibility": True,
        "registered_delay_gate": (
            not bool(full_rule["require_registered_delay_gate"])
            or bool(e3_decision.get("passed"))
        ),
        "mean_gain_vs_anchor": _mean(full_anchor)
        >= float(full_rule["minimum_mean_gain_vs_anchor_pp"]),
        "nonnegative_pairs_vs_anchor": sum(value >= 0.0 for value in full_anchor)
        >= int(full_rule["minimum_nonnegative_pairs_vs_anchor"]),
        "mean_snn_vs_ann": _mean(full_ann)
        >= float(full_rule["minimum_mean_snn_vs_ann_pp"]),
        "all_pairs_snn_vs_ann_floor": min(full_ann)
        >= float(full_rule["minimum_pair_snn_vs_ann_pp"]),
        "all_pairs_nondegenerate_firing": all(
            firing_low <= float(row["mean_full_firing_rate"]) <= firing_high
            for row in pair_rows
        ),
    }
    zero_criteria = {
        "matched_transport_prerequisite": True,
        "mean_gain_vs_anchor": _mean(zero_anchor)
        >= float(zero_rule["minimum_mean_gain_vs_anchor_pp"]),
        "nonnegative_pairs_vs_anchor": sum(value >= 0.0 for value in zero_anchor)
        >= int(zero_rule["minimum_nonnegative_pairs_vs_anchor"]),
        "mean_snn_vs_ann": _mean(zero_ann)
        >= float(zero_rule["minimum_mean_snn_vs_ann_pp"]),
        "all_pairs_snn_vs_ann_floor": min(zero_ann)
        >= float(zero_rule["minimum_pair_snn_vs_ann_pp"]),
    }
    full_passed = all(full_criteria.values())
    zero_passed = all(zero_criteria.values())
    selected = (
        "full_delay_routed_snn"
        if full_passed
        else (
            "matched_zero_routed_snn"
            if zero_passed
            else "confirmed_e4_sequence_residual"
        )
    )
    decision = {
        "status": "completed",
        "stage": "E3_FINAL_ARCHITECTURE_GATE",
        "protocol": PROTOCOL,
        "selected_branch": selected,
        "selection_reason": "predeclared_compound_session_t_gate",
        "full_delay_branch_passed": full_passed,
        "zero_delay_branch_passed": zero_passed,
        "full_delay_criteria": full_criteria,
        "zero_delay_criteria": zero_criteria,
        "summaries_pp": {
            "full_mean_vs_anchor": _mean(full_anchor),
            "zero_mean_vs_anchor": _mean(zero_anchor),
            "full_mean_snn_vs_ann": _mean(full_ann),
            "zero_mean_snn_vs_ann": _mean(zero_ann),
        },
        "e3_prior_feasibility_passed": True,
        "e3_delay_gate_evaluated": True,
        "e3_delay_gate_passed": bool(e3_decision.get("passed")),
        "e4_fallback_gate_passed": True,
        "post_session_e_selection": False,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    return pair_rows, decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feasibility-gate", required=True)
    parser.add_argument("--campaign")
    parser.add_argument("--e3-gate")
    parser.add_argument("--e4-confirmation-gate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config",
        default="configs/experiments/v8_e3_final_architecture_gate.yaml",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA:
        raise RuntimeError("final architecture gate schema changed")
    if config.get("preregistered_before_formal_e3_results") is not True:
        raise RuntimeError("final architecture branch priority was not preregistered")
    if config.get("data_access") != {
        "allowed_session": "T",
        "heldout_session_e_accessed": False,
        "openbmi_session_s2_accessed": False,
    }:
        raise RuntimeError("final architecture gate held-out lock changed")
    feasibility_gate = Path(args.feasibility_gate).resolve()
    campaign = Path(args.campaign).resolve() if args.campaign else None
    e3_gate = Path(args.e3_gate).resolve() if args.e3_gate else None
    e4_gate_path = Path(args.e4_confirmation_gate).resolve()
    pair_rows, decision = evaluate_final_gate(
        feasibility_gate=feasibility_gate,
        campaign=campaign,
        e3_gate=e3_gate,
        e4_gate_path=e4_gate_path,
        config=config,
    )
    source_tree = collect_source_tree_manifest(ROOT)
    output = ensure_dir(Path(args.output).resolve())
    write_csv(output / "pair_diagnostics.csv", pair_rows)
    write_json(output / "decision.json", decision)
    write_json(output / "source_tree_manifest.json", source_tree)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    input_manifest = {
        "feasibility_gate_manifest_sha256": file_sha256(
            feasibility_gate / "manifest.json"
        ),
        "e4_confirmation_gate_sha256": file_sha256(e4_gate_path),
        "config_sha256": file_sha256(config_path),
        "evaluator_source_tree_sha256": source_tree_digest(source_tree),
    }
    if campaign is not None:
        input_manifest["campaign_manifest_sha256"] = file_sha256(
            campaign / "manifest.json"
        )
    if e3_gate is not None:
        input_manifest["e3_gate_manifest_sha256"] = file_sha256(
            e3_gate / "manifest.json"
        )
    write_json(output / "input_manifest.json", input_manifest)
    write_run_artifact_manifest(output, required_files=OUTPUT_FILES)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
