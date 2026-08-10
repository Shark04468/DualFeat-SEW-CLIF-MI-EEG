#!/usr/bin/env python3
"""Evaluate the locked entropy-gated residual decoder on Session-T OOF data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v8_fusion import (  # noqa: E402
    entropy_residual_probability,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from scripts.analyze_v8_e4_atc_sequence_canaries import (  # noqa: E402
    _load_e1,
    _load_fold_prediction,
    assemble_oof,
)


LOCK_SCHEMA = "dpc-snn-v8-e4-entropy-residual-lock/v1"


def confirmation_gate(
    lock: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    selected = str(lock["selection"]["selected_snn_variant"])
    matched = str(lock["selection"]["matched_ann_control"])
    confirmation = lock["confirmation"]
    expected_pairs = {
        (int(subject), int(seed))
        for subject in confirmation["subjects"]
        for seed in confirmation["seeds"]
    }
    by_key = {
        (int(row["subject"]), int(row["seed"]), str(row["variant"])): row
        for row in rows
    }
    if len(by_key) != len(rows):
        raise RuntimeError("duplicate entropy residual aggregate rows")

    pair_rows: list[dict[str, Any]] = []
    for subject, seed in sorted(expected_pairs):
        snn_key = (subject, seed, selected)
        ann_key = (subject, seed, matched)
        if snn_key not in by_key or ann_key not in by_key:
            raise RuntimeError(f"missing confirmation result for S{subject}/seed{seed}")
        snn = by_key[snn_key]
        ann = by_key[ann_key]
        snn_accuracy = float(snn["final_accuracy"])
        ann_accuracy = float(ann["final_accuracy"])
        anchor_accuracy = float(snn["anchor_accuracy"])
        if abs(anchor_accuracy - float(ann["anchor_accuracy"])) > 1e-12:
            raise RuntimeError("matched ANN and SNN rows use different anchors")
        delta_anchor = 100.0 * (snn_accuracy - anchor_accuracy)
        delta_ann = 100.0 * (snn_accuracy - ann_accuracy)
        firing_rate = float(snn["mean_firing_rate"])
        firing_low, firing_high = [float(value) for value in confirmation["firing_rate_interval"]]
        pair_rows.append(
            {
                "subject": subject,
                "seed": seed,
                "snn_accuracy": snn_accuracy,
                "matched_ann_accuracy": ann_accuracy,
                "anchor_accuracy": anchor_accuracy,
                "snn_delta_vs_anchor_pp": delta_anchor,
                "snn_delta_vs_matched_ann_pp": delta_ann,
                "snn_mean_firing_rate": firing_rate,
                "positive_vs_anchor": delta_anchor > 0.0,
                "passes_anchor_floor": delta_anchor
                >= float(confirmation["per_pair_snn_vs_anchor_floor_pp"]),
                "passes_firing_interval": firing_low <= firing_rate <= firing_high,
            }
        )

    macro_snn = float(np.mean([row["snn_accuracy"] for row in pair_rows]))
    macro_ann = float(np.mean([row["matched_ann_accuracy"] for row in pair_rows]))
    macro_anchor = float(np.mean([row["anchor_accuracy"] for row in pair_rows]))
    macro_ann_delta = 100.0 * (macro_snn - macro_ann)
    macro_anchor_delta = 100.0 * (macro_snn - macro_anchor)
    positive_pairs = sum(bool(row["positive_vs_anchor"]) for row in pair_rows)
    criteria = {
        "macro_snn_vs_matched_ann": macro_ann_delta
        >= float(confirmation["macro_snn_vs_matched_ann_minimum_delta_pp"]),
        "macro_snn_vs_anchor": macro_anchor_delta
        >= float(confirmation["macro_snn_vs_anchor_minimum_delta_pp"]),
        "minimum_positive_pairs": positive_pairs
        >= int(confirmation["minimum_positive_subject_seed_pairs"]),
        "all_pairs_above_anchor_floor": all(
            bool(row["passes_anchor_floor"]) for row in pair_rows
        ),
        "all_pairs_have_nondegenerate_firing": all(
            bool(row["passes_firing_interval"]) for row in pair_rows
        ),
    }
    return {
        "schema": "dpc-snn-v8-e4-entropy-residual-result/v1",
        "passed": all(criteria.values()),
        "selected_snn_variant": selected,
        "matched_ann_control": matched,
        "confirmation_pairs": len(pair_rows),
        "positive_subject_seed_pairs": positive_pairs,
        "macro_snn_accuracy": macro_snn,
        "macro_matched_ann_accuracy": macro_ann,
        "macro_anchor_accuracy": macro_anchor,
        "macro_snn_delta_vs_matched_ann_pp": macro_ann_delta,
        "macro_snn_delta_vs_anchor_pp": macro_anchor_delta,
        "criteria": criteria,
        "thresholds": dict(confirmation),
        "pair_rows": pair_rows,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--canary-root", required=True)
    parser.add_argument("--atc-root", required=True)
    parser.add_argument("--fbc-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scope", choices=("development", "confirmation"), required=True)
    args = parser.parse_args()

    lock_path = Path(args.lock).resolve()
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != LOCK_SCHEMA:
        raise RuntimeError("entropy residual lock schema is invalid")
    if lock["data_access"].get("heldout_session_e_accessed") is not False:
        raise RuntimeError("Session E must remain locked")
    selection = lock["selection"]
    if selection.get("mode") != "anchor_entropy":
        raise RuntimeError("only the locked anchor-entropy rule is permitted")
    if selection.get("temperature_calibration") is not False:
        raise RuntimeError("temperature calibration is not permitted")
    if selection.get("learned_fusion_weights") is not False:
        raise RuntimeError("learned fusion weights are not permitted")

    scope = lock[args.scope]
    subjects = [int(value) for value in scope["subjects"]]
    seeds = [int(value) for value in scope["seeds"]]
    variants = [
        str(selection["matched_ann_control"]),
        str(selection["selected_snn_variant"]),
    ]
    canary_root = Path(args.canary_root).resolve()
    atc_root = Path(args.atc_root).resolve()
    fbc_root = Path(args.fbc_root).resolve()
    output = ensure_dir(Path(args.output).resolve())
    maximum_weight = float(selection["maximum_decoder_weight"])

    rows: list[dict[str, Any]] = []
    prediction_payload: dict[str, np.ndarray] = {}
    for subject in subjects:
        for seed in seeds:
            atc = _load_e1(
                atc_root / "atcnet" / f"subject_{subject:02d}" / f"seed_{seed}" / "predictions.npz"
            )
            fbc = _load_e1(
                fbc_root / "fbcnet" / f"subject_{subject:02d}" / f"seed_{seed}" / "predictions.npz"
            )
            for key in ("label", "trial_id", "run", "session"):
                if not np.array_equal(atc[key], fbc[key]):
                    raise RuntimeError(f"ATC/FBC predictions differ on {key}")
            labels = np.asarray(atc["label"], dtype=np.int64)
            anchor = 0.5 * atc["probabilities"] + 0.5 * fbc["probabilities"]
            anchor_prediction = anchor.argmax(axis=1)
            anchor_metrics = classification_metrics(labels, anchor_prediction, n_classes=4)
            prediction_payload[f"s{subject}_seed{seed}_labels"] = labels
            prediction_payload[f"s{subject}_seed{seed}_anchor"] = anchor_prediction

            for variant in variants:
                parts: list[dict[str, np.ndarray]] = []
                firing_rates: list[float] = []
                for fold in range(6):
                    fold_root = canary_root / f"formal_s{subject}_seed{seed}_fold{fold}"
                    status = read_json(fold_root / "campaign_status.json")
                    if status.get("status") != "completed" or status.get("session_e_accessed") is not False:
                        raise RuntimeError(f"invalid decoder fold: {fold_root}")
                    parts.append(_load_fold_prediction(fold_root / variant / "outer_predictions.npz"))
                    metric = read_json(fold_root / variant / "metrics.json")
                    firing_rates.append(float(metric["decoder_mean_firing_rate"]))
                logits, decoder_labels = assemble_oof(parts, n_trials=288)
                if not np.array_equal(decoder_labels, labels):
                    raise RuntimeError("decoder labels do not match the anchor")
                decoder_probability = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
                final_probability, gate = entropy_residual_probability(
                    anchor, decoder_probability, maximum_weight=maximum_weight
                )
                final_prediction = final_probability.argmax(axis=1)
                final_metrics = classification_metrics(labels, final_prediction, n_classes=4)
                anchor_correct = anchor_prediction == labels
                final_correct = final_prediction == labels
                rows.append(
                    {
                        "subject": subject,
                        "seed": seed,
                        "variant": variant,
                        "folds": 6,
                        "anchor_accuracy": float(anchor_metrics["accuracy"]),
                        "final_accuracy": float(final_metrics["accuracy"]),
                        "final_kappa": float(final_metrics["kappa"]),
                        "delta_vs_anchor_pp": 100.0
                        * (float(final_metrics["accuracy"]) - float(anchor_metrics["accuracy"])),
                        "corrections_of_anchor_errors": int(np.sum(~anchor_correct & final_correct)),
                        "regressions_of_anchor_correct": int(np.sum(anchor_correct & ~final_correct)),
                        "mean_gate": float(np.mean(gate)),
                        "maximum_gate": float(np.max(gate)),
                        "mean_firing_rate": float(np.mean(firing_rates)),
                        "session_e_accessed": False,
                    }
                )
                prediction_payload[f"s{subject}_seed{seed}_{variant}"] = final_prediction

    gate = (
        confirmation_gate(lock, rows)
        if args.scope == "confirmation"
        else {
            "schema": "dpc-snn-v8-e4-entropy-residual-development-result/v1",
            "passed": None,
            "note": "development metrics only; confirmation thresholds are not applied",
            "session_e_accessed": False,
        }
    )
    write_csv(output / "aggregate.csv", rows)
    write_json(output / "gate.json", gate)
    (output / "lock.yaml").write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")
    np.savez_compressed(output / "predictions.npz", **prediction_payload)
    print(json.dumps({"status": "completed", "rows": rows, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
