#!/usr/bin/env python3
"""Aggregate fold-local ATC sequence decoder canaries without weight search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.models.v8_sequence_decoder import V8_SEQUENCE_DECODER_VARIANTS  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402


MATCHED_ANN_CONTROL = {
    "plif_plain": "ann_plain",
    "clif_plain": "ann_plain",
    "sew_clif": "ann_sew",
}


def assemble_oof(
    parts: Sequence[Mapping[str, np.ndarray]],
    *,
    n_trials: int,
    n_classes: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    logits = np.full((int(n_trials), int(n_classes)), np.nan, dtype=np.float32)
    labels = np.full(int(n_trials), -1, dtype=np.int64)
    seen = np.zeros(int(n_trials), dtype=bool)
    for part in parts:
        indices = np.asarray(part["indices"], dtype=np.int64)
        part_logits = np.asarray(part["logits"], dtype=np.float32)
        part_labels = np.asarray(part["labels"], dtype=np.int64)
        if indices.ndim != 1 or part_logits.shape != (indices.size, n_classes):
            raise RuntimeError("fold prediction part has an invalid shape")
        if part_labels.shape != indices.shape or np.any(indices < 0) or np.any(indices >= n_trials):
            raise RuntimeError("fold prediction indices or labels are invalid")
        if seen[indices].any():
            raise RuntimeError("fold predictions overlap")
        logits[indices] = part_logits
        labels[indices] = part_labels
        seen[indices] = True
    if not seen.all() or not np.isfinite(logits).all() or np.any(labels < 0):
        raise RuntimeError("fold predictions do not form complete finite OOF coverage")
    return logits, labels


def _load_fold_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    if set(values) != {"indices", "logits", "labels"}:
        raise RuntimeError(f"invalid decoder fold archive: {path}")
    return values


def _load_e1(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    required = {"probabilities", "label", "trial_id", "run", "session"}
    if not required <= values.keys() or values["probabilities"].shape != (288, 4):
        raise RuntimeError(f"invalid E1 OOF archive: {path}")
    if set(values["session"].astype(str).tolist()) != {"T"}:
        raise RuntimeError("E4 aggregation received non-Session-T predictions")
    return values


def _accuracy(labels: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.asarray(labels) == np.asarray(prediction)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary-root", required=True)
    parser.add_argument("--e1-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", default="0,1,2,3,4,5")
    parser.add_argument("--variants", default=",".join(V8_SEQUENCE_DECODER_VARIANTS))
    args = parser.parse_args()

    canary_root = Path(args.canary_root).resolve()
    e1_root = Path(args.e1_root).resolve()
    output = ensure_dir(Path(args.output).resolve())
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    if folds != list(range(6)):
        raise RuntimeError("formal S1 canary aggregation requires all six outer folds")
    if any(variant not in V8_SEQUENCE_DECODER_VARIANTS for variant in variants):
        raise ValueError("unknown decoder variant")

    atc = _load_e1(
        e1_root
        / "atcnet"
        / f"subject_{args.subject:02d}"
        / f"seed_{args.seed}"
        / "predictions.npz"
    )
    fbc = _load_e1(
        e1_root
        / "fbcnet"
        / f"subject_{args.subject:02d}"
        / f"seed_{args.seed}"
        / "predictions.npz"
    )
    for key in ("label", "trial_id", "run", "session"):
        if not np.array_equal(atc[key], fbc[key]):
            raise RuntimeError(f"ATC and FBC OOF archives are not aligned on {key}")
    labels = np.asarray(atc["label"], dtype=np.int64)
    anchor_probability = 0.5 * atc["probabilities"] + 0.5 * fbc["probabilities"]
    anchor_prediction = anchor_probability.argmax(axis=1)
    anchor_accuracy = _accuracy(labels, anchor_prediction)

    fold_rows: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []
    variant_predictions: dict[str, np.ndarray] = {}
    for variant in variants:
        parts: list[dict[str, np.ndarray]] = []
        firing_rates: list[float] = []
        for fold in folds:
            fold_root = canary_root / (
                f"formal_s{args.subject}_seed{args.seed}_fold{fold}"
            )
            status = read_json(fold_root / "campaign_status.json")
            if (
                status.get("status") != "completed"
                or status.get("session_e_accessed") is not False
                or int(status.get("fold", -1)) != fold
            ):
                raise RuntimeError(f"incomplete or invalid canary fold: {fold_root}")
            row = read_json(fold_root / variant / "metrics.json")
            prediction = _load_fold_prediction(fold_root / variant / "outer_predictions.npz")
            parts.append(prediction)
            firing_rates.append(float(row["decoder_mean_firing_rate"]))
            fold_rows.append(row)
        logits, decoder_labels = assemble_oof(parts, n_trials=288)
        if not np.array_equal(decoder_labels, labels):
            raise RuntimeError(f"decoder OOF labels differ from E1 for {variant}")
        decoder_probability = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        decoder_prediction = decoder_probability.argmax(axis=1)
        fused_probability = 0.5 * decoder_probability + 0.5 * fbc["probabilities"]
        fused_prediction = fused_probability.argmax(axis=1)
        three_way_probability = (
            atc["probabilities"] + fbc["probabilities"] + decoder_probability
        ) / 3.0
        three_way_prediction = three_way_probability.argmax(axis=1)
        decoder_metrics = classification_metrics(labels, decoder_prediction, n_classes=4)
        fused_metrics = classification_metrics(labels, fused_prediction, n_classes=4)
        three_way_metrics = classification_metrics(labels, three_way_prediction, n_classes=4)
        anchor_correct = anchor_prediction == labels
        fused_correct = fused_prediction == labels
        three_way_correct = three_way_prediction == labels
        variant_predictions[f"{variant}_decoder_plus_fbc"] = fused_prediction
        variant_predictions[f"{variant}_three_way"] = three_way_prediction
        aggregate.append(
            {
                "variant": variant,
                "subject": int(args.subject),
                "seed": int(args.seed),
                "folds": len(folds),
                "decoder_accuracy": decoder_metrics["accuracy"],
                "decoder_kappa": decoder_metrics["kappa"],
                "decoder_plus_fbc_accuracy": fused_metrics["accuracy"],
                "decoder_plus_fbc_kappa": fused_metrics["kappa"],
                "three_way_equal_accuracy": three_way_metrics["accuracy"],
                "three_way_equal_kappa": three_way_metrics["kappa"],
                "anchor_atc_plus_fbc_accuracy": anchor_accuracy,
                "fused_delta_vs_anchor_pp": 100.0
                * (float(fused_metrics["accuracy"]) - anchor_accuracy),
                "fused_corrections_of_anchor_errors": int(np.sum(~anchor_correct & fused_correct)),
                "fused_regressions_of_anchor_correct": int(np.sum(anchor_correct & ~fused_correct)),
                "three_way_delta_vs_anchor_pp": 100.0
                * (float(three_way_metrics["accuracy"]) - anchor_accuracy),
                "three_way_corrections_of_anchor_errors": int(
                    np.sum(~anchor_correct & three_way_correct)
                ),
                "three_way_regressions_of_anchor_correct": int(
                    np.sum(anchor_correct & ~three_way_correct)
                ),
                "mean_firing_rate": float(np.mean(firing_rates)),
                "session_e_accessed": False,
            }
        )

    eligible: list[str] = []
    for row in aggregate:
        variant = str(row["variant"])
        if variant not in MATCHED_ANN_CONTROL:
            continue
        matched_ann_name = MATCHED_ANN_CONTROL[variant]
        matched_ann = next(
            candidate for candidate in aggregate if candidate["variant"] == matched_ann_name
        )
        snn_gap_pp = 100.0 * (
            float(row["three_way_equal_accuracy"])
            - float(matched_ann["three_way_equal_accuracy"])
        )
        row["matched_ann_control"] = matched_ann_name
        row["snn_delta_vs_matched_ann_pp"] = snn_gap_pp
        row["within_0_3pp_of_matched_ann"] = snn_gap_pp >= -0.3
        row["within_0_5pp_of_accuracy_anchor"] = (
            float(row["three_way_delta_vs_anchor_pp"]) >= -0.5
        )
        row["nondegenerate_firing"] = 0.005 <= float(row["mean_firing_rate"]) <= 0.30
        if (
            row["within_0_3pp_of_matched_ann"]
            and row["within_0_5pp_of_accuracy_anchor"]
            and row["nondegenerate_firing"]
        ):
            eligible.append(str(row["variant"]))
    gate = {
        "passed": bool(eligible),
        "eligible_snn_variants": eligible,
        "scope": "S1 development canary only; fixed three-way rule requires untouched-subject confirmation",
        "fusion_rule": "equal probability mean of ATCNet, FBCNet, and the decoder",
        "criteria": {
            "snn_vs_matched_ann_minimum_delta_pp": -0.3,
            "snn_three_way_vs_accuracy_anchor_minimum_delta_pp": -0.5,
            "mean_firing_rate_interval": [0.005, 0.30],
        },
        "anchor_accuracy": anchor_accuracy,
        "session_e_accessed": False,
    }
    write_csv(output / "per_fold.csv", fold_rows)
    write_csv(output / "aggregate.csv", aggregate)
    write_json(output / "gate.json", gate)
    np.savez_compressed(
        output / "fused_predictions.npz",
        labels=labels,
        anchor=anchor_prediction,
        **variant_predictions,
    )
    print(json.dumps({"status": "completed", "aggregate": aggregate, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
