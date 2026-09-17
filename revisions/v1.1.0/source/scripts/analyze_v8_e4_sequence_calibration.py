#!/usr/bin/env python3
"""Inner-validation temperature calibration for ATC/FBC/SNN fold logits."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v8_sequence_decoder_training import (  # noqa: E402
    predict_v8_sequence_decoder,
)
from dpc_snn.models.v8_sequence_decoder import (  # noqa: E402
    V8_SEQUENCE_DECODER_VARIANTS,
    build_v8_sequence_decoder,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from scripts.analyze_v8_e4_atc_sequence_canaries import (  # noqa: E402
    MATCHED_ANN_CONTROL,
    _load_e1,
    _load_fold_prediction,
)


TEMPERATURE_GRID = (0.25, 0.35, 0.50, 0.70, 1.0, 1.4, 2.0, 2.8, 4.0)


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    values = torch.as_tensor(logits, dtype=torch.float64) / float(temperature)
    targets = torch.as_tensor(labels, dtype=torch.long)
    if values.ndim != 2 or targets.shape != (values.shape[0],):
        raise ValueError("temperature calibration logits and labels are not aligned")
    return float(F.cross_entropy(values, targets))


def select_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    grid: Sequence[float] = TEMPERATURE_GRID,
) -> tuple[float, list[dict[str, float]]]:
    rows = [
        {
            "temperature": float(value),
            "negative_log_likelihood": negative_log_likelihood(logits, labels, float(value)),
        }
        for value in grid
    ]
    selected = min(
        rows,
        key=lambda row: (
            row["negative_log_likelihood"],
            abs(math.log(row["temperature"])),
            row["temperature"],
        ),
    )
    return float(selected["temperature"]), rows


def calibrated_probability(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = torch.as_tensor(logits, dtype=torch.float32) / float(temperature)
    return torch.softmax(values, dim=1).numpy()


def _aligned_fold(
    path: Path,
    expected_indices: np.ndarray,
    expected_labels: np.ndarray,
) -> np.ndarray:
    prediction = _load_fold_prediction(path)
    if not np.array_equal(prediction["indices"], expected_indices) or not np.array_equal(
        prediction["labels"], expected_labels
    ):
        raise RuntimeError(f"calibration fold archive is not aligned: {path}")
    return np.asarray(prediction["logits"], dtype=np.float32)


def _fill(
    destination: np.ndarray,
    seen: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
) -> None:
    if seen[indices].any() or values.shape != (indices.size, destination.shape[1]):
        raise RuntimeError("calibrated OOF parts overlap or have an invalid shape")
    destination[indices] = values
    seen[indices] = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary-root", required=True)
    parser.add_argument("--e1-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--variants", default="ann_plain,ann_sew,plif_plain,clif_plain,sew_clif"
    )
    args = parser.parse_args()

    canary_root = Path(args.canary_root).resolve()
    e1_root = Path(args.e1_root).resolve()
    output = ensure_dir(Path(args.output).resolve())
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    if any(value not in V8_SEQUENCE_DECODER_VARIANTS for value in variants):
        raise ValueError("unknown sequence decoder variant")
    required_controls = {
        MATCHED_ANN_CONTROL[variant]
        for variant in variants
        if variant in MATCHED_ANN_CONTROL
    }
    if not required_controls <= set(variants):
        raise RuntimeError("calibration comparison is missing a topology-matched ANN control")

    atc_run = e1_root / "atcnet" / f"subject_{args.subject:02d}" / f"seed_{args.seed}"
    fbc_run = e1_root / "fbcnet" / f"subject_{args.subject:02d}" / f"seed_{args.seed}"
    atc = _load_e1(atc_run / "predictions.npz")
    fbc = _load_e1(fbc_run / "predictions.npz")
    for key in ("label", "trial_id", "run", "session"):
        if not np.array_equal(atc[key], fbc[key]):
            raise RuntimeError(f"ATC/FBC OOF mismatch on {key}")
    labels = np.asarray(atc["label"], dtype=np.int64)
    n_trials = labels.size
    atc_calibrated = np.full((n_trials, 4), np.nan, dtype=np.float32)
    fbc_calibrated = np.full((n_trials, 4), np.nan, dtype=np.float32)
    atc_seen = np.zeros(n_trials, dtype=bool)
    fbc_seen = np.zeros(n_trials, dtype=bool)
    decoder_calibrated = {
        variant: np.full((n_trials, 4), np.nan, dtype=np.float32) for variant in variants
    }
    decoder_seen = {variant: np.zeros(n_trials, dtype=bool) for variant in variants}
    calibration_rows: list[dict[str, Any]] = []

    for fold in range(6):
        fold_root = canary_root / f"formal_s{args.subject}_seed{args.seed}_fold{fold}"
        status = read_json(fold_root / "campaign_status.json")
        if status.get("status") != "completed" or status.get("session_e_accessed") is not False:
            raise RuntimeError(f"incomplete canary fold: {fold_root}")
        with np.load(fold_root / "frozen_sequence_cache.npz", allow_pickle=False) as cache:
            validation_indices = np.asarray(cache["inner_validation_indices"], dtype=np.int64)
            validation_sequence = np.asarray(
                cache["selection_validation_sequence"], dtype=np.float32
            )
            validation_teacher = np.asarray(
                cache["selection_validation_teacher"], dtype=np.float32
            )
            outer_indices = np.asarray(cache["outer_test_indices"], dtype=np.int64)
        validation_labels = labels[validation_indices]
        outer_labels = labels[outer_indices]

        atc_validation_logits = _aligned_fold(
            atc_run / f"fold_{fold}" / "selection_predictions.npz",
            validation_indices,
            validation_labels,
        )
        atc_outer_logits = _aligned_fold(
            atc_run / f"fold_{fold}" / "outer_test_predictions.npz",
            outer_indices,
            outer_labels,
        )
        fbc_validation_logits = _aligned_fold(
            fbc_run / f"fold_{fold}" / "selection_predictions.npz",
            validation_indices,
            validation_labels,
        )
        fbc_outer_logits = _aligned_fold(
            fbc_run / f"fold_{fold}" / "outer_test_predictions.npz",
            outer_indices,
            outer_labels,
        )
        atc_temperature, atc_grid = select_temperature(
            atc_validation_logits, validation_labels
        )
        fbc_temperature, fbc_grid = select_temperature(
            fbc_validation_logits, validation_labels
        )
        _fill(
            atc_calibrated,
            atc_seen,
            outer_indices,
            calibrated_probability(atc_outer_logits, atc_temperature),
        )
        _fill(
            fbc_calibrated,
            fbc_seen,
            outer_indices,
            calibrated_probability(fbc_outer_logits, fbc_temperature),
        )
        calibration_rows.extend(
            [
                {
                    "fold": fold,
                    "branch": "atcnet",
                    "temperature": atc_temperature,
                    "grid": json.dumps(atc_grid, separators=(",", ":")),
                },
                {
                    "fold": fold,
                    "branch": "fbcnet",
                    "temperature": fbc_temperature,
                    "grid": json.dumps(fbc_grid, separators=(",", ":")),
                },
            ]
        )

        for variant in variants:
            model = build_v8_sequence_decoder(variant)
            state = torch.load(
                fold_root / variant / "selection_best.pt",
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state, strict=True)
            validation = predict_v8_sequence_decoder(
                model,
                validation_sequence,
                validation_labels,
                validation_teacher,
                device="cpu",
            )
            decoder_temperature, decoder_grid = select_temperature(
                validation["logits"], validation_labels
            )
            outer = _load_fold_prediction(fold_root / variant / "outer_predictions.npz")
            if not np.array_equal(outer["indices"], outer_indices) or not np.array_equal(
                outer["labels"], outer_labels
            ):
                raise RuntimeError(f"decoder outer fold is not aligned for {variant}")
            _fill(
                decoder_calibrated[variant],
                decoder_seen[variant],
                outer_indices,
                calibrated_probability(outer["logits"], decoder_temperature),
            )
            calibration_rows.append(
                {
                    "fold": fold,
                    "branch": variant,
                    "temperature": decoder_temperature,
                    "grid": json.dumps(decoder_grid, separators=(",", ":")),
                }
            )

    if not atc_seen.all() or not fbc_seen.all() or not all(seen.all() for seen in decoder_seen.values()):
        raise RuntimeError("temperature calibration did not produce complete OOF coverage")
    original_anchor = 0.5 * atc["probabilities"] + 0.5 * fbc["probabilities"]
    calibrated_anchor = 0.5 * atc_calibrated + 0.5 * fbc_calibrated
    original_anchor_accuracy = float(np.mean(original_anchor.argmax(1) == labels))
    calibrated_anchor_accuracy = float(np.mean(calibrated_anchor.argmax(1) == labels))
    rows: list[dict[str, Any]] = []
    fused_predictions: dict[str, np.ndarray] = {}
    for variant in variants:
        fused = 0.5 * decoder_calibrated[variant] + 0.5 * fbc_calibrated
        prediction = fused.argmax(axis=1)
        metrics = classification_metrics(labels, prediction, n_classes=4)
        fused_predictions[variant] = prediction
        rows.append(
            {
                "variant": variant,
                "calibrated_decoder_plus_fbc_accuracy": metrics["accuracy"],
                "calibrated_decoder_plus_fbc_kappa": metrics["kappa"],
                "original_anchor_accuracy": original_anchor_accuracy,
                "calibrated_anchor_accuracy": calibrated_anchor_accuracy,
                "delta_vs_original_anchor_pp": 100.0
                * (float(metrics["accuracy"]) - original_anchor_accuracy),
                "delta_vs_calibrated_anchor_pp": 100.0
                * (float(metrics["accuracy"]) - calibrated_anchor_accuracy),
                "session_e_accessed": False,
            }
        )
    eligible: list[str] = []
    for row in rows:
        variant = str(row["variant"])
        if variant not in MATCHED_ANN_CONTROL:
            continue
        matched_ann_name = MATCHED_ANN_CONTROL[variant]
        ann = next(candidate for candidate in rows if candidate["variant"] == matched_ann_name)
        row["matched_ann_control"] = matched_ann_name
        row["delta_vs_matched_ann_pp"] = 100.0 * (
            float(row["calibrated_decoder_plus_fbc_accuracy"])
            - float(ann["calibrated_decoder_plus_fbc_accuracy"])
        )
        if (
            float(row["delta_vs_matched_ann_pp"]) >= -0.3
            and float(row["delta_vs_calibrated_anchor_pp"]) >= -0.5
        ):
            eligible.append(str(row["variant"]))
    gate = {
        "passed": bool(eligible),
        "eligible_snn_variants": eligible,
        "scope": "post-canary S1 development diagnosis; confirmation required on untouched subjects",
        "temperature_selection": "per fold, per branch, inner-validation NLL only",
        "temperature_grid": list(TEMPERATURE_GRID),
        "fusion_weight": 0.5,
        "criteria": {
            "snn_vs_matched_ann_minimum_delta_pp": -0.3,
            "snn_vs_calibrated_anchor_minimum_delta_pp": -0.5,
        },
        "session_e_accessed": False,
    }
    write_csv(output / "temperatures.csv", calibration_rows)
    write_csv(output / "aggregate.csv", rows)
    write_json(output / "gate.json", gate)
    np.savez_compressed(
        output / "calibrated_predictions.npz",
        labels=labels,
        original_anchor=original_anchor.argmax(1),
        calibrated_anchor=calibrated_anchor.argmax(1),
        **fused_predictions,
    )
    print(json.dumps({"status": "completed", "rows": rows, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
