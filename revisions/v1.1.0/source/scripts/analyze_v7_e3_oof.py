#!/usr/bin/env python3
"""Post-hoc diagnostic of paired V7 E3 out-of-fold delay logits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.utils.io import ensure_dir, write_csv, write_json  # noqa: E402


def _load_seed(seed_dir: Path) -> dict[str, np.ndarray]:
    parts = []
    for fold_dir in sorted(seed_dir.glob("fold_*")):
        path = fold_dir / "paired_validation_predictions.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as archive:
            parts.append({key: archive[key] for key in archive.files})
    if not parts:
        raise FileNotFoundError(f"no paired fold predictions under {seed_dir}")
    indices = np.concatenate([part["indices"] for part in parts])
    order = np.argsort(indices)
    indices = indices[order]
    if len(indices) != len(np.unique(indices)):
        raise RuntimeError("OOF diagnostic found duplicated trial indices")
    return {
        "indices": indices,
        "labels": np.concatenate([part["labels"] for part in parts])[order],
        "full_logits": np.concatenate([part["full_logits"] for part in parts])[order],
        "zero_logits": np.concatenate([part["locked_zero_logits"] for part in parts])[order],
    }


def _true_margin(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    true = logits[np.arange(len(labels)), labels]
    masked = logits.copy()
    masked[np.arange(len(labels)), labels] = -np.inf
    return true - masked.max(axis=1)


def _gain_row(
    zero: np.ndarray,
    delta: np.ndarray,
    labels: np.ndarray,
    alpha: float,
) -> dict[str, float | int]:
    prediction = (zero + float(alpha) * delta).argmax(axis=1)
    zero_prediction = zero.argmax(axis=1)
    return {
        "alpha": float(alpha),
        "accuracy": float(np.mean(prediction == labels)),
        "corrected": int(np.count_nonzero((prediction == labels) & (zero_prediction != labels))),
        "regressed": int(np.count_nonzero((prediction != labels) & (zero_prediction == labels))),
        "prediction_changes": int(np.count_nonzero(prediction != zero_prediction)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    output = ensure_dir(Path(args.output).resolve())
    seed_payloads: dict[str, dict[str, np.ndarray]] = {}
    summary_rows: list[dict[str, Any]] = []
    gain_rows: list[dict[str, Any]] = []
    alpha_grid = np.linspace(-2.0, 8.0, 201)
    registered_alphas = {0.0, 0.5, 1.0, 2.0, 4.0, 8.0}

    for subject_dir in sorted(input_root.glob("subject_*")):
        for seed_dir in sorted(subject_dir.glob("seed_*")):
            payload = _load_seed(seed_dir)
            key = f"{subject_dir.name}/{seed_dir.name}"
            seed_payloads[key] = payload
            labels = payload["labels"].astype(np.int64)
            zero = payload["zero_logits"].astype(np.float64)
            delta = payload["full_logits"].astype(np.float64) - zero
            rows = [_gain_row(zero, delta, labels, float(alpha)) for alpha in alpha_grid]
            best = max(rows, key=lambda row: (row["accuracy"], -abs(row["alpha"] - 1.0)))
            for row in rows:
                if any(abs(float(row["alpha"]) - value) < 1e-8 for value in registered_alphas):
                    gain_rows.append({"run": key, **row})

            zero_prediction = zero.argmax(axis=1)
            zero_correct = zero_prediction == labels
            zero_margin = _true_margin(zero, labels)
            full_margin = _true_margin(zero + delta, labels)
            margin_delta = full_margin - zero_margin
            summary_rows.append(
                {
                    "run": key,
                    "trials": int(len(labels)),
                    "zero_accuracy": float(np.mean(zero_correct)),
                    "full_accuracy": float(np.mean((zero + delta).argmax(axis=1) == labels)),
                    "registered_delta_pp": float(
                        100.0
                        * (
                            np.mean((zero + delta).argmax(axis=1) == labels)
                            - np.mean(zero_correct)
                        )
                    ),
                    "delay_logit_rms": float(np.sqrt(np.mean(delta**2))),
                    "delay_logit_mae": float(np.mean(np.abs(delta))),
                    "mean_true_margin_delta_zero_correct": float(margin_delta[zero_correct].mean()),
                    "mean_true_margin_delta_zero_incorrect": float(
                        margin_delta[~zero_correct].mean()
                    ),
                    "fraction_positive_margin_delta_zero_incorrect": float(
                        np.mean(margin_delta[~zero_correct] > 0.0)
                    ),
                    "posthoc_best_alpha": float(best["alpha"]),
                    "posthoc_best_accuracy": float(best["accuracy"]),
                    "posthoc_best_delta_pp": float(
                        100.0 * (float(best["accuracy"]) - np.mean(zero_correct))
                    ),
                    "posthoc_best_corrected": int(best["corrected"]),
                    "posthoc_best_regressed": int(best["regressed"]),
                }
            )

    consistency_rows = []
    keys = sorted(seed_payloads)
    for first_index, first_key in enumerate(keys):
        for second_key in keys[first_index + 1 :]:
            first = seed_payloads[first_key]
            second = seed_payloads[second_key]
            if not np.array_equal(first["indices"], second["indices"]):
                raise RuntimeError("seed OOF trial order does not match")
            first_delta = (first["full_logits"] - first["zero_logits"]).ravel()
            second_delta = (second["full_logits"] - second["zero_logits"]).ravel()
            consistency_rows.append(
                {
                    "first": first_key,
                    "second": second_key,
                    "delay_logit_correlation": float(np.corrcoef(first_delta, second_delta)[0, 1]),
                    "delay_logit_mae": float(np.mean(np.abs(first_delta - second_delta))),
                }
            )

    write_csv(output / "gain_scan_registered_points.csv", gain_rows)
    write_csv(output / "seed_summary.csv", summary_rows)
    write_csv(output / "seed_consistency.csv", consistency_rows)
    write_json(
        output / "diagnostic_manifest.json",
        {
            "status": "completed",
            "analysis_type": "posthoc_diagnostic_not_model_selection",
            "input": str(input_root),
            "alpha_grid": {
                "minimum": float(alpha_grid.min()),
                "maximum": float(alpha_grid.max()),
                "points": int(alpha_grid.size),
            },
            "runs": len(summary_rows),
            "heldout_session_e_accessed": False,
            "warning": "Best alpha is descriptive only and must not be used as a confirmatory setting.",
        },
    )


if __name__ == "__main__":
    main()
