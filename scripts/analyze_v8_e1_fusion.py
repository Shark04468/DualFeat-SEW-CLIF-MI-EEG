#!/usr/bin/env python3
"""Measure cross-fitted error complementarity among Session-T E1 models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import file_sha256  # noqa: E402
from dpc_snn.utils.io import ensure_dir, write_csv, write_json  # noqa: E402


def _csv(value: str, cast: Any = str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    required = {"probabilities", "pred", "label", "session", "run", "trial_id"}
    if not required <= values.keys():
        raise RuntimeError(f"prediction archive is missing fields: {path}")
    if set(values["session"].astype(str).tolist()) != {"T"}:
        raise RuntimeError(f"fusion analysis received non-Session-T trials: {path}")
    if values["probabilities"].shape != (288, 4):
        raise RuntimeError(f"unexpected prediction shape: {path}")
    return values


def _aligned(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> None:
    for key in ("trial_id", "label", "run", "session"):
        if not np.array_equal(reference[key], candidate[key]):
            raise RuntimeError(f"OOF predictions are not aligned on {key}")


def _accuracy(label: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.asarray(label) == np.asarray(prediction)))


def cross_fitted_probability_weight(
    anchor_probability: np.ndarray,
    branch_probability: np.ndarray,
    label: np.ndarray,
    runs: np.ndarray,
    *,
    weights: Sequence[float] = tuple(np.linspace(0.0, 1.0, 21)),
) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    """Select the anchor weight on five runs, then evaluate the omitted run.

    Ties prefer more anchor weight.  A branch therefore cannot enter merely
    because several weights produce the same development accuracy.
    """

    label = np.asarray(label)
    runs = np.asarray(runs).astype(str)
    output = np.empty(label.shape, dtype=np.int64)
    selections: list[dict[str, float | str]] = []
    for held_run in sorted(set(runs.tolist())):
        train = runs != held_run
        test = ~train
        ranked: list[tuple[float, float]] = []
        for weight in weights:
            probability = (
                float(weight) * anchor_probability
                + (1.0 - float(weight)) * branch_probability
            )
            ranked.append((_accuracy(label[train], probability[train].argmax(1)), float(weight)))
        train_accuracy, selected = max(ranked, key=lambda item: (item[0], item[1]))
        fused = selected * anchor_probability[test] + (1.0 - selected) * branch_probability[test]
        output[test] = fused.argmax(1)
        selections.append(
            {
                "held_run": held_run,
                "selected_anchor_weight": selected,
                "five_run_selection_accuracy": train_accuracy,
                "held_run_accuracy": _accuracy(label[test], output[test]),
            }
        )
    return output, selections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--anchor", default="atcnet")
    parser.add_argument(
        "--branches", default="fbcnet,tcformer,eeg_conformer,mi_snn_plif"
    )
    parser.add_argument("--subjects", default="1,3,8")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.e1_root).resolve()
    output = ensure_dir(Path(args.output).resolve())
    branches = _csv(args.branches)
    subjects = _csv(args.subjects, int)
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "protocol": "bci2a_session_t_nested_six_fold_oof_cross_fitted_fusion",
        "anchor": args.anchor,
        "seed": int(args.seed),
        "weight_grid": [float(value) for value in np.linspace(0.0, 1.0, 21)],
        "tie_break": "prefer_more_anchor_weight",
        "session_e_accessed": False,
        "comparisons": {},
    }
    for branch in branches:
        details["comparisons"][branch] = {}
        for subject in subjects:
            anchor_path = (
                root / args.anchor / f"subject_{subject:02d}" / f"seed_{args.seed}" / "predictions.npz"
            )
            branch_path = (
                root / branch / f"subject_{subject:02d}" / f"seed_{args.seed}" / "predictions.npz"
            )
            anchor = _load(anchor_path)
            candidate = _load(branch_path)
            _aligned(anchor, candidate)
            label = anchor["label"]
            anchor_pred = anchor["probabilities"].argmax(1)
            branch_pred = candidate["probabilities"].argmax(1)
            equal_pred = (anchor["probabilities"] + candidate["probabilities"]).argmax(1)
            cross_pred, selections = cross_fitted_probability_weight(
                anchor["probabilities"],
                candidate["probabilities"],
                label,
                anchor["run"],
            )
            anchor_correct = anchor_pred == label
            branch_correct = branch_pred == label
            row = {
                "anchor": args.anchor,
                "branch": branch,
                "subject": subject,
                "seed": int(args.seed),
                "anchor_accuracy": _accuracy(label, anchor_pred),
                "branch_accuracy": _accuracy(label, branch_pred),
                "equal_probability_accuracy": _accuracy(label, equal_pred),
                "cross_fitted_weight_accuracy": _accuracy(label, cross_pred),
                "oracle_either_correct_accuracy": float(np.mean(anchor_correct | branch_correct)),
                "branch_corrections_of_anchor_errors": int(np.sum(~anchor_correct & branch_correct)),
                "branch_regressions_of_anchor_correct": int(np.sum(anchor_correct & ~branch_correct)),
                "prediction_disagreements": int(np.sum(anchor_pred != branch_pred)),
                "session_e_accessed": False,
            }
            rows.append(row)
            details["comparisons"][branch][f"subject_{subject:02d}"] = {
                **row,
                "fold_weight_selections": selections,
                "anchor_predictions_sha256": file_sha256(anchor_path),
                "branch_predictions_sha256": file_sha256(branch_path),
            }

    aggregate: list[dict[str, Any]] = []
    for branch in branches:
        selected = [row for row in rows if row["branch"] == branch]
        aggregate.append(
            {
                "anchor": args.anchor,
                "branch": branch,
                "subjects": len(selected),
                **{
                    key: float(np.mean([float(row[key]) for row in selected]))
                    for key in (
                        "anchor_accuracy",
                        "branch_accuracy",
                        "equal_probability_accuracy",
                        "cross_fitted_weight_accuracy",
                        "oracle_either_correct_accuracy",
                    )
                },
                "equal_probability_delta_pp": 100.0
                * float(
                    np.mean(
                        [
                            float(row["equal_probability_accuracy"])
                            - float(row["anchor_accuracy"])
                            for row in selected
                        ]
                    )
                ),
                "cross_fitted_weight_delta_pp": 100.0
                * float(
                    np.mean(
                        [
                            float(row["cross_fitted_weight_accuracy"])
                            - float(row["anchor_accuracy"])
                            for row in selected
                        ]
                    )
                ),
                "session_e_accessed": False,
            }
        )
    details["aggregate"] = aggregate
    write_csv(output / "per_subject.csv", rows)
    write_csv(output / "aggregate.csv", aggregate)
    write_json(output / "analysis.json", details)
    print(json.dumps({"status": "completed", "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
