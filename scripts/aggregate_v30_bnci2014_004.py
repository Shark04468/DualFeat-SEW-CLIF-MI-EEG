#!/usr/bin/env python3
"""Aggregate and apply the predeclared V30 subject-level promotion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import wilcoxon
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v30_bnci2014_004 import v30_blind_gate  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402


def _evaluation_trial_bounds(
    root: Path, subject: int, config: dict[str, object]
) -> tuple[int, int]:
    dataset = config["dataset"]
    if not isinstance(dataset, dict):
        raise TypeError("V30 dataset configuration must be a mapping")
    minimum = int(dataset["evaluation_trials_minimum"])
    maximum = int(dataset["evaluation_trials_maximum"])
    if subject != 2:
        return minimum, maximum

    path = root / "post_barrier_metadata_erratum_subject_02.json"
    erratum = read_json(path)
    observed = erratum.get("observed_before_any_subject_02_prediction", {})
    if (
        erratum.get("schema")
        != "dpc-snn-v30-post-barrier-metadata-erratum/v1"
        or erratum.get("subject") != 2
        or erratum.get("original_evaluation_trials_minimum") != minimum
        or erratum.get("corrected_evaluation_trials_minimum_in_memory") != 280
        or erratum.get("evaluation_trials_maximum_unchanged") != maximum
        or erratum.get("prediction_logic_changed") is not False
        or erratum.get("preprocessing_changed") is not False
        or erratum.get("model_or_checkpoint_changed") is not False
        or observed.get("trials") != 280
        or observed.get("session_trial_counts") != {"04E": 120, "05E": 160}
        or observed.get("class_counts") != [140, 140]
    ):
        raise RuntimeError("Invalid V30 Subject 2 metadata-only erratum")
    return 280, maximum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = ensure_dir(Path(args.output).resolve())
    config = yaml.safe_load(
        (ROOT / "configs" / "experiments" / "v30_bnci2014_004_blind.yaml").read_text(
            encoding="utf-8"
        )
    )

    rows: list[dict[str, object]] = []
    reference_rows: list[dict[str, object]] = []
    for subject in config["dataset"]["subjects"]:
        minimum, maximum = _evaluation_trial_bounds(root, int(subject), config)
        status = read_json(root / f"subject_{subject:02d}" / "evaluation_status.json")
        if (
            status.get("status") != "completed"
            or status.get("evaluation_gradient_updates") is not False
        ):
            raise RuntimeError(f"V30 Subject {subject} evaluation is incomplete or invalid")
        for seed in config["seeds"]:
            seed_dir = root / f"subject_{subject:02d}" / f"seed_{seed}"
            reference = read_json(seed_dir / "evaluation" / "reference_metrics.json")
            for name in ("atcnet", "fbcnet", "equal_teacher"):
                reference_rows.append(
                    {
                        "subject": subject,
                        "seed": seed,
                        "variant": name,
                        "accuracy": float(reference[name]["accuracy"]),
                        "balanced_accuracy": float(reference[name]["balanced_accuracy"]),
                        "kappa": float(reference[name]["kappa"]),
                        "macro_f1": float(reference[name]["macro_f1"]),
                    }
                )
            reference_label: np.ndarray | None = None
            for variant in config["variants"]:
                directory = seed_dir / "students" / variant / "evaluation"
                metrics = read_json(directory / "metrics.json")
                if metrics.get("evaluation_gradient_updates") is not False:
                    raise RuntimeError(f"V30 evaluation update detected: {directory}")
                if metrics.get("state_unchanged_during_evaluation") is not True:
                    raise RuntimeError(f"V30 state mutation detected: {directory}")
                with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
                    label = np.asarray(archive["label"], dtype=np.int64)
                    pred = np.asarray(archive["pred"], dtype=np.int64)
                if (
                    label.ndim != 1
                    or pred.shape != label.shape
                    or not minimum <= label.size <= maximum
                ):
                    raise RuntimeError(f"Unexpected V30 prediction shape: {directory}")
                if reference_label is None:
                    reference_label = label
                elif not np.array_equal(reference_label, label):
                    raise RuntimeError(f"Paired V30 labels are not aligned: {directory}")
                if not np.isclose(float((label == pred).mean()), float(metrics["accuracy"])):
                    raise RuntimeError(f"V30 accuracy cannot be recomputed: {directory}")
                rows.append(
                    {
                        "subject": subject,
                        "seed": seed,
                        "variant": variant,
                        "accuracy": float(metrics["accuracy"]),
                        "balanced_accuracy": float(metrics["balanced_accuracy"]),
                        "kappa": float(metrics["kappa"]),
                        "macro_f1": float(metrics["macro_f1"]),
                        "mean_firing_rate": float(metrics["mean_firing_rate"]),
                    }
                )

    expected = (
        len(config["dataset"]["subjects"])
        * len(config["seeds"])
        * len(config["variants"])
    )
    if len(rows) != expected:
        raise RuntimeError(f"V30 metric row count mismatch: {len(rows)} != {expected}")
    write_csv(output / "subject_seed_metrics.csv", rows)
    write_csv(output / "reference_subject_seed_metrics.csv", reference_rows)

    lookup = {
        (int(row["subject"]), int(row["seed"]), str(row["variant"])): float(
            row["accuracy"]
        )
        for row in rows
    }
    subject_rows: list[dict[str, object]] = []
    gains: list[float] = []
    for subject in config["dataset"]["subjects"]:
        ann = np.asarray(
            [lookup[(subject, seed, "ann_sew_ce")] for seed in config["seeds"]],
            dtype=float,
        )
        snn = np.asarray(
            [lookup[(subject, seed, "sew_clif_ce")] for seed in config["seeds"]],
            dtype=float,
        )
        gain = 100.0 * float((snn - ann).mean())
        gains.append(gain)
        subject_rows.append(
            {
                "subject": subject,
                "ann_accuracy_seed_mean": float(ann.mean()),
                "snn_accuracy_seed_mean": float(snn.mean()),
                "snn_minus_ann_pp": gain,
            }
        )
    write_csv(output / "subject_metrics.csv", subject_rows)
    gain_array = np.asarray(gains, dtype=float)
    nonzero = gain_array[gain_array != 0.0]
    p_value = (
        float(wilcoxon(nonzero, alternative="greater", method="auto").pvalue)
        if nonzero.size
        else 1.0
    )
    rng = np.random.default_rng(30_004)
    indices = rng.integers(
        0, gain_array.size, size=(int(args.bootstrap_samples), gain_array.size)
    )
    bootstrap = gain_array[indices].mean(axis=1)
    decision = v30_blind_gate(gain_array, one_sided_wilcoxon_p=p_value)
    decision.update(
        {
            "subject_bootstrap_mean_ci95_low_pp": float(
                np.quantile(bootstrap, 0.025)
            ),
            "subject_bootstrap_mean_ci95_high_pp": float(
                np.quantile(bootstrap, 0.975)
            ),
            "ann_grand_accuracy": float(
                np.mean([row["ann_accuracy_seed_mean"] for row in subject_rows])
            ),
            "snn_grand_accuracy": float(
                np.mean([row["snn_accuracy_seed_mean"] for row in subject_rows])
            ),
            "atcnet_grand_accuracy": float(
                np.mean(
                    [
                        row["accuracy"]
                        for row in reference_rows
                        if row["variant"] == "atcnet"
                    ]
                )
            ),
            "fbcnet_grand_accuracy": float(
                np.mean(
                    [
                        row["accuracy"]
                        for row in reference_rows
                        if row["variant"] == "fbcnet"
                    ]
                )
            ),
            "equal_teacher_grand_accuracy": float(
                np.mean(
                    [
                        row["accuracy"]
                        for row in reference_rows
                        if row["variant"] == "equal_teacher"
                    ]
                )
            ),
            "inferential_unit": "subject_after_averaging_five_paired_seeds",
            "project_level_blind_confirmation": True,
        }
    )
    write_json(output / "gate_decision.json", decision)
    write_json(
        output / "aggregate_summary.json",
        {
            "status": "completed",
            "subjects": 9,
            "seeds": 5,
            "variants": config["variants"],
            "decision": decision,
        },
    )
    print(json.dumps(decision, indent=2))
    if decision["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
