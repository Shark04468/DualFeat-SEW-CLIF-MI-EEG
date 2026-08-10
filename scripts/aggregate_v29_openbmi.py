#!/usr/bin/env python3
"""Aggregate and gate the full 54-subject V29 OpenBMI replication."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import wilcoxon
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v29_openbmi import v29_replication_gate  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = ensure_dir(Path(args.output).resolve())
    config = yaml.safe_load(
        (ROOT / "configs" / "experiments" / "v29_openbmi_replication.yaml").read_text(
            encoding="utf-8"
        )
    )
    rows: list[dict[str, object]] = []
    for subject in config["subjects"]:
        status = read_json(root / f"subject_{subject:02d}" / "evaluation_status.json")
        if status.get("status") != "completed" or status.get("openbmi_s2_gradient_updates") is not False:
            raise RuntimeError(f"V29 Subject {subject} evaluation is incomplete or invalid")
        for seed in config["seeds"]:
            reference_label: np.ndarray | None = None
            for variant in config["variants"]:
                directory = root / f"subject_{subject:02d}" / f"seed_{seed}" / variant / "evaluation"
                metrics = read_json(directory / "metrics.json")
                if metrics.get("openbmi_s2_gradient_updates") is not False:
                    raise RuntimeError(f"S2 gradient update detected: {directory}")
                if float(metrics["teacher_replay_max_abs_error"]) > 1e-5:
                    raise RuntimeError(f"teacher replay drift detected: {directory}")
                with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
                    label = np.asarray(archive["label"], dtype=np.int64)
                    pred = np.asarray(archive["pred"], dtype=np.int64)
                if label.shape != (100,) or pred.shape != (100,):
                    raise RuntimeError(f"unexpected V29 prediction shape: {directory}")
                if reference_label is None:
                    reference_label = label
                elif not np.array_equal(reference_label, label):
                    raise RuntimeError(f"paired V29 labels are not aligned: {directory}")
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
    expected = len(config["subjects"]) * len(config["seeds"]) * len(config["variants"])
    if len(rows) != expected:
        raise RuntimeError(f"V29 metric row count mismatch: {len(rows)} != {expected}")
    write_csv(output / "subject_seed_metrics.csv", rows)
    lookup = {
        (int(row["subject"]), int(row["seed"]), str(row["variant"])): float(row["accuracy"])
        for row in rows
    }
    subject_rows: list[dict[str, object]] = []
    gains = []
    for subject in config["subjects"]:
        ann = np.asarray(
            [lookup[(subject, seed, "ann_sew_ce")] for seed in config["seeds"]], dtype=float
        )
        snn = np.asarray(
            [lookup[(subject, seed, "sew_clif_ce")] for seed in config["seeds"]], dtype=float
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
    rng = np.random.default_rng(29_000)
    indices = rng.integers(
        0, gain_array.size, size=(int(args.bootstrap_samples), gain_array.size)
    )
    bootstrap = gain_array[indices].mean(axis=1)
    decision = v29_replication_gate(gain_array, one_sided_wilcoxon_p=p_value)
    decision.update(
        {
            "subject_bootstrap_mean_ci95_low_pp": float(np.quantile(bootstrap, 0.025)),
            "subject_bootstrap_mean_ci95_high_pp": float(np.quantile(bootstrap, 0.975)),
            "ann_grand_accuracy": float(
                np.mean([row["ann_accuracy_seed_mean"] for row in subject_rows])
            ),
            "snn_grand_accuracy": float(
                np.mean([row["snn_accuracy_seed_mean"] for row in subject_rows])
            ),
            "inferential_unit": "subject_after_averaging_five_paired_seeds",
            "historical_exposure": config["historical_exposure"],
        }
    )
    write_json(output / "gate_decision.json", decision)
    write_json(
        output / "aggregate_summary.json",
        {
            "status": "completed",
            "subjects": 54,
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
