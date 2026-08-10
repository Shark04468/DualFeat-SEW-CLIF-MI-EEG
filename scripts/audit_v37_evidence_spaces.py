#!/usr/bin/env python
"""Compare classifier-free delay evidence spaces on Subject 1 Session T."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.analysis.evidence_space import (
    HurdleEvidenceConfig,
    apply_alignment_matrix,
    apply_euclidean_alignment,
    current_source_density,
    evaluate_evidence_space,
    fit_var_innovations,
    model_evidence_features,
    phase_surrogate,
    task_window,
    trial_band_lag_scores,
)
from dpc_snn.data.bci2a import load_processed_npz, subject_session_data
from dpc_snn.models.dpc_snn import DEFAULT_DELAY_EVIDENCE_EDGES
from dpc_snn.utils.io import ensure_dir, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/bci2a_strict")
    parser.add_argument("--output", required=True)
    parser.add_argument("--subject", default="1")
    parser.add_argument("--evidence-rate", type=float, default=125.0)
    parser.add_argument("--max-delay-steps", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-batch-size", type=int, default=16)
    args = parser.parse_args()
    output = ensure_dir(args.output)
    data = load_processed_npz(args.data)
    train = subject_session_data(data, args.subject, session="T")
    x = task_window(train)
    sfreq = float(train["sfreq"])
    ch_names = train.get("ch_names")
    if not ch_names:
        raise ValueError("Strict CSD audit requires EEG channel names")
    surrogate_raw = phase_surrogate(x, args.seed + 1009)
    ea, ea_matrix = apply_euclidean_alignment(x)
    surrogate_ea = apply_alignment_matrix(surrogate_raw, ea_matrix)
    csd = current_source_density(x, list(ch_names), sfreq)
    surrogate_csd = current_source_density(surrogate_raw, list(ch_names), sfreq)
    spaces = {
        "raw": (x, x[..., ::-1].copy(), surrogate_raw),
        "ea": (ea, ea[..., ::-1].copy(), surrogate_ea),
        "csd": (csd, csd[..., ::-1].copy(), surrogate_csd),
        "innovations": (
            fit_var_innovations(csd),
            fit_var_innovations(csd[..., ::-1].copy()),
            fit_var_innovations(surrogate_csd),
        ),
    }
    config = HurdleEvidenceConfig(
        max_delay_steps=args.max_delay_steps,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.seed,
    )
    summaries = []
    for name, (normal, reversed_x, surrogate) in spaces.items():
        print(
            json.dumps({"event": "evidence_space_started", "space": name}),
            flush=True,
        )
        graph_steps = int(round(normal.shape[-1] / sfreq * args.evidence_rate))
        normal_carrier, normal_envelope = model_evidence_features(
            normal, sfreq, DEFAULT_DELAY_EVIDENCE_EDGES, n_nodes=16,
            graph_steps=graph_steps, device=args.device
        )
        reversed_carrier, reversed_envelope = model_evidence_features(
            reversed_x, sfreq, DEFAULT_DELAY_EVIDENCE_EDGES, n_nodes=16,
            graph_steps=graph_steps, device=args.device
        )
        surrogate_carrier, surrogate_envelope = model_evidence_features(
            surrogate, sfreq, DEFAULT_DELAY_EVIDENCE_EDGES, n_nodes=16,
            graph_steps=graph_steps, device=args.device
        )
        normal_scores = trial_band_lag_scores(
            normal_carrier, normal_envelope, args.max_delay_steps,
            device=args.device, batch_size=args.score_batch_size
        )
        reversed_scores = trial_band_lag_scores(
            reversed_carrier, reversed_envelope, args.max_delay_steps,
            device=args.device, batch_size=args.score_batch_size
        )
        surrogate_scores = trial_band_lag_scores(
            surrogate_carrier, surrogate_envelope, args.max_delay_steps,
            device=args.device, batch_size=args.score_batch_size
        )
        summary, arrays = evaluate_evidence_space(
            normal_scores, reversed_scores, surrogate_scores, config
        )
        summary = {
            "evidence_space": name,
            "evidence_bands": 12,
            "evidence_nodes": 16,
            "route_entities": 192,
            "same_band_signal": "carrier",
            "cross_band_signal": "envelope",
            "statistic": "offline_bic_approximate_bayes_factor",
            **summary,
        }
        summaries.append(summary)
        np.savez_compressed(output / f"{name}_evidence.npz", **arrays)
        write_json(output / f"{name}_summary.json", summary)
        write_csv(output / "evidence_space_summary.csv", summaries)
        print(
            json.dumps(
                {
                    "event": "evidence_space_completed",
                    "space": name,
                    "passed": summary["passed"],
                    "accepted_edges": summary["accepted_edges"],
                }
            ),
            flush=True,
        )
    passing = [row["evidence_space"] for row in summaries if row["passed"]]
    write_csv(output / "evidence_space_summary.csv", summaries)
    write_json(
        output / "evidence_gate.json",
        {
            "status": "passed" if passing else "failed",
            "passing_spaces": passing,
            "classifier_training": False,
            "subject": str(args.subject),
            "session": "T",
            "heldout_session_E_accessed": False,
            "evidence_geometry": "12_band_x_16_fixed_nodes",
            "online_route_odds_used": False,
            "criteria": {
                "split_half_delay_correlation": 0.5,
                "bootstrap_edge_frequency": 0.7,
                "time_reversal_transpose_correlation": 0.5,
                "phase_surrogate_p": 0.05,
                "phase_surrogate_min_log_bf_drop": float(np.log(1.5)),
            },
            "spaces": summaries,
        },
    )
    print(json.dumps({"passing_spaces": passing, "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
