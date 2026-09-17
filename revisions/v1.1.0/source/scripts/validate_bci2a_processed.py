#!/usr/bin/env python3
"""Validate compact BCI2a session exports before skipping raw acquisition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.data.bci2a import load_processed_npz
from dpc_snn.experiments.v8_publication_baselines import array_sha256
from dpc_snn.experiments.v62_protocol import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--session", choices=("T", "E"), required=True)
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 10)))
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    records = []
    for subject in args.subjects:
        path = root / f"A{subject:02d}.npz"
        data = load_processed_npz(path)
        x = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64)
        sessions = np.asarray(data["session"]).astype(str)
        subjects = np.asarray(data["subject"]).astype(str)
        trial_ids = np.asarray(data["trial_id"]).astype(str)
        labels, counts = np.unique(y, return_counts=True)
        if x.shape != (288, 22, 1250) or y.shape != (288,):
            raise RuntimeError(f"BCI2a compact shape mismatch: {path} {x.shape} {y.shape}")
        if set(sessions.tolist()) != {args.session} or set(subjects.tolist()) != {str(subject)}:
            raise RuntimeError(f"BCI2a subject/session metadata mismatch: {path}")
        if labels.tolist() != [0, 1, 2, 3] or counts.tolist() != [72, 72, 72, 72]:
            raise RuntimeError(f"BCI2a class balance mismatch: {path}")
        if len(set(trial_ids.tolist())) != 288 or float(data["sfreq"]) != 250.0:
            raise RuntimeError(f"BCI2a trial identity or sampling-rate mismatch: {path}")
        if len(data["ch_names"]) != 22:
            raise RuntimeError(f"BCI2a channel contract mismatch: {path}")
        records.append(
            {
                "subject": subject,
                "session": args.session,
                "file": str(path),
                "file_sha256": file_sha256(path),
                "signal_sha256": array_sha256(x),
                "label_sha256": array_sha256(y),
            }
        )
    print(json.dumps({"status": "validated", "records": records}, indent=2))


if __name__ == "__main__":
    main()
