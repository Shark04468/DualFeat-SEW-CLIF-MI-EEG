#!/usr/bin/env python3
"""Aggregate five fixed-seed V8 evidence audits before classifier training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_evidence_stability import (  # noqa: E402
    evaluate_v8_evidence_seed_stability,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, help="Comma-separated audit roots")
    parser.add_argument("--space", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    roots = [Path(value).resolve() for value in args.inputs.split(",") if value.strip()]
    if len(roots) != 5 or len(set(roots)) != 5:
        raise ValueError("seed-stability analysis requires exactly five distinct roots")
    summaries = []
    arrays = []
    identities = []
    for root in roots:
        manifest = read_json(root / "manifest.json")
        validate_run_artifact_manifest(
            root,
            required_files=tuple(manifest["required_files"]),
            verify_hashes=True,
        )
        fingerprint = read_json(root / "fingerprint.json")
        identities.append(
            {
                "data_sha256": fingerprint["data_sha256"],
                "inner_train_trial_ids": fingerprint["inner_train_trial_ids"],
                "config_sha256": fingerprint["config_sha256"],
                "model_config_sha256": fingerprint["model_config_sha256"],
                "source_tree_sha256": fingerprint["source_tree_sha256"],
            }
        )
        summaries.append(read_json(root / args.space / "summary.json"))
        with np.load(root / args.space / "evidence.npz", allow_pickle=False) as archive:
            arrays.append({name: archive[name] for name in archive.files})
    if any(identity != identities[0] for identity in identities[1:]):
        raise RuntimeError("evidence seed replicates do not share data/config/source identity")
    report, pairs = evaluate_v8_evidence_seed_stability(summaries, arrays)
    report = {
        **report,
        "evidence_space": args.space,
        "input_roots": [str(root) for root in roots],
        "input_identity": identities[0],
    }
    output = ensure_dir(args.output)
    write_json(output / "stability_gate.json", report)
    write_csv(output / "pairwise_stability.csv", pairs)
    write_json(output / "input_summaries.json", {"rows": summaries})
    required = (
        "manifest.json",
        "stability_gate.json",
        "pairwise_stability.csv",
        "input_summaries.json",
    )
    write_run_artifact_manifest(output, required_files=required)
    print(report, flush=True)


if __name__ == "__main__":
    main()
