#!/usr/bin/env python3
"""Freeze V25 subject-seed epochs and residual scales from Session-T evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v25_confirmation import (  # noqa: E402
    select_aggregate_residual_scale,
    upper_median_epoch,
    validate_v25_freeze,
)
from dpc_snn.experiments.v62_protocol import file_sha256, sha256_fingerprint  # noqa: E402
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    source_tree_digest,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_json  # noqa: E402


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _roots(args: argparse.Namespace, seed: int) -> tuple[Path, Path, Path]:
    if seed == 0:
        return (
            Path(args.e1_seed0).resolve(),
            Path(args.v9_seed0).resolve(),
            Path(args.v9_seed0).resolve() / "evaluation_seed0",
        )
    campaign = Path(args.e24).resolve()
    v9 = campaign / f"v9_seed_{seed}"
    return campaign / f"e1_seed_{seed}", v9, v9 / "evaluation"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1-seed0", required=True)
    parser.add_argument("--v9-seed0", required=True)
    parser.add_argument("--e24", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"V25 freeze output must be new: {output}")
    ensure_dir(output)
    entries: dict[str, Any] = {}
    evidence_files: dict[str, str] = {}
    for seed in range(5):
        e1_root, v9_root, evaluation_root = _roots(args, seed)
        scale_rows = _rows(evaluation_root / "scale_search.csv")
        for subject in range(1, 10):
            fixed_epochs: dict[str, int] = {}
            epoch_evidence: dict[str, list[int]] = {}
            for model in ("atcnet", "fbcnet"):
                values = []
                for fold in range(6):
                    path = (
                        e1_root
                        / model
                        / f"subject_{subject:02d}"
                        / f"seed_{seed}"
                        / f"fold_{fold}"
                        / "result.json"
                    )
                    result = read_json(path)
                    values.append(int(result["selected_outer_retrain_epoch"]))
                    evidence_files[str(path)] = file_sha256(path)
                fixed_epochs[model] = upper_median_epoch(values)
                epoch_evidence[model] = values
            residual_scales: dict[str, float] = {}
            scale_evidence: dict[str, Any] = {}
            for variant in ("ann_plain_ce", "sew_clif_ce"):
                values = []
                for fold in range(6):
                    path = (
                        v9_root
                        / f"subject_{subject:02d}"
                        / f"seed_{seed}"
                        / f"fold_{fold}"
                        / variant
                        / "metrics.json"
                    )
                    result = read_json(path)
                    values.append(int(result["selected_epoch"]))
                    evidence_files[str(path)] = file_sha256(path)
                fixed_epochs[variant] = upper_median_epoch(values)
                epoch_evidence[variant] = values
                relevant = [
                    row
                    for row in scale_rows
                    if int(row["subject"]) == subject and row["variant"] == variant
                ]
                selected, summary = select_aggregate_residual_scale(relevant)
                residual_scales[variant] = selected
                scale_evidence[variant] = summary
            entries[f"subject_{subject:02d}_seed_{seed}"] = {
                "subject": subject,
                "seed": seed,
                "fixed_epochs": fixed_epochs,
                "epoch_evidence": epoch_evidence,
                "residual_scales": residual_scales,
                "scale_evidence": scale_evidence,
            }
        scale_path = evaluation_root / "scale_search.csv"
        evidence_files[str(scale_path)] = file_sha256(scale_path)

    source_tree = collect_source_tree_manifest(ROOT)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "architecture_id": "v25_equal_probability_dual_feature_sew_clif_residual",
        "protocol": "subject_dependent_bci2a_session_t_train_session_e_evaluate",
        "subjects": list(range(1, 10)),
        "seeds": list(range(5)),
        "variants": ["ann_plain_ce", "sew_clif_ce"],
        "teacher": "equal_atcnet_fbcnet_probability",
        "residual_scale_candidates": [0.0, 0.25, 0.5, 1.0],
        "selection_source": "Session-T nested six-fold OOF only",
        "historical_data_exposure": {
            "bci2a_session_e": True,
            "openbmi_session_s2": True,
            "disclosure": (
                "Prior project versions evaluated both held-out sets. V25 is locked against "
                "using them for current checkpoint or hyperparameter selection, but is not a "
                "project-level blind confirmation."
            ),
        },
        "current_v25_selection_used_session_e": False,
        "source_tree_sha256": source_tree_digest(source_tree),
        "evidence_files": dict(sorted(evidence_files.items())),
        "entries": entries,
    }
    payload["evidence_sha256"] = sha256_fingerprint(payload["evidence_files"])
    payload["combined_sha256"] = sha256_fingerprint(payload)
    validate_v25_freeze(payload)
    write_json(output / "freeze_manifest.json", payload)
    write_json(
        output / "freeze_summary.json",
        {
            "status": "frozen",
            "entries": len(entries),
            "source_tree_sha256": payload["source_tree_sha256"],
            "freeze_sha256": payload["combined_sha256"],
            "historical_data_exposure": payload["historical_data_exposure"],
        },
    )
    print(json.dumps(read_json(output / "freeze_summary.json"), indent=2))


if __name__ == "__main__":
    main()
