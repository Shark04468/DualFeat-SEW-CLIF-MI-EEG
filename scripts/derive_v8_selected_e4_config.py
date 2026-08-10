#!/usr/bin/env python3
"""Bind E4 to the exact E5-selected E2 training contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def derive(selected_e2: dict, base_e4: dict) -> dict:
    if bool(selected_e2.get("data_access", {}).get("heldout_session_e_accessed")):
        raise RuntimeError("selected E2 config accessed held-out Session E")
    if selected_e2.get("stage") != "development" or base_e4.get("stage") != "development":
        raise RuntimeError("E2/E4 derivation is restricted to development configs")
    resolved = dict(base_e4)
    resolved["architecture_version"] = selected_e2["architecture_version"]
    for field in ("training", "augmentation", "preprocessing"):
        resolved[field] = dict(selected_e2[field])
    resolved["selection"] = dict(base_e4["selection"])
    for field in (
        "max_epochs",
        "patience",
        "minimum_epochs",
        "minimum_outer_retrain_epochs",
    ):
        resolved["selection"][field] = selected_e2["selection"][field]
    resolved["selected_e2_binding"] = {
        "experiment_id": selected_e2["experiment_id"],
        "candidate_id": selected_e2["hpo_provenance"]["candidate_id"],
        "selection_scope": selected_e2["hpo_provenance"]["selection_scope"],
        "session_e_accessed": False,
    }
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-e2", required=True)
    parser.add_argument("--base-e4", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    selected = yaml.safe_load(Path(args.selected_e2).resolve().read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(args.base_e4).resolve().read_text(encoding="utf-8"))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(derive(selected, base), sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
