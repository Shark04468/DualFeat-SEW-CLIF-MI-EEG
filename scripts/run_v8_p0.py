#!/usr/bin/env python3
"""Materialize the V8 accuracy-first preregistration and held-out locks."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    V8_PROTOCOL_ID,
    build_v8_run_fingerprint,
    collect_source_tree_manifest,
    source_tree_digest,
    validate_v8_hpo_budget,
    validate_v8_resume_fingerprint,
    v8_heldout_lock_manifest,
    write_v8_fingerprint,
)
from dpc_snn.utils.io import ensure_dir, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402


P0_REQUIRED_FILES = (
    "manifest.json",
    "resolved_config.yaml",
    "source_fingerprint.json",
    "source_tree_manifest.json",
    "protocol_manifest.json",
    "heldout_lock.json",
    "metrics.json",
    "history.csv",
    "runtime_status.json",
    "stdout.log",
    "stderr.log",
)


def _environment(storage_root: Path | None) -> dict[str, object]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "torch", "scipy", "scikit-learn", "mne", "moabb"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "storage_root": str(storage_root) if storage_root is not None else None,
        "cache_environment": {
            name: os.environ.get(name)
            for name in (
                "DPC_SNN_STORAGE_ROOT",
                "MNE_DATA",
                "MOABB_DATA",
                "XDG_CACHE_HOME",
                "HF_HOME",
                "TORCH_HOME",
                "PIP_CACHE_DIR",
                "TMPDIR",
                "PYTHONPYCACHEPREFIX",
            )
        },
    }


def _default_output(storage_root: Path | None) -> Path:
    base = storage_root if storage_root is not None else ROOT
    return base / "runs" / "v8_accuracy_first" / "P0_protocol"


def _candidate_hpo_grid(config: dict[str, object]) -> list[dict[str, object]]:
    budget = config["hpo"]
    if not isinstance(budget, dict):
        raise ValueError("V8 hpo configuration must be a mapping")
    candidates: list[dict[str, object]] = []
    for index in range(int(budget["frontend_budget"])):
        candidates.append({"phase": "frontend", "index": index})
    for index in range(int(budget["decoder_budget"])):
        candidates.append({"phase": "decoder", "index": index})
    for index in range(int(budget["training_budget"])):
        candidates.append({"phase": "training", "index": index})
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/v8_p0_protocol.yaml")
    parser.add_argument("--output", default="")
    parser.add_argument("--storage-root", default="")
    args = parser.parse_args()

    started_at = time.time()
    storage_root = configure_cache_env(args.storage_root or None)
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["protocol"] != "dpc_snn_v8_accuracy_first":
        raise ValueError("the P0 runner received a non-V8 protocol config")
    maximum_hpo = int(config["hpo"]["maximum_unique_configurations"])
    candidates = _candidate_hpo_grid(config)
    validate_v8_hpo_budget(candidates, maximum=maximum_hpo)

    output = ensure_dir(Path(args.output).resolve() if args.output else _default_output(storage_root))
    if storage_root is not None:
        resolved_output = output.resolve()
        resolved_storage = storage_root.resolve()
        if resolved_storage not in resolved_output.parents:
            raise RuntimeError(
                f"formal V8 artifacts must be written under the data disk: {resolved_storage}"
            )

    source_tree = collect_source_tree_manifest(ROOT)
    environment = _environment(storage_root)
    fingerprint = build_v8_run_fingerprint(
        resolved_run_config=config,
        source_tree=source_tree,
        data={"status": "locked", "content_not_opened_at_p0": True},
        split=config["development"],
        augmentation={"status": "not_selected", "selection_session": "T"},
        prior={"status": "not_fitted", "scope": "inner_train_only"},
        checkpoint={"policy": "fresh_v8_namespace", "resume": False},
        environment=environment,
    )

    fingerprint_path = output / "source_fingerprint.json"
    if fingerprint_path.is_file():
        validate_v8_resume_fingerprint(fingerprint_path, fingerprint)
        validate_run_artifact_manifest(
            output,
            required_files=P0_REQUIRED_FILES,
            verify_hashes=True,
            verify_prediction_schema=False,
        )
        print(json.dumps({"status": "already_complete", "output": str(output)}, indent=2))
        return

    shutil.copyfile(config_path, output / "resolved_config.yaml")
    write_v8_fingerprint(fingerprint_path, fingerprint)
    write_json(output / "source_tree_manifest.json", source_tree)
    heldout_lock = v8_heldout_lock_manifest()
    write_json(output / "heldout_lock.json", heldout_lock)
    protocol_manifest = {
        "protocol": V8_PROTOCOL_ID,
        "status": "registered",
        "registered_at": started_at,
        "development": config["development"],
        "stages": config["stages"],
        "comparisons": config["comparisons"],
        "statistics": config["statistics"],
        "hpo_candidates": candidates,
        "heldout_lock": heldout_lock,
    }
    write_json(output / "protocol_manifest.json", protocol_manifest)
    metrics = {
        "status": "passed",
        "protocol": V8_PROTOCOL_ID,
        "source_file_count": len(source_tree),
        "source_tree_sha256": source_tree_digest(source_tree),
        "run_fingerprint_sha256": fingerprint["combined_sha256"],
        "hpo_configuration_budget": len(candidates),
        "bci2a_session_e_locked": True,
        "openbmi_session_s2_locked": True,
        "storage_root": str(storage_root) if storage_root is not None else None,
        "output": str(output),
        "elapsed_seconds": time.time() - started_at,
    }
    write_json(output / "metrics.json", metrics)
    write_csv(output / "history.csv", [{"stage": "P0", **metrics}])
    write_json(
        output / "runtime_status.json",
        {"status": "completed", "started_at": started_at, "completed_at": time.time()},
    )
    (output / "stdout.log").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output / "stderr.log").write_text("", encoding="utf-8")
    write_run_artifact_manifest(output, required_files=P0_REQUIRED_FILES)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
