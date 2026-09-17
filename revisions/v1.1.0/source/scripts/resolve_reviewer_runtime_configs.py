"""Write immutable runtime configs with recovered-parent paths resolved."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import file_sha256
from dpc_snn.utils.io import read_json, write_json


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(value, sort_keys=False)
    if path.is_file() and path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"refusing to overwrite a different runtime config: {path}")
    if not path.is_file():
        path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e31", required=True)
    parser.add_argument("--publication", required=True)
    parser.add_argument("--v30", required=True)
    parser.add_argument("--baseline-source", required=True)
    parser.add_argument("--training-labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        "e31": Path(args.e31).resolve(),
        "publication": Path(args.publication).resolve(),
        "v30": Path(args.v30).resolve(),
        "baseline_source": Path(args.baseline_source).resolve(),
        "training_labels": Path(args.training_labels).resolve(),
    }
    required = {
        "e31": paths["e31"] / "checkpoint_barrier.json",
        "v30": paths["v30"] / "checkpoint_barrier.json",
    }
    for name, path in required.items():
        if not path.is_file() or read_json(path).get("status") != "sealed":
            raise RuntimeError(f"{name} parent barrier is missing or unsealed: {path}")
    if not paths["baseline_source"].is_dir() or not paths["training_labels"].is_dir():
        raise RuntimeError("baseline source and recovery-label roots must exist")
    reviewer_template = ROOT / "configs" / "experiments" / "reviewer_controls.yaml"
    binary_template = ROOT / "configs" / "experiments" / "bci2a_binary_sensitivity.yaml"
    reviewer = _load(reviewer_template)
    reviewer["parents"] = {key: str(value) for key, value in paths.items()}
    binary = _load(binary_template)
    binary["parents"] = {
        "e31": str(paths["e31"]),
        "training_labels": str(paths["training_labels"]),
    }
    output = Path(args.output).resolve()
    reviewer_path = output / "reviewer_controls.resolved.yaml"
    binary_path = output / "bci2a_binary_sensitivity.resolved.yaml"
    _write(reviewer_path, reviewer)
    _write(binary_path, binary)
    manifest = {
        "schema": "dpc-snn-reviewer-runtime-configs/v1",
        "status": "resolved",
        "lineage": "RECOVERED_NEW_PARENT",
        "paths": {key: str(value) for key, value in paths.items()},
        "parent_barrier_sha256": {key: file_sha256(value) for key, value in required.items()},
        "configs": {
            str(reviewer_path): file_sha256(reviewer_path),
            str(binary_path): file_sha256(binary_path),
        },
    }
    write_json(output / "runtime_config_manifest.json", manifest)
    print(yaml.safe_dump(manifest, sort_keys=False))


if __name__ == "__main__":
    main()
