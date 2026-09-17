#!/usr/bin/env python3
"""Stream OpenBMI into compact V8 views and safely remove per-subject raw files."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.data.openbmi_v8_cache import (
    load_openbmi_v8_cache,
    openbmi_v8_cache_paths,
    write_openbmi_v8_cache,
)
from dpc_snn.experiments.v62_protocol import file_sha256
from dpc_snn.utils.io import ensure_dir, read_json, write_json
from dpc_snn.utils.storage import configure_cache_env
from scripts.run_v8_publication_baselines import _openbmi_view


def _subjects(text: str) -> list[int]:
    values: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first, last = (int(value) for value in token.split("-", maxsplit=1))
            values.extend(range(first, last + 1))
        else:
            values.append(int(token))
    selected = sorted(set(values))
    if not selected or selected[0] < 1 or selected[-1] > 54:
        raise ValueError("OpenBMI subjects must be a non-empty subset of 1..54")
    return selected


def _raw_paths(storage_root: Path, subject: int) -> list[Path]:
    base = (
        storage_root
        / "cache"
        / "mne_data"
        / "MNE-lee2019-mi-data"
        / "gigadb-datasets"
        / "live"
        / "pub"
        / "10.5524"
        / "100001_101000"
        / "100542"
    )
    return [
        base / f"session{session}" / f"s{subject}" / f"sess0{session}_subj{subject:02d}_EEG_MI.mat"
        for session in (1, 2)
    ]


def _validate_subject(
    output: Path,
    *,
    subject: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    dataset = dict(config["datasets"]["openbmi"])
    channels = [str(value) for value in config["channel_names"]]
    records: dict[str, Any] = {}
    for session, role, expected_key in (
        ("S1", "training", "expected_train_trials"),
        ("S2", "evaluation", "expected_evaluation_trials"),
    ):
        x, y, rows, manifest, identity = load_openbmi_v8_cache(
            output,
            subject=subject,
            session=session,
            role=role,
            expected_trials=int(dataset[expected_key]),
            channel_names=channels,
        )
        npz_path, json_path = openbmi_v8_cache_paths(output, subject, session)
        records[session] = {
            "npz": str(npz_path),
            "json": str(json_path),
            "npz_sha256": file_sha256(npz_path),
            "json_sha256": file_sha256(json_path),
            "shape": list(x.shape),
            "labels": int(y.size),
            "trial_ids": len({str(row["trial_id"]) for row in rows}),
            "signal_sha256": manifest["signal_sha256"],
            "identity": identity,
        }
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "v8_publication_baselines.yaml"),
    )
    parser.add_argument("--storage-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subjects", default="1-54")
    parser.add_argument("--cleanup-raw", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    storage_root = Path(args.storage_root).expanduser().resolve()
    output = ensure_dir(Path(args.output).expanduser().resolve())
    configure_cache_env(storage_root)
    config = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    selected = _subjects(args.subjects)
    if args.validate_only:
        for subject in selected:
            records = _validate_subject(output, subject=subject, config=config)
            ready = read_json(output / f"subject_{subject:02d}" / "READY.json")
            if ready.get("status") != "compact_cache_ready" or ready.get("sessions") != records:
                raise RuntimeError(f"OpenBMI compact READY contract changed for subject {subject}")
        print({"status": "validated", "subjects": selected})
        return

    dataset = dict(config["datasets"]["openbmi"])
    channels = [str(value) for value in config["channel_names"]]
    os.environ["DPC_SNN_OPENBMI_V8_CACHE_DISABLE"] = "1"
    for subject in selected:
        ready_path = output / f"subject_{subject:02d}" / "READY.json"
        if ready_path.is_file():
            _validate_subject(output, subject=subject, config=config)
            print({"subject": subject, "status": "already_ready"}, flush=True)
            continue
        for session, role, expected_key in (
            ("S1", "training", "expected_train_trials"),
            ("S2", "evaluation", "expected_evaluation_trials"),
        ):
            x, y, rows, manifest, identity = _openbmi_view(
                subject=subject,
                session=session,
                role=role,
                dataset_config=dataset,
                channel_names=channels,
                expected_trials=int(dataset[expected_key]),
            )
            write_openbmi_v8_cache(
                output,
                subject=subject,
                session=session,
                role=role,
                x=x,
                y=y,
                rows=rows,
                manifest=manifest,
                identity=identity,
            )
        records = _validate_subject(output, subject=subject, config=config)
        raw_records = []
        for path in _raw_paths(storage_root, subject):
            resolved = path.resolve(strict=True)
            cache_root = (storage_root / "cache" / "mne_data").resolve(strict=True)
            if not resolved.is_relative_to(cache_root):
                raise RuntimeError(f"unsafe OpenBMI cleanup target: {resolved}")
            raw_records.append(
                {
                    "path": str(resolved),
                    "size": resolved.stat().st_size,
                    "sha256": file_sha256(resolved),
                }
            )
        if args.cleanup_raw:
            for record in raw_records:
                Path(record["path"]).unlink()
        ready = {
            "schema": "dpc-snn-openbmi-v8-compact-subject/v1",
            "status": "compact_cache_ready",
            "subject": subject,
            "sessions": records,
            "raw_files": raw_records,
            "raw_cleanup_completed": bool(args.cleanup_raw),
            "raw_files_remaining": sum(Path(record["path"]).is_file() for record in raw_records),
        }
        write_json(ready_path, ready)
        print({"subject": subject, "status": "compact_cache_ready"}, flush=True)


if __name__ == "__main__":
    main()
