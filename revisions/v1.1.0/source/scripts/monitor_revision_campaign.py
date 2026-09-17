"""Write compact progress snapshots for the long four-GPU revision campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

STAGES = (
    "bci2a_training_data",
    "bci2a_evaluation_data",
    "v30_recovery_freeze",
    "publication_bci2a",
    "publication_openbmi",
    "v30_recovered",
    "v31_recovered",
    "resolve_reviewer_configs",
    "reviewer_train",
    "reviewer_seal",
    "reviewer_evaluate",
    "reviewer_aggregate",
    "binary_train",
    "binary_seal",
    "binary_evaluate",
    "binary_aggregate",
    "utility_profile",
    "final_validation",
)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _gpu_snapshot() -> list[dict[str, str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        index, name, utilization, used, total, temperature = [
            value.strip() for value in line.split(",")
        ]
        rows.append(
            {
                "index": index,
                "name": name,
                "utilization_percent": utilization,
                "memory_used_mib": used,
                "memory_total_mib": total,
                "temperature_c": temperature,
            }
        )
    return rows


def _snapshot(root: Path) -> dict:
    control = root / "control"
    completed = [stage for stage in STAGES if (control / f"{stage}.done").is_file()]
    checkpoint_count = sum(1 for _ in root.rglob("checkpoint.pt")) if root.exists() else 0
    prediction_count = sum(1 for _ in root.rglob("predictions.npz")) if root.exists() else 0
    ready = root / "READY_FOR_MANUSCRIPT_REVISION.json"
    return {
        "schema": "dpc-snn-revision-monitor/v1",
        "timestamp_unix": time.time(),
        "root": str(root),
        "stages_total": len(STAGES),
        "stages_completed": len(completed),
        "completed": completed,
        "next_stage": next((stage for stage in STAGES if stage not in completed), None),
        "checkpoint_files": checkpoint_count,
        "prediction_files": prediction_count,
        "ready": ready.is_file(),
        "gpus": _gpu_snapshot(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 10:
        raise ValueError("monitor interval must be at least 10 seconds")
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    while True:
        value = _snapshot(root)
        _atomic_json(output, value)
        print(json.dumps(value, indent=2), flush=True)
        if args.once or value["ready"]:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
