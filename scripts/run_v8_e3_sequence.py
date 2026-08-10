#!/usr/bin/env python3
"""Run the preregistered V8 delay stages with strict sequential stopping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.utils.io import ensure_dir, read_json, write_json  # noqa: E402


STAGES = (
    ("static_slow_within_band", "configs/experiments/v8_e3_static_delay.yaml"),
    ("static_slow_cross_band", "configs/experiments/v8_e3_cross_band_delay.yaml"),
    ("static_fast_within_band", "configs/experiments/v8_e3_fast_delay.yaml"),
    ("contextual_slow_residual", "configs/experiments/v8_e3_contextual_residual.yaml"),
    ("phase_fast_residual", "configs/experiments/v8_e3_phase_residual.yaml"),
)


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--parent-e2", required=True)
    parser.add_argument("--parent-variant", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = ensure_dir(Path(args.output).resolve())
    parent = Path(args.parent_e2).resolve()
    parent_status = read_json(parent / "campaign_status.json")
    if (
        parent_status.get("status") != "completed"
        or parent_status.get("stage") != "E2"
        or args.parent_variant not in parent_status.get("confirmed_variants", [])
        or bool(parent_status.get("session_e_accessed"))
    ):
        raise RuntimeError("E3 sequence requires a completed, held-out-locked E2 parent")

    results: list[dict[str, object]] = []
    selected_stage: str | None = None
    for index, (stage, relative_config) in enumerate(STAGES, start=1):
        stage_root = output / f"{index:02d}_{stage}"
        gate_root = output / f"{index:02d}_{stage}_gate"
        config = ROOT / relative_config
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_v8_e3_static_delay.py"),
                "--data",
                str(Path(args.data).resolve()),
                "--parent-e2",
                str(parent),
                "--parent-variant",
                args.parent_variant,
                "--model-config",
                str(Path(args.model_config).resolve()),
                "--output",
                str(stage_root),
                "--config",
                str(config),
                "--device",
                args.device,
            ]
        )
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_v8_e3_gate.py"),
                "--e3",
                str(stage_root),
                "--output",
                str(gate_root),
                "--config",
                str(config),
                "--parent-variant",
                args.parent_variant,
            ]
        )
        gate = read_json(gate_root / "gate_decision.json")
        if gate.get("delay_stage") != stage:
            raise RuntimeError("E3 gate reported a different delay stage")
        passed = bool(gate.get("passed"))
        results.append(
            {
                "order": index,
                "stage": stage,
                "passed": passed,
                "decision": gate.get("decision"),
                "campaign": str(stage_root),
                "gate": str(gate_root),
            }
        )
        if not passed:
            break
        selected_stage = stage

    status = {
        "status": "completed",
        "stage": "E3_SEQUENCE",
        "protocol": "bci2a_session_t_nested_six_fold_oof",
        "parent_e2": str(parent),
        "parent_variant": args.parent_variant,
        "registered_stages": [stage for stage, _ in STAGES],
        "executed_stages": [row["stage"] for row in results],
        "selected_delay_stage": selected_stage,
        "delay_promoted": selected_stage is not None,
        "stopped_after_first_failure": len(results) < len(STAGES),
        "results": results,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    write_json(output / "sequence_status.json", status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
