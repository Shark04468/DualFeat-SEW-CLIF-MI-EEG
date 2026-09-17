#!/usr/bin/env python
"""Run E0-E24 sequentially with per-experiment logs and a summary table."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "unreadable", "error": repr(exc)}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _experiment_ids() -> list[str]:
    return [f"E{i}" for i in range(25)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/all_experiments.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--log-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    log_root = Path(args.log_root) if args.log_root else output_root / "_logs"
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    experiments = args.experiments or _experiment_ids()
    summary_rows: list[dict[str, Any]] = []
    run_started = datetime.now(timezone.utc).isoformat()
    print(json.dumps({"event": "full_run_start", "output_root": str(output_root), "experiments": experiments, "started_utc": run_started}), flush=True)

    for exp_id in experiments:
        start = time.perf_counter()
        exp_output = output_root / exp_id
        log_path = log_root / f"{exp_id}.log"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_experiment.py"),
            "--config",
            args.config,
            "--experiment",
            exp_id,
            "--output",
            str(exp_output),
        ]
        if args.device:
            cmd.extend(["--device", args.device])
        if args.epochs is not None:
            cmd.extend(["--epochs", str(args.epochs)])
        if args.batch_size is not None:
            cmd.extend(["--batch-size", str(args.batch_size)])

        print(json.dumps({"event": "experiment_start", "experiment": exp_id, "output": str(exp_output), "log": str(log_path)}), flush=True)
        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        elapsed = time.perf_counter() - start
        manifest = _read_json(exp_output / "run_manifest.json")
        runner_status = _read_json(exp_output / "runner_status.json")
        status = runner_status.get("status") or manifest.get("status") or ("completed" if proc.returncode == 0 else "failed")
        message = runner_status.get("message") or manifest.get("message") or ""
        row = {
            "experiment": exp_id,
            "status": status,
            "returncode": proc.returncode,
            "elapsed_seconds": round(elapsed, 3),
            "output": str(exp_output),
            "log": str(log_path),
            "message": message,
        }
        summary_rows.append(row)
        _write_csv(output_root / "run_summary.csv", summary_rows)
        (output_root / "run_summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"event": "experiment_end", **row}), flush=True)
        if proc.returncode != 0 and not args.continue_on_error:
            break

    failures = [row for row in summary_rows if str(row.get("status", "")).lower() not in {"completed"} or int(row.get("returncode", 0)) != 0]
    failure_text = "# Full Run Failures\n\n"
    if failures:
        for row in failures:
            failure_text += f"- {row['experiment']}: status={row['status']} returncode={row['returncode']} message={row['message']} log={row['log']}\n"
    else:
        failure_text += "No non-completed experiments recorded by the orchestrator.\n"
    (output_root / "orchestrator_failures.md").write_text(failure_text, encoding="utf-8")
    print(json.dumps({"event": "full_run_end", "output_root": str(output_root), "n_experiments": len(summary_rows), "n_failures": len(failures)}), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
