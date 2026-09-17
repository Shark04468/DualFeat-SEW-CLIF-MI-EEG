"""Resume-safe dynamic GPU queue for REV-E5 BCI2a binary workers."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "evaluate"), required=True)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "bci2a_binary_sensitivity.yaml"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--checkpoint-barrier")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.phase == "evaluate" and not args.checkpoint_barrier:
        raise ValueError("REV-E5 evaluation launch requires --checkpoint-barrier")
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("at least one GPU is required")
    subjects = [int(value) for value in config["dataset"]["subjects"]]
    if args.smoke:
        subjects = subjects[:1]
    tasks = deque(subjects)
    output = Path(args.output).resolve()
    log_root = output / "logs" / args.phase
    log_root.mkdir(parents=True, exist_ok=True)
    active: dict[str, tuple[subprocess.Popen, object, int]] = {}
    failures: list[tuple[int, int]] = []
    while tasks or active:
        for gpu in gpu_ids:
            if gpu in active or not tasks:
                continue
            subject = tasks.popleft()
            log = (log_root / f"subject_{subject:02d}_gpu{gpu}.log").open("a", encoding="utf-8")
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_bci2a_binary_sensitivity_subject.py"),
                "--phase",
                args.phase,
                "--subject",
                str(subject),
                "--config",
                str(config_path),
                "--output",
                str(output),
                "--device",
                "cuda",
            ]
            if args.checkpoint_barrier:
                command.extend(("--checkpoint-barrier", args.checkpoint_barrier))
            if args.smoke:
                command.append("--smoke")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(
                command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
            active[gpu] = (process, log, subject)
            print(f"started {args.phase} REV-E5 subject {subject} on GPU {gpu}", flush=True)
        time.sleep(2.0)
        for gpu, (process, log, subject) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            if code:
                failures.append((subject, code))
                print(f"FAILED REV-E5 subject {subject} on GPU {gpu}: exit {code}", flush=True)
            else:
                print(f"completed REV-E5 subject {subject} on GPU {gpu}", flush=True)
            del active[gpu]
    if failures:
        raise RuntimeError(f"REV-E5 queue failures: {failures}")
    print("all REV-E5 subject jobs completed", flush=True)


if __name__ == "__main__":
    main()
