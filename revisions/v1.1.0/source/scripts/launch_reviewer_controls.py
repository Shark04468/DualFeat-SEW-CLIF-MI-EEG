"""Resume-safe dynamic GPU queue for reviewer-control subject workers."""

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
        "--config", default=str(ROOT / "configs" / "experiments" / "reviewer_controls.yaml")
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--checkpoint-barrier")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.phase == "evaluate" and not args.checkpoint_barrier:
        raise ValueError("evaluation launch requires --checkpoint-barrier")
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("at least one GPU is required")
    tasks = deque()
    for dataset, dataset_config in config["datasets"].items():
        subjects = [int(value) for value in dataset_config["subjects"]]
        if args.smoke:
            subjects = subjects[:1]
        tasks.extend((dataset, subject) for subject in subjects)
    output = Path(args.output).resolve()
    log_root = output / "logs" / args.phase
    log_root.mkdir(parents=True, exist_ok=True)
    active: dict[str, tuple[subprocess.Popen, object, tuple[str, int]]] = {}
    failures: list[tuple[str, int, int]] = []
    while tasks or active:
        for gpu in gpu_ids:
            if gpu in active or not tasks:
                continue
            dataset, subject = tasks.popleft()
            log = (log_root / f"{dataset}_subject_{subject:02d}_gpu{gpu}.log").open(
                "a", encoding="utf-8"
            )
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_reviewer_controls_subject.py"),
                "--phase",
                args.phase,
                "--dataset",
                dataset,
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
            active[gpu] = (process, log, (dataset, subject))
            print(f"started {args.phase} {dataset} subject {subject} on GPU {gpu}", flush=True)
        time.sleep(2.0)
        for gpu, (process, log, task) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            dataset, subject = task
            if code != 0:
                failures.append((dataset, subject, code))
                print(f"FAILED {dataset} subject {subject} on GPU {gpu}: exit {code}", flush=True)
            else:
                print(f"completed {dataset} subject {subject} on GPU {gpu}", flush=True)
            del active[gpu]
    if failures:
        raise RuntimeError(f"reviewer-control queue failures: {failures}")
    print("all reviewer-control subject jobs completed", flush=True)


if __name__ == "__main__":
    main()
