"""Dynamic multi-GPU queue for the BNCI2014-004 parent recovery."""

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


def _queue(args: argparse.Namespace, phase: str, subjects: list[int], gpus: list[str]) -> None:
    root = Path(args.output).resolve()
    log_root = root / "logs" / phase
    log_root.mkdir(parents=True, exist_ok=True)
    tasks = deque(subjects)
    active: dict[str, tuple[subprocess.Popen, object, int]] = {}
    failures: list[tuple[int, int]] = []
    while tasks or active:
        for gpu in gpus:
            if gpu in active or not tasks:
                continue
            subject = tasks.popleft()
            log = (log_root / f"subject_{subject:02d}_gpu{gpu}.log").open("a", encoding="utf-8")
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_v30_bnci2014_004_subject.py"),
                "--config",
                str(Path(args.config).resolve()),
                "--phase",
                phase,
                "--subject",
                str(subject),
                "--source-root",
                str(Path(args.source_root).resolve()),
                "--freeze",
                str(Path(args.freeze).resolve()),
                "--output",
                str(root),
                "--device",
                "cuda",
                "--feature-batch-size",
                str(args.feature_batch_size),
            ]
            if phase == "evaluate":
                command.extend(("--checkpoint-barrier", str(root / "checkpoint_barrier.json")))
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(
                command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
            active[gpu] = (process, log, subject)
            print(f"started V30 {phase} subject {subject} on GPU {gpu}", flush=True)
        time.sleep(2.0)
        for gpu, (process, log, subject) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            if code:
                failures.append((subject, code))
                print(f"FAILED V30 {phase} subject {subject}: exit {code}", flush=True)
            else:
                print(f"completed V30 {phase} subject {subject} on GPU {gpu}", flush=True)
            del active[gpu]
    if failures:
        raise RuntimeError(f"V30 queue failures: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "v30_bnci2014_004_blind.yaml"),
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--feature-batch-size", type=int, default=96)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    subjects = [int(value) for value in config["dataset"]["subjects"]]
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    root = Path(args.output).resolve()
    barrier = root / "checkpoint_barrier.json"
    if args.phase in ("train", "all"):
        _queue(args, "train", subjects, gpus)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "seal_v30_checkpoints.py"),
                "--config",
                str(Path(args.config).resolve()),
                "--root",
                str(root),
                "--freeze",
                str(Path(args.freeze).resolve()),
                "--output",
                str(barrier),
            ],
            cwd=ROOT,
            check=True,
        )
    if args.phase in ("evaluate", "all"):
        _queue(args, "evaluate", subjects, gpus)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "aggregate_v30_bnci2014_004.py"),
                "--config",
                str(Path(args.config).resolve()),
                "--root",
                str(root),
                "--output",
                str(root / "aggregate"),
                "--bootstrap-samples",
                str(args.bootstrap_samples),
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
