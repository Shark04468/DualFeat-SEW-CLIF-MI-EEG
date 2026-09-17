"""Dynamic multi-GPU queue for BCI2a/OpenBMI frozen-front-end recovery."""

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


def _base(args: argparse.Namespace, phase: str) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_v8_publication_baselines.py"),
        "--config",
        str(Path(args.config).resolve()),
        "--dataset",
        args.dataset,
        "--phase",
        phase,
        "--output-root",
        str(Path(args.output_root).resolve()),
        "--source-root",
        str(Path(args.source_root).resolve()),
        "--bci2a-train-root",
        str(Path(args.bci2a_train_root).resolve()),
        "--bci2a-eval-root",
        str(Path(args.bci2a_eval_root).resolve()),
        "--storage-root",
        str(Path(args.storage_root).resolve()),
        "--subjects",
        args.subjects,
        "--models",
        args.models,
        "--seeds",
        args.seeds,
        "--device",
        "cuda",
    ]
    if args.fixed_epochs:
        command.extend(("--fixed-epochs", str(args.fixed_epochs)))
    if args.canary:
        command.append("--canary")
    return command


def _run_queue(
    args: argparse.Namespace, phase: str, subjects: list[int], gpu_ids: list[str]
) -> None:
    output = Path(args.output_root).resolve()
    log_root = output / "launcher_logs" / args.dataset / phase
    log_root.mkdir(parents=True, exist_ok=True)
    tasks = deque(subjects)
    active: dict[str, tuple[subprocess.Popen, object, int]] = {}
    failures: list[tuple[int, int]] = []
    while tasks or active:
        for gpu in gpu_ids:
            if gpu in active or not tasks:
                continue
            subject = tasks.popleft()
            log = (log_root / f"subject_{subject:02d}_gpu{gpu}.log").open("a", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(
                [*_base(args, phase), "--worker-subject", str(subject)],
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            active[gpu] = (process, log, subject)
            print(f"started {args.dataset} {phase} subject {subject} on GPU {gpu}", flush=True)
        time.sleep(2.0)
        for gpu, (process, log, subject) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            if code:
                failures.append((subject, code))
                print(f"FAILED {args.dataset} {phase} subject {subject}: exit {code}", flush=True)
            else:
                print(
                    f"completed {args.dataset} {phase} subject {subject} on GPU {gpu}", flush=True
                )
            del active[gpu]
    if failures:
        raise RuntimeError(f"publication queue failures: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("bci2a", "openbmi"), required=True)
    parser.add_argument("--phase", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "experiments" / "v8_publication_baselines.yaml")
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--storage-root", required=True)
    parser.add_argument("--bci2a-train-root", required=True)
    parser.add_argument("--bci2a-eval-root", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--subjects", default="")
    parser.add_argument("--models", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--fixed-epochs", type=int, default=0)
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    dataset_config = config["datasets"][args.dataset]
    subjects = (
        [int(value) for value in args.subjects.split(",") if value]
        if args.subjects
        else [int(value) for value in dataset_config["subjects"]]
    )
    if args.canary and not args.subjects:
        subjects = [int(value) for value in config["canary"][f"{args.dataset}_subjects"]]
    args.subjects = ",".join(str(value) for value in subjects)
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("at least one GPU is required")
    if args.phase in ("train", "all"):
        subprocess.run(_base(args, "init"), cwd=ROOT, check=True)
        _run_queue(args, "train", subjects, gpu_ids)
        subprocess.run(_base(args, "seal"), cwd=ROOT, check=True)
    if args.phase in ("evaluate", "all"):
        _run_queue(args, "evaluate", subjects, gpu_ids)
        subprocess.run(_base(args, "aggregate"), cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
