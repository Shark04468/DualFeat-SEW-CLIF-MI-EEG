"""Dynamic multi-GPU queue for the recovered V31 parent campaign."""

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


def _tasks(config: dict) -> deque[tuple[str, int]]:
    return deque(
        (dataset, int(subject))
        for dataset, dataset_config in config["datasets"].items()
        for subject in dataset_config["subjects"]
    )


def _queue(args: argparse.Namespace, config: dict, phase: str, gpus: list[str]) -> None:
    tasks = _tasks(config)
    output = Path(args.output).resolve()
    log_root = output / "logs" / phase
    log_root.mkdir(parents=True, exist_ok=True)
    active: dict[str, tuple[subprocess.Popen, object, tuple[str, int]]] = {}
    failures: list[tuple[str, int, int]] = []
    while tasks or active:
        for gpu in gpus:
            if gpu in active or not tasks:
                continue
            dataset, subject = tasks.popleft()
            log = (log_root / f"{dataset}_subject_{subject:02d}_gpu{gpu}.log").open(
                "a", encoding="utf-8"
            )
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_v31_decoder_learning_curve_subject.py"),
                "--config",
                str(Path(args.config).resolve()),
                "--phase",
                phase,
                "--dataset",
                dataset,
                "--subject",
                str(subject),
                "--publication-root",
                str(Path(args.publication_root).resolve()),
                "--v30-root",
                str(Path(args.v30_root).resolve()),
                "--source-root",
                str(Path(args.source_root).resolve()),
                "--output",
                str(output),
                "--bci2a-train-root",
                str(Path(args.bci2a_train_root).resolve()),
                "--bci2a-eval-root",
                str(Path(args.bci2a_eval_root).resolve()),
                "--recovery-label-root",
                str(Path(args.recovery_label_root).resolve()),
                "--device",
                "cuda",
                "--feature-batch-size",
                str(args.feature_batch_size),
            ]
            if phase == "evaluate":
                command.extend(("--checkpoint-barrier", str(output / "checkpoint_barrier.json")))
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(
                command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
            active[gpu] = (process, log, (dataset, subject))
            print(f"started V31 {phase} {dataset} subject {subject} on GPU {gpu}", flush=True)
        time.sleep(2.0)
        for gpu, (process, log, task) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            dataset, subject = task
            if code:
                failures.append((dataset, subject, code))
                print(f"FAILED V31 {phase} {dataset} subject {subject}: exit {code}", flush=True)
            else:
                print(f"completed V31 {phase} {dataset} subject {subject} on GPU {gpu}", flush=True)
            del active[gpu]
    if failures:
        raise RuntimeError(f"V31 queue failures: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "v31_decoder_learning_curve.yaml"),
    )
    parser.add_argument("--publication-root", required=True)
    parser.add_argument("--v30-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bci2a-train-root", required=True)
    parser.add_argument("--bci2a-eval-root", required=True)
    parser.add_argument("--recovery-label-root", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--feature-batch-size", type=int, default=96)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    output = Path(args.output).resolve()
    barrier = output / "checkpoint_barrier.json"
    if args.phase in ("train", "all"):
        _queue(args, config, "train", gpus)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "seal_v31_learning_curve.py"),
                "--config",
                str(config_path),
                "--output",
                str(output),
                "--barrier",
                str(barrier),
            ],
            cwd=ROOT,
            check=True,
        )
    if args.phase in ("evaluate", "all"):
        _queue(args, config, "evaluate", gpus)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "aggregate_v31_learning_curve.py"),
                "--config",
                str(config_path),
                "--input",
                str(output),
                "--output",
                str(output / "aggregate"),
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
