#!/usr/bin/env python
"""Run ordered V3.9 Subject-T ablations without touching Session E."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import os
from pathlib import Path
import signal
import sys
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.config import load_experiment_config, load_yaml  # noqa: E402
from dpc_snn.data.bci2a import load_processed_npz, subject_session_data  # noqa: E402
from dpc_snn.experiments.common import resolve_dataset_cfg, resolve_model_cfg  # noqa: E402
from dpc_snn.experiments.runners import _run_train_validation_model  # noqa: E402
from dpc_snn.utils.io import ensure_dir, write_csv, write_json  # noqa: E402


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    paths = sorted((ROOT / "src").rglob("*.py"))
    paths.extend(
        [
            ROOT / "scripts" / "run_v39_single_variable_ablation.py",
            ROOT / "configs" / "models" / "dpc_snn.yaml",
            ROOT / "configs" / "experiments" / "v39_single_variable_ablation.yaml",
        ]
    )
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/all_experiments.yaml")
    parser.add_argument(
        "--ablation-config",
        default="configs/experiments/v39_single_variable_ablation.yaml",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--subjects", default="1")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--representation-epochs", type=int, default=6)
    parser.add_argument("--delay-pretrain-epochs", type=int, default=8)
    # Batch 2 with accumulation 16 preserves the effective batch of 32 while
    # keeping the learned-hurdle stage below a 32 GiB device boundary.
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    stages = load_yaml(args.ablation_config)["stages"]
    output = ensure_dir(args.output)
    source_fingerprint = _source_fingerprint()
    manifest_path = output / "run_manifest.json"
    manifest_base = {
        "protocol": "v39_session_T_ordered_single_variable_ablation",
        "subjects": args.subjects,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "representation_epochs": args.representation_epochs,
        "delay_pretrain_epochs": args.delay_pretrain_epochs,
        "source_fingerprint": source_fingerprint,
        "heldout_session_E_accessed": False,
    }
    write_json(manifest_path, {**manifest_base, "status": "running"})

    def record_failure(exc_type, exc, tb) -> None:
        write_json(
            manifest_path,
            {
                **manifest_base,
                "status": "failed",
                "error_type": exc_type.__name__,
                "error": str(exc),
                "traceback": "".join(traceback.format_exception(exc_type, exc, tb)),
            },
        )
        sys.__excepthook__(exc_type, exc, tb)

    def record_interrupt(signum, _frame) -> None:
        write_json(
            manifest_path,
            {
                **manifest_base,
                "status": "interrupted",
                "signal": int(signum),
            },
        )
        raise SystemExit(128 + int(signum))

    sys.excepthook = record_failure
    signal.signal(signal.SIGTERM, record_interrupt)
    signal.signal(signal.SIGINT, record_interrupt)
    subjects = [value.strip() for value in args.subjects.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    cfg = load_experiment_config(args.config, "E14", overrides={"device": args.device})
    dataset_cfg = resolve_dataset_cfg(cfg, key="bci2a_config")
    data = load_processed_npz(dataset_cfg["root"])
    rows = []
    for subject in subjects:
        session_t = subject_session_data(data, subject, session="T")
        for seed in seeds:
            for stage_index, stage in enumerate(stages):
                run_cfg = copy.deepcopy(cfg)
                run_cfg["seed"] = seed
                run_cfg["experiment_id"] = "V39_SINGLE_VARIABLE_ABLATION"
                learned_delay = not bool(stage["freeze_delay_posterior"])
                run_cfg["training"] = {
                    **run_cfg.get("training", {}),
                    "epochs": args.epochs,
                    "representation_warmup_epochs": (
                        args.representation_epochs if learned_delay else 0
                    ),
                    "delay_pretrain_epochs": args.delay_pretrain_epochs if learned_delay else 0,
                    "batch_size": args.batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "tet_loss_weight": 0.0,
                    "select_last_checkpoint": False,
                    "minimum_lag_anneal_joint_epochs": 10 if learned_delay else 0,
                }
                model_cfg = resolve_model_cfg(run_cfg, "dpc_snn")
                model_cfg.update(stage)
                model_cfg.update(
                    {
                        "baseline_tmin": -1.0,
                        "physical_reference": "csd",
                        "euclidean_alignment": False,
                        "fold_local_evidence_prior": True,
                        "cumulative_readout_seconds": [],
                        "fold_local_evidence_prior_path": str(
                            output / "fold_priors" / f"subject_{subject}_seed_{seed}.npz"
                        ),
                    }
                )
                stage_name = str(stage["name"])
                result = _run_train_validation_model(
                    "dpc_snn",
                    session_t,
                    run_cfg,
                    output / f"subject_{subject}" / f"seed_{seed}" / stage_name,
                    dataset_name="bci2a",
                    protocol="v39_session_T_ordered_single_variable_ablation",
                    subject=subject,
                    model_cfg=model_cfg,
                    evaluation_split="session_T_inner_validation",
                    variant=stage_name,
                    evidence_role="inner_train_prior_and_validation_only",
                    evidence_space="csd",
                    evidence_audit_path="inner_train_recomputed",
                )
                rows.append(
                    {
                        **result,
                        "status": "completed",
                        "subject": subject,
                        "seed": seed,
                        "stage_index": stage_index,
                        "stage": stage_name,
                        "heldout_session_E_accessed": False,
                    }
                )
                write_csv(output / "ablation_results.csv", rows)
                # Each stage builds a large, independent delay graph. Release
                # allocator reservations before constructing the next model.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    write_json(
        output / "run_manifest.json",
        {
            **manifest_base,
            "status": "completed",
            "subjects": subjects,
            "seeds": seeds,
            "stages": stages,
        },
    )


if __name__ == "__main__":
    main()
