#!/usr/bin/env python
"""Run paired real-EEG full/zero/within-band delay pilots."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.config import load_experiment_config  # noqa: E402
from dpc_snn.data.bci2a import load_processed_npz, subject_session_data  # noqa: E402
from dpc_snn.experiments.common import resolve_dataset_cfg, resolve_model_cfg  # noqa: E402
from dpc_snn.experiments.runners import _run_train_test_model, _run_train_validation_model  # noqa: E402
from dpc_snn.utils.io import ensure_dir, write_csv, write_json  # noqa: E402


VARIANTS: dict[str, dict[str, Any]] = {
    "full": {},
    "zero_delay_matched": {"force_zero_delay": True},
    "within_band_delay": {"use_cross_band_routes": False},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/all_experiments.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--subjects", default="1,3,8")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--delay-pretrain-epochs", type=int, default=3)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--shared-joint-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--delay-pretrain-gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--protocol-version", default="v39")
    parser.add_argument(
        "--evaluation-role",
        choices=("development", "confirmatory"),
        default="development",
    )
    parser.add_argument("--frozen-architecture-id", default="")
    parser.add_argument(
        "--evidence-audit-path",
        default="",
        help="Optional global diagnostic audit; the model prior is always recomputed on inner train.",
    )
    parser.add_argument("--evidence-space", default="csd", choices=("csd",))
    parser.add_argument(
        "--shared-only",
        action="store_true",
        help="Train only the common Subject/seed checkpoints; do not run paired variants.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Retrain variants even when a complete metrics.json already exists.",
    )
    return parser.parse_args()


def _result_row(
    metrics: dict[str, Any],
    subject: str,
    seed: int,
    variant: str,
    protocol: str = "v35_paired_T_to_E_pilot",
    evaluated_on_heldout_test: bool = True,
    evaluation_split: str = "heldout_session_E",
    evidence_role: str = "development",
) -> dict[str, Any]:
    return {
        **metrics,
        "dataset": "bci2a",
        "protocol": protocol,
        "subject": subject,
        "seed": seed,
        "model": "dpc_snn",
        "variant": variant,
        "status": "completed",
        "evaluated_on_heldout_test": evaluated_on_heldout_test,
        "evaluation_split": evaluation_split,
        "evidence_role": evidence_role,
    }


def _load_existing_result(
    output: Path,
    subject: str,
    seed: int,
    variant: str,
    protocol: str = "v35_paired_T_to_E_pilot",
) -> dict[str, Any] | None:
    metrics_path = (
        output / f"subject_{subject}" / f"seed_{seed}" / variant / "dpc_snn" / "metrics.json"
    )
    if not metrics_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    required = {"accuracy", "balanced_accuracy", "kappa", "macro_f1"}
    if not required.issubset(metrics):
        return None
    is_shared = variant == "shared_representation"
    if is_shared and metrics.get("evaluation_split") != "validation":
        # Never relabel a historical shared checkpoint that selected on E as a
        # Session-T-only checkpoint.
        return None
    return _result_row(
        metrics,
        subject,
        seed,
        variant,
        protocol,
        evaluated_on_heldout_test=not is_shared,
        evaluation_split=("session_T_inner_validation" if is_shared else "heldout_session_E"),
        evidence_role=("training_selection_only" if is_shared else "development"),
    )


def main() -> None:
    args = parse_args()
    audit_path = Path(args.evidence_audit_path) if args.evidence_audit_path else None
    if audit_path is not None and not (audit_path / "evidence_gate.json").exists():
        raise FileNotFoundError(
            f"Missing optional diagnostic audit: {audit_path / 'evidence_gate.json'}"
        )
    subjects = [value.strip() for value in args.subjects.split(",") if value.strip()]
    if args.evaluation_role == "confirmatory":
        if "1" in subjects or "A01" in subjects:
            raise ValueError("Subject 1 Session E is development evidence and cannot be confirmatory.")
        if not args.frozen_architecture_id.strip():
            raise ValueError("Confirmatory evaluation requires --frozen-architecture-id.")
    paired_protocol = f"{args.protocol_version}_{args.evaluation_role}_paired_T_to_E"
    shared_protocol = f"{args.protocol_version}_shared_session_T_inner_validation"
    output = ensure_dir(args.output)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    cfg = load_experiment_config(args.config, "E14", overrides={"device": args.device})
    cfg["experiment_id"] = f"{args.protocol_version.upper()}_PILOT"
    cfg["training"] = {
        **cfg.get("training", {}),
        "epochs": args.epochs,
        "delay_pretrain_epochs": args.delay_pretrain_epochs,
        "representation_warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "delay_pretrain_gradient_accumulation_steps": (
            args.delay_pretrain_gradient_accumulation_steps
        ),
    }
    dataset_cfg = resolve_dataset_cfg(cfg, key="bci2a_config")
    data = load_processed_npz(dataset_cfg["root"])
    rows: list[dict[str, Any]] = []
    shared_rows: list[dict[str, Any]] = []
    for subject in subjects:
        train_raw = subject_session_data(data, subject, session="T")
        for seed in seeds:
            shared_dir = output / f"subject_{subject}" / f"seed_{seed}" / "shared_representation"
            shared_checkpoint = shared_dir / "dpc_snn" / "model_checkpoint.pt"
            joint_epochs = max(1, args.epochs - args.warmup_epochs)
            existing_shared = None if args.rerun else _load_existing_result(
                output,
                subject,
                seed,
                "shared_representation",
                shared_protocol,
            )
            if not shared_checkpoint.exists() or existing_shared is None:
                shared_cfg = copy.deepcopy(cfg)
                shared_cfg["seed"] = seed
                shared_cfg["training"] = {
                    **shared_cfg.get("training", {}),
                    "epochs": max(1, args.warmup_epochs + args.shared_joint_epochs),
                    "delay_pretrain_epochs": args.delay_pretrain_epochs,
                    "representation_warmup_epochs": max(1, args.warmup_epochs),
                    "select_last_checkpoint": False,
                    "tet_loss_weight": 0.0,
                    "minimum_lag_anneal_joint_epochs": 10,
                }
                print(
                    f'__DS_PROGRESS__ {{"subject":"{subject}","seed":{seed},"variant":"shared_representation","status":"started"}}',
                    flush=True,
                )
                shared_metric = _run_train_validation_model(
                    "dpc_snn",
                    train_raw,
                    shared_cfg,
                    shared_dir,
                    dataset_name="bci2a",
                    protocol=shared_protocol,
                    subject=subject,
                    model_cfg={
                        **resolve_model_cfg(shared_cfg, "dpc_snn"),
                        "physical_reference": args.evidence_space,
                        "fold_local_evidence_prior": True,
                    },
                    evaluation_split="session_T_inner_validation",
                    variant="shared_representation",
                    evidence_role="training_selection_only",
                    evidence_space=args.evidence_space,
                    evidence_audit_path="inner_train_recomputed",
                )
                shared_rows.append(
                    _result_row(
                        shared_metric,
                        subject,
                        seed,
                        "shared_representation",
                        shared_protocol,
                        evaluated_on_heldout_test=False,
                        evaluation_split="session_T_inner_validation",
                        evidence_role="training_selection_only",
                    )
                )
                print(
                    f'__DS_PROGRESS__ {{"subject":"{subject}","seed":{seed},"variant":"shared_representation","status":"completed"}}',
                    flush=True,
                )
            else:
                shared_rows.append(existing_shared)
            write_csv(output / "shared_checkpoint_results.csv", shared_rows)
            if args.shared_only:
                continue
            test_raw = subject_session_data(data, subject, session="E")
            for variant, override in VARIANTS.items():
                existing = None if args.rerun else _load_existing_result(
                    output, subject, seed, variant, paired_protocol
                )
                if existing is not None:
                    rows.append(existing)
                    write_csv(output / "pilot_results.csv", rows)
                    print(
                        f'__DS_PROGRESS__ {{"subject":"{subject}","seed":{seed},"variant":"{variant}","status":"recovered"}}',
                        flush=True,
                    )
                    continue
                print(
                    f'__DS_PROGRESS__ {{"subject":"{subject}","seed":{seed},"variant":"{variant}","status":"started"}}',
                    flush=True,
                )
                run_cfg = copy.deepcopy(cfg)
                run_cfg["seed"] = seed
                run_cfg["training"] = {
                    **run_cfg.get("training", {}),
                    "epochs": joint_epochs,
                    "delay_pretrain_epochs": 0,
                    "representation_warmup_epochs": 0,
                    "initial_checkpoint": str(shared_checkpoint),
                    "initial_checkpoint_role": "common_delay_prior_and_representation",
                }
                model_cfg = resolve_model_cfg(run_cfg, "dpc_snn")
                model_cfg["physical_reference"] = args.evidence_space
                model_cfg["fold_local_evidence_prior"] = True
                model_cfg.update(override)
                try:
                    metric = _run_train_test_model(
                        "dpc_snn",
                        train_raw,
                        test_raw,
                        run_cfg,
                        output / f"subject_{subject}" / f"seed_{seed}" / variant,
                        dataset_name="bci2a",
                        protocol=paired_protocol,
                        subject=subject,
                        model_cfg=model_cfg,
                        evaluation_split="heldout_session_E",
                        variant=variant,
                        evidence_role=args.evaluation_role,
                        frozen_architecture_id=args.frozen_architecture_id,
                        evidence_space=args.evidence_space,
                        evidence_audit_path="inner_train_recomputed",
                    )
                    rows.append(
                        _result_row(
                            metric,
                            subject,
                            seed,
                            variant,
                            paired_protocol,
                            evidence_role=args.evaluation_role,
                        )
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "dataset": "bci2a",
                            "protocol": paired_protocol,
                            "subject": subject,
                            "seed": seed,
                            "model": "dpc_snn",
                            "variant": variant,
                            "status": "failed",
                            "error": repr(exc),
                        }
                    )
                write_csv(output / "pilot_results.csv", rows)
                print(
                    f'__DS_PROGRESS__ {{"subject":"{subject}","seed":{seed},"variant":"{variant}","status":"{rows[-1]["status"]}"}}',
                    flush=True,
                )
    expected_shared = len(subjects) * len(seeds)
    completed = (
        len(shared_rows) == expected_shared
        if args.shared_only
        else bool(rows) and all(row["status"] == "completed" for row in rows)
    )
    write_json(
        output / "run_manifest.json",
        {
            "status": "completed" if completed else "partial",
            "architecture_version": resolve_model_cfg(cfg, "dpc_snn").get(
                "architecture_version", "unknown"
            ),
            "subjects": subjects,
            "seeds": seeds,
            "variants": VARIANTS,
            "epochs": args.epochs,
            "delay_pretrain_epochs": args.delay_pretrain_epochs,
            "representation_warmup_epochs": args.warmup_epochs,
            "shared_joint_epochs": args.shared_joint_epochs,
            "joint_epochs_per_variant": max(1, args.epochs - args.warmup_epochs),
            "physical_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "delay_pretrain_gradient_accumulation_steps": (
                args.delay_pretrain_gradient_accumulation_steps
            ),
            "protocol": "shared_selection_on_session_T_only_then_explicit_session_E_evaluation",
            "evaluation_role": args.evaluation_role,
            "subject_1_session_E_role": "development_only",
            "frozen_architecture_id": args.frozen_architecture_id,
            "diagnostic_evidence_audit_path": "" if audit_path is None else str(audit_path),
            "evidence_audit_path": "inner_train_recomputed_per_seed",
            "selected_evidence_space": args.evidence_space,
            "zero_delay_capacity_matched": True,
            "shared_representation_checkpoint": True,
            "shared_only": args.shared_only,
            "paired_variants_executed": [] if args.shared_only else list(VARIANTS),
            "joint_optimizer_steps_matched": True,
        },
    )
    print(
        f"completed shared_rows={len(shared_rows)} paired_rows={len(rows)} output={output}"
    )


if __name__ == "__main__":
    main()
