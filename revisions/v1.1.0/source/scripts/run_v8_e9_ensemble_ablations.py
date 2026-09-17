#!/usr/bin/env python3
"""Run and aggregate the frozen V8 ensemble ablation campaign."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.analysis.v8_utility import multiclass_calibration_metrics  # noqa: E402
from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v62_baselines import task_carrier  # noqa: E402
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
    write_trial_predictions,
)
from dpc_snn.experiments.v8_ensemble import (  # noqa: E402
    validate_ensemble_freeze_contract,
)
from dpc_snn.experiments.v8_ensemble_followup import (  # noqa: E402
    component_state_digests,
    fit_ensemble_components,
    predict_ensemble_from_carrier,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    build_v8_run_fingerprint,
    collect_source_tree_manifest,
    source_tree_digest,
    validate_v8_freeze_manifest,
    validate_v8_resume_fingerprint,
    write_v8_fingerprint,
)
from dpc_snn.experiments.v8_statistics import (  # noqa: E402
    pair_subject_seed_rows,
    paired_delta_summary,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v8_e1_baselines import _environment  # noqa: E402
from scripts.run_v8_e2_zero_delay import _subject_file  # noqa: E402
from scripts.run_v8_e6_ensemble_frozen import (  # noqa: E402
    RUN_FILES as E6_RUN_FILES,
    _metadata,
)


ARMS = (
    "primary",
    "matched_ann",
    "anchor",
    "atcnet",
    "fbcnet",
    "decoder_snn",
    "decoder_ann",
)
PREDICTION_NAMES = {
    "primary": "predictions",
    "matched_ann": "matched_ann_predictions",
    "anchor": "anchor_predictions",
    "atcnet": "atcnet_predictions",
    "fbcnet": "fbcnet_predictions",
    "decoder_snn": "decoder_snn_predictions",
    "decoder_ann": "decoder_ann_predictions",
}
RUN_FILES = (
    "manifest.json",
    "source_fingerprint.json",
    "source_tree_manifest.json",
    "resolved_config.yaml",
    "data_identity.json",
    "data_access_manifest.json",
    "gain.npz",
    "atcnet_history.csv",
    "fbcnet_history.csv",
    "sew_clif_history.csv",
    "ann_sew_history.csv",
    "atcnet.pt",
    "fbcnet.pt",
    "sew_clif.pt",
    "ann_sew.pt",
    "state_audit.json",
    "metrics.json",
    "runtime_status.json",
) + tuple(
    f"{basename}.{suffix}"
    for basename in PREDICTION_NAMES.values()
    for suffix in ("npz", "csv")
)
CAMPAIGN_FILES = (
    "manifest.json",
    "campaign_status.json",
    "retrained_summary.csv",
    "reused_controls.csv",
    "paired_subject_seed.csv",
    "comparisons.json",
    "evidence_links.json",
    "source_tree_manifest.json",
    "resolved_campaign.yaml",
)


def _csv_values(value: str, cast: Any = str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _save_fit(run_dir: Path, fit: Any) -> None:
    for name, model in fit.components.as_dict().items():
        torch.save(
            {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            run_dir / f"{name}.pt",
        )
        write_csv(run_dir / f"{name}_history.csv", fit.histories[name])
    np.savez_compressed(
        run_dir / "gain.npz",
        values=np.asarray(fit.gain.values, dtype=np.float32),
        clip=np.asarray(fit.gain.clip, dtype=np.float32),
    )


def _run_one(
    *,
    output: Path,
    train_path: Path,
    evaluation_path: Path,
    e6: Path,
    source_root: Path,
    freeze: dict[str, Any],
    config: dict[str, Any],
    source_tree: dict[str, str],
    environment: dict[str, Any],
    subject: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    e6_run = e6 / f"subject_{subject:02d}" / f"seed_{seed}"
    validate_run_artifact_manifest(
        e6_run,
        required_files=E6_RUN_FILES,
        verify_hashes=True,
        verify_prediction_schema=True,
    )
    e6_metrics = read_json(e6_run / "metrics.json")
    model_config = dict(freeze["architecture"]["model_config"])
    component_rule = dict(freeze["checkpoint_rule"]["components"])
    augmentation = {**dict(freeze["augmentation"]), "enabled": False}
    run_seed = int(seed) * 100_003 + int(subject) * 1_009
    decoder_seed = run_seed + 8_000_003
    resolved = {
        "stage": "E9_ENSEMBLE_ABLATION",
        "variant": "no_augmentation",
        "subject": int(subject),
        "seed": int(seed),
        "run_seed": run_seed,
        "decoder_seed": decoder_seed,
        "architecture": model_config,
        "component_rule": component_rule,
        "preprocessing": freeze["preprocessing"],
        "augmentation": augmentation,
        "reference_e6_run_fingerprint": e6_metrics["run_fingerprint"],
    }
    data_hashes = {
        f"session_t/{train_path.name}": file_sha256(train_path),
        f"session_e/{evaluation_path.name}": file_sha256(evaluation_path),
    }
    fingerprint = build_v8_run_fingerprint(
        resolved_run_config=resolved,
        source_tree=source_tree,
        data=data_hashes,
        split={
            "train_session": "T",
            "evaluation_session": "E",
            "evaluation_labels_used_for_checkpoint_selection": False,
        },
        augmentation=augmentation,
        prior={"policy": "none", "delay_enabled": False},
        checkpoint={
            "components": component_rule,
            "session_e_checkpoint_selection": False,
            "reference_e6_run_fingerprint": e6_metrics["run_fingerprint"],
        },
        environment=environment,
    )
    run_dir = ensure_dir(
        output / "no_augmentation" / f"subject_{subject:02d}" / f"seed_{seed}"
    )
    fingerprint_path = run_dir / "source_fingerprint.json"
    if (run_dir / "manifest.json").is_file():
        validate_v8_resume_fingerprint(fingerprint_path, fingerprint)
        validate_run_artifact_manifest(
            run_dir,
            required_files=RUN_FILES,
            verify_hashes=True,
            verify_prediction_schema=True,
        )
        return read_json(run_dir / "metrics.json")
    if fingerprint_path.is_file():
        validate_v8_resume_fingerprint(fingerprint_path, fingerprint)
    else:
        write_v8_fingerprint(fingerprint_path, fingerprint)
    write_json(run_dir / "source_tree_manifest.json", source_tree)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    write_json(
        run_dir / "runtime_status.json",
        {
            "status": "training_session_t",
            "started_at": time.time(),
            "session_e_arrays_loaded": False,
        },
    )

    train_data = load_processed_npz(train_path)
    metadata_t = _metadata(train_data, expected_session="T", role="training")
    sfreq = float(np.asarray(train_data["sfreq"]).item())
    epoch_tmin = float(np.asarray(train_data["epoch_tmin"]).item())
    fit = fit_ensemble_components(
        np.asarray(train_data["X"], dtype=np.float32),
        np.asarray(train_data["y"], dtype=np.int64),
        sfreq=sfreq,
        epoch_tmin=epoch_tmin,
        source_root=source_root,
        model_config=model_config,
        component_rule=component_rule,
        preprocessing=dict(freeze["preprocessing"]),
        augmentation=augmentation,
        n_classes=4,
        run_seed=run_seed,
        decoder_seed=decoder_seed,
        device=device,
        run_label=f"V8-E9-no-augmentation:S{subject}:seed{seed}",
    )
    before = component_state_digests(fit.components)
    _save_fit(run_dir, fit)
    write_json(
        run_dir / "runtime_status.json",
        {
            "status": "checkpointed_before_session_e",
            "checkpointed_at": time.time(),
            "session_e_arrays_loaded": False,
            "state_sha256": before,
        },
    )

    evaluation_data = load_processed_npz(evaluation_path)
    metadata_e = _metadata(evaluation_data, expected_session="E", role="evaluation")
    labels_e = np.asarray(evaluation_data["y"], dtype=np.int64)
    carrier_e = task_carrier(
        np.asarray(evaluation_data["X"], dtype=np.float32),
        sfreq=sfreq,
        epoch_tmin=epoch_tmin,
    )
    prediction = predict_ensemble_from_carrier(
        fit.components,
        carrier_e,
        labels_e,
        fit.gain,
        sfreq=sfreq,
        model_config=model_config,
        n_classes=4,
        device=device,
    )
    after = component_state_digests(fit.components)
    if before != after:
        raise RuntimeError("E9 model state changed during Session-E evaluation")
    for arm in ARMS:
        values = prediction["arms"][arm]
        write_trial_predictions(
            run_dir,
            logits=values["logits"],
            probabilities=values["probabilities"],
            pred=values["pred"],
            label=labels_e,
            subject=[row["subject"] for row in metadata_e],
            session="E",
            run=[row["run"] for row in metadata_e],
            trial_id=[row["trial_id"] for row in metadata_e],
            seed=seed,
            model=f"v8_e9_no_augmentation_{arm}",
            basename=PREDICTION_NAMES[arm],
        )
    arm_metrics = {}
    for arm in ARMS:
        values = prediction["arms"][arm]
        arm_metrics[arm] = {
            **{
                key: values[key]
                for key in ("accuracy", "balanced_accuracy", "kappa", "macro_f1")
            },
            **multiclass_calibration_metrics(values["logits"], labels_e),
        }
    write_json(
        run_dir / "data_identity.json",
        {
            "train": {
                "logical_key": f"session_t/{train_path.name}",
                "sha256": data_hashes[f"session_t/{train_path.name}"],
                "trials": len(metadata_t),
            },
            "evaluation": {
                "logical_key": f"session_e/{evaluation_path.name}",
                "sha256": data_hashes[f"session_e/{evaluation_path.name}"],
                "trials": len(metadata_e),
            },
        },
    )
    write_json(
        run_dir / "data_access_manifest.json",
        {
            "train_session": "T",
            "evaluation_session": "E",
            "session_e_first_semantic_load_after_checkpoint_sha256": before,
            "session_e_checkpoint_selection": False,
            "session_e_gradient_updates": False,
            "openbmi_s2_gradient_updates": False,
        },
    )
    write_json(
        run_dir / "state_audit.json",
        {
            "checkpoint_before_session_e": before,
            "state_after_session_e": after,
            "identical": True,
        },
    )
    metrics = {
        "status": "completed",
        "stage": "E9_ENSEMBLE_ABLATION",
        "variant": "no_augmentation",
        "subject": int(subject),
        "seed": int(seed),
        "accuracy": arm_metrics["primary"]["accuracy"],
        "balanced_accuracy": arm_metrics["primary"]["balanced_accuracy"],
        "kappa": arm_metrics["primary"]["kappa"],
        "macro_f1": arm_metrics["primary"]["macro_f1"],
        "arms": arm_metrics,
        "snn_mean_firing_rate": prediction["snn_mean_firing_rate"],
        "entropy_gate_mean": float(np.mean(prediction["primary_gate"])),
        "parameters": {
            name: sum(parameter.numel() for parameter in model.parameters())
            for name, model in fit.components.as_dict().items()
        },
        "optimizer_steps": fit.optimizer_steps,
        "train_seconds": fit.train_seconds,
        "freeze_sha256": freeze["combined_sha256"],
        "reference_e6_run_fingerprint": e6_metrics["run_fingerprint"],
        "run_fingerprint": fingerprint["combined_sha256"],
        "heldout_e_selected_checkpoint": False,
        "post_e6_explanatory_ablation": True,
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "runtime_status.json",
        {
            "status": "completed",
            "completed_at": time.time(),
            "session_e_arrays_loaded": True,
        },
    )
    write_run_artifact_manifest(run_dir, required_files=RUN_FILES)
    return metrics


def _worker_command(
    args: argparse.Namespace, output: Path, subject: int, seeds: list[int]
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--train-data",
        str(Path(args.train_data).resolve()),
        "--evaluation-data",
        str(Path(args.evaluation_data).resolve()),
        "--e6",
        str(Path(args.e6).resolve()),
        "--e6-audit",
        str(Path(args.e6_audit).resolve()),
        "--e7",
        str(Path(args.e7).resolve()),
        "--e8-audit",
        str(Path(args.e8_audit).resolve()),
        "--e3-final-gate",
        str(Path(args.e3_final_gate).resolve()),
        "--source-root",
        str(Path(args.source_root).resolve()),
        "--freeze",
        str(Path(args.freeze).resolve()),
        "--output",
        str(output),
        "--config",
        str(Path(args.config).resolve()),
        "--device",
        str(args.device),
        "--worker-subject",
        str(subject),
        "--worker-seeds",
        ",".join(str(seed) for seed in seeds),
    ]


def _run_worker(command: list[str], expected: list[Path]) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    threads = int(environment.get("DPC_SNN_E9_ENSEMBLE_WORKER_THREADS", "8"))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = str(threads)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"E9 worker failed ({completed.returncode}): {' '.join(command)}\n"
            + completed.stderr[-8000:]
        )
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError("E9 worker missed outputs: " + ", ".join(missing))
    return [read_json(path) for path in expected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--evaluation-data", required=True)
    parser.add_argument("--e6", required=True)
    parser.add_argument("--e6-audit", required=True)
    parser.add_argument("--e7", required=True)
    parser.add_argument("--e8-audit", required=True)
    parser.add_argument("--e3-final-gate", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default="configs/experiments/v8_e9_ensemble_ablation.yaml"
    )
    parser.add_argument("--subjects", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--worker-subject", type=int)
    parser.add_argument("--worker-seeds", default="")
    args = parser.parse_args()

    configure_cache_env()
    output = ensure_dir(Path(args.output).resolve())
    train_root = Path(args.train_data).resolve()
    evaluation_root = Path(args.evaluation_data).resolve()
    if train_root == evaluation_root:
        raise RuntimeError("E9 requires physically separated Session T and E roots")
    e6 = Path(args.e6).resolve()
    source_root = Path(args.source_root).resolve()
    config = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    source_tree = collect_source_tree_manifest(ROOT)
    freeze = validate_ensemble_freeze_contract(
        validate_v8_freeze_manifest(Path(args.freeze).resolve())
    )
    e6_audit = read_json(Path(args.e6_audit).resolve() / "audit_report.json")
    e7_gate = read_json(Path(args.e7).resolve() / "gate_decision.json")
    e8_audit = read_json(Path(args.e8_audit).resolve() / "audit_report.json")
    e3_gate = read_json(Path(args.e3_final_gate).resolve())
    if (
        e6_audit.get("status") != "passed"
        or e7_gate.get("status") != "completed"
        or e8_audit.get("status") != "passed"
        or e3_gate.get("status") != "completed"
    ):
        raise RuntimeError("E9 requires completed E3, E6, E7 and E8 evidence")
    subjects = (
        [int(args.worker_subject)]
        if args.worker_subject is not None
        else _csv_values(args.subjects, int)
        or ([1] if args.canary else list(range(1, 10)))
    )
    seeds = (
        _csv_values(args.worker_seeds, int)
        if args.worker_subject is not None
        else _csv_values(args.seeds, int)
        or ([0] if args.canary else list(range(5)))
    )
    environment = _environment()
    if args.worker_subject is not None:
        rows = [
            _run_one(
                output=output,
                train_path=_subject_file(train_root, subjects[0]),
                evaluation_path=_subject_file(evaluation_root, subjects[0]),
                e6=e6,
                source_root=source_root,
                freeze=freeze,
                config=config,
                source_tree=source_tree,
                environment=environment,
                subject=subjects[0],
                seed=seed,
                device=args.device,
            )
            for seed in seeds
        ]
        print(json.dumps({"status": "worker_completed", "runs": len(rows)}, indent=2))
        return

    write_json(output / "source_tree_manifest.json", source_tree)
    (output / "resolved_campaign.yaml").write_text(
        yaml.safe_dump(
            {
                **config,
                "active_subjects": subjects,
                "active_seeds": seeds,
                "source_tree_sha256": source_tree_digest(source_tree),
                "freeze_sha256": freeze["combined_sha256"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    if int(args.workers) > 1 and len(subjects) > 1:
        with ThreadPoolExecutor(max_workers=min(int(args.workers), len(subjects))) as pool:
            futures = {}
            for subject in subjects:
                expected = [
                    output
                    / "no_augmentation"
                    / f"subject_{subject:02d}"
                    / f"seed_{seed}"
                    / "metrics.json"
                    for seed in seeds
                ]
                future = pool.submit(
                    _run_worker,
                    _worker_command(args, output, subject, seeds),
                    expected,
                )
                futures[future] = subject
            for future in as_completed(futures):
                rows.extend(future.result())
    else:
        for subject in subjects:
            for seed in seeds:
                rows.append(
                    _run_one(
                        output=output,
                        train_path=_subject_file(train_root, subject),
                        evaluation_path=_subject_file(evaluation_root, subject),
                        e6=e6,
                        source_root=source_root,
                        freeze=freeze,
                        config=config,
                        source_tree=source_tree,
                        environment=environment,
                        subject=subject,
                        seed=seed,
                        device=args.device,
                    )
                )
    rows = sorted(rows, key=lambda row: (row["subject"], row["seed"]))
    retrained_rows = [
        {
            "variant": "no_augmentation",
            "subject": row["subject"],
            "seed": row["seed"],
            **row["arms"]["primary"],
        }
        for row in rows
    ]
    write_csv(output / "retrained_summary.csv", retrained_rows)

    e6_rows = []
    for subject in subjects:
        for seed in seeds:
            metrics = read_json(
                e6 / f"subject_{subject:02d}" / f"seed_{seed}" / "metrics.json"
            )
            e6_rows.append(metrics)
    reused_rows = [
        {
            "subject": row["subject"],
            "seed": row["seed"],
            "arm": arm,
            **row["arms"][arm],
        }
        for row in e6_rows
        for arm in ARMS
    ]
    write_csv(output / "reused_controls.csv", reused_rows)
    paired_rows = []
    comparisons = {}
    no_aug = [
        {"subject": row["subject"], "seed": row["seed"], "accuracy": row["arms"]["primary"]["accuracy"]}
        for row in rows
    ]
    e6_primary = [
        {"subject": row["subject"], "seed": row["seed"], "accuracy": row["arms"]["primary"]["accuracy"]}
        for row in e6_rows
    ]
    pairs = pair_subject_seed_rows(no_aug, e6_primary, value="accuracy")
    comparisons["augmentation_contribution_e6_minus_no_augmentation"] = paired_delta_summary(
        pairs, seed=20260901
    )
    paired_rows.extend(
        {"comparison": "augmentation_contribution_e6_minus_no_augmentation", **pair}
        for pair in pairs
    )
    for index, arm in enumerate(("matched_ann", "anchor", "atcnet", "fbcnet", "decoder_snn", "decoder_ann")):
        comparator = [
            {"subject": row["subject"], "seed": row["seed"], "accuracy": row["arms"][arm]["accuracy"]}
            for row in e6_rows
        ]
        pairs = pair_subject_seed_rows(comparator, e6_primary, value="accuracy")
        key = f"primary_minus_{arm}"
        comparisons[key] = paired_delta_summary(pairs, seed=20260902 + index)
        paired_rows.extend({"comparison": key, **pair} for pair in pairs)
    write_csv(output / "paired_subject_seed.csv", paired_rows)
    write_json(output / "comparisons.json", comparisons)
    evidence = {
        "e3_delay_and_cross_band_gate": {
            "path": str(Path(args.e3_final_gate).resolve()),
            "sha256": file_sha256(Path(args.e3_final_gate).resolve()),
            "selected_branch": e3_gate.get("selected_branch"),
            "delay_promoted": e3_gate.get("selected_branch") == "delay",
        },
        "e7_robustness_and_utility": {
            "path": str(Path(args.e7).resolve() / "gate_decision.json"),
            "sha256": file_sha256(Path(args.e7).resolve() / "gate_decision.json"),
            "gate": e7_gate,
        },
        "e8_external_confirmation_audit": {
            "path": str(Path(args.e8_audit).resolve() / "audit_report.json"),
            "sha256": file_sha256(Path(args.e8_audit).resolve() / "audit_report.json"),
            "status": e8_audit.get("status"),
        },
    }
    write_json(output / "evidence_links.json", evidence)
    full_contract = subjects == list(range(1, 10)) and seeds == list(range(5))
    status = {
        "status": "completed",
        "stage": "E9_ENSEMBLE_ABLATION",
        "subjects": subjects,
        "seeds": seeds,
        "runs": len(rows),
        "full_registered_contract": full_contract,
        "retrained_variants": ["no_augmentation"],
        "reused_controls": list(config["reused_exact_controls"]),
        "post_e6_explanatory_ablation": True,
        "session_e_checkpoint_selection": False,
        "openbmi_s2_model_updates": False,
        "freeze_sha256": freeze["combined_sha256"],
    }
    write_json(output / "campaign_status.json", status)
    write_run_artifact_manifest(output, required_files=CAMPAIGN_FILES)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
