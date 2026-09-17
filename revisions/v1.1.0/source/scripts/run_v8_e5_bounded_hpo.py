#!/usr/bin/env python3
"""Run the bounded V8 E5 Session-T inner-validation-only architecture search."""

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

from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_baselines import session_t_development_view  # noqa: E402
from dpc_snn.experiments.v8_hpo import (  # noqa: E402
    generate_balanced_v8_candidates,
    rank_v8_hpo_candidates,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    build_v8_run_fingerprint,
    collect_source_tree_manifest,
    source_tree_digest,
    validate_v8_resume_fingerprint,
    v8_heldout_lock_manifest,
    write_v8_fingerprint,
)
from dpc_snn.experiments.v8_training import (  # noqa: E402
    apply_v8_fold_gain,
    fit_v8,
    fit_v8_gain_from_cached_rates,
    predict_v8,
    seed_v8,
)
from dpc_snn.models.build import build_model  # noqa: E402
from dpc_snn.models.v8_accuracy_first import V8AccuracyFirstModel  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v8_e2_zero_delay import (  # noqa: E402
    SubjectBundle,
    _csv,
    _environment,
    _fit_kwargs,
    _load_or_build_base_rates,
    _nested_folds,
    _split_manifest,
    _subject_file,
)


FOLD_REQUIRED_FILES = (
    "manifest.json",
    "best.pt",
    "last.pt",
    "history.csv",
    "result.json",
    "predictions.npz",
    "source_fingerprint.json",
    "resolved_config.yaml",
)


def _build_candidate(
    model_config: dict[str, Any], candidate: dict[str, Any], *, seed: int
) -> V8AccuracyFirstModel:
    seed_v8(seed)
    model = build_model(
        "v8_accuracy_first", {**model_config, **candidate["model_overrides"]}
    )
    if not isinstance(model, V8AccuracyFirstModel):
        raise TypeError("V8 E5 model factory returned an unexpected model")
    if model.delay_auxiliary_enabled or model.delay_auxiliary is not None:
        raise RuntimeError("V8 E5 candidates may not enable delay")
    if model.decoder is None or model.decoder.decoder_kind != "ann":
        raise RuntimeError("V8 E5 searches only the zero-delay ANN backbone")
    if model.decoder.decoder_residual_mode != "sew_add":
        raise RuntimeError("V8 E5 ANN must retain the strong residual reference")
    return model


def _load_fold(
    directory: Path,
    *,
    expected_indices: np.ndarray,
    expected_labels: np.ndarray,
) -> dict[str, Any] | None:
    if not all((directory / name).is_file() for name in FOLD_REQUIRED_FILES):
        return None
    validate_run_artifact_manifest(
        directory,
        required_files=FOLD_REQUIRED_FILES,
        verify_hashes=True,
        verify_prediction_schema=False,
    )
    result = read_json(directory / "result.json")
    with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"indices", "labels", "logits", "prefix_logits"}:
            raise RuntimeError(f"invalid E5 prediction schema under {directory}")
        if not np.array_equal(archive["indices"], expected_indices):
            raise RuntimeError(f"E5 validation indices changed under {directory}")
        if not np.array_equal(archive["labels"], expected_labels):
            raise RuntimeError(f"E5 validation labels changed under {directory}")
        if not np.isfinite(archive["logits"]).all() or not np.isfinite(
            archive["prefix_logits"]
        ).all():
            raise RuntimeError(f"non-finite E5 predictions under {directory}")
    return result


def _run_fold(
    *,
    output: Path,
    stage_name: str,
    stage_config: dict[str, Any],
    config: dict[str, Any],
    model_config: dict[str, Any],
    source_tree: dict[str, str],
    environment: dict[str, Any],
    bundle: SubjectBundle,
    candidate: dict[str, Any],
    fold_index: int,
    device: str,
) -> dict[str, Any]:
    subject = int(bundle.metadata[0]["subject"])
    _, outer_test, inner_train, inner_validation, inner_run = bundle.nested_folds[
        int(fold_index)
    ]
    fold_seed = int(config["seed"]) * 100_003 + subject * 10_007 + int(fold_index) * 1_009
    training = {
        **dict(config["training"]),
        **dict(candidate["training_overrides"]),
    }
    resolved = {
        "stage": "E5",
        "search_stage": stage_name,
        "stage_config": stage_config,
        "candidate": candidate,
        "subject": subject,
        "fold": int(fold_index),
        "seed": int(config["seed"]),
        "resolved_model": {**model_config, **candidate["model_overrides"]},
        "resolved_training": training,
        "augmentation": config["augmentation"],
        "evaluation_role": "inner_validation_only",
    }
    fingerprint = build_v8_run_fingerprint(
        resolved_run_config=resolved,
        source_tree=source_tree,
        data={
            bundle.subject_path.name: bundle.data_sha256,
            "access": bundle.access_manifest,
            "physical_rate_cache": bundle.cache_manifest,
        },
        split={
            "subject": subject,
            "fold": int(fold_index),
            "inner_train_trial_ids": [
                bundle.metadata[int(index)]["trial_id"] for index in inner_train
            ],
            "inner_validation_trial_ids": [
                bundle.metadata[int(index)]["trial_id"] for index in inner_validation
            ],
            "outer_test_trial_ids_forbidden_for_metrics": [
                bundle.metadata[int(index)]["trial_id"] for index in outer_test
            ],
            "inner_validation_run": inner_run,
        },
        augmentation=config["augmentation"],
        prior={"policy": "none", "delay_auxiliary_enabled": False},
        checkpoint={
            "policy": "best inner-validation kappa, accuracy tie-break",
            "outer_test_evaluation": "forbidden",
            "session_e_accessed": False,
        },
        environment=environment,
    )
    directory = ensure_dir(
        output
        / stage_name
        / candidate["candidate_id"]
        / f"subject_{subject:02d}"
        / f"fold_{int(fold_index)}"
    )
    fingerprint_path = directory / "source_fingerprint.json"
    cached = _load_fold(
        directory,
        expected_indices=inner_validation,
        expected_labels=bundle.y[inner_validation],
    )
    if cached is not None:
        validate_v8_resume_fingerprint(fingerprint_path, fingerprint)
        return cached
    if fingerprint_path.is_file():
        validate_v8_resume_fingerprint(fingerprint_path, fingerprint)
    else:
        write_v8_fingerprint(fingerprint_path, fingerprint)
        (directory / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )

    model = _build_candidate(model_config, candidate, seed=fold_seed)
    gain = fit_v8_gain_from_cached_rates(bundle.base_rates, inner_train)
    train_rates = apply_v8_fold_gain(bundle.base_rates.subset(inner_train), gain)
    validation_rates = apply_v8_fold_gain(
        bundle.base_rates.subset(inner_validation), gain
    )
    fit = fit_v8(
        model,
        train_rates,
        bundle.y[inner_train],
        validation_rates=validation_rates,
        validation_labels=bundle.y[inner_validation],
        device=device,
        seed=fold_seed,
        epochs=int(stage_config["max_epochs"]),
        patience=int(stage_config["patience"]),
        minimum_epochs=int(stage_config["minimum_epochs"]),
        run_label=(
            f"V8-E5:{stage_name}:{candidate['candidate_id']}:"
            f"S{subject}:fold{fold_index}"
        ),
        **_fit_kwargs(training, dict(config["augmentation"])),
    )
    evaluation = predict_v8(
        fit.model,
        validation_rates,
        bundle.y[inner_validation],
        device=device,
        batch_size=int(training["batch_size"]),
    )
    result = {
        "status": "completed",
        "stage": "E5",
        "search_stage": stage_name,
        "candidate_id": candidate["candidate_id"],
        "subject": subject,
        "fold": int(fold_index),
        "fold_seed": fold_seed,
        "inner_validation_run": inner_run,
        "evaluation_role": "inner_validation",
        "outer_test_trials_evaluated": 0,
        "session_e_trials_evaluated": 0,
        "best_epoch": fit.best_epoch,
        "validation_accuracy": evaluation["accuracy"],
        "validation_balanced_accuracy": evaluation["balanced_accuracy"],
        "validation_kappa": evaluation["kappa"],
        "validation_macro_f1": evaluation["macro_f1"],
        "endpoint_metrics": evaluation["endpoint_metrics"],
        "parameters": fit.model.parameter_count,
        "trainable_parameters": fit.model.trainable_parameter_count,
        "optimizer_steps": fit.optimizer_steps,
        "elapsed_seconds": fit.elapsed_seconds,
        "gain": gain.tolist(),
        "run_fingerprint": fingerprint["combined_sha256"],
    }
    torch.save(fit.best_state, directory / "best.pt")
    torch.save(fit.last_state, directory / "last.pt")
    write_csv(directory / "history.csv", fit.history)
    write_json(directory / "result.json", result)
    np.savez_compressed(
        directory / "predictions.npz",
        indices=inner_validation.astype(np.int64),
        labels=evaluation["labels"].astype(np.int64),
        logits=evaluation["logits"].astype(np.float32),
        prefix_logits=evaluation["prefix_logits"].astype(np.float32),
    )
    write_run_artifact_manifest(directory, required_files=FOLD_REQUIRED_FILES)
    print(
        "__V8_E5_FOLD_DONE__ "
        f"stage={stage_name} candidate={candidate['candidate_id']} "
        f"subject={subject} fold={fold_index} kappa={evaluation['kappa']:.6f}",
        flush=True,
    )
    del model, fit, evaluation, train_rates, validation_rates
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def _selected_outputs(
    *,
    output: Path,
    selected: dict[str, Any],
    model_config: dict[str, Any],
    e2_config: dict[str, Any],
) -> None:
    selected_model = {**model_config, **selected["model_overrides"]}
    selected_training = {
        **dict(e2_config["training"]),
        **dict(selected["training_overrides"]),
    }
    selected_e2 = {
        **e2_config,
        "experiment_id": "V8_E2_HPO_SELECTED_ZERO_DELAY",
        "architecture_version": selected_model.get("architecture_version"),
        "screening_seed": 0,
        "confirmation_top_k": 1,
        "confirmation_seeds": [0, 1, 2],
        "required_confirmation_variants": ["hpo_selected_full_ann"],
        "variants": {
            "hpo_selected_full_ann": {
                "use_statistical_branch": True,
                "use_covariance_branch": True,
                "use_temporal_branch": True,
                "decoder_kind": "ann",
                "decoder_residual_mode": "sew_add",
                "delay_auxiliary_enabled": False,
            }
        },
        "training": selected_training,
        "hpo_provenance": {
            "candidate_id": selected["candidate_id"],
            "selection_scope": "Session-T inner-validation only",
            "outer_test_accessed_for_selection": False,
            "session_e_accessed": False,
        },
    }
    (output / "selected_model.yaml").write_text(
        yaml.safe_dump(selected_model, sort_keys=False), encoding="utf-8"
    )
    (output / "selected_e2_config.yaml").write_text(
        yaml.safe_dump(selected_e2, sort_keys=False), encoding="utf-8"
    )


def _subject_bundle(
    *,
    data_root: Path,
    cache_root: Path,
    output: Path,
    subject: int,
    config: dict[str, Any],
    reference: V8AccuracyFirstModel,
    device: str,
) -> SubjectBundle:
    subject_path = _subject_file(data_root, int(subject))
    data_sha256 = file_sha256(subject_path)
    data = load_processed_npz(subject_path)
    x, y, metadata, access_manifest = session_t_development_view(data)
    nested = _nested_folds(
        metadata,
        n_splits=int(config["selection"]["n_splits"]),
        split_seed=int(config["selection"]["split_seed"]),
    )
    split_manifest = _split_manifest(
        metadata,
        nested,
        subject=int(subject),
        split_seed=int(config["selection"]["split_seed"]),
    )
    base_rates, cache_manifest = _load_or_build_base_rates(
        output=cache_root,
        subject=int(subject),
        subject_path=subject_path,
        data_sha256=data_sha256,
        x=x,
        metadata=metadata,
        model=reference,
        preprocessing=dict(config["preprocessing"]),
        canary_train_indices=nested[0][0],
        device=device,
    )
    del data
    return SubjectBundle(
        subject_path=subject_path,
        data_sha256=data_sha256,
        x=x,
        y=y,
        metadata=metadata,
        access_manifest=access_manifest,
        nested_folds=nested,
        split_manifest=split_manifest,
        base_rates=base_rates,
        cache_manifest=cache_manifest,
    )


def _worker_command(
    *,
    args: argparse.Namespace,
    output: Path,
    stage_name: str,
    subject: int,
    candidate_ids: list[str],
    folds: list[int],
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--data",
        str(Path(args.data).resolve()),
        "--output",
        str(output),
        "--config",
        str(Path(args.config).resolve()),
        "--model-config",
        str(Path(args.model_config).resolve()),
        "--e2-config",
        str(Path(args.e2_config).resolve()),
        "--physical-cache-root",
        str(Path(args.physical_cache_root).resolve()),
        "--device",
        str(args.device),
        "--worker-stage",
        stage_name,
        "--worker-subject",
        str(subject),
        "--worker-candidate-ids",
        ",".join(candidate_ids),
        "--worker-folds",
        ",".join(str(value) for value in folds),
    ]
    if args.max_epochs is not None:
        command.extend(("--max-epochs", str(int(args.max_epochs))))
    return command


def _run_worker_command(
    command: list[str], result_paths: list[Path]
) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    worker_threads = int(environment.get("DPC_SNN_E5_WORKER_THREADS", "8"))
    if worker_threads < 1 or worker_threads > 32:
        raise ValueError("DPC_SNN_E5_WORKER_THREADS must be between 1 and 32")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = str(worker_threads)
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
            "E5 fold worker failed with code "
            f"{completed.returncode}: {' '.join(command)}\n{completed.stderr[-8000:]}"
        )
    missing = [str(path) for path in result_paths if not path.is_file()]
    if missing:
        raise RuntimeError("E5 subject worker missed results: " + ", ".join(missing))
    return [read_json(path) for path in result_paths]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default="configs/experiments/v8_e5_bounded_hpo.yaml"
    )
    parser.add_argument("--model-config", default="configs/models/v8_accuracy_first.yaml")
    parser.add_argument("--e2-config", default="configs/experiments/v8_e2_zero_delay.yaml")
    parser.add_argument("--physical-cache-root", default="")
    parser.add_argument("--subjects", default="")
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--stages", default="")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-stage", default="")
    parser.add_argument("--worker-subject", type=int, default=None)
    parser.add_argument("--worker-candidate-ids", default="")
    parser.add_argument("--worker-folds", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--source-smoke-only", action="store_true")
    args = parser.parse_args()

    configure_cache_env()
    output = ensure_dir(Path(args.output).resolve())
    data_root = Path(args.data).resolve()
    config = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    model_config = yaml.safe_load(
        Path(args.model_config).resolve().read_text(encoding="utf-8")
    )
    e2_config = yaml.safe_load(
        Path(args.e2_config).resolve().read_text(encoding="utf-8")
    )
    if (
        config.get("stage") != "development"
        or config["selection"].get("evaluation_role") != "inner_validation_only"
        or not config["selection"].get("outer_test_predictions_forbidden")
        or config["data_access"].get("heldout_session_e_accessed")
    ):
        raise RuntimeError("V8 E5 data-access/selection contract is not locked")
    candidates = generate_balanced_v8_candidates(config)
    registered_candidate_count = len(candidates)
    if args.candidate_limit is not None:
        candidates = candidates[: int(args.candidate_limit)]
    subjects = _csv(args.subjects, int) if args.subjects else list(config["subjects"])
    stages = list(config["successive_halving"])
    if args.stages:
        selected_stages = _csv(args.stages)
        unknown = sorted(set(selected_stages) - set(stages))
        if unknown:
            raise ValueError(f"unknown E5 stages: {unknown}")
        stages = selected_stages
    if args.max_epochs is not None:
        for name in stages:
            config["successive_halving"][name]["max_epochs"] = int(args.max_epochs)
            config["successive_halving"][name]["minimum_epochs"] = min(
                int(config["successive_halving"][name]["minimum_epochs"]),
                int(args.max_epochs),
            )
            config["successive_halving"][name]["patience"] = min(
                int(config["successive_halving"][name]["patience"]),
                int(args.max_epochs),
            )
    workers = int(args.workers)
    if workers < 1 or workers > 8:
        raise ValueError("V8 E5 workers must be between 1 and 8")
    if args.worker_stage:
        if not args.physical_cache_root:
            raise RuntimeError("E5 fold workers require a read-only physical cache root")
        if (
            args.worker_subject is None
            or not args.worker_candidate_ids
            or not args.worker_folds
        ):
            raise ValueError("incomplete E5 subject-worker identity")
        candidate_index = {row["candidate_id"]: row for row in candidates}
        worker_candidate_ids = _csv(args.worker_candidate_ids)
        unknown_candidates = sorted(set(worker_candidate_ids) - set(candidate_index))
        if unknown_candidates:
            raise ValueError(f"unknown E5 subject-worker candidates: {unknown_candidates}")
        if args.worker_stage not in config["successive_halving"]:
            raise ValueError("unknown E5 subject-worker stage")
        stage_config = dict(config["successive_halving"][args.worker_stage])
        worker_folds = _csv(args.worker_folds, int)
        if worker_folds != [int(value) for value in stage_config["active_folds"]]:
            raise ValueError("E5 subject worker received the wrong active folds")
        source_tree = collect_source_tree_manifest(ROOT)
        reference = _build_candidate(model_config, candidates[0], seed=0)
        bundle = _subject_bundle(
            data_root=data_root,
            cache_root=Path(args.physical_cache_root).resolve(),
            output=output,
            subject=int(args.worker_subject),
            config=config,
            reference=reference,
            device=args.device,
        )
        worker_environment = _environment()
        results = []
        for candidate_id in worker_candidate_ids:
            for fold_index in worker_folds:
                results.append(
                    _run_fold(
                        output=output,
                        stage_name=args.worker_stage,
                        stage_config=stage_config,
                        config=config,
                        model_config=model_config,
                        source_tree=source_tree,
                        environment=worker_environment,
                        bundle=bundle,
                        candidate=candidate_index[candidate_id],
                        fold_index=fold_index,
                        device=args.device,
                    )
                )
        print(json.dumps({"status": "completed", "results": len(results)}, indent=2))
        return
    full_contract = bool(
        registered_candidate_count == int(config["maximum_unique_configurations"])
        and len(candidates) == registered_candidate_count
        and subjects == list(config["subjects"])
        and stages == list(config["successive_halving"])
        and args.max_epochs is None
    )
    if not full_contract and not args.canary and not args.source_smoke_only:
        raise RuntimeError("partial E5 execution requires --canary")
    if int(config["training"]["effective_batch_size"]) != int(
        config["training"]["batch_size"]
    ) * int(config["training"]["gradient_accumulation_steps"]):
        raise RuntimeError("V8 E5 effective batch size metadata is inconsistent")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")

    source_tree = collect_source_tree_manifest(ROOT)
    environment = _environment()
    write_json(output / "source_tree_manifest.json", source_tree)
    write_json(
        output / "source_tree_summary.json",
        {"files": len(source_tree), "sha256": source_tree_digest(source_tree)},
    )
    write_json(output / "heldout_lock_manifest.json", v8_heldout_lock_manifest())
    (output / "resolved_campaign.yaml").write_text(
        yaml.safe_dump(
            {
                **config,
                "active_subjects": subjects,
                "active_candidate_ids": [row["candidate_id"] for row in candidates],
                "active_stages": stages,
                "canary": bool(args.canary),
                "parallel_workers": workers,
                "worker_cpu_threads": (
                    int(os.environ.get("DPC_SNN_E5_WORKER_THREADS", "8"))
                    if workers > 1
                    else None
                ),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    write_json(
        output / "candidate_ledger.json",
        {
            "registered_count": registered_candidate_count,
            "active_count": len(candidates),
            "maximum_unique_configurations": int(config["maximum_unique_configurations"]),
            "balanced_complete_factorial": True,
            "candidates": candidates,
        },
    )
    capacity_rows = []
    frontend_fingerprints = set()
    for candidate in candidates:
        model = _build_candidate(model_config, candidate, seed=0)
        capacity_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "parameters": model.parameter_count,
                "trainable_parameters": model.trainable_parameter_count,
            }
        )
        frontend_fingerprints.add(model.physical_frontend_fingerprint())
    if len(frontend_fingerprints) != 1:
        raise RuntimeError("E5 candidates changed the physical cache front end")
    write_csv(output / "capacity_audit.csv", capacity_rows)
    if args.source_smoke_only:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "source_smoke_only": True,
                    "candidates": len(candidates),
                    "maximum_parameters": max(row["parameters"] for row in capacity_rows),
                },
                indent=2,
            )
        )
        return

    cache_root = output
    external_cache = bool(args.physical_cache_root)
    if external_cache:
        cache_root = Path(args.physical_cache_root).resolve()
        for subject in subjects:
            directory = cache_root / "shared_physical_rates" / f"subject_{subject:02d}"
            if not (directory / "unit_gain_rates.pt").is_file() or not (
                directory / "manifest.json"
            ).is_file():
                raise FileNotFoundError(f"incomplete external E5 cache: {directory}")
    write_json(
        output / "shared_cache_provenance.json",
        {
            "external_read_only_cache": external_cache,
            "root": str(cache_root),
            "physical_frontend_fingerprint": next(iter(frontend_fingerprints)),
        },
    )

    bundles: dict[int, SubjectBundle] = {}
    reference = _build_candidate(model_config, candidates[0], seed=0)
    if workers == 1:
        for subject in subjects:
            bundles[int(subject)] = _subject_bundle(
                data_root=data_root,
                cache_root=cache_root,
                output=output,
                subject=int(subject),
                config=config,
                reference=reference,
                device=args.device,
            )

    all_rows: list[dict[str, Any]] = []
    active = list(candidates)
    promotions: list[dict[str, Any]] = []
    completed_stages: list[str] = []
    for stage_name in stages:
        stage_config = dict(config["successive_halving"][stage_name])
        folds = [int(value) for value in stage_config["active_folds"]]
        stage_rows = []
        tasks = [
            (candidate, int(subject), int(fold_index))
            for candidate in active
            for subject in subjects
            for fold_index in folds
        ]
        if workers == 1:
            for candidate, subject, fold_index in tasks:
                row = _run_fold(
                    output=output,
                    stage_name=stage_name,
                    stage_config=stage_config,
                    config=config,
                    model_config=model_config,
                    source_tree=source_tree,
                    environment=environment,
                    bundle=bundles[subject],
                    candidate=candidate,
                    fold_index=fold_index,
                    device=args.device,
                )
                stage_rows.append(row)
                all_rows.append(row)
                write_csv(output / "summary.csv", all_rows)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {}
                active_ids = [candidate["candidate_id"] for candidate in active]
                for subject in subjects:
                    result_paths = [
                        output
                        / stage_name
                        / candidate_id
                        / f"subject_{int(subject):02d}"
                        / f"fold_{fold_index}"
                        / "result.json"
                        for candidate_id in active_ids
                        for fold_index in folds
                    ]
                    command = _worker_command(
                        args=args,
                        output=output,
                        stage_name=stage_name,
                        subject=int(subject),
                        candidate_ids=active_ids,
                        folds=folds,
                    )
                    futures[executor.submit(
                        _run_worker_command, command, result_paths
                    )] = int(subject)
                completed_rows = []
                for future in as_completed(futures):
                    completed_rows.extend(future.result())
                stage_rows = sorted(
                    completed_rows,
                    key=lambda row: (
                        str(row["candidate_id"]),
                        int(row["subject"]),
                        int(row["fold"]),
                    ),
                )
                all_rows.extend(stage_rows)
                write_csv(output / "summary.csv", all_rows)
        ranking = rank_v8_hpo_candidates(
            stage_rows,
            candidate_ids=[row["candidate_id"] for row in active],
            subjects=subjects,
            folds=folds,
        )
        write_csv(output / f"{stage_name}_ranking.csv", ranking)
        promote_count = min(int(stage_config["promote_top_k"]), len(active))
        promoted_ids = [row["candidate_id"] for row in ranking[:promote_count]]
        promotions.append(
            {
                "stage": stage_name,
                "input_candidates": [row["candidate_id"] for row in active],
                "promoted_candidates": promoted_ids,
                "selection_data": "Session-T inner-validation only",
                "outer_test_accessed": False,
            }
        )
        index = {row["candidate_id"]: row for row in candidates}
        active = [index[candidate_id] for candidate_id in promoted_ids]
        completed_stages.append(stage_name)
        write_json(output / "promotion_ledger.json", promotions)

    selected = active[0] if len(active) == 1 and completed_stages else None
    if full_contract and selected is None:
        raise RuntimeError("full E5 contract did not select exactly one candidate")
    if selected is not None:
        _selected_outputs(
            output=output,
            selected=selected,
            model_config=model_config,
            e2_config=e2_config,
        )
        write_json(
            output / "selected_candidate.json",
            {
                **selected,
                "selection_data": "BCI2a Session T inner-validation only",
                "outer_test_accessed": False,
                "session_e_accessed": False,
                "requires_fresh_e2_nested_oof": True,
            },
        )
    status = {
        "status": "completed" if full_contract else "canary_completed",
        "stage": "E5",
        "protocol": config["protocol"],
        "registered_candidates": registered_candidate_count,
        "active_candidates": len(candidates),
        "subjects": subjects,
        "completed_stages": completed_stages,
        "selected_candidate_id": None if selected is None else selected["candidate_id"],
        "selection_role": "inner_validation_only",
        "outer_test_predictions_created": False,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
        "full_registered_contract": full_contract,
        "source_tree_sha256": source_tree_digest(source_tree),
        "parallel_workers": workers,
        "worker_cpu_threads": (
            int(os.environ.get("DPC_SNN_E5_WORKER_THREADS", "8"))
            if workers > 1
            else None
        ),
        "completed_at": time.time(),
    }
    write_json(output / "campaign_status.json", status)
    required = [
        "manifest.json",
        "campaign_status.json",
        "summary.csv",
        "candidate_ledger.json",
        "capacity_audit.csv",
        "source_tree_manifest.json",
        "source_tree_summary.json",
        "heldout_lock_manifest.json",
        "shared_cache_provenance.json",
        "resolved_campaign.yaml",
        "promotion_ledger.json",
        *[f"{name}_ranking.csv" for name in completed_stages],
    ]
    if selected is not None:
        required.extend(
            ["selected_candidate.json", "selected_model.yaml", "selected_e2_config.yaml"]
        )
    write_run_artifact_manifest(output, required_files=required)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
