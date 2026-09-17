#!/usr/bin/env python3
"""Run bounded Session-T-only HPO for the matched-transport residual expert."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v8_anchor import (  # noqa: E402
    e1_selection_prediction_path,
    load_fused_e1_selection_anchor,
    probability_from_logits,
)
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_baselines import session_t_development_view  # noqa: E402
from dpc_snn.experiments.v8_delay_expert_training import (  # noqa: E402
    fit_delay_expert_input_gain,
    fit_v8_delay_expert,
    predict_v8_delay_expert,
)
from dpc_snn.experiments.v8_delay_hpo import (  # noqa: E402
    apply_delay_hpo_candidate,
    enumerate_delay_hpo_candidates,
    rank_delay_hpo_candidates,
)
from dpc_snn.experiments.v8_delay_prior import v8_delay_prior_seed  # noqa: E402
from dpc_snn.experiments.v8_evidence_audit import (  # noqa: E402
    build_v8_evidence_rates,
    pool_v8_anatomical_regions,
    prepare_v8_evidence_space,
)
from dpc_snn.experiments.v8_fusion import entropy_residual_probability  # noqa: E402
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    file_sha256,
    mapping_sha256,
    sha256_fingerprint,
    source_tree_digest,
)
from dpc_snn.experiments.v8_training import (  # noqa: E402
    apply_v8_fold_gain,
    fit_v8_gain_from_cached_rates,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v8_e1_baselines import _subject_file  # noqa: E402
from scripts.run_v8_e2_zero_delay import _nested_folds  # noqa: E402
from scripts.run_v8_e3_delay_residual_campaign import _input_inventory  # noqa: E402
from scripts.run_v8_e3_delay_residual_fold import (  # noqa: E402
    _build_expert,
    _reference_model,
)
from scripts.run_v8_e3_static_delay import (  # noqa: E402
    PRIOR_FILES,
    _load_or_fit_prior,
    _load_parent_rates,
)


CONFIG_SCHEMA = "dpc-snn-v8-e5-matched-transport-hpo/v1"
CANDIDATE_FILES = (
    "manifest.json",
    "candidate_contract.json",
    "best.pt",
    "last.pt",
    "history.csv",
    "metrics.json",
    "validation_predictions.npz",
)
ROOT_FILES = (
    "manifest.json",
    "campaign_contract.json",
    "campaign_status.json",
    "candidate_ledger.csv",
    "resolved_config.yaml",
    "selected_candidate.json",
    "selected_e3_config.yaml",
    "source_tree_manifest.json",
)


def _csv(value: str, cast: Any = str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _candidate_index(candidates: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {str(row["candidate_id"]): dict(row) for row in candidates}
    if len(indexed) != len(candidates):
        raise RuntimeError("duplicate E5 candidate identifiers")
    return indexed


def _campaign_contract(
    *,
    source_digest: str,
    config_path: Path,
    base_config_path: Path,
    model_config_path: Path,
    inputs: Mapping[str, str],
    candidates: Sequence[Mapping[str, Any]],
    device: str,
) -> dict[str, Any]:
    payload = {
        "schema": "dpc-snn-v8-e5-matched-transport-campaign-contract/v1",
        "source_tree_sha256": source_digest,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "base_e3_config_path": str(base_config_path),
        "base_e3_config_sha256": file_sha256(base_config_path),
        "model_config_path": str(model_config_path),
        "model_config_sha256": file_sha256(model_config_path),
        "input_sha256": dict(sorted(inputs.items())),
        "candidate_order": [str(row["candidate_id"]) for row in candidates],
        "candidate_payload_sha256": sha256_fingerprint(list(candidates)),
        "device": str(device),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
        },
    }
    return {**payload, "combined_sha256": sha256_fingerprint(payload)}


def _prepare_root(output: Path, contract: Mapping[str, Any]) -> None:
    path = output / "campaign_contract.json"
    if output.exists() and any(output.iterdir()):
        if not path.is_file() or read_json(path) != dict(contract):
            raise RuntimeError("non-empty E5 output has no exact campaign contract")
        return
    ensure_dir(output)
    write_json(path, dict(contract))


def _candidate_seed(subject: int, fold: int, classifier_seed: int) -> int:
    """Use common random numbers across candidates within each subject/fold."""

    return (
        int(classifier_seed) * 100_000_007
        + int(subject) * 1_000_003
        + int(fold) * 10_007
        + 8_500_003
    )


def _candidate_dir(
    output: Path, stage: str, subject: int, fold: int, candidate_id: str
) -> Path:
    return (
        output
        / stage
        / f"subject_{subject:02d}"
        / f"fold_{fold}"
        / str(candidate_id)
    )


def _load_completed_candidate(
    directory: Path, expected_contract: Mapping[str, Any]
) -> dict[str, Any] | None:
    if not (directory / "manifest.json").is_file():
        return None
    if read_json(directory / "candidate_contract.json") != dict(expected_contract):
        raise RuntimeError("completed E5 candidate contract changed")
    validate_run_artifact_manifest(
        directory,
        required_files=CANDIDATE_FILES,
        verify_hashes=True,
        verify_prediction_schema=False,
    )
    return read_json(directory / "metrics.json")


def _candidate_contract(
    *,
    campaign_contract_sha256: str,
    stage: str,
    stage_config: Mapping[str, Any],
    candidate: Mapping[str, Any],
    subject: int,
    fold: int,
    shared_input_sha256: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": "dpc-snn-v8-e5-matched-transport-candidate/v1",
        "campaign_contract_sha256": str(campaign_contract_sha256),
        "stage": str(stage),
        "stage_config": dict(stage_config),
        "candidate": dict(candidate),
        "subject": int(subject),
        "fold": int(fold),
        "shared_input_sha256": dict(shared_input_sha256),
        "outer_test_metrics_used": False,
        "full_delay_metrics_used": False,
    }
    return {**payload, "combined_sha256": sha256_fingerprint(payload)}


def _worker(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    base_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    source_digest: str,
    campaign_contract: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    stage = str(args.worker_stage)
    subject = int(args.worker_subject)
    fold = int(args.worker_fold)
    candidate_ids = _csv(args.worker_candidates)
    stage_config = dict(config["successive_halving"][stage])
    if fold not in [int(value) for value in stage_config["active_folds"]]:
        raise ValueError("worker fold is outside the active HPO stage")
    if any(candidate_id not in candidates for candidate_id in candidate_ids):
        raise ValueError("worker received an unknown HPO candidate")

    data_root = Path(args.data).resolve()
    cache_parent = Path(args.cache_parent).resolve()
    atc_root = Path(args.atc_root).resolve()
    fbc_root = Path(args.fbc_root).resolve()
    output = Path(args.output).resolve()
    subject_path = _subject_file(data_root, subject)
    data_sha = file_sha256(subject_path)
    data = load_processed_npz(subject_path)
    x, labels, metadata, access = session_t_development_view(data)
    if bool(access.get("heldout_signals_used_by_development")) or bool(
        access.get("heldout_labels_used_by_development")
    ):
        raise RuntimeError("E5 development view consumed held-out data")
    nested = _nested_folds(metadata, n_splits=6, split_seed=0)
    _, outer_test, inner_train, inner_validation, inner_run = nested[fold]
    if set(inner_train.tolist()) & set(outer_test.tolist()) or set(
        inner_validation.tolist()
    ) & set(outer_test.tolist()):
        raise RuntimeError("E5 inner selection overlaps its outer test fold")

    reference = _reference_model(model_config)
    base_rates, cache_manifest = _load_parent_rates(
        cache_parent,
        subject=subject,
        data_sha256=data_sha,
        model=reference,
    )
    gain = fit_v8_gain_from_cached_rates(base_rates, inner_train)
    train_rates = apply_v8_fold_gain(base_rates.subset(inner_train), gain)
    validation_rates = apply_v8_fold_gain(base_rates.subset(inner_validation), gain)
    delay_config = dict(base_config["delay"])
    channel_names = [str(value) for value in data["ch_names"]]
    if delay_config.get("transport_space") != "car_anatomical_regions":
        raise RuntimeError("E5 matched-transport HPO requires the fixed regional transport")
    train_rates, _ = pool_v8_anatomical_regions(train_rates, channel_names)
    validation_rates, _ = pool_v8_anatomical_regions(validation_rates, channel_names)

    transformed, adjusted_epoch_tmin, transform = prepare_v8_evidence_space(
        x[inner_train],
        str(delay_config["evidence_space"]),
        channel_names=channel_names,
        sfreq=float(np.asarray(data["sfreq"]).item()),
        epoch_tmin=float(np.asarray(data["epoch_tmin"]).item()),
        task_tmin=float(model_config["task_tmin"]),
    )
    evidence_rates = build_v8_evidence_rates(
        transformed,
        model_config,
        epoch_tmin=adjusted_epoch_tmin,
        transform_sha256=str(transform["output_sha256"]),
        device=args.device,
        batch_size=int(base_config["training"]["batch_size"]),
    )
    evidence_rates, pooled = pool_v8_anatomical_regions(evidence_rates, channel_names)
    evidence_identity = {
        **transform,
        **pooled,
        "scope": "inner_training_fold_only",
        "analytic_representation": (
            "v8_csd_unwhitened_var_innovations_fixed_anatomical_regions"
        ),
    }
    replicates = int(delay_config["prior"]["audit_replicates"])
    seeds = [
        v8_delay_prior_seed(subject, fold, replicate, scope="inner")
        for replicate in range(replicates)
    ]
    split_strata = [
        f"run={metadata[int(index)]['run']}|class={int(labels[int(index)])}"
        for index in inner_train
    ]
    prior_dir = output / "_priors" / f"subject_{subject:02d}" / f"fold_{fold}" / "inner"
    prior, prior_summary = _load_or_fit_prior(
        prior_dir,
        scope="inner_training_fold_only",
        trial_ids=[metadata[int(index)]["trial_id"] for index in inner_train],
        rates=train_rates,
        evidence_rates=evidence_rates,
        evidence_identity=evidence_identity,
        gain=gain,
        model_config=dict(model_config),
        delay_config=delay_config,
        source_sha256=source_digest,
        seed=seeds[0],
        ensemble_seeds=seeds,
        split_strata=split_strata,
    )
    if not bool(prior_summary["stability_gate"]["passed"]):
        raise RuntimeError("E5 fold-local prior failed its stability gate")

    classifier_seed = int(config["classifier_seed"])
    atc_selection_path = e1_selection_prediction_path(
        atc_root,
        model="atcnet",
        subject=subject,
        seed=classifier_seed,
        fold=fold,
    )
    fbc_selection_path = e1_selection_prediction_path(
        fbc_root,
        model="fbcnet",
        subject=subject,
        seed=classifier_seed,
        fold=fold,
    )
    validation_anchor = load_fused_e1_selection_anchor(
        atc_selection_path,
        fbc_selection_path,
        expected_indices=inner_validation,
        expected_labels=labels[inner_validation],
    )

    cache_dir = cache_parent / "shared_physical_rates" / f"subject_{subject:02d}"
    shared_input_sha256 = {
        "data": data_sha,
        "cache": file_sha256(cache_dir / "unit_gain_rates.pt"),
        "cache_manifest": file_sha256(cache_dir / "manifest.json"),
        "atc_selection_predictions": file_sha256(atc_selection_path),
        "fbc_selection_predictions": file_sha256(fbc_selection_path),
        "atc_selection_path": str(atc_selection_path),
        "fbc_selection_path": str(fbc_selection_path),
        "prior": {name: file_sha256(prior_dir / name) for name in PRIOR_FILES},
        "inner_train_indices": sha256_fingerprint(inner_train.tolist()),
        "inner_validation_indices": sha256_fingerprint(inner_validation.tolist()),
    }
    rows = []
    for candidate_id in candidate_ids:
        candidate = dict(candidates[candidate_id])
        candidate_config = apply_delay_hpo_candidate(base_config, candidate)
        contract = _candidate_contract(
            campaign_contract_sha256=str(campaign_contract["combined_sha256"]),
            stage=stage,
            stage_config=stage_config,
            candidate=candidate,
            subject=subject,
            fold=fold,
            shared_input_sha256=shared_input_sha256,
        )
        directory = _candidate_dir(output, stage, subject, fold, candidate_id)
        completed = _load_completed_candidate(directory, contract)
        if completed is not None:
            rows.append(completed)
            continue
        ensure_dir(directory)
        contract_path = directory / "candidate_contract.json"
        if contract_path.is_file() and read_json(contract_path) != contract:
            raise RuntimeError("partial E5 candidate contract changed")
        allowed_partial = {
            "candidate_contract.json",
            "best.pt",
            "last.pt",
            "history.csv",
            "metrics.json",
            "validation_predictions.npz",
        }
        extras = {
            path.name for path in directory.iterdir() if path.is_file()
        }.difference(allowed_partial)
        if extras:
            raise RuntimeError(f"E5 candidate contains unknown partial artifacts: {extras}")
        write_json(contract_path, contract)
        run_seed = _candidate_seed(subject, fold, classifier_seed)
        model = _build_expert(str(config["variant"]), candidate_config, seed=run_seed)
        model.load_fold_prior(**prior)
        fit_delay_expert_input_gain(
            model,
            train_rates,
            device=args.device,
            batch_size=int(base_config["training"]["batch_size"]),
        )
        fit = fit_v8_delay_expert(
            model,
            train_rates,
            labels[inner_train],
            validation_rates=validation_rates,
            validation_labels=labels[inner_validation],
            validation_anchor_probability=validation_anchor,
            maximum_residual_weight=float(
                base_config["residual_fusion"]["maximum_expert_weight"]
            ),
            delay_override="zero",
            device=args.device,
            seed=run_seed,
            epochs=int(stage_config["max_epochs"]),
            patience=int(stage_config["patience"]),
            minimum_epochs=int(stage_config["minimum_epochs"]),
            scheduler_epochs=int(config["fixed_contract"]["scheduler_epochs"]),
            batch_size=int(config["fixed_contract"]["batch_size"]),
            gradient_accumulation_steps=int(
                config["fixed_contract"]["gradient_accumulation_steps"]
            ),
            learning_rate=float(candidate["learning_rate"]),
            weight_decay=float(config["fixed_contract"]["weight_decay"]),
            firing_rate_weight=float(config["fixed_contract"]["firing_rate_weight"]),
            run_label=f"V8-E5:{stage}:{candidate_id}:S{subject}:fold{fold}",
        )
        best = dict(fit.history[int(fit.best_epoch) - 1])
        validation_evaluation = predict_v8_delay_expert(
            fit.model,
            validation_rates,
            labels[inner_validation],
            delay_override="zero",
            device=args.device,
            batch_size=int(config["fixed_contract"]["batch_size"]),
        )
        validation_targets = np.asarray(validation_evaluation["labels"], dtype=np.int64)
        if not np.array_equal(validation_targets, labels[inner_validation]):
            raise RuntimeError("E5 best checkpoint changed inner-validation label order")
        expert_probability = probability_from_logits(validation_evaluation["logits"])
        fused_probability, residual_gate = entropy_residual_probability(
            validation_anchor,
            expert_probability,
            maximum_weight=float(
                base_config["residual_fusion"]["maximum_expert_weight"]
            ),
        )
        fused_metrics = classification_metrics(
            validation_targets,
            fused_probability.argmax(axis=1),
            n_classes=4,
        )
        expert_metrics = classification_metrics(
            validation_targets,
            expert_probability.argmax(axis=1),
            n_classes=4,
        )
        validation_nll = float(
            -np.log(
                np.clip(
                    fused_probability[
                        np.arange(validation_targets.size), validation_targets
                    ],
                    1e-12,
                    1.0,
                )
            ).mean()
        )
        expected_best = (
            float(best["validation_kappa"]),
            float(best["validation_accuracy"]),
            float(best["validation_nll"]),
        )
        recomputed_best = (
            float(fused_metrics["kappa"]),
            float(fused_metrics["accuracy"]),
            validation_nll,
        )
        if not np.allclose(expected_best, recomputed_best, rtol=0.0, atol=1e-7):
            raise RuntimeError("E5 best-checkpoint metrics do not reproduce from predictions")
        metrics = {
            "status": "completed",
            "stage": stage,
            "candidate_id": candidate_id,
            "candidate": candidate,
            "subject": subject,
            "fold": fold,
            "inner_validation_run": inner_run,
            "best_epoch": int(fit.best_epoch),
            "validation_kappa": float(fused_metrics["kappa"]),
            "validation_accuracy": float(fused_metrics["accuracy"]),
            "validation_nll": validation_nll,
            "validation_expert_kappa": float(expert_metrics["kappa"]),
            "validation_expert_accuracy": float(expert_metrics["accuracy"]),
            "validation_firing_rate": float(
                validation_evaluation["mean_firing_rate"]
            ),
            "parameters": fit.model.parameter_count,
            "optimizer_steps": int(fit.optimizer_steps),
            "delay_override": "zero",
            "full_delay_metrics_used": False,
            "outer_test_metrics_used": False,
            "prior_sha256": prior_summary["prior_sha256"],
            "prior_routes": int(prior_summary["selected_sparse_routes"]),
            "prior_stability_passed": True,
            "input_gain_sha256": mapping_sha256(
                {"input_gain": fit.model.input_gain}
            )["input_gain"],
            "cache_manifest": cache_manifest,
            "source_tree_sha256": source_digest,
            "session_e_accessed": False,
            "openbmi_s2_accessed": False,
        }
        torch.save(fit.best_state, directory / "best.pt")
        torch.save(fit.last_state, directory / "last.pt")
        write_csv(directory / "history.csv", fit.history)
        write_json(directory / "metrics.json", metrics)
        np.savez_compressed(
            directory / "validation_predictions.npz",
            indices=np.asarray(inner_validation, dtype=np.int64),
            labels=validation_targets,
            anchor_probability=np.asarray(validation_anchor, dtype=np.float32),
            expert_logits=np.asarray(validation_evaluation["logits"], dtype=np.float32),
            expert_probability=np.asarray(expert_probability, dtype=np.float32),
            fused_probability=np.asarray(fused_probability, dtype=np.float32),
            residual_gate=np.asarray(residual_gate, dtype=np.float32),
        )
        write_run_artifact_manifest(directory, required_files=CANDIDATE_FILES)
        rows.append(metrics)
    return rows


def _worker_command(
    *,
    args: argparse.Namespace,
    stage: str,
    subject: int,
    fold: int,
    candidate_ids: Sequence[str],
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--data",
        str(Path(args.data).resolve()),
        "--cache-parent",
        str(Path(args.cache_parent).resolve()),
        "--atc-root",
        str(Path(args.atc_root).resolve()),
        "--fbc-root",
        str(Path(args.fbc_root).resolve()),
        "--output",
        str(Path(args.output).resolve()),
        "--config",
        str(Path(args.config).resolve()),
        "--model-config",
        str(Path(args.model_config).resolve()),
        "--device",
        str(args.device),
        "--worker-stage",
        str(stage),
        "--worker-subject",
        str(subject),
        "--worker-fold",
        str(fold),
        "--worker-candidates",
        ",".join(candidate_ids),
    ]


def _run_worker(command: Sequence[str], log_out: Path, log_err: Path, threads: int) -> None:
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = str(threads)
    with log_out.open("a", encoding="utf-8") as stdout, log_err.open(
        "a", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            list(command),
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        tail = log_err.read_text(encoding="utf-8", errors="replace")[-8000:]
        raise RuntimeError(
            f"E5 worker failed with code {completed.returncode}: {' '.join(command)}\n{tail}"
        )


def _stage_rows(
    output: Path,
    *,
    stage: str,
    subjects: Sequence[int],
    folds: Sequence[int],
    candidate_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    for subject in subjects:
        for fold in folds:
            for candidate_id in candidate_ids:
                directory = _candidate_dir(output, stage, subject, fold, candidate_id)
                validate_run_artifact_manifest(
                    directory,
                    required_files=CANDIDATE_FILES,
                    verify_hashes=True,
                    verify_prediction_schema=False,
                )
                rows.append(read_json(directory / "metrics.json"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache-parent", required=True)
    parser.add_argument("--atc-root", required=True)
    parser.add_argument("--fbc-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config",
        default="configs/experiments/v8_e5_matched_transport_hpo.yaml",
    )
    parser.add_argument("--model-config", default="configs/models/v8_accuracy_first.yaml")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worker-stage", default="")
    parser.add_argument("--worker-subject", type=int, default=0)
    parser.add_argument("--worker-fold", type=int, default=-1)
    parser.add_argument("--worker-candidates", default="")
    args = parser.parse_args()

    configure_cache_env()
    config_path = Path(args.config).resolve()
    model_config_path = Path(args.model_config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA:
        raise RuntimeError("E5 matched-transport HPO schema changed")
    if config.get("delay_override_for_selection") != "zero":
        raise RuntimeError("E5 HPO is forbidden from selecting on full-delay metrics")
    if config.get("data_access") != {
        "allowed_session": "T",
        "heldout_session_e_accessed": False,
        "openbmi_session_s2_accessed": False,
    }:
        raise RuntimeError("E5 held-out data lock changed")
    base_config_path = (ROOT / str(config["fixed_contract"]["base_config"])).resolve()
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    candidates = enumerate_delay_hpo_candidates(config)
    candidate_by_id = _candidate_index(candidates)
    source_tree = collect_source_tree_manifest(ROOT)
    source_digest = source_tree_digest(source_tree)
    subjects = [int(value) for value in config["subjects"]]
    classifier_seed = int(config["classifier_seed"])
    inputs = _input_inventory(
        data_root=Path(args.data).resolve(),
        cache_parent=Path(args.cache_parent).resolve(),
        atc_root=Path(args.atc_root).resolve(),
        fbc_root=Path(args.fbc_root).resolve(),
        subjects=subjects,
        seeds=[classifier_seed],
        folds=sorted(
            {
                int(fold)
                for stage in config["successive_halving"].values()
                for fold in stage["active_folds"]
            }
        ),
    )
    campaign_contract = _campaign_contract(
        source_digest=source_digest,
        config_path=config_path,
        base_config_path=base_config_path,
        model_config_path=model_config_path,
        inputs=inputs,
        candidates=candidates,
        device=args.device,
    )
    output = Path(args.output).resolve()
    _prepare_root(output, campaign_contract)

    if args.worker_stage:
        rows = _worker(
            args=args,
            config=config,
            base_config=base_config,
            model_config=model_config,
            source_digest=source_digest,
            campaign_contract=campaign_contract,
            candidates=candidate_by_id,
        )
        print(json.dumps({"status": "completed", "rows": rows}, indent=2))
        return

    if (output / "manifest.json").is_file():
        manifest = read_json(output / "manifest.json")
        validate_run_artifact_manifest(
            output,
            required_files=tuple(manifest["required_files"]),
            verify_hashes=True,
            verify_prediction_schema=False,
        )
        print(json.dumps({"status": "already_complete", "output": str(output)}))
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    workers = int(args.workers or config["execution"]["workers"])
    if workers < 1 or workers > 8:
        raise ValueError("E5 workers must lie in [1, 8]")
    threads = int(config["execution"]["worker_cpu_threads"])
    write_json(output / "source_tree_manifest.json", source_tree)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {**config, "workers": workers, "device": args.device}, sort_keys=False
        ),
        encoding="utf-8",
    )
    write_csv(output / "candidate_ledger.csv", candidates)
    log_root = ensure_dir(output / "_logs")

    active_ids = [str(row["candidate_id"]) for row in candidates]
    stage_artifacts: list[str] = []
    stage_names = list(config["successive_halving"])
    for stage in stage_names:
        stage_config = config["successive_halving"][stage]
        folds = [int(value) for value in stage_config["active_folds"]]
        jobs = []
        for subject in subjects:
            for fold in folds:
                command = _worker_command(
                    args=args,
                    stage=stage,
                    subject=subject,
                    fold=fold,
                    candidate_ids=active_ids,
                )
                jobs.append(
                    (
                        command,
                        log_root / f"{stage}_s{subject:02d}_fold{fold}.stdout.log",
                        log_root / f"{stage}_s{subject:02d}_fold{fold}.stderr.log",
                    )
                )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_worker, command, out, err, threads): command
                for command, out, err in jobs
            }
            for future in as_completed(futures):
                future.result()
                print(
                    "__V8_E5_WORKER_DONE__ " + " ".join(futures[future][-8:]),
                    flush=True,
                )
        rows = _stage_rows(
            output,
            stage=stage,
            subjects=subjects,
            folds=folds,
            candidate_ids=active_ids,
        )
        ranking = rank_delay_hpo_candidates(
            rows,
            candidate_ids=active_ids,
            subjects=subjects,
            folds=folds,
        )
        ranking_path = output / f"{stage}_ranking.csv"
        promotion_path = output / f"{stage}_promotion.json"
        write_csv(ranking_path, ranking)
        promote_top_k = int(stage_config["promote_top_k"])
        promoted = [str(row["candidate_id"]) for row in ranking[:promote_top_k]]
        write_json(
            promotion_path,
            {
                "status": "completed",
                "stage": stage,
                "input_candidates": active_ids,
                "active_folds": folds,
                "promote_top_k": promote_top_k,
                "promoted_candidates": promoted,
                "selection_uses_inner_validation_only": True,
                "full_delay_metrics_used": False,
                "outer_test_metrics_used": False,
                "session_e_accessed": False,
                "openbmi_s2_accessed": False,
            },
        )
        stage_artifacts.extend([ranking_path.name, promotion_path.name])
        active_ids = promoted

    if len(active_ids) != 1:
        raise RuntimeError("E5 final stage did not select exactly one candidate")
    selected_id = active_ids[0]
    selected_candidate = candidate_by_id[selected_id]
    selected_config = apply_delay_hpo_candidate(base_config, selected_candidate)
    selected_config["experiment_id"] = "V8_E3_MATCHED_TRANSPORT_HPO_SELECTED"
    selected_config["hpo_selection"] = {
        "campaign_contract_sha256": campaign_contract["combined_sha256"],
        "candidate_id": selected_id,
        "candidate": selected_candidate,
        "selection_protocol": "Session-T inner-validation only",
        "requires_fresh_formal_outer_oof": True,
    }
    selected_path = output / str(config["output_contract"]["selected_e3_config"])
    selected_path.write_text(
        yaml.safe_dump(selected_config, sort_keys=False), encoding="utf-8"
    )
    selected = {
        "status": "completed",
        "candidate_id": selected_id,
        "candidate": selected_candidate,
        "selected_e3_config": str(selected_path),
        "selected_e3_config_sha256": file_sha256(selected_path),
        "selection_uses_inner_validation_only": True,
        "full_delay_metrics_used": False,
        "outer_test_metrics_used": False,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    write_json(output / "selected_candidate.json", selected)
    status = {
        "status": "completed",
        "stage": "E5_MATCHED_TRANSPORT_HPO",
        "protocol": str(config["protocol"]),
        "source_tree_sha256": source_digest,
        "campaign_contract_sha256": campaign_contract["combined_sha256"],
        "unique_candidates": len(candidates),
        "stages": stage_names,
        "selected_candidate": selected_id,
        "selected_e3_config_sha256": file_sha256(selected_path),
        "selection_uses_inner_validation_only": True,
        "requires_fresh_formal_e3_outer_oof": True,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    write_json(output / "campaign_status.json", status)

    candidate_artifacts = []
    active_by_stage = [str(row["candidate_id"]) for row in candidates]
    for stage in stage_names:
        folds = [int(value) for value in config["successive_halving"][stage]["active_folds"]]
        promotion = read_json(output / f"{stage}_promotion.json")
        for subject in subjects:
            for fold in folds:
                for candidate_id in active_by_stage:
                    relative = _candidate_dir(
                        output, stage, subject, fold, candidate_id
                    ).relative_to(output)
                    candidate_artifacts.extend(
                        [
                            (relative / "manifest.json").as_posix(),
                            (relative / "candidate_contract.json").as_posix(),
                            (relative / "metrics.json").as_posix(),
                        ]
                    )
        active_by_stage = [str(value) for value in promotion["promoted_candidates"]]
    write_run_artifact_manifest(
        output,
        required_files=ROOT_FILES
        + tuple(stage_artifacts)
        + tuple(sorted(candidate_artifacts)),
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
