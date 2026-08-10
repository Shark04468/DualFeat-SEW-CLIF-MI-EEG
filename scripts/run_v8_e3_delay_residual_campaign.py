#!/usr/bin/env python3
"""Run the registered V8 matched-transport E3 campaign with strict resume."""

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

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_anchor import e1_selection_prediction_path  # noqa: E402
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    file_sha256,
    sha256_fingerprint,
    source_tree_digest,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from scripts.run_v8_e1_baselines import _subject_file  # noqa: E402


CONFIG_SCHEMA = "dpc-snn-v8-e3-delay-residual-campaign/v1"
CAMPAIGN_FILES = (
    "manifest.json",
    "campaign_contract.json",
    "campaign_progress.json",
    "campaign_status.json",
    "resolved_config.yaml",
    "source_tree_manifest.json",
    "summary.csv",
)


def _csv(value: str, cast: Any = str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _registered_values(
    override: str,
    registered: Sequence[Any],
    *,
    cast: Any,
    name: str,
) -> list[Any]:
    values = _csv(override, cast) if override else [cast(value) for value in registered]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{name} must be a non-empty unique sequence")
    if set(values).difference(cast(value) for value in registered):
        raise ValueError(f"{name} contains values outside the registered campaign")
    return values


def _input_inventory(
    *,
    data_root: Path,
    cache_parent: Path,
    atc_root: Path,
    fbc_root: Path,
    subjects: Sequence[int],
    seeds: Sequence[int],
    folds: Sequence[int],
) -> dict[str, str]:
    paths: dict[str, Path] = {}
    for subject in subjects:
        paths[f"data/subject_{subject:02d}"] = _subject_file(data_root, int(subject))
        cache = cache_parent / "shared_physical_rates" / f"subject_{subject:02d}"
        paths[f"cache/subject_{subject:02d}/rates"] = cache / "unit_gain_rates.pt"
        paths[f"cache/subject_{subject:02d}/manifest"] = cache / "manifest.json"
        for seed in seeds:
            paths[f"atc/subject_{subject:02d}/seed_{seed}"] = (
                atc_root
                / "atcnet"
                / f"subject_{subject:02d}"
                / f"seed_{seed}"
                / "predictions.npz"
            )
            paths[f"fbc/subject_{subject:02d}/seed_{seed}"] = (
                fbc_root
                / "fbcnet"
                / f"subject_{subject:02d}"
                / f"seed_{seed}"
                / "predictions.npz"
            )
            for fold in folds:
                paths[
                    f"atc/subject_{subject:02d}/seed_{seed}/fold_{fold}/selection"
                ] = e1_selection_prediction_path(
                    atc_root,
                    model="atcnet",
                    subject=int(subject),
                    seed=int(seed),
                    fold=int(fold),
                )
                paths[
                    f"fbc/subject_{subject:02d}/seed_{seed}/fold_{fold}/selection"
                ] = e1_selection_prediction_path(
                    fbc_root,
                    model="fbcnet",
                    subject=int(subject),
                    seed=int(seed),
                    fold=int(fold),
                )
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("E3 campaign inputs are incomplete: " + ", ".join(missing))
    return {name: file_sha256(path) for name, path in sorted(paths.items())}


def build_campaign_contract(
    *,
    source_digest: str,
    config_path: Path,
    model_config_path: Path,
    inputs: Mapping[str, str],
    subjects: Sequence[int],
    seeds: Sequence[int],
    folds: Sequence[int],
    variants: Sequence[str],
    device: str,
    canary: bool,
) -> dict[str, Any]:
    payload = {
        "schema": "dpc-snn-v8-e3-delay-residual-campaign-contract/v1",
        "source_tree_sha256": str(source_digest),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "model_config_path": str(model_config_path),
        "model_config_sha256": file_sha256(model_config_path),
        "input_sha256": dict(sorted(inputs.items())),
        "subjects": [int(value) for value in subjects],
        "seeds": [int(value) for value in seeds],
        "folds": [int(value) for value in folds],
        "variants": [str(value) for value in variants],
        "device": str(device),
        "canary": bool(canary),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
        },
    }
    return {**payload, "combined_sha256": sha256_fingerprint(payload)}


def _prepare_output(output: Path, contract: Mapping[str, Any]) -> bool:
    contract_path = output / "campaign_contract.json"
    if output.exists() and any(output.iterdir()):
        if not contract_path.is_file() or read_json(contract_path) != dict(contract):
            raise RuntimeError("non-empty E3 campaign has no exact resume contract")
        return True
    ensure_dir(output)
    write_json(contract_path, dict(contract))
    return False


def _fold_dir(output: Path, subject: int, seed: int, fold: int) -> Path:
    return output / f"subject_{subject:02d}" / f"seed_{seed}" / f"fold_{fold}"


def _validate_fold(
    directory: Path,
    *,
    source_digest: str,
    subject: int,
    seed: int,
    fold: int,
    variants: Sequence[str],
) -> list[dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"E3 fold is incomplete: {directory}")
    manifest = read_json(manifest_path)
    validate_run_artifact_manifest(
        directory,
        required_files=tuple(manifest["required_files"]),
        verify_hashes=True,
        verify_prediction_schema=False,
    )
    status = read_json(directory / "campaign_status.json")
    expected = {
        "status": "completed",
        "stage": "E3_delay_residual_fold_canary",
        "subject": int(subject),
        "seed": int(seed),
        "fold": int(fold),
        "variants": [str(value) for value in variants],
        "source_tree_sha256": str(source_digest),
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    for field, value in expected.items():
        if status.get(field) != value:
            raise RuntimeError(
                f"E3 fold status mismatch at {directory}: {field}={status.get(field)!r}"
            )
    metrics = read_json(directory / "metrics.json")
    rows = list(metrics.get("rows", []))
    if metrics.get("session_e_accessed") is not False or len(rows) != len(variants):
        raise RuntimeError(f"E3 fold metrics are incomplete: {directory}")
    by_variant = {str(row["variant"]): row for row in rows}
    if set(by_variant) != set(variants):
        raise RuntimeError(f"E3 fold variant coverage changed: {directory}")
    return [dict(by_variant[variant]) for variant in variants]


def _tail(path: Path, limit: int = 8000) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-int(limit) :]


def _run_fold(
    *,
    output: Path,
    data_root: Path,
    cache_parent: Path,
    atc_root: Path,
    fbc_root: Path,
    config_path: Path,
    model_config_path: Path,
    source_digest: str,
    subject: int,
    seed: int,
    fold: int,
    variants: Sequence[str],
    device: str,
    prior_source: Path | None,
    cpu_threads: int,
) -> tuple[tuple[int, int, int], list[dict[str, Any]], bool]:
    directory = _fold_dir(output, subject, seed, fold)
    if (directory / "manifest.json").is_file():
        return (
            (subject, seed, fold),
            _validate_fold(
                directory,
                source_digest=source_digest,
                subject=subject,
                seed=seed,
                fold=fold,
                variants=variants,
            ),
            True,
        )
    log_root = ensure_dir(output / "_logs")
    stdout_path = log_root / f"s{subject:02d}_seed{seed}_fold{fold}.stdout.log"
    stderr_path = log_root / f"s{subject:02d}_seed{seed}_fold{fold}.stderr.log"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_v8_e3_delay_residual_fold.py"),
        "--data",
        str(data_root),
        "--cache-parent",
        str(cache_parent),
        "--atc-root",
        str(atc_root),
        "--fbc-root",
        str(fbc_root),
        "--output",
        str(directory),
        "--config",
        str(config_path),
        "--model-config",
        str(model_config_path),
        "--subject",
        str(subject),
        "--seed",
        str(seed),
        "--fold",
        str(fold),
        "--variants",
        ",".join(variants),
        "--device",
        str(device),
    ]
    if prior_source is not None:
        command.extend(["--import-fold-priors", str(prior_source)])
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = str(cpu_threads)
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"E3 fold failed with code {completed.returncode}: "
            f"subject={subject}, seed={seed}, fold={fold}\n{_tail(stderr_path)}"
        )
    return (
        (subject, seed, fold),
        _validate_fold(
            directory,
            source_digest=source_digest,
            subject=subject,
            seed=seed,
            fold=fold,
            variants=variants,
        ),
        False,
    )


def _run_phase(
    jobs: Sequence[dict[str, Any]],
    *,
    workers: int,
    progress: dict[str, Any],
    progress_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_fold, **job): job for job in jobs}
        for future in as_completed(futures):
            key, fold_rows, resumed = future.result()
            rows.extend(fold_rows)
            marker = f"S{key[0]:02d}/seed{key[1]}/fold{key[2]}"
            progress["completed"] = sorted(set(progress["completed"]) | {marker})
            progress["completed_fold_runs"] = len(progress["completed"])
            write_json(progress_path, progress)
            print(
                "__V8_E3_MATCHED_FOLD_DONE__ "
                f"subject={key[0]} seed={key[1]} fold={key[2]} resumed={int(resumed)}",
                flush=True,
            )
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
        default="configs/experiments/v8_e3_matched_transport_campaign.yaml",
    )
    parser.add_argument("--model-config", default="configs/models/v8_accuracy_first.yaml")
    parser.add_argument("--subjects", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--folds", default="")
    parser.add_argument("--variants", default="")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    model_config_path = Path(args.model_config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA:
        raise RuntimeError("E3 matched-transport campaign schema changed")
    if config.get("protocol") != "bci2a_session_t_nested_six_fold_oof":
        raise RuntimeError("E3 matched-transport campaign protocol changed")
    if config.get("data_access") != {
        "allowed_session": "T",
        "heldout_session_e_accessed": False,
        "openbmi_session_s2_accessed": False,
    }:
        raise RuntimeError("E3 matched-transport held-out lock changed")
    subjects = _registered_values(
        args.subjects, config["subjects"], cast=int, name="subjects"
    )
    seeds = _registered_values(args.seeds, config["seeds"], cast=int, name="seeds")
    folds = _registered_values(args.folds, config["folds"], cast=int, name="folds")
    variants = _registered_values(
        args.variants, config["variants"], cast=str, name="variants"
    )
    full_contract = (
        subjects == list(config["subjects"])
        and seeds == list(config["seeds"])
        and folds == list(config["folds"])
        and variants == list(config["variants"])
    )
    if not full_contract and not args.canary:
        raise RuntimeError("partial E3 campaign execution requires --canary")
    workers = int(args.workers or config["execution"]["workers"])
    if workers < 1 or workers > 8:
        raise ValueError("E3 campaign workers must lie in [1, 8]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")

    source_tree = collect_source_tree_manifest(ROOT)
    source_digest = source_tree_digest(source_tree)
    data_root = Path(args.data).resolve()
    cache_parent = Path(args.cache_parent).resolve()
    atc_root = Path(args.atc_root).resolve()
    fbc_root = Path(args.fbc_root).resolve()
    inputs = _input_inventory(
        data_root=data_root,
        cache_parent=cache_parent,
        atc_root=atc_root,
        fbc_root=fbc_root,
        subjects=subjects,
        seeds=seeds,
        folds=folds,
    )
    contract = build_campaign_contract(
        source_digest=source_digest,
        config_path=config_path,
        model_config_path=model_config_path,
        inputs=inputs,
        subjects=subjects,
        seeds=seeds,
        folds=folds,
        variants=variants,
        device=args.device,
        canary=bool(args.canary),
    )
    output = Path(args.output).resolve()
    _prepare_output(output, contract)
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
    source_path = output / "source_tree_manifest.json"
    if source_path.is_file() and read_json(source_path) != source_tree:
        raise RuntimeError("E3 campaign source manifest changed during resume")
    if not source_path.is_file():
        write_json(source_path, source_tree)
    resolved_path = output / "resolved_config.yaml"
    resolved_text = yaml.safe_dump(
        {
            **config,
            "active_subjects": subjects,
            "active_seeds": seeds,
            "active_folds": folds,
            "active_variants": variants,
            "workers": workers,
            "device": args.device,
            "canary": bool(args.canary),
        },
        sort_keys=False,
    )
    if resolved_path.is_file() and resolved_path.read_text(encoding="utf-8") != resolved_text:
        raise RuntimeError("E3 campaign resolved config changed during resume")
    if not resolved_path.is_file():
        resolved_path.write_text(resolved_text, encoding="utf-8")

    progress_path = output / "campaign_progress.json"
    progress = (
        read_json(progress_path)
        if progress_path.is_file()
        else {
            "schema": "dpc-snn-v8-e3-delay-residual-campaign-progress/v1",
            "status": "running",
            "expected_fold_runs": len(subjects) * len(seeds) * len(folds),
            "completed_fold_runs": 0,
            "completed": [],
            "session_e_accessed": False,
            "openbmi_s2_accessed": False,
        }
    )
    canonical_seed = int(config["execution"]["canonical_prior_seed"])
    reuse_priors = bool(config["execution"]["reuse_fold_priors_across_classifier_seeds"])
    if reuse_priors and canonical_seed not in seeds:
        raise RuntimeError("canonical prior seed must be present when prior reuse is enabled")
    cpu_threads = int(config["execution"]["worker_cpu_threads"])

    def job(subject: int, seed: int, fold: int) -> dict[str, Any]:
        prior_source = None
        if reuse_priors and seed != canonical_seed:
            prior_source = _fold_dir(output, subject, canonical_seed, fold)
        return {
            "output": output,
            "data_root": data_root,
            "cache_parent": cache_parent,
            "atc_root": atc_root,
            "fbc_root": fbc_root,
            "config_path": config_path,
            "model_config_path": model_config_path,
            "source_digest": source_digest,
            "subject": int(subject),
            "seed": int(seed),
            "fold": int(fold),
            "variants": variants,
            "device": str(args.device),
            "prior_source": prior_source,
            "cpu_threads": cpu_threads,
        }

    rows: list[dict[str, Any]] = []
    canonical_jobs = [
        job(subject, canonical_seed, fold) for subject in subjects for fold in folds
    ]
    rows.extend(
        _run_phase(
            canonical_jobs,
            workers=workers,
            progress=progress,
            progress_path=progress_path,
        )
    )
    remaining_jobs = [
        job(subject, seed, fold)
        for subject in subjects
        for seed in seeds
        if seed != canonical_seed
        for fold in folds
    ]
    if remaining_jobs:
        rows.extend(
            _run_phase(
                remaining_jobs,
                workers=workers,
                progress=progress,
                progress_path=progress_path,
            )
        )

    expected_runs = len(subjects) * len(seeds) * len(folds)
    expected_rows = expected_runs * len(variants)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"E3 campaign produced {len(rows)} rows; expected {expected_rows}"
        )
    rows.sort(key=lambda row: (int(row["subject"]), int(row["seed"]), int(row["fold"]), str(row["variant"])))
    write_csv(output / "summary.csv", rows)
    progress["status"] = "completed"
    progress["completed_fold_runs"] = expected_runs
    write_json(progress_path, progress)
    status = {
        "status": "completed",
        "stage": "E3_DELAY_RESIDUAL_CAMPAIGN",
        "protocol": "bci2a_session_t_nested_six_fold_oof",
        "source_tree_sha256": source_digest,
        "campaign_contract_sha256": contract["combined_sha256"],
        "subjects": subjects,
        "seeds": seeds,
        "folds": folds,
        "variants": variants,
        "fold_runs": expected_runs,
        "variant_fold_rows": expected_rows,
        "full_registered_contract": bool(full_contract),
        "canary": bool(args.canary),
        "prior_reuse": {
            "enabled": reuse_priors,
            "canonical_classifier_seed": canonical_seed,
            "scientific_seed_scope": config["delay"]["prior_seed_scope"],
        },
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    write_json(output / "campaign_status.json", status)
    shard_files = []
    for subject in subjects:
        for seed in seeds:
            for fold in folds:
                relative = _fold_dir(output, subject, seed, fold).relative_to(output)
                shard_files.extend(
                    [
                        (relative / "manifest.json").as_posix(),
                        (relative / "metrics.json").as_posix(),
                        (relative / "run_manifest.json").as_posix(),
                    ]
                )
    write_run_artifact_manifest(
        output,
        required_files=CAMPAIGN_FILES + tuple(sorted(shard_files)),
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
