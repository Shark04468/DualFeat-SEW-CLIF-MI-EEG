#!/usr/bin/env python3
"""Evaluate frozen V8 ensemble utility without changing any model state."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.analysis.v8_utility import (  # noqa: E402
    add_gaussian_noise_at_snr,
    drop_eeg_channels,
    fit_logit_calibrator,
    multiclass_calibration_metrics,
    stratified_kshot_indices,
    utility_win_summary,
)
from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v62_baselines import task_carrier  # noqa: E402
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_ensemble import (  # noqa: E402
    validate_ensemble_freeze_contract,
)
from dpc_snn.experiments.v8_ensemble_followup import (  # noqa: E402
    component_state_digests,
    decoder_operation_proxy,
    load_ensemble_components,
    load_fixed_gain,
    mask_carrier_after_endpoint,
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


ARMS = ("primary", "matched_ann")
RUN_FILES = (
    "manifest.json",
    "source_fingerprint.json",
    "source_tree_manifest.json",
    "resolved_config.yaml",
    "metrics.json",
    "clean_replay.json",
    "calibration.csv",
    "kshot.csv",
    "early_decision.csv",
    "early_predictions.npz",
    "robustness.csv",
    "robustness_predictions.npz",
    "operation_proxy.json",
    "state_audit.json",
    "runtime_status.json",
)
CAMPAIGN_FILES = (
    "manifest.json",
    "campaign_status.json",
    "summary.csv",
    "calibration_summary.csv",
    "kshot_summary.csv",
    "early_decision_summary.csv",
    "robustness_summary.csv",
    "operation_proxy_summary.csv",
    "paired_subject_seed.csv",
    "gate_decision.json",
    "source_tree_manifest.json",
    "resolved_campaign.yaml",
)


def _csv_values(value: str, cast: Any = str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"probabilities", "label", "trial_id", "subject", "seed"}
        if not required.issubset(archive.files):
            raise RuntimeError(f"E6 prediction schema is incomplete: {path}")
        return {name: archive[name] for name in archive.files}


def _assert_clean_replay(
    replay: dict[str, Any], e6_run: Path, *, tolerance: float = 1e-5
) -> dict[str, float]:
    mapping = {
        "primary": "predictions.npz",
        "matched_ann": "matched_ann_predictions.npz",
        "anchor": "anchor_predictions.npz",
        "atcnet": "atcnet_predictions.npz",
        "fbcnet": "fbcnet_predictions.npz",
        "decoder_snn": "decoder_snn_predictions.npz",
        "decoder_ann": "decoder_ann_predictions.npz",
    }
    maximum: dict[str, float] = {}
    for arm, filename in mapping.items():
        expected = _prediction(e6_run / filename)
        actual = replay["arms"][arm]
        difference = float(
            np.max(np.abs(actual["probabilities"] - expected["probabilities"]))
        )
        if difference > float(tolerance):
            raise RuntimeError(
                f"E7 clean replay differs from E6 for {arm}: {difference:.3e}"
            )
        if not np.array_equal(actual["pred"], expected["pred"]):
            raise RuntimeError(f"E7 clean predictions differ from E6 for {arm}")
        maximum[arm] = difference
    return maximum


def _calibration_rows(
    replay: dict[str, Any], labels: np.ndarray, *, bins: int
) -> list[dict[str, Any]]:
    rows = []
    for arm in ARMS:
        metrics = multiclass_calibration_metrics(
            replay["arms"][arm]["logits"], labels, n_bins=int(bins)
        )
        rows.append({"arm": arm, **metrics})
    return rows


def _kshot_rows(
    replay: dict[str, Any],
    labels: np.ndarray,
    *,
    subject: int,
    seed: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in config["k_per_class"]:
        for repeat in range(int(config["repeats"])):
            split_seed = (
                int(subject) * 10_000_019
                + int(seed) * 100_003
                + int(k) * 1_009
                + repeat
            )
            calibration_indices, evaluation_indices = stratified_kshot_indices(
                labels, k_per_class=int(k), seed=split_seed
            )
            for arm in ARMS:
                logits = replay["arms"][arm]["logits"]
                calibrator = fit_logit_calibrator(
                    logits[calibration_indices],
                    labels[calibration_indices],
                    l2_bias=float(config["bias_l2"]),
                )
                calibrated = calibrator.apply(logits[evaluation_indices])
                calibrated_metrics = multiclass_calibration_metrics(
                    calibrated,
                    labels[evaluation_indices],
                    n_bins=int(config["confidence_bins"]),
                )
                raw_metrics = multiclass_calibration_metrics(
                    logits[evaluation_indices],
                    labels[evaluation_indices],
                    n_bins=int(config["confidence_bins"]),
                )
                rows.append(
                    {
                        "arm": arm,
                        "k_per_class": int(k),
                        "repeat": repeat,
                        "split_seed": split_seed,
                        "calibration_trials": int(calibration_indices.size),
                        "evaluation_trials": int(evaluation_indices.size),
                        "log_temperature": float(calibrator.log_temperature),
                        "centered_bias": json.dumps(calibrator.bias.tolist()),
                        "raw_accuracy": raw_metrics["accuracy"],
                        "calibrated_accuracy": calibrated_metrics["accuracy"],
                        "calibrated_nll": calibrated_metrics[
                            "negative_log_likelihood"
                        ],
                        "calibrated_ece": calibrated_metrics["ece"],
                    }
                )
    return rows


def _run_one(
    *,
    output: Path,
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
    checkpoint_hashes = {
        name: file_sha256(e6_run / f"{name}.pt")
        for name in ("atcnet", "fbcnet", "sew_clif", "ann_sew")
    }
    resolved = {
        "stage": "E7_ENSEMBLE",
        "protocol": config["protocol"],
        "subject": int(subject),
        "seed": int(seed),
        "freeze_sha256": freeze["combined_sha256"],
        "parent_e6_run_fingerprint": e6_metrics["run_fingerprint"],
        "utility": config,
    }
    fingerprint = build_v8_run_fingerprint(
        resolved_run_config=resolved,
        source_tree=source_tree,
        data={f"session_e/{evaluation_path.name}": file_sha256(evaluation_path)},
        split={
            "train_session": "T",
            "evaluation_session": "E",
            "role": "frozen_posthoc_utility",
            "model_updates": False,
        },
        augmentation={"policy": "none_during_frozen_utility"},
        prior={"policy": "none", "delay_enabled": False},
        checkpoint={
            "parent_e6_run_fingerprint": e6_metrics["run_fingerprint"],
            "component_sha256": checkpoint_hashes,
            "model_updates": False,
        },
        environment=environment,
    )
    run_dir = ensure_dir(output / f"subject_{subject:02d}" / f"seed_{seed}")
    fingerprint_path = run_dir / "source_fingerprint.json"
    if (run_dir / "manifest.json").is_file():
        validate_v8_resume_fingerprint(fingerprint_path, fingerprint)
        validate_run_artifact_manifest(run_dir, required_files=RUN_FILES, verify_hashes=True)
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
        {"status": "running", "started_at": time.time(), "model_updates": False},
    )

    model_config = dict(freeze["architecture"]["model_config"])
    components = load_ensemble_components(
        e6_run,
        source_root=source_root,
        model_config=model_config,
        n_classes=4,
    ).to(device).eval()
    before = component_state_digests(components)
    gain = load_fixed_gain(e6_run / "gain.npz")
    data = load_processed_npz(evaluation_path)
    metadata = _metadata(data, expected_session="E", role="evaluation")
    labels = np.asarray(data["y"], dtype=np.int64)
    trial_ids = np.asarray([row["trial_id"] for row in metadata])
    expected_primary = _prediction(e6_run / "predictions.npz")
    if not np.array_equal(trial_ids.astype(str), expected_primary["trial_id"].astype(str)):
        raise RuntimeError("E7 Session-E trial order differs from E6")
    sfreq = float(np.asarray(data["sfreq"]).item())
    epoch_tmin = float(np.asarray(data["epoch_tmin"]).item())
    carrier = task_carrier(
        np.asarray(data["X"], dtype=np.float32),
        sfreq=sfreq,
        epoch_tmin=epoch_tmin,
    )
    replay = predict_ensemble_from_carrier(
        components,
        carrier,
        labels,
        gain,
        sfreq=sfreq,
        model_config=model_config,
        n_classes=4,
        device=device,
    )
    replay_errors = _assert_clean_replay(replay, e6_run)
    write_json(
        run_dir / "clean_replay.json",
        {"maximum_probability_error": replay_errors, "all_predictions_identical": True},
    )

    calibration_rows = _calibration_rows(
        replay, labels, bins=int(config["calibration"]["confidence_bins"])
    )
    write_csv(run_dir / "calibration.csv", calibration_rows)
    kshot_rows = _kshot_rows(
        replay,
        labels,
        subject=subject,
        seed=seed,
        config=dict(config["calibration"]),
    )
    write_csv(run_dir / "kshot.csv", kshot_rows)

    endpoints = [float(value) for value in config["early_decision"]["endpoints_seconds"]]
    early_rows: list[dict[str, Any]] = []
    early_logits: list[np.ndarray] = []
    for endpoint in endpoints:
        if np.isclose(endpoint, carrier.shape[-1] / sfreq):
            result = replay
        else:
            result = predict_ensemble_from_carrier(
                components,
                mask_carrier_after_endpoint(
                    carrier, endpoint_seconds=endpoint, sfreq=sfreq
                ),
                labels,
                gain,
                sfreq=sfreq,
                model_config=model_config,
                n_classes=4,
                device=device,
            )
        early_logits.append(
            np.stack([result["arms"][arm]["logits"] for arm in ARMS])
        )
        for arm in ARMS:
            early_rows.append(
                {
                    "arm": arm,
                    "endpoint_seconds": endpoint,
                    **{
                        key: result["arms"][arm][key]
                        for key in (
                            "accuracy",
                            "balanced_accuracy",
                            "kappa",
                            "macro_f1",
                        )
                    },
                }
            )
    write_csv(run_dir / "early_decision.csv", early_rows)
    np.savez_compressed(
        run_dir / "early_predictions.npz",
        arms=np.asarray(ARMS),
        endpoint_seconds=np.asarray(endpoints, dtype=np.float32),
        logits=np.stack(early_logits).astype(np.float32),
        labels=labels,
        trial_id=trial_ids,
    )

    conditions: list[tuple[str, np.ndarray, list[int]]] = [("clean", carrier, [])]
    for index, snr in enumerate(config["robustness"]["gaussian_snr_db"]):
        perturb_seed = int(subject) * 1_000_003 + int(seed) * 10_009 + index
        conditions.append(
            (
                f"noise_{float(snr):g}dB",
                add_gaussian_noise_at_snr(
                    carrier, snr_db=float(snr), seed=perturb_seed
                ),
                [],
            )
        )
    for index, count in enumerate(config["robustness"]["channel_drop_counts"]):
        perturb_seed = int(subject) * 1_000_003 + int(seed) * 10_009 + 100 + index
        dropped_carrier, dropped = drop_eeg_channels(
            carrier, count=int(count), seed=perturb_seed
        )
        conditions.append(
            (f"drop_{int(count)}ch", dropped_carrier, dropped.astype(int).tolist())
        )
    robustness_rows: list[dict[str, Any]] = []
    robustness_logits: list[np.ndarray] = []
    for condition, perturbed, dropped in conditions:
        result = (
            replay
            if condition == "clean"
            else predict_ensemble_from_carrier(
                components,
                perturbed,
                labels,
                gain,
                sfreq=sfreq,
                model_config=model_config,
                n_classes=4,
                device=device,
            )
        )
        robustness_logits.append(
            np.stack([result["arms"][arm]["logits"] for arm in ARMS])
        )
        for arm in ARMS:
            robustness_rows.append(
                {
                    "arm": arm,
                    "condition": condition,
                    "dropped_channel_indices": json.dumps(dropped),
                    **{
                        key: result["arms"][arm][key]
                        for key in (
                            "accuracy",
                            "balanced_accuracy",
                            "kappa",
                            "macro_f1",
                        )
                    },
                }
            )
    write_csv(run_dir / "robustness.csv", robustness_rows)
    np.savez_compressed(
        run_dir / "robustness_predictions.npz",
        arms=np.asarray(ARMS),
        conditions=np.asarray([item[0] for item in conditions]),
        logits=np.stack(robustness_logits).astype(np.float32),
        labels=labels,
        trial_id=trial_ids,
    )

    operation = {
        "primary": decoder_operation_proxy(
            components.sew_clif,
            mean_binary_firing_rate=float(replay["snn_mean_firing_rate"]),
        ),
        "matched_ann": decoder_operation_proxy(
            components.ann_sew, mean_binary_firing_rate=None
        ),
    }
    operation["reduction"] = float(
        1.0
        - operation["primary"]["activity_weighted_decoder_events"]
        / operation["matched_ann"]["activity_weighted_decoder_events"]
    )
    write_json(run_dir / "operation_proxy.json", operation)
    after = component_state_digests(components)
    if before != after:
        raise RuntimeError("E7 changed a frozen E6 component state")
    write_json(
        run_dir / "state_audit.json",
        {
            "checkpoint_before_utility": before,
            "state_after_utility": after,
            "identical": True,
            "model_updates": False,
        },
    )

    primary_endpoint = float(config["early_decision"]["primary_endpoint_seconds"])
    metrics: dict[str, Any] = {
        "status": "completed",
        "stage": "E7_ENSEMBLE",
        "subject": int(subject),
        "seed": int(seed),
        "freeze_sha256": freeze["combined_sha256"],
        "parent_e6_run_fingerprint": e6_metrics["run_fingerprint"],
        "run_fingerprint": fingerprint["combined_sha256"],
        "model_updates": False,
        "arms": {},
        "operation_proxy_reduction": operation["reduction"],
    }
    for arm in ARMS:
        arm_calibration = next(row for row in calibration_rows if row["arm"] == arm)
        arm_early = next(
            row
            for row in early_rows
            if row["arm"] == arm
            and np.isclose(float(row["endpoint_seconds"]), primary_endpoint)
        )
        arm_robust = [
            row
            for row in robustness_rows
            if row["arm"] == arm and row["condition"] != "clean"
        ]
        arm_kshot = [row for row in kshot_rows if row["arm"] == arm]
        metrics["arms"][arm] = {
            "final_accuracy": float(replay["arms"][arm]["accuracy"]),
            "early_accuracy": float(arm_early["accuracy"]),
            "robustness_mean_accuracy": float(
                np.mean([float(row["accuracy"]) for row in arm_robust])
            ),
            "kshot_mean_accuracy": float(
                np.mean([float(row["calibrated_accuracy"]) for row in arm_kshot])
            ),
            "raw_ece": float(arm_calibration["ece"]),
            "raw_nll": float(arm_calibration["negative_log_likelihood"]),
            "activity_weighted_decoder_events": float(
                operation[arm]["activity_weighted_decoder_events"]
            ),
        }
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "runtime_status.json",
        {
            "status": "completed",
            "completed_at": time.time(),
            "model_updates": False,
            "session_e_accessed": True,
            "openbmi_s2_accessed": False,
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
        "--evaluation-data",
        str(Path(args.evaluation_data).resolve()),
        "--e6",
        str(Path(args.e6).resolve()),
        "--e6-audit",
        str(Path(args.e6_audit).resolve()),
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
    threads = int(environment.get("DPC_SNN_E7_WORKER_THREADS", "8"))
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
            f"E7 worker failed ({completed.returncode}): {' '.join(command)}\n"
            + completed.stderr[-8000:]
        )
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError("E7 worker missed outputs: " + ", ".join(missing))
    return [read_json(path) for path in expected]


def _merge_run_csv(
    output: Path, subjects: list[int], seeds: list[int], filename: str
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for subject in subjects:
        for seed in seeds:
            path = output / f"subject_{subject:02d}" / f"seed_{seed}" / filename
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    merged.append({"subject": subject, "seed": seed, **row})
    return merged


def _paired(rows: list[dict[str, Any]], field: str, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ann = [
        {"subject": row["subject"], "seed": row["seed"], "value": row["arms"]["matched_ann"][field]}
        for row in rows
    ]
    snn = [
        {"subject": row["subject"], "seed": row["seed"], "value": row["arms"]["primary"][field]}
        for row in rows
    ]
    pairs = pair_subject_seed_rows(ann, snn, value="value")
    return pairs, paired_delta_summary(pairs, seed=int(seed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-data", required=True)
    parser.add_argument("--e6", required=True)
    parser.add_argument("--e6-audit", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default="configs/experiments/v8_e7_ensemble_utility.yaml"
    )
    parser.add_argument("--subjects", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-subject", type=int, default=None)
    parser.add_argument("--worker-seeds", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()

    configure_cache_env()
    output = ensure_dir(Path(args.output).resolve())
    evaluation_root = Path(args.evaluation_data).resolve()
    e6 = Path(args.e6).resolve()
    source_root = Path(args.source_root).resolve()
    config = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    source_tree = collect_source_tree_manifest(ROOT)
    freeze = validate_ensemble_freeze_contract(
        validate_v8_freeze_manifest(Path(args.freeze).resolve())
    )
    audit = read_json(Path(args.e6_audit).resolve() / "audit_report.json")
    if (
        audit.get("status") != "passed"
        or audit.get("runs_audited") != 45
        or audit.get("freeze_sha256") != freeze["combined_sha256"]
        or audit.get("post_session_e_tuning_detected") is not False
    ):
        raise RuntimeError("E7 requires the passed full audit of the exact E6 campaign")
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
        rows = []
        for seed in seeds:
            rows.append(
                _run_one(
                    output=output,
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
            )
        print(json.dumps({"status": "worker_completed", "runs": len(rows)}, indent=2))
        return

    write_json(output / "source_tree_manifest.json", source_tree)
    (output / "resolved_campaign.yaml").write_text(
        yaml.safe_dump(
            {
                **config,
                "active_subjects": subjects,
                "active_seeds": seeds,
                "workers": int(args.workers),
                "freeze_sha256": freeze["combined_sha256"],
                "source_tree_sha256": source_tree_digest(source_tree),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    if int(args.workers) > 1 and len(subjects) > 1:
        with ThreadPoolExecutor(max_workers=min(int(args.workers), len(subjects))) as pool:
            future_map = {}
            for subject in subjects:
                expected = [
                    output / f"subject_{subject:02d}" / f"seed_{seed}" / "metrics.json"
                    for seed in seeds
                ]
                future = pool.submit(
                    _run_worker, _worker_command(args, output, subject, seeds), expected
                )
                future_map[future] = subject
            for future in as_completed(future_map):
                rows.extend(future.result())
    else:
        for subject in subjects:
            for seed in seeds:
                rows.append(
                    _run_one(
                        output=output,
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
    summary_rows = []
    for row in rows:
        summary_rows.append(
            {
                "subject": row["subject"],
                "seed": row["seed"],
                **{
                    f"{arm}_{field}": value
                    for arm in ARMS
                    for field, value in row["arms"][arm].items()
                },
                "operation_proxy_reduction": row["operation_proxy_reduction"],
            }
        )
    write_csv(output / "summary.csv", summary_rows)
    for filename, merged_name in (
        ("calibration.csv", "calibration_summary.csv"),
        ("kshot.csv", "kshot_summary.csv"),
        ("early_decision.csv", "early_decision_summary.csv"),
        ("robustness.csv", "robustness_summary.csv"),
    ):
        write_csv(output / merged_name, _merge_run_csv(output, subjects, seeds, filename))
    operation_rows = []
    for row in rows:
        operation = read_json(
            output
            / f"subject_{int(row['subject']):02d}"
            / f"seed_{int(row['seed'])}"
            / "operation_proxy.json"
        )
        for arm in ARMS:
            operation_rows.append(
                {
                    "subject": row["subject"],
                    "seed": row["seed"],
                    "arm": arm,
                    **operation[arm],
                    "paired_reduction": operation["reduction"],
                }
            )
    write_csv(output / "operation_proxy_summary.csv", operation_rows)

    paired_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for index, field in enumerate(
        (
            "final_accuracy",
            "early_accuracy",
            "robustness_mean_accuracy",
            "kshot_mean_accuracy",
        )
    ):
        pairs, summary = _paired(rows, field, 20260803 + index)
        summaries[field] = summary
        paired_rows.extend({"metric": field, **pair} for pair in pairs)
    write_csv(output / "paired_subject_seed.csv", paired_rows)
    operation_reduction = float(
        np.mean([float(row["operation_proxy_reduction"]) for row in rows])
    )
    utility = utility_win_summary(
        final_gain_pp=100.0 * summaries["final_accuracy"]["subject_macro_mean_delta"],
        early_gain_pp=100.0 * summaries["early_accuracy"]["subject_macro_mean_delta"],
        robustness_gain_pp=100.0
        * summaries["robustness_mean_accuracy"]["subject_macro_mean_delta"],
        kshot_gain_pp=100.0
        * summaries["kshot_mean_accuracy"]["subject_macro_mean_delta"],
        operation_reduction=operation_reduction,
        maximum_final_gap_pp=float(config["gate"]["maximum_final_accuracy_gap_pp"]),
        minimum_accuracy_utility_pp=float(
            config["gate"]["minimum_early_or_robustness_or_kshot_gain_pp"]
        ),
        minimum_operation_reduction=float(
            config["gate"]["minimum_operation_proxy_reduction"]
        ),
    )
    full_contract = subjects == list(range(1, 10)) and seeds == list(range(5))
    gate = {
        "status": "completed",
        "stage": "E7_ENSEMBLE_GATE",
        "passed": bool(utility["passed"] and full_contract),
        "full_registered_contract": full_contract,
        "utility": utility,
        "paired_summaries": summaries,
        "operation_proxy_reduction": operation_reduction,
        "hardware_energy_claim_allowed": False,
        "snn_necessity_claim_allowed": False,
        "snn_necessity_reason": (
            "an operation proxy is supporting evidence, not neuromorphic hardware proof"
        ),
    }
    write_json(output / "gate_decision.json", gate)
    status = {
        "status": "completed",
        "stage": "E7_ENSEMBLE",
        "subjects": subjects,
        "seeds": seeds,
        "runs": len(rows),
        "full_registered_contract": full_contract,
        "freeze_sha256": freeze["combined_sha256"],
        "model_updates": False,
        "session_e_accessed": True,
        "openbmi_s2_accessed": False,
        "gate_passed": gate["passed"],
    }
    write_json(output / "campaign_status.json", status)
    write_run_artifact_manifest(output, required_files=CAMPAIGN_FILES)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
