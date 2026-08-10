#!/usr/bin/env python3
"""Evaluate frozen V8 early, robust, calibrated and sparse-compute utility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
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
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    sha256_fingerprint,
    validate_prediction_schema,
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    build_v8_run_fingerprint,
    collect_source_tree_manifest,
    mapping_sha256,
    source_tree_digest,
    validate_v8_freeze_manifest,
    validate_v8_resume_fingerprint,
    write_v8_fingerprint,
)
from dpc_snn.experiments.v8_statistics import (  # noqa: E402
    pair_subject_seed_rows,
    paired_delta_summary,
)
from dpc_snn.experiments.v8_training import (  # noqa: E402
    cache_v8_physical_rates,
    predict_v8,
    seed_v8,
)
from dpc_snn.models.build import build_model  # noqa: E402
from dpc_snn.models.v8_accuracy_first import V8AccuracyFirstModel  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v8_e2_zero_delay import _environment, _subject_file  # noqa: E402
from scripts.run_v8_e3_static_delay import _build_delay_model  # noqa: E402
from scripts.run_v8_e6_bci2a_frozen import _metadata, _session_indices  # noqa: E402


RUN_FILES = (
    "manifest.json",
    "source_fingerprint.json",
    "resolved_config.yaml",
    "metrics.json",
    "calibration.json",
    "kshot.csv",
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
    "robustness_summary.csv",
    "operation_proxy_summary.csv",
    "gate_decision.json",
    "source_tree_manifest.json",
    "resolved_campaign.yaml",
)


def _state_digest(model: V8AccuracyFirstModel) -> str:
    return sha256_fingerprint(mapping_sha256(model.state_dict()))


def _load_prediction(path: Path) -> dict[str, np.ndarray]:
    validate_prediction_schema(path, path.with_suffix(".csv"))
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _build_frozen_model(
    freeze: dict[str, Any], arm: str, checkpoint: Path, *, seed: int
) -> tuple[V8AccuracyFirstModel, str]:
    seed_v8(seed)
    if arm == "frozen_primary":
        config = dict(freeze["architecture"]["model_config"])
        delay = dict(freeze["architecture"]["delay"])
        if delay["enabled"]:
            model = _build_delay_model(config, dict(delay["config"]["delay"]), seed=seed)
            override = "full"
        else:
            model = build_model("v8_accuracy_first", config)
            override = "off"
    elif arm == "matched_ann":
        config = dict(freeze["architecture"]["matched_ann_control_config"])
        model = build_model("v8_accuracy_first", config)
        override = "off"
    else:
        raise ValueError(f"unknown E7 arm {arm!r}")
    if not isinstance(model, V8AccuracyFirstModel):
        raise TypeError("E7 model factory returned an unexpected model")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model, override


def _operation_proxy(
    model: V8AccuracyFirstModel, e6_metrics: dict[str, Any]
) -> dict[str, Any]:
    if model.decoder is None:
        raise RuntimeError("E7 operation proxy requires the frozen temporal decoder")
    decoder_weights = sum(
        parameter.numel()
        for parameter in model.decoder.parameters()
        if parameter.ndim >= 2
    )
    steps = int(model.endpoint_samples[-1])
    dense_uses = int(decoder_weights * steps)
    activity = e6_metrics.get("binary_spike_rate")
    if model.decoder.decoder_kind == "ann":
        activity_weighted = float(dense_uses)
    else:
        if activity is None or not 0.0 <= float(activity) <= 1.0:
            raise RuntimeError("SNN operation proxy requires a finite binary spike rate")
        activity_weighted = float(dense_uses) * float(activity)
    return {
        "decoder_kind": model.decoder.decoder_kind,
        "decoder_weight_parameters": int(decoder_weights),
        "time_steps": steps,
        "dense_decoder_weight_uses": dense_uses,
        "activity_weighted_decoder_events": activity_weighted,
        "binary_spike_rate": activity,
        "scope": "software operation proxy; front-end dense work excluded equally",
        "hardware_energy_claim_allowed": False,
    }


def _run_one(
    *,
    output: Path,
    e6: Path,
    data_path: Path,
    freeze: dict[str, Any],
    config: dict[str, Any],
    source_tree: dict[str, str],
    environment: dict[str, Any],
    arm: str,
    subject: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    e6_run = e6 / arm / f"subject_{subject:02d}" / f"seed_{seed}"
    e6_manifest = read_json(e6_run / "manifest.json")
    validate_run_artifact_manifest(
        e6_run,
        required_files=e6_manifest["required_files"],
        verify_hashes=True,
        verify_prediction_schema=True,
    )
    e6_metrics = read_json(e6_run / "metrics.json")
    checkpoint = e6_run / "best.pt"
    resolved = {
        "stage": "E7",
        "arm": arm,
        "subject": int(subject),
        "seed": int(seed),
        "freeze_sha256": freeze["combined_sha256"],
        "utility_config": config,
        "e6_run_fingerprint": e6_metrics["run_fingerprint"],
    }
    fingerprint = build_v8_run_fingerprint(
        resolved_run_config=resolved,
        source_tree=source_tree,
        data={data_path.name: file_sha256(data_path)},
        split={"session": "E", "role": "frozen post-hoc utility"},
        augmentation={"policy": "none"},
        prior={"policy": "frozen checkpoint only"},
        checkpoint={
            "e6_checkpoint_sha256": file_sha256(checkpoint),
            "model_updates": False,
        },
        environment=environment,
    )
    run_dir = ensure_dir(output / arm / f"subject_{subject:02d}" / f"seed_{seed}")
    fingerprint_path = run_dir / "source_fingerprint.json"
    if (run_dir / "manifest.json").is_file():
        validate_v8_resume_fingerprint(fingerprint_path, fingerprint)
        validate_run_artifact_manifest(
            run_dir, required_files=RUN_FILES, verify_hashes=True
        )
        return read_json(run_dir / "metrics.json")
    if fingerprint_path.is_file():
        validate_v8_resume_fingerprint(fingerprint_path, fingerprint)
    else:
        write_v8_fingerprint(fingerprint_path, fingerprint)
        (run_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )
    write_json(
        run_dir / "runtime_status.json",
        {"status": "running", "started_at": time.time(), "model_updates": False},
    )

    model, delay_override = _build_frozen_model(
        freeze, arm, checkpoint, seed=int(seed) * 100_003 + int(subject) * 1_009
    )
    model.to(device).eval()
    before = _state_digest(model)
    prediction = _load_prediction(e6_run / "predictions.npz")
    with np.load(e6_run / "prefix_predictions.npz", allow_pickle=False) as archive:
        prefix_logits = archive["logits"]
        prefix_labels = archive["labels"]
        endpoint_seconds = archive["endpoint_seconds"]
    if not np.array_equal(prefix_labels, prediction["label"]):
        raise RuntimeError("E7 prefix labels differ from E6 final labels")

    calibration = multiclass_calibration_metrics(
        prediction["logits"],
        prediction["label"],
        n_bins=int(config["calibration"]["confidence_bins"]),
    )
    write_json(run_dir / "calibration.json", calibration)
    kshot_rows: list[dict[str, Any]] = []
    for k in config["calibration"]["k_per_class"]:
        for repeat in range(int(config["calibration"]["repeats"])):
            split_seed = (
                int(subject) * 10_000_019
                + int(seed) * 100_003
                + int(k) * 1_009
                + repeat
            )
            calibration_indices, evaluation_indices = stratified_kshot_indices(
                prediction["label"], k_per_class=int(k), seed=split_seed
            )
            calibrator = fit_logit_calibrator(
                prediction["logits"][calibration_indices],
                prediction["label"][calibration_indices],
                l2_bias=float(config["calibration"]["bias_l2"]),
            )
            calibrated = calibrator.apply(prediction["logits"][evaluation_indices])
            calibrated_metrics = multiclass_calibration_metrics(
                calibrated,
                prediction["label"][evaluation_indices],
                n_bins=int(config["calibration"]["confidence_bins"]),
            )
            raw_metrics = multiclass_calibration_metrics(
                prediction["logits"][evaluation_indices],
                prediction["label"][evaluation_indices],
                n_bins=int(config["calibration"]["confidence_bins"]),
            )
            kshot_rows.append(
                {
                    "arm": arm,
                    "subject": subject,
                    "seed": seed,
                    "k_per_class": int(k),
                    "repeat": repeat,
                    "split_seed": split_seed,
                    "evaluation_trials": int(evaluation_indices.size),
                    "raw_accuracy": raw_metrics["accuracy"],
                    "calibrated_accuracy": calibrated_metrics["accuracy"],
                    "calibrated_nll": calibrated_metrics["negative_log_likelihood"],
                    "calibrated_ece": calibrated_metrics["ece"],
                }
            )
    write_csv(run_dir / "kshot.csv", kshot_rows)

    data = load_processed_npz(data_path)
    e_indices = _session_indices(data, "E")
    metadata_e = _metadata(data, e_indices, stage="bci2a_evaluation", role="evaluation")
    if [row["trial_id"] for row in metadata_e] != prediction["trial_id"].tolist():
        raise RuntimeError("E7 loaded Session-E trial order differs from E6 predictions")
    x_e = np.asarray(data["X"], dtype=np.float32)[e_indices]
    y_e = np.asarray(data["y"], dtype=np.int64)[e_indices]
    conditions: list[tuple[str, np.ndarray, list[int]]] = [("clean", x_e, [])]
    for index, snr in enumerate(config["robustness"]["gaussian_snr_db"]):
        perturb_seed = int(subject) * 1_000_003 + int(seed) * 10_009 + index
        conditions.append(
            (
                f"noise_{float(snr):g}dB",
                add_gaussian_noise_at_snr(x_e, snr_db=float(snr), seed=perturb_seed),
                [],
            )
        )
    for index, count in enumerate(config["robustness"]["channel_drop_counts"]):
        perturb_seed = int(subject) * 1_000_003 + int(seed) * 10_009 + 100 + index
        dropped_x, dropped = drop_eeg_channels(
            x_e, count=int(count), seed=perturb_seed
        )
        conditions.append(
            (f"drop_{int(count)}ch", dropped_x, dropped.astype(int).tolist())
        )
    robustness_rows: list[dict[str, Any]] = []
    condition_logits: list[np.ndarray] = []
    for condition, perturbed, dropped in conditions:
        rates = cache_v8_physical_rates(model, perturbed, device=device, batch_size=16)
        result = predict_v8(
            model,
            rates,
            y_e,
            device=device,
            batch_size=int(freeze["training"]["batch_size"]),
            delay_override=delay_override,
        )
        if condition == "clean":
            maximum = float(np.max(np.abs(result["logits"] - prediction["logits"])))
            if maximum > 1e-5:
                raise RuntimeError(f"E7 clean replay differs from E6 logits: {maximum:.3e}")
        condition_logits.append(result["logits"].astype(np.float32))
        robustness_rows.append(
            {
                "arm": arm,
                "subject": subject,
                "seed": seed,
                "condition": condition,
                "dropped_channel_indices": json.dumps(dropped),
                "accuracy": result["accuracy"],
                "balanced_accuracy": result["balanced_accuracy"],
                "kappa": result["kappa"],
                "macro_f1": result["macro_f1"],
            }
        )
    write_csv(run_dir / "robustness.csv", robustness_rows)
    np.savez_compressed(
        run_dir / "robustness_predictions.npz",
        condition=np.asarray([row[0] for row in conditions]),
        logits=np.stack(condition_logits),
        labels=y_e,
        trial_id=prediction["trial_id"],
    )
    operation = _operation_proxy(model, e6_metrics)
    write_json(run_dir / "operation_proxy.json", operation)
    after = _state_digest(model)
    if before != after:
        raise RuntimeError("E7 changed the frozen E6 checkpoint")
    write_json(
        run_dir / "state_audit.json",
        {"before": before, "after": after, "identical": True, "model_updates": False},
    )
    primary_endpoint = float(config["early_decision"]["primary_endpoint_seconds"])
    endpoint_index = int(np.where(np.isclose(endpoint_seconds, primary_endpoint))[0][0])
    early = classification_metrics(
        prefix_labels, prefix_logits[:, endpoint_index].argmax(axis=1), n_classes=4
    )
    metrics = {
        "status": "completed",
        "stage": "E7",
        "arm": arm,
        "subject": subject,
        "seed": seed,
        "final_accuracy": float(e6_metrics["accuracy"]),
        "early_accuracy": early["accuracy"],
        "robustness_mean_accuracy": float(
            np.mean([row["accuracy"] for row in robustness_rows if row["condition"] != "clean"])
        ),
        "kshot_mean_accuracy": float(
            np.mean([row["calibrated_accuracy"] for row in kshot_rows])
        ),
        "activity_weighted_decoder_events": operation[
            "activity_weighted_decoder_events"
        ],
        "raw_ece": calibration["ece"],
        "raw_nll": calibration["negative_log_likelihood"],
        "freeze_sha256": freeze["combined_sha256"],
        "e6_run_fingerprint": e6_metrics["run_fingerprint"],
        "run_fingerprint": fingerprint["combined_sha256"],
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "runtime_status.json",
        {"status": "completed", "completed_at": time.time(), "model_updates": False},
    )
    write_run_artifact_manifest(run_dir, required_files=RUN_FILES)
    return metrics


def _paired_metric(
    ann: list[dict[str, Any]], snn: list[dict[str, Any]], field: str, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = pair_subject_seed_rows(ann, snn, value=field)
    return pairs, paired_delta_summary(pairs, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--e6", required=True)
    parser.add_argument("--e6-audit", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/experiments/v8_e7_utility.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    configure_cache_env()
    output = ensure_dir(Path(args.output).resolve())
    data_root = Path(args.data).resolve()
    e6 = Path(args.e6).resolve()
    config = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    source_tree = collect_source_tree_manifest(ROOT)
    freeze = validate_v8_freeze_manifest(
        Path(args.freeze).resolve(),
        expected_source_tree_sha256=source_tree_digest(source_tree),
    )
    audit = read_json(Path(args.e6_audit).resolve() / "audit_report.json")
    if audit.get("status") != "passed" or audit.get("freeze_sha256") != freeze[
        "combined_sha256"
    ]:
        raise RuntimeError("E7 requires the passed audit of the exact frozen E6 campaign")
    e6_status = read_json(e6 / "campaign_status.json")
    if e6_status.get("status") != "completed":
        raise RuntimeError("E7 requires completed E6")
    arms = ["frozen_primary"]
    if freeze["architecture"]["primary_variant"] != "ann_residual":
        arms.append("matched_ann")
    environment = _environment()
    write_json(output / "source_tree_manifest.json", source_tree)
    (output / "resolved_campaign.yaml").write_text(
        yaml.safe_dump(
            {**config, "arms": arms, "freeze_sha256": freeze["combined_sha256"]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    for arm in arms:
        for subject in range(1, 10):
            data_path = _subject_file(data_root, subject)
            for seed in range(5):
                rows.append(
                    _run_one(
                        output=output,
                        e6=e6,
                        data_path=data_path,
                        freeze=freeze,
                        config=config,
                        source_tree=source_tree,
                        environment=environment,
                        arm=arm,
                        subject=subject,
                        seed=seed,
                        device=args.device,
                    )
                )
                write_csv(output / "summary.csv", rows)
    rows = sorted(rows, key=lambda row: (row["arm"], row["subject"], row["seed"]))
    write_csv(output / "summary.csv", rows)
    for source_name, output_name in (
        ("calibration.json", "calibration_summary.csv"),
        ("kshot.csv", "kshot_summary.csv"),
        ("robustness.csv", "robustness_summary.csv"),
        ("operation_proxy.json", "operation_proxy_summary.csv"),
    ):
        merged: list[dict[str, Any]] = []
        for arm in arms:
            for subject in range(1, 10):
                for seed in range(5):
                    path = output / arm / f"subject_{subject:02d}" / f"seed_{seed}" / source_name
                    if path.suffix == ".csv":
                        import csv

                        with path.open("r", encoding="utf-8", newline="") as handle:
                            merged.extend(csv.DictReader(handle))
                    else:
                        merged.append(
                            {"arm": arm, "subject": subject, "seed": seed, **read_json(path)}
                        )
        write_csv(output / output_name, merged)

    gate_result: dict[str, Any]
    if len(arms) == 2:
        primary = [row for row in rows if row["arm"] == "frozen_primary"]
        ann = [row for row in rows if row["arm"] == "matched_ann"]
        pair_summaries: dict[str, Any] = {}
        pairs_by_metric: dict[str, list[dict[str, Any]]] = {}
        for index, field in enumerate(
            (
                "final_accuracy",
                "early_accuracy",
                "robustness_mean_accuracy",
                "kshot_mean_accuracy",
                "activity_weighted_decoder_events",
            )
        ):
            pairs, summary = _paired_metric(ann, primary, field, 20260718 + index)
            pairs_by_metric[field] = pairs
            pair_summaries[field] = summary
        ann_ops = np.asarray([row["first"] for row in pairs_by_metric["activity_weighted_decoder_events"]])
        snn_ops = np.asarray([row["second"] for row in pairs_by_metric["activity_weighted_decoder_events"]])
        operation_reduction = float(1.0 - np.mean(snn_ops / ann_ops))
        utility = utility_win_summary(
            final_gain_pp=100.0 * pair_summaries["final_accuracy"]["subject_macro_mean_delta"],
            early_gain_pp=100.0 * pair_summaries["early_accuracy"]["subject_macro_mean_delta"],
            robustness_gain_pp=100.0
            * pair_summaries["robustness_mean_accuracy"]["subject_macro_mean_delta"],
            kshot_gain_pp=100.0
            * pair_summaries["kshot_mean_accuracy"]["subject_macro_mean_delta"],
            operation_reduction=operation_reduction,
            maximum_final_gap_pp=float(config["gate"]["maximum_final_accuracy_gap_pp"]),
            minimum_accuracy_utility_pp=float(
                config["gate"]["minimum_early_or_robustness_or_kshot_gain_pp"]
            ),
            minimum_operation_reduction=float(
                config["gate"]["minimum_operation_proxy_reduction"]
            ),
        )
        gate_result = {
            "status": "completed",
            "stage": "E7_GATE",
            "snn_comparison_available": True,
            "passed": utility["passed"],
            "utility": utility,
            "paired_summaries": pair_summaries,
            "operation_proxy_reduction": operation_reduction,
            "hardware_energy_claim_allowed": False,
        }
    else:
        gate_result = {
            "status": "completed",
            "stage": "E7_GATE",
            "snn_comparison_available": False,
            "passed": False,
            "decision": "ANN was frozen; no SNN-necessity claim is permitted",
            "hardware_energy_claim_allowed": False,
        }
    write_json(output / "gate_decision.json", gate_result)
    status = {
        "status": "completed",
        "stage": "E7",
        "arms": arms,
        "subjects": list(range(1, 10)),
        "seeds": list(range(5)),
        "runs": len(rows),
        "freeze_sha256": freeze["combined_sha256"],
        "model_updates": False,
        "session_e_accessed": True,
        "openbmi_s2_accessed": False,
    }
    write_json(output / "campaign_status.json", status)
    write_run_artifact_manifest(output, required_files=CAMPAIGN_FILES)
    print(json.dumps({**status, "gate_passed": gate_result["passed"]}, indent=2))


if __name__ == "__main__":
    main()
