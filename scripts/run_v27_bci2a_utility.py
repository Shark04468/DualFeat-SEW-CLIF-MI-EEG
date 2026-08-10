#!/usr/bin/env python3
"""Replay frozen V25 checkpoints for registered V27 utility endpoints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.analysis.v8_utility import add_gaussian_noise_at_snr, drop_eeg_channels  # noqa: E402
from dpc_snn.baselines.neural import build_v62_neural_baseline  # noqa: E402
from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v17_information_replay import residual_fusion_logits  # noqa: E402
from dpc_snn.experiments.v27_utility import (  # noqa: E402
    dual_feature_operation_proxy,
    mask_random_time_windows,
)
from dpc_snn.experiments.v62_baselines import (  # noqa: E402
    FixedGain,
    apply_fixed_gain,
    prepare_model_input,
    task_carrier,
)
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    sha256_fingerprint,
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v62_scaffold import (  # noqa: E402
    project_registered_max_norm_constraints_,
)
from dpc_snn.experiments.v8_ensemble import V8ATCAccuracyBackbone  # noqa: E402
from dpc_snn.experiments.v8_ensemble_followup import mask_carrier_after_endpoint  # noqa: E402
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    source_tree_digest,
)
from dpc_snn.experiments.v9_dual_feature_training import (  # noqa: E402
    FrozenFeatureStandardizer,
    V9_DUAL_FEATURE_EXPERIMENT_VARIANTS,
    equal_probability_teacher,
    predict_v9_dual_feature,
)
from dpc_snn.models.v9_dual_feature_student import (  # noqa: E402
    V9FBCFeatureBackbone,
    V9DualFeatureStudent,
    build_v9_dual_feature_student,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v8_e1_baselines import _subject_file  # noqa: E402
from scripts.run_v8_e6_ensemble_frozen import _metadata  # noqa: E402


ARMS = ("sew_clif_ce", "ann_plain_ce")
RUN_FILES = (
    "manifest.json",
    "resolved_run.yaml",
    "source_tree_manifest.json",
    "runtime_status.json",
    "clean_replay.json",
    "early_decision.csv",
    "early_predictions.npz",
    "robustness.csv",
    "robustness_predictions.npz",
    "operation_proxy.json",
    "state_audit.json",
    "metrics.json",
)


@dataclass
class FrozenComponents:
    atc: nn.Module
    fbc: nn.Module
    students: dict[str, V9DualFeatureStudent]

    def state_digests(self) -> dict[str, str]:
        values = {"atcnet": self.atc, "fbcnet": self.fbc, **self.students}
        return {
            name: sha256_fingerprint(
                {
                    key: file_tensor.detach().cpu().contiguous().numpy().tobytes().hex()
                    for key, file_tensor in model.state_dict().items()
                }
            )
            for name, model in values.items()
        }


def _standardizer(path: Path) -> FrozenFeatureStandardizer:
    value = read_json(path)
    return FrozenFeatureStandardizer(
        np.asarray(value["atc_mean"], dtype=np.float32).reshape(1, 1, 32),
        np.asarray(value["atc_scale"], dtype=np.float32).reshape(1, 1, 32),
        np.asarray(value["fbc_mean"], dtype=np.float32).reshape(1, 1, 288),
        np.asarray(value["fbc_scale"], dtype=np.float32).reshape(1, 1, 288),
    )


def _gain(path: Path) -> FixedGain:
    with np.load(path, allow_pickle=False) as archive:
        return FixedGain(
            values=np.asarray(archive["values"], dtype=np.float32),
            clip=float(np.asarray(archive["clip"]).item()),
        )


def _load_components(run_dir: Path, source_root: Path, device: str) -> FrozenComponents:
    atc_adapter = build_v62_neural_baseline(
        "atcnet", source_root=source_root, n_channels=22, n_classes=4, samples=1000
    )
    atc_adapter.load_state_dict(
        torch.load(run_dir / "atcnet.pt", map_location="cpu", weights_only=True), strict=True
    )
    fbc_adapter = build_v62_neural_baseline(
        "fbcnet", source_root=source_root, n_channels=22, n_classes=4, samples=1000
    )
    fbc_adapter.load_state_dict(
        torch.load(run_dir / "fbcnet.pt", map_location="cpu", weights_only=True), strict=True
    )
    students: dict[str, V9DualFeatureStudent] = {}
    for variant in ARMS:
        spec = V9_DUAL_FEATURE_EXPERIMENT_VARIANTS[variant]
        model = build_v9_dual_feature_student(spec.model_variant)
        model.load_state_dict(
            torch.load(run_dir / f"{variant}.pt", map_location="cpu", weights_only=True),
            strict=True,
        )
        students[variant] = model.eval().to(device)
    components = FrozenComponents(
        atc=V8ATCAccuracyBackbone(atc_adapter.module).eval().to(device),
        fbc=V9FBCFeatureBackbone(fbc_adapter.module).eval().to(device),
        students=students,
    )
    projection_count = project_registered_max_norm_constraints_(components.atc)
    if projection_count < 1:
        raise RuntimeError("official ATCNet inference constraints were not registered")
    return components


@torch.no_grad()
def _extract(wrapper: nn.Module, carrier: np.ndarray, *, device: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.ascontiguousarray(carrier, dtype=np.float32))),
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    sequences: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for (batch,) in loader:
        output = wrapper(batch.to(device, non_blocking=True))
        key = "aux" if kind == "atc" else "continuous_sequence"
        sequence = output[key]["continuous_sequence"] if kind == "atc" else output[key]
        sequences.append(sequence.float().cpu().numpy())
        logits.append(output["logits"].float().cpu().numpy())
    return np.concatenate(sequences), np.concatenate(logits)


def _predict(
    components: FrozenComponents,
    carrier: np.ndarray,
    labels: np.ndarray,
    *,
    gain: FixedGain,
    standardizer: FrozenFeatureStandardizer,
    sfreq: float,
    scales: dict[str, float],
    device: str,
) -> dict[str, Any]:
    normalized = apply_fixed_gain(carrier, gain)
    atc_input = prepare_model_input("atcnet", normalized, sfreq=sfreq)
    fbc_input = prepare_model_input("fbcnet", normalized, sfreq=sfreq)
    atc_sequence, atc_logits = _extract(components.atc, atc_input, device=device, kind="atc")
    fbc_sequence, fbc_logits = _extract(components.fbc, fbc_input, device=device, kind="fbc")
    atc_sequence, fbc_sequence = standardizer.transform(atc_sequence, fbc_sequence)
    teacher = equal_probability_teacher(atc_logits, fbc_logits)
    result: dict[str, Any] = {"arms": {}, "teacher": teacher}
    for variant in ARMS:
        prediction = predict_v9_dual_feature(
            components.students[variant],
            atc_sequence,
            fbc_sequence,
            labels,
            teacher,
            device=device,
            batch_size=64,
        )
        logits = residual_fusion_logits(teacher, prediction["logits"], float(scales[variant]))
        result["arms"][variant] = {
            "logits": logits,
            "mean_firing_rate": float(prediction["mean_firing_rate"]),
            **classification_metrics(labels, logits.argmax(axis=1), n_classes=4),
        }
    return result


def _parent_manifest(run_dir: Path) -> None:
    manifest = read_json(run_dir / "manifest.json")
    validate_run_artifact_manifest(
        run_dir,
        required_files=tuple(manifest["required_files"]),
        verify_hashes=True,
        verify_prediction_schema=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-data", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    configure_cache_env()
    if args.subject not in range(1, 10) or args.seed not in range(5):
        raise ValueError("V27 requires subject 1-9 and seed 0-4")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    freeze = read_json(args.freeze)
    source_tree = collect_source_tree_manifest(ROOT)
    if source_tree_digest(source_tree) != freeze["source_tree_sha256"]:
        raise RuntimeError("active source differs from the V27 freeze")
    config = freeze["utility_config"]
    parent = Path(args.campaign).resolve() / f"subject_{args.subject:02d}" / f"seed_{args.seed}"
    _parent_manifest(parent)
    run_dir = Path(args.output).resolve() / f"subject_{args.subject:02d}" / f"seed_{args.seed}"
    if (run_dir / "manifest.json").is_file():
        validate_run_artifact_manifest(run_dir, required_files=RUN_FILES, verify_hashes=True)
        print(json.dumps(read_json(run_dir / "metrics.json"), indent=2))
        return
    ensure_dir(run_dir)
    write_json(run_dir / "source_tree_manifest.json", source_tree)
    resolved = {
        "stage": "V27_FROZEN_UTILITY",
        "subject": args.subject,
        "seed": args.seed,
        "freeze_sha256": freeze["combined_sha256"],
        "parent_manifest_sha256": file_sha256(parent / "manifest.json"),
        "config": config,
    }
    (run_dir / "resolved_run.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    write_json(run_dir / "runtime_status.json", {"status": "running", "started_at": time.time()})

    data_path = _subject_file(Path(args.evaluation_data).resolve(), args.subject)
    data = load_processed_npz(data_path)
    metadata = _metadata(data, expected_session="E", role="evaluation")
    labels = np.asarray(data["y"], dtype=np.int64)
    trial_ids = np.asarray([row["trial_id"] for row in metadata])
    sfreq = float(np.asarray(data["sfreq"]).item())
    carrier = task_carrier(
        np.asarray(data["X"], dtype=np.float32),
        sfreq=sfreq,
        epoch_tmin=float(np.asarray(data["epoch_tmin"]).item()),
    )
    parent_metrics = read_json(parent / "metrics.json")
    scales = {name: float(parent_metrics["residual_scales"][name]) for name in ARMS}
    components = _load_components(parent, Path(args.source_root).resolve(), args.device)
    before = components.state_digests()
    gain = _gain(parent / "gain.npz")
    standardizer = _standardizer(parent / "feature_standardizer.json")
    clean = _predict(
        components, carrier, labels, gain=gain, standardizer=standardizer,
        sfreq=sfreq, scales=scales, device=args.device,
    )
    clean_errors: dict[str, float] = {}
    prediction_files = {
        "sew_clif_ce": "predictions.npz",
        "ann_plain_ce": "ann_plain_ce_predictions.npz",
    }
    for arm in ARMS:
        with np.load(parent / prediction_files[arm], allow_pickle=False) as archive:
            expected_logits = np.asarray(archive["logits"], dtype=np.float32)
            expected_trial_ids = np.asarray(archive["trial_id"])
        if not np.array_equal(expected_trial_ids.astype(str), trial_ids.astype(str)):
            raise RuntimeError("V27 clean replay trial order differs from E26")
        error = float(np.max(np.abs(expected_logits - clean["arms"][arm]["logits"])))
        if error > 1e-5 or not np.array_equal(
            expected_logits.argmax(axis=1), clean["arms"][arm]["logits"].argmax(axis=1)
        ):
            raise RuntimeError(f"V27 clean replay differs from E26 for {arm}: {error:.3e}")
        clean_errors[arm] = error
    write_json(run_dir / "clean_replay.json", {"valid": True, "maximum_logit_error": clean_errors})

    endpoints = [float(value) for value in config["early_decision"]["endpoints_seconds"]]
    early_rows: list[dict[str, Any]] = []
    early_logits: list[np.ndarray] = []
    for endpoint in endpoints:
        result = clean if np.isclose(endpoint, carrier.shape[-1] / sfreq) else _predict(
            components,
            mask_carrier_after_endpoint(carrier, endpoint_seconds=endpoint, sfreq=sfreq),
            labels,
            gain=gain,
            standardizer=standardizer,
            sfreq=sfreq,
            scales=scales,
            device=args.device,
        )
        early_logits.append(np.stack([result["arms"][arm]["logits"] for arm in ARMS]))
        for arm in ARMS:
            early_rows.append({"arm": arm, "endpoint_seconds": endpoint, **{
                key: result["arms"][arm][key]
                for key in ("accuracy", "balanced_accuracy", "kappa", "macro_f1")
            }})
    write_csv(run_dir / "early_decision.csv", early_rows)
    np.savez_compressed(
        run_dir / "early_predictions.npz", arms=np.asarray(ARMS),
        endpoint_seconds=np.asarray(endpoints, dtype=np.float32),
        logits=np.stack(early_logits).astype(np.float32), labels=labels, trial_id=trial_ids,
    )

    conditions: list[tuple[str, np.ndarray, Any]] = [("clean", carrier, {})]
    for index, snr in enumerate(config["robustness"]["gaussian_snr_db"]):
        seed = args.subject * 1_000_003 + args.seed * 10_009 + index
        conditions.append((f"noise_{float(snr):g}dB", add_gaussian_noise_at_snr(carrier, snr_db=float(snr), seed=seed), {"seed": seed}))
    for index, count in enumerate(config["robustness"]["channel_drop_counts"]):
        seed = args.subject * 1_000_003 + args.seed * 10_009 + 100 + index
        perturbed, dropped = drop_eeg_channels(carrier, count=int(count), seed=seed)
        conditions.append((f"drop_{int(count)}ch", perturbed, {"seed": seed, "channels": dropped.astype(int).tolist()}))
    for index, seconds in enumerate(config["robustness"]["time_mask_seconds"]):
        seed = args.subject * 1_000_003 + args.seed * 10_009 + 200 + index
        perturbed, starts = mask_random_time_windows(carrier, duration_seconds=float(seconds), sfreq=sfreq, seed=seed)
        conditions.append((f"mask_{int(round(1000 * float(seconds)))}ms", perturbed, {"seed": seed, "starts": starts.tolist()}))
    for scale in config["robustness"]["amplitude_scales"]:
        conditions.append((f"amplitude_{float(scale):g}x", np.ascontiguousarray(carrier * float(scale), dtype=np.float32), {"scale": float(scale)}))

    robust_rows: list[dict[str, Any]] = []
    robust_logits: list[np.ndarray] = []
    for condition, perturbed, detail in conditions:
        result = clean if condition == "clean" else _predict(
            components, perturbed, labels, gain=gain, standardizer=standardizer,
            sfreq=sfreq, scales=scales, device=args.device,
        )
        robust_logits.append(np.stack([result["arms"][arm]["logits"] for arm in ARMS]))
        for arm in ARMS:
            robust_rows.append({"arm": arm, "condition": condition, "detail": json.dumps(detail), **{
                key: result["arms"][arm][key]
                for key in ("accuracy", "balanced_accuracy", "kappa", "macro_f1")
            }})
    write_csv(run_dir / "robustness.csv", robust_rows)
    np.savez_compressed(
        run_dir / "robustness_predictions.npz", arms=np.asarray(ARMS),
        conditions=np.asarray([item[0] for item in conditions]),
        logits=np.stack(robust_logits).astype(np.float32), labels=labels, trial_id=trial_ids,
    )

    operation = {
        arm: dual_feature_operation_proxy(
            components.students[arm],
            mean_binary_firing_rate=clean["arms"][arm]["mean_firing_rate"] if arm == "sew_clif_ce" else None,
        ) for arm in ARMS
    }
    operation["decoder_reduction"] = 1.0 - operation["sew_clif_ce"]["activity_weighted_decoder_events"] / operation["ann_plain_ce"]["activity_weighted_decoder_events"]
    operation["full_student_reduction"] = 1.0 - operation["sew_clif_ce"]["activity_weighted_full_student_events"] / operation["ann_plain_ce"]["activity_weighted_full_student_events"]
    write_json(run_dir / "operation_proxy.json", operation)
    after = components.state_digests()
    if before != after:
        raise RuntimeError("V27 changed a frozen component state")
    write_json(run_dir / "state_audit.json", {"before": before, "after": after, "identical": True, "model_updates": False})

    arm_metrics: dict[str, Any] = {}
    for arm in ARMS:
        accuracies = np.asarray([float(row["accuracy"]) for row in early_rows if row["arm"] == arm])
        early_auc = float(np.trapezoid(accuracies, endpoints) / (endpoints[-1] - endpoints[0]))
        robustness_mean = float(np.mean([float(row["accuracy"]) for row in robust_rows if row["arm"] == arm and row["condition"] != "clean"]))
        arm_metrics[arm] = {
            "final_accuracy": float(clean["arms"][arm]["accuracy"]),
            "early_accuracy_auc": early_auc,
            "robustness_mean_accuracy": robustness_mean,
            "mean_firing_rate": float(clean["arms"][arm]["mean_firing_rate"]),
        }
    metrics = {
        "status": "completed", "stage": "V27_FROZEN_UTILITY",
        "subject": args.subject, "seed": args.seed, "freeze_sha256": freeze["combined_sha256"],
        "arms": arm_metrics, "decoder_operation_reduction": operation["decoder_reduction"],
        "full_student_operation_reduction": operation["full_student_reduction"],
        "model_updates": False, "historical_session_e_exposure": True,
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "runtime_status.json", {"status": "completed", "completed_at": time.time(), "model_updates": False})
    write_run_artifact_manifest(run_dir, required_files=RUN_FILES)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
