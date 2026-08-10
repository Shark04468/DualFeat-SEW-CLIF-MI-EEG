#!/usr/bin/env python3
"""Independently audit the V8 post-hoc publication baseline campaign."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.analysis.v8_utility import multiclass_calibration_metrics  # noqa: E402
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    sha256_fingerprint,
    validate_prediction_schema,
    validate_run_artifact_manifest,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    mapping_sha256,
    validate_v8_fingerprint,
)
from dpc_snn.experiments.v8_publication_baselines import (  # noqa: E402
    FREQUENCY_OCCLUSIONS_HZ,
    REGION_CHANNELS,
    array_sha256,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from scripts.run_v8_publication_baselines import (  # noqa: E402
    EVALUATION_FILES,
    TRAIN_FILES,
)


METRIC_KEYS = ("accuracy", "balanced_accuracy", "kappa", "macro_f1")


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    return sha256_fingerprint(mapping_sha256(state))


def _gain_digest(path: Path) -> str:
    with np.load(path, allow_pickle=False) as archive:
        values = np.asarray(archive["values"], dtype=np.float32)
        clip = float(np.asarray(archive["clip"]).item())
    return sha256_fingerprint({"values_sha256": array_sha256(values), "clip": clip})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _close(first: float, second: float, atol: float = 1e-9) -> bool:
    return bool(np.isclose(float(first), float(second), rtol=0.0, atol=atol))


def _run_base(output: Path, model: str, subject: int, seed: int) -> Path:
    return output / "runs" / model / f"subject_{subject:02d}" / f"seed_{seed}"


def _verify_barrier(output: Path, contract: dict[str, Any]) -> dict[str, Any]:
    barrier = read_json(output / "checkpoint_barrier.json")
    body = {key: value for key, value in barrier.items() if key != "combined_sha256"}
    if barrier.get("combined_sha256") != sha256_fingerprint(body):
        raise RuntimeError("checkpoint barrier digest is invalid")
    if barrier.get("contract_sha256") != contract["combined_sha256"]:
        raise RuntimeError("checkpoint barrier and campaign contract differ")
    if not barrier.get("all_training_complete_before_evaluation"):
        raise RuntimeError("checkpoint barrier does not seal all training")
    if not np.isfinite(float(barrier.get("sealed_at", np.nan))):
        raise RuntimeError("checkpoint barrier has no finite sealed_at timestamp")
    return barrier


def _audit_training(
    *,
    output: Path,
    scope: dict[str, Any],
    config: dict[str, Any],
    dataset: str,
    barrier: dict[str, Any],
    issues: list[dict[str, Any]],
) -> dict[tuple[str, int, int], dict[str, Any]]:
    barrier_records = {
        (str(row["model"]), int(row["subject"]), int(row["seed"])): row
        for row in barrier["records"]
    }
    expected_keys = {
        (model, int(subject), int(seed))
        for model in scope["models"]
        for subject in scope["subjects"]
        for seed in scope["seeds"]
    }
    if set(barrier_records) != expected_keys:
        issues.append(
            {
                "stage": "barrier",
                "error": "barrier run keys differ from contract scope",
                "missing": sorted(expected_keys - set(barrier_records)),
                "extra": sorted(set(barrier_records) - expected_keys),
            }
        )
    gains: dict[int, set[str]] = {}
    metrics_by_run: dict[tuple[str, int, int], dict[str, Any]] = {}
    expected_trials = int(config["datasets"][dataset]["expected_train_trials"])
    expected_steps = int(np.ceil(expected_trials / int(config["effective_batch_size"]))) * int(
        scope["fixed_epochs"]
    )
    for key in sorted(expected_keys):
        model, subject, seed = key
        training = _run_base(output, model, subject, seed) / "training"
        try:
            validate_run_artifact_manifest(training, required_files=TRAIN_FILES, verify_hashes=True)
            validate_v8_fingerprint(read_json(training / "source_fingerprint.json"))
            runtime = read_json(training / "runtime_status.json")
            if runtime.get("heldout_session_loaded") is not False:
                raise RuntimeError("training runtime says heldout data were loaded")
            if float(runtime["completed_at"]) > float(barrier["sealed_at"]):
                raise RuntimeError("training completed after the checkpoint barrier")
            record = barrier_records[key]
            checkpoint_path = output / record["checkpoint_path"]
            manifest_path = output / record["training_manifest_path"]
            if checkpoint_path != training / "checkpoint.pt":
                raise RuntimeError("barrier checkpoint path is not the active run checkpoint")
            if file_sha256(checkpoint_path) != record["checkpoint_file_sha256"]:
                raise RuntimeError("checkpoint file differs from barrier hash")
            if file_sha256(manifest_path) != record["training_manifest_file_sha256"]:
                raise RuntimeError("training manifest differs from barrier hash")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_digest = _state_digest(checkpoint["state_dict"])
            if checkpoint["state_sha256"] != state_digest:
                raise RuntimeError("checkpoint state_dict differs from its embedded state seal")
            state_audit = read_json(training / "state_audit.json")
            if not state_audit.get("identical") or state_audit["checkpoint_state_sha256"] != state_digest:
                raise RuntimeError("training state audit is inconsistent")
            metrics = read_json(training / "training_metrics.json")
            if (
                int(metrics["optimizer_steps"]) != expected_steps
                or int(metrics["expected_optimizer_steps"]) != expected_steps
            ):
                raise RuntimeError("optimizer update count differs from equal-budget expectation")
            if int(metrics["fixed_epochs"]) != int(scope["fixed_epochs"]):
                raise RuntimeError("training epochs differ from contract")
            if int(metrics["effective_batch_size"]) != int(config["effective_batch_size"]):
                raise RuntimeError("effective batch size differs from the contract")
            if int(metrics["physical_batch_size"]) != int(config["physical_batch_size"][model]):
                raise RuntimeError("physical batch size differs from the contract")
            if not np.isfinite(float(metrics["final_training_loss"])):
                raise RuntimeError("training loss is not finite")
            gains.setdefault(subject, set()).add(_gain_digest(training / "gain.npz"))
            metrics_by_run[key] = metrics
        except Exception as exc:  # audit must retain all failures
            issues.append(
                {
                    "stage": "training",
                    "model": model,
                    "subject": subject,
                    "seed": seed,
                    "error": str(exc),
                }
            )
    for subject, digests in gains.items():
        if len(digests) != 1:
            issues.append(
                {
                    "stage": "training",
                    "subject": subject,
                    "error": "training-derived fixed gain differs across models or seeds",
                    "digests": sorted(digests),
                }
            )
    return metrics_by_run


def _audit_perturbations(
    path: Path,
    metrics_path: Path,
    labels: np.ndarray,
    nominal_accuracy: float,
) -> None:
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    expected_fields = {
        "label",
        "frequency_names",
        "frequency_logits",
        "frequency_probabilities",
        "frequency_pred",
        "region_names",
        "region_logits",
        "region_probabilities",
        "region_pred",
    }
    if set(payload) != expected_fields:
        raise RuntimeError("perturbation archive has an invalid field set")
    if not np.array_equal(payload["label"], labels):
        raise RuntimeError("perturbation labels differ from nominal labels")
    names_by_kind = {
        "frequency": list(FREQUENCY_OCCLUSIONS_HZ),
        "region": list(REGION_CHANNELS),
    }
    csv_rows = _read_csv(metrics_path)
    if len(csv_rows) != sum(len(value) for value in names_by_kind.values()):
        raise RuntimeError("perturbation metric row count is invalid")
    csv_index = {(row["kind"], row["name"]): row for row in csv_rows}
    for kind, names in names_by_kind.items():
        observed = payload[f"{kind}_names"].astype(str).tolist()
        if observed != names:
            raise RuntimeError(f"{kind} perturbation names differ from the locked order")
        logits = np.asarray(payload[f"{kind}_logits"], dtype=np.float32)
        probabilities = np.asarray(payload[f"{kind}_probabilities"], dtype=np.float32)
        prediction = np.asarray(payload[f"{kind}_pred"], dtype=np.int64)
        expected_shape = (len(names), labels.size, probabilities.shape[-1])
        if logits.shape != expected_shape or probabilities.shape != expected_shape:
            raise RuntimeError(f"{kind} perturbation arrays have invalid shapes")
        if prediction.shape != (len(names), labels.size):
            raise RuntimeError(f"{kind} perturbation predictions have invalid shape")
        if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
            raise RuntimeError(f"{kind} perturbations contain NaN or Inf")
        if not np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-5):
            raise RuntimeError(f"{kind} perturbation probabilities are not normalised")
        shifted = logits - logits.max(axis=-1, keepdims=True)
        recomputed_probability = np.exp(shifted)
        recomputed_probability /= recomputed_probability.sum(axis=-1, keepdims=True)
        if not np.allclose(probabilities, recomputed_probability, atol=2e-6):
            raise RuntimeError(f"{kind} probabilities do not equal softmax(logits)")
        if not np.array_equal(prediction, probabilities.argmax(axis=-1)):
            raise RuntimeError(f"{kind} predictions are not probability argmax")
        for index, name in enumerate(names):
            recomputed = classification_metrics(
                labels, prediction[index], n_classes=probabilities.shape[-1]
            )
            row = csv_index[(kind, name)]
            for metric in METRIC_KEYS:
                if not _close(float(row[metric]), recomputed[metric]):
                    raise RuntimeError(f"{kind}/{name} {metric} was not independently reproducible")
            if not _close(float(row["nominal_accuracy"]), nominal_accuracy):
                raise RuntimeError(f"{kind}/{name} nominal accuracy differs")
            if not _close(
                float(row["accuracy_drop"]), nominal_accuracy - recomputed["accuracy"]
            ):
                raise RuntimeError(f"{kind}/{name} accuracy drop differs")


def _audit_saliency(path: Path, labels: np.ndarray, trial_ids: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as archive:
        fields = set(archive.files)
        if fields != {"channel_names", "raw", "normalized", "label", "trial_id"}:
            raise RuntimeError("saliency archive has an invalid field set")
        raw = np.asarray(archive["raw"], dtype=np.float32)
        normalized = np.asarray(archive["normalized"], dtype=np.float32)
        channel_names = archive["channel_names"].astype(str)
        saliency_labels = archive["label"]
        saliency_trials = archive["trial_id"].astype(str)
    if raw.shape != normalized.shape or raw.shape != (labels.size, 22):
        raise RuntimeError("saliency arrays must have shape [trials, 22]")
    if channel_names.size != 22 or len(set(channel_names.tolist())) != 22:
        raise RuntimeError("saliency channel basis is invalid")
    if not np.array_equal(saliency_labels, labels) or not np.array_equal(saliency_trials, trial_ids):
        raise RuntimeError("saliency labels or trial identities differ from predictions")
    if not np.isfinite(raw).all() or not np.isfinite(normalized).all():
        raise RuntimeError("saliency contains NaN or Inf")
    if np.any(raw < 0.0) or np.any(normalized < 0.0):
        raise RuntimeError("absolute saliency must be non-negative")
    raw_sum = raw.sum(axis=1)
    normalized_sum = normalized.sum(axis=1)
    active = raw_sum > np.finfo(np.float32).tiny
    if np.any(active) and not np.allclose(normalized_sum[active], 1.0, atol=1e-5):
        raise RuntimeError("non-zero saliency rows are not L1-normalised")
    if np.any(~active) and not np.allclose(normalized_sum[~active], 0.0, atol=1e-7):
        raise RuntimeError("zero saliency rows have non-zero normalised values")


def _audit_evaluation(
    *,
    output: Path,
    scope: dict[str, Any],
    barrier: dict[str, Any],
    issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in scope["models"]:
        for subject in scope["subjects"]:
            for seed in scope["seeds"]:
                evaluation = _run_base(output, model, subject, seed) / "evaluation"
                try:
                    validate_run_artifact_manifest(
                        evaluation,
                        required_files=EVALUATION_FILES,
                        verify_hashes=True,
                        verify_prediction_schema=True,
                    )
                    validate_v8_fingerprint(read_json(evaluation / "source_fingerprint.json"))
                    runtime = read_json(evaluation / "runtime_status.json")
                    if float(runtime["started_at"]) < float(barrier["sealed_at"]):
                        raise RuntimeError("evaluation started before the checkpoint barrier")
                    schema = validate_prediction_schema(
                        evaluation / "predictions.npz", evaluation / "predictions.csv"
                    )
                    with np.load(evaluation / "predictions.npz", allow_pickle=False) as archive:
                        labels = np.asarray(archive["label"], dtype=np.int64)
                        prediction = np.asarray(archive["pred"], dtype=np.int64)
                        logits = np.asarray(archive["logits"], dtype=np.float32)
                        probabilities = np.asarray(archive["probabilities"], dtype=np.float32)
                        trial_ids = archive["trial_id"].astype(str)
                        sessions = archive["session"].astype(str)
                    if len(set(trial_ids.tolist())) != labels.size:
                        raise RuntimeError("prediction trial identities are not unique")
                    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
                        raise RuntimeError("nominal probabilities are not normalised")
                    shifted = logits - logits.max(axis=1, keepdims=True)
                    recomputed_probability = np.exp(shifted)
                    recomputed_probability /= recomputed_probability.sum(axis=1, keepdims=True)
                    if not np.allclose(probabilities, recomputed_probability, atol=2e-6):
                        raise RuntimeError("nominal probabilities do not equal softmax(logits)")
                    if not np.array_equal(prediction, probabilities.argmax(axis=1)):
                        raise RuntimeError("nominal predictions are not probability argmax")
                    metrics = read_json(evaluation / "metrics.json")
                    barrier_record = next(
                        row
                        for row in barrier["records"]
                        if row["model"] == model
                        and int(row["subject"]) == int(subject)
                        and int(row["seed"]) == int(seed)
                    )
                    if metrics["checkpoint_file_sha256"] != barrier_record["checkpoint_file_sha256"]:
                        raise RuntimeError("evaluation metrics reference a checkpoint outside the barrier")
                    if metrics["checkpoint_barrier_sha256"] != barrier["combined_sha256"]:
                        raise RuntimeError("evaluation metrics reference another checkpoint barrier")
                    recomputed = classification_metrics(
                        labels, prediction, n_classes=int(schema["n_classes"])
                    )
                    for metric in METRIC_KEYS:
                        if not _close(metrics[metric], recomputed[metric]):
                            raise RuntimeError(f"nominal {metric} was not independently reproducible")
                    calibration = multiclass_calibration_metrics(logits, labels)
                    for metric, value in calibration.items():
                        if not _close(metrics["calibration"][metric], value):
                            raise RuntimeError(f"calibration metric {metric} differs")
                    expected_session = "E" if metrics["dataset"] == "bci2a" else "S2"
                    if set(sessions.tolist()) != {expected_session}:
                        raise RuntimeError("prediction session metadata violate the evaluation protocol")
                    _audit_perturbations(
                        evaluation / "perturbations.npz",
                        evaluation / "perturbation_metrics.csv",
                        labels,
                        float(recomputed["accuracy"]),
                    )
                    _audit_saliency(evaluation / "saliency.npz", labels, trial_ids)
                    state = read_json(evaluation / "state_audit.json")
                    if not state.get("identical") or state["before_sha256"] != state["after_sha256"]:
                        raise RuntimeError("model state changed during evaluation")
                    rows.append(metrics)
                except Exception as exc:  # audit must retain all failures
                    issues.append(
                        {
                            "stage": "evaluation",
                            "model": model,
                            "subject": int(subject),
                            "seed": int(seed),
                            "error": str(exc),
                        }
                    )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset", choices=("bci2a", "openbmi"), required=True)
    parser.add_argument("--canary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = Path(args.output_root) / ("canary" if args.canary else "") / args.dataset
    contract = read_json(output / "contract.json")
    contract_body = {key: value for key, value in contract.items() if key != "combined_sha256"}
    if contract.get("combined_sha256") != sha256_fingerprint(contract_body):
        raise RuntimeError("campaign contract digest is invalid")
    scope = dict(contract["scope"])
    config = dict(contract["resolved_config"])
    if config.get("status") != "posthoc_explanatory_after_bci2a_e_and_openbmi_s2_opened":
        raise RuntimeError("campaign does not explicitly declare its post-hoc status")
    barrier = _verify_barrier(output, contract)
    issues: list[dict[str, Any]] = []
    training = _audit_training(
        output=output,
        scope=scope,
        config=config,
        dataset=args.dataset,
        barrier=barrier,
        issues=issues,
    )
    evaluation = _audit_evaluation(
        output=output, scope=scope, barrier=barrier, issues=issues
    )
    expected = len(scope["models"]) * len(scope["subjects"]) * len(scope["seeds"])
    if len(training) != expected:
        issues.append(
            {"stage": "aggregate", "error": f"valid training runs {len(training)} != {expected}"}
        )
    if len(evaluation) != expected:
        issues.append(
            {"stage": "aggregate", "error": f"valid evaluation runs {len(evaluation)} != {expected}"}
        )
    audit_dir = ensure_dir(output / "audit")
    write_csv(audit_dir / "issues.csv", issues, fieldnames=None)
    report = {
        "schema": "dpc-snn-v8-posthoc-publication-audit/v1",
        "dataset": args.dataset,
        "status": "passed" if not issues else "failed",
        "posthoc_explanatory": True,
        "expected_runs": expected,
        "valid_training_runs": len(training),
        "valid_evaluation_runs": len(evaluation),
        "checkpoint_barrier_sha256": barrier["combined_sha256"],
        "contract_sha256": contract["combined_sha256"],
        "issues": issues,
    }
    write_json(audit_dir / "audit_report.json", report)
    print(json.dumps(report, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
