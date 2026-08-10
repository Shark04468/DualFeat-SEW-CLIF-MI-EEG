#!/usr/bin/env python3
"""Run the frozen V8 ensemble on BCI2a Session T to Session E exactly once."""

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
from dpc_snn.baselines.neural import verify_official_source_locks  # noqa: E402
from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v62_baselines import (  # noqa: E402
    BASELINE_OPTIMIZERS,
    apply_fixed_gain,
    fit_baseline,
    fit_fixed_gain,
    predict_baseline,
    prepare_model_input,
    task_carrier,
)
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    sha256_fingerprint,
    validate_run_artifact_manifest,
    validate_trial_metadata,
    write_run_artifact_manifest,
    write_trial_predictions,
)
from dpc_snn.experiments.v62_scaffold import (  # noqa: E402
    project_registered_max_norm_constraints_,
)
from dpc_snn.experiments.v8_ensemble import (  # noqa: E402
    entropy_residual_prediction,
    equal_probability_anchor,
    extract_atc_sequence,
    softmax_probability,
    state_digest,
    validate_ensemble_freeze_contract,
)
from dpc_snn.experiments.v8_maintenance import (  # noqa: E402
    validate_ensemble_maintenance_manifest,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    assert_v8_data_access,
    build_v8_run_fingerprint,
    collect_source_tree_manifest,
    mapping_sha256,
    source_tree_digest,
    validate_v8_freeze_manifest,
    validate_v8_resume_fingerprint,
    write_v8_fingerprint,
)
from dpc_snn.experiments.v8_sequence_decoder_training import (  # noqa: E402
    fit_v8_sequence_decoder,
    predict_v8_sequence_decoder,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v8_e1_baselines import _environment  # noqa: E402
from scripts.run_v8_e2_zero_delay import _subject_file  # noqa: E402


PREDICTION_BASES = (
    "predictions",
    "matched_ann_predictions",
    "anchor_predictions",
    "atcnet_predictions",
    "fbcnet_predictions",
    "decoder_snn_predictions",
    "decoder_ann_predictions",
)
RUN_FILES = (
    "manifest.json",
    "resolved_config.yaml",
    "source_fingerprint.json",
    "source_tree_manifest.json",
    "data_identity.json",
    "data_access_manifest.json",
    "state_audit.json",
    "gain.npz",
    "atcnet_history.csv",
    "fbcnet_history.csv",
    "sew_clif_history.csv",
    "ann_sew_history.csv",
    "atcnet.pt",
    "fbcnet.pt",
    "sew_clif.pt",
    "ann_sew.pt",
    "metrics.json",
    "runtime_status.json",
    "stdout.log",
    "stderr.log",
) + tuple(f"{base}.{suffix}" for base in PREDICTION_BASES for suffix in ("npz", "csv"))
CAMPAIGN_FILES = (
    "manifest.json",
    "campaign_status.json",
    "summary.csv",
    "resolved_campaign.yaml",
    "source_tree_manifest.json",
    "freeze_manifest.json",
    "source_maintenance_manifest.json",
    "official_source_locks.json",
)


def _csv(value: str, cast: Any = str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else value


def _session_data_hashes(train_path: Path, evaluation_path: Path) -> dict[str, str]:
    """Bind both physically separated sessions without basename collisions."""

    return {
        f"session_t/{train_path.name}": file_sha256(train_path),
        f"session_e/{evaluation_path.name}": file_sha256(evaluation_path),
    }


def _state_component_digests(model: torch.nn.Module) -> dict[str, str]:
    """Hash each persistent tensor so mutation failures name the changed state."""

    return mapping_sha256(model.state_dict())


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _metadata(data: dict[str, Any], *, expected_session: str, role: str) -> list[dict[str, Any]]:
    count = int(np.asarray(data["X"]).shape[0])
    if count != 288:
        raise RuntimeError(
            f"BCI2a Session-{expected_session} must contain 288 trials, got {count}"
        )
    rows = validate_trial_metadata(
        [
            {
                "dataset": str(_scalar(data.get("dataset_name", "bci2a"))),
                "subject": str(np.asarray(data["subject"]).astype(str)[index]),
                "session": str(np.asarray(data["session"]).astype(str)[index]),
                "run": str(np.asarray(data["run"]).astype(str)[index]),
                "trial_id": str(np.asarray(data["trial_id"]).astype(str)[index]),
                "class": int(np.asarray(data["y"])[index]),
                "sfreq": float(_scalar(data["sfreq"])),
                "ch_names": [str(name) for name in data["ch_names"]],
                "epoch_tmin": float(_scalar(data["epoch_tmin"])),
                "epoch_tmax": float(_scalar(data["epoch_tmax"])),
            }
            for index in range(count)
        ],
        allowed_sessions=(expected_session,),
    )
    if {row["session"] for row in rows} != {expected_session}:
        raise RuntimeError(f"data file is not Session-{expected_session} only")
    assert_v8_data_access(rows, stage="bci2a_evaluation", role=role)
    labels = np.asarray(data["y"], dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    if classes.tolist() != [0, 1, 2, 3] or counts.tolist() != [72, 72, 72, 72]:
        raise RuntimeError(f"Session-{expected_session} labels are not balanced four-class data")
    return rows


def _log_probability(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("probability matrix is invalid")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("probability rows are not normalized")
    return np.log(np.clip(values, 1e-12, 1.0)).astype(np.float32)


def _arm_metrics(probability: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    pred = np.asarray(probability).argmax(axis=1)
    return {
        **classification_metrics(labels, pred, n_classes=4),
        **multiclass_calibration_metrics(_log_probability(probability), labels),
    }


def _write_arm_predictions(
    run_dir: Path,
    *,
    basename: str,
    logits: np.ndarray,
    probability: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict[str, Any]],
    seed: int,
    model: str,
) -> None:
    write_trial_predictions(
        run_dir,
        logits=np.asarray(logits, dtype=np.float32),
        probabilities=np.asarray(probability, dtype=np.float32),
        pred=np.asarray(probability).argmax(axis=1),
        label=labels,
        subject=[row["subject"] for row in metadata],
        session="E",
        run=[row["run"] for row in metadata],
        trial_id=[row["trial_id"] for row in metadata],
        seed=seed,
        model=model,
        basename=basename,
    )


def _run_one(
    *,
    output: Path,
    train_path: Path,
    evaluation_path: Path,
    source_root: Path,
    freeze: dict[str, Any],
    maintenance: dict[str, Any],
    source_tree: dict[str, str],
    environment: dict[str, Any],
    official_locks: dict[str, Any],
    subject: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    component_rule = dict(freeze["checkpoint_rule"]["components"])
    model_config = dict(freeze["architecture"]["model_config"])
    preprocessing = dict(freeze["preprocessing"])
    augmentation = dict(freeze["augmentation"])
    decoder_config = dict(model_config["decoder"])
    run_seed = int(seed) * 100_003 + int(subject) * 1_009
    decoder_seed = run_seed + 8_000_003
    resolved = {
        "stage": "E6",
        "protocol": "bci2a_session_t_train_session_e_single_evaluation",
        "freeze_sha256": freeze["combined_sha256"],
        "source_maintenance_sha256": maintenance["combined_sha256"],
        "subject": int(subject),
        "seed": int(seed),
        "run_seed": run_seed,
        "decoder_seed": decoder_seed,
        "architecture": model_config,
        "checkpoint_rule": component_rule,
        "preprocessing": preprocessing,
        "augmentation": augmentation,
        "baseline_optimizers": {
            name: BASELINE_OPTIMIZERS[name] for name in ("atcnet", "fbcnet")
        },
    }
    data_hashes = _session_data_hashes(train_path, evaluation_path)
    train_hash = data_hashes[f"session_t/{train_path.name}"]
    evaluation_hash = data_hashes[f"session_e/{evaluation_path.name}"]
    train_data = load_processed_npz(train_path)
    metadata_t = _metadata(train_data, expected_session="T", role="training")
    split = {
        "train_session": "T",
        "evaluation_session": "E",
        "train_trial_ids": [row["trial_id"] for row in metadata_t],
        "evaluation_file_identity_sha256": evaluation_hash,
        "evaluation_labels_used_for_checkpoint_selection": False,
    }
    fingerprint = build_v8_run_fingerprint(
        resolved_run_config=resolved,
        source_tree=source_tree,
        data=data_hashes,
        split=split,
        augmentation=augmentation,
        prior={"policy": "none", "delay_enabled": False},
        checkpoint={
            "freeze_sha256": freeze["combined_sha256"],
            "source_maintenance_sha256": maintenance["combined_sha256"],
            **component_rule,
        },
        environment={**environment, "official_source_locks": official_locks},
    )
    run_dir = ensure_dir(output / f"subject_{subject:02d}" / f"seed_{seed}")
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
    write_json(
        run_dir / "data_identity.json",
        {
            "train": {
                "logical_key": f"session_t/{train_path.name}",
                "sha256": train_hash,
                "semantic_load_before_training": True,
            },
            "evaluation": {
                "logical_key": f"session_e/{evaluation_path.name}",
                "sha256": evaluation_hash,
                "byte_hash_before_training": True,
                "semantic_load_before_checkpoint": False,
            },
        },
    )
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    write_json(
        run_dir / "runtime_status.json",
        {
            "status": "training_session_t",
            "started_at": time.time(),
            "session_e_arrays_loaded": False,
            "session_e_identity_hash_computed": True,
        },
    )

    sfreq = float(_scalar(train_data["sfreq"]))
    epoch_tmin = float(_scalar(train_data["epoch_tmin"]))
    labels_t = np.asarray(train_data["y"], dtype=np.int64)
    carrier_t = task_carrier(
        np.asarray(train_data["X"], dtype=np.float32),
        sfreq=sfreq,
        epoch_tmin=epoch_tmin,
    )
    gain = fit_fixed_gain(carrier_t, clip=float(preprocessing["clip_after_gain"]))
    normalized_t = apply_fixed_gain(carrier_t, gain)
    atc_t = prepare_model_input("atcnet", normalized_t, sfreq=sfreq)
    fbc_t = prepare_model_input("fbcnet", normalized_t, sfreq=sfreq)

    atc_fit = fit_baseline(
        "atcnet",
        source_root=source_root,
        x_train=atc_t,
        y_train=labels_t,
        x_validation=None,
        y_validation=None,
        device=device,
        seed=run_seed,
        epochs=int(component_rule["atcnet_final_epoch"]),
        patience=int(component_rule["atcnet_final_epoch"]),
        augmentation=augmentation,
        fixed_epoch=int(component_rule["atcnet_final_epoch"]),
        scheduler_epochs=int(component_rule["atcnet_scheduler_horizon"]),
        run_label=f"V8-E6-ensemble:atcnet:S{subject}:seed{seed}",
    )
    fbc_fit = fit_baseline(
        "fbcnet",
        source_root=source_root,
        x_train=fbc_t,
        y_train=labels_t,
        x_validation=None,
        y_validation=None,
        device=device,
        seed=run_seed,
        epochs=int(component_rule["fbcnet_final_epoch"]),
        patience=int(component_rule["fbcnet_final_epoch"]),
        augmentation=augmentation,
        fixed_epoch=int(component_rule["fbcnet_final_epoch"]),
        scheduler_epochs=int(component_rule["fbcnet_scheduler_horizon"]),
        run_label=f"V8-E6-ensemble:fbcnet:S{subject}:seed{seed}",
    )
    sequence_t, atc_teacher_t = extract_atc_sequence(
        atc_fit.model, atc_t, device=device
    )
    atc_replay_t = predict_baseline(
        atc_fit.model,
        atc_t,
        labels_t,
        device=device,
        batch_size=int(BASELINE_OPTIMIZERS["atcnet"]["batch_size"]),
    )
    replay_error = float(np.max(np.abs(atc_replay_t["logits"] - atc_teacher_t)))
    if replay_error > 1e-5:
        raise RuntimeError(f"ATC sequence wrapper changed teacher logits by {replay_error}")

    decoder_kwargs = {
        "hidden_channels": int(decoder_config["hidden_channels"]),
        "decoder_layers": int(decoder_config["layers"]),
        "readout_features": int(decoder_config["readout_features"]),
        "dropout": float(decoder_config["dropout"]),
    }
    decoder_common = {
        "x_train": sequence_t,
        "y_train": labels_t,
        "teacher_train": atc_teacher_t,
        "x_validation": None,
        "y_validation": None,
        "teacher_validation": None,
        "device": device,
        "seed": decoder_seed,
        "epochs": int(component_rule["decoder_final_epoch"]),
        "fixed_epoch": int(component_rule["decoder_final_epoch"]),
        "scheduler_epochs": int(component_rule["decoder_scheduler_horizon"]),
        "distillation_weight": float(decoder_config["distillation_weight"]),
        "distillation_temperature": float(
            decoder_config["distillation_temperature"]
        ),
        "firing_rate_weight": float(decoder_config["firing_rate_weight"]),
        "model_kwargs": decoder_kwargs,
    }
    snn_fit = fit_v8_sequence_decoder(
        "sew_clif",
        **decoder_common,
        run_label=f"V8-E6-ensemble:sew_clif:S{subject}:seed{seed}",
    )
    ann_fit = fit_v8_sequence_decoder(
        "ann_sew",
        **decoder_common,
        run_label=f"V8-E6-ensemble:ann_sew:S{subject}:seed{seed}",
    )

    models = {
        "atcnet": atc_fit.model,
        "fbcnet": fbc_fit.model,
        "sew_clif": snn_fit.model,
        "ann_sew": ann_fit.model,
    }
    constraint_projection_counts = {
        name: project_registered_max_norm_constraints_(model)
        for name, model in models.items()
    }
    checkpoint_states = {name: _cpu_state(model) for name, model in models.items()}
    state_components_before_e = {
        name: _state_component_digests(model) for name, model in models.items()
    }
    state_before_e = {name: state_digest(model) for name, model in models.items()}
    checkpoint_state_sha256 = {
        name: sha256_fingerprint(mapping_sha256(state))
        for name, state in checkpoint_states.items()
    }
    if checkpoint_state_sha256 != state_before_e:
        raise RuntimeError("serialized checkpoint state differs from the active model")
    torch.save(checkpoint_states["atcnet"], run_dir / "atcnet.pt")
    torch.save(checkpoint_states["fbcnet"], run_dir / "fbcnet.pt")
    torch.save(checkpoint_states["sew_clif"], run_dir / "sew_clif.pt")
    torch.save(checkpoint_states["ann_sew"], run_dir / "ann_sew.pt")
    write_csv(run_dir / "atcnet_history.csv", atc_fit.history)
    write_csv(run_dir / "fbcnet_history.csv", fbc_fit.history)
    write_csv(run_dir / "sew_clif_history.csv", snn_fit.history)
    write_csv(run_dir / "ann_sew_history.csv", ann_fit.history)
    np.savez_compressed(
        run_dir / "gain.npz",
        values=np.asarray(gain.values, dtype=np.float32),
        clip=np.asarray(gain.clip, dtype=np.float32),
    )
    write_json(
        run_dir / "runtime_status.json",
        {
            "status": "checkpointed_before_session_e",
            "checkpointed_at": time.time(),
            "session_e_arrays_loaded": False,
            "session_e_identity_hash_computed": True,
            "state_sha256": state_before_e,
        },
    )

    # This is the first semantic load of Session E arrays and labels. Every
    # trainable state is already serialized and hashed before this boundary.
    evaluation_data = load_processed_npz(evaluation_path)
    metadata_e = _metadata(evaluation_data, expected_session="E", role="evaluation")
    if {row["subject"] for row in metadata_t} != {row["subject"] for row in metadata_e}:
        raise RuntimeError("Session T and E subject identities differ")
    for field in ("sfreq", "epoch_tmin", "epoch_tmax"):
        if float(_scalar(train_data[field])) != float(_scalar(evaluation_data[field])):
            raise RuntimeError(f"Session T and E acquisition field differs: {field}")
    if not np.array_equal(train_data["ch_names"], evaluation_data["ch_names"]):
        raise RuntimeError("Session T and E channel order differs")
    labels_e = np.asarray(evaluation_data["y"], dtype=np.int64)
    carrier_e = task_carrier(
        np.asarray(evaluation_data["X"], dtype=np.float32),
        sfreq=sfreq,
        epoch_tmin=epoch_tmin,
    )
    normalized_e = apply_fixed_gain(carrier_e, gain)
    atc_e = prepare_model_input("atcnet", normalized_e, sfreq=sfreq)
    fbc_e = prepare_model_input("fbcnet", normalized_e, sfreq=sfreq)
    sequence_e, atc_logits = extract_atc_sequence(atc_fit.model, atc_e, device=device)
    fbc_evaluation = predict_baseline(
        fbc_fit.model,
        fbc_e,
        labels_e,
        device=device,
        batch_size=int(BASELINE_OPTIMIZERS["fbcnet"]["batch_size"]),
    )
    snn_evaluation = predict_v8_sequence_decoder(
        snn_fit.model,
        sequence_e,
        labels_e,
        atc_logits,
        device=device,
    )
    ann_evaluation = predict_v8_sequence_decoder(
        ann_fit.model,
        sequence_e,
        labels_e,
        atc_logits,
        device=device,
    )

    atc_probability = softmax_probability(atc_logits)
    fbc_probability = softmax_probability(fbc_evaluation["logits"])
    anchor_probability = equal_probability_anchor(atc_logits, fbc_evaluation["logits"])
    snn_decoder_probability = softmax_probability(snn_evaluation["logits"])
    ann_decoder_probability = softmax_probability(ann_evaluation["logits"])
    maximum_weight = float(model_config["residual_fusion"]["maximum_decoder_weight"])
    primary = entropy_residual_prediction(
        anchor_probability,
        snn_evaluation["logits"],
        maximum_weight=maximum_weight,
    )
    matched_ann = entropy_residual_prediction(
        anchor_probability,
        ann_evaluation["logits"],
        maximum_weight=maximum_weight,
    )

    state_after_e = {name: state_digest(model) for name, model in models.items()}
    state_components_after_e = {
        name: _state_component_digests(model) for name, model in models.items()
    }
    changed_state_components = {
        name: sorted(
            key
            for key in set(state_components_before_e[name])
            | set(state_components_after_e[name])
            if state_components_before_e[name].get(key)
            != state_components_after_e[name].get(key)
        )
        for name in models
        if state_components_before_e[name] != state_components_after_e[name]
    }
    write_json(
        run_dir / "state_audit.json",
        {
            "checkpoint_before_session_e": state_before_e,
            "state_after_session_e": state_after_e,
            "changed_state_components": changed_state_components,
            "identical": not changed_state_components,
            "checkpoint_state_sha256": checkpoint_state_sha256,
            "constraint_projection_counts": constraint_projection_counts,
            "atc_train_wrapper_max_abs_logit_error": replay_error,
        },
    )
    if state_before_e != state_after_e:
        raise RuntimeError(
            "a frozen model state changed during Session E evaluation: "
            + json.dumps(changed_state_components, sort_keys=True)
        )
    arms = {
        "primary": primary["probabilities"],
        "matched_ann": matched_ann["probabilities"],
        "anchor": anchor_probability,
        "atcnet": atc_probability,
        "fbcnet": fbc_probability,
        "decoder_snn": snn_decoder_probability,
        "decoder_ann": ann_decoder_probability,
    }
    arm_metrics = {name: _arm_metrics(probability, labels_e) for name, probability in arms.items()}

    _write_arm_predictions(
        run_dir,
        basename="predictions",
        logits=_log_probability(primary["probabilities"]),
        probability=primary["probabilities"],
        labels=labels_e,
        metadata=metadata_e,
        seed=seed,
        model="v8_e6_entropy_residual_sew_clif",
    )
    prediction_specs = (
        ("matched_ann_predictions", matched_ann["probabilities"], "v8_e6_entropy_residual_ann_sew"),
        ("anchor_predictions", anchor_probability, "v8_e6_atc_fbc_anchor"),
        ("atcnet_predictions", atc_probability, "v8_e6_atcnet"),
        ("fbcnet_predictions", fbc_probability, "v8_e6_fbcnet"),
        ("decoder_snn_predictions", snn_decoder_probability, "v8_e6_sew_clif_decoder"),
        ("decoder_ann_predictions", ann_decoder_probability, "v8_e6_ann_sew_decoder"),
    )
    for basename, probability, model_name in prediction_specs:
        _write_arm_predictions(
            run_dir,
            basename=basename,
            logits=_log_probability(probability),
            probability=probability,
            labels=labels_e,
            metadata=metadata_e,
            seed=seed,
            model=model_name,
        )

    access = {
        "stage": "bci2a_evaluation",
        "train_session": "T",
        "train_trials": len(metadata_t),
        "evaluation_session": "E",
        "evaluation_trials": len(metadata_e),
        "session_e_byte_hash_computed_before_training": evaluation_hash,
        "session_e_arrays_first_loaded_after_all_checkpoint_hashes": state_before_e,
        "session_e_checkpoint_selection": False,
        "session_e_gradient_updates": False,
        "openbmi_s2_accessed": False,
    }
    write_json(run_dir / "data_access_manifest.json", access)
    metrics = {
        "status": "completed",
        "stage": "E6_ENSEMBLE",
        "protocol": "bci2a_session_t_train_session_e_single_evaluation",
        "subject": int(subject),
        "seed": int(seed),
        "accuracy": arm_metrics["primary"]["accuracy"],
        "balanced_accuracy": arm_metrics["primary"]["balanced_accuracy"],
        "kappa": arm_metrics["primary"]["kappa"],
        "macro_f1": arm_metrics["primary"]["macro_f1"],
        "negative_log_likelihood": arm_metrics["primary"]["negative_log_likelihood"],
        "ece": arm_metrics["primary"]["ece"],
        "arms": arm_metrics,
        "snn_mean_firing_rate": float(snn_evaluation["mean_firing_rate"]),
        "ann_mean_firing_rate": float(ann_evaluation["mean_firing_rate"]),
        "entropy_gate_mean": float(np.mean(primary["gate"])),
        "entropy_gate_minimum": float(np.min(primary["gate"])),
        "entropy_gate_maximum": float(np.max(primary["gate"])),
        "parameters": {
            name: sum(parameter.numel() for parameter in model.parameters())
            for name, model in models.items()
        },
        "constraint_projection_counts": constraint_projection_counts,
        "optimizer_steps": {
            "atcnet": atc_fit.optimizer_steps,
            "fbcnet": fbc_fit.optimizer_steps,
            "sew_clif": snn_fit.optimizer_steps,
            "ann_sew": ann_fit.optimizer_steps,
        },
        "train_seconds": {
            "atcnet": atc_fit.elapsed_seconds,
            "fbcnet": fbc_fit.elapsed_seconds,
            "sew_clif": snn_fit.elapsed_seconds,
            "ann_sew": ann_fit.elapsed_seconds,
        },
        "freeze_sha256": freeze["combined_sha256"],
        "source_maintenance_sha256": maintenance["combined_sha256"],
        "run_fingerprint": fingerprint["combined_sha256"],
        "heldout_e_selected_checkpoint": False,
        "logit_storage_semantics": "log_normalized_probability_for_probability_fusions",
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "runtime_status.json",
        {
            "status": "completed",
            "completed_at": time.time(),
            "session_e_arrays_loaded": True,
            "session_e_identity_hash_computed": True,
        },
    )
    (run_dir / "stdout.log").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    write_run_artifact_manifest(run_dir, required_files=RUN_FILES)
    return metrics


def _worker_command(
    *,
    args: argparse.Namespace,
    output: Path,
    subject: int,
    seeds: list[int],
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--train-data",
        str(Path(args.train_data).resolve()),
        "--evaluation-data",
        str(Path(args.evaluation_data).resolve()),
        "--source-root",
        str(Path(args.source_root).resolve()),
        "--freeze",
        str(Path(args.freeze).resolve()),
        "--maintenance",
        str(Path(args.maintenance).resolve()),
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
    worker_threads = int(environment.get("DPC_SNN_E6_ENSEMBLE_WORKER_THREADS", "8"))
    if worker_threads < 1 or worker_threads > 32:
        raise ValueError("DPC_SNN_E6_ENSEMBLE_WORKER_THREADS must lie in [1, 32]")
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
            "E6 ensemble worker failed with code "
            f"{completed.returncode}: {' '.join(command)}\n{completed.stderr[-8000:]}"
        )
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError("E6 ensemble worker missed results: " + ", ".join(missing))
    return [read_json(path) for path in expected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--evaluation-data", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--maintenance", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default="configs/experiments/v8_e6_ensemble_frozen.yaml"
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
    train_root = Path(args.train_data).resolve()
    evaluation_root = Path(args.evaluation_data).resolve()
    if train_root == evaluation_root:
        raise RuntimeError("Session T and E must be stored in physically separate roots")
    source_root = Path(args.source_root).resolve()
    output = ensure_dir(Path(args.output).resolve())
    config = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    project_sources = collect_source_tree_manifest(ROOT)
    project_digest = source_tree_digest(project_sources)
    freeze = validate_ensemble_freeze_contract(
        validate_v8_freeze_manifest(Path(args.freeze).resolve())
    )
    maintenance = validate_ensemble_maintenance_manifest(
        read_json(Path(args.maintenance).resolve()),
        expected_parent_freeze_sha256=freeze["combined_sha256"],
        expected_current_source_tree_sha256=project_digest,
    )
    official_locks = verify_official_source_locks(source_root)
    if official_locks != freeze["baselines"]["official_source_locks"]:
        raise RuntimeError("official baseline sources differ from the frozen sources")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    subjects = _csv(args.subjects, int) if args.subjects else list(config["subjects"])
    seeds = _csv(args.seeds, int) if args.seeds else list(config["seeds"])
    if not args.canary and (
        subjects != list(config["subjects"]) or seeds != list(config["seeds"])
    ):
        raise RuntimeError("formal E6 coverage must be exactly 9 subjects x 5 seeds")
    workers = int(args.workers)
    maximum_workers = int(config["execution"]["maximum_parallel_subject_workers"])
    if workers < 1 or workers > maximum_workers:
        raise ValueError(f"E6 workers must lie in [1, {maximum_workers}]")
    environment = _environment()

    if args.worker_subject is not None:
        worker_seeds = _csv(args.worker_seeds, int)
        if not worker_seeds:
            raise RuntimeError("worker mode requires --worker-seeds")
        subject = int(args.worker_subject)
        train_path = _subject_file(train_root, subject)
        evaluation_path = _subject_file(evaluation_root, subject)
        rows = [
            _run_one(
                output=output,
                train_path=train_path,
                evaluation_path=evaluation_path,
                source_root=source_root,
                freeze=freeze,
                maintenance=maintenance,
                source_tree=project_sources,
                environment=environment,
                official_locks=official_locks,
                subject=subject,
                seed=seed,
                device=args.device,
            )
            for seed in worker_seeds
        ]
        print(json.dumps({"status": "completed", "runs": len(rows)}, indent=2))
        return

    write_json(output / "source_tree_manifest.json", project_sources)
    write_json(output / "freeze_manifest.json", freeze)
    write_json(output / "source_maintenance_manifest.json", maintenance)
    write_json(output / "official_source_locks.json", official_locks)
    (output / "resolved_campaign.yaml").write_text(
        yaml.safe_dump(
            {
                **config,
                "active_subjects": subjects,
                "active_seeds": seeds,
                "workers": workers,
                "freeze_sha256": freeze["combined_sha256"],
                "source_maintenance_sha256": maintenance["combined_sha256"],
                "source_tree_sha256": project_digest,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    started = time.time()
    all_rows: list[dict[str, Any]] = []
    jobs = []
    for subject in subjects:
        expected = [
            output / f"subject_{subject:02d}" / f"seed_{seed}" / "metrics.json"
            for seed in seeds
        ]
        jobs.append(
            (
                _worker_command(args=args, output=output, subject=subject, seeds=seeds),
                expected,
            )
        )
    if workers == 1:
        for command, expected in jobs:
            all_rows.extend(_run_worker(command, expected))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_worker, command, expected): expected
                for command, expected in jobs
            }
            for future in as_completed(futures):
                all_rows.extend(future.result())

    all_rows.sort(key=lambda row: (int(row["subject"]), int(row["seed"])))
    expected_pairs = {(subject, seed) for subject in subjects for seed in seeds}
    observed_pairs = {(int(row["subject"]), int(row["seed"])) for row in all_rows}
    full_contract = (
        subjects == list(config["subjects"])
        and seeds == list(config["seeds"])
        and observed_pairs == expected_pairs
        and len(all_rows) == len(expected_pairs)
    )
    write_csv(
        output / "summary.csv",
        [
            {
                "subject": row["subject"],
                "seed": row["seed"],
                "accuracy": row["accuracy"],
                "balanced_accuracy": row["balanced_accuracy"],
                "kappa": row["kappa"],
                "macro_f1": row["macro_f1"],
                "negative_log_likelihood": row["negative_log_likelihood"],
                "ece": row["ece"],
                "anchor_accuracy": row["arms"]["anchor"]["accuracy"],
                "atcnet_accuracy": row["arms"]["atcnet"]["accuracy"],
                "fbcnet_accuracy": row["arms"]["fbcnet"]["accuracy"],
                "matched_ann_accuracy": row["arms"]["matched_ann"]["accuracy"],
                "snn_mean_firing_rate": row["snn_mean_firing_rate"],
            }
            for row in all_rows
        ],
    )
    status = {
        "status": "completed",
        "stage": "E6_ENSEMBLE",
        "protocol": config["protocol"],
        "subjects": subjects,
        "seeds": seeds,
        "runs": len(all_rows),
        "full_registered_contract": full_contract,
        "freeze_sha256": freeze["combined_sha256"],
        "source_maintenance_sha256": maintenance["combined_sha256"],
        "source_tree_sha256": project_digest,
        "session_e_used_for_selection": False,
        "session_e_gradient_updates": False,
        "openbmi_s2_accessed": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "campaign_status.json", status)
    write_run_artifact_manifest(output, required_files=CAMPAIGN_FILES)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
