#!/usr/bin/env python3
"""Run a Session-T-only paired delay stage for the V7 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.baselines.neural import verify_official_source_locks  # noqa: E402
from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    build_run_fingerprint,
    file_sha256,
    session_t_run_grouped_folds,
    validate_resume_fingerprint,
    validate_trial_metadata,
    write_fingerprint_manifest,
    write_trial_predictions,
)
from dpc_snn.experiments.v62_scaffold import (  # noqa: E402
    cache_analytic_fixed_channel_gain_rate_features,
    fit_official_fbc_channel_gain,
    fit_zero_scaffold,
    predict_scaffold,
)
from dpc_snn.experiments.v7_delay import (  # noqa: E402
    build_delay_stage_model,
    delay_diagnostics,
    paired_delay_gate,
    state_sha256,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import paired_prediction_comparison  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402


def _parse_csv(value: str, cast: Any = str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for root in (ROOT / "src", ROOT / "scripts", ROOT / "configs"):
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in {
                ".py", ".yaml", ".yml", ".toml", ".sh"
            }:
                hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = file_sha256(path)
    return hashes


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "platform": platform.platform(),
    }


_REGISTERED_MODEL_OVERRIDES = {
    "atc_delayed_statistics_uncertainty_gate",
    "atc_delayed_statistics_cp_rank",
    "atc_delayed_statistics_bins",
    "atc_delayed_statistics_directional_moments",
    "atc_delayed_route_statistics_cp_rank",
    "atc_delayed_route_statistics_bins",
    "phase_pair_current_enabled",
    "phase_pair_current_bound",
    "phase_pair_initial_scale",
}


def _apply_registered_model_overrides(
    model_config: dict[str, Any],
    overrides: dict[str, Any],
) -> None:
    unknown = set(overrides) - _REGISTERED_MODEL_OVERRIDES
    if unknown:
        raise RuntimeError(f"unregistered E3 model overrides: {sorted(unknown)}")
    model_config.update(overrides)


def _validate_e3_readout_config(
    readout_contract: str,
    model_config: dict[str, Any],
) -> None:
    if int(model_config.get("atc_delayed_statistics_bins", 0)) != 0:
        raise RuntimeError("E3 forbids the generic delayed-statistics classifier")
    route_bins = int(model_config.get("atc_delayed_route_statistics_bins", 0))
    route_rank = int(model_config.get("atc_delayed_route_statistics_cp_rank", 0))
    phase_enabled = bool(model_config.get("phase_pair_current_enabled", False))
    if readout_contract == "degree_normalized_audited_route_delay_contrast_only":
        if route_bins <= 0:
            raise RuntimeError("route-only E3 requires delayed route-statistics bins")
        if route_rank <= 0:
            raise RuntimeError("route-only E3 requires a positive route CP rank")
        if phase_enabled:
            raise RuntimeError("route-only E3 forbids the R9 phase current")
    elif readout_contract == "pre_decoder_audited_phase_pair_current_only":
        if route_bins != 0 or route_rank != 0:
            raise RuntimeError("R9 phase-current E3 forbids delayed logit heads")
        if not phase_enabled:
            raise RuntimeError("R9 phase-current E3 requires phase current")
        if float(model_config.get("phase_pair_current_bound", 0.0)) <= 0.0:
            raise RuntimeError("R9 phase-current E3 requires a positive adapter bound")
    else:
        raise RuntimeError(f"unregistered E3 readout contract: {readout_contract}")


def _validate_e3_readout_model(
    model: torch.nn.Module,
    readout_contract: str,
) -> None:
    readout = getattr(model, "atc_readout", None)
    if readout is None:
        raise RuntimeError("route-only E3 requires the ATC readout")
    if getattr(readout, "delayed_statistics_classifier", None) is not None:
        raise RuntimeError("route-only E3 instantiated a forbidden generic statistics head")
    route_head = getattr(readout, "delayed_route_statistics_classifier", None)
    phase_adapter = getattr(model, "phase_pair_adapter", None)
    if readout_contract == "degree_normalized_audited_route_delay_contrast_only":
        if route_head is None:
            raise RuntimeError("route-only E3 did not instantiate its route head")
        if phase_adapter is not None:
            raise RuntimeError("route-only E3 instantiated the R9 phase adapter")
    elif readout_contract == "pre_decoder_audited_phase_pair_current_only":
        if route_head is not None:
            raise RuntimeError("R9 phase-current E3 instantiated a logit head")
        if phase_adapter is None:
            raise RuntimeError("R9 phase-current E3 has no phase adapter")
    else:
        raise RuntimeError(f"unregistered E3 readout contract: {readout_contract}")


def _validate_route_only_readout_config(model_config: dict[str, Any]) -> None:
    """Backward-compatible strict validator used by the frozen R8 probes."""

    _validate_e3_readout_config(
        "degree_normalized_audited_route_delay_contrast_only",
        model_config,
    )


def _validate_route_only_readout_model(model: torch.nn.Module) -> None:
    """Backward-compatible strict model validator for frozen R8 probes."""

    _validate_e3_readout_model(
        model,
        "degree_normalized_audited_route_delay_contrast_only",
    )


def _fit_phase_amplitude_scale(fast: torch.Tensor) -> torch.Tensor:
    """Fit the R8 confidence scale from outer-training cached analytic data."""

    if fast.ndim != 4 or not fast.is_complex() or fast.shape[-1] % 2:
        raise ValueError("phase scale requires complex [N, B, K, 2*T] features")
    aligned = fast[..., ::2].abs()
    scale = aligned.median(dim=-1).values.median(dim=0).values
    if not bool(torch.isfinite(scale).all()):
        raise FloatingPointError("fold-local phase amplitude scale is non-finite")
    return scale.clamp_min(1e-6).float()


def _subject_file(data_root: Path, subject: int) -> Path:
    candidates = sorted(data_root.glob(f"*A{subject:02d}*.npz"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one processed file for subject {subject}, found {len(candidates)}"
        )
    return candidates[0]


def _metadata_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    count = len(data["y"])
    rows = []
    for index in range(count):
        rows.append(
            {
                "dataset": str(data.get("dataset_name", "bci2a")),
                "trial_id": str(np.asarray(data["trial_id"])[index]),
                "subject": int(np.asarray(data["subject"])[index]),
                "session": str(np.asarray(data["session"])[index]),
                "run": int(np.asarray(data["run"])[index]),
                "class": int(np.asarray(data["y"])[index]),
                "sfreq": float(data["sfreq"]),
                "ch_names": data["ch_names"],
                "epoch_tmin": float(data["epoch_tmin"]),
                "epoch_tmax": float(data["epoch_tmax"]),
            }
        )
    validate_trial_metadata(rows)
    return rows


def _parent_checkpoint(
    parent: Path,
    *,
    subject: int,
    seed: int,
    fold: int,
) -> Path:
    candidates = (
        parent / f"subject_{subject:02d}" / f"seed_{seed}" / f"fold_{fold}" / "best.pt",
        parent / f"subject_{subject:02d}" / "selection" / f"fold_{fold}" / "best.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "no matching parent checkpoint: " + ", ".join(str(path) for path in candidates)
    )


def _load_audited_fold_prior(
    root: Path,
    *,
    subject: int,
    fold: int,
    scope: str = "pooled",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    allowed_scopes = {"pooled", "class_0", "class_1", "class_2", "class_3"}
    if scope not in allowed_scopes:
        raise ValueError(f"unsupported evidence-prior scope: {scope!r}")
    subject_roots = (root / f"subject_{subject:02d}", root)
    for subject_root in subject_roots:
        scope_root = subject_root / f"fold_{fold}" / scope
        prior_path = scope_root / "fold_local_evidence_prior.npz"
        summary_path = scope_root / "fold_local_evidence_prior.json"
        manifest_path = subject_root / "manifest.json"
        if prior_path.is_file() and summary_path.is_file() and manifest_path.is_file():
            break
    else:
        raise FileNotFoundError(
            f"no {scope} audited prior for subject={subject}, fold={fold} under {root}"
        )
    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    if summary.get("subject") != subject or summary.get("fold") != fold:
        raise RuntimeError("audited prior subject/fold metadata does not match the run")
    if summary.get("scope") != scope:
        raise RuntimeError("audited prior scope metadata does not match the requested scope")
    required_checks = {
        "natural_nonzero_edges",
        "split_half_delay",
        "bootstrap_frequency",
        "time_reversal",
        "phase_surrogate",
    }
    evidence_checks = summary.get("evidence_checks")
    if (
        not bool(summary.get("evidence_pipeline_passed"))
        or not isinstance(evidence_checks, dict)
        or not required_checks.issubset(evidence_checks)
        or not all(bool(evidence_checks[key]) for key in required_checks)
    ):
        raise RuntimeError(
            f"the {scope} prior did not pass every registered evidence check"
        )
    if summary.get("fit_scope") != "inner_training_fold_only":
        raise RuntimeError("audited prior was not fitted exclusively on the inner-training fold")
    if bool(summary.get("session_e_accessed")) or bool(summary.get("heldout_data_accessed")):
        raise RuntimeError("audited prior accessed held-out data")
    if (
        manifest.get("status") != "completed"
        or not bool(manifest.get("scientific_gate_eligible"))
        or int(manifest.get("bootstrap_samples", 0)) < 128
        or bool(manifest.get("session_e_accessed"))
        or int(manifest.get("subject", -1)) != subject
        or fold not in [int(value) for value in manifest.get("folds", [])]
    ):
        raise RuntimeError("audited prior does not meet the 128-bootstrap scientific tier")
    manifest_scope_mode = str(manifest.get("scope_mode", ""))
    allowed_manifest_modes = {"all", "pooled"} if scope == "pooled" else {"all", "classes"}
    if manifest_scope_mode not in allowed_manifest_modes:
        raise RuntimeError("audit manifest does not cover the requested prior scope")
    if scope.startswith("class_") and "class_labels" in manifest:
        requested_label = int(scope.split("_", 1)[1])
        if requested_label not in [int(value) for value in manifest["class_labels"]]:
            raise RuntimeError("audit manifest class list does not cover the requested scope")
    expected_representation = (
        "v7_online_car_baseline_mean_fold_gain_causal_filterbank_"
        "exact_sensor_fast_aligned_to_slow"
    )
    if summary.get("analytic_representation") != expected_representation:
        raise RuntimeError("audited prior was not fitted on the registered V7 online frontend")
    with np.load(prior_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    required = {
        "route_probability",
        "positive_delay_probability",
        "fractional_delay_target",
    }
    if not required.issubset(arrays):
        raise RuntimeError("audited prior archive is missing the transport contract")
    provenance = {
        "policy": f"fold_local_v7_online_matched_{scope}_b128",
        "scope": scope,
        "prior_path": str(prior_path),
        "prior_sha256": file_sha256(prior_path),
        "summary_path": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "frontend_fingerprint": summary["frontend_fingerprint"],
        "evidence_pipeline_passed": True,
        "session_e_accessed": False,
    }
    return arrays, provenance


def _probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-output", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--evidence-prior-root", default="")
    parser.add_argument(
        "--evidence-prior-scope",
        choices=("pooled", "class_0", "class_1", "class_2", "class_3"),
        default="pooled",
    )
    parser.add_argument("--require-audited-prior", action="store_true")
    parser.add_argument(
        "--config", default="configs/experiments/v7_e3_static_slow_within.yaml"
    )
    parser.add_argument("--subjects", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--fold-ids", default="")
    parser.add_argument("--cv-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--readout-training-scope",
        choices=(
            "frozen",
            "delay_head",
            "fixed_delay_head",
            "fixed_head",
            "fixed_full",
            "fixed_phase_adapter",
            "head",
            "full",
        ),
        default=None,
    )
    parser.add_argument("--initial-slow-delay-fraction", type=float, default=None)
    parser.add_argument("--initial-atc-residual-scale", type=float, default=None)
    parser.add_argument("--slow-route-scale", type=float, default=None)
    parser.add_argument("--slow-carrier-residual-scale", type=float, default=None)
    parser.add_argument("--delay-lr-multiplier", type=float, default=None)
    parser.add_argument("--atc-core-lr-multiplier", type=float, default=None)
    parser.add_argument("--matched-zero-kl-weight", type=float, default=None)
    parser.add_argument(
        "--matched-zero-kl-teacher-correct-weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--matched-zero-kl-teacher-incorrect-weight",
        type=float,
        default=None,
    )
    parser.add_argument("--matched-zero-kl-temperature", type=float, default=None)
    parser.add_argument(
        "--delayed-statistics-interaction-gain",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--delayed-statistics-uncertainty-gate",
        choices=("config", "on", "off"),
        default="config",
    )
    parser.add_argument("--delayed-statistics-cp-rank", type=int, default=None)
    parser.add_argument("--delayed-statistics-bins", type=int, default=None)
    parser.add_argument(
        "--delayed-statistics-directional-moments",
        choices=("config", "on", "off"),
        default="config",
    )
    parser.add_argument("--delayed-route-statistics-cp-rank", type=int, default=None)
    parser.add_argument("--delayed-route-statistics-bins", type=int, default=None)
    parser.add_argument(
        "--fixed-epoch",
        type=int,
        default=None,
        help="train outer folds for a fixed epoch count without outer-fold selection",
    )
    parser.add_argument(
        "--fixed-epoch-policy",
        default=None,
        help="pre-declared provenance for the fixed epoch; required with --fixed-epoch",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.fixed_epoch is not None:
        if args.fixed_epoch <= 0:
            parser.error("--fixed-epoch must be positive")
        if not args.fixed_epoch_policy:
            parser.error("--fixed-epoch-policy is required with --fixed-epoch")
    elif args.fixed_epoch_policy is not None:
        parser.error("--fixed-epoch-policy requires --fixed-epoch")
    if args.require_audited_prior and not args.evidence_prior_root:
        parser.error("--require-audited-prior requires --evidence-prior-root")

    configure_cache_env()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    transport_contract = config.get("transport", {})
    readout_contract = str(transport_contract.get("readout", ""))
    registered_broadband_contracts = {
        "identity_d0_plus_degree_normalized_audited_route_contrasts",
        "identity_d0_plus_zero_safe_audited_phase_pair_current",
    }
    if (
        transport_contract.get("broadband_carrier")
        not in registered_broadband_contracts
        or transport_contract.get("matched_zero_intervention")
        != "delay_posterior_only"
    ):
        raise RuntimeError(
            "E3 config does not declare the registered bandwise matched-delay contract"
        )
    parent_config_path = (ROOT / config["parent_config"]).resolve()
    parent_config = yaml.safe_load(parent_config_path.read_text(encoding="utf-8"))
    model_config = dict(parent_config["model"])
    _apply_registered_model_overrides(
        model_config,
        dict(config.get("model_overrides", {})),
    )
    if args.delayed_statistics_uncertainty_gate != "config":
        model_config["atc_delayed_statistics_uncertainty_gate"] = (
            args.delayed_statistics_uncertainty_gate == "on"
        )
    if args.delayed_statistics_cp_rank is not None:
        if args.delayed_statistics_cp_rank < 0:
            parser.error("--delayed-statistics-cp-rank must be non-negative")
        model_config["atc_delayed_statistics_cp_rank"] = int(
            args.delayed_statistics_cp_rank
        )
    if args.delayed_statistics_bins is not None:
        if args.delayed_statistics_bins < 0:
            parser.error("--delayed-statistics-bins must be non-negative")
        model_config["atc_delayed_statistics_bins"] = int(args.delayed_statistics_bins)
    if args.delayed_statistics_directional_moments != "config":
        model_config["atc_delayed_statistics_directional_moments"] = (
            args.delayed_statistics_directional_moments == "on"
        )
    if args.delayed_route_statistics_cp_rank is not None:
        if args.delayed_route_statistics_cp_rank < 0:
            parser.error("--delayed-route-statistics-cp-rank must be non-negative")
        model_config["atc_delayed_route_statistics_cp_rank"] = int(
            args.delayed_route_statistics_cp_rank
        )
    if args.delayed_route_statistics_bins is not None:
        if args.delayed_route_statistics_bins < 0:
            parser.error("--delayed-route-statistics-bins must be non-negative")
        model_config["atc_delayed_route_statistics_bins"] = int(
            args.delayed_route_statistics_bins
        )
    _validate_e3_readout_config(readout_contract, model_config)
    data_root = Path(args.data).resolve()
    output = ensure_dir(args.output)
    parent_output = Path(args.parent_output).resolve()
    source_root = str(Path(args.source_root).resolve())
    evidence_prior_root = (
        Path(args.evidence_prior_root).resolve() if args.evidence_prior_root else None
    )
    subjects = _parse_csv(args.subjects, int) if args.subjects else list(config["subjects"])
    seeds = _parse_csv(args.seeds, int) if args.seeds else list(config["seeds"])
    folds_requested = int(args.max_folds or config["selection"]["n_splits"])
    requested_fold_ids = _parse_csv(args.fold_ids, int) if args.fold_ids else None
    epochs = int(args.cv_epochs or config["selection"]["max_epochs"])
    patience = int(args.patience or config["selection"]["patience"])
    training = dict(config["training"])
    if args.batch_size is not None:
        training["batch_size"] = int(args.batch_size)
    if args.accumulation_steps is not None:
        training["gradient_accumulation_steps"] = int(args.accumulation_steps)
    if args.learning_rate is not None:
        if args.learning_rate <= 0.0:
            parser.error("--learning-rate must be positive")
        training["learning_rate"] = float(args.learning_rate)
    if args.readout_training_scope is not None:
        training["readout_training_scope"] = args.readout_training_scope
    if args.initial_slow_delay_fraction is not None:
        training["initial_slow_delay_fraction"] = float(
            args.initial_slow_delay_fraction
        )
    if args.initial_atc_residual_scale is not None:
        training["initial_atc_residual_scale"] = float(args.initial_atc_residual_scale)
    if args.slow_route_scale is not None:
        training["slow_route_scale"] = float(args.slow_route_scale)
    if args.slow_carrier_residual_scale is not None:
        if not 0.0 <= args.slow_carrier_residual_scale <= 1.0:
            parser.error("--slow-carrier-residual-scale must lie in [0, 1]")
        training["slow_carrier_residual_scale"] = float(
            args.slow_carrier_residual_scale
        )
    if args.delay_lr_multiplier is not None:
        training["delay_learning_rate_multiplier"] = float(
            args.delay_lr_multiplier
        )
    if args.atc_core_lr_multiplier is not None:
        if args.atc_core_lr_multiplier <= 0.0:
            parser.error("--atc-core-lr-multiplier must be positive")
        training["atc_core_learning_rate_multiplier"] = float(
            args.atc_core_lr_multiplier
        )
    if args.matched_zero_kl_weight is not None:
        training["matched_zero_kl_weight"] = float(args.matched_zero_kl_weight)
    if args.matched_zero_kl_teacher_correct_weight is not None:
        training["matched_zero_kl_teacher_correct_weight"] = float(
            args.matched_zero_kl_teacher_correct_weight
        )
    if args.matched_zero_kl_teacher_incorrect_weight is not None:
        training["matched_zero_kl_teacher_incorrect_weight"] = float(
            args.matched_zero_kl_teacher_incorrect_weight
        )
    if args.matched_zero_kl_temperature is not None:
        training["matched_zero_kl_temperature"] = float(
            args.matched_zero_kl_temperature
        )
    if args.delayed_statistics_interaction_gain is not None:
        training["delayed_statistics_interaction_gain"] = float(
            args.delayed_statistics_interaction_gain
        )
    training["effective_batch_size"] = int(training["batch_size"]) * int(
        training["gradient_accumulation_steps"]
    )
    source = _source_hashes()
    source["official_source_locks"] = verify_official_source_locks(source_root)
    environment = _environment()
    pair_rows: list[dict[str, Any]] = []

    for subject in subjects:
        subject_path = _subject_file(data_root, subject)
        data = load_processed_npz(subject_path)
        metadata_all = _metadata_rows(data)
        session = np.asarray(data["session"]).astype(str)
        t_indices = np.flatnonzero(session == "T")
        if len(t_indices) != 288 or int(np.count_nonzero(session == "E")) != 288:
            raise RuntimeError("BCI2a Session-T/E seal requires 288 trials per session")
        x_t = np.asarray(data["X"])[t_indices]
        y_t = np.asarray(data["y"], dtype=np.int64)[t_indices]
        metadata_t = [metadata_all[int(index)] for index in t_indices]
        all_folds = session_t_run_grouped_folds(
            metadata_t,
            n_splits=int(config["selection"]["n_splits"]),
            seed=int(config["selection_seed"]),
            shuffle=True,
        )
        if requested_fold_ids is None:
            selected_fold_ids = list(range(min(folds_requested, len(all_folds))))
        else:
            selected_fold_ids = list(requested_fold_ids)
            if len(selected_fold_ids) != len(set(selected_fold_ids)):
                raise ValueError("--fold-ids must not contain duplicates")
            if any(fold < 0 or fold >= len(all_folds) for fold in selected_fold_ids):
                raise ValueError("--fold-ids contains an out-of-range fold")
        if not selected_fold_ids:
            raise ValueError("at least one fold must be selected")

        for fold in selected_fold_ids:
            train_indices, validation_indices = all_folds[fold]
            gain = fit_official_fbc_channel_gain(
                x_t[train_indices],
                sfreq=float(data["sfreq"]),
                epoch_tmin=float(data["epoch_tmin"]),
                clip=float(parent_config["preprocessing"]["clip_after_gain"]),
            )
            cache_template_checkpoint = _parent_checkpoint(
                parent_output, subject=subject, seed=seeds[0], fold=fold
            )
            cache_template = build_delay_stage_model(
                seed=seeds[0],
                model_config=model_config,
                parent_checkpoint=cache_template_checkpoint,
                stage=config["stage"],
                train_readout=bool(training.get("train_readout", True)),
                readout_training_scope=training.get("readout_training_scope"),
                initial_slow_delay_fraction=training.get(
                    "initial_slow_delay_fraction"
                ),
                initial_atc_residual_scale=training.get(
                    "initial_atc_residual_scale"
                ),
                slow_route_scale=training.get("slow_route_scale"),
                slow_carrier_residual_scale=training.get(
                    "slow_carrier_residual_scale"
                ),
                delayed_statistics_interaction_gain=training.get(
                    "delayed_statistics_interaction_gain"
                ),
                official_source_root=source_root,
            )
            train_rates = cache_analytic_fixed_channel_gain_rate_features(
                cache_template,
                x_t[train_indices],
                gain,
                device=args.device,
                batch_size=max(4, int(training["batch_size"])),
            )
            validation_rates = cache_analytic_fixed_channel_gain_rate_features(
                cache_template,
                x_t[validation_indices],
                gain,
                device=args.device,
                batch_size=max(4, int(training["batch_size"])),
            )
            del cache_template
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            if evidence_prior_root is None:
                prior_arrays = None
                prior_provenance: dict[str, Any] = {
                    "policy": "dense_deterministic_delay_stage"
                }
            else:
                prior_arrays, prior_provenance = _load_audited_fold_prior(
                    evidence_prior_root,
                    subject=subject,
                    fold=fold,
                    scope=args.evidence_prior_scope,
                )
                if (
                    prior_provenance["frontend_fingerprint"]
                    != train_rates.frontend_fingerprint
                ):
                    raise RuntimeError(
                        "audited prior frontend fingerprint does not match cached online features"
                    )
            phase_amplitude_scale = None
            if bool(model_config.get("phase_pair_current_enabled", False)):
                if prior_arrays is None:
                    raise RuntimeError("R9 phase current requires an audited fold prior")
                phase_amplitude_scale = _fit_phase_amplitude_scale(train_rates.fast)
                prior_provenance = {
                    **prior_provenance,
                    "phase_amplitude_scale_policy": (
                        "outer_train_median_of_trial_time_medians_fast_every_2"
                    ),
                    "phase_amplitude_scale_sha256": hashlib.sha256(
                        phase_amplitude_scale.numpy().tobytes()
                    ).hexdigest(),
                    "phase_amplitude_scale_min": float(
                        phase_amplitude_scale.min()
                    ),
                    "phase_amplitude_scale_max": float(
                        phase_amplitude_scale.max()
                    ),
                }

            for seed in seeds:
                parent_checkpoint = _parent_checkpoint(
                    parent_output, subject=subject, seed=seed, fold=fold
                )
                run_dir = ensure_dir(
                    output / f"subject_{subject:02d}" / f"seed_{seed}" / f"fold_{fold}"
                )
                split = {
                    "subject": subject,
                    "seed": seed,
                    "fold": fold,
                    "selection_session": "T",
                    "session_e_accessed": False,
                    "train_trial_ids": [metadata_t[int(i)]["trial_id"] for i in train_indices],
                    "validation_trial_ids": [
                        metadata_t[int(i)]["trial_id"] for i in validation_indices
                    ],
                }
                resolved = {
                    **config,
                    "active_subject": subject,
                    "active_seed": seed,
                    "active_fold": fold,
                    "max_epochs": epochs,
                    "patience": patience,
                    "fixed_epoch_no_outer_selection": args.fixed_epoch,
                    "scheduler_horizon_epochs": epochs,
                    "fixed_epoch_selection_policy": args.fixed_epoch_policy,
                    "training": training,
                    "model_overrides": {
                        "atc_delayed_statistics_uncertainty_gate": bool(
                            model_config.get(
                                "atc_delayed_statistics_uncertainty_gate",
                                False,
                            )
                        ),
                        "atc_delayed_statistics_cp_rank": int(
                            model_config.get("atc_delayed_statistics_cp_rank", 0)
                        ),
                        "atc_delayed_statistics_bins": int(
                            model_config.get("atc_delayed_statistics_bins", 0)
                        ),
                        "atc_delayed_statistics_directional_moments": bool(
                            model_config.get(
                                "atc_delayed_statistics_directional_moments",
                                False,
                            )
                        ),
                        "atc_delayed_route_statistics_cp_rank": int(
                            model_config.get(
                                "atc_delayed_route_statistics_cp_rank",
                                0,
                            )
                        ),
                        "atc_delayed_route_statistics_bins": int(
                            model_config.get("atc_delayed_route_statistics_bins", 0)
                        ),
                        "phase_pair_current_enabled": bool(
                            model_config.get("phase_pair_current_enabled", False)
                        ),
                        "phase_pair_current_bound": float(
                            model_config.get("phase_pair_current_bound", 0.0)
                        ),
                        "phase_pair_initial_scale": float(
                            model_config.get("phase_pair_initial_scale", 0.0)
                        ),
                    },
                    "fold_local_prior": prior_provenance,
                }
                fingerprint = build_run_fingerprint(
                    resolved_config=resolved,
                    source=source,
                    data={subject_path.name: file_sha256(subject_path)},
                    split=split,
                    augmentation=config["augmentation"],
                    prior=prior_provenance,
                    checkpoint={
                        "parent": str(parent_checkpoint),
                        "parent_sha256": file_sha256(parent_checkpoint),
                    },
                    environment=environment,
                )
                result_path = run_dir / "result.json"
                prediction_path = run_dir / "paired_validation_predictions.npz"
                if result_path.is_file() and prediction_path.is_file() and (
                    run_dir / "best.pt"
                ).is_file():
                    validate_resume_fingerprint(run_dir / "source_fingerprint.json", fingerprint)
                    print(
                        f"__V7_E3_RESUME_SKIP__ subject={subject} seed={seed} fold={fold}",
                        flush=True,
                    )
                    continue
                write_fingerprint_manifest(run_dir / "source_fingerprint.json", fingerprint)
                write_json(run_dir / "split_manifest.json", split)
                (run_dir / "resolved_config.yaml").write_text(
                    yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
                )
                write_json(
                    run_dir / "runtime_status.json",
                    {"status": "running", "started_at": time.time()},
                )
                model = build_delay_stage_model(
                    seed=seed,
                    model_config=model_config,
                    parent_checkpoint=parent_checkpoint,
                    stage=config["stage"],
                    train_readout=bool(training.get("train_readout", True)),
                    readout_training_scope=training.get("readout_training_scope"),
                    initial_slow_delay_fraction=training.get(
                        "initial_slow_delay_fraction"
                    ),
                    initial_atc_residual_scale=training.get(
                        "initial_atc_residual_scale"
                    ),
                    slow_route_scale=training.get("slow_route_scale"),
                    slow_carrier_residual_scale=training.get(
                        "slow_carrier_residual_scale"
                    ),
                    delayed_statistics_interaction_gain=training.get(
                        "delayed_statistics_interaction_gain"
                    ),
                    official_source_root=source_root,
                )
                _validate_e3_readout_model(model, readout_contract)
                if prior_arrays is not None:
                    model.load_fold_local_slow_prior(
                        torch.from_numpy(prior_arrays["route_probability"]),
                        torch.from_numpy(prior_arrays["positive_delay_probability"]),
                        torch.from_numpy(prior_arrays["fractional_delay_target"]),
                    )
                if phase_amplitude_scale is not None:
                    model.load_fold_local_phase_amplitude_scale(
                        phase_amplitude_scale
                    )
                delay_state_before_fit = state_sha256(model.delay)
                if args.device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                fit = fit_zero_scaffold(
                    model,
                    train_rates=train_rates,
                    y_train=y_t[train_indices],
                    validation_rates=(None if args.fixed_epoch is not None else validation_rates),
                    y_validation=(None if args.fixed_epoch is not None else y_t[validation_indices]),
                    device=args.device,
                    seed=seed + fold * 1009,
                    epochs=epochs,
                    patience=patience,
                    minimum_epochs=int(config["selection"]["minimum_epochs"]),
                    batch_size=int(training["batch_size"]),
                    accumulation_steps=int(training["gradient_accumulation_steps"]),
                    learning_rate=float(training["learning_rate"]),
                    delay_learning_rate_multiplier=float(
                        training.get("delay_learning_rate_multiplier", 1.0)
                    ),
                    atc_core_learning_rate_multiplier=float(
                        training.get("atc_core_learning_rate_multiplier", 1.0)
                    ),
                    matched_zero_kl_weight=float(
                        training.get("matched_zero_kl_weight", 0.0)
                    ),
                    matched_zero_kl_teacher_correct_weight=training.get(
                        "matched_zero_kl_teacher_correct_weight"
                    ),
                    matched_zero_kl_teacher_incorrect_weight=training.get(
                        "matched_zero_kl_teacher_incorrect_weight"
                    ),
                    matched_zero_kl_temperature=float(
                        training.get("matched_zero_kl_temperature", 1.0)
                    ),
                    weight_decay=float(training["weight_decay"]),
                    auxiliary_weights=training["auxiliary_endpoint_weights"],
                    beta1=float(training["beta1"]),
                    scheduler_warmup_epochs=int(training["scheduler_warmup_epochs"]),
                    use_scheduler=bool(training["schedule"]),
                    validation_interval=int(training.get("validation_interval", 1)),
                    augmentation=config["augmentation"],
                    fixed_epoch=args.fixed_epoch,
                    run_label=f"{config['stage']}:S{subject}:seed{seed}:fold{fold}",
                )
                delay_state_after_fit = state_sha256(fit.model.delay)
                training_scope = str(
                    training.get("readout_training_scope", "")
                ).strip().lower()
                delay_checkpoint_unchanged = (
                    delay_state_before_fit == delay_state_after_fit
                )
                if (
                    training_scope
                    in {
                        "fixed_delay_head",
                        "fixed_head",
                        "fixed_full",
                        "fixed_phase_adapter",
                    }
                    and not delay_checkpoint_unchanged
                ):
                    raise RuntimeError(
                        "fixed-delay training mutated the audited delay checkpoint"
                    )
                state_before = state_sha256(fit.model)
                full = predict_scaffold(
                    fit.model,
                    validation_rates,
                    y_t[validation_indices],
                    device=args.device,
                    batch_size=max(4, int(training["batch_size"])),
                    # Let the registered stage decide which delay branches are
                    # active. A common "learned" override would incorrectly
                    # enable fast delay during the static-slow experiment.
                    delay_override=None,
                    retain_diagnostic_tensors=True,
                )
                state_after_full = state_sha256(fit.model)
                if state_before != state_after_full:
                    raise RuntimeError("full evaluation mutated the trained checkpoint")
                locked_zero = predict_scaffold(
                    fit.model,
                    validation_rates,
                    y_t[validation_indices],
                    device=args.device,
                    batch_size=max(4, int(training["batch_size"])),
                    delay_override="zero",
                    retain_diagnostic_tensors=True,
                )
                state_after_zero = state_sha256(fit.model)
                if state_before != state_after_zero:
                    raise RuntimeError("locked-zero evaluation mutated the trained checkpoint")
                if full["broadband_delay_source"] != locked_zero[
                    "broadband_delay_source"
                ]:
                    raise RuntimeError(
                        "full and locked-zero controls used different broadband routes"
                    )
                if (
                    fit.model.slow_delay_enabled
                    and not fit.model.fast_delay_enabled
                    and full["broadband_delay_source"] != "slow_route_residual"
                ):
                    raise RuntimeError(
                        "static-slow stage did not preserve bandwise slow-delay transport"
                    )
                interaction_gain = float(
                    getattr(
                        fit.model.atc_readout,
                        "delayed_statistics_interaction_gain",
                        0.0,
                    )
                )
                if (
                    interaction_gain > 0.0
                    and locked_zero["delayed_statistics_logits_rms"] > 1e-8
                ):
                    raise RuntimeError(
                        "locked-zero delay-contrast head produced non-zero logits"
                    )
                if locked_zero["delayed_route_statistics_logits_rms"] > 1e-8:
                    raise RuntimeError(
                        "locked-zero route delay-contrast head produced non-zero logits"
                    )
                if readout_contract == "pre_decoder_audited_phase_pair_current_only":
                    if locked_zero["phase_pair_current_rms"] != 0.0:
                        raise RuntimeError(
                            "locked-zero phase-pair current is not exactly zero"
                        )
                    if full["phase_pair_current_rms"] <= 0.0:
                        raise RuntimeError("full R9 phase-pair current is identically zero")
                amplitude_difference = abs(
                    full["synthesized_carrier_rms"] - locked_zero["synthesized_carrier_rms"]
                ) / max(locked_zero["synthesized_carrier_rms"], np.finfo(float).tiny)
                full_synthesized = full["diagnostic_synthesized_carrier"]
                zero_synthesized = locked_zero["diagnostic_synthesized_carrier"]
                full_broadband = full["diagnostic_broadband_current"]
                zero_broadband = locked_zero["diagnostic_broadband_current"]
                if (
                    full_synthesized is None
                    or zero_synthesized is None
                    or full_broadband is None
                    or zero_broadband is None
                ):
                    raise RuntimeError("paired carrier diagnostics were not retained")
                synthesized_difference_rms = float(
                    np.sqrt(np.mean(np.square(full_synthesized - zero_synthesized)))
                )
                broadband_difference_rms = float(
                    np.sqrt(np.mean(np.square(full_broadband - zero_broadband)))
                )
                synthesized_relative_difference = synthesized_difference_rms / max(
                    locked_zero["synthesized_carrier_rms"], np.finfo(float).tiny
                )
                zero_broadband_rms = float(np.sqrt(np.mean(np.square(zero_broadband))))
                broadband_relative_difference = broadband_difference_rms / max(
                    zero_broadband_rms, np.finfo(float).tiny
                )
                result = {
                    "status": "completed",
                    "stage": config["stage"],
                    "subject": subject,
                    "seed": seed,
                    "fold": fold,
                    "best_epoch": fit.best_epoch,
                    "scheduler_horizon_epochs": epochs,
                    "optimizer_steps": fit.optimizer_steps,
                    "train_seconds": fit.elapsed_seconds,
                    "peak_cuda_memory_mib": (
                        float(torch.cuda.max_memory_allocated()) / (1024.0**2)
                        if args.device.startswith("cuda")
                        else 0.0
                    ),
                    "full_accuracy": full["accuracy"],
                    "locked_zero_accuracy": locked_zero["accuracy"],
                    "delta_accuracy": full["accuracy"] - locked_zero["accuracy"],
                    "full_kappa": full["kappa"],
                    "locked_zero_kappa": locked_zero["kappa"],
                    "full_synthesized_carrier_rms": full["synthesized_carrier_rms"],
                    "locked_zero_synthesized_carrier_rms": locked_zero[
                        "synthesized_carrier_rms"
                    ],
                    "full_delay_contrast_rms": full["delay_contrast_rms"],
                    "locked_zero_delay_contrast_rms": locked_zero[
                        "delay_contrast_rms"
                    ],
                    "full_route_delay_contrast_rms": full[
                        "route_delay_contrast_rms"
                    ],
                    "locked_zero_route_delay_contrast_rms": locked_zero[
                        "route_delay_contrast_rms"
                    ],
                    "full_phase_pair_current_rms": full[
                        "phase_pair_current_rms"
                    ],
                    "locked_zero_phase_pair_current_rms": locked_zero[
                        "phase_pair_current_rms"
                    ],
                    "full_delayed_statistics_logits_rms": full[
                        "delayed_statistics_logits_rms"
                    ],
                    "locked_zero_delayed_statistics_logits_rms": locked_zero[
                        "delayed_statistics_logits_rms"
                    ],
                    "full_delayed_route_statistics_logits_rms": full[
                        "delayed_route_statistics_logits_rms"
                    ],
                    "locked_zero_delayed_route_statistics_logits_rms": locked_zero[
                        "delayed_route_statistics_logits_rms"
                    ],
                    "full_delayed_statistics_gate_mean": full[
                        "delayed_statistics_gate_mean"
                    ],
                    "full_delayed_statistics_gate_min": full[
                        "delayed_statistics_gate_min"
                    ],
                    "full_delayed_statistics_gate_max": full[
                        "delayed_statistics_gate_max"
                    ],
                    "locked_zero_delayed_statistics_gate_mean": locked_zero[
                        "delayed_statistics_gate_mean"
                    ],
                    "amplitude_relative_difference": amplitude_difference,
                    "paired_synthesized_difference_rms": synthesized_difference_rms,
                    "paired_synthesized_relative_difference": (
                        synthesized_relative_difference
                    ),
                    "paired_broadband_difference_rms": broadband_difference_rms,
                    "paired_broadband_relative_difference": broadband_relative_difference,
                    "slow_carrier_residual_scale": float(
                        fit.model.delay.slow_carrier_residual_scale
                    ),
                    "broadband_delay_source": full["broadband_delay_source"],
                    "locked_zero_broadband_delay_source": locked_zero[
                        "broadband_delay_source"
                    ],
                    "checkpoint_state_unchanged_by_control": True,
                    "checkpoint_state_unchanged_by_full_evaluation": True,
                    "checkpoint_state_sha256": state_before,
                    "delay_checkpoint_unchanged_by_training": (
                        delay_checkpoint_unchanged
                    ),
                    "delay_checkpoint_sha256": delay_state_before_fit,
                    "session_e_accessed": False,
                    "run_fingerprint": fingerprint["combined_sha256"],
                    "parent_state_migration": getattr(
                        fit.model,
                        "_parent_state_migration",
                        {},
                    ),
                    "fold_local_prior": prior_provenance,
                    **delay_diagnostics(fit.model),
                }
                torch.save(fit.best_state, run_dir / "best.pt")
                torch.save(fit.last_state, run_dir / "last.pt")
                write_csv(run_dir / "history.csv", fit.history)
                write_json(result_path, result)
                np.savez_compressed(
                    prediction_path,
                    indices=np.asarray(validation_indices, dtype=np.int64),
                    labels=full["labels"].astype(np.int64),
                    full_logits=full["logits"].astype(np.float32),
                    locked_zero_logits=locked_zero["logits"].astype(np.float32),
                )
                write_json(
                    run_dir / "runtime_status.json",
                    {"status": "completed", "completed_at": time.time()},
                )
                print(
                    "__V7_E3_FOLD_DONE__ "
                    f"stage={config['stage']} subject={subject} seed={seed} fold={fold} "
                    f"full={full['accuracy']:.6f} zero={locked_zero['accuracy']:.6f}",
                    flush=True,
                )
                del model, fit
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            del train_rates, validation_rates

        for seed in seeds:
            seed_dir = ensure_dir(output / f"subject_{subject:02d}" / f"seed_{seed}")
            fold_results = [
                read_json(seed_dir / f"fold_{fold}" / "result.json")
                for fold in selected_fold_ids
            ]
            prediction_parts = []
            for fold in selected_fold_ids:
                with np.load(
                    seed_dir / f"fold_{fold}" / "paired_validation_predictions.npz",
                    allow_pickle=False,
                ) as archive:
                    prediction_parts.append({key: archive[key] for key in archive.files})
            indices = np.concatenate([part["indices"] for part in prediction_parts])
            order = np.argsort(indices)
            indices = indices[order]
            if len(indices) != len(np.unique(indices)):
                raise RuntimeError("OOF predictions contain duplicated trials")
            complete_oof = len(indices) == len(y_t)
            full_logits = np.concatenate([part["full_logits"] for part in prediction_parts])[order]
            zero_logits = np.concatenate(
                [part["locked_zero_logits"] for part in prediction_parts]
            )[order]
            labels = y_t[indices]
            full_pred = full_logits.argmax(axis=1)
            zero_pred = zero_logits.argmax(axis=1)
            full_accuracy = float(np.mean(full_pred == labels))
            zero_accuracy = float(np.mean(zero_pred == labels))
            amplitude_relative_difference = float(
                max(row["amplitude_relative_difference"] for row in fold_results)
            )
            pair = {
                "stage": config["stage"],
                "subject": subject,
                "seed": seed,
                "full_accuracy": full_accuracy,
                "locked_zero_accuracy": zero_accuracy,
                "delta_accuracy": full_accuracy - zero_accuracy,
                "amplitude_relative_difference": amplitude_relative_difference,
                "folds": len(selected_fold_ids),
                "fold_ids": selected_fold_ids,
                "complete_oof": complete_oof,
                "session_e_accessed": False,
                **paired_prediction_comparison(labels, full_pred, zero_pred),
            }
            pair_rows.append(pair)
            write_json(seed_dir / "paired_oof_metrics.json", pair)
            write_trial_predictions(
                seed_dir / "full_oof",
                logits=full_logits,
                probabilities=_probabilities(full_logits),
                pred=full_pred,
                label=labels,
                subject=[metadata_t[int(index)]["subject"] for index in indices],
                session="T",
                run=[metadata_t[int(index)]["run"] for index in indices],
                trial_id=[metadata_t[int(index)]["trial_id"] for index in indices],
                seed=seed,
                model=f"dasp_v7_{config['stage']}_full",
            )
            write_trial_predictions(
                seed_dir / "locked_zero_oof",
                logits=zero_logits,
                probabilities=_probabilities(zero_logits),
                pred=zero_pred,
                label=labels,
                subject=[metadata_t[int(index)]["subject"] for index in indices],
                session="T",
                run=[metadata_t[int(index)]["run"] for index in indices],
                trial_id=[metadata_t[int(index)]["trial_id"] for index in indices],
                seed=seed,
                model=f"dasp_v7_{config['stage']}_locked_zero",
            )

    write_csv(output / "paired_summary.csv", pair_rows)
    gate = paired_delay_gate(pair_rows, config)
    write_json(output / "gate_report.json", gate)
    write_json(
        output / "campaign_status.json",
        {
            "status": "completed",
            "stage": config["stage"],
            "subjects": subjects,
            "seeds": seeds,
            "session_e_accessed": False,
            "gate": gate,
        },
    )
    print(json.dumps({"status": "completed", "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
