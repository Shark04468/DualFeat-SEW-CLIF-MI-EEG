#!/usr/bin/env python3
"""Train and evaluate the frozen V25 dual-feature model across BCI2a sessions."""

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

from dpc_snn.baselines.neural import verify_official_source_locks  # noqa: E402
from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v17_information_replay import residual_fusion_logits  # noqa: E402
from dpc_snn.experiments.v25_confirmation import validate_v25_freeze  # noqa: E402
from dpc_snn.experiments.v62_baselines import (  # noqa: E402
    BASELINE_OPTIMIZERS,
    FixedGain,
    apply_fixed_gain,
    fit_baseline,
    fit_fixed_gain,
    prepare_model_input,
    task_carrier,
)
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    sha256_fingerprint,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    build_v8_run_fingerprint,
    collect_source_tree_manifest,
    mapping_sha256,
    source_tree_digest,
    validate_v8_resume_fingerprint,
    write_v8_fingerprint,
)
from dpc_snn.experiments.v9_dual_feature_training import (  # noqa: E402
    FrozenFeatureStandardizer,
    V9_DUAL_FEATURE_EXPERIMENT_VARIANTS,
    equal_probability_teacher,
    fit_feature_standardizer,
    fit_v9_dual_feature,
    predict_v9_dual_feature,
)
from dpc_snn.models.v9_dual_feature_student import build_v9_dual_feature_student  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v8_e1_baselines import _subject_file  # noqa: E402
from scripts.run_v8_e6_ensemble_frozen import (  # noqa: E402
    _metadata,
    _scalar,
    _write_arm_predictions,
)
from scripts.run_v9_e3_dual_feature_fold import (  # noqa: E402
    _capacity_audit,
    _environment_manifest,
    _extract_atc,
    _extract_fbc,
)


CHECKPOINT_FILES = {
    "atcnet": "atcnet.pt",
    "fbcnet": "fbcnet.pt",
    "ann_plain_ce": "ann_plain_ce.pt",
    "sew_clif_ce": "sew_clif_ce.pt",
}
TRAIN_FILES = (
    "manifest.json",
    "train_manifest.json",
    "run_fingerprint.json",
    "resolved_run.yaml",
    "source_tree_manifest.json",
    "official_source_locks.json",
    "gain.npz",
    "feature_standardizer.json",
    "checkpoint_hashes.json",
    "atcnet.pt",
    "fbcnet.pt",
    "ann_plain_ce.pt",
    "sew_clif_ce.pt",
    "atcnet_history.csv",
    "fbcnet_history.csv",
    "ann_plain_ce_history.csv",
    "sew_clif_ce_history.csv",
)


def _probabilities(logits: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.from_numpy(np.asarray(logits, dtype=np.float32)), dim=1).numpy()


def _metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    return classification_metrics(labels, np.asarray(logits).argmax(axis=1), n_classes=4)


def _state_digest(model: torch.nn.Module) -> str:
    return sha256_fingerprint(mapping_sha256(model.state_dict()))


def _checkpoint_hashes(run_dir: Path) -> dict[str, str]:
    return {name: file_sha256(run_dir / filename) for name, filename in CHECKPOINT_FILES.items()}


def _standardizer_from_json(path: Path) -> FrozenFeatureStandardizer:
    value = read_json(path)
    return FrozenFeatureStandardizer(
        np.asarray(value["atc_mean"], dtype=np.float32).reshape(1, 1, 32),
        np.asarray(value["atc_scale"], dtype=np.float32).reshape(1, 1, 32),
        np.asarray(value["fbc_mean"], dtype=np.float32).reshape(1, 1, 288),
        np.asarray(value["fbc_scale"], dtype=np.float32).reshape(1, 1, 288),
    )


def _resolved(
    args: argparse.Namespace, freeze: dict[str, Any], entry: dict[str, Any]
) -> dict[str, Any]:
    epochs = {name: int(value) for name, value in entry["fixed_epochs"].items()}
    if args.epoch_cap is not None:
        if not args.canary:
            raise ValueError("epoch overrides are restricted to T-only canaries")
        epochs = {name: min(int(args.epoch_cap), value) for name, value in epochs.items()}
    return {
        "stage": "V25_TRAIN" if args.stage == "train" else "V25_EVALUATE",
        "protocol": freeze["protocol"],
        "architecture_id": freeze["architecture_id"],
        "freeze_sha256": freeze["combined_sha256"],
        "subject": int(args.subject),
        "seed": int(args.seed),
        "fixed_epochs": epochs,
        "residual_scales": entry["residual_scales"],
        "teacher": freeze["teacher"],
        "historical_data_exposure": freeze["historical_data_exposure"],
        "canary": bool(args.canary),
    }


def _fingerprint(
    *,
    args: argparse.Namespace,
    resolved: dict[str, Any],
    source_tree: dict[str, str],
    train_path: Path,
    evaluation_path: Path,
    augmentation: dict[str, Any],
    official_locks: dict[str, Any],
) -> dict[str, Any]:
    return build_v8_run_fingerprint(
        resolved_run_config={**resolved, "stage": "V25_LOCKED_UNIT"},
        source_tree=source_tree,
        data={
            f"session_t/{train_path.name}": file_sha256(train_path),
            f"session_e/{evaluation_path.name}": file_sha256(evaluation_path),
        },
        split={
            "train_session": "T",
            "evaluation_session": "E",
            "global_checkpoint_barrier_required": not args.canary,
        },
        augmentation=augmentation,
        prior={"policy": "none", "delay_enabled": False},
        checkpoint={
            "fixed_epochs": resolved["fixed_epochs"],
            "residual_scales": resolved["residual_scales"],
            "session_e_checkpoint_selection": False,
        },
        environment={**_environment_manifest(args.device), "official_source_locks": official_locks},
    )


def _train(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    train_path: Path,
    evaluation_path: Path,
    source_root: Path,
    resolved: dict[str, Any],
    fingerprint: dict[str, Any],
    source_tree: dict[str, str],
    official_locks: dict[str, Any],
    augmentation: dict[str, Any],
) -> None:
    if (run_dir / "train_manifest.json").is_file():
        validate_v8_resume_fingerprint(run_dir / "run_fingerprint.json", fingerprint)
        return
    ensure_dir(run_dir)
    write_v8_fingerprint(run_dir / "run_fingerprint.json", fingerprint)
    write_json(run_dir / "source_tree_manifest.json", source_tree)
    write_json(run_dir / "official_source_locks.json", official_locks)
    (run_dir / "resolved_run.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    data = load_processed_npz(train_path)
    metadata_t = _metadata(data, expected_session="T", role="training")
    labels = np.asarray(data["y"], dtype=np.int64)
    carrier = task_carrier(
        np.asarray(data["X"], dtype=np.float32),
        sfreq=float(_scalar(data["sfreq"])),
        epoch_tmin=float(_scalar(data["epoch_tmin"])),
    )
    gain = fit_fixed_gain(carrier, clip=12.0)
    normalized = apply_fixed_gain(carrier, gain)
    prepared = {
        branch: prepare_model_input(branch, normalized, sfreq=float(_scalar(data["sfreq"])))
        for branch in ("atcnet", "fbcnet")
    }
    run_seed = int(args.seed) * 100_003 + int(args.subject) * 1_009
    fits: dict[str, Any] = {}
    for branch in ("atcnet", "fbcnet"):
        fits[branch] = fit_baseline(
            branch,
            source_root=source_root,
            x_train=prepared[branch],
            y_train=labels,
            x_validation=None,
            y_validation=None,
            device=args.device,
            seed=run_seed,
            epochs=int(resolved["fixed_epochs"][branch]),
            patience=int(resolved["fixed_epochs"][branch]),
            augmentation=augmentation,
            fixed_epoch=int(resolved["fixed_epochs"][branch]),
            scheduler_epochs=300,
            run_label=f"V25:{branch}:S{args.subject}:seed{args.seed}",
        )
        torch.save(fits[branch].last_state, run_dir / CHECKPOINT_FILES[branch])
        write_csv(run_dir / f"{branch}_history.csv", fits[branch].history)

    atc_sequence, atc_logits = _extract_atc(
        source_root=source_root,
        checkpoint=run_dir / CHECKPOINT_FILES["atcnet"],
        carrier=prepared["atcnet"],
        device=args.device,
        batch_size=64,
    )
    fbc_sequence, fbc_logits = _extract_fbc(
        source_root=source_root,
        checkpoint=run_dir / CHECKPOINT_FILES["fbcnet"],
        carrier=prepared["fbcnet"],
        device=args.device,
        batch_size=64,
    )
    standardizer = fit_feature_standardizer(atc_sequence, fbc_sequence)
    atc_standardized, fbc_standardized = standardizer.transform(atc_sequence, fbc_sequence)
    teacher = equal_probability_teacher(atc_logits, fbc_logits)
    student_seed = run_seed + 9_300_007
    for variant in ("ann_plain_ce", "sew_clif_ce"):
        fit = fit_v9_dual_feature(
            variant,
            atc_train=atc_standardized,
            fbc_train=fbc_standardized,
            y_train=labels,
            teacher_train=teacher,
            atc_validation=None,
            fbc_validation=None,
            y_validation=None,
            teacher_validation=None,
            device=args.device,
            seed=student_seed,
            epochs=160,
            fixed_epoch=int(resolved["fixed_epochs"][variant]),
            scheduler_epochs=160,
            batch_size=48,
            run_label=f"V25:{variant}:S{args.subject}:seed{args.seed}",
        )
        torch.save(fit.last_state, run_dir / CHECKPOINT_FILES[variant])
        write_csv(run_dir / f"{variant}_history.csv", fit.history)
        fits[variant] = fit

    np.savez_compressed(
        run_dir / "gain.npz",
        values=np.asarray(gain.values, dtype=np.float32),
        clip=np.asarray(gain.clip, dtype=np.float32),
    )
    write_json(run_dir / "feature_standardizer.json", standardizer.as_dict())
    hashes = _checkpoint_hashes(run_dir)
    write_json(run_dir / "checkpoint_hashes.json", hashes)
    smoke = {}
    if args.canary:
        for variant in ("ann_plain_ce", "sew_clif_ce"):
            replay = predict_v9_dual_feature(
                fits[variant].model,
                atc_standardized[:32],
                fbc_standardized[:32],
                labels[:32],
                teacher[:32],
                device=args.device,
                batch_size=32,
            )
            smoke[variant] = {
                "finite": bool(np.isfinite(replay["logits"]).all()),
                "shape": list(replay["logits"].shape),
            }
    write_json(
        run_dir / "train_manifest.json",
        {
            "status": "checkpointed",
            "subject": int(args.subject),
            "seed": int(args.seed),
            "train_trials": len(metadata_t),
            "checkpoint_hashes": hashes,
            "checkpoint_bundle_sha256": sha256_fingerprint(hashes),
            "session_e_semantically_loaded": False,
            "session_e_byte_hash": file_sha256(evaluation_path),
            "current_v25_session_e_used_for_selection": False,
            "historical_bci2a_session_e_accessed": True,
            "smoke_replay": smoke,
            "completed_at": time.time(),
        },
    )


def _evaluate(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    train_path: Path,
    evaluation_path: Path,
    source_root: Path,
    resolved: dict[str, Any],
    fingerprint: dict[str, Any],
    barrier: Path,
) -> None:
    if args.canary:
        raise RuntimeError("canaries may not semantically load Session E")
    if not barrier.is_file():
        raise FileNotFoundError("global 45-checkpoint barrier is missing")
    validate_v8_resume_fingerprint(run_dir / "run_fingerprint.json", fingerprint)
    train_manifest = read_json(run_dir / "train_manifest.json")
    hashes_before = _checkpoint_hashes(run_dir)
    if hashes_before != train_manifest["checkpoint_hashes"]:
        raise RuntimeError("checkpoint bundle differs from the T-only train manifest")
    data_t = load_processed_npz(train_path)
    data_e = load_processed_npz(evaluation_path)
    metadata_e = _metadata(data_e, expected_session="E", role="evaluation")
    for field in ("sfreq", "epoch_tmin", "epoch_tmax"):
        if float(_scalar(data_t[field])) != float(_scalar(data_e[field])):
            raise RuntimeError(f"Session T/E acquisition mismatch: {field}")
    if not np.array_equal(data_t["ch_names"], data_e["ch_names"]):
        raise RuntimeError("Session T/E channel order differs")
    labels = np.asarray(data_e["y"], dtype=np.int64)
    carrier = task_carrier(
        np.asarray(data_e["X"], dtype=np.float32),
        sfreq=float(_scalar(data_e["sfreq"])),
        epoch_tmin=float(_scalar(data_e["epoch_tmin"])),
    )
    with np.load(run_dir / "gain.npz", allow_pickle=False) as archive:
        gain = FixedGain(
            values=np.asarray(archive["values"], dtype=np.float32),
            clip=float(np.asarray(archive["clip"]).item()),
        )
    normalized = apply_fixed_gain(carrier, gain)
    atc_input = prepare_model_input("atcnet", normalized, sfreq=float(_scalar(data_e["sfreq"])))
    fbc_input = prepare_model_input("fbcnet", normalized, sfreq=float(_scalar(data_e["sfreq"])))
    atc_sequence, atc_logits = _extract_atc(
        source_root=source_root,
        checkpoint=run_dir / CHECKPOINT_FILES["atcnet"],
        carrier=atc_input,
        device=args.device,
        batch_size=64,
    )
    fbc_sequence, fbc_logits = _extract_fbc(
        source_root=source_root,
        checkpoint=run_dir / CHECKPOINT_FILES["fbcnet"],
        carrier=fbc_input,
        device=args.device,
        batch_size=64,
    )
    atc_standardized, fbc_standardized = _standardizer_from_json(
        run_dir / "feature_standardizer.json"
    ).transform(atc_sequence, fbc_sequence)
    teacher = equal_probability_teacher(atc_logits, fbc_logits)
    arms: dict[str, np.ndarray] = {
        "atcnet": atc_logits,
        "fbcnet": fbc_logits,
        "teacher": teacher,
    }
    firing_rates: dict[str, float] = {}
    student_states: dict[str, tuple[str, str]] = {}
    for variant in ("ann_plain_ce", "sew_clif_ce"):
        specification = V9_DUAL_FEATURE_EXPERIMENT_VARIANTS[variant]
        model = build_v9_dual_feature_student(specification.model_variant)
        model.load_state_dict(
            torch.load(run_dir / CHECKPOINT_FILES[variant], map_location="cpu", weights_only=True),
            strict=True,
        )
        before = _state_digest(model)
        prediction = predict_v9_dual_feature(
            model,
            atc_standardized,
            fbc_standardized,
            labels,
            teacher,
            device=args.device,
            batch_size=64,
        )
        after = _state_digest(model)
        student_states[variant] = (before, after)
        if before != after:
            raise RuntimeError(f"{variant} state changed during Session-E evaluation")
        arms[variant] = residual_fusion_logits(
            teacher, prediction["logits"], float(resolved["residual_scales"][variant])
        )
        firing_rates[variant] = float(prediction["mean_firing_rate"])
    hashes_after = _checkpoint_hashes(run_dir)
    if hashes_before != hashes_after:
        raise RuntimeError("checkpoint files changed during Session-E evaluation")
    arm_metrics = {name: _metrics(labels, logits) for name, logits in arms.items()}
    for name, logits in arms.items():
        _write_arm_predictions(
            run_dir,
            basename="predictions" if name == "sew_clif_ce" else f"{name}_predictions",
            logits=logits,
            probability=_probabilities(logits),
            labels=labels,
            metadata=metadata_e,
            seed=int(args.seed),
            model=f"v25_{name}",
        )
    metrics = {
        "status": "completed",
        "stage": "V25_LOCKED_CROSS_SESSION_EVALUATION",
        "subject": int(args.subject),
        "seed": int(args.seed),
        "accuracy": arm_metrics["sew_clif_ce"]["accuracy"],
        "kappa": arm_metrics["sew_clif_ce"]["kappa"],
        "macro_f1": arm_metrics["sew_clif_ce"]["macro_f1"],
        "arms": arm_metrics,
        "residual_scales": resolved["residual_scales"],
        "firing_rates": firing_rates,
        "checkpoint_hashes": hashes_after,
        "checkpoint_bundle_sha256": sha256_fingerprint(hashes_after),
        "student_state_unchanged": {
            name: before == after for name, (before, after) in student_states.items()
        },
        "session_e_checkpoint_selection": False,
        "session_e_gradient_updates": False,
        "historical_bci2a_session_e_accessed": True,
        "current_v25_session_e_used_for_selection": False,
        "run_fingerprint": fingerprint["combined_sha256"],
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "data_access_manifest.json",
        {
            "train_session": "T",
            "evaluation_session": "E",
            "global_checkpoint_barrier": file_sha256(barrier),
            "session_e_arrays_first_loaded_after_global_barrier": True,
            "session_e_checkpoint_selection": False,
            "session_e_gradient_updates": False,
            "historical_bci2a_session_e_accessed": True,
        },
    )
    write_json(run_dir / "evaluation_manifest.json", {"status": "completed", **metrics})
    required = TRAIN_FILES + (
        "metrics.json",
        "data_access_manifest.json",
        "evaluation_manifest.json",
        "predictions.npz",
        "predictions.csv",
        "ann_plain_ce_predictions.npz",
        "ann_plain_ce_predictions.csv",
        "teacher_predictions.npz",
        "teacher_predictions.csv",
        "atcnet_predictions.npz",
        "atcnet_predictions.csv",
        "fbcnet_predictions.npz",
        "fbcnet_predictions.csv",
    )
    write_run_artifact_manifest(run_dir, required_files=required)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--evaluation-data", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", choices=("train", "evaluate"), required=True)
    parser.add_argument("--barrier", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--epoch-cap", type=int, default=None)
    args = parser.parse_args()

    configure_cache_env()
    if args.subject not in range(1, 10) or args.seed not in range(5):
        raise ValueError("V25 requires subject 1-9 and seed 0-4")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    freeze = validate_v25_freeze(read_json(Path(args.freeze).resolve()))
    entry = freeze["entries"][f"subject_{args.subject:02d}_seed_{args.seed}"]
    resolved = _resolved(args, freeze, entry)
    source_tree = collect_source_tree_manifest(ROOT)
    if freeze["source_tree_sha256"] != source_tree_digest(source_tree):
        raise RuntimeError("source tree differs from the V25 freeze snapshot")
    source_root = Path(args.source_root).resolve()
    official_locks = verify_official_source_locks(source_root)
    train_path = _subject_file(Path(args.train_data).resolve(), args.subject)
    evaluation_path = _subject_file(Path(args.evaluation_data).resolve(), args.subject)
    augmentation = yaml.safe_load(
        (ROOT / "configs/experiments/v8_e1_baselines.yaml").read_text(encoding="utf-8")
    )["augmentation"]
    fingerprint = _fingerprint(
        args=args,
        resolved=resolved,
        source_tree=source_tree,
        train_path=train_path,
        evaluation_path=evaluation_path,
        augmentation=augmentation,
        official_locks=official_locks,
    )
    run_dir = Path(args.output).resolve() / f"subject_{args.subject:02d}" / f"seed_{args.seed}"
    if args.stage == "train":
        _train(
            args=args,
            run_dir=run_dir,
            train_path=train_path,
            evaluation_path=evaluation_path,
            source_root=source_root,
            resolved=resolved,
            fingerprint=fingerprint,
            source_tree=source_tree,
            official_locks=official_locks,
            augmentation=augmentation,
        )
        return
    _evaluate(
        args=args,
        run_dir=run_dir,
        train_path=train_path,
        evaluation_path=evaluation_path,
        source_root=source_root,
        resolved=resolved,
        fingerprint=fingerprint,
        barrier=Path(args.barrier).resolve(),
    )


if __name__ == "__main__":
    main()
