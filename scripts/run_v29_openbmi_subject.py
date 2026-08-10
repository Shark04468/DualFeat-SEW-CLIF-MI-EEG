#!/usr/bin/env python3
"""Train or evaluate one OpenBMI subject under the frozen V29 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.baselines.neural import build_v62_neural_baseline  # noqa: E402
from dpc_snn.experiments.v62_baselines import (  # noqa: E402
    apply_fixed_gain,
    prepare_model_input,
    task_carrier,
)
from dpc_snn.experiments.v62_protocol import file_sha256, sha256_fingerprint  # noqa: E402
from dpc_snn.experiments.v62_scaffold import (  # noqa: E402
    project_registered_max_norm_constraints_,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    source_tree_digest,
)
from dpc_snn.experiments.v9_dual_feature_training import (  # noqa: E402
    equal_probability_teacher,
    fit_feature_standardizer,
    fit_v9_dual_feature,
    predict_v9_dual_feature,
)
from dpc_snn.models.v8_atc_backbone import V8ATCAccuracyBackbone  # noqa: E402
from dpc_snn.models.v9_dual_feature_student import (  # noqa: E402
    V9FBCFeatureBackbone,
    build_v9_dual_feature_student,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v8_publication_baselines import _load_gain, _openbmi_view  # noqa: E402


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _save_standardizer(path: Path, standardizer: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            atc_mean=standardizer.atc_mean,
            atc_scale=standardizer.atc_scale,
            fbc_mean=standardizer.fbc_mean,
            fbc_scale=standardizer.fbc_scale,
        )
    temporary.replace(path)


def _load_standardizer(path: Path) -> Any:
    with np.load(path, allow_pickle=False) as archive:
        from dpc_snn.experiments.v9_dual_feature_training import FrozenFeatureStandardizer

        return FrozenFeatureStandardizer(
            atc_mean=np.asarray(archive["atc_mean"], dtype=np.float32),
            atc_scale=np.asarray(archive["atc_scale"], dtype=np.float32),
            fbc_mean=np.asarray(archive["fbc_mean"], dtype=np.float32),
            fbc_scale=np.asarray(archive["fbc_scale"], dtype=np.float32),
        )


@torch.no_grad()
def _extract_features(
    *,
    model_name: str,
    checkpoint_path: Path,
    carrier: np.ndarray,
    source_root: Path,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    adapter = build_v62_neural_baseline(
        model_name,
        source_root=source_root,
        n_channels=int(checkpoint["n_channels"]),
        n_classes=int(checkpoint["n_classes"]),
        samples=int(checkpoint["samples"]),
    )
    adapter.load_state_dict(checkpoint["state_dict"], strict=True)
    projection_count = project_registered_max_norm_constraints_(adapter)
    if model_name == "atcnet":
        wrapper = V8ATCAccuracyBackbone(adapter.module)
        expected = (carrier.shape[0], 18, 32)
    elif model_name == "fbcnet":
        wrapper = V9FBCFeatureBackbone(adapter.module)
        expected = (carrier.shape[0], 4, 288)
    else:
        raise ValueError(f"unsupported frozen feature model: {model_name}")
    wrapper.eval().to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.ascontiguousarray(carrier, dtype=np.float32))),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    sequences: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for (batch_x,) in loader:
        output = wrapper(batch_x.to(device, non_blocking=True))
        continuous = (
            output["aux"]["continuous_sequence"]
            if model_name == "atcnet"
            else output["continuous_sequence"]
        )
        sequences.append(continuous.float().cpu().numpy())
        logits.append(output["logits"].float().cpu().numpy())
    sequence = np.concatenate(sequences)
    all_logits = np.concatenate(logits)
    if sequence.shape != expected or all_logits.shape != (carrier.shape[0], 2):
        raise RuntimeError(f"{model_name} returned unexpected OpenBMI feature shapes")
    del wrapper, adapter, checkpoint
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return sequence, all_logits, projection_count


def _view(config: dict[str, Any], subject: int, session: str, role: str) -> tuple[Any, ...]:
    publication_config = config["publication_config"]
    dataset = dict(publication_config["datasets"]["openbmi"])
    expected = int(
        dataset["expected_train_trials"]
        if role == "training"
        else dataset["expected_evaluation_trials"]
    )
    return _openbmi_view(
        subject=subject,
        session=session,
        role=role,
        dataset_config=dataset,
        channel_names=list(publication_config["channel_names"]),
        expected_trials=expected,
    )


def _prepared_carriers(
    config: dict[str, Any], subject: int, session: str, role: str, publication: Path
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    x_raw, labels, rows, manifest, _ = _view(config, subject, session, role)
    sfreq = float(rows[0]["sfreq"])
    carrier = task_carrier(x_raw, sfreq=sfreq, epoch_tmin=float(rows[0]["epoch_tmin"]))
    gain = _load_gain(
        publication
        / "runs"
        / "atcnet"
        / f"subject_{subject:02d}"
        / "seed_0"
        / "training"
        / "gain.npz"
    )
    normalized = apply_fixed_gain(carrier, gain)
    prepared = {
        name: prepare_model_input(name, normalized, sfreq=sfreq)
        for name in ("atcnet", "fbcnet")
    }
    data_identity = {
        "session": session,
        "signal_sha256": _array_sha256(x_raw),
        "label_sha256": _array_sha256(labels),
        "trial_ids_sha256": sha256_fingerprint([row["trial_id"] for row in rows]),
        "adapter_manifest_sha256": sha256_fingerprint(manifest),
    }
    return prepared, np.asarray(labels, dtype=np.int64), rows, data_identity


def _checkpoint_path(publication: Path, model: str, subject: int, seed: int) -> Path:
    return (
        publication
        / "runs"
        / model
        / f"subject_{subject:02d}"
        / f"seed_{seed}"
        / "training"
        / "checkpoint.pt"
    )


def _variant_dir(output: Path, subject: int, seed: int, variant: str) -> Path:
    return output / f"subject_{subject:02d}" / f"seed_{seed}" / variant


def _base_fingerprint(
    *,
    freeze: dict[str, Any],
    config: dict[str, Any],
    data_identity: dict[str, Any],
    publication: Path,
    subject: int,
    seed: int,
) -> dict[str, Any]:
    payload = {
        "freeze_sha256": freeze["combined_sha256"],
        "source_tree_sha256": freeze["source_tree_sha256"],
        "config": config["v29_config"],
        "data": data_identity,
        "subject": subject,
        "seed": seed,
        "atc_checkpoint_sha256": file_sha256(
            _checkpoint_path(publication, "atcnet", subject, seed)
        ),
        "fbc_checkpoint_sha256": file_sha256(
            _checkpoint_path(publication, "fbcnet", subject, seed)
        ),
    }
    payload["combined_sha256"] = sha256_fingerprint(payload)
    return payload


def _train(args: argparse.Namespace, config: dict[str, Any], freeze: dict[str, Any]) -> None:
    publication = Path(args.publication_root).resolve()
    output = Path(args.output).resolve()
    prepared, labels, _, data_identity = _prepared_carriers(
        config, args.subject, "S1", "training", publication
    )
    for seed in config["v29_config"]["seeds"]:
        atc, atc_logits, atc_projection = _extract_features(
            model_name="atcnet",
            checkpoint_path=_checkpoint_path(publication, "atcnet", args.subject, seed),
            carrier=prepared["atcnet"],
            source_root=Path(args.source_root).resolve(),
            device=args.device,
            batch_size=args.feature_batch_size,
        )
        fbc, fbc_logits, fbc_projection = _extract_features(
            model_name="fbcnet",
            checkpoint_path=_checkpoint_path(publication, "fbcnet", args.subject, seed),
            carrier=prepared["fbcnet"],
            source_root=Path(args.source_root).resolve(),
            device=args.device,
            batch_size=args.feature_batch_size,
        )
        standardizer = fit_feature_standardizer(atc, fbc)
        train_atc, train_fbc = standardizer.transform(atc, fbc)
        teacher = equal_probability_teacher(atc_logits, fbc_logits)
        base = _base_fingerprint(
            freeze=freeze,
            config=config,
            data_identity=data_identity,
            publication=publication,
            subject=args.subject,
            seed=seed,
        )
        for variant in config["v29_config"]["variants"]:
            directory = ensure_dir(_variant_dir(output, args.subject, seed, variant))
            fingerprint = {**base, "variant": variant}
            fingerprint["combined_sha256"] = sha256_fingerprint(fingerprint)
            fingerprint_path = directory / "training_fingerprint.json"
            required = ("checkpoint.pt", "standardizer.npz", "training_metrics.json")
            if all((directory / name).is_file() for name in required):
                if read_json(fingerprint_path) != fingerprint:
                    raise RuntimeError(f"stale V29 training resume: {directory}")
                continue
            if fingerprint_path.is_file() and read_json(fingerprint_path) != fingerprint:
                raise RuntimeError(f"partial stale V29 training resume: {directory}")
            write_json(fingerprint_path, fingerprint)
            # Matched ANN/SNN variants share the same initialization seed. Their
            # only permitted difference is the registered state equation.
            run_seed = 2_900_000 + args.subject * 10_000 + seed * 101
            fit = fit_v9_dual_feature(
                variant,
                atc_train=train_atc,
                fbc_train=train_fbc,
                y_train=labels,
                teacher_train=teacher,
                atc_validation=None,
                fbc_validation=None,
                y_validation=None,
                teacher_validation=None,
                device=args.device,
                seed=run_seed,
                fixed_epoch=int(config["v29_config"]["fixed_epochs"]),
                scheduler_epochs=int(config["v29_config"]["fixed_epochs"]),
                batch_size=int(config["v29_config"]["batch_size"]),
                learning_rate=float(config["v29_config"]["learning_rate"]),
                weight_decay=float(config["v29_config"]["weight_decay"]),
                firing_rate_weight=float(config["v29_config"]["firing_rate_weight"]),
                model_kwargs=dict(config["v29_config"]["model"]),
                run_label=f"V29:S{args.subject}:seed{seed}:{variant}",
            )
            _atomic_torch_save(
                {
                    "state_dict": fit.last_state,
                    "variant": variant,
                    "subject": args.subject,
                    "seed": seed,
                    "n_classes": 2,
                    "training_fingerprint_sha256": fingerprint["combined_sha256"],
                    "freeze_sha256": freeze["combined_sha256"],
                },
                directory / "checkpoint.pt",
            )
            _save_standardizer(directory / "standardizer.npz", standardizer)
            write_csv(directory / "history.csv", fit.history)
            write_json(
                directory / "training_metrics.json",
                {
                    "status": "training_completed_checkpoint_sealed",
                    "subject": args.subject,
                    "seed": seed,
                    "variant": variant,
                    "fixed_epochs": fit.best_epoch,
                    "optimizer_steps": fit.optimizer_steps,
                    "parameters": fit.model.parameter_count,
                    "elapsed_seconds": fit.elapsed_seconds,
                    "atc_projection_count": atc_projection,
                    "fbc_projection_count": fbc_projection,
                    "openbmi_s2_accessed": False,
                },
            )
            del fit
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_json(
        ensure_dir(output / f"subject_{args.subject:02d}") / "training_status.json",
        {"status": "completed", "subject": args.subject, "openbmi_s2_accessed": False},
    )


def _published_logits(publication: Path, model: str, subject: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    path = (
        publication
        / "runs"
        / model
        / f"subject_{subject:02d}"
        / f"seed_{seed}"
        / "evaluation"
        / "predictions.npz"
    )
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive["logits"], dtype=np.float32), np.asarray(
            archive["label"], dtype=np.int64
        )


def _evaluate(args: argparse.Namespace, config: dict[str, Any], freeze: dict[str, Any]) -> None:
    output = Path(args.output).resolve()
    barrier = read_json(Path(args.checkpoint_barrier).resolve())
    if barrier.get("freeze_sha256") != freeze["combined_sha256"] or barrier.get("status") != "sealed":
        raise RuntimeError("V29 evaluation requires its complete sealed checkpoint barrier")
    publication = Path(args.publication_root).resolve()
    prepared, labels, rows, data_identity = _prepared_carriers(
        config, args.subject, "S2", "evaluation", publication
    )
    for seed in config["v29_config"]["seeds"]:
        atc, atc_logits, _ = _extract_features(
            model_name="atcnet",
            checkpoint_path=_checkpoint_path(publication, "atcnet", args.subject, seed),
            carrier=prepared["atcnet"],
            source_root=Path(args.source_root).resolve(),
            device=args.device,
            batch_size=args.feature_batch_size,
        )
        fbc, fbc_logits, _ = _extract_features(
            model_name="fbcnet",
            checkpoint_path=_checkpoint_path(publication, "fbcnet", args.subject, seed),
            carrier=prepared["fbcnet"],
            source_root=Path(args.source_root).resolve(),
            device=args.device,
            batch_size=args.feature_batch_size,
        )
        saved_atc, saved_labels = _published_logits(publication, "atcnet", args.subject, seed)
        saved_fbc, saved_fbc_labels = _published_logits(publication, "fbcnet", args.subject, seed)
        if not np.array_equal(labels, saved_labels) or not np.array_equal(labels, saved_fbc_labels):
            raise RuntimeError("current OpenBMI S2 labels do not align with sealed publication artifacts")
        replay_error = max(
            float(np.max(np.abs(atc_logits - saved_atc))),
            float(np.max(np.abs(fbc_logits - saved_fbc))),
        )
        if replay_error > 1e-5:
            raise RuntimeError(f"frozen teacher replay drifted by {replay_error}")
        teacher = equal_probability_teacher(atc_logits, fbc_logits)
        for variant in config["v29_config"]["variants"]:
            directory = _variant_dir(output, args.subject, seed, variant)
            checkpoint_path = directory / "checkpoint.pt"
            expected_sha = barrier["checkpoints"].get(str(checkpoint_path))
            if expected_sha is None or file_sha256(checkpoint_path) != expected_sha:
                raise RuntimeError(f"V29 student checkpoint changed after barrier: {checkpoint_path}")
            evaluation_dir = ensure_dir(directory / "evaluation")
            prediction_path = evaluation_dir / "predictions.npz"
            if prediction_path.is_file() and (evaluation_dir / "metrics.json").is_file():
                continue
            standardizer = _load_standardizer(directory / "standardizer.npz")
            eval_atc, eval_fbc = standardizer.transform(atc, fbc)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = build_v9_dual_feature_student(
                "ann_sew" if variant == "ann_sew_ce" else "sew_clif",
                **dict(config["v29_config"]["model"]),
            )
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            prediction = predict_v9_dual_feature(
                model,
                eval_atc,
                eval_fbc,
                labels,
                teacher,
                device=args.device,
                batch_size=int(config["v29_config"]["batch_size"]),
            )
            with prediction_path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    logits=prediction["logits"],
                    pred=prediction["logits"].argmax(axis=1),
                    label=labels,
                    trial_id=np.asarray([row["trial_id"] for row in rows]),
                )
            metrics = {
                key: float(prediction[key])
                for key in ("accuracy", "balanced_accuracy", "kappa", "macro_f1", "mean_firing_rate")
            }
            metrics.update(
                {
                    "status": "completed",
                    "subject": args.subject,
                    "seed": seed,
                    "variant": variant,
                    "teacher_replay_max_abs_error": replay_error,
                    "training_checkpoint_sha256": expected_sha,
                    "s2_data_identity": data_identity,
                    "openbmi_s2_gradient_updates": False,
                }
            )
            write_json(evaluation_dir / "metrics.json", metrics)
            del model, checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_json(
        ensure_dir(output / f"subject_{args.subject:02d}") / "evaluation_status.json",
        {
            "status": "completed",
            "subject": args.subject,
            "openbmi_s2_accessed": True,
            "openbmi_s2_gradient_updates": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "evaluate"), required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--publication-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-barrier")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.subject not in range(1, 55):
        raise ValueError("OpenBMI subject must lie in [1, 54]")
    if args.phase == "evaluate" and not args.checkpoint_barrier:
        raise ValueError("evaluation requires --checkpoint-barrier")
    configure_cache_env()
    freeze = read_json(Path(args.freeze).resolve())
    if freeze.get("status") != "frozen_before_current_campaign_s2_access":
        raise RuntimeError("invalid V29 freeze manifest")
    current_source = source_tree_digest(collect_source_tree_manifest(ROOT))
    if current_source != freeze["source_tree_sha256"]:
        raise RuntimeError("V29 source tree changed after protocol freeze")
    config_path = ROOT / "configs" / "experiments" / "v29_openbmi_replication.yaml"
    publication_config_path = ROOT / "configs" / "experiments" / "v8_publication_baselines.yaml"
    config = {
        "v29_config": yaml.safe_load(config_path.read_text(encoding="utf-8")),
        "publication_config": yaml.safe_load(
            publication_config_path.read_text(encoding="utf-8")
        ),
    }
    started = time.time()
    if args.phase == "train":
        _train(args, config, freeze)
    else:
        _evaluate(args, config, freeze)
    print(
        json.dumps(
            {"status": "completed", "phase": args.phase, "subject": args.subject, "elapsed_seconds": time.time() - started},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
