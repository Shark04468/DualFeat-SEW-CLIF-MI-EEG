#!/usr/bin/env python3
"""Run one OpenBMI subject for the E29 objective-pure replication."""

from __future__ import annotations

import argparse
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

from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    sha256_fingerprint,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    source_tree_digest,
)
from dpc_snn.experiments.v9_dual_feature_training import (  # noqa: E402
    equal_probability_teacher,
    fit_v9_dual_feature,
    predict_v9_dual_feature,
)
from dpc_snn.models.v9_dual_feature_student import (  # noqa: E402
    build_v9_dual_feature_student,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v29_openbmi_subject import (  # noqa: E402
    _checkpoint_path,
    _extract_features,
    _load_standardizer,
    _prepared_carriers,
    _published_logits,
)


DEFAULT_CONFIG = "configs/experiments/v33_e29_zero_penalty.yaml"
CHILD_VARIANT = "sew_clif_ce_fr0"


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _parent_variant(parent: Path, subject: int, seed: int) -> Path:
    return parent / f"subject_{subject:02d}" / f"seed_{seed}" / "ann_sew_ce"


def _child_variant(output: Path, subject: int, seed: int) -> Path:
    return output / f"subject_{subject:02d}" / f"seed_{seed}" / CHILD_VARIANT


def _validate_parent(parent: Path) -> None:
    decision = read_json(parent / "aggregate" / "gate_decision.json")
    statuses = sorted(parent.glob("subject_*/evaluation_status.json"))
    complete = len(statuses) == 54 and all(
        read_json(path).get("status") == "completed" for path in statuses
    )
    if not complete or decision.get("status") != "pass":
        raise RuntimeError("E29-ZP requires the completed passing E29 parent")


def _load_config_bundle(config: dict[str, Any]) -> dict[str, Any]:
    publication_path = ROOT / "configs" / "experiments" / "v8_publication_baselines.yaml"
    return {
        "v29_config": {
            **config,
            "fixed_epochs": config["training"]["fixed_epochs"],
            "batch_size": config["training"]["batch_size"],
            "learning_rate": config["training"]["learning_rate"],
            "weight_decay": config["training"]["weight_decay"],
            "firing_rate_weight": 0.0,
        },
        "publication_config": yaml.safe_load(publication_path.read_text(encoding="utf-8")),
    }


def _train(
    args: argparse.Namespace,
    config: dict[str, Any],
    bundle: dict[str, Any],
    config_sha: str,
    source_sha: str,
) -> None:
    publication = Path(args.publication_root).resolve()
    parent = Path(args.parent).resolve()
    output = Path(args.output).resolve()
    prepared, labels, _, data_identity = _prepared_carriers(
        bundle, args.subject, "S1", "training", publication
    )
    checkpoints = []
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        atc_checkpoint = _checkpoint_path(publication, "atcnet", args.subject, seed)
        fbc_checkpoint = _checkpoint_path(publication, "fbcnet", args.subject, seed)
        atc, atc_logits, _ = _extract_features(
            model_name="atcnet",
            checkpoint_path=atc_checkpoint,
            carrier=prepared["atcnet"],
            source_root=Path(args.source_root).resolve(),
            device=args.device,
            batch_size=args.feature_batch_size,
        )
        fbc, fbc_logits, _ = _extract_features(
            model_name="fbcnet",
            checkpoint_path=fbc_checkpoint,
            carrier=prepared["fbcnet"],
            source_root=Path(args.source_root).resolve(),
            device=args.device,
            batch_size=args.feature_batch_size,
        )
        parent_variant = _parent_variant(parent, args.subject, seed)
        standardizer_path = parent_variant / "standardizer.npz"
        standardizer = _load_standardizer(standardizer_path)
        train_atc, train_fbc = standardizer.transform(atc, fbc)
        teacher = equal_probability_teacher(atc_logits, fbc_logits)
        run_seed = 2_900_000 + args.subject * 10_000 + seed * 101
        fingerprint = {
            "schema": "dpc-snn-e29zp-training-run/v1",
            "config_sha256": config_sha,
            "source_tree_sha256": source_sha,
            "parent_run": parent.name,
            "parent_standardizer_sha256": file_sha256(standardizer_path),
            "atc_checkpoint_sha256": file_sha256(atc_checkpoint),
            "fbc_checkpoint_sha256": file_sha256(fbc_checkpoint),
            "training_data_identity": data_identity,
            "subject": args.subject,
            "seed": seed,
            "run_seed": run_seed,
            "firing_rate_weight": 0.0,
        }
        fingerprint["combined_sha256"] = sha256_fingerprint(fingerprint)
        variant_dir = _child_variant(output, args.subject, seed)
        checkpoint_path = variant_dir / "checkpoint.pt"
        metrics_path = variant_dir / "training_metrics.json"
        fingerprint_path = variant_dir / "training_fingerprint.json"
        if checkpoint_path.is_file() and metrics_path.is_file():
            if read_json(fingerprint_path) != fingerprint:
                raise RuntimeError(f"stale E29-ZP resume rejected: {variant_dir}")
            checkpoints.append(str(checkpoint_path.resolve()))
            continue
        ensure_dir(variant_dir)
        write_json(fingerprint_path, fingerprint)
        training = config["training"]
        fit = fit_v9_dual_feature(
            "sew_clif_ce",
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
            fixed_epoch=int(training["fixed_epochs"]),
            scheduler_epochs=int(training["fixed_epochs"]),
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            firing_rate_weight=0.0,
            model_kwargs=dict(config["model"]),
            run_label=f"E29ZP:S{args.subject}:seed{seed}",
        )
        _atomic_torch_save(
            {
                "state_dict": fit.last_state,
                "subject": args.subject,
                "seed": seed,
                "variant": CHILD_VARIANT,
                "n_classes": 2,
                "training_fingerprint_sha256": fingerprint["combined_sha256"],
            },
            checkpoint_path,
        )
        write_csv(variant_dir / "history.csv", fit.history)
        write_json(
            metrics_path,
            {
                "status": "training_completed_checkpoint_sealed",
                "subject": args.subject,
                "seed": seed,
                "variant": CHILD_VARIANT,
                "fixed_epochs": int(training["fixed_epochs"]),
                "optimizer_steps": fit.optimizer_steps,
                "parameters": fit.model.parameter_count,
                "elapsed_seconds": fit.elapsed_seconds,
                "firing_rate_weight": 0.0,
                "openbmi_s2_accessed": False,
            },
        )
        checkpoints.append(str(checkpoint_path.resolve()))
        del fit
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(
        ensure_dir(output / f"subject_{args.subject:02d}") / "training_status.json",
        {
            "status": "completed",
            "subject": args.subject,
            "checkpoints": sorted(checkpoints),
            "openbmi_s2_accessed": False,
        },
    )


def _evaluate(
    args: argparse.Namespace,
    config: dict[str, Any],
    bundle: dict[str, Any],
    config_sha: str,
    source_sha: str,
) -> None:
    barrier = read_json(Path(args.checkpoint_barrier).resolve())
    if (
        barrier.get("schema") != "dpc-snn-e29zp-checkpoint-barrier/v1"
        or barrier.get("status") != "sealed"
        or barrier.get("config_sha256") != config_sha
        or barrier.get("source_tree_sha256") != source_sha
    ):
        raise RuntimeError("E29-ZP evaluation requires its checkpoint barrier")
    publication = Path(args.publication_root).resolve()
    parent = Path(args.parent).resolve()
    output = Path(args.output).resolve()
    prepared, labels, rows, _ = _prepared_carriers(
        bundle, args.subject, "S2", "evaluation", publication
    )
    for seed_value in config["seeds"]:
        seed = int(seed_value)
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
        saved_atc, saved_labels = _published_logits(
            publication, "atcnet", args.subject, seed
        )
        saved_fbc, saved_fbc_labels = _published_logits(
            publication, "fbcnet", args.subject, seed
        )
        if not np.array_equal(labels, saved_labels) or not np.array_equal(
            labels, saved_fbc_labels
        ):
            raise RuntimeError("OpenBMI S2 labels drifted from publication artifacts")
        replay_error = max(
            float(np.max(np.abs(atc_logits - saved_atc))),
            float(np.max(np.abs(fbc_logits - saved_fbc))),
        )
        if replay_error > 1e-5:
            raise RuntimeError(f"OpenBMI teacher replay drifted by {replay_error}")
        teacher = equal_probability_teacher(atc_logits, fbc_logits)
        parent_variant = _parent_variant(parent, args.subject, seed)
        standardizer = _load_standardizer(parent_variant / "standardizer.npz")
        eval_atc, eval_fbc = standardizer.transform(atc, fbc)
        variant_dir = _child_variant(output, args.subject, seed)
        checkpoint_path = variant_dir / "checkpoint.pt"
        expected_sha = barrier["checkpoints"].get(str(checkpoint_path.resolve()))
        if expected_sha is None or file_sha256(checkpoint_path) != expected_sha:
            raise RuntimeError(f"E29-ZP checkpoint changed: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = build_v9_dual_feature_student("sew_clif", **dict(config["model"]))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        prediction = predict_v9_dual_feature(
            model,
            eval_atc,
            eval_fbc,
            labels,
            teacher,
            device=args.device,
            batch_size=int(config["training"]["batch_size"]),
        )
        evaluation_dir = ensure_dir(variant_dir / "evaluation")
        _atomic_npz(
            evaluation_dir / "predictions.npz",
            logits=np.asarray(prediction["logits"], dtype=np.float32),
            pred=np.asarray(prediction["logits"]).argmax(axis=1),
            label=labels,
            trial_id=np.asarray([row["trial_id"] for row in rows]),
        )
        write_json(
            evaluation_dir / "metrics.json",
            {
                "status": "completed",
                "subject": args.subject,
                "seed": seed,
                "variant": CHILD_VARIANT,
                **{
                    key: float(prediction[key])
                    for key in (
                        "accuracy",
                        "balanced_accuracy",
                        "kappa",
                        "macro_f1",
                        "mean_firing_rate",
                    )
                },
                "teacher_replay_max_abs_error": replay_error,
                "training_checkpoint_sha256": expected_sha,
                "parent_ann_predictions_sha256": file_sha256(
                    parent_variant / "evaluation" / "predictions.npz"
                ),
                "openbmi_s2_gradient_updates": False,
            },
        )
        del model, checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(
        ensure_dir(output / f"subject_{args.subject:02d}") / "evaluation_status.json",
        {
            "status": "completed",
            "subject": args.subject,
            "openbmi_s2_gradient_updates": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "evaluate"), required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--publication-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-barrier")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=96)
    args = parser.parse_args()
    if args.subject not in range(1, 55):
        raise ValueError("OpenBMI subject must lie in [1, 54]")
    if args.phase == "evaluate" and not args.checkpoint_barrier:
        raise ValueError("evaluation requires --checkpoint-barrier")
    configure_cache_env()
    config_path = (ROOT / args.config).resolve()
    config = _config(config_path)
    _validate_parent(Path(args.parent).resolve())
    config_sha = file_sha256(config_path)
    source_sha = source_tree_digest(collect_source_tree_manifest(ROOT))
    bundle = _load_config_bundle(config)
    started = time.time()
    if args.phase == "train":
        _train(args, config, bundle, config_sha, source_sha)
    else:
        _evaluate(args, config, bundle, config_sha, source_sha)
    print(
        {
            "status": "completed",
            "phase": args.phase,
            "subject": args.subject,
            "elapsed_seconds": time.time() - started,
        }
    )


if __name__ == "__main__":
    main()
