#!/usr/bin/env python3
"""Calibrate frozen V14 expert residuals on one Session-T development fold."""

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

from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v62_protocol import file_sha256  # noqa: E402
from dpc_snn.experiments.v8_baselines import session_t_development_view  # noqa: E402
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    build_v8_run_fingerprint,
    collect_source_tree_manifest,
    source_tree_digest,
    validate_v8_resume_fingerprint,
    v8_heldout_lock_manifest,
    write_v8_fingerprint,
)
from dpc_snn.experiments.v9_dual_feature_training import (  # noqa: E402
    fit_feature_standardizer,
)
from dpc_snn.experiments.v14_residual_training import predict_v14  # noqa: E402
from dpc_snn.experiments.v15_residual_calibration import (  # noqa: E402
    E15_METHODS,
    calibrated_logits,
    rescue_damage,
    select_alpha,
)
from dpc_snn.models.v14_shared_residual_student import build_v14_student  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import accuracy, cohen_kappa  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v10_e1_strong_control_fold import (  # noqa: E402
    _array_sha256,
    _environment_manifest,
    _subject_file,
)


V14_SOURCE_DIGEST = "46d2e270e3d7f74da8705ff60fe06388267a41202efe181f32e7af59383a0f11"
SOURCE_VARIANTS = ("r1_atc_residual", "r4_generic_residual")


def _metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    prediction = logits.argmax(axis=1)
    return {
        "accuracy": accuracy(labels, prediction),
        "kappa": cohen_kappa(labels, prediction, n_classes=4),
    }


def _alpha_grid(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if (
        not parsed
        or len(parsed) != len(set(parsed))
        or any(not np.isfinite(item) or item < 0.0 for item in parsed)
    ):
        raise ValueError("alpha grid must contain unique non-negative finite values")
    return parsed


def _method_spec(method: str) -> tuple[str, str, bool] | None:
    if method == "r0_shared_replay":
        return None
    if method == "c1_atc_raw":
        return "r1_atc_residual", "atc_residual_logits", False
    if method == "c2_atc_rms":
        return "r1_atc_residual", "atc_residual_logits", True
    if method == "c3_generic_rms":
        return "r4_generic_residual", "generic_residual_logits", True
    raise KeyError(method)


def _v14_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _max_difference(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        return float("inf")
    return float(np.max(np.abs(first.astype(np.float64) - second.astype(np.float64))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--v9-anchor-root", required=True)
    parser.add_argument("--v14-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument(
        "--alphas", default="0,0.05,0.1,0.2,0.4,0.6,0.8,1.0,1.5,2.0"
    )
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-v14-source-digest", default=V14_SOURCE_DIGEST)
    args = parser.parse_args()

    configure_cache_env()
    if args.subject not in range(1, 10) or args.fold not in range(6):
        raise ValueError("invalid E15 subject or fold")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    alphas = _alpha_grid(args.alphas)
    if 0.0 not in alphas or tuple(sorted(alphas)) != alphas:
        raise ValueError("E15 alpha grid must be sorted and include zero")

    data_root = Path(args.data).resolve()
    v9_root = Path(args.v9_anchor_root).resolve()
    v14_root = Path(args.v14_root).resolve()
    output = Path(args.output).resolve()
    relative_fold = (
        Path(f"subject_{args.subject:02d}") / f"seed_{args.seed}" / f"fold_{args.fold}"
    )
    v9_fold = v9_root / relative_fold
    v14_fold = v14_root / relative_fold
    v14_status = read_json(v14_fold / "campaign_status.json")
    v14_source = read_json(v14_fold / "source_tree_summary.json")
    v14_replay = read_json(v14_fold / "shared_replay.json")
    if (
        v14_status.get("status") != "completed"
        or v14_status.get("session_e_accessed") is not False
        or v14_status.get("openbmi_s2_accessed") is not False
        or v14_source.get("sha256") != args.expected_v14_source_digest
        or v14_replay.get("status") != "passed"
    ):
        raise RuntimeError("V14 source fold failed E15 validation")

    cache_path = v9_fold / "frozen_dual_feature_cache.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        arrays = {name: np.asarray(cache[name]) for name in cache.files}
    required = {
        "inner_validation_indices",
        "outer_test_indices",
        *{
            f"{branch}_{partition}_{kind}"
            for branch in ("atcnet", "fbcnet")
            for partition in (
                "selection_train",
                "selection_validation",
                "outer_train",
                "outer_test",
            )
            for kind in ("sequence", "logits")
        },
        "teacher_selection_validation",
        "teacher_outer_test",
    }
    if not required.issubset(arrays):
        raise RuntimeError("V9 cache lacks E15 calibration inputs")

    subject_path = _subject_file(data_root, int(args.subject))
    data = load_processed_npz(subject_path)
    _, labels, _, access = session_t_development_view(data)
    validation_indices = arrays["inner_validation_indices"].astype(np.int64)
    outer_indices = arrays["outer_test_indices"].astype(np.int64)
    selection_standardizer = fit_feature_standardizer(
        arrays["atcnet_selection_train_sequence"],
        arrays["fbcnet_selection_train_sequence"],
    )
    outer_standardizer = fit_feature_standardizer(
        arrays["atcnet_outer_train_sequence"],
        arrays["fbcnet_outer_train_sequence"],
    )
    selection_features = selection_standardizer.transform(
        arrays["atcnet_selection_validation_sequence"],
        arrays["fbcnet_selection_validation_sequence"],
    )
    outer_features = outer_standardizer.transform(
        arrays["atcnet_outer_test_sequence"], arrays["fbcnet_outer_test_sequence"]
    )

    def teacher(partition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            arrays[f"teacher_{partition}"],
            arrays[f"atcnet_{partition}_logits"],
            arrays[f"fbcnet_{partition}_logits"],
        )

    evaluations: dict[str, dict[str, dict[str, Any]]] = {}
    replay_rows: list[dict[str, Any]] = []
    for variant in SOURCE_VARIANTS:
        variant_dir = v14_fold / variant
        variant_evaluations: dict[str, dict[str, Any]] = {}
        for partition, features, indices, checkpoint in (
            (
                "selection_validation",
                selection_features,
                validation_indices,
                variant_dir / "selection_best.pt",
            ),
            ("outer_test", outer_features, outer_indices, variant_dir / "outer_last.pt"),
        ):
            model = build_v14_student(variant)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            values = predict_v14(
                model,
                features[0],
                features[1],
                labels[indices],
                teacher(partition)[0],
                teacher(partition)[1],
                teacher(partition)[2],
                device=args.device,
                batch_size=int(args.batch_size),
            )
            variant_evaluations[partition] = values
        saved = _v14_archive(variant_dir / "outer_predictions.npz")
        outer_values = variant_evaluations["outer_test"]
        fields = (
            "logits",
            "shared_logits",
            "atc_residual_logits",
            "fbc_residual_logits",
            "generic_residual_logits",
        )
        for field in fields:
            difference = _max_difference(outer_values[field], saved[field])
            replay_rows.append(
                {"variant": variant, "field": field, "max_absolute_difference": difference}
            )
            if difference > 1e-6:
                raise RuntimeError(f"E15 failed exact V14 replay for {variant}/{field}")
        evaluations[variant] = variant_evaluations

    v14_r0 = _v14_archive(v14_fold / "r0_shared_replay" / "outer_predictions.npz")
    if not np.array_equal(v14_r0["indices"].astype(np.int64), outer_indices):
        raise RuntimeError("E15 outer indices do not match V14 R0")
    r1_outer = evaluations["r1_atc_residual"]["outer_test"]
    if _max_difference(v14_r0["logits"], r1_outer["shared_logits"]) > 1e-6:
        raise RuntimeError("E15 shared logits do not replay V14 R0")

    source_tree = collect_source_tree_manifest(ROOT)
    resolved_config = {
        **vars(args),
        "data": str(data_root),
        "v9_anchor_root": str(v9_root),
        "v14_root": str(v14_root),
        "output": str(output),
        "methods": list(E15_METHODS),
        "alphas": list(alphas),
        "selection": "inner-validation kappa, accuracy, smallest alpha",
        "outer_labels_used_for_selection": False,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
    }
    fingerprint = build_v8_run_fingerprint(
        resolved_run_config=resolved_config,
        source_tree=source_tree,
        data={"path": str(subject_path), "sha256": file_sha256(subject_path)},
        split={
            "inner_validation": _array_sha256(validation_indices),
            "outer_test": _array_sha256(outer_indices),
        },
        augmentation={"enabled": False},
        prior={"enabled": False, "calibration": "fold-local-inner-validation"},
        checkpoint={
            "v14_run_fingerprint": v14_status["run_fingerprint"],
            **{
                f"{variant}_{name}_sha256": file_sha256(v14_fold / variant / filename)
                for variant in SOURCE_VARIANTS
                for name, filename in (
                    ("selection", "selection_best.pt"),
                    ("outer", "outer_last.pt"),
                )
            },
        },
        environment=_environment_manifest(args.device),
    )
    if (output / "campaign_status.json").is_file():
        validate_v8_resume_fingerprint(output / "run_fingerprint.json", fingerprint)
        if read_json(output / "campaign_status.json").get("status") == "completed":
            print(json.dumps({"status": "skipped_completed", "output": str(output)}))
            return

    ensure_dir(output)
    write_json(output / "heldout_lock_manifest.json", v8_heldout_lock_manifest())
    write_json(output / "source_tree_manifest.json", source_tree)
    write_json(
        output / "source_tree_summary.json",
        {"files": len(source_tree), "sha256": source_tree_digest(source_tree)},
    )
    write_json(
        output / "v14_replay_audit.json",
        {
            "status": "passed",
            "tolerance": 1e-6,
            "rows": replay_rows,
            "shared_r0_max_absolute_difference": _max_difference(
                v14_r0["logits"], r1_outer["shared_logits"]
            ),
        },
    )
    write_v8_fingerprint(output / "run_fingerprint.json", fingerprint)
    (output / "resolved_run.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
    )

    started = time.time()
    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for method in E15_METHODS:
        spec = _method_spec(method)
        if spec is None:
            selected_alpha = 0.0
            validation_shared = evaluations["r1_atc_residual"]["selection_validation"]
            validation_metrics = _metrics(
                labels[validation_indices], validation_shared["shared_logits"]
            )
            outer_shared = np.asarray(v14_r0["logits"], dtype=np.float64)
            residual = np.zeros_like(outer_shared)
            outer_logits = outer_shared.copy()
            rms_normalize = False
            source_variant = "r0_shared_replay"
        else:
            source_variant, residual_field, rms_normalize = spec
            validation = evaluations[source_variant]["selection_validation"]
            selection = select_alpha(
                validation["shared_logits"],
                validation[residual_field],
                labels[validation_indices],
                alphas,
                rms_normalize=rms_normalize,
            )
            selected_alpha = selection.alpha
            validation_metrics = {
                "accuracy": selection.accuracy,
                "kappa": selection.kappa,
            }
            for row in selection.curve:
                curve_rows.append({"method": method, **row})
            outer_values = evaluations[source_variant]["outer_test"]
            outer_shared = outer_values["shared_logits"]
            residual = outer_values[residual_field]
            outer_logits = calibrated_logits(
                outer_shared,
                residual,
                selected_alpha,
                rms_normalize=rms_normalize,
            )
        metrics = _metrics(labels[outer_indices], outer_logits)
        diagnostics = rescue_damage(labels[outer_indices], outer_shared, outer_logits)
        row = {
            "method": method,
            "subject": int(args.subject),
            "seed": int(args.seed),
            "fold": int(args.fold),
            "source_variant": source_variant,
            "rms_normalize": bool(rms_normalize),
            "selected_alpha": float(selected_alpha),
            "inner_accuracy": validation_metrics["accuracy"],
            "inner_kappa": validation_metrics["kappa"],
            **metrics,
            **diagnostics,
            "session_e_accessed": False,
        }
        summary_rows.append(row)
        method_dir = ensure_dir(output / method)
        write_json(method_dir / "metrics.json", row)
        np.savez_compressed(
            method_dir / "outer_predictions.npz",
            indices=outer_indices,
            labels=labels[outer_indices],
            logits=np.asarray(outer_logits, dtype=np.float32),
            shared_logits=np.asarray(outer_shared, dtype=np.float32),
            residual_logits=np.asarray(residual, dtype=np.float32),
            selected_alpha=np.asarray([selected_alpha], dtype=np.float32),
            equal_teacher_logits=np.asarray(teacher("outer_test")[0], dtype=np.float32),
        )

    write_csv(output / "selection_curves.csv", curve_rows)
    write_csv(output / "summary.csv", summary_rows)
    status = {
        "status": "completed",
        "stage": "V15-E0-fold-local-residual-calibration",
        "subject": int(args.subject),
        "seed": int(args.seed),
        "fold": int(args.fold),
        "methods": list(E15_METHODS),
        "v14_replay": "passed",
        "outer_labels_used_for_selection": False,
        "elapsed_seconds": time.time() - started,
        "data_access": access,
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
        "run_fingerprint": fingerprint["combined_sha256"],
    }
    write_json(output / "campaign_status.json", status)
    print(json.dumps({"status": "completed", "rows": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
