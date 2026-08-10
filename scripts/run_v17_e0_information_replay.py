#!/usr/bin/env python3
"""Audit exact branch-head replay and closed-form feature probes for E17-0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.baselines.neural import build_v62_neural_baseline  # noqa: E402
from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v8_baselines import session_t_development_view  # noqa: E402
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    source_tree_digest,
    v8_heldout_lock_manifest,
)
from dpc_snn.experiments.v17_information_replay import (  # noqa: E402
    fit_ridge_probe,
    select_ridge_alpha,
)
from dpc_snn.experiments.v62_protocol import file_sha256  # noqa: E402
from dpc_snn.models.v8_atc_backbone import V8ATCAccuracyBackbone  # noqa: E402
from dpc_snn.models.v9_dual_feature_student import V9FBCFeatureBackbone  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.metrics import classification_metrics  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v8_e1_baselines import _subject_file  # noqa: E402


PARTITIONS = (
    "selection_train",
    "selection_validation",
    "outer_train",
    "outer_test",
)
PROBES = ("atc_flat", "fbc_flat", "branch_logits", "full_concat")


def _metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    return classification_metrics(labels, logits.argmax(axis=1), n_classes=4)


def _probe_features(cache: dict[str, np.ndarray], partition: str) -> dict[str, np.ndarray]:
    atc = np.asarray(cache[f"atcnet_{partition}_sequence"], dtype=np.float32)
    fbc = np.asarray(cache[f"fbcnet_{partition}_sequence"], dtype=np.float32)
    atc_logits = np.asarray(cache[f"atcnet_{partition}_logits"], dtype=np.float32)
    fbc_logits = np.asarray(cache[f"fbcnet_{partition}_logits"], dtype=np.float32)
    return {
        "atc_flat": atc.reshape(atc.shape[0], -1),
        "fbc_flat": fbc.reshape(fbc.shape[0], -1),
        "branch_logits": np.concatenate((atc_logits, fbc_logits), axis=1),
        "full_concat": np.concatenate(
            (atc.reshape(atc.shape[0], -1), fbc.reshape(fbc.shape[0], -1)), axis=1
        ),
    }


@torch.no_grad()
def _replay_errors(
    *,
    cache: dict[str, np.ndarray],
    source_root: Path,
    checkpoints: dict[str, Path],
    device: str,
) -> dict[str, float]:
    errors: dict[str, float] = {}
    for scope, partitions in (
        ("selection", ("selection_train", "selection_validation")),
        ("outer", ("outer_train", "outer_test")),
    ):
        atc_adapter = build_v62_neural_baseline(
            "atcnet", source_root=source_root, n_channels=22, n_classes=4, samples=1000
        )
        atc_adapter.load_state_dict(
            torch.load(checkpoints[f"atcnet_{scope}"], map_location="cpu", weights_only=True),
            strict=True,
        )
        atc = V8ATCAccuracyBackbone(atc_adapter.module).eval().to(device)
        fbc_adapter = build_v62_neural_baseline(
            "fbcnet", source_root=source_root, n_channels=22, n_classes=4, samples=1000
        )
        fbc_adapter.load_state_dict(
            torch.load(checkpoints[f"fbcnet_{scope}"], map_location="cpu", weights_only=True),
            strict=True,
        )
        fbc = V9FBCFeatureBackbone(fbc_adapter.module).eval().to(device)
        for partition in partitions:
            atc_sequence = torch.from_numpy(
                np.asarray(cache[f"atcnet_{partition}_sequence"], dtype=np.float32)
            ).to(device)
            atc_logits = atc.anchor_logits_from_sequence(atc_sequence).cpu().numpy()
            saved_atc = np.asarray(cache[f"atcnet_{partition}_logits"], dtype=np.float32)
            errors[f"atcnet_{partition}"] = float(np.max(np.abs(atc_logits - saved_atc)))

            fbc_sequence = torch.from_numpy(
                np.asarray(cache[f"fbcnet_{partition}_sequence"], dtype=np.float32)
            ).to(device)
            temporal = fbc_sequence.transpose(1, 2).unsqueeze(-1)
            fbc_logits = fbc.official_core.lastLayer(torch.flatten(temporal, start_dim=1))
            saved_fbc = np.asarray(cache[f"fbcnet_{partition}_logits"], dtype=np.float32)
            errors[f"fbcnet_{partition}"] = float(
                np.max(np.abs(fbc_logits.cpu().numpy() - saved_fbc))
            )
        del atc, atc_adapter, fbc, fbc_adapter
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--e1-root", required=True)
    parser.add_argument("--v9-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subjects", default="1,3,8")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    configure_cache_env()
    data_root = Path(args.data).resolve()
    source_root = Path(args.source_root).resolve()
    e1_root = Path(args.e1_root).resolve()
    v9_root = Path(args.v9_root).resolve()
    output = Path(args.output).resolve()
    subjects = [int(item) for item in args.subjects.split(",") if item.strip()]
    if not subjects or any(subject not in range(1, 10) for subject in subjects):
        raise ValueError("subjects must lie in [1, 9]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if output.exists():
        raise FileExistsError(f"E17-0 output must be immutable and new: {output}")
    ensure_dir(output)
    write_json(output / "heldout_lock_manifest.json", v8_heldout_lock_manifest())
    source_tree = collect_source_tree_manifest(ROOT)
    write_json(output / "source_tree_manifest.json", source_tree)
    write_json(
        output / "source_tree_summary.json",
        {"files": len(source_tree), "sha256": source_tree_digest(source_tree)},
    )

    alphas = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    fold_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[int, str], list[np.ndarray]] = {}
    targets: dict[int, list[np.ndarray]] = {}
    indices: dict[int, list[np.ndarray]] = {}
    input_manifest: dict[str, Any] = {}
    for subject in subjects:
        data_path = _subject_file(data_root, subject)
        data = load_processed_npz(data_path)
        _, labels, _, access = session_t_development_view(data)
        labels = np.asarray(labels, dtype=np.int64)
        if (
            access.get("selected_session") != "T"
            or access.get("heldout_signals_used_by_development") is not False
            or access.get("heldout_labels_used_by_development") is not False
        ):
            raise RuntimeError("E17-0 must not access Session E")
        input_manifest[f"subject_{subject:02d}_data"] = file_sha256(data_path)
        targets[subject] = []
        indices[subject] = []
        for name in (*PROBES, "teacher", "atcnet", "fbcnet"):
            predictions[(subject, name)] = []

        for fold in range(6):
            v9_fold = v9_root / f"subject_{subject:02d}" / f"seed_{args.seed}" / f"fold_{fold}"
            if read_json(v9_fold / "campaign_status.json").get("status") != "completed":
                raise RuntimeError(f"incomplete V9 anchor: {v9_fold}")
            cache_path = v9_fold / "frozen_dual_feature_cache.npz"
            with np.load(cache_path, allow_pickle=False) as archive:
                cache = {name: archive[name] for name in archive.files}
            input_manifest[f"subject_{subject:02d}_fold_{fold}_cache"] = file_sha256(cache_path)
            fold_indices = {
                "selection_train": cache["inner_train_indices"].astype(np.int64),
                "selection_validation": cache["inner_validation_indices"].astype(np.int64),
                "outer_train": cache["outer_train_indices"].astype(np.int64),
                "outer_test": cache["outer_test_indices"].astype(np.int64),
            }
            checkpoints: dict[str, Path] = {}
            for branch in ("atcnet", "fbcnet"):
                e1_fold = (
                    e1_root
                    / branch
                    / f"subject_{subject:02d}"
                    / f"seed_{args.seed}"
                    / f"fold_{fold}"
                )
                checkpoints[f"{branch}_selection"] = e1_fold / "selection_best.pt"
                checkpoints[f"{branch}_outer"] = e1_fold / "best.pt"
                for scope in ("selection", "outer"):
                    path = checkpoints[f"{branch}_{scope}"]
                    input_manifest[f"subject_{subject:02d}_fold_{fold}_{branch}_{scope}"] = (
                        file_sha256(path)
                    )
            errors = _replay_errors(
                cache=cache,
                source_root=source_root,
                checkpoints=checkpoints,
                device=args.device,
            )
            maximum_error = max(errors.values())
            replay_rows.append(
                {
                    "subject": subject,
                    "seed": int(args.seed),
                    "fold": fold,
                    "maximum_absolute_logit_error": maximum_error,
                    "replay_passed": maximum_error < 1e-5,
                    **errors,
                }
            )
            if maximum_error >= 1e-5:
                raise RuntimeError(f"exact branch-head replay failed for subject {subject}, fold {fold}")

            feature_sets = {
                partition: _probe_features(cache, partition) for partition in PARTITIONS
            }
            outer_labels = labels[fold_indices["outer_test"]]
            fold_prediction: dict[str, np.ndarray] = {}
            for probe_name in PROBES:
                selected_alpha, search_rows = select_ridge_alpha(
                    feature_sets["selection_train"][probe_name],
                    labels[fold_indices["selection_train"]],
                    feature_sets["selection_validation"][probe_name],
                    labels[fold_indices["selection_validation"]],
                    alphas=alphas,
                )
                probe = fit_ridge_probe(
                    feature_sets["outer_train"][probe_name],
                    labels[fold_indices["outer_train"]],
                    alpha=selected_alpha,
                )
                logits = probe.predict_logits(feature_sets["outer_test"][probe_name])
                fold_prediction[probe_name] = logits
                metrics = _metrics(outer_labels, logits)
                selected_search = next(
                    row for row in search_rows if row["alpha"] == selected_alpha
                )
                fold_rows.append(
                    {
                        "subject": subject,
                        "seed": int(args.seed),
                        "fold": fold,
                        "probe": probe_name,
                        "selected_alpha": selected_alpha,
                        "inner_validation_accuracy": selected_search["validation_accuracy"],
                        "inner_validation_kappa": selected_search["validation_kappa"],
                        "outer_accuracy": metrics["accuracy"],
                        "outer_kappa": metrics["kappa"],
                        "session_e_accessed": False,
                    }
                )
                predictions[(subject, probe_name)].append(logits)

            teacher = np.asarray(cache["teacher_outer_test"], dtype=np.float32)
            atc_logits = np.asarray(cache["atcnet_outer_test_logits"], dtype=np.float32)
            fbc_logits = np.asarray(cache["fbcnet_outer_test_logits"], dtype=np.float32)
            for name, logits in (
                ("teacher", teacher),
                ("atcnet", atc_logits),
                ("fbcnet", fbc_logits),
            ):
                predictions[(subject, name)].append(logits)
                fold_prediction[name] = logits
            targets[subject].append(outer_labels)
            indices[subject].append(fold_indices["outer_test"])
            np.savez_compressed(
                output / f"subject_{subject:02d}_fold_{fold}_predictions.npz",
                indices=fold_indices["outer_test"],
                labels=outer_labels,
                **{f"{name}_logits": value for name, value in fold_prediction.items()},
            )

    subject_rows: list[dict[str, Any]] = []
    for subject in subjects:
        all_indices = np.concatenate(indices[subject])
        all_labels = np.concatenate(targets[subject])
        if len(all_indices) != 288 or len(np.unique(all_indices)) != 288:
            raise RuntimeError(f"subject {subject} outer-fold coverage is not 288 unique trials")
        for name in (*PROBES, "teacher", "atcnet", "fbcnet"):
            logits = np.concatenate(predictions[(subject, name)])
            metrics = _metrics(all_labels, logits)
            subject_rows.append(
                {
                    "subject": subject,
                    "seed": int(args.seed),
                    "model": name,
                    "accuracy": metrics["accuracy"],
                    "kappa": metrics["kappa"],
                    "trials": len(all_labels),
                }
            )
    model_summary: list[dict[str, Any]] = []
    for name in (*PROBES, "teacher", "atcnet", "fbcnet"):
        rows = [row for row in subject_rows if row["model"] == name]
        model_summary.append(
            {
                "model": name,
                "mean_subject_accuracy": float(np.mean([row["accuracy"] for row in rows])),
                "mean_subject_kappa": float(np.mean([row["kappa"] for row in rows])),
                "subjects": len(rows),
            }
        )
    write_csv(output / "head_replay.csv", replay_rows)
    write_csv(output / "probe_folds.csv", fold_rows)
    write_csv(output / "probe_subjects.csv", subject_rows)
    write_csv(output / "probe_summary.csv", model_summary)
    write_json(output / "input_manifest.json", input_manifest)
    decision = {
        "status": "completed",
        "stage": "E17-0-information-replay",
        "exact_replay_passed": all(row["replay_passed"] for row in replay_rows),
        "maximum_absolute_logit_error": max(
            row["maximum_absolute_logit_error"] for row in replay_rows
        ),
        "models": {row["model"]: row for row in model_summary},
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
        "next_stage": "E17-1-global-selection",
    }
    write_json(output / "decision.json", decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
