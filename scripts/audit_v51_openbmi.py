#!/usr/bin/env python
"""Classifier-free OpenBMI delay audit and real-background injection power."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import tarfile
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_v42_fold_evidence import (  # noqa: E402
    _audit_space,
    _fit_prior,
    _inject_task_delay_latent,
    _known_route_metrics,
)
from dpc_snn.config import load_experiment_config, load_yaml  # noqa: E402
from dpc_snn.data.openbmi import load_openbmi_subject  # noqa: E402
from dpc_snn.experiments.common import subset  # noqa: E402
from dpc_snn.experiments.runners import _crop_to_common_task_window  # noqa: E402
from dpc_snn.models.eeg_frontend import AnchoredSpatialProjection  # noqa: E402
from dpc_snn.preprocessing.standardize import (  # noqa: E402
    apply_channelwise_zscore,
    channelwise_train_stats,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from run_v40_delay_gate import (  # noqa: E402
    _environment_manifest,
    _model_cfg,
    _reference_space,
    _repeated_stratified_folds,
    _source_fingerprint,
)


def _parse_csv(values: str, cast: Any) -> list[Any]:
    return [cast(value.strip()) for value in values.split(",") if value.strip()]


def _subject_gate(report: dict[str, Any], gate: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if int(report.get("accepted_evidence_edges", 0)) < int(gate["minimum_routes_per_subject"]):
        failures.append("insufficient_stable_routes")
    correlation = float(report.get("split_half_delay_correlation", float("nan")))
    if not np.isfinite(correlation) or correlation < float(
        gate["minimum_split_half_delay_correlation"]
    ):
        failures.append("split_half_delay_unstable")
    if bool(gate["phase_surrogate_must_reduce_support"]):
        if float(report.get("phase_surrogate_support_drop", 0.0)) <= 0.0:
            failures.append("phase_surrogate_support_not_reduced")
        if int(report.get("normal_node_edges", 0)) <= int(
            report.get("phase_surrogate_node_edges", 0)
        ):
            failures.append("phase_surrogate_edges_not_reduced")
    if float(report.get("time_reversal_transpose_support_mae", float("inf"))) > float(
        gate["time_reversal_max_transpose_mae"]
    ):
        failures.append("time_reversal_direction_not_transposed")
    return not failures, failures


def _fixed_train_fold(data: dict[str, Any], seed: int) -> dict[str, Any]:
    folds = _repeated_stratified_folds(np.asarray(data["y"]), 3, 2, seed)
    train_idx, _ = folds[0]
    return subset(data, train_idx)


def _standardized_reference(
    data: dict[str, Any], model_cfg: dict[str, Any], reference: str, seed: int
) -> dict[str, Any]:
    referenced, _ = _reference_space(data, model_cfg, reference)
    train = _fixed_train_fold(referenced, seed)
    mean, std = channelwise_train_stats(train["X"])
    return {**train, "X": apply_channelwise_zscore(train["X"], mean, std)}


def _projection(data: dict[str, Any], model_cfg: dict[str, Any]) -> np.ndarray:
    return (
        AnchoredSpatialProjection(
            int(np.asarray(data["X"]).shape[1]),
            int(model_cfg["latent_nodes"]),
            max_deviation=0.0,
            electrode_coordinates=data.get("electrode_coordinates"),
            anchor_indices=data.get("spatial_anchor_indices"),
        )
        .anchors.detach()
        .cpu()
        .numpy()
    )


def _injection_nodes(data: dict[str, Any], anchors: np.ndarray) -> tuple[int, int]:
    channels = [str(value) for value in data["ch_names"]]
    if "C3" not in channels or "C4" not in channels:
        raise ValueError("OpenBMI injection requires C3 and C4")
    source = int(np.argmax(anchors[:, channels.index("C3")]))
    target = int(np.argmax(anchors[:, channels.index("C4")]))
    if source == target:
        raise RuntimeError("C3 and C4 collapsed onto the same physical latent node")
    return source, target


def _resume_state(
    output: Path, manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = output / "audit_manifest.json"
    state_path = output / "partial_state.json"
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
        return [], []
    if read_json(manifest_path) != manifest:
        raise RuntimeError("Refusing stale OpenBMI audit resume: resolved manifest changed")
    if not state_path.exists():
        return [], []
    state = read_json(state_path)
    return list(state.get("real_rows", [])), list(state.get("injection_rows", []))


def _checkpoint(
    output: Path,
    real_rows: list[dict[str, Any]],
    injection_rows: list[dict[str, Any]],
) -> None:
    write_json(
        output / "partial_state.json",
        {"real_rows": real_rows, "injection_rows": injection_rows},
    )
    if real_rows:
        write_csv(output / "real_evidence_subject_summary.csv", real_rows)
    if injection_rows:
        write_csv(output / "real_background_injection_power.csv", injection_rows)


def _write_source_snapshot(output: Path) -> None:
    destination = output / "source_snapshot.tar.gz"
    if destination.exists():
        return
    temporary = output / ".source_snapshot.tar.gz.tmp"
    with tarfile.open(temporary, "w:gz") as archive:
        for relative in (
            "src",
            "configs",
            "scripts",
            "tests",
            "requirements.txt",
            "pyproject.toml",
        ):
            path = ROOT / relative
            archive.add(path, arcname=relative)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/all_experiments.yaml")
    parser.add_argument(
        "--protocol-config",
        default="configs/experiments/v51_hardware_free_plan.yaml",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--subjects", default="")
    parser.add_argument("--stage", choices=("real", "injection", "both"), default="real")
    parser.add_argument("--spaces", default="car,csd")
    parser.add_argument("--bootstrap-samples", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--injection-delays-ms", default="")
    parser.add_argument("--injection-strengths", default="")
    parser.add_argument("--injection-seeds", default="")
    args = parser.parse_args()

    output = ensure_dir(args.output)
    protocol_path = (ROOT / args.protocol_config).resolve()
    config_path = (ROOT / args.config).resolve()
    protocol = load_yaml(protocol_path)
    dataset_path = (ROOT / protocol["dataset_config"]).resolve()
    dataset_cfg = load_yaml(dataset_path)
    base_cfg = load_experiment_config(config_path, "E14", overrides={"device": args.device})
    model_cfg = _model_cfg(base_cfg, protocol)
    subjects = (
        _parse_csv(args.subjects, int)
        if args.subjects
        else [int(value) for value in dataset_cfg["development_subjects"]]
    )
    spaces = _parse_csv(args.spaces, str)
    invalid = sorted(set(spaces).difference({"car", "csd"}))
    if invalid:
        raise ValueError(f"Unknown evidence spaces: {invalid}")
    gate = protocol["real_evidence_gate"]
    duration = float(model_cfg["task_tmax"]) - float(model_cfg["task_tmin"])
    graph_step_ms = 1000.0 * duration / int(model_cfg["graph_timesteps"])
    power = protocol["mechanism_power"]
    delays_ms = (
        _parse_csv(args.injection_delays_ms, float)
        if args.injection_delays_ms
        else [float(value) for value in power["delay_milliseconds"]]
    )
    strengths = (
        _parse_csv(args.injection_strengths, float)
        if args.injection_strengths
        else [float(value) for value in power["injection_strengths"]]
    )
    injection_seeds = (
        _parse_csv(args.injection_seeds, int)
        if args.injection_seeds
        else [int(value) for value in power["semi_synthetic_seeds"]]
    )
    source_fingerprint = _source_fingerprint(protocol_path, config_path)
    manifest = {
        "experiment_id": protocol["experiment_id"],
        "protocol": protocol["protocol"],
        "architecture_version": protocol["architecture_version"],
        "source_fingerprint": source_fingerprint,
        "subjects": subjects,
        "requested_session": "S1",
        "heldout_session_S2_accessed": False,
        "classifier_training": False,
        "stage": args.stage,
        "spaces": spaces,
        "bootstrap_samples": int(args.bootstrap_samples),
        "injection_delays_ms": delays_ms,
        "injection_strengths": strengths,
        "injection_seeds": injection_seeds,
        "device": args.device,
        "environment": _environment_manifest(args.device),
    }
    real_rows, injection_rows = _resume_state(output, manifest)
    _write_source_snapshot(output)
    completed_real = {(int(row["subject"]), str(row["evidence_space"])) for row in real_rows}
    completed_injection = {
        (
            int(row["subject"]),
            str(row["evidence_space"]),
            int(row["injection_seed"]),
            float(row["strength"]),
            float(row["true_delay_ms"]),
        )
        for row in injection_rows
    }

    for subject_index, subject in enumerate(subjects):
        print(json.dumps({"event": "subject_loading", "subject": subject}), flush=True)
        data = load_openbmi_subject(
            subject,
            sessions=("S1",),
            tmin=float(dataset_cfg["tmin"]),
            tmax=float(dataset_cfg["tmax"]),
            resample=float(dataset_cfg["evidence_sfreq"]),
        )
        if bool(data["heldout_session_accessed"]):
            raise RuntimeError("OpenBMI Session S2 was accessed before architecture freeze")
        cropped = _crop_to_common_task_window(data, {"model": model_cfg})

        for space_index, space in enumerate(spaces):
            if args.stage in {"real", "both"}:
                real_key = (subject, space)
                if real_key in completed_real:
                    print(
                        json.dumps(
                            {"event": "real_evidence_resumed", "subject": subject, "space": space}
                        ),
                        flush=True,
                    )
                else:
                    standardized = _standardized_reference(
                        cropped,
                        model_cfg,
                        space,
                        seed=5101 + 100 * subject_index + space_index,
                    )
                    report, arrays = _audit_space(
                        space,
                        standardized,
                        model_cfg,
                        seed=5101 + 1000 * subject_index + 100 * space_index,
                        device=args.device,
                        bootstraps=int(args.bootstrap_samples),
                    )
                    passed, failures = _subject_gate(report, gate)
                    row = {
                        "subject": subject,
                        "session": "S1",
                        "subject_gate_passed": passed,
                        "subject_gate_failures": ";".join(failures),
                        **report,
                    }
                    real_rows.append(row)
                    completed_real.add(real_key)
                    subject_dir = ensure_dir(output / f"subject_{subject:02d}")
                    np.savez_compressed(subject_dir / f"{space}_prior.npz", **arrays)
                    write_json(subject_dir / f"{space}_summary.json", row)
                    _checkpoint(output, real_rows, injection_rows)
                    print(
                        json.dumps(
                            {
                                "event": "real_evidence_completed",
                                "subject": subject,
                                "space": space,
                                "passed": passed,
                                "accepted_routes": report["accepted_evidence_edges"],
                            }
                        ),
                        flush=True,
                    )
                    del standardized, arrays

            if args.stage in {"injection", "both"}:
                for injection_seed in injection_seeds:
                    pending = [
                        (strength, delay_ms)
                        for strength in strengths
                        for delay_ms in delays_ms
                        if (
                            subject,
                            space,
                            int(injection_seed),
                            float(strength),
                            float(delay_ms),
                        )
                        not in completed_injection
                    ]
                    if not pending:
                        print(
                            json.dumps(
                                {
                                    "event": "injection_seed_resumed",
                                    "subject": subject,
                                    "space": space,
                                    "injection_seed": int(injection_seed),
                                }
                            ),
                            flush=True,
                        )
                        continue
                    fold_seed = (
                        510100
                        + 10000 * subject_index
                        + 1000 * space_index
                        + 100 * int(injection_seed)
                    )
                    standardized = _standardized_reference(
                        cropped,
                        model_cfg,
                        space,
                        seed=fold_seed,
                    )
                    anchors = _projection(standardized, model_cfg)
                    source_node, target_node = _injection_nodes(standardized, anchors)
                    for strength, delay_ms in pending:
                        delay_steps = float(delay_ms) / graph_step_ms
                        injected = _inject_task_delay_latent(
                            standardized,
                            anchors,
                            source_node,
                            target_node,
                            delay_steps,
                            float(strength),
                            int(model_cfg["graph_timesteps"]),
                            float(model_cfg["task_tmin"]),
                            float(model_cfg["task_tmax"]),
                        )
                        arrays, summary = _fit_prior(
                            injected,
                            model_cfg,
                            seed=(fold_seed + int(round(100 * strength)) + int(round(delay_ms))),
                            device=args.device,
                            bootstraps=min(64, int(args.bootstrap_samples)),
                        )
                        metrics = _known_route_metrics(
                            arrays, source_node, target_node, delay_steps
                        )
                        error_steps = float(metrics["known_delay_absolute_error"])
                        row = {
                            "subject": subject,
                            "session": "S1",
                            "evidence_space": space,
                            "injection_seed": int(injection_seed),
                            "inner_train_fold_seed": fold_seed,
                            "strength": float(strength),
                            "true_delay_ms": float(delay_ms),
                            "true_delay_graph_steps": delay_steps,
                            "graph_step_ms": graph_step_ms,
                            "source_node": source_node,
                            "target_node": target_node,
                            "known_delay_absolute_error_ms": (
                                error_steps * graph_step_ms
                                if np.isfinite(error_steps)
                                else float("nan")
                            ),
                            "accepted_edges": int(summary["accepted_evidence_edges"]),
                            **metrics,
                        }
                        injection_rows.append(row)
                        completed_injection.add(
                            (
                                subject,
                                space,
                                int(injection_seed),
                                float(strength),
                                float(delay_ms),
                            )
                        )
                        _checkpoint(output, real_rows, injection_rows)
                        print(
                            json.dumps({"event": "injection_completed", **row}),
                            flush=True,
                        )
                        del injected, arrays
                        torch.cuda.empty_cache()
                    del standardized

        del data, cropped
        gc.collect()
        torch.cuda.empty_cache()

    decisions: dict[str, Any] = {
        "source_fingerprint": source_fingerprint,
        "heldout_session_S2_accessed": False,
        "classifier_training": False,
    }
    stage_passes: list[bool] = []
    if args.stage in {"real", "both"}:
        passing_subjects_by_space = {
            space: sorted(
                subject
                for subject in subjects
                if any(
                    bool(row["subject_gate_passed"])
                    for row in real_rows
                    if int(row["subject"]) == subject and str(row["evidence_space"]) == space
                )
            )
            for space in spaces
        }
        passing_spaces = sorted(
            space
            for space, passing_subjects in passing_subjects_by_space.items()
            if len(passing_subjects) >= int(gate["minimum_passing_development_subjects"])
        )
        real_passed = bool(passing_spaces)
        decisions["real_evidence"] = {
            "passed": real_passed,
            "passing_evidence_spaces": passing_spaces,
            "passing_subjects_by_evidence_space": passing_subjects_by_space,
            "passing_subject_count_by_evidence_space": {
                space: len(passing_subjects)
                for space, passing_subjects in passing_subjects_by_space.items()
            },
            "required_subject_count": int(gate["minimum_passing_development_subjects"]),
            "selection_rule": "one_fixed_evidence_space_must_pass_across_subjects",
        }
        stage_passes.append(real_passed)
    if args.stage in {"injection", "both"}:
        evaluated = [
            row
            for row in injection_rows
            if float(row["strength"]) >= 0.25 and float(row["true_delay_ms"]) >= 4.0
        ]
        direction_accuracy = float(
            np.mean([bool(row["known_direction_detected"]) for row in evaluated])
        )
        errors = [
            float(row["known_delay_absolute_error_ms"])
            for row in evaluated
            if bool(row["known_direction_detected"])
            and np.isfinite(float(row["known_delay_absolute_error_ms"]))
        ]
        median_mae = float(np.median(errors)) if errors else float("inf")
        injection_passed = direction_accuracy >= float(
            power["minimum_direction_accuracy"]
        ) and median_mae <= float(power["maximum_median_delay_mae_ms"])
        decisions["real_background_injection"] = {
            "passed": injection_passed,
            "evaluated_conditions": len(evaluated),
            "direction_accuracy": direction_accuracy,
            "median_delay_mae_ms": median_mae,
        }
        stage_passes.append(injection_passed)
    decisions["status"] = "passed" if stage_passes and all(stage_passes) else "failed"
    write_json(output / "stage_decision.json", decisions)


if __name__ == "__main__":
    main()
