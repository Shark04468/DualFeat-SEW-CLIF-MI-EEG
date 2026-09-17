#!/usr/bin/env python3
"""Embed verified E1 ATC checkpoints in the mandatory-delay E2 scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shutil
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
    validate_trial_metadata,
    write_fingerprint_manifest,
    write_run_artifact_manifest,
    write_trial_predictions,
)
from dpc_snn.experiments.v62_scaffold import (  # noqa: E402
    build_zero_scaffold,
    cache_analytic_fixed_channel_gain_rate_features,
    fit_official_fbc_channel_gain,
    predict_scaffold,
)
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402


REQUIRED_FILES = (
    "manifest.json",
    "resolved_config.yaml",
    "source_fingerprint.json",
    "split_manifest.json",
    "augmentation_manifest.json",
    "frontend_manifest.json",
    "history.csv",
    "predictions.npz",
    "predictions.csv",
    "metrics.json",
    "best.pt",
    "last.pt",
    "runtime_status.json",
    "stdout.log",
    "stderr.log",
)


def _parse_csv(value: str, cast: Any = str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _source_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path)
        for base in (ROOT / "src", ROOT / "scripts", ROOT / "configs")
        for path in sorted(base.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".py", ".yaml", ".yml", ".toml"}
    }


def _subject_file(data_root: Path, subject: int) -> Path:
    candidates = sorted(data_root.glob(f"*A{subject:02d}*.npz"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one BCI2a file for subject {subject}")
    return candidates[0]


def _metadata(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
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
        for index in range(len(data["y"]))
    ]
    validate_trial_metadata(rows)
    return rows


def _probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _gate(rows: list[dict[str, Any]], baseline_summary: Path, gap_pp: float) -> dict[str, Any]:
    baseline_rows: list[dict[str, str]] = []
    import csv

    with baseline_summary.open("r", encoding="utf-8", newline="") as handle:
        baseline_rows.extend(csv.DictReader(handle))
    means: dict[str, float] = {}
    for model in sorted({row["model"] for row in baseline_rows}):
        selected = [float(row["accuracy"]) for row in baseline_rows if row["model"] == model]
        means[model] = float(np.mean(selected))
    strongest = max(means, key=means.get)
    scaffold = float(np.mean([float(row["accuracy"]) for row in rows]))
    gap = 100.0 * (means[strongest] - scaffold)
    return {
        "passed": bool(gap <= float(gap_pp)),
        "strongest_baseline_model": strongest,
        "strongest_baseline_accuracy": means[strongest],
        "zero_scaffold_accuracy": scaffold,
        "gap_pp": gap,
        "maximum_gap_pp": float(gap_pp),
        "checkpoint_source": "verified_E1_ATC_core_embedded_in_mandatory_delay_shell",
        "next_stage_allowed": bool(gap <= float(gap_pp)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--selection-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default="configs/experiments/v7_e2_zero_scaffold_core_atc.yaml"
    )
    parser.add_argument("--subjects", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    configure_cache_env()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    subjects = _parse_csv(args.subjects, int) if args.subjects else list(config["subjects"])
    seeds = _parse_csv(args.seeds, int) if args.seeds else list(config["seeds"])
    data_root = Path(args.data).resolve()
    baseline_root = Path(args.baseline_root).resolve()
    selection_root = Path(args.selection_root).resolve()
    output = ensure_dir(args.output)
    source_root = str(Path(args.source_root).resolve())
    source = _source_hashes()
    source["official_source_locks"] = verify_official_source_locks(source_root)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    rows: list[dict[str, Any]] = []

    for subject in subjects:
        selection_summary_path = (
            selection_root / f"subject_{subject:02d}" / "selection" / "selection_summary.json"
        )
        if not selection_summary_path.is_file():
            raise FileNotFoundError(f"missing completed E2 selection for subject {subject}")
        selection_summary = read_json(selection_summary_path)
        if bool(selection_summary.get("session_e_accessed", False)):
            raise RuntimeError("E2 selection artifact accessed Session E")
        subject_path = _subject_file(data_root, subject)
        data = load_processed_npz(subject_path)
        metadata = _metadata(data)
        session = np.asarray(data["session"]).astype(str)
        t_indices = np.flatnonzero(session == "T")
        e_indices = np.flatnonzero(session == "E")
        if len(t_indices) != 288 or len(e_indices) != 288:
            raise RuntimeError("BCI2a T/E protocol requires 288 trials per session")
        x_t = np.asarray(data["X"])[t_indices]
        x_e = np.asarray(data["X"])[e_indices]
        y_e = np.asarray(data["y"], dtype=np.int64)[e_indices]
        metadata_e = [metadata[int(index)] for index in e_indices]
        gain = fit_official_fbc_channel_gain(
            x_t,
            sfreq=float(data["sfreq"]),
            epoch_tmin=float(data["epoch_tmin"]),
            clip=float(config["preprocessing"]["clip_after_gain"]),
        )

        for seed in seeds:
            baseline_dir = baseline_root / "atcnet" / f"subject_{subject:02d}" / f"seed_{seed}"
            baseline_checkpoint = baseline_dir / "best.pt"
            baseline_metrics = read_json(baseline_dir / "metrics.json")
            run_dir = ensure_dir(output / f"subject_{subject:02d}" / f"seed_{seed}")
            model = build_zero_scaffold(
                seed=seed,
                model_config=dict(config["model"]),
                official_source_root=source_root,
            )
            core_state = torch.load(baseline_checkpoint, map_location="cpu", weights_only=True)
            model.atc_readout.core.load_state_dict(core_state, strict=True)
            rates = cache_analytic_fixed_channel_gain_rate_features(
                model,
                x_e,
                gain,
                device=args.device,
                batch_size=int(config["training"]["batch_size"]),
            )
            evaluation = predict_scaffold(
                model.to(args.device),
                rates,
                y_e,
                device=args.device,
                batch_size=int(config["training"]["batch_size"]),
                delay_override="zero",
            )
            with np.load(baseline_dir / "predictions.npz", allow_pickle=False) as archive:
                old_logits = np.asarray(archive["logits"])
                old_pred = np.asarray(archive["pred"])
            logit_error = float(np.max(np.abs(evaluation["logits"] - old_logits)))
            prediction_disagreements = int(np.count_nonzero(evaluation["pred"] != old_pred))
            if prediction_disagreements != 0 or logit_error > 1e-4:
                raise RuntimeError(
                    f"mandatory-delay embedding changed ATC predictions: "
                    f"disagreements={prediction_disagreements}, max_error={logit_error}"
                )
            split = read_json(baseline_dir / "split_manifest.json")
            resolved = {
                **config,
                "active_subject": subject,
                "active_seed": seed,
                "selected_final_epoch": int(baseline_metrics["selected_final_epoch"]),
                "materialization": "verified_E1_ATC_core_checkpoint",
            }
            fingerprint = build_run_fingerprint(
                resolved_config=resolved,
                source=source,
                data={subject_path.name: file_sha256(subject_path)},
                split=split,
                augmentation=config["augmentation"],
                prior={"policy": "zero_delay_identity_backbone"},
                checkpoint={
                    "baseline_checkpoint": str(baseline_checkpoint),
                    "baseline_checkpoint_sha256": file_sha256(baseline_checkpoint),
                    "selection_summary_sha256": file_sha256(selection_summary_path),
                },
                environment=environment,
            )
            metrics = {
                "status": "completed",
                "stage": "E2",
                "model": "dasp_snn_v7_zero_ann",
                "implementation_id": "mandatory_delay_shell_with_verified_official_atc_core",
                "subject": subject,
                "seed": seed,
                "accuracy": evaluation["accuracy"],
                "balanced_accuracy": evaluation["balanced_accuracy"],
                "kappa": evaluation["kappa"],
                "macro_f1": evaluation["macro_f1"],
                "selected_final_epoch": int(baseline_metrics["selected_final_epoch"]),
                "parameters": model.parameter_count,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "selection_session": "T",
                "evaluation_session": "E",
                "heldout_e_selected_checkpoint": False,
                "delay_override": "zero",
                "mandatory_delay_bottleneck": True,
                "source_atc_accuracy": float(baseline_metrics["accuracy"]),
                "source_atc_logit_max_error": logit_error,
                "source_atc_prediction_disagreements": prediction_disagreements,
                "run_fingerprint": fingerprint["combined_sha256"],
            }
            write_fingerprint_manifest(run_dir / "source_fingerprint.json", fingerprint)
            write_json(run_dir / "split_manifest.json", split)
            write_json(run_dir / "augmentation_manifest.json", config["augmentation"])
            write_json(
                run_dir / "frontend_manifest.json",
                {
                    "frontend": config["preprocessing"]["frontend"],
                    "gain": gain.values.reshape(-1).tolist(),
                    "fit_session": "T",
                    "mandatory_delay": True,
                },
            )
            (run_dir / "resolved_config.yaml").write_text(
                yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
            )
            shutil.copy2(baseline_dir / "history.csv", run_dir / "history.csv")
            full_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(full_state, run_dir / "best.pt")
            torch.save(full_state, run_dir / "last.pt")
            write_json(run_dir / "metrics.json", metrics)
            write_trial_predictions(
                run_dir,
                logits=evaluation["logits"],
                probabilities=_probabilities(evaluation["logits"]),
                pred=evaluation["pred"],
                label=evaluation["labels"],
                subject=[row["subject"] for row in metadata_e],
                session=[row["session"] for row in metadata_e],
                run=[row["run"] for row in metadata_e],
                trial_id=[row["trial_id"] for row in metadata_e],
                seed=seed,
                model="dasp_snn_v7_zero_ann",
            )
            write_json(run_dir / "runtime_status.json", {"status": "completed", "completed_at": time.time()})
            (run_dir / "stdout.log").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            (run_dir / "stderr.log").write_text("", encoding="utf-8")
            write_run_artifact_manifest(run_dir, required_files=REQUIRED_FILES)
            rows.append(metrics)
            write_csv(output / "summary.csv", rows)
            print(
                f"__V7_E2_EMBED_DONE__ subject={subject} seed={seed} "
                f"accuracy={metrics['accuracy']:.6f} max_logit_error={logit_error:.3e}",
                flush=True,
            )

    gate = _gate(rows, Path(args.baseline_summary).resolve(), float(config["gate"]["maximum_gap_pp"]))
    write_csv(output / "summary.csv", rows)
    write_json(output / "gate_report.json", gate)
    write_json(
        output / "campaign_status.json",
        {"status": "completed", "subjects": subjects, "seeds": seeds, "gate": gate},
    )
    print(json.dumps({"status": "completed", "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
