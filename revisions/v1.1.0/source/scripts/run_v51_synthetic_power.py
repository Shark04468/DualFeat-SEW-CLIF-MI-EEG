#!/usr/bin/env python
"""Run the preregistered V5.1 classifier-free synthetic delay power matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.analysis.continuous_delay import (  # noqa: E402
    ContinuousDelayConfig,
    fit_continuous_delay_evidence,
    trial_continuous_phase_scores,
)
from dpc_snn.analysis.evidence_space import (  # noqa: E402
    complex_phase_surrogate,
    trial_psi_imaginary_support,
)
from dpc_snn.analysis.mechanism_power import (  # noqa: E402
    SyntheticMechanismSpec,
    generate_common_source_analytic,
    generate_coupled_analytic,
)
from dpc_snn.config import load_yaml  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from run_v40_delay_gate import _source_fingerprint  # noqa: E402


def _parse_csv(values: str, cast: Any) -> list[Any]:
    return [cast(value.strip()) for value in values.split(",") if value.strip()]


def _fit(
    analytic: np.ndarray,
    frequencies: np.ndarray,
    timestep_seconds: float,
    config: ContinuousDelayConfig,
    null_seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    surrogate = complex_phase_surrogate(analytic, null_seed)
    score, grid, _ = trial_continuous_phase_scores(
        analytic,
        frequencies,
        timestep_seconds=timestep_seconds,
        config=config,
    )
    null_score, null_grid, _ = trial_continuous_phase_scores(
        surrogate,
        frequencies,
        timestep_seconds=timestep_seconds,
        config=config,
    )
    if not np.array_equal(grid, null_grid):
        raise RuntimeError("Observed and surrogate delay grids diverged")
    support, _, _ = trial_psi_imaginary_support(analytic)
    null_support, _, _ = trial_psi_imaginary_support(surrogate)
    return (
        fit_continuous_delay_evidence(
            score,
            null_score,
            grid,
            config,
            direction_support=support,
            null_direction_support=null_support,
        ),
        surrogate,
    )


def _route_metrics(
    fit: dict[str, np.ndarray], source: int, target: int, true_delay_steps: float
) -> dict[str, Any]:
    estimate = float(fit["signed_delay_map"][target, source])
    return {
        "known_route_accepted": bool(fit["accepted"][target, source]),
        "reverse_route_accepted": bool(fit["accepted"][source, target]),
        "known_delay_estimate_steps": estimate,
        "known_delay_absolute_error_steps": abs(estimate - float(true_delay_steps)),
        "known_direction_probability": float(fit["direction_probability"][target, source]),
        "known_route_bayes_factor": float(fit["bayes_factor"][target, source]),
        "known_positive_delay_bayes_factor": float(
            fit["positive_delay_bayes_factor"][target, source]
        ),
        "known_psi_bayes_factor": float(fit["psi_imaginary_bayes_factor"][target, source]),
        "known_bootstrap_frequency": float(fit["bootstrap_frequency"][target, source]),
        "known_positive_bootstrap_frequency": float(
            fit["positive_delay_bootstrap_frequency"][target, source]
        ),
        "known_direction_bootstrap_frequency": float(
            fit["direction_bootstrap_frequency"][target, source]
        ),
        "known_psi_bootstrap_frequency": float(
            fit["psi_imaginary_bootstrap_frequency"][target, source]
        ),
        "accepted_route_count": int(np.asarray(fit["accepted"], dtype=bool).sum()),
    }


def _resume_rows(output: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_path = output / "manifest.json"
    partial_path = output / "partial_rows.json"
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
        return []
    previous = read_json(manifest_path)
    if previous != manifest:
        raise RuntimeError("Refusing stale synthetic resume: resolved manifest changed")
    if not partial_path.exists():
        return []
    payload = read_json(partial_path)
    return list(payload.get("rows", []))


def _checkpoint(output: Path, rows: list[dict[str, Any]]) -> None:
    write_json(output / "partial_rows.json", {"rows": rows})
    write_csv(output / "synthetic_power_raw.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol-config", default="configs/experiments/v51_hardware_free_plan.yaml"
    )
    parser.add_argument("--base-config", default="configs/experiments/all_experiments.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=0)
    parser.add_argument("--delays-ms", default="")
    parser.add_argument("--strengths", default="")
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    args = parser.parse_args()

    output = ensure_dir(args.output)
    protocol_path = (ROOT / args.protocol_config).resolve()
    base_config_path = (ROOT / args.base_config).resolve()
    protocol = load_yaml(protocol_path)
    power = protocol["mechanism_power"]
    model = protocol["model_overrides"]
    frequencies = tuple(float(np.mean(band)) for band in protocol["transport_band_edges_hz"])
    repetitions = int(args.repetitions or power["synthetic_repetitions"])
    delays_ms = (
        _parse_csv(args.delays_ms, float)
        if args.delays_ms
        else [float(value) for value in power["delay_milliseconds"]]
    )
    strengths = (
        _parse_csv(args.strengths, float)
        if args.strengths
        else [float(value) for value in power["injection_strengths"]]
    )
    graph_step_seconds = 1.0 / float(model["graph_rate_hz"])
    spec = SyntheticMechanismSpec(
        frequencies_hz=frequencies,
        trials=int(power["synthetic_trials"]),
        nodes=int(power["synthetic_nodes"]),
        time_steps=int(power["synthetic_time_steps"]),
        timestep_seconds=graph_step_seconds,
        ar_coefficient=float(power["synthetic_ar_coefficient"]),
    )
    bootstrap_samples = int(args.bootstrap_samples or power["synthetic_bootstrap_samples"])
    fit_template = {
        "max_delay_steps": int(round(max(delays_ms) / (1000.0 * graph_step_seconds))),
        "grid_oversample": int(model["continuous_delay_grid_oversample"]),
        "bootstrap_samples": bootstrap_samples,
        "min_bayes_factor": float(model["continuous_delay_min_bayes_factor"]),
        "min_bootstrap_frequency": float(model["continuous_delay_min_bootstrap_frequency"]),
        "min_direction_probability": float(model["continuous_delay_min_direction_probability"]),
    }
    manifest = {
        "experiment_id": protocol["experiment_id"],
        "source_fingerprint": _source_fingerprint(protocol_path, base_config_path),
        "classifier_training": False,
        "heldout_session_S2_accessed": False,
        "repetitions": repetitions,
        "delays_ms": delays_ms,
        "strengths": strengths,
        "spec": spec.__dict__,
        "fit": fit_template,
        "controls": ["independent_null", "common_source", "phase_surrogate", "time_reversal"],
    }
    rows = _resume_rows(output, manifest)
    completed = {
        (
            str(row["condition"]),
            int(row["repetition"]),
            float(row["delay_ms"]),
            float(row["strength"]),
        )
        for row in rows
    }
    source, target = 0, 1
    representative_strength = max(strengths)

    for repetition in range(repetitions):
        for strength in strengths:
            for delay_ms in delays_ms:
                key = ("coupled", repetition, delay_ms, strength)
                if key in completed:
                    continue
                delay_steps = delay_ms / (1000.0 * graph_step_seconds)
                seed = 5_100_000 + 100_000 * repetition + int(1000 * strength) + int(delay_ms)
                analytic = generate_coupled_analytic(
                    spec,
                    delay_steps=delay_steps,
                    coupling_strength=strength,
                    seed=seed,
                    source_node=source,
                    target_node=target,
                )
                fit_config = ContinuousDelayConfig(**fit_template, random_seed=seed + 1)
                fit, surrogate = _fit(
                    analytic, np.asarray(frequencies), graph_step_seconds, fit_config, seed + 2
                )
                metrics = _route_metrics(fit, source, target, delay_steps)
                row: dict[str, Any] = {
                    "condition": "coupled",
                    "repetition": repetition,
                    "delay_ms": delay_ms,
                    "delay_steps": delay_steps,
                    "strength": strength,
                    **metrics,
                    "known_delay_absolute_error_ms": metrics["known_delay_absolute_error_steps"]
                    * 1000.0
                    * graph_step_seconds,
                    "phase_surrogate_known_route_accepted": None,
                    "time_reversal_transpose_accepted": None,
                    "time_reversal_delay_error_ms": None,
                }
                if strength == representative_strength:
                    surrogate_fit, _ = _fit(
                        surrogate,
                        np.asarray(frequencies),
                        graph_step_seconds,
                        ContinuousDelayConfig(**fit_template, random_seed=seed + 3),
                        seed + 4,
                    )
                    reversed_fit, _ = _fit(
                        analytic[..., ::-1].conj().copy(),
                        np.asarray(frequencies),
                        graph_step_seconds,
                        ContinuousDelayConfig(**fit_template, random_seed=seed + 5),
                        seed + 6,
                    )
                    reversed_estimate = float(reversed_fit["signed_delay_map"][source, target])
                    row["phase_surrogate_known_route_accepted"] = bool(
                        surrogate_fit["accepted"][target, source]
                    )
                    row["time_reversal_transpose_accepted"] = bool(
                        reversed_fit["accepted"][source, target]
                    )
                    row["time_reversal_delay_error_ms"] = (
                        abs(reversed_estimate - delay_steps) * 1000.0 * graph_step_seconds
                    )
                rows.append(row)
                completed.add(key)
                _checkpoint(output, rows)
                print(json.dumps({"event": "coupled_completed", **row}), flush=True)

        for condition in ("independent_null", "common_source"):
            key = (condition, repetition, 0.0, representative_strength)
            if key in completed:
                continue
            seed = 7_100_000 + 10_000 * repetition + (0 if condition == "independent_null" else 1)
            if condition == "independent_null":
                analytic = generate_coupled_analytic(
                    spec, delay_steps=0.0, coupling_strength=0.0, seed=seed
                )
            else:
                analytic = generate_common_source_analytic(
                    spec, mixing_strength=representative_strength, seed=seed
                )
            fit, _ = _fit(
                analytic,
                np.asarray(frequencies),
                graph_step_seconds,
                ContinuousDelayConfig(**fit_template, random_seed=seed + 1),
                seed + 2,
            )
            off_diagonal = ~np.eye(spec.nodes, dtype=bool)
            row = {
                "condition": condition,
                "repetition": repetition,
                "delay_ms": 0.0,
                "delay_steps": 0.0,
                "strength": representative_strength,
                **_route_metrics(fit, source, target, 0.0),
                "known_delay_absolute_error_ms": float(
                    abs(float(fit["signed_delay_map"][target, source]))
                    * 1000.0
                    * graph_step_seconds
                ),
                "null_pair_any_direction_accepted": bool(
                    fit["accepted"][target, source] or fit["accepted"][source, target]
                ),
                "null_route_false_positive_fraction": float(
                    np.asarray(fit["accepted"], dtype=bool)[off_diagonal].mean()
                ),
                "phase_surrogate_known_route_accepted": None,
                "time_reversal_transpose_accepted": None,
                "time_reversal_delay_error_ms": None,
            }
            rows.append(row)
            completed.add(key)
            _checkpoint(output, rows)
            print(json.dumps({"event": "control_completed", **row}), flush=True)

    coupled = [row for row in rows if row["condition"] == "coupled"]
    primary = [
        row for row in coupled if float(row["strength"]) >= 0.25 and float(row["delay_ms"]) >= 4.0
    ]
    null_rows = [row for row in rows if row["condition"] == "independent_null"]
    common_rows = [row for row in rows if row["condition"] == "common_source"]
    direction_accuracy = float(np.mean([bool(row["known_route_accepted"]) for row in primary]))
    median_mae_ms = float(
        np.median([float(row["known_delay_absolute_error_ms"]) for row in primary])
    )
    null_route_fpr = float(
        np.mean([float(row["null_route_false_positive_fraction"]) for row in null_rows])
    )
    common_pair_fpr = float(
        np.mean([bool(row["null_pair_any_direction_accepted"]) for row in common_rows])
    )
    representative = [row for row in coupled if float(row["strength"]) == representative_strength]
    time_reversal_accuracy = float(
        np.mean([bool(row["time_reversal_transpose_accepted"]) for row in representative])
    )
    phase_surrogate_fpr = float(
        np.mean([bool(row["phase_surrogate_known_route_accepted"]) for row in representative])
    )
    passed = (
        direction_accuracy >= float(power["minimum_direction_accuracy"])
        and median_mae_ms <= float(power["maximum_median_delay_mae_ms"])
        and null_route_fpr <= float(power["maximum_null_false_positive_rate"])
        and common_pair_fpr <= float(power["maximum_null_false_positive_rate"])
    )
    summary = {
        "status": "passed" if passed else "failed",
        "classifier_training": False,
        "heldout_session_S2_accessed": False,
        "primary_condition_count": len(primary),
        "direction_accuracy": direction_accuracy,
        "median_delay_mae_ms": median_mae_ms,
        "null_route_false_positive_rate": null_route_fpr,
        "common_source_pair_false_positive_rate": common_pair_fpr,
        "phase_surrogate_false_positive_rate": phase_surrogate_fpr,
        "time_reversal_transpose_accuracy": time_reversal_accuracy,
        "thresholds": {
            "minimum_direction_accuracy": power["minimum_direction_accuracy"],
            "maximum_median_delay_mae_ms": power["maximum_median_delay_mae_ms"],
            "maximum_null_false_positive_rate": power["maximum_null_false_positive_rate"],
        },
    }
    write_json(output / "stage_decision.json", summary)
    write_csv(output / "synthetic_power_raw.csv", rows)
    print(json.dumps({"event": "stage_completed", **summary}), flush=True)


if __name__ == "__main__":
    main()
