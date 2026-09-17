#!/usr/bin/env python3
"""Run and materialize the DASP-SNN V6.2-R1 E0 invariant gate."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    build_run_fingerprint,
    file_sha256,
    write_fingerprint_manifest,
    write_run_artifact_manifest,
    write_trial_predictions,
)
from dpc_snn.models.coupled_dual_delay import (  # noqa: E402
    CoupledDualDelay,
    fractional_causal_shift,
)
from dpc_snn.models.dasp_snn_v62 import DASPSNNV62  # noqa: E402
from dpc_snn.models.v62_filterbank import (  # noqa: E402
    CausalAnalyticFilterBank,
    DualRateCausalResampler,
    causal_linear_upsample_2x,
)
from dpc_snn.utils.io import ensure_dir, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _environment() -> dict[str, object]:
    packages = {}
    for name in ("numpy", "torch", "scipy", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "storage_root": os.environ.get("DPC_SNN_STORAGE_ROOT"),
    }


def _source_hashes() -> dict[str, str]:
    paths = [
        "src/dpc_snn/models/v62_filterbank.py",
        "src/dpc_snn/models/v62_spatial.py",
        "src/dpc_snn/models/coupled_dual_delay.py",
        "src/dpc_snn/models/delayed_temporal_pyramid.py",
        "src/dpc_snn/models/delayed_geometry_sketch.py",
        "src/dpc_snn/models/v62_snn_decoder.py",
        "src/dpc_snn/models/dasp_snn_v62.py",
        "configs/models/dasp_snn_v62.yaml",
        "configs/experiments/v62_e0_invariants.yaml",
        "scripts/run_v62_e0_invariants.py",
    ]
    return {name: file_sha256(ROOT / name) for name in paths}


def _filter_and_timestamp_checks(tolerances: dict[str, float]) -> dict[str, float]:
    bank = CausalAnalyticFilterBank()
    future_a = torch.zeros(1, 1, 500)
    future_b = future_a.clone()
    future_b[..., 300] = 1.0
    causal_error = float((bank(future_a)[..., :300] - bank(future_b)[..., :300]).abs().max())
    _assert(causal_error == 0.0, "causal analytic filter reads a future sample")

    response = bank.frequency_response(torch.tensor([40.0, -40.0]))[-1]
    positive_gain = float(response[0].abs())
    negative_gain = float(response[1].abs())
    _assert(positive_gain > 1.4, "40 Hz is excessively attenuated before 125 Hz sampling")
    _assert(negative_gain < 0.1, "analytic filter did not reject the negative-frequency image")
    _assert(bank.support_seconds <= 1.0, "analytic FIR support exceeds the pre-cue history")

    time_axis = torch.arange(1250, dtype=torch.float32) / 250.0 - 1.0
    sine = torch.sin(2.0 * math.pi * 40.0 * time_axis).view(1, 1, -1)
    analytic = bank(sine)[:, -1, 0]
    steady = analytic[..., 500:1200:2]
    spectrum = torch.fft.rfft(steady.real)
    frequencies = torch.fft.rfftfreq(steady.shape[-1], d=1.0 / 125.0)
    peak_hz = float(frequencies[spectrum.abs().argmax()])
    amplitude = float(steady.abs().median())
    _assert(
        abs(peak_hz - 40.0) <= float(tolerances["forty_hz_peak_error_hz"]),
        f"40 Hz moved to {peak_hz:.3f} Hz after fast-path sampling",
    )
    _assert(
        float(tolerances["forty_hz_analytic_amplitude_min"])
        <= amplitude
        <= float(tolerances["forty_hz_analytic_amplitude_max"]),
        f"40 Hz analytic amplitude {amplitude:.4f} is outside tolerance",
    )

    resampler = DualRateCausalResampler(
        1,
        1,
        analytic_group_delay_samples=bank.group_delay_samples,
        envelope_taps=33,
    )
    impulse = torch.full((1, 1, 1, 1250), 1e-5, dtype=torch.complex64)
    impulse[..., 600] = 1.0 + 0j
    rates = resampler(impulse)
    upsampled = causal_linear_upsample_2x(rates.slow)
    fast_peak = int(rates.fast.abs().argmax())
    slow_peak = int(upsampled.argmax())
    timestamp_peak_error = abs(fast_peak - slow_peak)
    _assert(
        timestamp_peak_error <= int(tolerances["timestamp_peak_error_fast_samples"]),
        "fast and slow impulse timestamps are not aligned",
    )
    _assert(
        bool(torch.allclose(rates.fast_timestamps[::2], rates.slow_timestamps)),
        "fast/slow effective timestamps disagree",
    )
    return {
        "analytic_future_prefix_error": causal_error,
        "forty_hz_positive_gain": positive_gain,
        "forty_hz_negative_gain": negative_gain,
        "forty_hz_peak_hz": peak_hz,
        "forty_hz_analytic_amplitude": amplitude,
        "filter_support_seconds": bank.support_seconds,
        "fast_slow_peak_error_samples": timestamp_peak_error,
        "total_frontend_group_delay_seconds": resampler.total_group_delay_seconds,
    }


def _delay_checks(tolerances: dict[str, float]) -> dict[str, float]:
    values = torch.arange(12, dtype=torch.float64).view(1, 1, -1)
    maximum_error = 0.0
    minimum_gradient = math.inf
    for fraction in (0.25, 0.5, 0.75):
        delay = torch.tensor([fraction], dtype=torch.float64, requires_grad=True)
        shifted = fractional_causal_shift(values, delay, max_delay=4)
        previous = torch.cat((torch.zeros_like(values[..., :1]), values[..., :-1]), dim=-1)
        expected = (1.0 - fraction) * values + fraction * previous
        maximum_error = max(maximum_error, float((shifted - expected).abs().max()))
        shifted.sum().backward()
        minimum_gradient = min(minimum_gradient, float(delay.grad.abs().min()))
    _assert(
        maximum_error <= float(tolerances["fractional_delay_atol"]),
        "fractional delay values are incorrect",
    )
    _assert(minimum_gradient > 0.0 and math.isfinite(minimum_gradient), "delay gradient vanished")

    torch.manual_seed(7)
    delay_module = CoupledDualDelay(
        n_bands=3,
        n_nodes=3,
        route_rank=3,
        delay_rank=2,
        fast_max_delay=2,
        slow_max_delay=4,
    )
    parameters = delay_module._route_parameters()
    coupled_rank = int(torch.linalg.matrix_rank(parameters["slow_weight"].reshape(9, 9)))
    _assert(coupled_rank > 1, "rank-R coupled routes collapsed to one separable product")
    slow = torch.randn(2, 3, 3, 20)
    fast = torch.complex(torch.randn(2, 3, 3, 40), torch.randn(2, 3, 3, 40))
    learned = delay_module(fast, slow, slow_target_reference=slow.flip(-1))
    shuffled = delay_module(fast, slow, slow_target_reference=slow.flip(-1).roll(1, 0))
    target_sensitivity = float((learned.slow_current - shuffled.slow_current).abs().mean())
    _assert(target_sensitivity > 0.0, "target reference has no interaction semantics")
    zero = delay_module(
        torch.zeros_like(fast),
        torch.zeros_like(slow),
        fast_target_reference=fast,
        slow_target_reference=slow,
    )
    _assert(torch.count_nonzero(zero.fused_current) == 0, "target reference formed a bypass")
    locked = delay_module(fast, slow, delay_override="zero")
    for field in ("slow_route_weight", "fast_route_weight", "slow_gate", "fast_gate"):
        _assert(
            torch.equal(getattr(learned, field), getattr(locked, field)),
            f"matched-zero changed {field}",
        )
    return {
        "fractional_delay_max_error": maximum_error,
        "fractional_delay_min_abs_gradient": minimum_gradient,
        "coupled_route_matrix_rank": coupled_rank,
        "target_shuffle_mean_abs_change": target_sensitivity,
        "max_delay_intermediate_elements_small_probe": delay_module.last_max_intermediate_elements,
    }


def _model_checks(
    device: torch.device,
    batch_size: int,
    tolerances: dict[str, float],
) -> tuple[dict[str, float | int | bool], DASPSNNV62, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    model = DASPSNNV62(dropout=0.0).to(device).train()
    count = model.parameter_count
    _assert(count <= int(tolerances["maximum_parameters"]), "parameter ceiling exceeded")
    _assert(
        int(tolerances["target_parameter_min"])
        <= count
        <= int(tolerances["target_parameter_max"]),
        "default model is outside the registered capacity target",
    )
    zero = torch.zeros(1, 12, 16, 500, device=device)
    model.eval()
    with torch.no_grad():
        decoded_zero, _ = model.decode_delayed_current(zero)
    bias = model.decoder.classifier.bias[None, None].expand_as(decoded_zero.logits)
    bottleneck_error = float((decoded_zero.logits - bias).abs().max())
    _assert(bottleneck_error == 0.0, "zero delayed current does not reduce to classifier bias")

    causal_input = torch.randn(1, 22, 1250, device=device) * 1e-5
    future = causal_input.clone()
    future[..., 500:] += torch.randn_like(future[..., 500:])
    with torch.no_grad():
        prefix_a = model(causal_input)["prefix_logits"][:, 0]
        prefix_b = model(future)["prefix_logits"][:, 0]
    prefix_error = float((prefix_a - prefix_b).abs().max())
    _assert(
        prefix_error <= float(tolerances["future_prefix_atol"]),
        "future task samples changed the 1-second prediction",
    )

    model.train()
    x = (torch.randn(batch_size, 22, 1250, device=device) * 1e-5).requires_grad_(True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    output = model(x)
    loss = output["logits"].square().mean() + 1e-3 * output["firing_rate_loss"]
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    finite_output = bool(torch.isfinite(output["logits"]).all())
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    _assert(finite_output and finite_gradients, "full-shape forward/backward is non-finite")
    for spikes in output["aux"]["binary_spikes"]:
        _assert(bool(((spikes == 0) | (spikes == 1)).all()), "non-binary tensor counted as spikes")
    peak_mib = (
        torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
    )
    if device.type == "cuda":
        _assert(
            peak_mib <= float(tolerances["maximum_cuda_peak_allocated_mib"]),
            "CUDA activation budget exceeded",
        )
    metrics: dict[str, float | int | bool] = {
        f"{device.type}_batch_size": batch_size,
        f"{device.type}_elapsed_seconds": elapsed,
        f"{device.type}_peak_allocated_mib": peak_mib,
        f"{device.type}_finite_output": finite_output,
        f"{device.type}_finite_gradients": finite_gradients,
        f"{device.type}_max_delay_intermediate_elements": model.delay.last_max_intermediate_elements,
        "parameter_count": count,
        "mandatory_bottleneck_max_error": bottleneck_error,
        "one_second_future_prefix_max_error": prefix_error,
    }
    return metrics, model, output["logits"].detach(), output["prefix_logits"].detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/v62_e0_invariants.yaml")
    parser.add_argument("--output", default="runs/v62/E0_invariants")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    configure_cache_env()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tolerances = config["tolerances"]
    output_dir = ensure_dir(args.output)
    started_at = time.time()

    metrics: dict[str, object] = {"status": "running"}
    metrics.update(_filter_and_timestamp_checks(tolerances))
    metrics.update(_delay_checks(tolerances))
    cpu_metrics, _, _, _ = _model_checks(
        torch.device("cpu"), int(config["cpu_smoke_batch"]), tolerances
    )
    metrics.update(cpu_metrics)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("E0 was configured to require CUDA, but CUDA is unavailable")
    if torch.cuda.is_available():
        cuda_metrics, model, logits, prefix_logits = _model_checks(
            torch.device("cuda"), int(config["cuda_smoke_batch"]), tolerances
        )
        metrics.update(cuda_metrics)
    else:
        model = DASPSNNV62(dropout=0.0)
        logits = torch.zeros(1, 4)
        metrics["cuda_skipped"] = True
    metrics["status"] = "passed"
    metrics["elapsed_seconds"] = time.time() - started_at

    environment = _environment()
    source_hashes = _source_hashes()
    fingerprint = build_run_fingerprint(
        resolved_config=config,
        source=source_hashes,
        data={"kind": "synthetic_e0", "seed": config["seed"]},
        split={"kind": "none", "synthetic": True},
        augmentation={"policy": "none"},
        prior={"policy": "none"},
        checkpoint={"policy": "fresh_e0_state"},
        environment=environment,
    )
    shutil.copyfile(config_path, output_dir / "resolved_config.yaml")
    write_fingerprint_manifest(output_dir / "source_fingerprint.json", fingerprint)
    write_json(output_dir / "split_manifest.json", {"kind": "synthetic", "seed": config["seed"]})
    write_json(output_dir / "augmentation_manifest.json", {"policy": "none", "parents": []})
    write_json(output_dir / "metrics.json", metrics)
    write_csv(output_dir / "history.csv", [{"stage": "E0", **metrics}])
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state, output_dir / "best.pt")
    torch.save(state, output_dir / "last.pt")
    probability = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    logits_array = logits.float().cpu().numpy()
    write_trial_predictions(
        output_dir,
        logits=logits_array,
        probabilities=probability,
        pred=probability.argmax(axis=1),
        label=np.zeros(logits_array.shape[0], dtype=np.int64),
        subject="synthetic",
        session="T",
        run="e0",
        trial_id=[f"e0:{index}" for index in range(logits_array.shape[0])],
        seed=int(config["seed"]),
        model="dasp_snn_v62_e0_synthetic_probe",
    )
    write_json(
        output_dir / "runtime_status.json",
        {"status": "completed", "started_at": started_at, "completed_at": time.time()},
    )
    (output_dir / "stdout.log").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output_dir / "stderr.log").write_text("", encoding="utf-8")
    write_run_artifact_manifest(
        output_dir,
        required_files=(
            "manifest.json",
            "resolved_config.yaml",
            "source_fingerprint.json",
            "split_manifest.json",
            "augmentation_manifest.json",
            "history.csv",
            "predictions.npz",
            "predictions.csv",
            "metrics.json",
            "best.pt",
            "last.pt",
            "runtime_status.json",
            "stdout.log",
            "stderr.log",
        ),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
