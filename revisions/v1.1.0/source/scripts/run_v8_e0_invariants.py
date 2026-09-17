#!/usr/bin/env python3
"""Run and materialize the V8 E0 engineering-invariant gate."""

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
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    validate_run_artifact_manifest,
    write_run_artifact_manifest,
    write_trial_predictions,
)
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    assert_delay_control_contract,
    build_v8_run_fingerprint,
    collect_source_tree_manifest,
    mapping_sha256,
    validate_v8_resume_fingerprint,
    write_v8_fingerprint,
)
from dpc_snn.models.build import build_model  # noqa: E402
from dpc_snn.models.v8_accuracy_first import V8AccuracyFirstModel  # noqa: E402
from dpc_snn.utils.io import ensure_dir, read_json, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402


E0_REQUIRED_FILES = (
    "manifest.json",
    "resolved_config.yaml",
    "source_fingerprint.json",
    "p0_reference.json",
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
P0_REQUIRED_FILES = (
    "manifest.json",
    "resolved_config.yaml",
    "source_fingerprint.json",
    "source_tree_manifest.json",
    "protocol_manifest.json",
    "heldout_lock.json",
    "metrics.json",
    "history.csv",
    "runtime_status.json",
    "stdout.log",
    "stderr.log",
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _environment(storage_root: Path | None) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
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
        "storage_root": str(storage_root) if storage_root is not None else None,
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def _default_output(storage_root: Path | None) -> Path:
    base = storage_root if storage_root is not None else ROOT
    return base / "runs" / "v8_accuracy_first" / "E0_invariants"


def _default_p0(storage_root: Path | None) -> Path:
    base = storage_root if storage_root is not None else ROOT
    return base / "runs" / "v8_accuracy_first" / "P0_protocol"


def _model_from_config(
    model_config: dict[str, Any],
    *,
    decoder_kind: str,
    compact: dict[str, Any] | None = None,
) -> V8AccuracyFirstModel:
    resolved = dict(model_config)
    name = str(resolved.pop("name"))
    resolved["decoder_kind"] = decoder_kind
    resolved["dropout"] = 0.0
    resolved["delay_auxiliary_enabled"] = True
    if compact:
        resolved.update(compact)
    model = build_model(name, resolved)
    if not isinstance(model, V8AccuracyFirstModel):
        raise TypeError("V8 E0 model factory returned the wrong architecture")
    model.set_training_gain(torch.full((model.n_bands, model.n_channels), 1.0e5))
    return model


def _load_synthetic_prior(model: V8AccuracyFirstModel) -> str:
    if model.delay_auxiliary is None:
        raise RuntimeError("E0 requires the delay auxiliary")
    maximum_delay = model.delay_auxiliary.maximum_delay
    count = min(4, model.n_bands)
    source_band = torch.arange(count)
    source_node = torch.tensor([7, 11, 8, 10])[:count] % model.n_channels
    target_node = torch.tensor([11, 7, 10, 8])[:count] % model.n_channels
    probability = torch.zeros(count, maximum_delay + 1)
    for route in range(count):
        probability[route, min(route + 1, maximum_delay)] = 1.0
    model.load_fold_delay_prior(
        source_band=source_band,
        source_node=source_node,
        target_band=source_band.clone(),
        target_node=target_node,
        delay_probability=probability,
        fractional_target=torch.linspace(0.2, 0.8, count),
        route_weight=torch.where(source_band % 2 == 0, 1.0, -1.0),
        route_confidence=torch.linspace(0.7, 1.0, count),
        phase_preference=torch.linspace(-0.2, 0.2, count),
        amplitude_scale=torch.ones(count),
    )
    return model.delay_auxiliary.prior_fingerprint()


def _filter_checks(model: V8AccuracyFirstModel, tolerances: dict[str, float]) -> dict[str, float]:
    bank = model.filterbank
    response = bank.frequency_response(torch.tensor([40.0, -40.0]))[-1]
    positive_gain = float(response[0].abs())
    negative_gain = float(response[1].abs())
    _assert(positive_gain > 1.4, "V8 filterbank attenuates 40 Hz excessively")
    _assert(negative_gain < 0.1, "V8 analytic filter did not reject the negative image")
    _assert(bank.support_seconds <= 1.0, "V8 analytic FIR exceeds the pre-cue history")

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
        f"V8 40 Hz signal moved to {peak_hz:.3f} Hz",
    )
    _assert(
        float(tolerances["forty_hz_analytic_amplitude_min"])
        <= amplitude
        <= float(tolerances["forty_hz_analytic_amplitude_max"]),
        f"V8 40 Hz amplitude {amplitude:.4f} is outside tolerance",
    )
    return {
        "forty_hz_positive_gain": positive_gain,
        "forty_hz_negative_gain": negative_gain,
        "forty_hz_peak_hz": peak_hz,
        "forty_hz_analytic_amplitude": amplitude,
        "filter_support_seconds": bank.support_seconds,
    }


def _physical_basis_checks(model: V8AccuracyFirstModel) -> dict[str, Any]:
    expected = torch.eye(model.n_channels)[None].expand(model.n_bands, -1, -1)
    identity_error = float((model.physical_basis.weight().cpu() - expected).abs().max())
    _assert(identity_error == 0.0, "V8 delay physical basis is not exact sensor identity")
    _assert(model.physical_basis.frozen, "V8 delay physical basis is trainable")
    fingerprint = model.physical_frontend_fingerprint()
    saved = model.classifier_spatial.shared_delta.detach().clone()
    with torch.no_grad():
        model.classifier_spatial.shared_delta.add_(0.01)
    drifted_classifier_fingerprint = model.physical_frontend_fingerprint()
    with torch.no_grad():
        model.classifier_spatial.shared_delta.copy_(saved)
    _assert(
        drifted_classifier_fingerprint == fingerprint,
        "learned classifier nodes changed the physical delay fingerprint",
    )
    return {
        "physical_basis_identity_max_error": identity_error,
        "physical_basis_frozen": model.physical_basis.frozen,
        "physical_frontend_fingerprint": fingerprint,
    }


def _control_and_causality_checks(
    model: V8AccuracyFirstModel,
    *,
    device: torch.device,
    tolerances: dict[str, float],
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    model = model.to(device).eval()
    prior_fingerprint = _load_synthetic_prior(model)
    total_samples = int(round((model.task_tmax - model.epoch_tmin) * model.sfreq))
    generator = torch.Generator().manual_seed(91)
    x = torch.randn(1, model.n_channels, total_samples, generator=generator).to(device) * 1e-5
    before = mapping_sha256(model.state_dict())
    with torch.no_grad():
        off_output = model(x, delay_override="off")
        zero_output = model(x, delay_override="zero")
        full_output = model(x, delay_override="full")
    off = off_output["prefix_logits"]
    zero = zero_output["prefix_logits"]
    full = full_output["prefix_logits"]
    after = mapping_sha256(model.state_dict())
    zero_delay = zero_output["aux"]["delay_auxiliary"]
    full_delay = full_output["aux"]["delay_auxiliary"]
    if zero_delay is None or full_delay is None or model.delay_auxiliary is None:
        raise RuntimeError("E0 matched delay outputs are missing")
    routing_fingerprint = model.delay_auxiliary.routing_fingerprint()
    assert_delay_control_contract(
        full_current=full_delay.physical_current.cpu().numpy(),
        zero_current=zero_delay.physical_current.cpu().numpy(),
        full_delay_probability=full_delay.transport_probability.cpu().numpy(),
        zero_delay_probability=zero_delay.transport_probability.cpu().numpy(),
        full_fractional_delay=full_delay.transport_fractional_delay.cpu().numpy(),
        zero_fractional_delay=zero_delay.transport_fractional_delay.cpu().numpy(),
        full_routing_fingerprint=routing_fingerprint,
        zero_routing_fingerprint=routing_fingerprint,
        shared_state_before=before,
        shared_state_after=after,
        minimum_current_rms=float(tolerances["minimum_matched_current_rms"]),
    )
    full_difference = float((full - zero).abs().max())
    _assert(
        full_difference >= float(tolerances["minimum_full_delay_logit_difference"]),
        "V8 full delay does not alter logits in the synthetic E0 control",
    )

    first_endpoint = model.endpoint_seconds[0]
    future_start = int(
        round((model.task_tmin - model.epoch_tmin + first_endpoint) * model.sfreq)
    )
    future = x.clone()
    future[..., future_start:] += torch.randn_like(future[..., future_start:]) * 1e-4
    with torch.no_grad():
        prefix_a = model(x, delay_override="off")["prefix_logits"][:, 0]
        prefix_b = model(future, delay_override="off")["prefix_logits"][:, 0]
    prefix_error = float((prefix_a - prefix_b).abs().max())
    _assert(
        prefix_error <= float(tolerances["future_prefix_atol"]),
        "future task samples changed the first V8 causal endpoint",
    )
    return (
        {
            "delay_prior_fingerprint": prior_fingerprint,
            "off_zero_max_abs_difference": float((off - zero).abs().max()),
            "full_zero_max_abs_difference": full_difference,
            "full_current_rms": float(full_delay.physical_current.square().mean().sqrt()),
            "zero_current_rms": float(zero_delay.physical_current.square().mean().sqrt()),
            "first_endpoint_future_prefix_max_error": prefix_error,
            "shared_state_unchanged": before == after,
        },
        full[:, -1].detach(),
        full.detach(),
    )


def _backward_smoke(
    model: V8AccuracyFirstModel,
    *,
    device: torch.device,
    batch_size: int,
    maximum_peak_mib: float,
) -> dict[str, Any]:
    model = model.to(device).train()
    _load_synthetic_prior(model)
    total_samples = int(round((model.task_tmax - model.epoch_tmin) * model.sfreq))
    x = torch.randn(batch_size, model.n_channels, total_samples, device=device) * 1e-5
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    output = model(x, delay_override="full")
    loss = (
        output["logits"].square().mean()
        + 1e-3 * output["firing_rate_loss"]
        + 1e-4 * output["spatial_orthogonality_loss"]
        + 1e-4 * output["statistical_orthogonality_loss"]
    )
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    finite_logits = bool(torch.isfinite(output["prefix_logits"]).all())
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    nonzero_gradients = sum(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in model.parameters()
    )
    _assert(finite_logits and finite_gradients, "V8 forward/backward contains NaN or Inf")
    _assert(nonzero_gradients > 0, "V8 backward produced no non-zero parameter gradients")
    for spikes in output["aux"]["binary_spikes"]:
        _assert(bool(((spikes == 0) | (spikes == 1)).all()), "non-binary activity counted as spike")
    peak_mib = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
    if device.type == "cuda":
        _assert(peak_mib <= maximum_peak_mib, "V8 CUDA memory budget exceeded")
    return {
        f"{device.type}_batch_size": batch_size,
        f"{device.type}_elapsed_seconds": elapsed,
        f"{device.type}_peak_allocated_mib": peak_mib,
        f"{device.type}_finite_logits": finite_logits,
        f"{device.type}_finite_gradients": finite_gradients,
        f"{device.type}_nonzero_gradient_tensors": nonzero_gradients,
        f"{device.type}_binary_spike_tensors": len(output["aux"]["binary_spikes"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/v8_e0_invariants.yaml")
    parser.add_argument("--output", default="")
    parser.add_argument("--p0", default="")
    parser.add_argument("--storage-root", default="")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    started_at = time.time()
    torch.manual_seed(0)
    storage_root = configure_cache_env(args.storage_root or None)
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config_path = (ROOT / config["model_config"]).resolve()
    model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    tolerances = config["tolerances"]
    output = ensure_dir(Path(args.output).resolve() if args.output else _default_output(storage_root))
    p0 = Path(args.p0).resolve() if args.p0 else _default_p0(storage_root)
    validate_run_artifact_manifest(
        p0,
        required_files=P0_REQUIRED_FILES,
        verify_hashes=True,
        verify_prediction_schema=False,
    )
    p0_fingerprint = read_json(p0 / "source_fingerprint.json")

    source_tree = collect_source_tree_manifest(ROOT)
    environment = _environment(storage_root)
    fingerprint = build_v8_run_fingerprint(
        resolved_run_config={"experiment": config, "model": model_config},
        source_tree=source_tree,
        data={"kind": "synthetic_e0", "seed": int(config["seed"])},
        split={"kind": "none", "synthetic": True},
        augmentation={"policy": "none"},
        prior={"policy": "synthetic_sparse_within_band", "fold_local": True},
        checkpoint={"policy": "fresh_e0_state", "p0": p0_fingerprint["combined_sha256"]},
        environment=environment,
    )
    fingerprint_path = output / "source_fingerprint.json"
    if fingerprint_path.is_file():
        validate_v8_resume_fingerprint(fingerprint_path, fingerprint)
        validate_run_artifact_manifest(
            output,
            required_files=E0_REQUIRED_FILES,
            verify_hashes=True,
            verify_prediction_schema=True,
        )
        print(json.dumps({"status": "already_complete", "output": str(output)}, indent=2))
        return

    metrics: dict[str, Any] = {"status": "running"}
    full_ann = _model_from_config(model_config, decoder_kind="ann")
    metrics.update(_filter_checks(full_ann, tolerances))
    metrics.update(_physical_basis_checks(full_ann))
    parameter_counts = {
        kind: _model_from_config(model_config, decoder_kind=kind).parameter_count
        for kind in ("ann", "plif", "clif")
    }
    parameter_spread = max(parameter_counts.values()) - min(parameter_counts.values())
    _assert(
        max(parameter_counts.values()) <= int(tolerances["maximum_parameters"]),
        "V8 parameter ceiling exceeded",
    )
    _assert(
        parameter_spread <= int(tolerances["maximum_decoder_parameter_count_difference"]),
        "ANN/PLIF/CLIF parameter counts are not matched",
    )
    metrics["decoder_parameter_counts"] = parameter_counts
    metrics["decoder_parameter_count_spread"] = parameter_spread

    compact_control = _model_from_config(
        model_config,
        decoder_kind="ann",
        compact=dict(config["cpu_smoke"]),
    )
    control_metrics, logits, _prefix_logits = _control_and_causality_checks(
        compact_control,
        device=torch.device("cpu"),
        tolerances=tolerances,
    )
    metrics.update(control_metrics)
    compact_clif = _model_from_config(
        model_config,
        decoder_kind="clif",
        compact=dict(config["cpu_smoke"]),
    )
    metrics.update(
        _backward_smoke(
            compact_clif,
            device=torch.device("cpu"),
            batch_size=1,
            maximum_peak_mib=float(tolerances["maximum_cuda_peak_allocated_mib"]),
        )
    )

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("E0 requires CUDA, but CUDA is unavailable")
    if torch.cuda.is_available():
        full_clif = _model_from_config(model_config, decoder_kind="clif")
        metrics.update(
            _backward_smoke(
                full_clif,
                device=torch.device("cuda"),
                batch_size=int(config["cuda_smoke_batch"]),
                maximum_peak_mib=float(tolerances["maximum_cuda_peak_allocated_mib"]),
            )
        )
        checkpoint_model = full_clif.cpu()
    else:
        metrics["cuda_skipped"] = True
        checkpoint_model = compact_clif

    metrics["status"] = "passed"
    metrics["elapsed_seconds"] = time.time() - started_at
    shutil.copyfile(config_path, output / "resolved_config.yaml")
    write_v8_fingerprint(fingerprint_path, fingerprint)
    write_json(
        output / "p0_reference.json",
        {"path": str(p0), "combined_sha256": p0_fingerprint["combined_sha256"]},
    )
    write_json(output / "metrics.json", metrics)
    write_csv(output / "history.csv", [{"stage": "E0", **metrics}])
    state = {key: value.detach().cpu() for key, value in checkpoint_model.state_dict().items()}
    torch.save(state, output / "best.pt")
    torch.save(state, output / "last.pt")
    probability = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    logits_array = logits.float().cpu().numpy()
    write_trial_predictions(
        output,
        logits=logits_array,
        probabilities=probability,
        pred=probability.argmax(axis=1),
        label=np.zeros(logits_array.shape[0], dtype=np.int64),
        subject="synthetic",
        session="T",
        run="e0",
        trial_id=[f"v8-e0:{index}" for index in range(logits_array.shape[0])],
        seed=int(config["seed"]),
        model="v8_accuracy_first_e0",
    )
    write_json(
        output / "runtime_status.json",
        {"status": "completed", "started_at": started_at, "completed_at": time.time()},
    )
    (output / "stdout.log").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output / "stderr.log").write_text("", encoding="utf-8")
    write_run_artifact_manifest(output, required_files=E0_REQUIRED_FILES)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
