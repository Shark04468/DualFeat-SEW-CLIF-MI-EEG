#!/usr/bin/env python
"""Bounded CPU/CUDA forward-backward smoke for the V5 event-native path."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.models.dpc_snn import DPCSNN  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--full-shape", action="store_true")
    args = parser.parse_args()

    if args.full_shape:
        bands = [
            [6.0, 8.0],
            [8.0, 10.0],
            [10.0, 12.0],
            [12.0, 14.0],
            [14.0, 17.0],
            [17.0, 20.0],
            [20.0, 23.0],
            [23.0, 26.0],
            [26.0, 30.0],
            [30.0, 34.0],
            [34.0, 37.0],
            [37.0, 40.0],
        ]
        channels, nodes, steps, samples, sfreq = 22, 16, 500, 1250, 250.0
        classes, snn_channels, experts, max_delay = 4, 64, 4, 8
        epoch_tmin, task_tmax = -1.0, 4.0
    else:
        bands = [[3.0, 5.0], [5.0, 8.0], [8.0, 12.0]]
        channels, nodes, steps, samples, sfreq = 3, 3, 64, 128, 128.0
        classes, snn_channels, experts, max_delay = 2, 12, 2, 2
        epoch_tmin, task_tmax = 0.0, 1.0
    model = DPCSNN(
        n_classes=classes,
        n_channels=channels,
        n_bands=len(bands),
        hidden_channels=4,
        timesteps=steps,
        graph_timesteps=steps,
        sfreq=sfreq,
        epoch_tmin=epoch_tmin,
        task_tmin=0.0,
        task_tmax=task_tmax,
        latent_nodes=nodes,
        snn_channels=snn_channels,
        temporal_pool_channels=16 if args.full_shape else 4,
        d_max=max_delay,
        slow_d_max=max_delay,
        band_edges_hz=bands,
        delay_evidence_band_edges_hz=bands,
        classification_band_edges_hz=bands,
        decoder_layers=2,
        n_delay_experts=experts,
        delay_expert_topk=min(2, experts),
        freeze_delay_posterior=True,
        freeze_shared_physical_basis=True,
        freeze_shared_filterbank=True,
        freeze_delay_after_pretrain=True,
        preserve_transport_band_pairs=True,
        delayed_stat_channels=0,
        event_native_transport=True,
        checkpoint_event_routes=args.full_shape,
        multiscale_snn=True,
        snn_native_readout=True,
        use_membrane_readout=False,
        reference_augmentation_prob=0.0,
        cumulative_readout_seconds=(1.0, 2.0, 4.0) if args.full_shape else (0.5, 1.0),
        architecture_version="dpc_snn_v5_event_native",
    ).to(args.device)
    route = 0.7 * model.synapse.route_mask
    positive = torch.zeros(*route.shape, model.synapse.d_max + 1)
    positive[..., 1] = model.synapse.route_mask
    fraction = 0.5 * model.synapse.route_mask
    model.load_fold_local_evidence_prior(route, positive, fraction, route)
    model.begin_task_training()
    model.set_training_stage("joint")
    model.set_train_fitted_route_rms(1.0)

    x = torch.randn(
        args.batch_size, channels, samples, device=args.device
    )
    delta = model.event_envelope_deltas(x)
    flat_delta = delta.permute(1, 0, 2, 3).reshape(delta.shape[1], -1)
    thresholds = torch.quantile(flat_delta, 0.90, dim=1).clamp_min(1e-5)
    model.set_train_fitted_event_thresholds(thresholds)
    model.train(not args.forward_only)
    target = torch.arange(args.batch_size, device=args.device) % classes
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    context = torch.no_grad() if args.forward_only else nullcontext()
    with context:
        output = model(x=x, delay_evidence_x=x)
        loss = F.cross_entropy(output["logits"], target)
        loss = loss + model.regularization_loss(output["aux"])
    finite_gradients = True
    gradient_parameters = 0
    if not args.forward_only:
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        gradient_parameters = len(gradients)
        finite_gradients = bool(
            gradients and all(torch.isfinite(value).all() for value in gradients)
        )
    payload = {
        "device": args.device,
        "full_shape": args.full_shape,
        "logits_shape": list(output["logits"].shape),
        "loss": float(loss.detach().cpu()),
        "phase_event_density": float(
            output["aux"]["phase_event_density"].detach().cpu()
        ),
        "envelope_event_density": float(
            output["aux"]["envelope_event_density"].detach().cpu()
        ),
        "binary_output": bool(
            torch.all(
                (output["aux"]["hidden_spikes"] == 0)
                | (output["aux"]["hidden_spikes"] == 1)
            )
        ),
        "gradient_parameters": gradient_parameters,
        "finite_gradients": finite_gradients,
        "peak_cuda_mb": (
            float(torch.cuda.max_memory_allocated() / 1024**2)
            if args.device.startswith("cuda")
            else None
        ),
    }
    if not torch.isfinite(output["logits"]).all() or not finite_gradients:
        raise RuntimeError(f"V5 smoke failed: {payload}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
