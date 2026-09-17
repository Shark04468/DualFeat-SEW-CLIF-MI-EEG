"""Synthetic validation for continuous sub-sample delay transport."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch

from dpc_snn.models.delay_phase_synapse import DelayPhaseGraphSynapse


def recover_fractional_delays(
    targets: Iterable[float] = (0.25, 0.5, 0.75),
    steps: int = 160,
    learning_rate: float = 0.15,
    seed: int = 39,
    device: str = "cpu",
) -> list[dict[str, float | bool]]:
    """Recover sub-sample offsets from independently generated delayed signals."""

    torch.manual_seed(seed)
    n_trials, n_channels, n_time = 16, 2, 256
    time = torch.arange(n_time, device=device, dtype=torch.float32)
    frequencies = torch.tensor((0.025, 0.05, 0.08), device=device)
    phases = torch.rand(n_trials, n_channels, len(frequencies), device=device) * (
        2.0 * math.pi
    )
    amplitudes = 0.5 + torch.rand_like(phases)

    def continuous_signal(sample_time: torch.Tensor) -> torch.Tensor:
        angle = (
            2.0
            * math.pi
            * frequencies[None, None, :, None]
            * sample_time
            + phases[..., None]
        )
        return (amplitudes[..., None] * torch.sin(angle)).sum(dim=2)

    source = continuous_signal(time)
    rows: list[dict[str, float | bool | str]] = []
    for target in targets:
        if not 0.0 < float(target) < 1.0:
            raise ValueError("This gate validates strictly sub-sample delays in (0, 1)")
        synapse = DelayPhaseGraphSynapse(
            n_bands=1,
            n_channels=2,
            d_max=2,
            n_delay_experts=1,
            expert_topk=1,
            route_rank=1,
        ).to(device)
        with torch.no_grad():
            synapse.fast_fraction_target_raw.zero_()
            synapse.fast_fraction_source_raw.zero_()
            synapse.fast_fraction_rank_raw.zero_()
            synapse.fast_fraction_raw.fill_(torch.logit(torch.tensor(0.1)))
        for parameter in synapse.parameters():
            parameter.requires_grad = parameter is synapse.fast_fraction_raw
        optimizer = torch.optim.Adam([synapse.fast_fraction_raw], lr=learning_rate)
        # The target is sampled from the underlying continuous waveform rather
        # than produced by the synapse interpolation code. This avoids directly
        # supervising the delay parameter and avoids an interpolation inverse
        # crime in the recovery gate.
        delayed_target = continuous_signal(time - float(target))[:, 1]
        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            fraction = synapse.fast_fraction_expert()[0, 0, 0]
            candidates, _ = synapse._fractional_pair_candidates(source, 2, fraction)
            recovered_signal = candidates[:, 0, 1, 0]
            # Exclude the causal padding boundary, where the finite observed
            # source cannot represent the continuous signal before t=0.
            loss = (recovered_signal[..., 2:] - delayed_target[..., 2:]).square().mean()
            loss.backward()
            optimizer.step()
        recovered = synapse.fast_fraction_expert()[0, 0, 0, 0, 1].detach()
        recovered_fraction = torch.full(
            (n_channels, n_channels), float(recovered), device=device
        )
        recovered_candidates, _ = synapse._fractional_pair_candidates(
            source, 2, recovered_fraction
        )
        recovered_signal = recovered_candidates[:, 0, 1, 0]
        signal_rmse = torch.sqrt(
            (recovered_signal[..., 2:] - delayed_target[..., 2:]).square().mean()
        )
        error = abs(float(recovered) - float(target))
        rows.append(
            {
                "target_delay_samples": float(target),
                "recovered_delay_samples": float(recovered),
                "absolute_error_samples": error,
                "transport_rmse": float(signal_rmse),
                "recovery_objective": "delayed_signal_fit",
                "passed": bool(error <= 0.03 and float(signal_rmse) <= 0.03),
            }
        )
    return rows


def summarize_fractional_recovery(
    rows: list[dict[str, float | bool | str]],
) -> dict[str, object]:
    errors = np.asarray([float(row["absolute_error_samples"]) for row in rows])
    return {
        "status": "passed" if rows and all(bool(row["passed"]) for row in rows) else "failed",
        "targets": rows,
        "mean_absolute_error_samples": float(errors.mean()) if errors.size else float("nan"),
        "max_absolute_error_samples": float(errors.max()) if errors.size else float("nan"),
    }
