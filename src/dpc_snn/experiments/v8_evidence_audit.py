"""V8 evidence-space preparation without classifier fitting."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch

from dpc_snn.analysis.evidence_space import (
    apply_euclidean_alignment,
    current_source_density,
    fit_var_innovations,
)
from dpc_snn.experiments.v8_training import (
    V8CachedRates,
    cache_v8_physical_rates,
    fit_v8_physical_gain,
)
from dpc_snn.models.build import build_model
from dpc_snn.models.v8_accuracy_first import V8AccuracyFirstModel


V8EvidenceSpaceName = Literal[
    "car",
    "csd",
    "ea",
    "car_innovations",
    "csd_innovations",
    "car_innovations_unwhitened",
    "csd_innovations_unwhitened",
    "car_innovations_unwhitened_regions",
    "csd_innovations_unwhitened_regions",
]
V8_EVIDENCE_SPACES = (
    "car",
    "csd",
    "ea",
    "car_innovations",
    "csd_innovations",
    "car_innovations_unwhitened",
    "csd_innovations_unwhitened",
    "car_innovations_unwhitened_regions",
    "csd_innovations_unwhitened_regions",
)

V8_ANATOMICAL_REGION_NAMES = (
    "anterior_left",
    "anterior_midline",
    "anterior_right",
    "central_left",
    "central_midline",
    "central_right",
    "posterior_left",
    "posterior_midline",
    "posterior_right",
)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _car_baseline(
    x: np.ndarray,
    *,
    sfreq: float,
    epoch_tmin: float,
    task_tmin: float,
) -> np.ndarray:
    array = np.asarray(x, dtype=np.float32)
    baseline_stop = int(round((float(task_tmin) - float(epoch_tmin)) * float(sfreq)))
    if baseline_stop <= 0 or baseline_stop >= array.shape[-1]:
        raise ValueError("V8 evidence preprocessing requires a valid pre-task baseline")
    car = array - array.mean(axis=1, keepdims=True)
    return (car - car[..., :baseline_stop].mean(axis=-1, keepdims=True)).astype(np.float32)


def v8_anatomical_region_indices(channel_names: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    """Map BCI2a 10-20 electrodes to fixed anterior/central/posterior regions."""

    groups: list[list[int]] = [[] for _ in V8_ANATOMICAL_REGION_NAMES]
    for index, raw_name in enumerate(channel_names):
        name = str(raw_name)
        if name.startswith(("F", "FC")):
            row = 0
        elif name.startswith("C") and not name.startswith("CP"):
            row = 1
        elif name.startswith(("CP", "P", "PO")):
            row = 2
        else:
            raise ValueError(f"unsupported electrode for fixed V8 regions: {name}")
        if name.endswith("z"):
            column = 1
        else:
            try:
                column = 0 if int(name[-1]) % 2 else 2
            except ValueError as exc:
                raise ValueError(f"electrode lacks a 10-20 side suffix: {name}") from exc
        groups[3 * row + column].append(index)
    if any(not group for group in groups):
        raise ValueError("fixed V8 anatomical partition contains an empty region")
    return tuple(tuple(group) for group in groups)


def pool_v8_anatomical_regions(
    rates: V8CachedRates,
    channel_names: Sequence[str],
) -> tuple[V8CachedRates, dict[str, Any]]:
    """Equal-pool fixed physical electrodes into nine interpretable regions."""

    groups = v8_anatomical_region_indices(channel_names)
    if rates.fast.shape[2] != len(channel_names):
        raise ValueError("V8 regional pooling channel axis does not match channel names")
    fast = torch.stack(
        [rates.fast[:, :, list(group)].mean(dim=2) for group in groups],
        dim=2,
    )
    slow = torch.stack(
        [rates.slow[:, :, list(group)].mean(dim=2) for group in groups],
        dim=2,
    )
    digest = hashlib.sha256()
    digest.update(rates.physical_frontend_fingerprint.encode("ascii"))
    digest.update(repr(groups).encode("ascii"))
    pooled = V8CachedRates(
        fast=fast,
        slow=slow,
        gain=torch.ones(fast.shape[1], fast.shape[2]),
        physical_frontend_fingerprint=digest.hexdigest(),
    )
    metadata = {
        "region_names": list(V8_ANATOMICAL_REGION_NAMES),
        "region_members": [
            [str(channel_names[index]) for index in group] for group in groups
        ],
        "region_pooling": "fixed_equal_complex_analytic_mean_after_fold_gain",
        "region_count": len(groups),
    }
    return pooled, metadata


def prepare_v8_evidence_space(
    x: np.ndarray,
    space: V8EvidenceSpaceName,
    *,
    channel_names: Sequence[str],
    sfreq: float,
    epoch_tmin: float,
    task_tmin: float,
    innovation_order: int = 6,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Prepare one inner-fold-only evidence space and record its node semantics."""

    if space not in V8_EVIDENCE_SPACES:
        raise ValueError(f"unsupported V8 evidence space: {space}")
    raw = np.asarray(x, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1] != len(channel_names):
        raise ValueError("V8 evidence input must be [trial, named_channel, time]")
    adjusted_epoch_tmin = float(epoch_tmin)
    fitted_state: dict[str, Any] = {}
    region_pooling = space.endswith("_regions")
    if space.startswith("csd"):
        transformed = current_source_density(raw, list(channel_names), float(sfreq))
        reference = "spherical_spline_csd"
    else:
        transformed = raw.copy()
        reference = "common_average_reference"

    transformed = _car_baseline(
        transformed,
        sfreq=sfreq,
        epoch_tmin=epoch_tmin,
        task_tmin=task_tmin,
    )
    exact_sensor_nodes = True
    if space == "ea":
        transformed, alignment = apply_euclidean_alignment(transformed)
        fitted_state["alignment_sha256"] = _array_sha256(alignment)
        reference = "inner_fold_euclidean_alignment_after_car"
        exact_sensor_nodes = False
    elif "_innovations" in space:
        whiten = "_unwhitened" not in space
        transformed = fit_var_innovations(
            transformed,
            order=int(innovation_order),
            whiten=whiten,
        )
        adjusted_epoch_tmin += float(innovation_order) / float(sfreq)
        fitted_state["innovation_order"] = int(innovation_order)
        fitted_state["innovation_whitened"] = whiten
        reference += "_whitened_var_innovations" if whiten else "_var_innovations"
        exact_sensor_nodes = not whiten and not region_pooling
    if region_pooling:
        if "_innovations_unwhitened" not in space:
            raise ValueError("regional V8 evidence currently requires unwhitened innovations")
        reference += "_fixed_anatomical_region_pool"

    metadata = {
        "space": space,
        "reference": reference,
        "fit_scope": "active_inner_training_fold_only",
        "input_trials": int(raw.shape[0]),
        "input_shape": list(raw.shape),
        "output_shape": list(transformed.shape),
        "output_sha256": _array_sha256(transformed),
        "epoch_tmin": adjusted_epoch_tmin,
        "exact_physical_sensor_nodes": exact_sensor_nodes,
        "eligible_for_physical_delay_routes": exact_sensor_nodes,
        "eligible_for_region_delay_routes": region_pooling,
        "eligible_for_interpretable_delay_routes": exact_sensor_nodes or region_pooling,
        "region_pooling_requested": region_pooling,
        **fitted_state,
    }
    return transformed, adjusted_epoch_tmin, metadata


def build_v8_evidence_rates(
    x: np.ndarray,
    model_config: Mapping[str, Any],
    *,
    epoch_tmin: float,
    transform_sha256: str,
    device: str,
    batch_size: int = 16,
) -> V8CachedRates:
    """Run a transformed inner fold through the frozen V8 analytic front end."""

    config = deepcopy(dict(model_config))
    config["epoch_tmin"] = float(epoch_tmin)
    model = build_model("v8_accuracy_first", config)
    if not isinstance(model, V8AccuracyFirstModel):
        raise TypeError("V8 evidence audit built an incompatible model")
    if model.delay_auxiliary is not None or not model.physical_basis.frozen:
        raise RuntimeError("V8 evidence audit requires the delay-free frozen physical front end")
    fit_v8_physical_gain(model, x, device=device, batch_size=int(batch_size))
    rates = cache_v8_physical_rates(model, x, device=device, batch_size=int(batch_size))
    digest = hashlib.sha256()
    digest.update(rates.physical_frontend_fingerprint.encode("ascii"))
    digest.update(str(transform_sha256).encode("ascii"))
    return V8CachedRates(
        fast=rates.fast,
        slow=rates.slow,
        gain=rates.gain,
        physical_frontend_fingerprint=digest.hexdigest(),
    )


def within_band_route_count(arrays: Mapping[str, np.ndarray]) -> int:
    accepted = np.asarray(arrays["band_pair_accepted"], dtype=bool)
    if accepted.ndim != 4 or accepted.shape[0] != accepted.shape[1]:
        raise ValueError("band-pair acceptance must be [target_band, source_band, target, source]")
    bands, _, nodes, source_nodes = accepted.shape
    if nodes != source_nodes:
        raise ValueError("V8 evidence routes require square physical node axes")
    mask = np.eye(bands, dtype=bool)[:, :, None, None]
    nonself = ~np.eye(nodes, dtype=bool)[None, None]
    return int((accepted & mask & nonself).sum())


def select_physical_evidence_space(
    summaries: Sequence[Mapping[str, Any]],
    *,
    priority: Sequence[str] = (
        "csd_innovations_unwhitened_regions",
        "car_innovations_unwhitened_regions",
        "csd_innovations_unwhitened",
        "car_innovations_unwhitened",
        "csd",
        "car",
    ),
) -> str | None:
    """Apply a predeclared physiological priority only among strict passing spaces."""

    passing = {
        str(row["evidence_space"])
        for row in summaries
        if bool(row.get("evidence_pipeline_passed"))
        and bool(
            row.get(
                "eligible_for_interpretable_delay_routes",
                row.get("eligible_for_physical_delay_routes"),
            )
        )
        and int(row.get("accepted_within_band_routes", 0)) > 0
    }
    return next((name for name in priority if name in passing), None)
