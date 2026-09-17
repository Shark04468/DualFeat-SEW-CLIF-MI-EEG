"""Frozen channel/time adapter from OpenBMI to the V8 physical input space."""

from __future__ import annotations

from math import gcd
from typing import Any, Mapping, Sequence

import numpy as np

from dpc_snn.experiments.v62_protocol import validate_trial_metadata
from dpc_snn.experiments.v8_protocol import assert_v8_data_access


OPENBMI_FIXED_INPUT_ADAPTER: dict[str, Any] = {
    "policy": "fixed_linear_missing_sensor_interpolation",
    "derived_channels": {
        "FCz": {
            "source_channels": ["FC1", "FC2"],
            "weights": [0.5, 0.5],
        }
    },
}


def prepare_openbmi_v8_view(
    data: dict[str, Any],
    *,
    channel_names: Sequence[str],
    input_adapter: Mapping[str, Any] | None = None,
    target_sfreq: float = 250.0,
    stage: str = "openbmi_confirmation",
    role: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Project to the frozen 22 sensors and anti-alias to the V8 250-Hz grid."""

    from scipy import signal

    required = {"X", "y", "subject", "session", "run", "trial_id", "ch_names"}
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"OpenBMI V8 view is missing fields: {missing}")
    source_names = [str(name) for name in data["ch_names"]]
    normalized: dict[str, int] = {}
    for index, name in enumerate(source_names):
        key = name.strip().upper()
        if key in normalized:
            raise ValueError(f"OpenBMI contains duplicate channel {name!r}")
        normalized[key] = index
    requested = [str(name) for name in channel_names]
    adapter = dict(input_adapter or {})
    derived_rules = dict(adapter.get("derived_channels", {}))
    if adapter and adapter.get("policy") != "fixed_linear_missing_sensor_interpolation":
        raise ValueError("OpenBMI input adapter policy is not supported")
    projection = np.zeros((len(requested), len(source_names)), dtype=np.float32)
    source_indices: list[int | None] = []
    applied_rules: dict[str, Any] = {}
    missing_channels: list[str] = []
    for output_index, name in enumerate(requested):
        key = name.strip().upper()
        if key in normalized:
            source_index = normalized[key]
            projection[output_index, source_index] = 1.0
            source_indices.append(source_index)
            continue
        rule = derived_rules.get(name)
        if not isinstance(rule, Mapping):
            missing_channels.append(name)
            source_indices.append(None)
            continue
        rule_sources = [str(value) for value in rule.get("source_channels", ())]
        weights = np.asarray(rule.get("weights", ()), dtype=np.float64)
        if (
            not rule_sources
            or weights.shape != (len(rule_sources),)
            or not np.isfinite(weights).all()
            or np.any(weights < 0.0)
            or not np.isclose(float(weights.sum()), 1.0)
        ):
            raise ValueError(f"invalid fixed interpolation rule for {name}")
        source_keys = [value.strip().upper() for value in rule_sources]
        if any(value not in normalized for value in source_keys):
            raise ValueError(f"interpolation sources are missing for {name}")
        for source_key, weight in zip(source_keys, weights, strict=True):
            projection[output_index, normalized[source_key]] = float(weight)
        source_indices.append(None)
        applied_rules[name] = {
            "source_channels": rule_sources,
            "weights": weights.tolist(),
        }
    if missing_channels:
        raise ValueError(f"OpenBMI is missing frozen V8 channels: {missing_channels}")
    raw = np.asarray(data["X"], dtype=np.float32)
    x = np.einsum("oc,nct->not", projection, raw, optimize=True).astype(
        np.float32, copy=False
    )
    source_sfreq = float(data["sfreq"])
    target_sfreq = float(target_sfreq)
    if not np.isfinite(source_sfreq) or not np.isfinite(target_sfreq):
        raise ValueError("OpenBMI/V8 sampling rates must be finite")
    source_integer = int(round(source_sfreq))
    target_integer = int(round(target_sfreq))
    if not np.isclose(source_sfreq, source_integer) or not np.isclose(
        target_sfreq, target_integer
    ):
        raise ValueError("OpenBMI/V8 adapter requires integer-Hz sample grids")
    divisor = gcd(source_integer, target_integer)
    up, down = target_integer // divisor, source_integer // divisor
    if up != down:
        x = signal.resample_poly(x, up=up, down=down, axis=-1).astype(np.float32)
    duration = float(data["epoch_tmax"]) - float(data["epoch_tmin"])
    expected = int(round(duration * target_sfreq))
    if x.shape[-1] < expected or x.shape[-1] > expected + 1:
        raise RuntimeError(
            f"OpenBMI anti-alias output has {x.shape[-1]} samples, expected {expected} or {expected + 1}"
        )
    x = np.ascontiguousarray(x[..., :expected], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    sessions = np.asarray(data["session"]).astype(str)
    subjects = np.asarray(data["subject"]).astype(str)
    runs = np.asarray(data["run"]).astype(str)
    trial_ids = np.asarray(data["trial_id"]).astype(str)
    if x.shape[0] != y.size or len(set(trial_ids.tolist())) != y.size:
        raise ValueError("OpenBMI V8 trials are misaligned or not uniquely identified")
    rows = validate_trial_metadata(
        [
            {
                "dataset": "OpenBMI_Lee2019_MI",
                "subject": subjects[index],
                "session": sessions[index],
                "run": runs[index],
                "trial_id": trial_ids[index],
                "class": int(y[index]),
                "sfreq": target_sfreq,
                "ch_names": requested,
                "epoch_tmin": float(data["epoch_tmin"]),
                "epoch_tmax": float(data["epoch_tmin"]) + expected / target_sfreq,
            }
            for index in range(y.size)
        ],
        allowed_sessions=("S1", "S2"),
    )
    assert_v8_data_access(rows, stage=stage, role=role)
    labels, counts = np.unique(y, return_counts=True)
    if labels.tolist() != [0, 1] or len(set(counts.tolist())) != 1:
        raise ValueError("OpenBMI V8 view must be balanced binary motor imagery")
    manifest = {
        "dataset": "OpenBMI_Lee2019_MI",
        "sessions": sorted(set(sessions.tolist())),
        "role": role,
        "trials": int(y.size),
        "source_sfreq": source_sfreq,
        "target_sfreq": target_sfreq,
        "resample_poly_up": up,
        "resample_poly_down": down,
        "source_channel_indices": source_indices,
        "channel_names": requested,
        "input_adapter": {
            "policy": adapter.get("policy", "exact_channel_selection"),
            "applied_derived_channels": applied_rules,
        },
        "anti_aliasing": "scipy.signal.resample_poly_polyphase_FIR",
    }
    return x, y, rows, manifest
