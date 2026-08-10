from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dpc_snn.experiments.v62_protocol import sha256_fingerprint
from dpc_snn.experiments.v8_ensemble_followup import (
    decoder_operation_proxy,
    mask_carrier_after_endpoint,
    resolve_frozen_channel_basis,
)
from dpc_snn.experiments.v8_sequence_decoder_training import (
    fit_v8_sequence_decoder,
    predict_v8_sequence_decoder,
)
from dpc_snn.models.v8_sequence_decoder import build_v8_sequence_decoder
from scripts.audit_v8_e8_ensemble_confirmation import (
    _validate_input_adapter_manifest,
)
from scripts.run_v8_e8_ensemble_confirmation import _validate_barrier


def test_mask_carrier_after_endpoint_has_no_future_samples() -> None:
    carrier = np.ones((2, 22, 1000), dtype=np.float32)
    masked = mask_carrier_after_endpoint(
        carrier, endpoint_seconds=1.0, sfreq=250.0
    )
    assert np.all(masked[..., :250] == 1.0)
    assert np.all(masked[..., 250:] == 0.0)
    assert np.all(carrier == 1.0)


def test_decoder_operation_proxy_is_explicit_and_lower_for_sparse_snn() -> None:
    kwargs = {
        "n_classes": 4,
        "hidden_channels": 16,
        "decoder_layers": 2,
        "readout_features": 12,
        "dropout": 0.0,
    }
    snn = build_v8_sequence_decoder("sew_clif", **kwargs)
    ann = build_v8_sequence_decoder("ann_sew", **kwargs)
    snn_proxy = decoder_operation_proxy(snn, mean_binary_firing_rate=0.1)
    ann_proxy = decoder_operation_proxy(ann, mean_binary_firing_rate=None)
    assert snn_proxy["activity_weighted_decoder_events"] < ann_proxy[
        "activity_weighted_decoder_events"
    ]
    assert snn_proxy["hardware_energy_claim_allowed"] is False
    assert "shared ATC/FBC" in snn_proxy["scope"]


def test_sequence_decoder_training_supports_binary_external_head() -> None:
    rng = np.random.default_rng(7)
    sequence = rng.normal(size=(8, 18, 32)).astype(np.float32)
    labels = np.asarray([0, 1] * 4, dtype=np.int64)
    teacher = rng.normal(size=(8, 2)).astype(np.float32)
    fit = fit_v8_sequence_decoder(
        "ann_sew",
        x_train=sequence,
        y_train=labels,
        teacher_train=teacher,
        x_validation=None,
        y_validation=None,
        teacher_validation=None,
        device="cpu",
        seed=11,
        fixed_epoch=1,
        scheduler_epochs=1,
        batch_size=4,
        model_kwargs={
            "n_classes": 2,
            "hidden_channels": 8,
            "decoder_layers": 1,
            "readout_features": 8,
            "dropout": 0.0,
        },
    )
    result = predict_v8_sequence_decoder(
        fit.model, sequence, labels, teacher, device="cpu", batch_size=4
    )
    assert result["logits"].shape == (8, 2)
    assert 0.0 <= result["accuracy"] <= 1.0


def test_e8_barrier_requires_exact_source_unlock_and_coverage(tmp_path: Path) -> None:
    body = {
        "schema": "dpc-snn-v8-e8-checkpoint-barrier/v1",
        "created_at": 1.0,
        "source_tree_sha256": "source",
        "external_unlock_sha256": "unlock",
        "subjects": [1, 2],
        "seeds": [0, 1],
        "runs": 4,
        "all_training_complete_before_s2": True,
        "records": [],
    }
    barrier = {**body, "combined_sha256": sha256_fingerprint(body)}
    path = tmp_path / "barrier.json"
    path.write_text(json.dumps(barrier), encoding="utf-8")
    loaded = _validate_barrier(
        path,
        source_digest="source",
        unlock_sha256="unlock",
        subjects=[1, 2],
        seeds=[0, 1],
    )
    assert loaded["all_training_complete_before_s2"] is True


def test_e8_unlock_resolves_legacy_freeze_channel_basis() -> None:
    freeze = {"architecture": {"model_config": {"architecture_id": "legacy"}}}
    channels = resolve_frozen_channel_basis(freeze)
    assert len(channels) == 22
    assert channels[:3] == ["Fz", "FC3", "FC1"]
    assert channels[-1] == "POz"


def test_e8_unlock_rejects_declared_channel_order_drift() -> None:
    freeze = {
        "architecture": {
            "model_config": {
                "channel_names": ["FC3", "Fz", "FC1"],
            }
        }
    }
    try:
        resolve_frozen_channel_basis(freeze)
    except RuntimeError as exc:
        assert "sensor order" in str(exc)
    else:
        raise AssertionError("channel order drift must be rejected")


def test_e8_audit_rejects_fcz_recorded_as_raw_channel() -> None:
    channels = resolve_frozen_channel_basis(
        {"architecture": {"model_config": {"architecture_id": "legacy"}}}
    )
    adapter = {
        "policy": "fixed_linear_missing_sensor_interpolation",
        "derived_channels": {
            "FCz": {
                "source_channels": ["FC1", "FC2"],
                "weights": [0.5, 0.5],
            }
        },
    }
    manifest = {
        "channel_names": channels,
        "source_channel_indices": list(range(22)),
        "input_adapter": {
            "policy": adapter["policy"],
            "applied_derived_channels": adapter["derived_channels"],
        },
    }
    try:
        _validate_input_adapter_manifest(
            manifest,
            expected_channels=channels,
            expected_adapter=adapter,
        )
    except RuntimeError as exc:
        assert "derived channel" in str(exc)
    else:
        raise AssertionError("audit must reject an FCz channel that was not derived")
