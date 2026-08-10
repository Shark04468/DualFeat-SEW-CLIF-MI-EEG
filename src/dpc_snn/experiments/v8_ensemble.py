"""Pure helpers for the frozen V8 ATC/FBC plus sequence-residual ensemble."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dpc_snn.experiments.v62_protocol import sha256_fingerprint
from dpc_snn.experiments.v8_fusion import entropy_residual_probability
from dpc_snn.experiments.v8_protocol import mapping_sha256
from dpc_snn.models.v8_atc_backbone import V8ATCAccuracyBackbone


def state_digest(model: nn.Module) -> str:
    """Hash all persistent model state for held-out mutation checks."""

    return sha256_fingerprint(mapping_sha256(model.state_dict()))


def softmax_probability(logits: np.ndarray) -> np.ndarray:
    array = np.asarray(logits, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 2 or not np.isfinite(array).all():
        raise ValueError("logits must be finite [trials, classes]")
    return torch.softmax(torch.from_numpy(array), dim=1).numpy()


def equal_probability_anchor(
    atc_logits: np.ndarray, fbc_logits: np.ndarray
) -> np.ndarray:
    atc = softmax_probability(atc_logits)
    fbc = softmax_probability(fbc_logits)
    if atc.shape != fbc.shape:
        raise ValueError("ATCNet and FBCNet predictions are not aligned")
    return np.ascontiguousarray(0.5 * atc + 0.5 * fbc, dtype=np.float32)


def entropy_residual_prediction(
    anchor_probability: np.ndarray,
    decoder_logits: np.ndarray,
    *,
    maximum_weight: float = 0.05,
) -> dict[str, np.ndarray]:
    decoder_probability = softmax_probability(decoder_logits)
    final_probability, gate = entropy_residual_probability(
        anchor_probability,
        decoder_probability,
        maximum_weight=float(maximum_weight),
    )
    return {
        "probabilities": final_probability,
        "pred": final_probability.argmax(axis=1).astype(np.int64),
        "gate": gate,
    }


@torch.no_grad()
def extract_atc_sequence(
    adapter: nn.Module,
    x: np.ndarray,
    *,
    device: str,
    batch_size: int = 64,
    n_classes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the exact official ATC post-convolution sequence in memory."""

    official_core = getattr(adapter, "module", None)
    if official_core is None:
        raise TypeError("ATC adapter does not expose its pinned official module")
    array = np.ascontiguousarray(x, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (22, 1000):
        raise ValueError("ATC input must have shape [trials, 22, 1000]")
    wrapper = V8ATCAccuracyBackbone(official_core).eval().to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(array)),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    sequences: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for (batch_x,) in loader:
        output: dict[str, Any] = wrapper(batch_x.to(device, non_blocking=True))
        sequences.append(output["aux"]["continuous_sequence"].float().cpu().numpy())
        logits.append(output["logits"].float().cpu().numpy())
    sequence = np.concatenate(sequences)
    all_logits = np.concatenate(logits)
    if sequence.shape != (array.shape[0], 18, 32):
        raise RuntimeError(f"unexpected ATC sequence shape: {sequence.shape}")
    expected_classes = int(n_classes) if n_classes is not None else int(all_logits.shape[1])
    if expected_classes < 2 or all_logits.shape != (array.shape[0], expected_classes):
        raise RuntimeError(f"unexpected ATC logit shape: {all_logits.shape}")
    if not np.isfinite(sequence).all() or not np.isfinite(all_logits).all():
        raise FloatingPointError("ATC sequence extraction produced non-finite values")
    return np.ascontiguousarray(sequence), np.ascontiguousarray(all_logits)


def validate_ensemble_freeze_contract(freeze: dict[str, Any]) -> dict[str, Any]:
    """Reject any E6 freeze that differs from the selected E4 ensemble."""

    architecture = freeze["architecture"]
    model = architecture["model_config"]
    checkpoint = freeze["checkpoint_rule"]
    baselines = freeze["baselines"]
    if architecture.get("primary_variant") != "sew_clif":
        raise RuntimeError("frozen primary must be SEW-CLIF")
    if architecture.get("delay") != {"enabled": False, "mode": "off"}:
        raise RuntimeError("the failed delay branch must remain disabled in E6")
    if model.get("architecture_id") != "v8_atc_fbc_entropy_residual_sequence_ensemble_r1":
        raise RuntimeError("unknown frozen ensemble architecture")
    if model.get("anchor", {}).get("components") != ["atcnet", "fbcnet"]:
        raise RuntimeError("frozen anchor components changed")
    if model.get("anchor", {}).get("probability_weights") != [0.5, 0.5]:
        raise RuntimeError("frozen anchor weights changed")
    if model.get("primary_decoder") != "sew_clif":
        raise RuntimeError("frozen primary sequence decoder changed")
    if model.get("matched_ann_decoder") != "ann_sew":
        raise RuntimeError("frozen matched ANN decoder changed")
    residual = model.get("residual_fusion", {})
    if residual.get("mode") != "anchor_entropy" or float(
        residual.get("maximum_decoder_weight", -1.0)
    ) != 0.05:
        raise RuntimeError("frozen entropy residual rule changed")
    if checkpoint.get("components") != {
        "selection_data": "BCI2a Session T development only",
        "atcnet_final_epoch": 79,
        "atcnet_scheduler_horizon": 300,
        "fbcnet_final_epoch": 70,
        "fbcnet_scheduler_horizon": 300,
        "decoder_final_epoch": 20,
        "decoder_scheduler_horizon": 120,
        "heldout_checkpoint_selection": False,
    }:
        raise RuntimeError("frozen component checkpoint rules changed")
    if baselines.get("models") != ["atcnet", "fbcnet"]:
        raise RuntimeError("frozen component baseline set changed")
    return freeze
