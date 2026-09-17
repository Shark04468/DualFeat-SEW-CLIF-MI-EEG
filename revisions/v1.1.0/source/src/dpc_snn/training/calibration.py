"""Few-shot and unlabeled calibration helpers."""

from __future__ import annotations

from typing import Iterable


def set_trainable_by_mode(model, mode: str) -> int:
    """Freeze parameters according to the calibration mode.

    Modes:
        readout: only classifier/readout layers
        delay: shared delay logits and the residual delay gate
        phase_delay: delay parameters and bounded residual phase parameters
        full: all parameters
    """

    mode = mode.lower()
    if hasattr(model, "set_calibration_mode"):
        return int(model.set_calibration_mode(mode))
    for _, p in model.named_parameters():
        p.requires_grad = mode == "full"

    keywords: Iterable[str]
    if mode == "readout":
        keywords = ("readout", "classifier", "feature")
    elif mode == "delay":
        keywords = ("delay_logits", "delay_gate_logits")
    elif mode == "phase_delay":
        keywords = ("delay_logits", "delay_gate_logits", "phase_pref", "theta", "beta")
    elif mode == "full":
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        raise ValueError(f"Unknown calibration mode: {mode}")

    for name, p in model.named_parameters():
        if any(key in name for key in keywords):
            p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
