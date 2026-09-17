"""Baseline fairness audit helpers."""

from __future__ import annotations

from typing import Any


FAIRNESS_FIELDS = [
    "model",
    "code_source",
    "split",
    "preprocess",
    "seed",
    "status",
    "accuracy",
    "balanced_accuracy",
    "kappa",
    "macro_f1",
    "hparam_budget",
    "params",
    "trainable_params",
    "train_seconds",
    "inference_ms",
    "peak_memory_mb",
    "notes",
]


def audit_row(model: str, cfg: dict[str, Any], metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics = metrics or {}
    return {
        "model": model,
        "code_source": cfg.get("code_source", "local"),
        "split": cfg.get("split", "shared"),
        "preprocess": cfg.get("preprocess", "shared"),
        "seed": cfg.get("seed", ""),
        "status": cfg.get("status", metrics.get("status", "completed" if metrics else "not_run")),
        "accuracy": metrics.get("accuracy", ""),
        "balanced_accuracy": metrics.get("balanced_accuracy", ""),
        "kappa": metrics.get("kappa", ""),
        "macro_f1": metrics.get("macro_f1", ""),
        "hparam_budget": cfg.get("hparam_budget", 20),
        "params": metrics.get("params", ""),
        "trainable_params": metrics.get("trainable_params", ""),
        "train_seconds": metrics.get("train_seconds", ""),
        "inference_ms": metrics.get("inference_ms", ""),
        "peak_memory_mb": metrics.get("peak_memory_mb", ""),
        "notes": cfg.get("notes", ""),
    }
