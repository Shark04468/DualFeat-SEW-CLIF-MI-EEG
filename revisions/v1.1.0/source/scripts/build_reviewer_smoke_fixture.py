"""Build a deterministic synthetic parent tree for reviewer-control dry runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.v9_dual_feature_training import fit_feature_standardizer
from dpc_snn.experiments.v62_protocol import sha256_fingerprint
from dpc_snn.utils.io import write_json


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def save_standardizer(path: Path, indices: np.ndarray, atc: np.ndarray, fbc: np.ndarray) -> None:
    value = fit_feature_standardizer(atc[indices], fbc[indices])
    atomic_npz(
        path,
        atc_mean=value.atc_mean,
        atc_scale=value.atc_scale,
        fbc_mean=value.fbc_mean,
        fbc_scale=value.fbc_scale,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    parent = root / "parent"
    label_root = root / "labels"
    output = root / "output"
    rng = np.random.default_rng(20260901)
    train_atc = rng.normal(size=(12, 18, 32)).astype(np.float32)
    train_fbc = rng.normal(size=(12, 4, 288)).astype(np.float32)
    eval_atc = rng.normal(size=(10, 18, 32)).astype(np.float32)
    eval_fbc = rng.normal(size=(10, 4, 288)).astype(np.float32)
    train_y = (np.arange(12) % 2).astype(np.int64)
    eval_y = (np.arange(10) % 2).astype(np.int64)
    feature_root = parent / "bci2a" / "subject_01" / "seed_0" / "feature_cache"
    atomic_npz(
        feature_root / "training.npz",
        atc=train_atc,
        fbc=train_fbc,
        teacher=np.zeros((12, 2), dtype=np.float32),
    )
    atomic_npz(
        feature_root / "evaluation.npz",
        atc=eval_atc,
        fbc=eval_fbc,
        teacher=np.zeros((10, 2), dtype=np.float32),
    )
    label_path = label_root / "bci2a" / "subject_01" / "training_labels.npz"
    atomic_npz(label_path, label=train_y)
    write_json(
        label_path.with_suffix(".json"),
        {
            "schema": "dpc-snn-recovery-labels/v1",
            "dataset": "bci2a",
            "subject": 1,
            "label_identity_sha256": sha256_fingerprint(
                {"shape": list(train_y.shape), "values": train_y.tolist()}
            ),
        },
    )
    subsets = {"n25": np.arange(8, dtype=np.int64), "all": np.arange(12, dtype=np.int64)}
    for budget, indices in subsets.items():
        directory = parent / "bci2a" / "subject_01" / "seed_0" / f"budget_{budget}"
        atomic_npz(directory / "subset.npz", indices=indices)
        write_json(
            directory / "subset.json",
            {
                "label": budget,
                "total_examples": int(indices.size),
                "examples_per_class": float(indices.size / 2),
                "class_counts": {
                    "0": int((train_y[indices] == 0).sum()),
                    "1": int((train_y[indices] == 1).sum()),
                },
            },
        )
        save_standardizer(directory / "standardizer.npz", indices, train_atc, train_fbc)
        prediction_path = directory / "ann_sew_ce" / "evaluation" / "predictions.npz"
        atomic_npz(
            prediction_path,
            logits=np.zeros((10, 2), dtype=np.float32),
            pred=np.zeros(10, dtype=np.int64),
            label=eval_y,
            trial_id=np.asarray([f"fixture_{index:03d}" for index in range(10)]),
        )
    config = {
        "experiment_id": "REVIEWER_CONTROL_SYNTHETIC_SMOKE",
        "schema": "dpc-snn-reviewer-controls/v1",
        "status": "synthetic_smoke_only",
        "lineage": "SYNTHETIC_SMOKE_NOT_PAPER_EVIDENCE",
        "parents": {
            "e31": str(parent),
            "publication": str(root / "unused_publication"),
            "v30": str(root / "unused_v30"),
            "baseline_source": str(root / "unused_sources"),
            "training_labels": str(label_root),
        },
        "datasets": {
            "bci2a": {
                "subjects": [1],
                "n_classes": 2,
                "equal_update_budgets": ["n25", "all"],
            }
        },
        "seeds": [0],
        "fixed_budget_controls": ["n25", "all"],
        "variants": {
            "soft_clif_exact": {
                "model": "soft_clif_exact",
                "experiment": "REV-E1",
                "budget_group": "fixed_budget_controls",
                "equal_update": False,
                "firing_rate_weight": 0.0,
                "gradient": "analytic_sigmoid_autograd",
            },
            "gru_stat": {
                "model": "gru_stat",
                "experiment": "REV-E2",
                "budget_group": "fixed_budget_controls",
                "equal_update": False,
                "firing_rate_weight": 0.0,
                "temporal_width": 31,
            },
            "tcn_stat": {
                "model": "tcn_stat",
                "experiment": "REV-E2",
                "budget_group": "fixed_budget_controls",
                "equal_update": False,
                "firing_rate_weight": 0.0,
                "temporal_width": 28,
            },
            "ann_sew_equal": {
                "model": "ann_sew",
                "experiment": "REV-E3",
                "budget_group": "equal_update_budgets",
                "equal_update": True,
                "firing_rate_weight": 0.0,
            },
            "sew_clif_equal": {
                "model": "sew_clif",
                "experiment": "REV-E3",
                "budget_group": "equal_update_budgets",
                "equal_update": True,
                "firing_rate_weight": 0.01,
            },
        },
        "training": {
            "fixed_epochs": 80,
            "smoke_epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.001,
            "weight_decay": 0.001,
            "gradient_clip_norm": 5.0,
            "dropout": 0.25,
            "soft_gate_slope": 10.0,
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "smoke_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump({"root": str(root), "config": str(config_path), "output": str(output)}))


if __name__ == "__main__":
    main()
