"""Profile REV-E4 dense-MAC proxies and synchronized GPU latency."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.experiments.reviewer_controls import (
    benchmark_latency,
    build_reviewer_model,
    dense_mac_proxy,
)
from dpc_snn.experiments.v62_protocol import file_sha256
from dpc_snn.utils.io import ensure_dir, write_json
from scripts.run_v31_decoder_learning_curve_subject import _load_standardizer


def _cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return (
            np.asarray(archive["atc"], dtype=np.float32),
            np.asarray(archive["fbc"], dtype=np.float32),
        )


def _checkpoint_model(
    path: Path,
    variant: str,
    n_classes: int,
    config: dict[str, Any],
) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_reviewer_model(
        variant,
        n_classes=n_classes,
        dropout=float(config["training"]["dropout"]),
        soft_gate_slope=float(config["training"]["soft_gate_slope"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "reviewer_controls.yaml"),
    )
    parser.add_argument("--reviewer-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", default="bci2a,openbmi")
    parser.add_argument("--budget", default="n25")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("REV-E4 publication latency requires a CUDA GPU")
    config = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    parent = Path(config["parents"]["e31"]).resolve()
    reviewer = Path(args.reviewer_root).resolve()
    output = ensure_dir(Path(args.output).resolve())
    hardware = {
        "schema": "dpc-snn-rev-e4-hardware-manifest/v1",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "batch_size": 1,
        "atc_steps": 18,
        "warmup": int(args.warmup),
        "repetitions": int(args.repetitions),
        "cuda_synchronized": True,
        "energy_claim_supported": False,
    }
    write_json(output / "hardware_manifest.json", hardware)
    summaries: list[dict[str, Any]] = []
    for dataset in [value.strip() for value in args.datasets.split(",") if value.strip()]:
        dataset_config = config["datasets"][dataset]
        subject = int(dataset_config["subjects"][0])
        n_classes = int(dataset_config["n_classes"])
        feature_root = (
            parent / dataset / f"subject_{subject:02d}" / f"seed_{args.seed}" / "feature_cache"
        )
        atc, fbc = _cache(feature_root / "evaluation.npz")
        budget_root = (
            parent
            / dataset
            / f"subject_{subject:02d}"
            / f"seed_{args.seed}"
            / f"budget_{args.budget}"
        )
        standardizer = _load_standardizer(budget_root / "standardizer.npz")
        atc, fbc = standardizer.transform(atc[:1], fbc[:1])
        tensor_atc = torch.from_numpy(atc).to(args.device)
        tensor_fbc = torch.from_numpy(fbc).to(args.device)
        candidates = {
            "ann_sew": budget_root / "ann_sew_ce" / "checkpoint.pt",
            "sew_clif": budget_root / "sew_clif_ce" / "checkpoint.pt",
            "soft_clif_exact": (
                reviewer
                / dataset
                / f"subject_{subject:02d}"
                / f"seed_{args.seed}"
                / f"budget_{args.budget}"
                / "soft_clif_exact"
                / "checkpoint.pt"
            ),
        }
        for variant, checkpoint_path in candidates.items():
            if not checkpoint_path.is_file():
                if variant == "soft_clif_exact":
                    continue
                raise FileNotFoundError(checkpoint_path)
            model = _checkpoint_model(checkpoint_path, variant, n_classes, config)
            operations = dense_mac_proxy(model, tensor_atc, tensor_fbc)
            timing = benchmark_latency(
                model,
                tensor_atc,
                tensor_fbc,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
            raw = np.asarray(timing.pop("samples_ms"), dtype=np.float64)
            raw_path = output / f"latency_{dataset}_{variant}.npz"
            with raw_path.open("wb") as handle:
                np.savez_compressed(handle, samples_ms=raw)
            summaries.append(
                {
                    "dataset": dataset,
                    "n_classes": n_classes,
                    "subject": subject,
                    "seed": args.seed,
                    "budget": args.budget,
                    "variant": variant,
                    "checkpoint_sha256": file_sha256(checkpoint_path),
                    "raw_timing_sha256": file_sha256(raw_path),
                    "dense_mac_proxy": operations,
                    "latency": timing,
                }
            )
            del model
            torch.cuda.empty_cache()
    payload = {
        "schema": "dpc-snn-rev-e4-utility-profile/v1",
        "status": "completed",
        "hardware": hardware,
        "profiles": summaries,
        "interpretation": "GPU latency and dense-operation proxies do not establish neuromorphic energy efficiency or closed-loop clinical latency.",
    }
    write_json(output / "utility_profile.json", payload)
    print(json.dumps({"status": "completed", "profiles": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
