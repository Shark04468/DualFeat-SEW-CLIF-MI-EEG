#!/usr/bin/env python3
"""Run parameter-matched ANN/PLIF/CLIF/SEW-CLIF Session-T controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dpc_snn.data.bci2a import load_processed_npz  # noqa: E402
from dpc_snn.experiments.v62_protocol import (  # noqa: E402
    file_sha256,
    write_run_artifact_manifest,
)
from dpc_snn.experiments.v8_baselines import session_t_development_view  # noqa: E402
from dpc_snn.experiments.v8_protocol import (  # noqa: E402
    collect_source_tree_manifest,
    source_tree_digest,
    v8_heldout_lock_manifest,
)
from dpc_snn.experiments.v8_training import seed_v8  # noqa: E402
from dpc_snn.models.build import build_model  # noqa: E402
from dpc_snn.models.v8_accuracy_first import V8AccuracyFirstModel  # noqa: E402
from dpc_snn.utils.io import ensure_dir, write_csv, write_json  # noqa: E402
from dpc_snn.utils.storage import configure_cache_env  # noqa: E402
from scripts.run_v8_e2_zero_delay import (  # noqa: E402
    SubjectBundle,
    _csv,
    _environment,
    _load_or_build_base_rates,
    _nested_folds,
    _run_one,
    _split_manifest,
    _subject_file,
    _variant_smoke,
)


EXPECTED_VARIANTS: dict[str, dict[str, Any]] = {
    "ann_residual": {
        "decoder_kind": "ann",
        "decoder_residual_mode": "sew_add",
        "delay_auxiliary_enabled": False,
    },
    "plif_plain": {
        "decoder_kind": "plif",
        "decoder_residual_mode": "plain",
        "delay_auxiliary_enabled": False,
    },
    "clif_plain": {
        "decoder_kind": "clif",
        "decoder_residual_mode": "plain",
        "delay_auxiliary_enabled": False,
    },
    "sew_clif": {
        "decoder_kind": "clif",
        "decoder_residual_mode": "sew_add",
        "delay_auxiliary_enabled": False,
    },
}

CAMPAIGN_REQUIRED_FILES = (
    "manifest.json",
    "campaign_status.json",
    "summary.csv",
    "capacity_audit.json",
    "variant_smoke.json",
    "source_tree_manifest.json",
    "source_tree_summary.json",
    "heldout_lock_manifest.json",
    "shared_cache_provenance.json",
    "resolved_campaign.yaml",
)


def _build_e4_variant(
    model_config: dict[str, Any],
    variant_name: str,
    overrides: dict[str, Any],
    *,
    seed: int,
) -> V8AccuracyFirstModel:
    expected = EXPECTED_VARIANTS.get(variant_name)
    if expected is None or dict(overrides) != expected:
        raise RuntimeError(f"E4 variant contract mismatch for {variant_name!r}")
    seed_v8(seed)
    model = build_model("v8_accuracy_first", {**model_config, **overrides})
    if not isinstance(model, V8AccuracyFirstModel):
        raise TypeError("V8 E4 model factory returned an unexpected model")
    if model.delay_auxiliary_enabled or model.delay_auxiliary is not None:
        raise RuntimeError(f"E4 variant {variant_name!r} enabled a delay branch")
    if model.decoder is None:
        raise RuntimeError(f"E4 variant {variant_name!r} disabled the temporal decoder")
    if model.decoder.decoder_kind != expected["decoder_kind"]:
        raise RuntimeError(f"E4 variant {variant_name!r} has the wrong decoder kind")
    if model.decoder.decoder_residual_mode != expected["decoder_residual_mode"]:
        raise RuntimeError(f"E4 variant {variant_name!r} has the wrong residual mode")
    return model


def _shape_signature(model: V8AccuracyFirstModel) -> list[list[int]]:
    return sorted(
        [list(parameter.shape) for parameter in model.parameters() if parameter.requires_grad],
        key=lambda shape: (len(shape), shape),
    )


def _capacity_audit(
    model_config: dict[str, Any], variants: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for index, (name, overrides) in enumerate(variants.items()):
        model = _build_e4_variant(model_config, name, overrides, seed=index)
        rows[name] = {
            "parameters": model.parameter_count,
            "trainable_parameters": model.trainable_parameter_count,
            "trainable_parameter_shape_multiset": _shape_signature(model),
            "decoder_kind": model.decoder.decoder_kind if model.decoder else None,
            "decoder_residual_mode": (
                model.decoder.decoder_residual_mode if model.decoder else None
            ),
            "physical_frontend_fingerprint": model.physical_frontend_fingerprint(),
        }
    counts = {row["parameters"] for row in rows.values()}
    trainable = {row["trainable_parameters"] for row in rows.values()}
    signatures = {
        json.dumps(row["trainable_parameter_shape_multiset"], separators=(",", ":"))
        for row in rows.values()
    }
    frontends = {row["physical_frontend_fingerprint"] for row in rows.values()}
    passed = len(counts) == len(trainable) == len(signatures) == len(frontends) == 1
    if not passed:
        raise RuntimeError("E4 decoder controls are not exactly capacity/front-end matched")
    return {
        "status": "passed",
        "comparison_scope": "identical V8 front end, branches, optimizer budget and readout",
        "permitted_differences": ["decoder_kind", "decoder_residual_mode"],
        "parameter_names_not_required_to_match": True,
        "parameter_shape_multiset_and_count_required_to_match": True,
        "variants": rows,
    }


def _external_cache_ready(root: Path, subjects: list[int]) -> None:
    for subject in subjects:
        directory = root / "shared_physical_rates" / f"subject_{subject:02d}"
        for name in ("unit_gain_rates.pt", "manifest.json"):
            if not (directory / name).is_file():
                raise FileNotFoundError(
                    f"external physical cache is incomplete for subject {subject}: {directory / name}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default="configs/experiments/v8_e4_decoder_controls.yaml"
    )
    parser.add_argument("--model-config", default="configs/models/v8_accuracy_first.yaml")
    parser.add_argument("--physical-cache-root", default="")
    parser.add_argument("--variants", default="")
    parser.add_argument("--subjects", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--source-smoke-only", action="store_true")
    args = parser.parse_args()

    configure_cache_env()
    output = ensure_dir(Path(args.output).resolve())
    data_root = Path(args.data).resolve()
    config = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    model_config = yaml.safe_load(
        Path(args.model_config).resolve().read_text(encoding="utf-8")
    )
    if config.get("stage") != "development" or config["data_access"].get(
        "heldout_session_e_accessed"
    ):
        raise RuntimeError("V8 E4 must keep Session E locked")
    if dict(config["variants"]) != EXPECTED_VARIANTS:
        raise RuntimeError("V8 E4 config does not contain the exact registered four arms")
    if config.get("reference_variant") != "ann_residual":
        raise RuntimeError("V8 E4 reference arm must be ann_residual")

    variants = dict(config["variants"])
    if args.variants:
        selected = _csv(args.variants)
        unknown = sorted(set(selected) - set(variants))
        if unknown:
            raise ValueError(f"unknown V8 E4 variants: {unknown}")
        variants = {name: variants[name] for name in selected}
    subjects = _csv(args.subjects, int) if args.subjects else list(config["subjects"])
    seeds = _csv(args.seeds, int) if args.seeds else list(config["seeds"])
    if args.max_epochs is not None:
        config["selection"]["max_epochs"] = int(args.max_epochs)
        config["selection"]["minimum_epochs"] = min(
            int(config["selection"]["minimum_epochs"]), int(args.max_epochs)
        )
        config["selection"]["minimum_outer_retrain_epochs"] = min(
            int(config["selection"]["minimum_outer_retrain_epochs"]),
            int(args.max_epochs),
        )
    if args.patience is not None:
        config["selection"]["patience"] = int(args.patience)
    full_contract = bool(
        variants == EXPECTED_VARIANTS
        and subjects == list(config["subjects"])
        and seeds == list(config["seeds"])
        and args.max_epochs is None
        and args.patience is None
    )
    if not full_contract and not args.canary and not args.source_smoke_only:
        raise RuntimeError("partial E4 execution requires --canary and cannot pass the gate")
    if int(config["training"]["effective_batch_size"]) != int(
        config["training"]["batch_size"]
    ) * int(config["training"]["gradient_accumulation_steps"]):
        raise RuntimeError("V8 E4 effective batch size metadata is inconsistent")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")

    source_tree = collect_source_tree_manifest(ROOT)
    environment = _environment()
    write_json(output / "source_tree_manifest.json", source_tree)
    write_json(
        output / "source_tree_summary.json",
        {"files": len(source_tree), "sha256": source_tree_digest(source_tree)},
    )
    write_json(output / "heldout_lock_manifest.json", v8_heldout_lock_manifest())
    (output / "resolved_campaign.yaml").write_text(
        yaml.safe_dump(
            {
                **config,
                "active_variants": list(variants),
                "active_subjects": subjects,
                "active_seeds": seeds,
                "canary": bool(args.canary),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    capacity = _capacity_audit(model_config, dict(config["variants"]))
    write_json(output / "capacity_audit.json", capacity)
    _variant_smoke(
        output,
        model_config,
        variants,
        args.device,
        build_variant_fn=_build_e4_variant,
        stage="E4",
    )
    if args.source_smoke_only:
        print(json.dumps({"status": "passed", "source_smoke_only": True}, indent=2))
        return

    cache_root = output
    external_cache = bool(args.physical_cache_root)
    if external_cache:
        cache_root = Path(args.physical_cache_root).resolve()
        _external_cache_ready(cache_root, subjects)
    write_json(
        output / "shared_cache_provenance.json",
        {
            "external_read_only_cache": external_cache,
            "root": str(cache_root),
            "physical_frontend_fingerprint": next(
                iter(capacity["variants"].values())
            )["physical_frontend_fingerprint"],
        },
    )

    bundles: dict[int, SubjectBundle] = {}
    for subject in subjects:
        subject_path = _subject_file(data_root, int(subject))
        data_sha256 = file_sha256(subject_path)
        data = load_processed_npz(subject_path)
        x, y, metadata, access_manifest = session_t_development_view(data)
        nested = _nested_folds(
            metadata,
            n_splits=int(config["selection"]["n_splits"]),
            split_seed=int(config["selection"]["split_seed"]),
        )
        split_manifest = _split_manifest(
            metadata,
            nested,
            subject=int(subject),
            split_seed=int(config["selection"]["split_seed"]),
        )
        reference = _build_e4_variant(
            model_config,
            "ann_residual",
            EXPECTED_VARIANTS["ann_residual"],
            seed=0,
        )
        base_rates, cache_manifest = _load_or_build_base_rates(
            output=cache_root,
            subject=int(subject),
            subject_path=subject_path,
            data_sha256=data_sha256,
            x=x,
            metadata=metadata,
            model=reference,
            preprocessing=dict(config["preprocessing"]),
            canary_train_indices=nested[0][0],
            device=args.device,
        )
        bundles[int(subject)] = SubjectBundle(
            subject_path=subject_path,
            data_sha256=data_sha256,
            x=x,
            y=y,
            metadata=metadata,
            access_manifest=access_manifest,
            nested_folds=nested,
            split_manifest=split_manifest,
            base_rates=base_rates,
            cache_manifest=cache_manifest,
        )
        del data, reference
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    for variant_name, overrides in variants.items():
        for seed in seeds:
            for subject in subjects:
                rows.append(
                    _run_one(
                        output=output,
                        config=config,
                        model_config=model_config,
                        source_tree=source_tree,
                        environment=environment,
                        bundle=bundles[int(subject)],
                        variant_name=variant_name,
                        variant_overrides=overrides,
                        seed=int(seed),
                        device=args.device,
                        stage="E4",
                        build_variant_fn=_build_e4_variant,
                    )
                )
                write_csv(output / "summary.csv", rows)
    rows = sorted(
        rows, key=lambda row: (row["variant"], int(row["subject"]), int(row["seed"]))
    )
    write_csv(output / "summary.csv", rows)
    status = {
        "status": "completed" if full_contract else "canary_completed",
        "stage": "E4",
        "protocol": config["protocol"],
        "variants": list(variants),
        "registered_variants": list(EXPECTED_VARIANTS),
        "subjects": subjects,
        "seeds": seeds,
        "runs": len(rows),
        "full_registered_contract": full_contract,
        "capacity_audit_passed": capacity["status"] == "passed",
        "session_e_accessed": False,
        "openbmi_s2_accessed": False,
        "source_tree_sha256": source_tree_digest(source_tree),
    }
    write_json(output / "campaign_status.json", status)
    write_run_artifact_manifest(output, required_files=CAMPAIGN_REQUIRED_FILES)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
