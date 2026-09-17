"""Static dry-run validation for the four-GPU revision launch graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"launch plan references missing files: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    configs = ROOT / "configs" / "experiments"
    scripts = ROOT / "scripts"
    required = [
        scripts / "cloud" / "run_revision_recovery_4gpu.sh",
        scripts / "launch_publication_baselines.py",
        scripts / "launch_v30_bnci2014_004.py",
        scripts / "launch_v31_learning_curve.py",
        scripts / "launch_reviewer_controls.py",
        scripts / "launch_bci2a_binary_sensitivity.py",
        scripts / "monitor_revision_campaign.py",
        scripts / "validate_revision_campaign.py",
        configs / "v8_publication_baselines.yaml",
        configs / "v30_bnci2014_004_recovery.yaml",
        configs / "v31_decoder_learning_curve.yaml",
        configs / "reviewer_controls.yaml",
        configs / "bci2a_binary_sensitivity.yaml",
    ]
    _require_files(required)
    publication = yaml.safe_load(
        (configs / "v8_publication_baselines.yaml").read_text(encoding="utf-8")
    )
    v30 = yaml.safe_load((configs / "v30_bnci2014_004_recovery.yaml").read_text(encoding="utf-8"))
    v31 = yaml.safe_load((configs / "v31_decoder_learning_curve.yaml").read_text(encoding="utf-8"))
    reviewer = yaml.safe_load((configs / "reviewer_controls.yaml").read_text(encoding="utf-8"))
    binary = yaml.safe_load((configs / "bci2a_binary_sensitivity.yaml").read_text(encoding="utf-8"))
    recovered_models = ["atcnet", "fbcnet"]
    recovered_seeds = [0, 1, 2]
    frontend_tasks = sum(
        len(value["subjects"]) for value in publication["datasets"].values()
    ) * len(recovered_models) * len(recovered_seeds) + len(v30["dataset"]["subjects"]) * len(
        v30["teacher_models"]
    ) * len(v30["seeds"])
    v31_budget_count = {"bci2a": 3, "openbmi": 2, "bnci2014_004": 4}
    v31_tasks = sum(
        len(dataset_config["subjects"])
        * len(v31["seeds"])
        * v31_budget_count[dataset]
        * len(v31["variants"])
        for dataset, dataset_config in v31["datasets"].items()
    )
    reviewer_tasks = 0
    for dataset_config in reviewer["datasets"].values():
        for spec in reviewer["variants"].values():
            budgets = (
                reviewer["fixed_budget_controls"]
                if spec["budget_group"] == "fixed_budget_controls"
                else dataset_config["equal_update_budgets"]
            )
            reviewer_tasks += (
                len(dataset_config["subjects"]) * len(reviewer["seeds"]) * len(budgets)
            )
    binary_tasks = (
        len(binary["dataset"]["subjects"])
        * len(binary["seeds"])
        * (len(binary["numeric_budgets_per_class"]) + 1)
        * len(binary["variants"])
    )
    if (frontend_tasks, v31_tasks, reviewer_tasks, binary_tasks) != (432, 1026, 2322, 162):
        raise RuntimeError(
            "frozen workload drift: "
            f"frontends={frontend_tasks}, v31={v31_tasks}, reviewer={reviewer_tasks}, binary={binary_tasks}"
        )
    total = frontend_tasks + v31_tasks + reviewer_tasks + binary_tasks
    payload = {
        "schema": "dpc-snn-revision-launch-dry-run/v1",
        "status": "PASS",
        "gpu_queue": [0, 1, 2, 3],
        "frontend_trainings": frontend_tasks,
        "v31_parent_trainings": v31_tasks,
        "reviewer_trainings_including_rev_e5": reviewer_tasks + binary_tasks,
        "total_trainings": total,
        "expected_total": 3942,
        "global_barriers": ["publication", "v30", "v31", "reviewer", "bci2a_binary"],
        "resume_policy": "stage_barriers_plus_per-run_fingerprint_validation",
    }
    if total != 3942:
        raise RuntimeError(f"total workload drift: {total}")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
