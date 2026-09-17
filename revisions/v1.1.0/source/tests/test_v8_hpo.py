from __future__ import annotations

from collections import Counter
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from dpc_snn.experiments.v8_hpo import (
    V8HPOContractError,
    generate_balanced_v8_candidates,
    rank_v8_hpo_candidates,
)
from scripts.run_v8_e5_bounded_hpo import (
    _build_candidate,
    _run_worker_command,
    _selected_outputs,
    _worker_command,
)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/experiments/v8_e5_bounded_hpo.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_e5_candidate_factorial_is_complete_balanced_and_order_deterministic() -> None:
    first = generate_balanced_v8_candidates(_config())
    second = generate_balanced_v8_candidates(_config())
    assert first == second
    assert len(first) == 24
    assert len({row["candidate_id"] for row in first}) == 24
    for factor, levels in _config()["candidate_factors"].items():
        counts = Counter(row["factor_levels"][factor] for row in first)
        assert set(counts) == set(levels)
        assert len(set(counts.values())) == 1
    assert all(row["model_overrides"]["decoder_kind"] == "ann" for row in first)
    assert all(row["training_overrides"].keys() == {"learning_rate"} for row in first)
    model_config = yaml.safe_load(
        (ROOT / "configs/models/v8_accuracy_first.yaml").read_text(encoding="utf-8")
    )
    counts = [_build_candidate(model_config, row, seed=0).parameter_count for row in first]
    assert max(counts) <= 700_000


def test_e5_factorial_may_not_be_truncated_by_candidate_order() -> None:
    config = _config()
    config["maximum_unique_configurations"] = 20
    with pytest.raises(V8HPOContractError, match="above limit"):
        generate_balanced_v8_candidates(config)


def test_e5_ranking_uses_exact_subject_fold_coverage() -> None:
    candidates = ["a", "b"]
    subjects = [1, 3]
    folds = [0, 2]
    rows = []
    for candidate in candidates:
        for subject in subjects:
            for fold in folds:
                rows.append(
                    {
                        "candidate_id": candidate,
                        "subject": subject,
                        "fold": fold,
                        "evaluation_role": "inner_validation",
                        "validation_kappa": 0.7 if candidate == "b" else 0.6,
                        "validation_accuracy": 0.75 if candidate == "b" else 0.65,
                        "parameters": 20 if candidate == "b" else 10,
                        "best_epoch": 5,
                    }
                )
    ranking = rank_v8_hpo_candidates(
        rows, candidate_ids=candidates, subjects=subjects, folds=folds
    )
    assert [row["candidate_id"] for row in ranking] == ["b", "a"]

    with pytest.raises(V8HPOContractError, match="coverage mismatch"):
        rank_v8_hpo_candidates(
            rows[:-1], candidate_ids=candidates, subjects=subjects, folds=folds
        )


def test_e5_ranking_rejects_outer_test_metrics() -> None:
    with pytest.raises(V8HPOContractError, match="non-inner-validation"):
        rank_v8_hpo_candidates(
            [
                {
                    "candidate_id": "a",
                    "subject": 1,
                    "fold": 0,
                    "evaluation_role": "outer_test",
                    "validation_kappa": 0.5,
                    "validation_accuracy": 0.6,
                    "parameters": 10,
                    "best_epoch": 2,
                }
            ],
            candidate_ids=["a"],
            subjects=[1],
            folds=[0],
        )


def test_e5_selected_output_requires_fresh_e2_oof(tmp_path: Path) -> None:
    candidate = generate_balanced_v8_candidates(_config())[0]
    model_config = yaml.safe_load(
        (ROOT / "configs/models/v8_accuracy_first.yaml").read_text(encoding="utf-8")
    )
    e2_config = yaml.safe_load(
        (ROOT / "configs/experiments/v8_e2_zero_delay.yaml").read_text(
            encoding="utf-8"
        )
    )
    _selected_outputs(
        output=tmp_path,
        selected=candidate,
        model_config=model_config,
        e2_config=e2_config,
    )
    selected_model = yaml.safe_load(
        (tmp_path / "selected_model.yaml").read_text(encoding="utf-8")
    )
    selected_e2 = yaml.safe_load(
        (tmp_path / "selected_e2_config.yaml").read_text(encoding="utf-8")
    )
    assert selected_model["parameter_ceiling"] == 700_000
    assert selected_e2["required_confirmation_variants"] == [
        "hpo_selected_full_ann"
    ]
    assert selected_e2["training"]["learning_rate"] == candidate[
        "training_overrides"
    ]["learning_rate"]
    assert not selected_e2["hpo_provenance"]["outer_test_accessed_for_selection"]


def test_e5_parallel_worker_command_preserves_exact_fold_identity(tmp_path: Path) -> None:
    args = Namespace(
        data=str(tmp_path / "data"),
        config=str(ROOT / "configs/experiments/v8_e5_bounded_hpo.yaml"),
        model_config=str(ROOT / "configs/models/v8_accuracy_first.yaml"),
        e2_config=str(ROOT / "configs/experiments/v8_e2_zero_delay.yaml"),
        physical_cache_root=str(tmp_path / "cache"),
        device="cuda",
        max_epochs=None,
    )
    command = _worker_command(
        args=args,
        output=tmp_path / "campaign",
        stage_name="stage_2",
        subject=3,
        candidate_ids=["hpo_a", "hpo_b"],
        folds=[0, 2, 4],
    )
    joined = " ".join(command)
    assert "--worker-stage stage_2" in joined
    assert "--worker-subject 3" in joined
    assert "--worker-candidate-ids hpo_a,hpo_b" in joined
    assert "--worker-folds 0,2,4" in joined
    assert "--canary" not in command


def test_e5_worker_rejects_invalid_cpu_thread_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DPC_SNN_E5_WORKER_THREADS", "0")
    with pytest.raises(ValueError, match="between 1 and 32"):
        _run_worker_command(["unused"], [tmp_path / "result.json"])
