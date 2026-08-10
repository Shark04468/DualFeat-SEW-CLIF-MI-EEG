from __future__ import annotations

from scripts.derive_v8_selected_e4_config import derive


def test_selected_e4_inherits_every_training_semantic_from_selected_e2() -> None:
    selected = {
        "experiment_id": "selected",
        "architecture_version": "v8",
        "stage": "development",
        "training": {"learning_rate": 0.002, "batch_size": 4},
        "augmentation": {"enabled": True, "segments": 8},
        "preprocessing": {"gain": "fold_train"},
        "selection": {
            "max_epochs": 101,
            "patience": 17,
            "minimum_epochs": 23,
            "minimum_outer_retrain_epochs": 23,
        },
        "hpo_provenance": {
            "candidate_id": "c17",
            "selection_scope": "Session-T inner-validation only",
        },
        "data_access": {"heldout_session_e_accessed": False},
    }
    base = {
        "stage": "development",
        "selection": {
            "max_epochs": 200,
            "patience": 30,
            "minimum_epochs": 30,
            "minimum_outer_retrain_epochs": 30,
            "n_splits": 6,
        },
        "training": {},
        "augmentation": {},
        "preprocessing": {},
    }
    resolved = derive(selected, base)
    assert resolved["training"] == selected["training"]
    assert resolved["augmentation"] == selected["augmentation"]
    assert resolved["preprocessing"] == selected["preprocessing"]
    assert resolved["selection"]["n_splits"] == 6
    assert resolved["selection"]["max_epochs"] == 101
    assert resolved["selected_e2_binding"]["candidate_id"] == "c17"
