import numpy as np

from dpc_snn.experiments.v31_learning_curve import (
    nested_stratified_subsets,
    paired_run_seed,
)


def test_nested_subsets_skip_unavailable_and_duplicate_full_budget() -> None:
    labels = np.repeat(np.arange(2), 50)
    subsets = nested_stratified_subsets(labels, [25, 50, 100], seed=7)
    assert [subset.label for subset in subsets] == ["n25", "all"]
    assert subsets[0].indices.size == 50
    assert subsets[1].indices.size == 100
    assert set(subsets[0].indices).issubset(set(subsets[1].indices))


def test_nested_subsets_keep_unequal_full_set() -> None:
    labels = np.concatenate([np.zeros(50, dtype=int), np.ones(55, dtype=int)])
    subsets = nested_stratified_subsets(labels, [25, 50, 100], seed=3)
    assert [subset.label for subset in subsets] == ["n25", "n50", "all"]
    assert subsets[-1].class_counts == {0: 50, 1: 55}
    assert subsets[-1].examples_per_class == 52.5


def test_sampling_is_reproducible_and_seeded() -> None:
    labels = np.repeat(np.arange(4), 72)
    first = nested_stratified_subsets(labels, [25, 50, 100], seed=11)
    replay = nested_stratified_subsets(labels, [25, 50, 100], seed=11)
    other = nested_stratified_subsets(labels, [25, 50, 100], seed=12)
    assert [value.index_sha256 for value in first] == [
        value.index_sha256 for value in replay
    ]
    assert first[0].index_sha256 != other[0].index_sha256
    assert [value.label for value in first] == ["n25", "n50", "all"]


def test_paired_seed_ignores_budget_and_variant_by_construction() -> None:
    assert paired_run_seed(1, 8, 2) == paired_run_seed(1, 8, 2)
    assert paired_run_seed(0, 8, 2) != paired_run_seed(1, 8, 2)
