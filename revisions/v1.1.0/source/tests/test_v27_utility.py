import numpy as np

from dpc_snn.experiments.v27_utility import mask_random_time_windows, utility_gate


def test_time_mask_is_deterministic_and_has_registered_width() -> None:
    carrier = np.ones((3, 2, 100), dtype=np.float32)
    first, starts = mask_random_time_windows(
        carrier, duration_seconds=0.2, sfreq=100.0, seed=7
    )
    second, repeated = mask_random_time_windows(
        carrier, duration_seconds=0.2, sfreq=100.0, seed=7
    )
    assert np.array_equal(first, second)
    assert np.array_equal(starts, repeated)
    assert np.all((first == 0.0).sum(axis=(1, 2)) == 40)


def test_utility_gate_requires_clean_replay_and_one_win() -> None:
    assert utility_gate(
        clean_replay_valid=True,
        early_auc_gain_pp=0.6,
        robustness_gain_pp=0.0,
        decoder_reduction=0.0,
        full_student_reduction=0.0,
    )["status"] == "pass"
    assert utility_gate(
        clean_replay_valid=False,
        early_auc_gain_pp=1.0,
        robustness_gain_pp=1.0,
        decoder_reduction=1.0,
        full_student_reduction=1.0,
    )["status"] == "fail"
