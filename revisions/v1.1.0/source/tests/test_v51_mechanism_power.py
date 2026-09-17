from __future__ import annotations

import numpy as np
import pytest

from dpc_snn.analysis.mechanism_power import (
    SyntheticMechanismSpec,
    fractional_causal_delay,
    generate_common_source_analytic,
    generate_coupled_analytic,
)


def _spec() -> SyntheticMechanismSpec:
    return SyntheticMechanismSpec(
        frequencies_hz=(7.0, 11.0, 18.0, 29.0),
        trials=4,
        nodes=3,
        time_steps=64,
        timestep_seconds=0.004,
        burn_in_steps=8,
    )


def test_fractional_causal_delay_matches_model_equation() -> None:
    source = np.arange(8, dtype=np.float32)
    delayed = fractional_causal_delay(source, 1.25)

    assert delayed[:2].tolist() == [0.0, 0.0]
    assert delayed[2] == pytest.approx(0.75)
    assert delayed[3] == pytest.approx(1.75)


def test_coupled_generator_is_deterministic_and_preserves_shape() -> None:
    first = generate_coupled_analytic(_spec(), delay_steps=0.5, coupling_strength=0.25, seed=17)
    second = generate_coupled_analytic(_spec(), delay_steps=0.5, coupling_strength=0.25, seed=17)

    assert first.shape == (4, 4, 3, 64)
    assert first.dtype == np.complex64
    np.testing.assert_array_equal(first, second)


def test_common_source_control_contains_no_explicit_delay() -> None:
    mixed = generate_common_source_analytic(_spec(), mixing_strength=0.5, seed=23)

    assert mixed.shape == (4, 4, 3, 64)
    assert np.isfinite(mixed.real).all()
    assert np.isfinite(mixed.imag).all()
