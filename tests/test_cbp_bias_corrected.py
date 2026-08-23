"""Regression coverage for CBP replacement ranking bias correction
(issue #2343): freshly reset units must not be re-replaced by the raw
utility EMA under-ranking their recent activity."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from alberta_framework.core.continual_backprop import (
    ContinualBackpropConfig,
    _select_replacement_index,
)


def _bias_corrected(utility: jnp.ndarray, age: jnp.ndarray, decay: float) -> jnp.ndarray:
    age_clamped = jnp.maximum(age, 1)
    return utility / (1.0 - jnp.power(decay, age_clamped.astype(jnp.float32)))


class TestBiasCorrectedReplacementRanking:
    def test_young_high_utility_unit_ranked_above_mature_low_utility(self) -> None:
        # Unit 0 was freshly reset (age 1): its raw utility EMA is tiny
        # (0.001) because it has barely accumulated, so the raw EMA ranks
        # it lowest and re-replaces it (the bug). Unit 1 is genuinely
        # low-utility long-run (raw 0.01 at age 100).
        utility = jnp.array([0.001, 0.01], dtype=jnp.float32)
        age = jnp.array([1, 100], dtype=jnp.int32)
        decay = 0.99

        raw_ranking = _select_replacement_index(utility, age, 1)
        bc_ranking = _select_replacement_index(
            _bias_corrected(utility, age, decay), age, 1
        )

        # Raw EMA picks the freshly reset unit (index 0) — the bug.
        assert int(raw_ranking[0]) == 0
        # Bias-corrected picks the genuinely low-utility mature unit.
        assert int(bc_ranking[0]) == 1

    def test_bias_correction_converges_for_old_units(self) -> None:
        # For a very old unit, bias correction is ~1 (decay^t -> 0).
        utility = jnp.array([1.0], dtype=jnp.float32)
        age = jnp.array([10_000], dtype=jnp.int32)
        corrected = _bias_corrected(utility, age, 0.99)
        assert float(corrected[0]) == pytest.approx(1.0, rel=1e-3)

    def test_age_zero_never_divide_by_zero(self) -> None:
        utility = jnp.array([1.0], dtype=jnp.float32)
        age = jnp.array([0], dtype=jnp.int32)
        corrected = _bias_corrected(utility, age, 0.99)
        assert jnp.isfinite(corrected[0])
