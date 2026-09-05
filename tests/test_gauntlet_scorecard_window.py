"""Tests for gauntlet scorecard and window scalar validation."""

import jax.numpy as jnp
import pytest

from alberta_framework.streams.gauntlet import (
    GauntletConfig,
    early_window_mse,
    lifetime_scorecard,
    savings_ratio,
    segment_slice,
)


def test_early_window_mse_and_savings_ratio_reject_boolean_and_invalid_window() -> None:
    errors = jnp.ones((400,), dtype=jnp.float32)

    # Boolean window
    with pytest.raises(ValueError, match="window"):
        early_window_mse(errors, segment=0, segment_length=100, window=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="window"):
        early_window_mse(errors, segment=0, segment_length=100, window=False)  # type: ignore[arg-type]

    # Non-integer / out-of-range window
    with pytest.raises(ValueError, match="window"):
        early_window_mse(errors, segment=0, segment_length=100, window=0)

    with pytest.raises(ValueError, match="window"):
        early_window_mse(errors, segment=0, segment_length=100, window=101)

    # Invalid segment / segment_length
    with pytest.raises(ValueError, match="segment"):
        early_window_mse(errors, segment=True, segment_length=100, window=50)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="segment"):
        early_window_mse(errors, segment=-1, segment_length=100, window=50)

    with pytest.raises(ValueError, match="segment_length"):
        early_window_mse(errors, segment=0, segment_length=0, window=50)

    # savings_ratio propagates window validation
    with pytest.raises(ValueError, match="window"):
        savings_ratio(errors, first_segment=0, revisit_segment=2, segment_length=100, window=True)  # type: ignore[arg-type]

    # Valid evaluation
    res = early_window_mse(errors, segment=0, segment_length=100, window=50)
    assert float(res) == pytest.approx(1.0)


def test_lifetime_scorecard_rejects_boolean_and_invalid_arguments() -> None:
    config = GauntletConfig(segment_length=50)
    errors = jnp.ones((2, 400), dtype=jnp.float32)

    # Boolean / invalid n_cycles
    with pytest.raises(ValueError, match="n_cycles"):
        lifetime_scorecard(errors, config, n_cycles=True, window=20)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="n_cycles"):
        lifetime_scorecard(errors, config, n_cycles=0, window=20)

    # Boolean / invalid window
    with pytest.raises(ValueError, match="window"):
        lifetime_scorecard(errors, config, n_cycles=2, window=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="window"):
        lifetime_scorecard(errors, config, n_cycles=2, window=0)

    with pytest.raises(ValueError, match="window"):
        lifetime_scorecard(errors, config, n_cycles=2, window=51)

    card = lifetime_scorecard(errors, config, n_cycles=2, window=20)
    assert "fresh_early" in card
    assert "savings_c" in card


def test_segment_slice_rejects_boolean_and_negative_indices() -> None:
    errors = jnp.ones((100,), dtype=jnp.float32)
    with pytest.raises(ValueError, match="segment"):
        segment_slice(errors, segment=True, segment_length=50)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="segment_length"):
        segment_slice(errors, segment=0, segment_length=False)  # type: ignore[arg-type]


def test_lifetime_scorecard_identical_floor_and_underflow_windows_score_one() -> None:
    import numpy as np

    config = GauntletConfig(segment_length=50)
    # 2 seeds, 3 cycles of (4 subsegments * 50) = 600 steps
    n_cycles = 3
    total_steps = n_cycles * 4 * config.segment_length

    # 1. Identical zeros (oracle / noiseless) must report 1.0 savings across cycles, not 0.0
    zeros = jnp.zeros((2, total_steps), dtype=jnp.float32)
    card_zeros = lifetime_scorecard(zeros, config, n_cycles=n_cycles, window=20)
    np.testing.assert_allclose(np.asarray(card_zeros["savings_c"]), np.ones((2, n_cycles - 1)))
    np.testing.assert_allclose(np.asarray(card_zeros["savings_d"]), np.ones((2, n_cycles - 1)))

    # 2. Identical sub-floor underflow (1e-12) must report 1.0, not 1e-12 / 1e-8 = 1e-4
    underflow = jnp.full((2, total_steps), 1e-12, dtype=jnp.float32)
    card_underflow = lifetime_scorecard(underflow, config, n_cycles=n_cycles, window=20)
    np.testing.assert_allclose(np.asarray(card_underflow["savings_c"]), np.ones((2, n_cycles - 1)))
    np.testing.assert_allclose(np.asarray(card_underflow["savings_d"]), np.ones((2, n_cycles - 1)))

    # 3. Memoryless identical non-zero (1.0) traces remain 1.0
    ones = jnp.ones((2, total_steps), dtype=jnp.float32)
    card_ones = lifetime_scorecard(ones, config, n_cycles=n_cycles, window=20)
    np.testing.assert_allclose(np.asarray(card_ones["savings_c"]), np.ones((2, n_cycles - 1)))
    np.testing.assert_allclose(np.asarray(card_ones["savings_d"]), np.ones((2, n_cycles - 1)))

    # 4. Perfect first exposure (0.0) with degraded revisit (1.0) correctly scores 0.0
    degraded = jnp.ones((2, total_steps), dtype=jnp.float32)
    # Cycle 0 subsegment 1 (task C) and subsegment 3 (task D) have zero error
    degraded = degraded.at[:, : 4 * config.segment_length].set(0.0)
    # Revisit in cycles 1 and 2 has error 1.0
    card_degraded = lifetime_scorecard(degraded, config, n_cycles=n_cycles, window=20)
    np.testing.assert_allclose(np.asarray(card_degraded["savings_c"]), np.zeros((2, n_cycles - 1)))
    np.testing.assert_allclose(np.asarray(card_degraded["savings_d"]), np.zeros((2, n_cycles - 1)))

    # 5. Genuine improvement (1.0 first exposure, 0.1 revisit) scores expected ratio (10.0)
    improved = jnp.full((2, total_steps), 0.1, dtype=jnp.float32)
    improved = improved.at[:, : 4 * config.segment_length].set(1.0)
    card_improved = lifetime_scorecard(improved, config, n_cycles=n_cycles, window=20)
    np.testing.assert_allclose(
        np.asarray(card_improved["savings_c"]), np.full((2, n_cycles - 1), 10.0), rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(card_improved["savings_d"]), np.full((2, n_cycles - 1), 10.0), rtol=1e-5
    )
