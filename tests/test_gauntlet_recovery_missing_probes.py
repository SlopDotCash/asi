"""Recovery clocks must not treat one missing probe as never recovered."""

import jax.numpy as jnp

from alberta_framework.streams.gauntlet import (
    GauntletConfig,
    ema_smooth,
    gauntlet_scorecard,
    steps_to_criterion,
)


def _recovery_trace(*, poison: bool) -> jnp.ndarray:
    # 20 high, one probe, then a long quiet tail — enough for the default
    # half-life-50 EMA to cross a 0.05 criterion after the quiet stretch.
    mid = jnp.nan if poison else 1.0
    return jnp.concatenate(
        [jnp.ones(20), jnp.asarray([mid]), jnp.zeros(400)]
    ).astype(jnp.float32)


def test_ema_smooth_all_finite_stays_bit_identical_to_recurrence() -> None:
    values = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    decay = 0.5 ** (1.0 / 1.0)
    expected = jnp.empty_like(values)
    carry = values[..., 0]
    for t in range(values.shape[-1]):
        carry = decay * carry + (1.0 - decay) * values[..., t]
        expected = expected.at[..., t].set(carry)
    actual = ema_smooth(values, halflife=1.0)
    assert jnp.array_equal(actual, expected)


def test_steps_to_criterion_recovers_after_a_mid_segment_nan() -> None:
    """A single NaN must not cap recovery at the segment length."""
    clean = _recovery_trace(poison=False)
    poison = _recovery_trace(poison=True)
    threshold = 0.05
    clean_steps = int(steps_to_criterion(clean, threshold))
    poison_steps = int(steps_to_criterion(poison, threshold))
    assert clean_steps < clean.shape[0]
    assert poison_steps < poison.shape[0]
    # Skipping one missing probe shifts the first-below index by at most one
    # step versus the all-finite twin.
    assert abs(poison_steps - clean_steps) <= 1


def test_gauntlet_scorecard_recovery_survives_one_missing_probe() -> None:
    """Public scorecard path: nan_steps counts the hole, recovery still fires."""
    length = 421
    config = GauntletConfig(segment_length=length)
    poison = _recovery_trace(poison=True)
    sq_errors = jnp.ones((1, 9 * length), dtype=jnp.float32)
    # Segment 2 is the first task-C exposure (recovery_steps_c).
    start = 2 * length
    sq_errors = sq_errors.at[0, start : start + length].set(poison)
    card = gauntlet_scorecard(sq_errors, config)
    rec_c = int(card["recovery_steps_c"][0])
    assert int(card["nan_steps"][0]) == 1
    assert rec_c < length
