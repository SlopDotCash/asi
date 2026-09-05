"""Zero-scale regression for Step 7 predecessor priority propagation."""

import jax.numpy as jnp

from alberta_framework.steps.step7 import _propagate_predecessor_priorities


def _queue() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    next_observations = jnp.zeros((3, 2), dtype=jnp.float32)
    priorities = jnp.array([0.5, 0.25, 0.125], dtype=jnp.float32)
    count = jnp.array(3, dtype=jnp.int32)
    anchor = jnp.zeros((2,), dtype=jnp.float32)
    return next_observations, priorities, count, anchor


def test_disabled_propagation_leaves_the_queue_intact_under_infinite_td() -> None:
    next_observations, priorities, count, anchor = _queue()
    td_error = jnp.array(jnp.inf, dtype=jnp.float32)
    scale = jnp.float32(0.0)

    # The raw scale*|td| product is the 0*inf that used to poison the queue,
    # and jnp.maximum propagates the resulting NaN over every live priority.
    raw = scale * jnp.abs(td_error)
    assert not bool(jnp.isfinite(raw))
    assert not bool(jnp.isfinite(jnp.maximum(priorities[0], raw)))

    updated = _propagate_predecessor_priorities(
        next_observations, priorities, count, anchor, td_error, 0.0
    )
    assert bool(jnp.all(jnp.isfinite(updated)))
    assert [float(value) for value in updated] == [0.5, 0.25, 0.125]


def test_enabled_propagation_still_raises_priorities() -> None:
    next_observations, priorities, count, anchor = _queue()
    td_error = jnp.array(-4.0, dtype=jnp.float32)

    updated = _propagate_predecessor_priorities(
        next_observations, priorities, count, anchor, td_error, 0.5
    )
    # distance is 0, so propagated = 0.5 * 4.0 / 1.0 = 2.0 everywhere.
    assert [float(value) for value in updated] == [2.0, 2.0, 2.0]


def test_entries_beyond_memory_count_are_untouched() -> None:
    next_observations, priorities, _, anchor = _queue()
    count = jnp.array(1, dtype=jnp.int32)
    td_error = jnp.array(-4.0, dtype=jnp.float32)

    updated = _propagate_predecessor_priorities(
        next_observations, priorities, count, anchor, td_error, 0.5
    )
    assert [float(value) for value in updated] == [2.0, 0.25, 0.125]
