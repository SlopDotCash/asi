"""Zero-blend regression for the Step 7 learned search-control utility."""

import jax.numpy as jnp

from alberta_framework.steps.step7 import _update_planning_utility


def _utilities() -> tuple[jnp.ndarray, jnp.ndarray]:
    return jnp.zeros(4, dtype=jnp.float32), jnp.array(1, dtype=jnp.int32)


def test_zero_step_size_keeps_the_stored_utility_under_infinite_td() -> None:
    utilities, index = _utilities()
    utilities = utilities.at[index].set(jnp.float32(0.75))
    td_signal = jnp.array(jnp.inf, dtype=jnp.float32)
    alpha = jnp.float32(0.0)

    # The raw applied term is the 0*inf that used to poison the utility.
    assert not bool(jnp.isfinite(alpha * jnp.abs(td_signal)))

    updated = _update_planning_utility(utilities, index, td_signal, 0.0)
    assert bool(jnp.all(jnp.isfinite(updated)))
    assert float(updated[index]) == 0.75


def test_unit_step_size_replaces_an_infinite_stored_utility() -> None:
    utilities, index = _utilities()
    utilities = utilities.at[index].set(jnp.array(jnp.inf, dtype=jnp.float32))
    td_signal = jnp.array(-2.0, dtype=jnp.float32)
    retained_scale = jnp.float32(1.0) - jnp.float32(1.0)

    # The raw retained term is the 0*inf that used to poison the utility.
    assert not bool(jnp.isfinite(retained_scale * utilities[index]))

    updated = _update_planning_utility(utilities, index, td_signal, 1.0)
    assert bool(jnp.all(jnp.isfinite(updated)))
    assert float(updated[index]) == 2.0


def test_interior_step_size_still_blends_exactly() -> None:
    utilities, index = _utilities()
    utilities = utilities.at[index].set(jnp.float32(1.0))
    td_signal = jnp.array(-3.0, dtype=jnp.float32)

    updated = _update_planning_utility(utilities, index, td_signal, 0.25)
    expected = jnp.float32(0.75) * jnp.float32(1.0) + jnp.float32(0.25) * jnp.float32(3.0)
    assert float(updated[index]) == float(expected)
    assert float(updated[0]) == 0.0
