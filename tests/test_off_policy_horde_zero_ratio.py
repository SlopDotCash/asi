"""Zero-importance-ratio guards in the nonlinear shared-trunk GTD Horde.

``update_with_ratios_and_discounts`` sums every demon's primary step into one
shared trunk step. A demon whose masked ratio is exactly zero must therefore
contribute *nothing*, even when the quantity it would have been multiplied by
has overflowed to infinity. The unguarded ``0 * inf`` product forms NaN,
which fails the whole-candidate finite check and rejects the update for every
healthy demon in the Horde.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.off_policy_horde import NonlinearSharedGTDHordeLearner
from alberta_framework.core.types import DemonType, GVFSpec, HordeSpec, create_horde_spec

pytestmark = pytest.mark.unit

_FLOAT32_NEAR_MAX = 3.0e38


def _spec(gammas: tuple[float, ...]) -> HordeSpec:
    demons = tuple(
        GVFSpec(
            name=f"demon_{i}",
            demon_type=DemonType.PREDICTION,
            gamma=gamma,
            lamda=0.0,
            cumulant_index=i,
        )  # type: ignore[call-arg]
        for i, gamma in enumerate(gammas)
    )
    return create_horde_spec(demons)


def _learner() -> NonlinearSharedGTDHordeLearner:
    return NonlinearSharedGTDHordeLearner(
        _spec((0.8, 0.8)),
        hidden_size=4,
        primary_step_size=0.01,
        secondary_step_size=0.01,
        ratio_clip=10.0,
    )


def test_inactive_demon_overflowing_secondary_dot_does_not_reject_the_horde() -> None:
    """An inactive demon's overflowed correction must not poison the trunk."""
    learner = _learner()
    state = learner.init(2, jax.random.key(5))
    observation = jnp.array([2.0, 1.0], dtype=jnp.float32)
    next_observation = jnp.array([-0.4, 1.0], dtype=jnp.float32)

    # Finite-but-huge secondary weights for demon 0 only.
    huge = jnp.full_like(state.secondary_trunk_w[0], _FLOAT32_NEAR_MAX)
    state = state.replace(  # type: ignore[attr-defined]
        secondary_trunk_w=state.secondary_trunk_w.at[0].set(huge),
    )
    assert bool(jnp.all(jnp.isfinite(state.secondary_trunk_w)))

    # The raw product the implementation must not form: the masked ratio for an
    # inactive demon is exactly zero and the secondary dot has overflowed.
    hidden = jnp.tanh(state.trunk_w @ observation + state.trunk_b)
    grad_hidden = state.head_w[0] * (1.0 - hidden**2)
    grad_trunk_w = grad_hidden[:, None] * observation[None, :]
    secondary_dot = jnp.vdot(huge, grad_trunk_w)
    assert not bool(jnp.isfinite(secondary_dot))
    assert not bool(jnp.isfinite(jnp.float32(0.0) * secondary_dot))

    result = learner.update_with_ratios_and_discounts(
        state,
        observation,
        jnp.array([jnp.nan, 1.0], dtype=jnp.float32),
        next_observation,
        jnp.array([1.0, 1.0], dtype=jnp.float32),
        jnp.array([0.8, 0.8], dtype=jnp.float32),
    )

    assert bool(result.update_applied)
    assert not bool(result.head_updates_applied[0])
    assert bool(result.head_updates_applied[1])
    assert bool(jnp.all(jnp.isfinite(result.state.trunk_w)))
    assert bool(jnp.all(jnp.isfinite(result.state.head_w)))
    # Demon 0 keeps its adopted secondary weights untouched.
    np.testing.assert_array_equal(
        np.asarray(result.state.secondary_trunk_w[0]),
        np.asarray(huge),
    )
    assert bool(jnp.any(result.state.head_w[1] != state.head_w[1]))


def test_inactive_demon_overflowing_gradient_does_not_reject_the_horde() -> None:
    """An inactive demon's overflowed gradient must not poison the trunk."""
    learner = _learner()
    state = learner.init(2, jax.random.key(11))
    observation = jnp.array([3.0, 1.0], dtype=jnp.float32)
    next_observation = jnp.array([-0.4, 1.0], dtype=jnp.float32)

    # Finite-but-huge output weights for demon 0 overflow its trunk gradient.
    huge_head = jnp.full_like(state.head_w[0], _FLOAT32_NEAR_MAX)
    state = state.replace(  # type: ignore[attr-defined]
        head_w=state.head_w.at[0].set(huge_head),
    )
    assert bool(jnp.all(jnp.isfinite(state.head_w)))

    hidden = jnp.tanh(state.trunk_w @ observation + state.trunk_b)
    grad_trunk_w = (huge_head * (1.0 - hidden**2))[:, None] * observation[None, :]
    assert not bool(jnp.all(jnp.isfinite(grad_trunk_w)))
    assert not bool(jnp.all(jnp.isfinite(jnp.float32(0.0) * grad_trunk_w)))

    result = learner.update_with_ratios_and_discounts(
        state,
        observation,
        jnp.array([jnp.nan, 1.0], dtype=jnp.float32),
        next_observation,
        jnp.array([1.0, 1.0], dtype=jnp.float32),
        jnp.array([0.8, 0.8], dtype=jnp.float32),
    )

    assert bool(result.update_applied)
    assert not bool(result.head_updates_applied[0])
    assert bool(result.head_updates_applied[1])
    assert bool(jnp.all(jnp.isfinite(result.state.trunk_w)))
    assert bool(jnp.all(jnp.isfinite(result.state.trunk_b)))
    assert bool(jnp.all(jnp.isfinite(result.state.secondary_trunk_w)))
    np.testing.assert_array_equal(
        np.asarray(result.state.head_w[0]),
        np.asarray(huge_head),
    )


def test_ordinary_gtd_step_still_matches_the_reference_algebra() -> None:
    """Skipping zero-scale products must not perturb an ordinary update."""
    gamma, alpha, beta, rho = 0.8, 0.01, 0.1, 1.5
    learner = NonlinearSharedGTDHordeLearner(
        _spec((gamma,)),
        hidden_size=3,
        primary_step_size=alpha,
        secondary_step_size=beta,
        ratio_clip=10.0,
    )
    state = learner.init(2, jax.random.key(3))

    trunk_w = np.asarray(state.trunk_w, dtype=np.float64)
    trunk_b = np.asarray(state.trunk_b, dtype=np.float64)
    head_w = np.asarray(state.head_w[0], dtype=np.float64)
    head_b = float(state.head_b[0])

    observation = np.array([1.0, -0.5])
    next_observation = np.array([-0.3, 1.0])
    cumulant = 1.0

    hidden = np.tanh(trunk_w @ observation + trunk_b)
    next_hidden = np.tanh(trunk_w @ next_observation + trunk_b)
    value = head_w @ hidden + head_b
    next_value = head_w @ next_hidden + head_b
    td_error = cumulant + gamma * next_value - value

    grad_hidden = head_w * (1.0 - hidden**2)
    grad_trunk_w = grad_hidden[:, None] * observation[None, :]
    # Secondary weights start at zero, so the correction term vanishes.
    expected_trunk_w = trunk_w + alpha * rho * td_error * grad_trunk_w
    expected_secondary_head_w = beta * rho * td_error * hidden

    result = learner.update_with_ratios_and_discounts(
        state,
        jnp.asarray(observation, dtype=jnp.float32),
        jnp.array([cumulant], dtype=jnp.float32),
        jnp.asarray(next_observation, dtype=jnp.float32),
        jnp.array([rho], dtype=jnp.float32),
        jnp.array([gamma], dtype=jnp.float32),
    )

    assert bool(result.update_applied)
    np.testing.assert_allclose(
        np.asarray(result.state.trunk_w), expected_trunk_w, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(result.state.secondary_head_w[0]),
        expected_secondary_head_w,
        atol=1e-6,
    )
