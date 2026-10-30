"""Continue Step 5 learning from its own returned scan state."""

import jax
import jax.numpy as jnp
import numpy as np

from alberta_framework.steps.step5 import make_step5_td_learner, run_step5_scan


def test_step5_scan_continues_without_resetting_learning() -> None:
    learner = make_step5_td_learner()
    initial = learner.init(2)
    observations = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 1.0]])
    rewards = jnp.array([1.0, 0.5, -0.25, 1.5])
    following = jnp.roll(observations, -1, axis=0)
    first = run_step5_scan(learner, initial, observations[:2], rewards[:2], following[:2])
    continued = run_step5_scan(
        learner, first.state, observations[2:], rewards[2:], following[2:],
    )
    whole = run_step5_scan(learner, initial, observations, rewards, following)
    assert int(continued.state.step_count) == 4
    assert float(continued.state.uptime_s) >= float(first.state.uptime_s)
    for left, right in zip(
        jax.tree.leaves(continued.state.replace(uptime_s=0.0)),
        jax.tree.leaves(whole.state.replace(uptime_s=0.0)), strict=True,
    ):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        jnp.concatenate([first.metrics, continued.metrics]), whole.metrics, rtol=1e-6, atol=1e-7,
    )
