"""Compiled finiteness regressions for the STOMP Gumbel-max importance ratio.

Under ``jax.jit`` on XLA:CPU, the intra-option modeled policy
``softmax(centered_q / 1e-6)`` in ``core/options.py`` returns NaN for finite Q
once ``|q / 1e-6|`` reaches roughly ``2**31``: XLA turns the length-one
``q_weights @ observation`` contraction into an elementwise multiply that is
recomputed inside both softmax fusions, so the contracted
multiply-then-subtract leaves a rounding residual at the argmax instead of an
exact zero (SlopDotCash/asi#2886).  The fixed ``_GUMBEL_TIE_BREAK_SCALE`` is
not a power of two and no non-default temperature is needed, so the default
configuration is affected: a finite observation near ``2.1e13`` with the
default 0.01-scale option weights yields a NaN importance ratio, the
transaction finiteness gate then rejects every subsequent update, and the
agent livelocks while eager evaluation keeps learning.

These tests pin the corrected construction: the compiled ratio stays finite
and equal to eager at the measured observation scale, an executing-option
transaction stays applied with all-finite committed state, ordinary near-tie
probabilities are unchanged, and exact ties still split uniformly.

Every defect probe is an unbatched compiled call on the real consumption path
(``_clipped_epsilon_greedy_importance_ratio``); the bare two-vector helper
shape was measured to mask the defect on some XLA builds, so it appears only
as a drift guard.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.options import (
    STOMPAgent,
    STOMPConfig,
    SubtaskSpec,
    _clipped_epsilon_greedy_importance_ratio,
    _epsilon_greedy_action_probabilities,
)

_N_PRIMITIVE = 2
_OPTION_ACTION = jnp.asarray(_N_PRIMITIVE, dtype=jnp.int32)
_MEASURED_OBSERVATION = jnp.asarray([2.1e13], dtype=jnp.float32)


def _compiled_ratio(
    q_weights: jnp.ndarray, observation: jnp.ndarray
) -> jnp.ndarray:
    """Return the compiled option importance ratio exactly as the update consumes it."""

    def ratio(
        q_weights: jnp.ndarray, observation: jnp.ndarray
    ) -> jnp.ndarray:
        return _clipped_epsilon_greedy_importance_ratio(
            q_weights,
            observation,
            jnp.asarray(0, dtype=jnp.int32),
            behavior_epsilon=0.1,
            target_epsilon=0.1,
            clip=10.0,
        )

    return jax.jit(ratio)(q_weights, observation)


def test_compiled_ratio_stays_finite_and_matches_eager_at_measured_scale() -> None:
    """The 2.1e13 observation from the issue report must not NaN under jit."""

    q_weights = jnp.asarray([[0.05], [0.03]], dtype=jnp.float32)
    compiled = _compiled_ratio(q_weights, _MEASURED_OBSERVATION)
    eager = _clipped_epsilon_greedy_importance_ratio(
        q_weights,
        _MEASURED_OBSERVATION,
        jnp.asarray(0, dtype=jnp.int32),
        behavior_epsilon=0.1,
        target_epsilon=0.1,
        clip=10.0,
    )
    assert bool(jnp.isfinite(compiled))
    np.testing.assert_allclose(compiled, eager, rtol=1e-6, atol=1e-6)


def test_compiled_ratio_stays_finite_for_explicit_overflow_scale_weights() -> None:
    """Deterministic weights put the ratio argument past the 2**31 threshold."""

    q_weights = jnp.asarray([[3000.0], [1000.0]], dtype=jnp.float32)
    observation = jnp.asarray([1.0], dtype=jnp.float32)
    compiled = _compiled_ratio(q_weights, observation)
    eager = _clipped_epsilon_greedy_importance_ratio(
        q_weights,
        observation,
        jnp.asarray(0, dtype=jnp.int32),
        behavior_epsilon=0.1,
        target_epsilon=0.1,
        clip=10.0,
    )
    assert float(jnp.max(jnp.abs(q_weights @ observation))) >= 2147.5
    assert bool(jnp.isfinite(compiled))
    np.testing.assert_allclose(compiled, eager, rtol=1e-6, atol=1e-6)


def _executing_agent_and_state() -> tuple[STOMPAgent, object]:
    """Return a default-config agent holding an executing option on a large Q row."""

    agent = STOMPAgent(
        STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0, threshold=1.0e30),),
            observation_dim=1,
            n_primitive_actions=_N_PRIMITIVE,
        )
    )
    state = agent.start(
        agent.init(jr.key(0)), jnp.zeros((1,), dtype=jnp.float32)
    )
    candidate = state.replace(
        base_last_obs=_MEASURED_OBSERVATION,
        base_last_action=_OPTION_ACTION,
        executing_option=jnp.asarray(0, dtype=jnp.int32),
        option_last_intra_action=state.last_primitive_action,
    )
    assert bool(agent.state_valid(candidate))
    return agent, candidate


def test_executing_option_transaction_applies_at_measured_observation_scale() -> None:
    """A finite observation must not wedge the option into permanent rejection."""

    agent, state = _executing_agent_and_state()
    reward = jnp.asarray(1.0, dtype=jnp.float32)
    result = agent.update(state, reward, _MEASURED_OBSERVATION)
    assert bool(result.update_applied)
    assert bool(jnp.isfinite(result.option_importance_ratio))
    assert bool(agent.state_valid(result.state))


def test_repeated_option_updates_stay_applied_and_finite() -> None:
    """Once fixed, consecutive intra-option updates keep committing finite state."""

    agent, state = _executing_agent_and_state()
    weights = jnp.asarray([[[3000.0], [1000.0]]], dtype=jnp.float32)
    state = state.replace(
        option_policies=state.option_policies.replace(q_weights=weights)
    )
    assert bool(agent.state_valid(state))
    observation = jnp.asarray([1.0], dtype=jnp.float32)
    reward = jnp.asarray(1.0, dtype=jnp.float32)
    for _ in range(3):
        result = agent.update(state, reward, observation)
        assert bool(result.update_applied)
        assert bool(jnp.isfinite(result.option_importance_ratio))
        state = result.state
    assert bool(agent.state_valid(state))
    for leaf in jax.tree_util.tree_leaves(state):
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            assert bool(jnp.all(jnp.isfinite(leaf)))


def test_modeled_probabilities_unchanged_for_near_ties_and_exact_ties() -> None:
    """The pinned near-tie mix and exact-tie uniform split must not drift."""

    epsilon = jnp.asarray(0.2, dtype=jnp.float32)

    def compiled(q: jnp.ndarray) -> jnp.ndarray:
        return jax.jit(_epsilon_greedy_action_probabilities)(q, epsilon)

    near_tie = compiled(jnp.asarray([0.0, 5.0e-7], dtype=jnp.float32))
    np.testing.assert_allclose(near_tie, [0.40203255, 0.5979675], atol=1.0e-7)

    exact_tie = compiled(jnp.asarray([1.0e38, 1.0e38], dtype=jnp.float32))
    np.testing.assert_array_equal(exact_tie, [0.5, 0.5])

    # Compiled and eager agree across an ordinary magnitude sweep.
    for base in (0.0, 1.0, 100.0, 1.0e4):
        q = jnp.asarray([base, base + 4096.0], dtype=jnp.float32)
        np.testing.assert_allclose(
            compiled(q),
            _epsilon_greedy_action_probabilities(q, epsilon),
            rtol=1e-6,
            atol=1e-7,
        )
