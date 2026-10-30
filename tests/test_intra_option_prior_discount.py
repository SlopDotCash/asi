"""Intra-option eligibility traces must decay by the prior transition discount.

The trace used for a TD error is built from earlier states, so its decay
factor is the discount of the transition *into* the current state (the
previous call's discount), not the discount of the transition *out of* it
(the current call's discount, which only bootstraps the TD error). This
mirrors TrueOnlineTD's ``previous_gamma`` and the prior-discount fixes in
the other trace learners. Decaying by the current discount turns the
terminating step into TD(0) exactly where sparse pseudo-reward needs
multi-step credit: with a zero terminating discount, earlier option
actions receive no credit at all from the final TD error.
"""

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.options import (
    STOMPAgent,
    STOMPConfig,
    SubtaskSpec,
    _update_intra_option_policy,
)


def _agent() -> STOMPAgent:
    return STOMPAgent(
        STOMPConfig(
            subtask_specs=(
                SubtaskSpec(
                    feature_index=0,
                    threshold=100.0,
                    max_option_steps=100,
                ),
            ),
            observation_dim=2,
            n_primitive_actions=2,
            base_step_size=0.0,
            base_avg_reward_step_size=0.0,
            base_trace_decay=0.0,
            option_step_size=0.5,
            option_avg_reward_step_size=0.0,
            option_trace_decay=0.75,
            option_model_decay=0.0,
            option_model_step_size=1.0,
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )


def _zero_policies(agent: STOMPAgent):
    state = agent.init(jr.key(0))
    return state.option_policies.replace(
        q_weights=jnp.zeros((1, 2, 2), dtype=jnp.float32),
        traces=jnp.zeros((1, 2, 2), dtype=jnp.float32),
        average_rewards=jnp.zeros((1,), dtype=jnp.float32),
    )


_KW = dict(
    step_size=0.5,
    avg_reward_step_size=0.0,
    trace_decay=0.75,
    n_primitive_actions=2,
    importance_ratio=jnp.asarray(1.0, dtype=jnp.float32),
)
_S0 = jnp.array([1.0, 0.0], dtype=jnp.float32)
_S1 = jnp.array([0.0, 1.0], dtype=jnp.float32)
_S2 = jnp.array([1.0, 1.0], dtype=jnp.float32)


def _first_step(policies):
    """One discount-1.0 step: action 0 on S0, zero TD error, trace = phi(S0)."""
    new_policies, td_error, applied = _update_intra_option_policy(
        policies,
        jnp.int32(0),
        _S0,
        jnp.int32(0),
        jnp.asarray(0.0, dtype=jnp.float32),
        _S1,
        jnp.asarray(False),
        jnp.asarray(1.0, dtype=jnp.float32),
        **_KW,
    )
    assert bool(applied)
    assert float(td_error) == 0.0
    return new_policies


def test_terminating_step_credits_earlier_actions() -> None:
    """A zero terminating discount must not erase the carried trace."""
    policies = _first_step(_zero_policies(_agent()))
    new_policies, td_error, applied = _update_intra_option_policy(
        policies,
        jnp.int32(0),
        _S1,
        jnp.int32(1),
        jnp.asarray(1.0, dtype=jnp.float32),
        _S2,
        jnp.asarray(True),
        jnp.asarray(0.0, dtype=jnp.float32),
        **_KW,
    )
    assert bool(applied)
    assert float(td_error) == 1.0
    # Action 0 (earlier step): alpha * td * lambda * prior_discount(1.0) * phi(S0)
    np.testing.assert_allclose(
        np.asarray(new_policies.q_weights[0, 0]),
        np.array([0.375, 0.0], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )
    # Action 1 (current step): alpha * td * phi(S1)
    np.testing.assert_allclose(
        np.asarray(new_policies.q_weights[0, 1]),
        np.array([0.0, 0.5], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )


def test_trace_decays_by_prior_discount_when_discount_changes() -> None:
    """A mid-option discount change must not rescale already-carried credit."""
    policies = _first_step(_zero_policies(_agent()))
    new_policies, td_error, applied = _update_intra_option_policy(
        policies,
        jnp.int32(0),
        _S1,
        jnp.int32(1),
        jnp.asarray(1.0, dtype=jnp.float32),
        _S2,
        jnp.asarray(False),
        jnp.asarray(0.5, dtype=jnp.float32),
        **_KW,
    )
    assert bool(applied)
    assert float(td_error) == 1.0
    # Prior discount into S1 is 1.0 (the first step's discount), not 0.5.
    np.testing.assert_allclose(
        np.asarray(new_policies.q_weights[0, 0]),
        np.array([0.375, 0.0], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )
