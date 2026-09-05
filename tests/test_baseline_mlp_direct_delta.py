"""Learner updates must preserve the baseline optimizer's loss-gradient trajectory."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.baseline_optimizers import Adam, RMSprop
from alberta_framework.core.learners import MLPLearner


@pytest.mark.parametrize(
    "optimizer",
    [
        Adam(step_size=0.01),
        Adam(step_size=10.0),
        Adam(step_size=0.01, weight_decay=0.2),
        RMSprop(step_size=0.01),
    ],
)
@pytest.mark.parametrize("error", [0.0, 1.2e-38, -1.2e-38, 0.5, -0.5])
def test_mlp_matches_loss_gradient_update_after_momentum(optimizer, error):
    learner = MLPLearner(hidden_sizes=(), optimizer=optimizer, sparsity=0.0)
    state = learner.init(1, jr.key(0))
    observation = jnp.array([1.0], dtype=jnp.float32)
    state = learner.update(state, observation, learner.predict(state, observation) + 1.0).state
    # Keep optimizer history but put the prediction at zero, so tiny residuals
    # remain representable rather than disappearing in target subtraction.
    state = state.replace(
        params=state.params.replace(
            weights=(jnp.zeros_like(state.params.weights[0]),),
            biases=(jnp.zeros_like(state.params.biases[0]),),
        )
    )
    expected = []
    for param, opt_state in zip(
        (state.params.weights[0], state.params.biases[0]), state.optimizer_states, strict=True
    ):
        kwargs = {"param": param} if isinstance(optimizer, Adam) else {}
        reference = optimizer.update_from_gradient_checked(
            opt_state, jnp.full_like(param, -error), error=None, **kwargs
        )
        assert bool(reference.update_applied)
        expected.append(param - reference.step)
    result = learner.update(state, observation, jnp.array(error, dtype=jnp.float32))
    assert bool(result.update_applied)
    for actual, reference in zip(
        (result.state.params.weights[0], result.state.params.biases[0]), expected, strict=True
    ):
        np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=0.0)


def test_adamw_mlp_decays_parameters_at_zero_error():
    optimizer = Adam(step_size=0.01, weight_decay=0.2)
    learner = MLPLearner(hidden_sizes=(), optimizer=optimizer, sparsity=0.0)
    state = learner.init(1, jr.key(0))
    state = state.replace(
        params=state.params.replace(weights=(jnp.ones((1, 1)),), biases=(jnp.ones((1,)),))
    )
    observation = jnp.array([1.0], dtype=jnp.float32)
    result = learner.update(state, observation, learner.predict(state, observation))
    assert bool(result.update_applied)
    np.testing.assert_allclose(result.state.params.weights[0], [[0.998]], rtol=1e-6)
    np.testing.assert_allclose(result.state.params.biases[0], [0.998], rtol=1e-6)


@pytest.mark.parametrize("kind", ["multi_head", "off_policy", "independent"])
def test_adamw_consumers_apply_zero_error_decay_to_trunk_and_heads(kind):
    import jax

    from alberta_framework.core.independent_demon_horde import IndependentDemonHorde
    from alberta_framework.core.multi_head_learner import MultiHeadMLPLearner
    from alberta_framework.core.off_policy_horde import OffPolicyHordeLearner
    from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec

    optimizer = Adam(step_size=0.01, weight_decay=0.2)
    spec = create_horde_spec(
        (
            GVFSpec(
                name="v", demon_type=DemonType.PREDICTION, gamma=0.0, lamda=0.0, cumulant_index=0
            ),
        )
    )
    kwargs = dict(hidden_sizes=(2,), optimizer=optimizer, sparsity=0.0, use_layer_norm=False)
    if kind == "multi_head":
        learner = MultiHeadMLPLearner(n_heads=1, **kwargs)
    elif kind == "off_policy":
        learner = OffPolicyHordeLearner(spec, **kwargs)
    else:
        learner = IndependentDemonHorde(spec, **kwargs)
    state = learner.init(2, jr.key(11))
    observation = jnp.array([0.25, -0.5], dtype=jnp.float32)
    targets = learner.predict(state, observation)
    if kind == "multi_head":
        result = learner.update(state, observation, targets)
    elif kind == "off_policy":
        result = learner.update(state, observation, targets, observation, jnp.ones(1))
    else:
        result = learner.update(state, observation, targets, observation)
    if kind == "independent":
        old_params = state.demon_states[0].params
        new_params = result.state.demon_states[0].params
    else:
        old_params = (state.trunk_params, state.head_params)
        new_params = (result.state.trunk_params, result.state.head_params)
    for old, new in zip(jax.tree.leaves(old_params), jax.tree.leaves(new_params), strict=True):
        np.testing.assert_allclose(new, 0.998 * old, rtol=2e-6, atol=1e-7)


def test_mixed_mlp_optimizers_bound_complete_delta_without_losing_zero_error_decay():
    from alberta_framework.core.optimizers import LMS, ObGDBounding

    learner = MLPLearner(
        hidden_sizes=(2,),
        optimizer=LMS(step_size=0.01),
        head_optimizer=Adam(step_size=0.01, weight_decay=0.2),
        bounder=ObGDBounding(kappa=2.0),
        sparsity=0.0,
    )
    state = learner.init(2, jr.key(21))
    observation = jnp.array([0.25, -0.5], dtype=jnp.float32)
    result = learner.update(state, observation, learner.predict(state, observation))
    assert bool(result.update_applied)
    np.testing.assert_array_equal(result.state.params.weights[0], state.params.weights[0])
    np.testing.assert_allclose(
        result.state.params.weights[1], 0.998 * state.params.weights[1], rtol=2e-6
    )


def test_nonlinear_actor_adamw_preserves_decay_at_zero_td_error():
    import jax

    from alberta_framework.core.horde import HordeLearner
    from alberta_framework.core.horde_actor_critic import (
        NonlinearHordeActorCriticAgent,
        NonlinearHordeActorCriticConfig,
    )
    from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec

    spec = create_horde_spec(
        (
            GVFSpec(
                name="v", demon_type=DemonType.PREDICTION, gamma=0.0, lamda=0.0, cumulant_index=0
            ),
        )
    )
    critic = HordeLearner(spec, hidden_sizes=())
    agent = NonlinearHordeActorCriticAgent(
        NonlinearHordeActorCriticConfig(n_actions=2, hidden_sizes=(2,), actor_sparsity=0.0),
        critic,
        actor_optimizer=Adam(step_size=0.01, weight_decay=0.2),
    )
    observation = jnp.array([0.25, -0.5], dtype=jnp.float32)
    state = agent.init(2, jr.key(31))
    state, _, _ = agent.start(state, observation)
    reward = critic.predict(state.critic_state, observation)[0]
    result = agent.update(state, reward, observation)
    assert bool(result.update_applied)
    assert float(result.td_error) == 0.0
    old_params = (state.actor_trunk, state.actor_head_w, state.actor_head_b)
    new_params = (result.state.actor_trunk, result.state.actor_head_w, result.state.actor_head_b)
    for old, new in zip(jax.tree.leaves(old_params), jax.tree.leaves(new_params), strict=True):
        np.testing.assert_allclose(new, 0.998 * old, rtol=2e-6, atol=1e-7)
