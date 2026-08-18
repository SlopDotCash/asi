"""Tests for the Step 4 continuous-action actor-critic preview."""

from __future__ import annotations

import warnings
from types import MappingProxyType

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework import ContinuousActorCriticAgent as TopLevelContinuousActorCriticAgent
from alberta_framework.core import (
    ContinuousActorCriticAgent as CoreContinuousActorCriticAgent,
)
from alberta_framework.core.actor_critic import (
    ContinuousActorCriticAgent,
    ContinuousActorCriticConfig,
    _require_continuous_state_resources,
    run_continuous_actor_critic_from_arrays,
)
from alberta_framework.core.optimizers import ObGDBounding


def _assert_continuous_actor_critic_state_finite(state) -> None:  # type: ignore[no-untyped-def]
    chex.assert_tree_all_finite(
        (
            state.mean_weights,
            state.mean_bias,
            state.log_sigma,
            state.critic_weights,
            state.critic_bias,
            state.mean_trace_weights,
            state.mean_trace_bias,
            state.log_sigma_trace,
            state.critic_trace_weights,
            state.critic_trace_bias,
            state.last_observation,
            state.last_action,
        )
    )


def test_continuous_actor_critic_top_level_alias() -> None:
    """Top-level and core re-exports refer to the same class."""
    assert TopLevelContinuousActorCriticAgent is ContinuousActorCriticAgent
    assert CoreContinuousActorCriticAgent is ContinuousActorCriticAgent


def test_continuous_actor_critic_init_and_select_action_shapes() -> None:
    agent = ContinuousActorCriticAgent(
        ContinuousActorCriticConfig(action_dim=2, action_low=-1.0, action_high=1.0)
    )
    state = agent.init(feature_dim=4, key=jr.key(0))
    obs = jnp.array([1.0, -1.0, 0.5, 2.0], dtype=jnp.float32)

    mean, sigma = agent.policy_params(state, obs)
    value = agent.value(state, obs)
    next_state, action, mean_start, sigma_start = agent.start(state, obs)

    chex.assert_shape(mean, (2,))
    chex.assert_shape(sigma, (2,))
    chex.assert_shape(value, ())
    chex.assert_shape(action, (2,))
    chex.assert_shape(mean_start, (2,))
    chex.assert_shape(sigma_start, (2,))
    assert jnp.all(sigma > 0.0)
    assert jnp.all(action >= -1.0)
    assert jnp.all(action <= 1.0)
    _assert_continuous_actor_critic_state_finite(next_state)


def test_continuous_actor_critic_update_produces_finite_outputs() -> None:
    config = ContinuousActorCriticConfig(
        action_dim=1,
        gamma=0.9,
        actor_step_size=0.01,
        critic_step_size=0.05,
        actor_lamda=0.5,
        critic_lamda=0.5,
    )
    agent = ContinuousActorCriticAgent(config)
    state = agent.init(feature_dim=3, key=jr.key(1))
    state, _action, _mean, _sigma = agent.start(
        state, jnp.array([1.0, 0.0, -1.0], dtype=jnp.float32)
    )

    result = agent.update(
        state,
        reward=jnp.array(1.0, dtype=jnp.float32),
        observation=jnp.array([0.0, 1.0, 0.5], dtype=jnp.float32),
        terminated=jnp.array(False),
    )

    assert int(result.state.step_count) == 1
    _assert_continuous_actor_critic_state_finite(result.state)
    chex.assert_tree_all_finite(
        (result.mean, result.sigma, result.value, result.next_value, result.td_error)
    )


def test_continuous_actor_critic_policy_gradient_sign() -> None:
    """A positive TD error with action above the mean should push the mean up.

    With a Gaussian score function ``(a - mu) / sigma^2`` and positive TD
    error, the mean parameters update along ``+ (a - mu) / sigma^2 . s``. For a
    one-dimensional input ``s = 1`` the mean weight should therefore increase.
    """
    config = ContinuousActorCriticConfig(
        action_dim=1,
        gamma=0.0,  # purely instantaneous TD = reward - value
        actor_step_size=0.5,
        critic_step_size=0.0,
        actor_lamda=0.0,
        critic_lamda=0.0,
        log_sigma_init=0.0,
    )
    agent = ContinuousActorCriticAgent(config)
    obs = jnp.array([1.0], dtype=jnp.float32)
    state = agent.init(feature_dim=1, key=jr.key(2)).replace(  # type: ignore[attr-defined]
        last_observation=obs,
        last_action=jnp.array([0.4], dtype=jnp.float32),  # action > mean (mean starts at 0)
    )
    # critic V(s) = 0 since weights are zero, so td_error = reward = +1.0 > 0.
    result = agent.update(
        state,
        reward=jnp.array(1.0, dtype=jnp.float32),
        observation=jnp.array([0.0], dtype=jnp.float32),
        terminated=jnp.array(True),
    )
    assert float(result.state.mean_weights[0, 0]) > 0.0
    assert float(result.state.mean_bias[0]) > 0.0
    assert float(result.td_error) > 0.0


def test_continuous_actor_critic_terminal_resets_traces() -> None:
    config = ContinuousActorCriticConfig(
        action_dim=2,
        actor_step_size=0.01,
        critic_step_size=0.01,
        actor_lamda=0.9,
        critic_lamda=0.9,
    )
    agent = ContinuousActorCriticAgent(config)
    state = agent.init(feature_dim=2, key=jr.key(3)).replace(  # type: ignore[attr-defined]
        last_observation=jnp.array([1.0, 0.5], dtype=jnp.float32),
        last_action=jnp.array([0.1, -0.1], dtype=jnp.float32),
        mean_trace_weights=jnp.ones((2, 2), dtype=jnp.float32),
        mean_trace_bias=jnp.ones((2,), dtype=jnp.float32),
        log_sigma_trace=jnp.ones((2,), dtype=jnp.float32),
        critic_trace_weights=jnp.ones((2,), dtype=jnp.float32),
        critic_trace_bias=jnp.array(1.0, dtype=jnp.float32),
    )

    result = agent.update(
        state,
        reward=jnp.array(0.5, dtype=jnp.float32),
        observation=jnp.array([0.0, 1.0], dtype=jnp.float32),
        terminated=jnp.array(True),
    )
    chex.assert_trees_all_close(
        (
            result.state.mean_trace_weights,
            result.state.mean_trace_bias,
            result.state.log_sigma_trace,
            result.state.critic_trace_weights,
            result.state.critic_trace_bias,
        ),
        (
            jnp.zeros_like(result.state.mean_trace_weights),
            jnp.zeros_like(result.state.mean_trace_bias),
            jnp.zeros_like(result.state.log_sigma_trace),
            jnp.zeros_like(result.state.critic_trace_weights),
            jnp.array(0.0, dtype=jnp.float32),
        ),
    )


def test_continuous_actor_critic_log_sigma_clipping() -> None:
    """Log-sigma stays within configured bounds even with large updates."""
    config = ContinuousActorCriticConfig(
        action_dim=1,
        gamma=0.0,
        actor_step_size=10.0,
        critic_step_size=0.0,
        actor_lamda=0.0,
        critic_lamda=0.0,
        log_sigma_init=0.0,
        log_sigma_min=-2.0,
        log_sigma_max=1.0,
    )
    agent = ContinuousActorCriticAgent(config)
    obs = jnp.array([1.0], dtype=jnp.float32)
    state = agent.init(feature_dim=1, key=jr.key(4)).replace(  # type: ignore[attr-defined]
        last_observation=obs,
        last_action=jnp.array([5.0], dtype=jnp.float32),
    )
    result = agent.update(
        state,
        reward=jnp.array(1.0, dtype=jnp.float32),
        observation=jnp.array([0.0], dtype=jnp.float32),
        terminated=jnp.array(True),
    )
    assert float(result.state.log_sigma[0]) <= 1.0
    assert float(result.state.log_sigma[0]) >= -2.0


@pytest.mark.parametrize(
    ("low", "high", "message"),
    [
        (1.0, -1.0, "action_low must be <= action_high"),
        (float("nan"), 1.0, "action_low must be finite"),
        (-1.0, float("inf"), "action_high must be finite"),
        (1e100, None, "action_low must remain finite once narrowed to float32"),
        (None, -1e100, "action_high must remain finite once narrowed to float32"),
        (-3.5e38, 3.5e38, "action_low must remain finite once narrowed to float32"),
        (0.5, 0.5, None),
        (-float(np.finfo(np.float32).max), float(np.finfo(np.float32).max), None),
    ],
)
def test_continuous_actor_critic_rejects_inverted_or_nonfinite_action_bounds(
    low: float | None, high: float | None, message: str | None
) -> None:
    """low > high makes jnp.clip return high for every input: a constant-action policy."""
    if message is None:
        config = ContinuousActorCriticConfig(action_dim=2, action_low=low, action_high=high)
        agent = ContinuousActorCriticAgent(config)
        state = agent.init(feature_dim=1, key=jr.key(2))
        _state, action, _mean, _sigma = agent.start(state, jnp.array([1.0], dtype=jnp.float32))
        assert bool(jnp.all(jnp.isfinite(action)))
        return
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=message):
            ContinuousActorCriticConfig(action_dim=2, action_low=low, action_high=high)


class _FloatSpoof:
    """Not a Real: reports float through __class__ and never compares as out of range."""

    @property
    def __class__(self) -> type[float]:  # type: ignore[override]
        return float

    def as_integer_ratio(self) -> tuple[int, int]:
        return (0, 1)

    def __float__(self) -> float:
        return 0.0

    def __lt__(self, other: object) -> bool:
        return False

    def __le__(self, other: object) -> bool:
        return False

    def __gt__(self, other: object) -> bool:
        return False

    def __ge__(self, other: object) -> bool:
        return False


@pytest.mark.parametrize("field", ["action_low", "action_high"])
def test_continuous_actor_critic_rejects_bounds_that_only_spoof_float(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be a finite real number"):
        ContinuousActorCriticConfig(  # type: ignore[arg-type]
            action_dim=1, **{field: _FloatSpoof()}
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action_low", (2**25 - 1) * 2**103 - 1),
        ("action_high", -((2**25 - 1) * 2**103 - 1)),
        ("action_low", 2**64 + 2**40 + 1),
    ],
)
def test_continuous_actor_critic_stores_exact_float32_for_integer_bounds(
    field: str, value: int
) -> None:
    """A built-in int must not be stored as float(value): that payload double-rounds in JAX."""
    from alberta_framework._float32 import round_real_to_float32

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        agent = ContinuousActorCriticAgent(
            ContinuousActorCriticConfig(action_dim=1, **{field: value})  # type: ignore[arg-type]
        )
        stored = getattr(agent.config, field)
        assert type(stored) is float
        assert stored == round_real_to_float32(value)
        assert float(np.float32(stored)) == stored
        state = agent.init(feature_dim=1, key=jr.key(3))
        _state, action, _mean, _sigma = agent.start(state, jnp.array([1.0], dtype=jnp.float32))
        assert bool(jnp.all(jnp.isfinite(action)))


def test_continuous_actor_critic_canonicalizes_real_bounds_to_float() -> None:
    from fractions import Fraction

    agent = ContinuousActorCriticAgent(
        ContinuousActorCriticConfig(
            action_dim=1, action_low=Fraction(-1, 4), action_high=np.float64(0.25)
        )
    )
    assert type(agent.config.action_low) is float and agent.config.action_low == -0.25
    assert type(agent.config.action_high) is float and agent.config.action_high == 0.25
    state = agent.init(feature_dim=1, key=jr.key(1))
    _state, action, _mean, _sigma = agent.start(state, jnp.array([1.0], dtype=jnp.float32))
    assert float(action[0]) <= 0.25 and float(action[0]) >= -0.25


def test_continuous_actor_critic_normalizes_conversion_hook_failures() -> None:
    from fractions import Fraction

    class BrokenFraction(Fraction):
        def as_integer_ratio(self) -> tuple[int, int]:
            raise RuntimeError("conversion hook failed")

    with pytest.raises(ValueError, match="action_low must be"):
        ContinuousActorCriticAgent(
            ContinuousActorCriticConfig(
                action_dim=1,
                action_low=BrokenFraction(-1, 1),
                action_high=1.0,
            )
        )


def test_continuous_actor_critic_action_clipping() -> None:
    """Sampled actions respect the configured action bounds."""
    config = ContinuousActorCriticConfig(
        action_dim=1,
        log_sigma_init=2.0,  # very wide policy
        action_low=-0.25,
        action_high=0.25,
    )
    agent = ContinuousActorCriticAgent(config)
    state = agent.init(feature_dim=1, key=jr.key(5))
    obs = jnp.array([1.0], dtype=jnp.float32)
    actions = []
    for _ in range(64):
        action, key, _mean, _sigma = agent.select_action(state, obs)
        state = state.replace(rng_key=key)  # type: ignore[attr-defined]
        actions.append(action)
    stacked = jnp.stack(actions)
    assert jnp.all(stacked >= -0.25)
    assert jnp.all(stacked <= 0.25)


def test_continuous_actor_critic_update_is_jittable() -> None:
    agent = ContinuousActorCriticAgent(ContinuousActorCriticConfig(action_dim=2))
    state = agent.init(feature_dim=2, key=jr.key(6))
    state, _action, _mean, _sigma = agent.start(state, jnp.array([1.0, 0.0], dtype=jnp.float32))
    update = jax.jit(agent.update)
    result = update(
        state,
        jnp.array(0.5, dtype=jnp.float32),
        jnp.array([0.0, 1.0], dtype=jnp.float32),
        jnp.array(False),
    )
    chex.assert_shape(result.mean, (2,))
    assert int(result.state.step_count) == 1


def test_continuous_actor_critic_run_from_arrays_scan() -> None:
    agent = ContinuousActorCriticAgent(ContinuousActorCriticConfig(action_dim=2))
    state = agent.init(feature_dim=2, key=jr.key(7))
    observations = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=jnp.float32)
    next_observations = jnp.array([[0.0, 1.0], [1.0, 1.0], [0.5, -0.5]], dtype=jnp.float32)
    rewards = jnp.array([1.0, 0.0, -1.0], dtype=jnp.float32)
    terminated = jnp.array([False, False, True])

    result = run_continuous_actor_critic_from_arrays(
        agent, state, observations, rewards, terminated, next_observations
    )

    chex.assert_shape(result.actions, (3, 2))
    chex.assert_shape(result.means, (3, 2))
    chex.assert_shape(result.sigmas, (3, 2))
    chex.assert_shape(result.values, (3,))
    chex.assert_shape(result.td_errors, (3,))
    assert int(result.state.step_count) == 3
    _assert_continuous_actor_critic_state_finite(result.state)
    chex.assert_tree_all_finite((result.means, result.sigmas, result.values, result.td_errors))


def test_continuous_actor_critic_config_roundtrip() -> None:
    config = ContinuousActorCriticConfig(
        action_dim=3,
        gamma=0.95,
        actor_step_size=0.002,
        critic_step_size=0.04,
        actor_lamda=0.7,
        critic_lamda=0.8,
        log_sigma_init=-0.25,
        log_sigma_min=-3.0,
        log_sigma_max=1.5,
        action_low=-2.0,
        action_high=2.0,
    )
    agent = ContinuousActorCriticAgent(config, bounder=ObGDBounding(kappa=2.5))
    serialised = agent.to_config()
    restored = ContinuousActorCriticAgent.from_config(serialised)
    assert restored.config.action_dim == config.action_dim
    assert restored.config.gamma == config.gamma
    assert restored.config.actor_step_size == config.actor_step_size
    assert restored.config.critic_step_size == config.critic_step_size
    assert restored.config.actor_lamda == config.actor_lamda
    assert restored.config.critic_lamda == config.critic_lamda
    assert restored.config.log_sigma_init == config.log_sigma_init
    assert restored.config.log_sigma_min == config.log_sigma_min
    assert restored.config.log_sigma_max == config.log_sigma_max
    assert restored.config.action_low == config.action_low
    assert restored.config.action_high == config.action_high
    assert restored.bounder is not None
    assert restored.bounder.to_config() == agent.bounder.to_config()  # type: ignore[union-attr]


def test_continuous_actor_critic_with_obgd_bounder_is_finite() -> None:
    agent = ContinuousActorCriticAgent(
        ContinuousActorCriticConfig(
            action_dim=2,
            actor_step_size=0.5,
            critic_step_size=0.5,
        ),
        bounder=ObGDBounding(kappa=2.0),
    )
    state = agent.init(feature_dim=3, key=jr.key(8))
    state, _action, _mean, _sigma = agent.start(
        state, jnp.array([1.0, 0.0, -1.0], dtype=jnp.float32)
    )
    result = agent.update(
        state,
        reward=jnp.array(5.0, dtype=jnp.float32),
        observation=jnp.array([0.0, 1.0, 0.5], dtype=jnp.float32),
        terminated=jnp.array(False),
    )
    _assert_continuous_actor_critic_state_finite(result.state)
    chex.assert_tree_all_finite(result.bound_metric)


def test_continuous_actor_critic_infinite_reward_with_obgd_does_not_poison() -> None:
    """Inf TD error zeros the ObGD step, then td_error*step is 0*inf=NaN."""
    agent = ContinuousActorCriticAgent(
        ContinuousActorCriticConfig(
            action_dim=2,
            actor_step_size=0.1,
            critic_step_size=0.1,
        ),
        bounder=ObGDBounding(kappa=2.0),
    )
    state = agent.init(feature_dim=3, key=jr.key(8))
    state, _, _, _ = agent.start(state, jnp.array([1.0, 0.0, -1.0], dtype=jnp.float32))
    poisoned = agent.update(
        state,
        reward=jnp.array(jnp.inf, dtype=jnp.float32),
        observation=jnp.array([0.0, 1.0, 0.5], dtype=jnp.float32),
        terminated=jnp.array(False),
    )
    assert bool(jnp.all(jnp.isfinite(poisoned.state.mean_weights)))
    assert bool(jnp.all(jnp.isfinite(poisoned.state.critic_weights)))
    chex.assert_trees_all_close(poisoned.state.mean_weights, state.mean_weights)
    chex.assert_trees_all_close(poisoned.state.critic_weights, state.critic_weights)
    chex.assert_trees_all_equal(jr.key_data(poisoned.state.rng_key), jr.key_data(state.rng_key))
    chex.assert_trees_all_close(
        poisoned.state.replace(rng_key=jr.key_data(poisoned.state.rng_key)),
        state.replace(rng_key=jr.key_data(state.rng_key)),
    )
    assert not bool(poisoned.update_applied)
    assert float(poisoned.td_error) == 0.0
    chex.assert_trees_all_close(poisoned.action, jnp.zeros_like(poisoned.action))

    recovered = agent.update(
        poisoned.state,
        reward=jnp.array(1.0, dtype=jnp.float32),
        observation=jnp.array([0.5, 0.0, 0.0], dtype=jnp.float32),
        terminated=jnp.array(False),
    )
    assert bool(jnp.all(jnp.isfinite(recovered.state.mean_weights)))
    assert bool(jnp.all(jnp.isfinite(recovered.state.critic_weights)))
    assert bool(recovered.update_applied)


def test_continuous_actor_critic_terminal_does_not_multiply_inf_next_value() -> None:
    """gamma=0 * inf V(s') is 0*inf = NaN and would freeze a terminal update."""
    agent = ContinuousActorCriticAgent(
        ContinuousActorCriticConfig(
            action_dim=1,
            actor_step_size=0.1,
            critic_step_size=0.1,
        )
    )
    huge = jnp.float32(1e38)
    state = agent.init(feature_dim=2, key=jr.key(1)).replace(  # type: ignore[attr-defined]
        last_observation=jnp.array([0.0, 1.0], dtype=jnp.float32),
        last_action=jnp.array([0.0], dtype=jnp.float32),
        critic_weights=jnp.array([huge, 0.0], dtype=jnp.float32),
        critic_bias=jnp.array(0.0, dtype=jnp.float32),
    )
    next_obs = jnp.array([huge, 0.0], dtype=jnp.float32)
    raw = jnp.asarray(0.0, dtype=jnp.float32) * (huge * huge)
    assert not bool(jnp.isfinite(raw))

    result = agent.update(
        state,
        reward=jnp.array(3.0, dtype=jnp.float32),
        observation=next_obs,
        terminated=jnp.array(True),
    )
    assert bool(result.update_applied)
    chex.assert_trees_all_close(result.td_error, jnp.array(3.0, dtype=jnp.float32))
    _assert_continuous_actor_critic_state_finite(result.state)


def test_continuous_actor_critic_integer_and_scalar_validation() -> None:
    with pytest.raises(ValueError, match="action_dim"):
        ContinuousActorCriticConfig(action_dim=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="action_dim"):
        ContinuousActorCriticConfig(action_dim=0)
    with pytest.raises(ValueError, match="action_dim"):
        ContinuousActorCriticConfig(action_dim=2.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="gamma"):
        ContinuousActorCriticConfig(action_dim=2, gamma=1.5)
    with pytest.raises(ValueError, match="actor_step_size"):
        ContinuousActorCriticConfig(action_dim=2, actor_step_size=-0.1)

    cfg = ContinuousActorCriticConfig(action_dim=np.int32(3))
    assert cfg.action_dim == 3
    assert type(cfg.action_dim) is int

    agent = ContinuousActorCriticAgent(cfg)
    with pytest.raises(ValueError, match="feature_dim"):
        agent.init(feature_dim=True, key=jr.key(0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="feature_dim"):
        agent.init(feature_dim=0, key=jr.key(0))

    state = agent.init(feature_dim=np.int32(5), key=jr.key(0))
    assert state.mean_weights.shape == (3, 5)


class _HostileFloat(float):
    def as_integer_ratio(self) -> tuple[int, int]:
        raise RuntimeError("hostile hook executed")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gamma", _HostileFloat(0.5)),
        ("actor_step_size", np.float64(1e-100)),
        ("log_sigma_init", 1e100),
        ("log_sigma_min", True),
        ("action_low", _HostileFloat(-1.0)),
        ("action_high", np.float64(1e100)),
    ],
)
def test_continuous_actor_critic_rejects_invalid_float32_config(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        config = ContinuousActorCriticConfig(action_dim=2, **{field: value})  # type: ignore[arg-type]
        ContinuousActorCriticAgent(config)


def test_continuous_actor_critic_mapping_config_and_resource_boundary() -> None:
    config = ContinuousActorCriticConfig(
        action_dim=np.uint16(2), action_low=np.float32(-1), action_high=np.float64(1)
    ).to_config()
    clone = ContinuousActorCriticConfig.from_config(MappingProxyType(config))
    assert clone.to_config() == config
    _require_continuous_state_resources(1, 107_374_180)
    with pytest.raises(ValueError, match="state exceeds"):
        _require_continuous_state_resources(1, 107_374_181)


class _HostileArray:
    def __jax_array__(self) -> jax.Array:
        raise RuntimeError("hostile array hook executed")


def test_continuous_actor_critic_schema_and_hostile_inputs_fail_closed() -> None:
    config = ContinuousActorCriticConfig(action_dim=2).to_config()
    with pytest.raises(ValueError, match="serialized schema"):
        ContinuousActorCriticConfig.from_config({**config, "unknown": 1})
    with pytest.raises(ValueError, match="exact JSON"):
        ContinuousActorCriticConfig.from_config({**config, "action_dim": np.int32(2)})

    agent = ContinuousActorCriticAgent(ContinuousActorCriticConfig(action_dim=2))
    payload = agent.to_config()
    with pytest.raises(ValueError, match="type differs"):
        ContinuousActorCriticAgent.from_config({**payload, "type": "wrong"})
    with pytest.raises(ValueError, match="Bounder"):
        ContinuousActorCriticAgent(agent.config, bounder=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Threefry"):
        agent.init(2, _HostileArray())  # type: ignore[arg-type]
    state = agent.init(2, jr.key(0))
    with pytest.raises(ValueError, match="trusted array metadata"):
        agent.policy_params.__wrapped__(agent, state, _HostileArray())  # type: ignore[attr-defined,arg-type]


@pytest.mark.parametrize("shape", [(), (1,), (1, 2), (2, 1), (3,)])
def test_continuous_actor_critic_rejects_wrong_observation_shapes(
    shape: tuple[int, ...]
) -> None:
    agent = ContinuousActorCriticAgent(ContinuousActorCriticConfig(action_dim=2))
    state = agent.init(2, jr.key(0))
    malformed = jnp.zeros(shape, dtype=jnp.float32)
    with pytest.raises(ValueError, match="observation"):
        agent.policy_params(state, malformed)
    with pytest.raises(ValueError, match="observation"):
        agent.update(state, jnp.asarray(0.0), malformed)


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.uint8, jnp.int64])
def test_run_continuous_actor_critic_from_arrays_rejects_integer_terminated(
    dtype: object,
) -> None:
    agent = ContinuousActorCriticAgent(ContinuousActorCriticConfig(action_dim=2))
    state = agent.init(2, jr.key(0))
    observations = jnp.zeros((2, 2), dtype=jnp.float32)
    rewards = jnp.zeros((2,), dtype=jnp.float32)
    with pytest.raises(TypeError, match="terminated must have dtype bool or float32"):
        run_continuous_actor_critic_from_arrays(
            agent,
            state,
            observations,
            rewards,
            jnp.array([0, 1], dtype=dtype),
            observations,
        )


def test_run_continuous_actor_critic_from_arrays_accepts_bool_and_float32_terminated() -> None:
    agent = ContinuousActorCriticAgent(ContinuousActorCriticConfig(action_dim=2))
    state = agent.init(2, jr.key(0))
    observations = jnp.zeros((2, 2), dtype=jnp.float32)
    rewards = jnp.zeros((2,), dtype=jnp.float32)
    for flags in (
        jnp.array([False, True], dtype=jnp.bool_),
        jnp.array([0.0, 1.0], dtype=jnp.float32),
    ):
        result = run_continuous_actor_critic_from_arrays(
            agent, state, observations, rewards, flags, observations
        )
        assert result.td_errors.shape == (2,)


def test_continuous_actor_critic_state_contract_and_counter_saturation() -> None:
    agent = ContinuousActorCriticAgent(ContinuousActorCriticConfig(action_dim=2))
    state = agent.init(2, jr.key(0))
    malformed = state.replace(mean_weights=jnp.zeros((2,), dtype=jnp.float32))
    with pytest.raises(ValueError, match="mean_weights"):
        agent.policy_params(malformed, jnp.zeros((2,), dtype=jnp.float32))
    saturated = state.replace(step_count=jnp.asarray(2**31 - 1, dtype=jnp.int32))
    result = agent.update(
        saturated,
        jnp.asarray(0.0),
        jnp.zeros((2,), dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    assert int(result.state.step_count) == 2**31 - 1
