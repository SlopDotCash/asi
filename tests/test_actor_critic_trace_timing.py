"""Equation-level discount timing and exact-resume checks for linear AC(lambda)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.actor_critic import (
    ActorCriticAgent,
    ActorCriticConfig,
    ContinuousActorCriticAgent,
    ContinuousActorCriticConfig,
    run_actor_critic_from_arrays,
    run_continuous_actor_critic_from_arrays,
)
from alberta_framework.core.checkpoints import load_checkpoint, save_checkpoint

pytestmark = pytest.mark.unit

_OBSERVATIONS = np.asarray(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)), dtype=np.float32)
_CASES = {
    "terminal": (0.9, 0.0),
    "variable": (0.25, 0.75),
    "constant": (0.9, 0.9),
    "td0": (0.25, 0.0),
}


def _agent(kind: str, case: str = "terminal") -> Any:
    kwargs = {
        "gamma": 0.9,
        "actor_step_size": 0.1,
        "critic_step_size": 0.2,
        "actor_lamda": 0.0 if case == "td0" else 0.8,
        "critic_lamda": 0.0 if case == "td0" else 0.7,
    }
    if kind == "discrete":
        return ActorCriticAgent(ActorCriticConfig(n_actions=2, **kwargs))
    return ContinuousActorCriticAgent(ContinuousActorCriticConfig(action_dim=1, **kwargs))


def _fixed_action(state: Any, kind: str) -> Any:
    if kind == "discrete":
        return state.replace(last_action=jnp.int32(0))
    action = jnp.asarray([0.5], dtype=jnp.float32)
    return state.replace(last_action=action, last_sample=action)


def _equations(
    kind: str,
    case: str,
    *,
    observations: Any = _OBSERVATIONS,
    rewards: tuple[float, ...] = (0.0, 1.0),
    discounts: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """NumPy backward view, with the discount *into* S_t on the old trace.

    Fixed actions make the policy score independent of the implementation's
    action sampler. Parameters, scores, traces, and TD errors are recomputed
    here after every update, including across an episode boundary.
    """
    actor_lambda = 0.0 if case == "td0" else 0.8
    critic_lambda = 0.0 if case == "td0" else 0.7
    width = 2 if kind == "discrete" else 1
    actor_w, actor_b = np.zeros((width, 2)), np.zeros(width)
    actor_ew, actor_eb = np.zeros_like(actor_w), np.zeros_like(actor_b)
    log_sigma = np.full(width, -0.5 if kind == "continuous" else 0.0)
    sigma_e = np.zeros(width)
    critic_w, critic_b = np.zeros(2), 0.0
    critic_ew, critic_eb = np.zeros(2), 0.0
    previous_discount = 0.9
    for index, discount in enumerate(_CASES[case] if discounts is None else discounts):
        obs, next_obs = observations[index:index + 2].astype(np.float64)
        reward = rewards[index]
        delta = reward + discount * (critic_w @ next_obs + critic_b)
        delta -= critic_w @ obs + critic_b
        if kind == "discrete":
            logits = actor_w @ obs + actor_b
            policy = np.exp(logits - logits.max())
            policy /= policy.sum()
            score = np.asarray([1.0, 0.0]) - policy
        else:
            standardized = (0.5 - (actor_w @ obs + actor_b)) / np.exp(log_sigma)
            score = standardized / np.exp(log_sigma)
            sigma_e = previous_discount * actor_lambda * sigma_e + standardized**2 - 1.0
        actor_ew = previous_discount * actor_lambda * actor_ew + score[:, None] * obs
        actor_eb = previous_discount * actor_lambda * actor_eb + score
        critic_ew = previous_discount * critic_lambda * critic_ew + obs
        critic_eb = previous_discount * critic_lambda * critic_eb + 1.0
        actor_w += 0.1 * delta * actor_ew
        actor_b += 0.1 * delta * actor_eb
        critic_w += 0.2 * delta * critic_ew
        critic_b += 0.2 * delta * critic_eb
        if kind == "continuous":
            log_sigma += 0.1 * delta * sigma_e
        if discount == 0.0:
            actor_ew, actor_eb = np.zeros_like(actor_ew), np.zeros_like(actor_eb)
            critic_ew, critic_eb = np.zeros_like(critic_ew), 0.0
            sigma_e = np.zeros_like(sigma_e)
        previous_discount = discount
    prefix = "actor" if kind == "discrete" else "mean"
    expected = {
        f"{prefix}_weights": actor_w,
        f"{prefix}_bias": actor_b,
        f"{prefix}_trace_weights": actor_ew,
        f"{prefix}_trace_bias": actor_eb,
        "critic_weights": critic_w,
        "critic_bias": critic_b,
        "critic_trace_weights": critic_ew,
        "critic_trace_bias": critic_eb,
    }
    if kind == "continuous":
        expected.update(log_sigma=log_sigma, log_sigma_trace=sigma_e)
    return expected


def _assert_equations(state: Any, kind: str, case: str) -> None:
    for name, expected in _equations(kind, case).items():
        np.testing.assert_allclose(
            np.asarray(getattr(state, name)), expected, rtol=2e-6, atol=1e-7,
            err_msg=name,
        )


def _assert_exact(left: Any, right: Any) -> None:
    assert repr(jax.tree.structure(left)) == repr(jax.tree.structure(right))
    for actual, expected in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if jnp.issubdtype(actual.dtype, jax.dtypes.prng_key):
            actual, expected = jr.key_data(actual), jr.key_data(expected)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("kind", ["discrete", "continuous"])
@pytest.mark.parametrize("eager", [True, False], ids=["eager", "jit"])
@pytest.mark.parametrize("case", _CASES)
def test_updates_match_discount_into_current_state_equations(
    kind: str, eager: bool, case: str,
) -> None:
    agent = _agent(kind, case)
    with jax.disable_jit(eager):
        state = agent.start(agent.init(2, jr.key(123)), jnp.asarray(_OBSERVATIONS[0]))[0]
        expected_key = state.rng_key
        for index, discount in enumerate(_CASES[case]):
            state = _fixed_action(state, kind)
            result = agent.update(
                state, jnp.float32(index), jnp.asarray(_OBSERVATIONS[index + 1]),
                discount=jnp.float32(discount),
            )
            assert bool(result.update_applied)
            state = result.state
            expected_key = jr.split(expected_key)[0]
        _assert_equations(state, kind, case)
        np.testing.assert_array_equal(jr.key_data(state.rng_key), jr.key_data(expected_key))


@pytest.mark.parametrize("kind", ["discrete", "continuous"])
@pytest.mark.parametrize("case", _CASES)
def test_public_array_runners_match_equations_and_preserve_rng_schedule(
    kind: str, case: str,
) -> None:
    agent = _agent(kind, case)
    state = agent.init(2, jr.key(123))
    kwargs = {
        "observations": jnp.asarray(_OBSERVATIONS[:-1]),
        "rewards": jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        "terminated": jnp.asarray([False, True]),
        "next_observations": jnp.asarray(_OBSERVATIONS[1:]),
        # Explicit discounts take precedence over the legacy terminal flags.
        "discounts": jnp.asarray(_CASES[case], dtype=jnp.float32),
    }
    result: Any
    if kind == "discrete":
        result = run_actor_critic_from_arrays(
            agent, state, actions=jnp.zeros(2, jnp.int32), **kwargs,
        )
    else:
        result = run_continuous_actor_critic_from_arrays(
            agent, state, actions=jnp.full((2, 1), 0.5, jnp.float32), **kwargs,
        )
    assert bool(jnp.all(result.updates_applied))
    _assert_equations(result.state, kind, case)
    expected_key = jr.split(jr.split(state.rng_key)[0])[0]
    np.testing.assert_array_equal(jr.key_data(result.state.rng_key), jr.key_data(expected_key))


@pytest.mark.parametrize("kind", ["discrete", "continuous"])
@pytest.mark.parametrize("eager", [True, False], ids=["eager", "jit"])
def test_learning_policy_and_critic_match_equations_across_terminal_boundary(
    kind: str, eager: bool,
) -> None:
    observations = np.asarray(
        ((1., 0.), (0., 1.), (0.5, -0.5), (-1., 0.5), (0.25, 1.)), dtype=np.float32,
    )
    rewards = (0.2, -0.3, 1.0, 0.4)
    discounts = (0.25, 0.75, 0.0, 0.9)
    agent = _agent(kind)
    with jax.disable_jit(eager):
        state = agent.start(agent.init(2, jr.key(123)), jnp.asarray(observations[0]))[0]
        for index, discount in enumerate(discounts):
            state = _fixed_action(state, kind)
            result = agent.update(
                state, rewards[index], jnp.asarray(observations[index + 1]), discount=discount,
            )
            assert bool(result.update_applied)
            state = result.state
            expected = _equations(
                kind, "terminal", observations=observations,
                rewards=rewards[:index + 1], discounts=discounts[:index + 1],
            )
            for name, value in expected.items():
                np.testing.assert_allclose(
                    np.asarray(getattr(state, name)), value, rtol=3e-6, atol=1e-7,
                    err_msg=f"step {index}: {name}",
                )
            assert float(state.previous_discount) == pytest.approx(discount)


@pytest.mark.parametrize("kind", ["discrete", "continuous"])
@pytest.mark.parametrize("failure", ["reward", "discount"])
def test_rejected_update_preserves_discount_cursor_and_rng(kind: str, failure: str) -> None:
    agent = _agent(kind)
    state = agent.start(agent.init(2, jr.key(123)), jnp.asarray(_OBSERVATIONS[0]))[0]
    state = agent.update(state, 0.0, jnp.asarray(_OBSERVATIONS[1]), discount=0.25).state
    rejected = agent.update(
        state, float("nan") if failure == "reward" else 1.0,
        jnp.asarray(_OBSERVATIONS[2]), discount=1.5 if failure == "discount" else 0.75,
    )
    assert not bool(rejected.update_applied)
    _assert_exact(rejected.state, state)
    assert float(rejected.state.previous_discount) == 0.25
    direct = agent.update(state, 1.0, jnp.asarray(_OBSERVATIONS[2]), discount=0.75)
    retried = agent.update(rejected.state, 1.0, jnp.asarray(_OBSERVATIONS[2]), discount=0.75)
    _assert_exact(retried, direct)


@pytest.mark.parametrize("kind", ["discrete", "continuous"])
@pytest.mark.parametrize("discount", [-0.1, 1.1, float("nan")])
def test_invalid_previous_discount_rejects_without_mutation(kind: str, discount: float) -> None:
    agent = _agent(kind)
    state = agent.start(agent.init(2, jr.key(123)), jnp.asarray(_OBSERVATIONS[0]))[0]
    state = state.replace(previous_discount=jnp.float32(discount))
    result = agent.update(state, 1.0, jnp.asarray(_OBSERVATIONS[1]))
    assert not bool(result.update_applied)
    _assert_exact(result.state, state)


@pytest.mark.parametrize("kind", ["discrete", "continuous"])
def test_checkpoint_restores_discount_cursor_and_exact_continuation(
    kind: str, tmp_path: Path,
) -> None:
    agent = _agent(kind)
    state = agent.start(agent.init(2, jr.key(123)), jnp.asarray(_OBSERVATIONS[0]))[0]
    state = agent.update(state, 0.0, jnp.asarray(_OBSERVATIONS[1]), discount=0.25).state
    path = tmp_path / "current"
    save_checkpoint(state, path)
    restored, _ = load_checkpoint(agent.init(2, jr.key(999)), path)
    assert float(restored.previous_discount) == 0.25
    _assert_exact(restored, state)
    direct = agent.update(state, 1.0, jnp.asarray(_OBSERVATIONS[2]), discount=0.75)
    resumed = agent.update(restored, 1.0, jnp.asarray(_OBSERVATIONS[2]), discount=0.75)
    _assert_exact(resumed, direct)


@pytest.mark.parametrize("kind", ["discrete", "continuous"])
def test_legacy_checkpoint_without_discount_cursor_is_explicitly_rejected(
    kind: str, tmp_path: Path,
) -> None:
    agent = _agent(kind)
    state = agent.init(2, jr.key(123))
    # The old serialized mapping has exactly these leaves, without the cursor.
    legacy = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(state) if field.name != "previous_discount"
    }
    path = tmp_path / "legacy"
    save_checkpoint(legacy, path)
    with pytest.raises(ValueError, match="checkpoint state leaf count"):
        load_checkpoint(state, path)
