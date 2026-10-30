"""Integer Box exploration must produce actions accepted by the environment."""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")

from alberta_framework.streams.gymnasium import (  # noqa: E402
    collect_trajectory,
    make_epsilon_greedy_policy,
    make_random_policy,
)

pytestmark = pytest.mark.unit


class _IntegerBoxEnv(gymnasium.Env):
    observation_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def __init__(self, dtype: Any = np.int32) -> None:
        self.action_space = gymnasium.spaces.Box(0, 1, shape=(2, 2), dtype=dtype)
        self.actions: list[np.ndarray] = []

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        value = np.asarray(action)
        if not self.action_space.contains(value):
            raise ValueError("random policy produced an action outside the integer Box")
        self.actions.append(value.copy())
        return np.zeros(1, dtype=np.float32), float(value.sum()), False, False, {}


@pytest.mark.parametrize(
    "dtype",
    (np.bool_, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64),
)
def test_random_integer_box_preserves_dtype_and_closed_bounds(dtype: Any) -> None:
    env = _IntegerBoxEnv(dtype)
    policy = make_random_policy(env, seed=42)
    actions = [np.asarray(policy(jnp.zeros(1))) for _ in range(16)]

    assert all(action.dtype == np.dtype(dtype) for action in actions)
    assert all(env.action_space.contains(action) for action in actions)
    # The integer Box is closed: its high value must not be silently excluded.
    assert set(np.stack(actions).ravel()) == {0, 1}


def test_random_integer_box_preserves_signed_per_coordinate_bounds_and_seed() -> None:
    env = _IntegerBoxEnv()
    env.action_space = gymnasium.spaces.Box(
        low=np.asarray(((-2, 7), (-5, 0)), dtype=np.int32),
        high=np.asarray(((2, 7), (-3, 1)), dtype=np.int32),
        dtype=np.int32,
    )
    first = make_random_policy(env, seed=17)
    second = make_random_policy(env, seed=17)
    observed: list[np.ndarray] = []
    for _ in range(16):
        action = np.asarray(first(jnp.zeros(1)))
        np.testing.assert_array_equal(action, second(jnp.zeros(1)))
        assert env.action_space.contains(action)
        assert action[0, 1] == 7
        observed.append(action)
    assert len({tuple(value.ravel()) for value in observed}) > 1


@pytest.mark.parametrize("exploration", ("random", "epsilon"))
def test_integer_box_exploration_collects_a_valid_trajectory(exploration: str) -> None:
    env = _IntegerBoxEnv()
    policy = None
    if exploration == "epsilon":
        policy = make_epsilon_greedy_policy(
            lambda _obs: pytest.fail("epsilon=1 must use the exploration policy"),
            env,
            epsilon=1.0,
            seed=17,
        )
    features, targets = collect_trajectory(env, policy, num_steps=8, seed=17)

    assert features.shape == (8, 5)
    assert targets.shape == (8, 1)
    assert len(env.actions) == 8
    np.testing.assert_array_equal(np.asarray(features[:, 1:]), np.stack(env.actions).reshape(8, 4))
    np.testing.assert_array_equal(np.asarray(targets[:, 0]), np.stack(env.actions).sum(axis=(1, 2)))


@pytest.mark.parametrize(
    ("low", "high"),
    ((2**31 - 1, 2**31 - 1), (-(2**31) - 1, 0), (-(2**31), 2**31 - 2)),
)
def test_integer_box_random_policy_rejects_unrepresentable_jax_bounds(
    low: int, high: int
) -> None:
    env = _IntegerBoxEnv()
    env.action_space = gymnasium.spaces.Box(low, high, shape=(1,), dtype=np.int64)
    with pytest.raises(ValueError, match="integer Box.*signed int32"):
        make_random_policy(env, seed=17)
