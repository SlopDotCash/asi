"""Episode-boundary targets and reset ownership through real Gymnasium wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

if TYPE_CHECKING:
    import gymnasium as gym
else:
    gym = pytest.importorskip("gymnasium")

from alberta_framework.streams.gymnasium import (  # noqa: E402
    GymnasiumStream,
    PredictionMode,
    TDStream,
    collect_trajectory,
)

pytestmark = pytest.mark.unit
CONSUMERS = ("collector_value", "stream_value", "td_value", "collector_next", "stream_next")
WRAPPERS = ("manual", "same_step", "official")


class _BoundaryEnv(gym.Env[NDArray[np.float32], Any]):
    """Known transitions: reset 10, intermediate 1, last observation 2, reward 1."""

    def __init__(self, terminated: bool, horizon: int = 1) -> None:
        self.observation_space = gym.spaces.Box(-100, 100, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.terminated = terminated
        self.horizon = horizon
        self.episode_step = 0
        self.seeds: list[int | None] = []
        self.actions: list[int] = []

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        self.seeds.append(seed)
        self.episode_step = 0
        return np.asarray([10], np.float32), {}

    def step(
        self, action: Any,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        self.actions.append(int(action))
        self.episode_step += 1
        done = self.episode_step == self.horizon
        return (
            np.asarray([2 if done else 1], np.float32), 1.0,
            done and self.terminated, done and not self.terminated, {},
        )


class _SameStep(gym.Wrapper[NDArray[np.float32], Any, NDArray[np.float32], Any]):
    """Keep the 0.29 scalar contract covered even when CI installs newer Gymnasium."""

    final_override: NDArray[np.float32] | None = None

    def step(
        self, action: Any,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            reset_obs, reset_info = self.env.reset()
            reset_info["final_observation"] = (
                obs if self.final_override is None else self.final_override
            )
            reset_info["final_info"] = info
            obs, info = reset_obs, reset_info
        return obs, float(reward), terminated, truncated, info


def _environment(
    wrapper: str, terminated: bool, horizon: int = 1,
) -> tuple[_BoundaryEnv, gym.Env[NDArray[np.float32], Any], bool]:
    raw = _BoundaryEnv(terminated, horizon)
    if wrapper == "manual":
        return raw, raw, False
    if wrapper == "same_step":
        return raw, _SameStep(raw), True
    old_contract = int(gym.__version__.split(".")[0]) == 0
    constructor = getattr(gym.wrappers, "AutoResetWrapper" if old_contract else "Autoreset")
    env: gym.Env[NDArray[np.float32], Any] = constructor(raw)
    return raw, env, old_contract


def _consume(
    env: gym.Env[NDArray[np.float32], Any], consumer: str, steps: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], GymnasiumStream | TDStream | None]:
    mode = PredictionMode.NEXT_STATE if consumer.endswith("next") else PredictionMode.VALUE
    if consumer.startswith("collector"):
        observations, targets = collect_trajectory(
            env, lambda _obs: 0, steps, mode=mode, include_action_in_features=False,
            seed=7, value_estimator=lambda x: float(x[0]), gamma=0.5,
        )
        return np.asarray(observations), np.asarray(targets), None
    stream: GymnasiumStream | TDStream
    if consumer == "td_value":
        stream = TDStream(env, policy=lambda _obs: 0, gamma=0.5, seed=7)
        stream.update_value_function(lambda x: float(x[0]))
    else:
        stream = GymnasiumStream(
            env, mode=mode, policy=lambda _obs: 0, gamma=0.5,
            include_action_in_features=False, seed=7,
        )
        stream.set_value_estimator(lambda x: float(x[0]))
    results = [next(stream) for _ in range(steps)]
    return (
        np.stack([np.asarray(x.observation) for x in results]),
        np.stack([np.asarray(x.target) for x in results]), stream,
    )


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("terminated", [False, True], ids=["truncated", "terminated"])
@pytest.mark.parametrize("consumer", CONSUMERS)
def test_last_observation_drives_the_learning_target(
    wrapper: str, terminated: bool, consumer: str,
) -> None:
    _, env, _ = _environment(wrapper, terminated)
    observations, targets, _ = _consume(env, consumer, 1)
    # V(last)=2: truncation bootstraps to 1 + .5*2; true termination does not.
    expected = 2.0 if consumer.endswith("next") or not terminated else 1.0
    np.testing.assert_array_equal(observations, [[10]])
    np.testing.assert_array_equal(targets, [[expected]])


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("terminated", [False, True], ids=["truncated", "terminated"])
@pytest.mark.parametrize("consumer", ["collector_value", "stream_value", "td_value"])
def test_boundaries_keep_reset_ownership_and_manual_seed_order(
    wrapper: str, terminated: bool, consumer: str,
) -> None:
    raw, env, same_step = _environment(wrapper, terminated, horizon=2)
    observations, targets, stream = _consume(env, consumer, 4)
    np.testing.assert_array_equal(observations[:, 0], [10, 1, 10, 1])
    boundary_target = 1.0 if terminated else 2.0
    np.testing.assert_array_equal(targets[:, 0], [1.5, boundary_target, 1.5, boundary_target])
    assert len(raw.actions) == 4  # No synthetic reset-only transition was learned.
    if same_step:
        assert raw.seeds == [7, None, None]  # Wrapper owns the two resets, without reseeding.
    else:
        assert raw.seeds == ([7, 8, 9] if stream is None else [7, 8])
    if stream is not None:
        assert stream.step_count == 4
        assert stream.episode_count == 2


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("terminated", [False, True], ids=["truncated", "terminated"])
def test_td_bootstrap_action_is_not_the_next_episode_action(
    wrapper: str, terminated: bool,
) -> None:
    raw, env, _ = _environment(wrapper, terminated)
    calls: list[float] = []

    def policy(obs: Any) -> int:
        calls.append(float(obs[0]))
        return int(obs[0] > 5)

    stream = TDStream(env, policy=policy, gamma=0.5, seed=7, include_action_in_features=True)
    stream.update_value_function(lambda x: float(x[0] + 100 * x[1]))
    steps = [next(stream), next(stream)]
    np.testing.assert_array_equal(
        [np.asarray(x.target) for x in steps], [[1 if terminated else 2]] * 2,
    )
    assert raw.actions == [1, 1]
    assert calls == ([10, 10] if terminated else [10, 2, 10, 2])


@pytest.mark.parametrize("terminated", [False, True], ids=["truncated", "terminated"])
def test_installed_official_wrapper_contract(terminated: bool) -> None:
    raw, env, same_step = _environment("official", terminated)
    env.reset(seed=7)
    obs, reward, term, trunc, info = env.step(0)
    assert (float(reward), term, trunc) == (1.0, terminated, not terminated)
    if same_step:
        np.testing.assert_array_equal(obs, [10])
        np.testing.assert_array_equal(info["final_observation"], [2])
        assert raw.seeds == [7, None]
    else:
        np.testing.assert_array_equal(obs, [2])
        assert "final_observation" not in info and "final_obs" not in info
        assert raw.seeds == [7]
        obs, reward, term, trunc, _ = env.step(0)
        np.testing.assert_array_equal(obs, [10])
        assert (float(reward), term, trunc) == (0.0, False, False)
        assert raw.seeds == [7, None]


@pytest.mark.parametrize("consumer", ["collector_value", "stream_value", "td_value"])
def test_invalid_final_observation_is_not_hidden_by_valid_reset_observation(consumer: str) -> None:
    raw = _BoundaryEnv(False)
    env = _SameStep(raw)
    env.final_override = np.asarray([2, 3], np.float32)
    with pytest.raises(ValueError, match="observation.*shape"):
        _consume(env, consumer, 1)


def test_nonterminal_info_does_not_claim_a_reset() -> None:
    class IncidentalInfo(_BoundaryEnv):
        def step(
            self, action: Any,
        ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
            obs, reward, term, trunc, info = super().step(action)
            info["final_observation"] = np.asarray([99], np.float32)
            return obs, reward, term, trunc, info

    raw = IncidentalInfo(False, horizon=2)
    stream = GymnasiumStream(raw, mode=PredictionMode.NEXT_STATE, policy=lambda _obs: 0)
    np.testing.assert_array_equal(next(stream).target, jnp.asarray([1]))


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_terminal_incidental_final_observation_does_not_claim_a_reset(consumer: str) -> None:
    class IncidentalInfo(_BoundaryEnv):
        def step(
            self, action: Any,
        ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
            if self.episode_step >= self.horizon:
                raise AssertionError("step called without a reset")
            obs, reward, term, trunc, info = super().step(action)
            info["final_observation"] = np.asarray([99], np.float32)
            return obs, reward, term, trunc, info

    raw = IncidentalInfo(False, horizon=1)
    observations, targets, _ = _consume(raw, consumer, 2)
    np.testing.assert_array_equal(observations, [[10], [10]])
    np.testing.assert_array_equal(targets, [[2], [2]])
