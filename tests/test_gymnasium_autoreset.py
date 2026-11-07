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
    _transition_observations,
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
    *, monkeypatch: pytest.MonkeyPatch | None = None,
) -> tuple[_BoundaryEnv, gym.Env[NDArray[np.float32], Any], bool]:
    raw = _BoundaryEnv(terminated, horizon)
    if wrapper == "manual":
        return raw, raw, False
    if wrapper == "same_step":
        # Explicit legacy-type test double, not metadata-based detection. Real
        # installed wrapper contracts are exercised by the "official" cases.
        assert monkeypatch is not None
        monkeypatch.setattr(gym.wrappers, "AutoResetWrapper", _SameStep, raising=False)
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
    wrapper: str, terminated: bool, consumer: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, env, _ = _environment(wrapper, terminated, monkeypatch=monkeypatch)
    observations, targets, _ = _consume(env, consumer, 1)
    # V(last)=2: truncation bootstraps to 1 + .5*2; true termination does not.
    expected = 2.0 if consumer.endswith("next") or not terminated else 1.0
    np.testing.assert_array_equal(observations, [[10]])
    np.testing.assert_array_equal(targets, [[expected]])


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("terminated", [False, True], ids=["truncated", "terminated"])
@pytest.mark.parametrize("consumer", ["collector_value", "stream_value", "td_value"])
def test_boundaries_keep_reset_ownership_and_manual_seed_order(
    wrapper: str, terminated: bool, consumer: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, env, same_step = _environment(wrapper, terminated, horizon=2, monkeypatch=monkeypatch)
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
    wrapper: str, terminated: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, env, _ = _environment(wrapper, terminated, monkeypatch=monkeypatch)
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
def test_invalid_final_observation_is_not_hidden_by_valid_reset_observation(
    consumer: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _BoundaryEnv(False)
    env = _SameStep(raw)
    monkeypatch.setattr(gym.wrappers, "AutoResetWrapper", _SameStep, raising=False)
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


class _MetadataOnlyEnv(_BoundaryEnv):
    """The reviewer's scalar counterexample: final metadata without any reset."""

    needs_reset = True

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        self.needs_reset = False
        return super().reset(seed=seed, options=options)

    def step(
        self, action: Any,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self.needs_reset:
            raise gym.error.ResetNeeded("step called on an ended environment before reset")
        obs, reward, term, trunc, info = super().step(action)
        self.needs_reset = term or trunc
        if self.needs_reset:
            info["final_observation"] = np.asarray([2], np.float32)
        return obs, reward, term, trunc, info


@pytest.mark.parametrize("consumer", CONSUMERS)
@pytest.mark.parametrize("terminated", [False, True], ids=["truncated", "terminated"])
@pytest.mark.parametrize("outer", [False, True], ids=["bare", "wrapped"])
def test_terminal_metadata_does_not_own_reset(
    consumer: str, terminated: bool, outer: bool,
) -> None:
    raw = _MetadataOnlyEnv(terminated)
    env = gym.Wrapper(raw) if outer else raw
    observations, targets, stream = _consume(env, consumer, 2)
    np.testing.assert_array_equal(observations, [[10], [10]])
    expected = 2 if consumer.endswith("next") or not terminated else 1
    np.testing.assert_array_equal(targets, [[expected], [expected]])
    assert raw.actions == [0, 0]
    assert raw.seeds == ([7, 8, 9] if stream is None else [7, 8])
    if stream is not None:
        assert stream.episode_count == stream.step_count == 2


@pytest.mark.parametrize("terminated", [False, True], ids=["truncated", "terminated"])
def test_terminal_metadata_does_not_transfer_bootstrap_action_to_new_episode(
    terminated: bool,
) -> None:
    raw = _MetadataOnlyEnv(terminated)
    calls: list[float] = []

    def policy(obs: Any) -> int:
        calls.append(float(obs[0]))
        return int(obs[0] > 5)

    stream = TDStream(raw, policy=policy, gamma=0.5, seed=7, include_action_in_features=True)
    stream.update_value_function(lambda x: float(x[0] + 100 * x[1]))
    results = [next(stream), next(stream)]
    np.testing.assert_array_equal([x.target for x in results], [[1 if terminated else 2]] * 2)
    assert calls == ([10, 10] if terminated else [10, 2, 10, 2])
    assert raw.actions == [1, 1]
    assert raw.seeds == [7, 8]


class _FlipAction(gym.ActionWrapper[Any, Any, Any]):
    def action(self, action: Any) -> int:
        return 1 - int(action)


class _DoubleReward(gym.RewardWrapper[Any, Any]):
    def reward(self, reward: Any) -> float:
        return 2 * float(reward)


@pytest.mark.parametrize("consumer", CONSUMERS)
@pytest.mark.parametrize("terminated", [False, True], ids=["truncated", "terminated"])
@pytest.mark.parametrize("outer", ["plain", "action", "reward", "nested"])
def test_real_official_wrapper_under_contract_preserving_outers(
    consumer: str, terminated: bool, outer: str,
) -> None:
    raw, env, same_step = _environment("official", terminated)
    if outer in ("action", "nested"):
        env = _FlipAction(env)
    if outer in ("reward", "nested"):
        env = _DoubleReward(env)
    env = gym.Wrapper(env)
    observations, targets, stream = _consume(env, consumer, 2)
    np.testing.assert_array_equal(observations, [[10], [10]])
    reward = 2 if outer in ("reward", "nested") else 1
    expected = 2 if consumer.endswith("next") else reward + (0 if terminated else 1)
    np.testing.assert_array_equal(targets, [[expected], [expected]])
    assert raw.actions == ([1, 1] if outer in ("action", "nested") else [0, 0])
    assert raw.seeds == ([7, None, None] if same_step else [7, 8, 9] if stream is None else [7, 8])


class _ObservationOffset(gym.ObservationWrapper[Any, Any, Any]):
    def observation(self, observation: Any) -> Any:
        return observation + 10


class _OuterBoundary(gym.Wrapper[Any, Any, Any, Any]):
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, term, _, info = self.env.step(action)
        info["final_observation"] = obs
        return obs, float(reward), term, True, info


@pytest.mark.parametrize("consumer", CONSUMERS)
@pytest.mark.parametrize("outer", ["observation", "boundary"])
def test_unverified_outer_same_step_contract_is_rejected(
    consumer: str, outer: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # These outers either leave final_observation in another coordinate system
    # or introduce a done flag the inner autoreset wrapper never saw.
    raw, env, _ = _environment("same_step", False, monkeypatch=monkeypatch)
    if outer == "observation":
        env = _ObservationOffset(env)
    else:
        raw.horizon = 2
        env = _OuterBoundary(env)
    with pytest.raises(ValueError, match="same-step AutoResetWrapper requires"):
        _consume(env, consumer, 1)
    assert len(raw.actions) == 1


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_info_flags_do_not_declare_same_step_reset(consumer: str) -> None:
    class ClaimedMetadata(_MetadataOnlyEnv):
        metadata = {"autoreset_mode": "SameStep", "autoreset": True}

        def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            obs, reward, term, trunc, info = super().step(action)
            info["final_info"] = {}
            return obs, reward, term, trunc, info

    raw = ClaimedMetadata(False)
    _consume(raw, consumer, 2)
    assert raw.seeds[:2] == [7, 8]


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_official_outer_time_limit_does_not_imply_an_inner_reset(consumer: str) -> None:
    class IncidentalInfo(_MetadataOnlyEnv):
        def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            obs, reward, term, trunc, info = super().step(action)
            info["final_observation"] = obs
            info["final_info"] = {}
            return obs, reward, term, trunc, info

    raw = IncidentalInfo(False, horizon=2)
    same_step = int(gym.__version__.split(".")[0]) == 0
    constructor = getattr(gym.wrappers, "AutoResetWrapper" if same_step else "Autoreset")
    env: gym.Env[NDArray[np.float32], Any] = gym.wrappers.TimeLimit(
        constructor(raw), max_episode_steps=1,
    )
    if same_step:
        with pytest.raises(ValueError, match="same-step AutoResetWrapper requires"):
            _consume(env, consumer, 2)
        assert raw.seeds == [7]  # The inner wrapper never saw the outer truncation.
        assert raw.actions == [0]
    else:
        observations, targets, stream = _consume(env, consumer, 2)
        np.testing.assert_array_equal(observations, [[10], [10]])
        np.testing.assert_array_equal(targets, [[1 if consumer.endswith("next") else 1.5]] * 2)
        assert raw.seeds == ([7, 8, 9] if stream is None else [7, 8])
        assert raw.actions == [0, 0]


def test_claimed_same_step_contract_requires_final_observation() -> None:
    raw = _BoundaryEnv(False)
    with pytest.raises(ValueError, match="same-step autoreset must provide final_observation"):
        _transition_observations(
            np.asarray([10], np.float32), {}, raw.observation_space,
            done=True, same_step_autoreset=True,
        )


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_manual_metadata_cannot_replace_the_returned_observation(consumer: str) -> None:
    class UnrelatedMetadata(_MetadataOnlyEnv):
        def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            obs, reward, term, trunc, info = super().step(action)
            info["final_observation"] = {"unrelated": "not an observation"}
            return obs, reward, term, trunc, info

    raw = UnrelatedMetadata(False)
    # The outer observation transform must apply to learning observations too;
    # incidental raw metadata must not override it.
    observations, targets, stream = _consume(_ObservationOffset(raw), consumer, 2)
    np.testing.assert_array_equal(observations, [[20], [20]])
    np.testing.assert_array_equal(targets, [[12 if consumer.endswith("next") else 7]] * 2)
    assert raw.seeds == ([7, 8, 9] if stream is None else [7, 8])


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_replaced_legacy_step_does_not_inherit_reset_ownership(
    consumer: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _MetadataOnlyEnv(False)
    env = _SameStep(raw)
    monkeypatch.setattr(gym.wrappers, "AutoResetWrapper", _SameStep, raising=False)
    monkeypatch.setattr(env, "step", raw.step)
    with pytest.raises(ValueError, match="same-step AutoResetWrapper requires"):
        _consume(env, consumer, 1)
    assert raw.seeds == [7]
    assert raw.actions == [0]
