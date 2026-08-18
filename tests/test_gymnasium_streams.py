"""Tests for Gymnasium experience streams."""

import jax.numpy as jnp
import numpy as np
import pytest

# Skip all tests if gymnasium is not installed
gymnasium = pytest.importorskip("gymnasium")

import alberta_framework.streams.gymnasium as gymnasium_stream_module  # noqa: E402
from alberta_framework import TimeStep  # noqa: E402
from alberta_framework.streams.gymnasium import (  # noqa: E402
    GymnasiumStream,
    PredictionMode,
    TDStream,
    _flatten_action,
    _flatten_observation,
    _flatten_space,
    collect_trajectory,
    make_epsilon_greedy_policy,
    make_gymnasium_stream,
    make_random_policy,
)


class TestPredictionMode:
    """Tests for PredictionMode enum."""

    def test_has_all_modes(self):
        """Enum should have all expected modes."""
        assert PredictionMode.REWARD.value == "reward"
        assert PredictionMode.NEXT_STATE.value == "next_state"
        assert PredictionMode.VALUE.value == "value"


@pytest.mark.parametrize("invalid", (True, float("nan"), float("inf"), -0.1, 1.1))
def test_gamma_entry_points_reject_invalid_float32_probabilities(invalid: object) -> None:
    env = gymnasium.make("CartPole-v1")
    try:
        with pytest.raises(ValueError, match="gamma"):
            GymnasiumStream(env, gamma=invalid)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="gamma"):
            TDStream(env, gamma=invalid)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="gamma"):
            collect_trajectory(  # type: ignore[arg-type]
                env, None, num_steps=1, gamma=invalid
            )
        with pytest.raises(ValueError, match="gamma"):
            make_gymnasium_stream(  # type: ignore[arg-type]
                "CartPole-v1", gamma=invalid
            )
    finally:
        env.close()


def test_probability_entry_points_normalize_hostile_real_failures_without_repr() -> None:
    class HostileFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            raise RuntimeError("untrusted ratio hook")

        def __repr__(self) -> str:
            raise AssertionError("repr hook executed")

    env = gymnasium.make("CartPole-v1")
    try:
        with pytest.raises(ValueError, match="gamma"):
            GymnasiumStream(env, gamma=HostileFloat(0.5))
        with pytest.raises(ValueError, match="epsilon"):
            make_epsilon_greedy_policy(
                lambda _observation: 0,
                env,
                epsilon=HostileFloat(0.5),
            )
    finally:
        env.close()


@pytest.mark.parametrize(
    "integer_type",
    tuple(
        dict.fromkeys(
            (
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
                np.longlong,
                np.ulonglong,
            )
        )
    ),
)
def test_num_steps_accepts_all_numpy_integer_families(integer_type: type[np.integer]) -> None:
    env = gymnasium.make("CartPole-v1")
    try:
        observations, targets = collect_trajectory(env, None, integer_type(1))
        assert observations.shape == (1, 5)
        assert targets.shape == (1, 1)
    finally:
        env.close()


def test_schedule_and_shape_contracts_fail_before_environment_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = gymnasium.make("CartPole-v1")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("environment executed before trajectory preflight")

    monkeypatch.setattr(env, "reset", forbidden)
    try:
        for num_steps in (0, -1, True, 1.5, "1"):
            with pytest.raises(ValueError, match="num_steps"):
                collect_trajectory(env, None, num_steps)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="trajectory output"):
            collect_trajectory(env, None, 2**31 - 1)
        with pytest.raises(ValueError, match="exact bool"):
            collect_trajectory(env, None, 1, include_action_in_features=1)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="PredictionMode"):
            collect_trajectory(env, None, 1, mode="reward")  # type: ignore[arg-type]
    finally:
        env.close()


def test_seed_contracts_reject_aliases_and_spoofs_without_shrinking_uint32_domain() -> None:
    class Spoof:
        @property
        def __class__(self) -> type[int]:  # type: ignore[override,misc]
            return int

        def __index__(self) -> int:
            raise AssertionError("unapproved index hook executed")

        def __repr__(self) -> str:
            raise AssertionError("error path invoked repr")

    env = gymnasium.make("CartPole-v1")
    try:
        for seed in (True, np.uint32(1), -1, 2**32, Spoof()):
            with pytest.raises(ValueError, match="seed"):
                make_random_policy(env, seed=seed)  # type: ignore[arg-type]
        policy = make_epsilon_greedy_policy(
            lambda _observation: 0,
            env,
            epsilon=1.0,
            seed=2**32 - 1,
        )
        assert env.action_space.contains(policy(jnp.zeros(4)))
    finally:
        env.close()


def test_random_policy_respects_nonzero_and_multiaxis_discrete_starts() -> None:
    class StubEnv:
        def __init__(self, action_space: object):
            self.action_space = action_space

    discrete = gymnasium.spaces.Discrete(3, start=5)
    discrete_policy = make_random_policy(StubEnv(discrete), seed=1)  # type: ignore[arg-type]
    assert {discrete_policy(jnp.zeros(1)) for _ in range(20)} <= {5, 6, 7}

    multi = gymnasium.spaces.MultiDiscrete(
        np.asarray(((2, 3), (4, 5))),
        start=np.asarray(((10, 20), (30, 40))),
    )
    multi_policy = make_random_policy(StubEnv(multi), seed=2)  # type: ignore[arg-type]
    action = multi_policy(jnp.zeros(1))
    assert action.shape == (2, 2)
    assert np.all(action >= multi.start)
    assert np.all(action < multi.start + multi.nvec)
    assert _flatten_space(multi) == 4
    assert _flatten_observation(multi.start, multi).shape == (4,)
    assert _flatten_action(action, multi).shape == (4,)


def test_random_policy_rejects_discrete_ranges_wider_than_jax_int32() -> None:
    class StubEnv:
        def __init__(self, action_space: object):
            self.action_space = action_space

    discrete = gymnasium.spaces.Discrete(2**31, start=-(2**30))
    with pytest.raises(ValueError, match="signed int32"):
        make_random_policy(StubEnv(discrete), seed=0)  # type: ignore[arg-type]

    multi = gymnasium.spaces.MultiDiscrete(
        np.asarray([2**31], dtype=np.uint32),
        start=np.asarray([-(2**30)], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="signed int32"):
        make_random_policy(StubEnv(multi), seed=0)  # type: ignore[arg-type]


def test_random_box_policy_rejects_nonfinite_bounds() -> None:
    class StubEnv:
        action_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)

    with pytest.raises(ValueError, match="finite ordered"):
        make_random_policy(StubEnv(), seed=0)  # type: ignore[arg-type]


def test_random_box_policy_samples_full_finite_float32_domain() -> None:
    maximum = np.finfo(np.float32).max

    class StubEnv:
        action_space = gymnasium.spaces.Box(
            np.asarray((-maximum, -maximum, maximum / 2), dtype=np.float32),
            np.asarray((maximum, -maximum / 2, maximum), dtype=np.float32),
            dtype=np.float32,
        )

    policy = make_random_policy(StubEnv(), seed=0)  # type: ignore[arg-type]
    for _ in range(20):
        action = policy(jnp.zeros(1, dtype=jnp.float32))
        assert bool(jnp.all(jnp.isfinite(action)))
        assert bool(jnp.all(action >= StubEnv.action_space.low))
        assert bool(jnp.all(action < StubEnv.action_space.high))


def test_random_box_preflights_dimension_before_jax_bound_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubEnv:
        action_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    monkeypatch.setattr(
        gymnasium_stream_module,
        "_flatten_space",
        lambda _space: (_ for _ in ()).throw(ValueError("dimension preflight")),
    )
    monkeypatch.setattr(
        gymnasium_stream_module.jnp,
        "asarray",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("JAX conversion ran before dimension preflight")
        ),
    )
    with pytest.raises(ValueError, match="dimension preflight"):
        make_random_policy(StubEnv(), seed=0)  # type: ignore[arg-type]


def test_epsilon_wrapper_rejects_base_policy_before_environment_access() -> None:
    class HostileEnv:
        @property
        def action_space(self) -> object:
            raise AssertionError("environment accessed before base-policy validation")

    with pytest.raises(ValueError, match="base_policy"):
        make_epsilon_greedy_policy(object(), HostileEnv())  # type: ignore[arg-type]


def test_runtime_values_must_match_declared_flattened_shapes_and_be_finite() -> None:
    box = gymnasium.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    with pytest.raises(ValueError, match="observation.*shape"):
        _flatten_observation(np.zeros((1,), dtype=np.float32), box)
    with pytest.raises(ValueError, match="action.*shape"):
        _flatten_action(np.zeros((3,), dtype=np.float32), box)
    with pytest.raises(ValueError, match="observation.*finite"):
        _flatten_observation(np.asarray((0.0, np.nan), dtype=np.float32), box)
    with pytest.raises(ValueError, match="action.*finite"):
        _flatten_action(np.asarray((0.0, np.inf), dtype=np.float32), box)


def test_runtime_shape_metadata_rejects_before_jax_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    box = gymnasium.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    class OversizedStub:
        shape = (2**31 - 1,)

        def __array__(self, *_args: object, **_kwargs: object) -> np.ndarray:
            raise AssertionError("array conversion ran before shape preflight")

    monkeypatch.setattr(
        gymnasium_stream_module.jnp,
        "asarray",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("JAX conversion ran before shape preflight")
        ),
    )
    with pytest.raises(ValueError, match="observation.*declared shape"):
        _flatten_observation(OversizedStub(), box)
    with pytest.raises(ValueError, match="action.*declared shape"):
        _flatten_action(OversizedStub(), box)


def test_factory_rejects_noncallable_policy_before_environment_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden_make(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("environment constructed before policy validation")

    monkeypatch.setattr(gymnasium, "make", forbidden_make)
    with pytest.raises(ValueError, match="policy"):
        make_gymnasium_stream("CartPole-v1", policy=object())  # type: ignore[arg-type]
    assert calls == 0


def test_factory_closes_environment_when_stream_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidEnv:
        observation_space = gymnasium.spaces.Tuple(
            (gymnasium.spaces.Discrete(2), gymnasium.spaces.Discrete(2))
        )
        action_space = gymnasium.spaces.Discrete(2)
        closed = False

        def close(self) -> None:
            self.closed = True

    env = InvalidEnv()
    monkeypatch.setattr(gymnasium, "make", lambda *args, **kwargs: env)
    with pytest.raises(ValueError, match="Unsupported space type"):
        make_gymnasium_stream("invalid-v0")
    assert env.closed


class TestGymnasiumStreamRewardMode:
    """Tests for GymnasiumStream with REWARD prediction mode."""

    def test_feature_dim_with_action(self):
        """Feature dim should include observation + action when enabled."""
        env = gymnasium.make("CartPole-v1")
        stream = GymnasiumStream(env, mode=PredictionMode.REWARD, include_action_in_features=True)
        # CartPole: obs=4, action=1 (Discrete)
        assert stream.feature_dim == 5

    def test_feature_dim_without_action(self):
        """Feature dim should be observation only when action excluded."""
        env = gymnasium.make("CartPole-v1")
        stream = GymnasiumStream(env, mode=PredictionMode.REWARD, include_action_in_features=False)
        # CartPole: obs=4
        assert stream.feature_dim == 4

    def test_target_dim_is_one(self):
        """Target dim should be 1 for REWARD mode."""
        env = gymnasium.make("CartPole-v1")
        stream = GymnasiumStream(env, mode=PredictionMode.REWARD)
        assert stream.target_dim == 1

    def test_generates_timesteps(self):
        """Stream should generate valid TimeStep instances."""
        env = gymnasium.make("CartPole-v1")
        stream = GymnasiumStream(env, mode=PredictionMode.REWARD)

        timestep = next(stream)

        assert isinstance(timestep, TimeStep)
        assert timestep.observation.shape == (stream.feature_dim,)
        assert timestep.target.shape == (1,)

    def test_observations_are_finite(self):
        """Generated observations should be finite."""
        env = gymnasium.make("CartPole-v1")
        stream = GymnasiumStream(env, mode=PredictionMode.REWARD, seed=42)

        for i, timestep in enumerate(stream):
            if i >= 100:
                break
            assert jnp.all(jnp.isfinite(timestep.observation))
            assert jnp.all(jnp.isfinite(timestep.target))


class TestGymnasiumStreamNextStateMode:
    """Tests for GymnasiumStream with NEXT_STATE prediction mode."""

    def test_target_dim_equals_obs_dim(self):
        """Target dim should equal observation dim for NEXT_STATE mode."""
        env = gymnasium.make("CartPole-v1")
        stream = GymnasiumStream(env, mode=PredictionMode.NEXT_STATE)
        # CartPole: obs=4
        assert stream.target_dim == 4

    def test_generates_valid_targets(self):
        """Targets should be valid next observations."""
        env = gymnasium.make("CartPole-v1")
        stream = GymnasiumStream(env, mode=PredictionMode.NEXT_STATE, seed=42)

        for i, timestep in enumerate(stream):
            if i >= 50:
                break
            assert timestep.target.shape == (4,)
            assert jnp.all(jnp.isfinite(timestep.target))


class TestGymnasiumStreamValueMode:
    """Tests for GymnasiumStream with VALUE prediction mode."""

    def test_target_dim_is_one(self):
        """Target dim should be 1 for VALUE mode."""
        env = gymnasium.make("CartPole-v1")
        stream = GymnasiumStream(env, mode=PredictionMode.VALUE)
        assert stream.target_dim == 1

    def test_generates_valid_targets(self):
        """Targets should be valid scalar values."""
        env = gymnasium.make("CartPole-v1")
        stream = GymnasiumStream(env, mode=PredictionMode.VALUE, gamma=0.99, seed=42)

        for i, timestep in enumerate(stream):
            if i >= 50:
                break
            assert timestep.target.shape == (1,)
            assert jnp.all(jnp.isfinite(timestep.target))

    def test_value_estimator_is_used(self):
        """Value estimator should be used for bootstrapping."""
        env = gymnasium.make("CartPole-v1")
        stream = GymnasiumStream(env, mode=PredictionMode.VALUE, gamma=0.99, seed=42)

        # Set a constant value estimator
        stream.set_value_estimator(lambda x: 10.0)

        # Collect targets
        targets_with_estimator = []
        for i, timestep in enumerate(stream):
            if i >= 20:
                break
            targets_with_estimator.append(float(timestep.target[0]))

        # Reset and run without estimator
        env2 = gymnasium.make("CartPole-v1")
        stream2 = GymnasiumStream(env2, mode=PredictionMode.VALUE, gamma=0.99, seed=42)

        targets_without = []
        for i, timestep in enumerate(stream2):
            if i >= 20:
                break
            targets_without.append(float(timestep.target[0]))

        # With estimator V(s')=10, non-terminal targets should be r + 0.99*10 ≈ r + 9.9
        # Without estimator, targets are just r + 0.99*0 = r
        # So targets with estimator should generally be larger
        assert sum(targets_with_estimator) > sum(targets_without)

    def test_zero_gamma_skips_inf_bootstrap(self) -> None:
        """gamma=0 is the next reward; 0 * inf V(s') is NaN.

        Fail-closed: a zero discount does not multiply the bootstrap value.
        """
        env = gymnasium.make("CartPole-v1")
        stream = GymnasiumStream(env, mode=PredictionMode.VALUE, gamma=0.0, seed=0)
        stream.set_value_estimator(lambda _obs: float("inf"))
        target = stream._construct_target(1.25, jnp.ones(4, dtype=jnp.float32), terminated=False)
        assert bool(jnp.isfinite(target).all())
        assert float(target[0]) == pytest.approx(1.25)


class TestGymnasiumStreamAutoReset:
    """Tests for auto-reset behavior on episode boundaries."""

    def test_infinite_stream_with_auto_reset(self):
        """Stream should continue indefinitely with auto-reset."""
        env = gymnasium.make("CartPole-v1")
        stream = GymnasiumStream(env, mode=PredictionMode.REWARD, seed=42)

        # Run for many steps (more than one episode)
        count = 0
        for timestep in stream:
            count += 1
            if count >= 500:
                break

        # Should have completed at least one episode
        assert stream.episode_count >= 1
        assert count == 500

    def test_episode_count_increments(self):
        """Episode count should increment on termination."""
        env = gymnasium.make("CartPole-v1")
        stream = GymnasiumStream(env, mode=PredictionMode.REWARD, seed=0)

        initial_episodes = stream.episode_count
        assert initial_episodes == 0

        # Run until at least one episode completes
        for _ in range(1000):
            _ = next(stream)
            if stream.episode_count > initial_episodes:
                break

        assert stream.episode_count >= 1


class TestGymnasiumStreamCustomPolicy:
    """Tests for custom policy support."""

    def test_uses_custom_policy(self):
        """Stream should use provided custom policy."""
        env = gymnasium.make("CartPole-v1")

        # Policy that always returns action 0
        def always_zero_policy(obs):
            return 0

        stream = GymnasiumStream(
            env,
            mode=PredictionMode.REWARD,
            policy=always_zero_policy,
            include_action_in_features=True,
        )

        # All actions should be 0
        for i, timestep in enumerate(stream):
            if i >= 50:
                break
            # Last element of features is the action
            action = timestep.observation[-1]
            assert float(action) == 0.0


class TestGymnasiumStreamReproducibility:
    """Tests for reproducibility with seeds."""

    def test_reproducible_with_seed(self):
        """Same seed should produce same sequence."""
        env1 = gymnasium.make("CartPole-v1")
        env2 = gymnasium.make("CartPole-v1")

        stream1 = GymnasiumStream(env1, mode=PredictionMode.REWARD, seed=123)
        stream2 = GymnasiumStream(env2, mode=PredictionMode.REWARD, seed=123)

        for i in range(20):
            ts1 = next(stream1)
            ts2 = next(stream2)
            assert jnp.allclose(ts1.observation, ts2.observation)
            assert jnp.allclose(ts1.target, ts2.target)


class TestContinuousActionSpaces:
    """Tests for environments with continuous action spaces."""

    def test_pendulum_works(self):
        """Should work with continuous action spaces like Pendulum."""
        env = gymnasium.make("Pendulum-v1")
        stream = GymnasiumStream(
            env,
            mode=PredictionMode.REWARD,
            include_action_in_features=True,
            seed=42,
        )

        # Pendulum: obs=3, action=1 (Box)
        assert stream.feature_dim == 4
        assert stream.target_dim == 1

        for i, timestep in enumerate(stream):
            if i >= 50:
                break
            assert timestep.observation.shape == (4,)
            assert jnp.all(jnp.isfinite(timestep.observation))


class TestTDStream:
    """Tests for TDStream with value function bootstrap."""

    def test_feature_dim_without_action(self):
        """Default TDStream should use observation only."""
        env = gymnasium.make("CartPole-v1")
        stream = TDStream(env, include_action_in_features=False)
        # CartPole: obs=4
        assert stream.feature_dim == 4

    def test_feature_dim_with_action(self):
        """TDStream can include action for Q-learning."""
        env = gymnasium.make("CartPole-v1")
        stream = TDStream(env, include_action_in_features=True)
        # CartPole: obs=4, action=1
        assert stream.feature_dim == 5

    def test_generates_timesteps(self):
        """TDStream should generate valid TimeStep instances."""
        env = gymnasium.make("CartPole-v1")
        stream = TDStream(env, seed=42)

        timestep = next(stream)

        assert isinstance(timestep, TimeStep)
        assert timestep.observation.shape == (stream.feature_dim,)
        assert timestep.target.shape == (1,)

    def test_value_function_update(self):
        """TDStream should use updated value function for bootstrap."""
        env = gymnasium.make("CartPole-v1")
        stream = TDStream(env, gamma=0.99, seed=42)

        # Collect targets with default (zero) value function
        targets_zero = []
        for i, timestep in enumerate(stream):
            if i >= 10:
                break
            targets_zero.append(float(timestep.target[0]))

        # Update value function and collect more targets
        stream.update_value_function(lambda x: 5.0)

        # The next targets should use the new value function
        targets_with_value = []
        for i, timestep in enumerate(stream):
            if i >= 10:
                break
            targets_with_value.append(float(timestep.target[0]))

        # Non-terminal targets with V(s')=5 should be larger: r + 0.99*5 vs r + 0.99*0
        # At least some should be larger (terminal states will be the same)
        assert sum(targets_with_value) > sum(targets_zero)

    def test_action_value_bootstrap_uses_next_policy_action(self):
        """Action-value targets should bootstrap from Q(s', a'), not Q(s', a)."""
        env = gymnasium.make("CartPole-v1")
        actions = iter((0, 1))
        stream = TDStream(
            env,
            policy=lambda _observation: next(actions),
            gamma=0.5,
            include_action_in_features=True,
            seed=42,
        )
        stream.update_value_function(lambda features: float(features[-1]))

        timestep = next(stream)

        assert float(timestep.observation[-1]) == 0.0
        assert float(timestep.target[0]) == pytest.approx(1.5)

    def test_zero_gamma_skips_inf_value_function(self) -> None:
        """gamma=0 times inf V(s') is NaN in TDStream targets.

        Fail-closed: a zero discount does not multiply the value function.
        """
        env = gymnasium.make("CartPole-v1")
        stream = TDStream(env, gamma=0.0, seed=0)
        stream.update_value_function(lambda _obs: float("inf"))
        target = next(stream).target
        assert bool(jnp.isfinite(target).all())

    def test_episode_tracking(self):
        """TDStream should track episode count."""
        env = gymnasium.make("CartPole-v1")
        stream = TDStream(env, seed=0)

        assert stream.episode_count == 0
        assert stream.step_count == 0

        # Run until episode completes
        for _ in range(1000):
            _ = next(stream)
            if stream.episode_count > 0:
                break

        assert stream.episode_count >= 1
        assert stream.step_count > 0


class TestMakeRandomPolicy:
    """Tests for make_random_policy factory."""

    def test_discrete_action_space(self):
        """Should work with discrete action spaces."""
        env = gymnasium.make("CartPole-v1")
        policy = make_random_policy(env, seed=42)

        obs = jnp.zeros(4)
        action = policy(obs)

        assert isinstance(action, int)
        assert 0 <= action < 2  # CartPole has 2 actions

    def test_continuous_action_space(self):
        """Should work with continuous action spaces."""
        env = gymnasium.make("Pendulum-v1")
        policy = make_random_policy(env, seed=42)

        obs = jnp.zeros(3)
        action = policy(obs)

        assert hasattr(action, "shape")
        assert action.shape == (1,)
        # Pendulum action bounds are [-2, 2]
        assert -2.0 <= float(action[0]) <= 2.0


class TestMakeEpsilonGreedyPolicy:
    """Tests for make_epsilon_greedy_policy factory."""

    def test_epsilon_zero_uses_base_policy(self):
        """With epsilon=0, should always use base policy."""
        env = gymnasium.make("CartPole-v1")

        # Base policy always returns 1
        def base_policy(obs):
            return 1

        policy = make_epsilon_greedy_policy(base_policy, env, epsilon=0.0, seed=42)

        obs = jnp.zeros(4)
        for _ in range(20):
            action = policy(obs)
            assert action == 1

    def test_epsilon_one_uses_random(self):
        """With epsilon=1, should always use random policy."""
        env = gymnasium.make("CartPole-v1")

        # Base policy always returns 1
        def base_policy(obs):
            return 1

        policy = make_epsilon_greedy_policy(base_policy, env, epsilon=1.0, seed=42)

        obs = jnp.zeros(4)
        actions = [policy(obs) for _ in range(100)]

        # Should have some 0s (random exploration)
        assert 0 in actions


class TestMakeGymnasiumStream:
    """Tests for make_gymnasium_stream factory."""

    def test_creates_stream_from_env_id(self):
        """Factory should create stream from environment ID."""
        stream = make_gymnasium_stream("CartPole-v1", mode=PredictionMode.REWARD)

        assert isinstance(stream, GymnasiumStream)
        assert stream.feature_dim == 5  # obs(4) + action(1)
        assert stream.mode == PredictionMode.REWARD

    def test_passes_env_kwargs(self):
        """Factory should pass kwargs to gymnasium.make()."""
        stream = make_gymnasium_stream(
            "CartPole-v1",
            mode=PredictionMode.REWARD,
            max_episode_steps=50,
        )

        # The max_episode_steps should limit episode length
        count = 0
        episodes = 0
        for _ in stream:
            count += 1
            if stream.episode_count > episodes:
                episodes = stream.episode_count
                if count <= 50:
                    break
            if count > 200:
                break

        # Should have had a truncated episode by 50 steps
        assert stream.episode_count >= 1


class TestStreamIterator:
    """Tests for stream iterator behavior."""

    def test_can_use_in_for_loop(self):
        """Streams should work with Python for loops."""
        env = gymnasium.make("CartPole-v1")
        stream = GymnasiumStream(env, mode=PredictionMode.REWARD)

        count = 0
        for timestep in stream:
            count += 1
            if count >= 10:
                break

        assert count == 10

    def test_iter_returns_self(self):
        """__iter__ should return self."""
        env = gymnasium.make("CartPole-v1")
        stream = GymnasiumStream(env, mode=PredictionMode.REWARD)
        assert iter(stream) is stream


class TestCollectTrajectoryValueMode:
    """collect_trajectory VALUE targets with and without a value estimator."""

    def test_without_estimator_matches_reward_targets(self):
        """Documented zero bootstrap: VALUE targets equal immediate reward."""
        env_a = gymnasium.make("CartPole-v1")
        env_b = gymnasium.make("CartPole-v1")

        _, reward_targets = collect_trajectory(
            env_a, None, num_steps=40, mode=PredictionMode.REWARD, seed=3
        )
        _, value_targets = collect_trajectory(
            env_b, None, num_steps=40, mode=PredictionMode.VALUE, seed=3
        )

        assert jnp.allclose(value_targets, reward_targets)

    def test_with_estimator_bootstraps_td_target(self):
        """With V(s)=1 the non-terminal targets shift by exactly gamma."""
        gamma = 0.5
        env_a = gymnasium.make("CartPole-v1")
        env_b = gymnasium.make("CartPole-v1")

        _, reward_targets = collect_trajectory(
            env_a, None, num_steps=60, mode=PredictionMode.REWARD, seed=7
        )
        _, value_targets = collect_trajectory(
            env_b,
            None,
            num_steps=60,
            mode=PredictionMode.VALUE,
            seed=7,
            value_estimator=lambda _obs: 1.0,
            gamma=gamma,
        )

        diffs = value_targets - reward_targets
        # Every step is either bootstrapped (+gamma * 1.0) or terminal (+0).
        bootstrapped = jnp.isclose(diffs, gamma)
        terminal = jnp.isclose(diffs, 0.0)
        assert bool(jnp.all(bootstrapped | terminal))
        # Random CartPole rollouts of 60 steps contain non-terminal steps.
        assert bool(jnp.any(bootstrapped))

    def test_zero_gamma_skips_inf_bootstrap(self) -> None:
        """gamma=0 times inf V(s') is NaN in collected VALUE targets.

        Fail-closed: a zero discount does not multiply the estimator.
        """
        env = gymnasium.make("CartPole-v1")
        _, targets = collect_trajectory(
            env,
            None,
            num_steps=20,
            mode=PredictionMode.VALUE,
            seed=11,
            value_estimator=lambda _obs: float("inf"),
            gamma=0.0,
        )
        assert bool(jnp.all(jnp.isfinite(targets)))

    def test_estimator_receives_next_observation(self):
        """The bootstrap value is computed from the next observation."""
        env = gymnasium.make("CartPole-v1")
        seen: list[tuple[float, ...]] = []

        def estimator(obs) -> float:
            seen.append(tuple(float(v) for v in obs))
            return 0.0

        observations, _ = collect_trajectory(
            env,
            None,
            num_steps=10,
            mode=PredictionMode.VALUE,
            include_action_in_features=False,
            seed=11,
            value_estimator=estimator,
        )

        assert seen
        # Each recorded estimator input is a full 4-dim CartPole observation.
        assert all(len(obs) == 4 for obs in seen)
        # Non-terminal steps: estimator input equals the following feature row.
        first_input = jnp.asarray(seen[0], dtype=jnp.float32)
        assert jnp.allclose(first_input, observations[1], atol=1e-6)

    def test_estimator_ignored_by_reward_mode(self):
        """value_estimator only affects VALUE mode."""
        env_a = gymnasium.make("CartPole-v1")
        env_b = gymnasium.make("CartPole-v1")

        _, plain = collect_trajectory(
            env_a, None, num_steps=20, mode=PredictionMode.REWARD, seed=5
        )
        _, with_estimator = collect_trajectory(
            env_b,
            None,
            num_steps=20,
            mode=PredictionMode.REWARD,
            seed=5,
            value_estimator=lambda _obs: 100.0,
        )

        assert jnp.allclose(plain, with_estimator)
