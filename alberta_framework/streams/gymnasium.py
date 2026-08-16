"""Gymnasium environment wrappers as experience streams.

This module wraps Gymnasium environments to provide temporally-uniform experience
streams compatible with the Alberta Framework's learners.

Gymnasium environments cannot be JIT-compiled, so this module provides:
1. Trajectory collection: Collect data using Python loop, then learn with scan
2. Online learning: Python loop for cases requiring real-time env interaction

Supports multiple prediction modes:
- REWARD: Predict immediate reward from (state, action)
- NEXT_STATE: Predict next state from (state, action)
- VALUE: TD-style value target ``reward + gamma * V(next)``.  How much
  genuine bootstrapping happens depends on the collector:
  :func:`collect_trajectory` bootstraps only when a ``value_estimator`` is
  passed (zero bootstrap otherwise, so targets reduce to immediate reward),
  :class:`GymnasiumStream` bootstraps only after ``set_value_estimator`` is
  called, and :class:`TDStream` bootstraps from a caller-updated value
  function.
"""

from __future__ import annotations

import math
import operator
import time
from collections.abc import Callable, Iterator
from enum import Enum
from typing import TYPE_CHECKING, Any, SupportsIndex, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework._seed_validation import JAX_KEY_SEED_MAX, require_jax_seed
from alberta_framework.core._float32_scalars import validated_float32_scalar
from alberta_framework.core.learners import LinearLearner
from alberta_framework.core.types import LearnerState, TimeStep

if TYPE_CHECKING:
    import gymnasium

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_ACTUAL_INT_TYPES = frozenset(
    {
        int,
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
    }
)


def _require_positive_int32(name: str, value: object) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [1, {_INT32_MAX}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not 1 <= canonical <= _INT32_MAX:
        raise ValueError(f"{name} must be an integer in [1, {_INT32_MAX}]")
    return canonical


def _require_allocation(name: str, scalars: int, *, itemsize: int) -> None:
    if scalars > _INT32_MAX or itemsize * scalars > _INT32_MAX:
        raise ValueError(f"derived {name} scalar and byte counts must fit signed int32")


def _require_float32_allocation(name: str, scalars: int) -> None:
    _require_allocation(name, scalars, itemsize=4)


def _require_exact_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be an exact bool")
    return value


def _require_mode(mode: object) -> PredictionMode:
    if type(mode) is not PredictionMode:
        raise ValueError("mode must be an exact PredictionMode")
    return mode


class PredictionMode(Enum):
    """Mode for what the stream predicts.

    REWARD: Predict immediate reward from (state, action)
    NEXT_STATE: Predict next state from (state, action)
    VALUE: TD value target ``reward + gamma * V(next)``; collectors without
        a value estimator fall back to a zero bootstrap, making the target
        identical to REWARD
    """

    REWARD = "reward"
    NEXT_STATE = "next_state"
    VALUE = "value"


def _discounted_bootstrap(gamma: float, next_value: float) -> float:
    """Return ``gamma * V(s')``, or 0 when gamma is 0 so 0*inf stays 0."""
    if float(gamma) == 0.0:
        return 0.0
    return float(gamma) * float(next_value)


def _flatten_space(space: gymnasium.spaces.Space[Any]) -> int:
    """Get the flattened dimension of a Gymnasium space.

    Args:
        space: A Gymnasium space (Box, Discrete, MultiDiscrete)

    Returns:
        Integer dimension of the flattened space

    Raises:
        ValueError: If space type is not supported
    """
    import gymnasium

    if isinstance(space, gymnasium.spaces.Box):
        dimension = math.prod(space.shape)
        return _require_positive_int32("flattened Box dimension", dimension)
    elif isinstance(space, gymnasium.spaces.Discrete):
        return 1
    elif isinstance(space, gymnasium.spaces.MultiDiscrete):
        return _require_positive_int32("flattened MultiDiscrete dimension", int(space.nvec.size))
    else:
        raise ValueError(
            f"Unsupported space type: {type(space).__name__}. "
            "Supported types: Box, Discrete, MultiDiscrete"
        )


def _flatten_observation(obs: Any, space: gymnasium.spaces.Space[Any]) -> Array:
    """Flatten an observation to a 1D JAX array.

    Args:
        obs: Observation from the environment
        space: The observation space

    Returns:
        Flattened observation as a 1D JAX array
    """
    import gymnasium

    if isinstance(space, (gymnasium.spaces.Box, gymnasium.spaces.MultiDiscrete)):
        _require_runtime_shape("observation", obs, tuple(space.shape))
    elif isinstance(space, gymnasium.spaces.Discrete):
        _require_runtime_shape("observation", obs, ())
    try:
        if isinstance(space, gymnasium.spaces.Box):
            flattened = jnp.asarray(obs, dtype=jnp.float32).reshape((-1,))
        elif isinstance(space, gymnasium.spaces.Discrete):
            flattened = jnp.asarray([obs], dtype=jnp.float32)
        elif isinstance(space, gymnasium.spaces.MultiDiscrete):
            flattened = jnp.asarray(obs, dtype=jnp.float32).reshape((-1,))
        else:
            raise ValueError(f"Unsupported space type: {type(space).__name__}")
    except Exception as error:
        raise ValueError("observation could not be converted to float32") from error
    expected = _flatten_space(space)
    if flattened.shape != (expected,) or not bool(jnp.all(jnp.isfinite(flattened))):
        raise ValueError("observation must match its declared flattened shape and be finite")
    return flattened


def _flatten_action(action: Any, space: gymnasium.spaces.Space[Any]) -> Array:
    """Flatten an action to a 1D JAX array.

    Args:
        action: Action for the environment
        space: The action space

    Returns:
        Flattened action as a 1D JAX array
    """
    import gymnasium

    if isinstance(space, (gymnasium.spaces.Box, gymnasium.spaces.MultiDiscrete)):
        _require_runtime_shape("action", action, tuple(space.shape))
    elif isinstance(space, gymnasium.spaces.Discrete):
        _require_runtime_shape("action", action, ())
    try:
        if isinstance(space, gymnasium.spaces.Box):
            flattened = jnp.asarray(action, dtype=jnp.float32).reshape((-1,))
        elif isinstance(space, gymnasium.spaces.Discrete):
            flattened = jnp.asarray([action], dtype=jnp.float32)
        elif isinstance(space, gymnasium.spaces.MultiDiscrete):
            flattened = jnp.asarray(action, dtype=jnp.float32).reshape((-1,))
        else:
            raise ValueError(f"Unsupported space type: {type(space).__name__}")
    except Exception as error:
        raise ValueError("action could not be converted to float32") from error
    expected = _flatten_space(space)
    if flattened.shape != (expected,) or not bool(jnp.all(jnp.isfinite(flattened))):
        raise ValueError("action must match its declared flattened shape and be finite")
    return flattened


def _require_runtime_shape(name: str, value: object, expected: tuple[int, ...]) -> None:
    """Check host shape metadata before conversion can allocate a device array."""
    try:
        metadata = getattr(value, "shape", None)
        actual = tuple(metadata) if metadata is not None else tuple(np.shape(cast(Any, value)))
    except Exception as error:
        raise ValueError(f"{name} shape metadata could not be read") from error
    if actual != expected:
        raise ValueError(f"{name} must have declared shape {expected}; got {actual}")


def _sample_finite_box(key: Array, low: Array, high: Array) -> Array:
    """Sample a finite float32 Box without forming the possibly-overflowing full span."""
    unit = jr.uniform(key, low.shape, dtype=jnp.float32)
    midpoint = low / 2.0 + high / 2.0
    lower = low + (midpoint - low) * (2.0 * unit)
    upper = midpoint + (high - midpoint) * (2.0 * unit - 1.0)
    sample = jnp.where(unit < 0.5, lower, upper)
    upper_open = jnp.nextafter(high, low)
    return jnp.where(low < high, jnp.minimum(sample, upper_open), low)


def _make_random_policy(
    env: gymnasium.Env[Any, Any], rng: Array
) -> Callable[[Array], Any]:
    """Build a bounded random policy from one already validated JAX key."""
    import gymnasium

    action_space = env.action_space
    action_dim = _flatten_space(action_space)

    if isinstance(action_space, gymnasium.spaces.Discrete):
        discrete_low = int(action_space.start)
        discrete_count = int(action_space.n)
        discrete_high = discrete_low + discrete_count
        if not (
            1 <= discrete_count <= _INT32_MAX
            and _INT32_MIN <= discrete_low < discrete_high <= _INT32_MAX
        ):
            raise ValueError("Discrete action bounds must fit JAX signed int32")
    elif isinstance(action_space, gymnasium.spaces.Box):
        _require_float32_allocation("random Box bounds and action", 3 * action_dim)
        low_array = np.asarray(action_space.low, dtype=np.float32)
        high_array = np.asarray(action_space.high, dtype=np.float32)
        if not bool(
            np.all(
                np.isfinite(low_array)
                & np.isfinite(high_array)
                & (low_array <= high_array)
            )
        ):
            raise ValueError("Box random actions require finite ordered float32 bounds")
        low = jnp.asarray(low_array)
        high = jnp.asarray(high_array)
    elif isinstance(action_space, gymnasium.spaces.MultiDiscrete):
        starts = np.asarray(action_space.start).reshape((-1,))
        counts = np.asarray(action_space.nvec).reshape((-1,))
        action_dtype = np.dtype(action_space.dtype)
        _require_allocation(
            "random MultiDiscrete action",
            action_dim,
            itemsize=action_dtype.itemsize,
        )
        for index in range(action_dim):
            start = int(starts[index])
            count = int(counts[index])
            if not (
                1 <= count <= _INT32_MAX
                and _INT32_MIN <= start < start + count <= _INT32_MAX
            ):
                raise ValueError("MultiDiscrete action bounds must fit JAX signed int32")
    else:
        raise ValueError(f"Unsupported action space: {type(action_space).__name__}")

    def policy(_obs: Array) -> Any:
        nonlocal rng
        rng, key = jr.split(rng)

        if isinstance(action_space, gymnasium.spaces.Discrete):
            return int(jr.randint(key, (), discrete_low, discrete_high))
        if isinstance(action_space, gymnasium.spaces.Box):
            return _sample_finite_box(key, low, high)
        if isinstance(action_space, gymnasium.spaces.MultiDiscrete):
            samples = np.fromiter(
                (
                    int(
                        jr.randint(
                            jr.fold_in(key, index),
                            (),
                            int(starts[index]),
                            int(starts[index]) + int(counts[index]),
                        )
                    )
                    for index in range(action_dim)
                ),
                dtype=action_dtype,
                count=action_dim,
            )
            return samples.reshape(action_space.shape)
        raise AssertionError("action-space validation and dispatch diverged")

    return policy


def make_random_policy(env: gymnasium.Env[Any, Any], seed: int = 0) -> Callable[[Array], Any]:
    """Create a random action policy for an environment.

    Args:
        env: Gymnasium environment
        seed: Random seed

    Returns:
        A callable that takes an observation and returns a random action
    """
    seed = require_jax_seed(seed)
    return _make_random_policy(env, jr.key(seed))


def make_epsilon_greedy_policy(
    base_policy: Callable[[Array], Any],
    env: gymnasium.Env[Any, Any],
    epsilon: float = 0.1,
    seed: int = 0,
) -> Callable[[Array], Any]:
    """Wrap a policy with epsilon-greedy exploration.

    Args:
        base_policy: The greedy policy to wrap
        env: Gymnasium environment (for random action sampling)
        epsilon: Probability of taking a random action
        seed: Random seed

    Returns:
        Epsilon-greedy policy
    """
    if not callable(base_policy):
        raise ValueError("base_policy must be callable")
    epsilon = validated_float32_scalar("epsilon", epsilon, lower=0.0, upper=1.0)
    seed = require_jax_seed(seed)
    rng = jr.key(seed)
    random_rng = jr.key(seed + 1) if seed < JAX_KEY_SEED_MAX else jr.fold_in(rng, 1)
    random_policy = _make_random_policy(env, random_rng)

    def policy(obs: Array) -> Any:
        nonlocal rng
        rng, key = jr.split(rng)

        if jr.uniform(key) < epsilon:
            return random_policy(obs)
        return base_policy(obs)

    return policy


def collect_trajectory(
    env: gymnasium.Env[Any, Any],
    policy: Callable[[Array], Any] | None,
    num_steps: int,
    mode: PredictionMode = PredictionMode.REWARD,
    include_action_in_features: bool = True,
    seed: int = 0,
    value_estimator: Callable[[Array], float] | None = None,
    gamma: float = 0.99,
) -> tuple[Array, Array]:
    """Collect a trajectory from a Gymnasium environment.

    This uses a Python loop to interact with the environment and collects
    observations and targets into JAX arrays that can be used with scan-based
    learning.

    Args:
        env: Gymnasium environment instance
        policy: Action selection function. If None, uses random policy
        num_steps: Number of steps to collect
        mode: What to predict (REWARD, NEXT_STATE, VALUE). VALUE targets
            are ``reward + gamma * value_estimator(next_obs)`` when a
            ``value_estimator`` is supplied (``reward`` alone on
            termination); without one the documented zero bootstrap makes
            targets equal immediate reward. Use :class:`TDStream` for
            targets that bootstrap from a learner updated during collection.
        include_action_in_features: If True, features = concat(obs, action)
        seed: Random seed for environment resets and random policy
        value_estimator: Optional fixed state-value function ``V(next_obs)``
            used only by VALUE mode for the TD bootstrap
        gamma: Discount factor used only by VALUE mode

    Returns:
        Tuple of (observations, targets) as JAX arrays with shape
        (num_steps, feature_dim) and (num_steps, target_dim)
    """
    num_steps = _require_positive_int32("num_steps", num_steps)
    mode = _require_mode(mode)
    include_action_in_features = _require_exact_bool(
        "include_action_in_features", include_action_in_features
    )
    seed = require_jax_seed(seed)
    gamma = validated_float32_scalar("gamma", gamma, lower=0.0, upper=1.0)
    if policy is not None and not callable(policy):
        raise ValueError("policy must be callable or None")
    if value_estimator is not None and not callable(value_estimator):
        raise ValueError("value_estimator must be callable or None")
    observation_dim = _flatten_space(env.observation_space)
    action_dim = _flatten_space(env.action_space)
    feature_dim = observation_dim + action_dim if include_action_in_features else observation_dim
    target_dim = observation_dim if mode is PredictionMode.NEXT_STATE else 1
    _require_float32_allocation(
        "trajectory output", num_steps * (feature_dim + target_dim)
    )
    if policy is None:
        policy = make_random_policy(env, seed)

    observations = []
    targets = []

    reset_count = 0
    raw_obs, _ = env.reset(seed=seed + reset_count)
    reset_count += 1
    current_obs = _flatten_observation(raw_obs, env.observation_space)

    for _ in range(num_steps):
        action = policy(current_obs)
        flat_action = _flatten_action(action, env.action_space)

        raw_next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = _flatten_observation(raw_next_obs, env.observation_space)

        # Construct features
        if include_action_in_features:
            features = jnp.concatenate([current_obs, flat_action])
        else:
            features = current_obs

        # Construct target based on mode
        if mode == PredictionMode.REWARD:
            target = jnp.atleast_1d(jnp.array(reward, dtype=jnp.float32))
        elif mode == PredictionMode.NEXT_STATE:
            target = next_obs
        else:  # VALUE mode
            # TD target reward + gamma * V(next); zero bootstrap without an
            # estimator (documented on PredictionMode.VALUE) and on
            # termination, where the post-terminal value is defined as zero.
            if value_estimator is not None and not terminated:
                bootstrap = _discounted_bootstrap(gamma, float(value_estimator(next_obs)))
            else:
                bootstrap = 0.0
            target = jnp.atleast_1d(
                jnp.array(float(reward) + bootstrap, dtype=jnp.float32)
            )

        observations.append(features)
        targets.append(target)

        if terminated or truncated:
            raw_obs, _ = env.reset(seed=seed + reset_count)
            reset_count += 1
            current_obs = _flatten_observation(raw_obs, env.observation_space)
        else:
            current_obs = next_obs

    return jnp.stack(observations), jnp.stack(targets)


def learn_from_trajectory(
    learner: LinearLearner,
    observations: Array,
    targets: Array,
    learner_state: LearnerState | None = None,
) -> tuple[LearnerState, Array]:
    """Learn from a pre-collected trajectory using jax.lax.scan.

    This is a JIT-compiled learning function that processes a trajectory
    collected from a Gymnasium environment.

    Args:
        learner: The learner to train
        observations: Array of observations with shape (num_steps, feature_dim)
        targets: Array of targets with shape (num_steps, target_dim)
        learner_state: Initial state (if None, will be initialized)

    Returns:
        Tuple of (final_state, metrics_array). For a plain learner the
        metrics have shape (num_steps, 3) with columns
        [squared_error, error, mean_step_size]; a learner constructed with
        a normalizer emits a fourth column, normalizer_mean_var.
    """
    if learner_state is None:
        learner_state = learner.init(observations.shape[1])

    def step_fn(state: LearnerState, inputs: tuple[Array, Array]) -> tuple[LearnerState, Array]:
        obs, target = inputs
        result = learner.update(state, obs, target)
        return result.state, result.metrics

    t0 = time.time()
    final_state, metrics = jax.lax.scan(step_fn, learner_state, (observations, targets))
    elapsed = time.time() - t0
    final_state = final_state.replace(uptime_s=final_state.uptime_s + elapsed)  # type: ignore[attr-defined]

    return final_state, metrics


def learn_from_trajectory_normalized(
    learner: LinearLearner,
    observations: Array,
    targets: Array,
    learner_state: LearnerState | None = None,
) -> tuple[LearnerState, Array]:
    """Learn from a pre-collected trajectory with normalization using jax.lax.scan.

    Backward-compatible alias of :func:`learn_from_trajectory`; normalization
    lives in the learner itself (e.g.
    ``LinearLearner(optimizer=..., normalizer=EMANormalizer())``), so the two
    functions are identical.

    Args:
        learner: The learner to train (should have a normalizer configured)
        observations: Array of observations with shape (num_steps, feature_dim)
        targets: Array of targets with shape (num_steps, target_dim)
        learner_state: Initial state (if None, will be initialized)

    Returns:
        Tuple of (final_state, metrics_array). The column count follows the
        learner: with a normalizer configured, shape (num_steps, 4) with
        columns [squared_error, error, mean_step_size, normalizer_mean_var];
        without one, only the first 3 columns.
    """
    return learn_from_trajectory(learner, observations, targets, learner_state)


class GymnasiumStream:
    """Experience stream from a Gymnasium environment using Python loop.

    This class maintains iterator-based access for online learning scenarios
    where you need to interact with the environment in real-time.

    For batch learning, use collect_trajectory() followed by learn_from_trajectory().

    Attributes:
        mode: Prediction mode (REWARD, NEXT_STATE, VALUE)
        gamma: Discount factor for VALUE mode
        include_action_in_features: Whether to include action in features
        episode_count: Number of completed episodes
    """

    def __init__(
        self,
        env: gymnasium.Env[Any, Any],
        mode: PredictionMode = PredictionMode.REWARD,
        policy: Callable[[Array], Any] | None = None,
        gamma: float = 0.99,
        include_action_in_features: bool = True,
        seed: int = 0,
    ):
        """Initialize the Gymnasium stream.

        Args:
            env: Gymnasium environment instance
            mode: What to predict (REWARD, NEXT_STATE, VALUE)
            policy: Action selection function. If None, uses random policy
            gamma: Discount factor for VALUE mode
            include_action_in_features: If True, features = concat(obs, action).
                If False, features = obs only
            seed: Random seed for environment resets and random policy
        """
        mode = _require_mode(mode)
        include_action_in_features = _require_exact_bool(
            "include_action_in_features", include_action_in_features
        )
        seed = require_jax_seed(seed)
        if policy is not None and not callable(policy):
            raise ValueError("policy must be callable or None")
        gamma = validated_float32_scalar("gamma", gamma, lower=0.0, upper=1.0)
        observation_dim = _flatten_space(env.observation_space)
        action_dim = _flatten_space(env.action_space)
        feature_dim = (
            observation_dim + action_dim
            if include_action_in_features
            else observation_dim
        )
        target_dim = observation_dim if mode is PredictionMode.NEXT_STATE else 1
        _require_float32_allocation("stream feature vector", feature_dim)
        _require_float32_allocation("stream target vector", target_dim)

        selected_policy = make_random_policy(env, seed) if policy is None else policy
        self._env = env
        self._mode = mode
        self._gamma = gamma
        self._include_action_in_features = include_action_in_features
        self._seed = seed
        self._reset_count = 0
        self._policy = selected_policy
        self._obs_dim = observation_dim
        self._action_dim = action_dim
        self._feature_dim = feature_dim
        self._target_dim = target_dim

        self._current_obs: Array | None = None
        self._episode_count = 0
        self._step_count = 0
        self._value_estimator: Callable[[Array], float] | None = None

    @property
    def feature_dim(self) -> int:
        """Return the dimension of feature vectors."""
        return self._feature_dim

    @property
    def target_dim(self) -> int:
        """Return the dimension of target vectors."""
        return self._target_dim

    @property
    def episode_count(self) -> int:
        """Return the number of completed episodes."""
        return self._episode_count

    @property
    def step_count(self) -> int:
        """Return the total number of steps taken."""
        return self._step_count

    @property
    def mode(self) -> PredictionMode:
        """Return the prediction mode."""
        return self._mode

    def set_value_estimator(self, estimator: Callable[[Array], float]) -> None:
        """Set the value estimator for proper TD learning in VALUE mode."""
        if not callable(estimator):
            raise ValueError("estimator must be callable")
        self._value_estimator = estimator

    def _get_reset_seed(self) -> int:
        """Get the seed for the next environment reset."""
        seed = self._seed + self._reset_count
        self._reset_count += 1
        return seed

    def _construct_features(self, obs: Array, action: Array) -> Array:
        """Construct feature vector from observation and action."""
        if self._include_action_in_features:
            return jnp.concatenate([obs, action])
        return obs

    def _construct_target(
        self,
        reward: float,
        next_obs: Array,
        terminated: bool,
    ) -> Array:
        """Construct target based on prediction mode."""
        if self._mode == PredictionMode.REWARD:
            return jnp.atleast_1d(jnp.array(reward, dtype=jnp.float32))

        elif self._mode == PredictionMode.NEXT_STATE:
            return next_obs

        elif self._mode == PredictionMode.VALUE:
            if terminated:
                return jnp.atleast_1d(jnp.array(reward, dtype=jnp.float32))

            if self._value_estimator is not None:
                next_value = self._value_estimator(next_obs)
            else:
                next_value = 0.0

            target = reward + _discounted_bootstrap(self._gamma, next_value)
            return jnp.atleast_1d(jnp.array(target, dtype=jnp.float32))

        else:
            raise ValueError(f"Unknown mode: {self._mode}")

    def __iter__(self) -> Iterator[TimeStep]:
        """Return self as iterator."""
        return self

    def __next__(self) -> TimeStep:
        """Generate the next time step."""
        if self._current_obs is None:
            raw_obs, _ = self._env.reset(seed=self._get_reset_seed())
            self._current_obs = _flatten_observation(raw_obs, self._env.observation_space)

        action = self._policy(self._current_obs)
        flat_action = _flatten_action(action, self._env.action_space)

        raw_next_obs, reward, terminated, truncated, _ = self._env.step(action)
        next_obs = _flatten_observation(raw_next_obs, self._env.observation_space)

        features = self._construct_features(self._current_obs, flat_action)
        target = self._construct_target(float(reward), next_obs, terminated)

        self._step_count += 1

        if terminated or truncated:
            self._episode_count += 1
            self._current_obs = None
        else:
            self._current_obs = next_obs

        return TimeStep(observation=features, target=target)


class TDStream:
    """Experience stream for proper TD learning with value function bootstrap.

    This stream integrates with a learner to use its predictions for
    bootstrapping in TD targets.

    Usage:
        stream = TDStream(env)
        learner = LinearLearner(optimizer=IDBD())
        state = learner.init(stream.feature_dim)

        for step, timestep in enumerate(stream):
            result = learner.update(state, timestep.observation, timestep.target)
            state = result.state
            stream.update_value_function(lambda x: learner.predict(state, x))
    """

    def __init__(
        self,
        env: gymnasium.Env[Any, Any],
        policy: Callable[[Array], Any] | None = None,
        gamma: float = 0.99,
        include_action_in_features: bool = False,
        seed: int = 0,
    ):
        """Initialize the TD stream.

        Args:
            env: Gymnasium environment instance
            policy: Action selection function. If None, uses random policy
            gamma: Discount factor
            include_action_in_features: If True, learn Q(s,a). If False, learn V(s)
            seed: Random seed
        """
        include_action_in_features = _require_exact_bool(
            "include_action_in_features", include_action_in_features
        )
        seed = require_jax_seed(seed)
        if policy is not None and not callable(policy):
            raise ValueError("policy must be callable or None")
        gamma = validated_float32_scalar("gamma", gamma, lower=0.0, upper=1.0)
        observation_dim = _flatten_space(env.observation_space)
        action_dim = _flatten_space(env.action_space)
        feature_dim = (
            observation_dim + action_dim
            if include_action_in_features
            else observation_dim
        )
        _require_float32_allocation("TD stream feature vector", feature_dim)

        selected_policy = make_random_policy(env, seed) if policy is None else policy
        self._env = env
        self._gamma = gamma
        self._include_action_in_features = include_action_in_features
        self._seed = seed
        self._reset_count = 0
        self._policy = selected_policy
        self._obs_dim = observation_dim
        self._action_dim = action_dim
        self._feature_dim = feature_dim

        self._current_obs: Array | None = None
        self._episode_count = 0
        self._step_count = 0
        self._value_fn: Callable[[Array], float] = lambda x: 0.0

    @property
    def feature_dim(self) -> int:
        """Return the dimension of feature vectors."""
        return self._feature_dim

    @property
    def episode_count(self) -> int:
        """Return the number of completed episodes."""
        return self._episode_count

    @property
    def step_count(self) -> int:
        """Return the total number of steps taken."""
        return self._step_count

    def update_value_function(self, value_fn: Callable[[Array], float]) -> None:
        """Update the value function used for TD bootstrapping."""
        if not callable(value_fn):
            raise ValueError("value_fn must be callable")
        self._value_fn = value_fn

    def _get_reset_seed(self) -> int:
        """Get the seed for the next environment reset."""
        seed = self._seed + self._reset_count
        self._reset_count += 1
        return seed

    def _construct_features(self, obs: Array, action: Array) -> Array:
        """Construct feature vector from observation and action."""
        if self._include_action_in_features:
            return jnp.concatenate([obs, action])
        return obs

    def __iter__(self) -> Iterator[TimeStep]:
        """Return self as iterator."""
        return self

    def __next__(self) -> TimeStep:
        """Generate the next time step with TD target.

        Caveat: with ``include_action_in_features=True`` the bootstrap
        features pair the *next* observation with the *current* action —
        Q(s', a) rather than Q(s', a') — because the next action has not
        been sampled yet.  The target is therefore biased relative to
        SARSA's ``reward + gamma * Q(s', a')``.
        """
        if self._current_obs is None:
            raw_obs, _ = self._env.reset(seed=self._get_reset_seed())
            self._current_obs = _flatten_observation(raw_obs, self._env.observation_space)

        action = self._policy(self._current_obs)
        flat_action = _flatten_action(action, self._env.action_space)

        raw_next_obs, reward, terminated, truncated, _ = self._env.step(action)
        next_obs = _flatten_observation(raw_next_obs, self._env.observation_space)

        features = self._construct_features(self._current_obs, flat_action)
        next_features = self._construct_features(next_obs, flat_action)

        if terminated:
            target = jnp.atleast_1d(jnp.array(reward, dtype=jnp.float32))
        else:
            bootstrap = self._value_fn(next_features)
            target_val = float(reward) + _discounted_bootstrap(self._gamma, float(bootstrap))
            target = jnp.atleast_1d(jnp.array(target_val, dtype=jnp.float32))

        self._step_count += 1

        if terminated or truncated:
            self._episode_count += 1
            self._current_obs = None
        else:
            self._current_obs = next_obs

        return TimeStep(observation=features, target=target)


def make_gymnasium_stream(
    env_id: str,
    mode: PredictionMode = PredictionMode.REWARD,
    policy: Callable[[Array], Any] | None = None,
    gamma: float = 0.99,
    include_action_in_features: bool = True,
    seed: int = 0,
    **env_kwargs: Any,
) -> GymnasiumStream:
    """Factory function to create a GymnasiumStream from an environment ID.

    Args:
        env_id: Gymnasium environment ID (e.g., "CartPole-v1")
        mode: What to predict (REWARD, NEXT_STATE, VALUE)
        policy: Action selection function. If None, uses random policy
        gamma: Discount factor for VALUE mode
        include_action_in_features: If True, features = concat(obs, action)
        seed: Random seed
        **env_kwargs: Additional arguments passed to gymnasium.make()

    Returns:
        GymnasiumStream wrapping the environment
    """
    if type(env_id) is not str or not env_id:
        raise ValueError("env_id must be a non-empty exact string")
    mode = _require_mode(mode)
    include_action_in_features = _require_exact_bool(
        "include_action_in_features", include_action_in_features
    )
    seed = require_jax_seed(seed)
    gamma = validated_float32_scalar("gamma", gamma, lower=0.0, upper=1.0)
    if policy is not None and not callable(policy):
        raise ValueError("policy must be callable or None")

    import gymnasium

    env = gymnasium.make(env_id, **env_kwargs)
    try:
        return GymnasiumStream(
            env=env,
            mode=mode,
            policy=policy,
            gamma=gamma,
            include_action_in_features=include_action_in_features,
            seed=seed,
        )
    except Exception:
        try:
            env.close()
        except Exception:
            pass
        raise
