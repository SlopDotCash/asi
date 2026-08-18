# mypy: disable-error-code="attr-defined,call-arg,comparison-overlap,redundant-cast,unused-ignore"
"""Public Step 5 average-reward prediction facade (Predict II).

Step 5 of the Alberta Plan moves prediction to the continuing, average-reward
setting: no discounting, no episodes.  The learner behind this facade is the
linear differential TD predictor in
:mod:`alberta_framework.core.average_reward`, which learns values relative to
an online reward-rate estimate via
``td_error = reward - average_reward + v(s') - v(s)`` and updates the rate
estimate from the same TD error.

The default step-sizes separate two timescales: values at ``step_size=0.05``,
the reward-rate estimate at ``average_reward_step_size=0.01``.  The rate
estimate enters every differential TD target, so it is moved more slowly than
the value weights — a fast-moving estimate would make every value target
non-stationary.

References:
    Wan, Naik, & Sutton (2021). "Learning and Planning in Average-Reward
        Markov Decision Processes."
    Sutton & Barto (2018). "Reinforcement Learning: An Introduction," §10.3-4.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework._seed_validation import require_jax_seed
from alberta_framework.core.average_reward import (
    DifferentialTDArrayResult,
    DifferentialTDConfig,
    DifferentialTDLearner,
    DifferentialTDState,
    run_differential_td_from_arrays,
)
from alberta_framework.steps._float32_validation import finite_real_and_float32
from alberta_framework.steps._smoke_record_validation import require_step_shape

_STEP5_CONFIG_KEYS = frozenset(
    {"step_size", "average_reward_step_size", "trace_decay"}
)
_STEP5_CONFIG_KEYS_ERROR = (
    "Step5AverageRewardTDConfig payload keys must be exactly "
    "['average_reward_step_size', 'step_size', 'trace_decay']"
)
_INT32_MAX = 2**31 - 1
_ACTUAL_INT_TYPES = (
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
)
_NUMPY_INTEGER_TYPES = _ACTUAL_INT_TYPES


def _require_int(
    name: str,
    value: object,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Reject non-integers (including bool and class-spoofed actual types)."""
    actual_type = type(value)
    if not any(actual_type is candidate for candidate in _ACTUAL_INT_TYPES):
        raise ValueError(f"{name} must be an integer")
    number = int(cast(Integral, value))
    if minimum is not None and number < minimum:
        if minimum == 1:
            raise ValueError(f"{name} must be positive")
        if minimum == 0:
            raise ValueError(f"{name} must be non-negative")
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return number


def _require_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a built-in bool")
    return value


def _finite_float32_scalar(name: str, value: object) -> tuple[int, int, float]:
    """Validate a real scalar and retain the exact ratio used for rounding."""
    _, numerator, denominator, narrowed = finite_real_and_float32(name, value)
    return numerator, denominator, narrowed


def _compatible_float32_storage(value: object, narrowed: float) -> float:
    """Preserve a builtin payload only when its eventual sink is proved equal."""
    if type(value) is float:
        return value
    if type(value) is int and value == narrowed:
        return value
    return narrowed


@dataclass(frozen=True)
class Step5AverageRewardTDConfig:
    """Config for the public Step 5 differential TD facade."""

    step_size: float = 0.05
    average_reward_step_size: float = 0.01
    trace_decay: float = 0.0

    def __post_init__(self) -> None:
        """Reject malformed scientific scalars before JAX execution."""
        if type(self) is not Step5AverageRewardTDConfig:
            raise ValueError("config must be an actual Step5AverageRewardTDConfig")
        step_numerator, _, step_size = _finite_float32_scalar(
            "step_size",
            self.step_size,
        )
        average_numerator, _, average_reward_step_size = _finite_float32_scalar(
            "average_reward_step_size", self.average_reward_step_size
        )
        trace_numerator, trace_denominator, trace_decay = _finite_float32_scalar(
            "trace_decay",
            self.trace_decay,
        )
        if self.step_size < 0.0 or step_numerator < 0 or step_size < 0.0:
            raise ValueError("step_size must be non-negative")
        if (
            self.average_reward_step_size < 0.0
            or average_numerator < 0
            or average_reward_step_size < 0.0
        ):
            raise ValueError("average_reward_step_size must be non-negative")
        if (
            not 0.0 <= self.trace_decay <= 1.0
            or trace_numerator < 0
            or trace_numerator > trace_denominator
            or not 0.0 <= trace_decay <= 1.0
        ):
            raise ValueError("trace_decay must be in [0, 1]")
        # Preserve builtin floats and sink-exact builtin integers. Other Reals
        # need the already-rounded value so the JAX sink cannot double-round.
        object.__setattr__(
            self,
            "step_size",
            _compatible_float32_storage(self.step_size, step_size),
        )
        object.__setattr__(
            self,
            "average_reward_step_size",
            _compatible_float32_storage(
                self.average_reward_step_size,
                average_reward_step_size,
            ),
        )
        object.__setattr__(
            self,
            "trace_decay",
            _compatible_float32_storage(self.trace_decay, trace_decay),
        )
        self.to_core_config()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Step5AverageRewardTDConfig:
        """Reconstruct from :meth:`to_dict` output."""
        if cls is not Step5AverageRewardTDConfig:
            raise ValueError("cls must be Step5AverageRewardTDConfig")
        if type(payload) is not dict:
            raise ValueError("payload must be an actual dict")
        if any(type(key) is not str for key in payload):
            raise ValueError("payload keys must be exact strings")
        if payload.keys() != _STEP5_CONFIG_KEYS:
            raise ValueError(_STEP5_CONFIG_KEYS_ERROR)
        return cls(**cast(Any, payload))

    def to_core_config(self) -> DifferentialTDConfig:
        """Return the core differential TD config."""
        return DifferentialTDConfig(
            step_size=self.step_size,
            average_reward_step_size=self.average_reward_step_size,
            trace_decay=self.trace_decay,
        )


@dataclass(frozen=True)
class Step5SmokeResult:
    """Summary returned by :func:`run_step5_smoke`."""

    config: Step5AverageRewardTDConfig
    steps: int
    seed: int
    predictions_shape: tuple[int, ...]
    td_errors_shape: tuple[int, ...]
    average_rewards_shape: tuple[int, ...]
    finite: bool
    learner_config: dict[str, Any]

    def __post_init__(self) -> None:
        if type(self) is not Step5SmokeResult:
            raise ValueError("result must be an actual Step5SmokeResult")
        if type(self.config) is not Step5AverageRewardTDConfig:
            raise ValueError("config must be an actual Step5AverageRewardTDConfig")
        object.__setattr__(
            self, "steps", _require_int("steps", self.steps, minimum=1, maximum=_INT32_MAX)
        )
        object.__setattr__(self, "seed", require_jax_seed(self.seed, name="seed"))
        for name in (
            "predictions_shape",
            "td_errors_shape",
            "average_rewards_shape",
        ):
            object.__setattr__(
                self,
                name,
                require_step_shape(name, getattr(self, name), steps=self.steps),
            )
        object.__setattr__(self, "finite", _require_bool("finite", self.finite))
        for name in ("predictions_shape", "td_errors_shape", "average_rewards_shape"):
            if getattr(self, name) != (self.steps,):
                raise ValueError(f"{name} must be exactly (steps,)")
        _require_closed_json_object("learner_config", self.learner_config)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["config"] = self.config.to_dict()
        payload["predictions_shape"] = list(self.predictions_shape)
        payload["td_errors_shape"] = list(self.td_errors_shape)
        payload["average_rewards_shape"] = list(self.average_rewards_shape)
        return payload


def _trusted_array(
    name: str,
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    actual_type = type(value)
    if not (
        actual_type is np.ndarray
        or issubclass(actual_type, jax.Array)
        or issubclass(actual_type, jax.core.Tracer)
    ):
        raise TypeError(f"{name} must be a trusted array")
    try:
        actual_shape = tuple(value.shape)  # type: ignore[attr-defined]
        actual_dtype = jnp.dtype(value.dtype)  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(f"{name} must expose trusted shape and dtype metadata") from error
    if actual_shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if actual_dtype != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}")
    return cast(Array, value)


def _preflight_smoke_resources(steps: int, feature_dim: int) -> None:
    # The smoke lane owns its (T+1, D) observation table and T rewards in
    # addition to the learner's two D-vectors and four scalar leaves.
    learner_scalars = 2 * feature_dim + 4
    input_scalars = (steps + 1) * feature_dim + steps
    output_float_scalars = 7 * steps
    output_bool_bytes = steps
    total_bytes = 4 * (learner_scalars + input_scalars + output_float_scalars)
    total_bytes += output_bool_bytes
    if (
        any(
            count > _INT32_MAX or 4 * count > _INT32_MAX
            for count in (learner_scalars, input_scalars, output_float_scalars)
        )
        or output_bool_bytes > _INT32_MAX
        or total_bytes > _INT32_MAX
    ):
        raise ValueError("derived Step 5 smoke float32 resources exceed signed-int32 bounds")


def _require_closed_json_object(name: str, value: object) -> dict[str, Any]:
    """Require an acyclic tree of exact built-in JSON identities."""
    if type(value) is not dict:
        raise ValueError(f"{name} must be an actual dict")
    active_containers: set[int] = set()
    stack: list[tuple[bool, object]] = [(True, value)]
    while stack:
        entering, item = stack.pop()
        if not entering:
            active_containers.remove(id(item))
            continue
        if type(item) is dict:
            identity = id(item)
            if identity in active_containers:
                raise ValueError(f"{name} must be an acyclic JSON object")
            active_containers.add(identity)
            mapping = cast(dict[object, object], item)
            if any(type(key) is not str for key in mapping):
                raise ValueError(f"{name} keys must be exact strings")
            stack.append((False, item))
            stack.extend((True, child) for child in mapping.values())
        elif type(item) is list:
            identity = id(item)
            if identity in active_containers:
                raise ValueError(f"{name} must be an acyclic JSON object")
            active_containers.add(identity)
            stack.append((False, item))
            stack.extend((True, child) for child in cast(list[object], item))
        elif item is None or type(item) in (str, bool, int):
            continue
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError(f"{name} floats must be finite")
        else:
            raise ValueError(f"{name} must contain only exact JSON identities")
    return cast(dict[str, Any], value)


def _state_feature_dim(state: DifferentialTDState) -> int:
    weights = state.weights
    actual_type = type(weights)
    if not (
        actual_type is np.ndarray
        or issubclass(actual_type, jax.Array)
        or issubclass(actual_type, jax.core.Tracer)
    ):
        raise TypeError("state.weights must be a trusted array")
    try:
        shape = tuple(weights.shape)
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError("state.weights must expose trusted shape metadata") from error
    if len(shape) != 1 or not 1 <= shape[0] <= _INT32_MAX:
        raise ValueError("state.weights must be a nonempty signed-int32 vector")
    feature_dim = shape[0]
    _trusted_array("state.weights", weights, shape=(feature_dim,), dtype=jnp.float32)
    _trusted_array(
        "state.eligibility_traces",
        state.eligibility_traces,
        shape=(feature_dim,),
        dtype=jnp.float32,
    )
    for name in ("bias", "bias_eligibility_trace", "average_reward"):
        _trusted_array(
            name=f"state.{name}",
            value=getattr(state, name),
            shape=(),
            dtype=jnp.float32,
        )
    _trusted_array("state.step_count", state.step_count, shape=(), dtype=jnp.int32)
    if type(state.birth_timestamp) is not float or not math.isfinite(state.birth_timestamp):
        raise ValueError("state.birth_timestamp must be a finite built-in float")
    if (
        type(state.uptime_s) is not float
        or not math.isfinite(state.uptime_s)
        or state.uptime_s < 0.0
    ):
        raise ValueError("state.uptime_s must be a finite non-negative built-in float")
    return feature_dim


def make_step5_td_learner(
    config: Step5AverageRewardTDConfig | None = None,
) -> DifferentialTDLearner:
    """Create the public Step 5 differential TD learner."""
    cfg = Step5AverageRewardTDConfig() if config is None else config
    if type(cfg) is not Step5AverageRewardTDConfig:
        raise TypeError("config must be an actual Step5AverageRewardTDConfig")
    return DifferentialTDLearner(cfg.to_core_config())


def run_step5_scan(
    learner: DifferentialTDLearner,
    state: object,
    observations: object,
    rewards: object,
    next_observations: object,
) -> DifferentialTDArrayResult:
    """Run Step 5 differential TD over pre-collected transition arrays."""
    if type(learner) is not DifferentialTDLearner:
        raise TypeError("learner must be an actual DifferentialTDLearner")
    if type(state) is not DifferentialTDState:
        raise TypeError("state must be an actual DifferentialTDState")
    feature_dim = _state_feature_dim(state)
    actual_type = type(observations)
    if not (
        actual_type is np.ndarray
        or issubclass(actual_type, jax.Array)
        or issubclass(actual_type, jax.core.Tracer)
    ):
        raise TypeError("observations must be a trusted array")
    try:
        steps = int(observations.shape[0])  # type: ignore[attr-defined]
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise TypeError("observations must expose trusted shape metadata") from error
    if not 1 <= steps <= _INT32_MAX:
        raise ValueError("observations must contain between 1 and signed-int32 steps")
    checked_observations = _trusted_array(
        "observations", observations, shape=(steps, feature_dim), dtype=jnp.float32
    )
    checked_rewards = _trusted_array("rewards", rewards, shape=(steps,), dtype=jnp.float32)
    checked_next = _trusted_array(
        "next_observations",
        next_observations,
        shape=(steps, feature_dim),
        dtype=jnp.float32,
    )
    return run_differential_td_from_arrays(
        learner, state, checked_observations, checked_rewards, checked_next
    )


def run_step5_smoke(
    config: Step5AverageRewardTDConfig | None = None,
    *,
    steps: int = 32,
    feature_dim: int = 6,
    seed: int = 0,
) -> Step5SmokeResult:
    """Run a tiny deterministic Step 5 integration probe."""
    steps = _require_int("steps", steps, minimum=1, maximum=_INT32_MAX)
    feature_dim = _require_int("feature_dim", feature_dim, minimum=1, maximum=_INT32_MAX)
    seed = require_jax_seed(seed, name="seed")
    _preflight_smoke_resources(steps, feature_dim)

    cfg = Step5AverageRewardTDConfig() if config is None else config
    if type(cfg) is not Step5AverageRewardTDConfig:
        raise TypeError("config must be an actual Step5AverageRewardTDConfig")
    learner = make_step5_td_learner(cfg)
    key = jr.key(seed)
    obs_key, reward_key = jr.split(key)
    observations = jr.normal(obs_key, (steps + 1, feature_dim), dtype=jnp.float32)
    rewards = 0.25 + 0.1 * jnp.tanh(
        observations[:-1, 0] + jr.normal(reward_key, (steps,), dtype=jnp.float32)
    )
    state = learner.init(feature_dim)
    result = run_step5_scan(learner, state, observations[:-1], rewards, observations[1:])
    result.td_errors.block_until_ready()
    finite = bool(
        jnp.all(jnp.isfinite(result.predictions))
        & jnp.all(jnp.isfinite(result.td_errors))
        & jnp.all(jnp.isfinite(result.average_rewards))
        & jnp.all(result.updates_applied)
    )
    return Step5SmokeResult(
        config=cfg,
        steps=steps,
        seed=seed,
        predictions_shape=tuple(int(dim) for dim in result.predictions.shape),
        td_errors_shape=tuple(int(dim) for dim in result.td_errors.shape),
        average_rewards_shape=tuple(int(dim) for dim in result.average_rewards.shape),
        finite=finite,
        learner_config=learner.to_config(),
    )
