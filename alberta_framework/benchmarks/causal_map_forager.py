"""Observation-causal cognitive-map planner for stationary Forager.

This module provides an Alberta field-of-view variant that learns a compact
world model online.  Its state consists only of:

* a relative, toroidal map assembled from egocentric observations;
* empirical reward means for the observation channels it has collected;
* collection and reappearance times used to learn respawn schedules; and
* visit counts and the policy's own previous action.

The arbitrary map origin is the agent's initial location.  No evaluator
context, global position, reward grid, object id, task label, biome id, hidden
clock, or privileged ``info`` value enters the policy transition.  The
benchmark runner sees environment state only to execute the ordinary public
environment API and to collect evaluator metrics.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.benchmarks.forager import (
    FORAGER_ENVIRONMENT_RNG_SCHEDULE,
    FORAGER_FOV_EMA_DECAY,
    FORAGER_FOV_EMA_SUBSAMPLE,
    ForagerAgentContext,
    ForagerBatchMode,
    ForagerBenchmarkConfig,
    ForagerRewardTraceSinkFactory,
    ForagerRunResult,
    _abort_reward_trace_sinks,
    _adjusted_ewm_chunk,
    _append_reward_trace,
    _create_reward_trace_sinks,
    _exact_float32_biome_regret,
    _finalize_reward_trace_sinks,
    _fov_last_tenth_ema_auc,
    _unadjusted_ema_chunk,
    environment_rng_schedule_sha256,
    forager_metric_contract,
)

CAUSAL_MAP_STATE_SCHEMA = "alberta.forager_causal_map_state.v5"
CAUSAL_MAP_VARIANT_KIND = "alberta_causal_map"
_CAUSAL_MAP_RNG_NAMESPACE = 0xCA05A14
_CAUSAL_MAP_PRNG_IMPL = "threefry2x32"
_CAUSAL_MAP_ENVIRONMENT_PRNG_IMPL = "threefry2x32"
_EXACT_NUMPY_INTEGER_TYPES = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
)
_EXACT_INTEGER_TYPES = (int, *_EXACT_NUMPY_INTEGER_TYPES)
_EXACT_REAL_TYPES = (
    int,
    float,
    *_EXACT_NUMPY_INTEGER_TYPES,
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
)


def _require_exact_str(name: object, value: object) -> str:
    if type(name) is not str:
        raise ValueError("name must be an exact string")
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    return value
_INT32_MAX = int(np.iinfo(np.int32).max)
_MAX_CAUSAL_MAP_CELLS = 4_096
_MAX_SAFE_FLOAT32_INT32 = float(
    np.nextafter(np.float32(_INT32_MAX), np.float32(-np.inf))
)
_FLOAT32_SCORE_HEADROOM = float(np.finfo(np.float32).max) / 8.0
_DIRECTION_STEPS = (
    (0, 1),   # action 0: +y
    (1, 0),   # action 1: +x
    (0, -1),  # action 2: -y
    (-1, 0),  # action 3: -x
)
_DIRECTIONS = jnp.asarray(
    _DIRECTION_STEPS,
    dtype=jnp.int32,
)


def causal_map_rng_contract() -> dict[str, Any]:
    """Describe the causal-map agent's seed-isolated PRNG root."""
    return {
        "root": (
            "jax.random.fold_in(jax.random.key(seed, impl=prng_impl), namespace)"
        ),
        "namespace": _CAUSAL_MAP_RNG_NAMESPACE,
        "prng_impl": _CAUSAL_MAP_PRNG_IMPL,
        "jax_threefry_partitionable": _threefry_partitionable_mode(),
        "environment_key_shared_with_agent": False,
    }


def _finite_float32(value: Any) -> bool:
    """Return whether a scalar remains finite in the planner's JAX dtype."""
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.float32(value)
    return bool(np.isfinite(converted))


def _prng_impl_name(key: Array) -> str:
    """Return the stable JAX PRNG implementation name for a typed key."""
    return str(jr.key_impl(key))


def _threefry_partitionable_mode() -> bool:
    """Return the JAX Threefry split mode that determines key trajectories."""
    return bool(jax.config.jax_threefry_partitionable)


def _causal_map_environment_key(seed: int | Array) -> Array:
    """Return the explicitly typed, default-PRNG-independent environment root."""
    key = jr.key(seed, impl=_CAUSAL_MAP_ENVIRONMENT_PRNG_IMPL)
    if _prng_impl_name(key) != _CAUSAL_MAP_ENVIRONMENT_PRNG_IMPL:
        raise RuntimeError("causal-map environment PRNG implementation mismatch")
    return key


@dataclass(frozen=True)
class CausalMapForagerConfig:
    """Configuration for the learned stationary field-of-view planner.

    ``world_shape`` and the four-action movement convention are public task
    semantics, not evaluator state.  ``initial_retry_delay`` is deliberately
    channel-agnostic.  Once any mapped cell is observed to reappear, the
    channel's schedule is estimated from observed collection-to-reappearance
    intervals rather than an object-specific constant.

    ``exploration_probability`` only arbitrates between an exploitation target
    and a genuinely unobserved cell reachable under cost-aware routing.  It
    never redirects the policy to generic coverage after every reachable cell
    has been observed.
    When neither kind of target exists, the planner still uses safe
    least-visited coverage as a fail-safe, independent of the exploration coin.

    With ``arrival_aware_readiness=True``, a pending collected cell is eligible
    when its predicted ready step is no later than the last decision state
    before the action that enters it.  This one-step lead is intentional:
    public Foragax transitions move and compute collection reward before
    decrementing respawn timers and returning the next observation.  Setting
    the option to ``False`` retains decision-time-only readiness.
    """

    world_shape: tuple[int, int] = (15, 15)
    optimistic_unknown_reward: float = 2.0
    negative_reward_threshold: float = -1e-6
    reward_observation_epsilon: float = 1e-7
    initial_retry_delay: int = 10
    maximum_retry_delay: int = 1_024
    maximum_retry_exponent: int = 10
    minimum_respawn_samples: int = 1
    maximum_exact_interval_width: int = 0
    respawn_safety_quantile: float = 0.75
    respawn_safety_factor: float = 1.0
    minimum_respawn_delay: int = 1
    maximum_respawn_delay: int = 4_096
    distance_cost: float = 0.08
    reverse_action_penalty: float = 0.35
    visit_penalty: float = 0.0
    retry_penalty: float = 0.2
    tie_break_scale: float = 1e-4
    exploration_probability: float = 0.05
    arrival_aware_readiness: bool = True
    one_hot_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if (
            type(self.world_shape) is not tuple
            or len(self.world_shape) != 2
            or any(
                type(value) is not int or value < 1
                or value > _INT32_MAX
                for value in self.world_shape
            )
            or math.prod(self.world_shape) > _MAX_CAUSAL_MAP_CELLS
        ):
            raise ValueError(
                "world_shape must contain two positive integers and at most "
                f"{_MAX_CAUSAL_MAP_CELLS} total cells"
            )
        object.__setattr__(
            self,
            "world_shape",
            tuple(int(value) for value in self.world_shape),
        )
        for name in (
            "optimistic_unknown_reward",
            "negative_reward_threshold",
        ):
            value = getattr(self, name)
            if (
                type(value) not in _EXACT_REAL_TYPES
                or not math.isfinite(value)
                or not _finite_float32(value)
            ):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))
        positive_float = (
            "reward_observation_epsilon",
            "respawn_safety_factor",
            "distance_cost",
            "tie_break_scale",
            "one_hot_tolerance",
        )
        for name in positive_float:
            value = getattr(self, name)
            if (
                type(value) not in _EXACT_REAL_TYPES
                or not math.isfinite(value)
                or value <= 0.0
                or not _finite_float32(value)
                or np.float32(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, float(value))
        if np.float32(self.respawn_safety_factor) < np.float32(1.0):
            raise ValueError(
                "respawn_safety_factor must be at least 1.0 after float32 conversion"
            )
        nonnegative_float = (
            "reverse_action_penalty",
            "visit_penalty",
            "retry_penalty",
        )
        for name in nonnegative_float:
            value = getattr(self, name)
            if (
                type(value) not in _EXACT_REAL_TYPES
                or not math.isfinite(value)
                or value < 0.0
                or not _finite_float32(value)
            ):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, float(value))
        if (
            type(self.exploration_probability) not in _EXACT_REAL_TYPES
            or not math.isfinite(self.exploration_probability)
            or not 0.0 <= self.exploration_probability <= 1.0
            or not _finite_float32(self.exploration_probability)
        ):
            raise ValueError("exploration_probability must lie in [0, 1]")
        object.__setattr__(
            self,
            "exploration_probability",
            float(self.exploration_probability),
        )
        if type(self.arrival_aware_readiness) is not bool:
            raise ValueError("arrival_aware_readiness must be a boolean")
        positive_int = (
            "initial_retry_delay",
            "maximum_retry_delay",
            "minimum_respawn_samples",
            "minimum_respawn_delay",
            "maximum_respawn_delay",
        )
        for name in positive_int:
            value = getattr(self, name)
            if (
                type(value) not in _EXACT_INTEGER_TYPES
                or value < 1
                or value > _INT32_MAX
            ):
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if (
            type(self.maximum_retry_exponent) not in _EXACT_INTEGER_TYPES
            or self.maximum_retry_exponent < 0
            or self.maximum_retry_exponent > _INT32_MAX
        ):
            raise ValueError("maximum_retry_exponent must be a non-negative integer")
        object.__setattr__(
            self,
            "maximum_retry_exponent",
            int(self.maximum_retry_exponent),
        )
        if (
            type(self.maximum_exact_interval_width) not in _EXACT_INTEGER_TYPES
            or self.maximum_exact_interval_width < 0
            or self.maximum_exact_interval_width > _INT32_MAX
        ):
            raise ValueError(
                "maximum_exact_interval_width must be a non-negative integer"
            )
        object.__setattr__(
            self,
            "maximum_exact_interval_width",
            int(self.maximum_exact_interval_width),
        )
        if self.maximum_exact_interval_width != 0:
            raise ValueError(
                "maximum_exact_interval_width must be zero; intervals with "
                "nonzero width are censored rather than exact"
            )
        if self.maximum_retry_delay < self.initial_retry_delay:
            raise ValueError("maximum_retry_delay must be at least initial_retry_delay")
        if self.maximum_respawn_delay < self.minimum_respawn_delay:
            raise ValueError(
                "maximum_respawn_delay must be at least minimum_respawn_delay"
            )
        if (
            type(self.respawn_safety_quantile) not in _EXACT_REAL_TYPES
            or not math.isfinite(self.respawn_safety_quantile)
            or not 0.5 <= self.respawn_safety_quantile < 1.0
            or not _finite_float32(self.respawn_safety_quantile)
        ):
            raise ValueError("respawn_safety_quantile must lie in [0.5, 1)")
        object.__setattr__(
            self,
            "respawn_safety_quantile",
            float(self.respawn_safety_quantile),
        )
        if not _finite_float32(self.respawn_quantile_z):
            raise ValueError("respawn_safety_quantile produces a non-finite float32 z")
        if (
            not 0.0 < self.one_hot_tolerance < 0.5
            or not np.float32(self.one_hot_tolerance) < np.float32(0.5)
        ):
            raise ValueError("one_hot_tolerance must lie strictly between 0 and 0.5")
        # Every configurable term below is combined in float32 inside the JAX
        # planner.  Individual float32 representability is insufficient: for
        # example, a finite retry penalty multiplied by a finite retry exponent
        # can overflow and turn every candidate score into -inf.  Reserve ample
        # headroom for learned rewards and for addition-order differences across
        # backends, while rejecting configurations whose own worst-case terms
        # cannot be scored finitely.
        maximum_path_distance = math.prod(self.world_shape) + 1
        score_bounds = {
            "target": (
                abs(self.optimistic_unknown_reward)
                + self.distance_cost * maximum_path_distance
                + self.retry_penalty * self.maximum_retry_exponent
                + self.tie_break_scale
            ),
            "pursuit": (
                maximum_path_distance
                + self.visit_penalty * math.log1p(_INT32_MAX)
                + self.reverse_action_penalty
                + self.tie_break_scale
            ),
            "exploration": (
                _INT32_MAX
                + self.reverse_action_penalty
                + self.tie_break_scale
            ),
            "respawn": (
                self.respawn_safety_factor
                * (1.0 + abs(self.respawn_quantile_z))
                * _INT32_MAX
            ),
        }
        invalid_bounds = [
            name
            for name, bound in score_bounds.items()
            if not math.isfinite(bound)
            or not _finite_float32(bound)
            or bound > _FLOAT32_SCORE_HEADROOM
        ]
        if invalid_bounds:
            raise ValueError(
                "configuration can overflow float32 planner arithmetic: "
                + ", ".join(sorted(invalid_bounds))
            )

    @property
    def height(self) -> int:
        """Map height."""
        return self.world_shape[0]

    @property
    def width(self) -> int:
        """Map width."""
        return self.world_shape[1]

    @property
    def respawn_quantile_z(self) -> float:
        """One-sided normal quantile used with online Welford statistics."""
        return float(NormalDist().inv_cdf(self.respawn_safety_quantile))

    def to_dict(self) -> dict[str, Any]:
        """Return a fully explicit JSON-compatible configuration."""
        data = dataclasses.asdict(self)
        data = {
            name: value.item() if isinstance(value, np.generic) else value
            for name, value in data.items()
        }
        data["world_shape"] = [
            value.item() if isinstance(value, np.generic) else value
            for value in self.world_shape
        ]
        data["respawn_quantile_z"] = self.respawn_quantile_z
        return data

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CausalMapForagerConfig:
        """Restore a configuration, rejecting unknown or derived fields."""
        if not isinstance(payload, Mapping):
            raise ValueError("causal-map config must be a mapping")
        data = dict(payload)
        declared_quantile_z = data.pop("respawn_quantile_z", None)
        if "world_shape" in data:
            value = data["world_shape"]
            if (
                not isinstance(value, Sequence)
                or isinstance(value, (str, bytes))
                or len(value) != 2
            ):
                raise ValueError("world_shape must contain two integers")
            data["world_shape"] = tuple(value)
        known = {item.name for item in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown causal-map config fields: {sorted(unknown)}")
        config = cls(**data)
        if declared_quantile_z is not None and (
            type(declared_quantile_z) not in (int, float)
            or not math.isclose(
                float(declared_quantile_z),
                config.respawn_quantile_z,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ValueError(
                "respawn_quantile_z does not match respawn_safety_quantile"
            )
        return config

    def fingerprint(self) -> str:
        """Return a canonical SHA-256 suitable for benchmark provenance."""
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


class CausalMapForagerState(NamedTuple):
    """Finite JAX state for the observation-causal map and planner."""

    step_count: Array
    initial_seed: Array
    position: Array
    last_action: Array
    last_target_channel: Array
    last_target_position: Array
    last_target_expected_active: Array
    rng_key: Array
    jax_threefry_partitionable: Array
    cell_channel: Array
    cell_active: Array
    cell_collection_step: Array
    cell_ready_step: Array
    cell_retry_count: Array
    cell_last_seen_step: Array
    cell_last_absent_step: Array
    visit_count: Array
    reward_sum: Array
    reward_count: Array
    respawn_interval_count: Array
    respawn_interval_lower_floor: Array
    respawn_interval_lower_remainder: Array
    respawn_interval_upper_floor: Array
    respawn_interval_upper_remainder: Array
    respawn_exact_count: Array
    respawn_exact_floor: Array
    respawn_exact_remainder: Array
    respawn_exact_mean: Array
    respawn_exact_m2: Array


class CausalMapStepDiagnostics(NamedTuple):
    """Small causal diagnostics emitted by one planner transition."""

    learned_reward: Array
    learned_respawn: Array
    retry_miss: Array
    known_cells: Array
    known_negative_cells: Array


_STATE_INTEGER_FIELDS = frozenset(
    {
        "step_count",
        "position",
        "last_action",
        "last_target_channel",
        "last_target_position",
        "cell_channel",
        "cell_collection_step",
        "cell_ready_step",
        "cell_retry_count",
        "cell_last_seen_step",
        "cell_last_absent_step",
        "visit_count",
        "reward_count",
        "respawn_interval_count",
        "respawn_interval_lower_floor",
        "respawn_interval_lower_remainder",
        "respawn_interval_upper_floor",
        "respawn_interval_upper_remainder",
        "respawn_exact_count",
        "respawn_exact_floor",
        "respawn_exact_remainder",
    }
)
_STATE_UINT32_FIELDS = frozenset({"initial_seed"})
_STATE_BOOLEAN_FIELDS = frozenset(
    {
        "last_target_expected_active",
        "jax_threefry_partitionable",
        "cell_active",
    }
)
_STATE_FLOAT_FIELDS = frozenset(CausalMapForagerState._fields) - (
    _STATE_INTEGER_FIELDS
    | _STATE_UINT32_FIELDS
    | _STATE_BOOLEAN_FIELDS
    | {"rng_key"}
)
_STATE_SERIALIZED_DTYPES = {
    name: (
        "uint32-key-data"
        if name == "rng_key"
        else (
            "uint32"
            if name in _STATE_UINT32_FIELDS
            else
            "int32"
            if name in _STATE_INTEGER_FIELDS
            else "bool"
            if name in _STATE_BOOLEAN_FIELDS
            else "float32"
        )
    )
    for name in CausalMapForagerState._fields
}


def _image(observation: Any) -> Array:
    if isinstance(observation, Mapping):
        if "image" not in observation:
            raise ValueError("mapping observation must contain an 'image' entry")
        image = observation["image"]
    else:
        image = observation
    try:
        raw_image = jnp.asarray(image)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            "causal-map observation must have a real numeric dtype"
        ) from exc
    if not (
        jnp.issubdtype(raw_image.dtype, jnp.integer)
        or jnp.issubdtype(raw_image.dtype, jnp.floating)
    ):
        raise ValueError("causal-map observation must have a real numeric dtype")
    return raw_image.astype(jnp.float32)


def _validated_seed(seed: int | Array) -> Array:
    """Return one uint32 seed without bool, range, shape, or cast laundering."""
    if not isinstance(seed, (jax.Array, jax.core.Tracer)):
        try:
            host_seed = np.asarray(seed)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                "causal-map seed must be one non-bool uint32-compatible integer"
            ) from exc
        if (
            host_seed.shape != ()
            or host_seed.dtype.kind not in ("i", "u")
            or not 0 <= int(host_seed) <= np.iinfo(np.uint32).max
        ):
            raise ValueError(
                "causal-map seed must be one non-bool uint32-compatible integer"
            )
        return jnp.asarray(int(host_seed), dtype=jnp.uint32)

    raw_seed = jnp.asarray(seed)
    if raw_seed.ndim != 0 or not jnp.issubdtype(raw_seed.dtype, jnp.integer):
        raise ValueError(
            "causal-map seed must be one non-bool uint32-compatible integer"
        )
    valid_range = raw_seed >= 0
    if raw_seed.dtype.itemsize > np.dtype(np.uint32).itemsize:
        valid_range = valid_range & (
            raw_seed
            <= jnp.asarray(np.iinfo(np.uint32).max, dtype=raw_seed.dtype)
        )
    checked_seed = _runtime_require(
        valid_range,
        raw_seed,
        message="causal-map seed must be uint32-compatible",
    )
    return checked_seed.astype(jnp.uint32)


def _validate_observation_host(
    observation: Any,
    config: CausalMapForagerConfig,
) -> tuple[int, int, int]:
    image = np.asarray(_image(observation), dtype=np.float32)
    if image.ndim != 3:
        raise ValueError("causal-map Forager observation must have rank 3")
    aperture_h, aperture_w, channels = image.shape
    if channels < 1:
        raise ValueError("causal-map Forager requires at least one object/color channel")
    if (
        aperture_h < 3
        or aperture_w < 3
        or aperture_h % 2 == 0
        or aperture_w % 2 == 0
    ):
        raise ValueError(
            "causal-map Forager requires an odd aperture of at least 3; "
            "aperture 1 cannot causally identify a collected destination channel"
        )
    if aperture_h > config.height or aperture_w > config.width:
        raise ValueError("aperture cannot exceed the configured toroidal world")
    if not np.all(np.isfinite(image)):
        raise ValueError("observation must be finite")
    tolerance = config.one_hot_tolerance
    if np.any(image < -tolerance) or np.any(image > 1.0 + tolerance):
        raise ValueError("observation channels must lie in [0, 1]")
    channel_sums = np.sum(image, axis=-1)
    if np.any(channel_sums > 1.0 + tolerance):
        raise ValueError("causal-map planner requires one-hot color/object channels")
    near_binary = np.isclose(image, 0.0, atol=tolerance) | np.isclose(
        image, 1.0, atol=tolerance
    )
    if not bool(np.all(near_binary)):
        raise ValueError("causal-map planner requires binary one-hot observations")
    return aperture_h, aperture_w, channels


def _validate_static_observation_shape(
    image: Array,
    config: CausalMapForagerConfig,
) -> None:
    """Validate shape-only invariants that remain static while tracing JAX."""
    if image.ndim != 3:
        raise ValueError("causal-map Forager observation must have rank 3")
    aperture_h, aperture_w, channels = image.shape
    if channels < 1:
        raise ValueError("causal-map Forager requires at least one object/color channel")
    if (
        aperture_h < 3
        or aperture_w < 3
        or aperture_h % 2 == 0
        or aperture_w % 2 == 0
    ):
        raise ValueError(
            "causal-map Forager requires an odd aperture of at least 3; "
            "aperture 1 cannot causally identify a collected destination channel"
        )
    if aperture_h > config.height or aperture_w > config.width:
        raise ValueError("aperture cannot exceed the configured toroidal world")


def _runtime_require(
    predicate: Array,
    value: Array,
    *,
    message: str,
) -> Array:
    """Return ``value`` while raising from eager and compiled invalid paths."""
    scalar_predicate = jnp.reshape(jnp.asarray(predicate, dtype=jnp.bool_), ())

    def valid_branch(operand: tuple[Array, Array]) -> Array:
        checked_value, _ = operand
        return checked_value

    def invalid_branch(operand: tuple[Array, Array]) -> Array:
        checked_value, runtime_predicate = operand

        def raise_if_false(concrete_predicate: Any) -> None:
            if not bool(concrete_predicate):
                raise ValueError(message)

        jax.debug.callback(raise_if_false, runtime_predicate, ordered=True)
        return checked_value

    return cast(
        Array,
        jax.lax.cond(
            scalar_predicate,
            valid_branch,
            invalid_branch,
            (value, scalar_predicate),
        ),
    )


def _validated_observation_image(
    observation: Any,
    config: CausalMapForagerConfig,
) -> Array:
    """Validate observation shape and values on eager and compiled paths."""
    image = _image(observation)
    _validate_static_observation_shape(image, config)
    tolerance = jnp.asarray(config.one_hot_tolerance, dtype=jnp.float32)
    finite = jnp.all(jnp.isfinite(image))
    in_range = jnp.all((image >= -tolerance) & (image <= 1.0 + tolerance))
    binary = jnp.all(
        (jnp.abs(image) <= tolerance) | (jnp.abs(image - 1.0) <= tolerance)
    )
    one_hot = jnp.all(jnp.sum(image, axis=-1) <= 1.0 + tolerance)
    return _runtime_require(
        finite & in_range & binary & one_hot,
        image,
        message=(
            "causal-map observation must be finite, [0, 1]-bounded, binary, "
            "and one-hot per pixel"
        ),
    )


def _empty_state(
    channels: int,
    config: CausalMapForagerConfig,
    seed: int | Array,
) -> CausalMapForagerState:
    map_shape = config.world_shape
    zero_i = jnp.asarray(0, dtype=jnp.int32)
    return CausalMapForagerState(
        step_count=zero_i,
        initial_seed=jnp.asarray(seed, dtype=jnp.uint32),
        position=jnp.zeros((2,), dtype=jnp.int32),
        last_action=jnp.asarray(-1, dtype=jnp.int32),
        last_target_channel=jnp.asarray(-1, dtype=jnp.int32),
        last_target_position=jnp.zeros((2,), dtype=jnp.int32),
        last_target_expected_active=jnp.asarray(False),
        rng_key=jr.fold_in(
            jr.key(seed, impl=_CAUSAL_MAP_PRNG_IMPL),
            _CAUSAL_MAP_RNG_NAMESPACE,
        ),
        jax_threefry_partitionable=jnp.asarray(
            _threefry_partitionable_mode(),
            dtype=jnp.bool_,
        ),
        cell_channel=jnp.full(map_shape, -1, dtype=jnp.int32),
        cell_active=jnp.zeros(map_shape, dtype=jnp.bool_),
        cell_collection_step=jnp.full(map_shape, -1, dtype=jnp.int32),
        cell_ready_step=jnp.full(map_shape, -1, dtype=jnp.int32),
        cell_retry_count=jnp.zeros(map_shape, dtype=jnp.int32),
        cell_last_seen_step=jnp.full(map_shape, -1, dtype=jnp.int32),
        cell_last_absent_step=jnp.full(map_shape, -1, dtype=jnp.int32),
        visit_count=jnp.zeros(map_shape, dtype=jnp.int32),
        reward_sum=jnp.zeros((channels,), dtype=jnp.float32),
        reward_count=jnp.zeros((channels,), dtype=jnp.int32),
        respawn_interval_count=jnp.zeros((channels,), dtype=jnp.int32),
        respawn_interval_lower_floor=jnp.zeros((channels,), dtype=jnp.int32),
        respawn_interval_lower_remainder=jnp.zeros((channels,), dtype=jnp.int32),
        respawn_interval_upper_floor=jnp.zeros((channels,), dtype=jnp.int32),
        respawn_interval_upper_remainder=jnp.zeros((channels,), dtype=jnp.int32),
        respawn_exact_count=jnp.zeros((channels,), dtype=jnp.int32),
        respawn_exact_floor=jnp.zeros((channels,), dtype=jnp.int32),
        respawn_exact_remainder=jnp.zeros((channels,), dtype=jnp.int32),
        respawn_exact_mean=jnp.zeros((channels,), dtype=jnp.float32),
        respawn_exact_m2=jnp.zeros((channels,), dtype=jnp.float32),
    )


def _saturating_add_int32(value: Array, increment: Array) -> Array:
    """Add non-negative int32 values without allowing signed wraparound."""
    value = jnp.asarray(value, dtype=jnp.int32)
    increment = jnp.asarray(increment, dtype=jnp.int32)
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    return jnp.where(
        value >= maximum - increment,
        maximum,
        value + increment,
    )


def _require_unsaturated_step_count(step_count: Array) -> Array:
    """Reject a transition once the finite int32 lifetime is exhausted."""
    predicate = jnp.reshape(
        jnp.asarray(step_count < _INT32_MAX, dtype=jnp.bool_),
        (),
    )

    def valid_branch(operand: tuple[Array, Array]) -> Array:
        value, _ = operand
        return value

    def invalid_branch(operand: tuple[Array, Array]) -> Array:
        value, runtime_predicate = operand

        def raise_if_false(concrete_predicate: Any) -> None:
            if not bool(concrete_predicate):
                raise OverflowError(
                    "causal-map step_count is saturated; no further transition "
                    "can preserve the finite-state timestamp contract"
                )

        jax.debug.callback(
            raise_if_false,
            runtime_predicate,
            ordered=True,
        )
        return value

    return cast(
        Array,
        jax.lax.cond(
            predicate,
            valid_branch,
            invalid_branch,
            (step_count, predicate),
        ),
    )


def _increment_retry_count(
    retry_count: Array,
    config: CausalMapForagerConfig,
) -> Array:
    """Increment a retry exponent without overflowing its int32 storage."""
    if config.maximum_retry_exponent == 0:
        return jnp.zeros_like(retry_count, dtype=jnp.int32)
    ceiling_minus_one = jnp.asarray(
        config.maximum_retry_exponent - 1,
        dtype=jnp.int32,
    )
    return jnp.minimum(jnp.asarray(retry_count, dtype=jnp.int32), ceiling_minus_one) + 1


def _retry_delay(
    retry_count: Array,
    config: CausalMapForagerConfig,
) -> Array:
    """Return a positive, saturating exponential retry delay."""
    maximum_safe_exponent = (
        _INT32_MAX // config.initial_retry_delay
    ).bit_length() - 1
    safe_exponent = jnp.minimum(
        jnp.asarray(retry_count, dtype=jnp.int32),
        jnp.asarray(maximum_safe_exponent, dtype=jnp.int32),
    )
    shifted = jnp.left_shift(
        jnp.asarray(config.initial_retry_delay, dtype=jnp.int32),
        safe_exponent,
    )
    saturated = jnp.where(
        jnp.asarray(retry_count, dtype=jnp.int32) > maximum_safe_exponent,
        jnp.asarray(config.maximum_retry_delay, dtype=jnp.int32),
        jnp.minimum(
            shifted,
            jnp.asarray(config.maximum_retry_delay, dtype=jnp.int32),
        ),
    )
    return jnp.maximum(saturated, jnp.asarray(1, dtype=jnp.int32))


def _estimated_respawn_delay(
    state: CausalMapForagerState,
    channel: Array,
    config: CausalMapForagerConfig,
) -> Array:
    """Return a conservative delay from interval-censored online evidence.

    For each observed reappearance ``i`` the policy knows only an integer
    interval ``L_i <= T_i <= U_i`` unless absence and presence were observed on
    consecutive steps.  Each endpoint mean is stored exactly as an int32 floor
    quotient and non-negative remainder, preserving ``sum = floor * n +
    remainder`` without x64.  The scheduler deliberately starts from the exact
    outward integer ceiling of the upper endpoint mean; it is an upper bound,
    never an imputed or exact sample.  Exact consecutive observations
    additionally maintain Welford statistics, whose configured Normal safety
    quantile can only increase the conservative interval endpoint.

    This distribution-free bound may be late when the policy returns late, but
    it cannot mistake visibility duration (only a lower censoring bound) for a
    respawn draw.
    """
    safe_channel = jnp.maximum(channel, 0)
    interval_count = state.respawn_interval_count[safe_channel]

    exact_count = state.respawn_exact_count[safe_channel]
    denominator = jnp.maximum(exact_count - 1, 1)
    exact_variance = jnp.where(
        exact_count > 1,
        state.respawn_exact_m2[safe_channel]
        / denominator.astype(jnp.float32),
        0.0,
    )
    exact_quantile = state.respawn_exact_mean[safe_channel] + jnp.asarray(
        config.respawn_quantile_z,
        dtype=jnp.float32,
    ) * jnp.sqrt(jnp.maximum(exact_variance, 0.0))
    identified_upper = _saturating_add_int32(
        state.respawn_interval_upper_floor[safe_channel],
        (state.respawn_interval_upper_remainder[safe_channel] > 0).astype(
            jnp.int32
        ),
    )
    conservative_upper = jnp.maximum(
        identified_upper.astype(jnp.float32),
        jnp.where(
            exact_count > 0,
            exact_quantile,
            0.0,
        ),
    )
    estimate = jnp.ceil(
        jnp.asarray(config.respawn_safety_factor, dtype=jnp.float32)
        * conservative_upper
    )
    # Do not cast a float32 value above its largest int32-safe representation:
    # the mathematical int32 maximum rounds upward in float32 and can wrap
    # negative on conversion.  Select the exact configured integer ceiling
    # without conversion whenever the estimate reaches it or becomes nonfinite.
    configured_ceiling_float = jnp.asarray(
        config.maximum_respawn_delay,
        dtype=jnp.float32,
    )
    use_ceiling = (~jnp.isfinite(estimate)) | (estimate >= configured_ceiling_float)
    safe_estimate = jnp.minimum(
        estimate,
        jnp.asarray(_MAX_SAFE_FLOAT32_INT32, dtype=jnp.float32),
    ).astype(jnp.int32)
    clipped = jnp.where(
        use_ceiling,
        jnp.asarray(config.maximum_respawn_delay, dtype=jnp.int32),
        jnp.clip(
            safe_estimate,
            config.minimum_respawn_delay,
            config.maximum_respawn_delay,
        ),
    )
    # Float32 cannot represent every int32.  Retain the exact rational
    # endpoint's outward integer ceiling independently of the float32 safety
    # calculation, qualified only by the configured operational ceiling.
    identified_clipped = jnp.clip(
        identified_upper,
        config.minimum_respawn_delay,
        config.maximum_respawn_delay,
    )
    clipped = jnp.maximum(clipped, identified_clipped)
    return jnp.where(
        interval_count >= config.minimum_respawn_samples,
        clipped,
        jnp.asarray(config.initial_retry_delay, dtype=jnp.int32),
    )


def _rational_mean_float32(
    floor: Array,
    remainder: Array,
    count: Array,
) -> Array:
    """Evaluate a quotient/remainder mean under one explicit float32 contract."""
    denominator = jnp.maximum(jnp.asarray(count, dtype=jnp.int32), 1)
    return jnp.asarray(floor, dtype=jnp.float32) + (
        jnp.asarray(remainder, dtype=jnp.float32)
        / denominator.astype(jnp.float32)
    )


def _add_exact_mean_sample(
    floor: Array,
    remainder: Array,
    sample: Array,
    next_count: Array,
) -> tuple[Array, Array]:
    """Add one positive int32 sample to an exact quotient/remainder mean."""
    # Let the old exact sum be q*n+r.  The new quotient correction is
    # floor((r + x - q) / (n + 1)).  Split positive and negative cases so no
    # signed intermediate can overflow; r+x fits exactly in uint32.
    floor_minus_sample = floor - sample
    nonnegative = (sample >= floor) | (
        (sample < floor) & (remainder >= floor_minus_sample)
    )
    positive_numerator = (
        remainder.astype(jnp.uint32)
        + sample.astype(jnp.uint32)
        - floor.astype(jnp.uint32)
    )
    unsigned_count = next_count.astype(jnp.uint32)
    positive_adjustment = (positive_numerator // unsigned_count).astype(jnp.int32)
    positive_remainder = (positive_numerator % unsigned_count).astype(jnp.int32)

    negative_magnitude = jnp.maximum(
        jnp.maximum(floor_minus_sample, 0) - remainder,
        0,
    )
    negative_adjustment = jnp.where(
        negative_magnitude > 0,
        1 + (negative_magnitude - 1) // next_count,
        0,
    )
    negative_modulus = negative_magnitude % next_count
    negative_remainder = jnp.where(
        negative_modulus == 0,
        0,
        next_count - negative_modulus,
    )
    return (
        jnp.where(
            nonnegative,
            floor + positive_adjustment,
            floor - negative_adjustment,
        ),
        jnp.where(nonnegative, positive_remainder, negative_remainder),
    )


def _merge_channel_interval_bounds(
    count: Array,
    lower_floor: Array,
    lower_remainder: Array,
    upper_floor: Array,
    upper_remainder: Array,
    channels: Array,
    lower_bounds: Array,
    upper_bounds: Array,
    sample_mask: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Merge interval-censored samples without imputing either endpoint.

    If every latent delay obeys ``L_i <= T_i <= U_i``, arithmetic-mean
    monotonicity guarantees ``mean(L) <= mean(T) <= mean(U)``.  Endpoint sums
    are retained exactly as ``floor * count + remainder`` using only int32 and
    uint32 intermediates.  Samples are streamed in aperture order so count
    saturation drops only the suffix that cannot be represented.
    """
    count = jnp.asarray(count, dtype=jnp.int32)
    lower_floor = jnp.asarray(lower_floor, dtype=jnp.int32)
    lower_remainder = jnp.asarray(lower_remainder, dtype=jnp.int32)
    upper_floor = jnp.asarray(upper_floor, dtype=jnp.int32)
    upper_remainder = jnp.asarray(upper_remainder, dtype=jnp.int32)
    channels = jnp.asarray(channels, dtype=jnp.int32)
    lower_bounds = jnp.asarray(lower_bounds, dtype=jnp.int32)
    upper_bounds = jnp.asarray(upper_bounds, dtype=jnp.int32)
    sample_mask = jnp.asarray(sample_mask, dtype=jnp.bool_)
    channel_count = count.shape[0]

    def merge_one(
        carry: tuple[Array, Array, Array, Array, Array],
        sample_data: tuple[Array, Array, Array, Array],
    ) -> tuple[
        tuple[Array, Array, Array, Array, Array],
        None,
    ]:
        (
            current_count,
            current_lower_floor,
            current_lower_remainder,
            current_upper_floor,
            current_upper_remainder,
        ) = carry
        channel, lower_bound, upper_bound, enabled = sample_data
        safe_channel = jnp.clip(channel, 0, channel_count - 1)
        old_count = current_count[safe_channel]
        accepted = (
            enabled
            & (channel >= 0)
            & (channel < channel_count)
            & (old_count < _INT32_MAX)
        )
        next_count = old_count + accepted.astype(jnp.int32)
        next_lower_floor, next_lower_remainder = _add_exact_mean_sample(
            current_lower_floor[safe_channel],
            current_lower_remainder[safe_channel],
            lower_bound,
            jnp.maximum(next_count, 1),
        )
        next_upper_floor, next_upper_remainder = _add_exact_mean_sample(
            current_upper_floor[safe_channel],
            current_upper_remainder[safe_channel],
            upper_bound,
            jnp.maximum(next_count, 1),
        )
        next_carry = (
            current_count.at[safe_channel].set(next_count),
            current_lower_floor.at[safe_channel].set(
                jnp.where(
                    accepted,
                    next_lower_floor,
                    current_lower_floor[safe_channel],
                )
            ),
            current_lower_remainder.at[safe_channel].set(
                jnp.where(
                    accepted,
                    next_lower_remainder,
                    current_lower_remainder[safe_channel],
                )
            ),
            current_upper_floor.at[safe_channel].set(
                jnp.where(
                    accepted,
                    next_upper_floor,
                    current_upper_floor[safe_channel],
                )
            ),
            current_upper_remainder.at[safe_channel].set(
                jnp.where(
                    accepted,
                    next_upper_remainder,
                    current_upper_remainder[safe_channel],
                )
            ),
        )
        return next_carry, None

    merged, _ = jax.lax.scan(
        merge_one,
        (
            count,
            lower_floor,
            lower_remainder,
            upper_floor,
            upper_remainder,
        ),
        (channels, lower_bounds, upper_bounds, sample_mask),
    )
    return merged


def _merge_exact_channel_samples(
    count: Array,
    floor: Array,
    remainder: Array,
    mean: Array,
    m2: Array,
    channels: Array,
    samples: Array,
    sample_mask: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Merge integer exact delays with a canonical mean and centered M2.

    The quotient/remainder pair is the authoritative exact sample sum.  The
    float32 mean is derived from it after every merge so checkpoint and batch
    boundaries cannot select a different rounded mean.  Welford's M2 is then
    translated from its provisional center to that canonical mean via
    ``M2_b = M2_a + n * (a - b)^2``.  This keeps future Welford merges centered
    consistently even when the provisional and rational means differ by one
    float32 ULP.
    """
    count = jnp.asarray(count, dtype=jnp.int32)
    floor = jnp.asarray(floor, dtype=jnp.int32)
    remainder = jnp.asarray(remainder, dtype=jnp.int32)
    mean = jnp.asarray(mean, dtype=jnp.float32)
    m2 = jnp.asarray(m2, dtype=jnp.float32)
    channels = jnp.asarray(channels, dtype=jnp.int32)
    samples = jnp.asarray(samples, dtype=jnp.int32)
    sample_mask = jnp.asarray(sample_mask, dtype=jnp.bool_)
    channel_count = count.shape[0]

    def merge_one(
        carry: tuple[Array, Array, Array, Array, Array],
        sample_data: tuple[Array, Array, Array],
    ) -> tuple[tuple[Array, Array, Array, Array, Array], None]:
        (
            current_count,
            current_floor,
            current_remainder,
            current_mean,
            current_m2,
        ) = carry
        channel, sample, enabled = sample_data
        safe_channel = jnp.clip(channel, 0, channel_count - 1)
        old_count = current_count[safe_channel]
        accepted = (
            enabled
            & (channel >= 0)
            & (channel < channel_count)
            & (old_count < _INT32_MAX)
        )
        next_count = old_count + accepted.astype(jnp.int32)
        denominator = jnp.maximum(next_count, 1)
        next_floor, next_remainder = _add_exact_mean_sample(
            current_floor[safe_channel],
            current_remainder[safe_channel],
            sample,
            denominator,
        )
        sample_float = sample.astype(jnp.float32)
        denominator_float = denominator.astype(jnp.float32)
        delta = sample_float - current_mean[safe_channel]
        provisional_mean = current_mean[safe_channel] + delta / denominator_float
        delta2 = sample_float - provisional_mean
        provisional_m2 = jnp.maximum(
            current_m2[safe_channel] + delta * delta2,
            0.0,
        )
        canonical_mean = _rational_mean_float32(
            next_floor,
            next_remainder,
            denominator,
        )
        center_delta = provisional_mean - canonical_mean
        centered_m2 = provisional_m2 + (
            denominator_float * jnp.square(center_delta)
        )
        return (
            (
                current_count.at[safe_channel].set(next_count),
                current_floor.at[safe_channel].set(
                    jnp.where(accepted, next_floor, current_floor[safe_channel])
                ),
                current_remainder.at[safe_channel].set(
                    jnp.where(
                        accepted,
                        next_remainder,
                        current_remainder[safe_channel],
                    )
                ),
                current_mean.at[safe_channel].set(
                    jnp.where(accepted, canonical_mean, current_mean[safe_channel])
                ),
                current_m2.at[safe_channel].set(
                    jnp.where(accepted, centered_m2, current_m2[safe_channel])
                ),
            ),
            None,
        )

    merged, _ = jax.lax.scan(
        merge_one,
        (count, floor, remainder, mean, m2),
        (channels, samples, sample_mask),
    )
    return merged


def _integrate_observation(
    state: CausalMapForagerState,
    observation: Any,
    config: CausalMapForagerConfig,
) -> tuple[CausalMapForagerState, Array, Array]:
    """Project an egocentric image into the relative map and learn reappearance."""
    image = _image(observation)
    _validate_static_observation_shape(image, config)
    aperture_h, aperture_w, _ = image.shape
    center_y, center_x = aperture_h // 2, aperture_w // 2
    active = jnp.sum(image, axis=-1) > 0.5
    channels = jnp.argmax(image, axis=-1).astype(jnp.int32)
    rows, cols = jnp.meshgrid(
        jnp.arange(aperture_h, dtype=jnp.int32),
        jnp.arange(aperture_w, dtype=jnp.int32),
        indexing="ij",
    )
    ys = jnp.mod(
        state.position[1] + rows - center_y,
        config.height,
    ).reshape(-1)
    xs = jnp.mod(
        state.position[0] + cols - center_x,
        config.width,
    ).reshape(-1)
    flat_active = active.reshape(-1)
    flat_channels = channels.reshape(-1)
    old_channel = state.cell_channel[ys, xs]
    old_collection = state.cell_collection_step[ys, xs]
    old_ready = state.cell_ready_step[ys, xs]
    old_retry = state.cell_retry_count[ys, xs]
    old_last_absent = state.cell_last_absent_step[ys, xs]
    reappeared = (
        flat_active
        & (old_collection >= 0)
        & (old_collection < state.step_count)
    )
    upper_elapsed = state.step_count - old_collection
    lower_elapsed = jnp.maximum(
        old_last_absent + 1 - old_collection,
        1,
    ).astype(jnp.int32)
    interval_width = upper_elapsed - lower_elapsed
    exact_reappearance = reappeared & (
        interval_width
        <= jnp.asarray(config.maximum_exact_interval_width, dtype=jnp.int32)
    )
    sample_channels = jnp.where(old_channel >= 0, old_channel, flat_channels)

    # Retain both endpoints of every causal observation interval.  Neither
    # endpoint is imputed as a respawn draw.  Exact consecutive observations
    # additionally enter their own exact-sum/Welford population.
    (
        respawn_interval_count,
        respawn_interval_lower_floor,
        respawn_interval_lower_remainder,
        respawn_interval_upper_floor,
        respawn_interval_upper_remainder,
    ) = _merge_channel_interval_bounds(
        state.respawn_interval_count,
        state.respawn_interval_lower_floor,
        state.respawn_interval_lower_remainder,
        state.respawn_interval_upper_floor,
        state.respawn_interval_upper_remainder,
        sample_channels,
        lower_elapsed,
        upper_elapsed,
        reappeared,
    )
    (
        respawn_exact_count,
        respawn_exact_floor,
        respawn_exact_remainder,
        respawn_exact_mean,
        respawn_exact_m2,
    ) = _merge_exact_channel_samples(
        state.respawn_exact_count,
        state.respawn_exact_floor,
        state.respawn_exact_remainder,
        state.respawn_exact_mean,
        state.respawn_exact_m2,
        sample_channels,
        upper_elapsed,
        exact_reappearance,
    )

    visible_retry_miss = (
        ~flat_active
        & (old_collection >= 0)
        & (old_ready >= 0)
        & (state.step_count >= old_ready)
    )
    next_retry = _increment_retry_count(old_retry, config)
    retry_delay = _retry_delay(next_retry, config)
    missed_ready = _saturating_add_int32(state.step_count, retry_delay)

    cell_channel = state.cell_channel.at[ys, xs].set(
        jnp.where(flat_active, flat_channels, old_channel)
    )
    cell_active = state.cell_active.at[ys, xs].set(flat_active)
    collection_step = state.cell_collection_step.at[ys, xs].set(
        jnp.where(reappeared, -1, old_collection)
    )
    ready_step = state.cell_ready_step.at[ys, xs].set(
        jnp.where(
            reappeared,
            -1,
            jnp.where(visible_retry_miss, missed_ready, old_ready),
        )
    )
    retry_count = state.cell_retry_count.at[ys, xs].set(
        jnp.where(
            reappeared,
            0,
            jnp.where(visible_retry_miss, next_retry, old_retry),
        )
    )
    last_seen = state.cell_last_seen_step.at[ys, xs].set(state.step_count)
    last_absent = state.cell_last_absent_step.at[ys, xs].set(
        jnp.where(~flat_active, state.step_count, old_last_absent)
    )
    learned_count = jnp.sum(reappeared, dtype=jnp.int32)
    visible_retry_count = jnp.sum(visible_retry_miss, dtype=jnp.int32)
    return state._replace(
        cell_channel=cell_channel,
        cell_active=cell_active,
        cell_collection_step=collection_step,
        cell_ready_step=ready_step,
        cell_retry_count=retry_count,
        cell_last_seen_step=last_seen,
        cell_last_absent_step=last_absent,
        respawn_interval_count=respawn_interval_count,
        respawn_interval_lower_floor=respawn_interval_lower_floor,
        respawn_interval_lower_remainder=respawn_interval_lower_remainder,
        respawn_interval_upper_floor=respawn_interval_upper_floor,
        respawn_interval_upper_remainder=respawn_interval_upper_remainder,
        respawn_exact_count=respawn_exact_count,
        respawn_exact_floor=respawn_exact_floor,
        respawn_exact_remainder=respawn_exact_remainder,
        respawn_exact_mean=respawn_exact_mean,
        respawn_exact_m2=respawn_exact_m2,
    ), learned_count, visible_retry_count


def _channel_values(
    state: CausalMapForagerState,
    config: CausalMapForagerConfig,
) -> Array:
    return jnp.where(
        state.reward_count > 0,
        state.reward_sum / jnp.maximum(state.reward_count, 1).astype(jnp.float32),
        jnp.asarray(config.optimistic_unknown_reward, dtype=jnp.float32),
    )


def _negative_channels(
    state: CausalMapForagerState,
    config: CausalMapForagerConfig,
) -> Array:
    return (state.reward_count > 0) & (
        _channel_values(state, config) < config.negative_reward_threshold
    )


def _cost_aware_route_grid(
    source: Array,
    negative_cells: Array,
    config: CausalMapForagerConfig,
) -> tuple[Array, Array]:
    """Return exact minimum-negative-entry counts and shortest tie paths.

    Objects in the stationary public Forager task are traversable, including
    collectable deathcaps.  Encode one path as ``negative_entries * base +
    steps``, where ``base`` exceeds every simple-path length.  Ordinary integer
    shortest-path relaxation then minimizes negative entries lexicographically
    before distance without treating a learned-negative cell as a wall.

    The source is never charged: an entry cost belongs to the destination of an
    action, and the policy is already standing at ``source``.  At most ``V - 1``
    edges are needed by a lexicographically optimal simple path.  ``V`` bounded
    relaxations preserve the same finite fail-safe used by the safe-only grid.
    """
    cell_count = config.height * config.width
    base = jnp.asarray(cell_count + 1, dtype=jnp.int32)
    infinity = jnp.asarray((cell_count + 1) * (cell_count + 2), dtype=jnp.int32)
    source_x, source_y = source[0], source[1]
    entry_cost = (
        negative_cells.astype(jnp.int32) * base
        + jnp.asarray(1, dtype=jnp.int32)
    )
    encoded = jnp.full(config.world_shape, infinity, dtype=jnp.int32)
    encoded = encoded.at[source_y, source_x].set(0)

    def relax(current: Array) -> Array:
        neighbor_minimum = jnp.minimum(
            jnp.minimum(
                jnp.roll(current, 1, axis=0),
                jnp.roll(current, -1, axis=0),
            ),
            jnp.minimum(
                jnp.roll(current, 1, axis=1),
                jnp.roll(current, -1, axis=1),
            ),
        )
        return jnp.minimum(current, neighbor_minimum + entry_cost)

    def not_fixed(carry: tuple[Array, Array, Array]) -> Array:
        _, changed, iteration = carry
        return changed & (iteration < cell_count)

    def relax_once(
        carry: tuple[Array, Array, Array],
    ) -> tuple[Array, Array, Array]:
        current, _, iteration = carry
        updated = relax(current)
        return updated, jnp.any(updated != current), iteration + 1

    encoded, _, _ = jax.lax.while_loop(
        not_fixed,
        relax_once,
        (
            encoded,
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    return encoded // base, encoded % base


def _choose_action(
    state: CausalMapForagerState,
    config: CausalMapForagerConfig,
) -> tuple[CausalMapForagerState, Array]:
    """Choose a safe one-step move toward the best learned available cell."""
    key, target_key, action_key, exploration_key = jr.split(state.rng_key, 4)
    channel_values = _channel_values(state, config)
    negative_channels = _negative_channels(state, config)
    safe_channel = jnp.maximum(state.cell_channel, 0)
    known = state.cell_channel >= 0
    negative_cells = known & negative_channels[safe_channel]
    route_negative_entries, route_steps = _cost_aware_route_grid(
        state.position,
        negative_cells,
        config,
    )
    path_infinity = jnp.asarray(
        config.height * config.width + 1,
        dtype=jnp.int32,
    )
    reachable = route_steps < path_infinity
    readiness_step = jnp.broadcast_to(
        state.step_count,
        state.cell_ready_step.shape,
    )
    if config.arrival_aware_readiness:
        # Foragax computes movement/collection reward from the pre-transition
        # object grid, then decrements respawn timers and emits the returned
        # observation.  A target at graph distance d is therefore collectible
        # on arrival only if it is ready in the decision state after d - 1
        # transitions.  Saturation prevents a near-lifetime-end wraparound from
        # making an otherwise due target look indefinitely unavailable.
        pre_entry_steps = jnp.maximum(
            route_steps - jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        readiness_step = _saturating_add_int32(
            state.step_count,
            pre_entry_steps,
        )
    predicted_ready = (
        (state.cell_collection_step >= 0)
        & (state.cell_ready_step >= 0)
        & (readiness_step >= state.cell_ready_step)
    )
    available = state.cell_active | predicted_ready
    candidate = known & available & ~negative_cells & reachable
    distances = route_steps.astype(jnp.float32)
    # The public stationary task has one learned-negative deathcap channel.
    # Price every necessary negative entry at the worst empirical negative
    # channel mean.  Cap the per-entry price so the bounded V-cell route cannot
    # overflow float32 scoring even for a corrupted but finite learned mean.
    maximum_entry_penalty = jnp.asarray(
        _FLOAT32_SCORE_HEADROOM / (config.height * config.width),
        dtype=jnp.float32,
    )
    negative_entry_penalty = jnp.minimum(
        jnp.max(
            jnp.where(
                negative_channels,
                -channel_values,
                jnp.asarray(0.0, dtype=jnp.float32),
            )
        ),
        maximum_entry_penalty,
    )
    route_negative_cost = (
        route_negative_entries.astype(jnp.float32) * negative_entry_penalty
    )
    target_noise = jr.uniform(
        target_key,
        state.cell_channel.shape,
        dtype=jnp.float32,
    ) * jnp.asarray(config.tie_break_scale, dtype=jnp.float32)
    cell_values = channel_values[safe_channel]
    target_scores = (
        cell_values
        - jnp.asarray(config.distance_cost, dtype=jnp.float32) * distances
        - route_negative_cost
        - jnp.asarray(config.retry_penalty, dtype=jnp.float32)
        * state.cell_retry_count.astype(jnp.float32)
        + target_noise
    )
    finite_candidate = candidate & jnp.isfinite(target_scores)
    target_scores = jnp.where(finite_candidate, target_scores, -jnp.inf)
    has_exploitation_target = jnp.any(finite_candidate)
    exploitation_flat = jnp.argmax(target_scores.reshape(-1))
    exploitation_target = jnp.asarray(
        (
            exploitation_flat % config.width,
            exploitation_flat // config.width,
        ),
        dtype=jnp.int32,
    )

    unobserved = state.cell_last_seen_step < 0
    unknown_reachable = unobserved & ~negative_cells & reachable
    coverage_scores = (
        -state.visit_count.astype(jnp.float32)
        - jnp.asarray(config.distance_cost, dtype=jnp.float32) * distances
        - route_negative_cost
        + target_noise
    )
    finite_unknown_candidate = unknown_reachable & jnp.isfinite(coverage_scores)
    unknown_scores = jnp.where(
        finite_unknown_candidate,
        coverage_scores,
        -jnp.inf,
    )
    has_unknown_target = jnp.any(finite_unknown_candidate)
    unknown_flat = jnp.argmax(unknown_scores.reshape(-1))
    unknown_target = jnp.asarray(
        (unknown_flat % config.width, unknown_flat // config.width),
        dtype=jnp.int32,
    )
    # Generic coverage is not an exploration target.  Retain it solely for the
    # no-exploitation/no-unknown fail-safe so a completed map cannot make the
    # exploration coin divert an otherwise useful exploitation trajectory.
    safe_reachable = ~negative_cells & reachable
    finite_failsafe_candidate = safe_reachable & jnp.isfinite(coverage_scores)
    failsafe_scores = jnp.where(
        finite_failsafe_candidate,
        coverage_scores,
        -jnp.inf,
    )
    has_failsafe_target = jnp.any(finite_failsafe_candidate)
    failsafe_flat = jnp.argmax(failsafe_scores.reshape(-1))
    failsafe_target = jnp.asarray(
        (failsafe_flat % config.width, failsafe_flat // config.width),
        dtype=jnp.int32,
    )
    explore = jr.uniform(exploration_key, (), dtype=jnp.float32) < jnp.asarray(
        config.exploration_probability,
        dtype=jnp.float32,
    )
    use_unknown = has_unknown_target & (explore | ~has_exploitation_target)
    use_failsafe = ~has_unknown_target & ~has_exploitation_target & has_failsafe_target
    target = jnp.where(
        use_unknown,
        unknown_target,
        jnp.where(use_failsafe, failsafe_target, exploitation_target),
    )
    has_target = jnp.where(
        use_unknown,
        has_unknown_target,
        has_exploitation_target | use_failsafe,
    )

    neighbor_positions = jnp.mod(
        state.position[None, :] + _DIRECTIONS,
        jnp.asarray((config.width, config.height), dtype=jnp.int32),
    )
    neighbor_x = neighbor_positions[:, 0]
    neighbor_y = neighbor_positions[:, 1]
    neighbor_negative = negative_cells[neighbor_y, neighbor_x]
    all_neighbors_negative = jnp.all(neighbor_negative)
    allowed = ~neighbor_negative | all_neighbors_negative

    target_negative_entries, target_distance_grid = _cost_aware_route_grid(
        target,
        negative_cells,
        config,
    )
    target_negative_entries = target_negative_entries[neighbor_y, neighbor_x]
    target_distances_int = target_distance_grid[neighbor_y, neighbor_x]
    target_distances = target_distances_int.astype(jnp.float32)
    route_available = jnp.any(target_distances_int < path_infinity)
    minimum_route_negative_entries = jnp.min(
        jnp.where(
            target_distances_int < path_infinity,
            target_negative_entries,
            jnp.asarray(path_infinity, dtype=jnp.int32),
        )
    )
    pursuit_allowed = (
        (target_distances_int < path_infinity)
        & (target_negative_entries == minimum_route_negative_entries)
    )
    neighbor_visits = state.visit_count[neighbor_y, neighbor_x].astype(jnp.float32)
    reverse_action = jnp.mod(state.last_action + 2, 4)
    reverse_cost = (
        (state.last_action >= 0)
        & (jnp.arange(4, dtype=jnp.int32) == reverse_action)
    ).astype(jnp.float32) * jnp.asarray(
        config.reverse_action_penalty,
        dtype=jnp.float32,
    )
    action_noise = jr.uniform(action_key, (4,), dtype=jnp.float32) * jnp.asarray(
        config.tie_break_scale,
        dtype=jnp.float32,
    )
    pursuit_cost = (
        target_distances
        + jnp.asarray(config.visit_penalty, dtype=jnp.float32)
        * jnp.log1p(neighbor_visits)
        + reverse_cost
        + action_noise
    )
    exploration_cost = (
        neighbor_visits
        + reverse_cost
        + action_noise
    )
    pursuing = has_target & route_available
    costs = jnp.where(pursuing, pursuit_cost, exploration_cost)
    selected_allowed = jnp.where(pursuing, pursuit_allowed, allowed)
    finite_allowed = selected_allowed & jnp.isfinite(costs)
    has_finite_action = jnp.any(finite_allowed)
    finite_costs = jnp.where(finite_allowed, costs, jnp.inf)
    # ``allowed`` is non-empty by construction: if every neighbor is learned
    # negative, the explicit escape fallback allows all four.  The index cost
    # is deterministic and finite even if corrupted learned values make every
    # ordinary planner cost non-finite.
    fallback_costs = jnp.where(
        selected_allowed,
        jnp.arange(4, dtype=jnp.float32),
        jnp.inf,
    )
    action = jnp.argmin(
        jnp.where(has_finite_action, finite_costs, fallback_costs)
    ).astype(jnp.int32)

    destination = neighbor_positions[action]
    destination_y, destination_x = destination[1], destination[0]
    destination_channel = state.cell_channel[destination_y, destination_x]
    destination_predicted = predicted_ready[destination_y, destination_x]
    destination_expected_active = (
        state.cell_active[destination_y, destination_x] | destination_predicted
    ) & (destination_channel >= 0)
    return state._replace(
        rng_key=key,
        last_action=action,
        last_target_channel=destination_channel,
        last_target_position=destination,
        last_target_expected_active=destination_expected_active,
    ), action


def causal_map_start(
    observation: Any,
    config: CausalMapForagerConfig,
    seed: int | Array,
) -> tuple[CausalMapForagerState, Array]:
    """Initialize from the first raw observation and choose the first action."""
    validated_seed = _validated_seed(seed)
    image = _validated_observation_image(observation, config)
    state = _empty_state(int(image.shape[-1]), config, validated_seed)
    state = state._replace(
        visit_count=state.visit_count.at[0, 0].set(1),
    )
    state, _, _ = _integrate_observation(state, image, config)
    return _choose_action(state, config)


def causal_map_step(
    state: CausalMapForagerState,
    reward: Any,
    observation: Any,
    config: CausalMapForagerConfig,
) -> tuple[CausalMapForagerState, Array, CausalMapStepDiagnostics]:
    """Consume one ordinary transition and choose the next action.

    The only transition inputs are the reward, current observation, and state
    generated from the policy's own prior inputs/actions.
    """
    raw_reward = jnp.asarray(reward)
    if raw_reward.ndim != 0 or not (
        jnp.issubdtype(raw_reward.dtype, jnp.integer)
        or jnp.issubdtype(raw_reward.dtype, jnp.floating)
    ):
        raise ValueError("causal-map reward must be one finite real scalar")
    reward = jnp.asarray(raw_reward, dtype=jnp.float32)
    reward = _runtime_require(
        jnp.isfinite(reward),
        reward,
        message="causal-map reward must be one finite float32 scalar",
    )
    image = _validated_observation_image(observation, config)
    if image.shape[-1] != state.reward_sum.shape[0]:
        raise ValueError(
            "causal-map observation channel count changed during the continuing run"
        )
    mode_checked_step_count = _runtime_require(
        state.jax_threefry_partitionable
        == jnp.asarray(_threefry_partitionable_mode(), dtype=jnp.bool_),
        state.step_count,
        message=(
            "causal-map state jax_threefry_partitionable mode does not match runtime"
        ),
    )
    checked_step_count = _require_unsaturated_step_count(mode_checked_step_count)
    position = jnp.mod(
        state.position + _DIRECTIONS[state.last_action],
        jnp.asarray((config.width, config.height), dtype=jnp.int32),
    )
    step_count = _saturating_add_int32(
        checked_step_count,
        jnp.asarray(1, dtype=jnp.int32),
    )
    next_visit_count = _saturating_add_int32(
        state.visit_count[position[1], position[0]],
        jnp.asarray(1, dtype=jnp.int32),
    )
    visit_count = state.visit_count.at[position[1], position[0]].set(
        next_visit_count
    )
    state = state._replace(
        step_count=step_count,
        position=position,
        visit_count=visit_count,
    )

    valid_channel = state.last_target_channel >= 0
    observed_reward = (
        valid_channel
        & jnp.isfinite(reward)
        & (jnp.abs(reward) > config.reward_observation_epsilon)
    )
    safe_channel = jnp.maximum(state.last_target_channel, 0)
    reward_sum = state.reward_sum.at[safe_channel].add(
        jnp.where(observed_reward, reward, 0.0)
    )
    reward_sum = _runtime_require(
        jnp.all(jnp.isfinite(reward_sum)),
        reward_sum,
        message="causal-map reward accumulation must remain finite",
    )
    next_reward_count = _saturating_add_int32(
        state.reward_count[safe_channel],
        observed_reward.astype(jnp.int32),
    )
    reward_count = state.reward_count.at[safe_channel].set(
        next_reward_count
    )

    # A reward collected from a cell that was still pending proves that the
    # previous object reappeared before it was immediately collected again.
    # Capture its causal interval before overwriting the old collection time.
    current_y, current_x = position[1], position[0]
    prior_collection = state.cell_collection_step[current_y, current_x]
    prior_last_absent = state.cell_last_absent_step[current_y, current_x]
    immediate_reappearance = observed_reward & (prior_collection >= 0)
    immediate_upper = step_count - prior_collection
    immediate_lower = jnp.maximum(
        prior_last_absent + 1 - prior_collection,
        1,
    ).astype(jnp.int32)
    immediate_exact = immediate_reappearance & (
        immediate_upper - immediate_lower
        <= jnp.asarray(config.maximum_exact_interval_width, dtype=jnp.int32)
    )
    sample_channel = jnp.reshape(safe_channel, (1,))
    sample_lower = jnp.reshape(immediate_lower, (1,))
    sample_upper = jnp.reshape(immediate_upper, (1,))
    sample_mask = jnp.reshape(immediate_reappearance, (1,))
    exact_mask = jnp.reshape(immediate_exact, (1,))
    (
        respawn_interval_count,
        respawn_interval_lower_floor,
        respawn_interval_lower_remainder,
        respawn_interval_upper_floor,
        respawn_interval_upper_remainder,
    ) = _merge_channel_interval_bounds(
        state.respawn_interval_count,
        state.respawn_interval_lower_floor,
        state.respawn_interval_lower_remainder,
        state.respawn_interval_upper_floor,
        state.respawn_interval_upper_remainder,
        sample_channel,
        sample_lower,
        sample_upper,
        sample_mask,
    )
    (
        respawn_exact_count,
        respawn_exact_floor,
        respawn_exact_remainder,
        respawn_exact_mean,
        respawn_exact_m2,
    ) = _merge_exact_channel_samples(
        state.respawn_exact_count,
        state.respawn_exact_floor,
        state.respawn_exact_remainder,
        state.respawn_exact_mean,
        state.respawn_exact_m2,
        sample_channel,
        sample_upper,
        exact_mask,
    )
    state = state._replace(
        respawn_interval_count=respawn_interval_count,
        respawn_interval_lower_floor=respawn_interval_lower_floor,
        respawn_interval_lower_remainder=respawn_interval_lower_remainder,
        respawn_interval_upper_floor=respawn_interval_upper_floor,
        respawn_interval_upper_remainder=respawn_interval_upper_remainder,
        respawn_exact_count=respawn_exact_count,
        respawn_exact_floor=respawn_exact_floor,
        respawn_exact_remainder=respawn_exact_remainder,
        respawn_exact_mean=respawn_exact_mean,
        respawn_exact_m2=respawn_exact_m2,
    )

    learned_delay = _estimated_respawn_delay(state, safe_channel, config)
    cell_active = state.cell_active.at[current_y, current_x].set(
        jnp.where(observed_reward, False, state.cell_active[current_y, current_x])
    )
    collection_step = state.cell_collection_step.at[current_y, current_x].set(
        jnp.where(
            observed_reward,
            step_count,
            state.cell_collection_step[current_y, current_x],
        )
    )
    ready_step = state.cell_ready_step.at[current_y, current_x].set(
        jnp.where(
            observed_reward,
            _saturating_add_int32(step_count, learned_delay),
            state.cell_ready_step[current_y, current_x],
        )
    )
    retry_count = state.cell_retry_count.at[current_y, current_x].set(
        jnp.where(observed_reward, 0, state.cell_retry_count[current_y, current_x])
    )
    state = state._replace(
        reward_sum=reward_sum,
        reward_count=reward_count,
        cell_active=cell_active,
        cell_collection_step=collection_step,
        cell_ready_step=ready_step,
        cell_retry_count=retry_count,
    )
    state, visible_learned_respawn, visible_retry_count = _integrate_observation(
        state,
        image,
        config,
    )
    learned_respawn = _saturating_add_int32(
        immediate_reappearance.astype(jnp.int32),
        visible_learned_respawn,
    )
    retry_miss = visible_retry_count > 0
    state, action = _choose_action(state, config)
    negative_channels = _negative_channels(state, config)
    known_channels = jnp.maximum(state.cell_channel, 0)
    diagnostics = CausalMapStepDiagnostics(
        learned_reward=observed_reward,
        learned_respawn=learned_respawn,
        retry_miss=retry_miss,
        known_cells=jnp.sum(state.cell_channel >= 0, dtype=jnp.int32),
        known_negative_cells=jnp.sum(
            (state.cell_channel >= 0) & negative_channels[known_channels],
            dtype=jnp.int32,
        ),
    )
    return state, action, diagnostics


def validate_causal_map_state(
    state: CausalMapForagerState,
    config: CausalMapForagerConfig,
    *,
    observation_channels: int | None = None,
) -> None:
    """Validate exact dtype, shape, domain, and cross-field state invariants."""
    if not isinstance(state, CausalMapForagerState):
        raise ValueError("state must be a CausalMapForagerState")
    if not isinstance(config, CausalMapForagerConfig):
        raise ValueError("config must be a CausalMapForagerConfig")
    arrays: dict[str, np.ndarray[Any, Any]] = {}
    for name, value in state._asdict().items():
        if name == "rng_key":
            try:
                typed_key = jnp.issubdtype(
                    value.dtype,
                    jax.dtypes.prng_key,
                )
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError("rng_key must be a typed JAX PRNG key") from exc
            if not typed_key:
                raise ValueError("rng_key must be a typed JAX PRNG key")
            try:
                array = np.asarray(jr.key_data(value))
                impl = _prng_impl_name(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("rng_key must be a valid JAX PRNG key") from exc
            host_impl = _require_exact_str("impl", impl)
            if host_impl != _CAUSAL_MAP_PRNG_IMPL:
                raise ValueError(
                    "rng_key must use the causal-map PRNG implementation "
                    f"'{_CAUSAL_MAP_PRNG_IMPL}'"
                )
            if array.shape != (2,) or array.dtype != np.dtype(np.uint32):
                raise ValueError("rng_key must contain exactly two uint32 words")
        else:
            array = np.asarray(value)
            expected_dtype = np.dtype(
                np.uint32
                if name in _STATE_UINT32_FIELDS
                else np.int32
                if name in _STATE_INTEGER_FIELDS
                else np.bool_
                if name in _STATE_BOOLEAN_FIELDS
                else np.float32
            )
            if array.dtype != expected_dtype:
                raise ValueError(
                    f"{name} must have dtype {expected_dtype}, found {array.dtype}"
                )
        arrays[name] = array

    scalar_fields = (
        "step_count",
        "initial_seed",
        "last_action",
        "last_target_channel",
        "last_target_expected_active",
        "jax_threefry_partitionable",
    )
    for name in scalar_fields:
        if arrays[name].shape != ():
            raise ValueError(f"{name} must be a scalar")
    map_shape = config.world_shape
    map_fields = (
        "cell_channel",
        "cell_active",
        "cell_collection_step",
        "cell_ready_step",
        "cell_retry_count",
        "cell_last_seen_step",
        "cell_last_absent_step",
        "visit_count",
    )
    for name in map_fields:
        if arrays[name].shape != map_shape:
            raise ValueError(f"{name} must have shape {map_shape}")
    channel_fields = (
        "reward_sum",
        "reward_count",
        "respawn_interval_count",
        "respawn_interval_lower_floor",
        "respawn_interval_lower_remainder",
        "respawn_interval_upper_floor",
        "respawn_interval_upper_remainder",
        "respawn_exact_count",
        "respawn_exact_floor",
        "respawn_exact_remainder",
        "respawn_exact_mean",
        "respawn_exact_m2",
    )
    if arrays["reward_sum"].ndim != 1:
        raise ValueError("reward_sum must be one-dimensional")
    channel_count = arrays["reward_sum"].shape[0]
    if channel_count < 1:
        raise ValueError("state must contain at least one observation channel")
    if observation_channels is not None and (
        type(observation_channels) not in _EXACT_INTEGER_TYPES
        or observation_channels < 1
    ):
        raise ValueError("observation_channels must be a positive integer")
    if observation_channels is not None and channel_count != int(observation_channels):
        raise ValueError("state channel count does not match observation")
    for name in channel_fields:
        if arrays[name].shape != (channel_count,):
            raise ValueError(f"{name} must have shape ({channel_count},)")
    if arrays["position"].shape != (2,) or arrays["last_target_position"].shape != (2,):
        raise ValueError("position fields must have shape (2,)")
    for name in ("position", "last_target_position"):
        x, y = (int(value) for value in arrays[name])
        if not (0 <= x < config.width and 0 <= y < config.height):
            raise ValueError(f"{name} lies outside configured toroidal world")

    step_count = int(arrays["step_count"])
    if step_count < 0:
        raise ValueError("step_count must be non-negative")
    initial_seed = int(arrays["initial_seed"])
    if not 0 <= initial_seed <= np.iinfo(np.uint32).max:
        raise ValueError("initial_seed must be uint32-compatible")
    if bool(arrays["jax_threefry_partitionable"]) != _threefry_partitionable_mode():
        raise ValueError(
            "state jax_threefry_partitionable mode does not match runtime"
        )
    last_action = int(arrays["last_action"])
    if last_action not in (-1, 0, 1, 2, 3):
        raise ValueError("last_action must be -1 or one of the four public actions")
    if step_count > 0 and last_action < 0:
        raise ValueError("a non-initial state must record its previous action")
    last_target_channel = int(arrays["last_target_channel"])
    if not -1 <= last_target_channel < channel_count:
        raise ValueError("last_target_channel contains an invalid channel index")
    if bool(arrays["last_target_expected_active"]) and last_target_channel < 0:
        raise ValueError("an expected-active target must have a known channel")

    channels = arrays["cell_channel"]
    if np.any((channels < -1) | (channels >= channel_count)):
        raise ValueError("cell_channel contains an invalid channel index")
    if np.any(arrays["cell_active"] & (channels < 0)):
        raise ValueError("active cells must have a known channel")

    if (
        np.any(arrays["reward_count"] < 0)
        or np.any(arrays["respawn_interval_count"] < 0)
        or np.any(arrays["respawn_exact_count"] < 0)
    ):
        raise ValueError("learning counts must be non-negative")
    if (
        np.any(arrays["cell_retry_count"] < 0)
        or np.any(arrays["cell_retry_count"] > config.maximum_retry_exponent)
        or np.any(arrays["visit_count"] < 0)
    ):
        raise ValueError("map counts must be non-negative")
    if np.any(arrays["reward_count"] > step_count):
        raise ValueError("reward_count cannot exceed step_count")
    if int(np.sum(arrays["reward_count"], dtype=np.int64)) > step_count:
        raise ValueError("total reward_count cannot exceed step_count")
    maximum_reappearances = step_count * config.height * config.width
    for name in ("respawn_interval_count", "respawn_exact_count"):
        if np.any(arrays[name] > maximum_reappearances):
            raise ValueError(f"{name} exceeds the causal map/step bound")
        if int(np.sum(arrays[name], dtype=np.int64)) > maximum_reappearances:
            raise ValueError(f"total {name} exceeds the causal map/step bound")
    if np.any(
        arrays["respawn_exact_count"] > arrays["respawn_interval_count"]
    ):
        raise ValueError("exact respawn counts cannot exceed all reappearance counts")
    if np.any(arrays["visit_count"] > min(step_count + 1, _INT32_MAX)):
        raise ValueError("visit_count cannot exceed the number of visited states")
    total_visits = int(np.sum(arrays["visit_count"], dtype=np.int64))
    if step_count < _INT32_MAX and total_visits != step_count + 1:
        raise ValueError("visit_count total must equal step_count + 1")
    if step_count == _INT32_MAX and total_visits < _INT32_MAX:
        raise ValueError("saturated step_count requires at least int32-max visits")

    timestamp_fields = (
        "cell_collection_step",
        "cell_ready_step",
        "cell_last_seen_step",
        "cell_last_absent_step",
    )
    for name in timestamp_fields:
        if np.any(arrays[name] < -1):
            raise ValueError(f"{name} timestamps must be -1 or non-negative")
    for name in (
        "cell_collection_step",
        "cell_last_seen_step",
        "cell_last_absent_step",
    ):
        if np.any(arrays[name] > step_count):
            raise ValueError(f"{name} cannot exceed step_count")
    collection = arrays["cell_collection_step"]
    ready = arrays["cell_ready_step"]
    retry = arrays["cell_retry_count"]
    has_collection = collection >= 0
    if np.any(has_collection & (collection < 1)):
        raise ValueError("collection timestamps must begin at transition step one")
    observed_collection_steps = collection[has_collection]
    if np.unique(observed_collection_steps).size != observed_collection_steps.size:
        raise ValueError("pending collection timestamps must be unique")
    if np.any((ready >= 0) != has_collection):
        raise ValueError("collection and ready timestamps must be present together")
    if np.any(has_collection & (ready <= collection)):
        raise ValueError("ready timestamps must follow their collection timestamps")
    if np.any(has_collection & arrays["cell_active"]):
        raise ValueError("a collected cell cannot simultaneously be active")
    if np.any(has_collection & (channels < 0)):
        raise ValueError("collected cells must have a known channel")
    if np.any(~has_collection & (retry != 0)):
        raise ValueError("retry counts require a pending collected cell")
    last_seen = arrays["cell_last_seen_step"]
    last_absent = arrays["cell_last_absent_step"]
    if np.any((last_absent >= 0) & (last_seen < last_absent)):
        raise ValueError("last absence cannot follow the last observation")
    if np.any((channels >= 0) & (last_seen < 0)):
        raise ValueError("known cells must have an observation timestamp")
    position_x, position_y = (int(value) for value in arrays["position"])
    if arrays["visit_count"][position_y, position_x] < 1:
        raise ValueError("current position must have a positive visit count")
    if last_seen[position_y, position_x] != step_count:
        raise ValueError("current position must be observed at step_count")
    expected_cell_active = last_seen > last_absent
    if not np.array_equal(arrays["cell_active"], expected_cell_active):
        raise ValueError(
            "cell_active must exactly match whether the last observation "
            "followed the last absence"
        )
    if np.any(has_collection & (last_absent < collection)):
        raise ValueError("collected cells must be observed absent at collection time")
    pending_by_channel = np.bincount(
        channels[has_collection],
        minlength=channel_count,
    ).astype(np.int64)
    accounted_collections = np.minimum(
        arrays["respawn_interval_count"].astype(np.int64) + pending_by_channel,
        _INT32_MAX,
    )
    if not np.array_equal(
        accounted_collections,
        arrays["reward_count"].astype(np.int64),
    ):
        raise ValueError(
            "reward counts must equal reappearances plus pending collections"
        )

    if last_action >= 0:
        direction_x, direction_y = _DIRECTION_STEPS[last_action]
        expected_target = np.asarray(
            (
                (int(arrays["position"][0]) + direction_x) % config.width,
                (int(arrays["position"][1]) + direction_y) % config.height,
            ),
            dtype=np.int32,
        )
        if not np.array_equal(arrays["last_target_position"], expected_target):
            raise ValueError(
                "last_target_position must be the destination of last_action"
            )
        target_x, target_y = (int(value) for value in expected_target)
        expected_channel = int(channels[target_y, target_x])
        if last_target_channel != expected_channel:
            raise ValueError(
                "last_target_channel must match the destination map cell"
            )
        predicted_ready = bool(
            collection[target_y, target_x] >= 0
            and ready[target_y, target_x] >= 0
            and step_count >= ready[target_y, target_x]
        )
        expected_active = bool(
            expected_channel >= 0
            and (
                arrays["cell_active"][target_y, target_x]
                or predicted_ready
            )
        )
        if bool(arrays["last_target_expected_active"]) != expected_active:
            raise ValueError(
                "last_target_expected_active does not match destination state"
            )
    elif (
        last_target_channel != -1
        or bool(arrays["last_target_expected_active"])
        or not np.array_equal(arrays["last_target_position"], arrays["position"])
    ):
        raise ValueError("an action-free initial state cannot declare a target")

    for name in (
        "reward_sum",
        "respawn_exact_mean",
        "respawn_exact_m2",
    ):
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"{name} must remain finite")
    if np.any(arrays["respawn_exact_m2"] < 0.0):
        raise ValueError("exact respawn M2 statistics must be non-negative")
    if np.any((arrays["reward_count"] == 0) & (arrays["reward_sum"] != 0.0)):
        raise ValueError("zero-count reward statistics must have zero sums")
    interval_empty = arrays["respawn_interval_count"] == 0
    interval_fields = (
        "respawn_interval_lower_floor",
        "respawn_interval_lower_remainder",
        "respawn_interval_upper_floor",
        "respawn_interval_upper_remainder",
    )
    if any(np.any(interval_empty & (arrays[name] != 0)) for name in interval_fields):
        raise ValueError("zero-count respawn interval bounds must be zero")
    interval_populated = ~interval_empty
    if np.any(
        interval_populated & (arrays["respawn_interval_lower_floor"] < 1)
    ):
        raise ValueError("respawn interval lower floors must be at least one step")
    if np.any(
        interval_populated
        & (
            (arrays["respawn_interval_lower_remainder"] < 0)
            | (arrays["respawn_interval_upper_remainder"] < 0)
            | (
                arrays["respawn_interval_lower_remainder"]
                >= arrays["respawn_interval_count"]
            )
            | (
                arrays["respawn_interval_upper_remainder"]
                >= arrays["respawn_interval_count"]
            )
        )
    ):
        raise ValueError("respawn interval remainders must lie in [0, count)")
    lower_floor = arrays["respawn_interval_lower_floor"]
    lower_remainder = arrays["respawn_interval_lower_remainder"]
    upper_floor = arrays["respawn_interval_upper_floor"]
    upper_remainder = arrays["respawn_interval_upper_remainder"]
    if np.any(
        interval_populated
        & (
            (lower_floor > upper_floor)
            | ((lower_floor == upper_floor) & (lower_remainder > upper_remainder))
        )
    ):
        raise ValueError("respawn interval lower means cannot exceed upper means")
    if np.any(
        interval_populated
        & ((upper_floor > step_count) | ((upper_floor == step_count) & (upper_remainder > 0)))
    ):
        raise ValueError("respawn interval upper means cannot exceed step_count")

    exact_empty = arrays["respawn_exact_count"] == 0
    if np.any(
        exact_empty
        & (
            (arrays["respawn_exact_floor"] != 0)
            | (arrays["respawn_exact_remainder"] != 0)
            | (arrays["respawn_exact_mean"] != 0.0)
            | (arrays["respawn_exact_m2"] != 0.0)
        )
    ):
        raise ValueError("zero-count exact respawn statistics must be zero")
    if np.any(
        (arrays["respawn_exact_count"] <= 1)
        & (arrays["respawn_exact_m2"] != 0.0)
    ):
        raise ValueError("exact respawn M2 must be zero with at most one sample")
    exact_populated = ~exact_empty
    exact_floor = arrays["respawn_exact_floor"]
    exact_remainder = arrays["respawn_exact_remainder"]
    exact_count = arrays["respawn_exact_count"]
    if np.any(exact_populated & (exact_floor < 1)):
        raise ValueError("populated exact respawn floors must be at least one step")
    if np.any(
        exact_populated
        & ((exact_remainder < 0) | (exact_remainder >= exact_count))
    ):
        raise ValueError("exact respawn remainders must lie in [0, count)")
    if np.any(
        exact_populated
        & ((exact_floor > step_count) | ((exact_floor == step_count) & (exact_remainder > 0)))
    ):
        raise ValueError("populated exact respawn means cannot exceed step_count")
    exact_fraction = np.divide(
        exact_remainder.astype(np.float32),
        np.maximum(exact_count, 1).astype(np.float32),
        dtype=np.float32,
    )
    canonical_exact_mean = np.add(
        exact_floor.astype(np.float32),
        exact_fraction,
        dtype=np.float32,
    )
    if np.any(
        canonical_exact_mean.view(np.uint32)
        != arrays["respawn_exact_mean"].view(np.uint32)
    ):
        raise ValueError(
            "exact respawn mean must match its rational sample sum under the "
            "schema float32 contract"
        )

    exact_population = (
        arrays["respawn_exact_count"] == arrays["respawn_interval_count"]
    )
    if np.any(
        exact_population
        & (
            (lower_floor != upper_floor)
            | (lower_remainder != upper_remainder)
        )
    ):
        raise ValueError(
            "all-exact respawn intervals must have identical rational endpoints"
        )
    if np.any(
        exact_population
        & (
            (lower_floor != exact_floor)
            | (lower_remainder != exact_remainder)
        )
    ):
        raise ValueError(
            "all-exact respawn interval and exact-sample rational means must match"
        )


def causal_map_state_to_dict(
    state: CausalMapForagerState,
    config: CausalMapForagerConfig,
) -> dict[str, Any]:
    """Serialize a validated state without host-specific paths or objects."""
    validate_causal_map_state(state, config)
    fields: dict[str, Any] = {}
    for name, value in state._asdict().items():
        if name == "rng_key":
            array = np.asarray(jr.key_data(value), dtype=np.uint32)
        else:
            array = np.asarray(value)
        fields[name] = array.tolist()
    return {
        "schema": CAUSAL_MAP_STATE_SCHEMA,
        "prng_impl": _prng_impl_name(state.rng_key),
        "jax_threefry_partitionable": bool(
            np.asarray(state.jax_threefry_partitionable)
        ),
        "config": config.to_dict(),
        "config_sha256": config.fingerprint(),
        "dtypes": dict(_STATE_SERIALIZED_DTYPES),
        "fields": fields,
    }


def _raw_json_array(
    raw: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    kind: str,
) -> None:
    """Validate a JSON field without relying on coercive NumPy conversion."""
    def require_json_shape(value: Any, dimensions: tuple[int, ...]) -> None:
        if not dimensions:
            if isinstance(value, (list, tuple, Mapping)):
                raise ValueError(f"serialized {name} must contain JSON scalars")
            return
        if type(value) is not list or len(value) != dimensions[0]:
            raise ValueError(
                f"serialized {name} must use JSON arrays with shape {shape}"
            )
        for child in value:
            require_json_shape(child, dimensions[1:])

    require_json_shape(raw, shape)
    try:
        array = np.asarray(raw, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"serialized {name} is not rectangular") from exc
    if array.shape != shape:
        raise ValueError(f"serialized {name} must have shape {shape}")

    for value in array.reshape(-1):
        if kind == "int32":
            valid = (
                type(value) is int
                and np.iinfo(np.int32).min <= value <= np.iinfo(np.int32).max
            )
        elif kind == "uint32-key-data":
            valid = type(value) is int and 0 <= value <= np.iinfo(np.uint32).max
        elif kind == "uint32":
            valid = type(value) is int and 0 <= value <= np.iinfo(np.uint32).max
        elif kind == "bool":
            valid = type(value) is bool
        elif kind == "float32":
            valid = type(value) is float and math.isfinite(value)
            if valid:
                valid = (
                    abs(value) <= np.finfo(np.float32).max
                    and float(np.float32(value)) == value
                )
        else:  # pragma: no cover - internal schema table is closed
            host_kind = _require_exact_str("kind", kind)
            raise RuntimeError(f"unknown serialized state kind '{host_kind}'")
        if not valid:
            raise ValueError(
                f"serialized {name} contains a value incompatible with {kind}"
            )


def _require_raw_json_value(value: Any, *, path: str) -> None:
    """Reject Python-only containers and scalar types in serialized payloads."""
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise ValueError(f"{path} JSON object keys must be strings")
            _require_raw_json_value(child, path=f"{path}.{key}")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _require_raw_json_value(child, path=f"{path}[{index}]")
        return
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    raise ValueError(f"{path} must contain only raw JSON values")


def causal_map_state_from_dict(
    payload: Mapping[str, Any],
    config: CausalMapForagerConfig,
) -> CausalMapForagerState:
    """Restore and validate a state serialized by :func:`causal_map_state_to_dict`."""
    if not isinstance(payload, Mapping):
        raise ValueError("serialized state must be a mapping")
    if not isinstance(config, CausalMapForagerConfig):
        raise ValueError("config must be a CausalMapForagerConfig")
    expected_top_level = {
        "schema",
        "prng_impl",
        "jax_threefry_partitionable",
        "config",
        "config_sha256",
        "dtypes",
        "fields",
    }
    if set(payload) != expected_top_level:
        raise ValueError("serialized state top-level fields do not match the schema")
    raw_schema = payload.get("schema")
    if type(raw_schema) is not str or raw_schema != CAUSAL_MAP_STATE_SCHEMA:
        raise ValueError("unsupported causal-map state schema")
    raw_prng_impl = payload.get("prng_impl")
    if type(raw_prng_impl) is not str or raw_prng_impl != _CAUSAL_MAP_PRNG_IMPL:
        raise ValueError("serialized state PRNG implementation does not match")
    checkpoint_partitionable = payload.get("jax_threefry_partitionable")
    if type(checkpoint_partitionable) is not bool:
        raise ValueError(
            "serialized state jax_threefry_partitionable mode must be boolean"
        )
    if checkpoint_partitionable != _threefry_partitionable_mode():
        raise ValueError(
            "serialized state jax_threefry_partitionable mode does not match runtime"
        )
    declared_config_value = payload.get("config")
    _require_raw_json_value(
        declared_config_value,
        path="serialized state config",
    )
    try:
        declared_config = json.dumps(
            declared_config_value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        expected_config = json.dumps(
            config.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        legacy_config_value = config.to_dict()
        legacy_config_value.pop("arrival_aware_readiness")
        legacy_config = json.dumps(
            legacy_config_value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("serialized state config is not canonical JSON") from exc
    declared_config_sha256 = hashlib.sha256(declared_config.encode()).hexdigest()
    raw_config_sha256 = payload.get("config_sha256")
    declared_hash_matches = (
        type(raw_config_sha256) is str
        and raw_config_sha256 == declared_config_sha256
    )
    current_config_matches = (
        declared_config == expected_config
        and declared_config_sha256 == config.fingerprint()
    )
    # State v5 predates the scheduling option but its state tuple is unchanged.
    # A field-absent checkpoint therefore remains resumable only under the
    # explicit legacy decision-time policy.  Never silently resume it under the
    # new arrival-aware default, which would change continuation behavior.
    legacy_config_matches = (
        not config.arrival_aware_readiness
        and declared_config == legacy_config
    )
    if not declared_hash_matches or not (
        current_config_matches or legacy_config_matches
    ):
        raise ValueError("serialized state embeds a different configuration")
    raw_dtypes = payload.get("dtypes")
    if not isinstance(raw_dtypes, Mapping) or dict(raw_dtypes) != _STATE_SERIALIZED_DTYPES:
        raise ValueError("serialized state dtype table does not match the schema")
    raw_fields = payload.get("fields")
    if not isinstance(raw_fields, Mapping):
        raise ValueError("serialized state fields must be a mapping")
    expected = set(CausalMapForagerState._fields)
    if set(raw_fields) != expected:
        raise ValueError("serialized state fields do not match the schema")

    raw_reward_sum = raw_fields["reward_sum"]
    if not isinstance(raw_reward_sum, list) or not raw_reward_sum:
        raise ValueError("serialized reward_sum must be a non-empty JSON array")
    channel_count = len(raw_reward_sum)
    map_fields = {
        "cell_channel",
        "cell_active",
        "cell_collection_step",
        "cell_ready_step",
        "cell_retry_count",
        "cell_last_seen_step",
        "cell_last_absent_step",
        "visit_count",
    }
    channel_fields = {
        "reward_sum",
        "reward_count",
        "respawn_interval_count",
        "respawn_interval_lower_floor",
        "respawn_interval_lower_remainder",
        "respawn_interval_upper_floor",
        "respawn_interval_upper_remainder",
        "respawn_exact_count",
        "respawn_exact_floor",
        "respawn_exact_remainder",
        "respawn_exact_mean",
        "respawn_exact_m2",
    }
    vector_fields = {"position", "last_target_position", "rng_key"}
    for name in CausalMapForagerState._fields:
        shape = (
            config.world_shape
            if name in map_fields
            else (channel_count,)
            if name in channel_fields
            else (2,)
            if name in vector_fields
            else ()
        )
        _raw_json_array(
            raw_fields[name],
            name=name,
            shape=shape,
            kind=_STATE_SERIALIZED_DTYPES[name],
        )
    if (
        raw_fields["jax_threefry_partitionable"]
        is not checkpoint_partitionable
    ):
        raise ValueError(
            "serialized state-bound and top-level jax_threefry_partitionable "
            "modes do not match"
        )

    values: dict[str, Array] = {}
    for name in CausalMapForagerState._fields:
        raw = raw_fields[name]
        if name == "rng_key":
            key_data = jnp.asarray(raw, dtype=jnp.uint32)
            values[name] = jr.wrap_key_data(
                key_data,
                impl=_CAUSAL_MAP_PRNG_IMPL,
            )
        elif name in _STATE_UINT32_FIELDS:
            values[name] = jnp.asarray(raw, dtype=jnp.uint32)
        elif name in _STATE_INTEGER_FIELDS:
            values[name] = jnp.asarray(raw, dtype=jnp.int32)
        elif name in _STATE_BOOLEAN_FIELDS:
            values[name] = jnp.asarray(raw, dtype=jnp.bool_)
        else:
            values[name] = jnp.asarray(raw, dtype=jnp.float32)
    state = CausalMapForagerState(**values)
    validate_causal_map_state(state, config)
    return state


class CausalMapForagerAgent:
    """Host policy facade over the pure causal-map transition."""

    def __init__(
        self,
        config: CausalMapForagerConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        if config is not None and not isinstance(config, CausalMapForagerConfig):
            raise TypeError("config must be a CausalMapForagerConfig")
        self.config = config if config is not None else CausalMapForagerConfig()
        if type(seed) not in _EXACT_INTEGER_TYPES:
            raise ValueError("seed must be a uint32-compatible integer")
        self.seed = int(seed)
        if not 0 <= self.seed <= np.iinfo(np.uint32).max:
            raise ValueError("seed must be a uint32-compatible integer")
        self._state: CausalMapForagerState | None = None

    @property
    def name(self) -> str:
        """Stable benchmark method name."""
        return CAUSAL_MAP_VARIANT_KIND

    @property
    def privileged(self) -> bool:
        """The policy never receives evaluator context."""
        return False

    @property
    def state(self) -> CausalMapForagerState:
        """Current finite planner state."""
        if self._state is None:
            raise RuntimeError("start() must be called before state")
        return self._state

    def start(
        self,
        observation: Any,
        context: ForagerAgentContext | None = None,
    ) -> int:
        """Build the initial relative map and select an action."""
        del context
        _validate_observation_host(observation, self.config)
        self._state, action = causal_map_start(observation, self.config, self.seed)
        return int(action)

    def step(
        self,
        reward: float,
        observation: Any,
        context: ForagerAgentContext | None = None,
    ) -> int:
        """Learn once from the transition and replan."""
        del context
        if self._state is None:
            raise RuntimeError("start() must be called before step()")
        if isinstance(reward, (bool, np.bool_)) or not isinstance(
            reward,
            (int, float, np.integer, np.floating),
        ):
            raise ValueError("reward must be one finite real scalar")
        try:
            reward_value = float(reward)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("reward must be one finite real scalar") from exc
        if (
            not math.isfinite(reward_value)
            or abs(reward_value) > float(np.finfo(np.float32).max)
            or not _finite_float32(reward_value)
        ):
            raise ValueError("reward must be one finite float32 scalar")
        _, _, channels = _validate_observation_host(observation, self.config)
        if channels != self._state.reward_sum.shape[0]:
            raise ValueError(
                "observation channel count changed during the continuing run"
            )
        self._state, action, _ = causal_map_step(
            self._state,
            jnp.asarray(reward_value, dtype=jnp.float32),
            observation,
            self.config,
        )
        return int(action)

    def metadata(self) -> Mapping[str, Any]:
        """Return explicit algorithm and non-privilege provenance."""
        partitionable_mode = _threefry_partitionable_mode()
        if self._state is not None:
            state_mode = bool(np.asarray(self._state.jax_threefry_partitionable))
            if state_mode != partitionable_mode:
                raise ValueError(
                    "agent state jax_threefry_partitionable mode does not match runtime"
                )
            partitionable_mode = state_mode
        return {
            "name": self.name,
            "privileged": False,
            "seed": self.seed,
            "variant_kind": CAUSAL_MAP_VARIANT_KIND,
            "state_schema": CAUSAL_MAP_STATE_SCHEMA,
            "prng_impl": _CAUSAL_MAP_PRNG_IMPL,
            "jax_threefry_partitionable": partitionable_mode,
            "config": self.config.to_dict(),
            "config_sha256": self.config.fingerprint(),
            "status_claim": "candidate; no SOTA claim without matched-seed evidence",
            "update_semantics": (
                "one causal map/reward/schedule update per transition; no replay"
            ),
            "world_model": {
                "kind": "relative_toroidal_cognitive_map",
                "origin": "arbitrary initial agent location",
                "maximum_map_cells": _MAX_CAUSAL_MAP_CELLS,
                "reward_model": "online empirical mean per observed channel",
                "unknown_value": "optimistic configurable prior",
                "respawn_model": (
                    "channel-agnostic initial retry prior, exponential empty-retry "
                    "backoff on every visible due miss, a distribution-free "
                    "identification interval for censored reappearance delays, "
                    "and a separate exact-delay rational sum with streamed "
                    "Welford M2 centered on its canonical float32 mean; "
                    "scheduling uses the exact outward ceiling of the rational "
                    "interval upper mean and may only be raised by exact-delay "
                    "safety statistics, subject to the configured maximum delay "
                    "ceiling; optionally, pending targets become eligible when "
                    "exact cost-aware path travel predicts they will be collectible on "
                    "arrival under the public move/reward-before-respawn step order"
                ),
                "negative_avoidance": (
                    "exact toroidal lexicographic routing that minimizes entries into "
                    "known learned-negative traversable cells before path length; "
                    "target scores charge the worst empirical negative-channel mean "
                    "per necessary entry, while no-target motion retains an explicit "
                    "all-neighbors-negative fallback"
                ),
                "negative_route_semantics": (
                    "known-negative cells are costly, never impassable; a clean route "
                    "always wins for the same target, but the minimum-loss crossing "
                    "remains available to profitable or unexplored regions"
                ),
                "arrival_aware_readiness": self.config.arrival_aware_readiness,
                "arrival_readiness_semantics": (
                    "ready_step <= saturating(step_count + max(route_step_distance "
                    "- 1, 0)); the subtraction reflects public move/reward-before-"
                    "respawn transition order"
                    if self.config.arrival_aware_readiness
                    else "ready_step <= step_count"
                ),
                "coverage": (
                    "seeded exploration only toward genuinely unobserved reachable "
                    "cells; generic safe least-visited coverage is reserved for the "
                    "no-exploitation/no-unknown fail-safe"
                ),
                "exploration_probability_semantics": (
                    "probability of choosing an unobserved reachable target when an "
                    "exploitation target also exists; unknown coverage is mandatory "
                    "when no exploitation target exists and never broadens to already "
                    "observed cells"
                ),
            },
            "nonprivilege_contract": {
                "policy_inputs": [
                    "raw current observation",
                    "current scalar reward",
                    "own previous action",
                    "own finite causal state",
                    "public 4-action and 15x15 toroidal task semantics",
                ],
                "forbidden_inputs": [
                    "ForagerAgentContext",
                    "environment state",
                    "global position",
                    "global object or reward grid",
                    "biome id or task label",
                    "evaluator info",
                    "hidden environment time",
                ],
                "context_consumed": False,
            },
        }

    def state_dict(self) -> dict[str, Any]:
        """Serialize the current validated state."""
        if int(self.state.initial_seed) != self.seed:
            raise ValueError("state initial_seed does not match agent seed")
        return causal_map_state_to_dict(self.state, self.config)

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        """Restore a validated state."""
        restored = causal_map_state_from_dict(payload, self.config)
        if int(restored.initial_seed) != self.seed:
            raise ValueError("checkpoint initial_seed does not match agent seed")
        self._state = restored


def causal_map_variant_spec(
    config: CausalMapForagerConfig,
) -> dict[str, Any]:
    """Return a matrix-runner-friendly immutable variant description."""
    if not isinstance(config, CausalMapForagerConfig):
        raise TypeError("config must be a CausalMapForagerConfig")
    return {
        "kind": CAUSAL_MAP_VARIANT_KIND,
        "agent": CAUSAL_MAP_VARIANT_KIND,
        "privileged": False,
        "state_schema": CAUSAL_MAP_STATE_SCHEMA,
        "prng_impl": _CAUSAL_MAP_PRNG_IMPL,
        "jax_threefry_partitionable": _threefry_partitionable_mode(),
        "config": config.to_dict(),
        "config_sha256": config.fingerprint(),
        "implementation": (
            "alberta_framework.benchmarks.causal_map_forager:"
            "CausalMapForagerAgent"
        ),
    }


def _validate_benchmark_contract(
    agent_config: CausalMapForagerConfig,
    benchmark_config: ForagerBenchmarkConfig,
) -> None:
    env = benchmark_config.environment
    if (
        type(benchmark_config.steps) is not int
        or benchmark_config.steps < 1
        or benchmark_config.steps >= _INT32_MAX
    ):
        raise ValueError(
            "causal-map benchmark steps must be a positive int below int32 maximum"
        )
    if (
        type(benchmark_config.jax_chunk_size) is not int
        or benchmark_config.jax_chunk_size < 1
        or benchmark_config.jax_chunk_size >= _INT32_MAX
    ):
        raise ValueError(
            "causal-map jax_chunk_size must be a positive int below int32 maximum"
        )
    if env.preset != "field_of_view" or env.resolved_env_id != "ForagaxTwoBiomeLarge-v1":
        raise ValueError(
            "causal-map variant is defined only for stationary Forager field_of_view"
        )
    if env.require_exact_version is not True:
        raise ValueError("causal-map benchmark requires the exact pinned Foragax build")
    if env.resolved_observation_type != "color":
        raise ValueError("causal-map variant requires one-hot color observations")
    if env.reward_delay != 0:
        raise ValueError("causal-map variant currently requires immediate rewards")
    if env.random_shift_max_steps != 0:
        raise ValueError("stationary causal-map protocol forbids random time shifts")
    if env.extra_kwargs:
        raise ValueError("stationary causal-map protocol forbids extra environment kwargs")
    if agent_config.world_shape != (15, 15):
        raise ValueError("ForagaxTwoBiomeLarge-v1 has public world_shape (15, 15)")
    if env.aperture_size == -1:
        raise ValueError("causal-map variant is a partial field-of-view policy")
    if env.aperture_size < 3:
        raise ValueError(
            "causal-map Forager requires an odd aperture of at least 3; "
            "aperture 1 cannot causally identify a collected destination channel"
        )
    if env.aperture_size % 2 != 1:
        raise ValueError("causal-map Forager requires an odd centered aperture")
    if env.aperture_size > min(agent_config.world_shape):
        raise ValueError("aperture cannot exceed the configured world")
    maximum_batch_observations = env.aperture_size * env.aperture_size
    if benchmark_config.steps * maximum_batch_observations >= _INT32_MAX:
        raise ValueError(
            "causal-map horizon and aperture can overflow int32 reappearance counts"
        )


class _LaneMetrics:
    """Bounded host-side metric accumulator for one device lane."""

    def __init__(self, config: ForagerBenchmarkConfig) -> None:
        self.config = config
        self.target_steps = [1]
        self.target_steps.extend(
            range(config.record_every, config.steps + 1, config.record_every)
        )
        if self.target_steps[-1] != config.steps:
            self.target_steps.append(config.steps)
        self.target_steps = sorted(set(self.target_steps))
        self.target_index = 0
        self.curve_steps: list[int] = []
        self.curve_ewm: list[float] = []
        self.curve_window: list[float] = []
        self.reward_tail = np.zeros((0,), dtype=np.float64)
        self.total_reward = 0.0
        self.ewm_total = 0.0
        self.ewm_filter_state = np.zeros((1,), dtype=np.float64)
        self.fov_filter_state = np.zeros((1,), dtype=np.float64)
        self.fov_samples: list[float] = []
        self.final_ewm = math.nan
        self.regret_total = 0.0
        self.regret_count = 0
        self.final_regret = math.nan
        self.all_finite = True

    def add(
        self,
        rewards: np.ndarray,
        regrets: np.ndarray,
        finite: np.ndarray,
        *,
        completed: int,
    ) -> None:
        active = rewards.size
        ewm, self.ewm_filter_state = _adjusted_ewm_chunk(
            rewards,
            decay=self.config.ewm_decay,
            completed_steps=completed,
            filter_state=self.ewm_filter_state,
        )
        fov, self.fov_filter_state = _unadjusted_ema_chunk(
            rewards,
            decay=FORAGER_FOV_EMA_DECAY,
            filter_state=self.fov_filter_state,
        )
        fov_mask = (
            np.arange(completed, completed + active) % FORAGER_FOV_EMA_SUBSAMPLE
            == 0
        )
        self.fov_samples.extend(float(value) for value in fov[fov_mask])
        self.final_ewm = float(ewm[-1])
        self.all_finite = self.all_finite and bool(np.all(finite))
        self.total_reward += float(np.sum(rewards, dtype=np.float64))
        self.ewm_total += float(np.sum(ewm, dtype=np.float64))
        finite_regrets = regrets[np.isfinite(regrets)]
        if finite_regrets.size:
            self.regret_total += float(np.sum(finite_regrets, dtype=np.float64))
            self.regret_count += int(finite_regrets.size)
            self.final_regret = float(regrets[-1])

        combined = np.concatenate((self.reward_tail, rewards))
        prefix = np.concatenate(
            (np.zeros((1,), dtype=np.float64), np.cumsum(combined, dtype=np.float64))
        )
        while (
            self.target_index < len(self.target_steps)
            and self.target_steps[self.target_index] <= completed + active
        ):
            step_number = self.target_steps[self.target_index]
            local_count = step_number - completed
            end = self.reward_tail.size + local_count
            start = max(0, end - self.config.final_window)
            self.curve_steps.append(step_number)
            self.curve_ewm.append(float(ewm[local_count - 1]))
            self.curve_window.append(
                float((prefix[end] - prefix[start]) / (end - start))
            )
            self.target_index += 1
        self.reward_tail = combined[
            -min(self.config.final_window, combined.size) :
        ]


def _make_scan_chunk(
    env: Any,
    params: Any,
    agent_config: CausalMapForagerConfig,
    benchmark_config: ForagerBenchmarkConfig,
) -> Any:
    def scan_chunk(
        initial_carry: tuple[Any, Array, CausalMapForagerState, Array],
        active_steps: Array,
    ) -> tuple[
        tuple[Any, Array, CausalMapForagerState, Array],
        tuple[Array, Array, Array, Array],
    ]:
        def active_step(
            carry: tuple[Any, Array, CausalMapForagerState, Array],
        ) -> tuple[
            tuple[Any, Array, CausalMapForagerState, Array],
            tuple[Array, Array, Array, Array],
        ]:
            env_state, env_key, agent_state, action = carry
            env_key, step_key = jr.split(env_key)
            observation, next_env_state, reward, done, info = env.step(
                step_key,
                env_state,
                action,
                params,
            )
            next_agent_state, next_action, _ = causal_map_step(
                agent_state,
                reward,
                observation,
                agent_config,
            )
            finite = jnp.isfinite(reward) & jnp.all(
                jnp.asarray(
                    (
                        jnp.all(jnp.isfinite(next_agent_state.reward_sum)),
                        jnp.all(jnp.isfinite(next_agent_state.respawn_exact_mean)),
                        jnp.all(jnp.isfinite(next_agent_state.respawn_exact_m2)),
                    )
                )
            )
            return (
                next_env_state,
                env_key,
                next_agent_state,
                next_action,
            ), (
                reward,
                _exact_float32_biome_regret(info),
                done,
                finite,
            )

        def inactive_step(
            carry: tuple[Any, Array, CausalMapForagerState, Array],
        ) -> tuple[
            tuple[Any, Array, CausalMapForagerState, Array],
            tuple[Array, Array, Array, Array],
        ]:
            zero = jnp.asarray(0.0, dtype=jnp.float32)
            return carry, (
                zero,
                zero,
                jnp.asarray(False),
                jnp.asarray(True),
            )

        def body(
            carry: tuple[Any, Array, CausalMapForagerState, Array],
            index: Array,
        ) -> tuple[
            tuple[Any, Array, CausalMapForagerState, Array],
            tuple[Array, Array, Array, Array],
        ]:
            return cast(
                tuple[
                    tuple[Any, Array, CausalMapForagerState, Array],
                    tuple[Array, Array, Array, Array],
                ],
                jax.lax.cond(
                    index < active_steps,
                    active_step,
                    inactive_step,
                    carry,
                ),
            )

        return jax.lax.scan(
            body,
            initial_carry,
            jnp.arange(benchmark_config.jax_chunk_size, dtype=jnp.int32),
        )

    return scan_chunk


def _host_metric_array(
    value: Any,
    *,
    dtype: Any | None = None,
) -> np.ndarray[Any, Any]:
    """Materialize one compiled evaluator output for guarded host processing."""
    return np.asarray(value, dtype=dtype)


def _require_host_threefry_mode(expected: bool) -> None:
    """Reject host-side runtime drift from a trajectory's state-bound mode."""
    if _threefry_partitionable_mode() != expected:
        raise RuntimeError(
            "causal-map runner jax_threefry_partitionable mode drifted "
            "before trace finalization"
        )


def _build_causal_map_result(
    *,
    agent_config: CausalMapForagerConfig,
    cfg: ForagerBenchmarkConfig,
    seeds: tuple[int, ...],
    mode: ForagerBatchMode,
    seed: int,
    lane_metrics: _LaneMetrics,
    lane_state: CausalMapForagerState,
    base_metadata: Mapping[str, Any],
    trace_metadata: Mapping[str, Any] | None,
    overall_duration: float,
    overall_started: float,
    compile_started: float,
    compile_duration: float,
    execution_duration: float,
) -> ForagerRunResult:
    """Build one fully materialized lane result from validated host state."""
    count = len(seeds)
    aggregate_fps = count * cfg.steps / max(execution_duration, 1e-12)
    effective_seed_fps = cfg.steps / max(execution_duration, 1e-12)
    metadata = dict(base_metadata)
    metadata["environment_rng_schedule"] = FORAGER_ENVIRONMENT_RNG_SCHEDULE
    metadata["environment_rng_schedule_sha256"] = environment_rng_schedule_sha256()
    metadata["environment_prng_impl"] = _CAUSAL_MAP_ENVIRONMENT_PRNG_IMPL
    metadata["environment_prng_impl_explicit"] = True
    lane_reward_sum = np.asarray(lane_state.reward_sum, dtype=np.float64)
    lane_reward_count = np.asarray(lane_state.reward_count, dtype=np.int64)
    lane_respawn_interval_count = np.asarray(
        lane_state.respawn_interval_count,
        dtype=np.int64,
    )
    lane_respawn_interval_lower_floor = np.asarray(
        lane_state.respawn_interval_lower_floor,
        dtype=np.int64,
    )
    lane_respawn_interval_lower_remainder = np.asarray(
        lane_state.respawn_interval_lower_remainder,
        dtype=np.int64,
    )
    lane_respawn_interval_upper_floor = np.asarray(
        lane_state.respawn_interval_upper_floor,
        dtype=np.int64,
    )
    lane_respawn_interval_upper_remainder = np.asarray(
        lane_state.respawn_interval_upper_remainder,
        dtype=np.int64,
    )
    lane_respawn_exact_count = np.asarray(
        lane_state.respawn_exact_count,
        dtype=np.int64,
    )
    lane_respawn_exact_floor = np.asarray(
        lane_state.respawn_exact_floor,
        dtype=np.int64,
    )
    lane_respawn_exact_remainder = np.asarray(
        lane_state.respawn_exact_remainder,
        dtype=np.int64,
    )
    lane_respawn_exact_mean = np.asarray(
        lane_state.respawn_exact_mean,
        dtype=np.float64,
    )
    lane_respawn_exact_m2 = np.asarray(
        lane_state.respawn_exact_m2,
        dtype=np.float64,
    )
    learned_reward_means = [
        (
            float(lane_reward_sum[index] / lane_reward_count[index])
            if lane_reward_count[index] > 0
            else None
        )
        for index in range(lane_reward_count.size)
    ]
    learned_respawn_exact_std = [
        (
            float(
                math.sqrt(
                    max(
                        0.0,
                        lane_respawn_exact_m2[index]
                        / (lane_respawn_exact_count[index] - 1),
                    )
                )
            )
            if lane_respawn_exact_count[index] > 1
            else 0.0
        )
        for index in range(lane_respawn_exact_count.size)
    ]
    lane_cell_channel = np.asarray(lane_state.cell_channel, dtype=np.int32)
    negative_channel_mask = np.asarray(
        [
            channel_count > 0
            and mean is not None
            and mean < agent_config.negative_reward_threshold
            for channel_count, mean in zip(
                lane_reward_count,
                learned_reward_means,
                strict=True,
            )
        ],
        dtype=np.bool_,
    )
    known_mask = lane_cell_channel >= 0
    known_negative_mask = known_mask & negative_channel_mask[
        np.maximum(lane_cell_channel, 0)
    ]
    metadata["final_causal_diagnostics"] = {
        "reward_count_by_observed_channel": lane_reward_count.tolist(),
        "reward_sum_by_observed_channel": lane_reward_sum.tolist(),
        "reward_mean_by_observed_channel": learned_reward_means,
        "respawn_interval_count_by_observed_channel": (
            lane_respawn_interval_count.tolist()
        ),
        "respawn_interval_lower_floor_by_observed_channel": (
            lane_respawn_interval_lower_floor.tolist()
        ),
        "respawn_interval_lower_remainder_by_observed_channel": (
            lane_respawn_interval_lower_remainder.tolist()
        ),
        "respawn_interval_upper_floor_by_observed_channel": (
            lane_respawn_interval_upper_floor.tolist()
        ),
        "respawn_interval_upper_remainder_by_observed_channel": (
            lane_respawn_interval_upper_remainder.tolist()
        ),
        "respawn_interval_upper_outward_ceil_by_observed_channel": (
            (
                lane_respawn_interval_upper_floor
                + (lane_respawn_interval_upper_remainder > 0).astype(np.int64)
            ).tolist()
        ),
        "exact_respawn_count_by_observed_channel": lane_respawn_exact_count.tolist(),
        "exact_respawn_floor_by_observed_channel": lane_respawn_exact_floor.tolist(),
        "exact_respawn_remainder_by_observed_channel": (
            lane_respawn_exact_remainder.tolist()
        ),
        "exact_respawn_delay_mean_by_observed_channel": (
            lane_respawn_exact_mean.tolist()
        ),
        "exact_respawn_delay_sample_std_by_observed_channel": (
            learned_respawn_exact_std
        ),
        "known_fixed_cells": int(np.sum(known_mask)),
        "known_learned_negative_cells": int(np.sum(known_negative_mask)),
        "map_cells": int(agent_config.height * agent_config.width),
    }
    metadata["runner"] = {
        "kind": "jax_scan" if count == 1 else "jax_batched_scan",
        "batch_mode": mode,
        "batch_size": count,
        "batch_seeds": list(seeds),
        "chunk_size": cfg.jax_chunk_size,
        "environment_prng_impl": _CAUSAL_MAP_ENVIRONMENT_PRNG_IMPL,
        "environment_prng_impl_explicit": True,
        "overall_duration_s": overall_duration,
        "setup_duration_s": compile_started - overall_started,
        "compile_duration_s": compile_duration,
        "execution_duration_s": execution_duration,
        "aggregate_transitions_per_second": aggregate_fps,
        "per_seed_effective_frames_per_second": effective_seed_fps,
        "bounded_reward_buffer_steps_per_seed": min(cfg.final_window, cfg.steps),
        "full_reward_history_retained": False,
        "rounding_contract": (
            "pure per-lane map transition; vmap and lax.map trajectories are exact"
        ),
    }
    if trace_metadata is not None:
        metadata["raw_metric_trace"] = dict(trace_metadata)
    return ForagerRunResult(
        agent=CAUSAL_MAP_VARIANT_KIND,
        privileged=False,
        seed=seed,
        steps=cfg.steps,
        total_reward=lane_metrics.total_reward,
        mean_reward=lane_metrics.total_reward / cfg.steps,
        final_window_mean_reward=float(np.mean(lane_metrics.reward_tail)),
        final_ewm_reward=lane_metrics.final_ewm,
        mean_ewm_reward=lane_metrics.ewm_total / cfg.steps,
        fov_last_10pct_ema_auc=_fov_last_tenth_ema_auc(lane_metrics.fov_samples),
        mean_biome_regret=(
            lane_metrics.regret_total / lane_metrics.regret_count
            if lane_metrics.regret_count
            else math.nan
        ),
        final_biome_regret=lane_metrics.final_regret,
        curve_steps=tuple(lane_metrics.curve_steps),
        curve_ewm_reward=tuple(lane_metrics.curve_ewm),
        curve_window_reward=tuple(lane_metrics.curve_window),
        duration_s=execution_duration,
        frames_per_second=effective_seed_fps,
        environment=cfg.environment.to_dict(),
        metric_contract=forager_metric_contract(
            ewm_decay=cfg.ewm_decay,
            final_window=cfg.final_window,
            record_every=cfg.record_every,
            steps=cfg.steps,
        ),
        agent_metadata=metadata,
    )


def _run_causal_map_lanes(
    agent_config: CausalMapForagerConfig,
    benchmark_config: ForagerBenchmarkConfig,
    seeds: tuple[int, ...],
    *,
    mode: ForagerBatchMode,
    reward_trace_sink_factory: ForagerRewardTraceSinkFactory | None = None,
) -> tuple[tuple[ForagerRunResult, ...], CausalMapForagerState | None]:
    _validate_benchmark_contract(agent_config, benchmark_config)
    cfg = benchmark_config
    trace_sinks = _create_reward_trace_sinks(
        reward_trace_sink_factory,
        seeds,
        steps=cfg.steps,
    )
    try:
        overall_started = time.perf_counter()
        env, params = cfg.environment.make()
        seed_values = jnp.asarray(seeds, dtype=jnp.uint32)

        def init_one(
            seed: Array,
        ) -> tuple[Any, Array, CausalMapForagerState, Array]:
            env_key = _causal_map_environment_key(seed)
            env_key, reset_key = jr.split(env_key)
            observation, env_state = env.reset(reset_key, params)
            agent_state, action = causal_map_start(observation, agent_config, seed)
            return env_state, env_key, agent_state, action

        initialize = jax.jit(jax.vmap(init_one))
        carry = initialize(seed_values)
        jax.block_until_ready(carry)  # type: ignore[no-untyped-call]
        seed_chunk = _make_scan_chunk(env, params, agent_config, cfg)
        if mode == "vmap":
            chunk_function = jax.jit(jax.vmap(seed_chunk, in_axes=(0, None)))
        else:

            def strict_chunk(carries: Any, active_steps: Array) -> Any:
                return jax.lax.map(
                    lambda lane_carry: seed_chunk(lane_carry, active_steps),
                    carries,
                )

            chunk_function = jax.jit(strict_chunk)

        compile_started = time.perf_counter()
        compiled_chunk = chunk_function.lower(
            carry,
            jnp.asarray(cfg.jax_chunk_size, dtype=jnp.int32),
        ).compile()
        compile_duration = time.perf_counter() - compile_started
        metrics = [_LaneMetrics(cfg) for _ in seeds]
        completed = 0
        execution_started = time.perf_counter()
    except BaseException:
        _abort_reward_trace_sinks(trace_sinks)
        raise
    while completed < cfg.steps:
        active = min(cfg.jax_chunk_size, cfg.steps - completed)
        try:
            carry, outputs = compiled_chunk(
                carry,
                jnp.asarray(active, dtype=jnp.int32),
            )
            rewards_device, regrets_device, done_device, finite_device = outputs
            jax.block_until_ready(outputs)  # type: ignore[no-untyped-call]
            raw_rewards = _host_metric_array(rewards_device[:, :active])
            raw_regrets = _host_metric_array(regrets_device[:, :active])
            if (
                raw_rewards.dtype != np.dtype(np.float32)
                or raw_regrets.dtype != np.dtype(np.float32)
            ):
                raise TypeError(
                    "Foragax evaluator outputs must retain exact float32 dtype"
                )
            rewards = raw_rewards.astype(np.float64)
            regrets = raw_regrets.astype(np.float64)
            done = _host_metric_array(done_device[:, :active], dtype=np.bool_)
            finite = _host_metric_array(finite_device[:, :active], dtype=np.bool_)
            if np.any(done):
                raise RuntimeError("Foragax paper presets must remain continuing")
            for lane, lane_metrics in enumerate(metrics):
                _append_reward_trace(
                    trace_sinks,
                    lane,
                    raw_rewards[lane],
                    raw_regrets[lane],
                )
                lane_metrics.add(
                    rewards[lane],
                    regrets[lane],
                    finite[lane],
                    completed=completed,
                )
            completed += active
        except BaseException:
            _abort_reward_trace_sinks(trace_sinks)
            raise
    try:
        execution_duration = time.perf_counter() - execution_started
        overall_duration = time.perf_counter() - overall_started
        final_agent_states = carry[2]
        final_leaves = jax.device_get(jax.tree_util.tree_leaves(final_agent_states))
        state_finite = all(
            bool(np.all(np.isfinite(leaf)))
            for leaf in final_leaves
            if jnp.issubdtype(leaf.dtype, jnp.inexact)
        )
        if not state_finite or not all(item.all_finite for item in metrics):
            raise FloatingPointError(
                "causal-map Alberta variant produced non-finite values"
            )
        validated_lane_states: list[CausalMapForagerState] = []
        for lane in range(len(seeds)):
            lane_state = jax.device_get(
                jax.tree_util.tree_map(lambda value: value[lane], final_agent_states)
            )
            validate_causal_map_state(lane_state, agent_config)
            if int(lane_state.step_count) != cfg.steps:
                raise ValueError(
                    "causal-map final state step_count does not match requested horizon"
                )
            if int(lane_state.initial_seed) != seeds[lane]:
                raise ValueError(
                    "causal-map final state initial_seed does not match requested lane"
                )
            validated_lane_states.append(lane_state)
        bound_partitionable_modes = {
            bool(np.asarray(state.jax_threefry_partitionable))
            for state in validated_lane_states
        }
        if len(bound_partitionable_modes) != 1:
            raise ValueError(
                "causal-map lanes disagree on jax_threefry_partitionable mode"
            )
        bound_partitionable_mode = bound_partitionable_modes.pop()
    except BaseException:
        _abort_reward_trace_sinks(trace_sinks)
        raise
    try:
        base_metadata_by_lane: list[Mapping[str, Any]] = []
        for seed in seeds:
            base_metadata = dict(
                CausalMapForagerAgent(agent_config, seed=seed).metadata()
            )
            base_metadata["jax_threefry_partitionable"] = (
                bound_partitionable_mode
            )
            base_metadata_by_lane.append(base_metadata)
        # This is the final fallible runtime-mode check before trace sealing.
        # Result construction below uses only the captured state-bound mode.
        _require_host_threefry_mode(bound_partitionable_mode)
        trace_metadata = _finalize_reward_trace_sinks(trace_sinks)
        if trace_metadata and len(trace_metadata) != len(seeds):
            raise ValueError("reward trace metadata lane count does not match seeds")
        results = tuple(
            _build_causal_map_result(
                agent_config=agent_config,
                cfg=cfg,
                seeds=seeds,
                mode=mode,
                seed=seed,
                lane_metrics=lane_metrics,
                lane_state=lane_state,
                base_metadata=base_metadata_by_lane[lane],
                trace_metadata=(
                    trace_metadata[lane] if trace_metadata else None
                ),
                overall_duration=overall_duration,
                overall_started=overall_started,
                compile_started=compile_started,
                compile_duration=compile_duration,
                execution_duration=execution_duration,
            )
            for lane, (seed, lane_metrics, lane_state) in enumerate(
                zip(seeds, metrics, validated_lane_states, strict=True)
            )
        )
        single_state = validated_lane_states[0] if len(seeds) == 1 else None
    except BaseException:
        _abort_reward_trace_sinks(trace_sinks)
        raise
    return results, single_state


def run_causal_map_forager(
    policy: CausalMapForagerAgent,
    benchmark_config: ForagerBenchmarkConfig,
) -> ForagerRunResult:
    """Run one seed with a compiled bounded-memory JAX scan."""
    if not isinstance(policy, CausalMapForagerAgent):
        raise TypeError("policy must be a CausalMapForagerAgent")
    if not isinstance(benchmark_config, ForagerBenchmarkConfig):
        raise TypeError("benchmark_config must be a ForagerBenchmarkConfig")
    if policy.seed != benchmark_config.seed:
        raise ValueError("policy seed must equal benchmark_config.seed")
    results, final_state = _run_causal_map_lanes(
        policy.config,
        benchmark_config,
        (policy.seed,),
        mode="vmap",
    )
    if final_state is None:  # pragma: no cover - count is statically one
        raise RuntimeError("single-lane runner did not return final state")
    policy._state = final_state
    return results[0]


def run_causal_map_forager_seeds(
    agent_config: CausalMapForagerConfig,
    benchmark_config: ForagerBenchmarkConfig,
    seeds: Sequence[int],
    *,
    mode: ForagerBatchMode = "vmap",
    reward_trace_sink_factory: ForagerRewardTraceSinkFactory | None = None,
) -> tuple[ForagerRunResult, ...]:
    """Run unique seeds with one compiled executable and bounded host memory."""
    if not isinstance(agent_config, CausalMapForagerConfig):
        raise TypeError("agent_config must be a CausalMapForagerConfig")
    if not isinstance(benchmark_config, ForagerBenchmarkConfig):
        raise TypeError("benchmark_config must be a ForagerBenchmarkConfig")
    raw_seeds = tuple(seeds)
    if not raw_seeds:
        raise ValueError("seeds must be non-empty")
    if any(
        type(seed) not in _EXACT_INTEGER_TYPES
        for seed in raw_seeds
    ):
        raise ValueError("seeds must be uint32-compatible integers without coercion")
    ordered = tuple(int(seed) for seed in raw_seeds)
    if len(set(ordered)) != len(ordered):
        raise ValueError("seeds must be unique")
    if any(seed < 0 or seed > np.iinfo(np.uint32).max for seed in ordered):
        raise ValueError("seeds must be uint32-compatible non-negative integers")
    if mode not in ("vmap", "strict"):
        raise ValueError("mode must be 'vmap' or 'strict'")
    results, _ = _run_causal_map_lanes(
        agent_config,
        benchmark_config,
        ordered,
        mode=mode,
        reward_trace_sink_factory=reward_trace_sink_factory,
    )
    return results


__all__ = [
    "CAUSAL_MAP_STATE_SCHEMA",
    "CAUSAL_MAP_VARIANT_KIND",
    "CausalMapForagerAgent",
    "CausalMapForagerConfig",
    "CausalMapForagerState",
    "CausalMapStepDiagnostics",
    "causal_map_start",
    "causal_map_rng_contract",
    "causal_map_state_from_dict",
    "causal_map_state_to_dict",
    "causal_map_step",
    "causal_map_variant_spec",
    "run_causal_map_forager",
    "run_causal_map_forager_seeds",
    "validate_causal_map_state",
]
