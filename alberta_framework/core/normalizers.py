"""Online feature normalization for continual learning.

Implements online (streaming) normalization that updates estimates of mean
and variance at every time step, following the principle of temporal uniformity.

Three normalizer variants are provided:

- ``EMANormalizer``: Exponential moving average estimates of mean and variance.
  Suitable for non-stationary distributions where recent observations should
  be weighted more heavily.

- ``WelfordNormalizer``: Welford's online algorithm for numerically stable
  estimation of cumulative sample mean and variance with Bessel's correction.
  Suitable for stationary distributions up through its explicit float32
  cumulative horizon of ``2**24`` accepted samples; it then fail-stops.

- ``StreamingBatchNormalizer``: BatchNorm-style running moments with no warmup
  schedule, useful as a direct streaming normalization comparator. Momentum
  below one is continual-recency; momentum one is static after initialization.
"""

import dataclasses
import math
import operator
from abc import ABC, abstractmethod
from collections.abc import Mapping
from numbers import Real
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.update_safety import (
    floating_tree_is_finite as _floating_tree_is_finite,
)


def _require_exact_str(name: object, value: object) -> str:
    if type(name) is not str:
        raise ValueError("name must be an exact string")
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    return value


NORMALIZER_STATE_SCHEMA = "alberta.normalizer-state.v2"
WELFORD_ESTIMATOR_SCHEMA = "alberta.welford-cumulative-float32-fail-stop-at-2p24.v2"
BOUNDED_RECENCY_ESTIMATOR_SEMANTICS = "bounded-recency"
CUMULATIVE_FLOAT32_ESTIMATOR_SEMANTICS = "cumulative-float32-fail-stop-at-2p24"
EMA_PRIOR_BOUNDED_RECENCY_ESTIMATOR_SEMANTICS = (
    "ema-zero-mean-unit-variance-prior-bounded-recency"
)
EMA_PRIOR_CUMULATIVE_FLOAT32_ESTIMATOR_SEMANTICS = (
    "ema-zero-mean-unit-variance-prior-cumulative-float32-fail-stop-at-2p24"
)
STATIC_AFTER_FIRST_ESTIMATOR_SEMANTICS = "static-after-first"
NORMALIZER_LIFETIME_COUNTER_NBYTES = 12
NORMALIZER_LIFETIME_COUNTER_DELTA_NBYTES = 8

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_FLOAT32_CONSECUTIVE_INTEGER_LIMIT = 2**24

_ACTUAL_INT_TYPES: tuple[type, ...] = (
    int,
    *(np.dtype(code).type for code in ("b", "B", "h", "H", "i", "I", "l", "L", "q", "Q")),
)


def _require_int(
    name: str,
    value: object,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer")
    number = operator.index(cast(SupportsIndex, value))
    if minimum is not None and number < minimum:
        if minimum == 1:
            raise ValueError(f"{name} must be positive")
        if minimum == 0:
            raise ValueError(f"{name} must be non-negative")
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return number


def _stored_float32_equals_one(value: float) -> bool:
    """Return whether a validated host scalar is stored as float32 one."""

    return float(jnp.asarray(value, dtype=jnp.float32)) == 1.0


def _saturating_int32_counter_increment(value: Array) -> Int[Array, ""]:
    """Increment non-negative int32 telemetry without signed wraparound."""

    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    counter = jnp.asarray(value, dtype=jnp.int32)
    return jnp.minimum(jnp.maximum(counter, 0), maximum - 1) + 1


def _checked_lifetime_words_increment(
    words: Array,
) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Increment a big-endian uint32 pair, refusing its all-ones value."""

    if getattr(words, "shape", None) != (2,):
        raise ValueError("lifetime counter words must have shape (2,)")
    if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("lifetime counter words must have dtype uint32")
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity_available = ~jnp.all(words == maximum)
    one = jnp.asarray(1, dtype=jnp.uint32)
    low = words[1] + one
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    candidate = jnp.stack((words[0] + carry, low))
    return (
        jnp.where(capacity_available, candidate, words).astype(jnp.uint32),
        capacity_available,
    )


def _lifetime_counter_valid(
    words: Array,
    telemetry: Array,
) -> Bool[Array, ""]:
    """Validate an exact word clock against saturating int32 telemetry."""

    if getattr(words, "shape", None) != (2,):
        raise ValueError("lifetime counter words must have shape (2,)")
    if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("lifetime counter words must have dtype uint32")
    if getattr(telemetry, "shape", None) != ():
        raise ValueError("lifetime counter telemetry must be scalar")
    if getattr(telemetry, "dtype", None) != jnp.dtype(jnp.int32):
        raise TypeError("lifetime counter telemetry must have dtype int32")
    maximum_i32 = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    maximum_u32 = jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    fits_telemetry = (words[0] == 0) & (words[1] <= maximum_u32)
    expected = jnp.where(
        fits_telemetry,
        words[1].astype(jnp.int32),
        maximum_i32,
    )
    return (telemetry >= 0) & (telemetry == expected)


def _lifetime_words_are_zero(words: Array) -> Bool[Array, ""]:
    """Return whether an exact lifetime clock is at its initial event."""

    return jnp.all(words == jnp.asarray(0, dtype=jnp.uint32))


def _lifetime_words_to_float32(words: Array) -> Float[Array, ""]:
    """Return the best float32 representation of an exact word counter."""

    high_scale = jnp.asarray(2.0**32, dtype=jnp.float32)
    return words[0].astype(jnp.float32) * high_scale + words[1].astype(jnp.float32)


@chex.dataclass(frozen=True)
class EMANormalizerState:
    """State for EMA-based online feature normalization.

    Uses exponential moving average to estimate running mean and variance,
    suitable for non-stationary distributions.

    Attributes:
        mean: Running mean estimate per feature
        var: Running variance estimate per feature
        sample_count: Saturating int32 compatibility telemetry.
        sample_count_words: Exact big-endian ``[high, low]`` uint32 sample
            identity. The all-ones value is terminal.
        decay: Exponential decay factor for estimates. ``1.0`` retains the
            initial zero-mean, unit-variance pseudo-sample with cumulative
            weight instead of exponentially forgetting it.
    """

    mean: Float[Array, " feature_dim"]
    var: Float[Array, " feature_dim"]
    sample_count: Int[Array, ""]
    sample_count_words: UInt[Array, " 2"]
    decay: Float[Array, ""]


@chex.dataclass(frozen=True)
class WelfordNormalizerState:
    """State for Welford's online normalization algorithm.

    Uses Welford's algorithm for numerically stable estimation of cumulative
    sample mean and variance with Bessel's correction.

    Attributes:
        mean: Running mean estimate per feature
        var: Running variance estimate per feature (Bessel-corrected)
        sample_count: Saturating int32 compatibility telemetry.
        sample_count_words: Exact big-endian ``[high, low]`` uint32 sample
            identity. The cumulative float32 estimator accepts the event that
            reaches ``2**24`` and then fail-stops before any later mutation.
        p: Sum of squared deviations from the current mean (M2 accumulator)
    """

    mean: Float[Array, " feature_dim"]
    var: Float[Array, " feature_dim"]
    sample_count: Int[Array, ""]
    sample_count_words: UInt[Array, " 2"]
    p: Float[Array, " feature_dim"]


@chex.dataclass(frozen=True)
class StreamingBatchNormalizerState:
    """State for BatchNorm-style streaming feature normalization."""

    mean: Float[Array, " feature_dim"]
    var: Float[Array, " feature_dim"]
    sample_count: Int[Array, ""]
    sample_count_words: UInt[Array, " 2"]
    momentum: Float[Array, ""]


AnyNormalizerState = EMANormalizerState | WelfordNormalizerState | StreamingBatchNormalizerState


@chex.dataclass(frozen=True)
class NormalizerCounterStatus:
    """Read-only preflight status for one normalizer update transaction."""

    counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    estimator_capacity_available: Bool[Array, ""]
    update_available: Bool[Array, ""]


@chex.dataclass(frozen=True)
class NormalizerUpdateResult:
    """Normalizer output with explicit lifetime and estimator availability."""

    normalized: Float[Array, " feature_dim"]
    state: AnyNormalizerState
    pre_sample_count_words: UInt[Array, " 2"]
    post_sample_count_words: UInt[Array, " 2"]
    counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    estimator_capacity_available: Bool[Array, ""]
    update_applied: Bool[Array, ""]


class Normalizer[
    StateT: (EMANormalizerState, WelfordNormalizerState, StreamingBatchNormalizerState)
](ABC):
    """Abstract base class for online feature normalizers.

    Normalizes features using running estimates of mean and standard deviation:
    ``x_normalized = (x - mean) / (std + epsilon)``

    The normalizer updates its estimates at every time step, following
    temporal uniformity.

    Subclasses must implement ``init`` and ``normalize``. The ``normalize_only``
    and ``update_only`` methods have default implementations.

    Attributes:
        epsilon: Small constant for numerical stability
    """

    _epsilon: float

    def __init__(self, epsilon: float = 1e-8):
        """Initialize the normalizer.

        Args:
            epsilon: Small constant added to std for numerical stability
        """
        epsilon_type = type(epsilon)
        if issubclass(epsilon_type, bool) or not issubclass(epsilon_type, Real):
            raise TypeError("epsilon must be a finite real number")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        self._epsilon = float(epsilon)

    @abstractmethod
    def to_config(self) -> dict[str, Any]:
        """Serialize normalizer configuration to dict."""
        ...

    @abstractmethod
    def init(self, feature_dim: int) -> StateT:
        """Initialize normalizer state.

        Args:
            feature_dim: Dimension of feature vectors

        Returns:
            Initial normalizer state with zero mean and unit variance
        """
        ...

    def normalize(
        self,
        state: StateT,
        observation: Array,
    ) -> tuple[Array, StateT]:
        """Normalize observation and update running statistics.

        This method both normalizes the current observation AND updates
        the running statistics, maintaining temporal uniformity.

        Args:
            state: Current normalizer state
            observation: Raw feature vector

        Returns:
            Tuple of (normalized_observation, new_state)
        """
        result = self.normalize_with_diagnostics(state, observation)
        return result.normalized, result.state  # type: ignore[return-value]

    @abstractmethod
    def normalize_with_diagnostics(
        self,
        state: StateT,
        observation: Array,
    ) -> NormalizerUpdateResult:
        """Normalize with explicit clock validity and capacity diagnostics."""
        ...

    def counter_status(self, state: StateT) -> NormalizerCounterStatus:
        """Return preflight status without changing estimator state.

        Estimator parameters live in both the host-side normalizer and its
        persisted JAX state.  The two copies are deliberately required to
        agree after float32 storage; otherwise resuming a checkpoint with the
        wrong normalizer config could silently change both the estimator
        semantics and its lifetime horizon.
        """

        valid = _lifetime_counter_valid(
            state.sample_count_words,
            state.sample_count,
        )
        _, lifetime_capacity = _checked_lifetime_words_increment(state.sample_count_words)
        below_float32_cumulative_limit = (state.sample_count_words[0] == 0) & (
            state.sample_count_words[1]
            < jnp.asarray(
                _FLOAT32_CONSECUTIVE_INTEGER_LIMIT,
                dtype=jnp.uint32,
            )
        )
        estimator_capacity = jnp.asarray(True, dtype=jnp.bool_)
        if isinstance(state, WelfordNormalizerState):
            if not isinstance(self, WelfordNormalizer):
                raise TypeError(
                    "WelfordNormalizerState requires a WelfordNormalizer"
                )
            estimator_capacity = below_float32_cumulative_limit
        elif isinstance(state, EMANormalizerState):
            if not isinstance(self, EMANormalizer):
                raise TypeError("EMANormalizerState requires an EMANormalizer")
            if getattr(state.decay, "shape", None) != ():
                raise ValueError("persisted EMA decay must be scalar")
            if getattr(state.decay, "dtype", None) != jnp.dtype(jnp.float32):
                raise TypeError("persisted EMA decay must have dtype float32")
            decay_valid = jnp.isfinite(state.decay) & (state.decay >= 0.0) & (state.decay <= 1.0)
            configured_decay = jnp.asarray(self._decay, dtype=jnp.float32)
            valid = valid & decay_valid & (state.decay == configured_decay)
            estimator_capacity = jnp.where(
                state.decay == jnp.asarray(1.0, dtype=jnp.float32),
                below_float32_cumulative_limit,
                jnp.asarray(True, dtype=jnp.bool_),
            )
        elif isinstance(state, StreamingBatchNormalizerState):
            if not isinstance(self, StreamingBatchNormalizer):
                raise TypeError(
                    "StreamingBatchNormalizerState requires a StreamingBatchNormalizer"
                )
            if getattr(state.momentum, "shape", None) != ():
                raise ValueError("persisted streaming momentum must be scalar")
            if getattr(state.momentum, "dtype", None) != jnp.dtype(jnp.float32):
                raise TypeError("persisted streaming momentum must have dtype float32")
            momentum_valid = (
                jnp.isfinite(state.momentum) & (state.momentum >= 0.0) & (state.momentum <= 1.0)
            )
            configured_momentum = jnp.asarray(self._momentum, dtype=jnp.float32)
            valid = valid & momentum_valid & (state.momentum == configured_momentum)
        else:
            raise TypeError(f"unsupported normalizer state type: '{type(state).__name__}'")
        return NormalizerCounterStatus(
            counter_valid=valid,
            lifetime_capacity_available=lifetime_capacity,
            estimator_capacity_available=estimator_capacity,
            update_available=valid & lifetime_capacity & estimator_capacity,
        )

    def normalize_only(
        self,
        state: StateT,
        observation: Array,
    ) -> Array:
        """Normalize observation without updating statistics.

        Useful for inference or when you want to normalize multiple
        observations with the same statistics.

        Args:
            state: Current normalizer state
            observation: Raw feature vector

        Returns:
            Normalized observation
        """
        std = jnp.sqrt(state.var)
        normalized = (observation - state.mean) / (std + self._epsilon)
        # Inf observation minus an inf mean is inf-inf = NaN. Zero only
        # those coordinates. Finite-input overflow must stay visible.
        return jnp.where(
            jnp.isfinite(observation),
            normalized,
            jnp.zeros_like(normalized),
        )

    def update_only(
        self,
        state: StateT,
        observation: Array,
    ) -> StateT:
        """Update statistics without returning normalized observation.

        Args:
            state: Current normalizer state
            observation: Raw feature vector

        Returns:
            Updated normalizer state
        """
        _, new_state = self.normalize(state, observation)
        return new_state


class EMANormalizer(Normalizer[EMANormalizerState]):
    """Online feature normalizer using exponential moving average.

    Estimates mean and variance via EMA, suitable for non-stationary
    environments where recent observations should be weighted more heavily.

    The initialized zero mean and unit variance are one explicit prior
    pseudo-sample. The first effective decay is therefore
    ``min(decay, 1/2)`` and then ramps toward the target decay over early
    steps. ``decay < 1`` exponentially forgets that prior as a
    bounded-recency estimator and can run until the uint64 word clock is
    exhausted. ``decay == 1`` retains the prior with cumulative weight in
    float32: it is not an ordinary sample mean or a Welford estimator, and it
    accepts the event that reaches ``2**24`` before explicitly fail-stopping.

    Attributes:
        epsilon: Small constant for numerical stability
        decay: Exponential decay for running estimates (0.99 = slower adaptation)
    """

    _decay: float

    def __init__(
        self,
        epsilon: float = 1e-8,
        decay: float = 0.99,
    ):
        """Initialize the EMA normalizer.

        Args:
            epsilon: Small constant added to std for numerical stability
            decay: Exponential decay factor for running estimates. Lower
                values adapt faster to changes. ``1.0`` means cumulative
                prior-regularized moments with no exponential forgetting.
        """
        super().__init__(epsilon=epsilon)
        decay_type = type(decay)
        if issubclass(decay_type, bool) or not issubclass(decay_type, Real):
            raise TypeError("decay must be a finite real number")
        if not math.isfinite(decay):
            raise ValueError("decay must be finite")
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        self._decay = float(decay)

    def to_config(self) -> dict[str, Any]:
        """Serialize configuration to dict."""
        return {
            "type": "EMANormalizer",
            "state_schema": NORMALIZER_STATE_SCHEMA,
            "estimator_semantics": (
                EMA_PRIOR_CUMULATIVE_FLOAT32_ESTIMATOR_SEMANTICS
                if _stored_float32_equals_one(self._decay)
                else EMA_PRIOR_BOUNDED_RECENCY_ESTIMATOR_SEMANTICS
            ),
            "epsilon": self._epsilon,
            "decay": self._decay,
        }

    def init(self, feature_dim: int) -> EMANormalizerState:
        """Initialize EMA normalizer state.

        Args:
            feature_dim: Dimension of feature vectors

        Returns:
            Initial normalizer state whose zero mean and unit variance form
            the estimator's one explicit prior pseudo-sample.
        """
        feature_dim = _require_int(
            "feature_dim", feature_dim, minimum=1, maximum=_INT32_MAX
        )
        return EMANormalizerState(
            mean=jnp.zeros(feature_dim, dtype=jnp.float32),
            var=jnp.ones(feature_dim, dtype=jnp.float32),
            sample_count=jnp.array(0, dtype=jnp.int32),
            sample_count_words=jnp.zeros((2,), dtype=jnp.uint32),
            decay=jnp.array(self._decay, dtype=jnp.float32),
        )

    def normalize_with_diagnostics(
        self,
        state: EMANormalizerState,
        observation: Array,
    ) -> NormalizerUpdateResult:
        """Normalize and conditionally commit EMA running statistics.

        Args:
            state: Current EMA normalizer state
            observation: Raw feature vector

        Returns:
            Tuple of (normalized_observation, new_state)
        """
        status = self.counter_status(state)
        observation_valid = jnp.all(jnp.isfinite(observation))
        zero_decay = state.decay == jnp.asarray(0.0, dtype=jnp.float32)
        checked_state = EMANormalizerState(
            mean=jnp.where(zero_decay, jnp.zeros_like(state.mean), state.mean),
            var=jnp.where(zero_decay, jnp.ones_like(state.var), state.var),
            sample_count=state.sample_count,
            sample_count_words=state.sample_count_words,
            decay=state.decay,
        )
        source_state_finite = _floating_tree_is_finite(checked_state)
        update_available = (
            status.update_available & observation_valid & source_state_finite
        )

        def accepted(_: None) -> tuple[Array, EMANormalizerState]:
            new_words, ignored_capacity = _checked_lifetime_words_increment(
                state.sample_count_words
            )
            del ignored_capacity
            new_count_float = _lifetime_words_to_float32(new_words)
            effective_decay = jnp.minimum(
                state.decay,
                1.0 - 1.0 / (new_count_float + 1.0),
            )
            mean_for_update = jnp.where(
                zero_decay & ~jnp.isfinite(state.mean),
                observation,
                state.mean,
            )
            delta = observation - mean_for_update
            new_mean = mean_for_update + (1.0 - effective_decay) * delta
            delta2 = observation - new_mean
            var_for_update = jnp.where(
                zero_decay & ~jnp.isfinite(state.var),
                jnp.zeros_like(state.var),
                state.var,
            )
            new_var = (
                effective_decay * var_for_update
                + (1.0 - effective_decay) * delta * delta2
            )
            new_var = jnp.maximum(new_var, self._epsilon)
            normalized = (observation - new_mean) / (jnp.sqrt(new_var) + self._epsilon)
            return normalized, EMANormalizerState(
                mean=new_mean,
                var=new_var,
                sample_count=_saturating_int32_counter_increment(state.sample_count),
                sample_count_words=new_words,
                decay=state.decay,
            )

        def refused(_: None) -> tuple[Array, EMANormalizerState]:
            return self.normalize_only(state, observation), state

        candidate_normalized, candidate_state = jax.lax.cond(
            update_available,
            accepted,
            refused,
            operand=None,
        )
        update_applied = (
            update_available
            & jnp.all(jnp.isfinite(candidate_normalized))
            & _floating_tree_is_finite(candidate_state)
        )
        new_state = jax.lax.cond(
            update_applied,
            lambda: candidate_state,
            lambda: state,
        )
        normalized = jnp.where(
            update_applied | (observation_valid & source_state_finite),
            candidate_normalized,
            jnp.zeros_like(candidate_normalized),
        )
        return NormalizerUpdateResult(
            normalized=normalized,
            state=new_state,
            pre_sample_count_words=state.sample_count_words,
            post_sample_count_words=new_state.sample_count_words,
            counter_valid=status.counter_valid,
            lifetime_capacity_available=status.lifetime_capacity_available,
            estimator_capacity_available=status.estimator_capacity_available,
            update_applied=update_applied,
        )


class WelfordNormalizer(Normalizer[WelfordNormalizerState]):
    """Online feature normalizer using Welford's algorithm.

    Computes cumulative sample mean and variance with Bessel's correction for
    at most ``2**24`` samples.  Float32 cannot represent every later integer,
    so this estimator explicitly accepts the event reaching that boundary and
    then fail-stops as a bit-exact state no-op.  Use ``EMANormalizer`` or
    ``StreamingBatchNormalizer`` for an indefinitely running recency estimator.

    Reference: Welford 1962, "Note on a Method for Calculating Corrected
    Sums of Squares and Products"

    Attributes:
        epsilon: Small constant for numerical stability
    """

    def to_config(self) -> dict[str, Any]:
        """Serialize configuration to dict."""
        return {
            "type": "WelfordNormalizer",
            "state_schema": NORMALIZER_STATE_SCHEMA,
            "estimator_schema": WELFORD_ESTIMATOR_SCHEMA,
            "estimator_semantics": CUMULATIVE_FLOAT32_ESTIMATOR_SEMANTICS,
            "epsilon": self._epsilon,
        }

    def init(self, feature_dim: int) -> WelfordNormalizerState:
        """Initialize Welford normalizer state.

        Args:
            feature_dim: Dimension of feature vectors

        Returns:
            Initial normalizer state with zero mean and unit variance
        """
        feature_dim = _require_int(
            "feature_dim", feature_dim, minimum=1, maximum=_INT32_MAX
        )
        return WelfordNormalizerState(
            mean=jnp.zeros(feature_dim, dtype=jnp.float32),
            var=jnp.ones(feature_dim, dtype=jnp.float32),
            sample_count=jnp.array(0, dtype=jnp.int32),
            sample_count_words=jnp.zeros((2,), dtype=jnp.uint32),
            p=jnp.zeros(feature_dim, dtype=jnp.float32),
        )

    def normalize_with_diagnostics(
        self,
        state: WelfordNormalizerState,
        observation: Array,
    ) -> NormalizerUpdateResult:
        """Normalize and update while the cumulative estimator is exact.

        Uses Welford's online algorithm:
        1. Increment count
        2. Update mean incrementally
        3. Update sum of squared deviations (p / M2)
        4. Compute variance with Bessel's correction when count >= 2

        Args:
            state: Current Welford normalizer state
            observation: Raw feature vector

        Returns:
            Tuple of (normalized_observation, new_state)
        """
        status = self.counter_status(state)
        observation_valid = jnp.all(jnp.isfinite(observation))
        source_state_finite = _floating_tree_is_finite(state)
        update_available = (
            status.update_available & observation_valid & source_state_finite
        )

        def accepted(_: None) -> tuple[Array, WelfordNormalizerState]:
            new_words, ignored_capacity = _checked_lifetime_words_increment(
                state.sample_count_words
            )
            del ignored_capacity
            new_count_float = _lifetime_words_to_float32(new_words)
            delta = observation - state.mean
            new_mean = state.mean + delta / new_count_float
            delta2 = observation - new_mean
            new_p = state.p + delta * delta2
            new_var = jnp.where(
                new_count_float >= 2.0,
                new_p / (new_count_float - 1.0),
                jnp.ones_like(new_p),
            )
            normalized = (observation - new_mean) / (jnp.sqrt(new_var) + self._epsilon)
            return normalized, WelfordNormalizerState(
                mean=new_mean,
                var=new_var,
                sample_count=_saturating_int32_counter_increment(state.sample_count),
                sample_count_words=new_words,
                p=new_p,
            )

        def refused(_: None) -> tuple[Array, WelfordNormalizerState]:
            return self.normalize_only(state, observation), state

        candidate_normalized, candidate_state = jax.lax.cond(
            update_available,
            accepted,
            refused,
            operand=None,
        )
        update_applied = (
            update_available
            & jnp.all(jnp.isfinite(candidate_normalized))
            & _floating_tree_is_finite(candidate_state)
        )
        new_state = jax.lax.cond(
            update_applied,
            lambda: candidate_state,
            lambda: state,
        )
        normalized = jnp.where(
            update_applied | (observation_valid & source_state_finite),
            candidate_normalized,
            jnp.zeros_like(candidate_normalized),
        )
        return NormalizerUpdateResult(
            normalized=normalized,
            state=new_state,
            pre_sample_count_words=state.sample_count_words,
            post_sample_count_words=new_state.sample_count_words,
            counter_valid=status.counter_valid,
            lifetime_capacity_available=status.lifetime_capacity_available,
            estimator_capacity_available=status.estimator_capacity_available,
            update_applied=update_applied,
        )


class StreamingBatchNormalizer(Normalizer[StreamingBatchNormalizerState]):
    """BatchNorm-style online normalizer with exponential running moments.

    This is the single-sample streaming analogue of BatchNorm's running
    statistics: each time step is treated as a batch of one, with running mean
    and variance updated by ``momentum``. Unlike :class:`EMANormalizer`, this
    intentionally has no warmup schedule, making it a distinct Step 1
    normalization comparator.  ``momentum < 1`` is a bounded-recency
    estimator.  ``momentum == 1`` is retained as an intentionally static
    comparator: it initializes on the first sample and never adapts again.
    """

    _momentum: float

    def __init__(self, epsilon: float = 1e-5, momentum: float = 0.99):
        """Initialize the streaming BatchNorm-style normalizer."""
        super().__init__(epsilon=epsilon)
        momentum_type = type(momentum)
        if issubclass(momentum_type, bool) or not issubclass(momentum_type, Real):
            raise TypeError("momentum must be a finite real number")
        if not math.isfinite(momentum):
            raise ValueError("momentum must be finite")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be in [0, 1]")
        self._momentum = float(momentum)

    def to_config(self) -> dict[str, Any]:
        """Serialize configuration to dict."""
        return {
            "type": "StreamingBatchNormalizer",
            "state_schema": NORMALIZER_STATE_SCHEMA,
            "estimator_semantics": (
                STATIC_AFTER_FIRST_ESTIMATOR_SEMANTICS
                if _stored_float32_equals_one(self._momentum)
                else BOUNDED_RECENCY_ESTIMATOR_SEMANTICS
            ),
            "epsilon": self._epsilon,
            "momentum": self._momentum,
        }

    def init(self, feature_dim: int) -> StreamingBatchNormalizerState:
        """Initialize running moments."""
        feature_dim = _require_int(
            "feature_dim", feature_dim, minimum=1, maximum=_INT32_MAX
        )
        return StreamingBatchNormalizerState(
            mean=jnp.zeros(feature_dim, dtype=jnp.float32),
            var=jnp.ones(feature_dim, dtype=jnp.float32),
            sample_count=jnp.array(0, dtype=jnp.int32),
            sample_count_words=jnp.zeros((2,), dtype=jnp.uint32),
            momentum=jnp.array(self._momentum, dtype=jnp.float32),
        )

    def normalize_with_diagnostics(
        self,
        state: StreamingBatchNormalizerState,
        observation: Array,
    ) -> NormalizerUpdateResult:
        """Normalize and conditionally commit BatchNorm-style moments."""
        status = self.counter_status(state)
        observation_valid = jnp.all(jnp.isfinite(observation))
        zero_momentum = state.momentum == jnp.asarray(0.0, dtype=jnp.float32)
        checked_state = StreamingBatchNormalizerState(
            mean=state.mean,
            var=jnp.where(zero_momentum, jnp.ones_like(state.var), state.var),
            sample_count=state.sample_count,
            sample_count_words=state.sample_count_words,
            momentum=state.momentum,
        )
        source_state_finite = _floating_tree_is_finite(checked_state)
        update_available = (
            status.update_available & observation_valid & source_state_finite
        )

        def accepted(_: None) -> tuple[Array, StreamingBatchNormalizerState]:
            new_words, ignored_capacity = _checked_lifetime_words_increment(
                state.sample_count_words
            )
            del ignored_capacity
            is_first = _lifetime_words_are_zero(state.sample_count_words)
            one_minus_m = 1.0 - state.momentum
            candidate_mean = state.momentum * state.mean + one_minus_m * observation
            new_mean = jnp.where(is_first, observation, candidate_mean)
            centered = observation - state.mean
            var_for_update = jnp.where(
                zero_momentum & ~jnp.isfinite(state.var),
                jnp.zeros_like(state.var),
                state.var,
            )
            candidate_var = (
                state.momentum * var_for_update
                + one_minus_m * (centered * centered)
            )
            new_var = jnp.where(
                is_first,
                jnp.ones_like(state.var),
                candidate_var,
            )
            new_var = jnp.maximum(new_var, self._epsilon)
            normalized = (observation - new_mean) / (jnp.sqrt(new_var) + self._epsilon)
            return normalized, StreamingBatchNormalizerState(
                mean=new_mean,
                var=new_var,
                sample_count=_saturating_int32_counter_increment(state.sample_count),
                sample_count_words=new_words,
                momentum=state.momentum,
            )

        def refused(_: None) -> tuple[Array, StreamingBatchNormalizerState]:
            return self.normalize_only(state, observation), state

        candidate_normalized, candidate_state = jax.lax.cond(
            update_available,
            accepted,
            refused,
            operand=None,
        )
        update_applied = (
            update_available
            & jnp.all(jnp.isfinite(candidate_normalized))
            & _floating_tree_is_finite(candidate_state)
        )
        new_state = jax.lax.cond(
            update_applied,
            lambda: candidate_state,
            lambda: state,
        )
        normalized = jnp.where(
            update_applied | (observation_valid & source_state_finite),
            candidate_normalized,
            jnp.zeros_like(candidate_normalized),
        )
        return NormalizerUpdateResult(
            normalized=normalized,
            state=new_state,
            pre_sample_count_words=state.sample_count_words,
            post_sample_count_words=new_state.sample_count_words,
            counter_valid=status.counter_valid,
            lifetime_capacity_available=status.lifetime_capacity_available,
            estimator_capacity_available=status.estimator_capacity_available,
            update_applied=update_applied,
        )


def normalizer_state_nbytes_formula(
    normalizer_type: object,
    feature_dim: object,
) -> int:
    """Return exact persistent JAX-array bytes for a normalizer state."""

    host_type = _require_exact_str("normalizer_type", normalizer_type)
    feature_dim = _require_int(
        "feature_dim", feature_dim, minimum=1, maximum=_INT32_MAX
    )
    if host_type in {"EMANormalizer", "StreamingBatchNormalizer"}:
        return 8 * feature_dim + 16
    if host_type == "WelfordNormalizer":
        return 12 * feature_dim + 12
    raise ValueError(f"Unknown normalizer type: '{host_type}'")


def measure_normalizer_state_nbytes(state: AnyNormalizerState) -> int:
    """Measure persistent JAX-array bytes in one concrete normalizer state."""

    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(state)
        if isinstance(leaf, Array)
    )


def _legacy_state_mapping(legacy_state: Any) -> dict[str, Any]:
    """Return a shallow field mapping for one host-side legacy dataclass."""

    if isinstance(legacy_state, Mapping):
        return dict(legacy_state)
    if dataclasses.is_dataclass(legacy_state) and not isinstance(legacy_state, type):
        return {
            field.name: getattr(legacy_state, field.name)
            for field in dataclasses.fields(legacy_state)
        }
    raise TypeError("legacy normalizer state must be a mapping or dataclass instance")


def migrate_legacy_normalizer_state(
    legacy_state: Any,
    *,
    normalizer_type: object,
) -> AnyNormalizerState:
    """Migrate an exact pre-v2 normalizer state on the host.

    Legacy float32 counts at or above ``2**24`` are ambiguous: the stored
    value may represent that exact event or any number of later updates lost
    to float32 rounding.  Such states, as well as negative, non-finite, or
    fractional counts, are rejected rather than assigned invented history.
    """

    classes: dict[str, type[AnyNormalizerState]] = {
        "EMANormalizer": EMANormalizerState,
        "WelfordNormalizer": WelfordNormalizerState,
        "StreamingBatchNormalizer": StreamingBatchNormalizerState,
    }
    host_type = _require_exact_str("normalizer_type", normalizer_type)
    state_class = classes.get(host_type)
    if state_class is None:
        raise ValueError(f"Unknown normalizer type: '{host_type}'")
    fields = _legacy_state_mapping(legacy_state)
    current_names = {
        field.name
        for field in dataclasses.fields(state_class)  # type: ignore[arg-type]
    }
    legacy_names = current_names - {"sample_count_words"}
    supplied_names = set(fields)
    if supplied_names != legacy_names:
        missing = sorted(legacy_names - supplied_names)
        extra = sorted(supplied_names - legacy_names)
        raise ValueError(
            f"legacy normalizer field manifest is not exact; missing={missing}, extra={extra}"
        )
    count_array = jnp.asarray(fields["sample_count"])
    if count_array.shape != () or count_array.dtype != jnp.dtype(jnp.float32):
        raise TypeError("legacy normalizer sample_count must be scalar float32")
    count_float = float(count_array)
    if not count_float.is_integer() or count_float < 0.0:
        raise ValueError("legacy normalizer sample_count is negative or fractional")
    if count_float >= _FLOAT32_CONSECUTIVE_INTEGER_LIMIT:
        raise ValueError("legacy normalizer sample_count at or above 2**24 has ambiguous history")
    count = int(count_float)
    fields["sample_count"] = jnp.asarray(count, dtype=jnp.int32)
    fields["sample_count_words"] = jnp.asarray((0, count), dtype=jnp.uint32)
    return state_class(**fields)


# =============================================================================
# Config serialization dispatcher
# =============================================================================

_NORMALIZER_REGISTRY: dict[str, type] = {
    "EMANormalizer": EMANormalizer,
    "StreamingBatchNormalizer": StreamingBatchNormalizer,
    "WelfordNormalizer": WelfordNormalizer,
}


def normalizer_from_config(config: dict[str, Any]) -> Normalizer[Any]:
    """Reconstruct a normalizer from a config dict.

    Args:
        config: Dict with ``"type"`` key and constructor kwargs

    Returns:
        Reconstructed normalizer instance

    Raises:
        ValueError: If the normalizer type is unknown
    """
    config = dict(config)
    type_name = config.pop("type")
    host_type_name = _require_exact_str("type_name", type_name)
    state_schema = config.pop("state_schema", NORMALIZER_STATE_SCHEMA)
    if state_schema != NORMALIZER_STATE_SCHEMA:
        host_schema = _require_exact_str("state_schema", state_schema)
        raise ValueError(f"Unsupported normalizer state schema: '{host_schema}'")
    estimator_schema = config.pop("estimator_schema", None)
    estimator_semantics = config.pop("estimator_semantics", None)
    legacy_estimator_semantics: str | None = None
    if host_type_name == "WelfordNormalizer":
        if estimator_schema not in {None, WELFORD_ESTIMATOR_SCHEMA}:
            host_est = _require_exact_str("estimator_schema", estimator_schema)
            raise ValueError(f"Unsupported Welford estimator schema: '{host_est}'")
        expected_semantics = CUMULATIVE_FLOAT32_ESTIMATOR_SEMANTICS
    elif estimator_schema is not None:
        raise ValueError(f"estimator_schema is not valid for normalizer type '{host_type_name}'")
    elif host_type_name == "EMANormalizer":
        if _stored_float32_equals_one(config.get("decay", 0.99)):
            expected_semantics = EMA_PRIOR_CUMULATIVE_FLOAT32_ESTIMATOR_SEMANTICS
            legacy_estimator_semantics = CUMULATIVE_FLOAT32_ESTIMATOR_SEMANTICS
        else:
            expected_semantics = EMA_PRIOR_BOUNDED_RECENCY_ESTIMATOR_SEMANTICS
            legacy_estimator_semantics = BOUNDED_RECENCY_ESTIMATOR_SEMANTICS
    elif host_type_name == "StreamingBatchNormalizer":
        expected_semantics = (
            STATIC_AFTER_FIRST_ESTIMATOR_SEMANTICS
            if _stored_float32_equals_one(config.get("momentum", 0.99))
            else BOUNDED_RECENCY_ESTIMATOR_SEMANTICS
        )
    else:
        expected_semantics = None
    if estimator_semantics is not None and (
        type(estimator_semantics) is not str
        or (
            estimator_semantics != expected_semantics
            and estimator_semantics != legacy_estimator_semantics
        )
    ):
        if type(estimator_semantics) is str:
            host_est_sem = _require_exact_str("estimator_semantics", estimator_semantics)
            host_exp = expected_semantics if type(expected_semantics) is str else "None"
            # expected is internal constant, safe to treat as exact str via host check
            if type(expected_semantics) is str:
                host_exp = _require_exact_str("expected_semantics", expected_semantics)
            raise ValueError(
                "normalizer estimator_semantics does not match its parameters: "
                f"expected '{host_exp}', got '{host_est_sem}'"
            )
        raise ValueError(
            "normalizer estimator_semantics does not match its parameters: "
            "expected exact string, got non-string"
        )
    cls = _NORMALIZER_REGISTRY.get(host_type_name)
    if cls is None:
        raise ValueError(f"Unknown normalizer type: '{host_type_name}'")
    result: Normalizer[Any] = cls(**config)
    return result


__all__ = [
    "BOUNDED_RECENCY_ESTIMATOR_SEMANTICS",
    "CUMULATIVE_FLOAT32_ESTIMATOR_SEMANTICS",
    "EMA_PRIOR_BOUNDED_RECENCY_ESTIMATOR_SEMANTICS",
    "EMA_PRIOR_CUMULATIVE_FLOAT32_ESTIMATOR_SEMANTICS",
    "NORMALIZER_LIFETIME_COUNTER_DELTA_NBYTES",
    "NORMALIZER_LIFETIME_COUNTER_NBYTES",
    "NORMALIZER_STATE_SCHEMA",
    "STATIC_AFTER_FIRST_ESTIMATOR_SEMANTICS",
    "WELFORD_ESTIMATOR_SCHEMA",
    "AnyNormalizerState",
    "EMANormalizer",
    "EMANormalizerState",
    "Normalizer",
    "NormalizerCounterStatus",
    "NormalizerUpdateResult",
    "StreamingBatchNormalizer",
    "StreamingBatchNormalizerState",
    "WelfordNormalizer",
    "WelfordNormalizerState",
    "measure_normalizer_state_nbytes",
    "migrate_legacy_normalizer_state",
    "normalizer_from_config",
    "normalizer_state_nbytes_formula",
]
