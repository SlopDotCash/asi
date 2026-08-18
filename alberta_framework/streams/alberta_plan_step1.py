"""Streams that exercise the Alberta Plan Step 1 supervised-learning spec.

The Alberta Plan Step 1 specifies a non-stationary supervised problem in which
the desired output is

    y*_t = w*_t . x_t + b*_t + eta_t

with ``eta_t`` an independent mean-zero noise signal. The problem is non-
stationary if ``w*_t`` or ``b*_t`` change over time, OR if the distribution of
``x_t`` changes over time. This module provides two streams that cover the
two non-stationarity cases:

* :class:`AlbertaPlanStep1Stream` — the canonical Step 1 task: ``w*_t`` and
  ``b*_t`` follow Gaussian random walks and ``eta_t`` is included.
* :class:`XDistShiftStream` — fixes the target function and shifts only the
  input distribution (per-feature scales redrawn at fixed intervals).

Both streams follow the :class:`~alberta_framework.streams.base.ScanStream`
protocol and are JIT-friendly (no Python control flow on traced values).

Reference: Sutton et al., "The Alberta Plan for AI Research", Step 1.
"""

import math
from numbers import Integral, Real
from typing import cast

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Float, Int, PRNGKeyArray

from alberta_framework._float32 import round_real_to_float32_with_ratio
from alberta_framework.core.normalizers import _saturating_int32_counter_increment
from alberta_framework.core.types import TimeStep

_INT32_MAX = 2**31 - 1
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


def _finite_real_and_float32(name: str, value: object) -> tuple[Real, int, int, float]:
    """Return the original real, exact ratio, and finite binary32 rounding."""
    actual_type = type(value)
    if issubclass(actual_type, bool) or not issubclass(actual_type, Real):
        raise ValueError(f"{name} must be a real number")
    real = cast(Real, value)
    try:
        numerator, denominator, narrowed = round_real_to_float32_with_ratio(real)
    except (FloatingPointError, OverflowError, TypeError, ValueError):
        raise ValueError(f"{name} must narrow to a finite float32") from None
    if not math.isfinite(narrowed):
        raise ValueError(f"{name} must narrow to a finite float32")
    return real, numerator, denominator, narrowed


def _canonical_float32_storage(value: Real, narrowed: float) -> float:
    if not isinstance(value, (int, float, np.floating)):
        return narrowed
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return narrowed
    if not math.isfinite(number):
        raise ValueError("scalar must be finite")
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        renarrowed = np.asarray(number, dtype=np.float32)
    if not bool(np.array_equal(narrowed, renarrowed)):
        number = float(narrowed)
    return number


def _require_nonnegative_real(name: str, value: object) -> float:
    real, numerator, _, narrowed = _finite_real_and_float32(name, value)
    if real < 0.0 or numerator < 0 or narrowed < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return _canonical_float32_storage(real, narrowed)


def _require_positive_real(name: str, value: object) -> float:
    real, numerator, _, narrowed = _finite_real_and_float32(name, value)
    if real <= 0.0 or numerator <= 0 or narrowed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return _canonical_float32_storage(real, narrowed)


def _require_int(
    name: str,
    value: object,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    actual_type = type(value)
    if actual_type not in _ACTUAL_INT_TYPES:
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


@chex.dataclass(frozen=True)
class AlbertaPlanStep1State:
    """State for :class:`AlbertaPlanStep1Stream`.

    Attributes:
        key: JAX random key for generating randomness
        true_weights: Current target weight vector ``w*_t`` (only the first
            ``num_relevant`` entries are nonzero; the rest stay at zero)
        true_bias: Current scalar target bias ``b*_t``
        step_count: Number of steps taken so far
    """

    key: PRNGKeyArray
    true_weights: Float[Array, " feature_dim"]
    true_bias: Float[Array, ""]
    step_count: Int[Array, ""]


class AlbertaPlanStep1Stream:
    """Canonical Alberta Plan Step 1 supervised stream.

    Generates targets

        y*_t = w*_t . x_t + b*_t + eta_t,    eta_t ~ N(0, noise_std^2)

    where the first ``num_relevant`` entries of ``w*_t`` follow independent
    Gaussian random walks with std ``drift_rate_w`` per step (the remaining
    entries stay identically zero, mirroring the Sutton 1992 sparse-relevance
    setup), and ``b*_t`` follows a Gaussian random walk with std
    ``drift_rate_b`` per step. Inputs ``x_t`` are drawn iid from
    ``N(0, feature_std^2)``.

    With ``drift_rate_w = drift_rate_b = 0.0`` this becomes a stationary
    target with additive observation noise; the stream is non-stationary
    whenever either drift rate is positive.

    Attributes:
        feature_dim: Dimension of observation vectors (default 20)
        num_relevant: Number of relevant inputs whose weights are nonzero
            (default 5)
        drift_rate_w: Std dev of Gaussian random walk on the relevant
            entries of ``w*_t`` per step (default 0.001)
        drift_rate_b: Std dev of Gaussian random walk on ``b*_t`` per step
            (default 0.001)
        noise_std: Std dev of additive mean-zero target noise ``eta_t``
            (default 1.0)
        feature_std: Std dev of input features ``x_t`` (default 1.0)
    """

    def __init__(
        self,
        feature_dim: int = 20,
        num_relevant: int = 5,
        drift_rate_w: float = 0.001,
        drift_rate_b: float = 0.001,
        noise_std: float = 1.0,
        feature_std: float = 1.0,
    ):
        """Initialize the Alberta Plan Step 1 stream.

        Args:
            feature_dim: Dimension of feature vectors
            num_relevant: Number of relevant inputs (must be <= feature_dim)
            drift_rate_w: Std dev of weight drift per step
            drift_rate_b: Std dev of bias drift per step
            noise_std: Std dev of additive target noise
            feature_std: Std dev of input features

        Raises:
            ValueError: If ``num_relevant > feature_dim`` or either is
                non-positive.
        """
        feature_dim_val = _require_int("feature_dim", feature_dim, minimum=1, maximum=_INT32_MAX)
        num_relevant_val = _require_int("num_relevant", num_relevant, minimum=1, maximum=_INT32_MAX)
        if num_relevant_val > feature_dim_val:
            raise ValueError(
                f"num_relevant ({num_relevant_val}) must not exceed "
                f"feature_dim ({feature_dim_val})"
            )
        drift_rate_w_val = _require_nonnegative_real("drift_rate_w", drift_rate_w)
        drift_rate_b_val = _require_nonnegative_real("drift_rate_b", drift_rate_b)
        noise_std_val = _require_nonnegative_real("noise_std", noise_std)
        feature_std_val = _require_positive_real("feature_std", feature_std)

        self._feature_dim = feature_dim_val
        self._num_relevant = num_relevant_val
        self._drift_rate_w = drift_rate_w_val
        self._drift_rate_b = drift_rate_b_val
        self._noise_std = noise_std_val
        self._feature_std = feature_std_val

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        return self._feature_dim

    @property
    def num_relevant(self) -> int:
        """Return the number of relevant input dimensions."""
        return self._num_relevant

    @property
    def drift_rate_w(self) -> float:
        """Return the weight drift rate."""
        return self._drift_rate_w

    @property
    def drift_rate_b(self) -> float:
        """Return the bias drift rate."""
        return self._drift_rate_b

    @property
    def noise_std(self) -> float:
        """Return the target noise standard deviation."""
        return self._noise_std

    @property
    def feature_std(self) -> float:
        """Return the feature standard deviation."""
        return self._feature_std

    def init(self, key: Array) -> AlbertaPlanStep1State:
        """Initialize stream state.

        Args:
            key: JAX random key

        Returns:
            Initial stream state with random relevant weights and zero bias.
        """
        key, k_init = jr.split(key)
        relevant_init = jr.normal(k_init, (self._num_relevant,), dtype=jnp.float32)
        weights = jnp.zeros(self._feature_dim, dtype=jnp.float32)
        weights = weights.at[: self._num_relevant].set(relevant_init)
        return AlbertaPlanStep1State(
            key=key,
            true_weights=weights,
            true_bias=jnp.array(0.0, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def step(
        self, state: AlbertaPlanStep1State, idx: Array
    ) -> tuple[TimeStep, AlbertaPlanStep1State]:
        """Generate one time step.

        Args:
            state: Current stream state
            idx: Current step index (unused)

        Returns:
            Tuple of (timestep, new_state)
        """
        del idx  # unused
        key, k_w_drift, k_b_drift, k_x, k_eta = jr.split(state.key, 5)

        # Random walk on the relevant slice of w*_t. Use a full-length zero
        # vector with a relevant-only update so the irrelevant entries never
        # leave zero (preserves the sparse-relevance setting).
        relevant_drift = self._drift_rate_w * jr.normal(
            k_w_drift, (self._num_relevant,), dtype=jnp.float32
        )
        weight_drift = jnp.zeros(self._feature_dim, dtype=jnp.float32)
        weight_drift = weight_drift.at[: self._num_relevant].set(relevant_drift)
        new_weights = state.true_weights + weight_drift

        # Random walk on b*_t.
        bias_drift = self._drift_rate_b * jr.normal(k_b_drift, (), dtype=jnp.float32)
        new_bias = state.true_bias + bias_drift

        # Sample input features.
        x = self._feature_std * jr.normal(k_x, (self._feature_dim,), dtype=jnp.float32)

        # Compute target: y* = w* . x + b* + eta.
        eta = self._noise_std * jr.normal(k_eta, (), dtype=jnp.float32)
        target = jnp.dot(new_weights, x) + new_bias + eta

        timestep = TimeStep(observation=x, target=jnp.atleast_1d(target))
        new_state = AlbertaPlanStep1State(
            key=key,
            true_weights=new_weights,
            true_bias=new_bias,
            step_count=_saturating_int32_counter_increment(state.step_count),
        )
        return timestep, new_state


@chex.dataclass(frozen=True)
class XDistShiftState:
    """State for :class:`XDistShiftStream`.

    Attributes:
        key: JAX random key for generating randomness
        true_weights: Fixed target weight vector (only the first
            ``num_relevant`` entries are nonzero; sampled once at init)
        current_scales: Current per-feature scale vector ``s_t``
        step_count: Number of steps taken so far
    """

    key: PRNGKeyArray
    true_weights: Float[Array, " feature_dim"]
    current_scales: Float[Array, " feature_dim"]
    step_count: Int[Array, ""]


class XDistShiftStream:
    """Step 1 stream that holds the target fixed and shifts the x distribution.

    Implements the third Step 1 case from the Alberta Plan: "The problem is
    non-stationary if w*_t or b*_t change over time OR if the distribution of
    x_t changes over time." This stream isolates input-distribution
    non-stationarity from target non-stationarity.

    The TARGET function is fixed: ``w*`` is sampled once at ``init`` and never
    changes. Only the INPUT distribution shifts: every
    ``scale_change_interval`` steps a new per-feature scale vector
    ``s ~ Uniform[scale_min, scale_max]`` is drawn, and observations are

        x_t = s * z_t,    z_t ~ N(0, 1)

    The target is computed from the SCALED observation ``x_t`` so that the
    learner sees the same (observation, target) relationship the underlying
    affine map describes; the scale changes induce non-stationarity through
    the distribution of features (their variances and norms) rather than
    through the target function itself.

    Attributes:
        feature_dim: Dimension of observation vectors
        num_relevant: Number of relevant inputs whose weights are nonzero
        noise_std: Std dev of additive mean-zero target noise (default 0.1).
            Only added to the target when ``noise_in_target=True``.
        scale_change_interval: Steps between scale resamplings (default 2000)
        scale_min: Minimum per-feature scale, inclusive (default 0.1)
        scale_max: Maximum per-feature scale, exclusive (default 10.0)
        noise_in_target: If True, add Gaussian noise to the target. If False,
            the target is exactly ``w* . x``.
    """

    def __init__(
        self,
        feature_dim: int,
        num_relevant: int,
        noise_std: float = 0.1,
        scale_change_interval: int = 2000,
        scale_min: float = 0.1,
        scale_max: float = 10.0,
        noise_in_target: bool = True,
    ):
        """Initialize the x-distribution-shift stream.

        Args:
            feature_dim: Dimension of feature vectors
            num_relevant: Number of relevant inputs (must be <= feature_dim)
            noise_std: Std dev of additive target noise (only used if
                ``noise_in_target=True``)
            scale_change_interval: Steps between abrupt scale resamplings
            scale_min: Lower bound of uniform scale distribution
            scale_max: Upper bound of uniform scale distribution
            noise_in_target: Whether to add Gaussian noise to the target

        Raises:
            ValueError: If ``num_relevant > feature_dim``,
                ``scale_min >= scale_max``, ``scale_change_interval <= 0``, or
                if ``feature_dim`` / ``num_relevant`` are non-positive.
        """
        feature_dim_val = _require_int("feature_dim", feature_dim, minimum=1, maximum=_INT32_MAX)
        num_relevant_val = _require_int("num_relevant", num_relevant, minimum=1, maximum=_INT32_MAX)
        if num_relevant_val > feature_dim_val:
            raise ValueError(
                f"num_relevant ({num_relevant_val}) must not exceed "
                f"feature_dim ({feature_dim_val})"
            )
        noise_std_val = _require_nonnegative_real("noise_std", noise_std)
        scale_change_interval_val = _require_int(
            "scale_change_interval",
            scale_change_interval,
            minimum=1,
            maximum=_INT32_MAX,
        )
        scale_min_val = _require_positive_real("scale_min", scale_min)
        scale_max_val = _require_positive_real("scale_max", scale_max)
        if scale_min_val >= scale_max_val:
            raise ValueError(
                f"scale_min ({scale_min_val}) must be less than scale_max ({scale_max_val})"
            )
        narrowed_scale_min = float(np.float32(scale_min_val))
        narrowed_scale_max = float(np.float32(scale_max_val))
        if narrowed_scale_min >= narrowed_scale_max:
            raise ValueError(
                f"scale_min ({scale_min_val}) must remain less than "
                f"scale_max ({scale_max_val}) after float32 narrowing; both narrow "
                f"to [{narrowed_scale_min}, {narrowed_scale_max}]"
            )
        if type(noise_in_target) is bool or type(noise_in_target) is np.bool_:
            noise_in_target_val = bool(noise_in_target)
        else:
            raise TypeError("noise_in_target must be a boolean")

        self._feature_dim = feature_dim_val
        self._num_relevant = num_relevant_val
        self._noise_std = noise_std_val
        self._scale_change_interval = scale_change_interval_val
        self._scale_min = scale_min_val
        self._scale_max = scale_max_val
        self._noise_in_target = noise_in_target_val

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        return self._feature_dim

    @property
    def num_relevant(self) -> int:
        """Return the number of relevant input dimensions."""
        return self._num_relevant

    @property
    def noise_std(self) -> float:
        """Return the target noise standard deviation."""
        return self._noise_std

    @property
    def scale_change_interval(self) -> int:
        """Return the scale change interval."""
        return self._scale_change_interval

    @property
    def scale_min(self) -> float:
        """Return the minimum scale."""
        return self._scale_min

    @property
    def scale_max(self) -> float:
        """Return the maximum scale."""
        return self._scale_max

    @property
    def noise_in_target(self) -> bool:
        """Return whether target noise is enabled."""
        return self._noise_in_target

    def init(self, key: Array) -> XDistShiftState:
        """Initialize stream state.

        Args:
            key: JAX random key

        Returns:
            Initial stream state with a fixed target function and an initial
            per-feature scale vector.
        """
        key, k_w, k_scales = jr.split(key, 3)
        relevant_w = jr.normal(k_w, (self._num_relevant,), dtype=jnp.float32)
        weights = jnp.zeros(self._feature_dim, dtype=jnp.float32)
        weights = weights.at[: self._num_relevant].set(relevant_w)
        initial_scales = jr.uniform(
            k_scales,
            (self._feature_dim,),
            minval=self._scale_min,
            maxval=self._scale_max,
            dtype=jnp.float32,
        )
        return XDistShiftState(
            key=key,
            true_weights=weights,
            current_scales=initial_scales,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def step(self, state: XDistShiftState, idx: Array) -> tuple[TimeStep, XDistShiftState]:
        """Generate one time step.

        Args:
            state: Current stream state
            idx: Current step index (unused)

        Returns:
            Tuple of (timestep, new_state)
        """
        del idx  # unused
        key, k_scales, k_z, k_eta = jr.split(state.key, 4)

        # Decide whether to redraw scales this step. Always sample candidate
        # scales (jit-friendly) and use jnp.where to commit conditionally.
        should_change = (state.step_count > 0) & (
            state.step_count % self._scale_change_interval == 0
        )
        candidate_scales = jr.uniform(
            k_scales,
            (self._feature_dim,),
            minval=self._scale_min,
            maxval=self._scale_max,
            dtype=jnp.float32,
        )
        new_scales = jnp.where(should_change, candidate_scales, state.current_scales)

        # Sample latent z ~ N(0, 1), then form x = s * z.
        z = jr.normal(k_z, (self._feature_dim,), dtype=jnp.float32)
        x = new_scales * z

        # Compute target from the SCALED observation.
        target = jnp.dot(state.true_weights, x)
        eta = self._noise_std * jr.normal(k_eta, (), dtype=jnp.float32)
        target = target + jnp.where(self._noise_in_target, eta, jnp.float32(0.0))

        timestep = TimeStep(observation=x, target=jnp.atleast_1d(target))
        new_state = XDistShiftState(
            key=key,
            true_weights=state.true_weights,
            current_scales=new_scales,
            step_count=_saturating_int32_counter_increment(state.step_count),
        )
        return timestep, new_state
