"""Off-policy TD learners with importance sampling.

Implements per-decision importance sampling with optional ratio clipping for
off-policy linear value-function learning.

Theoretical background:
    TD with linear function approximation is **not** guaranteed to
    converge under off-policy distributions (Baird 1995, Counterexample
    to TD with FA). Several remedies exist:

    1. Per-decision importance sampling (Precup, Sutton, Singh 2000):
       include each rho_t = pi(a_t|s_t) / b(a_t|s_t) once in the
       eligibility-trace recursion so that on average we are simulating the
       on-policy distribution. Variance can be very large.
    2. Importance-ratio clipping: use ``min(c, rho_t)`` to bound individual
       updates at the cost of bias. This is not the multi-step Retrace
       operator of Munos et al. (2016).
    3. Gradient-TD (TDC, GQ-lambda) (Sutton, Maei, et al. 2009-2010):
       gradient descent on the projected Bellman error.
    4. Emphatic TD (Sutton, Mahmood, White 2016): emphasis traces F_t
       restore on-policy convergence proofs without a secondary weight
       vector.

    This module implements (1), (2), TDC-style Gradient-TD from (3) via
    :class:`GradientTDLinearLearner` (which maintains the required secondary
    weight vector), and ETD(lambda) from (4).

The learner has a simple interface::

    learner = OffPolicyTDLinearLearner(step_size=0.05, retrace_clip=1.0)
    state = learner.init(feature_dim)
    for t in range(T):
        rho_t = pi(a_t | s_t) / b(a_t | s_t)
        result = learner.update(state, obs_t, reward, next_obs, gamma, rho_t)
        state = result.state

Setting ``rho_t = 1.0`` reduces this to standard semi-gradient TD(0).

Use cases (Step 3 DoD-5):
    - Counterfactual prediction: "what would value be under target policy?"
    - Auxiliary Horde demons learning about hand-specified target policies.
    - Baird counterexample / divergence-prevention demonstrations.

Reference:
    Precup, D., Sutton, R.S., & Singh, S. (2000). Eligibility traces for
    off-policy policy evaluation. *ICML*.
    Munos, R., Stepleton, T., Harutyunyan, A., & Bellemare, M. (2016).
    Safe and efficient off-policy reinforcement learning. *NeurIPS*.
"""

from __future__ import annotations

import functools
import math
import operator
import time
from collections.abc import Mapping
from fractions import Fraction
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

from alberta_framework.core._float32_scalars import validated_float32_scalar
from alberta_framework.core.types import Observation
from alberta_framework.core.update_safety import (
    floating_tree_is_finite as _floating_tree_is_finite,
)

_INT32_MAX = 2**31 - 1
# Public last-fit is the 600-step off-policy positive control. Leftover INT32
# still admits T=10_000_000 (feature_dim=2) into jax.lax.scan.
_MAX_LEARNING_LOOP_STEPS = 10_000
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
_ACTUAL_FLOAT_TYPES = frozenset(
    {float, Fraction, np.float16, np.float32, np.float64, np.longdouble}
)
_TRUSTED_REAL_TYPES = _ACTUAL_INT_TYPES | _ACTUAL_FLOAT_TYPES
_INFINITY_TYPES = _ACTUAL_FLOAT_TYPES - {Fraction}


def _require_feature_dim(
    value: object,
    *,
    vectors: int,
    fixed_scalars: int,
    update_vectors: int,
    update_scalars: int = 16,
    augmented: bool = False,
) -> int:
    maximum = _INT32_MAX - 1 if augmented else _INT32_MAX
    actual_type = type(value)
    if not any(actual_type is allowed_type for allowed_type in _ACTUAL_INT_TYPES):
        raise ValueError(f"feature_dim must be an integer in [1, {maximum}]")
    feature_dim = operator.index(cast(SupportsIndex, value))
    if not 1 <= feature_dim <= maximum:
        raise ValueError(f"feature_dim must be an integer in [1, {maximum}]")
    width = feature_dim + 1 if augmented else feature_dim
    scalar_count = vectors * width + fixed_scalars
    if 4 * scalar_count > _INT32_MAX:
        raise ValueError("derived state_nbytes must fit in signed int32")
    update_scalar_count = update_vectors * width + update_scalars
    if 4 * update_scalar_count > _INT32_MAX:
        raise ValueError("derived update working set byte count must fit in signed int32")
    return feature_dim


def _positive_float32_or_infinity(name: str, value: object) -> float:
    """Preserve the documented infinity sentinel but reject finite overflow."""
    actual_type = type(value)
    if not any(actual_type is allowed_type for allowed_type in _TRUSTED_REAL_TYPES):
        raise ValueError(f"{name} must be positive or infinity")
    try:
        numeric_value = cast(Any, value)
        if any(actual_type is allowed_type for allowed_type in _INFINITY_TYPES) and bool(
            np.isinf(numeric_value)
        ):
            if bool(np.signbit(numeric_value)):
                raise ValueError(f"{name} must be positive or infinity")
            return math.inf
    except Exception as error:
        raise ValueError(f"{name} must be positive or infinity") from error
    return validated_float32_scalar(name, value, positive=True)


def _validated_config_float(name: str, value: object, **bounds: Any) -> float:
    actual_type = type(value)
    if not any(actual_type is allowed_type for allowed_type in _TRUSTED_REAL_TYPES):
        raise ValueError(f"{name} must be a finite real scalar")
    return validated_float32_scalar(name, value, **bounds)


def _skip_zero_scale(scale: Array, value: Array) -> Array:
    """Return 0 when ``scale`` is 0 so a 0*inf product cannot form."""
    return jnp.where(scale == 0.0, jnp.zeros_like(value), scale * value)


def _zero_if_unused(scale: Array, value: Array) -> Array:
    """Sanitize leftover inf only in the finite-state check when unused."""
    return jnp.where(scale == 0.0, jnp.zeros_like(value), value)


def _array_metadata(name: str, value: object, shape: tuple[int, ...]) -> Array:
    """Require exact float32 host metadata before a value reaches JAX."""
    actual_type = type(value)
    if not (
        actual_type is np.ndarray
        or issubclass(actual_type, jax.Array)
        or issubclass(actual_type, jax.core.Tracer)
        or actual_type is jax.ShapeDtypeStruct
    ):
        raise ValueError(f"{name} must expose trusted array metadata")
    trusted_value = cast(Any, value)
    actual_shape = tuple(trusted_value.shape)
    actual_dtype = np.dtype(trusted_value.dtype)
    if actual_shape != shape or actual_dtype != np.dtype(np.float32):
        raise ValueError(f"{name} must have shape {shape} and dtype float32")
    return cast(Array, value)


def _scalar_operand(name: str, value: object) -> Array:
    actual_type = type(value)
    if any(actual_type is allowed_type for allowed_type in _TRUSTED_REAL_TYPES):
        canonical = validated_float32_scalar(name, value)
        return jnp.asarray(canonical, dtype=jnp.float32)
    return _array_metadata(name, value, ())


def _discount_operand(value: object) -> Array:
    """Validate a discount in its exact host and consumed float32 domains."""
    actual_type = type(value)
    if any(actual_type is allowed_type for allowed_type in _TRUSTED_REAL_TYPES):
        canonical = validated_float32_scalar("gamma", value, lower=0.0, upper=1.0)
        return jnp.asarray(canonical, dtype=jnp.float32)
    return _array_metadata("gamma", value, ())


def _stable_rms(values: Array) -> Array:
    scale = jnp.max(jnp.abs(values), initial=0.0)
    absolute = jnp.abs(values)
    normalized = jnp.where(
        absolute == scale,
        jnp.sign(values),
        jnp.where(scale > 0.0, values / scale, 0.0),
    )
    return scale * jnp.sqrt(jnp.mean(jnp.square(normalized)))


def _require_scan_resources(num_steps: int, feature_dim: int) -> None:
    # Five inputs plus five outputs (four scalar vectors and six-column metrics),
    # with observation-width inputs and the complete three-vector carry/work set.
    logical_scalars = num_steps * (2 * feature_dim + 15) + 12 * feature_dim + 32
    if logical_scalars > _INT32_MAX or 4 * logical_scalars > _INT32_MAX:
        raise ValueError("learning-loop aggregate resources exceed signed-int32 bounds")


def _serialized_payload(
    config: object, *, type_name: str, fields: frozenset[str]
) -> dict[str, Any]:
    if not issubclass(type(config), Mapping):
        raise ValueError("config must be an actual mapping")
    try:
        payload = dict(cast(Mapping[str, Any], config))
    except Exception as error:
        raise ValueError("config must be a readable mapping") from error
    if any(type(key) is not str for key in payload) or set(payload) != fields | {"type"}:
        raise ValueError("config fields do not match the serialized schema")
    marker = payload.pop("type")
    if type(marker) is not str or marker != type_name:
        raise ValueError("config type differs")
    if any(type(value) is not int and type(value) is not float for value in payload.values()):
        raise ValueError("serialized scalar fields must be exact JSON numbers")
    return payload


# =============================================================================
# State / result types
# =============================================================================


@chex.dataclass(frozen=True)
class OffPolicyTDState:
    """State for the off-policy linear TD learner.

    Attributes:
        weights: Weight vector for linear value approximation
        bias: Bias term
        eligibility_traces: Per-feature importance-sampling trace ``z_t``
        bias_eligibility_trace: Bias importance-sampling trace
        step_count: Number of updates applied
        birth_timestamp: Wall-clock seconds at init
        uptime_s: Cumulative wall-clock seconds spent in update calls
    """

    weights: Float[Array, " feature_dim"]
    bias: Float[Array, ""]
    eligibility_traces: Float[Array, " feature_dim"]
    bias_eligibility_trace: Float[Array, ""]
    step_count: Int[Array, ""] = None  # type: ignore[assignment]
    birth_timestamp: float = 0.0
    uptime_s: float = 0.0


@chex.dataclass(frozen=True)
class OffPolicyTDUpdateResult:
    """Result of an off-policy TD update.

    Attributes:
        state: Updated learner state
        prediction: V(s) computed before the update
        td_error: TD error delta = R + gamma * V(s') - V(s)
        rho_clipped: Importance-sampling ratio after clipping (so it can
            be logged for variance diagnostics)
        metrics: Array of shape (5,) with columns
            [squared_td_error, td_error, rho_clipped, mean_alpha, mean_trace]
    """

    state: OffPolicyTDState
    prediction: Float[Array, " 1"]
    td_error: Float[Array, ""]
    rho_clipped: Float[Array, ""]
    metrics: Float[Array, " 5"]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ETDState:
    """State for the emphatic TD(lambda) linear learner.

    Attributes:
        weights: Weight vector for linear value approximation
        bias: Bias term
        eligibility_traces: Emphatic eligibility trace
        bias_eligibility_trace: Emphatic eligibility trace for the bias
        follow_on_trace: Scalar follow-on trace ``F_t``
        emphasis: Scalar emphasis ``M_t`` from the latest update
        previous_rho: Importance ratio from the prior update call, carried
            forward so the *next* call can advance ``F_t`` on ``rho_{t-1}``
            (Sutton, Mahmood & White 2016, eq. 20) rather than on the ratio
            of the transition it is currently processing.
        step_count: Number of updates applied
        birth_timestamp: Wall-clock seconds at init
        uptime_s: Cumulative wall-clock seconds spent in update calls
    """

    weights: Float[Array, " feature_dim"]
    bias: Float[Array, ""]
    eligibility_traces: Float[Array, " feature_dim"]
    bias_eligibility_trace: Float[Array, ""]
    follow_on_trace: Float[Array, ""]
    emphasis: Float[Array, ""]
    previous_rho: Float[Array, ""] = None  # type: ignore[assignment]
    step_count: Int[Array, ""] = None  # type: ignore[assignment]
    birth_timestamp: float = 0.0
    uptime_s: float = 0.0


@chex.dataclass(frozen=True)
class ETDUpdateResult:
    """Result of an emphatic TD(lambda) update.

    Attributes:
        state: Updated learner state
        prediction: V(s) computed before the update
        td_error: TD error delta = R + gamma * V(s') - V(s)
        follow_on_trace: Updated follow-on trace ``F_t``
        emphasis: Updated scalar emphasis ``M_t``
        metrics: Array of shape (7,) with columns
            [squared_td_error, td_error, rho, mean_alpha, mean_trace,
            follow_on_trace, emphasis]
    """

    state: ETDState
    prediction: Float[Array, " 1"]
    td_error: Float[Array, ""]
    follow_on_trace: Float[Array, ""]
    emphasis: Float[Array, ""]
    metrics: Float[Array, " 7"]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class GradientTDState:
    """State for linear off-policy Gradient-TD/TDC prediction.

    The bias is represented by an appended constant feature, so all vectors have
    shape ``feature_dim + 1``.
    """

    weights: Float[Array, " augmented_feature_dim"]
    secondary_weights: Float[Array, " augmented_feature_dim"]
    eligibility_traces: Float[Array, " augmented_feature_dim"]
    step_count: Int[Array, ""] = None  # type: ignore[assignment]
    birth_timestamp: float = 0.0
    uptime_s: float = 0.0


@chex.dataclass(frozen=True)
class GradientTDUpdateResult:
    """Result of one linear Gradient-TD/TDC update."""

    state: GradientTDState
    prediction: Float[Array, " 1"]
    td_error: Float[Array, ""]
    rho_clipped: Float[Array, ""]
    metrics: Float[Array, " 6"]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class GradientTDArrayResult:
    """Result from scanning Gradient-TD/TDC over transition arrays."""

    state: GradientTDState
    predictions: Float[Array, " num_steps"]
    td_errors: Float[Array, " num_steps"]
    rho_clipped: Float[Array, " num_steps"]
    metrics: Float[Array, "num_steps 6"]
    updates_applied: Bool[Array, " num_steps"]


def _off_policy_state_contract(state: object) -> tuple[OffPolicyTDState, int]:
    if type(state) is not OffPolicyTDState:
        raise ValueError("state must be an OffPolicyTDState")
    checked = state
    try:
        feature_dim = tuple(checked.weights.shape)[0]
    except Exception as error:
        raise ValueError("state.weights must expose array metadata") from error
    _array_metadata("state.weights", checked.weights, (feature_dim,))
    _array_metadata("state.bias", checked.bias, ())
    _array_metadata("state.eligibility_traces", checked.eligibility_traces, (feature_dim,))
    _array_metadata("state.bias_eligibility_trace", checked.bias_eligibility_trace, ())
    try:
        step_shape = tuple(checked.step_count.shape)
        step_dtype = np.dtype(checked.step_count.dtype)
    except Exception as error:
        raise ValueError("state.step_count must expose array metadata") from error
    if step_shape != () or step_dtype != np.dtype(np.int32):
        raise ValueError("state.step_count must be a scalar int32")
    return checked, feature_dim


def _etd_state_contract(state: object) -> tuple[ETDState, int]:
    if type(state) is not ETDState:
        raise ValueError("state must be an ETDState")
    checked = state
    try:
        feature_dim = tuple(checked.weights.shape)[0]
    except Exception as error:
        raise ValueError("state.weights must expose array metadata") from error
    _array_metadata("state.weights", checked.weights, (feature_dim,))
    _array_metadata("state.bias", checked.bias, ())
    _array_metadata("state.eligibility_traces", checked.eligibility_traces, (feature_dim,))
    for name in ("bias_eligibility_trace", "follow_on_trace", "emphasis", "previous_rho"):
        _array_metadata(f"state.{name}", getattr(checked, name), ())
    try:
        step_shape = tuple(checked.step_count.shape)
        step_dtype = np.dtype(checked.step_count.dtype)
    except Exception as error:
        raise ValueError("state.step_count must expose array metadata") from error
    if step_shape != () or step_dtype != np.dtype(np.int32):
        raise ValueError("state.step_count must be a scalar int32")
    return checked, feature_dim


def _gradient_state_contract(state: object) -> tuple[GradientTDState, int]:
    if type(state) is not GradientTDState:
        raise ValueError("state must be a GradientTDState")
    checked = state
    try:
        augmented_dim = tuple(checked.weights.shape)[0]
    except Exception as error:
        raise ValueError("state.weights must expose array metadata") from error
    if augmented_dim < 2:
        raise ValueError("state augmented feature dimension must be at least two")
    for name in ("weights", "secondary_weights", "eligibility_traces"):
        _array_metadata(f"state.{name}", getattr(checked, name), (augmented_dim,))
    try:
        step_shape = tuple(checked.step_count.shape)
        step_dtype = np.dtype(checked.step_count.dtype)
    except Exception as error:
        raise ValueError("state.step_count must expose array metadata") from error
    if step_shape != () or step_dtype != np.dtype(np.int32):
        raise ValueError("state.step_count must be a scalar int32")
    return checked, augmented_dim - 1


# =============================================================================
# Learner
# =============================================================================


class OffPolicyTDLinearLearner:
    """Off-policy linear TD(lambda) with clipped per-decision IS.

    The update rule is::

        rho_t = pi(a_t|s_t) / b(a_t|s_t)               (provided externally)
        rho_clipped = min(c, rho_t)                     (ratio clipping)
        delta_t = R_{t+1} + gamma_t * V(s_{t+1}) - V(s_t)
        z_t = rho_clipped * (gamma_t * lambda_t * z_{t-1} + phi_t)
        w_{t+1} = w_t + alpha * delta_t * z_t

    Setting the historically named ``retrace_clip`` to infinity recovers
    naive per-decision IS. Setting ``rho_t = 1`` recovers on-policy
    semi-gradient TD(lambda). This update is not the Retrace operator and
    does not inherit its convergence guarantees.

    Attributes:
        step_size: Learning rate alpha
        trace_decay: Eligibility trace decay lambda
        retrace_clip: Maximum allowed importance ratio (Inf to disable)
    """

    def __init__(
        self,
        step_size: float = 0.05,
        trace_decay: float = 0.0,
        retrace_clip: float = 1.0,
    ):
        """Initialize the off-policy TD learner.

        Args:
            step_size: Learning rate alpha (scalar)
            trace_decay: Eligibility trace decay lambda in [0, 1]
            retrace_clip: Maximum allowed importance ratio. The name is kept
                for configuration compatibility; pass ``float("inf")`` to
                disable clipping.
        """
        self._step_size = _validated_config_float("step_size", step_size, positive=True)
        self._trace_decay = _validated_config_float(
            "trace_decay", trace_decay, lower=0.0, upper=1.0
        )
        self._retrace_clip = _positive_float32_or_infinity("retrace_clip", retrace_clip)

    @property
    def step_size(self) -> float:
        """Learning rate alpha."""
        return self._step_size

    @property
    def trace_decay(self) -> float:
        """Trace decay lambda."""
        return self._trace_decay

    @property
    def retrace_clip(self) -> float:
        """Maximum per-decision importance ratio (compatibility name)."""
        return self._retrace_clip

    def init(self, feature_dim: int) -> OffPolicyTDState:
        """Initialize learner state with zero weights and zero traces."""
        feature_dim = _require_feature_dim(
            feature_dim, vectors=2, fixed_scalars=3, update_vectors=9
        )
        return OffPolicyTDState(  # type: ignore[call-arg]
            weights=jnp.zeros(feature_dim, dtype=jnp.float32),
            bias=jnp.array(0.0, dtype=jnp.float32),
            eligibility_traces=jnp.zeros(feature_dim, dtype=jnp.float32),
            bias_eligibility_trace=jnp.array(0.0, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
            birth_timestamp=time.time(),
            uptime_s=0.0,
        )

    def predict(self, state: OffPolicyTDState, observation: Observation) -> Float[Array, " 1"]:
        """Compute V(s) = w . phi(s) + b."""
        state, feature_dim = _off_policy_state_contract(state)
        observation = _array_metadata("observation", observation, (feature_dim,))
        return cast(Array, self._predict_jit(state, observation))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _predict_jit(self, state: OffPolicyTDState, observation: Observation) -> Float[Array, " 1"]:
        return jnp.atleast_1d(jnp.dot(state.weights, observation) + state.bias)

    def update(
        self,
        state: OffPolicyTDState,
        observation: Observation,
        reward: Array,
        next_observation: Observation,
        gamma: Array,
        rho: Array,
    ) -> OffPolicyTDUpdateResult:
        """Apply one off-policy TD update.

        Args:
            state: Current learner state
            observation: Current feature vector phi(s_t)
            reward: Reward R_{t+1}
            next_observation: Next feature vector phi(s_{t+1})
            gamma: State-dependent discount gamma_t (0 at terminal)
            rho: Importance-sampling ratio pi(a_t|s_t) / b(a_t|s_t).
                Pass 1.0 for on-policy data.

        Returns:
            ``OffPolicyTDUpdateResult`` with updated state, prediction,
            TD error, clipped IS ratio, and a metrics array of shape (5,).
        """
        state, feature_dim = _off_policy_state_contract(state)
        observation = _array_metadata("observation", observation, (feature_dim,))
        next_observation = _array_metadata("next_observation", next_observation, (feature_dim,))
        reward = _scalar_operand("reward", reward)
        gamma = _discount_operand(gamma)
        rho = _scalar_operand("rho", rho)
        result = self._update_jit(state, observation, reward, next_observation, gamma, rho)
        return cast(
            OffPolicyTDUpdateResult,
            result.replace(
                state=result.state.replace(
                    birth_timestamp=state.birth_timestamp,
                    uptime_s=state.uptime_s,
                )
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_jit(
        self,
        state: OffPolicyTDState,
        observation: Observation,
        reward: Array,
        next_observation: Observation,
        gamma: Array,
        rho: Array,
    ) -> OffPolicyTDUpdateResult:
        alpha = jnp.asarray(self._step_size, dtype=jnp.float32)
        lam = jnp.asarray(self._trace_decay, dtype=jnp.float32)
        clip = jnp.asarray(self._retrace_clip, dtype=jnp.float32)
        gamma_s = jnp.squeeze(gamma).astype(jnp.float32)
        reward_s = jnp.squeeze(reward).astype(jnp.float32)
        rho_s = jnp.squeeze(rho).astype(jnp.float32)

        rho_clipped = jnp.minimum(rho_s, clip)

        v_t = jnp.dot(state.weights, observation) + state.bias
        v_next = jnp.dot(state.weights, next_observation) + state.bias
        td_error = reward_s + _skip_zero_scale(gamma_s, v_next) - v_t

        # Canonical per-decision IS trace.  Each transition's ratio enters once:
        # the prior ratios are already represented in the stored trace.
        decay = gamma_s * lam
        new_e = _skip_zero_scale(
            rho_clipped, _skip_zero_scale(decay, state.eligibility_traces) + observation
        )
        new_e_b = _skip_zero_scale(
            rho_clipped, _skip_zero_scale(decay, state.bias_eligibility_trace) + 1.0
        )

        # rho_clipped is already represented in the trace.
        scaled_update = alpha * td_error
        proposed_state = OffPolicyTDState(  # type: ignore[call-arg]
            weights=state.weights + scaled_update * new_e,
            bias=state.bias + scaled_update * new_e_b,
            eligibility_traces=new_e,
            bias_eligibility_trace=new_e_b,
            step_count=jnp.minimum(state.step_count, _INT32_MAX - 1) + 1,
            birth_timestamp=state.birth_timestamp,
            uptime_s=state.uptime_s,
        )
        # Inf reward makes scaled_update * e = 0*inf = NaN on a silent feature.
        inputs_valid = (
            jnp.all(jnp.isfinite(observation))
            & jnp.isfinite(reward_s)
            & ((gamma_s == 0.0) | jnp.all(jnp.isfinite(next_observation)))
            & (gamma_s >= 0.0)
            & (gamma_s <= 1.0)
            & jnp.isfinite(rho_s)
        )
        previous_checked = state.replace(  # type: ignore[attr-defined]
            eligibility_traces=_zero_if_unused(decay, state.eligibility_traces),
            bias_eligibility_trace=_zero_if_unused(decay, state.bias_eligibility_trace),
        )
        squared_td = td_error**2
        mean_e = jnp.mean(jnp.abs(proposed_state.eligibility_traces))
        candidate_metrics = jnp.array(
            [squared_td, td_error, rho_clipped, alpha, mean_e], dtype=jnp.float32
        )
        update_applied = (
            inputs_valid
            & _floating_tree_is_finite(previous_checked)
            & _floating_tree_is_finite(proposed_state)
            & jnp.all(jnp.isfinite(candidate_metrics))
        )
        new_state = jax.lax.cond(
            update_applied,
            lambda: proposed_state,
            lambda: state,
        )

        return OffPolicyTDUpdateResult(  # type: ignore[call-arg]
            state=new_state,
            prediction=jnp.where(
                update_applied, jnp.atleast_1d(v_t), jnp.zeros_like(jnp.atleast_1d(v_t))
            ),
            td_error=jnp.where(update_applied, td_error, jnp.zeros_like(td_error)),
            rho_clipped=jnp.where(update_applied, rho_clipped, jnp.zeros_like(rho_clipped)),
            metrics=jnp.where(update_applied, candidate_metrics, jnp.zeros_like(candidate_metrics)),
            update_applied=update_applied,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "type": "OffPolicyTDLinearLearner",
            "step_size": self._step_size,
            "trace_decay": self._trace_decay,
            "retrace_clip": self._retrace_clip,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> OffPolicyTDLinearLearner:
        """Reconstruct from dict."""
        return cls(**_serialized_payload(
            config,
            type_name=cls.__name__,
            fields=frozenset({"step_size", "trace_decay", "retrace_clip"}),
        ))


class ETDLinearLearner:
    """Off-policy linear emphatic TD(lambda).

    Unlike the clipped per-decision learner above, ETD(lambda) uses a
    follow-on trace and scalar emphasis. Per Sutton, Mahmood & White (2016),
    eqs. 17-20, the follow-on trace advances on the *previous* step's ratio
    while the eligibility trace uses the *current* one -- this offset is
    deliberate, not a typo:

    ``F_t = rho_{t-1} * gamma_t * F_{t-1} + i_t``
    ``M_t = lambda * i_t + (1 - lambda) * F_t``
    ``e_t = rho_t * (gamma_t * lambda * e_{t-1} + M_t * phi_t)``
    ``w_{t+1} = w_t + alpha * delta_t * e_t``

    The single-step API advances the follow-on trace with the current
    transition's ratio and discount. With ``rho=1``, ``gamma=0``, and
    ``lambda=0``, this reduces to the standard LMS/TD(0) terminating update.

    Attributes:
        step_size: Learning rate alpha
        trace_decay: Eligibility trace decay lambda
    """

    def __init__(
        self,
        step_size: float = 0.05,
        trace_decay: float = 0.0,
    ):
        """Initialize the emphatic TD learner.

        Args:
            step_size: Learning rate alpha (scalar)
            trace_decay: Eligibility trace decay lambda in [0, 1]
        """
        self._step_size = _validated_config_float("step_size", step_size, positive=True)
        self._trace_decay = _validated_config_float(
            "trace_decay", trace_decay, lower=0.0, upper=1.0
        )

    @property
    def step_size(self) -> float:
        """Learning rate alpha."""
        return self._step_size

    @property
    def trace_decay(self) -> float:
        """Trace decay lambda."""
        return self._trace_decay

    def init(self, feature_dim: int) -> ETDState:
        """Initialize learner state with zero weights and zero traces."""
        feature_dim = _require_feature_dim(
            feature_dim, vectors=2, fixed_scalars=6, update_vectors=9
        )
        return ETDState(  # type: ignore[call-arg]
            weights=jnp.zeros(feature_dim, dtype=jnp.float32),
            bias=jnp.array(0.0, dtype=jnp.float32),
            eligibility_traces=jnp.zeros(feature_dim, dtype=jnp.float32),
            bias_eligibility_trace=jnp.array(0.0, dtype=jnp.float32),
            follow_on_trace=jnp.array(0.0, dtype=jnp.float32),
            emphasis=jnp.array(0.0, dtype=jnp.float32),
            previous_rho=jnp.array(1.0, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
            birth_timestamp=time.time(),
            uptime_s=0.0,
        )

    def predict(self, state: ETDState, observation: Observation) -> Float[Array, " 1"]:
        """Compute V(s) = w . phi(s) + b."""
        state, feature_dim = _etd_state_contract(state)
        observation = _array_metadata("observation", observation, (feature_dim,))
        return cast(Array, self._predict_jit(state, observation))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _predict_jit(self, state: ETDState, observation: Observation) -> Float[Array, " 1"]:
        return jnp.atleast_1d(jnp.dot(state.weights, observation) + state.bias)

    def update(
        self,
        state: ETDState,
        observation: Observation,
        reward: Array,
        next_observation: Observation,
        gamma: Array,
        rho: Array,
        interest: Array | float = 1.0,
    ) -> ETDUpdateResult:
        """Apply one ETD(lambda) update.

        Args:
            state: Current learner state
            observation: Current feature vector phi(s_t)
            reward: Reward R_{t+1}
            next_observation: Next feature vector phi(s_{t+1})
            gamma: State-dependent discount gamma (0 at terminal)
            rho: Importance-sampling ratio pi(a_t|s_t) / b(a_t|s_t).
            interest: State interest i_t. Defaults to 1.0.

        Returns:
            ``ETDUpdateResult`` with updated state, prediction, TD error,
            follow-on trace, emphasis, and a metrics array of shape (7,).
        """
        state, feature_dim = _etd_state_contract(state)
        observation = _array_metadata("observation", observation, (feature_dim,))
        next_observation = _array_metadata("next_observation", next_observation, (feature_dim,))
        reward = _scalar_operand("reward", reward)
        gamma = _discount_operand(gamma)
        rho = _scalar_operand("rho", rho)
        interest = _scalar_operand("interest", interest)
        result = self._update_jit(
            state, observation, reward, next_observation, gamma, rho, interest
        )
        return cast(
            ETDUpdateResult,
            result.replace(
                state=result.state.replace(
                    birth_timestamp=state.birth_timestamp,
                    uptime_s=state.uptime_s,
                )
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_jit(
        self,
        state: ETDState,
        observation: Observation,
        reward: Array,
        next_observation: Observation,
        gamma: Array,
        rho: Array,
        interest: Array,
    ) -> ETDUpdateResult:
        alpha = jnp.asarray(self._step_size, dtype=jnp.float32)
        lam = jnp.asarray(self._trace_decay, dtype=jnp.float32)
        gamma_s = jnp.squeeze(gamma).astype(jnp.float32)
        reward_s = jnp.squeeze(reward).astype(jnp.float32)
        rho_s = jnp.squeeze(rho).astype(jnp.float32)
        interest_s = jnp.squeeze(jnp.asarray(interest, dtype=jnp.float32))

        v_t = jnp.dot(state.weights, observation) + state.bias
        v_next = jnp.dot(state.weights, next_observation) + state.bias
        td_error = reward_s + _skip_zero_scale(gamma_s, v_next) - v_t

        # F_t = rho_{t-1} * gamma_t * F_{t-1} + i_t (Sutton, Mahmood & White
        # 2016, eq. 20): the follow-on trace advances on the PRIOR call's
        # ratio (state.previous_rho), not rho_s -- only e_t below uses rho_s.
        follow_on = (
            _skip_zero_scale(gamma_s, state.previous_rho * state.follow_on_trace) + interest_s
        )
        emphasis = _skip_zero_scale(lam, interest_s) + _skip_zero_scale(1.0 - lam, follow_on)

        trace_decay = gamma_s * lam
        new_e = rho_s * (
            _skip_zero_scale(trace_decay, state.eligibility_traces) + emphasis * observation
        )
        new_e_b = rho_s * (_skip_zero_scale(trace_decay, state.bias_eligibility_trace) + emphasis)

        proposed_state = ETDState(  # type: ignore[call-arg]
            weights=state.weights + alpha * td_error * new_e,
            bias=state.bias + alpha * td_error * new_e_b,
            eligibility_traces=new_e,
            bias_eligibility_trace=new_e_b,
            follow_on_trace=follow_on,
            emphasis=emphasis,
            previous_rho=rho_s,
            step_count=jnp.minimum(state.step_count, _INT32_MAX - 1) + 1,
            birth_timestamp=state.birth_timestamp,
            uptime_s=state.uptime_s,
        )
        inputs_valid = (
            jnp.all(jnp.isfinite(observation))
            & jnp.isfinite(reward_s)
            & ((gamma_s == 0.0) | jnp.all(jnp.isfinite(next_observation)))
            & (gamma_s >= 0.0)
            & (gamma_s <= 1.0)
            & jnp.isfinite(rho_s)
            & jnp.isfinite(interest_s)
        )
        previous_checked = state.replace(  # type: ignore[attr-defined]
            eligibility_traces=_zero_if_unused(trace_decay, state.eligibility_traces),
            bias_eligibility_trace=_zero_if_unused(trace_decay, state.bias_eligibility_trace),
            follow_on_trace=_zero_if_unused(gamma_s, state.follow_on_trace),
            previous_rho=_zero_if_unused(gamma_s, state.previous_rho),
        )
        squared_td = td_error**2
        mean_e = jnp.mean(jnp.abs(proposed_state.eligibility_traces))
        candidate_metrics = jnp.array(
            [squared_td, td_error, rho_s, alpha, mean_e, follow_on, emphasis],
            dtype=jnp.float32,
        )
        update_applied = (
            inputs_valid
            & _floating_tree_is_finite(previous_checked)
            & _floating_tree_is_finite(proposed_state)
            & jnp.all(jnp.isfinite(candidate_metrics))
        )
        new_state = jax.lax.cond(
            update_applied,
            lambda: proposed_state,
            lambda: state,
        )

        return ETDUpdateResult(  # type: ignore[call-arg]
            state=new_state,
            prediction=jnp.where(
                update_applied, jnp.atleast_1d(v_t), jnp.zeros_like(jnp.atleast_1d(v_t))
            ),
            td_error=jnp.where(update_applied, td_error, jnp.zeros_like(td_error)),
            follow_on_trace=jnp.where(update_applied, follow_on, jnp.zeros_like(follow_on)),
            emphasis=jnp.where(update_applied, emphasis, jnp.zeros_like(emphasis)),
            metrics=jnp.where(update_applied, candidate_metrics, jnp.zeros_like(candidate_metrics)),
            update_applied=update_applied,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "type": "ETDLinearLearner",
            "step_size": self._step_size,
            "trace_decay": self._trace_decay,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ETDLinearLearner:
        """Reconstruct from dict."""
        return cls(**_serialized_payload(
            config,
            type_name=cls.__name__,
            fields=frozenset({"step_size", "trace_decay"}),
        ))


class GradientTDLinearLearner:
    """Linear off-policy Gradient-TD/TDC learner with secondary weights.

    This implements the linear TDC/GTD(lambda)-style correction with an
    auxiliary weight vector, descending the projected Bellman-error objective in
    the standard linear setting:

    ``delta = r + gamma theta^T phi' - theta^T phi``
    ``e = rho * (phi + gamma * lambda * e)``
    ``theta += alpha * (delta * e - gamma * (1 - lambda) * (h^T e) * phi')``
    ``h += beta * (delta * e - (h^T phi) * phi)``

    The implementation is intentionally linear. Nonlinear shared-trunk GTD is a
    separate approximation problem; this class supplies the exact secondary
    weight correction missing from semi-gradient off-policy TD/Horde.
    """

    def __init__(
        self,
        step_size: float = 0.01,
        secondary_step_size: float = 0.05,
        trace_decay: float = 0.0,
        ratio_clip: float = 10.0,
    ):
        """Initialize the learner."""
        self._step_size = _validated_config_float("step_size", step_size, positive=True)
        self._secondary_step_size = _validated_config_float(
            "secondary_step_size", secondary_step_size, lower=0.0
        )
        self._trace_decay = _validated_config_float(
            "trace_decay", trace_decay, lower=0.0, upper=1.0
        )
        self._ratio_clip = _positive_float32_or_infinity("ratio_clip", ratio_clip)

    @property
    def step_size(self) -> float:
        """Primary learning rate."""
        return self._step_size

    @property
    def secondary_step_size(self) -> float:
        """Secondary-weight learning rate."""
        return self._secondary_step_size

    @property
    def trace_decay(self) -> float:
        """Eligibility trace decay."""
        return self._trace_decay

    @property
    def ratio_clip(self) -> float:
        """Importance-ratio clip."""
        return self._ratio_clip

    def init(self, feature_dim: int) -> GradientTDState:
        """Initialize primary weights, secondary weights, and traces."""
        feature_dim = _require_feature_dim(
            feature_dim,
            vectors=3,
            fixed_scalars=1,
            update_vectors=15,
            augmented=True,
        )
        augmented_dim = feature_dim + 1
        return GradientTDState(  # type: ignore[call-arg]
            weights=jnp.zeros(augmented_dim, dtype=jnp.float32),
            secondary_weights=jnp.zeros(augmented_dim, dtype=jnp.float32),
            eligibility_traces=jnp.zeros(augmented_dim, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
            birth_timestamp=time.time(),
            uptime_s=0.0,
        )

    @staticmethod
    def _augment(observation: Observation) -> Array:
        """Append the bias feature."""
        return jnp.concatenate(
            (
                jnp.asarray(observation, dtype=jnp.float32),
                jnp.ones((1,), dtype=jnp.float32),
            )
        )

    def predict(self, state: GradientTDState, observation: Observation) -> Float[Array, " 1"]:
        """Compute ``theta^T phi`` with an appended bias feature."""
        state, feature_dim = _gradient_state_contract(state)
        observation = _array_metadata("observation", observation, (feature_dim,))
        return cast(Array, self._predict_jit(state, observation))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _predict_jit(self, state: GradientTDState, observation: Observation) -> Float[Array, " 1"]:
        return jnp.atleast_1d(jnp.dot(state.weights, self._augment(observation)))

    def update(
        self,
        state: GradientTDState,
        observation: Observation,
        reward: Array,
        next_observation: Observation,
        gamma: Array,
        rho: Array,
    ) -> GradientTDUpdateResult:
        """Apply one off-policy Gradient-TD/TDC update."""
        state, feature_dim = _gradient_state_contract(state)
        observation = _array_metadata("observation", observation, (feature_dim,))
        next_observation = _array_metadata("next_observation", next_observation, (feature_dim,))
        reward = _scalar_operand("reward", reward)
        gamma = _discount_operand(gamma)
        rho = _scalar_operand("rho", rho)
        result = self._update_jit(state, observation, reward, next_observation, gamma, rho)
        return cast(
            GradientTDUpdateResult,
            result.replace(
                state=result.state.replace(
                    birth_timestamp=state.birth_timestamp,
                    uptime_s=state.uptime_s,
                )
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_jit(
        self,
        state: GradientTDState,
        observation: Observation,
        reward: Array,
        next_observation: Observation,
        gamma: Array,
        rho: Array,
    ) -> GradientTDUpdateResult:
        alpha = jnp.asarray(self._step_size, dtype=jnp.float32)
        beta = jnp.asarray(self._secondary_step_size, dtype=jnp.float32)
        lam = jnp.asarray(self._trace_decay, dtype=jnp.float32)
        ratio_clip = jnp.asarray(self._ratio_clip, dtype=jnp.float32)
        gamma_s = jnp.squeeze(gamma).astype(jnp.float32)
        reward_s = jnp.squeeze(reward).astype(jnp.float32)
        rho_s = jnp.squeeze(rho).astype(jnp.float32)
        rho_clipped = jnp.minimum(jnp.maximum(rho_s, 0.0), ratio_clip)

        phi = self._augment(observation)
        next_phi = self._augment(next_observation)
        prediction = jnp.dot(state.weights, phi)
        next_prediction = jnp.dot(state.weights, next_phi)
        td_error = reward_s + _skip_zero_scale(gamma_s, next_prediction) - prediction

        traces = rho_clipped * (phi + _skip_zero_scale(gamma_s * lam, state.eligibility_traces))
        secondary_dot_trace = jnp.dot(state.secondary_weights, traces)
        secondary_dot_phi = jnp.dot(state.secondary_weights, phi)

        correction_coefficient = gamma_s * (1.0 - lam)
        scaled_secondary_trace = _skip_zero_scale(correction_coefficient, secondary_dot_trace)
        bootstrap_correction = _skip_zero_scale(scaled_secondary_trace, next_phi)
        primary_step = alpha * (td_error * traces - bootstrap_correction)
        secondary_step = beta * (td_error * traces - secondary_dot_phi * phi)

        proposed_state = GradientTDState(  # type: ignore[call-arg]
            weights=state.weights + primary_step,
            secondary_weights=state.secondary_weights + secondary_step,
            eligibility_traces=traces,
            step_count=jnp.minimum(state.step_count, _INT32_MAX - 1) + 1,
            birth_timestamp=state.birth_timestamp,
            uptime_s=state.uptime_s,
        )
        inputs_valid = (
            jnp.all(jnp.isfinite(observation))
            & jnp.isfinite(reward_s)
            & ((gamma_s == 0.0) | jnp.all(jnp.isfinite(next_observation)))
            & (gamma_s >= 0.0)
            & (gamma_s <= 1.0)
            & jnp.isfinite(rho_s)
        )
        previous_checked = state.replace(  # type: ignore[attr-defined]
            eligibility_traces=_zero_if_unused(gamma_s * lam, state.eligibility_traces),
        )
        candidate_metrics = jnp.array(
            [
                td_error**2,
                td_error,
                rho_clipped,
                _stable_rms(proposed_state.weights),
                _stable_rms(proposed_state.secondary_weights),
                jnp.mean(jnp.abs(proposed_state.eligibility_traces)),
            ],
            dtype=jnp.float32,
        )
        update_applied = (
            inputs_valid
            & _floating_tree_is_finite(previous_checked)
            & _floating_tree_is_finite(proposed_state)
            & jnp.all(jnp.isfinite(candidate_metrics))
        )
        new_state = jax.lax.cond(
            update_applied,
            lambda: proposed_state,
            lambda: state,
        )
        return GradientTDUpdateResult(  # type: ignore[call-arg]
            state=new_state,
            prediction=jnp.where(
                update_applied,
                jnp.atleast_1d(prediction),
                jnp.zeros_like(jnp.atleast_1d(prediction)),
            ),
            td_error=jnp.where(update_applied, td_error, jnp.zeros_like(td_error)),
            rho_clipped=jnp.where(update_applied, rho_clipped, jnp.zeros_like(rho_clipped)),
            metrics=jnp.where(update_applied, candidate_metrics, jnp.zeros_like(candidate_metrics)),
            update_applied=update_applied,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "type": "GradientTDLinearLearner",
            "step_size": self._step_size,
            "secondary_step_size": self._secondary_step_size,
            "trace_decay": self._trace_decay,
            "ratio_clip": self._ratio_clip,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> GradientTDLinearLearner:
        """Reconstruct from dict."""
        return cls(**_serialized_payload(
            config,
            type_name=cls.__name__,
            fields=frozenset({"step_size", "secondary_step_size", "trace_decay", "ratio_clip"}),
        ))


def run_gradient_td_learning_loop(
    learner: GradientTDLinearLearner,
    state: GradientTDState,
    observations: Array,
    rewards: Array,
    next_observations: Array,
    gammas: Array,
    rhos: Array,
) -> GradientTDArrayResult:
    """Run Gradient-TD/TDC over arrays using ``jax.lax.scan``."""

    state, feature_dim = _gradient_state_contract(state)
    try:
        observations_shape = tuple(observations.shape)
    except Exception as error:
        raise ValueError("observations must expose array metadata") from error
    if len(observations_shape) != 2 or observations_shape[0] < 1:
        raise ValueError("observations must have shape (num_steps, feature_dim)")
    num_steps = observations_shape[0]
    _require_scan_resources(num_steps, feature_dim)
    if num_steps > _MAX_LEARNING_LOOP_STEPS:
        raise ValueError("num_steps exceeds the learning-loop scan limit")
    _array_metadata("observations", observations, (num_steps, feature_dim))
    _array_metadata("next_observations", next_observations, (num_steps, feature_dim))
    for name, value in (("rewards", rewards), ("gammas", gammas), ("rhos", rhos)):
        _array_metadata(name, value, (num_steps,))

    def step_fn(
        carry: GradientTDState,
        inputs: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[GradientTDState, tuple[Array, Array, Array, Array, Array]]:
        obs, reward, next_obs, gamma, rho = inputs
        result = learner._update_jit(carry, obs, reward, next_obs, gamma, rho)
        return (
            result.state,
            (
                result.prediction[0],
                result.td_error,
                result.rho_clipped,
                result.metrics,
                result.update_applied,
            ),
        )

    t0 = time.time()
    final_state, (predictions, td_errors, rho_clipped, metrics, updates_applied) = jax.lax.scan(
        step_fn,
        state,
        (observations, rewards, next_observations, gammas, rhos),
    )
    elapsed = time.time() - t0
    final_state = final_state.replace(  # type: ignore[attr-defined]
        uptime_s=final_state.uptime_s + elapsed
    )
    return GradientTDArrayResult(  # type: ignore[call-arg]
        state=final_state,
        predictions=predictions,
        td_errors=td_errors,
        rho_clipped=rho_clipped,
        metrics=metrics,
        updates_applied=updates_applied,
    )
