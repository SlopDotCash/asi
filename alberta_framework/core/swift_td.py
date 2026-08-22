# mypy: disable-error-code="call-arg"
"""SwiftTD optimizer for temporal-difference learning.

Implements SwiftTD (Javed, Sharifnassab & Sutton, RLC 2024): True Online
TD(lambda) combined with three ideas that together give fast AND robust
learning from a stream:

1. Per-feature step-size optimization -- IDBD-style meta-learned log
   step-sizes ``beta[i]`` with ``alpha[i] = exp(beta[i])``.
2. An overshoot bound on the update to the eligibility vector -- the
   correction ratio ``tau = sum_i alpha_i * phi_i^2`` of each update is
   capped at ``eta``, so a single update can never move the prediction
   past its target (for ``eta <= 1``).
3. Step-size decay -- whenever the bound is triggered (``tau > eta``)
   the step-sizes of the responsible features are multiplicatively
   decayed, proportional to their contribution ``phi_i^2``.

The implementation follows the author's dense reference implementation
(``SwiftTDNonSparse`` in https://github.com/khurramjaved96/SwiftTD),
restructured into a scan-compatible pure ``update`` over TD transitions
``(observation, td_error, next_observation, gamma)`` so it plugs into the
same loops as :class:`~alberta_framework.core.optimizers.TDIDBD` (e.g.
``TDLinearLearner`` / ``run_td_learning_loop``). Two adaptations:

- The paper's per-time-step loop uses ``delta' = r + gamma*v - v_old``
  where ``v_old`` is the prediction of the previous observation made
  BEFORE the previous weight update. The learner hands us
  ``td_error = r + gamma*V(phi') - V(phi)`` computed with current
  weights, so ``delta' = td_error + v_delta`` where
  ``v_delta = delta_w_prev . phi`` is the change the previous update
  made to the prediction of ``phi``; ``delta_w_prev`` is carried in the
  state. In the supervised limit (``gamma = 0``, fresh traces) the
  correction cancels exactly and SwiftTD reduces to per-feature bounded
  LMS with IDBD-style meta-learning.
- The bias is handled as one extra always-on feature (``phi_bias = 1``)
  appended internally, exactly as the paper treats any other feature --
  state arrays therefore have ``feature_dim + 1`` entries with the bias
  last.

Reference: Javed, K., Sharifnassab, A., & Sutton, R.S. (2024). "SwiftTD:
A Fast and Robust Algorithm for Temporal Difference Learning." RLJ/RLC 2024.
"""

import math
import operator
from typing import Any, SupportsIndex, cast

import chex
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float

from alberta_framework.core._float32_scalars import validated_float32_scalar
from alberta_framework.core.update_safety import (
    floating_tree_is_finite,
    neutralize_array,
    neutralize_metrics,
    select_transaction,
)


def _skip_zero_scale(scale: Array, value: Array) -> Array:
    """Return 0 when ``scale`` is 0 so IEEE ``0 * inf`` does not become NaN."""
    return jnp.where(scale == 0.0, jnp.zeros_like(value), scale * value)


# Paper default for the step-size floor: eta_min = e^-15.
_INT32_MAX = 2**31 - 1
_FLOAT32_MIN_NORMAL = float.fromhex("0x1.0p-126")
_ACTUAL_INT_TYPES = frozenset({int, *(np.dtype(code).type for code in "bBhHiIlLqQpP")})


def _require_int32(name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


def _require_float32_resource(name: str, *, float32_scalars: int) -> None:
    if float32_scalars > _INT32_MAX:
        raise ValueError(f"{name} scalar count must fit signed int32")
    if 4 * float32_scalars > _INT32_MAX:
        raise ValueError(f"{name} byte count must fit signed int32")


def _swift_td_persistent_bytes(feature_dim: int) -> int:
    """Named persist already preflighted by ``SwiftTD.init``."""
    return 4 * (8 * (feature_dim + 1) + 5)


def _swift_td_update_result_extras_bytes(feature_dim: int) -> int:
    """Returned ``SwiftTDUpdate`` extras excluding persist.

    Nested ``new_state`` is persist and is already counted in the simultaneous
    persist copies. These extras are the published weight/bias deltas, metric
    scalars, and diagnostic leaf.
    """
    return 4 * feature_dim + 29


def _swift_td_update_working_set_bytes(feature_dim: int) -> int:
    """Source persist, proposed persist, committed persist, and returned extras."""
    return 3 * _swift_td_persistent_bytes(
        feature_dim
    ) + _swift_td_update_result_extras_bytes(feature_dim)


def _preflight_swift_td_update_working_set(feature_dim: int) -> None:
    """Reject an update envelope the host cannot name in signed int32."""
    if _swift_td_update_working_set_bytes(feature_dim) > _INT32_MAX:
        raise ValueError("SwiftTD update working set byte count must fit signed int32")


_DEFAULT_ETA_MIN = math.exp(-15.0)

_SUPPORTED_CONFIG_REAL_TYPES: tuple[type[object], ...] = (
    int,
    float,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.longlong,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.ulonglong,
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
)


def _require_finite_config_float(
    value: object,
    name: str,
    *,
    minimum: float,
    minimum_inclusive: bool,
    maximum: float | None = None,
) -> float:
    """Return one trusted finite configuration scalar inside its host domain."""
    message = f"{name} must be a finite real scalar in its documented range"
    actual_type = type(value)
    if not any(actual_type is supported_type for supported_type in _SUPPORTED_CONFIG_REAL_TYPES):
        raise ValueError(message)
    try:
        concrete = validated_float32_scalar(
            name,
            value,
            positive=not minimum_inclusive and minimum == 0.0,
            lower=minimum if minimum_inclusive else None,
            upper=maximum,
        )
    except Exception as error:
        raise ValueError(message) from error
    return concrete


def _require_positive_normal_config_float(
    value: object, name: str, *, maximum: float | None = None
) -> float:
    if type(value) not in _SUPPORTED_CONFIG_REAL_TYPES:
        raise ValueError(f"{name} must be a finite real scalar in its documented range")
    try:
        return validated_float32_scalar(name, value, lower=_FLOAT32_MIN_NORMAL, upper=maximum)
    except Exception as error:
        raise ValueError(f"{name} must be a finite real scalar in its documented range") from error


@chex.dataclass(frozen=True)
class SwiftTDState:
    """State for the SwiftTD optimizer.

    All per-feature arrays have ``feature_dim + 1`` entries: the last entry
    is the internal bias feature (``phi_bias = 1``).

    Attributes:
        log_step_sizes: Per-feature log step-sizes ``beta`` (``alpha = exp(beta)``)
        eligibility_traces: Dutch-style eligibility traces ``z``
        z_bar_traces: Secondary eligibility traces ``z_bar`` for the meta-gradient
        p_traces: Meta-gradient traces ``p`` (accumulated ``h_old * phi``)
        h_traces: Step-size adaptation traces ``h``
        h_old_traces: Previous-generation ``h`` traces
        h_temp_traces: Work-in-progress ``h`` traces for the next step
        prev_weight_update: Weight update applied at the previous step
            (used for the True Online ``v_delta`` correction)
        meta_step_size: Meta learning rate theta
        trace_decay: Eligibility trace decay lambda
        eta: Overshoot bound -- maximum correction ratio per update
        eta_min: Floor on per-feature step-sizes
        step_size_decay: Multiplicative decay rate applied when the bound triggers
    """

    log_step_sizes: Float[Array, " aug_dim"]
    eligibility_traces: Float[Array, " aug_dim"]
    z_bar_traces: Float[Array, " aug_dim"]
    p_traces: Float[Array, " aug_dim"]
    h_traces: Float[Array, " aug_dim"]
    h_old_traces: Float[Array, " aug_dim"]
    h_temp_traces: Float[Array, " aug_dim"]
    prev_weight_update: Float[Array, " aug_dim"]
    meta_step_size: Float[Array, ""]
    trace_decay: Float[Array, ""]
    eta: Float[Array, ""]
    eta_min: Float[Array, ""]
    step_size_decay: Float[Array, ""]


@chex.dataclass(frozen=True)
class SwiftTDUpdate:
    """Result of a SwiftTD update step.

    Mirrors :class:`~alberta_framework.core.optimizers.TDOptimizerUpdate`
    so SwiftTD plugs into the same TD learning loops.

    Attributes:
        weight_delta: Change to apply to weights
        bias_delta: Change to apply to bias
        new_state: Updated optimizer state
        metrics: Dictionary of metrics for logging
    """

    weight_delta: Float[Array, " feature_dim"]
    bias_delta: Float[Array, ""]
    new_state: SwiftTDState
    metrics: dict[str, Array]
    update_applied: Bool[Array, ""]


class SwiftTD:
    """SwiftTD optimizer (Javed, Sharifnassab & Sutton, RLC 2024).

    Follows the :class:`~alberta_framework.core.optimizers.TDOptimizer`
    interface (``init(feature_dim)`` and scan-compatible
    ``update(state, td_error, observation, next_observation, gamma)``) so
    it can drive ``TDLinearLearner`` and the TD learning loops. It is not
    a nominal subclass only because its state type lives here rather than
    in ``core/types.py``.

    Attributes:
        initial_step_size: Initial per-feature step-size ``alpha_init``
            (paper default 1e-7; problems with few features can afford
            much larger values)
        meta_step_size: Meta learning rate theta for step-size optimization
        trace_decay: Eligibility trace decay lambda (0 = TD(0))
        eta: Overshoot bound -- maximum correction ratio of one update
            (paper default 0.1)
        step_size_decay: Multiplicative step-size decay rate applied when
            the overshoot bound triggers (paper default 0.99; called
            ``epsilon`` in the paper)
        eta_min: Floor on per-feature step-sizes (paper default ``e^-15``)
    """

    def __init__(
        self,
        initial_step_size: float = 1e-7,
        meta_step_size: float = 1e-3,
        trace_decay: float = 0.0,
        eta: float = 0.1,
        step_size_decay: float = 0.99,
        eta_min: float = _DEFAULT_ETA_MIN,
    ):
        """Initialize SwiftTD optimizer.

        Args:
            initial_step_size: Initial value for per-feature step-sizes
            meta_step_size: Meta learning rate theta for adapting step-sizes
            trace_decay: Eligibility trace decay lambda (0 = TD(0))
            eta: Overshoot bound on the per-update correction ratio
            step_size_decay: Decay rate applied to step-sizes of active
                features when the overshoot bound triggers
            eta_min: Floor on per-feature step-sizes

        Raises:
            ValueError: If a hyperparameter is outside its valid range
        """
        self._initial_step_size = _require_positive_normal_config_float(
            initial_step_size, "initial_step_size"
        )
        self._meta_step_size = _require_finite_config_float(
            meta_step_size,
            "meta_step_size",
            minimum=0.0,
            minimum_inclusive=True,
        )
        self._trace_decay = _require_finite_config_float(
            trace_decay,
            "trace_decay",
            minimum=0.0,
            minimum_inclusive=True,
            maximum=1.0,
        )
        self._eta = _require_positive_normal_config_float(eta, "eta")
        self._step_size_decay = _require_positive_normal_config_float(
            step_size_decay, "step_size_decay", maximum=1.0
        )
        self._eta_min = _require_positive_normal_config_float(eta_min, "eta_min")
        if self._eta_min > self._eta:
            raise ValueError("eta_min must be <= eta")

    def to_config(self) -> dict[str, Any]:
        """Serialize configuration to dict."""
        return {
            "type": "SwiftTD",
            "initial_step_size": self._initial_step_size,
            "meta_step_size": self._meta_step_size,
            "trace_decay": self._trace_decay,
            "eta": self._eta,
            "step_size_decay": self._step_size_decay,
            "eta_min": self._eta_min,
        }

    def init(self, feature_dim: int) -> SwiftTDState:
        """Initialize SwiftTD state.

        Args:
            feature_dim: Dimension of weight vector (excluding the bias;
                internal arrays get one extra entry for the bias feature)

        Returns:
            SwiftTD state with per-feature log step-sizes and zeroed traces
        """
        feature_dim = _require_int32("feature_dim", feature_dim, minimum=1, maximum=_INT32_MAX - 1)
        aug_dim = feature_dim + 1
        _require_float32_resource(
            "SwiftTD state",
            float32_scalars=8 * aug_dim + 5,
        )
        _preflight_swift_td_update_working_set(feature_dim)
        # The dense reference implementation clips beta into
        # [ln(eta_min), ln(eta)] on every pass, including the (otherwise
        # trivial) very first one -- so step-sizes are capped BEFORE the
        # first trace extension. Clip at init to match.
        initial_log_step_size = jnp.clip(
            jnp.log(jnp.float32(self._initial_step_size)),
            jnp.log(jnp.float32(self._eta_min)),
            jnp.log(jnp.float32(self._eta)),
        )
        return SwiftTDState(
            log_step_sizes=jnp.full(aug_dim, initial_log_step_size, dtype=jnp.float32),
            eligibility_traces=jnp.zeros(aug_dim, dtype=jnp.float32),
            z_bar_traces=jnp.zeros(aug_dim, dtype=jnp.float32),
            p_traces=jnp.zeros(aug_dim, dtype=jnp.float32),
            h_traces=jnp.zeros(aug_dim, dtype=jnp.float32),
            h_old_traces=jnp.zeros(aug_dim, dtype=jnp.float32),
            h_temp_traces=jnp.zeros(aug_dim, dtype=jnp.float32),
            prev_weight_update=jnp.zeros(aug_dim, dtype=jnp.float32),
            meta_step_size=jnp.array(self._meta_step_size, dtype=jnp.float32),
            trace_decay=jnp.array(self._trace_decay, dtype=jnp.float32),
            eta=jnp.array(self._eta, dtype=jnp.float32),
            eta_min=jnp.array(self._eta_min, dtype=jnp.float32),
            step_size_decay=jnp.array(self._step_size_decay, dtype=jnp.float32),
        )

    @staticmethod
    def _validate_state_static_contract(state: SwiftTDState) -> int:
        """Return feature_dim after rejecting malformed adopted state metadata."""
        if type(state) is not SwiftTDState:
            raise TypeError("state must be a SwiftTDState")
        vectors = (
            ("state.log_step_sizes", state.log_step_sizes),
            ("state.eligibility_traces", state.eligibility_traces),
            ("state.z_bar_traces", state.z_bar_traces),
            ("state.p_traces", state.p_traces),
            ("state.h_traces", state.h_traces),
            ("state.h_old_traces", state.h_old_traces),
            ("state.h_temp_traces", state.h_temp_traces),
            ("state.prev_weight_update", state.prev_weight_update),
        )
        try:
            augmented_dim = int(state.log_step_sizes.shape[0])
        except Exception as error:
            raise TypeError("state.log_step_sizes must expose array metadata") from error
        if augmented_dim < 2:
            raise ValueError("state vector length must include a positive feature dimension")
        _require_float32_resource("SwiftTD adopted state", float32_scalars=8 * augmented_dim + 5)
        _preflight_swift_td_update_working_set(augmented_dim - 1)
        for name, value in vectors:
            if not hasattr(value, "shape") or not hasattr(value, "dtype"):
                raise TypeError(f"{name} must expose array shape and dtype metadata")
            if tuple(value.shape) != (augmented_dim,):
                raise ValueError(f"{name} must have shape ({augmented_dim},)")
            if jnp.dtype(value.dtype) != jnp.dtype(jnp.float32):
                raise TypeError(f"{name} must have dtype float32")
        for name, value in (
            ("state.meta_step_size", state.meta_step_size),
            ("state.trace_decay", state.trace_decay),
            ("state.eta", state.eta),
            ("state.eta_min", state.eta_min),
            ("state.step_size_decay", state.step_size_decay),
        ):
            if not hasattr(value, "shape") or not hasattr(value, "dtype"):
                raise TypeError(f"{name} must expose array shape and dtype metadata")
            if tuple(value.shape) != ():
                raise ValueError(f"{name} must have scalar shape")
            if jnp.dtype(value.dtype) != jnp.dtype(jnp.float32):
                raise TypeError(f"{name} must have dtype float32")
        return augmented_dim - 1

    @staticmethod
    def _state_is_valid(state: SwiftTDState) -> Bool[Array, ""]:
        return (
            floating_tree_is_finite(state)
            & (state.meta_step_size >= 0.0)
            & (state.trace_decay >= 0.0)
            & (state.trace_decay <= 1.0)
            & (state.eta > 0.0)
            & (state.eta_min > 0.0)
            & (state.eta_min <= state.eta)
            & (state.step_size_decay > 0.0)
            & (state.step_size_decay <= 1.0)
            & jnp.all(state.log_step_sizes >= jnp.log(state.eta_min))
            & jnp.all(state.log_step_sizes <= jnp.log(state.eta))
        )

    def update(
        self,
        state: SwiftTDState,
        td_error: Array,
        observation: Array,
        next_observation: Array,
        gamma: Array,
    ) -> SwiftTDUpdate:
        """Compute one SwiftTD weight update.

        One call performs one iteration of the reference algorithm's
        while-loop, re-phased around the transition ``(phi, r, phi')``:
        first the trace-extension half-step for ``phi`` (which the
        reference performed at the END of its previous iteration --
        overshoot bound, dutch-trace extension, step-size decay), then the
        weight/meta half-step for the TD error of this transition. The
        eligibility decay ``gamma * lambda`` is applied at the end with
        THIS transition's ``gamma``, so traces reset cleanly at terminal
        transitions (``gamma = 0``).

        Args:
            state: Current SwiftTD state
            td_error: TD error ``delta = R + gamma*V(s') - V(s)`` computed
                with current weights (the True Online ``v_old`` correction
                is recovered internally from ``prev_weight_update``)
            observation: Current observation phi(s)
            next_observation: Next observation phi(s') (unused -- its value
                estimate is already folded into ``td_error``)
            gamma: Discount factor gamma (0 at terminal)

        Returns:
            SwiftTDUpdate with weight deltas and updated state
        """
        feature_dim = self._validate_state_static_contract(state)
        for name, value, shape in (
            ("td_error", td_error, ()),
            ("observation", observation, (feature_dim,)),
            ("next_observation", next_observation, (feature_dim,)),
            ("gamma", gamma, ()),
        ):
            if not hasattr(value, "shape") or not hasattr(value, "dtype"):
                raise TypeError(f"{name} must expose array shape and dtype metadata")
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if jnp.dtype(value.dtype) != jnp.dtype(jnp.float32):
                raise TypeError(f"{name} must have dtype float32")
        # V(s') is already folded into td_error, but the public transition is
        # accepted only when every supplied component is finite.
        next_observation_finite = jnp.all(jnp.isfinite(next_observation))
        phi = jnp.concatenate(
            [jnp.asarray(observation, dtype=jnp.float32), jnp.ones(1, dtype=jnp.float32)]
        )
        gamma_scalar = jnp.squeeze(gamma)
        eta = state.eta
        theta = state.meta_step_size

        # --- Trace extension for phi (second loop of Algorithm 1) ---
        # True Online correction: change the previous update made to V(phi).
        v_delta = jnp.dot(state.prev_weight_update, phi)
        alphas = jnp.exp(state.log_step_sizes)
        # Correction ratio tau = sum_i alpha_i * phi_i^2; the overshoot bound
        # caps the effective ratio of this update at eta.
        tau = jnp.sum(alphas * phi**2)
        bound_scale = jnp.minimum(1.0, eta / tau)
        z_delta = bound_scale * alphas * phi
        trace_dot = jnp.dot(state.eligibility_traces, phi)
        z_ext = state.eligibility_traces + z_delta * (1.0 - trace_dot)
        p_ext = state.p_traces + state.h_old_traces * phi
        z_bar_ext = state.z_bar_traces + z_delta * (
            1.0 - trace_dot - state.z_bar_traces * phi
        )
        h_temp_ext = (
            state.h_traces
            - state.h_old_traces * phi * (z_ext - z_delta)
            - state.h_traces * z_delta * phi
        )

        # Step-size decay: when the bound is active, decay the step-sizes of
        # the features responsible (proportional to phi_i^2) and reset the
        # meta-learning traces.
        decay_triggered = tau > eta
        log_alphas = jnp.where(
            decay_triggered,
            state.log_step_sizes + jnp.log(state.step_size_decay) * phi**2,
            state.log_step_sizes,
        )
        h_ext = jnp.where(decay_triggered, 0.0, state.h_traces)
        h_temp_ext = jnp.where(decay_triggered, 0.0, h_temp_ext)
        z_bar_ext = jnp.where(decay_triggered, 0.0, z_bar_ext)
        # (The reference also zeroes h_old here, but that value is always
        # overwritten by the h-shift below before it is next read.)

        # --- Weight and meta update (first loop of Algorithm 1) ---
        # delta' = r + gamma*V(phi') - v_old, with v_old the prediction of
        # phi made before the previous weight update.
        delta = jnp.squeeze(td_error) + v_delta
        delta_w = delta * z_ext - z_delta * v_delta

        # Meta-update: beta_i += (theta / alpha_i) * (delta' - v_delta) * p_i,
        # clipped to [ln(eta_min), ln(eta)].
        meta_delta = (theta / jnp.exp(log_alphas)) * (delta - v_delta) * p_ext
        new_log_step_sizes = jnp.where(
            jnp.isfinite(meta_delta),
            jnp.clip(
                log_alphas + meta_delta,
                jnp.log(state.eta_min),
                jnp.log(eta),
            ),
            log_alphas,
        )

        # h-trace generation shift.
        new_h_old = h_ext
        new_h = h_temp_ext + delta * z_bar_ext - z_delta * v_delta
        new_h_temp = new_h

        # Decay eligibility-side traces for the next transition.
        decay_factor = jnp.where(
            (gamma_scalar == 0.0) | (state.trace_decay == 0.0),
            jnp.zeros_like(gamma_scalar),
            gamma_scalar * state.trace_decay,
        )
        new_z = _skip_zero_scale(decay_factor, z_ext)
        new_p = _skip_zero_scale(decay_factor, p_ext)
        new_z_bar = _skip_zero_scale(decay_factor, z_bar_ext)

        candidate_state = SwiftTDState(
            log_step_sizes=new_log_step_sizes,
            eligibility_traces=new_z,
            z_bar_traces=new_z_bar,
            p_traces=new_p,
            h_traces=new_h,
            h_old_traces=new_h_old,
            h_temp_traces=new_h_temp,
            prev_weight_update=delta_w,
            meta_step_size=theta,
            trace_decay=state.trace_decay,
            eta=eta,
            eta_min=state.eta_min,
            step_size_decay=state.step_size_decay,
        )

        weight_alphas = jnp.exp(new_log_step_sizes[:-1])
        candidate_metrics = {
            "mean_step_size": jnp.mean(weight_alphas),
            "min_step_size": jnp.min(weight_alphas),
            "max_step_size": jnp.max(weight_alphas),
            "mean_eligibility_trace": jnp.mean(jnp.abs(new_z[:-1])),
            "bound_scale": bound_scale,
            "decay_triggered": decay_triggered.astype(jnp.float32),
        }
        inputs_valid = (
            jnp.all(jnp.isfinite(td_error))
            & jnp.all(jnp.isfinite(observation))
            & (next_observation_finite | (gamma_scalar == 0.0))
            & jnp.isfinite(gamma_scalar)
            & (gamma_scalar >= 0.0)
            & (gamma_scalar <= 1.0)
        )
        update_applied = (
            self._state_is_valid(state)
            & inputs_valid
            & jnp.all(jnp.isfinite(meta_delta))
            & jnp.all(jnp.isfinite(delta_w))
            & floating_tree_is_finite(candidate_state)
            & floating_tree_is_finite(candidate_metrics)
        )
        new_state = select_transaction(update_applied, candidate_state, state)
        metrics = neutralize_metrics(update_applied, candidate_metrics)
        return SwiftTDUpdate(
            weight_delta=neutralize_array(update_applied, delta_w[:-1]),
            bias_delta=neutralize_array(update_applied, delta_w[-1]),
            new_state=new_state,
            metrics=metrics,
            update_applied=update_applied,
        )
