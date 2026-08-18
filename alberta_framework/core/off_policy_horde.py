"""Off-policy nonlinear Horde backend for Step 3.

This module adds a JAX/scan-compatible Horde-style learner that accepts one
importance-sampling ratio per demon on every transition.  The implemented
backend is a stable first nonlinear backend: clipped, per-demon, weighted
semi-gradient TD with a shared nonlinear trunk and per-head traces.  It is not
full Gradient-TD/GQ/TDC; those algorithms require secondary weights and MSPBE
correction terms that are still separate from this shared-trunk backend.
"""

from __future__ import annotations

import functools
import math
import operator
import time
from collections.abc import Mapping
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

from alberta_framework.core._float32_scalars import validated_float32_scalar
from alberta_framework.core.learners import _update_from_gradient_with_diagnostics
from alberta_framework.core.multi_head_learner import (
    AnyOptimizer,
    MultiHeadMLPLearner,
    MultiHeadMLPState,
)
from alberta_framework.core.normalizers import (
    EMANormalizerState,
    Normalizer,
    WelfordNormalizerState,
    _saturating_int32_counter_increment,
)
from alberta_framework.core.optimizers import LMS, Bounder
from alberta_framework.core.types import HordeSpec, MLPParams, TraceMode
from alberta_framework.core.update_safety import (
    floating_tree_is_finite as _floating_tree_is_finite,
)

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
_TRUSTED_REAL_TYPES = (
    _ACTUAL_INT_TYPES | frozenset(np.dtype(code).type for code in ("e", "f", "d", "g")) | {float}
)


def _has_exact_type(value: object, allowed: frozenset[type]) -> bool:
    """Match a concrete type without invoking an untrusted metaclass hook."""
    actual_type = type(value)
    return any(actual_type is allowed_type for allowed_type in allowed)


def _require_int32(name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX) -> int:
    if not _has_exact_type(value, _ACTUAL_INT_TYPES):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


def _require_float32(name: str, value: object, **bounds: Any) -> float:
    if not _has_exact_type(value, _TRUSTED_REAL_TYPES):
        raise ValueError(f"{name} must be a finite real scalar")
    return validated_float32_scalar(name, value, **bounds)


def _require_positive_clip(name: str, value: object) -> float:
    """Positive float32 clip that keeps +inf legal as the no-clip sentinel.

    The Baird counterexample tests drive the learner with effectively
    infinite clips; rejecting +inf breaks that documented usage.
    """
    if type(value) is float and math.isinf(value) and value > 0.0:
        return value
    return _require_float32(name, value, positive=True)


def _preflight_nonlinear_state(*, n_demons: int, hidden_size: int, feature_dim: int) -> None:
    hidden_features = hidden_size * feature_dim
    demon_hidden_features = n_demons * hidden_features
    demon_hidden = n_demons * hidden_size
    float_scalars = (
        hidden_features * (n_demons + 1) + hidden_size + 3 * demon_hidden + 2 * n_demons + 2
    )
    logical_scalars = float_scalars + 1
    for name, value in (
        ("trunk weight scalars", hidden_features),
        ("secondary trunk weight scalars", demon_hidden_features),
        ("demon-hidden scalars", demon_hidden),
        ("persistent state scalars", logical_scalars),
        ("persistent state bytes", 4 * logical_scalars),
    ):
        if not 1 <= value <= _INT32_MAX:
            raise ValueError(f"derived nonlinear Horde {name} must fit signed int32")
    # Conservatively charge every source-level aggregate that can coexist at
    # publication: source, proposed, and selected states; the primary steps
    # plus retained per-demon secondary proposals; and the current, next, and
    # correction gradient families.  The tail covers the two observations and
    # all per-demon masks, predictions, targets, errors, norms, and result
    # diagnostics.  Charging logical leaves as four-byte scalars also covers
    # the int32 counter and boolean predicates.
    parameter_scalars = (
        hidden_features * (n_demons + 1)
        + hidden_size
        + 3 * demon_hidden
        + 2 * n_demons
    )
    one_gradient_scalars = hidden_features + 2 * hidden_size + 1
    update_scalars = (
        3 * logical_scalars
        + parameter_scalars
        + 3 * one_gradient_scalars
        # Current observation, next observation, and the finite-safe next copy.
        + 3 * feature_dim
        # Current/next hidden activations and their two tanh-derivative vectors.
        + 4 * hidden_size
        + 24 * n_demons
        + 32
    )
    if 4 * update_scalars > _INT32_MAX:
        raise ValueError(
            "derived nonlinear Horde update working set byte count must fit signed int32"
        )


def _require_typed_threefry_key(name: str, value: object) -> Array:
    """Reject legacy or non-Threefry keys before random operations."""
    if not isinstance(value, jax.Array) or tuple(value.shape) != ():
        raise ValueError(f"{name} must be a typed scalar threefry2x32 key")
    try:
        implementation = str(jax.random.key_impl(value))
        words = jax.random.key_data(value)
    except Exception as error:
        raise ValueError(f"{name} must be a typed scalar threefry2x32 key") from error
    if implementation != "threefry2x32" or tuple(words.shape) != (2,):
        raise ValueError(f"{name} must be a typed scalar threefry2x32 key")
    return value


def _stable_l2_norm(*values: Array) -> Array:
    """Compute a joint float32 L2 norm without squaring at the input scale."""
    scale = jnp.asarray(0.0, dtype=jnp.float32)
    for value in values:
        scale = jnp.maximum(scale, jnp.max(jnp.abs(value), initial=0.0))
    square_sum = jnp.asarray(0.0, dtype=jnp.float32)
    for value in values:
        normalized = jnp.where(scale > 0.0, value / scale, jnp.zeros_like(value))
        square_sum = square_sum + jnp.sum(jnp.square(normalized))
    return scale * jnp.sqrt(square_sum)


def _skip_zero_scale(scale: Array, value: Array) -> Array:
    """Return 0 when ``scale`` is 0 so a 0*inf product cannot form."""
    return jnp.where(scale == 0.0, jnp.zeros_like(value), scale * value)


def _report_demon_values(
    values: Array,
    requested: Array,
    updates_applied: Array,
) -> Array:
    """Preserve inactive NaN sentinels and neutralize rejected demons."""
    mask_shape = (requested.shape[0],) + (1,) * (values.ndim - 1)
    requested_mask = jnp.reshape(requested, mask_shape)
    applied_mask = jnp.reshape(updates_applied, mask_shape)
    return jnp.where(
        applied_mask,
        values,
        jnp.where(
            requested_mask,
            jnp.zeros_like(values),
            jnp.full_like(values, jnp.nan),
        ),
    )


def _extract_mean_step_size(opt_state: Any) -> Array:
    """Extract a scalar mean step-size from an optimizer state."""
    if hasattr(opt_state, "step_sizes"):
        return jnp.mean(opt_state.step_sizes)
    if hasattr(opt_state, "log_step_sizes"):
        return jnp.mean(jnp.exp(opt_state.log_step_sizes))
    if hasattr(opt_state, "step_size"):
        return jnp.asarray(opt_state.step_size, dtype=jnp.float32)
    return jnp.array(0.0, dtype=jnp.float32)


@chex.dataclass(frozen=True)
class OffPolicyHordeUpdateResult:
    """Result of one off-policy Horde update.

    Attributes:
        state: Updated shared-trunk multi-head learner state.
        predictions: Predictions at ``s_t``, shape ``(n_demons,)``.
        next_predictions: Bootstrap predictions at ``s_{t+1}``.
        td_targets: TD targets ``c_t + gamma_t V(s_{t+1})``.
        td_errors: Unweighted TD errors.
        rhos: Raw importance-sampling ratios.
        clipped_rhos: Ratios after update clipping.
        trace_coefficients: Ratios after trace clipping.
        per_demon_metrics: Shape ``(n_demons, 6)`` with columns
            ``[squared_td_error, td_error, rho, clipped_rho, trace_coeff,
            mean_step_size]``.
        trunk_bounding_metric: Scalar metric returned by the bounder.
    """

    state: MultiHeadMLPState
    predictions: Float[Array, " n_demons"]
    next_predictions: Float[Array, " n_demons"]
    td_targets: Float[Array, " n_demons"]
    td_errors: Float[Array, " n_demons"]
    rhos: Float[Array, " n_demons"]
    clipped_rhos: Float[Array, " n_demons"]
    trace_coefficients: Float[Array, " n_demons"]
    per_demon_metrics: Float[Array, "n_demons 6"]
    trunk_bounding_metric: Float[Array, ""]
    head_updates_applied: Bool[Array, " n_demons"]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class OffPolicyHordeLearningResult:
    """Result from a scan-based off-policy Horde learning loop."""

    state: MultiHeadMLPState
    per_demon_metrics: Float[Array, "num_steps n_demons 6"]
    td_errors: Float[Array, "num_steps n_demons"]
    clipped_rhos: Float[Array, "num_steps n_demons"]
    head_updates_applied: Bool[Array, "num_steps n_demons"]
    updates_applied: Bool[Array, " num_steps"]


@chex.dataclass(frozen=True)
class NonlinearSharedGTDHordeState:
    """State for a single-hidden-layer shared-trunk Gradient-TD Horde.

    The secondary weights are stored per demon and match the nonzero gradient
    support for that demon: shared trunk parameters plus that demon's output
    head. This is the corrected off-policy backend; unlike
    :class:`OffPolicyHordeLearner`, it carries secondary weights.
    """

    trunk_w: Float[Array, "hidden_dim feature_dim"]
    trunk_b: Float[Array, " hidden_dim"]
    head_w: Float[Array, "n_demons hidden_dim"]
    head_b: Float[Array, " n_demons"]
    secondary_trunk_w: Float[Array, "n_demons hidden_dim feature_dim"]
    secondary_trunk_b: Float[Array, "n_demons hidden_dim"]
    secondary_head_w: Float[Array, "n_demons hidden_dim"]
    secondary_head_b: Float[Array, " n_demons"]
    step_count: Int[Array, ""]
    birth_timestamp: Float[Array, ""]
    uptime_s: Float[Array, ""]


@chex.dataclass(frozen=True)
class NonlinearSharedGTDHordeUpdateResult:
    """Result from one corrected nonlinear shared-trunk off-policy update."""

    state: NonlinearSharedGTDHordeState
    predictions: Float[Array, " n_demons"]
    next_predictions: Float[Array, " n_demons"]
    td_targets: Float[Array, " n_demons"]
    td_errors: Float[Array, " n_demons"]
    clipped_rhos: Float[Array, " n_demons"]
    correction_norms: Float[Array, " n_demons"]
    secondary_norms: Float[Array, " n_demons"]
    head_updates_applied: Bool[Array, " n_demons"]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class NonlinearSharedGTDHordeLearningResult:
    """Scan result for corrected nonlinear shared-trunk off-policy Horde."""

    state: NonlinearSharedGTDHordeState
    td_errors: Float[Array, "num_steps n_demons"]
    clipped_rhos: Float[Array, "num_steps n_demons"]
    correction_norms: Float[Array, "num_steps n_demons"]
    secondary_norms: Float[Array, "num_steps n_demons"]


class OffPolicyHordeLearner:
    """Nonlinear off-policy Horde with per-demon importance ratios.

    The update is a clipped per-decision importance-sampled semi-gradient
    TD(lambda) backend (Sutton & Barto 2nd ed., eq. 12.23; GQ(lambda),
    Maei & Sutton 2010):

    ``delta_i = c_i + gamma_i V_i(s') - V_i(s)``

    ``z_i <- gamma_i lambda_i min(rho_i, trace_ratio_clip) z_i
    + min(rho_i, ratio_clip) grad V_i(s)``

    ``w_i <- w_i + alpha delta_i z_i``

    Each decision's ratio enters the trace exactly once; with the clips
    inactive this is the canonical ``z_t = rho_t (gamma lambda z_{t-1}
    + grad)`` with update ``delta_t z_t``.  The trace-free shared trunk
    receives the summed current-step cotangent
    ``sum_i min(rho_i, ratio_clip) delta_i grad_h V_i(s)``.  This keeps the
    nonlinear shared trunk on the same conservative footing as
    ``HordeLearner`` while making head traces and all demon updates
    ratio-aware.

    Full GTD/GQ/TDC MSPBE correction is intentionally out of scope for this
    first backend because it requires secondary weights and a different
    objective; the corrected variant with secondary weights is
    :class:`NonlinearSharedGTDHordeLearner` in this module.
    """

    def __init__(
        self,
        horde_spec: HordeSpec,
        hidden_sizes: tuple[int, ...] = (128, 128),
        optimizer: AnyOptimizer | None = None,
        step_size: float = 0.01,
        bounder: Bounder | None = None,
        normalizer: (
            Normalizer[EMANormalizerState] | Normalizer[WelfordNormalizerState] | None
        ) = None,
        sparsity: float = 0.9,
        leaky_relu_slope: float = 0.01,
        use_layer_norm: bool = True,
        head_optimizer: AnyOptimizer | None = None,
        trace_mode: TraceMode = TraceMode.ACCUMULATING,
        utility_decay: float = 0.99,
        ratio_clip: float = 1.0,
        trace_ratio_clip: float = 1.0,
        min_behavior_probability: float = 1e-6,
    ):
        """Initialize an off-policy Horde backend.

        Args:
            horde_spec: GVF metadata, one demon per head.
            hidden_sizes: Shared trunk hidden sizes. ``()`` gives linear heads.
            optimizer: Optimizer for trunk and heads unless ``head_optimizer``
                is provided.
            step_size: LMS step-size used when ``optimizer`` is omitted.
            bounder: Optional update bounder.
            normalizer: Optional online input normalizer.
            sparsity: Sparse initialization fraction.
            leaky_relu_slope: LeakyReLU negative slope.
            use_layer_norm: Whether the trunk uses parameterless layer norm.
            head_optimizer: Optional separate output-head optimizer.
            trace_mode: Accumulating or replacing head traces.
            utility_decay: Hidden-unit utility EMA decay.
            ratio_clip: Clip for the current TD update ratio.
            trace_ratio_clip: Clip for the eligibility-trace ratio.
            min_behavior_probability: Denominator floor for probability API.
        """
        ratio_clip = _require_positive_clip("ratio_clip", ratio_clip)
        trace_ratio_clip = _require_positive_clip("trace_ratio_clip", trace_ratio_clip)
        min_behavior_probability = _require_float32(
            "min_behavior_probability", min_behavior_probability, positive=True
        )

        self._horde_spec = horde_spec
        self._hidden_sizes = hidden_sizes
        self._optimizer: AnyOptimizer = optimizer or LMS(step_size=step_size)
        self._head_optimizer = head_optimizer
        self._bounder = bounder
        self._normalizer = normalizer
        self._sparsity = sparsity
        self._leaky_relu_slope = leaky_relu_slope
        self._use_layer_norm = use_layer_norm
        self._trace_mode = trace_mode
        self._utility_decay = utility_decay
        self._ratio_clip = ratio_clip
        self._trace_ratio_clip = trace_ratio_clip
        self._min_behavior_probability = min_behavior_probability

        # The wrapped learner supplies initialization, prediction, optimizer
        # states, normalizer state, and MLP forward utilities.  This backend
        # owns the update rule because off-policy ratios are transition-local.
        self._learner = MultiHeadMLPLearner(
            n_heads=len(horde_spec.demons),
            hidden_sizes=hidden_sizes,
            optimizer=self._optimizer,
            step_size=step_size,
            bounder=bounder,
            gamma=0.0,
            lamda=0.0,
            normalizer=normalizer,
            sparsity=sparsity,
            leaky_relu_slope=leaky_relu_slope,
            use_layer_norm=use_layer_norm,
            head_optimizer=head_optimizer,
            per_head_gamma_lamda=tuple(0.0 for _ in horde_spec.demons),
            trace_mode=trace_mode,
            utility_decay=utility_decay,
        )

    @property
    def horde_spec(self) -> HordeSpec:
        """The GVF specification."""
        return self._horde_spec

    @property
    def n_demons(self) -> int:
        """Number of demons."""
        return len(self._horde_spec.demons)

    @property
    def learner(self) -> MultiHeadMLPLearner:
        """Underlying multi-head MLP learner used for init/predict."""
        return self._learner

    @property
    def ratio_clip(self) -> float:
        """Current-step update ratio clip."""
        return self._ratio_clip

    @property
    def trace_ratio_clip(self) -> float:
        """Eligibility-trace ratio clip."""
        return self._trace_ratio_clip

    def init(self, feature_dim: int, key: Array) -> MultiHeadMLPState:
        """Initialize learner state."""
        return self._learner.init(feature_dim, key)

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(self, state: MultiHeadMLPState, observation: Array) -> Array:
        """Predict all demon values for one observation."""
        return self._learner.predict(state, observation)  # type: ignore[no-any-return]

    def update(
        self,
        state: MultiHeadMLPState,
        observation: Array,
        cumulants: Array,
        next_observation: Array,
        rhos: Array,
    ) -> OffPolicyHordeUpdateResult:
        """Alias for :meth:`update_with_ratios`."""
        return self.update_with_ratios(
            state,
            observation,
            cumulants,
            next_observation,
            rhos,
        )

    def update_with_ratios(
        self,
        state: MultiHeadMLPState,
        observation: Array,
        cumulants: Array,
        next_observation: Array,
        rhos: Array,
    ) -> OffPolicyHordeUpdateResult:
        """Update using explicit per-demon importance ratios."""
        return cast(
            OffPolicyHordeUpdateResult,
            self.update_with_ratios_and_discounts(
                state,
                observation,
                cumulants,
                next_observation,
                rhos,
                self._horde_spec.gammas,
            ),
        )

    def update_with_probabilities(
        self,
        state: MultiHeadMLPState,
        observation: Array,
        cumulants: Array,
        next_observation: Array,
        target_probabilities: Array,
        behavior_probabilities: Array,
    ) -> OffPolicyHordeUpdateResult:
        """Update from target/behavior probabilities instead of ratios."""
        behavior = jnp.maximum(
            behavior_probabilities,
            jnp.asarray(self._min_behavior_probability, dtype=jnp.float32),
        )
        rhos = target_probabilities / behavior
        return self.update_with_ratios(
            state,
            observation,
            cumulants,
            next_observation,
            rhos,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update_with_ratios_and_discounts(
        self,
        state: MultiHeadMLPState,
        observation: Array,
        cumulants: Array,
        next_observation: Array,
        rhos: Array,
        discounts: Array,
    ) -> OffPolicyHordeUpdateResult:
        """Update using explicit ratios and transition discounts."""
        n_demons = self.n_demons
        replacing = self._trace_mode == TraceMode.REPLACING
        counter_status = self._learner._counter_status(state)

        rhos = jnp.asarray(rhos, dtype=jnp.float32)
        discounts = jnp.asarray(discounts, dtype=jnp.float32)
        clipped_rhos = jnp.minimum(
            jnp.maximum(rhos, 0.0),
            jnp.asarray(self._ratio_clip, dtype=jnp.float32),
        )
        trace_coefficients = jnp.minimum(
            jnp.maximum(rhos, 0.0),
            jnp.asarray(self._trace_ratio_clip, dtype=jnp.float32),
        )

        next_predictions = self._learner.predict(state, next_observation)
        zero_discount_mask = discounts == 0.0
        bootstrap_predictions = jnp.where(
            zero_discount_mask,
            jnp.zeros_like(next_predictions),
            next_predictions,
        )
        td_targets = cumulants + discounts * bootstrap_predictions
        requested_mask = ~jnp.isnan(cumulants)
        checked_state = state
        if self._utility_decay == 0.0:
            checked_state = checked_state.replace(  # type: ignore[attr-defined]
                hidden_unit_utilities=tuple(
                    jnp.zeros_like(utility) for utility in state.hidden_unit_utilities
                )
            )
        if replacing:
            checked_state = checked_state.replace(  # type: ignore[attr-defined]
                trunk_traces=tuple(jnp.zeros_like(trace) for trace in state.trunk_traces),
            )
        lamdas = self._horde_spec.lamdas
        head_decay_unused = jnp.all(
            (discounts == 0.0) | (jnp.asarray(lamdas, dtype=jnp.float32) == 0.0)
        )
        checked_state = checked_state.replace(  # type: ignore[attr-defined]
            head_traces=tuple(
                (
                    jnp.where(head_decay_unused, jnp.zeros_like(old_w), old_w),
                    jnp.where(head_decay_unused, jnp.zeros_like(old_b), old_b),
                )
                for old_w, old_b in state.head_traces
            )
        )
        source_state_finite = _floating_tree_is_finite(checked_state)
        global_inputs_valid = jnp.all(jnp.isfinite(observation))
        next_observation_valid = jnp.all(jnp.isfinite(next_observation))
        head_inputs_valid = (
            requested_mask
            & jnp.isfinite(cumulants)
            & jnp.isfinite(rhos)
            & jnp.isfinite(discounts)
            & (discounts >= 0.0)
            & (discounts <= 1.0)
            & (zero_discount_mask | (next_observation_valid & jnp.isfinite(next_predictions)))
            & jnp.isfinite(td_targets)
        )
        active_mask = head_inputs_valid & global_inputs_valid & source_state_finite
        safe_targets = jnp.where(active_mask, td_targets, 0.0)
        safe_clipped_rhos = jnp.where(active_mask, clipped_rhos, 0.0)
        safe_trace_coefficients = jnp.where(active_mask, trace_coefficients, 0.0)
        safe_discounts = jnp.where(active_mask, discounts, 0.0)

        obs = observation
        new_normalizer_state = state.normalizer_state
        normalizer_update_applied = jnp.asarray(True, dtype=jnp.bool_)
        if self._normalizer is not None and state.normalizer_state is not None:
            normalizer: Any = self._normalizer
            normalizer_result = normalizer.normalize_with_diagnostics(
                state.normalizer_state,
                observation,
            )
            obs = normalizer_result.normalized
            new_normalizer_state = normalizer_result.state
            normalizer_update_applied = normalizer_result.update_applied

        slope = self._leaky_relu_slope
        use_layer_norm = self._use_layer_norm

        def trunk_fn(
            weights: tuple[Array, ...],
            biases: tuple[Array, ...],
        ) -> Array:
            return MultiHeadMLPLearner._trunk_forward(
                weights,
                biases,
                obs,
                slope,
                use_layer_norm,
            )

        hidden, trunk_vjp_fn = jax.vjp(
            trunk_fn,
            state.trunk_params.weights,
            state.trunk_params.biases,
        )
        _, activations = MultiHeadMLPLearner._trunk_forward_with_activations(
            state.trunk_params.weights,
            state.trunk_params.biases,
            obs,
            self._leaky_relu_slope,
            self._use_layer_norm,
        )

        cotangent = jnp.zeros(hidden.shape[0], dtype=jnp.float32)
        predictions_list: list[Array] = []
        td_errors_list: list[Array] = []
        masked_td_errors_list: list[Array] = []

        for i in range(n_demons):
            pred_i = MultiHeadMLPLearner._head_forward(
                state.head_params.weights[i],
                state.head_params.biases[i],
                hidden,
            )
            td_error_i = safe_targets[i] - pred_i
            masked_td_error_i = jnp.where(active_mask[i], td_error_i, 0.0)
            effective_error_i = safe_clipped_rhos[i] * masked_td_error_i

            predictions_list.append(pred_i)
            td_errors_list.append(jnp.where(active_mask[i], td_error_i, jnp.nan))
            masked_td_errors_list.append(masked_td_error_i)
            cotangent = cotangent + effective_error_i * jnp.squeeze(state.head_params.weights[i])

        predictions = jnp.stack(predictions_list)
        td_errors = jnp.stack(td_errors_list)
        masked_td_errors = jnp.stack(masked_td_errors_list)

        trunk_weight_grads, trunk_bias_grads = trunk_vjp_fn(cotangent)

        utility_decay = jnp.asarray(self._utility_decay, dtype=jnp.float32)
        new_hidden_unit_utilities: list[Array] = []
        for i in range(len(activations)):
            old_utility = (
                state.hidden_unit_utilities[i]
                if len(state.hidden_unit_utilities) > i
                else jnp.zeros_like(activations[i])
            )
            utility_signal = jnp.abs(activations[i] * trunk_bias_grads[i])
            new_hidden_unit_utilities.append(
                _skip_zero_scale(utility_decay, old_utility)
                + (1.0 - utility_decay) * utility_signal
            )

        new_trunk_traces: list[Array] = []
        trunk_steps: list[Array] = []
        new_trunk_opt_states: list[Any] = []
        optimizer_updates_applied: list[Array] = []
        n_trunk_layers = len(state.trunk_params.weights)

        for i in range(n_trunk_layers):
            w_grad_i = trunk_weight_grads[i]
            old_w_trace = state.trunk_traces[2 * i]
            if replacing:
                new_w_trace = jnp.where(w_grad_i != 0.0, w_grad_i, jnp.zeros_like(old_w_trace))
            else:
                new_w_trace = w_grad_i
            new_trunk_traces.append(new_w_trace)
            w_step, new_w_opt, w_update_applied = _update_from_gradient_with_diagnostics(
                self._optimizer,
                state.trunk_optimizer_states[2 * i],
                new_w_trace,
                error=None,
            )
            trunk_steps.append(w_step)
            new_trunk_opt_states.append(new_w_opt)
            optimizer_updates_applied.append(w_update_applied)

            b_grad_i = trunk_bias_grads[i]
            old_b_trace = state.trunk_traces[2 * i + 1]
            if replacing:
                new_b_trace = jnp.where(b_grad_i != 0.0, b_grad_i, jnp.zeros_like(old_b_trace))
            else:
                new_b_trace = b_grad_i
            new_trunk_traces.append(new_b_trace)
            b_step, new_b_opt, b_update_applied = _update_from_gradient_with_diagnostics(
                self._optimizer,
                state.trunk_optimizer_states[2 * i + 1],
                new_b_trace,
                error=None,
            )
            trunk_steps.append(b_step)
            new_trunk_opt_states.append(new_b_opt)
            optimizer_updates_applied.append(b_update_applied)

        trunk_bounding_metric = jnp.array(1.0, dtype=jnp.float32)
        if self._bounder is not None and n_trunk_layers > 0:
            trunk_params_flat: list[Array] = []
            for i in range(n_trunk_layers):
                trunk_params_flat.append(state.trunk_params.weights[i])
                trunk_params_flat.append(state.trunk_params.biases[i])
            bounded_trunk_steps, trunk_bounding_metric = self._bounder.bound(
                tuple(trunk_steps),
                jnp.array(1.0, dtype=jnp.float32),
                tuple(trunk_params_flat),
            )
            trunk_steps = list(bounded_trunk_steps)
            new_trunk_traces = [trunk_bounding_metric * t for t in new_trunk_traces]

        new_trunk_weights: list[Array] = []
        new_trunk_biases: list[Array] = []
        for i in range(n_trunk_layers):
            new_trunk_weights.append(state.trunk_params.weights[i] + trunk_steps[2 * i])
            new_trunk_biases.append(state.trunk_params.biases[i] + trunk_steps[2 * i + 1])

        new_trunk_params = MLPParams(
            weights=tuple(new_trunk_weights),
            biases=tuple(new_trunk_biases),
        )  # type: ignore[call-arg]

        new_head_weights: list[Array] = []
        new_head_biases: list[Array] = []
        new_head_traces: list[tuple[Array, Array]] = []
        new_head_opt_states: list[tuple[Any, Any]] = []
        per_demon_metrics: list[Array] = []
        head_candidates_finite: list[Array] = []
        head_optimizer = self._head_optimizer or self._optimizer
        lamdas = self._horde_spec.lamdas

        for i in range(n_demons):
            head_w = state.head_params.weights[i]
            head_b = state.head_params.biases[i]
            old_w_trace, old_b_trace = state.head_traces[i]
            old_w_opt, old_b_opt = state.head_optimizer_states[i]

            # Per-decision IS: the current ratio scales this step's gradient
            # as it enters the trace, and the trace decay carries the
            # (trace-clipped) ratio forward, so ``z = rho (gl z + grad)``
            # when the two clips agree.  The update below is ``delta * z``
            # with no additional ratio.
            safe_hidden = jnp.where(active_mask[i], hidden, jnp.zeros_like(hidden))
            w_grad = safe_clipped_rhos[i] * safe_hidden.reshape(1, -1)
            b_grad = safe_clipped_rhos[i] * jnp.ones(1, dtype=jnp.float32)
            head_gl = safe_discounts[i] * lamdas[i] * safe_trace_coefficients[i]

            if replacing:
                new_w_trace = jnp.where(
                    w_grad != 0.0,
                    w_grad,
                    _skip_zero_scale(head_gl, old_w_trace),
                )
                new_b_trace = jnp.where(
                    b_grad != 0.0,
                    b_grad,
                    _skip_zero_scale(head_gl, old_b_trace),
                )
            else:
                new_w_trace = _skip_zero_scale(head_gl, old_w_trace) + w_grad
                new_b_trace = _skip_zero_scale(head_gl, old_b_trace) + b_grad

            error_i = masked_td_errors[i]
            w_step, new_w_opt, w_update_applied = _update_from_gradient_with_diagnostics(
                head_optimizer,
                old_w_opt,
                new_w_trace,
                error=error_i,
            )
            b_step, new_b_opt, b_update_applied = _update_from_gradient_with_diagnostics(
                head_optimizer,
                old_b_opt,
                new_b_trace,
                error=error_i,
            )
            optimizer_updates_applied.extend((w_update_applied, b_update_applied))

            if self._bounder is not None:
                bounded_head_steps, bound_scale = self._bounder.bound(
                    (w_step, b_step),
                    error_i,
                    (head_w, head_b),
                )
                w_step, b_step = bounded_head_steps
                new_w_trace = bound_scale * new_w_trace
                new_b_trace = bound_scale * new_b_trace

            new_w = head_w + error_i * w_step
            new_b = head_b + error_i * b_step
            # Inf TD error zeros the ObGD step, then error_i * step is 0*inf=NaN.
            # Hold that head's previous finite params/traces/opt like a NaN cumulant.
            head_ok = (
                active_mask[i]
                & jnp.isfinite(error_i)
                & _floating_tree_is_finite(
                    (
                        new_w,
                        new_b,
                        new_w_trace,
                        new_b_trace,
                        new_w_opt,
                        new_b_opt,
                    )
                )
            )
            head_candidates_finite.append((~active_mask[i]) | head_ok)
            new_w = jnp.where(head_ok, new_w, head_w)
            new_b = jnp.where(head_ok, new_b, head_b)
            new_w_trace = jnp.where(head_ok, new_w_trace, old_w_trace)
            new_b_trace = jnp.where(head_ok, new_b_trace, old_b_trace)
            new_w_opt = jax.tree.map(
                lambda new, old: jnp.where(head_ok, new, old),
                new_w_opt,
                old_w_opt,
            )
            new_b_opt = jax.tree.map(
                lambda new, old: jnp.where(head_ok, new, old),
                new_b_opt,
                old_b_opt,
            )

            new_head_weights.append(new_w)
            new_head_biases.append(new_b)
            new_head_traces.append((new_w_trace, new_b_trace))
            new_head_opt_states.append((new_w_opt, new_b_opt))

            se_i = jnp.where(active_mask[i], td_errors[i] ** 2, jnp.nan)
            raw_error_i = jnp.where(active_mask[i], td_errors[i], jnp.nan)
            rho_i = jnp.where(active_mask[i], rhos[i], jnp.nan)
            clipped_rho_i = jnp.where(active_mask[i], clipped_rhos[i], jnp.nan)
            trace_coeff_i = jnp.where(
                active_mask[i],
                trace_coefficients[i],
                jnp.nan,
            )
            mean_ss_i = jnp.where(
                active_mask[i],
                _extract_mean_step_size(new_w_opt),
                jnp.nan,
            )
            per_demon_metrics.append(
                jnp.array(
                    [
                        se_i,
                        raw_error_i,
                        rho_i,
                        clipped_rho_i,
                        trace_coeff_i,
                        mean_ss_i,
                    ]
                )
            )

        new_head_params = MLPParams(
            weights=tuple(new_head_weights),
            biases=tuple(new_head_biases),
        )  # type: ignore[call-arg]
        new_state = MultiHeadMLPState(
            trunk_params=new_trunk_params,
            head_params=new_head_params,
            trunk_optimizer_states=tuple(new_trunk_opt_states),
            head_optimizer_states=tuple(new_head_opt_states),
            trunk_traces=tuple(new_trunk_traces),
            head_traces=tuple(new_head_traces),
            hidden_unit_utilities=tuple(new_hidden_unit_utilities),
            normalizer_state=new_normalizer_state,
            step_count=_saturating_int32_counter_increment(state.step_count),
            step_words=counter_status.proposed_step_words,
            birth_timestamp=state.birth_timestamp,
            uptime_s=state.uptime_s,
        )  # type: ignore[call-arg]

        candidate_state_finite = _floating_tree_is_finite(new_state)
        all_active_candidates_finite = jnp.all(jnp.stack(head_candidates_finite))
        has_accepted_work = jnp.any(active_mask) | jnp.all(~requested_mask)
        update_applied = (
            counter_status.update_available
            & global_inputs_valid
            & source_state_finite
            & normalizer_update_applied
            & all_active_candidates_finite
            & candidate_state_finite
            & has_accepted_work
            & jnp.all(jnp.stack(optimizer_updates_applied))
        )
        head_updates_applied = active_mask & update_applied
        committed_state = jax.lax.cond(update_applied, lambda: new_state, lambda: state)
        raw_metrics = jnp.stack(per_demon_metrics)
        reported_predictions = jnp.where(
            update_applied,
            jnp.where(
                requested_mask & ~head_updates_applied,
                jnp.zeros_like(predictions),
                predictions,
            ),
            jnp.zeros_like(predictions),
        )
        sanitized_next_predictions = jnp.where(
            head_updates_applied & zero_discount_mask & ~jnp.isfinite(next_predictions),
            jnp.zeros_like(next_predictions),
            next_predictions,
        )
        reported_next_predictions = jnp.where(
            update_applied,
            jnp.where(
                requested_mask & ~head_updates_applied,
                jnp.zeros_like(next_predictions),
                sanitized_next_predictions,
            ),
            jnp.zeros_like(next_predictions),
        )

        return OffPolicyHordeUpdateResult(
            state=committed_state,
            predictions=reported_predictions,
            next_predictions=reported_next_predictions,
            td_targets=_report_demon_values(td_targets, requested_mask, head_updates_applied),
            td_errors=_report_demon_values(td_errors, requested_mask, head_updates_applied),
            rhos=jnp.where(head_updates_applied, rhos, jnp.zeros_like(rhos)),
            clipped_rhos=_report_demon_values(clipped_rhos, requested_mask, head_updates_applied),
            trace_coefficients=_report_demon_values(
                trace_coefficients, requested_mask, head_updates_applied
            ),
            per_demon_metrics=_report_demon_values(
                raw_metrics, requested_mask, head_updates_applied
            ),
            trunk_bounding_metric=jnp.where(
                update_applied,
                trunk_bounding_metric,
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            head_updates_applied=head_updates_applied,
            update_applied=update_applied,
        )  # type: ignore[call-arg]

    def to_config(self) -> dict[str, Any]:
        """Serialize learner configuration."""
        return {
            "type": "OffPolicyHordeLearner",
            "horde_spec": self._horde_spec.to_config(),
            "hidden_sizes": list(self._hidden_sizes),
            "optimizer": self._optimizer.to_config(),
            "bounder": self._bounder.to_config() if self._bounder else None,
            "normalizer": self._normalizer.to_config() if self._normalizer else None,
            "sparsity": self._sparsity,
            "leaky_relu_slope": self._leaky_relu_slope,
            "use_layer_norm": self._use_layer_norm,
            "head_optimizer": (self._head_optimizer.to_config() if self._head_optimizer else None),
            "trace_mode": self._trace_mode.value,
            "utility_decay": self._utility_decay,
            "ratio_clip": self._ratio_clip,
            "trace_ratio_clip": self._trace_ratio_clip,
            "min_behavior_probability": self._min_behavior_probability,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> OffPolicyHordeLearner:
        """Reconstruct a learner from :meth:`to_config`."""
        from alberta_framework.core.normalizers import normalizer_from_config
        from alberta_framework.core.optimizers import (
            bounder_from_config,
            optimizer_from_config,
        )

        config = dict(config)
        config.pop("type", None)
        horde_spec = HordeSpec.from_config(config.pop("horde_spec"))
        optimizer = optimizer_from_config(config.pop("optimizer"))
        bounder_cfg = config.pop("bounder", None)
        bounder = bounder_from_config(bounder_cfg) if bounder_cfg else None
        normalizer_cfg = config.pop("normalizer", None)
        normalizer = normalizer_from_config(normalizer_cfg) if normalizer_cfg else None
        head_optimizer_cfg = config.pop("head_optimizer", None)
        head_optimizer = optimizer_from_config(head_optimizer_cfg) if head_optimizer_cfg else None
        trace_mode = TraceMode(config.pop("trace_mode", TraceMode.ACCUMULATING.value))

        return cls(
            horde_spec=horde_spec,
            hidden_sizes=tuple(config.pop("hidden_sizes")),
            optimizer=optimizer,
            bounder=bounder,
            normalizer=normalizer,
            head_optimizer=head_optimizer,
            trace_mode=trace_mode,
            **config,
        )


class NonlinearSharedGTDHordeLearner:
    """Corrected nonlinear shared-trunk off-policy Horde.

    This learner implements a compact TDC/GTD-style correction for a
    single-hidden-layer shared trunk with one head per demon. It is intentionally
    separate from :class:`OffPolicyHordeLearner`, whose state is a
    ``MultiHeadMLPState`` without secondary weights.
    """

    def __init__(
        self,
        horde_spec: HordeSpec,
        hidden_size: int = 16,
        primary_step_size: float = 0.002,
        secondary_step_size: float = 1e-5,
        ratio_clip: float = 10.0,
        init_scale: float = 0.25,
    ) -> None:
        if type(horde_spec) is not HordeSpec:
            raise ValueError("horde_spec must be an exact HordeSpec")
        if not horde_spec.demons:
            raise ValueError("horde_spec must contain at least one demon")
        if type(horde_spec.demons) is not tuple:
            raise ValueError("horde_spec.demons must be an exact tuple")
        n_demons = len(horde_spec.demons)
        for name, value in (
            ("horde_spec.gammas", horde_spec.gammas),
            ("horde_spec.lamdas", horde_spec.lamdas),
        ):
            if not isinstance(value, jax.Array):
                raise TypeError(f"{name} must be a JAX array")
            if tuple(value.shape) != (n_demons,):
                raise ValueError(f"{name} must have shape ({n_demons},)")
            if jnp.dtype(value.dtype) != jnp.dtype(jnp.float32):
                raise TypeError(f"{name} must have dtype float32")
            host = np.asarray(value)
            if not bool(np.all(np.isfinite(host))) or not bool(
                np.all((host >= 0.0) & (host <= 1.0))
            ):
                raise ValueError(f"{name} must contain values in [0, 1]")
        hidden_size = _require_int32("hidden_size", hidden_size, minimum=1)
        self._horde_spec = horde_spec
        self._hidden_size = hidden_size
        self._primary_step_size = _require_float32(
            "primary_step_size", primary_step_size, positive=True
        )
        self._secondary_step_size = _require_float32(
            "secondary_step_size", secondary_step_size, positive=True
        )
        self._ratio_clip = _require_positive_clip("ratio_clip", ratio_clip)
        self._init_scale = _require_float32("init_scale", init_scale, positive=True)
        _preflight_nonlinear_state(
            n_demons=len(horde_spec.demons), hidden_size=hidden_size, feature_dim=1
        )

    @property
    def horde_spec(self) -> HordeSpec:
        """The GVF specification."""
        return self._horde_spec

    @property
    def n_demons(self) -> int:
        """Number of demons."""
        return len(self._horde_spec.demons)

    def to_config(self) -> dict[str, Any]:
        """Serialize the complete nonlinear Horde construction."""
        return {
            "type": self.__class__.__name__,
            "horde_spec": self._horde_spec.to_config(),
            "hidden_size": self._hidden_size,
            "primary_step_size": self._primary_step_size,
            "secondary_step_size": self._secondary_step_size,
            "ratio_clip": self._ratio_clip,
            "init_scale": self._init_scale,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> NonlinearSharedGTDHordeLearner:
        """Reconstruct the exact serialized nonlinear Horde schema."""
        if not issubclass(type(config), Mapping):
            raise ValueError("config must be an actual mapping")
        try:
            payload = dict(config)
        except Exception as error:
            raise ValueError("config must be a readable mapping") from error
        expected = {
            "type",
            "horde_spec",
            "hidden_size",
            "primary_step_size",
            "secondary_step_size",
            "ratio_clip",
            "init_scale",
        }
        if any(type(key) is not str for key in payload) or set(payload) != expected:
            raise ValueError("config fields do not match the serialized schema")
        marker = payload.pop("type")
        if type(marker) is not str or marker != cls.__name__:
            raise ValueError("config type differs")
        raw_horde = payload.pop("horde_spec")
        if type(raw_horde) is not dict or set(raw_horde) != {"demons"}:
            raise ValueError("serialized horde_spec does not match its schema")
        if type(raw_horde["demons"]) is not list:
            raise ValueError("serialized horde_spec demons must be an exact list")
        demon_fields = {
            "name",
            "demon_type",
            "gamma",
            "lamda",
            "cumulant_index",
            "terminal_reward",
        }
        for demon in raw_horde["demons"]:
            if (
                type(demon) is not dict
                or any(type(key) is not str for key in demon)
                or set(demon) != demon_fields
            ):
                raise ValueError("serialized demon does not match the exact schema")
            if (
                type(demon["name"]) is not str
                or type(demon["demon_type"]) is not str
                or type(demon["cumulant_index"]) is not int
                or any(
                    type(demon[name]) is not float for name in ("gamma", "lamda", "terminal_reward")
                )
            ):
                raise ValueError("serialized demon scalar types must be exact JSON values")
        if type(payload["hidden_size"]) is not int or any(
            type(payload[name]) is not float
            for name in (
                "primary_step_size",
                "secondary_step_size",
                "ratio_clip",
                "init_scale",
            )
        ):
            raise ValueError("serialized scalar fields must be exact JSON numbers")
        return cls(horde_spec=HordeSpec.from_config(raw_horde), **payload)

    def init(self, feature_dim: int, key: Array) -> NonlinearSharedGTDHordeState:
        """Initialize primary and secondary weights."""
        feature_dim = _require_int32("feature_dim", feature_dim, minimum=1)
        key = _require_typed_threefry_key("key", key)
        _preflight_nonlinear_state(
            n_demons=self.n_demons,
            hidden_size=self._hidden_size,
            feature_dim=feature_dim,
        )
        trunk_key, head_key = jax.random.split(key)
        trunk_w = self._init_scale * jax.random.normal(
            trunk_key,
            (self._hidden_size, feature_dim),
            dtype=jnp.float32,
        )
        head_w = self._init_scale * jax.random.normal(
            head_key,
            (self.n_demons, self._hidden_size),
            dtype=jnp.float32,
        )
        state = NonlinearSharedGTDHordeState(  # type: ignore[call-arg]
            trunk_w=trunk_w,
            trunk_b=jnp.zeros(self._hidden_size, dtype=jnp.float32),
            head_w=head_w,
            head_b=jnp.zeros(self.n_demons, dtype=jnp.float32),
            secondary_trunk_w=jnp.zeros(
                (self.n_demons, self._hidden_size, feature_dim),
                dtype=jnp.float32,
            ),
            secondary_trunk_b=jnp.zeros(
                (self.n_demons, self._hidden_size),
                dtype=jnp.float32,
            ),
            secondary_head_w=jnp.zeros(
                (self.n_demons, self._hidden_size),
                dtype=jnp.float32,
            ),
            secondary_head_b=jnp.zeros(self.n_demons, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
            birth_timestamp=jnp.asarray(time.time(), dtype=jnp.float32),
            uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
        )
        if not bool(_floating_tree_is_finite(state)):
            raise ValueError("initialized nonlinear Horde state must be finite")
        return state

    def _validate_state_static_contract(self, state: NonlinearSharedGTDHordeState) -> int:
        """Return feature_dim after rejecting malformed adopted state metadata."""
        if type(state) is not NonlinearSharedGTDHordeState:
            raise TypeError("state must be a NonlinearSharedGTDHordeState")
        if not isinstance(state.trunk_w, jax.Array):
            raise TypeError("state.trunk_w must be a JAX array")
        shape = tuple(state.trunk_w.shape)
        if len(shape) != 2 or shape[0] != self._hidden_size or shape[1] < 1:
            raise ValueError("state.trunk_w has an invalid shape")
        feature_dim = int(shape[1])
        d = self.n_demons
        h = self._hidden_size
        expected = (
            ("state.trunk_w", state.trunk_w, (h, feature_dim), jnp.float32),
            ("state.trunk_b", state.trunk_b, (h,), jnp.float32),
            ("state.head_w", state.head_w, (d, h), jnp.float32),
            ("state.head_b", state.head_b, (d,), jnp.float32),
            (
                "state.secondary_trunk_w",
                state.secondary_trunk_w,
                (d, h, feature_dim),
                jnp.float32,
            ),
            ("state.secondary_trunk_b", state.secondary_trunk_b, (d, h), jnp.float32),
            ("state.secondary_head_w", state.secondary_head_w, (d, h), jnp.float32),
            ("state.secondary_head_b", state.secondary_head_b, (d,), jnp.float32),
            ("state.step_count", state.step_count, (), jnp.int32),
            ("state.birth_timestamp", state.birth_timestamp, (), jnp.float32),
            ("state.uptime_s", state.uptime_s, (), jnp.float32),
        )
        for name, value, expected_shape, dtype in expected:
            if not isinstance(value, jax.Array):
                raise TypeError(f"{name} must be a JAX array")
            if tuple(value.shape) != expected_shape:
                raise ValueError(f"{name} has an invalid shape")
            if jnp.dtype(value.dtype) != jnp.dtype(dtype):
                raise TypeError(f"{name} has an invalid dtype")
        return feature_dim

    @staticmethod
    def _state_is_valid(state: NonlinearSharedGTDHordeState) -> Bool[Array, ""]:
        return _floating_tree_is_finite(state) & (state.step_count >= 0)

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(self, state: NonlinearSharedGTDHordeState, observation: Array) -> Array:
        """Predict all demon values for one observation."""
        feature_dim = self._validate_state_static_contract(state)
        if not isinstance(observation, jax.Array):
            raise TypeError("observation must be a JAX array")
        if tuple(observation.shape) != (feature_dim,):
            raise ValueError(f"observation must have shape ({feature_dim},)")
        if jnp.dtype(observation.dtype) != jnp.dtype(jnp.float32):
            raise TypeError("observation must have dtype float32")
        hidden = jnp.tanh(state.trunk_w @ observation + state.trunk_b)
        return state.head_w @ hidden + state.head_b

    @functools.partial(jax.jit, static_argnums=(0,))
    def update_with_ratios_and_discounts(
        self,
        state: NonlinearSharedGTDHordeState,
        observation: Array,
        cumulants: Array,
        next_observation: Array,
        rhos: Array,
        discounts: Array,
    ) -> NonlinearSharedGTDHordeUpdateResult:
        """Update with explicit per-demon ratios and discounts."""
        feature_dim = self._validate_state_static_contract(state)
        for name, value, shape in (
            ("observation", observation, (feature_dim,)),
            ("cumulants", cumulants, (self.n_demons,)),
            ("next_observation", next_observation, (feature_dim,)),
            ("rhos", rhos, (self.n_demons,)),
            ("discounts", discounts, (self.n_demons,)),
        ):
            if not isinstance(value, jax.Array):
                raise TypeError(f"{name} must be a JAX array")
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if jnp.dtype(value.dtype) != jnp.dtype(jnp.float32):
                raise TypeError(f"{name} must have dtype float32")
        hidden = jnp.tanh(state.trunk_w @ observation + state.trunk_b)
        next_observation_valid = jnp.all(jnp.isfinite(next_observation))
        safe_next_observation = jnp.where(
            next_observation_valid,
            next_observation,
            jnp.zeros_like(next_observation),
        )
        next_hidden = jnp.tanh(state.trunk_w @ safe_next_observation + state.trunk_b)
        predictions = state.head_w @ hidden + state.head_b
        next_predictions = state.head_w @ next_hidden + state.head_b
        zero_discount_mask = discounts == 0.0
        bootstrap_predictions = jnp.where(
            zero_discount_mask,
            jnp.zeros_like(next_predictions),
            next_predictions,
        )
        td_targets = cumulants + discounts * bootstrap_predictions
        td_errors = td_targets - predictions
        requested = ~jnp.isnan(cumulants)
        active_mask = (
            requested
            & jnp.isfinite(cumulants)
            & jnp.isfinite(rhos)
            & (rhos >= 0.0)
            & jnp.isfinite(discounts)
            & (discounts >= 0.0)
            & (discounts <= 1.0)
            & ((discounts == 0.0) | next_observation_valid)
            & jnp.isfinite(td_targets)
        )
        inputs_valid = jnp.all(jnp.isfinite(observation))
        source_state_finite = self._state_is_valid(state)
        effective_mask = active_mask & inputs_valid & source_state_finite
        safe_td_errors = jnp.where(active_mask, td_errors, 0.0)
        clipped_rhos = jnp.minimum(
            jnp.maximum(jnp.asarray(rhos, dtype=jnp.float32), 0.0),
            jnp.asarray(self._ratio_clip, dtype=jnp.float32),
        )

        primary_alpha = jnp.asarray(self._primary_step_size, dtype=jnp.float32)
        secondary_beta = jnp.asarray(self._secondary_step_size, dtype=jnp.float32)
        trunk_w_step = jnp.zeros_like(state.trunk_w)
        trunk_b_step = jnp.zeros_like(state.trunk_b)
        head_w_step = jnp.zeros_like(state.head_w)
        head_b_step = jnp.zeros_like(state.head_b)
        new_secondary_trunk_w = []
        new_secondary_trunk_b = []
        new_secondary_head_w = []
        new_secondary_head_b = []
        correction_norms = []
        secondary_norms = []

        for i in range(self.n_demons):
            one_minus_hidden_sq = 1.0 - hidden**2
            next_one_minus_hidden_sq = 1.0 - next_hidden**2
            grad_head_w = hidden
            grad_head_b = jnp.array(1.0, dtype=jnp.float32)
            grad_hidden = state.head_w[i] * one_minus_hidden_sq
            grad_trunk_w = grad_hidden[:, None] * observation[None, :]
            grad_trunk_b = grad_hidden

            next_grad_head_w = next_hidden
            next_grad_head_b = jnp.array(1.0, dtype=jnp.float32)
            next_grad_hidden = state.head_w[i] * next_one_minus_hidden_sq
            next_grad_trunk_w = next_grad_hidden[:, None] * safe_next_observation[None, :]
            next_grad_trunk_b = next_grad_hidden

            secondary_dot = (
                jnp.vdot(state.secondary_trunk_w[i], grad_trunk_w)
                + jnp.vdot(state.secondary_trunk_b[i], grad_trunk_b)
                + jnp.vdot(state.secondary_head_w[i], grad_head_w)
                + state.secondary_head_b[i] * grad_head_b
            )
            # The correction is rho-weighted like the main term (TDC with
            # importance sampling, Sutton & Barto 2nd ed., Section 11.7;
            # GQ(0) with e = rho grad, Maei & Sutton 2010).  Inactive demons
            # (NaN cumulant) contribute nothing this step.
            masked_rho = jnp.where(effective_mask[i], clipped_rhos[i], 0.0)
            terminated_i = discounts[i] == 0.0
            rho_dot = jnp.where(
                terminated_i,
                jnp.zeros_like(secondary_dot),
                masked_rho * discounts[i] * secondary_dot,
            )
            correction_trunk_w = jnp.where(
                terminated_i,
                jnp.zeros_like(next_grad_trunk_w),
                rho_dot * next_grad_trunk_w,
            )
            correction_trunk_b = jnp.where(
                terminated_i,
                jnp.zeros_like(next_grad_trunk_b),
                rho_dot * next_grad_trunk_b,
            )
            correction_head_w = jnp.where(
                terminated_i,
                jnp.zeros_like(next_grad_head_w),
                rho_dot * next_grad_head_w,
            )
            correction_head_b = jnp.where(
                terminated_i,
                jnp.zeros_like(next_grad_head_b),
                rho_dot * next_grad_head_b,
            )
            rho_delta = masked_rho * safe_td_errors[i]

            trunk_w_step = trunk_w_step + primary_alpha * (
                rho_delta * grad_trunk_w - correction_trunk_w
            )
            trunk_b_step = trunk_b_step + primary_alpha * (
                rho_delta * grad_trunk_b - correction_trunk_b
            )
            head_w_step = head_w_step.at[i].add(
                primary_alpha * (rho_delta * grad_head_w - correction_head_w)
            )
            head_b_step = head_b_step.at[i].add(
                primary_alpha * (rho_delta * grad_head_b - correction_head_b)
            )

            masked_beta = jnp.where(effective_mask[i], secondary_beta, 0.0)
            sec_trunk_w = state.secondary_trunk_w[i] + masked_beta * (
                rho_delta * grad_trunk_w - secondary_dot * grad_trunk_w
            )
            sec_trunk_b = state.secondary_trunk_b[i] + masked_beta * (
                rho_delta * grad_trunk_b - secondary_dot * grad_trunk_b
            )
            sec_head_w = state.secondary_head_w[i] + masked_beta * (
                rho_delta * grad_head_w - secondary_dot * grad_head_w
            )
            sec_head_b = state.secondary_head_b[i] + masked_beta * (
                rho_delta * grad_head_b - secondary_dot * grad_head_b
            )
            new_secondary_trunk_w.append(sec_trunk_w)
            new_secondary_trunk_b.append(sec_trunk_b)
            new_secondary_head_w.append(sec_head_w)
            new_secondary_head_b.append(sec_head_b)
            correction_norms.append(
                _stable_l2_norm(
                    correction_trunk_w,
                    correction_trunk_b,
                    correction_head_w,
                    correction_head_b,
                )
            )
            secondary_norms.append(
                _stable_l2_norm(sec_trunk_w, sec_trunk_b, sec_head_w, sec_head_b)
            )

        proposed_state = state.replace(  # type: ignore[attr-defined]
            trunk_w=state.trunk_w + trunk_w_step,
            trunk_b=state.trunk_b + trunk_b_step,
            head_w=state.head_w + head_w_step,
            head_b=state.head_b + head_b_step,
            secondary_trunk_w=jnp.stack(new_secondary_trunk_w),
            secondary_trunk_b=jnp.stack(new_secondary_trunk_b),
            secondary_head_w=jnp.stack(new_secondary_head_w),
            secondary_head_b=jnp.stack(new_secondary_head_b),
            step_count=_saturating_int32_counter_increment(state.step_count),
        )
        candidate_state_finite = _floating_tree_is_finite(proposed_state)
        correction_norms_array = jnp.stack(correction_norms)
        secondary_norms_array = jnp.stack(secondary_norms)
        reported_next_predictions = jnp.where(
            zero_discount_mask, jnp.zeros_like(next_predictions), next_predictions
        )
        active_outputs_finite = jnp.all(
            (~active_mask)
            | (
                jnp.isfinite(predictions)
                & jnp.isfinite(reported_next_predictions)
                & jnp.isfinite(td_targets)
                & jnp.isfinite(td_errors)
                & jnp.isfinite(clipped_rhos)
                & jnp.isfinite(correction_norms_array)
                & jnp.isfinite(secondary_norms_array)
            )
        )
        update_applied = (
            inputs_valid
            & source_state_finite
            & candidate_state_finite
            & active_outputs_finite
            & (jnp.any(active_mask) | jnp.all(~requested))
        )
        head_updates_applied = effective_mask & update_applied
        new_state = jax.lax.cond(
            update_applied,
            lambda: proposed_state,
            lambda: state,
        )
        return NonlinearSharedGTDHordeUpdateResult(  # type: ignore[call-arg]
            state=new_state,
            predictions=jnp.where(update_applied, predictions, jnp.zeros_like(predictions)),
            next_predictions=jnp.where(
                update_applied,
                reported_next_predictions,
                jnp.zeros_like(reported_next_predictions),
            ),
            td_targets=jnp.where(update_applied, td_targets, jnp.zeros_like(td_targets)),
            td_errors=jnp.where(update_applied, td_errors, jnp.zeros_like(td_errors)),
            clipped_rhos=jnp.where(update_applied, clipped_rhos, jnp.zeros_like(clipped_rhos)),
            correction_norms=jnp.where(
                update_applied,
                correction_norms_array,
                jnp.zeros_like(correction_norms_array),
            ),
            secondary_norms=jnp.where(
                update_applied,
                secondary_norms_array,
                jnp.zeros_like(secondary_norms_array),
            ),
            head_updates_applied=head_updates_applied,
            update_applied=update_applied,
        )


def run_off_policy_horde_learning_loop(
    learner: OffPolicyHordeLearner,
    state: MultiHeadMLPState,
    observations: Array,
    cumulants: Array,
    next_observations: Array,
    rhos: Array,
    discounts: Array | None = None,
) -> OffPolicyHordeLearningResult:
    """Run an off-policy Horde scan over transition arrays."""
    if discounts is None:
        discounts = jnp.broadcast_to(learner.horde_spec.gammas, cumulants.shape)

    def step_fn(
        carry: MultiHeadMLPState,
        inputs: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[MultiHeadMLPState, tuple[Array, Array, Array, Array, Array]]:
        obs, cums, next_obs, rho_t, discount_t = inputs
        result = learner.update_with_ratios_and_discounts(
            carry,
            obs,
            cums,
            next_obs,
            rho_t,
            discount_t,
        )
        return (
            result.state,
            (
                result.per_demon_metrics,
                result.td_errors,
                result.clipped_rhos,
                result.head_updates_applied,
                result.update_applied,
            ),
        )

    t0 = time.time()
    (
        final_state,
        (
            per_demon_metrics,
            td_errors,
            clipped_rhos,
            head_updates_applied,
            updates_applied,
        ),
    ) = jax.lax.scan(
        step_fn,
        state,
        (observations, cumulants, next_observations, rhos, discounts),
    )
    elapsed = time.time() - t0
    final_state = final_state.replace(  # type: ignore[attr-defined]
        uptime_s=final_state.uptime_s + elapsed
    )

    return OffPolicyHordeLearningResult(
        state=final_state,
        per_demon_metrics=per_demon_metrics,
        td_errors=td_errors,
        clipped_rhos=clipped_rhos,
        head_updates_applied=head_updates_applied,
        updates_applied=updates_applied,
    )  # type: ignore[call-arg]


def run_off_policy_horde_learning_loop_batched(
    learner: OffPolicyHordeLearner,
    observations: Array,
    cumulants: Array,
    next_observations: Array,
    rhos: Array,
    keys: Array,
    discounts: Array | None = None,
) -> OffPolicyHordeLearningResult:
    """Run the off-policy Horde loop for multiple initialization keys."""

    def single_run(
        key: Array,
    ) -> tuple[MultiHeadMLPState, Array, Array, Array, Array, Array]:
        init_state = learner.init(observations.shape[1], key)
        result = run_off_policy_horde_learning_loop(
            learner,
            init_state,
            observations,
            cumulants,
            next_observations,
            rhos,
            discounts,
        )
        return (
            result.state,
            result.per_demon_metrics,
            result.td_errors,
            result.clipped_rhos,
            result.head_updates_applied,
            result.updates_applied,
        )

    (
        states,
        per_demon_metrics,
        td_errors,
        clipped_rhos,
        head_updates_applied,
        updates_applied,
    ) = jax.vmap(single_run)(keys)
    return OffPolicyHordeLearningResult(
        state=states,
        per_demon_metrics=per_demon_metrics,
        td_errors=td_errors,
        clipped_rhos=clipped_rhos,
        head_updates_applied=head_updates_applied,
        updates_applied=updates_applied,
    )  # type: ignore[call-arg]


__all__ = [
    "NonlinearSharedGTDHordeLearner",
    "NonlinearSharedGTDHordeLearningResult",
    "NonlinearSharedGTDHordeState",
    "NonlinearSharedGTDHordeUpdateResult",
    "OffPolicyHordeLearner",
    "OffPolicyHordeLearningResult",
    "OffPolicyHordeUpdateResult",
    "run_off_policy_horde_learning_loop",
    "run_off_policy_horde_learning_loop_batched",
]
