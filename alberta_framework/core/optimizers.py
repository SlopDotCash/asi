"""Optimizers for continual learning.

Implements LMS (fixed step-size baseline), IDBD (meta-learned step-sizes),
Autostep (tuning-free step-size adaptation), and ObGD (observation-bounded)
for the Alberta Plan.

Also provides the ``Bounder`` ABC for decoupled update bounding (e.g. ObGDBounding).

References:
- Sutton 1992, "Adapting Bias by Gradient Descent: An Incremental
  Version of Delta-Bar-Delta"
- Mahmood et al. 2012, "Tuning-free step-size adaptation"
- Elsayed et al. 2024, "Streaming Deep Reinforcement Learning Finally Works"
"""

import operator
from abc import ABC, abstractmethod
from math import prod
from typing import Any, SupportsIndex, cast

import chex
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float

from alberta_framework.core._float32_scalars import (
    validated_float32_scalar,
    validated_float32_scalar_with_ratio,
)
from alberta_framework.core.types import (
    AutostepGTDLambdaState,
    AutostepParamState,
    AutostepState,
    AutoTDIDBDState,
    IDBDParamState,
    IDBDState,
    LMSState,
    ObGDState,
    TDIDBDState,
)
from alberta_framework.core.update_safety import (
    floating_tree_is_finite,
    neutralize_array,
    neutralize_metrics,
    select_transaction,
    zero_if_collapsed_infinity,
)


def _require_exact_str(name: object, value: object) -> str:
    if type(name) is not str:
        raise ValueError("name must be an exact string")
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    return value

def _skip_zero_scale(scale: Array, value: Array) -> Array:
    """Return 0 when ``scale`` is 0 so IEEE ``0 * inf`` does not become NaN."""
    return jnp.where(scale == 0.0, jnp.zeros_like(value), scale * value)


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


def _require_int32(name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX) -> int:
    actual_type = type(value)
    if not any(actual_type is allowed_type for allowed_type in _ACTUAL_INT_TYPES):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


def _require_shape(shape: object) -> tuple[int, ...]:
    if type(shape) is not tuple:
        raise ValueError("shape must be an actual tuple")
    return tuple(_require_int32(f"shape[{i}]", s, minimum=1) for i, s in enumerate(shape))


def _require_float32_state(name: str, scalar_count: int) -> None:
    if scalar_count > _INT32_MAX:
        raise ValueError(f"{name} scalar count must fit signed int32")
    if 4 * scalar_count > _INT32_MAX:
        raise ValueError(f"{name} byte count must fit signed int32")


def _require_float32_update_working_set(name: str, scalar_count: int) -> None:
    """Reject simultaneous update tensors this optimizer cannot name."""

    _require_float32_state(name, scalar_count)


def _shape_size(shape: tuple[int, ...]) -> int:
    return prod(shape, start=1)


def _validated_idbd_step_size(
    name: str,
    value: object,
    *,
    positive: bool = False,
) -> float:
    """Validate an IDBD rate without silently erasing a positive host value."""
    stored, numerator, _ = validated_float32_scalar_with_ratio(
        name,
        value,
        positive=positive,
        lower=None if positive else 0.0,
    )
    narrowed = float(np.float32(stored))
    if numerator != 0 and narrowed == 0.0:
        raise ValueError(f"{name} must remain nonzero once narrowed to float32")
    return stored


# =============================================================================
# Bounder ABC
# =============================================================================


class Bounder(ABC):
    """Base class for update bounding strategies.

    A bounder takes the proposed per-parameter step arrays from an optimizer
    and optionally scales them down to prevent overshooting.
    """

    @abstractmethod
    def to_config(self) -> dict[str, Any]:
        """Serialize bounding configuration to dict."""
        ...

    @abstractmethod
    def bound(
        self,
        steps: tuple[Array, ...],
        error: Array,
        params: tuple[Array, ...],
    ) -> tuple[tuple[Array, ...], Array]:
        """Bound proposed update steps.

        Args:
            steps: Per-parameter step arrays from the optimizer
            error: Prediction error scalar
            params: Current parameter values (needed by some bounders like AGC)

        Returns:
            ``(bounded_steps, metric)`` where metric is a scalar for reporting
            (e.g., scale factor for ObGD, mean clip ratio for AGC)
        """
        ...


def _apply_obgd_bound(
    steps: tuple[Array, ...],
    error: Array,
    kappa: float,
) -> tuple[tuple[Array, ...], Array]:
    """Apply the ObGD global bounding formula. Returns (bounded_steps, scale)."""
    error_scalar = jnp.squeeze(error)
    total_step = jnp.array(0.0)
    for s in steps:
        total_step = total_step + jnp.sum(jnp.abs(s))
    delta_bar = jnp.maximum(jnp.abs(error_scalar), 1.0)
    scale = 1.0 / jnp.maximum(kappa * delta_bar * total_step, 1.0)
    collapsed = scale == 0
    return (
        tuple(zero_if_collapsed_infinity(scale * step, step, collapsed) for step in steps),
        scale,
    )


class ObGDBounding(Bounder):
    """ObGD-style global update bounding (Elsayed et al. 2024).

    Computes a global bounding factor from the L1 norm of all proposed
    steps and the error magnitude, then uniformly scales all steps down
    if the combined update would be too large.

    For LMS with a single scalar step-size ``alpha``:
    ``total_step = alpha * z_sum``, giving
    ``M = alpha * kappa * max(|error|, 1) * z_sum`` -- identical to
    the original Elsayed et al. 2024 formula.

    Attributes:
        kappa: Bounding sensitivity parameter (higher = more conservative)
    """

    def __init__(self, kappa: float = 2.0):
        self._kappa = kappa

    def to_config(self) -> dict[str, Any]:
        """Serialize configuration to dict."""
        return {"type": "ObGDBounding", "kappa": self._kappa}

    def bound(
        self,
        steps: tuple[Array, ...],
        error: Array,
        params: tuple[Array, ...],
    ) -> tuple[tuple[Array, ...], Array]:
        """Bound proposed steps using ObGD formula.

        Args:
            steps: Per-parameter step arrays
            error: Prediction error scalar
            params: Current parameter values (unused by ObGD)

        Returns:
            ``(bounded_steps, scale)`` where scale is the bounding factor
        """
        del params  # ObGD bounds based on step/error magnitude only
        return _apply_obgd_bound(steps, error, self._kappa)


def _unitwise_norm(x: Array) -> Array:
    """Compute unit-wise L2 norm.

    For 2D+ arrays (weight matrices laid out ``(fan_out, fan_in)`` throughout
    this framework, so each row is one output unit's incoming weights): L2
    norm over all axes except the first, with keepdims for broadcasting.
    For 1D arrays (biases): absolute value per element.
    For scalars: absolute value.
    """
    if x.ndim >= 2:
        return jnp.sqrt(jnp.sum(x**2, axis=tuple(range(1, x.ndim)), keepdims=True))
    return jnp.abs(x)


class AdaptiveObGDBounding(Bounder):
    """ObGD bounding with RMS per-parameter normalization.

    Extends :class:`ObGDBounding` with a second adaptive stage: after the
    global ObGD scale is applied, each per-parameter step is divided by the
    root-mean-square of all bounded steps (floored at 1).  This keeps the
    per-parameter relative magnitudes in check: parameters whose bounded
    updates are large compared to others are scaled down further without
    requiring any cross-step running average.

    The bounding factor returned is the ObGD global scale; the RMS stage is
    implicit in the returned steps.

    Attributes:
        kappa: ObGD sensitivity (higher = more conservative). Default 2.0.
        eps: Floor for the RMS denominator to avoid division by zero.

    Reference:
        Elsayed, M., Lan, Q., Lyle, C., & Mahmood, A.R. (2024).
        "Streaming Deep Reinforcement Learning Finally Works." Appendix B.
    """

    def __init__(self, kappa: float = 2.0, eps: float = 1e-8):
        self._kappa = kappa
        self._eps = eps

    def to_config(self) -> dict[str, Any]:
        """Serialize configuration to dict."""
        return {"type": "AdaptiveObGDBounding", "kappa": self._kappa, "eps": self._eps}

    def bound(
        self,
        steps: tuple[Array, ...],
        error: Array,
        params: tuple[Array, ...],
    ) -> tuple[tuple[Array, ...], Array]:
        """Apply ObGD global bound then per-weight RMS normalisation.

        Args:
            steps: Per-parameter step arrays
            error: Prediction error scalar
            params: Current parameter values (unused)

        Returns:
            ``(bounded_steps, obgd_scale)`` where ``obgd_scale`` is the global
            ObGD bounding factor before the RMS stage
        """
        del params
        bounded, scale = _apply_obgd_bound(steps, error, self._kappa)

        # Per-weight RMS normalization across all bounded steps.
        sum_sq = jnp.array(0.0)
        n_weights = jnp.array(0)
        for s in bounded:
            sum_sq = sum_sq + jnp.sum(s**2)
            n_weights = n_weights + s.size
        rms = jnp.sqrt(sum_sq / jnp.maximum(n_weights.astype(jnp.float32), 1.0) + self._eps)
        rms_scale = jnp.maximum(rms, 1.0)
        adaptive = tuple(s / rms_scale for s in bounded)
        return adaptive, scale


class AGCBounding(Bounder):
    """Adaptive Gradient Clipping (Brock et al. 2021).

    Clips per-output-unit based on the ratio of gradient norm to weight norm.
    Units where ``||grad|| / max(||weight||, eps) > clip_factor`` get scaled
    down to respect the constraint.

    Unlike ObGDBounding which applies a single global scale factor, AGC
    applies fine-grained, per-unit clipping that adapts to each layer's
    weight magnitude.

    The metric returned is the fraction of units that were clipped (0.0 = no
    clipping, 1.0 = all units clipped).

    Reference: Brock, A., De, S., Smith, S.L., & Simonyan, K. (2021).
    "High-Performance Large-Scale Image Recognition Without Normalization"
    (arXiv: 2102.06171)

    Attributes:
        clip_factor: Maximum allowed gradient-to-weight ratio (lambda). Default 0.01.
        eps: Floor for weight norm to avoid division by zero. Default 1e-3.
    """

    def __init__(self, clip_factor: float = 0.01, eps: float = 1e-3):
        self._clip_factor = clip_factor
        self._eps = eps

    def to_config(self) -> dict[str, Any]:
        """Serialize configuration to dict."""
        return {"type": "AGCBounding", "clip_factor": self._clip_factor, "eps": self._eps}

    def bound(
        self,
        steps: tuple[Array, ...],
        error: Array,
        params: tuple[Array, ...],
    ) -> tuple[tuple[Array, ...], Array]:
        """Bound proposed steps using per-unit adaptive gradient clipping.

        For each parameter/step pair, computes unit-wise norms and clips
        units where ``|error| * ||step|| > clip_factor * max(||param||, eps)``.

        Args:
            steps: Per-parameter step arrays from the optimizer
            error: Prediction error scalar
            params: Current parameter values (used for weight norms)

        Returns:
            ``(clipped_steps, frac_clipped)`` where frac_clipped is the
            fraction of units that were clipped
        """
        error_abs = jnp.abs(jnp.squeeze(error))
        clipped = []
        total_units = 0
        clipped_units = jnp.array(0.0)

        for step, param in zip(steps, params):
            p_norm = _unitwise_norm(param)
            s_norm = _unitwise_norm(step)
            g_norm = error_abs * s_norm
            max_norm = jnp.maximum(p_norm, self._eps) * self._clip_factor
            scale = max_norm / jnp.maximum(g_norm, 1e-6)
            needs_clip = g_norm > max_norm
            clipped_step = jnp.where(needs_clip, step * scale, step)
            collapsed = needs_clip & (scale == 0)
            clipped.append(zero_if_collapsed_infinity(clipped_step, step, collapsed))

            total_units += needs_clip.size
            clipped_units = clipped_units + jnp.sum(needs_clip.astype(jnp.float32))

        frac_clipped = clipped_units / jnp.maximum(total_units, 1)
        return tuple(clipped), frac_clipped


# =============================================================================
# Supervised Learning Optimizers
# =============================================================================


@chex.dataclass(frozen=True)
class OptimizerUpdate:
    """Result of an optimizer update step.

    Attributes:
        weight_delta: Change to apply to weights
        bias_delta: Change to apply to bias
        new_state: Updated optimizer state
        metrics: Dictionary of metrics for logging (values are JAX arrays for scan compatibility)
    """

    weight_delta: Float[Array, " feature_dim"]
    bias_delta: Float[Array, ""]
    new_state: (
        LMSState | IDBDState | AutostepState | AutostepGTDLambdaState | ObGDState | IDBDParamState
    )
    metrics: dict[str, Array]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ParamOptimizerUpdate:
    """Checked result for an arbitrary-shape parameter update.

    ``update_from_gradient`` retains its historical two-tuple API.  Callers
    that own a larger state transaction use ``update_from_gradient_checked``
    so a rejected parameter update cannot be mistaken for a legitimate zero
    step.
    """

    step: Array
    new_state: Any
    update_applied: Bool[Array, ""]


class Optimizer[
    StateT: (
        LMSState,
        IDBDState,
        AutostepState,
        AutostepGTDLambdaState,
        ObGDState,
        AutostepParamState,
        IDBDParamState,
    )
](ABC):
    """Base class for optimizers."""

    @abstractmethod
    def to_config(self) -> dict[str, Any]:
        """Serialize optimizer configuration to dict."""
        ...

    @abstractmethod
    def init(self, feature_dim: int) -> StateT:
        """Initialize optimizer state.

        Args:
            feature_dim: Dimension of weight vector

        Returns:
            Initial optimizer state
        """
        ...

    @abstractmethod
    def update(
        self,
        state: StateT,
        error: Array,
        observation: Array,
    ) -> OptimizerUpdate:
        """Compute weight updates given prediction error.

        Args:
            state: Current optimizer state
            error: Prediction error (target - prediction)
            observation: Current observation/feature vector

        Returns:
            OptimizerUpdate with deltas and new state
        """
        ...

    def supported_for_mlp(self) -> bool:
        """Whether this optimizer supports the per-parameter MLP path.

        MLP, multi-head, and head-optimizer construction sites should call
        this before accepting an optimizer, so an unsupported choice fails at
        construction time instead of raising ``NotImplementedError`` deep
        inside a jitted update.  The capability is detected structurally: an
        optimizer supports the MLP path exactly when it overrides both
        :meth:`init_for_shape` and :meth:`update_from_gradient`.

        Returns:
            True when both shape-generic hooks are overridden by the
            subclass (currently LMS, IDBD, and Autostep).
        """
        cls = type(self)
        return (
            cls.init_for_shape is not Optimizer.init_for_shape
            and cls.update_from_gradient is not Optimizer.update_from_gradient
        )

    def init_for_shape(self, shape: tuple[int, ...]) -> Any:
        """Initialize optimizer state for parameters of arbitrary shape.

        Used by MLP learners where parameters are matrices/vectors of
        varying shapes. Not all optimizers support this; check
        :meth:`supported_for_mlp` at construction time.

        The return type varies by subclass (e.g. ``LMSState`` for LMS,
        ``AutostepParamState`` for Autostep) so the base signature uses
        ``Any``.

        Args:
            shape: Shape of the parameter array

        Returns:
            Initial optimizer state with arrays matching the given shape

        Raises:
            NotImplementedError: If the optimizer does not support this
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support init_for_shape. "
            "Only LMS, IDBD, and Autostep currently implement this."
        )

    def update_from_gradient(
        self, state: Any, gradient: Array, error: Array | None = None
    ) -> tuple[Array, Any]:
        """Compute step delta from pre-computed gradient.

        The returned delta does NOT include the error -- the caller is
        responsible for multiplying ``error * delta`` before applying.

        The state type varies by subclass (e.g. ``LMSState`` for LMS,
        ``AutostepParamState`` for Autostep) so the base signature uses
        ``Any``.

        Args:
            state: Current optimizer state
            gradient: Pre-computed gradient (e.g. eligibility trace)
            error: Optional prediction error scalar. Optimizers with
                meta-learning (e.g. Autostep) use this for meta-gradient
                computation. LMS ignores it.

        Returns:
            ``(step, new_state)`` where step has the same shape as gradient

        Raises:
            NotImplementedError: If the optimizer does not support this
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support update_from_gradient. "
            "Only LMS, IDBD, and Autostep currently implement this."
        )

    def update_from_gradient_checked(
        self, state: Any, gradient: Array, error: Array | None = None
    ) -> ParamOptimizerUpdate:
        """Return a checked parameter update without changing the legacy API.

        This fallback makes externally defined optimizers transaction-aware.
        Built-in adaptive optimizers override it so finite candidate overflow
        can be reported explicitly after their internal guard runs.
        """

        step, new_state = self.update_from_gradient(state, gradient, error=error)
        error_is_finite = (
            jnp.asarray(True, dtype=jnp.bool_) if error is None else jnp.all(jnp.isfinite(error))
        )
        update_applied = (
            floating_tree_is_finite(state)
            & jnp.all(jnp.isfinite(gradient))
            & error_is_finite
            & jnp.all(jnp.isfinite(step))
            & floating_tree_is_finite(new_state)
        )
        return ParamOptimizerUpdate(
            step=neutralize_array(update_applied, step),
            new_state=select_transaction(update_applied, new_state, state),
            update_applied=update_applied,
        )


class LMS(Optimizer[LMSState]):
    """Least Mean Square optimizer with fixed step-size.

    The simplest gradient-based optimizer: ``w_{t+1} = w_t + alpha * delta * x_t``

    This serves as a baseline. The challenge is that the optimal step-size
    depends on the problem and changes as the task becomes non-stationary.

    Attributes:
        step_size: Fixed learning rate alpha
    """

    def __init__(self, step_size: float = 0.01):
        """Initialize LMS optimizer.

        Args:
            step_size: Fixed learning rate

        Raises:
            ValueError: If ``step_size`` is not a finite non-bool positive real.
        """
        self._step_size = validated_float32_scalar(
            "step_size", step_size, positive=True
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize configuration to dict."""
        return {"type": "LMS", "step_size": self._step_size}

    def init(self, feature_dim: int) -> LMSState:
        """Initialize LMS state.

        Args:
            feature_dim: Dimension of weight vector (unused for LMS)

        Returns:
            LMS state containing the step-size
        """
        _require_int32("feature_dim", feature_dim, minimum=1)
        return LMSState(step_size=jnp.array(self._step_size, dtype=jnp.float32))

    def init_for_shape(self, shape: tuple[int, ...]) -> LMSState:
        """Initialize LMS state for arbitrary-shape parameters.

        LMS state is shape-independent (single scalar), so this returns
        the same state regardless of shape.
        """
        _require_shape(shape)
        return LMSState(step_size=jnp.array(self._step_size, dtype=jnp.float32))

    def update_from_gradient(
        self, state: LMSState, gradient: Array, error: Array | None = None
    ) -> tuple[Array, LMSState]:
        """Compute step from gradient: ``step = alpha * gradient``.

        Args:
            state: Current LMS state
            gradient: Pre-computed gradient (any shape)
            error: Unused by LMS (accepted for interface compatibility)

        Returns:
            ``(step, state)`` -- state is unchanged for LMS
        """
        result = self.update_from_gradient_checked(state, gradient, error)
        return result.step, cast(LMSState, result.new_state)

    def update_from_gradient_checked(
        self, state: LMSState, gradient: Array, error: Array | None = None
    ) -> ParamOptimizerUpdate:
        """Compute an LMS parameter step with an explicit transaction signal."""

        step = state.step_size * gradient
        error_is_finite = (
            jnp.asarray(True, dtype=jnp.bool_) if error is None else jnp.all(jnp.isfinite(error))
        )
        update_applied = (
            floating_tree_is_finite(state)
            & jnp.all(jnp.isfinite(gradient))
            & error_is_finite
            & jnp.all(jnp.isfinite(step))
        )
        return ParamOptimizerUpdate(
            step=neutralize_array(update_applied, step),
            new_state=state,
            update_applied=update_applied,
        )

    def update(
        self,
        state: LMSState,
        error: Array,
        observation: Array,
    ) -> OptimizerUpdate:
        """Compute LMS weight update.

        Update rule: ``delta_w = alpha * error * x``

        Args:
            state: Current LMS state
            error: Prediction error (scalar)
            observation: Feature vector

        Returns:
            OptimizerUpdate with weight and bias deltas
        """
        alpha = state.step_size
        error_scalar = jnp.squeeze(error)

        # Weight update: alpha * error * x
        weight_delta = alpha * error_scalar * observation

        # Bias update: alpha * error
        bias_delta = alpha * error_scalar
        candidate_metrics = {"step_size": alpha}

        update_applied = (
            floating_tree_is_finite(state)
            & jnp.isfinite(error_scalar)
            & jnp.all(jnp.isfinite(observation))
            & jnp.all(jnp.isfinite(weight_delta))
            & jnp.isfinite(bias_delta)
            & floating_tree_is_finite(candidate_metrics)
        )
        metrics = neutralize_metrics(update_applied, candidate_metrics)

        return OptimizerUpdate(
            weight_delta=neutralize_array(update_applied, weight_delta),
            bias_delta=neutralize_array(update_applied, bias_delta),
            new_state=state,  # LMS state doesn't change
            metrics=metrics,
            update_applied=update_applied,
        )


class IDBD(Optimizer[IDBDState]):
    """Incremental Delta-Bar-Delta optimizer.

    IDBD maintains per-weight adaptive step-sizes that are meta-learned
    based on gradient correlation. When successive gradients agree in sign,
    the step-size for that weight increases. When they disagree, it decreases.

    This implements Sutton's 1992 algorithm for adapting step-sizes online
    without requiring manual tuning.

    Reference: Sutton, R.S. (1992). "Adapting Bias by Gradient Descent:
    An Incremental Version of Delta-Bar-Delta"

    Attributes:
        initial_step_size: Initial per-weight step-size
        meta_step_size: Meta learning rate beta for adapting step-sizes
    """

    def __init__(
        self,
        initial_step_size: float = 0.01,
        meta_step_size: float = 0.01,
        h_decay_mode: object = "prediction_grads",
    ):
        """Initialize IDBD optimizer.

        Args:
            initial_step_size: Initial value for per-weight step-sizes
            meta_step_size: Meta learning rate beta for adapting step-sizes
            h_decay_mode: Mode for computing the h-decay term in MLP path.
                ``"prediction_grads"``: h_decay = z^2 (squared prediction
                gradients). This is the principled generalization — for
                linear models, z = x so z^2 = x^2, recovering Sutton 1992.
                ``"loss_grads"``: h_decay = (error * z)^2 (Fisher
                approximation of the Hessian diagonal).
                Only affects the MLP path (``update_from_gradient``);
                the linear ``update()`` method always uses x^2.

        Raises:
            ValueError: If ``h_decay_mode`` is not one of the valid modes,
                or if a step-size is not a finite non-bool real in the
                declared domain (``initial_step_size`` positive;
                ``meta_step_size`` nonnegative).
        """
        host_mode = _require_exact_str("h_decay_mode", h_decay_mode)
        if host_mode not in ("prediction_grads", "loss_grads"):
            raise ValueError(
                f"Invalid h_decay_mode: '{host_mode}'. "
                "Must be 'prediction_grads' or 'loss_grads'."
            )
        self._initial_step_size = _validated_idbd_step_size(
            "initial_step_size", initial_step_size, positive=True
        )
        self._meta_step_size = _validated_idbd_step_size("meta_step_size", meta_step_size)
        self._h_decay_mode = host_mode

    def to_config(self) -> dict[str, Any]:
        """Serialize configuration to dict."""
        config: dict[str, Any] = {
            "type": "IDBD",
            "initial_step_size": self._initial_step_size,
            "meta_step_size": self._meta_step_size,
        }
        if self._h_decay_mode != "prediction_grads":
            config["h_decay_mode"] = self._h_decay_mode
        return config

    def init(self, feature_dim: int) -> IDBDState:
        """Initialize IDBD state.

        Args:
            feature_dim: Dimension of weight vector

        Returns:
            IDBD state with per-weight step-sizes and traces
        """
        feature_dim = _require_int32("feature_dim", feature_dim, minimum=1)
        _require_float32_state("IDBD state", 2 * feature_dim + 3)
        # Complete named logical envelope for both linear and parameter
        # paths, including checked source banks, selected result banks, and
        # the neutral returned step/delta.
        _require_float32_update_working_set(
            "IDBD update working set", 18 * feature_dim + 8
        )
        return IDBDState(
            log_step_sizes=jnp.full(
                feature_dim, jnp.log(self._initial_step_size), dtype=jnp.float32
            ),
            traces=jnp.zeros(feature_dim, dtype=jnp.float32),
            meta_step_size=jnp.array(self._meta_step_size, dtype=jnp.float32),
            bias_step_size=jnp.array(self._initial_step_size, dtype=jnp.float32),
            bias_trace=jnp.array(0.0, dtype=jnp.float32),
        )

    def init_for_shape(self, shape: tuple[int, ...]) -> IDBDParamState:
        """Initialize IDBD state for arbitrary-shape parameters.

        Args:
            shape: Shape of the parameter array

        Returns:
            IDBDParamState with arrays matching the given shape
        """
        shape = _require_shape(shape)
        n = _shape_size(shape)
        _require_float32_state("IDBD parameter state", 2 * n + 1)
        _require_float32_update_working_set("IDBD update working set", 18 * n + 8)
        return IDBDParamState(
            log_step_sizes=jnp.full(shape, jnp.log(self._initial_step_size), dtype=jnp.float32),
            traces=jnp.zeros(shape, dtype=jnp.float32),
            meta_step_size=jnp.array(self._meta_step_size, dtype=jnp.float32),
        )

    def update_from_gradient(
        self,
        state: IDBDParamState,
        gradient: Array,
        error: Array | None = None,
    ) -> tuple[Array, IDBDParamState]:
        """Return the historical unchecked pair for compatibility."""

        result = self.update_from_gradient_checked(state, gradient, error)
        return result.step, cast(IDBDParamState, result.new_state)

    def update_from_gradient_checked(
        self,
        state: IDBDParamState,
        gradient: Array,
        error: Array | None = None,
    ) -> ParamOptimizerUpdate:
        """Compute IDBD update from pre-computed gradient (MLP path).

        Implements Meyer's adaptation of IDBD to nonlinear models. The key
        insight: replace ``x^2`` in the h-decay term with ``(dy/dw)^2``
        (squared prediction gradients), which generalizes IDBD to arbitrary
        architectures.

        This follows Meyer's implementation, which differs from the linear
        IDBD (Sutton 1992) in two ways to better handle deep networks:

        1. The meta-update uses ``z * h`` (prediction gradient times trace)
           without the current error, rather than ``error * z * h``.
        2. The h-trace accumulates loss gradients (``-error * z``) rather
           than error-scaled prediction gradients (``error * z``).

        These changes address problems with IDBD in deep networks where
        the step-size being factored into both h and beta updates causes
        compounding effects.

        Reference: Meyer, https://github.com/ejmejm/phd_research

        Operation order (meta-update first, then new alpha for trace):

        1. Compute h_decay based on mode: ``z^2`` or ``(error * z)^2``
        2. Meta-update with OLD traces: ``log_alpha += beta * z * h``
        3. Clip log step-sizes to ``[-10.0, 2.0]``
        4. New step-sizes: ``alpha = exp(log_alpha)``
        5. Step: ``alpha * z`` (error applied externally by caller)
        6. Trace update: ``h = h * max(0, 1 - alpha * h_decay) + alpha * g``
           where ``g = -error * z`` (loss gradient direction)

        When ``error`` is None (trunk path in multi-head), the gradient
        is already in loss gradient direction (accumulated cotangents),
        so the trace uses ``alpha * z`` directly.

        Args:
            state: Current IDBD param state
            gradient: Pre-computed prediction gradient / eligibility trace
                (same shape as state arrays)
            error: Optional prediction error scalar. When provided,
                used for h_decay (loss_grads mode) and h-trace sign.

        Returns:
            ``(step, new_state)`` where step has the same shape as gradient
        """
        beta = state.meta_step_size
        z = gradient

        # 1. Compute h_decay based on mode
        if self._h_decay_mode == "loss_grads" and error is not None:
            h_decay = (jnp.squeeze(error) * z) ** 2
        else:
            # prediction_grads mode, or loss_grads without error
            h_decay = z**2

        # 2. Meta-update with OLD traces (Meyer: prediction_grads * h, no error).
        # Skip 0 * inf when meta-learning is disabled.
        meta_gradient = z * state.traces
        meta_delta = _skip_zero_scale(beta, meta_gradient)

        # 3. Clip log step-sizes; a non-finite meta-gradient (inf z against a
        # zero trace) skips adaptation instead of poisoning the step-size.
        new_log_step_sizes = jnp.where(
            jnp.isfinite(meta_delta),
            jnp.clip(state.log_step_sizes + meta_delta, -10.0, 2.0),
            state.log_step_sizes,
        )

        # 4. New step-sizes
        new_alphas = jnp.exp(new_log_step_sizes)

        # 5. Step: alpha * z (error applied externally)
        step = new_alphas * z

        # 6. Trace update: h = h * decay + alpha * loss_grads
        # Meyer uses loss_grads = -error * z when error is available;
        # when error is None (trunk path), z is already loss gradient direction.
        decay = jnp.maximum(0.0, 1.0 - new_alphas * h_decay)
        decayed_traces = _skip_zero_scale(decay, state.traces)
        if error is not None:
            new_traces = decayed_traces - new_alphas * jnp.squeeze(error) * z
        else:
            new_traces = decayed_traces + new_alphas * z

        new_state = IDBDParamState(
            log_step_sizes=new_log_step_sizes,
            traces=new_traces,
            meta_step_size=beta,
        )
        error_is_finite = (
            jnp.asarray(True, dtype=jnp.bool_) if error is None else jnp.all(jnp.isfinite(error))
        )
        unused_traces = (beta == 0.0) & (decay == 0.0)
        previous_checked = state.replace(  # type: ignore[attr-defined]
            traces=jnp.where(unused_traces, jnp.zeros_like(state.traces), state.traces),
        )
        update_applied = (
            floating_tree_is_finite(previous_checked)
            & jnp.all(jnp.isfinite(gradient))
            & error_is_finite
            & jnp.all(jnp.isfinite(meta_delta))
            & jnp.all(jnp.isfinite(step))
            & floating_tree_is_finite(new_state)
        )

        return ParamOptimizerUpdate(
            step=neutralize_array(update_applied, step),
            new_state=select_transaction(update_applied, new_state, state),
            update_applied=update_applied,
        )

    def update(
        self,
        state: IDBDState,
        error: Array,
        observation: Array,
    ) -> OptimizerUpdate:
        """Compute IDBD weight update with adaptive step-sizes.

        Following Sutton 1992, Figure 2, the operation ordering is:

        1. Meta-update: ``log_alpha_i += beta * error * x_i * h_i`` (using OLD traces)
        2. Compute NEW step-sizes: ``alpha_i = exp(log_alpha_i)``
        3. Update weights: ``w_i += alpha_i * error * x_i`` (using NEW alpha)
        4. Update traces: ``h_i = h_i * max(0, 1 - alpha_i * x_i^2) + alpha_i * error * x_i``
           (using NEW alpha)

        The trace h_i tracks the correlation between current and past gradients.
        When gradients consistently point the same direction, h_i grows,
        leading to larger step-sizes.

        Args:
            state: Current IDBD state
            error: Prediction error (scalar)
            observation: Feature vector

        Returns:
            OptimizerUpdate with weight deltas and updated state
        """
        error_scalar = jnp.squeeze(error)
        beta = state.meta_step_size

        # 1. Meta-update: adapt step-sizes using OLD traces. The product is
        # textually unchanged so ordinary finite trajectories stay
        # bit-identical; the guard below is the only behavioral change.
        gradient_correlation = error_scalar * observation * state.traces
        meta_delta = _skip_zero_scale(beta, gradient_correlation)

        # Clip log step-sizes to prevent numerical issues. clip(NaN) is NaN,
        # so a non-finite correlation (e.g. an inf error against a zero trace)
        # must skip adaptation for that weight instead of poisoning it for
        # every later finite update.
        new_log_step_sizes = jnp.where(
            jnp.isfinite(meta_delta),
            jnp.clip(state.log_step_sizes + meta_delta, -10.0, 2.0),
            state.log_step_sizes,
        )

        # 2. Compute NEW step-sizes
        new_alphas = jnp.exp(new_log_step_sizes)

        # 3. Weight updates using NEW alpha: alpha_i * error * x_i
        weight_delta = new_alphas * error_scalar * observation

        # 4. Update traces using NEW alpha: h_i = h_i * decay + alpha_i * error * x_i
        # decay = max(0, 1 - alpha_i * x_i^2)
        decay = jnp.maximum(0.0, 1.0 - new_alphas * observation**2)
        new_traces = _skip_zero_scale(decay, state.traces) + new_alphas * error_scalar * observation

        # Bias updates (same ordering: meta-update first, then new alpha).
        # Same guard: a non-finite correlation keeps the previous step-size.
        bias_gradient_correlation = error_scalar * state.bias_trace
        bias_meta_delta = _skip_zero_scale(beta, bias_gradient_correlation)
        new_bias_step_size = jnp.where(
            jnp.isfinite(bias_meta_delta),
            jnp.clip(state.bias_step_size * jnp.exp(bias_meta_delta), 1e-6, 1.0),
            state.bias_step_size,
        )

        bias_delta = new_bias_step_size * error_scalar

        bias_decay = jnp.maximum(0.0, 1.0 - new_bias_step_size)
        new_bias_trace = (
            _skip_zero_scale(bias_decay, state.bias_trace) + new_bias_step_size * error_scalar
        )

        candidate_state = IDBDState(
            log_step_sizes=new_log_step_sizes,
            traces=new_traces,
            meta_step_size=beta,
            bias_step_size=new_bias_step_size,
            bias_trace=new_bias_trace,
        )
        candidate_metrics = {
            "mean_step_size": jnp.mean(new_alphas),
            "min_step_size": jnp.min(new_alphas),
            "max_step_size": jnp.max(new_alphas),
        }

        unused_traces = (beta == 0.0) & (decay == 0.0)
        unused_bias_trace = (beta == 0.0) & (bias_decay == 0.0)
        previous_checked = state.replace(  # type: ignore[attr-defined]
            traces=jnp.where(unused_traces, jnp.zeros_like(state.traces), state.traces),
            bias_trace=jnp.where(
                unused_bias_trace, jnp.zeros_like(state.bias_trace), state.bias_trace
            ),
        )
        update_applied = (
            floating_tree_is_finite(previous_checked)
            & jnp.isfinite(error_scalar)
            & jnp.all(jnp.isfinite(observation))
            & jnp.all(jnp.isfinite(meta_delta))
            & jnp.isfinite(bias_meta_delta)
            & jnp.all(jnp.isfinite(weight_delta))
            & jnp.isfinite(bias_delta)
            & floating_tree_is_finite(candidate_state)
            & floating_tree_is_finite(candidate_metrics)
        )
        new_state = select_transaction(update_applied, candidate_state, state)
        metrics = neutralize_metrics(update_applied, candidate_metrics)

        return OptimizerUpdate(
            weight_delta=neutralize_array(update_applied, weight_delta),
            bias_delta=neutralize_array(update_applied, bias_delta),
            new_state=new_state,
            metrics=metrics,
            update_applied=update_applied,
        )


class Autostep(Optimizer[AutostepState]):
    """Autostep optimizer with tuning-free step-size adaptation.

    Implements the exact algorithm from Mahmood et al. 2012, Table 1.

    The algorithm maintains per-weight step-sizes that adapt based on
    meta-gradient correlation. The key innovations are:
    - Self-regulated normalizers (v_i) that track meta-gradient magnitude
      ``|delta * x_i * h_i|`` for stable meta-updates
    - Overshoot prevention via effective step-size normalization
      ``M = max(sum(alpha_i * x_i^2), 1)``

    Per-sample update (Table 1):

    1. ``v_i = max(|delta*x_i*h_i|, v_i + (1/tau)*alpha_i*x_i^2*(|delta*x_i*h_i| - v_i))``
    2. ``alpha_i *= exp(mu * delta*x_i*h_i / v_i)`` where ``v_i > 0``
    3. ``M = max(sum(alpha_i * x_i^2), 1)``; ``alpha_i /= M``
    4. ``w_i += alpha_i * delta * x_i`` (weight update with NEW alpha)
    5. ``h_i = h_i * (1 - alpha_i * x_i^2) + alpha_i * delta * x_i`` (trace update)

    Reference: Mahmood, A.R., Sutton, R.S., Degris, T., & Pilarski, P.M. (2012).
    "Tuning-free step-size adaptation"

    Attributes:
        initial_step_size: Initial per-weight step-size
        meta_step_size: Meta learning rate mu for adapting step-sizes
        tau: Time constant for normalizer adaptation (default: 10000)
    """

    def __init__(
        self,
        initial_step_size: float = 0.01,
        meta_step_size: float = 0.01,
        tau: float = 10000.0,
    ):
        """Initialize Autostep optimizer.

        Args:
            initial_step_size: Initial value for per-weight step-sizes
            meta_step_size: Meta learning rate for adapting step-sizes
            tau: Time constant for normalizer adaptation (default: 10000).
                Higher values mean slower normalizer decay.

        Raises:
            ValueError: If ``initial_step_size`` is not a finite non-bool
                positive real, or ``meta_step_size`` is not a finite non-bool
                nonnegative real.
        """
        self._initial_step_size = validated_float32_scalar(
            "initial_step_size", initial_step_size, positive=True
        )
        self._meta_step_size = validated_float32_scalar(
            "meta_step_size", meta_step_size, lower=0.0
        )
        self._tau = tau

    def to_config(self) -> dict[str, Any]:
        """Serialize configuration to dict."""
        return {
            "type": "Autostep",
            "initial_step_size": self._initial_step_size,
            "meta_step_size": self._meta_step_size,
            "tau": self._tau,
        }

    def init(self, feature_dim: int) -> AutostepState:
        """Initialize Autostep state.

        Normalizers (v_i) and traces (h_i) are initialized to 0 per the paper.

        Args:
            feature_dim: Dimension of weight vector

        Returns:
            Autostep state with per-weight step-sizes, traces, and normalizers
        """
        feature_dim = _require_int32("feature_dim", feature_dim, minimum=1)
        _require_float32_state("Autostep state", 3 * feature_dim + 5)
        # Table-1's parameter path retains the source banks plus its squared
        # feature, meta, normalizer, normalized-step, trace, checked-source,
        # selected-result, and neutral-output buffers.
        _require_float32_update_working_set(
            "Autostep update working set", 30 * feature_dim + 8
        )
        return AutostepState(
            step_sizes=jnp.full(feature_dim, self._initial_step_size, dtype=jnp.float32),
            traces=jnp.zeros(feature_dim, dtype=jnp.float32),
            normalizers=jnp.zeros(feature_dim, dtype=jnp.float32),
            meta_step_size=jnp.array(self._meta_step_size, dtype=jnp.float32),
            tau=jnp.array(self._tau, dtype=jnp.float32),
            bias_step_size=jnp.array(self._initial_step_size, dtype=jnp.float32),
            bias_trace=jnp.array(0.0, dtype=jnp.float32),
            bias_normalizer=jnp.array(0.0, dtype=jnp.float32),
        )

    def init_for_shape(self, shape: tuple[int, ...]) -> AutostepParamState:
        """Initialize Autostep state for arbitrary-shape parameters.

        Args:
            shape: Shape of the parameter array

        Returns:
            AutostepParamState with arrays matching the given shape
        """
        shape = _require_shape(shape)
        n = _shape_size(shape)
        _require_float32_state("Autostep parameter state", 3 * n + 2)
        _require_float32_update_working_set("Autostep update working set", 30 * n + 8)
        return AutostepParamState(
            step_sizes=jnp.full(shape, self._initial_step_size, dtype=jnp.float32),
            traces=jnp.zeros(shape, dtype=jnp.float32),
            normalizers=jnp.zeros(shape, dtype=jnp.float32),
            meta_step_size=jnp.array(self._meta_step_size, dtype=jnp.float32),
            tau=jnp.array(self._tau, dtype=jnp.float32),
        )

    def update_from_gradient(
        self,
        state: AutostepParamState,
        gradient: Array,
        error: Array | None = None,
    ) -> tuple[Array, AutostepParamState]:
        """Return the historical unchecked pair for compatibility."""

        result = self.update_from_gradient_checked(state, gradient, error)
        return result.step, cast(AutostepParamState, result.new_state)

    def update_from_gradient_checked(
        self,
        state: AutostepParamState,
        gradient: Array,
        error: Array | None = None,
    ) -> ParamOptimizerUpdate:
        """Compute Autostep update from pre-computed gradient (MLP path).

        Implements the Table 1 algorithm generalized for arbitrary-shape
        parameters, where ``gradient`` plays the role of the eligibility
        trace ``z`` (prediction gradient).

        When ``error`` is provided, the full paper algorithm is used:
        meta-gradient is ``error * z * h``. When ``error`` is None,
        falls back to error-free approximation (``z * h``).

        The returned step does NOT include the error -- the caller applies
        ``param += error * step`` after optional bounding.

        Args:
            state: Current Autostep param state
            gradient: Pre-computed gradient / eligibility trace (same shape as state arrays)
            error: Optional prediction error scalar. When provided, enables
                the full paper algorithm with error-scaled meta-gradients.

        Returns:
            ``(step, new_state)`` where step has the same shape as gradient
        """
        mu = state.meta_step_size
        tau = state.tau

        z = gradient  # eligibility trace
        z_sq = z**2

        # Compute meta-gradient: δ*z*h (or z*h if error is None)
        if error is not None:
            error_scalar = jnp.squeeze(error)
            meta_gradient = error_scalar * z * state.traces
        else:
            meta_gradient = z * state.traces

        abs_meta_gradient = jnp.abs(meta_gradient)

        # Eq. 4: v_i = max(|meta_grad|, v_i + (1/τ)*α_i*z_i²*(|meta_grad| - v_i)).
        # max(NaN) is NaN, so a non-finite meta-gradient (inf*0 or overflow)
        # must keep the previous normalizer instead of poisoning v.
        v_update = state.normalizers + (1.0 / tau) * state.step_sizes * z_sq * (
            abs_meta_gradient - state.normalizers
        )
        normalizer_candidate = jnp.maximum(abs_meta_gradient, v_update)
        valid_meta_update = jnp.logical_and(
            jnp.isfinite(meta_gradient), jnp.isfinite(normalizer_candidate)
        )
        new_normalizers = jnp.where(
            valid_meta_update,
            normalizer_candidate,
            state.normalizers,
        )

        # Eq. 5: α_i *= exp(μ * meta_grad / v_i) where v_i > 0. The v>0 gate
        # is not a finiteness guard: inf/inf = NaN and clip(NaN) is NaN.
        safe_v = jnp.maximum(new_normalizers, 1e-38)
        new_step_sizes = jnp.where(
            jnp.logical_and(valid_meta_update, new_normalizers > 0),
            state.step_sizes * jnp.exp(mu * meta_gradient / safe_v),
            state.step_sizes,
        )

        # Eq. 6-7: M = max(Σ α_i*z_i², 1); α_i /= M
        effective_terms = new_step_sizes * z_sq
        effective_step = jnp.sum(effective_terms)
        m_factor = jnp.maximum(effective_step, 1.0)
        new_step_sizes = new_step_sizes / m_factor

        # Clip step-sizes for numerical safety
        new_step_sizes = jnp.clip(new_step_sizes, 1e-8, 1.0)

        # Compute step: α_i * z_i (error applied externally)
        step = new_step_sizes * z

        # Trace update: h_i = h_i*(1 - α_i*z_i²) + α_i*δ*z_i
        trace_decay = 1.0 - new_step_sizes * z_sq
        decayed_traces = _skip_zero_scale(trace_decay, state.traces)
        if error is not None:
            trace_candidate = decayed_traces + new_step_sizes * error_scalar * z
        else:
            trace_candidate = decayed_traces + new_step_sizes * z
        new_traces = jnp.where(jnp.isfinite(trace_candidate), trace_candidate, state.traces)

        candidate_state = AutostepParamState(
            step_sizes=new_step_sizes,
            traces=new_traces,
            normalizers=new_normalizers,
            meta_step_size=mu,
            tau=tau,
        )
        error_is_finite = (
            jnp.asarray(True, dtype=jnp.bool_) if error is None else jnp.all(jnp.isfinite(error))
        )
        unused_traces = (mu == 0.0) & (trace_decay == 0.0)
        previous_checked = state.replace(  # type: ignore[attr-defined]
            traces=jnp.where(unused_traces, jnp.zeros_like(state.traces), state.traces),
        )
        meta_ok = jnp.logical_or(mu == 0.0, jnp.all(valid_meta_update))
        update_applied = (
            floating_tree_is_finite(previous_checked)
            & jnp.all(jnp.isfinite(gradient))
            & error_is_finite
            & meta_ok
            & jnp.all(jnp.isfinite(trace_candidate))
            & jnp.all(jnp.isfinite(step))
            & floating_tree_is_finite(candidate_state)
        )

        return ParamOptimizerUpdate(
            step=neutralize_array(update_applied, step),
            new_state=select_transaction(update_applied, candidate_state, state),
            update_applied=update_applied,
        )

    def update(
        self,
        state: AutostepState,
        error: Array,
        observation: Array,
    ) -> OptimizerUpdate:
        """Compute Autostep weight update following Mahmood et al. 2012, Table 1.

        The algorithm per sample:

        1. Eq. 4: ``v_i = max(|δ*x_i*h_i|, v_i + (1/τ)*α_i*x_i²*(|δ*x_i*h_i| - v_i))``
        2. Eq. 5: ``α_i *= exp(μ * δ*x_i*h_i / v_i)`` where ``v_i > 0``
        3. Eq. 6-7: ``M = max(Σ α_i*x_i² + α_bias, 1)``; ``α_i /= M``, ``α_bias /= M``
        4. Weight update: ``w_i += α_i * δ * x_i`` (with NEW alpha)
        5. Trace update: ``h_i = h_i*(1 - α_i*x_i²) + α_i*δ*x_i``

        Args:
            state: Current Autostep state
            error: Prediction error (scalar)
            observation: Feature vector

        Returns:
            OptimizerUpdate with weight deltas and updated state
        """
        error_scalar = jnp.squeeze(error)
        mu = state.meta_step_size
        tau = state.tau

        x = observation
        x_sq = x**2

        # --- Weights ---
        # Meta-gradient: δ*x_i*h_i
        meta_gradient = error_scalar * x * state.traces
        abs_meta_gradient = jnp.abs(meta_gradient)

        # Eq. 4: v_i update (self-regulated EMA). A non-finite
        # meta-gradient must keep the previous normalizer instead of
        # poisoning v through max(NaN).
        v_update = state.normalizers + (1.0 / tau) * state.step_sizes * x_sq * (
            abs_meta_gradient - state.normalizers
        )
        normalizer_candidate = jnp.maximum(abs_meta_gradient, v_update)
        valid_meta_update = jnp.logical_and(
            jnp.isfinite(meta_gradient), jnp.isfinite(normalizer_candidate)
        )
        new_normalizers = jnp.where(
            valid_meta_update,
            normalizer_candidate,
            state.normalizers,
        )

        # Eq. 5: α_i *= exp(μ * meta_grad / v_i) where v_i > 0. The v>0 gate
        # alone lets inf/inf produce a NaN step-size.
        safe_v = jnp.maximum(new_normalizers, 1e-38)
        new_step_sizes = jnp.where(
            jnp.logical_and(valid_meta_update, new_normalizers > 0),
            state.step_sizes * jnp.exp(mu * meta_gradient / safe_v),
            state.step_sizes,
        )

        # --- Bias ---
        # Meta-gradient for bias (implicit x=1): δ*h_bias
        bias_meta_gradient = error_scalar * state.bias_trace
        abs_bias_meta_gradient = jnp.abs(bias_meta_gradient)

        # Eq. 4 for bias. Apply the same non-finite correlation guard.
        bias_v_update = state.bias_normalizer + (1.0 / tau) * state.bias_step_size * (
            abs_bias_meta_gradient - state.bias_normalizer
        )
        bias_normalizer_candidate = jnp.maximum(abs_bias_meta_gradient, bias_v_update)
        valid_bias_meta_update = jnp.logical_and(
            jnp.isfinite(bias_meta_gradient),
            jnp.isfinite(bias_normalizer_candidate),
        )
        new_bias_normalizer = jnp.where(
            valid_bias_meta_update,
            bias_normalizer_candidate,
            state.bias_normalizer,
        )

        # Eq. 5 for bias. Keep inf/inf out of the exponential update.
        safe_bias_v = jnp.maximum(new_bias_normalizer, 1e-38)
        new_bias_step_size = jnp.where(
            jnp.logical_and(valid_bias_meta_update, new_bias_normalizer > 0),
            state.bias_step_size * jnp.exp(mu * bias_meta_gradient / safe_bias_v),
            state.bias_step_size,
        )

        # Eq. 6-7: Overshoot prevention (joint over weights + bias)
        # M = max(Σ α_i*x_i² + α_bias*1², 1)
        effective_terms = new_step_sizes * x_sq
        effective_step = jnp.sum(effective_terms) + new_bias_step_size
        m_factor = jnp.maximum(effective_step, 1.0)
        new_step_sizes = new_step_sizes / m_factor
        new_bias_step_size = new_bias_step_size / m_factor

        # Clip step-sizes for numerical safety
        new_step_sizes = jnp.clip(new_step_sizes, 1e-8, 1.0)
        new_bias_step_size = jnp.clip(new_bias_step_size, 1e-8, 1.0)

        # Weight update with NEW alpha: α_i * δ * x_i
        weight_delta = new_step_sizes * error_scalar * x

        # Bias update: α_bias * δ
        bias_delta = new_bias_step_size * error_scalar

        # Trace update: h_i = h_i*(1 - α_i*x_i²) + α_i*δ*x_i
        trace_decay = 1.0 - new_step_sizes * x_sq
        trace_candidate = (
            _skip_zero_scale(trace_decay, state.traces) + new_step_sizes * error_scalar * x
        )
        new_traces = jnp.where(jnp.isfinite(trace_candidate), trace_candidate, state.traces)

        # Bias trace: h_bias = h_bias*(1 - α_bias) + α_bias*δ
        bias_trace_decay = 1.0 - new_bias_step_size
        bias_trace_candidate = (
            _skip_zero_scale(bias_trace_decay, state.bias_trace) + new_bias_step_size * error_scalar
        )
        new_bias_trace = jnp.where(
            jnp.isfinite(bias_trace_candidate),
            bias_trace_candidate,
            state.bias_trace,
        )

        candidate_state = AutostepState(
            step_sizes=new_step_sizes,
            traces=new_traces,
            normalizers=new_normalizers,
            meta_step_size=mu,
            tau=tau,
            bias_step_size=new_bias_step_size,
            bias_trace=new_bias_trace,
            bias_normalizer=new_bias_normalizer,
        )
        candidate_metrics = {
            "mean_step_size": jnp.mean(new_step_sizes),
            "min_step_size": jnp.min(new_step_sizes),
            "max_step_size": jnp.max(new_step_sizes),
            "mean_normalizer": jnp.mean(new_normalizers),
        }

        unused_traces = (mu == 0.0) & (trace_decay == 0.0)
        unused_bias_trace = (mu == 0.0) & (bias_trace_decay == 0.0)
        previous_checked = state.replace(  # type: ignore[attr-defined]
            traces=jnp.where(unused_traces, jnp.zeros_like(state.traces), state.traces),
            bias_trace=jnp.where(
                unused_bias_trace, jnp.zeros_like(state.bias_trace), state.bias_trace
            ),
        )
        meta_ok = jnp.logical_or(mu == 0.0, jnp.all(valid_meta_update))
        bias_meta_ok = jnp.logical_or(mu == 0.0, valid_bias_meta_update)
        update_applied = (
            floating_tree_is_finite(previous_checked)
            & jnp.isfinite(error_scalar)
            & jnp.all(jnp.isfinite(observation))
            & meta_ok
            & bias_meta_ok
            & jnp.all(jnp.isfinite(trace_candidate))
            & jnp.isfinite(bias_trace_candidate)
            & jnp.all(jnp.isfinite(weight_delta))
            & jnp.isfinite(bias_delta)
            & floating_tree_is_finite(candidate_state)
            & floating_tree_is_finite(candidate_metrics)
        )
        new_state = select_transaction(update_applied, candidate_state, state)
        metrics = neutralize_metrics(update_applied, candidate_metrics)

        return OptimizerUpdate(
            weight_delta=neutralize_array(update_applied, weight_delta),
            bias_delta=neutralize_array(update_applied, bias_delta),
            new_state=new_state,
            metrics=metrics,
            update_applied=update_applied,
        )


class AutostepGTDLambda(Optimizer[AutostepGTDLambdaState]):
    """Autostep-for-GTD(lambda) supervised-limit optimizer.

    Kearney et al. (2019) apply Autostep-style normalized meta-descent in a
    GTD(lambda) setting. Step 1 uses the supervised limit, where ``rho=1`` and
    ``gamma=0``; this wrapper keeps an eligibility-trace state so the public
    name is explicit while reusing the vetted Autostep linear update.
    """

    def __init__(
        self,
        initial_step_size: float = 0.01,
        meta_step_size: float = 0.01,
        tau: float = 10000.0,
        trace_decay: float = 0.0,
    ):
        """Initialize Autostep-for-GTD(lambda).

        Args:
            initial_step_size: Initial per-weight step-size
            meta_step_size: Meta learning rate for adapting step-sizes
            tau: Time constant for Autostep normalizer adaptation
            trace_decay: Eligibility trace decay; ``0`` recovers supervised
                Autostep exactly.
        """
        self._base = Autostep(
            initial_step_size=initial_step_size,
            meta_step_size=meta_step_size,
            tau=tau,
        )
        self._initial_step_size = initial_step_size
        self._meta_step_size = meta_step_size
        self._tau = tau
        self._trace_decay = trace_decay

    def to_config(self) -> dict[str, Any]:
        """Serialize configuration to dict."""
        return {
            "type": "AutostepGTDLambda",
            "initial_step_size": self._initial_step_size,
            "meta_step_size": self._meta_step_size,
            "tau": self._tau,
            "trace_decay": self._trace_decay,
        }

    def init(self, feature_dim: int) -> AutostepGTDLambdaState:
        """Initialize optimizer state."""
        feature_dim = _require_int32("feature_dim", feature_dim, minimum=1)
        _require_float32_state("AutostepGTDLambda state", 4 * feature_dim + 7)
        _require_float32_update_working_set(
            "AutostepGTDLambda update working set", 40 * feature_dim + 8
        )
        base_state = self._base.init(feature_dim)
        return AutostepGTDLambdaState(
            step_sizes=base_state.step_sizes,
            traces=base_state.traces,
            normalizers=base_state.normalizers,
            eligibility_traces=jnp.zeros(feature_dim, dtype=jnp.float32),
            meta_step_size=base_state.meta_step_size,
            tau=base_state.tau,
            trace_decay=jnp.array(self._trace_decay, dtype=jnp.float32),
            bias_step_size=base_state.bias_step_size,
            bias_trace=base_state.bias_trace,
            bias_normalizer=base_state.bias_normalizer,
            bias_eligibility_trace=jnp.array(0.0, dtype=jnp.float32),
        )

    def update(
        self,
        state: AutostepGTDLambdaState,
        error: Array,
        observation: Array,
    ) -> OptimizerUpdate:
        """Compute one supervised-limit Autostep-for-GTD(lambda) update."""
        eligibility = _skip_zero_scale(state.trace_decay, state.eligibility_traces) + observation
        bias_eligibility = _skip_zero_scale(state.trace_decay, state.bias_eligibility_trace) + 1.0
        base_state = AutostepState(
            step_sizes=state.step_sizes,
            traces=state.traces,
            normalizers=state.normalizers,
            meta_step_size=state.meta_step_size,
            tau=state.tau,
            bias_step_size=state.bias_step_size,
            bias_trace=state.bias_trace,
            bias_normalizer=state.bias_normalizer,
        )
        base_update = self._base.update(base_state, error, eligibility)
        new_base_state = cast(AutostepState, base_update.new_state)
        candidate_state = AutostepGTDLambdaState(
            step_sizes=new_base_state.step_sizes,
            traces=new_base_state.traces,
            normalizers=new_base_state.normalizers,
            eligibility_traces=eligibility,
            meta_step_size=new_base_state.meta_step_size,
            tau=new_base_state.tau,
            trace_decay=state.trace_decay,
            bias_step_size=new_base_state.bias_step_size,
            bias_trace=new_base_state.bias_trace,
            bias_normalizer=new_base_state.bias_normalizer,
            bias_eligibility_trace=bias_eligibility,
        )
        previous_checked = state
        if self._trace_decay == 0.0:
            previous_checked = state.replace(  # type: ignore[attr-defined]
                eligibility_traces=jnp.zeros_like(state.eligibility_traces),
                bias_eligibility_trace=jnp.zeros_like(state.bias_eligibility_trace),
            )
        update_applied = (
            base_update.update_applied
            & floating_tree_is_finite(previous_checked)
            & jnp.all(jnp.isfinite(observation))
            & floating_tree_is_finite(candidate_state)
        )
        new_state = select_transaction(update_applied, candidate_state, state)
        return OptimizerUpdate(
            weight_delta=neutralize_array(update_applied, base_update.weight_delta),
            bias_delta=neutralize_array(update_applied, base_update.bias_delta),
            new_state=new_state,
            metrics=neutralize_metrics(update_applied, base_update.metrics),
            update_applied=update_applied,
        )


class ObGD(Optimizer[ObGDState]):
    """Observation-bounded Gradient Descent optimizer.

    ObGD prevents overshooting by dynamically bounding the effective step-size
    based on the magnitude of the prediction error and eligibility traces.
    When the combined update magnitude would be too large, the step-size is
    scaled down to prevent the prediction from overshooting the target.

    This is the deep-network generalization of Autostep's overshooting
    prevention, designed for streaming reinforcement learning.

    For supervised learning (gamma=0, lamda=0), traces equal the current
    observation each step, making ObGD equivalent to LMS with dynamic
    step-size bounding.

    The ObGD algorithm:

    1. Update traces: ``z = gamma * lamda * z + observation``
    2. Compute bound: ``M = alpha * kappa * max(|error|, 1) * (||z_w||_1 + |z_b|)``
    3. Effective step: ``alpha_eff = min(alpha, alpha / M)`` (i.e. ``alpha / max(M, 1)``)
    4. Weight delta: ``delta_w = alpha_eff * error * z_w``
    5. Bias delta: ``delta_b = alpha_eff * error * z_b``

    Reference: Elsayed et al. 2024, "Streaming Deep Reinforcement Learning
    Finally Works"

    Attributes:
        step_size: Base learning rate alpha
        kappa: Bounding sensitivity parameter (higher = more conservative)
        gamma: Discount factor for trace decay (0 for supervised learning)
        lamda: Eligibility trace decay parameter (0 for supervised learning)
    """

    def __init__(
        self,
        step_size: float = 1.0,
        kappa: float = 2.0,
        gamma: float = 0.0,
        lamda: float = 0.0,
    ):
        """Initialize ObGD optimizer.

        Args:
            step_size: Base learning rate (default: 1.0)
            kappa: Bounding sensitivity parameter (default: 2.0)
            gamma: Discount factor for trace decay (default: 0.0 for supervised)
            lamda: Eligibility trace decay parameter (default: 0.0 for supervised)

        Raises:
            ValueError: If ``step_size`` is not a finite non-bool positive real,
                ``kappa`` is not a finite non-bool nonnegative real, or
                ``gamma`` / ``lamda`` are not finite non-bool reals in ``[0, 1]``.
        """
        self._step_size = validated_float32_scalar("step_size", step_size, positive=True)
        self._kappa = validated_float32_scalar("kappa", kappa, lower=0.0)
        self._gamma = validated_float32_scalar("gamma", gamma, lower=0.0, upper=1.0)
        self._lamda = validated_float32_scalar("lamda", lamda, lower=0.0, upper=1.0)

    def to_config(self) -> dict[str, Any]:
        """Serialize configuration to dict."""
        return {
            "type": "ObGD",
            "step_size": self._step_size,
            "kappa": self._kappa,
            "gamma": self._gamma,
            "lamda": self._lamda,
        }

    def init(self, feature_dim: int) -> ObGDState:
        """Initialize ObGD state.

        Args:
            feature_dim: Dimension of weight vector

        Returns:
            ObGD state with eligibility traces
        """
        feature_dim = _require_int32("feature_dim", feature_dim, minimum=1)
        _require_float32_state("ObGD state", feature_dim + 5)
        _require_float32_update_working_set("ObGD update working set", 12 * feature_dim + 8)
        return ObGDState(
            step_size=jnp.array(self._step_size, dtype=jnp.float32),
            kappa=jnp.array(self._kappa, dtype=jnp.float32),
            traces=jnp.zeros(feature_dim, dtype=jnp.float32),
            bias_trace=jnp.array(0.0, dtype=jnp.float32),
            gamma=jnp.array(self._gamma, dtype=jnp.float32),
            lamda=jnp.array(self._lamda, dtype=jnp.float32),
        )

    def update(
        self,
        state: ObGDState,
        error: Array,
        observation: Array,
    ) -> OptimizerUpdate:
        """Compute ObGD weight update with overshooting prevention.

        The bounding mechanism scales down the step-size when the combined
        effect of error magnitude, trace norm, and step-size would cause
        the prediction to overshoot the target.

        Args:
            state: Current ObGD state
            error: Prediction error (target - prediction)
            observation: Current observation/feature vector

        Returns:
            OptimizerUpdate with bounded weight deltas and updated state
        """
        error_scalar = jnp.squeeze(error)
        alpha = state.step_size
        kappa = state.kappa

        # Update eligibility traces: z = gamma * lamda * z + observation.
        # Skip 0 * inf before adding the current observation.
        decay_scale = jnp.where(
            (state.gamma == 0.0) | (state.lamda == 0.0),
            jnp.zeros_like(state.gamma),
            state.gamma * state.lamda,
        )
        new_traces = _skip_zero_scale(decay_scale, state.traces) + observation
        new_bias_trace = _skip_zero_scale(decay_scale, state.bias_trace) + 1.0

        # Compute z_sum (L1 norm of all traces)
        z_sum = jnp.sum(jnp.abs(new_traces)) + jnp.abs(new_bias_trace)

        # Compute bounding factor: M = alpha * kappa * max(|error|, 1) * z_sum
        delta_bar = jnp.maximum(jnp.abs(error_scalar), 1.0)
        dot_product = delta_bar * z_sum * alpha * kappa

        # Effective step-size: alpha / max(M, 1)
        alpha_eff = alpha / jnp.maximum(dot_product, 1.0)

        # Weight and bias deltas. An infinite bound collapses ``alpha_eff``
        # to zero, so the resulting 0*inf is the bound's exact no-op. A NaN
        # from any other source remains visible.
        collapsed = alpha_eff == 0
        raw_weight_step = error_scalar * new_traces
        raw_bias_step = error_scalar * new_bias_trace
        weight_delta = zero_if_collapsed_infinity(
            alpha_eff * raw_weight_step,
            error_scalar,
            collapsed,
        )
        bias_delta = zero_if_collapsed_infinity(
            alpha_eff * raw_bias_step,
            error_scalar,
            collapsed,
        )

        candidate_state = ObGDState(
            step_size=alpha,
            kappa=kappa,
            traces=new_traces,
            bias_trace=new_bias_trace,
            gamma=state.gamma,
            lamda=state.lamda,
        )
        candidate_metrics = {
            "step_size": alpha,
            "effective_step_size": alpha_eff,
            "bounding_factor": dot_product,
        }

        previous_checked = state
        if self._gamma == 0.0 or self._lamda == 0.0:
            previous_checked = state.replace(  # type: ignore[attr-defined]
                traces=jnp.zeros_like(state.traces),
                bias_trace=jnp.zeros_like(state.bias_trace),
            )
        update_applied = (
            floating_tree_is_finite(previous_checked)
            & jnp.isfinite(error_scalar)
            & jnp.all(jnp.isfinite(observation))
            & jnp.all(jnp.isfinite(weight_delta))
            & jnp.isfinite(bias_delta)
            & floating_tree_is_finite(candidate_state)
            & floating_tree_is_finite(candidate_metrics)
        )
        new_state = select_transaction(update_applied, candidate_state, state)
        metrics = neutralize_metrics(update_applied, candidate_metrics)

        return OptimizerUpdate(
            weight_delta=neutralize_array(update_applied, weight_delta),
            bias_delta=neutralize_array(update_applied, bias_delta),
            new_state=new_state,
            metrics=metrics,
            update_applied=update_applied,
        )


# =============================================================================
# TD Optimizers (for Step 3+ of Alberta Plan)
# =============================================================================


@chex.dataclass(frozen=True)
class TDOptimizerUpdate:
    """Result of a TD optimizer update step.

    Attributes:
        weight_delta: Change to apply to weights
        bias_delta: Change to apply to bias
        new_state: Updated optimizer state
        metrics: Dictionary of metrics for logging
    """

    weight_delta: Float[Array, " feature_dim"]
    bias_delta: Float[Array, ""]
    new_state: TDIDBDState | AutoTDIDBDState
    metrics: dict[str, Array]
    update_applied: Bool[Array, ""]


class TDOptimizer[StateT: (TDIDBDState, AutoTDIDBDState)](ABC):
    """Base class for TD optimizers.

    TD optimizers handle temporal-difference learning with eligibility traces.
    They take TD error and both current and next observations as input.
    """

    @abstractmethod
    def init(self, feature_dim: int) -> StateT:
        """Initialize optimizer state.

        Args:
            feature_dim: Dimension of weight vector

        Returns:
            Initial optimizer state
        """
        ...

    @abstractmethod
    def update(
        self,
        state: StateT,
        td_error: Array,
        observation: Array,
        next_observation: Array,
        gamma: Array,
    ) -> TDOptimizerUpdate:
        """Compute weight updates given TD error.

        Args:
            state: Current optimizer state
            td_error: TD error delta = R + gamma*V(s') - V(s)
            observation: Current observation phi(s)
            next_observation: Next observation phi(s')
            gamma: Discount factor gamma (0 at terminal)

        Returns:
            TDOptimizerUpdate with deltas and new state
        """
        ...


class TDIDBD(TDOptimizer[TDIDBDState]):
    """TD-IDBD optimizer for temporal-difference learning.

    Extends IDBD to TD learning with eligibility traces. Maintains per-weight
    adaptive step-sizes that are meta-learned based on gradient correlation
    in the TD setting.

    Two variants are supported:
    - Semi-gradient (default): Uses only phi(s) in meta-update, more stable
    - Ordinary gradient: Uses both phi(s) and phi(s'), more accurate but sensitive

    Reference: Kearney et al. 2019, "Learning Feature Relevance Through Step Size
    Adaptation in Temporal-Difference Learning"

    Attributes:
        initial_step_size: Initial per-weight step-size
        meta_step_size: Meta learning rate theta
        trace_decay: Eligibility trace decay lambda
        use_semi_gradient: If True, use semi-gradient variant (default)
    """

    def __init__(
        self,
        initial_step_size: float = 0.01,
        meta_step_size: float = 0.01,
        trace_decay: float = 0.0,
        use_semi_gradient: bool = True,
    ):
        """Initialize TD-IDBD optimizer.

        Args:
            initial_step_size: Initial value for per-weight step-sizes
            meta_step_size: Meta learning rate theta for adapting step-sizes
            trace_decay: Eligibility trace decay lambda (0 = TD(0))
            use_semi_gradient: If True, use semi-gradient variant (recommended)
        """
        self._initial_step_size = initial_step_size
        self._meta_step_size = meta_step_size
        self._trace_decay = trace_decay
        self._use_semi_gradient = use_semi_gradient

    def init(self, feature_dim: int) -> TDIDBDState:
        """Initialize TD-IDBD state.

        Args:
            feature_dim: Dimension of weight vector

        Returns:
            TD-IDBD state with per-weight step-sizes, traces, and h traces
        """
        feature_dim = _require_int32("feature_dim", feature_dim, minimum=1)
        _require_float32_state("TDIDBD state", 3 * feature_dim + 5)
        _require_float32_update_working_set(
            "TDIDBD update working set", 22 * feature_dim + 8
        )
        return TDIDBDState(
            log_step_sizes=jnp.full(
                feature_dim, jnp.log(self._initial_step_size), dtype=jnp.float32
            ),
            eligibility_traces=jnp.zeros(feature_dim, dtype=jnp.float32),
            h_traces=jnp.zeros(feature_dim, dtype=jnp.float32),
            meta_step_size=jnp.array(self._meta_step_size, dtype=jnp.float32),
            trace_decay=jnp.array(self._trace_decay, dtype=jnp.float32),
            bias_log_step_size=jnp.array(jnp.log(self._initial_step_size), dtype=jnp.float32),
            bias_eligibility_trace=jnp.array(0.0, dtype=jnp.float32),
            bias_h_trace=jnp.array(0.0, dtype=jnp.float32),
        )

    def update(
        self,
        state: TDIDBDState,
        td_error: Array,
        observation: Array,
        next_observation: Array,
        gamma: Array,
    ) -> TDOptimizerUpdate:
        """Compute TD-IDBD weight update with adaptive step-sizes.

        Implements Algorithm 3 (semi-gradient) or Algorithm 4 (ordinary gradient)
        from Kearney et al. 2019.

        Args:
            state: Current TD-IDBD state
            td_error: TD error delta = R + gamma*V(s') - V(s)
            observation: Current observation phi(s)
            next_observation: Next observation phi(s')
            gamma: Discount factor gamma (0 at terminal)

        Returns:
            TDOptimizerUpdate with weight deltas and updated state
        """
        delta = jnp.squeeze(td_error)
        theta = state.meta_step_size
        lam = state.trace_decay
        gamma_scalar = jnp.squeeze(gamma)

        if self._use_semi_gradient:
            gradient_correlation = delta * observation * state.h_traces
            meta_delta = theta * gradient_correlation
        else:
            feature_diff = _skip_zero_scale(gamma_scalar, next_observation) - observation
            gradient_correlation = delta * feature_diff * state.h_traces
            meta_delta = -theta * gradient_correlation

        new_log_step_sizes = jnp.where(
            jnp.isfinite(meta_delta),
            jnp.clip(state.log_step_sizes + meta_delta, -10.0, 2.0),
            state.log_step_sizes,
        )
        new_alphas = jnp.exp(new_log_step_sizes)

        decay_scale = jnp.where(
            (gamma_scalar == 0.0) | (lam == 0.0),
            jnp.zeros_like(gamma_scalar),
            gamma_scalar * lam,
        )
        new_eligibility_traces = (
            _skip_zero_scale(decay_scale, state.eligibility_traces) + observation
        )
        weight_delta = new_alphas * delta * new_eligibility_traces

        if self._use_semi_gradient:
            h_decay = jnp.maximum(0.0, 1.0 - new_alphas * observation * new_eligibility_traces)
            new_h_traces = state.h_traces * h_decay + new_alphas * delta * new_eligibility_traces
        else:
            feature_diff = _skip_zero_scale(gamma_scalar, next_observation) - observation
            h_decay = jnp.maximum(0.0, 1.0 + new_alphas * new_eligibility_traces * feature_diff)
            new_h_traces = state.h_traces * h_decay + new_alphas * delta * new_eligibility_traces

        # Bias updates
        if self._use_semi_gradient:
            bias_gradient_correlation = delta * state.bias_h_trace
            bias_meta_delta = theta * bias_gradient_correlation
        else:
            bias_feature_diff = gamma_scalar - 1.0
            bias_gradient_correlation = delta * bias_feature_diff * state.bias_h_trace
            bias_meta_delta = -theta * bias_gradient_correlation

        new_bias_log_step_size = jnp.where(
            jnp.isfinite(bias_meta_delta),
            jnp.clip(state.bias_log_step_size + bias_meta_delta, -10.0, 2.0),
            state.bias_log_step_size,
        )
        new_bias_alpha = jnp.exp(new_bias_log_step_size)

        new_bias_eligibility_trace = (
            _skip_zero_scale(decay_scale, state.bias_eligibility_trace) + 1.0
        )
        bias_delta = new_bias_alpha * delta * new_bias_eligibility_trace

        if self._use_semi_gradient:
            bias_h_decay = jnp.maximum(0.0, 1.0 - new_bias_alpha * new_bias_eligibility_trace)
            new_bias_h_trace = (
                state.bias_h_trace * bias_h_decay
                + new_bias_alpha * delta * new_bias_eligibility_trace
            )
        else:
            bias_feature_diff = gamma_scalar - 1.0
            bias_h_decay = jnp.maximum(
                0.0, 1.0 + new_bias_alpha * new_bias_eligibility_trace * bias_feature_diff
            )
            new_bias_h_trace = (
                state.bias_h_trace * bias_h_decay
                + new_bias_alpha * delta * new_bias_eligibility_trace
            )

        candidate_state = TDIDBDState(
            log_step_sizes=new_log_step_sizes,
            eligibility_traces=new_eligibility_traces,
            h_traces=new_h_traces,
            meta_step_size=theta,
            trace_decay=lam,
            bias_log_step_size=new_bias_log_step_size,
            bias_eligibility_trace=new_bias_eligibility_trace,
            bias_h_trace=new_bias_h_trace,
        )
        candidate_metrics = {
            "mean_step_size": jnp.mean(new_alphas),
            "min_step_size": jnp.min(new_alphas),
            "max_step_size": jnp.max(new_alphas),
            "mean_eligibility_trace": jnp.mean(jnp.abs(new_eligibility_traces)),
        }

        inputs_valid = (
            jnp.isfinite(delta)
            & jnp.all(jnp.isfinite(observation))
            & (jnp.all(jnp.isfinite(next_observation)) | (gamma_scalar == 0.0))
            & jnp.isfinite(gamma_scalar)
        )
        previous_checked = state.replace(  # type: ignore[attr-defined]
            eligibility_traces=jnp.where(
                (gamma_scalar == 0.0) | (lam == 0.0),
                jnp.zeros_like(state.eligibility_traces),
                state.eligibility_traces,
            ),
            bias_eligibility_trace=jnp.where(
                (gamma_scalar == 0.0) | (lam == 0.0),
                jnp.zeros_like(state.bias_eligibility_trace),
                state.bias_eligibility_trace,
            ),
        )
        update_applied = (
            floating_tree_is_finite(previous_checked)
            & inputs_valid
            & jnp.all(jnp.isfinite(meta_delta))
            & jnp.isfinite(bias_meta_delta)
            & jnp.all(jnp.isfinite(weight_delta))
            & jnp.isfinite(bias_delta)
            & floating_tree_is_finite(candidate_state)
            & floating_tree_is_finite(candidate_metrics)
        )
        new_state = select_transaction(update_applied, candidate_state, state)
        metrics = neutralize_metrics(update_applied, candidate_metrics)

        return TDOptimizerUpdate(
            weight_delta=neutralize_array(update_applied, weight_delta),
            bias_delta=neutralize_array(update_applied, bias_delta),
            new_state=new_state,
            metrics=metrics,
            update_applied=update_applied,
        )


class AutoTDIDBD(TDOptimizer[AutoTDIDBDState]):
    """AutoStep-style normalized TD-IDBD optimizer.

    Adds AutoStep-style normalization to TDIDBD for improved stability and
    reduced sensitivity to the meta step-size theta.

    Reference: Kearney et al. 2019, Algorithm 6 "AutoStep Style Normalized TIDBD(lambda)"

    Attributes:
        initial_step_size: Initial per-weight step-size
        meta_step_size: Meta learning rate theta
        trace_decay: Eligibility trace decay lambda
        normalizer_decay: Decay parameter tau for normalizers
    """

    def __init__(
        self,
        initial_step_size: float = 0.01,
        meta_step_size: float = 0.01,
        trace_decay: float = 0.0,
        normalizer_decay: float = 10000.0,
    ):
        """Initialize AutoTDIDBD optimizer.

        Args:
            initial_step_size: Initial value for per-weight step-sizes
            meta_step_size: Meta learning rate theta for adapting step-sizes
            trace_decay: Eligibility trace decay lambda (0 = TD(0))
            normalizer_decay: Decay parameter tau for normalizers (default: 10000)
        """
        self._initial_step_size = initial_step_size
        self._meta_step_size = meta_step_size
        self._trace_decay = trace_decay
        self._normalizer_decay = normalizer_decay

    def init(self, feature_dim: int) -> AutoTDIDBDState:
        """Initialize AutoTDIDBD state.

        Args:
            feature_dim: Dimension of weight vector

        Returns:
            AutoTDIDBD state with per-weight step-sizes, traces, h traces, and normalizers
        """
        feature_dim = _require_int32("feature_dim", feature_dim, minimum=1)
        _require_float32_state("AutoTDIDBD state", 4 * feature_dim + 7)
        _require_float32_update_working_set(
            "AutoTDIDBD update working set", 32 * feature_dim + 8
        )
        return AutoTDIDBDState(
            log_step_sizes=jnp.full(
                feature_dim, jnp.log(self._initial_step_size), dtype=jnp.float32
            ),
            eligibility_traces=jnp.zeros(feature_dim, dtype=jnp.float32),
            h_traces=jnp.zeros(feature_dim, dtype=jnp.float32),
            normalizers=jnp.ones(feature_dim, dtype=jnp.float32),
            meta_step_size=jnp.array(self._meta_step_size, dtype=jnp.float32),
            trace_decay=jnp.array(self._trace_decay, dtype=jnp.float32),
            normalizer_decay=jnp.array(self._normalizer_decay, dtype=jnp.float32),
            bias_log_step_size=jnp.array(jnp.log(self._initial_step_size), dtype=jnp.float32),
            bias_eligibility_trace=jnp.array(0.0, dtype=jnp.float32),
            bias_h_trace=jnp.array(0.0, dtype=jnp.float32),
            bias_normalizer=jnp.array(1.0, dtype=jnp.float32),
        )

    def update(
        self,
        state: AutoTDIDBDState,
        td_error: Array,
        observation: Array,
        next_observation: Array,
        gamma: Array,
    ) -> TDOptimizerUpdate:
        """Compute AutoTDIDBD weight update with normalized adaptive step-sizes.

        Implements Algorithm 6 from Kearney et al. 2019.

        Args:
            state: Current AutoTDIDBD state
            td_error: TD error delta = R + gamma*V(s') - V(s)
            observation: Current observation phi(s)
            next_observation: Next observation phi(s')
            gamma: Discount factor gamma (0 at terminal)

        Returns:
            TDOptimizerUpdate with weight deltas and updated state
        """
        delta = jnp.squeeze(td_error)
        theta = state.meta_step_size
        lam = state.trace_decay
        tau = state.normalizer_decay
        gamma_scalar = jnp.squeeze(gamma)

        feature_diff = _skip_zero_scale(gamma_scalar, next_observation) - observation
        alphas = jnp.exp(state.log_step_sizes)

        # Update normalizers
        abs_weight_update = jnp.abs(delta * feature_diff * state.h_traces)
        normalizer_decay_term = (
            (1.0 / tau)
            * alphas
            * feature_diff
            * state.eligibility_traces
            * (jnp.abs(delta * observation * state.h_traces) - state.normalizers)
        )
        new_normalizers = jnp.maximum(abs_weight_update, state.normalizers - normalizer_decay_term)
        new_normalizers = jnp.maximum(new_normalizers, 1e-8)

        # Normalized meta-update
        normalized_gradient = delta * feature_diff * state.h_traces / new_normalizers
        new_log_step_sizes = state.log_step_sizes - theta * normalized_gradient

        # Effective step-size normalization
        effective_step_size = -jnp.sum(
            jnp.exp(new_log_step_sizes) * feature_diff * state.eligibility_traces
        )
        normalization_factor = jnp.maximum(effective_step_size, 1.0)
        new_log_step_sizes = new_log_step_sizes - jnp.log(normalization_factor)

        new_log_step_sizes = jnp.clip(new_log_step_sizes, -10.0, 2.0)
        new_alphas = jnp.exp(new_log_step_sizes)

        decay_scale = jnp.where(
            (gamma_scalar == 0.0) | (lam == 0.0),
            jnp.zeros_like(gamma_scalar),
            gamma_scalar * lam,
        )
        new_eligibility_traces = (
            _skip_zero_scale(decay_scale, state.eligibility_traces) + observation
        )
        weight_delta = new_alphas * delta * new_eligibility_traces

        # Update h traces
        h_decay = jnp.maximum(0.0, 1.0 + new_alphas * feature_diff * new_eligibility_traces)
        new_h_traces = state.h_traces * h_decay + new_alphas * delta * new_eligibility_traces

        # Bias updates
        bias_alpha = jnp.exp(state.bias_log_step_size)
        bias_feature_diff = gamma_scalar - 1.0

        abs_bias_weight_update = jnp.abs(delta * bias_feature_diff * state.bias_h_trace)
        bias_normalizer_decay_term = (
            (1.0 / tau)
            * bias_alpha
            * bias_feature_diff
            * state.bias_eligibility_trace
            * (jnp.abs(delta * state.bias_h_trace) - state.bias_normalizer)
        )
        new_bias_normalizer = jnp.maximum(
            abs_bias_weight_update, state.bias_normalizer - bias_normalizer_decay_term
        )
        new_bias_normalizer = jnp.maximum(new_bias_normalizer, 1e-8)

        normalized_bias_gradient = (
            delta * bias_feature_diff * state.bias_h_trace / new_bias_normalizer
        )
        new_bias_log_step_size = state.bias_log_step_size - theta * normalized_bias_gradient

        bias_effective_step_size = (
            -jnp.exp(new_bias_log_step_size) * bias_feature_diff * state.bias_eligibility_trace
        )
        bias_norm_factor = jnp.maximum(bias_effective_step_size, 1.0)
        new_bias_log_step_size = new_bias_log_step_size - jnp.log(bias_norm_factor)

        new_bias_log_step_size = jnp.clip(new_bias_log_step_size, -10.0, 2.0)
        new_bias_alpha = jnp.exp(new_bias_log_step_size)

        new_bias_eligibility_trace = (
            _skip_zero_scale(decay_scale, state.bias_eligibility_trace) + 1.0
        )
        bias_delta = new_bias_alpha * delta * new_bias_eligibility_trace

        bias_h_decay = jnp.maximum(
            0.0, 1.0 + new_bias_alpha * bias_feature_diff * new_bias_eligibility_trace
        )
        new_bias_h_trace = (
            state.bias_h_trace * bias_h_decay + new_bias_alpha * delta * new_bias_eligibility_trace
        )

        candidate_state = AutoTDIDBDState(
            log_step_sizes=new_log_step_sizes,
            eligibility_traces=new_eligibility_traces,
            h_traces=new_h_traces,
            normalizers=new_normalizers,
            meta_step_size=theta,
            trace_decay=lam,
            normalizer_decay=tau,
            bias_log_step_size=new_bias_log_step_size,
            bias_eligibility_trace=new_bias_eligibility_trace,
            bias_h_trace=new_bias_h_trace,
            bias_normalizer=new_bias_normalizer,
        )
        candidate_metrics = {
            "mean_step_size": jnp.mean(new_alphas),
            "min_step_size": jnp.min(new_alphas),
            "max_step_size": jnp.max(new_alphas),
            "mean_eligibility_trace": jnp.mean(jnp.abs(new_eligibility_traces)),
            "mean_normalizer": jnp.mean(new_normalizers),
        }

        inputs_valid = (
            jnp.isfinite(delta)
            & jnp.all(jnp.isfinite(observation))
            & (jnp.all(jnp.isfinite(next_observation)) | (gamma_scalar == 0.0))
            & jnp.isfinite(gamma_scalar)
        )
        previous_checked = state.replace(  # type: ignore[attr-defined]
            eligibility_traces=jnp.where(
                (gamma_scalar == 0.0) | (lam == 0.0),
                jnp.zeros_like(state.eligibility_traces),
                state.eligibility_traces,
            ),
            bias_eligibility_trace=jnp.where(
                (gamma_scalar == 0.0) | (lam == 0.0),
                jnp.zeros_like(state.bias_eligibility_trace),
                state.bias_eligibility_trace,
            ),
        )
        update_applied = (
            floating_tree_is_finite(previous_checked)
            & inputs_valid
            & jnp.all(jnp.isfinite(weight_delta))
            & jnp.isfinite(bias_delta)
            & floating_tree_is_finite(candidate_state)
            & floating_tree_is_finite(candidate_metrics)
        )
        new_state = select_transaction(update_applied, candidate_state, state)
        metrics = neutralize_metrics(update_applied, candidate_metrics)

        return TDOptimizerUpdate(
            weight_delta=neutralize_array(update_applied, weight_delta),
            bias_delta=neutralize_array(update_applied, bias_delta),
            new_state=new_state,
            metrics=metrics,
            update_applied=update_applied,
        )


# =============================================================================
# Config serialization dispatchers
# =============================================================================

_OPTIMIZER_REGISTRY: dict[str, type] = {
    "LMS": LMS,
    "IDBD": IDBD,
    "Autostep": Autostep,
    "AutostepGTDLambda": AutostepGTDLambda,
    "ObGD": ObGD,
}

_BOUNDER_REGISTRY: dict[str, type] = {
    "ObGDBounding": ObGDBounding,
    "AdaptiveObGDBounding": AdaptiveObGDBounding,
    "AGCBounding": AGCBounding,
}


def _ensure_registry_extras() -> None:
    """Register optimizers defined outside this module.

    ``baseline_optimizers`` imports from this module, so registering its
    classes at import time would create a circular import. All
    externally-defined optimizers (including ``SwiftTD``) are therefore
    added lazily on the first ``optimizer_from_config`` call.
    """
    if "SwiftTD" in _OPTIMIZER_REGISTRY:
        return
    from alberta_framework.core.baseline_optimizers import (
        NADALINE,
        AdaGain,
        Adam,
        RMSprop,
    )
    from alberta_framework.core.swift_td import SwiftTD

    _OPTIMIZER_REGISTRY.update(
        {
            "AdaGain": AdaGain,
            "Adam": Adam,
            "RMSprop": RMSprop,
            "NADALINE": NADALINE,
            "SwiftTD": SwiftTD,
        }
    )


def optimizer_from_config(config: dict[str, Any]) -> Optimizer[Any]:
    """Reconstruct an optimizer from a config dict.

    Args:
        config: Dict with ``"type"`` key and constructor kwargs

    Returns:
        Reconstructed optimizer instance

    Raises:
        ValueError: If the optimizer type is unknown
    """
    _ensure_registry_extras()
    config = dict(config)
    type_name = config.pop("type")
    host_type = _require_exact_str("type_name", type_name)
    cls = _OPTIMIZER_REGISTRY.get(host_type)
    if cls is None:
        raise ValueError(f"Unknown optimizer type: '{host_type}'")
    result: Optimizer[Any] = cls(**config)
    return result


def bounder_from_config(config: dict[str, Any]) -> Bounder:
    """Reconstruct a bounder from a config dict.

    Args:
        config: Dict with ``"type"`` key and constructor kwargs

    Returns:
        Reconstructed bounder instance

    Raises:
        ValueError: If the bounder type is unknown
    """
    config = dict(config)
    type_name = config.pop("type")
    host_type = _require_exact_str("type_name", type_name)
    cls = _BOUNDER_REGISTRY.get(host_type)
    if cls is None:
        raise ValueError(f"Unknown bounder type: '{host_type}'")
    result: Bounder = cls(**config)
    return result
