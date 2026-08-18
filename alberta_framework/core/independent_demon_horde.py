"""IndependentDemonHorde: reference Horde with one independent MLP per demon.

A Horde architecture (Sutton et al. 2011) variant where every GVF demon
owns its own independent ``MLPLearner`` (separate trunk + separate head).
Because there is no parameter sharing across demons, full per-parameter
eligibility traces with ``gamma * lamda > 0`` are forward-view-correct
for every layer (trunk and head), unlike the shared-trunk
``HordeLearner`` which must force trunk ``gamma * lamda = 0``.

Why this exists
---------------
The original Horde paper (Sutton et al. 2011) used shared linear features
(tile-coded) with separate linear value functions per demon. The shared
trunk in our ``HordeLearner`` (``MultiHeadMLPLearner`` underneath) folds
per-head error into the trunk cotangent BEFORE trace accumulation, so
trunk traces with ``gamma * lamda > 0`` would carry biased
error-gradient products across steps, violating forward-view equivalence
(Sutton & Barto Ch. 12). Forcing trunk ``gamma * lamda = 0`` sidesteps
this but constrains how much temporal credit the trunk can get.

This module provides the **reference architecture** where each demon
has its own MLP — fully independent — so we can validate empirically
how much accuracy is lost (or gained!) by sharing a nonlinear trunk.

Performance characteristics
---------------------------
This is intentionally slow. Each demon runs its own forward and
backward pass; there is no vectorization across demons (different
demons have different network architectures conceptually, even if
practically they share the constructor arguments). Use this as a
correctness reference, not a production deployment target.

Reference: Sutton et al. 2011, "Horde: A Scalable Real-time Architecture
for Learning Knowledge from Unsupervised Sensorimotor Interaction"
"""

import functools
import operator
import time
from collections.abc import Mapping
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float

from alberta_framework.core._float32_scalars import validated_float32_scalar
from alberta_framework.core.horde import HordeUpdateResult
from alberta_framework.core.initializers import sparse_init
from alberta_framework.core.learners import (
    AnyOptimizer,
    _update_from_gradient_with_diagnostics,
)
from alberta_framework.core.normalizers import Normalizer
from alberta_framework.core.optimizers import LMS, Bounder
from alberta_framework.core.types import (
    AutostepParamState,
    AutostepState,
    HordeSpec,
    IDBDParamState,
    LMSState,
    MLPLearnerState,
    MLPParams,
    TraceMode,
)
from alberta_framework.core.update_safety import (
    floating_tree_is_finite as _floating_tree_is_finite,
)

# =============================================================================
# Types
# =============================================================================


_INT32_MAX = 2**31 - 1
_ACTUAL_INT_TYPES = frozenset(
    {int, *(np.dtype(code).type for code in "bBhHiIlLqQpP")}
)
_ACTUAL_REAL_TYPES = _ACTUAL_INT_TYPES | frozenset(
    {float, *(np.dtype(code).type for code in "efdg")}
)


def _require_int32(name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


def _require_float32(
    name: str,
    value: object,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    if type(value) not in _ACTUAL_REAL_TYPES:
        raise ValueError(f"{name} must be a finite real number")
    return validated_float32_scalar(name, value, lower=lower, upper=upper)


def _read_mapping(name: str, value: object) -> dict[str, Any]:
    if not issubclass(type(value), Mapping):
        raise ValueError(f"{name} must be a mapping")
    try:
        return dict(cast(Mapping[str, Any], value))
    except Exception as error:
        raise ValueError(f"{name} must be a readable mapping") from error


def _decode_tuple(name: str, value: object) -> tuple[Any, ...]:
    if type(value) not in {list, tuple}:
        raise ValueError(f"{name} must be an exact list or tuple")
    return tuple(cast(list[Any] | tuple[Any, ...], value))


def _require_direct_state_resources(
    n_demons: int, hidden_sizes: tuple[int, ...], feature_dim: int
) -> None:
    """Bound arrays directly owned by the independent networks before allocation."""
    layer_sizes = (feature_dim, *hidden_sizes, 1)
    per_demon_parameters = sum(
        fan_out * (fan_in + 1)
        for fan_in, fan_out in zip(layer_sizes, layer_sizes[1:], strict=False)
    )
    n_layers = len(layer_sizes) - 1
    direct_state_scalars = n_demons * (2 * per_demon_parameters + 2 * n_layers + 1) + 1
    for name, value in (
        ("per_demon_parameter_count", per_demon_parameters),
        ("direct_state_scalars", direct_state_scalars),
        ("direct_state_bytes", 4 * direct_state_scalars),
    ):
        if not 1 <= value <= _INT32_MAX:
            raise ValueError(f"derived {name} must be at most {_INT32_MAX}")


def _require_loop_output_resources(num_steps: int, n_demons: int, n_seeds: int = 1) -> None:
    logical_scalars = n_seeds * num_steps * (5 * n_demons + 1)
    output_bytes = n_seeds * num_steps * (17 * n_demons + 1)
    if logical_scalars > _INT32_MAX or output_bytes > _INT32_MAX:
        raise ValueError("derived learning-loop outputs exceed the signed-int32 budget")


def _require_learning_arrays(
    observations: object,
    cumulants: object,
    next_observations: object,
    n_demons: int,
) -> tuple[Array, Array, Array, int, int]:
    obs = jnp.asarray(observations, dtype=jnp.float32)
    cums = jnp.asarray(cumulants, dtype=jnp.float32)
    next_obs = jnp.asarray(next_observations, dtype=jnp.float32)
    if obs.ndim != 2 or obs.shape[0] < 1 or obs.shape[1] < 1:
        raise ValueError("observations must have shape (num_steps, feature_dim)")
    if next_obs.shape != obs.shape:
        raise ValueError("next_observations must match observations shape")
    if cums.shape != (obs.shape[0], n_demons):
        raise ValueError("cumulants must have shape (num_steps, n_demons)")
    return obs, cums, next_obs, obs.shape[0], obs.shape[1]


@chex.dataclass(frozen=True)
class IndependentDemonHordeState:
    """State for an :class:`IndependentDemonHorde`.

    Each demon has its own ``MLPLearnerState`` stored in the
    ``demon_states`` tuple. The tuple is JAX pytree-compatible, but the
    leading dimension across demons is a Python tuple (not a JAX axis),
    so demons are processed sequentially under JIT.

    Attributes:
        demon_states: Per-demon ``MLPLearnerState`` (one entry per
            demon). Length equals ``n_demons``.
        step_count: Scalar step counter shared by all demons.
        birth_timestamp: Wall-clock seconds at construction.
        uptime_s: Cumulative wall-clock seconds spent in scan loops.
    """

    demon_states: tuple[MLPLearnerState, ...]
    step_count: Array = None  # type: ignore[assignment]
    birth_timestamp: float = 0.0
    uptime_s: float = 0.0


@chex.dataclass(frozen=True)
class IndependentDemonHordeLearningResult:
    """Result from an independent-demon Horde scan-based learning loop.

    Attributes:
        state: Final ``IndependentDemonHordeState``.
        per_demon_metrics: Per-demon metrics over time,
            shape ``(num_steps, n_demons, 3)``. Columns:
            ``[squared_error, raw_error, mean_step_size]``.
        td_errors: TD errors over time, shape ``(num_steps, n_demons)``.
    """

    state: IndependentDemonHordeState
    per_demon_metrics: Float[Array, "num_steps n_demons 3"]
    td_errors: Float[Array, "num_steps n_demons"]
    head_updates_applied: Bool[Array, "num_steps n_demons"]
    updates_applied: Bool[Array, " num_steps"]


@chex.dataclass(frozen=True)
class BatchedIndependentDemonHordeResult:
    """Result from batched independent-demon Horde learning loop.

    Attributes:
        states: Batched ``IndependentDemonHordeState``.
        per_demon_metrics: Per-demon metrics,
            shape ``(n_seeds, num_steps, n_demons, 3)``.
        td_errors: TD errors, shape ``(n_seeds, num_steps, n_demons)``.
    """

    states: IndependentDemonHordeState
    per_demon_metrics: Float[Array, "n_seeds num_steps n_demons 3"]
    td_errors: Float[Array, "n_seeds num_steps n_demons"]
    head_updates_applied: Bool[Array, "n_seeds num_steps n_demons"]
    updates_applied: Bool[Array, "n_seeds num_steps"]


# =============================================================================
# Helpers
# =============================================================================


def _extract_mean_step_size(
    opt_state: LMSState | AutostepParamState | IDBDParamState | AutostepState,
) -> Array:
    """Extract mean step-size from an optimizer state (mirrors multi_head)."""
    if hasattr(opt_state, "step_sizes"):
        return jnp.mean(opt_state.step_sizes)
    if hasattr(opt_state, "log_step_sizes"):
        return jnp.mean(jnp.exp(opt_state.log_step_sizes))
    if hasattr(opt_state, "step_size"):
        return opt_state.step_size
    return jnp.array(0.0, dtype=jnp.float32)


def _forward_mlp(
    weights: tuple[Array, ...],
    biases: tuple[Array, ...],
    observation: Array,
    leaky_relu_slope: float,
    use_layer_norm: bool,
) -> Array:
    """Pure forward pass through an MLP, returning a scalar prediction.

    Mirrors ``MLPLearner._forward`` so it stays consistent with the
    architecture that ``MLPLearner`` would build for the same config.
    """
    x = observation
    n_layers = len(weights)
    for i in range(n_layers - 1):
        x = weights[i] @ x + biases[i]
        if use_layer_norm:
            mean = jnp.mean(x)
            var = jnp.var(x)
            x = (x - mean) / jnp.sqrt(var + 1e-5)
        x = jnp.where(x >= 0, x, leaky_relu_slope * x)
    # Output layer (no activation)
    x = weights[-1] @ x + biases[-1]
    return jnp.squeeze(x)


# =============================================================================
# IndependentDemonHorde
# =============================================================================


class IndependentDemonHorde:
    """Reference Horde where every demon has its own independent MLP.

    Constructor signature mirrors :class:`HordeLearner` so that experiments
    can swap implementations by name. Each demon receives a separate copy
    of the architecture and optimizer configuration; the demon's own
    ``gamma`` / ``lamda`` from the ``GVFSpec`` drive its eligibility traces.

    Because there is no shared trunk, full per-parameter eligibility
    traces (trunk + head) with ``gamma * lamda > 0`` are correct for
    every demon, with no trace-error coupling pathology.

    Single-Step (Daemon) Usage
    --------------------------
    Both :meth:`predict` and :meth:`update` accept an unbatched 1D
    observation and are JIT-compiled automatically.

    Attributes:
        horde_spec: The ``HordeSpec`` defining all demons.
        n_demons: Number of demons.
    """

    def __init__(
        self,
        horde_spec: HordeSpec,
        hidden_sizes: tuple[int, ...] = (128, 128),
        optimizer: AnyOptimizer | None = None,
        step_size: float = 1.0,
        bounder: Bounder | None = None,
        normalizer: Normalizer[Any] | None = None,
        sparsity: float = 0.9,
        leaky_relu_slope: float = 0.01,
        use_layer_norm: bool = True,
        head_optimizer: AnyOptimizer | None = None,
        trace_mode: TraceMode = TraceMode.ACCUMULATING,
    ):
        """Initialize the independent-demon Horde.

        Args:
            horde_spec: Specification of all GVF demons.
            hidden_sizes: Tuple of hidden layer sizes (default: two layers
                of 128). Pass ``()`` for a per-demon linear baseline.
            optimizer: Optimizer for weight updates. Defaults to
                ``LMS(step_size)``.
            step_size: Base learning rate (used only when ``optimizer`` is
                ``None``).
            bounder: Optional update bounder (e.g. ``ObGDBounding``).
            normalizer: Optional feature normalizer (independent per demon).
            sparsity: Fraction of weights zeroed out per neuron.
            leaky_relu_slope: Negative slope for LeakyReLU.
            use_layer_norm: Whether to apply parameterless layer
                normalization between hidden layers.
            head_optimizer: Optional separate optimizer for each demon's
                output (head) layer.
            trace_mode: Eligibility trace mode (``ACCUMULATING`` or
                ``REPLACING``). Applies independently inside every
                demon's network.
        """
        if type(horde_spec) is not HordeSpec or not horde_spec.demons:
            raise ValueError("horde_spec must be a nonempty HordeSpec")
        if type(hidden_sizes) is not tuple:
            raise ValueError("hidden_sizes must be an exact tuple")
        hidden_sizes = tuple(
            _require_int32(f"hidden_sizes[{i}]", v, minimum=1) for i, v in enumerate(hidden_sizes)
        )
        step_size = _require_float32("step_size", step_size, lower=0.0)
        sparsity = _require_float32("sparsity", sparsity, lower=0.0, upper=1.0)
        leaky_relu_slope = _require_float32(
            "leaky_relu_slope", leaky_relu_slope, lower=0.0
        )
        if type(use_layer_norm) is not bool:
            raise ValueError("use_layer_norm must be an exact bool")
        if type(trace_mode) is not TraceMode:
            raise ValueError("trace_mode must be a TraceMode")
        self._horde_spec = horde_spec
        self._hidden_sizes = hidden_sizes
        self._optimizer: AnyOptimizer = (
            optimizer if optimizer is not None else LMS(step_size=step_size)
        )
        self._head_optimizer: AnyOptimizer | None = head_optimizer
        self._step_size = step_size
        self._bounder = bounder
        self._normalizer = normalizer
        self._sparsity = sparsity
        self._leaky_relu_slope = leaky_relu_slope
        self._use_layer_norm = use_layer_norm
        self._trace_mode = trace_mode

    @property
    def horde_spec(self) -> HordeSpec:
        """The HordeSpec defining all demons."""
        return self._horde_spec

    @property
    def n_demons(self) -> int:
        """Number of demons."""
        return len(self._horde_spec.demons)

    # -------------------------------------------------------------------------
    # Config serialization
    # -------------------------------------------------------------------------

    def to_config(self) -> dict[str, Any]:
        """Serialize learner configuration to a dict.

        Returns:
            Dict containing every constructor argument needed to recreate
            this learner via :meth:`from_config`.
        """
        return {
            "type": "IndependentDemonHorde",
            "horde_spec": self._horde_spec.to_config(),
            "hidden_sizes": list(self._hidden_sizes),
            "optimizer": self._optimizer.to_config(),
            "bounder": (self._bounder.to_config() if self._bounder is not None else None),
            "normalizer": (self._normalizer.to_config() if self._normalizer is not None else None),
            "head_optimizer": (
                self._head_optimizer.to_config() if self._head_optimizer is not None else None
            ),
            "sparsity": self._sparsity,
            "leaky_relu_slope": self._leaky_relu_slope,
            "use_layer_norm": self._use_layer_norm,
            "trace_mode": self._trace_mode.value,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "IndependentDemonHorde":
        """Reconstruct learner from a config dict.

        Args:
            config: Dict as produced by :meth:`to_config`.

        Returns:
            Reconstructed ``IndependentDemonHorde``.
        """
        from alberta_framework.core.normalizers import normalizer_from_config
        from alberta_framework.core.optimizers import (
            bounder_from_config,
            optimizer_from_config,
        )

        config = _read_mapping("config", config)
        config.pop("type", None)

        horde_spec = HordeSpec.from_config(config.pop("horde_spec"))
        optimizer = optimizer_from_config(config.pop("optimizer"))
        bounder_cfg = config.pop("bounder", None)
        bounder = bounder_from_config(bounder_cfg) if bounder_cfg is not None else None
        normalizer_cfg = config.pop("normalizer", None)
        normalizer = normalizer_from_config(normalizer_cfg) if normalizer_cfg is not None else None
        head_opt_cfg = config.pop("head_optimizer", None)
        head_optimizer = optimizer_from_config(head_opt_cfg) if head_opt_cfg is not None else None

        trace_mode_str = config.pop("trace_mode", None)
        if trace_mode_str is not None and type(trace_mode_str) is not str:
            raise ValueError("trace_mode must be an exact string")
        try:
            trace_mode = (
                TraceMode(trace_mode_str)
                if trace_mode_str is not None
                else TraceMode.ACCUMULATING
            )
        except (TypeError, ValueError) as error:
            raise ValueError("trace_mode must name a supported mode") from error

        return cls(
            horde_spec=horde_spec,
            hidden_sizes=_decode_tuple("hidden_sizes", config.pop("hidden_sizes")),
            optimizer=optimizer,
            bounder=bounder,
            normalizer=normalizer,
            head_optimizer=head_optimizer,
            trace_mode=trace_mode,
            **config,
        )

    # -------------------------------------------------------------------------
    # Init
    # -------------------------------------------------------------------------

    def _init_single_demon(self, feature_dim: int, key: Array) -> MLPLearnerState:
        """Initialize one demon's ``MLPLearnerState``.

        Equivalent to the body of ``MLPLearner.init`` so that the same
        sparse-init / shape conventions apply.
        """
        layer_sizes = [feature_dim, *self._hidden_sizes, 1]

        weights_list: list[Array] = []
        biases_list: list[Array] = []
        traces_list: list[Array] = []
        opt_states_list: list[Any] = []

        n_total_layers = len(layer_sizes) - 1
        for i in range(n_total_layers):
            fan_out = layer_sizes[i + 1]
            fan_in = layer_sizes[i]
            key, subkey = jax.random.split(key)
            w = sparse_init(subkey, (fan_out, fan_in), sparsity=self._sparsity)
            b = jnp.zeros(fan_out, dtype=jnp.float32)
            weights_list.append(w)
            biases_list.append(b)
            traces_list.append(jnp.zeros_like(w))
            traces_list.append(jnp.zeros_like(b))
            is_output = i == n_total_layers - 1
            opt = (
                self._head_optimizer
                if (self._head_optimizer is not None and is_output)
                else self._optimizer
            )
            opt_states_list.append(opt.init_for_shape(w.shape))
            opt_states_list.append(opt.init_for_shape(b.shape))

        params = MLPParams(
            weights=tuple(weights_list),
            biases=tuple(biases_list),
        )

        normalizer_state = None
        if self._normalizer is not None:
            normalizer_state = self._normalizer.init(feature_dim)

        return MLPLearnerState(
            params=params,
            optimizer_states=tuple(opt_states_list),
            traces=tuple(traces_list),
            normalizer_state=normalizer_state,
            step_count=jnp.array(0, dtype=jnp.int32),
            birth_timestamp=time.time(),
            uptime_s=0.0,
        )

    def init(self, feature_dim: int, key: Array) -> IndependentDemonHordeState:
        """Initialize an :class:`IndependentDemonHordeState`.

        Args:
            feature_dim: Dimension of the input feature vector.
            key: JAX random key used to seed every demon's weight init
                (each demon gets its own split key).

        Returns:
            Initial state with one ``MLPLearnerState`` per demon.
        """
        feature_dim = _require_int32("feature_dim", feature_dim, minimum=1)
        _require_direct_state_resources(self.n_demons, self._hidden_sizes, feature_dim)
        keys = jax.random.split(key, self.n_demons)
        demon_states = tuple(self._init_single_demon(feature_dim, k) for k in keys)
        return IndependentDemonHordeState(
            demon_states=demon_states,
            step_count=jnp.array(0, dtype=jnp.int32),
            birth_timestamp=time.time(),
            uptime_s=0.0,
        )

    def _feature_dim_from_state(self, state: IndependentDemonHordeState) -> int:
        """Validate the static learner-owned state contract and return its input width."""
        if type(state) is not IndependentDemonHordeState:
            raise ValueError("state must be an IndependentDemonHordeState")
        if type(state.demon_states) is not tuple or len(state.demon_states) != self.n_demons:
            raise ValueError("state.demon_states must match the configured demons")
        if state.step_count.shape != () or state.step_count.dtype != jnp.int32:
            raise ValueError("state.step_count must be a scalar int32")
        feature_dim: int | None = None
        for demon_state in state.demon_states:
            if (
                type(demon_state) is not MLPLearnerState
                or type(demon_state.params) is not MLPParams
            ):
                raise ValueError("every demon state must be an MLPLearnerState")
            weights = demon_state.params.weights
            biases = demon_state.params.biases
            expected_layers = len(self._hidden_sizes) + 1
            if (
                type(weights) is not tuple
                or type(biases) is not tuple
                or len(weights) != expected_layers
                or len(biases) != expected_layers
            ):
                raise ValueError("demon parameter topology must match hidden_sizes")
            current_feature_dim = weights[0].shape[1] if weights[0].ndim == 2 else -1
            if feature_dim is None:
                feature_dim = current_feature_dim
            if current_feature_dim != feature_dim:
                raise ValueError("all demon states must share one feature dimension")
            layer_sizes = (current_feature_dim, *self._hidden_sizes, 1)
            parameter_shapes: list[tuple[int, ...]] = []
            for index, (fan_in, fan_out) in enumerate(
                zip(layer_sizes, layer_sizes[1:], strict=False)
            ):
                if weights[index].shape != (fan_out, fan_in) or weights[index].dtype != jnp.float32:
                    raise ValueError("demon weight shapes and dtypes must match the configuration")
                if biases[index].shape != (fan_out,) or biases[index].dtype != jnp.float32:
                    raise ValueError("demon bias shapes and dtypes must match the configuration")
                parameter_shapes.extend(((fan_out, fan_in), (fan_out,)))
            if type(demon_state.traces) is not tuple or tuple(
                trace.shape for trace in demon_state.traces
            ) != tuple(parameter_shapes):
                raise ValueError("demon traces must match every parameter shape")
            if any(trace.dtype != jnp.float32 for trace in demon_state.traces):
                raise ValueError("demon traces must use float32")
            if demon_state.step_count.shape != () or demon_state.step_count.dtype != jnp.int32:
                raise ValueError("demon step_count must be a scalar int32")
        if feature_dim is None or feature_dim < 1:
            raise ValueError("state feature dimension must be positive")
        return feature_dim

    @staticmethod
    def _operand(name: str, value: object, shape: tuple[int, ...]) -> Array:
        array = jnp.asarray(value, dtype=jnp.float32)
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        return array

    # -------------------------------------------------------------------------
    # Per-demon predict / update (pure functions)
    # -------------------------------------------------------------------------

    def _predict_single(self, demon_state: MLPLearnerState, observation: Array) -> Array:
        """Predict value for a single demon given its state and observation."""
        obs = observation
        if self._normalizer is not None and demon_state.normalizer_state is not None:
            obs = self._normalizer.normalize_only(demon_state.normalizer_state, observation)
        return _forward_mlp(
            demon_state.params.weights,
            demon_state.params.biases,
            obs,
            self._leaky_relu_slope,
            self._use_layer_norm,
        )

    def _update_single(
        self,
        demon_state: MLPLearnerState,
        observation: Array,
        target: Array,
        gamma_lamda: Array,
        active: Array,
    ) -> tuple[MLPLearnerState, Array, Array, Array, Array]:
        """Run a single demon's update step.

        Returns ``(new_state, prediction, error, mean_step_size)``. When
        ``active`` is ``False`` the demon's parameters, traces, and
        optimizer states are preserved unchanged; ``error`` and
        ``mean_step_size`` are returned as NaN to flag inactivity.
        Non-finite proposed weights (ObGD ``0 * inf``) are also held.

        This mirrors the body of ``MLPLearner.update`` but: (a) accepts
        an external active mask so NaN cumulants suppress the update,
        (b) supports the configured ``trace_mode``, and (c) accepts
        an externally-supplied ``gamma_lamda`` (per-demon).
        """
        replacing = self._trace_mode == TraceMode.REPLACING

        # 1. Normalize observation if needed (and update normalizer state)
        obs = observation
        new_normalizer_state = demon_state.normalizer_state
        normalizer_update_applied = jnp.asarray(True, dtype=jnp.bool_)
        if self._normalizer is not None and demon_state.normalizer_state is not None:
            normalizer_result = self._normalizer.normalize_with_diagnostics(
                demon_state.normalizer_state, observation
            )
            obs = normalizer_result.normalized
            new_normalizer_state = normalizer_result.state
            normalizer_update_applied = normalizer_result.update_applied

        # 2. Forward + prediction-gradient via jax.grad
        slope = self._leaky_relu_slope
        ln = self._use_layer_norm

        prediction_val = _forward_mlp(
            demon_state.params.weights,
            demon_state.params.biases,
            obs,
            slope,
            ln,
        )
        error = jnp.where(active, target - prediction_val, 0.0)

        def pred_fn(weights: tuple[Array, ...], biases: tuple[Array, ...]) -> Array:
            return _forward_mlp(weights, biases, obs, slope, ln)

        weight_grads, bias_grads = jax.grad(pred_fn, argnums=(0, 1))(
            demon_state.params.weights, demon_state.params.biases
        )

        # 3. Trace update (per-demon gamma * lamda)
        n_layers = len(demon_state.params.weights)
        new_traces: list[Array] = []
        for i in range(n_layers):
            old_wt = demon_state.traces[2 * i]
            grad_w = weight_grads[i]
            if replacing:
                new_wt = jnp.where(
                    grad_w != 0.0,
                    grad_w,
                    jnp.where(
                        gamma_lamda == 0.0,
                        jnp.zeros_like(old_wt),
                        gamma_lamda * old_wt,
                    ),
                )
            else:
                carried = jnp.where(
                    gamma_lamda == 0.0, jnp.zeros_like(old_wt), gamma_lamda * old_wt
                )
                new_wt = carried + grad_w
            new_traces.append(new_wt)

            old_bt = demon_state.traces[2 * i + 1]
            grad_b = bias_grads[i]
            if replacing:
                new_bt = jnp.where(
                    grad_b != 0.0,
                    grad_b,
                    jnp.where(
                        gamma_lamda == 0.0,
                        jnp.zeros_like(old_bt),
                        gamma_lamda * old_bt,
                    ),
                )
            else:
                carried_b = jnp.where(
                    gamma_lamda == 0.0, jnp.zeros_like(old_bt), gamma_lamda * old_bt
                )
                new_bt = carried_b + grad_b
            new_traces.append(new_bt)

        # 4. Per-parameter optimizer step from traces
        n_trace_entries = len(new_traces)
        all_steps: list[Array] = []
        new_opt_states: list[Any] = []
        optimizer_updates_applied: list[Array] = []
        for j in range(n_trace_entries):
            is_output = self._head_optimizer is not None and j >= n_trace_entries - 2
            opt: AnyOptimizer = (
                self._head_optimizer
                if (is_output and self._head_optimizer is not None)
                else self._optimizer
            )
            step, new_opt, optimizer_update_applied = _update_from_gradient_with_diagnostics(
                opt,
                demon_state.optimizer_states[j],
                new_traces[j],
                error=error,
            )
            all_steps.append(step)
            new_opt_states.append(new_opt)
            optimizer_updates_applied.append(optimizer_update_applied)

        # 5. Optional bounding (per-demon)
        if self._bounder is not None:
            all_params: list[Array] = []
            for i in range(n_layers):
                all_params.append(demon_state.params.weights[i])
                all_params.append(demon_state.params.biases[i])
            bounded_steps, bound_scale = self._bounder.bound(
                tuple(all_steps), error, tuple(all_params)
            )
            all_steps = list(bounded_steps)
            # Scale traces so future updates reflect the effective step
            new_traces = [bound_scale * t for t in new_traces]

        # 6. Apply: param += error * step
        new_weights: list[Array] = []
        new_biases: list[Array] = []
        for i in range(n_layers):
            new_weights.append(demon_state.params.weights[i] + error * all_steps[2 * i])
            new_biases.append(demon_state.params.biases[i] + error * all_steps[2 * i + 1])

        new_params = MLPParams(
            weights=tuple(new_weights),
            biases=tuple(new_biases),
        )

        # Inf error zeros the ObGD step, then error * step is 0*inf=NaN.
        # Treat that the same as an inactive demon: keep previous finite
        # params, traces, and optimizer states.
        previous_checked = demon_state.replace(  # type: ignore[attr-defined]
            traces=tuple(
                jnp.where(gamma_lamda == 0.0, jnp.zeros_like(trace), trace)
                for trace in demon_state.traces
            )
        )
        proposed_finite = (
            _floating_tree_is_finite(new_params)
            & _floating_tree_is_finite(tuple(new_traces))
            & _floating_tree_is_finite(tuple(new_opt_states))
        )
        commit = (
            active
            & jnp.all(jnp.isfinite(observation))
            & jnp.isfinite(error)
            & _floating_tree_is_finite(previous_checked)
            & proposed_finite
            & normalizer_update_applied
            & jnp.all(jnp.stack(optimizer_updates_applied))
        )

        # 7. Mask inactive / non-finite updates: keep old params/traces/opt
        new_params = MLPParams(
            weights=tuple(
                jnp.where(commit, new_w, old_w)
                for new_w, old_w in zip(new_params.weights, demon_state.params.weights, strict=True)
            ),
            biases=tuple(
                jnp.where(commit, new_b, old_b)
                for new_b, old_b in zip(new_params.biases, demon_state.params.biases, strict=True)
            ),
        )
        masked_traces = tuple(
            jnp.where(commit, new_t, old_t)
            for new_t, old_t in zip(new_traces, demon_state.traces, strict=True)
        )
        masked_opt_states = tuple(
            jax.tree.map(
                lambda new, old: jnp.where(commit, new, old),
                new_opt,
                old_opt,
            )
            for new_opt, old_opt in zip(new_opt_states, demon_state.optimizer_states, strict=True)
        )
        if (
            self._normalizer is not None
            and demon_state.normalizer_state is not None
            and new_normalizer_state is not None
        ):
            masked_normalizer_state = jax.tree.map(
                lambda new, old: jnp.where(commit, new, old),
                new_normalizer_state,
                demon_state.normalizer_state,
            )
        else:
            masked_normalizer_state = new_normalizer_state

        new_state = MLPLearnerState(
            params=new_params,
            optimizer_states=masked_opt_states,
            traces=masked_traces,
            normalizer_state=masked_normalizer_state,
            step_count=jnp.where(
                commit,
                jnp.minimum(demon_state.step_count, _INT32_MAX - 1) + 1,
                demon_state.step_count,
            ),
            birth_timestamp=demon_state.birth_timestamp,
            uptime_s=demon_state.uptime_s,
        )

        # Mean step-size for the first weight optimizer state (matches
        # multi_head_learner._extract_mean_step_size convention).
        mean_ss = _extract_mean_step_size(new_opt_states[0])
        # NaN out reporting metrics for inactive demons (matches
        # MultiHeadMLPLearner's NaN convention).
        reported_error = jnp.where(
            commit,
            target - prediction_val,
            jnp.where(active, 0.0, jnp.nan),
        )
        reported_mean_ss = jnp.where(
            commit,
            mean_ss,
            jnp.where(active, 0.0, jnp.nan),
        )
        reported_prediction = jnp.where(
            jnp.isfinite(prediction_val), prediction_val, jnp.zeros_like(prediction_val)
        )
        return new_state, reported_prediction, reported_error, reported_mean_ss, commit

    # -------------------------------------------------------------------------
    # Public predict / update
    # -------------------------------------------------------------------------

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(self, state: IndependentDemonHordeState, observation: Array) -> Array:
        """Compute predictions from all demons.

        Args:
            state: Current learner state.
            observation: Input feature vector, shape ``(feature_dim,)``.

        Returns:
            Array of shape ``(n_demons,)`` with one prediction per demon.
        """
        feature_dim = self._feature_dim_from_state(state)
        observation = self._operand("observation", observation, (feature_dim,))
        preds = [
            self._predict_single(state.demon_states[i], observation) for i in range(self.n_demons)
        ]
        return jnp.stack(preds)

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: IndependentDemonHordeState,
        observation: Array,
        cumulants: Array,
        next_observation: Array,
    ) -> HordeUpdateResult:
        """Update every demon's network given observation, cumulants, next obs.

        Computes per-demon TD targets ``r + gamma * V(s')`` using each
        demon's own gamma, then drives every demon's independent MLP with
        that target. NaN cumulants suppress the update on the affected
        demon (its params, traces, and optimizer states are preserved).

        Args:
            state: Current state.
            observation: Input feature vector, shape ``(feature_dim,)``.
            cumulants: Per-demon pseudo-rewards, shape ``(n_demons,)``.
                NaN entries flag inactive demons.
            next_observation: Next feature vector, shape
                ``(feature_dim,)``. Each demon evaluates its own
                ``V(s')`` with its own network.

        Returns:
            ``HordeUpdateResult`` (the same dataclass used by
            :class:`HordeLearner`) but with ``state`` set to the new
            ``IndependentDemonHordeState``.
        """
        feature_dim = self._feature_dim_from_state(state)
        n_demons = self.n_demons
        observation = self._operand("observation", observation, (feature_dim,))
        cumulants = self._operand("cumulants", cumulants, (n_demons,))
        next_observation = self._operand(
            "next_observation", next_observation, (feature_dim,)
        )
        gammas = self._horde_spec.gammas
        lamdas = self._horde_spec.lamdas

        # 1. Per-demon V(s') for bootstrapping (each demon uses its OWN net)
        next_preds = jnp.stack(
            [self._predict_single(state.demon_states[i], next_observation) for i in range(n_demons)]
        )

        # 2. TD targets: r + gamma * V(s'). gamma=0 must not multiply inf V(s').
        bootstrap = jnp.where(gammas == 0.0, 0.0, gammas * next_preds)
        targets = cumulants + bootstrap

        # 3. NaN means inactive; other non-finite values are rejected heads.
        requested_mask = ~jnp.isnan(cumulants)
        checked_demon_states = tuple(
            demon_state.replace(  # type: ignore[attr-defined]
                traces=tuple(
                    jnp.where(
                        (gammas[i] == 0.0) | (lamdas[i] == 0.0),
                        jnp.zeros_like(trace),
                        trace,
                    )
                    for trace in demon_state.traces
                )
            )
            for i, demon_state in enumerate(state.demon_states)
        )
        global_inputs_valid = (
            jnp.all(jnp.isfinite(observation))
            & jnp.all(jnp.isfinite(next_observation))
            & (state.step_count >= 0)
            & jnp.all(
                jnp.stack([demon_state.step_count >= 0 for demon_state in state.demon_states])
            )
            & _floating_tree_is_finite(
                state.replace(  # type: ignore[attr-defined]
                    demon_states=checked_demon_states
                )
            )
        )
        active_mask = (
            requested_mask & jnp.isfinite(cumulants) & jnp.isfinite(targets) & global_inputs_valid
        )
        safe_targets = jnp.where(active_mask, targets, 0.0)

        # 4. Per-demon update
        new_demon_states: list[MLPLearnerState] = []
        predictions_list: list[Array] = []
        errors_list: list[Array] = []
        mean_ss_list: list[Array] = []
        commits_list: list[Array] = []
        for i in range(n_demons):
            gamma_lamda_i = gammas[i] * lamdas[i]
            new_ds, pred_i, err_i, mss_i, commit_i = self._update_single(
                state.demon_states[i],
                observation,
                safe_targets[i],
                gamma_lamda_i,
                active_mask[i],
            )
            new_demon_states.append(new_ds)
            predictions_list.append(pred_i)
            errors_list.append(err_i)
            mean_ss_list.append(mss_i)
            commits_list.append(commit_i)

        predictions = jnp.stack(predictions_list)
        td_errors = jnp.stack(errors_list)
        head_updates_applied = jnp.stack(commits_list)
        # Per-demon metrics: [squared_error, raw_error, mean_step_size]
        # NaN columns for inactive demons (matches HordeUpdateResult convention).
        squared_errors = jnp.where(
            head_updates_applied,
            td_errors**2,
            jnp.where(requested_mask, 0.0, jnp.nan),
        )
        # NaN-out td_targets for inactive demons to match HordeLearner.
        reported_targets = jnp.where(
            head_updates_applied,
            targets,
            jnp.where(requested_mask, jnp.zeros_like(targets), jnp.nan),
        )

        update_applied = global_inputs_valid & (
            jnp.any(head_updates_applied) | jnp.all(~requested_mask)
        )
        proposed_state = IndependentDemonHordeState(
            demon_states=tuple(new_demon_states),
            step_count=jnp.minimum(state.step_count, _INT32_MAX - 1) + 1,
            birth_timestamp=state.birth_timestamp,
            uptime_s=state.uptime_s,
        )
        new_state = jax.lax.cond(
            update_applied,
            lambda: proposed_state,
            lambda: state,
        )
        reported_errors = jnp.where(
            head_updates_applied,
            td_errors,
            jnp.where(requested_mask, jnp.zeros_like(td_errors), jnp.nan),
        )
        reported_metrics = jnp.stack(
            (
                squared_errors,
                reported_errors,
                jnp.where(
                    head_updates_applied,
                    jnp.stack(mean_ss_list),
                    jnp.where(requested_mask, 0.0, jnp.nan),
                ),
            ),
            axis=1,
        )

        return HordeUpdateResult(
            state=new_state,
            predictions=jnp.where(
                update_applied,
                jnp.where(
                    requested_mask & ~head_updates_applied,
                    jnp.zeros_like(predictions),
                    predictions,
                ),
                jnp.zeros_like(predictions),
            ),
            td_errors=reported_errors,
            td_targets=reported_targets,
            per_demon_metrics=reported_metrics,
            # No shared trunk -> no scalar trunk bounding metric.
            trunk_bounding_metric=jnp.where(
                update_applied,
                jnp.array(1.0, dtype=jnp.float32),
                jnp.array(0.0, dtype=jnp.float32),
            ),
            head_updates_applied=head_updates_applied,
            update_applied=update_applied,
        )


# =============================================================================
# Learning Loops
# =============================================================================


def run_independent_horde_learning_loop(
    horde: IndependentDemonHorde,
    state: IndependentDemonHordeState,
    observations: Array,
    cumulants: Array,
    next_observations: Array,
) -> IndependentDemonHordeLearningResult:
    """Run an independent-demon Horde learning loop using ``jax.lax.scan``.

    Args:
        horde: Independent-demon Horde learner.
        state: Initial learner state.
        observations: Input observations,
            shape ``(num_steps, feature_dim)``.
        cumulants: Per-demon cumulants,
            shape ``(num_steps, n_demons)``. NaN entries flag inactive
            demons for that step.
        next_observations: Next observations,
            shape ``(num_steps, feature_dim)``.

    Returns:
        ``IndependentDemonHordeLearningResult`` with final state,
        per-demon metrics, and TD errors over time.
    """

    observations, cumulants, next_observations, num_steps, feature_dim = (
        _require_learning_arrays(observations, cumulants, next_observations, horde.n_demons)
    )
    if horde._feature_dim_from_state(state) != feature_dim:
        raise ValueError("state feature dimension must match observations")
    _require_loop_output_resources(num_steps, horde.n_demons)

    def step_fn(
        carry: IndependentDemonHordeState,
        inputs: tuple[Array, Array, Array],
    ) -> tuple[IndependentDemonHordeState, tuple[Array, Array, Array, Array]]:
        l_state = carry
        obs, cums, next_obs = inputs
        result = horde.update(l_state, obs, cums, next_obs)
        return result.state, (
            result.per_demon_metrics,
            result.td_errors,
            result.head_updates_applied,
            result.update_applied,
        )

    t0 = time.time()
    (
        final_state,
        (
            per_demon_metrics,
            td_errors,
            head_updates_applied,
            updates_applied,
        ),
    ) = jax.lax.scan(step_fn, state, (observations, cumulants, next_observations))
    elapsed = time.time() - t0
    final_state = final_state.replace(  # type: ignore[attr-defined]
        uptime_s=final_state.uptime_s + elapsed
    )

    return IndependentDemonHordeLearningResult(  # type: ignore[call-arg]
        state=final_state,
        per_demon_metrics=per_demon_metrics,
        td_errors=td_errors,
        head_updates_applied=head_updates_applied,
        updates_applied=updates_applied,
    )


def run_independent_horde_learning_loop_batched(
    horde: IndependentDemonHorde,
    observations: Array,
    cumulants: Array,
    next_observations: Array,
    keys: Array,
) -> BatchedIndependentDemonHordeResult:
    """Run an independent-demon Horde learning loop across seeds via ``vmap``.

    Args:
        horde: Independent-demon Horde learner.
        observations: Shared observations,
            shape ``(num_steps, feature_dim)``.
        cumulants: Shared cumulants, shape ``(num_steps, n_demons)``.
        next_observations: Shared next observations,
            shape ``(num_steps, feature_dim)``.
        keys: JAX random keys, shape ``(n_seeds,)`` or ``(n_seeds, 2)``.

    Returns:
        ``BatchedIndependentDemonHordeResult`` with batched states,
        per-demon metrics, and TD errors.
    """
    observations, cumulants, next_observations, num_steps, feature_dim = (
        _require_learning_arrays(observations, cumulants, next_observations, horde.n_demons)
    )
    keys = jnp.asarray(keys)
    if (
        keys.ndim not in {1, 2}
        or keys.shape[0] < 1
        or (keys.ndim == 2 and keys.shape[1] != 2)
    ):
        raise ValueError("keys must contain at least one JAX key")
    n_seeds = keys.shape[0]
    _require_loop_output_resources(num_steps, horde.n_demons, n_seeds)

    def single_run(
        key: Array,
    ) -> tuple[IndependentDemonHordeState, Array, Array, Array, Array]:
        init_state = horde.init(feature_dim, key)
        result = run_independent_horde_learning_loop(
            horde, init_state, observations, cumulants, next_observations
        )
        return (
            result.state,
            result.per_demon_metrics,
            result.td_errors,
            result.head_updates_applied,
            result.updates_applied,
        )

    t0 = time.time()
    (
        batched_states,
        batched_metrics,
        batched_td_errors,
        batched_head_updates_applied,
        batched_updates_applied,
    ) = jax.vmap(single_run)(keys)
    elapsed = time.time() - t0
    batched_states = batched_states.replace(  # type: ignore[attr-defined]
        uptime_s=batched_states.uptime_s + elapsed
    )

    return BatchedIndependentDemonHordeResult(  # type: ignore[call-arg]
        states=batched_states,
        per_demon_metrics=batched_metrics,
        td_errors=batched_td_errors,
        head_updates_applied=batched_head_updates_applied,
        updates_applied=batched_updates_applied,
    )
