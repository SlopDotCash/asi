"""Streaming actor-critic with RTU-RTRL recurrent trace units.

This module is an Alberta-native, JAX-only implementation of the discrete
stream AC(lambda) construction described by Farr et al.,
`arXiv:2605.24709v2 <https://arxiv.org/abs/2605.24709>`_.  The RTU equations and
ObGD ordering were cross-checked against the Apache-2.0 Memorax implementation
at commit ``94dd21dba5ab6eee4a0281066a5f49643d052174`` (in particular
``networks/sequence_models/rtu.py`` and ``algorithms/stream_ac.py``).  The code
here has no runtime dependency on Memorax, Flax, or Optax.

The recurrent state uses a diagonal complex transition represented by paired
real arrays.  With fixed recurrent parameters, its compressed RTRL sensitivity
is exact: cross-unit recurrent derivatives are structurally zero, so the
retained tensors contain every non-zero derivative while requiring linear
memory in the RTU parameter count.
An explicit reverse/forward chain-rule contraction turns those sensitivities
into gradients of the policy or value head.  The sparse encoder receives the
paper's explicit one-step approximation rather than an inaccurately labelled
history trace.  The RTU input matrices themselves use the dense LeCun-normal
initialization of the audited reference.  These per-step gradients then enter
ordinary streaming eligibility traces; the TD error is applied exactly once,
by the final ObGD update.

This is a research core, not a claim of state-of-the-art performance.  Online
parameter changes make retained RTRL sensitivities stale in the precise sense
discussed by Farr et al.  The default path retains the ordinary stale trace;
an opt-in path implements the paper's parameter-wise diagonal, first-order
Taylor correction.  That path is a Taylor-corrected approximate sensitivity,
not exact RTRL under moving parameters.  Exactness tests therefore freeze
parameters while unrolling a comparison sequence.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import operator
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple, Self, SupportsIndex, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.nn.initializers import lecun_normal

from alberta_framework.core._float32_scalars import validated_float32_scalar_with_ratio
from alberta_framework.core.initializers import sparse_init
from alberta_framework.core.update_safety import (
    floating_tree_is_finite as _floating_tree_is_finite,
)

_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)
_MINIMUM_EXP_LOG = math.log(_FLOAT32_TINY)
_MAXIMUM_NU_LOG = math.log(-math.log(_FLOAT32_TINY))
_MAXIMUM_THETA_LOG = math.log(2.0 * math.pi)
_INT32_MAX = 2**31 - 1


def _validate_normal_float32_config_value(name: str, value: float) -> None:
    """Reject nonzero configuration values that are not normal float32 values.

    The complete actor-critic state and all runtime scalar arithmetic use
    float32.  Python-finite values outside this interval otherwise become
    infinity, zero, or backend-dependent subnormals only after validation.
    Exact zero remains available to fields whose later domain checks allow it.
    """
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    magnitude = abs(value)
    if magnitude > _FLOAT32_MAX or (magnitude != 0.0 and magnitude < _FLOAT32_TINY):
        raise ValueError(f"{name} must be exactly zero or representable as a finite normal float32")


class RTUParameters(NamedTuple):
    """Trainable parameters of one diagonal complex RTU layer.

    ``nu_log`` and ``theta_log`` parameterize the radius and phase as
    ``r = exp(-exp(nu_log))`` and ``theta = exp(theta_log)``.  ``b_real`` and
    ``b_imag`` are the real and imaginary input projections.  Their names use
    lower case to follow Python conventions; they are the ``B`` parameters in
    the RTU equations.
    """

    nu_log: Array
    theta_log: Array
    b_real: Array
    b_imag: Array

    @property
    def B_real(self) -> Array:  # noqa: N802 - matches the published equation
        """Alias for the real input matrix using the paper's ``B`` notation."""
        return self.b_real

    @property
    def B_imag(self) -> Array:  # noqa: N802 - matches the published equation
        """Alias for the imaginary input matrix using the paper's notation."""
        return self.b_imag


class RTUState(NamedTuple):
    """Paired real representation of a complex RTU hidden state."""

    real: Array
    imaginary: Array


class RTUSensitivities(NamedTuple):
    """Compressed sensitivity ``d RTUState / d RTUParameters``.

    The first axis of every leaf selects real or imaginary hidden state.
    Shapes are ``(2, H)`` for the two diagonal recurrent parameter vectors
    and ``(2, H, F)`` for the two input matrices.  The ordinary recurrence is
    exact for fixed parameters; a moving-parameter Taylor path remains an
    approximation.
    """

    nu_log: Array
    theta_log: Array
    b_real: Array
    b_imag: Array

    @property
    def B_real(self) -> Array:  # noqa: N802 - matches the published equation
        """Alias for sensitivities of the real ``B`` matrix."""
        return self.b_real

    @property
    def B_imag(self) -> Array:  # noqa: N802 - matches the published equation
        """Alias for sensitivities of the imaginary ``B`` matrix."""
        return self.b_imag


class RTUNetworkParameters(NamedTuple):
    """Sparse feedforward/head and dense-input RTU parameters."""

    encoder_weights: Array
    encoder_bias: Array
    rtu: RTUParameters
    output_weights: Array
    output_bias: Array
    head_weights: Array
    head_bias: Array


class RunningStatistics(NamedTuple):
    """Welford running moments for one scalar or feature vector."""

    sample_count: Array
    mean: Array
    m2: Array


class RewardStatistics(NamedTuple):
    """Running return moments used by canonical stream-AC reward scaling."""

    sample_count: Array
    mean: Array
    m2: Array
    discounted_return: Array


class RecurrentTraceActorCriticState(NamedTuple):
    """Immutable state of :class:`RecurrentTraceActorCriticAgent`.

    Actor and critic recurrence, RTRL sensitivities, optional diagonal Taylor
    traces, AC(lambda) eligibility traces, and optional adaptive-ObGD second
    moments are deliberately separate.  The second moments persist across
    episode resets and are present only for an adaptive-ObGD configuration.
    ``last_observation``, ``last_action``, ``rng_key``, and ``step_count``
    mirror the lifecycle fields used by Alberta's existing control cores.
    """

    actor_params: RTUNetworkParameters
    critic_params: RTUNetworkParameters
    actor_rtu_state: RTUState
    critic_rtu_state: RTUState
    actor_sensitivities: RTUSensitivities
    critic_sensitivities: RTUSensitivities
    actor_taylor_trace: RTUSensitivities | None
    critic_taylor_trace: RTUSensitivities | None
    actor_traces: RTUNetworkParameters
    critic_traces: RTUNetworkParameters
    observation_statistics: RunningStatistics
    reward_statistics: RewardStatistics
    last_normalized_observation: Array
    last_observation: Array
    last_action: Array
    rng_key: Array
    step_count: Array
    started: Array
    actor_second_moments: RTUNetworkParameters | None = None
    critic_second_moments: RTUNetworkParameters | None = None

    def replace(self, **changes: Any) -> Self:
        """Return a state with selected fields replaced, like ``chex.dataclass``."""
        return self._replace(**changes)


class RecurrentTraceActorCriticUpdateResult(NamedTuple):
    """Diagnostics and state returned from one consumed transition."""

    state: RecurrentTraceActorCriticState
    action: Array
    policy: Array
    value: Array
    next_value: Array
    td_error: Array
    entropy: Array
    normalized_reward: Array
    actor_step_size: Array
    critic_step_size: Array
    actor_obgd_scale: Array
    critic_obgd_scale: Array
    update_applied: Array


class ObGDUpdate(NamedTuple):
    """Tree update and scalar diagnostics from observation-bounded GD."""

    updates: Any
    step_size: Array
    scale: Array
    update_applied: Array


class AdaptiveObGDUpdate(NamedTuple):
    """Adaptive ObGD update, persisted raw second moment, and diagnostics."""

    updates: Any
    second_moment: Any
    step_size: Array
    scale: Array
    update_applied: Array


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
_ACTUAL_FLOAT_TYPES = frozenset(
    {float, *(np.dtype(code).type for code in ("e", "f", "d", "g"))}
)


def _require_int32(name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


def _validated_config_float(name: str, value: object, **bounds: Any) -> float:
    if type(value) is bool or type(value) is np.bool_:
        raise ValueError(f"{name} must be numeric, not bool")
    if type(value) not in (_ACTUAL_INT_TYPES | _ACTUAL_FLOAT_TYPES):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        magnitude = abs(float(cast(Any, value)))
    except (OverflowError, ValueError) as error:
        raise ValueError(
            f"{name} must be exactly zero or representable as a finite normal float32"
        ) from error
    if (
        not math.isfinite(magnitude)
        or magnitude > _FLOAT32_MAX
        or (magnitude != 0.0 and magnitude < _FLOAT32_TINY)
    ):
        raise ValueError(
            f"{name} must be exactly zero or representable as a finite normal float32"
        )
    normalized, numerator, _ = validated_float32_scalar_with_ratio(name, value, **bounds)
    narrowed_magnitude = abs(float(np.float32(normalized)))
    if numerator != 0 and narrowed_magnitude < _FLOAT32_TINY:
        raise ValueError(
            f"{name} must be exactly zero or representable as a finite normal float32"
        )
    _validate_normal_float32_config_value(name, normalized)
    return normalized


def _network_parameter_scalars(
    *,
    feature_dim: int,
    output_dim: int,
    hidden_size: int,
    encoder_width: int,
    output_width: int,
) -> int:
    products = {
        "encoder matrix": encoder_width * feature_dim,
        "RTU input matrices": 2 * hidden_size * encoder_width,
        "output matrix": 2 * output_width * hidden_size,
        "head matrix": output_dim * output_width,
    }
    for name, value in products.items():
        if value > _INT32_MAX:
            raise ValueError(f"derived {name} scalars must fit signed int32")
    return (
        products["encoder matrix"]
        + encoder_width
        + 2 * hidden_size
        + products["RTU input matrices"]
        + products["output matrix"]
        + output_width
        + products["head matrix"]
        + output_dim
    )


def _preflight_state_resources(
    config: RecurrentTraceActorCriticConfig,
    feature_dim: object,
) -> dict[str, int]:
    width = _require_int32("feature_dim", feature_dim, minimum=1)
    actor_parameters = _network_parameter_scalars(
        feature_dim=width,
        output_dim=config.n_actions,
        hidden_size=config.hidden_size,
        encoder_width=config.encoder_width,
        output_width=config.output_width,
    )
    critic_parameters = _network_parameter_scalars(
        feature_dim=width,
        output_dim=1,
        hidden_size=config.hidden_size,
        encoder_width=config.encoder_width,
        output_width=config.output_width,
    )
    parameters = actor_parameters + critic_parameters
    sensitivity_scalars = 4 * config.hidden_size * (config.encoder_width + 1)
    float32_scalars = (
        parameters * (3 if config.adaptive_obgd else 2)
        + 2 * sensitivity_scalars * (2 if config.rtrl_taylor_correction else 1)
        + 4 * config.hidden_size
        + 4 * width
        + 3
    )
    # Two statistics counters, last action, typed key, step counter, and the
    # started flag are six logical array elements. The typed key occupies two
    # uint32 words physically, which is reflected separately in state_nbytes.
    logical_scalars = float32_scalars + 6
    state_nbytes = 4 * (float32_scalars + 6) + 1
    resources = {
        "parameter_scalars": parameters,
        "sensitivity_scalars_per_network": sensitivity_scalars,
        "float32_state_scalars": float32_scalars,
        "state_scalars": logical_scalars,
        "state_nbytes": state_nbytes,
    }
    for name, value in resources.items():
        if value > _INT32_MAX:
            raise ValueError(f"derived {name} must fit signed int32")
    return resources


def _preflight_update_working_set(
    config: RecurrentTraceActorCriticConfig,
    state_resources: Mapping[str, int],
) -> int:
    """Preflight the source state and complete returned update result."""

    # The result owns its returned state, one float32 policy value per action,
    # one int32 action, nine float32 diagnostics, and one boolean verdict.
    result_output_bytes = (
        state_resources["state_nbytes"] + 4 * config.n_actions + 4 + 9 * 4 + 1
    )
    update_working_set_bytes = state_resources["state_nbytes"] + result_output_bytes
    if update_working_set_bytes > _INT32_MAX:
        raise ValueError(
            "recurrent-trace actor-critic update working set byte count must fit signed int32"
        )
    return update_working_set_bytes


def _copy_config_mapping(name: str, config: object) -> dict[str, Any]:
    """Copy a legacy-compatible mapping while normalizing hostile hooks."""
    if type(config) is not dict:
        raise ValueError(f"{name} must be a mapping")
    try:
        payload = dict(config)
    except Exception as error:
        raise ValueError(f"{name} must be a readable mapping") from error
    if any(type(key) is not str for key in payload):
        raise ValueError(f"{name} fields must be strings")
    return cast(dict[str, Any], payload)


@dataclasses.dataclass(frozen=True)
class RecurrentTraceActorCriticConfig:
    """Configuration for an RTU-RTRL discrete streaming actor-critic.

    Each independent network has a sparse input block, one RTU layer with
    dense LeCun-normal input matrices, a sparse output block, and a sparse
    linear head.  Parameterless layer normalization precedes LeakyReLU in both
    feedforward blocks.  As in Farr et al., the RTU parameters receive their
    retained full-history RTRL gradient while the encoder receives the
    explicit one-step gradient through the current RTU transition.  The RTU
    history is exact only while its parameters are fixed; retaining exact
    history sensitivities for the encoder would defeat the intended
    linear-memory construction.

    ``actor_alpha``/``critic_alpha`` and ``actor_kappa``/``critic_kappa`` are
    independent because actor and critic trace magnitudes generally differ.
    Normalization follows the audited Memorax stream-AC wrappers: the current
    observation updates Welford population moments before it is centered and
    scaled; rewards are divided (without centering) by the running standard
    deviation of the discounted return.  Both are causal because only the
    current and earlier samples enter their statistics.

    ``rtrl_taylor_correction`` opts into the parameter-wise diagonal correction
    from Appendix C.2 of arXiv:2605.24709v2.  It is disabled by default.  The
    trace corrects the parameter-diagonal component of leading-order staleness.
    Omitted mixed-parameter Hessian terms can remain first order when several
    recurrent parameters move, so total staleness need not decrease.

    ``adaptive_obgd`` opts into the second-moment-normalized ObGD rule in the
    audited Memorax ``stream_ac.py`` implementation at commit ``94dd21d``.
    The default remains the historical unnormalized Alberta update.  Adaptive
    runs retain one raw second-moment tree for each of actor and critic.
    """

    n_actions: int
    hidden_size: int = 192
    encoder_width: int = 64
    output_width: int = 64
    gamma: float = 0.99
    actor_lamda: float = 0.8
    critic_lamda: float = 0.8
    actor_alpha: float = 1.0
    critic_alpha: float = 1.0
    actor_kappa: float = 3.0
    critic_kappa: float = 2.0
    entropy_coefficient: float = 0.095
    temperature: float = 1.0
    sparsity: float = 0.9
    r_min: float = 0.0
    r_max: float = 1.0
    max_phase: float = 6.28
    rtu_epsilon: float = 1e-8
    layer_norm_epsilon: float = 1e-5
    leaky_relu_slope: float = 0.01
    normalize_observations: bool = True
    normalize_rewards: bool = True
    normalization_epsilon: float = 1e-8
    # Diagonal Taylor trace correction, NOT exact RTRL — a derivation, not a
    # port (no upstream reference implementation exists); see
    # docs/design/rtu-taylor-correction.md for the derivation and its limits.
    rtrl_taylor_correction: bool = False
    adaptive_obgd: bool = False
    beta2: float = 0.999
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        """Validate shape and numerical invariants eagerly."""
        n_actions = _require_int32("n_actions", self.n_actions, minimum=1)
        hidden_size = _require_int32("hidden_size", self.hidden_size, minimum=1)
        encoder_width = _require_int32("encoder_width", self.encoder_width, minimum=2)
        output_width = _require_int32("output_width", self.output_width, minimum=2)

        object.__setattr__(self, "n_actions", n_actions)
        object.__setattr__(self, "hidden_size", hidden_size)
        object.__setattr__(self, "encoder_width", encoder_width)
        object.__setattr__(self, "output_width", output_width)

        float_domains: dict[str, dict[str, Any]] = {
            "gamma": {"lower": 0.0, "upper": 1.0},
            "actor_lamda": {"lower": 0.0, "upper": 1.0},
            "critic_lamda": {"lower": 0.0, "upper": 1.0},
            "actor_alpha": {"lower": 0.0},
            "critic_alpha": {"lower": 0.0},
            "actor_kappa": {"positive": True},
            "critic_kappa": {"positive": True},
            "entropy_coefficient": {"lower": 0.0},
            "temperature": {"positive": True},
            "sparsity": {"lower": 0.0, "upper": 1.0, "upper_inclusive": False},
            "r_min": {"lower": 0.0, "upper": 1.0},
            "r_max": {"positive": True, "upper": 1.0},
            "max_phase": {"positive": True, "upper": 2.0 * math.pi},
            "rtu_epsilon": {"positive": True, "upper": 1.0, "upper_inclusive": False},
            "layer_norm_epsilon": {"positive": True},
            "leaky_relu_slope": {"lower": 0.0},
            "normalization_epsilon": {"positive": True},
            "beta2": {"lower": 0.0, "upper": 1.0, "upper_inclusive": False},
            "epsilon": {"positive": True},
        }
        for name, bounds in float_domains.items():
            object.__setattr__(
                self,
                name,
                _validated_config_float(name, getattr(self, name), **bounds),
            )

        for interval_name, interval_value in (
            ("gamma", self.gamma),
            ("actor_lamda", self.actor_lamda),
            ("critic_lamda", self.critic_lamda),
        ):
            if not math.isfinite(interval_value) or not 0.0 <= interval_value <= 1.0:
                raise ValueError(f"{interval_name} must be finite and lie in [0, 1]")

        for nonnegative_name, nonnegative_value in (
            ("actor_alpha", self.actor_alpha),
            ("critic_alpha", self.critic_alpha),
            ("entropy_coefficient", self.entropy_coefficient),
            ("leaky_relu_slope", self.leaky_relu_slope),
        ):
            if not math.isfinite(nonnegative_value) or nonnegative_value < 0.0:
                raise ValueError(f"{nonnegative_name} must be finite and non-negative")

        for positive_name, positive_value in (
            ("actor_kappa", self.actor_kappa),
            ("critic_kappa", self.critic_kappa),
            ("temperature", self.temperature),
            ("max_phase", self.max_phase),
            ("rtu_epsilon", self.rtu_epsilon),
            ("layer_norm_epsilon", self.layer_norm_epsilon),
            ("normalization_epsilon", self.normalization_epsilon),
            ("epsilon", self.epsilon),
        ):
            if not math.isfinite(positive_value) or positive_value <= 0.0:
                raise ValueError(f"{positive_name} must be finite and positive")

        if not math.isfinite(self.sparsity) or not 0.0 <= self.sparsity < 1.0:
            raise ValueError("sparsity must be finite and lie in [0, 1)")
        if (
            not math.isfinite(self.r_min)
            or not math.isfinite(self.r_max)
            or not 0.0 <= self.r_min < self.r_max <= 1.0
        ):
            raise ValueError("r_min and r_max must satisfy 0 <= r_min < r_max <= 1")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            minimum_radius_squared = float(np.float32(self.r_min**2))
            maximum_radius_squared = float(np.float32(self.r_max**2))
        if self.r_min != 0.0 and minimum_radius_squared < _FLOAT32_TINY:
            raise ValueError("nonzero r_min squared must be a finite normal float32")
        if maximum_radius_squared < _FLOAT32_TINY:
            raise ValueError("r_max squared must be a finite normal float32")
        effective_minimum_radius_squared = max(
            minimum_radius_squared,
            _FLOAT32_TINY,
        )
        if not effective_minimum_radius_squared < maximum_radius_squared:
            raise ValueError(
                "r_min and r_max must define a non-empty float32 radius-squared "
                "interval above the numerical floor"
            )
        if self.max_phase > 2.0 * math.pi:
            raise ValueError("max_phase must not exceed 2*pi")
        if self.rtu_epsilon >= 1.0:
            raise ValueError("rtu_epsilon must be less than 1")
        if not 0.0 <= self.beta2 < 1.0 or not float(np.float32(self.beta2)) < 1.0:
            raise ValueError("beta2 must remain in [0, 1) after conversion to float32")
        for name, value in (
            ("normalize_observations", self.normalize_observations),
            ("normalize_rewards", self.normalize_rewards),
            ("rtrl_taylor_correction", self.rtrl_taylor_correction),
            ("adaptive_obgd", self.adaptive_obgd),
        ):
            if type(value) is not bool:
                raise ValueError(f"{name} must be a bool")
        _preflight_state_resources(self, 1)

    def state_resource_budget(self, feature_dim: object) -> dict[str, int]:
        """Return exact persistent-state counts for an observation width."""
        return _preflight_state_resources(self, feature_dim)

    def to_config(self) -> dict[str, Any]:
        """Return a JSON-compatible configuration mapping.

        Exact adaptive-ObGD defaults are omitted so historical default
        configuration bytes and their canonical hashes remain unchanged.
        """
        payload = dataclasses.asdict(self)
        if self.adaptive_obgd is False:
            payload.pop("adaptive_obgd")
        if self.beta2 == 0.999:
            payload.pop("beta2")
        if self.epsilon == 1e-8:
            payload.pop("epsilon")
        return payload

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> RecurrentTraceActorCriticConfig:
        """Reconstruct and validate a configuration mapping."""
        payload = _copy_config_mapping("config", config)
        try:
            return cls(**payload)
        except TypeError as error:
            raise ValueError("config fields do not match the serialized schema") from error


def parameterless_layer_norm(inputs: Array, epsilon: float = 1e-5) -> Array:
    """Normalize the final axis without learned scale or offset parameters."""
    mean = jnp.mean(inputs, axis=-1, keepdims=True)
    centered = inputs - mean
    variance = jnp.mean(jnp.square(centered), axis=-1, keepdims=True)
    return centered * jax.lax.rsqrt(variance + jnp.asarray(epsilon, inputs.dtype))


def leaky_relu(inputs: Array, negative_slope: float = 0.01) -> Array:
    """Apply a parameterless LeakyReLU with an explicit slope at zero."""
    slope = jnp.asarray(negative_slope, dtype=inputs.dtype)
    return jnp.where(inputs >= 0.0, inputs, slope * inputs)


def zero_rtu_state(hidden_size: int, dtype: jnp.dtype[Any] = jnp.float32) -> RTUState:
    """Construct a zero RTU state."""
    hidden_size = _require_int32("hidden_size", hidden_size, minimum=1)
    zeros = jnp.zeros((hidden_size,), dtype=dtype)
    return RTUState(real=zeros, imaginary=zeros)


def zero_rtu_sensitivities(
    hidden_size: int,
    input_dim: int,
    dtype: jnp.dtype[Any] = jnp.float32,
) -> RTUSensitivities:
    """Construct zero compressed RTRL sensitivities."""
    hidden_size = _require_int32("hidden_size", hidden_size, minimum=1)
    input_dim = _require_int32("input_dim", input_dim, minimum=1)
    recurrent = jnp.zeros((2, hidden_size), dtype=dtype)
    inputs = jnp.zeros((2, hidden_size, input_dim), dtype=dtype)
    return RTUSensitivities(
        nu_log=recurrent,
        theta_log=recurrent,
        b_real=inputs,
        b_imag=inputs,
    )


def _rtu_coefficients(
    params: RTUParameters,
    epsilon: float,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Return coefficients and derivatives under representable log bounds.

    The input norm uses an explicit piecewise floor.  Its analytic derivative
    uses the identical branch predicate, so the saturated-radius branch is
    constant with derivative zero rather than silently differentiating the
    unfloored expression.  Bounds correspond to the useful float32
    exponential range, the smallest useful radius, and one full phase
    rotation.  They prevent ``exp`` overflow, underflow-driven parameter
    drift, and meaningless large-angle trigonometry after long online runs;
    the transform is constant, with zero derivative, beyond either bound.
    """
    minimum_exp_log = jnp.asarray(
        _MINIMUM_EXP_LOG,
        dtype=params.nu_log.dtype,
    )
    maximum_nu_log = jnp.asarray(_MAXIMUM_NU_LOG, dtype=params.nu_log.dtype)
    nu_branch_active = (params.nu_log >= minimum_exp_log) & (params.nu_log <= maximum_nu_log)
    bounded_nu_log = jnp.where(
        params.nu_log < minimum_exp_log,
        minimum_exp_log,
        jnp.where(
            params.nu_log > maximum_nu_log,
            maximum_nu_log,
            params.nu_log,
        ),
    )
    exp_nu = jnp.exp(bounded_nu_log)
    d_exp_nu_d_nu_log = jnp.where(
        nu_branch_active,
        exp_nu,
        jnp.zeros_like(exp_nu),
    )
    radius = jnp.exp(-exp_nu)
    maximum_theta_log = jnp.asarray(
        _MAXIMUM_THETA_LOG,
        dtype=params.theta_log.dtype,
    )
    theta_branch_active = (params.theta_log >= minimum_exp_log) & (
        params.theta_log <= maximum_theta_log
    )
    bounded_theta_log = jnp.where(
        params.theta_log < minimum_exp_log,
        minimum_exp_log,
        jnp.where(
            params.theta_log > maximum_theta_log,
            maximum_theta_log,
            params.theta_log,
        ),
    )
    theta = jnp.exp(bounded_theta_log)
    d_theta_d_theta_log = jnp.where(
        theta_branch_active,
        theta,
        jnp.zeros_like(theta),
    )
    g = radius * jnp.cos(theta)
    phi = radius * jnp.sin(theta)
    raw_one_minus_radius_sq = 1.0 - jnp.square(radius)
    epsilon_array = jnp.asarray(epsilon, dtype=radius.dtype)
    norm_branch_active = raw_one_minus_radius_sq > epsilon_array
    stabilized_one_minus_radius_sq = jnp.where(
        norm_branch_active,
        raw_one_minus_radius_sq,
        epsilon_array,
    )
    norm_denominator = jnp.sqrt(stabilized_one_minus_radius_sq)
    norm = norm_denominator + epsilon_array
    dnorm_d_nu_log = jnp.where(
        norm_branch_active,
        d_exp_nu_d_nu_log * jnp.square(radius) / norm_denominator,
        jnp.zeros_like(radius),
    )
    return (
        g,
        phi,
        norm,
        d_exp_nu_d_nu_log,
        d_theta_d_theta_log,
        dnorm_d_nu_log,
    )


def rtu_forward(
    params: RTUParameters,
    state: RTUState,
    inputs: Array,
    *,
    epsilon: float = 1e-8,
) -> RTUState:
    """Advance the nonlinear diagonal complex recurrence by one sample."""
    g, phi, norm, _, _, _ = _rtu_coefficients(params, epsilon)
    input_real = params.b_real @ inputs
    input_imaginary = params.b_imag @ inputs
    pre_real = g * state.real - phi * state.imaginary + norm * input_real
    pre_imaginary = g * state.imaginary + phi * state.real + norm * input_imaginary
    return RTUState(
        real=jnp.tanh(pre_real),
        imaginary=jnp.tanh(pre_imaginary),
    )


def _propagate_sensitivity(
    sensitivity: Array,
    direct: Array,
    g: Array,
    phi: Array,
    derivative_real: Array,
    derivative_imaginary: Array,
) -> Array:
    """Apply the diagonal state Jacobian and add an immediate Jacobian."""
    trailing_axes = sensitivity.ndim - 2
    coefficient_shape = (g.shape[0],) + (1,) * trailing_axes
    g_broadcast = jnp.reshape(g, coefficient_shape)
    phi_broadcast = jnp.reshape(phi, coefficient_shape)
    derivative_real_broadcast = jnp.reshape(derivative_real, coefficient_shape)
    derivative_imaginary_broadcast = jnp.reshape(
        derivative_imaginary,
        coefficient_shape,
    )
    propagated_real = g_broadcast * sensitivity[0] - phi_broadcast * sensitivity[1] + direct[0]
    propagated_imaginary = phi_broadcast * sensitivity[0] + g_broadcast * sensitivity[1] + direct[1]
    return jnp.stack(
        (
            derivative_real_broadcast * propagated_real,
            derivative_imaginary_broadcast * propagated_imaginary,
        ),
        axis=0,
    )


def _propagate_postactivation_trace(
    trace: Array,
    direct: Array,
    g: Array,
    phi: Array,
    derivative_real: Array,
    derivative_imaginary: Array,
) -> Array:
    """Apply the state Jacobian to a trace and add a postactivation term."""
    trailing_axes = trace.ndim - 2
    coefficient_shape = (g.shape[0],) + (1,) * trailing_axes
    g_broadcast = jnp.reshape(g, coefficient_shape)
    phi_broadcast = jnp.reshape(phi, coefficient_shape)
    derivative_real_broadcast = jnp.reshape(derivative_real, coefficient_shape)
    derivative_imaginary_broadcast = jnp.reshape(
        derivative_imaginary,
        coefficient_shape,
    )
    propagated_real = g_broadcast * trace[0] - phi_broadcast * trace[1]
    propagated_imaginary = phi_broadcast * trace[0] + g_broadcast * trace[1]
    return (
        jnp.stack(
            (
                derivative_real_broadcast * propagated_real,
                derivative_imaginary_broadcast * propagated_imaginary,
            ),
            axis=0,
        )
        + direct
    )


def _rtu_coefficient_second_derivatives(
    params: RTUParameters,
    epsilon: float,
    g: Array,
    phi: Array,
    d_exp_nu_d_nu_log: Array,
    d_theta_d_theta_log: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Return diagonal second derivatives of the transformed coefficients.

    The branch conventions exactly match :func:`_rtu_coefficients`: clamped
    transforms and the floored input-normalization branch have zero first and
    second derivative outside their active intervals.
    """
    d2g_d_nu_log2 = (jnp.square(d_exp_nu_d_nu_log) - d_exp_nu_d_nu_log) * g
    d2phi_d_nu_log2 = (jnp.square(d_exp_nu_d_nu_log) - d_exp_nu_d_nu_log) * phi

    minimum_exp_log = jnp.asarray(_MINIMUM_EXP_LOG, dtype=params.nu_log.dtype)
    maximum_nu_log = jnp.asarray(_MAXIMUM_NU_LOG, dtype=params.nu_log.dtype)
    bounded_nu_log = jnp.where(
        params.nu_log < minimum_exp_log,
        minimum_exp_log,
        jnp.where(
            params.nu_log > maximum_nu_log,
            maximum_nu_log,
            params.nu_log,
        ),
    )
    exp_nu = jnp.exp(bounded_nu_log)
    radius = jnp.exp(-exp_nu)
    radius_squared = jnp.square(radius)
    raw_one_minus_radius_sq = 1.0 - radius_squared
    epsilon_array = jnp.asarray(epsilon, dtype=radius.dtype)
    norm_branch_active = raw_one_minus_radius_sq > epsilon_array
    norm_denominator = jnp.sqrt(
        jnp.where(
            norm_branch_active,
            raw_one_minus_radius_sq,
            epsilon_array,
        )
    )
    d2norm_d_nu_log2 = jnp.where(
        norm_branch_active,
        d_exp_nu_d_nu_log * radius_squared * (1.0 - 2.0 * d_exp_nu_d_nu_log) / norm_denominator
        - jnp.square(d_exp_nu_d_nu_log)
        * jnp.square(radius_squared)
        / jnp.power(norm_denominator, 3),
        jnp.zeros_like(radius),
    )

    d2g_d_theta_log2 = -g * jnp.square(d_theta_d_theta_log) - phi * d_theta_d_theta_log
    d2phi_d_theta_log2 = -phi * jnp.square(d_theta_d_theta_log) + g * d_theta_d_theta_log
    return (
        d2g_d_nu_log2,
        d2phi_d_nu_log2,
        d2norm_d_nu_log2,
        d2g_d_theta_log2,
        d2phi_d_theta_log2,
    )


def _postactivation_diagonal_hessian(
    direct: Array,
    direct_second: Array,
    derivative_real: Array,
    derivative_imaginary: Array,
    second_derivative_real: Array,
    second_derivative_imaginary: Array,
) -> Array:
    """Convert preactivation derivatives into ``diag(d I / d psi)``."""
    trailing_axes = direct.ndim - 2
    coefficient_shape = (direct.shape[1],) + (1,) * trailing_axes
    first = jnp.stack(
        (
            jnp.reshape(derivative_real, coefficient_shape),
            jnp.reshape(derivative_imaginary, coefficient_shape),
        ),
        axis=0,
    )
    second = jnp.stack(
        (
            jnp.reshape(second_derivative_real, coefficient_shape),
            jnp.reshape(second_derivative_imaginary, coefficient_shape),
        ),
        axis=0,
    )
    return second * jnp.square(direct) + first * direct_second


def rtu_step(
    params: RTUParameters,
    state: RTUState,
    sensitivities: RTUSensitivities,
    inputs: Array,
    *,
    epsilon: float = 1e-8,
) -> tuple[RTUState, RTUSensitivities]:
    """Advance an RTU state and its exact forward RTRL sensitivities.

    This is the chain rule ``S_t = J_t S_{t-1} + I_t`` specialized to a
    diagonal complex recurrence.  Only structurally non-zero derivatives are
    retained; no gradient truncation or approximation occurs.
    """
    (
        g,
        phi,
        norm,
        d_exp_nu_d_nu_log,
        d_theta_d_theta_log,
        dnorm_d_nu_log,
    ) = _rtu_coefficients(
        params,
        epsilon,
    )
    input_real = params.b_real @ inputs
    input_imaginary = params.b_imag @ inputs
    pre_real = g * state.real - phi * state.imaginary + norm * input_real
    pre_imaginary = g * state.imaginary + phi * state.real + norm * input_imaginary
    next_state = RTUState(
        real=jnp.tanh(pre_real),
        imaginary=jnp.tanh(pre_imaginary),
    )
    derivative_real = 1.0 - jnp.square(next_state.real)
    derivative_imaginary = 1.0 - jnp.square(next_state.imaginary)

    dg_d_nu_log = -d_exp_nu_d_nu_log * g
    dphi_d_nu_log = -d_exp_nu_d_nu_log * phi
    direct_nu = jnp.stack(
        (
            dg_d_nu_log * state.real
            - dphi_d_nu_log * state.imaginary
            + dnorm_d_nu_log * input_real,
            dg_d_nu_log * state.imaginary
            + dphi_d_nu_log * state.real
            + dnorm_d_nu_log * input_imaginary,
        ),
        axis=0,
    )

    dg_d_theta_log = -phi * d_theta_d_theta_log
    dphi_d_theta_log = g * d_theta_d_theta_log
    direct_theta = jnp.stack(
        (
            dg_d_theta_log * state.real - dphi_d_theta_log * state.imaginary,
            dg_d_theta_log * state.imaginary + dphi_d_theta_log * state.real,
        ),
        axis=0,
    )

    projected_inputs = norm[:, None] * inputs[None, :]
    zeros = jnp.zeros_like(projected_inputs)
    direct_b_real = jnp.stack((projected_inputs, zeros), axis=0)
    direct_b_imaginary = jnp.stack((zeros, projected_inputs), axis=0)

    next_sensitivities = RTUSensitivities(
        nu_log=_propagate_sensitivity(
            sensitivities.nu_log,
            direct_nu,
            g,
            phi,
            derivative_real,
            derivative_imaginary,
        ),
        theta_log=_propagate_sensitivity(
            sensitivities.theta_log,
            direct_theta,
            g,
            phi,
            derivative_real,
            derivative_imaginary,
        ),
        b_real=_propagate_sensitivity(
            sensitivities.b_real,
            direct_b_real,
            g,
            phi,
            derivative_real,
            derivative_imaginary,
        ),
        b_imag=_propagate_sensitivity(
            sensitivities.b_imag,
            direct_b_imaginary,
            g,
            phi,
            derivative_real,
            derivative_imaginary,
        ),
    )
    return next_state, next_sensitivities


def rtu_taylor_step(
    params: RTUParameters,
    state: RTUState,
    sensitivities: RTUSensitivities,
    taylor_trace: RTUSensitivities,
    parameter_delta: RTUParameters,
    inputs: Array,
    *,
    epsilon: float = 1e-8,
) -> tuple[RTUState, RTUSensitivities, RTUSensitivities]:
    """Advance the paper's diagonal Taylor-corrected approximate sensitivity.

    With ``I_t = partial h_t / partial psi`` and
    ``J_t = partial h_t / partial h_{t-1}``, arXiv:2605.24709v2 defines

    ``omega_t = J_t omega_{t-1} + diag_psi(partial I_t / partial psi)``

    and

    ``S_t = J_t (S_{t-1} + omega_{t-1} * Delta psi_{t-1}) + I_t``.

    ``parameter_delta`` must be the effective parameter change from the
    parameters used for ``state`` to ``params``.  Every RTU parameter leaf is
    included.  The returned trace has the compressed shape of ``S_t`` and is
    the parameter-wise diagonal approximation to the paper's otherwise
    rank-three ``Omega_t``.  It corrects the parameter-diagonal component of
    leading-order staleness.  Mixed-parameter Hessian terms are omitted, can
    leave a first-order residual, and mean that total staleness need not
    decrease.  This is not an exact moving-parameter RTRL sensitivity.
    """
    corrected_previous = RTUSensitivities(
        nu_log=(sensitivities.nu_log + taylor_trace.nu_log * parameter_delta.nu_log[None, :]),
        theta_log=(
            sensitivities.theta_log + taylor_trace.theta_log * parameter_delta.theta_log[None, :]
        ),
        b_real=(sensitivities.b_real + taylor_trace.b_real * parameter_delta.b_real[None, :, :]),
        b_imag=(sensitivities.b_imag + taylor_trace.b_imag * parameter_delta.b_imag[None, :, :]),
    )
    next_state, next_sensitivities = rtu_step(
        params,
        state,
        corrected_previous,
        inputs,
        epsilon=epsilon,
    )

    (
        g,
        phi,
        norm,
        d_exp_nu_d_nu_log,
        d_theta_d_theta_log,
        dnorm_d_nu_log,
    ) = _rtu_coefficients(params, epsilon)
    derivative_real = 1.0 - jnp.square(next_state.real)
    derivative_imaginary = 1.0 - jnp.square(next_state.imaginary)
    second_derivative_real = -2.0 * next_state.real * derivative_real
    second_derivative_imaginary = -2.0 * next_state.imaginary * derivative_imaginary
    input_real = params.b_real @ inputs
    input_imaginary = params.b_imag @ inputs

    dg_d_nu_log = -d_exp_nu_d_nu_log * g
    dphi_d_nu_log = -d_exp_nu_d_nu_log * phi
    direct_nu = jnp.stack(
        (
            dg_d_nu_log * state.real
            - dphi_d_nu_log * state.imaginary
            + dnorm_d_nu_log * input_real,
            dg_d_nu_log * state.imaginary
            + dphi_d_nu_log * state.real
            + dnorm_d_nu_log * input_imaginary,
        ),
        axis=0,
    )
    (
        d2g_d_nu_log2,
        d2phi_d_nu_log2,
        d2norm_d_nu_log2,
        d2g_d_theta_log2,
        d2phi_d_theta_log2,
    ) = _rtu_coefficient_second_derivatives(
        params,
        epsilon,
        g,
        phi,
        d_exp_nu_d_nu_log,
        d_theta_d_theta_log,
    )
    direct_second_nu = jnp.stack(
        (
            d2g_d_nu_log2 * state.real
            - d2phi_d_nu_log2 * state.imaginary
            + d2norm_d_nu_log2 * input_real,
            d2g_d_nu_log2 * state.imaginary
            + d2phi_d_nu_log2 * state.real
            + d2norm_d_nu_log2 * input_imaginary,
        ),
        axis=0,
    )

    dg_d_theta_log = -phi * d_theta_d_theta_log
    dphi_d_theta_log = g * d_theta_d_theta_log
    direct_theta = jnp.stack(
        (
            dg_d_theta_log * state.real - dphi_d_theta_log * state.imaginary,
            dg_d_theta_log * state.imaginary + dphi_d_theta_log * state.real,
        ),
        axis=0,
    )
    direct_second_theta = jnp.stack(
        (
            d2g_d_theta_log2 * state.real - d2phi_d_theta_log2 * state.imaginary,
            d2g_d_theta_log2 * state.imaginary + d2phi_d_theta_log2 * state.real,
        ),
        axis=0,
    )

    projected_inputs = norm[:, None] * inputs[None, :]
    zeros = jnp.zeros_like(projected_inputs)
    direct_b_real = jnp.stack((projected_inputs, zeros), axis=0)
    direct_b_imaginary = jnp.stack((zeros, projected_inputs), axis=0)
    zero_second_b = jnp.zeros_like(direct_b_real)

    immediate_hessian = RTUSensitivities(
        nu_log=_postactivation_diagonal_hessian(
            direct_nu,
            direct_second_nu,
            derivative_real,
            derivative_imaginary,
            second_derivative_real,
            second_derivative_imaginary,
        ),
        theta_log=_postactivation_diagonal_hessian(
            direct_theta,
            direct_second_theta,
            derivative_real,
            derivative_imaginary,
            second_derivative_real,
            second_derivative_imaginary,
        ),
        b_real=_postactivation_diagonal_hessian(
            direct_b_real,
            zero_second_b,
            derivative_real,
            derivative_imaginary,
            second_derivative_real,
            second_derivative_imaginary,
        ),
        b_imag=_postactivation_diagonal_hessian(
            direct_b_imaginary,
            zero_second_b,
            derivative_real,
            derivative_imaginary,
            second_derivative_real,
            second_derivative_imaginary,
        ),
    )
    next_taylor_trace = RTUSensitivities(
        nu_log=_propagate_postactivation_trace(
            taylor_trace.nu_log,
            immediate_hessian.nu_log,
            g,
            phi,
            derivative_real,
            derivative_imaginary,
        ),
        theta_log=_propagate_postactivation_trace(
            taylor_trace.theta_log,
            immediate_hessian.theta_log,
            g,
            phi,
            derivative_real,
            derivative_imaginary,
        ),
        b_real=_propagate_postactivation_trace(
            taylor_trace.b_real,
            immediate_hessian.b_real,
            g,
            phi,
            derivative_real,
            derivative_imaginary,
        ),
        b_imag=_propagate_postactivation_trace(
            taylor_trace.b_imag,
            immediate_hessian.b_imag,
            g,
            phi,
            derivative_real,
            derivative_imaginary,
        ),
    )
    return next_state, next_sensitivities, next_taylor_trace


def rtu_network_encode(
    params: RTUNetworkParameters,
    observation: Array,
    *,
    layer_norm_epsilon: float = 1e-5,
    negative_slope: float = 0.01,
) -> Array:
    """Apply the sparse input block that supplies one RTU input vector."""
    preactivation = params.encoder_weights @ observation + params.encoder_bias
    return leaky_relu(
        parameterless_layer_norm(preactivation, layer_norm_epsilon),
        negative_slope,
    )


def rtu_network_output(
    params: RTUNetworkParameters,
    state: RTUState,
    *,
    layer_norm_epsilon: float = 1e-5,
    negative_slope: float = 0.01,
) -> Array:
    """Apply the sparse output block and linear policy or value head."""
    hidden = jnp.concatenate((state.real, state.imaginary), axis=-1)
    features = leaky_relu(
        parameterless_layer_norm(
            params.output_weights @ hidden + params.output_bias,
            layer_norm_epsilon,
        ),
        negative_slope,
    )
    return params.head_weights @ features + params.head_bias


def exact_rtrl_gradient(
    params: RTUNetworkParameters,
    state: RTUState,
    sensitivities: RTUSensitivities,
    objective: Callable[[Array], Array],
    *,
    encoder_input: Array | None = None,
    rtu_epsilon: float = 1e-8,
    layer_norm_epsilon: float = 1e-5,
    negative_slope: float = 0.01,
) -> tuple[Array, RTUNetworkParameters]:
    """Differentiate a scalar objective using RTU-RTRL and local encoder AD.

    Reverse-mode differentiation supplies the objective cotangent with
    respect to the current paired RTU state.  Explicit contraction with the
    forward sensitivities supplies gradients for every RTU parameter.  This
    is equivalent to phantom-gradient injection but makes the chain rule
    visible and testable without a neural-network framework.  When
    ``encoder_input`` is supplied, encoder parameters receive the paper's
    one-step gradient through the current RTU transition.  Their older
    influence on recurrent state is intentionally not retained.
    """

    def scalar_objective(
        output_weights: Array,
        output_bias: Array,
        head_weights: Array,
        head_bias: Array,
        real: Array,
        imaginary: Array,
    ) -> Array:
        candidate_params = RTUNetworkParameters(
            encoder_weights=params.encoder_weights,
            encoder_bias=params.encoder_bias,
            rtu=params.rtu,
            output_weights=output_weights,
            output_bias=output_bias,
            head_weights=head_weights,
            head_bias=head_bias,
        )
        output = rtu_network_output(
            candidate_params,
            RTUState(real=real, imaginary=imaginary),
            layer_norm_epsilon=layer_norm_epsilon,
            negative_slope=negative_slope,
        )
        return jnp.reshape(objective(output), ())

    value, direct_gradients = jax.value_and_grad(
        scalar_objective,
        argnums=(0, 1, 2, 3, 4, 5),
    )(
        params.output_weights,
        params.output_bias,
        params.head_weights,
        params.head_bias,
        state.real,
        state.imaginary,
    )
    (
        output_weights_gradient,
        output_bias_gradient,
        head_weights_gradient,
        head_bias_gradient,
        real_cotangent,
        imaginary_cotangent,
    ) = direct_gradients
    state_cotangent = jnp.stack((real_cotangent, imaginary_cotangent), axis=0)
    rtu_gradient = RTUParameters(
        nu_log=jnp.sum(state_cotangent * sensitivities.nu_log, axis=0),
        theta_log=jnp.sum(
            state_cotangent * sensitivities.theta_log,
            axis=0,
        ),
        b_real=jnp.sum(
            state_cotangent[:, :, None] * sensitivities.b_real,
            axis=0,
        ),
        b_imag=jnp.sum(
            state_cotangent[:, :, None] * sensitivities.b_imag,
            axis=0,
        ),
    )

    if encoder_input is None:
        encoder_weights_gradient = jnp.zeros_like(params.encoder_weights)
        encoder_bias_gradient = jnp.zeros_like(params.encoder_bias)
    else:
        _, _, norm, _, _, _ = _rtu_coefficients(params.rtu, rtu_epsilon)
        preactivation_real_cotangent = real_cotangent * (1.0 - jnp.square(state.real))
        preactivation_imaginary_cotangent = imaginary_cotangent * (
            1.0 - jnp.square(state.imaginary)
        )
        rtu_input_cotangent = params.rtu.b_real.T @ (
            norm * preactivation_real_cotangent
        ) + params.rtu.b_imag.T @ (norm * preactivation_imaginary_cotangent)

        def encoder_features(
            encoder_weights: Array,
            encoder_bias: Array,
        ) -> Array:
            preactivation = encoder_weights @ encoder_input + encoder_bias
            return leaky_relu(
                parameterless_layer_norm(
                    preactivation,
                    layer_norm_epsilon,
                ),
                negative_slope,
            )

        _, encoder_pullback = jax.vjp(
            encoder_features,
            params.encoder_weights,
            params.encoder_bias,
        )
        encoder_weights_gradient, encoder_bias_gradient = encoder_pullback(rtu_input_cotangent)

    return value, RTUNetworkParameters(
        encoder_weights=encoder_weights_gradient,
        encoder_bias=encoder_bias_gradient,
        rtu=rtu_gradient,
        output_weights=output_weights_gradient,
        output_bias=output_bias_gradient,
        head_weights=head_weights_gradient,
        head_bias=head_bias_gradient,
    )


def obgd_update(
    traces: Any,
    signal: Array,
    *,
    alpha: float,
    kappa: float,
) -> ObGDUpdate:
    """Apply one observation-bounded gradient update to a trace tree.

    For ``z_sum = sum(abs(z))``, the effective step size is
    ``alpha / max(1, alpha * kappa * max(abs(signal), 1) * z_sum)``.
    The returned parameter change is then ``step_size * signal * z``.  The
    signal is intentionally absent from ``traces`` and is multiplied here
    exactly once.
    """
    signal = jnp.reshape(jnp.asarray(signal), ())
    z_sum = jnp.asarray(0.0, dtype=signal.dtype)
    for leaf in jax.tree_util.tree_leaves(traces):
        z_sum = z_sum + jnp.sum(jnp.abs(leaf))
    alpha_array = jnp.asarray(alpha, dtype=signal.dtype)
    kappa_array = jnp.asarray(kappa, dtype=signal.dtype)
    bound_term = alpha_array * kappa_array * jnp.maximum(jnp.abs(signal), 1.0) * z_sum
    denominator = jnp.maximum(1.0, bound_term)
    scale = jnp.reciprocal(denominator)
    step_size = alpha_array * scale
    proposed_updates = jax.tree_util.tree_map(
        lambda trace: step_size * signal * trace,
        traces,
    )
    update_applied = (
        jnp.isfinite(signal)
        & jnp.isfinite(alpha_array)
        & (alpha_array >= 0.0)
        & jnp.isfinite(kappa_array)
        & (kappa_array >= 0.0)
        & _floating_tree_is_finite(traces)
        & _floating_tree_is_finite(proposed_updates)
        & jnp.isfinite(step_size)
        & jnp.isfinite(scale)
    )
    updates = jax.tree_util.tree_map(
        lambda proposed: jnp.where(update_applied, proposed, jnp.zeros_like(proposed)),
        proposed_updates,
    )
    return ObGDUpdate(
        updates=updates,
        step_size=jnp.where(update_applied, step_size, jnp.zeros_like(step_size)),
        scale=jnp.where(update_applied, scale, jnp.zeros_like(scale)),
        update_applied=update_applied,
    )


def adaptive_obgd_update(
    traces: Any,
    second_moment: Any,
    signal: Array,
    *,
    alpha: float,
    kappa: float,
    beta2: float,
    epsilon: float,
    step: Array | int,
) -> AdaptiveObGDUpdate:
    """Apply the adaptive stream-AC ObGD rule from Memorax ``94dd21d``.

    For every trace element ``z``, the raw moment is
    ``v = beta2*v + (1-beta2)*(signal*z)^2`` and its bias-corrected value is
    ``v_hat = v/(1-beta2**step)``.  The ObGD bound uses one global
    ``z_sum = sum(abs(z)/(sqrt(v_hat)+epsilon))`` across the complete tree;
    the same normalized trace supplies each parameter update.  Alberta's RTU
    core has one streaming environment, so the reference implementation's
    leading-environment mean is the identity here.
    """
    signal = jnp.reshape(jnp.asarray(signal), ())
    step_array = jnp.reshape(jnp.asarray(step), ())
    if not jnp.issubdtype(step_array.dtype, jnp.integer):
        raise ValueError("adaptive ObGD step must have integer dtype")
    if not isinstance(step_array, jax.core.Tracer) and int(step_array) < 1:
        raise ValueError("adaptive ObGD step must be positive")

    beta = jnp.asarray(beta2, dtype=signal.dtype)
    epsilon_array = jnp.asarray(epsilon, dtype=signal.dtype)
    proposed_second_moment = jax.tree_util.tree_map(
        lambda previous, trace: (
            jnp.where(beta == 0.0, jnp.zeros_like(previous), beta * previous)
            + (1.0 - beta) * jnp.square(signal * trace)
        ),
        second_moment,
        traces,
    )
    bias_correction = 1.0 - beta**step_array
    corrected_second_moment = jax.tree_util.tree_map(
        lambda moment: moment / bias_correction,
        proposed_second_moment,
    )
    normalized_traces = jax.tree_util.tree_map(
        lambda trace, corrected: trace / (jnp.sqrt(corrected) + epsilon_array),
        traces,
        corrected_second_moment,
    )

    z_sum = jnp.asarray(0.0, dtype=signal.dtype)
    for leaf in jax.tree_util.tree_leaves(normalized_traces):
        z_sum = z_sum + jnp.sum(jnp.abs(leaf))
    bound_term = (
        jnp.maximum(jnp.abs(signal), 1.0)
        * z_sum
        * jnp.asarray(alpha, dtype=signal.dtype)
        * jnp.asarray(kappa, dtype=signal.dtype)
    )
    denominator = jnp.maximum(1.0, bound_term)
    scale = jnp.reciprocal(denominator)
    step_size = jnp.asarray(alpha, dtype=signal.dtype) / denominator
    proposed_updates = jax.tree_util.tree_map(
        lambda normalized_trace: step_size * signal * normalized_trace,
        normalized_traces,
    )
    alpha_array = jnp.asarray(alpha, dtype=signal.dtype)
    kappa_array = jnp.asarray(kappa, dtype=signal.dtype)
    checked_second_moment = jax.tree_util.tree_map(
        lambda previous: jnp.where(beta == 0.0, jnp.zeros_like(previous), previous),
        second_moment,
    )
    update_applied = (
        jnp.isfinite(signal)
        & jnp.isfinite(alpha_array)
        & (alpha_array >= 0.0)
        & jnp.isfinite(kappa_array)
        & (kappa_array >= 0.0)
        & jnp.isfinite(beta)
        & (beta >= 0.0)
        & (beta < 1.0)
        & jnp.isfinite(epsilon_array)
        & (epsilon_array > 0.0)
        & _floating_tree_is_finite(traces)
        & _floating_tree_is_finite(checked_second_moment)
        & _floating_tree_is_finite(proposed_second_moment)
        & _floating_tree_is_finite(proposed_updates)
        & jnp.isfinite(step_size)
        & jnp.isfinite(scale)
    )
    updates = jax.tree_util.tree_map(
        lambda proposed: jnp.where(update_applied, proposed, jnp.zeros_like(proposed)),
        proposed_updates,
    )
    new_second_moment = jax.tree_util.tree_map(
        lambda proposed, previous: jnp.where(update_applied, proposed, previous),
        proposed_second_moment,
        second_moment,
    )
    return AdaptiveObGDUpdate(
        updates=updates,
        second_moment=new_second_moment,
        step_size=jnp.where(update_applied, step_size, jnp.zeros_like(step_size)),
        scale=jnp.where(update_applied, scale, jnp.zeros_like(scale)),
        update_applied=update_applied,
    )


def _initialize_rtu_parameters(
    key: Array,
    input_dim: int,
    config: RecurrentTraceActorCriticConfig,
) -> RTUParameters:
    """Initialize stable polar dynamics and dense LeCun-normal RTU inputs."""
    nu_key, theta_key, real_key, imaginary_key = jr.split(key, 4)
    radius_squared = jr.uniform(
        nu_key,
        (config.hidden_size,),
        minval=config.r_min**2,
        maxval=config.r_max**2,
        dtype=jnp.float32,
    )
    radius_squared = jnp.maximum(
        radius_squared,
        jnp.asarray(_FLOAT32_TINY, dtype=jnp.float32),
    )
    nu_log = jnp.log(-0.5 * jnp.log(radius_squared))
    phase = config.max_phase * jr.uniform(
        theta_key,
        (config.hidden_size,),
        dtype=jnp.float32,
    )
    phase = jnp.maximum(
        phase,
        jnp.asarray(_FLOAT32_TINY, dtype=jnp.float32),
    )
    theta_log = jnp.log(phase)
    dense_input_initializer = lecun_normal()
    return RTUParameters(
        nu_log=nu_log,
        theta_log=theta_log,
        b_real=dense_input_initializer(
            real_key,
            (config.hidden_size, input_dim),
            jnp.float32,
        ),
        b_imag=dense_input_initializer(
            imaginary_key,
            (config.hidden_size, input_dim),
            jnp.float32,
        ),
    )


def _live_sparse_init(
    key: Array,
    shape: tuple[int, int],
    sparsity: float,
) -> Array:
    """Sparse-initialize while retaining at least one input per output row."""
    fan_in = shape[1]
    maximum_live_sparsity = (fan_in - 1) / fan_in
    return sparse_init(
        key,
        shape,
        sparsity=min(sparsity, maximum_live_sparsity),
    )


def initialize_rtu_network_parameters(
    key: Array,
    input_dim: int,
    output_dim: int,
    config: RecurrentTraceActorCriticConfig,
) -> RTUNetworkParameters:
    """Initialize sparse feedforward/head blocks and a dense-input RTU."""
    input_dim = _require_int32("input_dim", input_dim, minimum=1)
    output_dim = _require_int32("output_dim", output_dim, minimum=1)
    encoder_key, rtu_key, output_key, head_key = jr.split(key, 4)
    return RTUNetworkParameters(
        encoder_weights=_live_sparse_init(
            encoder_key,
            (config.encoder_width, input_dim),
            config.sparsity,
        ),
        encoder_bias=jnp.zeros((config.encoder_width,), dtype=jnp.float32),
        rtu=_initialize_rtu_parameters(
            rtu_key,
            config.encoder_width,
            config,
        ),
        output_weights=_live_sparse_init(
            output_key,
            (config.output_width, 2 * config.hidden_size),
            config.sparsity,
        ),
        output_bias=jnp.zeros((config.output_width,), dtype=jnp.float32),
        head_weights=_live_sparse_init(
            head_key,
            (output_dim, config.output_width),
            config.sparsity,
        ),
        head_bias=jnp.zeros((output_dim,), dtype=jnp.float32),
    )


def _zero_network_like(params: RTUNetworkParameters) -> RTUNetworkParameters:
    """Construct a zero tree with the same network parameter structure."""
    return cast(
        RTUNetworkParameters,
        jax.tree_util.tree_map(jnp.zeros_like, params),
    )


def _zero_rtu_parameters_like(params: RTUParameters) -> RTUParameters:
    """Construct a zero RTU-parameter tree for an effective parameter delta."""
    return cast(
        RTUParameters,
        jax.tree_util.tree_map(jnp.zeros_like, params),
    )


def _initial_running_statistics(shape: tuple[int, ...]) -> RunningStatistics:
    """Initialize moments exactly as the stream-AC observation wrapper."""
    return RunningStatistics(
        sample_count=jnp.asarray(1, dtype=jnp.int32),
        mean=jnp.zeros(shape, dtype=jnp.float32),
        m2=jnp.ones(shape, dtype=jnp.float32),
    )


def _initial_reward_statistics() -> RewardStatistics:
    """Initialize moments exactly as the stream-AC reward wrapper."""
    scalar_zero = jnp.asarray(0.0, dtype=jnp.float32)
    return RewardStatistics(
        sample_count=jnp.asarray(1, dtype=jnp.int32),
        mean=scalar_zero,
        m2=jnp.asarray(1.0, dtype=jnp.float32),
        discounted_return=scalar_zero,
    )


def _saturating_int32_increment(value: Array) -> tuple[Array, Array]:
    """Increment a non-negative int32 counter without allowing wraparound.

    The boolean result marks whether the increment was accepted.  Callers that
    maintain running moments use it to freeze those moments once the largest
    portable JAX int32 count has been represented exactly.
    """
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    can_increment = value < maximum
    return value + can_increment.astype(jnp.int32), can_increment


def _update_running_statistics(
    statistics: RunningStatistics,
    sample: Array,
) -> RunningStatistics:
    """Consume one sample with Welford's stable online recurrence."""
    sample = jnp.asarray(sample, dtype=statistics.mean.dtype)
    sample_count, can_increment = _saturating_int32_increment(
        statistics.sample_count,
    )
    delta = sample - statistics.mean
    candidate_mean = statistics.mean + delta / sample_count
    delta_after = sample - candidate_mean
    candidate_m2 = statistics.m2 + delta * delta_after
    return RunningStatistics(
        sample_count=sample_count,
        mean=jnp.where(can_increment, candidate_mean, statistics.mean),
        m2=jnp.where(can_increment, candidate_m2, statistics.m2),
    )


def _population_standard_deviation(
    m2: Array,
    sample_count: Array,
    epsilon: float,
) -> Array:
    """Return the population standard deviation used by stream-AC wrappers."""
    variance = m2 / sample_count
    return jnp.sqrt(variance + jnp.asarray(epsilon, dtype=variance.dtype))


def _project_rtu_parameter_logs(
    params: RTUNetworkParameters,
) -> RTUNetworkParameters:
    """Project learned RTU logs back into the stable forward-transform range."""
    nu_limit = jnp.asarray(_MAXIMUM_NU_LOG, dtype=params.rtu.nu_log.dtype)
    theta_limit = jnp.asarray(
        _MAXIMUM_THETA_LOG,
        dtype=params.rtu.theta_log.dtype,
    )
    return params._replace(
        rtu=params.rtu._replace(
            nu_log=jnp.clip(
                params.rtu.nu_log,
                jnp.asarray(_MINIMUM_EXP_LOG, dtype=params.rtu.nu_log.dtype),
                nu_limit,
            ),
            theta_log=jnp.clip(
                params.rtu.theta_log,
                jnp.asarray(
                    _MINIMUM_EXP_LOG,
                    dtype=params.rtu.theta_log.dtype,
                ),
                theta_limit,
            ),
        )
    )


def _validated_scalar(
    name: str,
    value: Array | bool | float,
    *,
    boolean: bool,
    unit_interval: bool = False,
) -> Array:
    """Validate the static facts available before entering a jitted update."""
    scalar = jnp.asarray(value)
    if scalar.shape != ():
        raise ValueError(f"{name} must be scalar-shaped")
    if boolean:
        if scalar.dtype != jnp.bool_:
            raise ValueError(f"{name} must have boolean dtype")
    else:
        if not jnp.issubdtype(scalar.dtype, jnp.number) or jnp.issubdtype(
            scalar.dtype,
            jnp.bool_,
        ):
            raise ValueError(f"{name} must have numeric dtype")
        if not isinstance(scalar, jax.core.Tracer):
            concrete = float(scalar)
            if not math.isfinite(concrete):
                if unit_interval:
                    raise ValueError(f"{name} must be finite and lie in [0, 1]")
                raise ValueError(f"{name} must be finite")
            if unit_interval and not 0.0 <= concrete <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return scalar


def _validated_finite_array(
    name: str,
    value: Array,
    *,
    expected_shape: tuple[int, ...],
) -> Array:
    """Validate a numeric, finite array at an eager public boundary."""
    array = jnp.asarray(value)
    if array.shape != expected_shape:
        raise ValueError(f"{name} shape must match the initialized feature shape")
    if not jnp.issubdtype(array.dtype, jnp.number) or jnp.issubdtype(
        array.dtype,
        jnp.bool_,
    ):
        raise ValueError(f"{name} must have numeric dtype")
    if not isinstance(array, jax.core.Tracer) and not bool(jnp.all(jnp.isfinite(array))):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _runtime_checked_value(
    value: Array,
    predicate: Array,
    message: str,
) -> Array:
    """Return ``value`` while enforcing a dynamic assertion inside JAX transforms.

    JAX's functional ``checkify`` errors cannot be thrown from a function that
    an outer caller subsequently wraps in ``jax.jit``.  A conditional debug
    callback is therefore used at this public API boundary.  The callback is
    absent from the ordinary valid scalar execution branch.  Under ``vmap``,
    JAX may evaluate the callback branch for every lane, so the host callback
    rechecks the lane predicate before raising.

    A failed compiled check surfaces on synchronization as
    :class:`jax.errors.JaxRuntimeError` wrapping the ``ValueError`` raised by
    the host callback.
    """
    predicate = jnp.reshape(jnp.asarray(predicate, dtype=jnp.bool_), ())

    def valid_branch(operand: tuple[Array, Array]) -> Array:
        checked_value, _ = operand
        return checked_value

    def invalid_branch(operand: tuple[Array, Array]) -> Array:
        checked_value, runtime_predicate = operand

        def raise_if_false(concrete_predicate: Any) -> None:
            if not bool(concrete_predicate):
                raise ValueError(message)

        jax.debug.callback(
            raise_if_false,
            runtime_predicate,
            ordered=True,
        )
        return checked_value

    return cast(
        Array,
        jax.lax.cond(
            predicate,
            valid_branch,
            invalid_branch,
            (value, predicate),
        ),
    )


class RecurrentTraceActorCriticAgent:
    """One-transition-at-a-time softmax actor-critic with RTU-RTRL.

    ``start`` advances both independent RTUs on the initial observation and
    samples the first action.  Each call to ``update`` consumes exactly the
    transition held in ``state.last_observation``/``state.last_action`` plus
    the supplied reward and next observation.  No replay or trajectory buffer
    exists in the state.

    A boundary transition still updates from its current gradient, then clears
    stored AC(lambda) traces.  Its supplied observation is always the
    history-continuing bootstrap observation.  A distinct ``reset_observation``
    may seed the next episode's zeroed recurrence; for legacy callers that
    omit it, the bootstrap observation is reused as that first observation.
    """

    def __init__(self, config: RecurrentTraceActorCriticConfig):
        self._config = config

    @property
    def config(self) -> RecurrentTraceActorCriticConfig:
        """Return the immutable agent configuration."""
        return self._config

    def to_config(self) -> dict[str, Any]:
        """Serialize the agent without array state."""
        return {
            "type": type(self).__name__,
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> RecurrentTraceActorCriticAgent:
        """Reconstruct an agent from :meth:`to_config` output."""
        payload = _copy_config_mapping("serialized agent", config)
        agent_type = payload.pop("type", cls.__name__)
        if type(agent_type) is not str or agent_type != cls.__name__:
            raise ValueError("unsupported agent type")
        if "config" in payload:
            raw_config = payload.pop("config")
            if payload:
                raise ValueError("serialized agent fields do not match the schema")
        else:
            raw_config = payload
        if not isinstance(raw_config, Mapping):
            raise ValueError("serialized config must be a mapping")
        return cls(RecurrentTraceActorCriticConfig.from_config(raw_config))

    def init(self, feature_dim: int, key: Array) -> RecurrentTraceActorCriticState:
        """Initialize independent actor/critic parameters and streaming state."""
        feature_dim = _require_int32("feature_dim", feature_dim, minimum=1)
        resources = self._config.state_resource_budget(feature_dim)
        _preflight_update_working_set(self._config, resources)
        actor_key, critic_key, policy_key = jr.split(key, 3)
        actor_params = initialize_rtu_network_parameters(
            actor_key,
            feature_dim,
            self._config.n_actions,
            self._config,
        )
        critic_params = initialize_rtu_network_parameters(
            critic_key,
            feature_dim,
            1,
            self._config,
        )
        hidden_size = self._config.hidden_size
        return RecurrentTraceActorCriticState(
            actor_params=actor_params,
            critic_params=critic_params,
            actor_rtu_state=zero_rtu_state(hidden_size),
            critic_rtu_state=zero_rtu_state(hidden_size),
            actor_sensitivities=zero_rtu_sensitivities(
                hidden_size,
                self._config.encoder_width,
            ),
            critic_sensitivities=zero_rtu_sensitivities(
                hidden_size,
                self._config.encoder_width,
            ),
            actor_taylor_trace=(
                zero_rtu_sensitivities(hidden_size, self._config.encoder_width)
                if self._config.rtrl_taylor_correction
                else None
            ),
            critic_taylor_trace=(
                zero_rtu_sensitivities(hidden_size, self._config.encoder_width)
                if self._config.rtrl_taylor_correction
                else None
            ),
            actor_traces=_zero_network_like(actor_params),
            critic_traces=_zero_network_like(critic_params),
            actor_second_moments=(
                _zero_network_like(actor_params) if self._config.adaptive_obgd else None
            ),
            critic_second_moments=(
                _zero_network_like(critic_params) if self._config.adaptive_obgd else None
            ),
            observation_statistics=_initial_running_statistics((feature_dim,)),
            reward_statistics=_initial_reward_statistics(),
            last_normalized_observation=jnp.zeros(
                (feature_dim,),
                dtype=jnp.float32,
            ),
            last_observation=jnp.zeros((feature_dim,), dtype=jnp.float32),
            last_action=jnp.asarray(-1, dtype=jnp.int32),
            rng_key=policy_key,
            step_count=jnp.asarray(0, dtype=jnp.int32),
            started=jnp.asarray(False),
        )

    def _network_output(
        self,
        params: RTUNetworkParameters,
        rtu_state: RTUState,
    ) -> Array:
        """Evaluate one configured network head."""
        return rtu_network_output(
            params,
            rtu_state,
            layer_norm_epsilon=self._config.layer_norm_epsilon,
            negative_slope=self._config.leaky_relu_slope,
        )

    def _network_encode(
        self,
        params: RTUNetworkParameters,
        observation: Array,
    ) -> Array:
        """Evaluate one configured sparse input block."""
        return rtu_network_encode(
            params,
            observation,
            layer_norm_epsilon=self._config.layer_norm_epsilon,
            negative_slope=self._config.leaky_relu_slope,
        )

    def _normalize_observation(
        self,
        state: RecurrentTraceActorCriticState,
        observation: Array,
    ) -> tuple[Array, RunningStatistics]:
        """Update population moments with the observation, then normalize it."""
        observation = jnp.asarray(
            observation,
            dtype=state.observation_statistics.mean.dtype,
        )
        statistics = _update_running_statistics(
            state.observation_statistics,
            observation,
        )
        if self._config.normalize_observations:
            normalized = (observation - statistics.mean) / (
                _population_standard_deviation(
                    statistics.m2,
                    statistics.sample_count,
                    self._config.normalization_epsilon,
                )
            )
        else:
            normalized = observation
        return normalized, statistics

    def _normalize_reward(
        self,
        state: RecurrentTraceActorCriticState,
        reward: Array,
        boundary: Array,
    ) -> tuple[Array, RewardStatistics]:
        """Scale reward by current discounted-return population moments."""
        statistics = state.reward_statistics
        reward = jnp.reshape(
            jnp.asarray(reward, dtype=statistics.mean.dtype),
            (),
        )
        continuing = 1.0 - boundary.astype(reward.dtype)
        gamma = jnp.asarray(self._config.gamma, dtype=reward.dtype)
        bootstrap_return = jnp.where(
            (continuing == 0.0) | (gamma == 0.0),
            jnp.zeros_like(statistics.discounted_return),
            gamma * statistics.discounted_return,
        )
        discounted_return = reward + bootstrap_return
        sample_count, can_increment = _saturating_int32_increment(
            statistics.sample_count,
        )
        delta = discounted_return - statistics.mean
        candidate_mean = statistics.mean + delta / sample_count
        delta_after = discounted_return - candidate_mean
        candidate_m2 = statistics.m2 + delta * delta_after
        mean = jnp.where(can_increment, candidate_mean, statistics.mean)
        m2 = jnp.where(can_increment, candidate_m2, statistics.m2)
        if self._config.normalize_rewards:
            normalized = reward / _population_standard_deviation(
                m2,
                sample_count,
                self._config.normalization_epsilon,
            )
        else:
            normalized = reward
        next_statistics = RewardStatistics(
            sample_count=sample_count,
            mean=mean,
            m2=m2,
            discounted_return=jnp.where(
                continuing == 0.0,
                jnp.zeros_like(discounted_return),
                discounted_return,
            ),
        )
        return normalized, next_statistics

    @functools.partial(jax.jit, static_argnums=(0,))
    def policy(
        self,
        state: RecurrentTraceActorCriticState,
    ) -> Array:
        """Return the softmax policy of the already-advanced actor state."""
        logits = self._network_output(state.actor_params, state.actor_rtu_state)
        return jax.nn.softmax(logits / self._config.temperature)

    @functools.partial(jax.jit, static_argnums=(0,))
    def value(
        self,
        state: RecurrentTraceActorCriticState,
    ) -> Array:
        """Return the scalar value represented by the current critic state."""
        return jnp.reshape(
            self._network_output(state.critic_params, state.critic_rtu_state)[0],
            (),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def select_action(
        self,
        state: RecurrentTraceActorCriticState,
    ) -> tuple[Array, Array, Array]:
        """Sample from the current recurrent policy without advancing it."""
        logits = self._network_output(state.actor_params, state.actor_rtu_state)
        tempered_logits = logits / self._config.temperature
        probabilities = jax.nn.softmax(tempered_logits)
        key, sample_key = jr.split(state.rng_key)
        action = jr.categorical(sample_key, tempered_logits).astype(jnp.int32)
        return action, key, probabilities

    def _advance_actor(
        self,
        state: RecurrentTraceActorCriticState,
        inputs: Array,
        *,
        reset: Array,
        parameter_delta: RTUParameters | None,
    ) -> tuple[RTUState, RTUSensitivities, RTUSensitivities | None]:
        """Advance actor recurrence, optionally from a zero episode state."""
        hidden_size = self._config.hidden_size
        input_dim = inputs.shape[0]
        zero_sensitivities = zero_rtu_sensitivities(hidden_size, input_dim)
        source_state = jax.tree_util.tree_map(
            lambda current, zero: jnp.where(reset, zero, current),
            state.actor_rtu_state,
            zero_rtu_state(hidden_size),
        )
        source_sensitivities = jax.tree_util.tree_map(
            lambda current, zero: jnp.where(reset, zero, current),
            state.actor_sensitivities,
            zero_sensitivities,
        )
        if self._config.rtrl_taylor_correction:
            if state.actor_taylor_trace is None:
                raise ValueError("Taylor-enabled actor state requires a Taylor trace")
            if parameter_delta is None:
                raise ValueError("Taylor-enabled actor advance requires a parameter delta")
            source_taylor_trace = jax.tree_util.tree_map(
                lambda current, zero: jnp.where(reset, zero, current),
                state.actor_taylor_trace,
                zero_sensitivities,
            )
            effective_delta = jax.tree_util.tree_map(
                lambda delta: jnp.where(reset, jnp.zeros_like(delta), delta),
                parameter_delta,
            )
            return rtu_taylor_step(
                state.actor_params.rtu,
                cast(RTUState, source_state),
                cast(RTUSensitivities, source_sensitivities),
                cast(RTUSensitivities, source_taylor_trace),
                cast(RTUParameters, effective_delta),
                inputs,
                epsilon=self._config.rtu_epsilon,
            )
        next_state, next_sensitivities = rtu_step(
            state.actor_params.rtu,
            cast(RTUState, source_state),
            cast(RTUSensitivities, source_sensitivities),
            inputs,
            epsilon=self._config.rtu_epsilon,
        )
        return next_state, next_sensitivities, None

    def _advance_critic(
        self,
        state: RecurrentTraceActorCriticState,
        inputs: Array,
        *,
        reset: Array,
        parameter_delta: RTUParameters | None,
    ) -> tuple[RTUState, RTUSensitivities, RTUSensitivities | None]:
        """Advance critic recurrence, optionally from a zero episode state."""
        hidden_size = self._config.hidden_size
        input_dim = inputs.shape[0]
        zero_sensitivities = zero_rtu_sensitivities(hidden_size, input_dim)
        source_state = jax.tree_util.tree_map(
            lambda current, zero: jnp.where(reset, zero, current),
            state.critic_rtu_state,
            zero_rtu_state(hidden_size),
        )
        source_sensitivities = jax.tree_util.tree_map(
            lambda current, zero: jnp.where(reset, zero, current),
            state.critic_sensitivities,
            zero_sensitivities,
        )
        if self._config.rtrl_taylor_correction:
            if state.critic_taylor_trace is None:
                raise ValueError("Taylor-enabled critic state requires a Taylor trace")
            if parameter_delta is None:
                raise ValueError("Taylor-enabled critic advance requires a parameter delta")
            source_taylor_trace = jax.tree_util.tree_map(
                lambda current, zero: jnp.where(reset, zero, current),
                state.critic_taylor_trace,
                zero_sensitivities,
            )
            effective_delta = jax.tree_util.tree_map(
                lambda delta: jnp.where(reset, jnp.zeros_like(delta), delta),
                parameter_delta,
            )
            return rtu_taylor_step(
                state.critic_params.rtu,
                cast(RTUState, source_state),
                cast(RTUSensitivities, source_sensitivities),
                cast(RTUSensitivities, source_taylor_trace),
                cast(RTUParameters, effective_delta),
                inputs,
                epsilon=self._config.rtu_epsilon,
            )
        next_state, next_sensitivities = rtu_step(
            state.critic_params.rtu,
            cast(RTUState, source_state),
            cast(RTUSensitivities, source_sensitivities),
            inputs,
            epsilon=self._config.rtu_epsilon,
        )
        return next_state, next_sensitivities, None

    def start(
        self,
        state: RecurrentTraceActorCriticState,
        observation: Array,
    ) -> tuple[RecurrentTraceActorCriticState, Array, Array]:
        """Validate, reset recurrent episode state, consume an observation, and act."""
        observation_array = _validated_finite_array(
            "observation",
            observation,
            expected_shape=state.last_observation.shape,
        )
        return cast(
            tuple[RecurrentTraceActorCriticState, Array, Array],
            self._start(state, observation_array),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _start(
        self,
        state: RecurrentTraceActorCriticState,
        observation: Array,
    ) -> tuple[RecurrentTraceActorCriticState, Array, Array]:
        """Jitted implementation of one checked episode start."""
        observation = _runtime_checked_value(
            observation,
            jnp.all(jnp.isfinite(observation)),
            "observation must contain only finite values",
        )
        normalized, statistics = self._normalize_observation(state, observation)
        actor_inputs = self._network_encode(state.actor_params, normalized)
        critic_inputs = self._network_encode(state.critic_params, normalized)
        reset = jnp.asarray(True)
        actor_parameter_delta = (
            _zero_rtu_parameters_like(state.actor_params.rtu)
            if self._config.rtrl_taylor_correction
            else None
        )
        critic_parameter_delta = (
            _zero_rtu_parameters_like(state.critic_params.rtu)
            if self._config.rtrl_taylor_correction
            else None
        )
        actor_rtu_state, actor_sensitivities, actor_taylor_trace = self._advance_actor(
            state,
            actor_inputs,
            reset=reset,
            parameter_delta=actor_parameter_delta,
        )
        critic_rtu_state, critic_sensitivities, critic_taylor_trace = self._advance_critic(
            state,
            critic_inputs,
            reset=reset,
            parameter_delta=critic_parameter_delta,
        )
        advanced = state.replace(
            actor_rtu_state=actor_rtu_state,
            critic_rtu_state=critic_rtu_state,
            actor_sensitivities=actor_sensitivities,
            critic_sensitivities=critic_sensitivities,
            actor_taylor_trace=actor_taylor_trace,
            critic_taylor_trace=critic_taylor_trace,
            actor_traces=_zero_network_like(state.actor_params),
            critic_traces=_zero_network_like(state.critic_params),
            observation_statistics=statistics,
            reward_statistics=state.reward_statistics._replace(
                discounted_return=jnp.zeros_like(state.reward_statistics.discounted_return)
            ),
            last_normalized_observation=normalized,
            last_observation=jnp.asarray(observation, dtype=jnp.float32),
            started=jnp.asarray(True),
        )
        action, key, probabilities = self.select_action(advanced)
        return (
            advanced.replace(last_action=action, rng_key=key),
            action,
            probabilities,
        )

    def _critic_gradient(
        self,
        state: RecurrentTraceActorCriticState,
    ) -> tuple[Array, RTUNetworkParameters]:
        """Return value with retained RTU-history and local encoder gradients."""

        def value_objective(output: Array) -> Array:
            return output[0]

        return exact_rtrl_gradient(
            state.critic_params,
            state.critic_rtu_state,
            state.critic_sensitivities,
            value_objective,
            encoder_input=state.last_normalized_observation,
            rtu_epsilon=self._config.rtu_epsilon,
            layer_norm_epsilon=self._config.layer_norm_epsilon,
            negative_slope=self._config.leaky_relu_slope,
        )

    def _actor_gradient(
        self,
        state: RecurrentTraceActorCriticState,
        td_error: Array,
    ) -> tuple[Array, RTUNetworkParameters]:
        """Return entropy-regularized log-policy objective and its gradient."""
        action = state.last_action
        entropy_sign = jax.lax.stop_gradient(jnp.sign(td_error))

        def policy_objective(logits: Array) -> Array:
            log_probabilities = jax.nn.log_softmax(logits / self._config.temperature)
            probabilities = jnp.exp(log_probabilities)
            entropy = -jnp.sum(probabilities * log_probabilities)
            return (
                log_probabilities[action]
                + self._config.entropy_coefficient * entropy_sign * entropy
            )

        return exact_rtrl_gradient(
            state.actor_params,
            state.actor_rtu_state,
            state.actor_sensitivities,
            policy_objective,
            encoder_input=state.last_normalized_observation,
            rtu_epsilon=self._config.rtu_epsilon,
            layer_norm_epsilon=self._config.layer_norm_epsilon,
            negative_slope=self._config.leaky_relu_slope,
        )

    def update(
        self,
        state: RecurrentTraceActorCriticState,
        reward: Array,
        observation: Array,
        terminated: Array | None = None,
        discount: Array | None = None,
        episode_boundary: Array | None = None,
        reset_observation: Array | None = None,
    ) -> RecurrentTraceActorCriticUpdateResult:
        """Validate and consume one transition.

        ``observation`` is always advanced from the current critic history for
        the bootstrap target.  On an episode boundary, stored recurrence and
        traces reset independently; ``reset_observation`` optionally supplies
        the first observation to process from zeros.  Omitting it reuses
        ``observation`` for backward compatibility.  When both observations
        are supplied on a boundary, both enter observation-normalization
        statistics in that order.

        ``discount`` takes precedence over ``terminated`` only for the numeric
        TD bootstrap.  Boundary precedence is ``episode_boundary``, then
        ``terminated``, then an explicitly supplied zero ``discount``.  A call
        that supplies none of them is continuing even when configured
        ``gamma`` is zero.

        Examples:
            A true environment termination can use ``terminated=True``.
            A time-limit truncation can use
            ``discount=config.gamma, episode_boundary=True`` and pass the
            reset observation separately.

        Raises:
            RuntimeError: If called eagerly before :meth:`start`.
            ValueError: If eager transition inputs violate their scalar, dtype,
                shape, or range contracts.
            jax.errors.JaxRuntimeError: If a dynamic lifecycle or discount
                violation occurs when an outer ``jax.jit`` wraps this complete
                public method.  The wrapped ``ValueError`` message identifies
                the failed contract; the error surfaces when the result is
                synchronized and no updated state is returned.
        """
        return self._validated_update(
            state,
            reward,
            observation,
            terminated,
            discount,
            episode_boundary,
            reset_observation,
            check_dynamic_started=True,
        )

    def update_from_started_state(
        self,
        state: RecurrentTraceActorCriticState,
        reward: Array,
        observation: Array,
        terminated: Array | None = None,
        discount: Array | None = None,
        episode_boundary: Array | None = None,
        reset_observation: Array | None = None,
    ) -> RecurrentTraceActorCriticUpdateResult:
        """Consume a transition when an enclosing scan proves lifecycle order.

        This entry point has the same eager validation and numerical semantics
        as :meth:`update`, including rejecting a concrete unstarted state.  It
        omits host-callback lifecycle and finiteness assertions from the
        compiled graph.  That omission lets a CUDA-only JAX runtime execute a
        long-lived scan without requiring a local CPU device solely to host
        ``jax.debug.callback``; eager calls still receive the ordinary input
        checks.

        Callers must establish that every traced lane was produced by
        :meth:`start` and that rewards and observations are finite before
        entering the enclosing transform.  Public code that cannot prove those
        invariants must use :meth:`update`.
        """
        return self._validated_update(
            state,
            reward,
            observation,
            terminated,
            discount,
            episode_boundary,
            reset_observation,
            check_dynamic_started=False,
        )

    def _validated_update(
        self,
        state: RecurrentTraceActorCriticState,
        reward: Array,
        observation: Array,
        terminated: Array | None,
        discount: Array | None,
        episode_boundary: Array | None,
        reset_observation: Array | None,
        *,
        check_dynamic_started: bool,
    ) -> RecurrentTraceActorCriticUpdateResult:
        """Validate one public transition and dispatch the compiled kernel."""
        started = state.started
        if started.shape != ():
            raise ValueError("state.started must be scalar-shaped")
        if started.dtype != jnp.bool_:
            raise ValueError("state.started must have boolean dtype")
        if not isinstance(started, jax.core.Tracer) and not bool(started):
            raise RuntimeError("start must be called before update")

        reward_scalar = _validated_scalar(
            "reward",
            reward,
            boolean=False,
        )
        expected_observation_shape = state.last_observation.shape
        observation_array = _validated_finite_array(
            "observation",
            observation,
            expected_shape=expected_observation_shape,
        )
        if terminated is not None:
            terminated = _validated_scalar(
                "terminated",
                terminated,
                boolean=True,
            )
        if discount is not None:
            discount = _validated_scalar(
                "discount",
                discount,
                boolean=False,
                unit_interval=True,
            )
        if episode_boundary is not None:
            episode_boundary = _validated_scalar(
                "episode_boundary",
                episode_boundary,
                boolean=True,
            )
        if reset_observation is not None:
            reset_observation = _validated_finite_array(
                "reset_observation",
                reset_observation,
                expected_shape=expected_observation_shape,
            )
        return cast(
            RecurrentTraceActorCriticUpdateResult,
            self._update(
                state,
                reward_scalar,
                observation_array,
                terminated,
                discount,
                episode_boundary,
                reset_observation,
                check_dynamic_started,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0, 8))
    def _update(
        self,
        state: RecurrentTraceActorCriticState,
        reward: Array,
        observation: Array,
        terminated: Array | None,
        discount: Array | None,
        episode_boundary: Array | None,
        reset_observation: Array | None,
        check_dynamic_started: bool,
    ) -> RecurrentTraceActorCriticUpdateResult:
        """Jitted implementation of one checked streaming update."""
        if check_dynamic_started:
            reward = _runtime_checked_value(
                reward,
                state.started,
                "start must be called before update",
            )
            reward = _runtime_checked_value(
                reward,
                jnp.isfinite(reward),
                "reward must be finite",
            )
            observation = _runtime_checked_value(
                observation,
                jnp.all(jnp.isfinite(observation)),
                "observation must contain only finite values",
            )
            if reset_observation is not None:
                reset_observation = _runtime_checked_value(
                    reset_observation,
                    jnp.all(jnp.isfinite(reset_observation)),
                    "reset_observation must contain only finite values",
                )
        if discount is not None:
            discount = _runtime_checked_value(
                discount,
                jnp.isfinite(discount) & (discount >= 0.0) & (discount <= 1.0),
                "discount must be finite and lie in [0, 1]",
            )
        if discount is None:
            if terminated is None:
                transition_discount = jnp.asarray(
                    self._config.gamma,
                    dtype=jnp.float32,
                )
            else:
                transition_discount = jnp.where(
                    terminated,
                    jnp.asarray(0.0, dtype=jnp.float32),
                    jnp.asarray(self._config.gamma, dtype=jnp.float32),
                )
        else:
            transition_discount = jnp.reshape(
                jnp.asarray(discount, dtype=jnp.float32),
                (),
            )
        if episode_boundary is None:
            if terminated is not None:
                boundary = jnp.reshape(
                    jnp.asarray(terminated, dtype=jnp.bool_),
                    (),
                )
            elif discount is not None:
                boundary = transition_discount == 0.0
            else:
                boundary = jnp.asarray(False)
        else:
            boundary = jnp.reshape(
                jnp.asarray(episode_boundary, dtype=jnp.bool_),
                (),
            )
        current_logits = self._network_output(
            state.actor_params,
            state.actor_rtu_state,
        )
        current_log_policy = jax.nn.log_softmax(current_logits / self._config.temperature)
        current_policy = jnp.exp(current_log_policy)
        current_entropy = -jnp.sum(current_policy * current_log_policy)

        bootstrap_observation, bootstrap_statistics = self._normalize_observation(
            state, observation
        )
        if reset_observation is None:
            stored_observation = bootstrap_observation
            observation_statistics = bootstrap_statistics
            stored_raw_observation = jnp.asarray(
                observation,
                dtype=jnp.float32,
            )
        else:
            reset_statistics_state = state.replace(
                observation_statistics=bootstrap_statistics,
            )
            normalized_reset_observation, reset_statistics = self._normalize_observation(
                reset_statistics_state,
                reset_observation,
            )
            stored_observation = jnp.where(
                boundary,
                normalized_reset_observation,
                bootstrap_observation,
            )
            observation_statistics = RunningStatistics(
                sample_count=jnp.where(
                    boundary,
                    reset_statistics.sample_count,
                    bootstrap_statistics.sample_count,
                ),
                mean=jnp.where(
                    boundary,
                    reset_statistics.mean,
                    bootstrap_statistics.mean,
                ),
                m2=jnp.where(
                    boundary,
                    reset_statistics.m2,
                    bootstrap_statistics.m2,
                ),
            )
            stored_raw_observation = jnp.where(
                boundary,
                jnp.asarray(reset_observation, dtype=jnp.float32),
                jnp.asarray(observation, dtype=jnp.float32),
            )
        normalized_reward, reward_statistics = self._normalize_reward(
            state,
            reward,
            boundary,
        )

        bootstrap_critic_inputs = self._network_encode(
            state.critic_params,
            bootstrap_observation,
        )
        bootstrap_critic_state = rtu_forward(
            state.critic_params.rtu,
            state.critic_rtu_state,
            bootstrap_critic_inputs,
            epsilon=self._config.rtu_epsilon,
        )

        value, critic_gradient = self._critic_gradient(state)
        next_value = jnp.reshape(
            self._network_output(
                state.critic_params,
                bootstrap_critic_state,
            )[0],
            (),
        )
        bootstrap = jnp.where(
            transition_discount == 0.0,
            jnp.zeros_like(next_value),
            transition_discount * jax.lax.stop_gradient(next_value),
        )
        td_error = normalized_reward + bootstrap - value
        _, actor_gradient = self._actor_gradient(state, td_error)

        # The terminating reward must still credit traces accumulated within
        # the episode.  Stream AC(lambda) therefore applies gamma*lambda first
        # and clears the stored trace only after the terminal update.
        trace_discount = jnp.where(
            boundary,
            jnp.asarray(self._config.gamma, dtype=transition_discount.dtype),
            transition_discount,
        )
        actor_decay = trace_discount * self._config.actor_lamda
        critic_decay = trace_discount * self._config.critic_lamda
        actor_traces = cast(
            RTUNetworkParameters,
            jax.tree_util.tree_map(
                lambda trace, gradient: (
                    jnp.where(actor_decay == 0.0, jnp.zeros_like(trace), actor_decay * trace)
                    + gradient
                ),
                state.actor_traces,
                actor_gradient,
            ),
        )
        critic_traces = cast(
            RTUNetworkParameters,
            jax.tree_util.tree_map(
                lambda trace, gradient: (
                    jnp.where(critic_decay == 0.0, jnp.zeros_like(trace), critic_decay * trace)
                    + gradient
                ),
                state.critic_traces,
                critic_gradient,
            ),
        )

        step_count, _ = _saturating_int32_increment(state.step_count)
        actor_updates: Any
        critic_updates: Any
        actor_second_moments: RTUNetworkParameters | None
        critic_second_moments: RTUNetworkParameters | None
        if self._config.adaptive_obgd:
            if state.actor_second_moments is None:
                raise ValueError("adaptive-ObGD actor state requires a second-moment tree")
            if state.critic_second_moments is None:
                raise ValueError("adaptive-ObGD critic state requires a second-moment tree")
            actor_adaptive_obgd = adaptive_obgd_update(
                actor_traces,
                state.actor_second_moments,
                td_error,
                alpha=self._config.actor_alpha,
                kappa=self._config.actor_kappa,
                beta2=self._config.beta2,
                epsilon=self._config.epsilon,
                step=step_count,
            )
            critic_adaptive_obgd = adaptive_obgd_update(
                critic_traces,
                state.critic_second_moments,
                td_error,
                alpha=self._config.critic_alpha,
                kappa=self._config.critic_kappa,
                beta2=self._config.beta2,
                epsilon=self._config.epsilon,
                step=step_count,
            )
            actor_second_moments = cast(
                RTUNetworkParameters,
                actor_adaptive_obgd.second_moment,
            )
            critic_second_moments = cast(
                RTUNetworkParameters,
                critic_adaptive_obgd.second_moment,
            )
            actor_updates = actor_adaptive_obgd.updates
            critic_updates = critic_adaptive_obgd.updates
            actor_step_size = actor_adaptive_obgd.step_size
            critic_step_size = critic_adaptive_obgd.step_size
            actor_obgd_scale = actor_adaptive_obgd.scale
            critic_obgd_scale = critic_adaptive_obgd.scale
            actor_update_applied = actor_adaptive_obgd.update_applied
            critic_update_applied = critic_adaptive_obgd.update_applied
        else:
            actor_legacy_obgd = obgd_update(
                actor_traces,
                td_error,
                alpha=self._config.actor_alpha,
                kappa=self._config.actor_kappa,
            )
            critic_legacy_obgd = obgd_update(
                critic_traces,
                td_error,
                alpha=self._config.critic_alpha,
                kappa=self._config.critic_kappa,
            )
            actor_second_moments = state.actor_second_moments
            critic_second_moments = state.critic_second_moments
            actor_updates = actor_legacy_obgd.updates
            critic_updates = critic_legacy_obgd.updates
            actor_step_size = actor_legacy_obgd.step_size
            critic_step_size = critic_legacy_obgd.step_size
            actor_obgd_scale = actor_legacy_obgd.scale
            critic_obgd_scale = critic_legacy_obgd.scale
            actor_update_applied = actor_legacy_obgd.update_applied
            critic_update_applied = critic_legacy_obgd.update_applied
        actor_params = cast(
            RTUNetworkParameters,
            jax.tree_util.tree_map(
                lambda parameter, change: parameter + change,
                state.actor_params,
                actor_updates,
            ),
        )
        actor_params = _project_rtu_parameter_logs(actor_params)
        critic_params = cast(
            RTUNetworkParameters,
            jax.tree_util.tree_map(
                lambda parameter, change: parameter + change,
                state.critic_params,
                critic_updates,
            ),
        )
        critic_params = _project_rtu_parameter_logs(critic_params)
        if self._config.rtrl_taylor_correction:
            actor_rtu_delta = cast(
                RTUParameters,
                jax.tree_util.tree_map(
                    lambda current, previous: current - previous,
                    actor_params.rtu,
                    state.actor_params.rtu,
                ),
            )
            critic_rtu_delta = cast(
                RTUParameters,
                jax.tree_util.tree_map(
                    lambda current, previous: current - previous,
                    critic_params.rtu,
                    state.critic_params.rtu,
                ),
            )
        else:
            actor_rtu_delta = None
            critic_rtu_delta = None

        stored_actor_traces = cast(
            RTUNetworkParameters,
            jax.tree_util.tree_map(
                lambda trace: jnp.where(boundary, jnp.zeros_like(trace), trace),
                actor_traces,
            ),
        )
        stored_critic_traces = cast(
            RTUNetworkParameters,
            jax.tree_util.tree_map(
                lambda trace: jnp.where(boundary, jnp.zeros_like(trace), trace),
                critic_traces,
            ),
        )

        parameter_state = state.replace(
            actor_params=actor_params,
            critic_params=critic_params,
            actor_second_moments=actor_second_moments,
            critic_second_moments=critic_second_moments,
        )
        next_actor_inputs = self._network_encode(
            actor_params,
            stored_observation,
        )
        next_critic_inputs = self._network_encode(
            critic_params,
            stored_observation,
        )
        (
            next_actor_state,
            next_actor_sensitivities,
            next_actor_taylor_trace,
        ) = self._advance_actor(
            parameter_state,
            next_actor_inputs,
            reset=boundary,
            parameter_delta=actor_rtu_delta,
        )
        (
            next_critic_state,
            next_critic_sensitivities,
            next_critic_taylor_trace,
        ) = self._advance_critic(
            parameter_state,
            next_critic_inputs,
            reset=boundary,
            parameter_delta=critic_rtu_delta,
        )

        proposed_state = parameter_state.replace(
            actor_rtu_state=next_actor_state,
            critic_rtu_state=next_critic_state,
            actor_sensitivities=next_actor_sensitivities,
            critic_sensitivities=next_critic_sensitivities,
            actor_taylor_trace=next_actor_taylor_trace,
            critic_taylor_trace=next_critic_taylor_trace,
            actor_traces=stored_actor_traces,
            critic_traces=stored_critic_traces,
            observation_statistics=observation_statistics,
            reward_statistics=reward_statistics,
            last_normalized_observation=stored_observation,
            last_observation=stored_raw_observation,
            step_count=step_count,
        )
        candidate_applied = (
            actor_update_applied
            & critic_update_applied
            & _floating_tree_is_finite(
                state.replace(
                    reward_statistics=state.reward_statistics._replace(
                        discounted_return=jnp.where(
                            boundary,
                            jnp.zeros_like(state.reward_statistics.discounted_return),
                            state.reward_statistics.discounted_return,
                        )
                    )
                )
            )
            & _floating_tree_is_finite(proposed_state)
            & state.started
            & (state.last_action >= 0)
            & (state.last_action < self._config.n_actions)
        )
        action_state = jax.lax.cond(
            candidate_applied,
            lambda: proposed_state,
            lambda: state,
        )
        next_action, key, next_policy = self.select_action(action_state)
        committed_state = proposed_state.replace(
            last_action=next_action,
            rng_key=key,
        )
        update_applied = (
            candidate_applied
            & _floating_tree_is_finite(committed_state)
            & jnp.all(jnp.isfinite(next_policy))
            & (next_action >= 0)
            & (next_action < self._config.n_actions)
        )
        updated = jax.lax.cond(
            update_applied,
            lambda: committed_state,
            lambda: state,
        )
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return RecurrentTraceActorCriticUpdateResult(
            state=updated,
            action=jnp.where(
                update_applied,
                next_action,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            policy=jnp.where(update_applied, next_policy, jnp.zeros_like(next_policy)),
            value=jnp.where(update_applied, value, zero),
            next_value=jnp.where(update_applied & jnp.isfinite(next_value), next_value, zero),
            td_error=jnp.where(update_applied, td_error, zero),
            entropy=jnp.where(update_applied, current_entropy, zero),
            normalized_reward=jnp.where(update_applied, normalized_reward, zero),
            actor_step_size=jnp.where(update_applied, actor_step_size, zero),
            critic_step_size=jnp.where(update_applied, critic_step_size, zero),
            actor_obgd_scale=jnp.where(update_applied, actor_obgd_scale, zero),
            critic_obgd_scale=jnp.where(update_applied, critic_obgd_scale, zero),
            update_applied=update_applied,
        )


# Short aliases make integration with existing Alberta naming less cumbersome.
RTUActorCriticConfig = RecurrentTraceActorCriticConfig
RTUActorCriticState = RecurrentTraceActorCriticState
RTUActorCriticUpdateResult = RecurrentTraceActorCriticUpdateResult
RTUActorCriticAgent = RecurrentTraceActorCriticAgent
