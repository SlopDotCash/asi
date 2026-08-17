"""Source-profiled first-order Utility-based Perturbed Gradient Descent.

This module implements the weight-wise UPGD update from Elsayed and Mahmood,
ICLR 2024, together with the two distinct implementations published in the
authors' repository, as a small JAX PyTree transform.  It is intentionally
separate from ``core.upgd.UPGDLearner``: that historical Alberta learner uses
absolute ``|w g|`` utility and a power gate, and changing it in place would
invalidate existing checkpoints and results.

The protecting update is

``w <- (1 - alpha * weight_decay) * w
       - k * alpha * (gradient + noise) * (1 - scaled_utility)``.

The paper and experiment code use ``k=1``.  The repository README's advertised
implementation uses ``k=2`` while retaining one-alpha decoupled weight decay.
The non-protecting ablation gates only the perturbation.  Utility is the signed
first-order Taylor approximation ``-gradient * weight``, accumulated with an
EMA and bias correction.

The source profiles deliberately keep source discrepancies visible:

``paper_global``
    Corrected global maximum, one-alpha direction, and the paper's global
    time-step clock.
``official_readme_global``
    Uncorrected global maximum and the README's two-alpha gated direction.
``official_experiment_global``
    Uncorrected global maximum and the experiment code's one-alpha direction.
``official_experiment_local``
    Corrected row-local L2 normalization along the final axis with PyTorch's
    ``1e-12`` normalization floor.
``paper_local_literal``
    Appendix E's literal corrected numerator divided by the raw-EMA row norm.
``safe_extended``
    Explicit global or local normalization with numerical guards, masks,
    missing/non-finite-gradient skipping, and active-element clocks.

Source profiles are equation-exact only for finite, all-active gradients.  They
reject masks and Python ``None`` gradients.  Dynamic non-finite values fail
closed because raising from a JIT-compiled update is not portable.

Reference:
    Elsayed, M. & Mahmood, A. R. (2024). Addressing Loss of Plasticity and
    Catastrophic Forgetting in Continual Learning. ICLR 2024.
    https://openreview.net/forum?id=sKPzAXoylB

    Released MIT implementation audited at commit
    ``b75e90ad4b09c28971ac9dbb902a8fd86709b28c``:
    https://github.com/mohmdelsayed/upgd
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Int

from alberta_framework.core._float32_scalars import validated_float32_scalar_with_ratio

UPGDMode = Literal["protecting", "non_protecting"]
UPGDNormalization = Literal["global", "local"]
UPGDProfile = Literal[
    "paper_global",
    "official_readme_global",
    "official_experiment_global",
    "official_experiment_local",
    "paper_local_literal",
    "safe_extended",
]

_SOURCE_PROFILES = frozenset(
    {
        "paper_global",
        "official_readme_global",
        "official_experiment_global",
        "official_experiment_local",
        "paper_local_literal",
    }
)
_GLOBAL_PROFILES = frozenset(
    {
        "paper_global",
        "official_readme_global",
        "official_experiment_global",
    }
)
_RAW_GLOBAL_PROFILES = frozenset(
    {
        "official_readme_global",
        "official_experiment_global",
    }
)
_INT32_MAX = 2_147_483_647
_ACTUAL_INT_TYPES: tuple[type, ...] = (
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
_ACTUAL_REAL_TYPES = frozenset(
    (*_ACTUAL_INT_TYPES, float, np.float16, np.float32, np.float64, np.longdouble)
)


@dataclass(frozen=True)
class _ParameterResources:
    scalars: int
    nbytes: int


def _copy_config_mapping(name: str, config: object) -> dict[str, Any]:
    """Copy a legacy-compatible mapping while normalizing hostile hooks."""
    if not isinstance(config, Mapping):
        raise ValueError(f"{name} must be a mapping")
    try:
        payload = dict(config)
    except Exception as error:
        raise ValueError(f"{name} must be a readable mapping") from error
    if any(type(key) is not str for key in payload):
        raise ValueError(f"{name} fields must be strings")
    return cast(dict[str, Any], payload)


def _array_metadata(name: str, value: object) -> tuple[tuple[int, ...], np.dtype[Any]]:
    """Read bounded host shape/dtype metadata without invoking JAX conversion."""
    if type(value) is float:
        raw_shape: object = ()
        raw_dtype: object = np.dtype(float)
    else:
        try:
            raw_shape = getattr(value, "shape")
            raw_dtype = getattr(value, "dtype")
        except Exception as error:
            raise ValueError(f"{name} must expose array shape and dtype metadata") from error
    if type(raw_shape) is not tuple:
        raise ValueError(f"{name} shape must be a tuple")
    shape: list[int] = []
    for dimension in raw_shape:
        if type(dimension) not in _ACTUAL_INT_TYPES:
            raise ValueError(f"{name} dimensions must be actual integers")
        size = operator.index(cast(SupportsIndex, dimension))
        if not 0 <= size <= _INT32_MAX:
            raise ValueError(f"{name} dimensions must fit signed int32")
        shape.append(size)
    try:
        dtype = np.dtype(
            jax.dtypes.canonicalize_dtype(np.dtype(cast(Any, raw_dtype)))
        )
    except Exception as error:
        raise ValueError(f"{name} must expose a supported array dtype") from error
    return tuple(shape), dtype


def _floating_parameter_metadata(
    name: str, value: object
) -> tuple[tuple[int, ...], np.dtype[Any]]:
    shape, dtype = _array_metadata(name, value)
    if not jnp.issubdtype(dtype, jnp.floating):
        raise ValueError(f"{name} must have floating-point dtype")
    return shape, dtype


def _matching_array(
    name: str,
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> Array:
    actual_shape, actual_dtype = _array_metadata(name, value)
    if actual_shape != shape:
        raise ValueError(f"{name} must match its parameter shape")
    if actual_dtype != dtype:
        raise TypeError(f"{name} dtype must match its parameter dtype")
    try:
        return jnp.asarray(value)
    except Exception as error:
        raise ValueError(f"{name} could not be converted to a JAX array") from error


def _coerced_numeric_array(
    name: str,
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    actual_shape, actual_dtype = _array_metadata(name, value)
    if actual_shape != shape:
        raise ValueError(f"{name} must match its parameter shape")
    if actual_dtype.kind not in {"i", "u", "f"}:
        raise TypeError(f"{name} must have real numeric dtype")
    try:
        return jnp.asarray(value, dtype=dtype)
    except Exception as error:
        raise ValueError(f"{name} could not be converted to a JAX array") from error


def _broadcast_bool_array(name: str, value: object, shape: tuple[int, ...]) -> Array:
    actual_shape, actual_dtype = _array_metadata(name, value)
    if actual_dtype != np.dtype(np.bool_):
        raise TypeError(f"{name} must have boolean dtype")
    try:
        if np.broadcast_shapes(actual_shape, shape) != shape:
            raise ValueError
    except ValueError as error:
        raise ValueError(f"{name} must broadcast to its parameter shape") from error
    try:
        return jnp.broadcast_to(jnp.asarray(value), shape)
    except Exception as error:
        raise ValueError(f"{name} could not be converted to a JAX array") from error


def _parameter_arrays(
    params: Any,
    *,
    owner: str,
    persistent_nbytes: Callable[[int, int], int],
    update_nbytes: Callable[[int, int], int],
    working_scalars: Callable[[int], int],
    working_nbytes: Callable[[int, int], int],
) -> tuple[list[Array], jax.tree_util.PyTreeDef, _ParameterResources]:
    leaves, structure = _flatten_with_none(params)
    if not leaves or any(value is None for value in leaves):
        raise ValueError("params must contain at least one array leaf")
    metadata = [
        _floating_parameter_metadata(f"{owner} parameter", value) for value in leaves
    ]
    scalars = sum(math.prod(shape) for shape, _ in metadata)
    nbytes = sum(math.prod(shape) * dtype.itemsize for shape, dtype in metadata)
    for resource_name, amount in (
        ("parameter_count", scalars),
        ("persistent_state_nbytes", persistent_nbytes(scalars, nbytes)),
        ("update_result_nbytes", update_nbytes(scalars, nbytes)),
        ("working_set_scalars", working_scalars(scalars)),
        ("working_set_nbytes", working_nbytes(scalars, nbytes)),
    ):
        if amount > _INT32_MAX:
            raise ValueError(f"derived {owner} {resource_name} must fit signed int32")
    arrays: list[Array] = []
    for value in leaves:
        try:
            arrays.append(jnp.asarray(value))
        except Exception as error:
            raise ValueError(f"{owner} parameter could not be converted to a JAX array") from error
    return arrays, structure, _ParameterResources(scalars=scalars, nbytes=nbytes)


def _validated_float32_scalar(
    name: str,
    value: object,
    *,
    positive: bool = False,
    lower: float | None = None,
    upper: float | None = None,
    upper_inclusive: bool = True,
) -> float:
    """Validate an exact trusted scalar before any user-defined ratio hook."""
    if type(value) not in _ACTUAL_REAL_TYPES:
        raise ValueError(f"{name} must be a finite real number")
    stored, numerator, denominator = validated_float32_scalar_with_ratio(
        name,
        value,
        positive=positive,
        lower=lower,
        upper=upper,
        upper_inclusive=upper_inclusive,
    )
    if numerator != 0 and abs(numerator) * (1 << 150) <= denominator:
        raise ValueError(f"{name} must remain nonzero once narrowed to float32")
    return stored


def _static_zero_scale(scale: float, value: Array) -> Array:
    """Skip a statically disabled EMA without changing enabled JAX graphs."""
    if scale == 0.0:
        return jnp.zeros_like(value)
    return scale * value


def _saturating_increment(value: Array, increment: Array | int = 1) -> Array:
    """Increment a non-negative int32 counter without lifetime wraparound."""

    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    increment_array = jnp.asarray(increment, dtype=jnp.int32)
    return jnp.where(
        value <= maximum - increment_array,
        value + increment_array,
        maximum,
    )


def _floating_leaves_are_finite(*trees: Any) -> Array:
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for tree in trees:
        for leaf in jax.tree.leaves(tree):
            array = jnp.asarray(leaf)
            if jnp.issubdtype(array.dtype, jnp.floating):
                valid = valid & jnp.all(jnp.isfinite(array))
    return valid


@dataclass(frozen=True)
class CanonicalUPGDConfig:
    """Configuration for canonical first-order UPGD.

    Args:
        step_size: Base gradient step size ``alpha``.
        utility_decay: EMA decay ``beta`` for signed utility.
        noise_std: Standard deviation ``sigma`` of Gaussian perturbations.
        weight_decay: Decoupled UPGD-W decay coefficient.
        mode: ``"protecting"`` gates gradient and noise;
            ``"non_protecting"`` gates noise only.
        profile: Named equation source.  Profiles select equations, not source
            constructor hyperparameter defaults.
        normalization: Required explicit choice for ``"safe_extended"``.
            Source profiles accept ``None`` or their fixed matching value so
            serialized configurations remain explicit without changing the
            cited equation.
        epsilon: Positive numerical floor used only by ``"safe_extended"``.
    """

    step_size: float = 1e-3
    utility_decay: float = 0.999
    noise_std: float = 1e-3
    weight_decay: float = 0.0
    mode: UPGDMode = "protecting"
    profile: UPGDProfile = "safe_extended"
    normalization: UPGDNormalization | None = "global"
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        step_size = _validated_float32_scalar(
            "step_size", self.step_size, lower=0.0, positive=True
        )
        utility_decay = _validated_float32_scalar(
            "utility_decay",
            self.utility_decay,
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
        )
        noise_std = _validated_float32_scalar("noise_std", self.noise_std, lower=0.0)
        weight_decay = _validated_float32_scalar(
            "weight_decay", self.weight_decay, lower=0.0
        )
        epsilon = _validated_float32_scalar(
            "epsilon", self.epsilon, lower=0.0, positive=True
        )
        object.__setattr__(self, "step_size", step_size)
        object.__setattr__(self, "utility_decay", utility_decay)
        object.__setattr__(self, "noise_std", noise_std)
        object.__setattr__(self, "weight_decay", weight_decay)
        object.__setattr__(self, "epsilon", epsilon)
        if type(self.mode) is not str or self.mode not in {"protecting", "non_protecting"}:
            raise ValueError("mode must be protecting or non_protecting")
        if type(self.profile) is not str or self.profile not in _SOURCE_PROFILES | {
            "safe_extended"
        }:
            raise ValueError(
                "profile must name a paper, official implementation, or safe_extended profile"
            )
        if self.profile == "safe_extended":
            if type(self.normalization) is not str or self.normalization not in {
                "global",
                "local",
            }:
                raise ValueError("safe_extended requires normalization=global or local")
        else:
            fixed_normalization = "global" if self.profile in _GLOBAL_PROFILES else "local"
            if self.normalization is not None and (
                type(self.normalization) is not str
                or self.normalization != fixed_normalization
            ):
                raise ValueError(
                    f"{self.profile} fixes normalization to {fixed_normalization}"
                )
        if self.profile == "official_readme_global" and self.mode != "protecting":
            raise ValueError("official_readme_global only defines protecting UPGD")

    @property
    def is_source_profile(self) -> bool:
        """Whether this configuration follows a cited source equation."""

        return self.profile in _SOURCE_PROFILES

    @property
    def resolved_normalization(self) -> UPGDNormalization:
        """Return the normalization fixed by the selected profile."""

        if self.profile == "safe_extended":
            if self.normalization is None:  # guarded by __post_init__
                raise AssertionError("safe_extended normalization was not validated")
            return self.normalization
        if self.profile in _GLOBAL_PROFILES:
            return "global"
        return "local"

    @property
    def direction_multiplier(self) -> float:
        """Return the source coefficient on the gated learning direction."""

        return 2.0 if self.profile == "official_readme_global" else 1.0

    @property
    def uses_global_clock(self) -> bool:
        """Whether bias correction uses the global optimizer step."""

        return self.is_source_profile

    def to_config(self) -> dict[str, object]:
        """Return a JSON-serializable configuration."""

        return {
            "type": "CanonicalUPGD",
            "step_size": self.step_size,
            "utility_decay": self.utility_decay,
            "noise_std": self.noise_std,
            "weight_decay": self.weight_decay,
            "mode": self.mode,
            "profile": self.profile,
            "normalization": self.normalization,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> CanonicalUPGDConfig:
        """Strictly reconstruct from the complete serialized schema."""

        payload = _copy_config_mapping("CanonicalUPGD config", config)
        expected = {
            "type",
            "step_size",
            "utility_decay",
            "noise_std",
            "weight_decay",
            "mode",
            "profile",
            "normalization",
            "epsilon",
        }
        if set(payload) != expected:
            raise ValueError("config fields do not match the CanonicalUPGD schema")
        type_name = payload.pop("type")
        if type(type_name) is not str or type_name != "CanonicalUPGD":
            raise ValueError("expected CanonicalUPGD config, got invalid value")
        try:
            return cls(**payload)
        except Exception as error:
            raise ValueError("CanonicalUPGD config values are invalid") from error


@chex.dataclass(frozen=True)
class CanonicalUPGDState:
    """Optimizer state for an arbitrary parameter PyTree."""

    utility_ema: Any
    utility_age: Any
    step: Int[Array, ""]


@chex.dataclass(frozen=True)
class CanonicalUPGDUpdate:
    """Result of one functional UPGD update."""

    params: Any
    state: CanonicalUPGDState
    next_key: Array
    scaled_utility: Any
    corrected_utility: Any
    perturbation: Any
    metrics: dict[str, Array]


def _flatten_with_none(tree: Any) -> tuple[list[Any], jax.tree_util.PyTreeDef]:
    """Flatten a PyTree while treating a missing gradient as a leaf."""

    leaves, structure = jax.tree_util.tree_flatten(
        tree,
        is_leaf=lambda value: value is None,
    )
    return list(leaves), structure


def _local_normalize(
    numerator: Array,
    denominator_utility: Array,
    active: Array,
    epsilon: float | None,
) -> Array:
    """Normalize active entries row-wise along a parameter leaf's final axis."""

    active_denominator = jnp.where(active, denominator_utility, 0.0)
    if numerator.ndim == 0:
        denominator = jnp.abs(active_denominator)
    else:
        denominator = jnp.linalg.norm(
            active_denominator,
            axis=-1,
            keepdims=True,
        )
    if epsilon is not None:
        denominator = jnp.maximum(denominator, epsilon)
    normalized = numerator / denominator
    return jnp.where(active, normalized, 0.0)


def _safe_signed_denominator(value: Array, epsilon: float) -> Array:
    """Floor a scalar magnitude without changing its sign."""

    signed_floor = jnp.where(value < 0.0, -epsilon, epsilon)
    return jnp.where(jnp.abs(value) >= epsilon, value, signed_floor)


class CanonicalUPGD:
    """Functional, JIT-compatible first-order UPGD PyTree transform."""

    def __init__(self, config: CanonicalUPGDConfig | None = None):
        self._config = CanonicalUPGDConfig() if config is None else config

    @property
    def config(self) -> CanonicalUPGDConfig:
        """Return the immutable optimizer configuration."""

        return self._config

    def to_config(self) -> dict[str, object]:
        """Serialize the optimizer configuration."""

        return self._config.to_config()

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> CanonicalUPGD:
        """Reconstruct an optimizer from serialized configuration."""

        return cls(CanonicalUPGDConfig.from_config(config))

    @staticmethod
    def _parameter_contract(
        params: Any,
    ) -> tuple[list[Array], jax.tree_util.PyTreeDef, _ParameterResources]:
        return _parameter_arrays(
            params,
            owner="CanonicalUPGD",
            persistent_nbytes=lambda scalars, nbytes: nbytes + 4 * scalars + 4,
            # params, state, three diagnostic trees, key, and six metrics.
            update_nbytes=lambda scalars, nbytes: 5 * nbytes + 4 * scalars + 33,
            working_scalars=lambda scalars: 22 * scalars + 32,
            working_nbytes=lambda scalars, nbytes: 23 * nbytes + 4 * scalars + 128,
        )

    @staticmethod
    def _state_contract(
        state: CanonicalUPGDState,
        params: Any,
    ) -> tuple[list[Array], list[Array], list[Array], jax.tree_util.PyTreeDef]:
        if not isinstance(state, CanonicalUPGDState):
            raise TypeError("state must be a CanonicalUPGDState")
        param_leaves, structure, _ = CanonicalUPGD._parameter_contract(params)
        utility_leaves, utility_structure = _flatten_with_none(state.utility_ema)
        age_leaves, age_structure = _flatten_with_none(state.utility_age)
        if not structure == utility_structure == age_structure:
            raise ValueError("params and CanonicalUPGD state must share a PyTree structure")
        utilities: list[Array] = []
        ages: list[Array] = []
        for param, utility, age in zip(
            param_leaves, utility_leaves, age_leaves, strict=True
        ):
            param_dtype = np.dtype(param.dtype)
            utilities.append(
                _matching_array(
                    "utility state leaf",
                    utility,
                    shape=param.shape,
                    dtype=param_dtype,
                )
            )
            ages.append(
                _matching_array(
                    "utility-age state leaf",
                    age,
                    shape=param.shape,
                    dtype=np.dtype(np.int32),
                )
            )
        _matching_array(
            "state.step",
            state.step,
            shape=(),
            dtype=np.dtype(np.int32),
        )
        return param_leaves, utilities, ages, structure

    def init(self, params: Any) -> CanonicalUPGDState:
        """Initialize signed-utility traces matching ``params``."""

        leaves, structure, _ = self._parameter_contract(params)
        array_params = jax.tree_util.tree_unflatten(structure, leaves)

        return CanonicalUPGDState(  # type: ignore[call-arg]
            utility_ema=jax.tree.map(jnp.zeros_like, array_params),
            utility_age=jax.tree.map(
                lambda value: jnp.zeros(value.shape, dtype=jnp.int32),
                array_params,
            ),
            step=jnp.array(0, dtype=jnp.int32),
        )

    def update(
        self,
        state: CanonicalUPGDState,
        params: Any,
        gradients: Any,
        key: Array,
        *,
        mask: Any | None = None,
        noise: Any | None = None,
    ) -> CanonicalUPGDUpdate:
        """Apply one source-profiled UPGD-W step.

        With ``safe_extended``, ``mask`` controls where UPGD utility/noise
        gating applies.  Masked-out parameters still receive ordinary gradient
        and decoupled weight-decay updates, allowing callers to protect a trunk
        while training heads and biases with plain SGDW.  A ``None`` gradient
        or a non-finite gradient skips that parameter element and is reported
        in metrics.  Source profiles reject masks and Python ``None`` leaves;
        dynamic non-finite arrays fail closed under eager execution and JIT.

        ``noise`` may supply the already-scaled perturbation PyTree ``xi`` for
        equation-level parity tests.  Normal operation omits it and samples
        ``N(0, noise_std**2)`` from ``key``.

        The final representable int32 clock event is applied.  A saturated
        clock may continue only after its active leaf-dtype bias-correction
        denominator is exactly one.  If an active saturated clock still needs
        to advance that denominator, the update rejects atomically: parameters,
        optimizer state, and the key remain exact; computed payloads are
        neutral; and ``metrics["update_applied"]`` is false.  The refusal
        branch excludes the numerical proposal from reverse-mode
        differentiation, preserving the same identity boundary there.
        """

        param_leaves, utility_leaves, age_leaves, structure = self._state_contract(
            state, params
        )
        params = jax.tree_util.tree_unflatten(structure, param_leaves)
        key = _adaupgd_typed_threefry_key(key, name="key")
        grad_leaves, grad_structure = _flatten_with_none(gradients)
        if structure != grad_structure:
            raise ValueError("params, gradients, and UPGD state must share a PyTree structure")

        if self._config.is_source_profile and mask is not None:
            raise ValueError("source profiles do not accept masks")
        if self._config.is_source_profile and any(gradient is None for gradient in grad_leaves):
            raise ValueError("source profiles require every gradient leaf")

        if mask is None:
            mask_leaves = [jnp.ones_like(value, dtype=jnp.bool_) for value in param_leaves]
        else:
            mask_leaves, mask_structure = _flatten_with_none(mask)
            if mask_structure != structure:
                raise ValueError("mask must share the parameter PyTree structure")

        if noise is None:
            supplied_noise_leaves: list[Any] = [None] * len(param_leaves)
        else:
            supplied_noise_leaves, noise_structure = _flatten_with_none(noise)
            if noise_structure != structure:
                raise ValueError("noise must share the parameter PyTree structure")

        beta = self._config.utility_decay
        maximum_counter = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
        next_step = _saturating_increment(state.step)

        finite_masks: list[Array] = []
        active_masks: list[Array] = []
        clean_gradients: list[Array] = []
        new_age_leaves: list[Array] = []
        correction_denominators: list[Array] = []
        capacity_available = jnp.asarray(True, dtype=jnp.bool_)
        eligible_count = jnp.array(0.0, dtype=jnp.float32)
        nonfinite_count = jnp.array(0.0, dtype=jnp.float32)

        for param, gradient, age, mask_leaf in zip(
            param_leaves,
            grad_leaves,
            age_leaves,
            mask_leaves,
            strict=True,
        ):
            eligible = _broadcast_bool_array("mask leaf", mask_leaf, param.shape)
            if gradient is None:
                finite = jnp.zeros(param.shape, dtype=jnp.bool_)
                clean_gradient = jnp.zeros_like(param)
            else:
                gradient_array = _coerced_numeric_array(
                    "gradient leaf", gradient, shape=param.shape, dtype=param.dtype
                )
                finite = jnp.isfinite(param) & jnp.isfinite(gradient_array)
                clean_gradient = jnp.where(finite, gradient_array, 0.0)

            active = eligible & finite
            if self._config.uses_global_clock:
                next_age = jnp.full_like(age, next_step)
                correction_clock = next_step.astype(param.dtype)
            else:
                next_age = _saturating_increment(age, active.astype(jnp.int32))
                correction_clock = next_age.astype(param.dtype)
            bias_correction = 1.0 - jnp.power(beta, correction_clock)
            correction_denominator = (
                jnp.maximum(bias_correction, self._config.epsilon)
                if self._config.profile == "safe_extended"
                else bias_correction
            )
            denominator_converged = correction_denominator == jnp.asarray(
                1.0, dtype=bias_correction.dtype
            )
            if self._config.uses_global_clock:
                saturated_active = (state.step >= maximum_counter) & jnp.any(active)
                leaf_capacity_available = (~saturated_active) | jnp.all(
                    denominator_converged
                )
            else:
                leaf_capacity_available = ~jnp.any(
                    active
                    & (age >= maximum_counter)
                    & ~denominator_converged
                )
            capacity_available = capacity_available & leaf_capacity_available

            clean_gradients.append(clean_gradient)
            finite_masks.append(finite)
            active_masks.append(active)
            new_age_leaves.append(next_age)
            count = jnp.sum(active.astype(jnp.float32))
            eligible_count = eligible_count + count
            nonfinite_count = nonfinite_count + jnp.sum((~finite).astype(jnp.float32))
            correction_denominators.append(correction_denominator)

        def apply_update(_: None) -> CanonicalUPGDUpdate:
            split_keys = jr.split(key, len(param_leaves) + 1)
            next_key = split_keys[0]
            noise_keys = split_keys[1:]

            corrected_leaves: list[Array] = []
            new_utility_leaves: list[Array] = []
            for param, gradient, utility, next_age, active, correction_denominator in zip(
                param_leaves,
                clean_gradients,
                utility_leaves,
                new_age_leaves,
                active_masks,
                correction_denominators,
                strict=True,
            ):
                instantaneous = -gradient * param
                next_utility = jnp.where(
                    active,
                    _static_zero_scale(beta, utility)
                    + (1.0 - beta) * instantaneous,
                    utility,
                )
                corrected = jnp.where(
                    next_age > 0,
                    next_utility / correction_denominator,
                    0.0,
                )
                new_utility_leaves.append(next_utility)
                corrected_leaves.append(corrected)

            if self._config.resolved_normalization == "global":
                maximum = jnp.array(-jnp.inf, dtype=jnp.float32)
                has_active = jnp.array(False)
                reference_leaves = (
                    new_utility_leaves
                    if self._config.profile in _RAW_GLOBAL_PROFILES
                    else corrected_leaves
                )
                for reference, active in zip(
                    reference_leaves,
                    active_masks,
                    strict=True,
                ):
                    leaf_maximum = jnp.max(
                        jnp.where(active, reference, -jnp.inf),
                        initial=-jnp.inf,
                    )
                    maximum = jnp.maximum(maximum, leaf_maximum.astype(jnp.float32))
                    has_active = has_active | jnp.any(active)
                global_maximum = jnp.where(has_active, maximum, 0.0)
                denominator = (
                    _safe_signed_denominator(
                        global_maximum,
                        self._config.epsilon,
                    )
                    if self._config.profile == "safe_extended"
                    else global_maximum
                )
                normalized_leaves = [
                    jnp.where(
                        active,
                        corrected / denominator.astype(corrected.dtype),
                        0.0,
                    )
                    for corrected, active in zip(
                        corrected_leaves,
                        active_masks,
                        strict=True,
                    )
                ]
            else:
                global_maximum = jnp.array(jnp.nan, dtype=jnp.float32)
                if self._config.profile == "paper_local_literal":
                    denominator_leaves = new_utility_leaves
                    local_epsilon: float | None = None
                elif self._config.profile == "official_experiment_local":
                    denominator_leaves = corrected_leaves
                    local_epsilon = 1e-12
                else:
                    denominator_leaves = corrected_leaves
                    local_epsilon = self._config.epsilon
                normalized_leaves = [
                    _local_normalize(
                        corrected,
                        denominator_utility,
                        active,
                        local_epsilon,
                    )
                    for corrected, denominator_utility, active in zip(
                        corrected_leaves,
                        denominator_leaves,
                        active_masks,
                        strict=True,
                    )
                ]

            new_param_leaves: list[Array] = []
            gate_leaves: list[Array] = []
            perturbation_leaves: list[Array] = []
            mean_gate = jnp.array(0.0, dtype=jnp.float32)
            mean_utility = jnp.array(0.0, dtype=jnp.float32)
            count_floor = jnp.maximum(eligible_count, 1.0)

            for (
                param,
                gradient,
                normalized,
                corrected,
                finite,
                active,
                noise_key,
                supplied_noise,
            ) in zip(
                param_leaves,
                clean_gradients,
                normalized_leaves,
                corrected_leaves,
                finite_masks,
                active_masks,
                noise_keys,
                supplied_noise_leaves,
                strict=True,
            ):
                gate = jnp.where(active, jax.nn.sigmoid(normalized), 0.0)
                if supplied_noise is None:
                    sampled_noise = (
                        jr.normal(noise_key, param.shape, dtype=param.dtype)
                        * self._config.noise_std
                    )
                else:
                    sampled_noise = _coerced_numeric_array(
                        "noise leaf",
                        supplied_noise,
                        shape=param.shape,
                        dtype=param.dtype,
                    )
                finite_noise = jnp.where(jnp.isfinite(sampled_noise), sampled_noise, 0.0)
                perturbation = jnp.where(active, finite_noise, 0.0)

                if self._config.mode == "protecting":
                    direction = (gradient + perturbation) * (1.0 - gate)
                else:
                    direction = gradient + perturbation * (1.0 - gate)

                decayed = param * (1.0 - self._config.step_size * self._config.weight_decay)
                direction_step = self._config.step_size * self._config.direction_multiplier
                candidate = decayed - direction_step * direction
                updated = jnp.where(finite, candidate, param)

                mean_gate = mean_gate + jnp.sum(
                    gate.astype(jnp.float32) / count_floor
                )
                mean_utility = mean_utility + jnp.sum(
                    jnp.where(active, jnp.abs(corrected), 0.0).astype(jnp.float32)
                    / count_floor
                )

                new_param_leaves.append(updated)
                gate_leaves.append(gate)
                perturbation_leaves.append(perturbation)

            proposed_state = CanonicalUPGDState(  # type: ignore[call-arg]
                utility_ema=jax.tree_util.tree_unflatten(structure, new_utility_leaves),
                utility_age=jax.tree_util.tree_unflatten(structure, new_age_leaves),
                step=next_step,
            )
            proposed_result = CanonicalUPGDUpdate(  # type: ignore[call-arg]
                params=jax.tree_util.tree_unflatten(structure, new_param_leaves),
                state=proposed_state,
                next_key=next_key,
                scaled_utility=jax.tree_util.tree_unflatten(structure, gate_leaves),
                corrected_utility=jax.tree_util.tree_unflatten(
                    structure, corrected_leaves
                ),
                perturbation=jax.tree_util.tree_unflatten(
                    structure, perturbation_leaves
                ),
                metrics={
                    "update_applied": jnp.asarray(True, dtype=jnp.bool_),
                    "mean_scaled_utility": mean_gate,
                    "mean_absolute_utility": mean_utility,
                    "global_maximum_utility": global_maximum,
                    "eligible_parameter_count": eligible_count,
                    "nonfinite_or_missing_count": nonfinite_count,
                },
            )
            if self._config.profile != "safe_extended":
                return proposed_result
            finite_payload = _floating_leaves_are_finite(
                proposed_result.params,
                proposed_result.state,
                proposed_result.scaled_utility,
                proposed_result.corrected_utility,
                proposed_result.perturbation,
                mean_gate,
                mean_utility,
            )
            if self._config.resolved_normalization == "global":
                finite_payload = finite_payload & jnp.isfinite(global_maximum)
            return cast(
                CanonicalUPGDUpdate,
                jax.lax.cond(
                    finite_payload,
                    lambda _: proposed_result,
                    reject_update,
                    operand=None,
                ),
            )

        def reject_update(_: None) -> CanonicalUPGDUpdate:
            zero_tree = jax.tree.map(jnp.zeros_like, params)
            zero_metric = jnp.array(0.0, dtype=jnp.float32)
            return CanonicalUPGDUpdate(  # type: ignore[call-arg]
                params=params,
                state=state,
                next_key=key,
                scaled_utility=zero_tree,
                corrected_utility=zero_tree,
                perturbation=zero_tree,
                metrics={
                    "update_applied": jnp.asarray(False, dtype=jnp.bool_),
                    "mean_scaled_utility": zero_metric,
                    "mean_absolute_utility": zero_metric,
                    "global_maximum_utility": zero_metric,
                    "eligible_parameter_count": eligible_count,
                    "nonfinite_or_missing_count": nonfinite_count,
                },
            )

        result: CanonicalUPGDUpdate = jax.lax.cond(
            capacity_available,
            apply_update,
            reject_update,
            operand=None,
        )
        return result


# This Alberta-derived transform was introduced separately from the official RL
# AdaptiveUPGD profile implemented below and remains intentionally distinct.  It
# extends the numerically guarded first-order UPGD equations above with an
# Adam/RMSProp-style gradient second moment and applies that same denominator to
# both the ordinary direction and the perturbation.  The released implementation
# instead tracks a first moment, leaves noise outside the denominator, uses a raw
# utility-EMA maximum, and applies a two-alpha direction.  Keeping separate types
# prevents either set of semantics from changing CanonicalUPGD or each other.
AlbertaAdaUPGDProfile = Literal["alberta_derived_first_order_adaptive_v1"]
ALBERTA_ADAUPGD_PROFILE: AlbertaAdaUPGDProfile = (
    "alberta_derived_first_order_adaptive_v1"
)

@dataclass(frozen=True)
class AlbertaAdaUPGDConfig:
    """Configuration for the opt-in Alberta-derived adaptive UPGD transform.

    This is not claimed to reproduce the pinned official AdaUPGD variant from
    Elsayed and Mahmood.  It deliberately keeps the guarded first-order signed
    utility, normalization, sigmoid gate, protecting and non-protecting
    directions, and decoupled UPGD-W decay from :class:`CanonicalUPGD`, then
    divides gradient and perturbation directions by a bias-corrected
    per-parameter gradient second moment:

    ``v_t = beta_2 v_(t-1) + (1-beta_2) g_t^2``

    ``d_t = sqrt(v_t / (1-beta_2^t)) + epsilon``.

    In protecting mode the direction is ``(g + xi) / d * (1-gate)``.  In
    non-protecting mode it is ``g/d + xi/d * (1-gate)``.  Decoupled weight
    decay remains ``(1-alpha*weight_decay)*w`` and is never divided by ``d``.

    A parameter mask selects where UPGD utility protection and perturbation
    apply.  A false mask does *not* freeze a parameter: it receives ordinary
    adaptive gradient descent and decoupled decay.  Its utility trace and age
    pause, while its second moment continues to update.  Missing or non-finite
    gradients reject the complete transaction even under a false mask because
    the adaptive base update still owns those gradients.
    """

    profile: AlbertaAdaUPGDProfile = ALBERTA_ADAUPGD_PROFILE
    step_size: float = 1e-3
    utility_decay: float = 0.999
    second_moment_decay: float = 0.999
    noise_std: float = 1e-3
    weight_decay: float = 0.0
    mode: UPGDMode = "protecting"
    normalization: UPGDNormalization = "global"
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if type(self.profile) is not str or self.profile != ALBERTA_ADAUPGD_PROFILE:
            raise ValueError(
                "profile must be the explicit Alberta-derived AdaUPGD profile"
            )
        step_size = _validated_float32_scalar(
            "step_size", self.step_size, lower=0.0, positive=True
        )
        utility_decay = _validated_float32_scalar(
            "utility_decay", self.utility_decay, lower=0.0, upper=1.0, upper_inclusive=False
        )
        second_moment_decay = _validated_float32_scalar(
            "second_moment_decay",
            self.second_moment_decay,
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
        )
        noise_std = _validated_float32_scalar("noise_std", self.noise_std, lower=0.0)
        weight_decay = _validated_float32_scalar(
            "weight_decay", self.weight_decay, lower=0.0
        )
        epsilon = _validated_float32_scalar(
            "epsilon", self.epsilon, lower=0.0, positive=True
        )
        object.__setattr__(self, "step_size", step_size)
        object.__setattr__(self, "utility_decay", utility_decay)
        object.__setattr__(self, "second_moment_decay", second_moment_decay)
        object.__setattr__(self, "noise_std", noise_std)
        object.__setattr__(self, "weight_decay", weight_decay)
        object.__setattr__(self, "epsilon", epsilon)
        if type(self.mode) is not str or self.mode not in {"protecting", "non_protecting"}:
            raise ValueError("mode must be protecting or non_protecting")
        if type(self.normalization) is not str or self.normalization not in {
            "global",
            "local",
        }:
            raise ValueError("normalization must be global or local")

    @property
    def official_reference_parity(self) -> bool:
        """Whether this profile claims parity with the released AdaUPGD code."""

        return False

    @property
    def source_attribution(self) -> str:
        """Return the non-reference provenance attached to this equation."""

        return (
            "Alberta-derived adaptive extension of canonical first-order UPGD; "
            "not the pinned official AdaUPGD parity profile"
        )

    def to_config(self) -> dict[str, object]:
        """Return the complete, strict JSON-compatible configuration."""

        return {
            "type": "AlbertaAdaUPGD",
            "profile": self.profile,
            "step_size": self.step_size,
            "utility_decay": self.utility_decay,
            "second_moment_decay": self.second_moment_decay,
            "noise_std": self.noise_std,
            "weight_decay": self.weight_decay,
            "mode": self.mode,
            "normalization": self.normalization,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> AlbertaAdaUPGDConfig:
        """Strictly reconstruct the adaptive extension configuration."""

        payload = _copy_config_mapping("AlbertaAdaUPGD config", config)
        expected = {
            "type",
            "profile",
            "step_size",
            "utility_decay",
            "second_moment_decay",
            "noise_std",
            "weight_decay",
            "mode",
            "normalization",
            "epsilon",
        }
        if set(payload) != expected:
            raise ValueError("config fields do not match the AlbertaAdaUPGD schema")
        type_name = payload.pop("type")
        if type(type_name) is not str or type_name != "AlbertaAdaUPGD":
            raise ValueError("expected AlbertaAdaUPGD config, got invalid value")
        try:
            return cls(**payload)
        except Exception as error:
            raise ValueError("AlbertaAdaUPGD config values are invalid") from error


@chex.dataclass(frozen=True)
class AlbertaAdaUPGDState:
    """Persistent state owned by :class:`AlbertaAdaUPGD`."""

    utility_ema: Any
    utility_age: Any
    gradient_second_moment: Any
    step: Int[Array, ""]


@chex.dataclass(frozen=True)
class AlbertaAdaUPGDUpdate:
    """Atomic result of one adaptive UPGD proposal and commit attempt."""

    params: Any
    state: AlbertaAdaUPGDState
    next_key: Array
    accepted: Array
    scaled_utility: Any
    corrected_utility: Any
    adaptive_denominator: Any
    perturbation: Any
    metrics: dict[str, Array]


@dataclass(frozen=True)
class AlbertaAdaUPGDResources:
    """Exact persistent-array accounting for one adaptive optimizer state."""

    profile: str
    official_reference_parity: bool
    parameter_count: int
    persistent_array_count: int
    persistent_state_nbytes: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible resource record."""

        return {
            "profile": self.profile,
            "official_reference_parity": self.official_reference_parity,
            "parameter_count": self.parameter_count,
            "persistent_array_count": self.persistent_array_count,
            "persistent_state_nbytes": self.persistent_state_nbytes,
        }


def _adaupgd_typed_threefry_key(value: object, *, name: str) -> Array:
    """Require one typed scalar Threefry key without accepting legacy words."""

    try:
        key = jnp.asarray(value)
        implementation = str(jr.key_impl(value))  # type: ignore[arg-type]
        words = jr.key_data(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a typed scalar threefry2x32 key") from exc
    if (
        key.shape != ()
        or implementation != "threefry2x32"
        or words.shape != (2,)
        or words.dtype != jnp.uint32
    ):
        raise TypeError(f"{name} must be a typed scalar threefry2x32 key")
    return key


def _adaupgd_tree_array_nbytes(tree: Any) -> int:
    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(tree)
        if hasattr(leaf, "size") and hasattr(leaf, "dtype")
    )


def measure_alberta_adaupgd_state_nbytes(state: AlbertaAdaUPGDState) -> int:
    """Measure every persistent JAX-array byte in an adaptive UPGD state."""

    return _adaupgd_tree_array_nbytes(state)


def _adaupgd_select_tree(condition: Array, candidate: Any, fallback: Any) -> Any:
    """Select a complete same-structure array PyTree with one scalar condition."""

    return jax.tree.map(
        lambda new, old: jnp.where(condition, new, old),
        candidate,
        fallback,
    )


class AlbertaAdaUPGD:
    """Opt-in, functional Alberta-derived adaptive first-order UPGD.

    The update is JIT/scan compatible.  Structural contract violations raise
    before arithmetic.  Dynamic numeric failures (invalid state, missing or
    non-finite gradients, non-finite noise, counter exhaustion, or a non-finite
    candidate) reject atomically: parameters, state, and key all retain their
    input values and diagnostic output trees are zeroed.
    """

    def __init__(self, config: AlbertaAdaUPGDConfig | None = None):
        self._config = AlbertaAdaUPGDConfig() if config is None else config

    @property
    def config(self) -> AlbertaAdaUPGDConfig:
        """Return the immutable optimizer configuration."""

        return self._config

    def to_config(self) -> dict[str, object]:
        """Serialize the exact optimizer configuration."""

        return self._config.to_config()

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> AlbertaAdaUPGD:
        """Reconstruct the optimizer from its strict configuration."""

        return cls(AlbertaAdaUPGDConfig.from_config(config))

    @staticmethod
    def _parameter_contract(params: Any) -> tuple[list[Array], jax.tree_util.PyTreeDef]:
        arrays, structure, _ = _parameter_arrays(
            params,
            owner="AlbertaAdaUPGD",
            persistent_nbytes=lambda scalars, nbytes: 2 * nbytes + 4 * scalars + 4,
            # params, state, four diagnostics, key/accepted, and ten metrics.
            update_nbytes=lambda scalars, nbytes: 7 * nbytes + 4 * scalars + 50,
            working_scalars=lambda scalars: 33 * scalars + 32,
            working_nbytes=lambda scalars, nbytes: 34 * nbytes + 4 * scalars + 128,
        )
        return arrays, structure

    @staticmethod
    def _state_contract(
        state: AlbertaAdaUPGDState,
        params: Any,
    ) -> tuple[
        list[Array],
        list[Array],
        list[Array],
        list[Array],
        jax.tree_util.PyTreeDef,
    ]:
        if not isinstance(state, AlbertaAdaUPGDState):
            raise TypeError("state must be an AlbertaAdaUPGDState")
        param_leaves, structure = AlbertaAdaUPGD._parameter_contract(params)
        utility_leaves, utility_structure = _flatten_with_none(state.utility_ema)
        age_leaves, age_structure = _flatten_with_none(state.utility_age)
        moment_leaves, moment_structure = _flatten_with_none(
            state.gradient_second_moment
        )
        if not (
            structure == utility_structure == age_structure == moment_structure
        ):
            raise ValueError("params and AlbertaAdaUPGD state must share a PyTree structure")
        utilities: list[Array] = []
        ages: list[Array] = []
        moments: list[Array] = []
        for param, utility, age, moment in zip(
            param_leaves,
            utility_leaves,
            age_leaves,
            moment_leaves,
            strict=True,
        ):
            param_dtype = np.dtype(param.dtype)
            utility_array = _matching_array(
                "utility state leaf", utility, shape=param.shape, dtype=param_dtype
            )
            age_array = _matching_array(
                "utility-age state leaf",
                age,
                shape=param.shape,
                dtype=np.dtype(np.int32),
            )
            moment_array = _matching_array(
                "second-moment state leaf", moment, shape=param.shape, dtype=param_dtype
            )
            utilities.append(utility_array)
            ages.append(age_array)
            moments.append(moment_array)
        _matching_array("state.step", state.step, shape=(), dtype=np.dtype(np.int32))
        return param_leaves, utilities, ages, moments, structure

    def init(self, params: Any) -> AlbertaAdaUPGDState:
        """Initialize utility and adaptive second-moment traces."""

        leaves, structure = self._parameter_contract(params)
        array_params = jax.tree_util.tree_unflatten(structure, leaves)
        return AlbertaAdaUPGDState(  # type: ignore[call-arg]
            utility_ema=jax.tree.map(jnp.zeros_like, array_params),
            utility_age=jax.tree.map(
                lambda value: jnp.zeros(value.shape, dtype=jnp.int32),
                array_params,
            ),
            gradient_second_moment=jax.tree.map(jnp.zeros_like, array_params),
            step=jnp.asarray(0, dtype=jnp.int32),
        )

    def state_valid(self, state: AlbertaAdaUPGDState, params: Any) -> Array:
        """Return whether state and parameter values satisfy the numeric contract."""

        param_leaves, utility_leaves, age_leaves, moment_leaves, _ = self._state_contract(
            state, params
        )
        step = jnp.asarray(state.step)
        valid = (step >= 0) & (step <= _INT32_MAX)
        for param, utility, age, moment in zip(
            param_leaves,
            utility_leaves,
            age_leaves,
            moment_leaves,
            strict=True,
        ):
            valid = (
                valid
                & jnp.all(jnp.isfinite(param))
                & jnp.all(jnp.isfinite(utility))
                & jnp.all(jnp.isfinite(moment))
                & jnp.all(moment >= 0.0)
                & jnp.all(age >= 0)
                & jnp.all(age <= step)
            )
        return valid

    def resource_budget(self, state: AlbertaAdaUPGDState) -> AlbertaAdaUPGDResources:
        """Return exact persistent-array resources for ``state``."""

        parameter_count = sum(
            int(jnp.asarray(leaf).size)
            for leaf in jax.tree.leaves(state.utility_ema)
        )
        array_leaves = [
            leaf for leaf in jax.tree.leaves(state) if isinstance(leaf, Array)
        ]
        return AlbertaAdaUPGDResources(
            profile=self._config.profile,
            official_reference_parity=False,
            parameter_count=parameter_count,
            persistent_array_count=len(array_leaves),
            persistent_state_nbytes=measure_alberta_adaupgd_state_nbytes(state),
        )

    def update(
        self,
        state: AlbertaAdaUPGDState,
        params: Any,
        gradients: Any,
        key: Array,
        *,
        mask: Any | None = None,
        noise: Any | None = None,
    ) -> AlbertaAdaUPGDUpdate:
        """Attempt one atomic adaptive first-order UPGD-W update.

        ``noise=None`` samples Gaussian perturbations and advances the supplied
        typed Threefry key only if the transaction commits.  A supplied noise
        PyTree is interpreted as the already-scaled perturbation ``xi``; it is
        intended for fixed-noise equation tests and follows the same key
        splitting/commit contract as sampled noise.
        """

        (
            param_leaves,
            utility_leaves,
            age_leaves,
            moment_leaves,
            structure,
        ) = self._state_contract(state, params)
        params = jax.tree_util.tree_unflatten(structure, param_leaves)
        checked_key = _adaupgd_typed_threefry_key(key, name="key")
        grad_leaves, grad_structure = _flatten_with_none(gradients)
        if grad_structure != structure:
            raise ValueError("params, gradients, and state must share a PyTree structure")

        if mask is None:
            mask_leaves = [jnp.ones(param.shape, dtype=jnp.bool_) for param in param_leaves]
        else:
            mask_leaves, mask_structure = _flatten_with_none(mask)
            if mask_structure != structure:
                raise ValueError("mask must share the parameter PyTree structure")
        eligible_leaves: list[Array] = []
        for param, mask_leaf in zip(param_leaves, mask_leaves, strict=True):
            eligible_leaves.append(
                _broadcast_bool_array("mask leaf", mask_leaf, param.shape)
            )

        if noise is None:
            supplied_noise_leaves: list[Any] = [None] * len(param_leaves)
        else:
            supplied_noise_leaves, noise_structure = _flatten_with_none(noise)
            if noise_structure != structure:
                raise ValueError("noise must share the parameter PyTree structure")
            if any(value is None for value in supplied_noise_leaves):
                raise ValueError("supplied noise must contain every parameter leaf")

        split_keys = jr.split(checked_key, len(param_leaves) + 1)
        proposed_next_key = split_keys[0]
        noise_keys = split_keys[1:]
        state_is_valid = self.state_valid(state, params)
        if self._config.utility_decay == 0.0 or self._config.second_moment_decay == 0.0:
            checked_utility = (
                jax.tree.map(
                    lambda value: jnp.where(
                        jnp.isfinite(value), value, jnp.zeros_like(value)
                    ),
                    state.utility_ema,
                )
                if self._config.utility_decay == 0.0
                else state.utility_ema
            )
            checked_moment = (
                jax.tree.map(
                    lambda value: jnp.where(
                        jnp.isfinite(value), value, jnp.zeros_like(value)
                    ),
                    state.gradient_second_moment,
                )
                if self._config.second_moment_decay == 0.0
                else state.gradient_second_moment
            )
            state_is_valid = self.state_valid(
                state.replace(  # type: ignore[attr-defined]
                    utility_ema=checked_utility,
                    gradient_second_moment=checked_moment,
                ),
                params,
            )
        capacity_available = state.step < jnp.asarray(_INT32_MAX, dtype=jnp.int32)
        proposed_step = jnp.where(capacity_available, state.step + 1, state.step)
        input_is_valid = jnp.asarray(True, dtype=jnp.bool_)
        missing_gradient_leaf_count = jnp.asarray(0, dtype=jnp.int32)
        nonfinite_gradient_count = jnp.asarray(0, dtype=jnp.int32)
        nonfinite_noise_count = jnp.asarray(0, dtype=jnp.int32)

        clean_params: list[Array] = []
        clean_gradients: list[Array] = []
        clean_utilities: list[Array] = []
        clean_ages: list[Array] = []
        clean_moments: list[Array] = []
        sampled_noise_leaves: list[Array] = []
        for param, gradient, utility, age, moment, noise_key, supplied_noise in zip(
            param_leaves,
            grad_leaves,
            utility_leaves,
            age_leaves,
            moment_leaves,
            noise_keys,
            supplied_noise_leaves,
            strict=True,
        ):
            clean_params.append(jnp.where(jnp.isfinite(param), param, 0.0))
            clean_utilities.append(jnp.where(jnp.isfinite(utility), utility, 0.0))
            clean_ages.append(jnp.clip(age, 0, _INT32_MAX - 1))
            clean_moments.append(
                jnp.where(jnp.isfinite(moment) & (moment >= 0.0), moment, 0.0)
            )
            if gradient is None:
                clean_gradient = jnp.zeros_like(param)
                gradient_finite = jnp.zeros(param.shape, dtype=jnp.bool_)
                missing_gradient_leaf_count = missing_gradient_leaf_count + jnp.asarray(
                    1, dtype=jnp.int32
                )
            else:
                gradient_array = _coerced_numeric_array(
                    "gradient leaf", gradient, shape=param.shape, dtype=param.dtype
                )
                gradient_finite = jnp.isfinite(gradient_array)
                clean_gradient = jnp.where(gradient_finite, gradient_array, 0.0)
            nonfinite_gradient_count = nonfinite_gradient_count + jnp.sum(
                (~gradient_finite).astype(jnp.int32), dtype=jnp.int32
            )
            input_is_valid = input_is_valid & jnp.all(gradient_finite)
            clean_gradients.append(clean_gradient)

            if supplied_noise is None:
                sampled_noise = (
                    jr.normal(noise_key, param.shape, dtype=param.dtype)
                    * self._config.noise_std
                )
            else:
                sampled_noise = _coerced_numeric_array(
                    "noise leaf",
                    supplied_noise,
                    shape=param.shape,
                    dtype=param.dtype,
                )
            noise_finite = jnp.isfinite(sampled_noise)
            nonfinite_noise_count = nonfinite_noise_count + jnp.sum(
                (~noise_finite).astype(jnp.int32), dtype=jnp.int32
            )
            input_is_valid = input_is_valid & jnp.all(noise_finite)
            sampled_noise_leaves.append(jnp.where(noise_finite, sampled_noise, 0.0))

        beta_utility = self._config.utility_decay
        beta_second = self._config.second_moment_decay
        proposed_utility_leaves: list[Array] = []
        proposed_age_leaves: list[Array] = []
        proposed_moment_leaves: list[Array] = []
        corrected_leaves: list[Array] = []
        denominator_leaves: list[Array] = []

        for param, gradient, utility, age, moment, eligible in zip(
            clean_params,
            clean_gradients,
            clean_utilities,
            clean_ages,
            clean_moments,
            eligible_leaves,
            strict=True,
        ):
            instantaneous_utility = -gradient * param
            proposed_utility = jnp.where(
                eligible,
                _static_zero_scale(beta_utility, utility)
                + (1.0 - beta_utility) * instantaneous_utility,
                utility,
            )
            proposed_age = jnp.where(eligible, age + 1, age)
            utility_clock = jnp.maximum(proposed_age, 1).astype(param.dtype)
            utility_correction = 1.0 - jnp.power(beta_utility, utility_clock)
            corrected = jnp.where(
                eligible & (proposed_age > 0),
                proposed_utility
                / jnp.maximum(
                    utility_correction,
                    jnp.asarray(self._config.epsilon, dtype=param.dtype),
                ),
                0.0,
            )

            proposed_moment = _static_zero_scale(beta_second, moment) + (
                1.0 - beta_second
            ) * jnp.square(gradient)
            moment_clock = jnp.maximum(proposed_step, 1).astype(param.dtype)
            moment_correction = 1.0 - jnp.power(beta_second, moment_clock)
            corrected_moment = proposed_moment / jnp.maximum(
                moment_correction,
                jnp.asarray(self._config.epsilon, dtype=param.dtype),
            )
            denominator = jnp.sqrt(corrected_moment) + jnp.asarray(
                self._config.epsilon, dtype=param.dtype
            )

            proposed_utility_leaves.append(proposed_utility)
            proposed_age_leaves.append(proposed_age)
            proposed_moment_leaves.append(proposed_moment)
            corrected_leaves.append(corrected)
            denominator_leaves.append(denominator)

        if self._config.normalization == "global":
            maximum = jnp.asarray(-jnp.inf, dtype=jnp.float32)
            has_eligible = jnp.asarray(False, dtype=jnp.bool_)
            for corrected, eligible in zip(
                corrected_leaves, eligible_leaves, strict=True
            ):
                maximum = jnp.maximum(
                    maximum,
                    jnp.max(
                        jnp.where(eligible, corrected, -jnp.inf),
                        initial=-jnp.inf,
                    ).astype(jnp.float32),
                )
                has_eligible = has_eligible | jnp.any(eligible)
            global_maximum = jnp.where(has_eligible, maximum, 0.0)
            global_denominator = _safe_signed_denominator(
                global_maximum, self._config.epsilon
            )
            normalized_leaves = [
                jnp.where(
                    eligible,
                    corrected / global_denominator.astype(corrected.dtype),
                    0.0,
                )
                for corrected, eligible in zip(
                    corrected_leaves, eligible_leaves, strict=True
                )
            ]
        else:
            global_maximum = jnp.asarray(jnp.nan, dtype=jnp.float32)
            normalized_leaves = [
                _local_normalize(
                    corrected,
                    corrected,
                    eligible,
                    self._config.epsilon,
                )
                for corrected, eligible in zip(
                    corrected_leaves, eligible_leaves, strict=True
                )
            ]

        proposed_param_leaves: list[Array] = []
        gate_leaves: list[Array] = []
        perturbation_leaves: list[Array] = []
        for param, gradient, denominator, normalized, eligible, sampled_noise in zip(
            clean_params,
            clean_gradients,
            denominator_leaves,
            normalized_leaves,
            eligible_leaves,
            sampled_noise_leaves,
            strict=True,
        ):
            safe_normalized = jnp.where(jnp.isfinite(normalized), normalized, 0.0)
            gate = jnp.where(eligible, jax.nn.sigmoid(safe_normalized), 0.0)
            perturbation = jnp.where(eligible, sampled_noise, 0.0)
            adaptive_gradient = gradient / denominator
            adaptive_noise = perturbation / denominator
            if self._config.mode == "protecting":
                direction = (adaptive_gradient + adaptive_noise) * (1.0 - gate)
            else:
                direction = adaptive_gradient + adaptive_noise * (1.0 - gate)
            decayed = param * (
                1.0 - self._config.step_size * self._config.weight_decay
            )
            proposed_param_leaves.append(decayed - self._config.step_size * direction)
            gate_leaves.append(gate)
            perturbation_leaves.append(perturbation)

        proposed_params = jax.tree_util.tree_unflatten(structure, proposed_param_leaves)
        proposed_state = AlbertaAdaUPGDState(  # type: ignore[call-arg]
            utility_ema=jax.tree_util.tree_unflatten(
                structure, proposed_utility_leaves
            ),
            utility_age=jax.tree_util.tree_unflatten(structure, proposed_age_leaves),
            gradient_second_moment=jax.tree_util.tree_unflatten(
                structure, proposed_moment_leaves
            ),
            step=proposed_step,
        )
        eligible_count = sum(
            (
                jnp.sum(eligible.astype(jnp.int32), dtype=jnp.int32)
                for eligible in eligible_leaves
            ),
            start=jnp.asarray(0, dtype=jnp.int32),
        )
        count_floor = jnp.maximum(eligible_count.astype(jnp.float32), 1.0)
        mean_gate = sum(
            (
                jnp.sum(gate.astype(jnp.float32) / count_floor)
                for gate in gate_leaves
            ),
            start=jnp.asarray(0.0, dtype=jnp.float32),
        )
        mean_utility = sum(
            (
                jnp.sum(
                    jnp.where(eligible, jnp.abs(corrected), 0.0).astype(jnp.float32)
                    / count_floor
                )
                for corrected, eligible in zip(
                    corrected_leaves, eligible_leaves, strict=True
                )
            ),
            start=jnp.asarray(0.0, dtype=jnp.float32),
        )
        candidate_is_valid = self.state_valid(proposed_state, proposed_params)
        diagnostics_are_valid = _floating_leaves_are_finite(
            corrected_leaves,
            denominator_leaves,
            gate_leaves,
            perturbation_leaves,
            mean_gate,
            mean_utility,
        )
        if self._config.normalization == "global":
            diagnostics_are_valid = diagnostics_are_valid & jnp.isfinite(global_maximum)
        accepted = (
            state_is_valid
            & capacity_available
            & input_is_valid
            & candidate_is_valid
            & diagnostics_are_valid
        )
        committed_params = _adaupgd_select_tree(accepted, proposed_params, params)
        committed_state = _adaupgd_select_tree(accepted, proposed_state, state)
        committed_key = jax.lax.cond(
            accepted,
            lambda _: proposed_next_key,
            lambda _: checked_key,
            operand=None,
        )
        zero_tree = jax.tree.map(jnp.zeros_like, params)
        scaled_utility = _adaupgd_select_tree(
            accepted,
            jax.tree_util.tree_unflatten(structure, gate_leaves),
            zero_tree,
        )
        corrected_utility = _adaupgd_select_tree(
            accepted,
            jax.tree_util.tree_unflatten(structure, corrected_leaves),
            zero_tree,
        )
        adaptive_denominator = _adaupgd_select_tree(
            accepted,
            jax.tree_util.tree_unflatten(structure, denominator_leaves),
            zero_tree,
        )
        perturbation = _adaupgd_select_tree(
            accepted,
            jax.tree_util.tree_unflatten(structure, perturbation_leaves),
            zero_tree,
        )
        parameter_count = sum(param.size for param in param_leaves)
        reported_global_maximum = jnp.where(
            accepted & (self._config.normalization == "global"),
            global_maximum,
            0.0,
        )
        return AlbertaAdaUPGDUpdate(  # type: ignore[call-arg]
            params=committed_params,
            state=committed_state,
            next_key=committed_key,
            accepted=accepted,
            scaled_utility=scaled_utility,
            corrected_utility=corrected_utility,
            adaptive_denominator=adaptive_denominator,
            perturbation=perturbation,
            metrics={
                "accepted": accepted,
                "mean_scaled_utility": jnp.where(accepted, mean_gate, 0.0),
                "mean_absolute_utility": jnp.where(accepted, mean_utility, 0.0),
                "global_maximum_utility": reported_global_maximum,
                "eligible_parameter_count": eligible_count,
                "parameter_count": jnp.asarray(parameter_count, dtype=jnp.int32),
                "missing_gradient_leaf_count": missing_gradient_leaf_count,
                "nonfinite_gradient_count": nonfinite_gradient_count,
                "nonfinite_noise_count": nonfinite_noise_count,
                "persistent_state_nbytes": jnp.asarray(
                    measure_alberta_adaupgd_state_nbytes(state), dtype=jnp.int32
                ),
            },
        )


# Exact finite/all-active equation profile for the released RL implementation.
# This is intentionally separate from both CanonicalUPGD and AlbertaAdaUPGD:
# the released AdaptiveUPGD has materially different moments, normalization,
# noise scaling, and a two-alpha learning direction.  The constants below bind
# the implementation to the audited immutable source location.
OfficialAdaUPGDProfile = Literal["official_rl_adaptive_upgd_b75e90a"]
OFFICIAL_ADAUPGD_PROFILE: OfficialAdaUPGDProfile = (
    "official_rl_adaptive_upgd_b75e90a"
)
OFFICIAL_ADAUPGD_COMMIT = "b75e90ad4b09c28971ac9dbb902a8fd86709b28c"
OFFICIAL_ADAUPGD_PATH = "core/run/rl/adaupgd.py"


@dataclass(frozen=True)
class OfficialAdaUPGDConfig:
    """Configuration for the pinned official RL ``AdaptiveUPGD`` equation.

    Provenance is the released MIT implementation at commit
    ``b75e90ad4b09c28971ac9dbb902a8fd86709b28c``, file
    ``core/run/rl/adaupgd.py``.  Defaults match that constructor exactly.

    This profile deliberately preserves its source quirks:

    * corrected utility is divided by the *uncorrected* raw utility-EMA maximum;
    * corrected first/second Adam moments drive the task direction;
    * perturbation noise is outside the adaptive denominator;
    * the task/noise direction is multiplied by ``2 * step_size`` while
      decoupled weight decay uses ``1 * step_size``; and
    * the raw maximum has no zero or non-finite guard.

    Consequently all-zero utility produces zero-over-zero and non-finite values
    propagate just as they do in the source.  Parity is equation-level for one
    float32 parameter group with finite, all-active gradients and supplied fixed
    perturbations; JAX and PyTorch random samplers are not claimed bitwise
    identical.
    """

    profile: OfficialAdaUPGDProfile = OFFICIAL_ADAUPGD_PROFILE
    source_commit: str = OFFICIAL_ADAUPGD_COMMIT
    source_path: str = OFFICIAL_ADAUPGD_PATH
    step_size: float = 1e-5
    weight_decay: float = 0.001
    utility_decay: float = 0.999
    noise_std: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if type(self.profile) is not str or self.profile != OFFICIAL_ADAUPGD_PROFILE:
            raise ValueError("profile must name the pinned official RL AdaptiveUPGD")
        if type(self.source_commit) is not str or self.source_commit != OFFICIAL_ADAUPGD_COMMIT:
            raise ValueError("source_commit must match the pinned AdaptiveUPGD commit")
        if type(self.source_path) is not str or self.source_path != OFFICIAL_ADAUPGD_PATH:
            raise ValueError("source_path must match the pinned AdaptiveUPGD path")
        step_size = _validated_float32_scalar(
            "step_size", self.step_size, lower=0.0, positive=True
        )
        weight_decay = _validated_float32_scalar(
            "weight_decay", self.weight_decay, lower=0.0
        )
        utility_decay = _validated_float32_scalar(
            "utility_decay", self.utility_decay, lower=0.0, upper=1.0, upper_inclusive=False
        )
        noise_std = _validated_float32_scalar("noise_std", self.noise_std, lower=0.0)
        beta1 = _validated_float32_scalar(
            "beta1", self.beta1, lower=0.0, upper=1.0, upper_inclusive=False
        )
        beta2 = _validated_float32_scalar(
            "beta2", self.beta2, lower=0.0, upper=1.0, upper_inclusive=False
        )
        epsilon = _validated_float32_scalar(
            "epsilon", self.epsilon, lower=0.0, positive=True
        )
        object.__setattr__(self, "step_size", step_size)
        object.__setattr__(self, "weight_decay", weight_decay)
        object.__setattr__(self, "utility_decay", utility_decay)
        object.__setattr__(self, "noise_std", noise_std)
        object.__setattr__(self, "beta1", beta1)
        object.__setattr__(self, "beta2", beta2)
        object.__setattr__(self, "epsilon", epsilon)

    @property
    def official_reference_parity(self) -> bool:
        """Whether this is an explicitly source-bound equation profile."""

        return True

    @property
    def parity_scope(self) -> str:
        """Return the bounded scope of the official-source parity claim."""

        return "finite_float32_all_active_single_group_fixed_noise_equation_parity"

    def to_config(self) -> dict[str, object]:
        """Return the complete source-bound configuration."""

        return {
            "type": "OfficialAdaUPGD",
            "profile": self.profile,
            "source_commit": self.source_commit,
            "source_path": self.source_path,
            "step_size": self.step_size,
            "weight_decay": self.weight_decay,
            "utility_decay": self.utility_decay,
            "noise_std": self.noise_std,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> OfficialAdaUPGDConfig:
        """Strictly reconstruct the pinned official profile."""

        payload = _copy_config_mapping("OfficialAdaUPGD config", config)
        expected = {
            "type",
            "profile",
            "source_commit",
            "source_path",
            "step_size",
            "weight_decay",
            "utility_decay",
            "noise_std",
            "beta1",
            "beta2",
            "epsilon",
        }
        if set(payload) != expected:
            raise ValueError("config fields do not match the OfficialAdaUPGD schema")
        type_name = payload.pop("type")
        if type(type_name) is not str or type_name != "OfficialAdaUPGD":
            raise ValueError("expected OfficialAdaUPGD config, got invalid value")
        try:
            return cls(**payload)
        except Exception as error:
            raise ValueError("OfficialAdaUPGD config values are invalid") from error


@chex.dataclass(frozen=True)
class OfficialAdaUPGDState:
    """Utility, first-moment, and second-moment state of official AdaUPGD."""

    utility_ema: Any
    first_moment: Any
    second_moment: Any
    step: Int[Array, ""]


@chex.dataclass(frozen=True)
class OfficialAdaUPGDUpdate:
    """Result of one functional official-source AdaUPGD update."""

    params: Any
    state: OfficialAdaUPGDState
    next_key: Array
    scaled_utility: Any
    corrected_utility: Any
    corrected_first_moment: Any
    corrected_second_moment: Any
    perturbation: Any
    metrics: dict[str, Array]


@dataclass(frozen=True)
class OfficialAdaUPGDResources:
    """Exact state resources plus immutable official-source provenance."""

    profile: str
    source_commit: str
    source_path: str
    official_reference_parity: bool
    parameter_count: int
    persistent_array_count: int
    persistent_state_nbytes: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible resource record."""

        return {
            "profile": self.profile,
            "source_commit": self.source_commit,
            "source_path": self.source_path,
            "official_reference_parity": self.official_reference_parity,
            "parameter_count": self.parameter_count,
            "persistent_array_count": self.persistent_array_count,
            "persistent_state_nbytes": self.persistent_state_nbytes,
        }


def measure_official_adaupgd_state_nbytes(state: OfficialAdaUPGDState) -> int:
    """Measure every persistent JAX-array byte in official AdaUPGD state."""

    return _adaupgd_tree_array_nbytes(state)


class OfficialAdaUPGD:
    """Functional JAX port of the pinned official RL ``AdaptiveUPGD``.

    Masks and missing gradients are rejected because the source defines neither
    behavior.  Structural contracts fail before arithmetic.  Numeric values are
    otherwise left unguarded on purpose, including raw-maximum zero division
    and non-finite propagation.  Use :class:`AlbertaAdaUPGD` when an atomic,
    guarded adaptive extension is required instead of source parity.
    """

    def __init__(self, config: OfficialAdaUPGDConfig | None = None):
        self._config = OfficialAdaUPGDConfig() if config is None else config

    @property
    def config(self) -> OfficialAdaUPGDConfig:
        """Return the immutable official-source configuration."""

        return self._config

    def to_config(self) -> dict[str, object]:
        """Serialize the exact optimizer configuration and provenance."""

        return self._config.to_config()

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> OfficialAdaUPGD:
        """Reconstruct the transform from a strict source-bound config."""

        return cls(OfficialAdaUPGDConfig.from_config(config))

    @staticmethod
    def _parameter_contract(params: Any) -> tuple[list[Array], jax.tree_util.PyTreeDef]:
        arrays, structure, _ = _parameter_arrays(
            params,
            owner="OfficialAdaUPGD",
            persistent_nbytes=lambda _scalars, nbytes: 3 * nbytes + 4,
            # params, state, five diagnostics, key, and three metrics.
            update_nbytes=lambda _scalars, nbytes: 9 * nbytes + 24,
            working_scalars=lambda scalars: 33 * scalars + 32,
            working_nbytes=lambda scalars, nbytes: 34 * nbytes + 4 * scalars + 128,
        )
        return arrays, structure

    @staticmethod
    def _state_contract(
        state: OfficialAdaUPGDState,
        params: Any,
    ) -> tuple[
        list[Array],
        list[Array],
        list[Array],
        list[Array],
        jax.tree_util.PyTreeDef,
    ]:
        if not isinstance(state, OfficialAdaUPGDState):
            raise TypeError("state must be an OfficialAdaUPGDState")
        param_leaves, structure = OfficialAdaUPGD._parameter_contract(params)
        utility_leaves, utility_structure = _flatten_with_none(state.utility_ema)
        first_leaves, first_structure = _flatten_with_none(state.first_moment)
        second_leaves, second_structure = _flatten_with_none(state.second_moment)
        if not (
            structure == utility_structure == first_structure == second_structure
        ):
            raise ValueError("params and OfficialAdaUPGD state must share a PyTree structure")
        utilities: list[Array] = []
        first_moments: list[Array] = []
        second_moments: list[Array] = []
        for param, utility, first, second in zip(
            param_leaves,
            utility_leaves,
            first_leaves,
            second_leaves,
            strict=True,
        ):
            param_dtype = np.dtype(param.dtype)
            utility_array = _matching_array(
                "utility state leaf", utility, shape=param.shape, dtype=param_dtype
            )
            first_array = _matching_array(
                "first-moment state leaf", first, shape=param.shape, dtype=param_dtype
            )
            second_array = _matching_array(
                "second-moment state leaf", second, shape=param.shape, dtype=param_dtype
            )
            utilities.append(utility_array)
            first_moments.append(first_array)
            second_moments.append(second_array)
        _matching_array("state.step", state.step, shape=(), dtype=np.dtype(np.int32))
        return param_leaves, utilities, first_moments, second_moments, structure

    def init(self, params: Any) -> OfficialAdaUPGDState:
        """Initialize the three source-defined moment PyTrees."""

        leaves, structure = self._parameter_contract(params)
        array_params = jax.tree_util.tree_unflatten(structure, leaves)
        return OfficialAdaUPGDState(  # type: ignore[call-arg]
            utility_ema=jax.tree.map(jnp.zeros_like, array_params),
            first_moment=jax.tree.map(jnp.zeros_like, array_params),
            second_moment=jax.tree.map(jnp.zeros_like, array_params),
            step=jnp.asarray(0, dtype=jnp.int32),
        )

    def state_valid(self, state: OfficialAdaUPGDState, params: Any) -> Array:
        """Return whether a state is finite and its second moments are valid."""

        param_leaves, utility_leaves, first_leaves, second_leaves, _ = (
            self._state_contract(state, params)
        )
        valid = (state.step >= 0) & (state.step <= _INT32_MAX)
        for param, utility, first, second in zip(
            param_leaves,
            utility_leaves,
            first_leaves,
            second_leaves,
            strict=True,
        ):
            valid = (
                valid
                & jnp.all(jnp.isfinite(param))
                & jnp.all(jnp.isfinite(utility))
                & jnp.all(jnp.isfinite(first))
                & jnp.all(jnp.isfinite(second))
                & jnp.all(second >= 0.0)
            )
        return valid

    def resource_budget(self, state: OfficialAdaUPGDState) -> OfficialAdaUPGDResources:
        """Return exact state resources with source provenance."""

        parameter_count = sum(
            int(jnp.asarray(leaf).size)
            for leaf in jax.tree.leaves(state.utility_ema)
        )
        array_leaves = [
            leaf for leaf in jax.tree.leaves(state) if isinstance(leaf, Array)
        ]
        return OfficialAdaUPGDResources(
            profile=self._config.profile,
            source_commit=self._config.source_commit,
            source_path=self._config.source_path,
            official_reference_parity=True,
            parameter_count=parameter_count,
            persistent_array_count=len(array_leaves),
            persistent_state_nbytes=measure_official_adaupgd_state_nbytes(state),
        )

    def update(
        self,
        state: OfficialAdaUPGDState,
        params: Any,
        gradients: Any,
        key: Array,
        *,
        mask: Any | None = None,
        noise: Any | None = None,
    ) -> OfficialAdaUPGDUpdate:
        """Apply one pinned official AdaptiveUPGD equation step.

        A supplied noise PyTree is the already-scaled perturbation used for
        fixed-noise parity.  Sampling uses a typed Threefry key as the explicit
        JAX port of the source's implicit PyTorch generator.
        """

        if mask is not None:
            raise ValueError("the official AdaptiveUPGD profile does not accept masks")
        (
            param_leaves,
            utility_leaves,
            first_leaves,
            second_leaves,
            structure,
        ) = self._state_contract(state, params)
        params = jax.tree_util.tree_unflatten(structure, param_leaves)
        checked_key = _adaupgd_typed_threefry_key(key, name="key")
        grad_leaves, grad_structure = _flatten_with_none(gradients)
        if grad_structure != structure:
            raise ValueError("params, gradients, and state must share a PyTree structure")
        if any(gradient is None for gradient in grad_leaves):
            raise ValueError("the official AdaptiveUPGD profile requires every gradient")
        if noise is None:
            supplied_noise_leaves: list[Any] = [None] * len(param_leaves)
        else:
            supplied_noise_leaves, noise_structure = _flatten_with_none(noise)
            if noise_structure != structure:
                raise ValueError("noise must share the parameter PyTree structure")
            if any(value is None for value in supplied_noise_leaves):
                raise ValueError("supplied noise must contain every parameter leaf")

        gradient_arrays: list[Array] = []
        for param, gradient in zip(param_leaves, grad_leaves, strict=True):
            gradient_array = _coerced_numeric_array(
                "gradient leaf", gradient, shape=param.shape, dtype=param.dtype
            )
            gradient_arrays.append(gradient_array)

        split_keys = jr.split(checked_key, len(param_leaves) + 1)
        next_key = split_keys[0]
        noise_keys = split_keys[1:]
        next_step = _saturating_increment(state.step)
        proposed_utility_leaves: list[Array] = []
        proposed_first_leaves: list[Array] = []
        proposed_second_leaves: list[Array] = []
        for param, gradient, utility, first, second in zip(
            param_leaves,
            gradient_arrays,
            utility_leaves,
            first_leaves,
            second_leaves,
            strict=True,
        ):
            proposed_utility_leaves.append(
                _static_zero_scale(self._config.utility_decay, utility)
                + (1.0 - self._config.utility_decay) * (-gradient * param)
            )
            proposed_first_leaves.append(
                _static_zero_scale(self._config.beta1, first)
                + (1.0 - self._config.beta1) * gradient
            )
            proposed_second_leaves.append(
                _static_zero_scale(self._config.beta2, second)
                + (1.0 - self._config.beta2) * jnp.square(gradient)
            )

        # Match the source's scalar-tensor ``if current > maximum`` fold rather
        # than jnp.maximum: a NaN leaf comparison is false and leaves the prior
        # maximum untouched.
        raw_global_maximum = jnp.asarray(-jnp.inf, dtype=jnp.float32)
        for utility in proposed_utility_leaves:
            leaf_maximum = jnp.max(utility).astype(jnp.float32)
            raw_global_maximum = jnp.where(
                leaf_maximum > raw_global_maximum,
                leaf_maximum,
                raw_global_maximum,
            )

        corrected_utility_leaves: list[Array] = []
        corrected_first_leaves: list[Array] = []
        corrected_second_leaves: list[Array] = []
        gate_leaves: list[Array] = []
        perturbation_leaves: list[Array] = []
        proposed_param_leaves: list[Array] = []
        for (
            param,
            proposed_utility,
            proposed_first,
            proposed_second,
            noise_key,
            supplied_noise,
        ) in zip(
            param_leaves,
            proposed_utility_leaves,
            proposed_first_leaves,
            proposed_second_leaves,
            noise_keys,
            supplied_noise_leaves,
            strict=True,
        ):
            clock = next_step.astype(param.dtype)
            utility_correction = 1.0 - jnp.power(self._config.utility_decay, clock)
            first_correction = 1.0 - jnp.power(self._config.beta1, clock)
            second_correction = 1.0 - jnp.power(self._config.beta2, clock)
            corrected_utility = proposed_utility / utility_correction
            corrected_first = proposed_first / first_correction
            corrected_second = proposed_second / second_correction
            if supplied_noise is None:
                perturbation = (
                    jr.normal(noise_key, param.shape, dtype=param.dtype)
                    * self._config.noise_std
                )
            else:
                perturbation = _coerced_numeric_array(
                    "noise leaf",
                    supplied_noise,
                    shape=param.shape,
                    dtype=param.dtype,
                )
            gate = jax.nn.sigmoid(
                corrected_utility / raw_global_maximum.astype(param.dtype)
            )
            one_minus_gate = 1.0 - gate
            direction = (
                corrected_first * one_minus_gate
                / (jnp.sqrt(corrected_second) + self._config.epsilon)
                + perturbation * one_minus_gate
            )
            decayed = param * (
                1.0 - self._config.step_size * self._config.weight_decay
            )
            proposed_param_leaves.append(
                decayed - 2.0 * self._config.step_size * direction
            )
            corrected_utility_leaves.append(corrected_utility)
            corrected_first_leaves.append(corrected_first)
            corrected_second_leaves.append(corrected_second)
            gate_leaves.append(gate)
            perturbation_leaves.append(perturbation)

        next_state = OfficialAdaUPGDState(  # type: ignore[call-arg]
            utility_ema=jax.tree_util.tree_unflatten(
                structure, proposed_utility_leaves
            ),
            first_moment=jax.tree_util.tree_unflatten(
                structure, proposed_first_leaves
            ),
            second_moment=jax.tree_util.tree_unflatten(
                structure, proposed_second_leaves
            ),
            step=next_step,
        )
        parameter_count = sum(param.size for param in param_leaves)
        return OfficialAdaUPGDUpdate(  # type: ignore[call-arg]
            params=jax.tree_util.tree_unflatten(structure, proposed_param_leaves),
            state=next_state,
            next_key=next_key,
            scaled_utility=jax.tree_util.tree_unflatten(structure, gate_leaves),
            corrected_utility=jax.tree_util.tree_unflatten(
                structure, corrected_utility_leaves
            ),
            corrected_first_moment=jax.tree_util.tree_unflatten(
                structure, corrected_first_leaves
            ),
            corrected_second_moment=jax.tree_util.tree_unflatten(
                structure, corrected_second_leaves
            ),
            perturbation=jax.tree_util.tree_unflatten(
                structure, perturbation_leaves
            ),
            metrics={
                "raw_global_maximum_utility": raw_global_maximum,
                "parameter_count": jnp.asarray(parameter_count, dtype=jnp.int32),
                "persistent_state_nbytes": jnp.asarray(
                    measure_official_adaupgd_state_nbytes(state), dtype=jnp.int32
                ),
            },
        )
