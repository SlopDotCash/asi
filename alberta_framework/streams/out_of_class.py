"""Out-of-hypothesis-class Step 2 benchmark streams.

These streams test feature *construction* rather than feature *selection*.
The existing Step 2 streams (``InteractionFeatureDiscoveryStream``,
``NonlinearFeatureDiscoveryStream``) place the oracle features inside the
learner's hypothesis class -- the learner is given pair-products and the
stream uses pair-products, so "discovery" reduces to selecting the right
items from a known pool.

The Alberta Plan Step 2 demands construction of features from existing
features in the general case.  To probe that, the streams here generate
targets whose minimal representation lies *outside* a 1-layer pair-product
or tanh feature bank:

* ``OutOfClassPolynomialStream`` -- degree-3 polynomial targets requiring
  triple products ``x_i * x_j * x_l``.  A pair-product learner can only fit
  the marginal pair structure; a learner that composes pair-products with
  raw features (``(x_i * x_j) * x_l``) can fit the targets exactly.

* ``FrequencyMismatchStream`` -- targets are sums of trigonometric features
  ``sin(omega * x + phi)``.  Tanh / pair-product banks cannot represent
  sin(x) and must compose many surrogate features to approximate it.

* ``CompositionalStream`` -- targets are 2-hidden-layer tanh networks.
  A 1-layer feature bank cannot represent the targets exactly; only a
  compositional DAG that builds features-of-features can.
"""

import math
from fractions import Fraction
from numbers import Real
from typing import cast

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Float, Int, PRNGKeyArray

from alberta_framework._fixed_count_selection import (
    require_positive_builtin_int,
    stable_smallest_mask,
)
from alberta_framework._float32 import (
    round_real_to_float32,
    round_real_to_float32_with_ratio,
)
from alberta_framework.core.types import TimeStep

_FLOAT32_MULTIPLIER_MAX = float(np.sqrt(np.finfo(np.float32).max))
_INT32_MAX = int(np.iinfo(np.int32).max)
_MAX_OUT_OF_CLASS_STATE_BYTES = 64 * 1024 * 1024


def _require_positive_dimension(name: str, value: object) -> int:
    """Return a positive builtin int inside the int32 host-dimension domain."""
    canonical = require_positive_builtin_int(value, name=name)
    if canonical > _INT32_MAX:
        raise ValueError(f"{name} must be at most int32 max")
    return canonical


def _require_state_budget(name: str, array_scalars: int) -> dict[str, int]:
    """Preflight one resident float32/int32 state plus its typed PRNG key."""
    state_scalars = array_scalars + 2
    state_bytes = 4 * state_scalars
    if state_scalars > _INT32_MAX:
        raise ValueError(f"{name} state scalar count must fit signed int32")
    if state_bytes > _INT32_MAX:
        raise ValueError(f"{name} state byte count must fit signed int32")
    if state_bytes > _MAX_OUT_OF_CLASS_STATE_BYTES:
        raise ValueError(f"{name} state exceeds the 64 MiB budget")
    return {
        "array_scalars": array_scalars,
        "key_words": 2,
        "state_scalars": state_scalars,
        "state_bytes": state_bytes,
    }


def _frequency_state_budget(
    *, n_contexts: int, n_tasks: int, n_components_per_task: int
) -> dict[str, int]:
    tensor_scalars = n_contexts * n_tasks * n_components_per_task
    budget = _require_state_budget("frequency-mismatch", 4 * tensor_scalars + 1)
    return {"tensor_scalars": tensor_scalars, **budget}


def _compositional_state_budget(
    *,
    feature_dim: int,
    n_tasks: int,
    inner_hidden: int,
    outer_components: int,
    n_contexts: int,
) -> dict[str, int]:
    component_scalars = n_contexts * n_tasks * outer_components
    outer_scalars = component_scalars * inner_hidden
    inner_weight_scalars = outer_scalars * feature_dim
    state_scalars = inner_weight_scalars + 2 * outer_scalars + 2 * component_scalars + 1
    budget = _require_state_budget("compositional", state_scalars)
    return {
        "inner_weight_scalars": inner_weight_scalars,
        "outer_scalars": outer_scalars,
        "component_scalars": component_scalars,
        **budget,
    }


def _polynomial_state_budget(
    *,
    feature_dim: int,
    n_contexts: int,
    n_tasks: int,
    include_squares: bool,
) -> dict[str, int]:
    """Preflight the oracle-triple enumeration before it ever runs.

    ``OutOfClassPolynomialStream._triples()`` enumerates all (i, j, l) triples
    in plain Python -- O(feature_dim ** 3) work and memory for the strict
    ``i < j < l`` case (``i <= j <= l`` with ``include_squares`` is the same
    order). A caller-supplied ``feature_dim`` well inside the int32 domain
    (e.g. a few thousand) already makes that enumeration run for an
    unbounded amount of time before any JAX array exists, let alone before
    the sibling classes' own ``_require_state_budget`` gate would catch the
    resulting tensor size. Compute the exact triple count analytically via
    ``math.comb`` -- O(1), no enumeration -- and apply the same 64 MiB
    resident-state budget the sibling ``FrequencyMismatchStream`` and
    ``CompositionalStream`` constructors already enforce, before
    ``_triples()`` is ever invoked from ``init()``.
    """
    n_triples = math.comb(feature_dim + 2, 3) if include_squares else math.comb(feature_dim, 3)
    tensor_scalars = n_contexts * n_tasks * n_triples
    # Resident OutOfClassPolynomialState holds one (n_contexts, n_tasks,
    # n_triples) ``context_weights`` tensor, 3 * n_triples int32 indices
    # (triples_left/middle/right), n_tasks * feature_dim ``linear_weights``
    # scalars, and a scalar step_count; ``_require_state_budget`` adds the
    # 2-word PRNG key. ``init()`` also transiently allocates a same-shaped
    # ``dense_context_weights``/``mask_scores``/``mask`` working set, which
    # is smaller than or comparable to this resident bound and is caught by
    # the same gate in practice.
    state_scalars = tensor_scalars + 3 * n_triples + n_tasks * feature_dim + 1
    budget = _require_state_budget("out-of-class-polynomial", state_scalars)
    return {"n_triples": n_triples, "tensor_scalars": tensor_scalars, **budget}


def _saturating_step_count(step_count: Array) -> Array:
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    return jnp.minimum(jnp.maximum(step_count, 0), maximum - 1) + 1


_SUPPORTED_NUMPY_REAL_SCALAR_TYPES: tuple[type[object], ...] = (
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


def _trusted_multiplier_real(value: object, message: str) -> Real:
    """Return a hook-free concrete real for Polynomial stream multipliers."""
    actual_type = type(value)
    if actual_type is int or actual_type is float:
        return cast(Real, value)
    for supported_type in _SUPPORTED_NUMPY_REAL_SCALAR_TYPES:
        if actual_type is supported_type:
            return cast(Real, value)
    if actual_type is not Fraction:
        raise ValueError(message)

    fraction = cast(Fraction, value)
    try:
        numerator: object = object.__getattribute__(fraction, "_numerator")
        denominator: object = object.__getattribute__(fraction, "_denominator")
    except Exception as error:
        raise ValueError(message) from error
    if type(numerator) is not int or type(denominator) is not int:
        raise ValueError(message)
    if denominator <= 0:
        raise ValueError(message)
    try:
        return Fraction(numerator, denominator)
    except Exception as error:
        raise ValueError(message) from error


def _require_positive_float32(value: object, name: str) -> float:
    """Return a positive finite value representable in the stream execution dtype."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite float32 value")
    try:
        converted = round_real_to_float32(value)
    except (FloatingPointError, OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a positive finite float32 value"
        ) from error
    if not np.isfinite(converted) or converted <= np.float32(0.0):
        raise ValueError(f"{name} must be a positive finite float32 value")
    return converted


def _require_finite_float32_multiplier(
    value: object,
    *,
    name: str,
    nonnegative: bool = False,
) -> float:
    """Return a canonical finite multiplier with conservative float32 headroom."""
    message = f"{name} must be a finite real in the safe float32 multiplier range"
    try:
        trusted_value = _trusted_multiplier_real(value, message)
        numerator, _, narrowed = round_real_to_float32_with_ratio(trusted_value)
    except Exception as error:
        raise ValueError(message) from error
    if nonnegative and numerator < 0:
        raise ValueError(message)
    if not np.isfinite(narrowed) or abs(narrowed) > _FLOAT32_MULTIPLIER_MAX:
        raise ValueError(message)
    return narrowed


def _require_safe_float32_product(*values: float, names: str) -> None:
    """Reject configured multiplier products that consume float32 headroom."""
    product = np.float32(1.0)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for value in values:
            product = np.float32(product * np.float32(value))
    if not np.isfinite(product) or abs(product) > _FLOAT32_MULTIPLIER_MAX:
        raise ValueError(
            f"{names} must stay within the safe combined float32 multiplier range"
        )


# =============================================================================
# OutOfClassPolynomialStream -- degree-3 polynomial targets
# =============================================================================


@chex.dataclass(frozen=True)
class OutOfClassPolynomialState:
    """State for ``OutOfClassPolynomialStream``.

    The hidden oracle features are unordered triple products
    ``x_i * x_j * x_l``.  The triple list is fixed; per-context weights
    determine which triples are useful for each task.

    Attributes:
        key: PRNG key for sample generation.
        triples_left: First index of each triple, shape ``(n_triples,)``.
        triples_middle: Middle index of each triple, shape ``(n_triples,)``.
        triples_right: Third index of each triple, shape ``(n_triples,)``.
        context_weights: Per-context task weights over triples,
            shape ``(n_contexts, n_tasks, n_triples)``.
        linear_weights: Small direct linear component, shape
            ``(n_tasks, feature_dim)``.
        step_count: Number of generated samples so far.
    """

    key: PRNGKeyArray
    triples_left: Int[Array, " n_triples"]
    triples_middle: Int[Array, " n_triples"]
    triples_right: Int[Array, " n_triples"]
    context_weights: Float[Array, "n_contexts n_tasks n_triples"]
    linear_weights: Float[Array, "n_tasks feature_dim"]
    step_count: Int[Array, ""]


class OutOfClassPolynomialStream:
    """Non-stationary stream whose useful features are triple products.

    Targets are degree-3 polynomial combinations:

    ``y*_k(x) = sum_{i<=j<=l} W_k[i,j,l] x_i x_j x_l + L_k . x + noise``

    where ``W_k`` is sparse (only ``active_triples_per_context`` triples
    nonzero per task per context).  A learner whose feature bank contains
    only pair products ``x_i * x_j`` cannot fit this exactly; a learner
    able to compose features (``(x_i * x_j) * x_l``) can.
    """

    def __init__(
        self,
        feature_dim: int = 8,
        n_tasks: int = 3,
        n_contexts: int = 4,
        context_length: int = 500,
        active_triples_per_context: int = 3,
        feature_std: float = 1.0,
        linear_scale: float = 0.05,
        noise_std: float = 0.05,
        include_squares: bool = False,
    ):
        """Initialize the out-of-class polynomial stream.

        Args:
            feature_dim: Raw observation dimension.
            n_tasks: Number of supervised output heads.
            n_contexts: Number of recurring relevance contexts.
            context_length: Steps before switching context.
            active_triples_per_context: Number of active triple products per
                task/context, capped at the available triple count.
            feature_std: Standard deviation of raw observations.
            linear_scale: Scale of the small direct linear component.
            noise_std: Standard deviation of target noise.
            include_squares: Whether to include triples with repeated
                indices (``x_i^2 * x_j``, ``x_i^3``).  Default False, so
                the oracle uses strict ``i < j < l`` triples only.
        """
        feature_dim = require_positive_builtin_int(feature_dim, name="feature_dim")
        if feature_dim < 3:
            raise ValueError("feature_dim must be at least 3")
        n_tasks = require_positive_builtin_int(n_tasks, name="n_tasks")
        n_contexts = require_positive_builtin_int(n_contexts, name="n_contexts")
        context_length = require_positive_builtin_int(context_length, name="context_length")
        active_triples_per_context = require_positive_builtin_int(
            active_triples_per_context,
            name="active_triples_per_context",
        )
        for name, value in (
            ("feature_dim", feature_dim),
            ("n_tasks", n_tasks),
            ("n_contexts", n_contexts),
            ("context_length", context_length),
            ("active_triples_per_context", active_triples_per_context),
        ):
            if value > _INT32_MAX:
                raise ValueError(f"{name} must be at most int32 max")
        feature_std = _require_finite_float32_multiplier(
            feature_std,
            name="feature_std",
            nonnegative=True,
        )
        linear_scale = _require_finite_float32_multiplier(
            linear_scale,
            name="linear_scale",
        )
        noise_std = _require_finite_float32_multiplier(
            noise_std,
            name="noise_std",
            nonnegative=True,
        )
        _require_safe_float32_product(
            feature_std,
            feature_std,
            feature_std,
            names="feature_std cubed",
        )
        _require_safe_float32_product(
            linear_scale,
            feature_std,
            names="linear_scale and feature_std",
        )
        if type(include_squares) is not bool:
            raise ValueError("include_squares must be a built-in bool")
        resource_budget = _polynomial_state_budget(
            feature_dim=feature_dim,
            n_contexts=n_contexts,
            n_tasks=n_tasks,
            include_squares=include_squares,
        )

        self._feature_dim = feature_dim
        self._n_tasks = n_tasks
        self._n_contexts = n_contexts
        self._context_length = context_length
        self._active_triples_per_context = active_triples_per_context
        self._feature_std = feature_std
        self._linear_scale = linear_scale
        self._noise_std = noise_std
        self._include_squares = include_squares
        self._resource_budget = resource_budget

    @property
    def feature_dim(self) -> int:
        """Return the raw observation dimension."""
        return self._feature_dim

    @property
    def resource_budget(self) -> dict[str, int]:
        """Complete resident state payload accounting."""
        return dict(self._resource_budget)

    @property
    def target_dim(self) -> int:
        """Return the number of supervised tasks."""
        return self._n_tasks

    def _triples(self) -> tuple[Array, Array, Array]:
        """Enumerate all unordered triples (i, j, l).

        With ``include_squares=False``, returns strict ``i < j < l`` triples
        (count ``C(feature_dim, 3)``).  With ``include_squares=True``,
        returns ``i <= j <= l`` triples (count ``C(feature_dim+2, 3)``).

        The enumeration runs at construction time in plain Python and is
        materialized into JAX integer arrays, so it does not interact with
        JIT.

        Returns:
            Three Int arrays of equal length: left, middle, right indices.
        """
        triples: list[tuple[int, int, int]] = []
        for i in range(self._feature_dim):
            j_start = i if self._include_squares else i + 1
            for j in range(j_start, self._feature_dim):
                l_start = j if self._include_squares else j + 1
                for ell in range(l_start, self._feature_dim):
                    triples.append((i, j, ell))
        if not triples:
            raise ValueError(
                "feature_dim too small to enumerate any oracle triples"
            )
        arr = jnp.array(triples, dtype=jnp.int32)
        return arr[:, 0], arr[:, 1], arr[:, 2]

    def init(self, key: Array) -> OutOfClassPolynomialState:
        """Initialize stream state.

        Args:
            key: JAX PRNG key.

        Returns:
            Initialized ``OutOfClassPolynomialState``.
        """
        key, k_ctx, k_mask, k_linear = jr.split(key, 4)
        triples_left, triples_middle, triples_right = self._triples()
        n_triples = triples_left.shape[0]

        dense_context_weights = jr.normal(
            k_ctx,
            (self._n_contexts, self._n_tasks, n_triples),
            dtype=jnp.float32,
        )
        active_count = min(self._active_triples_per_context, n_triples)
        mask_scores = jr.uniform(
            k_mask,
            (self._n_contexts, self._n_tasks, n_triples),
            dtype=jnp.float32,
        )
        mask = stable_smallest_mask(mask_scores, active_count)
        context_weights = dense_context_weights * mask.astype(jnp.float32)
        norm = jnp.sqrt(jnp.maximum(jnp.sum(mask, axis=-1, keepdims=True), 1.0))
        context_weights = context_weights / norm

        linear_weights = self._linear_scale * jr.normal(
            k_linear, (self._n_tasks, self._feature_dim), dtype=jnp.float32
        )

        return OutOfClassPolynomialState(
            key=key,
            triples_left=triples_left,
            triples_middle=triples_middle,
            triples_right=triples_right,
            context_weights=context_weights,
            linear_weights=linear_weights,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def step(
        self,
        state: OutOfClassPolynomialState,
        idx: Array,
    ) -> tuple[TimeStep, OutOfClassPolynomialState]:
        """Generate one multitask polynomial sample.

        Args:
            state: Current stream state.
            idx: Step index (unused; kept for ``ScanStream`` protocol).

        Returns:
            Tuple of (``TimeStep``, new stream state).
        """
        del idx
        key, k_x, k_noise = jr.split(state.key, 3)
        x = self._feature_std * jr.normal(
            k_x, (self._feature_dim,), dtype=jnp.float32
        )
        triples = (
            x[state.triples_left]
            * x[state.triples_middle]
            * x[state.triples_right]
        )
        context_idx = (state.step_count // self._context_length) % self._n_contexts
        task_weights = state.context_weights[context_idx]
        target = task_weights @ triples + state.linear_weights @ x
        noise = self._noise_std * jr.normal(
            k_noise, (self._n_tasks,), dtype=jnp.float32
        )
        target = target + noise

        timestep = TimeStep(observation=x, target=target)
        new_state = state.replace(  # type: ignore[attr-defined]
            key=key, step_count=_saturating_step_count(state.step_count)
        )
        return timestep, new_state


# =============================================================================
# FrequencyMismatchStream -- trigonometric targets
# =============================================================================


@chex.dataclass(frozen=True)
class FrequencyMismatchState:
    """State for ``FrequencyMismatchStream``.

    Each context defines a different set of trigonometric oracle features
    (frequency, phase, active input dimension, amplitude) per task and
    component.

    Attributes:
        key: PRNG key for sample generation.
        omegas: Per-context per-task per-component frequencies, shape
            ``(n_contexts, n_tasks, n_components_per_task)``.
        phases: Per-context per-task per-component phase offsets, shape
            ``(n_contexts, n_tasks, n_components_per_task)``.
        active_indices: Which input dimension each component listens to,
            shape ``(n_contexts, n_tasks, n_components_per_task)``.
        amplitudes: Per-context per-task per-component amplitudes,
            shape ``(n_contexts, n_tasks, n_components_per_task)``.
        step_count: Number of generated samples so far.
    """

    key: PRNGKeyArray
    omegas: Float[Array, "n_contexts n_tasks n_components"]
    phases: Float[Array, "n_contexts n_tasks n_components"]
    active_indices: Int[Array, "n_contexts n_tasks n_components"]
    amplitudes: Float[Array, "n_contexts n_tasks n_components"]
    step_count: Int[Array, ""]


class FrequencyMismatchStream:
    """Non-stationary stream whose targets are sums of sinusoids.

    Targets are sums of trigonometric features:

    ``y*_k(x) = sum_c A_kc sin(omega_kc x[i_kc] + phi_kc) + noise``

    where the per-context ``omega``, ``phi``, ``i``, and ``A`` are all
    fixed at ``init`` time.  A learner whose hypothesis class is built from
    tanh / pair-products cannot represent ``sin`` exactly; it must compose
    many surrogate features to approximate this oracle.
    """

    def __init__(
        self,
        feature_dim: int = 4,
        n_tasks: int = 2,
        n_components_per_task: int = 3,
        n_contexts: int = 4,
        context_length: int = 500,
        omega_min: float = 0.5,
        omega_max: float = 3.0,
        amplitude_scale: float = 1.0,
        noise_std: float = 0.05,
    ):
        """Initialize the frequency-mismatch stream.

        Args:
            feature_dim: Raw observation dimension.
            n_tasks: Number of supervised output heads.
            n_components_per_task: Number of sinusoidal components combined
                in each task target.
            n_contexts: Number of recurring relevance contexts.
            context_length: Steps before switching context.
            omega_min: Minimum sinusoid angular frequency.
            omega_max: Maximum sinusoid angular frequency.
            amplitude_scale: Scale of per-component amplitudes (drawn from
                a centered Gaussian times this factor).
            noise_std: Standard deviation of target noise.
        """
        feature_dim = _require_positive_dimension("feature_dim", feature_dim)
        n_tasks = _require_positive_dimension("n_tasks", n_tasks)
        n_components_per_task = _require_positive_dimension(
            "n_components_per_task",
            n_components_per_task,
        )
        n_contexts = _require_positive_dimension("n_contexts", n_contexts)
        context_length = _require_positive_dimension("context_length", context_length)
        resource_budget = _frequency_state_budget(
            n_contexts=n_contexts,
            n_tasks=n_tasks,
            n_components_per_task=n_components_per_task,
        )
        omega_min_float32 = _require_positive_float32(omega_min, "omega_min")
        omega_max_float32 = _require_positive_float32(omega_max, "omega_max")
        if omega_max_float32 <= omega_min_float32:
            raise ValueError("omega_max must exceed omega_min in float32")

        self._feature_dim = feature_dim
        self._n_tasks = n_tasks
        self._n_components_per_task = n_components_per_task
        self._n_contexts = n_contexts
        self._context_length = context_length
        self._omega_min = omega_min_float32
        self._omega_max = omega_max_float32
        self._amplitude_scale = _require_finite_float32_multiplier(
            amplitude_scale,
            name="amplitude_scale",
            nonnegative=True,
        )
        self._noise_std = _require_finite_float32_multiplier(
            noise_std,
            name="noise_std",
            nonnegative=True,
        )
        self._resource_budget = resource_budget

    @property
    def feature_dim(self) -> int:
        """Return the raw observation dimension."""
        return self._feature_dim

    @property
    def target_dim(self) -> int:
        """Return the number of supervised tasks."""
        return self._n_tasks

    @property
    def resource_budget(self) -> dict[str, int]:
        """Complete resident state payload accounting."""
        return dict(self._resource_budget)

    def init(self, key: Array) -> FrequencyMismatchState:
        """Initialize stream state.

        Args:
            key: JAX PRNG key.

        Returns:
            Initialized ``FrequencyMismatchState``.
        """
        key, k_omega, k_phase, k_active, k_amp = jr.split(key, 5)
        shape = (
            self._n_contexts,
            self._n_tasks,
            self._n_components_per_task,
        )
        omegas = jr.uniform(
            k_omega, shape, dtype=jnp.float32,
            minval=self._omega_min, maxval=self._omega_max,
        )
        phases = jr.uniform(
            k_phase, shape, dtype=jnp.float32,
            minval=0.0, maxval=float(2 * jnp.pi),
        )
        active_indices = jr.randint(
            k_active, shape, minval=0, maxval=self._feature_dim, dtype=jnp.int32
        )
        amplitudes = self._amplitude_scale * jr.normal(
            k_amp, shape, dtype=jnp.float32
        )
        return FrequencyMismatchState(
            key=key,
            omegas=omegas,
            phases=phases,
            active_indices=active_indices,
            amplitudes=amplitudes,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def step(
        self,
        state: FrequencyMismatchState,
        idx: Array,
    ) -> tuple[TimeStep, FrequencyMismatchState]:
        """Generate one multitask sinusoidal sample.

        Args:
            state: Current stream state.
            idx: Step index (unused; kept for ``ScanStream`` protocol).

        Returns:
            Tuple of (``TimeStep``, new stream state).
        """
        del idx
        key, k_x, k_noise = jr.split(state.key, 3)
        # Sample x ~ Uniform[-pi, pi] so sin/cos cover their full range.
        x = jr.uniform(
            k_x, (self._feature_dim,), dtype=jnp.float32,
            minval=-float(jnp.pi), maxval=float(jnp.pi),
        )
        context_idx = (state.step_count // self._context_length) % self._n_contexts

        omegas = state.omegas[context_idx]              # (n_tasks, n_components)
        phases = state.phases[context_idx]              # (n_tasks, n_components)
        active = state.active_indices[context_idx]      # (n_tasks, n_components)
        amps = state.amplitudes[context_idx]            # (n_tasks, n_components)

        x_active = x[active]                            # (n_tasks, n_components)
        components = jnp.sin(omegas * x_active + phases)
        target = jnp.sum(amps * components, axis=-1)    # (n_tasks,)
        noise = self._noise_std * jr.normal(
            k_noise, (self._n_tasks,), dtype=jnp.float32
        )
        target = target + noise

        timestep = TimeStep(observation=x, target=target)
        new_state = state.replace(  # type: ignore[attr-defined]
            key=key,
            step_count=_saturating_step_count(state.step_count),
        )
        return timestep, new_state


# =============================================================================
# CompositionalStream -- 2-hidden-layer tanh oracle
# =============================================================================


def _require_compositional_weight_scale_float32(value: object) -> float:
    """Return a signed scale in the stream's safe float32 execution domain.

    The scale is canonicalized from its exact integer ratio to a built-in float
    carrying the nearest-even float32 value.  Magnitudes at or below half the
    smallest float32 subnormal become signed zero.  A finite float32 square is
    required because the scale controls initialization variance and is
    multiplied before fan-in normalization; that conservative headroom
    prevents eager/XLA reassociation from turning an accepted scale into
    non-finite oracle weights.
    """
    message = "weight_scale must be finite in float32 with finite float32 squared magnitude"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(message)
    try:
        rounded = round_real_to_float32(value)
    except (FloatingPointError, OverflowError, TypeError, ValueError) as error:
        raise ValueError(message) from error
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        narrowed = np.float32(rounded)
        squared = np.float32(narrowed * narrowed)
    if not bool(np.isfinite(narrowed)) or not bool(np.isfinite(squared)):
        raise ValueError(message)
    return rounded


@chex.dataclass(frozen=True)
class CompositionalState:
    """State for ``CompositionalStream``.

    Each context defines a 2-hidden-layer tanh network whose output is
    summed against per-output amplitudes.  The targets are therefore
    representable only by composing features-of-features.

    Attributes:
        key: PRNG key for sample generation.
        inner_w: Inner weight matrices, shape
            ``(n_contexts, n_tasks, outer_components, inner_hidden, feature_dim)``.
        inner_b: Inner biases, shape
            ``(n_contexts, n_tasks, outer_components, inner_hidden)``.
        outer_w: Outer weight vectors, shape
            ``(n_contexts, n_tasks, outer_components, inner_hidden)``.
        outer_b: Outer biases, shape
            ``(n_contexts, n_tasks, outer_components)``.
        amplitudes: Per-component output scalings, shape
            ``(n_contexts, n_tasks, outer_components)``.
        step_count: Number of generated samples so far.
    """

    key: PRNGKeyArray
    inner_w: Float[Array, "n_contexts n_tasks outer inner feature_dim"]
    inner_b: Float[Array, "n_contexts n_tasks outer inner"]
    outer_w: Float[Array, "n_contexts n_tasks outer inner"]
    outer_b: Float[Array, "n_contexts n_tasks outer"]
    amplitudes: Float[Array, "n_contexts n_tasks outer"]
    step_count: Int[Array, ""]


class CompositionalStream:
    """Non-stationary stream whose targets are 2-hidden-layer tanh nets.

    Targets are computed as:

    ``inner = tanh(V x + c)``
    ``outer = tanh(W inner + b)``
    ``y*_k = a . outer + noise``

    A 1-layer feature bank (raw features, single-layer tanh, or pair
    products) cannot represent the targets exactly; only a compositional
    DAG that builds features-of-features can.
    """

    def __init__(
        self,
        feature_dim: int = 6,
        n_tasks: int = 3,
        inner_hidden: int = 4,
        outer_components: int = 5,
        n_contexts: int = 4,
        context_length: int = 500,
        feature_std: float = 1.0,
        weight_scale: float = 1.0,
        amplitude_scale: float = 1.0,
        noise_std: float = 0.05,
    ):
        """Initialize the compositional stream.

        Args:
            feature_dim: Raw observation dimension.
            n_tasks: Number of supervised output heads.
            inner_hidden: Width of the inner tanh layer per outer component.
            outer_components: Number of outer tanh components combined per
                task.
            n_contexts: Number of recurring relevance contexts.
            context_length: Steps before switching context.
            feature_std: Standard deviation of raw observations.
            weight_scale: Scale of per-layer weights (divided by sqrt(fan-in)
                for unit-variance pre-activations). Real values are rounded
                once from their exact integer ratio to nearest-even float32 at
                construction. Magnitudes at or below half the smallest
                float32 subnormal become signed zero; zero and negative scales
                remain valid. Values whose float32 square overflows are
                rejected so eager and compiled initialization stay finite.
            amplitude_scale: Scale of per-component output amplitudes.
            noise_std: Standard deviation of target noise.
        """
        feature_dim = _require_positive_dimension("feature_dim", feature_dim)
        n_tasks = _require_positive_dimension("n_tasks", n_tasks)
        inner_hidden = _require_positive_dimension("inner_hidden", inner_hidden)
        outer_components = _require_positive_dimension(
            "outer_components",
            outer_components,
        )
        n_contexts = _require_positive_dimension("n_contexts", n_contexts)
        context_length = _require_positive_dimension("context_length", context_length)
        resource_budget = _compositional_state_budget(
            feature_dim=feature_dim,
            n_tasks=n_tasks,
            inner_hidden=inner_hidden,
            outer_components=outer_components,
            n_contexts=n_contexts,
        )
        feature_std_float32 = _require_finite_float32_multiplier(
            feature_std,
            name="feature_std",
            nonnegative=True,
        )
        weight_scale_float32 = _require_compositional_weight_scale_float32(weight_scale)
        amplitude_scale_float32 = _require_finite_float32_multiplier(
            amplitude_scale,
            name="amplitude_scale",
            nonnegative=True,
        )
        noise_std_float32 = _require_finite_float32_multiplier(
            noise_std,
            name="noise_std",
            nonnegative=True,
        )
        _require_safe_float32_product(
            feature_std_float32,
            weight_scale_float32,
            names="feature_std and weight_scale",
        )

        self._feature_dim = feature_dim
        self._n_tasks = n_tasks
        self._inner_hidden = inner_hidden
        self._outer_components = outer_components
        self._n_contexts = n_contexts
        self._context_length = context_length
        self._feature_std = feature_std_float32
        self._weight_scale = weight_scale_float32
        self._amplitude_scale = amplitude_scale_float32
        self._noise_std = noise_std_float32
        self._resource_budget = resource_budget

    @property
    def feature_dim(self) -> int:
        """Return the raw observation dimension."""
        return self._feature_dim

    @property
    def target_dim(self) -> int:
        """Return the number of supervised tasks."""
        return self._n_tasks

    @property
    def resource_budget(self) -> dict[str, int]:
        """Complete resident state payload accounting."""
        return dict(self._resource_budget)

    def init(self, key: Array) -> CompositionalState:
        """Initialize stream state.

        Args:
            key: JAX PRNG key.

        Returns:
            Initialized ``CompositionalState``.
        """
        key, k_iw, k_ib, k_ow, k_ob, k_amp = jr.split(key, 6)

        inner_shape = (
            self._n_contexts,
            self._n_tasks,
            self._outer_components,
            self._inner_hidden,
            self._feature_dim,
        )
        outer_shape = (
            self._n_contexts,
            self._n_tasks,
            self._outer_components,
            self._inner_hidden,
        )
        component_shape = (
            self._n_contexts,
            self._n_tasks,
            self._outer_components,
        )

        inner_w = (
            self._weight_scale
            * jr.normal(k_iw, inner_shape, dtype=jnp.float32)
            / jnp.sqrt(float(self._feature_dim))
        )
        inner_b = 0.25 * jr.normal(k_ib, outer_shape, dtype=jnp.float32)
        outer_w = (
            self._weight_scale
            * jr.normal(k_ow, outer_shape, dtype=jnp.float32)
            / jnp.sqrt(float(self._inner_hidden))
        )
        outer_b = 0.25 * jr.normal(k_ob, component_shape, dtype=jnp.float32)
        amplitudes = self._amplitude_scale * jr.normal(
            k_amp, component_shape, dtype=jnp.float32
        )

        return CompositionalState(
            key=key,
            inner_w=inner_w,
            inner_b=inner_b,
            outer_w=outer_w,
            outer_b=outer_b,
            amplitudes=amplitudes,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def step(
        self,
        state: CompositionalState,
        idx: Array,
    ) -> tuple[TimeStep, CompositionalState]:
        """Generate one multitask compositional sample.

        Args:
            state: Current stream state.
            idx: Step index (unused; kept for ``ScanStream`` protocol).

        Returns:
            Tuple of (``TimeStep``, new stream state).
        """
        del idx
        key, k_x, k_noise = jr.split(state.key, 3)
        x = self._feature_std * jr.normal(
            k_x, (self._feature_dim,), dtype=jnp.float32
        )
        context_idx = (state.step_count // self._context_length) % self._n_contexts

        # Pull this context's per-task per-component subnetworks.
        inner_w = state.inner_w[context_idx]    # (T, O, H, F)
        inner_b = state.inner_b[context_idx]    # (T, O, H)
        outer_w = state.outer_w[context_idx]    # (T, O, H)
        outer_b = state.outer_b[context_idx]    # (T, O)
        amps = state.amplitudes[context_idx]    # (T, O)

        # inner = tanh(V x + c) -> (T, O, H)
        inner_pre = jnp.einsum("tohf,f->toh", inner_w, x) + inner_b
        inner = jnp.tanh(inner_pre)
        # outer = tanh(W inner + b) -> (T, O)
        outer_pre = jnp.sum(outer_w * inner, axis=-1) + outer_b
        outer = jnp.tanh(outer_pre)
        # target_t = sum_o amps[t, o] * outer[t, o]
        target = jnp.sum(amps * outer, axis=-1)
        noise = self._noise_std * jr.normal(
            k_noise, (self._n_tasks,), dtype=jnp.float32
        )
        target = target + noise

        timestep = TimeStep(observation=x, target=target)
        new_state = state.replace(  # type: ignore[attr-defined]
            key=key,
            step_count=_saturating_step_count(state.step_count),
        )
        return timestep, new_state
