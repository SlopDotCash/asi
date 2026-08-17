"""Step 2 feature-discovery streams.

These streams are designed for the Alberta Plan's Step 2 setting:
continual supervised learning with vector-valued targets, nonlinear latent
features, and changing feature relevance.  The latent features are known to
the stream but hidden from learners.

Scope caveat: both oracles here (tanh units, pairwise products) lie *inside*
the hypothesis class of the corresponding Step 2 learners, so "discovery"
reduces to selecting and replacing the right features from a representable
pool under a fixed budget.  Feature *construction* proper — targets whose
minimal representation lies outside a one-layer feature bank — is probed by
:mod:`alberta_framework.streams.out_of_class`.
"""

import math
from numbers import Integral, Real
from typing import Any, cast

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
from alberta_framework._float32 import round_real_to_float32_with_ratio
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
class NonlinearFeatureDiscoveryState:
    """State for ``NonlinearFeatureDiscoveryStream``.

    Attributes:
        key: PRNG key for sample generation.
        latent_weights: Hidden feature weights, shape ``(n_latents, feature_dim)``.
        latent_biases: Hidden feature biases, shape ``(n_latents,)``.
        context_weights: Per-context task weights over latent features,
            shape ``(n_contexts, n_tasks, n_latents)``.
        linear_weights: Small direct linear component, shape
            ``(n_tasks, feature_dim)``.
        step_count: Number of generated samples.
    """

    key: PRNGKeyArray
    latent_weights: Float[Array, "n_latents feature_dim"]
    latent_biases: Float[Array, " n_latents"]
    context_weights: Float[Array, "n_contexts n_tasks n_latents"]
    linear_weights: Float[Array, "n_tasks feature_dim"]
    step_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class InteractionFeatureDiscoveryState:
    """State for ``InteractionFeatureDiscoveryStream``.

    The hidden oracle features are pairwise products ``x_i * x_j``.  The pair
    list is fixed, while context weights determine which products are useful.
    """

    key: PRNGKeyArray
    pair_left: Int[Array, " n_pairs"]
    pair_right: Int[Array, " n_pairs"]
    context_weights: Float[Array, "n_contexts n_tasks n_pairs"]
    linear_weights: Float[Array, "n_tasks feature_dim"]
    step_count: Int[Array, ""]


class NonlinearFeatureDiscoveryStream:
    """Non-stationary multitask stream with hidden nonlinear features.

    Observations are raw vectors ``x_t``.  Targets are vector-valued:

    ``y*_t = W_c phi(x_t) + L x_t + noise``

    where ``phi`` is a fixed bank of hidden nonlinear features and ``W_c``
    changes by context.  This creates a controlled Step 2 benchmark: useful
    nonlinear features exist, relevance shifts over time, and the learner has
    only a limited budget of representable features.
    """

    def __init__(
        self,
        feature_dim: int,
        n_tasks: int = 4,
        n_latents: int = 32,
        n_contexts: int = 8,
        context_length: int = 500,
        active_latents_per_context: int = 6,
        feature_std: float = 1.0,
        latent_scale: float = 1.0,
        linear_scale: float = 0.05,
        noise_std: float = 0.01,
    ):
        """Initialize the nonlinear feature-discovery stream.

        Args:
            feature_dim: Raw observation dimension.
            n_tasks: Number of supervised output heads.
            n_latents: Number of hidden oracle nonlinear features.
            n_contexts: Number of recurring relevance contexts.
            context_length: Number of steps before switching context.
            active_latents_per_context: Expected number of useful latent
                features per task/context.
            feature_std: Standard deviation of raw observations.
            latent_scale: Scale of oracle latent weights.
            linear_scale: Scale of the direct linear target component.
            noise_std: Standard deviation of target noise.
        """
        feature_dim_val = _require_int("feature_dim", feature_dim, minimum=1, maximum=_INT32_MAX)
        n_tasks_val = _require_int("n_tasks", n_tasks, minimum=1, maximum=_INT32_MAX)
        n_latents_val = _require_int("n_latents", n_latents, minimum=1, maximum=_INT32_MAX)
        n_contexts_val = _require_int("n_contexts", n_contexts, minimum=1, maximum=_INT32_MAX)
        context_length_val = _require_int(
            "context_length", context_length, minimum=1, maximum=_INT32_MAX
        )
        active_latents_per_context_val = _require_int(
            "active_latents_per_context",
            active_latents_per_context,
            minimum=1,
            maximum=_INT32_MAX,
        )
        feature_std_val = _require_positive_real("feature_std", feature_std)
        latent_scale_val = _require_positive_real("latent_scale", latent_scale)
        linear_scale_val = _require_nonnegative_real("linear_scale", linear_scale)
        noise_std_val = _require_nonnegative_real("noise_std", noise_std)

        self._feature_dim = feature_dim_val
        self._n_tasks = n_tasks_val
        self._n_latents = n_latents_val
        self._n_contexts = n_contexts_val
        self._context_length = context_length_val
        self._active_latents_per_context = active_latents_per_context_val
        self._feature_std = feature_std_val
        self._latent_scale = latent_scale_val
        self._linear_scale = linear_scale_val
        self._noise_std = noise_std_val

    @property
    def feature_dim(self) -> int:
        """Return the raw observation dimension."""
        return self._feature_dim

    @property
    def target_dim(self) -> int:
        """Return the number of supervised tasks."""
        return self._n_tasks

    @property
    def n_tasks(self) -> int:
        """Return the number of supervised tasks."""
        return self._n_tasks

    @property
    def n_latents(self) -> int:
        """Return the number of hidden oracle features."""
        return self._n_latents

    @property
    def n_contexts(self) -> int:
        """Return the number of recurring relevance contexts."""
        return self._n_contexts

    @property
    def context_length(self) -> int:
        """Return the context length."""
        return self._context_length

    @property
    def active_latents_per_context(self) -> int:
        """Return the active latents per context."""
        return self._active_latents_per_context

    @property
    def feature_std(self) -> float:
        """Return the feature standard deviation."""
        return self._feature_std

    @property
    def latent_scale(self) -> float:
        """Return the latent scale."""
        return self._latent_scale

    @property
    def linear_scale(self) -> float:
        """Return the linear scale."""
        return self._linear_scale

    @property
    def noise_std(self) -> float:
        """Return the noise standard deviation."""
        return self._noise_std

    def init(self, key: Array) -> NonlinearFeatureDiscoveryState:
        """Initialize stream state."""
        key, k_latent, k_bias, k_ctx, k_mask, k_linear = jr.split(key, 6)

        # LeCun-style 1/sqrt(feature_dim) scaling gives each latent a roughly
        # unit-variance preactivation (at latent_scale = feature_std = 1). The
        # bias std of 0.25 is small against that, spreading the tanh operating
        # points across latents without pushing units into saturation.
        latent_weights = (
            self._latent_scale
            * jr.normal(k_latent, (self._n_latents, self._feature_dim), dtype=jnp.float32)
            / jnp.sqrt(float(self._feature_dim))
        )
        latent_biases = 0.25 * jr.normal(k_bias, (self._n_latents,), dtype=jnp.float32)

        dense_context_weights = jr.normal(
            k_ctx,
            (self._n_contexts, self._n_tasks, self._n_latents),
            dtype=jnp.float32,
        )
        keep_prob = min(1.0, self._active_latents_per_context / self._n_latents)
        mask = jr.bernoulli(
            k_mask,
            keep_prob,
            (self._n_contexts, self._n_tasks, self._n_latents),
        )
        # Dividing each head's weights by sqrt(#active latents) keeps target
        # variance roughly constant across contexts: a context switch changes
        # WHICH features matter, not the scale of the regression problem.
        context_weights = dense_context_weights * mask.astype(jnp.float32)
        norm = jnp.sqrt(jnp.maximum(jnp.sum(mask, axis=-1, keepdims=True), 1.0))
        context_weights = context_weights / norm

        linear_weights = self._linear_scale * jr.normal(
            k_linear, (self._n_tasks, self._feature_dim), dtype=jnp.float32
        )

        return NonlinearFeatureDiscoveryState(
            key=key,
            latent_weights=latent_weights,
            latent_biases=latent_biases,
            context_weights=context_weights,
            linear_weights=linear_weights,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def step(
        self,
        state: NonlinearFeatureDiscoveryState,
        idx: Array,
    ) -> tuple[TimeStep, NonlinearFeatureDiscoveryState]:
        """Generate one multitask supervised sample."""
        del idx
        key, k_x, k_noise = jr.split(state.key, 3)

        x = self._feature_std * jr.normal(
            k_x, (self._feature_dim,), dtype=jnp.float32
        )
        latents = jnp.tanh(state.latent_weights @ x + state.latent_biases)

        context_idx = (state.step_count // self._context_length) % self._n_contexts
        task_weights = state.context_weights[context_idx]
        target = task_weights @ latents + state.linear_weights @ x
        noise = self._noise_std * jr.normal(k_noise, (self._n_tasks,), dtype=jnp.float32)
        target = target + noise

        timestep = TimeStep(observation=x, target=target)
        new_state = state.replace(key=key, step_count=state.step_count + 1)  # type: ignore[attr-defined]
        return timestep, new_state


def collect_feature_discovery_stream(
    stream: Any,
    num_steps: int,
    key: Array,
) -> tuple[Array, Array]:
    """Collect a fixed array view of a feature-discovery stream.

    This helper is for controlled experiments where multiple learners should
    see the exact same stream.  It still uses the one-step stream interface and
    ``jax.lax.scan``; it does not imply experience replay inside a learner.
    """
    import jax

    num_steps = _require_int("num_steps", num_steps, minimum=1, maximum=_INT32_MAX)
    if not hasattr(stream, "init") or not callable(stream.init):
        raise TypeError("stream must provide a callable init method")
    if not hasattr(stream, "step") or not callable(stream.step):
        raise TypeError("stream must provide a callable step method")

    state = stream.init(key)

    def step_fn(
        carry: Any,
        idx: Array,
    ) -> tuple[Any, tuple[Array, Array]]:
        timestep, new_state = stream.step(carry, idx)
        return new_state, (timestep.observation, timestep.target)

    _, (observations, targets) = jax.lax.scan(
        step_fn, state, jnp.arange(num_steps)
    )
    return observations, targets


class InteractionFeatureDiscoveryStream:
    """Non-stationary stream whose useful features are pairwise products.

    This benchmark gives Step 2 a sharper target than generic MLP learning.
    The useful nonlinear features are literal combinations of existing raw
    features:

    ``phi_ij(x_t) = x_t[i] * x_t[j]``

    The learner observes only ``x_t`` and vector target ``y*_t``.  Contexts
    change which products matter, so a bounded learner must rank and replace
    features rather than merely grow capacity.
    """

    def __init__(
        self,
        feature_dim: int,
        n_tasks: int = 4,
        n_contexts: int = 8,
        context_length: int = 500,
        active_pairs_per_context: int = 6,
        feature_std: float = 1.0,
        linear_scale: float = 0.01,
        noise_std: float = 0.01,
        include_squares: bool = False,
    ):
        """Initialize the interaction stream.

        Args:
            feature_dim: Raw observation dimension.
            n_tasks: Number of supervised output heads.
            n_contexts: Number of recurring relevance contexts.
            context_length: Steps before switching context.
            active_pairs_per_context: Number of active pair-products per
                task/context, capped at the available pair count.
            feature_std: Standard deviation of raw observations.
            linear_scale: Scale of the small direct linear component.
            noise_std: Standard deviation of target noise.
            include_squares: Whether to include ``x_i * x_i`` oracle features.
        """
        feature_dim_val = _require_int("feature_dim", feature_dim, minimum=2, maximum=_INT32_MAX)
        n_tasks_val = _require_int("n_tasks", n_tasks, minimum=1, maximum=_INT32_MAX)
        n_contexts_val = _require_int("n_contexts", n_contexts, minimum=1, maximum=_INT32_MAX)
        context_length_val = _require_int(
            "context_length", context_length, minimum=1, maximum=_INT32_MAX
        )
        active_pairs_per_context_val = require_positive_builtin_int(
            active_pairs_per_context,
            name="active_pairs_per_context",
        )
        if active_pairs_per_context_val > _INT32_MAX:
            raise ValueError(f"active_pairs_per_context must be <= {_INT32_MAX}")
        feature_std_val = _require_positive_real("feature_std", feature_std)
        linear_scale_val = _require_nonnegative_real("linear_scale", linear_scale)
        noise_std_val = _require_nonnegative_real("noise_std", noise_std)
        if type(include_squares) is bool or type(include_squares) is np.bool_:
            include_squares_val = bool(include_squares)
        else:
            raise TypeError("include_squares must be a boolean")

        self._feature_dim = feature_dim_val
        self._n_tasks = n_tasks_val
        self._n_contexts = n_contexts_val
        self._context_length = context_length_val
        self._active_pairs_per_context = active_pairs_per_context_val
        self._feature_std = feature_std_val
        self._linear_scale = linear_scale_val
        self._noise_std = noise_std_val
        self._include_squares = include_squares_val

    @property
    def feature_dim(self) -> int:
        """Return the raw observation dimension."""
        return self._feature_dim

    @property
    def target_dim(self) -> int:
        """Return the number of supervised tasks."""
        return self._n_tasks

    @property
    def n_tasks(self) -> int:
        """Return the number of supervised tasks."""
        return self._n_tasks

    @property
    def n_contexts(self) -> int:
        """Return the number of recurring relevance contexts."""
        return self._n_contexts

    @property
    def context_length(self) -> int:
        """Return the context length."""
        return self._context_length

    @property
    def active_pairs_per_context(self) -> int:
        """Return the active pairs per context."""
        return self._active_pairs_per_context

    @property
    def feature_std(self) -> float:
        """Return the feature standard deviation."""
        return self._feature_std

    @property
    def linear_scale(self) -> float:
        """Return the linear scale."""
        return self._linear_scale

    @property
    def noise_std(self) -> float:
        """Return the noise standard deviation."""
        return self._noise_std

    @property
    def include_squares(self) -> bool:
        """Return whether square interactions are included."""
        return self._include_squares

    def _pairs(self) -> tuple[Array, Array]:
        pairs = []
        for i in range(self._feature_dim):
            start = i if self._include_squares else i + 1
            for j in range(start, self._feature_dim):
                pairs.append((i, j))
        arr = jnp.array(pairs, dtype=jnp.int32)
        return arr[:, 0], arr[:, 1]

    def init(self, key: Array) -> InteractionFeatureDiscoveryState:
        """Initialize stream state."""
        key, k_ctx, k_mask, k_linear = jr.split(key, 4)
        pair_left, pair_right = self._pairs()
        n_pairs = pair_left.shape[0]

        dense_context_weights = jr.normal(
            k_ctx,
            (self._n_contexts, self._n_tasks, n_pairs),
            dtype=jnp.float32,
        )
        active_count = min(self._active_pairs_per_context, n_pairs)
        mask_scores = jr.uniform(
            k_mask,
            (self._n_contexts, self._n_tasks, n_pairs),
            dtype=jnp.float32,
        )
        mask = stable_smallest_mask(mask_scores, active_count)
        # Same sqrt(#active) normalization as the nonlinear stream: context
        # switches redirect relevance without changing the target scale.
        context_weights = dense_context_weights * mask.astype(jnp.float32)
        norm = jnp.sqrt(jnp.maximum(jnp.sum(mask, axis=-1, keepdims=True), 1.0))
        context_weights = context_weights / norm

        linear_weights = self._linear_scale * jr.normal(
            k_linear, (self._n_tasks, self._feature_dim), dtype=jnp.float32
        )

        return InteractionFeatureDiscoveryState(
            key=key,
            pair_left=pair_left,
            pair_right=pair_right,
            context_weights=context_weights,
            linear_weights=linear_weights,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def step(
        self,
        state: InteractionFeatureDiscoveryState,
        idx: Array,
    ) -> tuple[TimeStep, InteractionFeatureDiscoveryState]:
        """Generate one multitask interaction sample."""
        del idx
        key, k_x, k_noise = jr.split(state.key, 3)
        x = self._feature_std * jr.normal(
            k_x, (self._feature_dim,), dtype=jnp.float32
        )
        interactions = x[state.pair_left] * x[state.pair_right]
        context_idx = (state.step_count // self._context_length) % self._n_contexts
        task_weights = state.context_weights[context_idx]
        target = task_weights @ interactions + state.linear_weights @ x
        noise = self._noise_std * jr.normal(k_noise, (self._n_tasks,), dtype=jnp.float32)
        target = target + noise

        timestep = TimeStep(observation=x, target=target)
        new_state = state.replace(key=key, step_count=state.step_count + 1)  # type: ignore[attr-defined]
        return timestep, new_state
