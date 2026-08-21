# mypy: disable-error-code="call-arg,name-defined"
"""Online behavior/action prediction for discrete-action agents.

The behavior model is a temporally uniform supervised learner for
``P(A_t | features_t)``.  It is deliberately separate from control: SARSA,
actor-critic, scripted policies, external logs, and future dream rollouts can
all feed the same observed ``(features, action)`` stream into this model.
"""

from __future__ import annotations

import dataclasses
import functools
import operator
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

from alberta_framework.core._float32_scalars import validated_float32_scalar

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
_ACTUAL_FLOAT_TYPES = frozenset((float, np.float16, np.float32, np.float64, np.longdouble))
_CONFIG_FIELDS = frozenset(
    {
        "n_actions",
        "step_size",
        "temperature",
        "l2_penalty",
        "max_gradient_norm",
        "min_probability",
        "ratio_clip",
        "diagnostic_decay",
    }
)


def _has_exact_type(value: object, allowed: tuple[type, ...] | frozenset[type]) -> bool:
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


def _validated_config_float(name: str, value: object, **bounds: Any) -> float:
    """Validate only trusted concrete host scalar types at the float32 sink."""
    if not _has_exact_type(value, _ACTUAL_INT_TYPES | _ACTUAL_FLOAT_TYPES):
        raise ValueError(f"{name} must be a finite real scalar")
    return validated_float32_scalar(name, value, **bounds)


def _resource_counts(n_actions: int, feature_dim: int) -> tuple[int, int]:
    """Return exact trainable scalars and bytes, rejecting unsafe derived counts."""
    trainable = n_actions * feature_dim + n_actions
    state_nbytes = 4 * (trainable + 3 + 1 + 2)
    if trainable > _INT32_MAX:
        raise ValueError(f"derived trainable scalars must be at most {_INT32_MAX}")
    if state_nbytes > _INT32_MAX:
        raise ValueError(f"derived state_nbytes must be at most {_INT32_MAX}")
    return trainable, state_nbytes


# Matches the class of ceiling already established for other scan-driven
# array-loop runners in ``core`` (e.g. ``average_reward._AVERAGE_REWARD_SEQUENCE_MAX_STEPS``,
# ``learners._LEARNING_LOOP_MAX_STEPS``). Set with generous headroom above
# this module's largest exercised sequence (12 steps, see
# ``test_scan_loop_and_jit_compatibility``) while still bounding the leading
# axis that ``run_behavior_model_from_arrays`` hands straight to
# ``jax.lax.scan``.
_BEHAVIOR_MODEL_SEQUENCE_MAX_STEPS = 50_000


def _require_behavior_model_sequence_length(name: str, value: object) -> int:
    """Reject an oversized or malformed leading axis before it drives a scan.

    ``run_behavior_model_from_arrays`` hands ``observations`` straight to
    ``jax.lax.scan`` with no bound on the leading (step) axis. A hostile or
    mistaken caller supplying a huge leading length forces JAX to materialize
    per-step outputs (and, for large enough arrays, the inputs) at that
    length, exhausting memory or hanging the process well before any step
    executes.
    """
    if not (type(value) is np.ndarray or isinstance(value, jax.Array)):
        raise TypeError(f"{name} must be a trusted array")
    array = np.asarray(value) if type(value) is np.ndarray else value
    if array.ndim < 1:
        raise ValueError(f"{name} must have a leading step axis")
    length = int(array.shape[0])
    if length < 1 or length > _BEHAVIOR_MODEL_SEQUENCE_MAX_STEPS:
        raise ValueError(
            f"{name} length must be an integer in [1, {_BEHAVIOR_MODEL_SEQUENCE_MAX_STEPS}]"
        )
    return length


def _behavior_model_update_working_set_bytes(n_actions: int, feature_dim: int) -> int:
    """Source persist, proposed persist, committed persist, and returned extras.

    ``update`` keeps the source state, the proposed state, and the
    transaction-selected result simultaneously live.  The worst-case path
    also materializes a weight/bias gradient plus an L2 or clip copy,
    logits / probabilities / one-hot / logit-error, the observation, and
    neutralized returned logits/probabilities beside the committed state.
    """

    trainable = n_actions * feature_dim + n_actions
    persist_scalars = trainable + 6
    update_scalars = (
        3 * persist_scalars
        + 2 * n_actions * feature_dim
        + 2 * n_actions
        + 4 * n_actions
        + feature_dim
        + 2 * n_actions
        + 16
    )
    return 4 * update_scalars


def _preflight_behavior_model_update_working_set(
    n_actions: int,
    feature_dim: int,
) -> None:
    """Reject an update envelope the behavior model cannot name."""

    working_set_bytes = _behavior_model_update_working_set_bytes(
        n_actions,
        feature_dim,
    )
    if working_set_bytes > _INT32_MAX:
        raise ValueError(
            "behavior-model update working set byte count must fit signed int32"
        )


def _saturating_int32_increment(value: Array) -> Array:
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    counter = jnp.asarray(value, dtype=jnp.int32)
    return jnp.minimum(jnp.maximum(counter, 0), maximum - 1) + 1


def _integer_action_ids(
    actions: object,
    *,
    n_actions: int,
    expected_shape: tuple[int, ...],
) -> tuple[Array, Bool[Array, " *shape"]]:
    """Validate original-width action IDs before exposing safe int32 indices."""

    actual_type = type(actions)
    if issubclass(actual_type, jax.core.Tracer):
        traced_actions = cast(Any, actions)
        raw_shape = tuple(traced_actions.shape)
        raw_dtype = np.dtype(traced_actions.dtype)
    else:
        trusted_host = (
            actual_type is np.ndarray
            or any(actual_type is allowed_type for allowed_type in _ACTUAL_INT_TYPES)
            or issubclass(actual_type, jax.Array)
        )
        if not trusted_host:
            raise TypeError("actions must be a trusted array")
        host_actions = np.asarray(actions)
        raw_shape = tuple(host_actions.shape)
        raw_dtype = host_actions.dtype
        if not np.issubdtype(raw_dtype, np.integer):
            raise ValueError("actions must have an integer dtype")
        if not bool(np.all((host_actions >= 0) & (host_actions < n_actions))):
            raise ValueError(f"actions must lie in [0, {n_actions})")
    if not np.issubdtype(raw_dtype, np.integer):
        raise ValueError("actions must have an integer dtype")
    try:
        broadcast_shape = np.broadcast_shapes(raw_shape, expected_shape)
    except ValueError as error:
        raise ValueError(f"actions must broadcast to shape {expected_shape}") from error
    if broadcast_shape != expected_shape:
        raise ValueError(f"actions must broadcast to shape {expected_shape}")
    raw_actions = jnp.broadcast_to(jnp.asarray(actions), expected_shape)
    valid = (raw_actions >= 0) & (raw_actions < n_actions)
    safe = jnp.where(valid, raw_actions, 0).astype(jnp.int32)
    return safe, valid


def floor_and_renormalize_probabilities(
    probabilities: Array,
    min_probability: float = 1e-6,
) -> Array:
    """Floor probabilities and return a valid simplex along the last axis.

    This helper is for sampling or reporting a proper simplex distribution
    whose entries are at least ``min_probability``. Importance-ratio denominators
    should use :func:`selected_action_probabilities`, which floors only the
    selected action probability and does not change other actions.
    """
    probs = jnp.asarray(probabilities, dtype=jnp.float32)
    n_actions = probabilities.shape[-1]
    if min_probability * n_actions >= 1.0:
        return jnp.ones_like(probs) / n_actions
    clipped = jnp.maximum(probs, 0.0)
    normalizer = jnp.sum(clipped, axis=-1, keepdims=True)
    uniform = jnp.ones_like(probs) / n_actions
    # A zero, non-finite, or precision-collapsed normalizer cannot yield a
    # usable simplex (all-zero mass, or float32 ratios that underflow to
    # zero): fall back to the uniform distribution so the documented
    # sum-to-one invariant holds. Well-formed inputs keep their exact output.
    raw_normalized = clipped / normalizer
    normalized_sum = jnp.sum(raw_normalized, axis=-1, keepdims=True)
    usable = jnp.isfinite(normalized_sum) & (normalized_sum > 0.5)
    safe_normalizer = jnp.where(usable, normalizer, jnp.ones_like(normalizer))
    normalized = jnp.where(usable, clipped / safe_normalizer, uniform)
    floor_mass = jnp.asarray(min_probability * n_actions, dtype=jnp.float32)
    return jnp.asarray(min_probability, dtype=jnp.float32) + (1.0 - floor_mass) * normalized


def selected_action_probabilities(
    probabilities: Array,
    actions: Array,
    min_probability: float = 1e-6,
) -> Array:
    """Return floor-clipped probabilities for selected discrete actions.

    ``probabilities`` may be a single action distribution with shape
    ``(n_actions,)`` or a batch with actions on the last axis. ``actions`` must
    broadcast to ``probabilities.shape[:-1]``.
    """
    probs = jnp.asarray(probabilities, dtype=jnp.float32)
    action_ids, actions_valid = _integer_action_ids(
        actions,
        n_actions=probs.shape[-1],
        expected_shape=tuple(probs.shape[:-1]),
    )
    one_hot = jax.nn.one_hot(action_ids, probs.shape[-1], dtype=jnp.float32)
    selected = jnp.sum(probs * one_hot, axis=-1)
    floor = jnp.asarray(min_probability, dtype=jnp.float32)
    return jnp.where(actions_valid, jnp.maximum(selected, floor), floor)


def action_log_likelihoods(
    probabilities: Array,
    actions: Array,
    min_probability: float = 1e-6,
) -> Array:
    """Return log-likelihoods for selected actions under a behavior model."""
    return jnp.log(
        selected_action_probabilities(
            probabilities,
            actions,
            min_probability=min_probability,
        )
    )


def clipped_importance_ratios(
    target_probabilities: Array,
    behavior_probabilities: Array,
    actions: Array,
    *,
    clip: float | None = 10.0,
    min_behavior_probability: float = 1e-6,
) -> Array:
    """Compute selected-action target/behavior ratios with safe denominators.

    Args:
        target_probabilities: Target policy probabilities with actions on the
            last axis.
        behavior_probabilities: Behavior model probabilities with actions on
            the last axis.
        actions: Discrete selected actions.
        clip: Optional upper bound on ratios. ``None`` disables clipping.
        min_behavior_probability: Lower bound for behavior denominators.

    Returns:
        Per-sample ratios with shape ``target_probabilities.shape[:-1]``.
    """
    target = selected_action_probabilities(
        target_probabilities,
        actions,
        min_probability=0.0,
    )
    behavior = selected_action_probabilities(
        behavior_probabilities,
        actions,
        min_probability=min_behavior_probability,
    )
    ratios = target / behavior
    if clip is None:
        return ratios
    return jnp.minimum(ratios, jnp.asarray(clip, dtype=jnp.float32))


def epsilon_greedy_probabilities(
    q_values: Array,
    epsilon: Array,
    tie_tolerance: float = 1e-6,
) -> Array:
    """Return the exact epsilon-greedy action distribution for Q-values.

    This mirrors the SARSA/Q-learning policy surface: exploration is uniform
    over all actions and exploitation is uniform over maximal actions.
    """
    q = jnp.asarray(q_values, dtype=jnp.float32)
    n_actions = q.shape[-1]
    eps = jnp.asarray(epsilon, dtype=jnp.float32)
    max_q = jnp.max(q, axis=-1, keepdims=True)
    greedy_mask = jnp.isclose(q, max_q, atol=tie_tolerance, rtol=0.0).astype(jnp.float32)
    n_greedy = jnp.sum(greedy_mask, axis=-1, keepdims=True)
    explore = eps / n_actions
    exploit = (1.0 - eps) * greedy_mask / jnp.maximum(n_greedy, 1.0)
    return exploit + explore


@dataclasses.dataclass(frozen=True)
class BehaviorModelConfig:
    """Configuration for a linear online discrete behavior model.

    Attributes:
        n_actions: Number of discrete actions.
        step_size: Cross-entropy gradient step-size.
        temperature: Softmax temperature for behavior probabilities.
        l2_penalty: Optional L2 shrinkage on weights and biases.
        max_gradient_norm: Optional global gradient-norm clip before applying
            ``step_size``.
        min_probability: Probability floor for likelihood and ratio helpers.
        ratio_clip: Default ratio clip for off-policy helper methods.
        diagnostic_decay: EMA decay used for online reliability diagnostics.
    """

    n_actions: int
    step_size: float = 0.05
    temperature: float = 1.0
    l2_penalty: float = 0.0
    max_gradient_norm: float | None = None
    min_probability: float = 1e-6
    ratio_clip: float = 10.0
    diagnostic_decay: float = 0.99

    def __post_init__(self) -> None:
        """Validate scalar hyperparameters."""
        object.__setattr__(
            self,
            "n_actions",
            _require_int32("n_actions", self.n_actions, minimum=1),
        )
        _resource_counts(self.n_actions, 1)
        step_size = _validated_config_float("step_size", self.step_size, lower=0.0)
        temperature = _validated_config_float("temperature", self.temperature, positive=True)
        l2_penalty = _validated_config_float("l2_penalty", self.l2_penalty, lower=0.0)
        max_gradient_norm = (
            _validated_config_float(
                "max_gradient_norm",
                self.max_gradient_norm,
                positive=True,
            )
            if self.max_gradient_norm is not None
            else None
        )
        min_probability = _validated_config_float(
            "min_probability",
            self.min_probability,
            positive=True,
            upper=1.0,
            upper_inclusive=False,
        )
        ratio_clip = _validated_config_float("ratio_clip", self.ratio_clip, positive=True)
        diagnostic_decay = _validated_config_float(
            "diagnostic_decay",
            self.diagnostic_decay,
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
        )
        object.__setattr__(self, "step_size", step_size)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "l2_penalty", l2_penalty)
        object.__setattr__(self, "max_gradient_norm", max_gradient_norm)
        object.__setattr__(self, "min_probability", min_probability)
        object.__setattr__(self, "ratio_clip", ratio_clip)
        object.__setattr__(self, "diagnostic_decay", diagnostic_decay)

    def to_config(self) -> dict[str, Any]:
        """Serialize configuration to a JSON-compatible dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> BehaviorModelConfig:
        """Reconstruct from :meth:`to_config` output."""
        if type(config) is not dict:
            raise ValueError("BehaviorModelConfig payload must be an actual dict")
        if not all(type(key) is str for key in config) or set(config) != _CONFIG_FIELDS:
            raise ValueError("BehaviorModelConfig fields do not match the schema")
        return cls(**config)


@chex.dataclass(frozen=True)
class BehaviorModelState:
    """Immutable state for the behavior/action predictor."""

    weights: Float[Array, "n_actions feature_dim"]
    bias: Float[Array, " n_actions"]
    rng_key: Array
    step_count: Int[Array, ""]
    nll_ema: Float[Array, ""]
    accuracy_ema: Float[Array, ""]
    confidence_ema: Float[Array, ""]


@dataclasses.dataclass(frozen=True)
class BehaviorModelResourceBudget:
    """Exact persistent-state accounting for a configured feature width.

    The byte count covers the arrays in :class:`BehaviorModelState` when
    initialized with JAX's default two-word typed PRNG key. It excludes Python
    objects, compiled executables, and transient update buffers. The model has
    no replay storage.
    """

    feature_dim: int
    n_actions: int
    trainable_float32_scalars: int
    diagnostic_float32_scalars: int
    administrative_int32_scalars: int
    rng_uint32_scalars: int
    state_nbytes: int
    learned_float32_scalars_touched_per_update: int
    replay_capacity: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "feature_dim",
            _require_int32("feature_dim", self.feature_dim, minimum=1),
        )
        object.__setattr__(
            self,
            "n_actions",
            _require_int32("n_actions", self.n_actions, minimum=1),
        )
        object.__setattr__(
            self,
            "trainable_float32_scalars",
            _require_int32(
                "trainable_float32_scalars",
                self.trainable_float32_scalars,
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "diagnostic_float32_scalars",
            _require_int32(
                "diagnostic_float32_scalars",
                self.diagnostic_float32_scalars,
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "administrative_int32_scalars",
            _require_int32(
                "administrative_int32_scalars",
                self.administrative_int32_scalars,
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "rng_uint32_scalars",
            _require_int32("rng_uint32_scalars", self.rng_uint32_scalars, minimum=0),
        )
        object.__setattr__(
            self,
            "state_nbytes",
            _require_int32("state_nbytes", self.state_nbytes, minimum=0),
        )
        object.__setattr__(
            self,
            "learned_float32_scalars_touched_per_update",
            _require_int32(
                "learned_float32_scalars_touched_per_update",
                self.learned_float32_scalars_touched_per_update,
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "replay_capacity",
            _require_int32("replay_capacity", self.replay_capacity, minimum=0),
        )
        trainable, state_nbytes = _resource_counts(self.n_actions, self.feature_dim)
        expected = {
            "trainable_float32_scalars": trainable,
            "diagnostic_float32_scalars": 3,
            "administrative_int32_scalars": 1,
            "rng_uint32_scalars": 2,
            "state_nbytes": state_nbytes,
            "learned_float32_scalars_touched_per_update": trainable + 3,
            "replay_capacity": 0,
        }
        for name, expected_value in expected.items():
            if getattr(self, name) != expected_value:
                raise ValueError(
                    f"{name} must equal the exact behavior-model resource formula "
                    f"({expected_value})"
                )

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-compatible resource record."""
        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class BehaviorModelInputGradient:
    """Pre-update cross-entropy gradient with respect to input features.

    This result is read-only: computing it does not advance diagnostics, RNG,
    or parameters. ``gradient`` is the derivative of the unfloored softmax
    cross entropy. Probability flooring remains confined to reporting and
    importance-ratio safety. ``valid`` is the traced transaction verdict for
    the complete result: callers must gate any downstream state mutation on
    it. Numeric fields are neutralized when it is false so an invalid operand
    cannot poison a speculative update.
    """

    logits: Float[Array, " n_actions"]
    probabilities: Float[Array, " n_actions"]
    loss: Float[Array, ""]
    gradient: Float[Array, " feature_dim"]
    gradient_norm: Float[Array, ""]
    valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class BehaviorModelUpdateResult:
    """Result of one online behavior-model update."""

    state: BehaviorModelState
    logits: Float[Array, " n_actions"]
    probabilities: Float[Array, " n_actions"]
    action_probability: Float[Array, ""]
    log_likelihood: Float[Array, ""]
    loss: Float[Array, ""]
    entropy: Float[Array, ""]
    confidence: Float[Array, ""]
    predicted_action: Int[Array, ""]
    correct: Float[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class BehaviorModelSampleResult:
    """Result of sampling an action from the learned behavior model."""

    state: BehaviorModelState
    action: Int[Array, ""]
    probabilities: Float[Array, " n_actions"]
    action_probability: Float[Array, ""]
    log_likelihood: Float[Array, ""]


@chex.dataclass(frozen=True)
class BehaviorModelArrayResult:
    """Result from scan-based behavior-model learning."""

    state: BehaviorModelState
    probabilities: Float[Array, "num_steps n_actions"]
    action_probabilities: Float[Array, " num_steps"]
    log_likelihoods: Float[Array, " num_steps"]
    losses: Float[Array, " num_steps"]
    entropies: Float[Array, " num_steps"]
    confidences: Float[Array, " num_steps"]
    correct: Float[Array, " num_steps"]
    updates_applied: Bool[Array, " num_steps"]


class BehaviorModel:
    """Online softmax model of the behavior policy.

    The model learns from the actually executed action at every step using a
    one-step cross-entropy update.  It is suitable for estimating behavior
    denominators in off-policy ratios and for sampling plausible actions during
    short model-based rollouts.
    """

    def __init__(self, config: BehaviorModelConfig):
        """Initialize the model."""
        self._config = config

    @property
    def config(self) -> BehaviorModelConfig:
        """Behavior-model configuration."""
        return self._config

    def to_config(self) -> dict[str, Any]:
        """Serialize model configuration."""
        return {
            "type": "BehaviorModel",
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> BehaviorModel:
        """Reconstruct a behavior model from :meth:`to_config` output."""
        if type(config) is not dict:
            raise ValueError("BehaviorModel construction must be an actual dict")
        if not all(type(key) is str for key in config) or set(config) != {"type", "config"}:
            raise ValueError("BehaviorModel construction fields do not match the schema")
        if type(config["type"]) is not str or config["type"] != "BehaviorModel":
            raise ValueError("unexpected BehaviorModel construction type")
        nested = config["config"]
        if type(nested) is not dict:
            raise ValueError("BehaviorModel nested config must be an actual dict")
        return cls(BehaviorModelConfig.from_config(nested))

    def init(self, feature_dim: int, key: Array) -> BehaviorModelState:
        """Initialize parameters and diagnostics."""
        feature_dim = _require_int32("feature_dim", feature_dim, minimum=1)
        _resource_counts(self._config.n_actions, feature_dim)
        _preflight_behavior_model_update_working_set(
            self._config.n_actions,
            feature_dim,
        )
        return BehaviorModelState(
            weights=jnp.zeros(
                (self._config.n_actions, feature_dim),
                dtype=jnp.float32,
            ),
            bias=jnp.zeros((self._config.n_actions,), dtype=jnp.float32),
            rng_key=key,
            step_count=jnp.array(0, dtype=jnp.int32),
            nll_ema=jnp.array(0.0, dtype=jnp.float32),
            accuracy_ema=jnp.array(0.0, dtype=jnp.float32),
            confidence_ema=jnp.array(0.0, dtype=jnp.float32),
        )

    def resource_budget(self, feature_dim: int) -> BehaviorModelResourceBudget:
        """Return exact fixed-state accounting for ``feature_dim``.

        The implementation initializes ``weights`` and ``bias`` as trainable
        float32 arrays, keeps three float32 diagnostics and one int32 counter,
        and stores a default JAX typed key backed by two uint32 words.
        """
        feature_dim = _require_int32("feature_dim", feature_dim, minimum=1)
        trainable, state_nbytes = _resource_counts(self._config.n_actions, feature_dim)
        _preflight_behavior_model_update_working_set(
            self._config.n_actions,
            feature_dim,
        )
        diagnostics = 3
        administrative = 1
        rng_words = 2
        return BehaviorModelResourceBudget(
            feature_dim=feature_dim,
            n_actions=self._config.n_actions,
            trainable_float32_scalars=trainable,
            diagnostic_float32_scalars=diagnostics,
            administrative_int32_scalars=administrative,
            rng_uint32_scalars=rng_words,
            state_nbytes=state_nbytes,
            learned_float32_scalars_touched_per_update=trainable + diagnostics,
            replay_capacity=0,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict_logits(
        self,
        state: BehaviorModelState,
        observation: Array,
    ) -> Float[Array, " n_actions"]:
        """Predict behavior logits for one feature vector."""
        obs = jnp.asarray(observation, dtype=jnp.float32)
        return state.weights @ obs + state.bias

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict_probabilities(
        self,
        state: BehaviorModelState,
        observation: Array,
    ) -> Float[Array, " n_actions"]:
        """Predict behavior action probabilities for one feature vector."""
        logits = self.predict_logits(state, observation)
        return jax.nn.softmax(logits / self._config.temperature)

    def action_probability(
        self,
        state: BehaviorModelState,
        observation: Array,
        action: Array,
    ) -> Float[Array, ""]:
        """Return the floor-clipped probability of ``action``."""
        probs = self.predict_probabilities(state, observation)
        return selected_action_probabilities(
            probs,
            action,
            min_probability=self._config.min_probability,
        )

    def action_log_likelihood(
        self,
        state: BehaviorModelState,
        observation: Array,
        action: Array,
    ) -> Float[Array, ""]:
        """Return the floor-clipped log-likelihood of ``action``."""
        return jnp.log(self.action_probability(state, observation, action))

    def input_loss_gradient(
        self,
        state: BehaviorModelState,
        observation: Array,
        action: Array,
    ) -> BehaviorModelInputGradient:
        """Differentiate the pre-update prediction loss through its features.

        This is the supported causal bridge from partner prediction into a
        trainable state builder. Callers must invoke it before
        :meth:`update`, and before advancing a recurrent state builder to the
        next observation, so the gradient refers to the representation that
        produced the scored prediction. Any consumer that proposes a state
        mutation from this result must commit it only when ``result.valid`` is
        true.
        """
        action_id, action_valid = _integer_action_ids(
            action,
            n_actions=self._config.n_actions,
            expected_shape=(),
        )
        return cast(
            BehaviorModelInputGradient,
            self._input_loss_gradient_jit(
                state,
                observation,
                action_id,
                action_valid,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _input_loss_gradient_jit(
        self,
        state: BehaviorModelState,
        observation: Array,
        action_id: Array,
        action_valid: Array,
    ) -> BehaviorModelInputGradient:
        """Execute one already identity-checked input-gradient transaction."""
        cfg = self._config
        obs = jnp.asarray(observation, dtype=jnp.float32)
        logits = state.weights @ obs + state.bias
        scaled_logits = logits / cfg.temperature
        probabilities = jax.nn.softmax(scaled_logits)
        one_hot = jax.nn.one_hot(action_id, cfg.n_actions, dtype=jnp.float32)
        loss = -jnp.sum(one_hot * jax.nn.log_softmax(scaled_logits))
        logit_gradient = (probabilities - one_hot) / cfg.temperature
        gradient = state.weights.T @ logit_gradient
        gradient_norm = jnp.linalg.norm(gradient)
        # Inf observation makes softmax NaN and W.T @ (p - one_hot)
        # non-finite, including 0*inf on a silent feature. The parameter
        # update already no-ops; this pre-update bridge must not hand a
        # NaN gradient to the state builder.
        inputs_valid = (
            jnp.all(jnp.isfinite(obs))
            & action_valid
        )
        source_finite = jnp.all(jnp.isfinite(state.weights)) & jnp.all(
            jnp.isfinite(state.bias)
        )
        proposed_finite = (
            jnp.all(jnp.isfinite(logits))
            & jnp.all(jnp.isfinite(probabilities))
            & jnp.isfinite(loss)
            & jnp.all(jnp.isfinite(gradient))
            & jnp.isfinite(gradient_norm)
        )
        valid = source_finite & inputs_valid & proposed_finite
        zero_logits = jnp.zeros_like(logits)
        zero_gradient = jnp.zeros_like(gradient)
        zero_loss = jnp.asarray(0.0, dtype=jnp.float32)
        return BehaviorModelInputGradient(
            logits=jnp.where(valid, logits, zero_logits),
            probabilities=jnp.where(valid, probabilities, jnp.zeros_like(probabilities)),
            loss=jnp.where(valid, loss, zero_loss),
            gradient=jnp.where(valid, gradient, zero_gradient),
            gradient_norm=jnp.where(valid, gradient_norm, zero_loss),
            valid=valid,
        )

    def update(
        self,
        state: BehaviorModelState,
        observation: Array,
        action: Array,
    ) -> BehaviorModelUpdateResult:
        """Update the behavior model from one observed action."""
        action_id, action_valid = _integer_action_ids(
            action,
            n_actions=self._config.n_actions,
            expected_shape=(),
        )
        return cast(
            BehaviorModelUpdateResult,
            self._update_jit(
                state,
                observation,
                action_id,
                action_valid,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_jit(
        self,
        state: BehaviorModelState,
        observation: Array,
        action_id: Array,
        action_valid: Array,
    ) -> BehaviorModelUpdateResult:
        """Execute one already identity-checked atomic update."""
        cfg = self._config
        obs = jnp.asarray(observation, dtype=jnp.float32)
        logits = state.weights @ obs + state.bias
        probabilities = jax.nn.softmax(logits / cfg.temperature)
        one_hot = jax.nn.one_hot(action_id, cfg.n_actions, dtype=jnp.float32)

        logit_error = (one_hot - probabilities) / cfg.temperature
        weight_gradient = logit_error[:, None] * obs[None, :]
        bias_gradient = logit_error
        if cfg.l2_penalty > 0.0:
            weight_gradient = weight_gradient - cfg.l2_penalty * state.weights
            bias_gradient = bias_gradient - cfg.l2_penalty * state.bias

        if cfg.max_gradient_norm is not None:
            grad_norm = jnp.sqrt(
                jnp.sum(weight_gradient * weight_gradient) + jnp.sum(bias_gradient * bias_gradient)
            )
            grad_scale = jnp.minimum(
                1.0,
                jnp.asarray(cfg.max_gradient_norm, dtype=jnp.float32)
                / jnp.maximum(grad_norm, 1e-12),
            )
            weight_gradient = grad_scale * weight_gradient
            bias_gradient = grad_scale * bias_gradient

        action_prob = selected_action_probabilities(
            probabilities,
            action_id,
            min_probability=cfg.min_probability,
        )
        log_likelihood = jnp.log(action_prob)
        loss = -log_likelihood
        entropy = -jnp.sum(probabilities * jnp.log(jnp.maximum(probabilities, cfg.min_probability)))
        confidence = jnp.max(probabilities)
        predicted_action = jnp.argmax(probabilities).astype(jnp.int32)
        correct = (predicted_action == action_id).astype(jnp.float32)

        decay = jnp.asarray(cfg.diagnostic_decay, dtype=jnp.float32)
        first = state.step_count == 0
        carried_nll = jnp.where(decay == 0.0, jnp.zeros_like(state.nll_ema), decay * state.nll_ema)
        carried_accuracy = jnp.where(
            decay == 0.0, jnp.zeros_like(state.accuracy_ema), decay * state.accuracy_ema
        )
        carried_confidence = jnp.where(
            decay == 0.0, jnp.zeros_like(state.confidence_ema), decay * state.confidence_ema
        )
        nll_ema = jnp.where(
            first,
            loss,
            carried_nll + (1.0 - decay) * loss,
        )
        accuracy_ema = jnp.where(
            first,
            correct,
            carried_accuracy + (1.0 - decay) * correct,
        )
        confidence_ema = jnp.where(
            first,
            confidence,
            carried_confidence + (1.0 - decay) * confidence,
        )

        new_state = state.replace(  # type: ignore[attr-defined]
            weights=state.weights + cfg.step_size * weight_gradient,
            bias=state.bias + cfg.step_size * bias_gradient,
            step_count=_saturating_int32_increment(state.step_count),
            nll_ema=nll_ema,
            accuracy_ema=accuracy_ema,
            confidence_ema=confidence_ema,
        )
        # Inf observation makes softmax NaN and logit_error * x = 0*inf = NaN
        # on silent features. Hold the previous finite state.
        diagnostics_required = jnp.asarray(cfg.diagnostic_decay != 0.0, dtype=jnp.bool_)
        source_finite = (
            jnp.all(jnp.isfinite(state.weights))
            & jnp.all(jnp.isfinite(state.bias))
            & (
                (~diagnostics_required)
                | (
                    jnp.isfinite(state.nll_ema)
                    & jnp.isfinite(state.accuracy_ema)
                    & jnp.isfinite(state.confidence_ema)
                )
            )
        )
        inputs_valid = (
            jnp.all(jnp.isfinite(obs))
            & action_valid
        )
        proposed_finite = (
            jnp.all(jnp.isfinite(new_state.weights))
            & jnp.all(jnp.isfinite(new_state.bias))
            & jnp.isfinite(new_state.nll_ema)
            & jnp.isfinite(new_state.accuracy_ema)
            & jnp.isfinite(new_state.confidence_ema)
        )
        update_applied = source_finite & inputs_valid & proposed_finite
        committed = jax.lax.cond(
            update_applied,
            lambda: new_state,
            lambda: state,
        )
        neutral_float = jnp.asarray(0.0, dtype=jnp.float32)
        return BehaviorModelUpdateResult(
            state=committed,
            logits=jnp.where(update_applied, logits, jnp.zeros_like(logits)),
            probabilities=jnp.where(
                update_applied, probabilities, jnp.zeros_like(probabilities)
            ),
            action_probability=jnp.where(update_applied, action_prob, neutral_float),
            log_likelihood=jnp.where(update_applied, log_likelihood, neutral_float),
            loss=jnp.where(update_applied, loss, neutral_float),
            entropy=jnp.where(update_applied, entropy, neutral_float),
            confidence=jnp.where(update_applied, confidence, neutral_float),
            predicted_action=jnp.where(
                update_applied,
                predicted_action,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            correct=jnp.where(update_applied, correct, neutral_float),
            update_applied=update_applied,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def sample_action(
        self,
        state: BehaviorModelState,
        observation: Array,
    ) -> BehaviorModelSampleResult:
        """Sample one action from the learned behavior distribution."""
        key, sample_key = jr.split(state.rng_key)
        probabilities = floor_and_renormalize_probabilities(
            self.predict_probabilities(state, observation),
            min_probability=self._config.min_probability,
        )
        action = jr.categorical(
            sample_key,
            jnp.log(probabilities),
        ).astype(jnp.int32)
        action_prob = selected_action_probabilities(
            probabilities,
            action,
            min_probability=self._config.min_probability,
        )
        return BehaviorModelSampleResult(
            state=state.replace(rng_key=key),  # type: ignore[attr-defined]
            action=action,
            probabilities=probabilities,
            action_probability=action_prob,
            log_likelihood=jnp.log(action_prob),
        )

    def importance_ratio(
        self,
        state: BehaviorModelState,
        observation: Array,
        action: Array,
        target_probabilities: Array,
    ) -> Float[Array, ""]:
        """Compute a clipped target/behavior ratio for one transition."""
        behavior = self.predict_probabilities(state, observation)
        ratio = clipped_importance_ratios(
            target_probabilities,
            behavior,
            action,
            clip=self._config.ratio_clip,
            min_behavior_probability=self._config.min_probability,
        )
        return ratio


def run_behavior_model_from_arrays(
    model: BehaviorModel,
    state: BehaviorModelState,
    observations: Float[Array, "num_steps feature_dim"],
    actions: Int[Array, " num_steps"],
) -> BehaviorModelArrayResult:
    """Run online behavior prediction over arrays with ``jax.lax.scan``.

    Raises:
        TypeError: If ``observations`` is not a trusted array.
        ValueError: If ``observations`` does not have shape
            ``(num_steps, feature_dim)``, is empty, or its leading (step)
            length exceeds the documented scan-length ceiling
            (``_BEHAVIOR_MODEL_SEQUENCE_MAX_STEPS``).
    """

    if len(observations.shape) != 2:
        raise ValueError("observations must have shape (num_steps, feature_dim)")
    _require_behavior_model_sequence_length("observations", observations)
    _, _ = _integer_action_ids(
        actions,
        n_actions=model.config.n_actions,
        expected_shape=(observations.shape[0],),
    )

    def _scan_fn(
        carry: BehaviorModelState,
        inputs: tuple[Array, Array],
    ) -> tuple[
        BehaviorModelState,
        tuple[Array, Array, Array, Array, Array, Array, Array, Array],
    ]:
        obs, action = inputs
        result = model.update(carry, obs, action)
        return result.state, (
            result.probabilities,
            result.action_probability,
            result.log_likelihood,
            result.loss,
            result.entropy,
            result.confidence,
            result.correct,
            result.update_applied,
        )

    (
        final_state,
        (
            probabilities,
            action_probabilities,
            log_likelihoods,
            losses,
            entropies,
            confidences,
            correct,
            updates_applied,
        ),
    ) = jax.lax.scan(_scan_fn, state, (observations, actions))
    return BehaviorModelArrayResult(
        state=final_state,
        probabilities=probabilities,
        action_probabilities=action_probabilities,
        log_likelihoods=log_likelihoods,
        losses=losses,
        entropies=entropies,
        confidences=confidences,
        correct=correct,
        updates_applied=updates_applied,
    )


__all__ = [
    "BehaviorModel",
    "BehaviorModelArrayResult",
    "BehaviorModelConfig",
    "BehaviorModelInputGradient",
    "BehaviorModelResourceBudget",
    "BehaviorModelSampleResult",
    "BehaviorModelState",
    "BehaviorModelUpdateResult",
    "action_log_likelihoods",
    "clipped_importance_ratios",
    "epsilon_greedy_probabilities",
    "floor_and_renormalize_probabilities",
    "run_behavior_model_from_arrays",
    "selected_action_probabilities",
]
