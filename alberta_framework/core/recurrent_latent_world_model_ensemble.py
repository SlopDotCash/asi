# mypy: disable-error-code="call-arg"
"""Bounded recurrent latent world-model ensemble for development research.

The public protocol is deliberately explicit.  :meth:`start` captures the
observation and exact state that own a decision, :meth:`decide` advances each
member's GRU once *without mutating state* and caches the resulting
predict-before-update distribution, and :meth:`update` accepts only an
authoritative transition that matches that cached observation/action pair.

Targets are grounded ``[final/bootstrap observation, reward, continuation]``
values.  On an episode boundary the final/bootstrap observation remains the
learning target while ``next_decision_observation`` becomes the observation
in the returned reset cache.  It is never substituted into the target.  No
task or regime identifier is accepted anywhere in this module.

Every fixed-width ensemble member has a separately initialized trainable GRU,
grounded mean heads, and bounded heteroscedastic variance heads trained with a
Gaussian negative log likelihood.  Bootstrap masks use a distinct persisted
RNG stream.  Epistemic and aleatoric values are raw development diagnostics;
their availability is explicitly false during warmup and this module makes no
claim that either is calibrated or improves control.

The transaction is fail-closed.  Invalid inputs, stale/tampered caches,
exhausted counters, non-finite losses/gradients, or an invalid candidate leave
the entire persistent state (including RNG and recurrent context) unchanged.
All arrays have fixed shapes and each accepted event evaluates one candidate
gradient per member, commits at most one update per member, and records exactly
one recurrent advance before optionally resetting the persistent context.

References:
    Lakshminarayanan, Pritzel, & Blundell (2017). "Simple and Scalable
        Predictive Uncertainty Estimation using Deep Ensembles."
        (Heteroscedastic Gaussian NLL heads; ensemble disagreement as
        epistemic signal.)
    Chua et al. (2018). "Deep Reinforcement Learning in a Handful of Trials
        using Probabilistic Dynamics Models."  (Probabilistic dynamics-model
        ensembles; the aleatoric/epistemic split for model-based RL.)
    Cho et al. (2014). "Learning Phrase Representations using RNN
        Encoder-Decoder for Statistical Machine Translation."  (GRU.)
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
import operator
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)

EVIDENCE_LEVEL = "L0"
SCIENTIFIC_PROMOTION_ALLOWED = False
RECURRENT_LATENT_WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA = (
    "alberta.recurrent_latent_world_model_ensemble.v1"
)

_MAX_STATE_NBYTES = 256 * 1024 * 1024
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)
_LOG_TWO_PI = float(math.log(2.0 * math.pi))
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


def _positive_int(value: object, *, name: str, maximum: int = _INT32_MAX) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")
    canonical = operator.index(cast(SupportsIndex, value))
    if not 1 <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")
    return canonical


def _finite_float32(
    value: object,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    actual_type = type(value)
    if issubclass(actual_type, bool | np.bool_) or not issubclass(actual_type, Real):
        raise ValueError(f"{name} must be a real non-boolean scalar")
    canonical = float(cast(Real, value))
    lower_ok = canonical > 0.0 if positive else canonical >= 0.0 if nonnegative else True
    if not math.isfinite(canonical) or not lower_ok or abs(canonical) > _FLOAT32_MAX:
        qualifier = "positive " if positive else "nonnegative " if nonnegative else ""
        raise ValueError(f"{name} must be a {qualifier}finite float32 scalar")
    narrowed = float(np.float32(canonical))
    if not math.isfinite(narrowed):
        raise ValueError(f"{name} must remain finite in float32")
    if positive and narrowed < _FLOAT32_TINY:
        raise ValueError(f"{name} must remain a positive normal float32")
    return narrowed


def _strict_json_equal(actual: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if expected is None:
        return actual is None
    if isinstance(expected, int):
        return type(actual) is int and actual == expected
    if isinstance(expected, float):
        return type(actual) is float and math.isfinite(actual) and actual == expected
    if isinstance(expected, str):
        return type(actual) is str and actual == expected
    if isinstance(expected, list):
        return (
            type(actual) is list
            and len(actual) == len(expected)
            and all(
                _strict_json_equal(left, right)
                for left, right in zip(actual, expected, strict=True)
            )
        )
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_strict_json_equal(actual[key], expected[key]) for key in expected)
        )
    return False


def _array_contract(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    array = jnp.asarray(value)
    if array.shape != shape or array.dtype != jnp.dtype(dtype):
        raise ValueError(f"{name} must have shape {shape} and dtype {jnp.dtype(dtype)}")
    return array


def _tree_all_finite(tree: Any) -> Array:
    checks: list[Array] = []
    for leaf in jax.tree_util.tree_leaves(tree):
        array = jnp.asarray(leaf)
        if jnp.issubdtype(array.dtype, jnp.inexact):
            checks.append(jnp.all(jnp.isfinite(array)))
    if not checks:
        return jnp.asarray(True, dtype=jnp.bool_)
    return jnp.all(jnp.stack(checks))


def _tree_max_abs(tree: Any) -> Array:
    maxima = [
        jnp.max(jnp.abs(jnp.asarray(leaf)))
        for leaf in jax.tree_util.tree_leaves(tree)
        if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact)
    ]
    if not maxima:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.max(jnp.stack(maxima))


def _tree_equal(left: Any, right: Any) -> Array:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    if cast(Any, left_structure) != right_structure or len(left_leaves) != len(right_leaves):
        return jnp.asarray(False, dtype=jnp.bool_)
    return jnp.all(
        jnp.stack(
            [
                jnp.array_equal(jnp.asarray(left_leaf), jnp.asarray(right_leaf))
                for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True)
            ]
        )
    )


def _tree_l2_norm(tree: Any) -> Array:
    squared = [
        jnp.sum(jnp.square(jnp.asarray(leaf, dtype=jnp.float32)))
        for leaf in jax.tree_util.tree_leaves(tree)
    ]
    return jnp.sqrt(jnp.sum(jnp.stack(squared)))


def _saturating_increment(value: Array, maximum: int) -> Array:
    bound = jnp.asarray(maximum, dtype=jnp.int32)
    return jnp.minimum(jnp.maximum(value, 0), bound - 1) + 1


def _static_signature(tree: Any) -> tuple[Any, tuple[tuple[tuple[int, ...], Any], ...]]:
    leaves, structure = jax.tree_util.tree_flatten(tree)
    return structure, tuple((jnp.asarray(leaf).shape, jnp.asarray(leaf).dtype) for leaf in leaves)


def _validate_static_signature(
    tree: Any,
    signature: tuple[Any, tuple[tuple[tuple[int, ...], Any], ...]],
    *,
    name: str,
) -> None:
    leaves, structure = jax.tree_util.tree_flatten(tree)
    expected_structure, expected_leaves = signature
    if structure != expected_structure or len(leaves) != len(expected_leaves):
        raise ValueError(f"{name} tree structure does not match the configured model")
    for index, (leaf, (shape, dtype)) in enumerate(zip(leaves, expected_leaves, strict=True)):
        array = jnp.asarray(leaf)
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(f"{name} leaf {index} must have shape {shape} and dtype {dtype}")


@dataclasses.dataclass(frozen=True)
class RecurrentLatentWorldModelEnsembleConfig:
    """Fixed architecture, optimizer, warmup, and numeric bounds.

    The ``max_*`` fields are fail-closed validity gates, not tuned
    hyperparameters: an observation, reward, parameter, prediction, or loss
    whose magnitude exceeds its bound causes the whole event to be rejected
    with persistent state untouched.  Their internal consistency is validated
    at construction: ``variance_floor < max_variance``, ``gradient_clip_norm
    <= max_raw_gradient_norm``, and ``learning_rate * gradient_clip_norm <=
    max_parameter_magnitude`` (so a single clipped step cannot itself breach
    the parameter bound).  The default magnitudes are coarse sanity ceilings
    far above any healthy operating range, not derived quantities.

    Per-member raw NLL gradients are validated for finiteness only, not
    against ``max_raw_gradient_norm``: the reachable raw gradient scales as
    ``residual / variance``, so a well-trained low-noise member can legally
    exceed any fixed ceiling on an ordinary in-bounds transition, and because
    a rejection leaves state unchanged, a norm ceiling here creates a
    deadlock rather than a safety gate.  ``gradient_clip_norm`` already
    bounds the committed step magnitude, and ``candidate_parameters_valid`` /
    ``candidate_state_valid`` independently confirm the post-clip candidate is
    finite and in-bounds before it is committed.  ``max_raw_gradient_norm``
    still bounds the representation gradient (a returned diagnostic signal,
    never itself clipped) and constrains ``gradient_clip_norm`` at
    construction.
    """

    observation_dim: int
    n_actions: int
    latent_dim: int = 16
    ensemble_size: int = 3
    learning_rate: float = 1.0e-3
    bootstrap_probability: float = 0.8
    uncertainty_warmup_steps: int = 1
    variance_floor: float = 1.0e-3
    max_variance: float = 100.0
    initialization_scale: float = 0.2
    gradient_clip_norm: float = 10.0
    max_raw_gradient_norm: float = 100_000.0
    max_input_magnitude: float = 1_000.0
    max_parameter_magnitude: float = 10_000.0
    max_prediction_magnitude: float = 10_000.0
    max_loss_magnitude: float = 100_000_000.0
    max_updates: int = _INT32_MAX

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_dim",
            _positive_int(self.observation_dim, name="observation_dim"),
        )
        object.__setattr__(
            self,
            "n_actions",
            _positive_int(self.n_actions, name="n_actions"),
        )
        object.__setattr__(
            self,
            "latent_dim",
            _positive_int(self.latent_dim, name="latent_dim"),
        )
        object.__setattr__(
            self,
            "ensemble_size",
            _positive_int(self.ensemble_size, name="ensemble_size"),
        )
        if self.ensemble_size < 2:
            raise ValueError("ensemble_size must be at least 2 for epistemic disagreement")
        object.__setattr__(
            self,
            "max_updates",
            _positive_int(self.max_updates, name="max_updates"),
        )
        if (
            type(self.uncertainty_warmup_steps) not in _ACTUAL_INT_TYPES
            or not 0
            <= operator.index(cast(SupportsIndex, self.uncertainty_warmup_steps))
            <= self.max_updates
        ):
            raise ValueError("uncertainty_warmup_steps must be in 0..max_updates")
        object.__setattr__(
            self,
            "uncertainty_warmup_steps",
            operator.index(cast(SupportsIndex, self.uncertainty_warmup_steps)),
        )

        positive_fields = (
            "learning_rate",
            "variance_floor",
            "max_variance",
            "initialization_scale",
            "gradient_clip_norm",
            "max_raw_gradient_norm",
            "max_input_magnitude",
            "max_parameter_magnitude",
            "max_prediction_magnitude",
            "max_loss_magnitude",
        )
        for name in positive_fields:
            object.__setattr__(
                self,
                name,
                _finite_float32(getattr(self, name), name=name, positive=True),
            )
        object.__setattr__(
            self,
            "bootstrap_probability",
            _finite_float32(
                self.bootstrap_probability,
                name="bootstrap_probability",
                positive=True,
            ),
        )
        if not 0.0 < self.bootstrap_probability < 1.0:
            raise ValueError("bootstrap_probability must be in (0, 1)")
        if not self.variance_floor < self.max_variance:
            raise ValueError("variance_floor must be smaller than max_variance")
        if self.gradient_clip_norm > self.max_raw_gradient_norm:
            raise ValueError("gradient_clip_norm cannot exceed max_raw_gradient_norm")
        if self.learning_rate * self.gradient_clip_norm > self.max_parameter_magnitude:
            raise ValueError("one clipped update exceeds max_parameter_magnitude")
        initial_variance_bias = self.initial_variance_bias_magnitude
        if initial_variance_bias > self.max_parameter_magnitude:
            raise ValueError(
                f"initial variance bias magnitude {initial_variance_bias} exceeds "
                f"max_parameter_magnitude={self.max_parameter_magnitude}"
            )
        if self.state_nbytes > _MAX_STATE_NBYTES:
            raise ValueError(
                f"configured persistent state needs {self.state_nbytes} bytes; "
                f"the limit is {_MAX_STATE_NBYTES}"
            )

    @property
    def recurrent_input_dim(self) -> int:
        """Observation plus one-hot action width."""
        return self.observation_dim + self.n_actions

    @property
    def target_dim(self) -> int:
        """Grounded next-observation, reward, and continuation width."""
        return self.observation_dim + 2

    @property
    def raw_output_dim(self) -> int:
        """Mean logits and variance logits for every grounded head."""
        return 2 * self.target_dim

    @property
    def trainable_scalars_per_member(self) -> int:
        """Exact trainable float32 scalar count for one member."""
        joined = self.recurrent_input_dim + self.latent_dim
        recurrent = 2 * self.latent_dim * joined + 2 * self.latent_dim
        recurrent += self.latent_dim * joined + self.latent_dim
        heads = 2 * (self.target_dim * self.latent_dim + self.target_dim)
        return recurrent + heads

    @property
    def initial_variance_logit(self) -> float:
        """Variance-head bias that starts every member at sqrt(floor * max) variance."""
        desired_variance = math.sqrt(self.variance_floor * self.max_variance)
        variance_ratio = (desired_variance - self.variance_floor) / (
            self.max_variance - self.variance_floor
        )
        return math.log(variance_ratio / (1.0 - variance_ratio))

    @property
    def initial_variance_bias_magnitude(self) -> float:
        """Magnitude of the stored float32 variance bias, as ``_parameters_valid`` sees it."""
        return float(np.float32(abs(self.initial_variance_logit)))

    @property
    def state_nbytes(self) -> int:
        """Exact persistent logical array bytes for this configuration."""
        float32_scalars = self.ensemble_size * (self.trainable_scalars_per_member + self.latent_dim)
        int32_scalars = self.ensemble_size + 3
        uint32_scalars = 2
        bool_scalars = self.ensemble_size
        return 4 * (float32_scalars + int32_scalars + uint32_scalars) + bool_scalars

    def to_config(self) -> dict[str, object]:
        """Return the exact JSON-compatible static construction."""
        payload: dict[str, object] = {"type": type(self).__name__}
        payload.update(dataclasses.asdict(self))
        return payload

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> RecurrentLatentWorldModelEnsembleConfig:
        """Strictly reconstruct :meth:`to_config` output."""
        expected = {field.name for field in dataclasses.fields(cls)} | {"type"}
        if set(config) != expected:
            raise ValueError("config fields do not match the serialized schema")
        if config.get("type") != cls.__name__:
            raise ValueError("unexpected recurrent latent ensemble config type")
        kwargs = {field.name: config[field.name] for field in dataclasses.fields(cls)}
        restored = cls(**cast(dict[str, Any], kwargs))
        if not _strict_json_equal(dict(config), restored.to_config()):
            raise ValueError("config is not the canonical serialized construction")
        return restored


@chex.dataclass(frozen=True)
class RecurrentLatentMemberParameters:
    """Trainable GRU and heteroscedastic grounded-head parameters."""

    gate_kernel: Float[Array, "two_latent joined_input"]
    gate_bias: Float[Array, " two_latent"]
    candidate_kernel: Float[Array, "latent_dim joined_input"]
    candidate_bias: Float[Array, " latent_dim"]
    mean_kernel: Float[Array, "target_dim latent_dim"]
    mean_bias: Float[Array, " target_dim"]
    variance_kernel: Float[Array, "target_dim latent_dim"]
    variance_bias: Float[Array, " target_dim"]


@chex.dataclass(frozen=True)
class RecurrentLatentWorldModelEnsembleState:
    """Complete fixed-budget parameters, recurrent context, RNG, and counters."""

    member_parameters: tuple[RecurrentLatentMemberParameters, ...]
    member_hidden_states: Float[Array, "ensemble_size latent_dim"]
    bootstrap_key: Array
    last_bootstrap_mask: Bool[Array, " ensemble_size"]
    member_update_counts: Int[Array, " ensemble_size"]
    event_count: Int[Array, ""]
    recurrent_advance_count: Int[Array, ""]
    boundary_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class RecurrentLatentPredictionAvailability:
    """Explicit usability flags; uncertainty remains unavailable in warmup."""

    prediction: Bool[Array, ""]
    epistemic: Bool[Array, ""]
    aleatoric: Bool[Array, ""]


@chex.dataclass(frozen=True)
class RecurrentLatentWorldModelPrediction:
    """One pre-update ensemble distribution and recurrent candidates."""

    member_raw_outputs: Float[Array, "ensemble_size raw_output_dim"]
    mean_raw_output: Float[Array, " raw_output_dim"]
    member_mean_predictions: Float[Array, "ensemble_size target_dim"]
    mean_prediction: Float[Array, " target_dim"]
    member_next_observations: Float[Array, "ensemble_size observation_dim"]
    mean_next_observation: Float[Array, " observation_dim"]
    member_rewards: Float[Array, " ensemble_size"]
    mean_reward: Float[Array, ""]
    member_continuations: Float[Array, " ensemble_size"]
    mean_continuation: Float[Array, ""]
    member_aleatoric_variances: Float[Array, "ensemble_size target_dim"]
    mean_aleatoric_variance: Float[Array, " target_dim"]
    aleatoric_uncertainty: Float[Array, ""]
    per_head_epistemic_variance: Float[Array, " target_dim"]
    epistemic_disagreement: Float[Array, ""]
    member_next_hidden_states: Float[Array, "ensemble_size latent_dim"]
    warmup_ready: Bool[Array, ""]
    availability: RecurrentLatentPredictionAvailability
    valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class RecurrentLatentStartCache:
    """Exact state/observation ownership record before action selection."""

    owner_event_count: Int[Array, ""]
    owner_hidden_states: Float[Array, "ensemble_size latent_dim"]
    observation: Float[Array, " observation_dim"]
    valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class RecurrentLatentDecisionCache:
    """Exact predict-before-update decision record consumed by ``update``."""

    owner_event_count: Int[Array, ""]
    owner_hidden_states: Float[Array, "ensemble_size latent_dim"]
    observation: Float[Array, " observation_dim"]
    action: Int[Array, ""]
    prediction: RecurrentLatentWorldModelPrediction
    valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class RecurrentLatentTransitionRecord:
    """Authoritative outcome for one exact cached observation/action decision.

    ``bootstrap_observation`` is the final observation used in every model
    target.  Off a boundary it must equal ``next_decision_observation``.  On a
    terminated or truncated transition, ``next_decision_observation`` is the
    reset observation and is used only by the returned next start cache.
    """

    observation: Float[Array, " observation_dim"]
    action: Int[Array, ""]
    reward: Float[Array, ""]
    discount: Float[Array, ""]
    terminated: Bool[Array, ""]
    truncated: Bool[Array, ""]
    bootstrap_observation: Float[Array, " observation_dim"]
    next_decision_observation: Float[Array, " observation_dim"]


@chex.dataclass(frozen=True)
class RecurrentLatentWorldModelDiagnostics:
    """Fail-closed checks and explicit one-advance/boundary verdicts."""

    state_valid: Bool[Array, ""]
    cache_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    ownership_valid: Bool[Array, ""]
    boundary_semantics_valid: Bool[Array, ""]
    capacity_available: Bool[Array, ""]
    cached_prediction_exact: Bool[Array, ""]
    predictions_valid: Bool[Array, ""]
    losses_valid: Bool[Array, ""]
    representation_gradient_valid: Bool[Array, ""]
    member_gradients_valid: Bool[Array, " ensemble_size"]
    candidate_parameters_valid: Bool[Array, " ensemble_size"]
    candidate_state_valid: Bool[Array, ""]
    recurrent_advanced_once: Bool[Array, ""]
    recurrent_reset: Bool[Array, ""]
    applied: Bool[Array, ""]
    rejected: Bool[Array, ""]


@chex.dataclass(frozen=True)
class RecurrentLatentWorldModelUpdateResult:
    """One bounded predict-before-update recurrent ensemble transaction."""

    state: RecurrentLatentWorldModelEnsembleState
    prediction: RecurrentLatentWorldModelPrediction
    targets: Float[Array, " target_dim"]
    member_negative_log_likelihoods: Float[Array, " ensemble_size"]
    mean_negative_log_likelihood: Float[Array, ""]
    representation_objective: Float[Array, ""]
    representation_gradient: Float[Array, " observation_dim"]
    representation_gradient_available: Bool[Array, ""]
    bootstrap_mask: Bool[Array, " ensemble_size"]
    member_updates_applied: Bool[Array, " ensemble_size"]
    next_start_cache: RecurrentLatentStartCache
    diagnostics: RecurrentLatentWorldModelDiagnostics


@dataclasses.dataclass(frozen=True)
class _LogicalAccounting:
    float32_scalars: int = 0
    int32_scalars: int = 0
    uint32_scalars: int = 0
    bool_scalars: int = 0

    @property
    def logical_scalars(self) -> int:
        return self.float32_scalars + self.int32_scalars + self.uint32_scalars + self.bool_scalars

    @property
    def logical_bytes(self) -> int:
        return 4 * (self.float32_scalars + self.int32_scalars + self.uint32_scalars) + (
            self.bool_scalars
        )


@dataclasses.dataclass(frozen=True)
class RecurrentLatentWorldModelResourceBudget:
    """Exact fixed-PyTree logical storage/output and per-event work bounds."""

    ensemble_size: int
    observation_dim: int
    latent_dim: int
    target_dim: int
    trainable_scalars_per_member: int
    total_trainable_scalars: int
    persistent_float32_scalars: int
    persistent_int32_scalars: int
    persistent_uint32_scalars: int
    persistent_bool_scalars: int
    persistent_state_scalars: int
    persistent_state_bytes: int
    bootstrap_prng_keys: int
    bootstrap_prng_uint32_scalars: int
    start_cache_logical_scalars: int
    start_cache_logical_bytes: int
    decision_cache_logical_scalars: int
    decision_cache_logical_bytes: int
    update_result_logical_scalars: int
    update_result_logical_bytes: int
    member_gradient_candidates_per_event: int
    max_member_parameter_updates_per_event: int
    recurrent_advances_per_accepted_event: int
    max_event_count: int
    max_member_update_count: int
    replay_capacity: int

    def to_config(self) -> dict[str, int]:
        """Return exact JSON-compatible accounting."""
        return dataclasses.asdict(self)


def _logical_accounting(tree: Any) -> _LogicalAccounting:
    counts = {"float32": 0, "int32": 0, "uint32": 0, "bool": 0}
    for leaf in jax.tree_util.tree_leaves(tree):
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jax.dtypes.prng_key):
            array = jr.key_data(leaf)
        else:
            array = jnp.asarray(leaf)
        size = int(array.size)
        if array.dtype == jnp.float32:
            counts["float32"] += size
        elif array.dtype == jnp.int32:
            counts["int32"] += size
        elif array.dtype == jnp.uint32:
            counts["uint32"] += size
        elif array.dtype == jnp.bool_:
            counts["bool"] += size
        else:
            raise TypeError(f"unsupported logical resource dtype {array.dtype}")
    return _LogicalAccounting(
        float32_scalars=counts["float32"],
        int32_scalars=counts["int32"],
        uint32_scalars=counts["uint32"],
        bool_scalars=counts["bool"],
    )


class RecurrentLatentWorldModelEnsemble:
    """Fixed-width online GRU ensemble with heteroscedastic grounded heads."""

    def __init__(self, config: RecurrentLatentWorldModelEnsembleConfig):
        self._config = config
        template = self._initial_state(jr.key(0))
        self._member_signature = _static_signature(template.member_parameters[0])
        self._state_signature = _static_signature(template)
        self._start_signature = _static_signature(self._zero_start_cache())
        self._decision_signature = _static_signature(self._zero_decision_cache())

    @property
    def config(self) -> RecurrentLatentWorldModelEnsembleConfig:
        """Return immutable construction and bound configuration."""
        return self._config

    def to_config(self) -> dict[str, object]:
        """Serialize the exact model construction."""
        return {"type": type(self).__name__, "config": self._config.to_config()}

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> RecurrentLatentWorldModelEnsemble:
        """Strictly reconstruct :meth:`to_config` output."""
        if set(config) != {"type", "config"}:
            raise ValueError("model config fields do not match the serialized schema")
        if config.get("type") != cls.__name__:
            raise ValueError("unexpected recurrent latent ensemble model type")
        nested = config.get("config")
        if not isinstance(nested, Mapping):
            raise ValueError("model config.config must be a mapping")
        model = cls(RecurrentLatentWorldModelEnsembleConfig.from_config(nested))
        if not _strict_json_equal(dict(config), model.to_config()):
            raise ValueError("model config is not canonical")
        return model

    def _init_member(self, key: Array) -> RecurrentLatentMemberParameters:
        cfg = self._config
        keys = jr.split(key, 4)
        joined = cfg.recurrent_input_dim + cfg.latent_dim
        scale = jnp.asarray(cfg.initialization_scale, dtype=jnp.float32)

        def kernel(current_key: Array, shape: tuple[int, int]) -> Array:
            fan_in = jnp.asarray(shape[1], dtype=jnp.float32)
            return jr.normal(current_key, shape, dtype=jnp.float32) * scale / jnp.sqrt(fan_in)

        variance_logit = cfg.initial_variance_logit
        return RecurrentLatentMemberParameters(
            gate_kernel=kernel(keys[0], (2 * cfg.latent_dim, joined)),
            gate_bias=jnp.zeros((2 * cfg.latent_dim,), dtype=jnp.float32),
            candidate_kernel=kernel(keys[1], (cfg.latent_dim, joined)),
            candidate_bias=jnp.zeros((cfg.latent_dim,), dtype=jnp.float32),
            mean_kernel=kernel(keys[2], (cfg.target_dim, cfg.latent_dim)),
            mean_bias=jnp.zeros((cfg.target_dim,), dtype=jnp.float32),
            variance_kernel=kernel(keys[3], (cfg.target_dim, cfg.latent_dim)),
            variance_bias=jnp.full(
                (cfg.target_dim,),
                variance_logit,
                dtype=jnp.float32,
            ),
        )

    def init(self, key: Array) -> RecurrentLatentWorldModelEnsembleState:
        """Initialize distinct member parameters and an isolated bootstrap key.

        This is a deliberately host-side entry point: it evaluates the drawn
        parameters against ``max_parameter_magnitude`` eagerly and raises, so
        it must not be wrapped in ``jax.jit``; compile the ``start`` /
        ``decide`` / ``update`` transitions instead.

        Raises:
            ValueError: If ``key`` is not one JAX PRNG key, or the drawn
                initial parameters would already fail the configured
                ``max_parameter_magnitude`` bound (an initialized state that
                the module itself rejects would make every later event a
                silent no-op).
        """
        state = self._initial_state(key)
        if not bool(self._state_valid(state)):
            raise ValueError(
                "initialized parameters exceed max_parameter_magnitude="
                f"{self._config.max_parameter_magnitude}; lower initialization_scale "
                "or raise the bound"
            )
        return state

    def _initial_state(self, key: Array) -> RecurrentLatentWorldModelEnsembleState:
        key_data = jr.key_data(key)
        if key_data.shape != (2,) or key_data.dtype != jnp.uint32:
            raise ValueError("key must be one JAX PRNG key")
        keys = jr.split(key, self._config.ensemble_size + 1)
        return RecurrentLatentWorldModelEnsembleState(
            member_parameters=tuple(
                self._init_member(keys[index]) for index in range(self._config.ensemble_size)
            ),
            member_hidden_states=jnp.zeros(
                (self._config.ensemble_size, self._config.latent_dim),
                dtype=jnp.float32,
            ),
            bootstrap_key=keys[-1],
            last_bootstrap_mask=jnp.zeros((self._config.ensemble_size,), dtype=jnp.bool_),
            member_update_counts=jnp.zeros((self._config.ensemble_size,), dtype=jnp.int32),
            event_count=jnp.asarray(0, dtype=jnp.int32),
            recurrent_advance_count=jnp.asarray(0, dtype=jnp.int32),
            boundary_count=jnp.asarray(0, dtype=jnp.int32),
        )

    def _validate_state_static(self, state: RecurrentLatentWorldModelEnsembleState) -> None:
        if not isinstance(state, RecurrentLatentWorldModelEnsembleState):
            raise TypeError("state must be a RecurrentLatentWorldModelEnsembleState")
        if len(state.member_parameters) != self._config.ensemble_size:
            raise ValueError("state member count does not match ensemble_size")
        _validate_static_signature(state, self._state_signature, name="state")
        for index, parameters in enumerate(state.member_parameters):
            _validate_static_signature(
                parameters,
                self._member_signature,
                name=f"state.member_parameters[{index}]",
            )
        key_data = jr.key_data(state.bootstrap_key)
        if key_data.shape != (2,) or key_data.dtype != jnp.uint32:
            raise ValueError("state.bootstrap_key must be one JAX PRNG key")

    def _validate_start_static(self, cache: RecurrentLatentStartCache) -> None:
        if not isinstance(cache, RecurrentLatentStartCache):
            raise TypeError("start_cache must be a RecurrentLatentStartCache")
        _validate_static_signature(cache, self._start_signature, name="start_cache")

    def _validate_decision_static(self, cache: RecurrentLatentDecisionCache) -> None:
        if not isinstance(cache, RecurrentLatentDecisionCache):
            raise TypeError("decision_cache must be a RecurrentLatentDecisionCache")
        _validate_static_signature(cache, self._decision_signature, name="decision_cache")

    def _parameters_valid(self, parameters: RecurrentLatentMemberParameters) -> Array:
        return _tree_all_finite(parameters) & (
            _tree_max_abs(parameters) <= self._config.max_parameter_magnitude
        )

    def _state_valid(self, state: RecurrentLatentWorldModelEnsembleState) -> Array:
        parameters_valid = jnp.all(
            jnp.stack([self._parameters_valid(item) for item in state.member_parameters])
        )
        return (
            parameters_valid
            & jnp.all(jnp.isfinite(state.member_hidden_states))
            & jnp.all(jnp.abs(state.member_hidden_states) <= 1.0 + 1.0e-6)
            & (state.event_count >= 0)
            & (state.event_count <= self._config.max_updates)
            & (state.recurrent_advance_count == state.event_count)
            & (state.boundary_count >= 0)
            & (state.boundary_count <= state.event_count)
            & jnp.all(state.member_update_counts >= 0)
            & jnp.all(state.member_update_counts <= state.event_count)
            & jnp.all(state.member_update_counts <= self._config.max_updates)
        )

    def state_valid(self, state: RecurrentLatentWorldModelEnsembleState) -> Array:
        """Return dynamic validity after enforcing the fixed static contract."""
        self._validate_state_static(state)
        return self._state_valid(state)

    def _zero_availability(self) -> RecurrentLatentPredictionAvailability:
        false = jnp.asarray(False, dtype=jnp.bool_)
        return RecurrentLatentPredictionAvailability(
            prediction=false,
            epistemic=false,
            aleatoric=false,
        )

    def _zero_prediction(self) -> RecurrentLatentWorldModelPrediction:
        cfg = self._config
        return RecurrentLatentWorldModelPrediction(
            member_raw_outputs=jnp.zeros(
                (cfg.ensemble_size, cfg.raw_output_dim), dtype=jnp.float32
            ),
            mean_raw_output=jnp.zeros((cfg.raw_output_dim,), dtype=jnp.float32),
            member_mean_predictions=jnp.zeros(
                (cfg.ensemble_size, cfg.target_dim), dtype=jnp.float32
            ),
            mean_prediction=jnp.zeros((cfg.target_dim,), dtype=jnp.float32),
            member_next_observations=jnp.zeros(
                (cfg.ensemble_size, cfg.observation_dim), dtype=jnp.float32
            ),
            mean_next_observation=jnp.zeros((cfg.observation_dim,), dtype=jnp.float32),
            member_rewards=jnp.zeros((cfg.ensemble_size,), dtype=jnp.float32),
            mean_reward=jnp.asarray(0.0, dtype=jnp.float32),
            member_continuations=jnp.zeros((cfg.ensemble_size,), dtype=jnp.float32),
            mean_continuation=jnp.asarray(0.0, dtype=jnp.float32),
            member_aleatoric_variances=jnp.zeros(
                (cfg.ensemble_size, cfg.target_dim), dtype=jnp.float32
            ),
            mean_aleatoric_variance=jnp.zeros((cfg.target_dim,), dtype=jnp.float32),
            aleatoric_uncertainty=jnp.asarray(0.0, dtype=jnp.float32),
            per_head_epistemic_variance=jnp.zeros((cfg.target_dim,), dtype=jnp.float32),
            epistemic_disagreement=jnp.asarray(0.0, dtype=jnp.float32),
            member_next_hidden_states=jnp.zeros(
                (cfg.ensemble_size, cfg.latent_dim), dtype=jnp.float32
            ),
            warmup_ready=jnp.asarray(False, dtype=jnp.bool_),
            availability=self._zero_availability(),
            valid=jnp.asarray(False, dtype=jnp.bool_),
        )

    def _zero_start_cache(self) -> RecurrentLatentStartCache:
        return RecurrentLatentStartCache(
            owner_event_count=jnp.asarray(0, dtype=jnp.int32),
            owner_hidden_states=jnp.zeros(
                (self._config.ensemble_size, self._config.latent_dim), dtype=jnp.float32
            ),
            observation=jnp.zeros((self._config.observation_dim,), dtype=jnp.float32),
            valid=jnp.asarray(False, dtype=jnp.bool_),
        )

    def _zero_decision_cache(self) -> RecurrentLatentDecisionCache:
        return RecurrentLatentDecisionCache(
            owner_event_count=jnp.asarray(0, dtype=jnp.int32),
            owner_hidden_states=jnp.zeros(
                (self._config.ensemble_size, self._config.latent_dim), dtype=jnp.float32
            ),
            observation=jnp.zeros((self._config.observation_dim,), dtype=jnp.float32),
            action=jnp.asarray(0, dtype=jnp.int32),
            prediction=self._zero_prediction(),
            valid=jnp.asarray(False, dtype=jnp.bool_),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def start(
        self,
        state: RecurrentLatentWorldModelEnsembleState,
        observation: Array,
    ) -> RecurrentLatentStartCache:
        """Capture one exact observation/state owner without advancing the GRU."""
        self._validate_state_static(state)
        obs = _array_contract(
            observation,
            name="observation",
            shape=(self._config.observation_dim,),
            dtype=jnp.float32,
        )
        valid = (
            self._state_valid(state)
            & jnp.all(jnp.isfinite(obs))
            & jnp.all(jnp.abs(obs) <= self._config.max_input_magnitude)
        )

        def accepted(_: None) -> RecurrentLatentStartCache:
            return RecurrentLatentStartCache(
                owner_event_count=state.event_count,
                owner_hidden_states=state.member_hidden_states,
                observation=obs,
                valid=jnp.asarray(True, dtype=jnp.bool_),
            )

        return cast(
            RecurrentLatentStartCache,
            jax.lax.cond(valid, accepted, lambda _: self._zero_start_cache(), operand=None),
        )

    def _member_predict(
        self,
        parameters: RecurrentLatentMemberParameters,
        hidden: Array,
        observation: Array,
        action: Array,
    ) -> tuple[Array, Array, Array, Array]:
        cfg = self._config
        action_one_hot = jax.nn.one_hot(action, cfg.n_actions, dtype=jnp.float32)
        recurrent_input = jnp.concatenate((observation, action_one_hot), axis=0)
        gate_input = jnp.concatenate((recurrent_input, hidden), axis=0)
        reset_and_update = jax.nn.sigmoid(
            parameters.gate_kernel @ gate_input + parameters.gate_bias
        )
        reset_gate, update_gate = jnp.split(reset_and_update, 2, axis=0)
        candidate_input = jnp.concatenate((recurrent_input, reset_gate * hidden), axis=0)
        candidate = jnp.tanh(
            parameters.candidate_kernel @ candidate_input + parameters.candidate_bias
        )
        next_hidden = (1.0 - update_gate) * hidden + update_gate * candidate

        mean_logits = parameters.mean_kernel @ next_hidden + parameters.mean_bias
        variance_logits = parameters.variance_kernel @ next_hidden + parameters.variance_bias
        grounded = self._config.max_prediction_magnitude * jnp.tanh(
            mean_logits[:-1] / self._config.max_prediction_magnitude
        )
        continuation = jax.nn.sigmoid(mean_logits[-1])
        means = jnp.concatenate((grounded, continuation[None]), axis=0)
        variances = self._config.variance_floor + (
            self._config.max_variance - self._config.variance_floor
        ) * jax.nn.sigmoid(variance_logits)
        raw = jnp.concatenate((mean_logits, variance_logits), axis=0)
        return next_hidden, raw, means, variances

    def _predict_unchecked(
        self,
        state: RecurrentLatentWorldModelEnsembleState,
        observation: Array,
        action: Array,
    ) -> RecurrentLatentWorldModelPrediction:
        member_outputs = [
            self._member_predict(
                state.member_parameters[index],
                state.member_hidden_states[index],
                observation,
                action,
            )
            for index in range(self._config.ensemble_size)
        ]
        next_hidden = jnp.stack([item[0] for item in member_outputs], axis=0)
        raw = jnp.stack([item[1] for item in member_outputs], axis=0)
        means = jnp.stack([item[2] for item in member_outputs], axis=0)
        variances = jnp.stack([item[3] for item in member_outputs], axis=0)
        epistemic = jnp.var(means, axis=0)
        mean_variance = jnp.mean(variances, axis=0)
        warmup = state.event_count >= self._config.uncertainty_warmup_steps
        finite = (
            jnp.all(jnp.isfinite(next_hidden))
            & jnp.all(jnp.abs(next_hidden) <= 1.0 + 1.0e-6)
            & jnp.all(jnp.isfinite(raw))
            & jnp.all(jnp.abs(raw) <= self._config.max_prediction_magnitude)
            & jnp.all(jnp.isfinite(means))
            & jnp.all(jnp.abs(means) <= self._config.max_prediction_magnitude)
            & jnp.all(jnp.isfinite(variances))
            & jnp.all(variances >= self._config.variance_floor)
            & jnp.all(variances <= self._config.max_variance)
            & jnp.all(jnp.isfinite(epistemic))
            & jnp.all(jnp.isfinite(mean_variance))
        )
        uncertainty_available = finite & warmup
        return RecurrentLatentWorldModelPrediction(
            member_raw_outputs=raw,
            mean_raw_output=jnp.mean(raw, axis=0),
            member_mean_predictions=means,
            mean_prediction=jnp.mean(means, axis=0),
            member_next_observations=means[:, : self._config.observation_dim],
            mean_next_observation=jnp.mean(means[:, : self._config.observation_dim], axis=0),
            member_rewards=means[:, -2],
            mean_reward=jnp.mean(means[:, -2]),
            member_continuations=means[:, -1],
            mean_continuation=jnp.mean(means[:, -1]),
            member_aleatoric_variances=variances,
            mean_aleatoric_variance=mean_variance,
            aleatoric_uncertainty=jnp.mean(mean_variance),
            per_head_epistemic_variance=epistemic,
            epistemic_disagreement=jnp.mean(epistemic),
            member_next_hidden_states=next_hidden,
            warmup_ready=warmup,
            availability=RecurrentLatentPredictionAvailability(
                prediction=finite,
                epistemic=uncertainty_available,
                aleatoric=uncertainty_available,
            ),
            valid=finite,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def decide(
        self,
        state: RecurrentLatentWorldModelEnsembleState,
        start_cache: RecurrentLatentStartCache,
        action: Array,
    ) -> RecurrentLatentDecisionCache:
        """Form and cache a read-only recurrent predict-before-update decision."""
        self._validate_state_static(state)
        self._validate_start_static(start_cache)
        act = _array_contract(action, name="action", shape=(), dtype=jnp.int32)
        ownership = (
            start_cache.valid
            & (start_cache.owner_event_count == state.event_count)
            & jnp.array_equal(start_cache.owner_hidden_states, state.member_hidden_states)
        )
        input_valid = (
            (act >= 0)
            & (act < self._config.n_actions)
            & jnp.all(jnp.isfinite(start_cache.observation))
            & jnp.all(jnp.abs(start_cache.observation) <= self._config.max_input_magnitude)
        )
        can_predict = self._state_valid(state) & ownership & input_valid

        def accepted(_: None) -> RecurrentLatentDecisionCache:
            prediction = self._predict_unchecked(state, start_cache.observation, act)
            return RecurrentLatentDecisionCache(
                owner_event_count=state.event_count,
                owner_hidden_states=state.member_hidden_states,
                observation=start_cache.observation,
                action=act,
                prediction=prediction,
                valid=prediction.valid,
            )

        result = cast(
            RecurrentLatentDecisionCache,
            jax.lax.cond(
                can_predict,
                accepted,
                lambda _: self._zero_decision_cache(),
                operand=None,
            ),
        )
        return cast(
            RecurrentLatentDecisionCache,
            jax.lax.cond(
                result.valid,
                lambda: result,
                self._zero_decision_cache,
            ),
        )

    def _member_nll(
        self,
        parameters: RecurrentLatentMemberParameters,
        hidden: Array,
        observation: Array,
        action: Array,
        stopped_targets: Array,
    ) -> Array:
        _, _, means, variances = self._member_predict(
            parameters,
            hidden,
            observation,
            action,
        )
        residuals = means - stopped_targets
        return 0.5 * jnp.mean(
            jnp.asarray(_LOG_TWO_PI, dtype=jnp.float32)
            + jnp.log(variances)
            + jnp.square(residuals) / variances
        )

    def _zero_diagnostics(
        self,
        *,
        state_valid: Array,
        cache_valid: Array,
        input_valid: Array,
        ownership_valid: Array,
        boundary_semantics_valid: Array,
        capacity_available: Array,
    ) -> RecurrentLatentWorldModelDiagnostics:
        false = jnp.asarray(False, dtype=jnp.bool_)
        return RecurrentLatentWorldModelDiagnostics(
            state_valid=state_valid,
            cache_valid=cache_valid,
            input_valid=input_valid,
            ownership_valid=ownership_valid,
            boundary_semantics_valid=boundary_semantics_valid,
            capacity_available=capacity_available,
            cached_prediction_exact=false,
            predictions_valid=false,
            losses_valid=false,
            representation_gradient_valid=false,
            member_gradients_valid=jnp.zeros((self._config.ensemble_size,), dtype=jnp.bool_),
            candidate_parameters_valid=jnp.zeros((self._config.ensemble_size,), dtype=jnp.bool_),
            candidate_state_valid=false,
            recurrent_advanced_once=false,
            recurrent_reset=false,
            applied=false,
            rejected=jnp.asarray(True, dtype=jnp.bool_),
        )

    def _rejected_result(
        self,
        state: RecurrentLatentWorldModelEnsembleState,
        diagnostics: RecurrentLatentWorldModelDiagnostics,
        next_decision_observation: Array | None = None,
        transition_boundary: Array | None = None,
    ) -> RecurrentLatentWorldModelUpdateResult:
        # A rejection never mutates `state`, but the *cache* previously
        # defaulted to the invalid zero cache unconditionally.  Since `decide`
        # and `update` both require a valid, owned start cache, that made
        # every rejection -- regardless of cause -- a permanent deadlock: no
        # later event, however legal, could ever be accepted again.
        #
        # Re-owning the unchanged state is safe only after an authoritative,
        # in-domain, off-boundary transition reached a *late* numerical or
        # candidate-state rejection.  In particular, `cache_valid` says only
        # that the caller supplied a cache whose own flags are true; ownership
        # and exact cached-prediction checks are what bind it to this state,
        # observation, and action.  Minting a valid cache before those checks
        # would launder an arbitrary `next_decision_observation` through a
        # stale or tampered decision.  A rejected boundary also cannot recover
        # this way because the unchanged state has not committed the required
        # recurrent reset.
        if next_decision_observation is None or transition_boundary is None:
            recoverable_cache = self._zero_start_cache()
        else:
            recoverable = (
                diagnostics.state_valid
                & diagnostics.cache_valid
                & diagnostics.input_valid
                & diagnostics.ownership_valid
                & diagnostics.boundary_semantics_valid
                & diagnostics.capacity_available
                & diagnostics.cached_prediction_exact
                & diagnostics.predictions_valid
                & ~transition_boundary
            )
            recoverable_cache = cast(
                RecurrentLatentStartCache,
                jax.lax.cond(
                    recoverable,
                    lambda: RecurrentLatentStartCache(
                        owner_event_count=state.event_count,
                        owner_hidden_states=state.member_hidden_states,
                        observation=next_decision_observation,
                        valid=jnp.asarray(True, dtype=jnp.bool_),
                    ),
                    self._zero_start_cache,
                ),
            )
        return RecurrentLatentWorldModelUpdateResult(
            state=state,
            prediction=self._zero_prediction(),
            targets=jnp.zeros((self._config.target_dim,), dtype=jnp.float32),
            member_negative_log_likelihoods=jnp.zeros(
                (self._config.ensemble_size,), dtype=jnp.float32
            ),
            mean_negative_log_likelihood=jnp.asarray(0.0, dtype=jnp.float32),
            representation_objective=jnp.asarray(0.0, dtype=jnp.float32),
            representation_gradient=jnp.zeros((self._config.observation_dim,), dtype=jnp.float32),
            representation_gradient_available=jnp.asarray(False, dtype=jnp.bool_),
            bootstrap_mask=jnp.zeros((self._config.ensemble_size,), dtype=jnp.bool_),
            member_updates_applied=jnp.zeros((self._config.ensemble_size,), dtype=jnp.bool_),
            next_start_cache=recoverable_cache,
            diagnostics=diagnostics,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: RecurrentLatentWorldModelEnsembleState,
        decision_cache: RecurrentLatentDecisionCache,
        transition: RecurrentLatentTransitionRecord,
    ) -> RecurrentLatentWorldModelUpdateResult:
        """Atomically learn one authoritative cached transition.

        The target is stopped in both parameter and representation
        derivatives.  The cached GRU advance is committed exactly once on an
        accepted event, then reset to zero only at a declared boundary.
        """
        self._validate_state_static(state)
        self._validate_decision_static(decision_cache)
        if not isinstance(transition, RecurrentLatentTransitionRecord):
            raise TypeError("transition must be a RecurrentLatentTransitionRecord")
        cfg = self._config
        observation = _array_contract(
            transition.observation,
            name="transition.observation",
            shape=(cfg.observation_dim,),
            dtype=jnp.float32,
        )
        action = _array_contract(
            transition.action,
            name="transition.action",
            shape=(),
            dtype=jnp.int32,
        )
        reward = _array_contract(
            transition.reward,
            name="transition.reward",
            shape=(),
            dtype=jnp.float32,
        )
        discount = _array_contract(
            transition.discount,
            name="transition.discount",
            shape=(),
            dtype=jnp.float32,
        )
        terminated = _array_contract(
            transition.terminated,
            name="transition.terminated",
            shape=(),
            dtype=jnp.bool_,
        )
        truncated = _array_contract(
            transition.truncated,
            name="transition.truncated",
            shape=(),
            dtype=jnp.bool_,
        )
        bootstrap_observation = _array_contract(
            transition.bootstrap_observation,
            name="transition.bootstrap_observation",
            shape=(cfg.observation_dim,),
            dtype=jnp.float32,
        )
        next_decision_observation = _array_contract(
            transition.next_decision_observation,
            name="transition.next_decision_observation",
            shape=(cfg.observation_dim,),
            dtype=jnp.float32,
        )
        state_valid = self._state_valid(state)
        cache_valid = decision_cache.valid & decision_cache.prediction.valid
        boundary = terminated | truncated
        ownership_valid = (
            (decision_cache.owner_event_count == state.event_count)
            & jnp.array_equal(decision_cache.owner_hidden_states, state.member_hidden_states)
            & jnp.array_equal(observation, decision_cache.observation)
            & (action == decision_cache.action)
        )
        input_valid = (
            jnp.all(jnp.isfinite(observation))
            & jnp.all(jnp.abs(observation) <= cfg.max_input_magnitude)
            & (action >= 0)
            & (action < cfg.n_actions)
            & jnp.isfinite(reward)
            & (jnp.abs(reward) <= cfg.max_input_magnitude)
            & jnp.isfinite(discount)
            & (discount >= 0.0)
            & (discount <= 1.0)
            & jnp.all(jnp.isfinite(bootstrap_observation))
            & jnp.all(jnp.abs(bootstrap_observation) <= cfg.max_input_magnitude)
            & jnp.all(jnp.isfinite(next_decision_observation))
            & jnp.all(jnp.abs(next_decision_observation) <= cfg.max_input_magnitude)
        )
        boundary_semantics_valid = (
            ((discount == 0.0) == terminated)
            & ((~(truncated & ~terminated)) | (discount > 0.0))
            & (boundary | jnp.array_equal(bootstrap_observation, next_decision_observation))
        )
        capacity_available = (
            (state.event_count < cfg.max_updates)
            & (state.recurrent_advance_count < cfg.max_updates)
            & jnp.all(state.member_update_counts < cfg.max_updates)
        )
        can_attempt = (
            state_valid
            & cache_valid
            & input_valid
            & ownership_valid
            & boundary_semantics_valid
            & capacity_available
        )
        rejected_diagnostics = self._zero_diagnostics(
            state_valid=state_valid,
            cache_valid=cache_valid,
            input_valid=input_valid,
            ownership_valid=ownership_valid,
            boundary_semantics_valid=boundary_semantics_valid,
            capacity_available=capacity_available,
        )

        def accepted_branch(_: None) -> RecurrentLatentWorldModelUpdateResult:
            targets = jnp.concatenate((bootstrap_observation, reward[None], discount[None]), axis=0)
            stopped_targets = jax.lax.stop_gradient(targets)
            prediction = self._predict_unchecked(state, observation, action)
            cached_prediction_exact = _tree_equal(prediction, decision_cache.prediction)

            member_losses: list[Array] = []
            member_gradients: list[RecurrentLatentMemberParameters] = []
            gradient_norms: list[Array] = []
            member_gradients_valid: list[Array] = []
            for index, parameters in enumerate(state.member_parameters):
                loss, gradient = jax.value_and_grad(self._member_nll)(
                    parameters,
                    state.member_hidden_states[index],
                    observation,
                    action,
                    stopped_targets,
                )
                gradient_norm = _tree_l2_norm(gradient)
                member_losses.append(loss)
                member_gradients.append(gradient)
                gradient_norms.append(gradient_norm)
                # Finiteness only -- see the class docstring.  The raw norm is
                # not compared against max_raw_gradient_norm: a well-trained
                # low-noise member legally produces residual / variance
                # gradients that dwarf any fixed ceiling on an ordinary
                # in-bounds transition, and gradient_clip_norm (applied below)
                # already bounds the committed step regardless of raw scale.
                member_gradients_valid.append(
                    _tree_all_finite(gradient) & jnp.isfinite(gradient_norm)
                )
            losses = jnp.stack(member_losses)
            gradient_norm_array = jnp.stack(gradient_norms)
            gradient_valid_array = jnp.stack(member_gradients_valid)
            mean_loss = jnp.mean(losses)
            losses_valid = (
                jnp.all(jnp.isfinite(losses))
                & jnp.all(jnp.abs(losses) <= cfg.max_loss_magnitude)
                & jnp.isfinite(mean_loss)
                & (jnp.abs(mean_loss) <= cfg.max_loss_magnitude)
            )

            def representation_loss(current_observation: Array) -> Array:
                current_losses = [
                    self._member_nll(
                        state.member_parameters[index],
                        state.member_hidden_states[index],
                        current_observation,
                        action,
                        stopped_targets,
                    )
                    for index in range(cfg.ensemble_size)
                ]
                return jnp.mean(jnp.stack(current_losses))

            representation_objective, representation_gradient = jax.value_and_grad(
                representation_loss
            )(observation)
            representation_gradient_valid = (
                jnp.isfinite(representation_objective)
                & (jnp.abs(representation_objective) <= cfg.max_loss_magnitude)
                & jnp.all(jnp.isfinite(representation_gradient))
                & (_tree_l2_norm(representation_gradient) <= cfg.max_raw_gradient_norm)
            )

            next_bootstrap_key, mask_key = jr.split(state.bootstrap_key)
            bootstrap_mask = jr.bernoulli(
                mask_key,
                p=cfg.bootstrap_probability,
                shape=(cfg.ensemble_size,),
            )
            candidate_parameters: list[RecurrentLatentMemberParameters] = []
            candidate_parameter_valid: list[Array] = []
            for index, parameters in enumerate(state.member_parameters):
                scale = jnp.minimum(
                    1.0,
                    cfg.gradient_clip_norm / jnp.maximum(gradient_norm_array[index], 1.0e-12),
                )
                updated = jax.tree_util.tree_map(
                    lambda parameter, gradient: (
                        parameter
                        - jnp.asarray(cfg.learning_rate, dtype=jnp.float32) * scale * gradient
                    ),
                    parameters,
                    member_gradients[index],
                )
                candidate = cast(
                    RecurrentLatentMemberParameters,
                    jax.lax.cond(
                        bootstrap_mask[index],
                        lambda: updated,
                        lambda: parameters,
                    ),
                )
                candidate_parameters.append(candidate)
                candidate_parameter_valid.append(self._parameters_valid(candidate))
            candidate_parameter_valid_array = jnp.stack(candidate_parameter_valid)

            next_hidden = jnp.where(
                boundary,
                jnp.zeros_like(prediction.member_next_hidden_states),
                prediction.member_next_hidden_states,
            )
            member_counts = state.member_update_counts + bootstrap_mask.astype(jnp.int32)
            candidate_state = RecurrentLatentWorldModelEnsembleState(
                member_parameters=tuple(candidate_parameters),
                member_hidden_states=next_hidden,
                bootstrap_key=next_bootstrap_key,
                last_bootstrap_mask=bootstrap_mask,
                member_update_counts=member_counts,
                event_count=_saturating_increment(state.event_count, cfg.max_updates),
                recurrent_advance_count=_saturating_increment(
                    state.recurrent_advance_count, cfg.max_updates
                ),
                boundary_count=jnp.where(
                    boundary,
                    _saturating_increment(state.boundary_count, cfg.max_updates),
                    state.boundary_count,
                ),
            )
            predictions_valid = prediction.valid
            candidate_state_valid = self._state_valid(candidate_state)
            applied = (
                cached_prediction_exact
                & predictions_valid
                & losses_valid
                & representation_gradient_valid
                & jnp.all(gradient_valid_array)
                & jnp.all(candidate_parameter_valid_array)
                & candidate_state_valid
            )
            committed_state = cast(
                RecurrentLatentWorldModelEnsembleState,
                jax.lax.cond(applied, lambda: candidate_state, lambda: state),
            )
            next_cache = RecurrentLatentStartCache(
                owner_event_count=candidate_state.event_count,
                owner_hidden_states=candidate_state.member_hidden_states,
                observation=next_decision_observation,
                valid=applied,
            )
            diagnostics = RecurrentLatentWorldModelDiagnostics(
                state_valid=state_valid,
                cache_valid=cache_valid,
                input_valid=input_valid,
                ownership_valid=ownership_valid,
                boundary_semantics_valid=boundary_semantics_valid,
                capacity_available=capacity_available,
                cached_prediction_exact=cached_prediction_exact,
                predictions_valid=predictions_valid,
                losses_valid=losses_valid,
                representation_gradient_valid=representation_gradient_valid,
                member_gradients_valid=gradient_valid_array,
                candidate_parameters_valid=candidate_parameter_valid_array,
                candidate_state_valid=candidate_state_valid,
                recurrent_advanced_once=applied
                & (candidate_state.recurrent_advance_count == state.recurrent_advance_count + 1),
                recurrent_reset=applied & boundary,
                applied=applied,
                rejected=~applied,
            )
            result = RecurrentLatentWorldModelUpdateResult(
                state=committed_state,
                prediction=prediction,
                targets=targets,
                member_negative_log_likelihoods=losses,
                mean_negative_log_likelihood=mean_loss,
                representation_objective=representation_objective,
                representation_gradient=representation_gradient,
                representation_gradient_available=applied & representation_gradient_valid,
                bootstrap_mask=bootstrap_mask,
                member_updates_applied=bootstrap_mask & applied,
                next_start_cache=next_cache,
                diagnostics=diagnostics,
            )
            return cast(
                RecurrentLatentWorldModelUpdateResult,
                jax.lax.cond(
                    applied,
                    lambda: result,
                    lambda: self._rejected_result(
                        state,
                        diagnostics,
                        next_decision_observation,
                        boundary,
                    ),
                ),
            )

        return cast(
            RecurrentLatentWorldModelUpdateResult,
            jax.lax.cond(
                can_attempt,
                accepted_branch,
                lambda _: self._rejected_result(
                    state,
                    rejected_diagnostics,
                    next_decision_observation,
                    boundary,
                ),
                operand=None,
            ),
        )

    def resource_budget(
        self,
        state: RecurrentLatentWorldModelEnsembleState | None = None,
    ) -> RecurrentLatentWorldModelResourceBudget:
        """Measure exact logical persistent/cache/result PyTree resources."""
        measured = self.init(jr.key(0)) if state is None else state
        self._validate_state_static(measured)
        if not bool(jax.device_get(self._state_valid(measured))):
            raise ValueError("cannot account an invalid state")
        persistent = _logical_accounting(measured)
        start = _logical_accounting(self._zero_start_cache())
        decision = _logical_accounting(self._zero_decision_cache())
        false = jnp.asarray(False, dtype=jnp.bool_)
        result = _logical_accounting(
            self._rejected_result(
                measured,
                self._zero_diagnostics(
                    state_valid=false,
                    cache_valid=false,
                    input_valid=false,
                    ownership_valid=false,
                    boundary_semantics_valid=false,
                    capacity_available=false,
                ),
            )
        )
        trainable = _logical_accounting(measured.member_parameters[0])
        if trainable.float32_scalars != self._config.trainable_scalars_per_member:
            raise ValueError("measured trainable scalar count differs from config")
        if persistent.logical_bytes != self._config.state_nbytes:
            raise ValueError("measured persistent state bytes differ from config")
        key_words = int(jr.key_data(measured.bootstrap_key).size)
        return RecurrentLatentWorldModelResourceBudget(
            ensemble_size=self._config.ensemble_size,
            observation_dim=self._config.observation_dim,
            latent_dim=self._config.latent_dim,
            target_dim=self._config.target_dim,
            trainable_scalars_per_member=trainable.float32_scalars,
            total_trainable_scalars=(self._config.ensemble_size * trainable.float32_scalars),
            persistent_float32_scalars=persistent.float32_scalars,
            persistent_int32_scalars=persistent.int32_scalars,
            persistent_uint32_scalars=persistent.uint32_scalars,
            persistent_bool_scalars=persistent.bool_scalars,
            persistent_state_scalars=persistent.logical_scalars,
            persistent_state_bytes=persistent.logical_bytes,
            bootstrap_prng_keys=1,
            bootstrap_prng_uint32_scalars=key_words,
            start_cache_logical_scalars=start.logical_scalars,
            start_cache_logical_bytes=start.logical_bytes,
            decision_cache_logical_scalars=decision.logical_scalars,
            decision_cache_logical_bytes=decision.logical_bytes,
            update_result_logical_scalars=result.logical_scalars,
            update_result_logical_bytes=result.logical_bytes,
            member_gradient_candidates_per_event=self._config.ensemble_size,
            max_member_parameter_updates_per_event=self._config.ensemble_size,
            recurrent_advances_per_accepted_event=1,
            max_event_count=self._config.max_updates,
            max_member_update_count=self._config.max_updates,
            replay_capacity=0,
        )


def _config_digest(config: Mapping[str, object]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_recurrent_latent_world_model_ensemble_checkpoint(
    model: RecurrentLatentWorldModelEnsemble,
    state: RecurrentLatentWorldModelEnsembleState,
    path: str | Path,
) -> None:
    """Persist the exact construction, resource record, RNG, and state."""
    if not bool(jax.device_get(model.state_valid(state))):
        raise ValueError("cannot save an invalid recurrent latent ensemble state")
    config = model.to_config()
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": RECURRENT_LATENT_WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA,
            "model_config": config,
            "config_sha256": _config_digest(config),
            "resource_budget": model.resource_budget(state).to_config(),
            "evidence_level": EVIDENCE_LEVEL,
            "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
        },
    )


def load_recurrent_latent_world_model_ensemble_checkpoint(
    path: str | Path,
    *,
    template_key: Array | None = None,
) -> tuple[RecurrentLatentWorldModelEnsemble, RecurrentLatentWorldModelEnsembleState]:
    """Strictly restore a digest-bound v1 recurrent latent ensemble."""
    metadata = load_checkpoint_metadata(path)
    expected_fields = {
        "schema",
        "model_config",
        "config_sha256",
        "resource_budget",
        "evidence_level",
        "scientific_promotion_allowed",
    }
    if set(metadata) != expected_fields:
        raise ValueError("checkpoint metadata fields do not match the v1 schema")
    if metadata.get("schema") != RECURRENT_LATENT_WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint is not a recurrent latent ensemble v1 checkpoint")
    config = metadata.get("model_config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint model_config must be a mapping")
    digest = metadata.get("config_sha256")
    if type(digest) is not str or digest != _config_digest(config):
        raise ValueError("checkpoint config digest does not match")
    model = RecurrentLatentWorldModelEnsemble.from_config(config)
    if not _strict_json_equal(dict(config), model.to_config()):
        raise ValueError("checkpoint model_config is not canonical")
    if metadata.get("evidence_level") != EVIDENCE_LEVEL:
        raise ValueError("checkpoint evidence level is not L0")
    if metadata.get("scientific_promotion_allowed") is not False:
        raise ValueError("checkpoint cannot claim scientific promotion")
    key = jr.key(0) if template_key is None else template_key
    # The template supplies only the checkpoint's static PyTree contract.  It
    # must not make restoration depend on whether this unrelated random draw
    # happens to fit the configured parameter bound; the persisted state is
    # validated immediately after it is reconstructed.
    template = model._initial_state(key)
    restored, second_metadata = load_checkpoint(template, path)
    if second_metadata != metadata:
        raise ValueError("checkpoint metadata changed between reads")
    state = cast(RecurrentLatentWorldModelEnsembleState, restored)
    if not bool(jax.device_get(model.state_valid(state))):
        raise ValueError("restored recurrent latent ensemble state is invalid")
    expected_budget = model.resource_budget(state).to_config()
    if metadata.get("resource_budget") != expected_budget:
        raise ValueError("checkpoint resource budget does not match config and state")
    return model, state


__all__ = [
    "EVIDENCE_LEVEL",
    "RECURRENT_LATENT_WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "RecurrentLatentDecisionCache",
    "RecurrentLatentMemberParameters",
    "RecurrentLatentPredictionAvailability",
    "RecurrentLatentStartCache",
    "RecurrentLatentTransitionRecord",
    "RecurrentLatentWorldModelDiagnostics",
    "RecurrentLatentWorldModelEnsemble",
    "RecurrentLatentWorldModelEnsembleConfig",
    "RecurrentLatentWorldModelEnsembleState",
    "RecurrentLatentWorldModelPrediction",
    "RecurrentLatentWorldModelResourceBudget",
    "RecurrentLatentWorldModelUpdateResult",
    "load_recurrent_latent_world_model_ensemble_checkpoint",
    "save_recurrent_latent_world_model_ensemble_checkpoint",
]
