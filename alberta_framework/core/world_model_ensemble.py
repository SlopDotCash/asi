# mypy: disable-error-code="call-arg"
"""Bounded predict-before-update ensembles for one-step world models.

This module composes a fixed number of
:class:`~alberta_framework.core.world_model.ActionConditionedWorldModel`
states.  Every event first evaluates every member and emits causal typed
learning signals.  Only after that observation has been recorded are
bootstrap-masked member updates and residual-variance statistics formed.

Bootstrap masking follows the bootstrapped-ensemble recipe (Osband et al.
2016): each member commits a given event's update only with probability
``bootstrap_probability``, so the members train on different subsamples of
the same stream and stay decorrelated.  The variance across member
predictions (``epistemic_disagreement``) then serves as an inexpensive
epistemic-uncertainty proxy — members agree where data has constrained them
all and disagree where it has not (Lakshminarayanan et al. 2017).

The residual variance is a per-member, per-head EMA of earlier squared
residuals.  It is only a development residual-variance proxy, not a learned or
calibrated aleatoric-uncertainty estimate.  Uncertainty channels remain
unavailable until the configured warmup has put enough observed residuals into
the proxy used for a prediction.

The subsystem is transactional.  Invalid runtime input, corrupt dynamic
state, a non-finite prediction, or a non-finite candidate update leaves the
complete state (including RNG, masks, counters, models, and signal estimator)
unchanged.  Static shape and dtype mismatches raise before compiled execution.

References:
    Osband, Blundell, Pritzel, & Van Roy (2016). "Deep Exploration via
        Bootstrapped DQN."
    Lakshminarayanan, Pritzel, & Blundell (2017). "Simple and Scalable
        Predictive Uncertainty Estimation Using Deep Ensembles."
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import operator
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

from alberta_framework.core._float32_scalars import validated_float32_scalar
from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.learning_signals import (
    LearningSignalAvailability,
    LearningSignalEstimator,
    LearningSignalEstimatorConfig,
    LearningSignalEstimatorState,
    TypedLearningSignals,
)
from alberta_framework.core.world_model import (
    ActionConditionedWorldModel,
    ActionConditionedWorldModelConfig,
    ActionConditionedWorldModelState,
)

WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA = "alberta.world_model_ensemble.v2"
_WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA_V1 = "alberta.world_model_ensemble.v1"
_INT32_MAX = 2**31 - 1
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
_ACTUAL_FLOAT_TYPES = frozenset(
    {float, Fraction, *(np.dtype(code).type for code in ("e", "f", "d", "g"))}
)


def _require_int(
    name: str,
    value: object,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer")
    number = operator.index(cast(SupportsIndex, value))
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return number


def _validated_config_float(name: str, value: object, **bounds: Any) -> float:
    if type(value) not in (frozenset(_ACTUAL_INT_TYPES) | _ACTUAL_FLOAT_TYPES):
        raise ValueError(f"{name} must be a finite real scalar")
    return validated_float32_scalar(name, value, **bounds)


def _member_state_scalars(config: ActionConditionedWorldModelConfig) -> int:
    """Return exact default-LMS member state scalars counted by the budget surface."""
    input_dim = config.observation_dim + config.n_actions
    if config.include_action_interactions:
        input_dim += config.observation_dim * config.n_actions
    layer_sizes = (input_dim, *config.hidden_sizes)
    trunk_parameters = sum(
        fan_out * (fan_in + 1)
        for fan_in, fan_out in zip(layer_sizes, layer_sizes[1:], strict=False)
    )
    n_heads = config.observation_dim + 2
    final_width = config.hidden_sizes[-1] if config.hidden_sizes else input_dim
    parameters = trunk_parameters + n_heads * (final_width + 1)
    tensor_count = 2 * len(config.hidden_sizes) + 2 * n_heads
    return (
        2 * parameters
        + tensor_count
        + sum(config.hidden_sizes)
        + 2 * config.observation_dim
        + 9
    )


def _preflight_ensemble_state_resources(
    *, model: ActionConditionedWorldModelConfig, ensemble_size: int
) -> None:
    target_dim = model.observation_dim + 2
    members = ensemble_size * _member_state_scalars(model)
    residuals = ensemble_size * target_dim
    logical_scalars = members + residuals + 4 * ensemble_size + 15
    logical_bytes = 4 * members + 4 * residuals + 10 * ensemble_size + 60
    for name, value in (
        ("ensemble member state scalars", members),
        ("ensemble residual scalars", residuals),
        ("ensemble persistent state scalars", logical_scalars),
        ("ensemble persistent state bytes", logical_bytes),
    ):
        if not 1 <= value <= _INT32_MAX:
            raise ValueError(f"derived {name} must fit signed int32")

    update_float32_scalars = (
        2 * ensemble_size * target_dim
        + 3 * target_dim
        + ensemble_size * model.observation_dim
        + 2 * model.observation_dim
        + 3 * ensemble_size
        + 13
    )
    update_bool_scalars = 2 * ensemble_size + 20
    update_scalars = logical_scalars + update_float32_scalars + update_bool_scalars
    update_bytes = logical_bytes + 4 * update_float32_scalars + update_bool_scalars
    for name, value in (
        ("ensemble update-result scalars", update_scalars),
        ("ensemble update-result bytes", update_bytes),
    ):
        if not 1 <= value <= _INT32_MAX:
            raise ValueError(f"derived {name} must fit signed int32")


def _safe_mean(values: Array, *, axis: int | None = None) -> Array:
    """Compute a float32 mean without first forming an overflowing sum."""
    divisor = values.size if axis is None else values.shape[axis]
    return jnp.sum(values / jnp.asarray(divisor, dtype=values.dtype), axis=axis)


_REPLAY_KEY_FOLD_IN = 0x5245504C
_V1_REPLAY_KEY_FOLD_IN = 0x50525632


def _saturating_int32_increment(value: Array) -> Array:
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    return jnp.minimum(jnp.maximum(value, 0), maximum - 1) + 1


def _saturating_int32_sum(values: Array) -> Array:
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)

    def add(total: Array, value: Array) -> Array:
        return total + jnp.minimum(value, maximum - total)

    return cast(
        Array,
        jax.lax.fori_loop(
            0,
            values.shape[0],
            lambda index, total: add(total, values[index]),
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )


@dataclasses.dataclass(frozen=True)
class WorldModelEnsembleConfig:
    """Static ensemble, bootstrap, and residual-proxy contract.

    ``bootstrap_probability`` must be strictly between zero and one so the
    persisted mask stream can expose members to genuinely different subsets.
    A particular event may still draw an all-true or all-false mask.  Each
    member commits, in expectation, ``bootstrap_probability`` of all valid
    events.

    ``residual_variance_decay`` is the EMA decay of the squared-residual
    proxy; the default 0.99 averages over an effective window of roughly
    ``1 / (1 - decay)`` = 100 events.  ``residual_variance_warmup_steps``
    gates the uncertainty channels; the default of 1 requires only that one
    real residual has been observed, so the floor-initialized prior variance
    is never reported as if it were measured.

    ``signal_estimator.ensemble_size`` and ``target_dim`` must exactly match
    this ensemble and the normalized raw world-model head vector.
    """

    model: ActionConditionedWorldModelConfig
    signal_estimator: LearningSignalEstimatorConfig
    ensemble_size: int = 3
    bootstrap_probability: float = 0.8
    residual_variance_decay: float = 0.99
    residual_variance_warmup_steps: int = 1
    residual_variance_floor: float = 1.0e-6

    def __post_init__(self) -> None:
        if type(self.model) is not ActionConditionedWorldModelConfig:
            raise ValueError("model must be an exact ActionConditionedWorldModelConfig")
        if type(self.signal_estimator) is not LearningSignalEstimatorConfig:
            raise ValueError("signal_estimator must be an exact LearningSignalEstimatorConfig")
        ensemble_size = _require_int(
            "ensemble_size", self.ensemble_size, minimum=2, maximum=_INT32_MAX - 1
        )
        bootstrap_probability = _validated_config_float(
            "bootstrap_probability",
            self.bootstrap_probability,
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
            positive=True,
        )
        residual_variance_decay = _validated_config_float(
            "residual_variance_decay",
            self.residual_variance_decay,
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
        )
        residual_variance_warmup_steps = _require_int(
            "residual_variance_warmup_steps",
            self.residual_variance_warmup_steps,
            minimum=1,
            maximum=_INT32_MAX,
        )
        residual_variance_floor = _validated_config_float(
            "residual_variance_floor",
            self.residual_variance_floor,
            positive=True,
        )
        if residual_variance_floor < self.signal_estimator.variance_floor:
            raise ValueError("residual_variance_floor must be >= signal_estimator.variance_floor")
        if residual_variance_floor > self.signal_estimator.max_predicted_variance:
            raise ValueError("residual_variance_floor exceeds the signal estimator variance bound")
        if self.signal_estimator.ensemble_size != ensemble_size:
            raise ValueError("signal_estimator.ensemble_size must match ensemble_size")
        expected_target_dim = self.model.observation_dim + 2
        if self.signal_estimator.target_dim != expected_target_dim:
            raise ValueError("signal_estimator.target_dim must equal model.observation_dim + 2")
        object.__setattr__(self, "ensemble_size", ensemble_size)
        object.__setattr__(self, "bootstrap_probability", bootstrap_probability)
        object.__setattr__(self, "residual_variance_decay", residual_variance_decay)
        object.__setattr__(
            self, "residual_variance_warmup_steps", residual_variance_warmup_steps
        )
        object.__setattr__(self, "residual_variance_floor", residual_variance_floor)
        _preflight_ensemble_state_resources(model=self.model, ensemble_size=ensemble_size)

    @property
    def target_dim(self) -> int:
        """Number of normalized mean-prediction heads per member."""
        return self.model.observation_dim + 2

    def to_config(self) -> dict[str, Any]:
        """Return a JSON-compatible configuration."""
        return {
            "type": "WorldModelEnsembleConfig",
            "model": self.model.to_config(),
            "signal_estimator": self.signal_estimator.to_config(),
            "ensemble_size": self.ensemble_size,
            "bootstrap_probability": self.bootstrap_probability,
            "residual_variance_decay": self.residual_variance_decay,
            "residual_variance_warmup_steps": self.residual_variance_warmup_steps,
            "residual_variance_floor": self.residual_variance_floor,
            "development_only": True,
            "accepted_scientific_evidence": False,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> WorldModelEnsembleConfig:
        """Reconstruct the exact development-only configuration."""
        if not issubclass(type(config), Mapping):
            raise ValueError("config must be an actual mapping")
        try:
            payload = dict(config)
        except Exception as error:
            raise ValueError("config must be a readable mapping") from error
        if any(type(key) is not str for key in payload):
            raise ValueError("config keys must be exact strings")
        type_name = payload.pop("type", None)
        if type_name is not None and (
            type(type_name) is not str or type_name != "WorldModelEnsembleConfig"
        ):
            raise ValueError("type must be WorldModelEnsembleConfig")
        if payload.pop("development_only", True) is not True:
            raise ValueError("world-model ensemble is development_only")
        if payload.pop("accepted_scientific_evidence", False) is not False:
            raise ValueError("world-model ensemble is not accepted scientific evidence")
        raw_model = payload.pop("model")
        raw_signals = payload.pop("signal_estimator")
        if not issubclass(type(raw_model), Mapping) or not issubclass(
            type(raw_signals), Mapping
        ):
            raise ValueError("nested configs must be actual mappings")
        model = ActionConditionedWorldModelConfig.from_config(raw_model)
        try:
            signal_payload = dict(raw_signals)
        except Exception as error:
            raise ValueError("signal_estimator must be a readable mapping") from error
        if any(type(key) is not str for key in signal_payload):
            raise ValueError("signal_estimator keys must be exact strings")
        signal_estimator = LearningSignalEstimatorConfig.from_config(signal_payload)
        return cls(model=model, signal_estimator=signal_estimator, **payload)


@dataclasses.dataclass(frozen=True)
class WorldModelEnsembleResourceBudget:
    """Exact fixed-state and bounded-output logical resource accounting.

    Logical byte counts use the canonical JAX scalar width for every PyTree
    leaf and count a typed Threefry PRNG key as its two underlying uint32
    words.  They exclude Python object overhead, device alignment, compiled
    code, autodiff tapes, and other transient compiler/runtime buffers.

    ``update_result_output_*`` and ``replay_update_result_output_*`` count the
    complete returned results, including their returned persistent state.  The
    update-count fields distinguish exact candidate evaluations from the
    maximum number bootstrap masking can commit.  Real and replay bootstrap
    keys/counters are separate so rehearsal cannot perturb the real-event mask
    stream or evidence calibration.  ``replay_capacity`` is zero because this
    class owns no transition storage; a rehearsal composer supplies separately
    accounted bounded replay slots.  It does not mean replay updates are
    unsupported.
    """

    ensemble_size: int
    observation_dim: int
    target_dim: int
    member_state_scalars_per_member: int
    member_state_bytes_per_member: int
    member_trainable_scalars: int
    total_trainable_scalars: int
    persistent_float32_scalars: int
    persistent_float64_scalars: int
    persistent_int32_scalars: int
    persistent_int64_scalars: int
    persistent_uint32_scalars: int
    persistent_bool_scalars: int
    persistent_state_scalars: int
    persistent_state_bytes: int
    bootstrap_prng_keys: int
    bootstrap_prng_uint32_scalars: int
    bootstrap_prng_bytes: int
    prediction_output_logical_scalars: int
    prediction_output_logical_bytes: int
    update_result_output_logical_scalars: int
    update_result_output_logical_bytes: int
    replay_update_result_output_logical_scalars: int
    replay_update_result_output_logical_bytes: int
    member_update_candidates_per_valid_event: int
    max_member_updates_per_event: int
    replay_member_update_candidates_per_available_sample: int
    max_replay_member_updates_per_available_sample: int
    max_event_count: int
    max_member_update_count: int
    max_replay_event_count: int
    max_replay_member_update_count: int
    replay_capacity: int

    def to_config(self) -> dict[str, int]:
        """Return a JSON-compatible exact accounting record."""
        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class WorldModelEnsembleState:
    """Complete fixed-budget ensemble and causal-signal state."""

    member_states: tuple[ActionConditionedWorldModelState, ...]
    residual_variances: Float[Array, "ensemble_size target_dim"]
    signal_state: LearningSignalEstimatorState
    bootstrap_key: Array
    replay_bootstrap_key: Array
    last_bootstrap_mask: Bool[Array, " ensemble_size"]
    last_replay_bootstrap_mask: Bool[Array, " ensemble_size"]
    member_update_counts: Int[Array, " ensemble_size"]
    replay_member_update_counts: Int[Array, " ensemble_size"]
    event_count: Int[Array, ""]
    replay_event_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class _WorldModelEnsembleStateV1:
    """Exact pre-rehearsal checkpoint tree used only for strict migration."""

    member_states: tuple[ActionConditionedWorldModelState, ...]
    residual_variances: Array
    signal_state: LearningSignalEstimatorState
    bootstrap_key: Array
    last_bootstrap_mask: Array
    member_update_counts: Array
    event_count: Array


@chex.dataclass(frozen=True)
class WorldModelEnsemblePrediction:
    """Pre-update member predictions and disagreement diagnostics."""

    member_raw_predictions: Float[Array, "ensemble_size target_dim"]
    mean_raw_prediction: Float[Array, " target_dim"]
    member_next_observations: Float[Array, "ensemble_size observation_dim"]
    mean_next_observation: Float[Array, " observation_dim"]
    member_rewards: Float[Array, " ensemble_size"]
    mean_reward: Float[Array, ""]
    member_discounts: Float[Array, " ensemble_size"]
    mean_discount: Float[Array, ""]
    per_head_epistemic_variance: Float[Array, " target_dim"]
    epistemic_disagreement: Float[Array, ""]
    residual_variances: Float[Array, "ensemble_size target_dim"]
    residual_proxy_ready: Bool[Array, ""]
    valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class WorldModelEnsembleDiagnostics:
    """Atomic transaction checks for one ensemble event."""

    state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    capacity_available: Bool[Array, ""]
    predictions_valid: Bool[Array, ""]
    representation_gradient_valid: Bool[Array, ""]
    signals_valid: Bool[Array, ""]
    residual_update_valid: Bool[Array, ""]
    member_updates_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    applied: Bool[Array, ""]
    rejected: Bool[Array, ""]


@chex.dataclass(frozen=True)
class WorldModelEnsembleUpdateResult:
    """One bounded predict-signal-update transaction.

    ``representation_objective`` is half the mean normalized squared residual
    across the pre-update member/head grid.  ``representation_gradient`` is
    its derivative with respect to the current observation representation,
    with the already-formed target held constant.  The gradient is zero unless
    ``representation_gradient_valid`` is true.
    """

    state: WorldModelEnsembleState
    prediction: WorldModelEnsemblePrediction
    signals: TypedLearningSignals
    targets: Float[Array, " target_dim"]
    observed_loss: Float[Array, ""]
    member_prediction_losses: Float[Array, " ensemble_size"]
    representation_objective: Float[Array, ""]
    representation_gradient: Float[Array, " observation_dim"]
    representation_gradient_valid: Bool[Array, ""]
    bootstrap_mask: Bool[Array, " ensemble_size"]
    member_updates_applied: Bool[Array, " ensemble_size"]
    diagnostics: WorldModelEnsembleDiagnostics


@chex.dataclass(frozen=True)
class WorldModelEnsembleReplayDiagnostics:
    """Checks for one model-only rehearsal sample.

    A padding position has ``sample_available=False``, ``applied=False``, and
    ``rejected=False``.  An available invalid sample is rejected atomically.
    """

    state_valid: Bool[Array, ""]
    sample_available: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    capacity_available: Bool[Array, ""]
    predictions_valid: Bool[Array, ""]
    member_updates_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    calibration_unchanged: Bool[Array, ""]
    applied: Bool[Array, ""]
    rejected: Bool[Array, ""]


@chex.dataclass(frozen=True)
class WorldModelEnsembleReplayUpdateResult:
    """One replay-only member update with no learning-signal observation."""

    state: WorldModelEnsembleState
    prediction: WorldModelEnsemblePrediction
    targets: Float[Array, " target_dim"]
    observed_loss: Float[Array, ""]
    member_prediction_losses: Float[Array, " ensemble_size"]
    bootstrap_mask: Bool[Array, " ensemble_size"]
    member_updates_applied: Bool[Array, " ensemble_size"]
    diagnostics: WorldModelEnsembleReplayDiagnostics


@dataclasses.dataclass(frozen=True)
class _LogicalTreeAccounting:
    float32_scalars: int = 0
    float64_scalars: int = 0
    int32_scalars: int = 0
    int64_scalars: int = 0
    uint32_scalars: int = 0
    bool_scalars: int = 0

    @property
    def logical_scalars(self) -> int:
        return (
            self.float32_scalars
            + self.float64_scalars
            + self.int32_scalars
            + self.int64_scalars
            + self.uint32_scalars
            + self.bool_scalars
        )

    @property
    def logical_bytes(self) -> int:
        return (
            4 * (self.float32_scalars + self.int32_scalars + self.uint32_scalars)
            + 8 * (self.float64_scalars + self.int64_scalars)
            + self.bool_scalars
        )


def _zero_signals() -> TypedLearningSignals:
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    false = jnp.asarray(False, dtype=jnp.bool_)
    return TypedLearningSignals(
        epistemic_disagreement=zero,
        epistemic_surprise=zero,
        aleatoric_uncertainty=zero,
        normalized_residual=zero,
        learning_progress=zero,
        calibrated_residual_z=zero,
        instantaneous_change_probability=zero,
        change_probability=zero,
        availability=LearningSignalAvailability(
            input_valid=false,
            epistemic=false,
            aleatoric=false,
            normalized_residual=false,
            learning_progress=false,
            change_probability=false,
        ),
    )


def _logical_tree_accounting(tree: Any) -> _LogicalTreeAccounting:
    """Count canonical numeric PyTree leaves without counting runtime buffers."""
    counts = {
        "float32": 0,
        "float64": 0,
        "int32": 0,
        "int64": 0,
        "uint32": 0,
        "bool": 0,
    }
    for leaf in jax.tree_util.tree_leaves(tree):
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jnp.issubdtype(
            dtype,
            jax.dtypes.prng_key,
        ):
            array = jr.key_data(leaf)
        else:
            array = jnp.asarray(leaf)
        size = int(array.size)
        if array.dtype == jnp.dtype(jnp.float32):
            counts["float32"] += size
        elif array.dtype == jnp.dtype(jnp.float64):
            counts["float64"] += size
        elif array.dtype == jnp.dtype(jnp.int32):
            counts["int32"] += size
        elif array.dtype == jnp.dtype(jnp.int64):
            counts["int64"] += size
        elif array.dtype == jnp.dtype(jnp.uint32):
            counts["uint32"] += size
        elif array.dtype == jnp.dtype(jnp.bool_):
            counts["bool"] += size
        else:
            raise TypeError(
                "world-model ensemble resource accounting does not support "
                f"logical dtype {array.dtype}"
            )
    return _LogicalTreeAccounting(
        float32_scalars=counts["float32"],
        float64_scalars=counts["float64"],
        int32_scalars=counts["int32"],
        int64_scalars=counts["int64"],
        uint32_scalars=counts["uint32"],
        bool_scalars=counts["bool"],
    )


def _tree_all_finite(tree: Any) -> Array:
    checks: list[Array] = []
    for leaf in jax.tree_util.tree_leaves(tree):
        value = jnp.asarray(leaf)
        if jnp.issubdtype(value.dtype, jnp.inexact):
            checks.append(jnp.all(jnp.isfinite(value)))
    if not checks:
        return jnp.asarray(True, dtype=jnp.bool_)
    return jnp.all(jnp.stack(checks))


def _real_array(value: Any, shape: tuple[int, ...], *, name: str) -> Array:
    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if (
        jnp.issubdtype(array.dtype, jnp.bool_)
        or jnp.issubdtype(array.dtype, jnp.complexfloating)
        or not (
            jnp.issubdtype(array.dtype, jnp.floating) or jnp.issubdtype(array.dtype, jnp.integer)
        )
    ):
        raise ValueError(f"{name} must have a real numeric dtype")
    return jnp.asarray(array, dtype=jnp.float32)


def _action_scalar(value: Any, *, name: str) -> Array:
    array = jnp.asarray(value)
    if array.shape != () or array.dtype != jnp.int32:
        raise ValueError(f"{name} must be a scalar with dtype int32")
    return array


def _tree_static_signature(tree: Any) -> tuple[Any, tuple[tuple[tuple[int, ...], Any], ...]]:
    leaves, structure = jax.tree_util.tree_flatten(tree)
    specifications = tuple((jnp.asarray(leaf).shape, jnp.asarray(leaf).dtype) for leaf in leaves)
    return structure, specifications


def _validate_tree_static_signature(
    tree: Any,
    expected: tuple[Any, tuple[tuple[tuple[int, ...], Any], ...]],
    *,
    name: str,
) -> None:
    leaves, structure = jax.tree_util.tree_flatten(tree)
    expected_structure, expected_specifications = expected
    if structure != expected_structure:
        raise ValueError(f"{name} structure does not match the configured model")
    if len(leaves) != len(expected_specifications):
        raise ValueError(f"{name} leaf count does not match the configured model")
    for index, (leaf, (shape, dtype)) in enumerate(
        zip(leaves, expected_specifications, strict=True)
    ):
        array = jnp.asarray(leaf)
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(f"{name} leaf {index} must have shape {shape} and dtype {dtype}")


class WorldModelEnsemble:
    """Fixed-size bootstrap ensemble with causal typed learning signals."""

    def __init__(self, config: WorldModelEnsembleConfig):
        if type(config) is not WorldModelEnsembleConfig:
            raise ValueError("config must be an exact WorldModelEnsembleConfig")
        self._config = config
        self._model = ActionConditionedWorldModel(config.model)
        self._signals = LearningSignalEstimator(config.signal_estimator)
        self._member_state_static_signature = _tree_static_signature(self._model.init(jr.key(0)))
        self._signal_state_static_signature = _tree_static_signature(self._signals.init())

    @property
    def config(self) -> WorldModelEnsembleConfig:
        """Return the immutable ensemble configuration."""
        return self._config

    @property
    def member_model(self) -> ActionConditionedWorldModel:
        """Return the shared member implementation."""
        return self._model

    @property
    def signal_estimator(self) -> LearningSignalEstimator:
        """Return the causal signal estimator."""
        return self._signals

    def to_config(self) -> dict[str, Any]:
        """Serialize the complete static ensemble construction."""
        return {
            "type": "WorldModelEnsemble",
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> WorldModelEnsemble:
        """Reconstruct from :meth:`to_config` output."""
        if not issubclass(type(config), Mapping):
            raise ValueError("ensemble payload must be an actual mapping")
        try:
            payload = dict(config)
        except Exception as error:
            raise ValueError("ensemble payload must be a readable mapping") from error
        if any(type(key) is not str for key in payload):
            raise ValueError("ensemble payload keys must be exact strings")
        type_name = payload.pop("type", None)
        if type_name is not None and (
            type(type_name) is not str or type_name != "WorldModelEnsemble"
        ):
            raise ValueError("type must be WorldModelEnsemble")
        if set(payload) != {"config"}:
            raise ValueError("world-model ensemble config keys are invalid")
        nested = payload["config"]
        if not issubclass(type(nested), Mapping):
            raise ValueError("nested ensemble config must be an actual mapping")
        return cls(WorldModelEnsembleConfig.from_config(nested))

    def init(self, key: Array) -> WorldModelEnsembleState:
        """Initialize distinct members and isolated real/replay mask streams."""
        keys = jr.split(key, self._config.ensemble_size + 1)
        member_states = tuple(
            self._model.init(keys[index]) for index in range(self._config.ensemble_size)
        )
        return WorldModelEnsembleState(
            member_states=member_states,
            residual_variances=jnp.full(
                (self._config.ensemble_size, self._config.target_dim),
                self._config.residual_variance_floor,
                dtype=jnp.float32,
            ),
            signal_state=self._signals.init(),
            bootstrap_key=keys[-1],
            replay_bootstrap_key=jr.fold_in(key, _REPLAY_KEY_FOLD_IN),
            last_bootstrap_mask=jnp.zeros(
                (self._config.ensemble_size,),
                dtype=jnp.bool_,
            ),
            last_replay_bootstrap_mask=jnp.zeros(
                (self._config.ensemble_size,),
                dtype=jnp.bool_,
            ),
            member_update_counts=jnp.zeros(
                (self._config.ensemble_size,),
                dtype=jnp.int32,
            ),
            replay_member_update_counts=jnp.zeros(
                (self._config.ensemble_size,),
                dtype=jnp.int32,
            ),
            event_count=jnp.asarray(0, dtype=jnp.int32),
            replay_event_count=jnp.asarray(0, dtype=jnp.int32),
        )

    def resource_budget(
        self,
        state: WorldModelEnsembleState | None = None,
    ) -> WorldModelEnsembleResourceBudget:
        """Return exact logical counts measured from the fixed-shape PyTrees.

        Passing a state verifies its complete static tree contract before
        measuring it.  Omitting the state measures a freshly initialized
        state, which is the canonical checkpoint budget for this config.
        """
        measured_state = self.init(jr.key(0)) if state is None else state
        self._validate_state_static_contract(measured_state)

        member_accounts = tuple(
            _logical_tree_accounting(member) for member in measured_state.member_states
        )
        if any(account != member_accounts[0] for account in member_accounts[1:]):
            raise ValueError("ensemble members do not have identical resource shapes")
        member_account = member_accounts[0]

        trainable_accounts = tuple(
            _logical_tree_accounting(
                (
                    member.learner_state.trunk_params,
                    member.learner_state.head_params,
                )
            )
            for member in measured_state.member_states
        )
        if any(account != trainable_accounts[0] for account in trainable_accounts[1:]):
            raise ValueError("ensemble members do not have identical trainable shapes")
        trainable_account = trainable_accounts[0]
        if (
            trainable_account.int32_scalars
            or trainable_account.int64_scalars
            or trainable_account.uint32_scalars
            or trainable_account.bool_scalars
        ):
            raise TypeError("world-model trainable parameters must be floating point")
        member_trainable_scalars = (
            trainable_account.float32_scalars + trainable_account.float64_scalars
        )

        persistent = _logical_tree_accounting(measured_state)
        bootstrap_key_data = jr.key_data(measured_state.bootstrap_key)
        replay_key_data = jr.key_data(measured_state.replay_bootstrap_key)
        if bootstrap_key_data.dtype != jnp.uint32 or replay_key_data.dtype != jnp.uint32:
            raise TypeError("bootstrap PRNG key storage must use uint32 words")
        bootstrap_words = int(bootstrap_key_data.size + replay_key_data.size)
        bootstrap_bytes = int(bootstrap_key_data.nbytes + replay_key_data.nbytes)
        expected_uint32 = (
            bootstrap_words + member_account.uint32_scalars * self._config.ensemble_size
        )
        if persistent.uint32_scalars != expected_uint32 or bootstrap_bytes != 4 * bootstrap_words:
            raise ValueError("bootstrap PRNG key accounting does not match state")

        prediction = _logical_tree_accounting(self._zero_prediction())
        false = jnp.asarray(False, dtype=jnp.bool_)
        update_result = _logical_tree_accounting(
            self._rejected_update_result(
                measured_state,
                state_valid=false,
                input_valid=false,
                capacity_available=false,
            )
        )
        replay_update_result = _logical_tree_accounting(
            self._rejected_replay_update_result(
                measured_state,
                state_valid=false,
                sample_available=false,
                input_valid=false,
                capacity_available=false,
            )
        )
        return WorldModelEnsembleResourceBudget(
            ensemble_size=self._config.ensemble_size,
            observation_dim=self._config.model.observation_dim,
            target_dim=self._config.target_dim,
            member_state_scalars_per_member=member_account.logical_scalars,
            member_state_bytes_per_member=member_account.logical_bytes,
            member_trainable_scalars=member_trainable_scalars,
            total_trainable_scalars=(self._config.ensemble_size * member_trainable_scalars),
            persistent_float32_scalars=persistent.float32_scalars,
            persistent_float64_scalars=persistent.float64_scalars,
            persistent_int32_scalars=persistent.int32_scalars,
            persistent_int64_scalars=persistent.int64_scalars,
            persistent_uint32_scalars=persistent.uint32_scalars,
            persistent_bool_scalars=persistent.bool_scalars,
            persistent_state_scalars=persistent.logical_scalars,
            persistent_state_bytes=persistent.logical_bytes,
            bootstrap_prng_keys=2,
            bootstrap_prng_uint32_scalars=bootstrap_words,
            bootstrap_prng_bytes=bootstrap_bytes,
            prediction_output_logical_scalars=prediction.logical_scalars,
            prediction_output_logical_bytes=prediction.logical_bytes,
            update_result_output_logical_scalars=update_result.logical_scalars,
            update_result_output_logical_bytes=update_result.logical_bytes,
            replay_update_result_output_logical_scalars=(
                replay_update_result.logical_scalars
            ),
            replay_update_result_output_logical_bytes=replay_update_result.logical_bytes,
            member_update_candidates_per_valid_event=self._config.ensemble_size,
            max_member_updates_per_event=self._config.ensemble_size,
            replay_member_update_candidates_per_available_sample=(
                self._config.ensemble_size
            ),
            max_replay_member_updates_per_available_sample=self._config.ensemble_size,
            max_event_count=_INT32_MAX,
            max_member_update_count=_INT32_MAX,
            max_replay_event_count=_INT32_MAX,
            max_replay_member_update_count=_INT32_MAX,
            replay_capacity=0,
        )

    def _validate_state_static_contract(self, state: WorldModelEnsembleState) -> None:
        if not isinstance(state, WorldModelEnsembleState):
            raise TypeError("state must be a WorldModelEnsembleState")
        if len(state.member_states) != self._config.ensemble_size:
            raise ValueError("state.member_states length does not match ensemble_size")
        for index, member_state in enumerate(state.member_states):
            _validate_tree_static_signature(
                member_state,
                self._member_state_static_signature,
                name=f"state.member_states[{index}]",
            )
        _validate_tree_static_signature(
            state.signal_state,
            self._signal_state_static_signature,
            name="state.signal_state",
        )
        expected_arrays = {
            "state.residual_variances": (
                state.residual_variances,
                (self._config.ensemble_size, self._config.target_dim),
                jnp.float32,
            ),
            "state.last_bootstrap_mask": (
                state.last_bootstrap_mask,
                (self._config.ensemble_size,),
                jnp.bool_,
            ),
            "state.last_replay_bootstrap_mask": (
                state.last_replay_bootstrap_mask,
                (self._config.ensemble_size,),
                jnp.bool_,
            ),
            "state.member_update_counts": (
                state.member_update_counts,
                (self._config.ensemble_size,),
                jnp.int32,
            ),
            "state.replay_member_update_counts": (
                state.replay_member_update_counts,
                (self._config.ensemble_size,),
                jnp.int32,
            ),
            "state.event_count": (state.event_count, (), jnp.int32),
            "state.replay_event_count": (state.replay_event_count, (), jnp.int32),
        }
        for name, (value, shape, dtype) in expected_arrays.items():
            array = jnp.asarray(value)
            if array.shape != shape or array.dtype != jnp.dtype(dtype):
                raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")

        key_data = jr.key_data(state.bootstrap_key)
        if key_data.shape != (2,) or key_data.dtype != jnp.uint32:
            raise ValueError("state.bootstrap_key must be one JAX PRNG key")
        replay_key_data = jr.key_data(state.replay_bootstrap_key)
        if replay_key_data.shape != (2,) or replay_key_data.dtype != jnp.uint32:
            raise ValueError("state.replay_bootstrap_key must be one JAX PRNG key")

    @staticmethod
    def _signal_state_valid(state: LearningSignalEstimatorState) -> Array:
        counters = jnp.stack(
            (
                state.valid_count,
                state.invalid_count,
            )
        )
        return (
            (state.step_count >= 0)
            & (state.valid_count >= 0)
            & (state.invalid_count >= 0)
            & (state.step_count == _saturating_int32_sum(counters))
            & (state.calibration_count >= 0)
            & jnp.isfinite(state.calibration_mean)
            & (state.calibration_mean >= 0.0)
            & jnp.isfinite(state.calibration_m2)
            & (state.calibration_m2 >= 0.0)
            & jnp.isfinite(state.fast_loss_ema)
            & (state.fast_loss_ema >= 0.0)
            & jnp.isfinite(state.slow_loss_ema)
            & (state.slow_loss_ema >= 0.0)
            & jnp.isfinite(state.sustained_change_probability)
            & (state.sustained_change_probability >= 0.0)
            & (state.sustained_change_probability <= 1.0)
        )

    @staticmethod
    def _member_state_valid(state: ActionConditionedWorldModelState) -> Array:
        step_count = jnp.asarray(state.step_count, dtype=jnp.int32)
        pristine_bounds = (
            jnp.all(jnp.isposinf(state.observation_min))
            & jnp.all(jnp.isneginf(state.observation_max))
            & jnp.isposinf(state.reward_min)
            & jnp.isneginf(state.reward_max)
        )
        learned_bounds = (
            jnp.all(jnp.isfinite(state.observation_min))
            & jnp.all(jnp.isfinite(state.observation_max))
            & jnp.all(state.observation_min <= state.observation_max)
            & jnp.isfinite(state.reward_min)
            & jnp.isfinite(state.reward_max)
            & (state.reward_min <= state.reward_max)
        )
        return (
            (step_count >= 0)
            & (step_count <= _INT32_MAX)
            & (state.learner_state.step_count == step_count)
            & _tree_all_finite(state.learner_state)
            & jnp.isfinite(state.model_error_ema)
            & (state.model_error_ema >= 0.0)
            & jnp.where(step_count == 0, pristine_bounds, learned_bounds)
        )

    def _state_valid(self, state: WorldModelEnsembleState) -> Array:
        member_valid = []
        member_counts_match = []
        for index, member_state in enumerate(state.member_states):
            member_valid.append(self._member_state_valid(member_state))
            member_counts_match.append(
                member_state.step_count
                == (
                    state.member_update_counts[index]
                    + state.replay_member_update_counts[index]
                )
            )
        signal_cfg = self._config.signal_estimator
        signal_state = state.signal_state
        signal_bounds_valid = (
            (signal_state.calibration_count <= signal_cfg.change_calibration_steps)
            & (signal_state.calibration_count <= signal_state.valid_count)
            & (signal_state.calibration_mean <= signal_cfg.max_normalized_residual)
            & (signal_state.fast_loss_ema <= signal_cfg.max_observed_loss)
            & (signal_state.slow_loss_ema <= signal_cfg.max_observed_loss)
        )
        return (
            (state.event_count >= 0)
            & (state.event_count <= _INT32_MAX)
            & (state.replay_event_count >= 0)
            & (state.replay_event_count <= _INT32_MAX)
            & jnp.all(state.member_update_counts >= 0)
            & jnp.all(state.member_update_counts <= state.event_count)
            & jnp.all(state.replay_member_update_counts >= 0)
            & jnp.all(state.replay_member_update_counts <= state.replay_event_count)
            & jnp.all(
                state.member_update_counts
                <= jnp.asarray(_INT32_MAX, dtype=jnp.int32)
                - state.replay_member_update_counts
            )
            & jnp.all(jnp.stack(member_valid))
            & jnp.all(jnp.stack(member_counts_match))
            & jnp.all(jnp.isfinite(state.residual_variances))
            & jnp.all(state.residual_variances >= self._config.residual_variance_floor)
            & jnp.all(state.residual_variances <= signal_cfg.max_predicted_variance)
            & self._signal_state_valid(signal_state)
            & signal_bounds_valid
            & (signal_state.step_count == state.event_count)
        )

    def state_valid(self, state: WorldModelEnsembleState) -> Array:
        """Return the dynamic state-validity verdict after static validation."""
        self._validate_state_static_contract(state)
        return self._state_valid(state)

    def _zero_prediction(self) -> WorldModelEnsemblePrediction:
        ensemble_size = self._config.ensemble_size
        observation_dim = self._config.model.observation_dim
        target_dim = self._config.target_dim
        return WorldModelEnsemblePrediction(
            member_raw_predictions=jnp.zeros((ensemble_size, target_dim), dtype=jnp.float32),
            mean_raw_prediction=jnp.zeros((target_dim,), dtype=jnp.float32),
            member_next_observations=jnp.zeros((ensemble_size, observation_dim), dtype=jnp.float32),
            mean_next_observation=jnp.zeros((observation_dim,), dtype=jnp.float32),
            member_rewards=jnp.zeros((ensemble_size,), dtype=jnp.float32),
            mean_reward=jnp.asarray(0.0, dtype=jnp.float32),
            member_discounts=jnp.zeros((ensemble_size,), dtype=jnp.float32),
            mean_discount=jnp.asarray(0.0, dtype=jnp.float32),
            per_head_epistemic_variance=jnp.zeros((target_dim,), dtype=jnp.float32),
            epistemic_disagreement=jnp.asarray(0.0, dtype=jnp.float32),
            residual_variances=jnp.zeros((ensemble_size, target_dim), dtype=jnp.float32),
            residual_proxy_ready=jnp.asarray(False),
            valid=jnp.asarray(False),
        )

    def _predict_unchecked(
        self,
        state: WorldModelEnsembleState,
        observation: Array,
        action: Array,
    ) -> WorldModelEnsemblePrediction:
        predictions = [
            self._model.predict(member_state, observation, action)
            for member_state in state.member_states
        ]
        raw = jnp.stack([prediction.raw_predictions for prediction in predictions], axis=0)
        next_observations = jnp.stack(
            [prediction.next_observation for prediction in predictions], axis=0
        )
        rewards = jnp.stack([prediction.reward for prediction in predictions])
        discounts = jnp.stack([prediction.discount for prediction in predictions])
        mean_raw = _safe_mean(raw, axis=0)
        mean_next_observation = _safe_mean(next_observations, axis=0)
        mean_reward = _safe_mean(rewards)
        mean_discount = _safe_mean(discounts)
        per_head_epistemic = _safe_mean(jnp.square(raw - mean_raw[None, :]), axis=0)
        epistemic_disagreement = _safe_mean(per_head_epistemic)
        aggregates_finite = (
            jnp.all(jnp.isfinite(mean_raw))
            & jnp.all(jnp.isfinite(mean_next_observation))
            & jnp.isfinite(mean_reward)
            & jnp.isfinite(mean_discount)
            & jnp.all(jnp.isfinite(per_head_epistemic))
            & jnp.isfinite(epistemic_disagreement)
        )
        finite = (
            jnp.all(jnp.isfinite(raw))
            & jnp.all(jnp.isfinite(next_observations))
            & jnp.all(jnp.isfinite(rewards))
            & jnp.all(jnp.isfinite(discounts))
            & jnp.all(jnp.abs(raw) <= self._config.signal_estimator.max_input_magnitude)
            & aggregates_finite
        )
        return WorldModelEnsemblePrediction(
            member_raw_predictions=raw,
            mean_raw_prediction=mean_raw,
            member_next_observations=next_observations,
            mean_next_observation=mean_next_observation,
            member_rewards=rewards,
            mean_reward=mean_reward,
            member_discounts=discounts,
            mean_discount=mean_discount,
            per_head_epistemic_variance=per_head_epistemic,
            epistemic_disagreement=epistemic_disagreement,
            residual_variances=state.residual_variances,
            residual_proxy_ready=(state.event_count >= self._config.residual_variance_warmup_steps),
            valid=finite,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        state: WorldModelEnsembleState,
        observation: Array,
        action: Array,
    ) -> WorldModelEnsemblePrediction:
        """Return a fail-closed read-only ensemble prediction."""
        self._validate_state_static_contract(state)
        obs = _real_array(
            observation,
            (self._config.model.observation_dim,),
            name="observation",
        )
        act = _action_scalar(action, name="action")
        input_valid = (
            jnp.all(jnp.isfinite(obs))
            & jnp.all(jnp.abs(obs) <= self._config.signal_estimator.max_input_magnitude)
            & (act >= 0)
            & (act < self._config.model.n_actions)
        )
        valid = self._state_valid(state) & input_valid

        def do_predict(_: None) -> WorldModelEnsemblePrediction:
            result = self._predict_unchecked(state, obs, act)
            result_valid = result.valid & valid
            return cast(
                WorldModelEnsemblePrediction,
                jax.lax.cond(
                    result_valid,
                    lambda: dataclasses.replace(cast(Any, result), valid=result_valid),
                    self._zero_prediction,
                ),
            )

        return cast(
            WorldModelEnsemblePrediction,
            jax.lax.cond(
                valid,
                do_predict,
                lambda _: self._zero_prediction(),
                operand=None,
            ),
        )

    @staticmethod
    def _gate_signals(
        signals: TypedLearningSignals,
        *,
        available: Array,
        residual_proxy_ready: Array,
    ) -> TypedLearningSignals:
        false = jnp.asarray(False, dtype=jnp.bool_)
        model_uncertainty_available = (
            available
            & residual_proxy_ready
            & signals.availability.epistemic
            & signals.availability.aleatoric
            & signals.availability.normalized_residual
        )
        progress_available = available & signals.availability.learning_progress
        change_available = (
            available & residual_proxy_ready & signals.availability.change_probability
        )
        zero = jnp.asarray(0.0, dtype=jnp.float32)

        def gated(value: Array, flag: Array) -> Array:
            return jnp.where(flag, value, zero)

        return TypedLearningSignals(
            epistemic_disagreement=gated(
                signals.epistemic_disagreement, model_uncertainty_available
            ),
            epistemic_surprise=gated(signals.epistemic_surprise, model_uncertainty_available),
            aleatoric_uncertainty=gated(signals.aleatoric_uncertainty, model_uncertainty_available),
            normalized_residual=gated(signals.normalized_residual, model_uncertainty_available),
            learning_progress=gated(signals.learning_progress, progress_available),
            calibrated_residual_z=gated(signals.calibrated_residual_z, change_available),
            instantaneous_change_probability=gated(
                signals.instantaneous_change_probability, change_available
            ),
            change_probability=gated(signals.change_probability, change_available),
            availability=LearningSignalAvailability(
                input_valid=jnp.where(available, signals.availability.input_valid, false),
                epistemic=model_uncertainty_available,
                aleatoric=model_uncertainty_available,
                normalized_residual=model_uncertainty_available,
                learning_progress=progress_available,
                change_probability=change_available,
            ),
        )

    def _rejected_update_result(
        self,
        state: WorldModelEnsembleState,
        *,
        state_valid: Array,
        input_valid: Array,
        capacity_available: Array,
    ) -> WorldModelEnsembleUpdateResult:
        false = jnp.asarray(False, dtype=jnp.bool_)
        diagnostics = WorldModelEnsembleDiagnostics(
            state_valid=state_valid,
            input_valid=input_valid,
            capacity_available=capacity_available,
            predictions_valid=false,
            representation_gradient_valid=false,
            signals_valid=false,
            residual_update_valid=false,
            member_updates_valid=false,
            candidate_state_valid=false,
            applied=false,
            rejected=jnp.asarray(True),
        )
        return WorldModelEnsembleUpdateResult(
            state=state,
            prediction=self._zero_prediction(),
            signals=_zero_signals(),
            targets=jnp.zeros((self._config.target_dim,), dtype=jnp.float32),
            observed_loss=jnp.asarray(0.0, dtype=jnp.float32),
            member_prediction_losses=jnp.zeros((self._config.ensemble_size,), dtype=jnp.float32),
            representation_objective=jnp.asarray(0.0, dtype=jnp.float32),
            representation_gradient=jnp.zeros(
                (self._config.model.observation_dim,), dtype=jnp.float32
            ),
            representation_gradient_valid=false,
            bootstrap_mask=jnp.zeros((self._config.ensemble_size,), dtype=jnp.bool_),
            member_updates_applied=jnp.zeros((self._config.ensemble_size,), dtype=jnp.bool_),
            diagnostics=diagnostics,
        )

    def _rejected_replay_update_result(
        self,
        state: WorldModelEnsembleState,
        *,
        state_valid: Array,
        sample_available: Array,
        input_valid: Array,
        capacity_available: Array,
    ) -> WorldModelEnsembleReplayUpdateResult:
        false = jnp.asarray(False, dtype=jnp.bool_)
        diagnostics = WorldModelEnsembleReplayDiagnostics(
            state_valid=state_valid,
            sample_available=sample_available,
            input_valid=input_valid,
            capacity_available=capacity_available,
            predictions_valid=false,
            member_updates_valid=false,
            candidate_state_valid=false,
            calibration_unchanged=jnp.asarray(True),
            applied=false,
            rejected=sample_available,
        )
        return WorldModelEnsembleReplayUpdateResult(
            state=state,
            prediction=self._zero_prediction(),
            targets=jnp.zeros((self._config.target_dim,), dtype=jnp.float32),
            observed_loss=jnp.asarray(0.0, dtype=jnp.float32),
            member_prediction_losses=jnp.zeros(
                (self._config.ensemble_size,), dtype=jnp.float32
            ),
            bootstrap_mask=jnp.zeros((self._config.ensemble_size,), dtype=jnp.bool_),
            member_updates_applied=jnp.zeros(
                (self._config.ensemble_size,), dtype=jnp.bool_
            ),
            diagnostics=diagnostics,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def replay_update(
        self,
        state: WorldModelEnsembleState,
        observation: Array,
        action: Array,
        reward: Array,
        discount: Array,
        next_observation: Array,
        sample_available: Array,
    ) -> WorldModelEnsembleReplayUpdateResult:
        """Update member models from replay without observing fresh evidence.

        This method deliberately does not call ``LearningSignalEstimator`` and
        does not change ``signal_state``, ``residual_variances``, real
        ``event_count``, the real bootstrap key/mask, or real member-update
        counts.  Padding positions are explicit exact no-ops.  Available
        invalid samples and numeric candidate failures roll back the complete
        state, including the replay PRNG.
        """
        self._validate_state_static_contract(state)
        observation_dim = self._config.model.observation_dim
        obs = _real_array(observation, (observation_dim,), name="observation")
        act = _action_scalar(action, name="action")
        rew = _real_array(reward, (), name="reward")
        disc = _real_array(discount, (), name="discount")
        next_obs = _real_array(
            next_observation,
            (observation_dim,),
            name="next_observation",
        )
        available = jnp.asarray(sample_available)
        if available.shape != () or available.dtype != jnp.bool_:
            raise ValueError("sample_available must be a scalar with dtype bool")
        magnitude_bound = self._config.signal_estimator.max_input_magnitude
        input_valid = (
            jnp.all(jnp.isfinite(obs))
            & jnp.all(jnp.abs(obs) <= magnitude_bound)
            & (act >= 0)
            & (act < self._config.model.n_actions)
            & jnp.isfinite(rew)
            & (jnp.abs(rew) <= magnitude_bound)
            & jnp.isfinite(disc)
            & (disc >= 0.0)
            & (disc <= 1.0)
            & jnp.all(jnp.isfinite(next_obs))
            & jnp.all(jnp.abs(next_obs) <= magnitude_bound)
        )
        state_valid = self._state_valid(state)
        capacity_available = (state.replay_event_count < _INT32_MAX) & jnp.all(
            state.member_update_counts + state.replay_member_update_counts < _INT32_MAX
        )
        can_attempt = state_valid & available & input_valid & capacity_available

        def do_update(_: None) -> WorldModelEnsembleReplayUpdateResult:
            prediction = self._predict_unchecked(state, obs, act)
            targets = self._model.targets(obs, rew, disc, next_obs)
            residuals = prediction.member_raw_predictions - targets[None, :]
            squared_residuals = jnp.square(residuals)
            member_losses = _safe_mean(squared_residuals, axis=1)
            observed_loss = _safe_mean(member_losses)
            signal_cfg = self._config.signal_estimator
            predictions_valid = (
                prediction.valid
                & jnp.all(jnp.isfinite(targets))
                & jnp.all(jnp.abs(targets) <= signal_cfg.max_input_magnitude)
                & jnp.all(jnp.isfinite(squared_residuals))
                & jnp.all(squared_residuals <= signal_cfg.max_predicted_variance)
                & jnp.isfinite(observed_loss)
                & (observed_loss >= 0.0)
                & (observed_loss <= signal_cfg.max_observed_loss)
            )
            next_replay_key, mask_key = jr.split(state.replay_bootstrap_key)
            bootstrap_mask = jr.bernoulli(
                mask_key,
                p=self._config.bootstrap_probability,
                shape=(self._config.ensemble_size,),
            )
            candidate_members: list[ActionConditionedWorldModelState] = []
            candidate_members_valid: list[Array] = []
            for index, member_state in enumerate(state.member_states):
                update_result = self._model.update(
                    member_state,
                    obs,
                    act,
                    rew,
                    disc,
                    next_obs,
                )
                candidate = cast(
                    ActionConditionedWorldModelState,
                    jax.lax.cond(
                        bootstrap_mask[index],
                        lambda: update_result.state,
                        lambda: member_state,
                    ),
                )
                candidate_members.append(candidate)
                candidate_members_valid.append(self._member_state_valid(candidate))

            replay_counts = (
                state.replay_member_update_counts + bootstrap_mask.astype(jnp.int32)
            )
            total_counts = state.member_update_counts + replay_counts
            member_updates_valid = (
                jnp.all(jnp.stack(candidate_members_valid))
                & jnp.all(replay_counts >= state.replay_member_update_counts)
                & jnp.all(total_counts <= _INT32_MAX)
                & jnp.all(
                    jnp.stack(
                        [
                            candidate_members[index].step_count == total_counts[index]
                            for index in range(self._config.ensemble_size)
                        ]
                    )
                )
            )
            candidate_state = WorldModelEnsembleState(
                member_states=tuple(candidate_members),
                residual_variances=state.residual_variances,
                signal_state=state.signal_state,
                bootstrap_key=state.bootstrap_key,
                replay_bootstrap_key=next_replay_key,
                last_bootstrap_mask=state.last_bootstrap_mask,
                last_replay_bootstrap_mask=bootstrap_mask,
                member_update_counts=state.member_update_counts,
                replay_member_update_counts=replay_counts,
                event_count=state.event_count,
                replay_event_count=_saturating_int32_increment(state.replay_event_count),
            )
            candidate_state_valid = self._state_valid(candidate_state)
            applied = predictions_valid & member_updates_valid & candidate_state_valid
            next_state = cast(
                WorldModelEnsembleState,
                jax.lax.cond(applied, lambda: candidate_state, lambda: state),
            )
            diagnostics = WorldModelEnsembleReplayDiagnostics(
                state_valid=state_valid,
                sample_available=available,
                input_valid=input_valid,
                capacity_available=capacity_available,
                predictions_valid=predictions_valid,
                member_updates_valid=member_updates_valid,
                candidate_state_valid=candidate_state_valid,
                calibration_unchanged=jnp.asarray(True),
                applied=applied,
                rejected=~applied,
            )
            return WorldModelEnsembleReplayUpdateResult(
                state=next_state,
                prediction=cast(
                    WorldModelEnsemblePrediction,
                    jax.lax.cond(applied, lambda: prediction, self._zero_prediction),
                ),
                targets=jnp.where(applied, targets, jnp.zeros_like(targets)),
                observed_loss=jnp.where(applied, observed_loss, 0.0),
                member_prediction_losses=jnp.where(
                    applied, member_losses, jnp.zeros_like(member_losses)
                ),
                bootstrap_mask=bootstrap_mask,
                member_updates_applied=bootstrap_mask & applied,
                diagnostics=diagnostics,
            )

        return cast(
            WorldModelEnsembleReplayUpdateResult,
            jax.lax.cond(
                can_attempt,
                do_update,
                lambda _: self._rejected_replay_update_result(
                    state,
                    state_valid=state_valid,
                    sample_available=available,
                    input_valid=input_valid,
                    capacity_available=capacity_available,
                ),
                operand=None,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: WorldModelEnsembleState,
        observation: Array,
        action: Array,
        reward: Array,
        discount: Array,
        next_observation: Array,
    ) -> WorldModelEnsembleUpdateResult:
        """Atomically process one bounded predict-before-update event.

        The representation objective and gradient are formed from pre-update
        members before the signal estimator or any residual/model update.
        """
        self._validate_state_static_contract(state)
        observation_dim = self._config.model.observation_dim
        obs = _real_array(observation, (observation_dim,), name="observation")
        act = _action_scalar(action, name="action")
        rew = _real_array(reward, (), name="reward")
        disc = _real_array(discount, (), name="discount")
        next_obs = _real_array(
            next_observation,
            (observation_dim,),
            name="next_observation",
        )
        magnitude_bound = self._config.signal_estimator.max_input_magnitude
        input_valid = (
            jnp.all(jnp.isfinite(obs))
            & jnp.all(jnp.abs(obs) <= magnitude_bound)
            & (act >= 0)
            & (act < self._config.model.n_actions)
            & jnp.isfinite(rew)
            & (jnp.abs(rew) <= magnitude_bound)
            & jnp.isfinite(disc)
            & (disc >= 0.0)
            & (disc <= 1.0)
            & jnp.all(jnp.isfinite(next_obs))
            & jnp.all(jnp.abs(next_obs) <= magnitude_bound)
        )
        state_valid = self._state_valid(state)
        capacity_available = (state.event_count < _INT32_MAX) & jnp.all(
            state.member_update_counts + state.replay_member_update_counts < _INT32_MAX
        )
        can_attempt = state_valid & input_valid & capacity_available

        def do_update(_: None) -> WorldModelEnsembleUpdateResult:
            targets = self._model.targets(obs, rew, disc, next_obs)
            stopped_targets = jax.lax.stop_gradient(targets)

            def representation_loss(
                representation: Array,
            ) -> tuple[Array, WorldModelEnsemblePrediction]:
                preupdate_prediction = self._predict_unchecked(
                    state,
                    representation,
                    act,
                )
                residual = preupdate_prediction.member_raw_predictions - stopped_targets[None, :]
                return 0.5 * _safe_mean(jnp.square(residual)), preupdate_prediction

            (
                (representation_objective, prediction),
                representation_gradient,
            ) = jax.value_and_grad(representation_loss, has_aux=True)(obs)
            residuals = prediction.member_raw_predictions - stopped_targets[None, :]
            squared_residuals = jnp.square(residuals)
            member_losses = _safe_mean(squared_residuals, axis=1)
            observed_loss = _safe_mean(member_losses)
            signal_cfg = self._config.signal_estimator
            predictions_valid = (
                prediction.valid
                & jnp.all(jnp.isfinite(targets))
                & jnp.all(jnp.abs(targets) <= signal_cfg.max_input_magnitude)
                & jnp.all(jnp.isfinite(squared_residuals))
                & jnp.all(squared_residuals <= signal_cfg.max_predicted_variance)
                & jnp.isfinite(observed_loss)
                & (observed_loss >= 0.0)
                & (observed_loss <= signal_cfg.max_observed_loss)
            )
            representation_gradient_valid = (
                predictions_valid
                & jnp.isfinite(representation_objective)
                & (representation_objective >= 0.0)
                & jnp.all(jnp.isfinite(representation_gradient))
            )

            candidate_signal_state, raw_signals = self._signals.observe(
                state.signal_state,
                prediction.member_raw_predictions,
                state.residual_variances,
                targets,
                observed_loss,
            )
            signals_valid = raw_signals.availability.input_valid

            residual_floor = jnp.asarray(self._config.residual_variance_floor, dtype=jnp.float32)
            decay = jnp.asarray(self._config.residual_variance_decay, dtype=jnp.float32)
            candidate_residual_variances = jnp.where(
                state.event_count == 0,
                jnp.maximum(squared_residuals, residual_floor),
                jnp.maximum(
                    decay * state.residual_variances + (1.0 - decay) * squared_residuals,
                    residual_floor,
                ),
            )
            residual_update_valid = (
                jnp.all(jnp.isfinite(candidate_residual_variances))
                & jnp.all(candidate_residual_variances >= residual_floor)
                & jnp.all(candidate_residual_variances <= signal_cfg.max_predicted_variance)
            )

            next_bootstrap_key, mask_key = jr.split(state.bootstrap_key)
            bootstrap_mask = jr.bernoulli(
                mask_key,
                p=self._config.bootstrap_probability,
                shape=(self._config.ensemble_size,),
            )
            candidate_members: list[ActionConditionedWorldModelState] = []
            candidate_members_valid: list[Array] = []
            for index, member_state in enumerate(state.member_states):
                update_result = self._model.update(
                    member_state,
                    obs,
                    act,
                    rew,
                    disc,
                    next_obs,
                )
                candidate = cast(
                    ActionConditionedWorldModelState,
                    jax.lax.cond(
                        bootstrap_mask[index],
                        lambda: update_result.state,
                        lambda: member_state,
                    ),
                )
                candidate_members.append(candidate)
                candidate_members_valid.append(self._member_state_valid(candidate))

            member_update_counts = state.member_update_counts + bootstrap_mask.astype(jnp.int32)
            member_updates_valid = (
                jnp.all(jnp.stack(candidate_members_valid))
                & jnp.all(member_update_counts >= state.member_update_counts)
                & jnp.all(
                    member_update_counts
                    <= jnp.asarray(_INT32_MAX, dtype=jnp.int32)
                    - state.replay_member_update_counts
                )
                & jnp.all(
                    jnp.stack(
                        [
                            candidate_members[index].step_count
                            == (
                                member_update_counts[index]
                                + state.replay_member_update_counts[index]
                            )
                            for index in range(self._config.ensemble_size)
                        ]
                    )
                )
            )
            candidate_state = WorldModelEnsembleState(
                member_states=tuple(candidate_members),
                residual_variances=candidate_residual_variances,
                signal_state=candidate_signal_state,
                bootstrap_key=next_bootstrap_key,
                replay_bootstrap_key=state.replay_bootstrap_key,
                last_bootstrap_mask=bootstrap_mask,
                last_replay_bootstrap_mask=state.last_replay_bootstrap_mask,
                member_update_counts=member_update_counts,
                replay_member_update_counts=state.replay_member_update_counts,
                event_count=_saturating_int32_increment(state.event_count),
                replay_event_count=state.replay_event_count,
            )
            candidate_state_valid = self._state_valid(candidate_state)
            applied = (
                predictions_valid
                & representation_gradient_valid
                & signals_valid
                & residual_update_valid
                & member_updates_valid
                & candidate_state_valid
            )
            next_state = cast(
                WorldModelEnsembleState,
                jax.lax.cond(applied, lambda: candidate_state, lambda: state),
            )
            reported_signals = self._gate_signals(
                raw_signals,
                available=applied,
                residual_proxy_ready=prediction.residual_proxy_ready,
            )
            zero_target = jnp.zeros_like(targets)
            zero_losses = jnp.zeros_like(member_losses)
            diagnostics = WorldModelEnsembleDiagnostics(
                state_valid=state_valid,
                input_valid=input_valid,
                capacity_available=capacity_available,
                predictions_valid=predictions_valid,
                representation_gradient_valid=representation_gradient_valid,
                signals_valid=signals_valid,
                residual_update_valid=residual_update_valid,
                member_updates_valid=member_updates_valid,
                candidate_state_valid=candidate_state_valid,
                applied=applied,
                rejected=~applied,
            )
            return WorldModelEnsembleUpdateResult(
                state=next_state,
                prediction=cast(
                    WorldModelEnsemblePrediction,
                    jax.lax.cond(applied, lambda: prediction, self._zero_prediction),
                ),
                signals=reported_signals,
                targets=jnp.where(applied, targets, zero_target),
                observed_loss=jnp.where(applied, observed_loss, 0.0),
                member_prediction_losses=jnp.where(applied, member_losses, zero_losses),
                representation_objective=jnp.where(
                    applied,
                    representation_objective,
                    0.0,
                ),
                representation_gradient=jnp.where(
                    applied,
                    representation_gradient,
                    jnp.zeros_like(representation_gradient),
                ),
                representation_gradient_valid=(applied & representation_gradient_valid),
                bootstrap_mask=bootstrap_mask,
                member_updates_applied=bootstrap_mask & applied,
                diagnostics=diagnostics,
            )

        return cast(
            WorldModelEnsembleUpdateResult,
            jax.lax.cond(
                can_attempt,
                do_update,
                lambda _: self._rejected_update_result(
                    state,
                    state_valid=state_valid,
                    input_valid=input_valid,
                    capacity_available=capacity_available,
                ),
                operand=None,
            ),
        )


def _ensemble_config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _legacy_v1_template(state: WorldModelEnsembleState) -> _WorldModelEnsembleStateV1:
    return _WorldModelEnsembleStateV1(
        member_states=state.member_states,
        residual_variances=state.residual_variances,
        signal_state=state.signal_state,
        bootstrap_key=state.bootstrap_key,
        last_bootstrap_mask=state.last_bootstrap_mask,
        member_update_counts=state.member_update_counts,
        event_count=state.event_count,
    )


def _legacy_v1_resource_budget(
    ensemble: WorldModelEnsemble,
    current_template: WorldModelEnsembleState,
) -> dict[str, int]:
    """Reconstruct the exact resource payload written by the v1 saver."""
    legacy = _legacy_v1_template(current_template)
    current = ensemble.resource_budget(current_template)
    persistent = _logical_tree_accounting(legacy)
    prediction = _logical_tree_accounting(ensemble._zero_prediction())
    false = jnp.asarray(False, dtype=jnp.bool_)
    update = cast(
        Any,
        ensemble._rejected_update_result(
            current_template,
            state_valid=false,
            input_valid=false,
            capacity_available=false,
        ),
    ).replace(state=legacy)
    update_account = _logical_tree_accounting(update)
    key_data = jr.key_data(legacy.bootstrap_key)
    return {
        "ensemble_size": current.ensemble_size,
        "observation_dim": current.observation_dim,
        "target_dim": current.target_dim,
        "member_state_scalars_per_member": current.member_state_scalars_per_member,
        "member_state_bytes_per_member": current.member_state_bytes_per_member,
        "member_trainable_scalars": current.member_trainable_scalars,
        "total_trainable_scalars": current.total_trainable_scalars,
        "persistent_float32_scalars": persistent.float32_scalars,
        "persistent_float64_scalars": persistent.float64_scalars,
        "persistent_int32_scalars": persistent.int32_scalars,
        "persistent_int64_scalars": persistent.int64_scalars,
        "persistent_uint32_scalars": persistent.uint32_scalars,
        "persistent_bool_scalars": persistent.bool_scalars,
        "persistent_state_scalars": persistent.logical_scalars,
        "persistent_state_bytes": persistent.logical_bytes,
        "bootstrap_prng_keys": 1,
        "bootstrap_prng_uint32_scalars": int(key_data.size),
        "bootstrap_prng_bytes": int(key_data.nbytes),
        "prediction_output_logical_scalars": prediction.logical_scalars,
        "prediction_output_logical_bytes": prediction.logical_bytes,
        "update_result_output_logical_scalars": update_account.logical_scalars,
        "update_result_output_logical_bytes": update_account.logical_bytes,
        "member_update_candidates_per_valid_event": (
            current.member_update_candidates_per_valid_event
        ),
        "max_member_updates_per_event": current.max_member_updates_per_event,
        "max_event_count": current.max_event_count,
        "max_member_update_count": current.max_member_update_count,
    }


def _migrate_v1_state(legacy: _WorldModelEnsembleStateV1) -> WorldModelEnsembleState:
    return WorldModelEnsembleState(
        member_states=legacy.member_states,
        residual_variances=legacy.residual_variances,
        signal_state=legacy.signal_state,
        bootstrap_key=legacy.bootstrap_key,
        replay_bootstrap_key=jr.fold_in(
            legacy.bootstrap_key,
            _V1_REPLAY_KEY_FOLD_IN,
        ),
        last_bootstrap_mask=legacy.last_bootstrap_mask,
        last_replay_bootstrap_mask=jnp.zeros_like(
            legacy.last_bootstrap_mask,
            dtype=jnp.bool_,
        ),
        member_update_counts=legacy.member_update_counts,
        replay_member_update_counts=jnp.zeros_like(
            legacy.member_update_counts,
            dtype=jnp.int32,
        ),
        event_count=legacy.event_count,
        replay_event_count=jnp.asarray(0, dtype=jnp.int32),
    )


def save_world_model_ensemble_checkpoint(
    ensemble: WorldModelEnsemble,
    state: WorldModelEnsembleState,
    path: str | Path,
) -> None:
    """Persist ensemble construction, complete state, masks, and RNG."""
    if not bool(ensemble.state_valid(state)):
        raise ValueError("cannot save an invalid WorldModelEnsemble state")
    config = ensemble.to_config()
    resource_budget = ensemble.resource_budget(state).to_config()
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA,
            "ensemble_config": config,
            "config_sha256": _ensemble_config_digest(config),
            "resource_budget": resource_budget,
        },
    )


def load_world_model_ensemble_checkpoint(
    path: str | Path,
    *,
    template_key: Array | None = None,
) -> tuple[WorldModelEnsemble, WorldModelEnsembleState]:
    """Restore v2 or strictly migrate a pre-rehearsal v1 ensemble.

    Migration preserves every v1 member, residual, causal-signal, real-key,
    real-mask, and real-counter field exactly.  Only the replay lane is added:
    its key is deterministically folded from the stored real key and all replay
    masks/counters start at zero.  New saves always use the v2 schema.
    """
    metadata = load_checkpoint_metadata(path)
    schema = metadata.get("schema")
    if schema not in {
        WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA,
        _WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA_V1,
    }:
        raise ValueError("checkpoint is not a WorldModelEnsemble v1/v2 checkpoint")
    config = metadata.get("ensemble_config")
    if not isinstance(config, dict):
        raise ValueError("ensemble checkpoint is missing ensemble_config")
    digest = metadata.get("config_sha256")
    if not isinstance(digest, str) or digest != _ensemble_config_digest(config):
        raise ValueError("ensemble checkpoint config digest does not match")
    ensemble = WorldModelEnsemble.from_config(config)
    if ensemble.to_config() != config:
        raise ValueError("ensemble checkpoint config is not canonical")
    key = jr.key(0) if template_key is None else template_key
    template = ensemble.init(key)
    if schema == WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA:
        expected_budget = ensemble.resource_budget(template).to_config()
        restore_template: WorldModelEnsembleState | _WorldModelEnsembleStateV1 = template
    else:
        expected_budget = _legacy_v1_resource_budget(ensemble, template)
        restore_template = _legacy_v1_template(template)
    if metadata.get("resource_budget") != expected_budget:
        raise ValueError("ensemble checkpoint resource budget does not match config")
    restored, restored_metadata = load_checkpoint(restore_template, path)
    if restored_metadata != metadata:
        raise ValueError("ensemble checkpoint metadata changed between reads")
    if schema == _WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA_V1:
        state = _migrate_v1_state(cast(_WorldModelEnsembleStateV1, restored))
    else:
        state = cast(WorldModelEnsembleState, restored)
    if not bool(ensemble.state_valid(state)):
        raise ValueError("restored WorldModelEnsemble state is invalid")
    if (
        schema == WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA
        and ensemble.resource_budget(state).to_config() != expected_budget
    ):
        raise ValueError("restored WorldModelEnsemble state resource budget is invalid")
    return ensemble, state


__all__ = [
    "WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA",
    "WorldModelEnsemble",
    "WorldModelEnsembleConfig",
    "WorldModelEnsembleDiagnostics",
    "WorldModelEnsemblePrediction",
    "WorldModelEnsembleReplayDiagnostics",
    "WorldModelEnsembleReplayUpdateResult",
    "WorldModelEnsembleResourceBudget",
    "WorldModelEnsembleState",
    "WorldModelEnsembleUpdateResult",
    "load_world_model_ensemble_checkpoint",
    "save_world_model_ensemble_checkpoint",
]
