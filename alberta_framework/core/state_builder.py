# mypy: disable-error-code="call-arg,name-defined"
r"""Causal state-construction contracts and small reference implementations.

The existing Alberta components expose several useful forms of history:
``HistoryFeatureExtractor`` and ``WorkingMemoryFeaturizer`` provide fixed
traces, while ``PrototypeAgent`` optionally uses a fixed-weight echo-state
GRU.  This module gives those design points a narrow common contract and adds
one genuinely trainable recurrent reference.

The trainable reference is deliberately modest.  It is a diagonal bank of
write/hold units,

.. math::

    g_t &= \sigma(W_g u_t + b_g) \\
    c_t &= \tanh(W_c u_t + b_c) \\
    h_t &= (1-g_t) h_{t-1} + g_t c_t ,

where ``u_t`` contains the current observation and the preceding transition's
action, reward, and discount.  It carries an RTRL-style online eligibility
matrix (real-time recurrent learning; Williams & Zipser 1989).  With
parameters held fixed, that matrix is the exact unrolled ``dh_t / dtheta``;
after an online parameter update, carrying it forward is the
usual changing-parameter eligibility approximation.  A downstream prediction
or control head can therefore pass the gradient of its loss with respect to
the emitted state to :meth:`OnlineGatedStateBuilder.learn` without replay or a
backward sweep through stored experience.

This is a learnable recurrent baseline, not a general state-discovery result:

* units do not interact recurrently with one another;
* the caller must supply useful online auxiliary/control gradients;
* carried sensitivities are a fixed, potentially substantial memory cost; and
* there is no generate-and-test or feature-recycling mechanism here.

All builders are pure PyTree transformations, have fixed output/state budgets,
serialize their configuration, and round-trip through Alberta's generic Orbax
checkpoint utilities.
"""

from __future__ import annotations

import functools
import hashlib
import json
import operator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol, SupportsIndex, TypeVar, cast, runtime_checkable

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core._float32_scalars import validated_float32_scalar
from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.update_safety import safe_discrete_action, select_transaction
from alberta_framework.core.working_memory import (
    WorkingMemoryConfig,
    WorkingMemoryFeaturizer,
    WorkingMemoryState,
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


def _require_integer(
    name: str,
    value: object,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer")
    canonical = operator.index(cast(SupportsIndex, value))
    if canonical < minimum or (maximum is not None and canonical > maximum):
        domain = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be an integer in {domain}")
    return canonical


def _require_int32(name: str, value: object, *, minimum: int) -> int:
    return _require_integer(name, value, minimum=minimum, maximum=_INT32_MAX)


def _require_decay_rates(name: str, value: object) -> tuple[float, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be an actual tuple")
    rates = cast(tuple[object, ...], value)
    return tuple(
        validated_float32_scalar(
            f"{name}[{index}]",
            rate,
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
        )
        for index, rate in enumerate(rates)
    )


_FLOAT32_MAX = 3.4028234663852886e38


def _saturating_int32_increment(value: Array) -> Array:
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    counter = jnp.asarray(value, dtype=jnp.int32)
    return jnp.minimum(jnp.maximum(counter, 0), maximum - 1) + 1


StateT = TypeVar("StateT")

STATE_BUILDER_CHECKPOINT_SCHEMA = "alberta.state_builder.v2"


@dataclass(frozen=True)
class StateBuilderBudget:
    """Exact history-independent resource counts for a state builder.

    ``state_scalars`` counts every scalar carried in the builder state,
    including parameters and integer counters.  ``state_bytes`` assumes the
    implementations in this module: float32 and int32 arrays throughout.
    It excludes transient compiler buffers and a downstream learning head.
    """

    output_scalars: int
    trainable_scalars: int
    state_scalars: int
    state_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_scalars",
            _require_integer("output_scalars", self.output_scalars, minimum=0),
        )
        object.__setattr__(
            self,
            "trainable_scalars",
            _require_integer("trainable_scalars", self.trainable_scalars, minimum=0),
        )
        object.__setattr__(
            self,
            "state_scalars",
            _require_integer("state_scalars", self.state_scalars, minimum=0),
        )
        object.__setattr__(
            self,
            "state_bytes",
            _require_integer("state_bytes", self.state_bytes, minimum=0),
        )

    def to_config(self) -> dict[str, int]:
        """Return a JSON-compatible budget description."""
        return asdict(self)


@chex.dataclass(frozen=True)
class StateBuilderLearningDiagnostics:
    """Diagnostics from one representation-learning update."""

    gradient_norm: Float[Array, ""]
    clipped_gradient_norm: Float[Array, ""]
    parameter_update_norm: Float[Array, ""]
    proposal_valid: Bool[Array, ""]
    source_matches: Bool[Array, ""]
    capacity_available: Bool[Array, ""]
    candidate_parameters_valid: Bool[Array, ""]
    applied: Bool[Array, ""]
    fixed_noop: Bool[Array, ""]
    valid: Bool[Array, ""]
    rejected: Bool[Array, ""]


@chex.dataclass(frozen=True)
class StateBuilderLearningProposal:
    """Pure, source-bound proposal for one causal representation update.

    ``source_parameters`` and ``source_update_count`` bind the proposal to the
    exact parameter version that produced it. ``candidate_parameter_update``
    is already clipped, step-size-scaled, and ready for a later atomic commit.
    A downstream safety layer may replace that vector only through
    :func:`replace_state_builder_learning_proposal_update`, which preserves the
    source binding and recomputes all candidate diagnostics.
    """

    builder_fingerprint: UInt[Array, " 8"]
    source_parameters: Float[Array, " parameter_count"]
    source_update_count: Int[Array, ""]
    raw_parameter_gradient: Float[Array, " parameter_count"]
    clipped_parameter_gradient: Float[Array, " parameter_count"]
    candidate_parameter_update: Float[Array, " parameter_count"]
    gradient_norm: Float[Array, ""]
    clipped_gradient_norm: Float[Array, ""]
    parameter_update_norm: Float[Array, ""]
    source_state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    raw_parameter_gradient_valid: Bool[Array, ""]
    clipped_parameter_gradient_valid: Bool[Array, ""]
    candidate_parameter_update_valid: Bool[Array, ""]
    candidate_parameters_valid: Bool[Array, ""]
    capacity_available: Bool[Array, ""]
    candidate_update_transformed: Bool[Array, ""]
    candidate_update_approved: Bool[Array, ""]
    fixed_noop: Bool[Array, ""]
    valid: Bool[Array, ""]
    rejected: Bool[Array, ""]


@runtime_checkable
class StateBuilder(Protocol[StateT]):
    """Minimal causal state-construction contract.

    ``start`` consumes the first observation and optional preceding-transition
    values.  Thereafter, ``update`` consumes ``(current_observation,
    previous_action, previous_reward, previous_discount)`` and advances state
    exactly once.  The action and outcomes must be from the transition that
    produced the current observation; passing a current action before selecting
    it would leak future information.  Both methods return the representation
    associated with the supplied current observation.

    ``encode`` is pure: it pairs a supplied raw observation with the recurrent
    memory already present in ``state`` and never advances history.  For
    history-dependent builders it is valid for the most recently consumed
    observation (or as an explicitly counterfactual raw-observation query);
    callers must not mistake it for a second recurrent transition.

    ``learn`` accepts ``d(loss) / d(representation)`` from any downstream head.
    Builders without trainable representation parameters implement it as a
    no-op.  Separating ``update`` and ``learn`` prevents target information
    from entering the emitted state before its prediction is scored.
    """

    def init(self, key: Array) -> StateT:
        """Return a fresh builder state."""
        ...

    def start(
        self,
        state: StateT,
        raw_observation: Array,
        last_action: Array | int = -1,
        last_reward: Array | float = 0.0,
        last_discount: Array | float = 1.0,
    ) -> tuple[StateT, Array]:
        """Consume the first observation and emit its representation."""
        ...

    def reset_episode(self, state: StateT) -> StateT:
        """Clear episode-local memory without resetting lifetime state.

        Learned parameters, lifetime learning counters, and the monotonic
        observation-event counter are preserved.  The next :meth:`start` call
        consumes the first observation of the new episode and increments that
        counter exactly once.
        """
        ...

    def encode(self, state: StateT, raw_observation: Array) -> Array:
        """Emit a representation without advancing state."""
        ...

    def update(
        self,
        state: StateT,
        raw_observation: Array,
        previous_action: Array | int,
        previous_reward: Array | float,
        previous_discount: Array | float,
    ) -> tuple[StateT, Array]:
        """Consume one continuing transition and emit the new representation."""
        ...

    def learn(
        self,
        state: StateT,
        representation_gradient: Array,
    ) -> tuple[StateT, StateBuilderLearningDiagnostics]:
        """Apply an online representation gradient."""
        ...

    def propose_learning_update(
        self,
        source_state: StateT,
        dL_drepresentation: Array,  # noqa: N803 - mathematical dL notation
    ) -> StateBuilderLearningProposal:
        """Form a pure parameter update bound to ``source_state``."""
        ...

    def commit_learning_update(
        self,
        destination_state: StateT,
        proposal: StateBuilderLearningProposal,
    ) -> tuple[StateT, StateBuilderLearningDiagnostics]:
        """Atomically apply a still-current proposal to ``destination_state``."""
        ...

    def feature_dim(self) -> int:
        """Return the fixed representation dimension."""
        ...

    def observation_dim(self) -> int:
        """Return the raw observation dimension consumed by the builder."""
        ...

    def resource_budget(self) -> StateBuilderBudget:
        """Return exact persistent-state and trainable-parameter counts."""
        ...

    def state_valid(self, state: StateT) -> Bool[Array, ""]:
        """Validate the complete dynamic state contract."""
        ...

    def to_config(self) -> dict[str, Any]:
        """Return a JSON-compatible builder configuration."""
        ...


def _zero_learning_diagnostics(
    *,
    valid: Array | bool = True,
    proposal_valid: Array | bool | None = None,
    source_matches: Array | bool = True,
    capacity_available: Array | bool = True,
    candidate_parameters_valid: Array | bool = True,
    fixed_noop: Array | bool = True,
) -> StateBuilderLearningDiagnostics:
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    valid_array = jnp.asarray(valid, dtype=jnp.bool_)
    proposal_valid_array = (
        valid_array if proposal_valid is None else jnp.asarray(proposal_valid, dtype=jnp.bool_)
    )
    return StateBuilderLearningDiagnostics(
        gradient_norm=zero,
        clipped_gradient_norm=zero,
        parameter_update_norm=zero,
        proposal_valid=proposal_valid_array,
        source_matches=jnp.asarray(source_matches, dtype=jnp.bool_),
        capacity_available=jnp.asarray(capacity_available, dtype=jnp.bool_),
        candidate_parameters_valid=jnp.asarray(
            candidate_parameters_valid,
            dtype=jnp.bool_,
        ),
        applied=jnp.asarray(False),
        fixed_noop=jnp.asarray(fixed_noop, dtype=jnp.bool_),
        valid=valid_array,
        rejected=~valid_array,
    )


def _static_array_contract(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> None:
    """Validate shape and dtype using metadata available during JAX tracing."""
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must be an array with static shape and dtype metadata")
    actual_shape = tuple(value.shape)
    if actual_shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {actual_shape}")
    expected_dtype = jnp.dtype(dtype)
    actual_dtype = jnp.dtype(value.dtype)
    if actual_dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}; got {actual_dtype}")


def _scale_safe_l2_norm(values: Array) -> Float[Array, ""]:
    """Return a finite, saturating float32 L2 norm without squaring overflow."""
    vector = jnp.asarray(values, dtype=jnp.float32).reshape((-1,))
    maximum = jnp.max(jnp.abs(vector), initial=jnp.asarray(0.0, dtype=jnp.float32))
    scaled = _divide_by_positive_scale(vector, maximum)
    scaled_norm = jnp.sqrt(jnp.sum(scaled * scaled))
    safe_scaled_norm = jnp.where(scaled_norm > 0.0, scaled_norm, 1.0)
    float32_max = jnp.asarray(_FLOAT32_MAX, dtype=jnp.float32)
    overflow = maximum > float32_max / safe_scaled_norm
    product = maximum * scaled_norm
    return jnp.where(maximum == 0.0, 0.0, jnp.where(overflow, float32_max, product))


def _divide_by_positive_scale(values: Array, scale: Array) -> Array:
    """Divide by a positive float32 scale without reciprocal underflow."""
    mantissa, exponent = jnp.frexp(scale)
    safe_mantissa = jnp.where(scale > 0.0, mantissa, 1.0)
    scaled = jnp.ldexp(values, -exponent) / safe_mantissa
    return jnp.where(scale > 0.0, scaled, 0.0)


def _scale_safe_clip_by_l2_norm(
    values: Array,
    clip: Array,
) -> tuple[Array, Float[Array, ""]]:
    """Clip a finite float32 vector without forming an overflowing norm."""
    vector = jnp.asarray(values, dtype=jnp.float32)
    flat = vector.reshape((-1,))
    maximum = jnp.max(jnp.abs(flat), initial=jnp.asarray(0.0, dtype=jnp.float32))
    scaled = _divide_by_positive_scale(vector, maximum)
    scaled_norm = jnp.sqrt(jnp.sum(scaled.reshape((-1,)) ** 2))
    safe_scaled_norm = jnp.where(scaled_norm > 0.0, scaled_norm, 1.0)
    within_limit = maximum <= clip / safe_scaled_norm
    clipped = scaled * (clip / safe_scaled_norm)
    result = jnp.where(within_limit, vector, clipped)
    return result, _scale_safe_l2_norm(vector)


def _builder_learning_fingerprint(config: dict[str, Any]) -> UInt[Array, " 8"]:
    """Return a compact, deterministic binding for one exact builder config."""
    canonical = json.dumps(
        config,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).digest()
    return jnp.asarray(
        [int.from_bytes(digest[offset : offset + 4], "big") for offset in range(0, 32, 4)],
        dtype=jnp.uint32,
    )


def _validate_learning_proposal_static_contract(
    proposal: StateBuilderLearningProposal,
    parameter_count: int,
) -> None:
    """Reject structural proposal corruption before eager or compiled execution."""
    if not isinstance(proposal, StateBuilderLearningProposal):
        raise TypeError("proposal must be a StateBuilderLearningProposal")
    _static_array_contract(
        proposal.builder_fingerprint,
        name="proposal.builder_fingerprint",
        shape=(8,),
        dtype=jnp.uint32,
    )
    for name in (
        "source_parameters",
        "raw_parameter_gradient",
        "clipped_parameter_gradient",
        "candidate_parameter_update",
    ):
        _static_array_contract(
            getattr(proposal, name),
            name=f"proposal.{name}",
            shape=(parameter_count,),
            dtype=jnp.float32,
        )
    _static_array_contract(
        proposal.source_update_count,
        name="proposal.source_update_count",
        shape=(),
        dtype=jnp.int32,
    )
    for name in (
        "gradient_norm",
        "clipped_gradient_norm",
        "parameter_update_norm",
    ):
        _static_array_contract(
            getattr(proposal, name),
            name=f"proposal.{name}",
            shape=(),
            dtype=jnp.float32,
        )
    for name in (
        "source_state_valid",
        "input_valid",
        "raw_parameter_gradient_valid",
        "clipped_parameter_gradient_valid",
        "candidate_parameter_update_valid",
        "candidate_parameters_valid",
        "capacity_available",
        "candidate_update_transformed",
        "candidate_update_approved",
        "fixed_noop",
        "valid",
        "rejected",
    ):
        _static_array_contract(
            getattr(proposal, name),
            name=f"proposal.{name}",
            shape=(),
            dtype=jnp.bool_,
        )


def _finite_or_max_norm(values: Array, valid: Array) -> Float[Array, ""]:
    """Return a finite norm, conservatively saturating invalid vectors."""
    safe_values = jnp.where(valid, values, 0.0)
    return jnp.where(
        valid,
        _scale_safe_l2_norm(safe_values),
        jnp.asarray(_FLOAT32_MAX, dtype=jnp.float32),
    )


def _float32_vectors_bitwise_equal(left: Array, right: Array) -> Bool[Array, ""]:
    """Compare exact float32 encodings, including signed zero payloads."""
    left_bits = jax.lax.bitcast_convert_type(left, jnp.uint32)
    right_bits = jax.lax.bitcast_convert_type(right, jnp.uint32)
    return jnp.all(left_bits == right_bits)


def _diagnostic_scalar_matches(actual: Array, expected: Array) -> Bool[Array, ""]:
    """Allow only backend-rounding tolerance in recomputed finite diagnostics."""
    exact = actual == expected
    rounded = (expected > 0.0) & jnp.isclose(actual, expected, rtol=1.0e-6, atol=0.0)
    return jnp.isfinite(actual) & (actual >= 0.0) & (exact | rounded)


def replace_state_builder_learning_proposal_update(
    proposal: StateBuilderLearningProposal,
    candidate_parameter_update: Array,
    approved: Array,
) -> StateBuilderLearningProposal:
    """Replace only a valid proposal's candidate update and refresh its checks.

    This is the supported boundary for a downstream update filter.  It keeps
    the source binding and raw/clipped gradients immutable, requires the exact
    float32 vector contract, requires an exact scalar-bool approval decision,
    and cannot turn an already-rejected proposal into an accepted one.  A
    false approval is an explicit veto even when the replacement vector is
    zero.
    """
    if not isinstance(proposal, StateBuilderLearningProposal):
        raise TypeError("proposal must be a StateBuilderLearningProposal")
    if not hasattr(proposal.source_parameters, "shape"):
        raise TypeError("proposal.source_parameters must expose static shape metadata")
    source_shape = tuple(proposal.source_parameters.shape)
    if len(source_shape) != 1:
        raise ValueError(
            f"proposal.source_parameters must be a rank-one parameter vector; got {source_shape}"
        )
    parameter_count = source_shape[0]
    _validate_learning_proposal_static_contract(proposal, parameter_count)
    _static_array_contract(
        candidate_parameter_update,
        name="candidate_parameter_update",
        shape=(parameter_count,),
        dtype=jnp.float32,
    )
    _static_array_contract(
        approved,
        name="approved",
        shape=(),
        dtype=jnp.bool_,
    )

    update = jnp.asarray(candidate_parameter_update, dtype=jnp.float32)
    approved_array = jnp.asarray(approved, dtype=jnp.bool_)
    update_valid = jnp.all(jnp.isfinite(update))
    candidate_parameters = proposal.source_parameters + update
    candidate_parameters_valid = jnp.all(jnp.isfinite(candidate_parameters))
    parameter_update_norm = _finite_or_max_norm(update, update_valid)
    valid = (
        proposal.valid
        & ~proposal.fixed_noop
        & approved_array
        & update_valid
        & candidate_parameters_valid
    )
    return cast(
        StateBuilderLearningProposal,
        replace(
            cast(Any, proposal),
            candidate_parameter_update=update,
            parameter_update_norm=parameter_update_norm,
            candidate_parameter_update_valid=update_valid,
            candidate_parameters_valid=candidate_parameters_valid,
            candidate_update_transformed=jnp.asarray(True),
            candidate_update_approved=approved_array,
            valid=valid,
            rejected=~valid,
        ),
    )


def _action_features(action: Array | int, n_actions: int) -> Array:
    if n_actions == 0:
        return jnp.zeros((0,), dtype=jnp.float32)
    action_id = jnp.asarray(action, dtype=jnp.int32)
    return jax.nn.one_hot(action_id, n_actions, dtype=jnp.float32)


def _fixed_learning_proposal(
    builder_fingerprint: Array,
    source_state_valid: Array,
    input_valid: Array,
) -> StateBuilderLearningProposal:
    """Build an honest empty proposal for a non-trainable representation."""
    empty = jnp.zeros((0,), dtype=jnp.float32)
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    true = jnp.asarray(True)
    false = jnp.asarray(False)
    valid = source_state_valid & input_valid
    return StateBuilderLearningProposal(
        builder_fingerprint=jnp.asarray(builder_fingerprint, dtype=jnp.uint32),
        source_parameters=empty,
        source_update_count=jnp.asarray(0, dtype=jnp.int32),
        raw_parameter_gradient=empty,
        clipped_parameter_gradient=empty,
        candidate_parameter_update=empty,
        gradient_norm=zero,
        clipped_gradient_norm=zero,
        parameter_update_norm=zero,
        source_state_valid=source_state_valid,
        input_valid=input_valid,
        raw_parameter_gradient_valid=true,
        clipped_parameter_gradient_valid=true,
        candidate_parameter_update_valid=true,
        candidate_parameters_valid=true,
        capacity_available=true,
        candidate_update_transformed=false,
        candidate_update_approved=true,
        fixed_noop=true,
        valid=valid,
        rejected=~valid,
    )


def _fixed_learning_proposal_integrity(proposal: StateBuilderLearningProposal) -> Array:
    """Validate all derived fields of an empty fixed-builder proposal."""
    expected_valid = proposal.source_state_valid & proposal.input_valid
    empty_vectors = (
        (proposal.source_parameters.size == 0)
        & (proposal.raw_parameter_gradient.size == 0)
        & (proposal.clipped_parameter_gradient.size == 0)
        & (proposal.candidate_parameter_update.size == 0)
    )
    return (
        empty_vectors
        & (proposal.source_update_count == 0)
        & (proposal.gradient_norm == 0.0)
        & (proposal.clipped_gradient_norm == 0.0)
        & (proposal.parameter_update_norm == 0.0)
        & proposal.raw_parameter_gradient_valid
        & proposal.clipped_parameter_gradient_valid
        & proposal.candidate_parameter_update_valid
        & proposal.candidate_parameters_valid
        & proposal.capacity_available
        & ~proposal.candidate_update_transformed
        & proposal.candidate_update_approved
        & proposal.fixed_noop
        & (proposal.valid == expected_valid)
        & (proposal.rejected == ~expected_valid)
    )


def _fixed_learning_commit_diagnostics(
    proposal: StateBuilderLearningProposal,
    builder_fingerprint: Array,
    destination_state_valid: Array,
) -> StateBuilderLearningDiagnostics:
    """Return acceptance diagnostics for an atomic fixed-builder no-op."""
    source_matches = jnp.array_equal(
        proposal.builder_fingerprint,
        builder_fingerprint,
    )
    proposal_valid = _fixed_learning_proposal_integrity(proposal) & proposal.valid
    valid = destination_state_valid & source_matches & proposal_valid
    return _zero_learning_diagnostics(
        valid=valid,
        proposal_valid=proposal_valid,
        source_matches=source_matches,
        capacity_available=proposal.capacity_available,
        candidate_parameters_valid=proposal.candidate_parameters_valid,
        fixed_noop=True,
    )


@dataclass(frozen=True)
class IdentityStateBuilderConfig:
    """Configuration for the observation-only state baseline."""

    observation_dim: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_dim",
            _require_int32("observation_dim", self.observation_dim, minimum=1),
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "type": "IdentityStateBuilder",
            "observation_dim": self.observation_dim,
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> IdentityStateBuilderConfig:
        """Reconstruct from :meth:`to_config` output."""
        data = dict(payload)
        data.pop("type", None)
        return cls(observation_dim=data["observation_dim"])


@chex.dataclass(frozen=True)
class IdentityStateBuilderState:
    """The observation-only baseline carries only a continuing step counter."""

    step_count: Array


class IdentityStateBuilder:
    """Observation-only state builder and lower memory control."""

    def __init__(self, config: IdentityStateBuilderConfig):
        self._config = config
        self._learning_fingerprint = _builder_learning_fingerprint(config.to_config())

    @property
    def config(self) -> IdentityStateBuilderConfig:
        """Return the immutable configuration."""
        return self._config

    def init(self, key: Array) -> IdentityStateBuilderState:
        """Return a fresh state; ``key`` is accepted for protocol parity."""
        del key
        return IdentityStateBuilderState(step_count=jnp.asarray(0, dtype=jnp.int32))

    def feature_dim(self) -> int:
        """Return the raw observation dimension."""
        return self._config.observation_dim

    def observation_dim(self) -> int:
        """Return the raw observation dimension."""
        return self._config.observation_dim

    def resource_budget(self) -> StateBuilderBudget:
        """Return the one-counter persistent budget."""
        return StateBuilderBudget(
            output_scalars=self.feature_dim(),
            trainable_scalars=0,
            state_scalars=1,
            state_bytes=4,
        )

    def state_valid(self, state: IdentityStateBuilderState) -> Bool[Array, ""]:
        """Validate the exact identity-builder state contract."""
        self._validate_state_static_contract(state)
        return state.step_count >= 0

    def to_config(self) -> dict[str, Any]:
        """Serialize the builder configuration."""
        return self._config.to_config()

    @functools.partial(jax.jit, static_argnums=(0,))
    def encode(
        self,
        state: IdentityStateBuilderState,
        raw_observation: Array,
    ) -> Float[Array, " observation_dim"]:
        """Return the raw observation without touching ``state``."""
        del state
        return jnp.asarray(raw_observation, dtype=jnp.float32).reshape(
            (self._config.observation_dim,)
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: IdentityStateBuilderState,
        raw_observation: Array,
        previous_action: Array | int,
        previous_reward: Array | float,
        previous_discount: Array | float,
    ) -> tuple[IdentityStateBuilderState, Float[Array, " observation_dim"]]:
        """Advance the counter and emit the raw observation."""
        del previous_action, previous_reward, previous_discount
        features = self.encode(state, raw_observation)
        next_state = IdentityStateBuilderState(
            step_count=_saturating_int32_increment(state.step_count)
        )
        return next_state, features

    def start(
        self,
        state: IdentityStateBuilderState,
        raw_observation: Array,
        last_action: Array | int = -1,
        last_reward: Array | float = 0.0,
        last_discount: Array | float = 1.0,
    ) -> tuple[IdentityStateBuilderState, Array]:
        """Consume the first observation."""
        return cast(
            tuple[IdentityStateBuilderState, Array],
            self.update(
                state,
                raw_observation,
                last_action,
                last_reward,
                last_discount,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset_episode(
        self,
        state: IdentityStateBuilderState,
    ) -> IdentityStateBuilderState:
        """Preserve the lifetime event counter; there is no recurrent memory."""
        return state

    def _validate_state_static_contract(self, state: IdentityStateBuilderState) -> None:
        if not isinstance(state, IdentityStateBuilderState):
            raise TypeError("state must be an IdentityStateBuilderState")
        _static_array_contract(
            state.step_count,
            name="state.step_count",
            shape=(),
            dtype=jnp.int32,
        )

    def _validate_gradient_static_contract(self, representation_gradient: Array) -> None:
        _static_array_contract(
            representation_gradient,
            name="representation_gradient",
            shape=(self.feature_dim(),),
            dtype=jnp.float32,
        )

    def propose_learning_update(
        self,
        source_state: IdentityStateBuilderState,
        dL_drepresentation: Array,  # noqa: N803 - mathematical dL notation
    ) -> StateBuilderLearningProposal:
        """Return an honest source-checked no-op proposal."""
        self._validate_state_static_contract(source_state)
        self._validate_gradient_static_contract(dL_drepresentation)
        return cast(
            StateBuilderLearningProposal,
            self._propose_learning_update_jit(source_state, dL_drepresentation),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _propose_learning_update_jit(
        self,
        source_state: IdentityStateBuilderState,
        representation_gradient: Array,
    ) -> StateBuilderLearningProposal:
        source_state_valid = source_state.step_count >= 0
        input_valid = jnp.all(jnp.isfinite(representation_gradient))
        return _fixed_learning_proposal(
            self._learning_fingerprint,
            source_state_valid,
            input_valid,
        )

    def commit_learning_update(
        self,
        destination_state: IdentityStateBuilderState,
        proposal: StateBuilderLearningProposal,
    ) -> tuple[IdentityStateBuilderState, StateBuilderLearningDiagnostics]:
        """Validate and accept a fixed-representation no-op atomically."""
        self._validate_state_static_contract(destination_state)
        _validate_learning_proposal_static_contract(proposal, 0)
        return cast(
            tuple[IdentityStateBuilderState, StateBuilderLearningDiagnostics],
            self._commit_learning_update_jit(destination_state, proposal),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _commit_learning_update_jit(
        self,
        destination_state: IdentityStateBuilderState,
        proposal: StateBuilderLearningProposal,
    ) -> tuple[IdentityStateBuilderState, StateBuilderLearningDiagnostics]:
        diagnostics = _fixed_learning_commit_diagnostics(
            proposal,
            self._learning_fingerprint,
            destination_state.step_count >= 0,
        )
        return destination_state, diagnostics

    def learn(
        self,
        state: IdentityStateBuilderState,
        representation_gradient: Array,
    ) -> tuple[IdentityStateBuilderState, StateBuilderLearningDiagnostics]:
        """Propose and commit the fixed representation's no-op update."""
        proposal = self.propose_learning_update(state, representation_gradient)
        return self.commit_learning_update(state, proposal)


@dataclass(frozen=True)
class FixedTraceStateBuilderConfig:
    """Configuration for a fixed observation/action/outcome trace bank."""

    observation_dim: int
    n_actions: int = 0
    observation_decay_rates: tuple[float, ...] = (0.5, 0.9, 0.99)
    action_decay_rates: tuple[float, ...] = (0.5, 0.9)
    outcome_decay_rates: tuple[float, ...] = (0.5, 0.9)
    include_raw_observation: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_dim",
            _require_int32("observation_dim", self.observation_dim, minimum=1),
        )
        object.__setattr__(
            self,
            "n_actions",
            _require_int32("n_actions", self.n_actions, minimum=0),
        )
        if type(self.include_raw_observation) is not bool:
            raise ValueError("include_raw_observation must be a bool")
        object.__setattr__(
            self,
            "observation_decay_rates",
            _require_decay_rates("observation_decay_rates", self.observation_decay_rates),
        )
        object.__setattr__(
            self,
            "action_decay_rates",
            _require_decay_rates("action_decay_rates", self.action_decay_rates),
        )
        object.__setattr__(
            self,
            "outcome_decay_rates",
            _require_decay_rates("outcome_decay_rates", self.outcome_decay_rates),
        )
        if (
            not self.include_raw_observation
            and not self.observation_decay_rates
            and not self.action_decay_rates
            and not self.outcome_decay_rates
        ):
            raise ValueError("configuration must emit at least one feature")

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "type": "FixedTraceStateBuilder",
            "observation_dim": self.observation_dim,
            "n_actions": self.n_actions,
            "observation_decay_rates": list(self.observation_decay_rates),
            "action_decay_rates": list(self.action_decay_rates),
            "outcome_decay_rates": list(self.outcome_decay_rates),
            "include_raw_observation": self.include_raw_observation,
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> FixedTraceStateBuilderConfig:
        """Reconstruct from :meth:`to_config` output."""
        data = dict(payload)
        data.pop("type", None)
        return cls(
            observation_dim=data["observation_dim"],
            n_actions=data.get("n_actions", 0),
            observation_decay_rates=tuple(data.get("observation_decay_rates", ())),
            action_decay_rates=tuple(data.get("action_decay_rates", ())),
            outcome_decay_rates=tuple(data.get("outcome_decay_rates", ())),
            include_raw_observation=data.get("include_raw_observation", True),
        )


class FixedTraceStateBuilder:
    """Fixed multi-timescale trace baseline using ``WorkingMemoryFeaturizer``.

    Reward and discount are treated as a two-channel outcome vector.  Returned
    traces are post-update, so :meth:`encode` reproduces the current recurrent
    state without applying a transition twice.
    """

    def __init__(self, config: FixedTraceStateBuilderConfig):
        self._config = config
        self._learning_fingerprint = _builder_learning_fingerprint(config.to_config())
        self._memory = WorkingMemoryFeaturizer(
            WorkingMemoryConfig(
                observation_dim=config.observation_dim,
                action_dim=config.n_actions,
                reward_dim=2,
                observation_decay_rates=config.observation_decay_rates,
                action_decay_rates=config.action_decay_rates,
                reward_decay_rates=config.outcome_decay_rates,
                include_current_observation=config.include_raw_observation,
                include_current_action=False,
                include_current_reward=False,
                include_traces=True,
                include_innovations=False,
            )
        )

    @property
    def config(self) -> FixedTraceStateBuilderConfig:
        """Return the immutable configuration."""
        return self._config

    def init(self, key: Array) -> WorkingMemoryState:
        """Return an all-zero trace state; ``key`` is unused."""
        del key
        return self._memory.init()

    def feature_dim(self) -> int:
        """Return the fixed trace representation dimension."""
        return int(self._memory.feature_dim())

    def observation_dim(self) -> int:
        """Return the raw observation dimension."""
        return self._config.observation_dim

    def resource_budget(self) -> StateBuilderBudget:
        """Return exact trace-bank and counter storage."""
        cfg = self._config
        trace_scalars = (
            cfg.observation_dim * len(cfg.observation_decay_rates)
            + cfg.n_actions * len(cfg.action_decay_rates)
            + 2 * len(cfg.outcome_decay_rates)
        )
        state_scalars = trace_scalars + 4  # step_count and three last gates
        return StateBuilderBudget(
            output_scalars=self.feature_dim(),
            trainable_scalars=0,
            state_scalars=state_scalars,
            state_bytes=4 * state_scalars,
        )

    def state_valid(self, state: WorkingMemoryState) -> Bool[Array, ""]:
        """Validate every trace, gate, and lifetime counter."""
        self._validate_state_static_contract(state)
        return self._state_is_valid(state)

    def to_config(self) -> dict[str, Any]:
        """Serialize the builder configuration."""
        return self._config.to_config()

    @functools.partial(jax.jit, static_argnums=(0,))
    def encode(
        self,
        state: WorkingMemoryState,
        raw_observation: Array,
    ) -> Float[Array, " feature_dim"]:
        """Combine a raw observation with the already-current trace state."""
        observation = jnp.asarray(raw_observation, dtype=jnp.float32).reshape(
            (self._config.observation_dim,)
        )
        return cast(
            Float[Array, " feature_dim"],
            self._memory.features(
                state,
                observation,
                self._memory.zero_action(),
                self._memory.zero_reward(),
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: WorkingMemoryState,
        raw_observation: Array,
        previous_action: Array | int,
        previous_reward: Array | float,
        previous_discount: Array | float,
    ) -> tuple[WorkingMemoryState, Float[Array, " feature_dim"]]:
        """Advance all trace banks and emit the post-update memory state."""
        observation = jnp.asarray(raw_observation, dtype=jnp.float32).reshape(
            (self._config.observation_dim,)
        )
        safe_action, action_valid = safe_discrete_action(
            previous_action,
            self._config.n_actions,
            allow_unset=True,
        )
        action_vector = _action_features(safe_action, self._config.n_actions)
        outcomes = jnp.stack(
            [
                jnp.asarray(previous_reward, dtype=jnp.float32),
                jnp.asarray(previous_discount, dtype=jnp.float32),
            ]
        )
        memory_update = self._memory.update_checked(
            state,
            observation,
            action_vector,
            outcomes,
        )
        update_applied = action_valid & memory_update.update_applied
        next_state = select_transaction(update_applied, memory_update.state, state)
        representation = self.encode(next_state, observation)
        reported_representation = jnp.where(
            update_applied,
            representation,
            jnp.full_like(representation, jnp.nan),
        )
        return next_state, reported_representation

    def start(
        self,
        state: WorkingMemoryState,
        raw_observation: Array,
        last_action: Array | int = -1,
        last_reward: Array | float = 0.0,
        last_discount: Array | float = 1.0,
    ) -> tuple[WorkingMemoryState, Array]:
        """Consume the first observation and seed the trace bank."""
        return cast(
            tuple[WorkingMemoryState, Array],
            self.update(
                state,
                raw_observation,
                last_action,
                last_reward,
                last_discount,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset_episode(self, state: WorkingMemoryState) -> WorkingMemoryState:
        """Clear trace banks and gates while preserving the lifetime counter."""
        reset = self._memory.reset()
        return WorkingMemoryState(
            observation_traces=reset.observation_traces,
            action_traces=reset.action_traces,
            reward_traces=reset.reward_traces,
            step_count=state.step_count,
            last_gate=reset.last_gate,
        )

    def _validate_state_static_contract(self, state: WorkingMemoryState) -> None:
        if not isinstance(state, WorkingMemoryState):
            raise TypeError("state must be a WorkingMemoryState")
        cfg = self._config
        for name, shape in (
            (
                "observation_traces",
                (len(cfg.observation_decay_rates), cfg.observation_dim),
            ),
            ("action_traces", (len(cfg.action_decay_rates), cfg.n_actions)),
            ("reward_traces", (len(cfg.outcome_decay_rates), 2)),
            ("last_gate", (3,)),
        ):
            _static_array_contract(
                getattr(state, name),
                name=f"state.{name}",
                shape=shape,
                dtype=jnp.float32,
            )
        _static_array_contract(
            state.step_count,
            name="state.step_count",
            shape=(),
            dtype=jnp.int32,
        )

    def _validate_gradient_static_contract(self, representation_gradient: Array) -> None:
        _static_array_contract(
            representation_gradient,
            name="representation_gradient",
            shape=(self.feature_dim(),),
            dtype=jnp.float32,
        )

    @staticmethod
    def _state_is_valid(state: WorkingMemoryState) -> Array:
        return (
            jnp.all(jnp.isfinite(state.observation_traces))
            & jnp.all(jnp.isfinite(state.action_traces))
            & jnp.all(jnp.isfinite(state.reward_traces))
            & jnp.all(jnp.isfinite(state.last_gate))
            & jnp.all(state.last_gate >= 0.0)
            & jnp.all(state.last_gate <= 1.0)
            & (state.step_count >= 0)
        )

    def propose_learning_update(
        self,
        source_state: WorkingMemoryState,
        dL_drepresentation: Array,  # noqa: N803 - mathematical dL notation
    ) -> StateBuilderLearningProposal:
        """Return an honest source-checked no-op proposal."""
        self._validate_state_static_contract(source_state)
        self._validate_gradient_static_contract(dL_drepresentation)
        return cast(
            StateBuilderLearningProposal,
            self._propose_learning_update_jit(source_state, dL_drepresentation),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _propose_learning_update_jit(
        self,
        source_state: WorkingMemoryState,
        representation_gradient: Array,
    ) -> StateBuilderLearningProposal:
        return _fixed_learning_proposal(
            self._learning_fingerprint,
            self._state_is_valid(source_state),
            jnp.all(jnp.isfinite(representation_gradient)),
        )

    def commit_learning_update(
        self,
        destination_state: WorkingMemoryState,
        proposal: StateBuilderLearningProposal,
    ) -> tuple[WorkingMemoryState, StateBuilderLearningDiagnostics]:
        """Validate and accept a fixed-trace no-op atomically."""
        self._validate_state_static_contract(destination_state)
        _validate_learning_proposal_static_contract(proposal, 0)
        return cast(
            tuple[WorkingMemoryState, StateBuilderLearningDiagnostics],
            self._commit_learning_update_jit(destination_state, proposal),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _commit_learning_update_jit(
        self,
        destination_state: WorkingMemoryState,
        proposal: StateBuilderLearningProposal,
    ) -> tuple[WorkingMemoryState, StateBuilderLearningDiagnostics]:
        diagnostics = _fixed_learning_commit_diagnostics(
            proposal,
            self._learning_fingerprint,
            self._state_is_valid(destination_state),
        )
        return destination_state, diagnostics

    def learn(
        self,
        state: WorkingMemoryState,
        representation_gradient: Array,
    ) -> tuple[WorkingMemoryState, StateBuilderLearningDiagnostics]:
        """Propose and commit the fixed trace representation's no-op update."""
        proposal = self.propose_learning_update(state, representation_gradient)
        return self.commit_learning_update(state, proposal)


@dataclass(frozen=True)
class OnlineGatedStateBuilderConfig:
    """Configuration for the online learnable write/hold state builder."""

    observation_dim: int
    n_actions: int = 0
    hidden_dim: int = 8
    step_size: float = 0.01
    gradient_clip: float = 10.0
    initial_gate_bias: float = -2.0
    initialization_scale: float = 0.2
    include_raw_observation: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_dim",
            _require_int32("observation_dim", self.observation_dim, minimum=1),
        )
        object.__setattr__(
            self,
            "n_actions",
            _require_int32("n_actions", self.n_actions, minimum=0),
        )
        object.__setattr__(
            self,
            "hidden_dim",
            _require_int32("hidden_dim", self.hidden_dim, minimum=1),
        )
        if type(self.include_raw_observation) is not bool:
            raise ValueError("include_raw_observation must be a bool")
        object.__setattr__(
            self,
            "step_size",
            validated_float32_scalar("step_size", self.step_size, positive=True),
        )
        object.__setattr__(
            self,
            "gradient_clip",
            validated_float32_scalar("gradient_clip", self.gradient_clip, positive=True),
        )
        object.__setattr__(
            self,
            "initial_gate_bias",
            validated_float32_scalar("initial_gate_bias", self.initial_gate_bias),
        )
        object.__setattr__(
            self,
            "initialization_scale",
            validated_float32_scalar(
                "initialization_scale", self.initialization_scale, positive=True
            ),
        )

    def event_dim(self) -> int:
        """Return observation + one-hot action + reward + discount width."""
        return self.observation_dim + self.n_actions + 2

    def parameter_count(self) -> int:
        """Return write/candidate weights and biases."""
        return 2 * self.hidden_dim * (self.event_dim() + 1)

    def feature_dim(self) -> int:
        """Return raw-observation plus hidden-state width."""
        return self.hidden_dim + (self.observation_dim if self.include_raw_observation else 0)

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        payload = asdict(self)
        payload["type"] = "OnlineGatedStateBuilder"
        return payload

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> OnlineGatedStateBuilderConfig:
        """Reconstruct from :meth:`to_config` output."""
        data = dict(payload)
        data.pop("type", None)
        return cls(**data)


@chex.dataclass(frozen=True)
class OnlineGatedStateBuilderState:
    """Parameters, recurrent state, and RTRL-style eligibility sensitivities."""

    parameters: Float[Array, " parameter_count"]
    hidden: Float[Array, " hidden_dim"]
    parameter_sensitivity: Float[Array, "hidden_dim parameter_count"]
    step_count: Array
    update_count: Array
    last_gradient_norm: Float[Array, ""]


class OnlineGatedStateBuilder:
    """Learnable gated recurrent state with an online sensitivity trace.

    The recurrence is advanced by :meth:`start`/:meth:`update`.  Learning is a
    separate call because a prediction must be scored before its target-derived
    gradient is allowed to modify the representation.  ``learn`` updates only
    recurrent parameters; a downstream head owns and checkpoints its own
    parameters. Sensitivities are exact for an unroll with fixed parameters.
    Carrying them across :meth:`learn` calls is an online eligibility
    approximation; it is not the derivative of the stored hidden state with
    respect to the newly updated parameter vector.
    """

    def __init__(self, config: OnlineGatedStateBuilderConfig):
        self._config = config
        self._learning_fingerprint = _builder_learning_fingerprint(config.to_config())

    @property
    def config(self) -> OnlineGatedStateBuilderConfig:
        """Return the immutable configuration."""
        return self._config

    def feature_dim(self) -> int:
        """Return the fixed emitted representation width."""
        return self._config.feature_dim()

    def observation_dim(self) -> int:
        """Return the raw observation dimension."""
        return self._config.observation_dim

    def resource_budget(self) -> StateBuilderBudget:
        """Return exact persistent state, including recurrent sensitivities."""
        parameter_count = self._config.parameter_count()
        state_scalars = (
            parameter_count
            + self._config.hidden_dim
            + self._config.hidden_dim * parameter_count
            + 3
        )
        return StateBuilderBudget(
            output_scalars=self.feature_dim(),
            trainable_scalars=parameter_count,
            state_scalars=state_scalars,
            state_bytes=4 * state_scalars,
        )

    def state_valid(
        self,
        state: OnlineGatedStateBuilderState,
    ) -> Bool[Array, ""]:
        """Validate parameters, recurrence, sensitivities, norms, and counters."""
        self._validate_state_static_contract(state)
        return self._state_is_valid(state)

    def to_config(self) -> dict[str, Any]:
        """Serialize the builder configuration."""
        return self._config.to_config()

    def init(self, key: Array) -> OnlineGatedStateBuilderState:
        """Initialize trainable parameters, hidden state, and sensitivities."""
        gate_key, candidate_key = jr.split(key)
        cfg = self._config
        event_dim = cfg.event_dim()
        hidden_dim = cfg.hidden_dim
        scale = jnp.asarray(cfg.initialization_scale, dtype=jnp.float32)
        gate_weights = scale * jr.normal(
            gate_key,
            (hidden_dim, event_dim),
            dtype=jnp.float32,
        )
        gate_bias = jnp.full(
            (hidden_dim,),
            cfg.initial_gate_bias,
            dtype=jnp.float32,
        )
        candidate_weights = scale * jr.normal(
            candidate_key,
            (hidden_dim, event_dim),
            dtype=jnp.float32,
        )
        candidate_bias = jnp.zeros((hidden_dim,), dtype=jnp.float32)
        parameters = jnp.concatenate(
            [
                gate_weights.reshape(-1),
                gate_bias,
                candidate_weights.reshape(-1),
                candidate_bias,
            ]
        )
        return OnlineGatedStateBuilderState(
            parameters=parameters,
            hidden=jnp.zeros((hidden_dim,), dtype=jnp.float32),
            parameter_sensitivity=jnp.zeros(
                (hidden_dim, cfg.parameter_count()),
                dtype=jnp.float32,
            ),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            update_count=jnp.asarray(0, dtype=jnp.int32),
            last_gradient_norm=jnp.asarray(0.0, dtype=jnp.float32),
        )

    def _unpack_parameters(
        self,
        parameters: Array,
    ) -> tuple[Array, Array, Array, Array]:
        cfg = self._config
        matrix_size = cfg.hidden_dim * cfg.event_dim()
        offset = 0
        gate_weights = parameters[offset : offset + matrix_size].reshape(
            (cfg.hidden_dim, cfg.event_dim())
        )
        offset += matrix_size
        gate_bias = parameters[offset : offset + cfg.hidden_dim]
        offset += cfg.hidden_dim
        candidate_weights = parameters[offset : offset + matrix_size].reshape(
            (cfg.hidden_dim, cfg.event_dim())
        )
        offset += matrix_size
        candidate_bias = parameters[offset : offset + cfg.hidden_dim]
        return gate_weights, gate_bias, candidate_weights, candidate_bias

    def _transition(self, parameters: Array, hidden: Array, event: Array) -> Array:
        gate_weights, gate_bias, candidate_weights, candidate_bias = self._unpack_parameters(
            parameters
        )
        gate = jax.nn.sigmoid(gate_weights @ event + gate_bias)
        candidate = jnp.tanh(candidate_weights @ event + candidate_bias)
        return hidden + gate * (candidate - hidden)

    def _event(
        self,
        raw_observation: Array,
        previous_action: Array | int,
        previous_reward: Array | float,
        previous_discount: Array | float,
    ) -> Array:
        observation = jnp.asarray(raw_observation, dtype=jnp.float32).reshape(
            (self._config.observation_dim,)
        )
        return jnp.concatenate(
            [
                observation,
                _action_features(previous_action, self._config.n_actions),
                jnp.atleast_1d(jnp.asarray(previous_reward, dtype=jnp.float32)),
                jnp.atleast_1d(jnp.asarray(previous_discount, dtype=jnp.float32)),
            ]
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def encode(
        self,
        state: OnlineGatedStateBuilderState,
        raw_observation: Array,
    ) -> Float[Array, " feature_dim"]:
        """Pair raw input with the current hidden state without advancing it."""
        observation = jnp.asarray(raw_observation, dtype=jnp.float32).reshape(
            (self._config.observation_dim,)
        )
        if self._config.include_raw_observation:
            return jnp.concatenate([observation, state.hidden])
        return state.hidden

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: OnlineGatedStateBuilderState,
        raw_observation: Array,
        previous_action: Array | int,
        previous_reward: Array | float,
        previous_discount: Array | float,
    ) -> tuple[OnlineGatedStateBuilderState, Float[Array, " feature_dim"]]:
        """Advance recurrence and its RTRL-style eligibility sensitivity."""
        safe_action, action_valid = safe_discrete_action(
            previous_action,
            self._config.n_actions,
            allow_unset=True,
        )
        event = self._event(
            raw_observation,
            safe_action,
            previous_reward,
            previous_discount,
        )
        event_valid = action_valid & jnp.all(jnp.isfinite(event))
        safe_event = jnp.where(event_valid, event, jnp.zeros_like(event))
        new_hidden = self._transition(state.parameters, state.hidden, safe_event)
        direct_sensitivity = jax.jacfwd(self._transition, argnums=0)(
            state.parameters,
            state.hidden,
            safe_event,
        )

        gate_weights, gate_bias, _, _ = self._unpack_parameters(state.parameters)
        gate = jax.nn.sigmoid(gate_weights @ safe_event + gate_bias)
        new_sensitivity = direct_sensitivity + ((1.0 - gate)[:, None] * state.parameter_sensitivity)
        candidate_state = OnlineGatedStateBuilderState(
            parameters=state.parameters,
            hidden=new_hidden,
            parameter_sensitivity=new_sensitivity,
            step_count=_saturating_int32_increment(state.step_count),
            update_count=state.update_count,
            last_gradient_norm=state.last_gradient_norm,
        )
        update_applied = (
            event_valid & self._state_is_valid(state) & self._state_is_valid(candidate_state)
        )
        next_state = select_transaction(update_applied, candidate_state, state)
        representation = self.encode(next_state, raw_observation)
        reported_representation = jnp.where(
            update_applied,
            representation,
            jnp.full_like(representation, jnp.nan),
        )
        return next_state, reported_representation

    def start(
        self,
        state: OnlineGatedStateBuilderState,
        raw_observation: Array,
        last_action: Array | int = -1,
        last_reward: Array | float = 0.0,
        last_discount: Array | float = 1.0,
    ) -> tuple[OnlineGatedStateBuilderState, Array]:
        """Consume the initial observation."""
        return cast(
            tuple[OnlineGatedStateBuilderState, Array],
            self.update(
                state,
                raw_observation,
                last_action,
                last_reward,
                last_discount,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset_episode(
        self,
        state: OnlineGatedStateBuilderState,
    ) -> OnlineGatedStateBuilderState:
        """Clear recurrence and sensitivities while retaining lifetime learning."""
        return OnlineGatedStateBuilderState(
            parameters=state.parameters,
            hidden=jnp.zeros_like(state.hidden),
            parameter_sensitivity=jnp.zeros_like(state.parameter_sensitivity),
            step_count=state.step_count,
            update_count=state.update_count,
            last_gradient_norm=state.last_gradient_norm,
        )

    def _validate_state_static_contract(
        self,
        state: OnlineGatedStateBuilderState,
    ) -> None:
        """Reject structural state corruption before eager or compiled execution."""
        if not isinstance(state, OnlineGatedStateBuilderState):
            raise TypeError("state must be an OnlineGatedStateBuilderState")
        cfg = self._config
        parameter_count = cfg.parameter_count()
        _static_array_contract(
            state.parameters,
            name="state.parameters",
            shape=(parameter_count,),
            dtype=jnp.float32,
        )
        _static_array_contract(
            state.hidden,
            name="state.hidden",
            shape=(cfg.hidden_dim,),
            dtype=jnp.float32,
        )
        _static_array_contract(
            state.parameter_sensitivity,
            name="state.parameter_sensitivity",
            shape=(cfg.hidden_dim, parameter_count),
            dtype=jnp.float32,
        )
        _static_array_contract(
            state.step_count,
            name="state.step_count",
            shape=(),
            dtype=jnp.int32,
        )
        _static_array_contract(
            state.update_count,
            name="state.update_count",
            shape=(),
            dtype=jnp.int32,
        )
        _static_array_contract(
            state.last_gradient_norm,
            name="state.last_gradient_norm",
            shape=(),
            dtype=jnp.float32,
        )

    def _validate_gradient_static_contract(self, representation_gradient: Array) -> None:
        _static_array_contract(
            representation_gradient,
            name="representation_gradient",
            shape=(self.feature_dim(),),
            dtype=jnp.float32,
        )

    @staticmethod
    def _state_is_valid(state: OnlineGatedStateBuilderState) -> Array:
        return (
            jnp.all(jnp.isfinite(state.parameters))
            & jnp.all(jnp.isfinite(state.hidden))
            & jnp.all(jnp.isfinite(state.parameter_sensitivity))
            & jnp.isfinite(state.last_gradient_norm)
            & (state.last_gradient_norm >= 0.0)
            & (state.step_count >= 0)
            & (state.update_count >= 0)
        )

    def propose_learning_update(
        self,
        source_state: OnlineGatedStateBuilderState,
        dL_drepresentation: Array,  # noqa: N803 - mathematical dL notation
    ) -> StateBuilderLearningProposal:
        """Form a pure clipped update from the source state's sensitivity."""
        self._validate_state_static_contract(source_state)
        self._validate_gradient_static_contract(dL_drepresentation)
        return cast(
            StateBuilderLearningProposal,
            self._propose_learning_update_jit(source_state, dL_drepresentation),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _propose_learning_update_jit(
        self,
        source_state: OnlineGatedStateBuilderState,
        representation_gradient: Array,
    ) -> StateBuilderLearningProposal:
        """Numerically validate and form one source-bound online proposal."""
        gradient = jnp.asarray(representation_gradient, dtype=jnp.float32)
        hidden_gradient = gradient[-self._config.hidden_dim :]

        source_state_valid = self._state_is_valid(source_state)
        input_valid = jnp.all(jnp.isfinite(gradient))
        raw_parameter_gradient = source_state.parameter_sensitivity.T @ hidden_gradient
        raw_parameter_gradient_valid = jnp.all(jnp.isfinite(raw_parameter_gradient))
        safe_parameter_gradient = jnp.where(
            input_valid & raw_parameter_gradient_valid,
            raw_parameter_gradient,
            0.0,
        )

        clip = jnp.asarray(self._config.gradient_clip, dtype=jnp.float32)
        clipped_parameter_gradient, safe_gradient_norm = _scale_safe_clip_by_l2_norm(
            safe_parameter_gradient,
            clip,
        )
        gradient_norm = jnp.where(
            input_valid & raw_parameter_gradient_valid,
            safe_gradient_norm,
            jnp.asarray(_FLOAT32_MAX, dtype=jnp.float32),
        )
        clipped_parameter_gradient_valid = jnp.all(jnp.isfinite(clipped_parameter_gradient))
        candidate_parameter_update = (
            -jnp.asarray(self._config.step_size, dtype=jnp.float32) * clipped_parameter_gradient
        )
        candidate_parameter_update_valid = jnp.all(jnp.isfinite(candidate_parameter_update))
        candidate_parameters = source_state.parameters + candidate_parameter_update
        candidate_parameters_valid = jnp.all(jnp.isfinite(candidate_parameters))
        capacity_available = (source_state.update_count >= 0) & (
            source_state.update_count < _INT32_MAX
        )
        valid = (
            source_state_valid
            & input_valid
            & raw_parameter_gradient_valid
            & clipped_parameter_gradient_valid
            & candidate_parameter_update_valid
            & candidate_parameters_valid
            & capacity_available
        )
        return StateBuilderLearningProposal(
            builder_fingerprint=self._learning_fingerprint,
            source_parameters=source_state.parameters,
            source_update_count=source_state.update_count,
            raw_parameter_gradient=raw_parameter_gradient,
            clipped_parameter_gradient=clipped_parameter_gradient,
            candidate_parameter_update=candidate_parameter_update,
            gradient_norm=gradient_norm,
            clipped_gradient_norm=_finite_or_max_norm(
                clipped_parameter_gradient,
                clipped_parameter_gradient_valid,
            ),
            parameter_update_norm=_finite_or_max_norm(
                candidate_parameter_update,
                candidate_parameter_update_valid,
            ),
            source_state_valid=source_state_valid,
            input_valid=input_valid,
            raw_parameter_gradient_valid=raw_parameter_gradient_valid,
            clipped_parameter_gradient_valid=clipped_parameter_gradient_valid,
            candidate_parameter_update_valid=candidate_parameter_update_valid,
            candidate_parameters_valid=candidate_parameters_valid,
            capacity_available=capacity_available,
            candidate_update_transformed=jnp.asarray(False),
            candidate_update_approved=jnp.asarray(True),
            fixed_noop=jnp.asarray(False),
            valid=valid,
            rejected=~valid,
        )

    def _proposal_has_integrity(self, proposal: StateBuilderLearningProposal) -> Array:
        """Recompute proposal diagnostics and update formation at commit time."""
        raw_valid = jnp.all(jnp.isfinite(proposal.raw_parameter_gradient))
        safe_raw_gradient = jnp.where(
            proposal.input_valid & raw_valid,
            proposal.raw_parameter_gradient,
            0.0,
        )
        expected_clipped, safe_gradient_norm = _scale_safe_clip_by_l2_norm(
            safe_raw_gradient,
            jnp.asarray(self._config.gradient_clip, dtype=jnp.float32),
        )
        expected_gradient_norm = jnp.where(
            proposal.input_valid & raw_valid,
            safe_gradient_norm,
            jnp.asarray(_FLOAT32_MAX, dtype=jnp.float32),
        )
        clipped_valid = jnp.all(jnp.isfinite(proposal.clipped_parameter_gradient))
        expected_formed_update = (
            -jnp.asarray(self._config.step_size, dtype=jnp.float32)
            * proposal.clipped_parameter_gradient
        )
        formed_update_matches = proposal.candidate_update_transformed | (
            _float32_vectors_bitwise_equal(
                proposal.candidate_parameter_update,
                expected_formed_update,
            )
        )
        update_valid = jnp.all(jnp.isfinite(proposal.candidate_parameter_update))
        candidate_parameters = proposal.source_parameters + proposal.candidate_parameter_update
        candidate_parameters_valid = jnp.all(jnp.isfinite(candidate_parameters))
        capacity_available = (proposal.source_update_count >= 0) & (
            proposal.source_update_count < _INT32_MAX
        )
        expected_valid = (
            proposal.source_state_valid
            & proposal.input_valid
            & raw_valid
            & clipped_valid
            & update_valid
            & candidate_parameters_valid
            & capacity_available
            & proposal.candidate_update_approved
        )
        return (
            ~proposal.fixed_noop
            & (proposal.candidate_update_transformed | proposal.candidate_update_approved)
            & (~proposal.source_state_valid | jnp.all(jnp.isfinite(proposal.source_parameters)))
            & (proposal.raw_parameter_gradient_valid == raw_valid)
            & _float32_vectors_bitwise_equal(
                proposal.clipped_parameter_gradient,
                expected_clipped,
            )
            & (proposal.clipped_parameter_gradient_valid == clipped_valid)
            & formed_update_matches
            & (proposal.candidate_parameter_update_valid == update_valid)
            & (proposal.candidate_parameters_valid == candidate_parameters_valid)
            & (proposal.capacity_available == capacity_available)
            & _diagnostic_scalar_matches(
                proposal.gradient_norm,
                expected_gradient_norm,
            )
            & _diagnostic_scalar_matches(
                proposal.clipped_gradient_norm,
                _finite_or_max_norm(proposal.clipped_parameter_gradient, clipped_valid),
            )
            & _diagnostic_scalar_matches(
                proposal.parameter_update_norm,
                _finite_or_max_norm(proposal.candidate_parameter_update, update_valid),
            )
            & (proposal.valid == expected_valid)
            & (proposal.rejected == ~expected_valid)
        )

    def commit_learning_update(
        self,
        destination_state: OnlineGatedStateBuilderState,
        proposal: StateBuilderLearningProposal,
    ) -> tuple[OnlineGatedStateBuilderState, StateBuilderLearningDiagnostics]:
        """Atomically commit a valid proposal if its parameter source is current."""
        self._validate_state_static_contract(destination_state)
        _validate_learning_proposal_static_contract(
            proposal,
            self._config.parameter_count(),
        )
        return cast(
            tuple[OnlineGatedStateBuilderState, StateBuilderLearningDiagnostics],
            self._commit_learning_update_jit(destination_state, proposal),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _commit_learning_update_jit(
        self,
        destination_state: OnlineGatedStateBuilderState,
        proposal: StateBuilderLearningProposal,
    ) -> tuple[OnlineGatedStateBuilderState, StateBuilderLearningDiagnostics]:
        proposal_integrity = self._proposal_has_integrity(proposal)
        fingerprint_matches = jnp.array_equal(
            proposal.builder_fingerprint,
            self._learning_fingerprint,
        )
        parameters_match = _float32_vectors_bitwise_equal(
            destination_state.parameters,
            proposal.source_parameters,
        )
        count_matches = destination_state.update_count == proposal.source_update_count
        source_matches = fingerprint_matches & parameters_match & count_matches
        destination_state_valid = self._state_is_valid(destination_state)
        capacity_available = (
            (destination_state.update_count >= 0)
            & (destination_state.update_count < _INT32_MAX)
            & proposal.capacity_available
        )
        candidate_parameters = destination_state.parameters + proposal.candidate_parameter_update
        candidate_parameters_valid = jnp.all(jnp.isfinite(candidate_parameters))
        proposal_valid = proposal_integrity & proposal.valid
        applied = (
            destination_state_valid
            & source_matches
            & proposal_valid
            & capacity_available
            & candidate_parameters_valid
        )
        candidate_state = OnlineGatedStateBuilderState(
            parameters=candidate_parameters,
            hidden=destination_state.hidden,
            parameter_sensitivity=destination_state.parameter_sensitivity,
            step_count=destination_state.step_count,
            update_count=_saturating_int32_increment(destination_state.update_count),
            last_gradient_norm=proposal.gradient_norm,
        )
        next_state = cast(
            OnlineGatedStateBuilderState,
            jax.lax.cond(applied, lambda: candidate_state, lambda: destination_state),
        )
        norms_valid = (
            jnp.isfinite(proposal.gradient_norm)
            & (proposal.gradient_norm >= 0.0)
            & jnp.isfinite(proposal.clipped_gradient_norm)
            & (proposal.clipped_gradient_norm >= 0.0)
            & jnp.isfinite(proposal.parameter_update_norm)
            & (proposal.parameter_update_norm >= 0.0)
        )
        safe_gradient_norm = jnp.where(
            norms_valid,
            proposal.gradient_norm,
            jnp.asarray(_FLOAT32_MAX, dtype=jnp.float32),
        )
        safe_clipped_norm = jnp.where(
            norms_valid,
            proposal.clipped_gradient_norm,
            jnp.asarray(_FLOAT32_MAX, dtype=jnp.float32),
        )
        safe_update_norm = jnp.where(
            norms_valid,
            proposal.parameter_update_norm,
            jnp.asarray(_FLOAT32_MAX, dtype=jnp.float32),
        )
        diagnostics = StateBuilderLearningDiagnostics(
            gradient_norm=safe_gradient_norm,
            clipped_gradient_norm=safe_clipped_norm,
            parameter_update_norm=safe_update_norm,
            proposal_valid=proposal_valid,
            source_matches=source_matches,
            capacity_available=capacity_available,
            candidate_parameters_valid=candidate_parameters_valid,
            applied=applied,
            fixed_noop=jnp.asarray(False),
            valid=applied,
            rejected=~applied,
        )
        return next_state, diagnostics

    def learn(
        self,
        state: OnlineGatedStateBuilderState,
        representation_gradient: Array,
    ) -> tuple[OnlineGatedStateBuilderState, StateBuilderLearningDiagnostics]:
        """Propose and commit one update against the same parameter version."""
        proposal = self.propose_learning_update(state, representation_gradient)
        return self.commit_learning_update(state, proposal)


StateBuilderConfig = (
    IdentityStateBuilderConfig | FixedTraceStateBuilderConfig | OnlineGatedStateBuilderConfig
)


def state_builder_config_from_config(payload: dict[str, Any]) -> StateBuilderConfig:
    """Parse one known state-builder configuration without creating state."""
    builder_type = payload.get("type")
    if builder_type == "IdentityStateBuilder":
        return IdentityStateBuilderConfig.from_config(payload)
    if builder_type == "FixedTraceStateBuilder":
        return FixedTraceStateBuilderConfig.from_config(payload)
    if builder_type == "OnlineGatedStateBuilder":
        return OnlineGatedStateBuilderConfig.from_config(payload)
    raise ValueError(f"unknown state builder type: {builder_type!r}")


def state_builder_from_config(payload: dict[str, Any]) -> StateBuilder[Any]:
    """Construct a known state builder from its serialized configuration."""
    config = state_builder_config_from_config(payload)
    if isinstance(config, IdentityStateBuilderConfig):
        return IdentityStateBuilder(config)
    if isinstance(config, FixedTraceStateBuilderConfig):
        return FixedTraceStateBuilder(config)
    return OnlineGatedStateBuilder(config)


def _state_builder_config_digest(config: dict[str, Any]) -> str:
    """Return the canonical SHA-256 digest of an exact serialized config."""
    canonical = json.dumps(
        config,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def save_state_builder_checkpoint(
    builder: StateBuilder[Any],
    state: Any,
    path: str | Path,
) -> None:
    """Save a builder's full configuration and PyTree state."""
    config = builder.to_config()
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": STATE_BUILDER_CHECKPOINT_SCHEMA,
            "builder_config": config,
            "config_sha256": _state_builder_config_digest(config),
            "resource_budget": builder.resource_budget().to_config(),
        },
    )


def load_state_builder_checkpoint(
    path: str | Path,
    *,
    template_key: Array | None = None,
) -> tuple[StateBuilder[Any], Any]:
    """Restore a state builder without requiring a caller-constructed template."""
    metadata = load_checkpoint_metadata(path)
    if metadata.get("schema") != STATE_BUILDER_CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint is not an Alberta state-builder v2 checkpoint")
    config = metadata.get("builder_config")
    if not isinstance(config, dict):
        raise ValueError("state-builder checkpoint is missing builder_config")
    expected_config_digest = _state_builder_config_digest(config)
    if metadata.get("config_sha256") != expected_config_digest:
        raise ValueError("state-builder checkpoint config digest does not match config")
    builder = state_builder_from_config(config)
    if _state_builder_config_digest(builder.to_config()) != expected_config_digest:
        raise ValueError("state-builder checkpoint config is not canonical for its builder")
    key = jr.key(0) if template_key is None else template_key
    template = builder.init(key)
    state, restored_metadata = load_checkpoint(template, path)
    if restored_metadata != metadata:
        raise ValueError("state-builder checkpoint metadata changed between reads")
    if restored_metadata.get("resource_budget") != builder.resource_budget().to_config():
        raise ValueError("state-builder checkpoint resource budget does not match config")
    return builder, state


__all__ = [
    "STATE_BUILDER_CHECKPOINT_SCHEMA",
    "FixedTraceStateBuilder",
    "FixedTraceStateBuilderConfig",
    "IdentityStateBuilder",
    "IdentityStateBuilderConfig",
    "IdentityStateBuilderState",
    "OnlineGatedStateBuilder",
    "OnlineGatedStateBuilderConfig",
    "OnlineGatedStateBuilderState",
    "StateBuilder",
    "StateBuilderBudget",
    "StateBuilderConfig",
    "StateBuilderLearningDiagnostics",
    "StateBuilderLearningProposal",
    "load_state_builder_checkpoint",
    "replace_state_builder_learning_proposal_update",
    "save_state_builder_checkpoint",
    "state_builder_config_from_config",
    "state_builder_from_config",
]
