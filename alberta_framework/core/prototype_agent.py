# mypy: disable-error-code="attr-defined,call-arg,no-any-return,union-attr"
"""Prototype agent assembling mechanisms across the Alberta Plan.

The :class:`PrototypeAgent` is an experimental integration of mechanisms
mapped to the Alberta Plan's retreat-and-return strategy:

- **OaK** (steps 5/6/10/11): Differential average-reward control with temporal
  abstraction, option utility tracking, and curation.
- **Horde** (step 3): Parallel GVF prediction demons sharing a learned trunk.
- **World Model** (step 8): One-step action-conditioned environment model.
- **Model-only rehearsal** (step 8): Optional bounded dual replay that updates
  only world-model ensemble members from stored real transitions.
- **Guarded Dreaming** (step 9): Model-generated Dyna transitions accepted only
  when the world model's error EMA is below a configurable gate.
- **Intelligence Amplification** (step 12, optional): Exo-cerebellum +
  exo-cortex companion that augments a partner agent's decisions.

The integrated control and prediction paths are online, batch-size-one
learners. Step 1/2 adaptive optimizers and bounded updates are available
elsewhere in the framework, but the default STOMP base learner here uses LMS;
this integration alone is therefore not evidence that every Step 1/2 variant
is active.

References:
    Sutton, Bowling, & Pilarski (2022). "The Alberta Plan for AI Research."
    Barreto et al. (2019). "The Option Keyboard: Combining Skills in RL."
    Sutton et al. (2011). "Horde: A Scalable Real-time Architecture."
    Elsayed & Sutton (2024). "Streaming Backpropagation through Time."
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
import operator
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, SupportsIndex, cast

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
from alberta_framework.core.delight import (
    GradientJoyApplicationResult,
    GradientJoyConfig,
    GradientJoyEvidence,
    LearningValue,
    LearningValueAvailability,
    apply_gradient_joy_update,
)
from alberta_framework.core.dreaming import (
    DreamingConfig,
    GuardedDreamer,
    RecentObservationBuffer,
)
from alberta_framework.core.experiential_memory import (
    ExperientialMemory,
    ExperientialMemoryConfig,
    ExperientialMemoryEntry,
    ExperientialMemoryState,
    ExperientialMemoryStepResult,
)
from alberta_framework.core.experiential_memory_policy import (
    ExperientialMemoryPolicy,
    ExperientialMemoryPolicyProposal,
)
from alberta_framework.core.horde import HordeLearner
from alberta_framework.core.intelligence_amplification import (
    IAAgent,
    IAConfig,
)
from alberta_framework.core.learning_signals import (
    LearningSignalAvailability,
    LearningSignalEstimator,
    LearningSignalEstimatorConfig,
    LearningSignalEstimatorState,
    TypedLearningSignals,
)
from alberta_framework.core.model_replay_rehearsal import (
    ModelReplayRehearsal,
    ModelReplayRehearsalConfig,
    RealModelReplayEvent,
)
from alberta_framework.core.multi_head_learner import _validate_direct_state_resources
from alberta_framework.core.normalizers import _lifetime_counter_valid
from alberta_framework.core.oak import (
    OaKAgent,
    OaKConfig,
    OaKKeyboardPolicyProposal,
    OaKState,
)
from alberta_framework.core.option_search_control import (
    OptionSearchControl,
    OptionSearchControlConfig,
    OptionSearchControlDiagnostics,
    OptionSearchControlResourceBudget,
)
from alberta_framework.core.options import (
    DispatchedPrimitiveActionDecision,
    STOMPConfig,
    SubtaskSpec,
    check_option_terminated,
    compute_pseudo_reward,
    replace_dispatched_primitive_action,
)
from alberta_framework.core.partner_policy_fusion import (
    OptionKeyboardProposal,
    PartnerFusionDecision,
    PartnerFusionFeedbackResult,
    PartnerMessageBatch,
    PartnerPolicyFusion,
    PartnerPolicyFusionConfig,
    PartnerPolicyFusionFeedback,
    PartnerPolicyFusionState,
)
from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureConsumerBinding,
    PrototypeFeatureLifecycle,
    PrototypeFeatureLifecycleConfig,
    PrototypeFeatureLifecycleDiagnostics,
    PrototypeFeatureLifecycleEvent,
    PrototypeFeatureLifecycleState,
    PrototypePairGradientPullback,
)
from alberta_framework.core.recurrent_latent_world_model_ensemble import (
    RecurrentLatentDecisionCache,
    RecurrentLatentPredictionAvailability,
    RecurrentLatentStartCache,
    RecurrentLatentTransitionRecord,
    RecurrentLatentWorldModelDiagnostics,
    RecurrentLatentWorldModelEnsemble,
    RecurrentLatentWorldModelEnsembleConfig,
    RecurrentLatentWorldModelEnsembleState,
)
from alberta_framework.core.representation_gradient_mixer import (
    RepresentationGradientMixDiagnostics,
    RepresentationGradientMixerConfig,
    RepresentationGradientMixResult,
    mix_representation_gradients,
)
from alberta_framework.core.state_builder import (
    IdentityStateBuilderConfig,
    OnlineGatedStateBuilderConfig,
    StateBuilder,
    StateBuilderConfig,
    StateBuilderLearningDiagnostics,
    replace_state_builder_learning_proposal_update,
    state_builder_from_config,
)
from alberta_framework.core.types import HordeSpec
from alberta_framework.core.world_model import (
    ActionConditionedWorldModel,
    ActionConditionedWorldModelConfig,
)
from alberta_framework.core.world_model_ensemble import (
    WorldModelEnsemble,
    WorldModelEnsembleConfig,
    WorldModelEnsembleDiagnostics,
    WorldModelEnsembleState,
)

PROTOTYPE_CHECKPOINT_SCHEMA = "alberta.prototype_agent.v3"
_PROTOTYPE_CHECKPOINT_SCHEMA_V2 = "alberta.prototype_agent.v2"
_PROTOTYPE_CHECKPOINT_SCHEMA_V1 = "alberta.prototype_agent.v1"
_PROTOTYPE_EMPTY_ARRAY_CODEC = "alberta.prototype_agent.empty_array_projection.v1"
_PROTOTYPE_SUPPORTED_PRNG_IMPLS = frozenset({"threefry2x32", "rbg"})
_GUARDED_DREAM_STREAM_TAG = 0x44524D53
_DREAM_NEXT_OBSERVATION_STREAM_TAG = 0x44524D4F
_PROTOTYPE_V2_REPLAY_MIGRATION_TAG = 0x50525632
_PROTOTYPE_FEATURE_LIFECYCLE_KEY_TAG = 0x50464C43
_UINT32_MAX = 2**32 - 1
_INT32_MAX = 2**31 - 1

# ---------------------------------------------------------------------------
# Standalone utility
# ---------------------------------------------------------------------------


def _guarded_dream_rng_root(stored_real_key: Array) -> Array:
    """Derive dreaming randomness without consuming the future real stream."""
    return jr.fold_in(stored_real_key, _GUARDED_DREAM_STREAM_TAG)


def feature_to_subtask_specs(
    oak_state: OaKState,
    *,
    n_subtasks: int = 4,
    threshold: float = 0.5,
    pseudo_reward_scale: float = 1.0,
    max_option_steps: int = 20,
) -> tuple[SubtaskSpec, ...]:
    """Extract top-k subtask specs from OaK Q-weight feature importance.

    Ranks observation dimensions by the maximum absolute Q-weight across all
    base and option policies.  The top-k most important features become subtask
    targets for the next curation cycle.

    Args:
        oak_state: Current OaK state.
        n_subtasks: Non-negative integer scalar number of subtask specs to return.
        threshold: Pseudo-reward threshold for subtask completion.
        pseudo_reward_scale: Pseudo-reward multiplier for generated specs.
        max_option_steps: Hard cap on option duration.

    Returns:
        Tuple of up to ``n_subtasks`` :class:`SubtaskSpec` instances, ordered
        by descending feature importance.
    """
    if isinstance(n_subtasks, bool):
        raise ValueError("n_subtasks must be a non-negative integer")
    try:
        normalized_n_subtasks = operator.index(n_subtasks)
    except TypeError as exc:
        raise ValueError("n_subtasks must be a non-negative integer") from exc
    if normalized_n_subtasks < 0:
        raise ValueError("n_subtasks must be a non-negative integer")

    bls = oak_state.stomp_state.base_learner_state
    trunk_ws = bls.trunk_params.weights
    if len(trunk_ws) == 0:
        # Linear base Q: head_params.weights[i] has shape (1, obs_dim)
        base_q_mat = jnp.stack([w[0] for w in bls.head_params.weights])
    else:
        # Nonlinear: use first trunk layer as feature-importance proxy
        base_q_mat = trunk_ws[0]  # (hidden_size, obs_dim)
    feature_importance = jnp.max(jnp.abs(base_q_mat), axis=0)  # (obs_dim,)

    opt_q = oak_state.stomp_state.option_policies.q_weights      # (n_opts, n_prim, obs_dim)
    opt_q_abs = jnp.abs(opt_q)
    obs_dim = int(opt_q.shape[-1])
    opt_importance = jnp.max(opt_q_abs.reshape(-1, obs_dim), axis=0)  # (obs_dim,)

    combined = jnp.maximum(feature_importance, opt_importance)
    n = min(normalized_n_subtasks, obs_dim)
    ranking = sorted(range(obs_dim), key=lambda i: float(combined[i]), reverse=True)[:n]

    return tuple(
        SubtaskSpec(
            feature_index=int(idx),
            threshold=threshold,
            pseudo_reward_scale=pseudo_reward_scale,
            max_option_steps=max_option_steps,
        )
        for idx in ranking
    )


# ---------------------------------------------------------------------------
# Validation helpers (hostile-safe, evidence-grade)
# ---------------------------------------------------------------------------


_INT32_MAX = 2**31 - 1
_UINT32_MAX = 4_294_967_295
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
_ACTUAL_FLOAT_TYPES: tuple[type, ...] = (
    float,
    Fraction,
    *(np.dtype(code).type for code in ("e", "f", "d", "g")),
)
_ALLOWED_REAL_TYPES: tuple[type, ...] = _ACTUAL_INT_TYPES + _ACTUAL_FLOAT_TYPES


def _require_int(
    name: str,
    value: object,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    actual_type = type(value)
    if not any(actual_type is candidate for candidate in _ACTUAL_INT_TYPES):
        raise ValueError(f"{name} must be an integer")
    number = operator.index(cast(SupportsIndex, value))
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return number


def _validated_config_float(name: str, value: object, **bounds: Any) -> float:
    actual_type = type(value)
    if not any(actual_type is candidate for candidate in _ALLOWED_REAL_TYPES):
        raise ValueError(f"{name} must be a finite real scalar")
    return validated_float32_scalar(name, value, **bounds)


def _require_float32_resource(
    name: str,
    *,
    vector_scalars: int,
    fixed_scalars: int = 0,
) -> None:
    if vector_scalars < 0 or fixed_scalars < 0:
        raise ValueError(f"{name} scalar counts must be non-negative")
    total = vector_scalars + fixed_scalars
    if total > _INT32_MAX:
        raise ValueError(f"{name} scalar count must fit signed int32")
    if 4 * total > _INT32_MAX:
        raise ValueError(f"{name} byte count must fit signed int32")


def _gru_perception_persistent_bytes(observation_dim: int, hidden_dim: int) -> int:
    """Named persist already preflighted by ``GRUPerceptionConfig``."""
    return 4 * (
        3 * hidden_dim * observation_dim + 3 * hidden_dim * hidden_dim + 4 * hidden_dim
    )


def _gru_perception_update_result_extras_bytes(
    observation_dim: int, hidden_dim: int
) -> int:
    """Returned ``_gru_step`` extras excluding persist.

    Nested new ``GRUPerceptionState`` is persist and is already counted in the
    simultaneous persist copies. These extras are the published augmented
    observation ``[obs, new_hidden]``.
    """
    return 4 * (observation_dim + hidden_dim)


def _gru_perception_update_working_set_bytes(
    observation_dim: int, hidden_dim: int
) -> int:
    """Source persist, proposed persist, committed persist, and returned extras."""
    return 3 * _gru_perception_persistent_bytes(
        observation_dim, hidden_dim
    ) + _gru_perception_update_result_extras_bytes(observation_dim, hidden_dim)


def _preflight_gru_perception_update_working_set(
    observation_dim: int, hidden_dim: int
) -> None:
    """Reject an update envelope the host cannot name in signed int32."""
    if _gru_perception_update_working_set_bytes(observation_dim, hidden_dim) > _INT32_MAX:
        raise ValueError(
            "GRUPerception update working set byte count must fit signed int32"
        )


def _copy_mapping(payload: object, *, name: str) -> dict[str, Any]:
    if not issubclass(type(payload), Mapping):
        raise ValueError(f"{name} payload must be a mapping")
    try:
        data = dict(cast(Mapping[str, Any], payload))
    except Exception as error:
        raise ValueError(f"{name} payload could not be read") from error
    for key in data:
        if type(key) is not str:
            raise ValueError(f"{name} payload has exact strings as keys")
    return data


# ---------------------------------------------------------------------------
# GRU Perception (Step 8 sub-component a — recursive state update)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GRUPerceptionConfig:
    """Configuration for the fixed-weight GRU perception layer.

    A minimal echo-state GRU that provides the recursive state-update
    (perception) sub-component required by Alberta Plan Step 8.  Weights are
    sampled once at :meth:`PrototypeAgent.init` and remain fixed; the hidden
    state is updated at every step.  The downstream Q-function (OaK) learns to
    use the temporal context encoded in ``hidden``.

    Args:
        observation_dim: Raw observation dimensionality (GRU input).
        hidden_dim: GRU hidden-state dimensionality (GRU output).

    Note:
        When this config is present, the effective observation dimensionality
        seen by OaK, the Horde, the world model, and IA is
        ``observation_dim + hidden_dim``.  Set ``oak.observation_dim``
        (and ``world_model.observation_dim`` when applicable) accordingly.
    """

    observation_dim: int
    hidden_dim: int = 32

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_dim",
            _require_int("observation_dim", self.observation_dim, minimum=1, maximum=_INT32_MAX),
        )
        object.__setattr__(
            self,
            "hidden_dim",
            _require_int("hidden_dim", self.hidden_dim, minimum=1, maximum=_INT32_MAX),
        )
        # GRU weights: 3*(h*obs + h*h + h) + hidden state h
        h = int(self.hidden_dim)
        obs = int(self.observation_dim)
        _require_float32_resource(
            "GRUPerception state",
            vector_scalars=3 * h * obs + 3 * h * h,
            fixed_scalars=4 * h,
        )
        _preflight_gru_perception_update_working_set(obs, h)

    def augmented_dim(self) -> int:
        """Return ``observation_dim + hidden_dim``."""
        return self.observation_dim + self.hidden_dim

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "type": "GRUPerceptionConfig",
            "observation_dim": self.observation_dim,
            "hidden_dim": self.hidden_dim,
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> GRUPerceptionConfig:
        """Reconstruct from :meth:`to_config` output."""
        data = _copy_mapping(payload, name="GRUPerceptionConfig")
        type_name = data.pop("type", None)
        if type(type_name) is not str or type_name != "GRUPerceptionConfig":
            raise ValueError("GRUPerceptionConfig payload type is invalid")
        if set(data) != {"observation_dim", "hidden_dim"}:
            raise ValueError("GRUPerceptionConfig payload fields are invalid")
        return cls(**data)


@chex.dataclass(frozen=True)
class GRUPerceptionState:
    """State for the fixed-weight GRU perception layer.

    Weight matrices are initialised once and never updated; ``hidden`` is the
    only mutable component and is replaced at every step.

    Attributes:
        W_z, U_z, b_z: Update-gate input/recurrent weights and bias.
        W_r, U_r, b_r: Reset-gate input/recurrent weights and bias.
        W_h, U_h, b_h: Candidate-hidden input/recurrent weights and bias.
        hidden: Running GRU hidden state ``h_t``.
    """

    W_z: Float[Array, "hidden_dim obs_dim"]
    U_z: Float[Array, "hidden_dim hidden_dim"]
    b_z: Float[Array, " hidden_dim"]
    W_r: Float[Array, "hidden_dim obs_dim"]
    U_r: Float[Array, "hidden_dim hidden_dim"]
    b_r: Float[Array, " hidden_dim"]
    W_h: Float[Array, "hidden_dim obs_dim"]
    U_h: Float[Array, "hidden_dim hidden_dim"]
    b_h: Float[Array, " hidden_dim"]
    hidden: Float[Array, " hidden_dim"]


def _glorot_uniform(key: Array, shape: tuple[int, int]) -> Array:
    fan_in, fan_out = shape[-1], shape[0]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jr.uniform(key, shape, dtype=jnp.float32, minval=-limit, maxval=limit)


def _init_gru_state(cfg: GRUPerceptionConfig, key: Array) -> GRUPerceptionState:
    """Glorot-uniform weight init + zero hidden state."""
    keys = jr.split(key, 6)
    d_obs, d_h = cfg.observation_dim, cfg.hidden_dim
    return GRUPerceptionState(
        W_z=_glorot_uniform(keys[0], (d_h, d_obs)),
        U_z=_glorot_uniform(keys[1], (d_h, d_h)),
        b_z=jnp.zeros((d_h,), dtype=jnp.float32),
        W_r=_glorot_uniform(keys[2], (d_h, d_obs)),
        U_r=_glorot_uniform(keys[3], (d_h, d_h)),
        b_r=jnp.zeros((d_h,), dtype=jnp.float32),
        W_h=_glorot_uniform(keys[4], (d_h, d_obs)),
        U_h=_glorot_uniform(keys[5], (d_h, d_h)),
        b_h=jnp.zeros((d_h,), dtype=jnp.float32),
        hidden=jnp.zeros((d_h,), dtype=jnp.float32),
    )


def _gru_step(
    gru: GRUPerceptionState,
    obs: Float[Array, " obs_dim"],
) -> tuple[GRUPerceptionState, Float[Array, " augmented_dim"]]:
    """One GRU step: update hidden state and return augmented observation.

    Returns the *new* GRU state and the concatenation
    ``[obs, new_hidden]`` as the augmented observation.
    """
    h = gru.hidden
    z = jax.nn.sigmoid(gru.W_z @ obs + gru.U_z @ h + gru.b_z)
    r = jax.nn.sigmoid(gru.W_r @ obs + gru.U_r @ h + gru.b_r)
    h_tilde = jnp.tanh(gru.W_h @ obs + gru.U_h @ (r * h) + gru.b_h)
    new_h = (1.0 - z) * h + z * h_tilde
    new_gru = cast(GRUPerceptionState, gru.replace(hidden=new_h))
    return new_gru, jnp.concatenate([obs, new_h], axis=0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _default_oak_config() -> OaKConfig:
    return OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0),),
            observation_dim=4,
            n_primitive_actions=2,
        )
    )


def _sample_one_hot_dream_observation(
    prediction: Array,
    key: Array,
) -> tuple[Array, Array]:
    """Sample a categorical one-hot state from an expectation-valued prediction.

    Finite coordinates are first projected into ``[0, 1]``. Categorical
    sampling normalizes the remaining mass implicitly and preserves exact zero
    support. Non-finite or zero-mass predictions return a safe dummy one-hot
    value together with ``False`` so the caller can reject the backup.
    """

    values = jnp.asarray(prediction, dtype=jnp.float32)
    if values.ndim != 1 or values.shape[0] == 0:
        raise ValueError(
            "sampled dream observation prediction must be a non-empty vector"
        )
    finite = jnp.all(jnp.isfinite(values))
    weights = jnp.clip(
        jnp.where(jnp.isfinite(values), values, jnp.zeros_like(values)),
        0.0,
        1.0,
    )
    valid = finite & (jnp.sum(weights) > 0.0)
    fallback = jax.nn.one_hot(0, values.shape[0], dtype=values.dtype)
    safe_weights = jnp.where(valid, weights, fallback)
    logits = jnp.where(safe_weights > 0.0, jnp.log(safe_weights), -jnp.inf)
    index = jr.categorical(key, logits).astype(jnp.int32)
    sampled = jax.nn.one_hot(index, values.shape[0], dtype=values.dtype)
    return sampled, valid


@dataclasses.dataclass(frozen=True)
class PrototypeAgentConfig:
    """Configuration for the experimental PrototypeAgent composition.

    All components are designed for *continuing* average-reward settings —
    no episode resets, no offline training phases.

    Args:
        oak: OaK agent configuration (steps 5/6/10/11).  Required.
        option_search_control: Optional stateless support-aware Bellman-residual
            search over learned option models. It applies a fixed number of
            planner-only base-value backups at the next decision
            representation without refreshing the dispatch OaK already
            cached. Value changes become behavior-eligible only at the next
            extended-action selection boundary. The legacy STOMP
            option-planning budget must be zero so the two planners cannot
            silently double the work budget.
        world_model: World model configuration (step 8).  When ``None``,
            dreaming is disabled regardless of ``n_dreams_per_step``.
        world_model_ensemble: Mutually exclusive bounded bootstrap ensemble.
            It emits causal typed signals and a pre-update representation
            gradient. Dreaming remains disabled on this development lane until
            uncertainty and rollout-validity gates are calibrated.
        model_replay_rehearsal: Mutually exclusive composition of the bounded
            bootstrap ensemble and fixed-capacity dual replay. Rehearsal updates
            only ensemble member models; its replay samples never update the
            actor, critic, state builder, or causal learning-signal calibrator.
        recurrent_latent_world_model_ensemble: Mutually exclusive recurrent
            GRU ensemble. Its exact dispatched representation/action prediction
            is cached at decision time and consumed once by the authoritative
            transition. Only its real predict-before-update NLL gradient may
            reach the representation mixer; raw uncertainty is uncalibrated.
        dreaming: Dreaming guard configuration (step 9).  Only honoured when
            ``world_model`` is not ``None``.
        buffer_capacity: Number of real observations retained as dream anchors.
        n_dreams_per_step: Dyna-style imagined transitions per real step.
            Zero disables dreaming even when a world model is configured.
        dream_next_observation_mode: How a model-predicted next observation is
            exposed to the base-Q learner. ``"model_prediction"`` preserves
            the regression output. ``"sample_one_hot"`` clips its coordinates
            to ``[0, 1]`` and samples one categorical one-hot state; this mode
            is only valid when the complete control observation is one-hot.
        horde_spec: GVF Horde specification (step 3).  When ``None``, the
            prediction-demon pathway is disabled.
        horde_hidden_sizes: Trunk layer widths for the Horde MLP.
        horde_step_size: Base step-size for the Horde learner.
        ia: IA agent configuration (step 12).  When ``None``, the
            intelligence-amplification companion is disabled.
        partner_policy_fusion: Optional bounded contextual partner-message
            policy fusion. Its action count must match OaK's primitive action
            count and its context width must match OaK's representation width.
            The initial :meth:`start` decision remains OaK-only; the first
            fusion opportunity is the next decision after a real transition.
        experiential_memory: Optional fixed-capacity experiential memory used
            only through its conservative categorical policy boundary. Stored
            observations and keys must match OaK's representation width,
            stored action mass must match the primitive action count, and the
            outcome stores the bootstrap representation plus scalar reward.
        state_builder: Canonical causal representation builder. Its output
            width must equal the feature lifecycle's base width when that lane
            is enabled, and ``oak.observation_dim`` otherwise. Mutually
            exclusive with the deprecated fixed-weight ``gru_perception`` path.
        learn_state_builder_from_world_model: Apply the configured ensemble
            lane's causal, real pre-update representation gradient to an
            online-gated builder. Model replay gradients are never routed here.
            The proposal is formed from the source state that emitted the
            modeled representation and committed into the already-advanced
            destination state, preserving its recurrent history.
        representation_gradient_mixer: Optional successor path that explicitly
            mixes the current transition's causal control TD-loss gradient with
            the real grounded world-model gradient. Its presence enables online
            builder learning without changing the legacy flag. ``behavior_only``
            and ``discard`` can be used without an ensemble; modes with an active
            grounded-world contribution require an ensemble or model-rehearsal
            lane. Replay gradients are never eligible inputs.
        gradient_joy: Historical configuration name for the optional
            multi-objective candidate-update safety audit applied to each
            proposed builder update. Missing or incomplete probe evidence
            vetoes only the representation update; control and model learning
            continue.
        prototype_feature_lifecycle: Optional fixed-budget pair-feature bank
            between a supported base state builder and linear OaK.  This
            narrow lane owns one scalar, owner-bound control-TD discovery
            target and atomically routes every OaK observation axis when a
            pair descriptor changes.  Consumers whose representation state
            is not routed and versioned are rejected by configuration.
    """

    oak: OaKConfig = dataclasses.field(default_factory=_default_oak_config)
    world_model: ActionConditionedWorldModelConfig | None = None
    world_model_ensemble: WorldModelEnsembleConfig | None = None
    model_replay_rehearsal: ModelReplayRehearsalConfig | None = None
    recurrent_latent_world_model_ensemble: (
        RecurrentLatentWorldModelEnsembleConfig | None
    ) = None
    dreaming: DreamingConfig | None = None
    buffer_capacity: int = 200
    n_dreams_per_step: int = 0
    dream_next_observation_mode: Literal[
        "model_prediction",
        "sample_one_hot",
    ] = "model_prediction"
    horde_spec: HordeSpec | None = None
    horde_hidden_sizes: tuple[int, ...] = (64, 64)
    horde_step_size: float = 0.1
    ia: IAConfig | None = None
    partner_policy_fusion: PartnerPolicyFusionConfig | None = None
    experiential_memory: ExperientialMemoryConfig | None = None
    gru_perception: GRUPerceptionConfig | None = None
    state_builder: StateBuilderConfig | None = None
    learn_state_builder_from_world_model: bool = False
    representation_gradient_mixer: RepresentationGradientMixerConfig | None = None
    gradient_joy: GradientJoyConfig | None = None
    auto_curate_every: int = 0
    option_search_control: OptionSearchControlConfig | None = None
    prototype_feature_lifecycle: PrototypeFeatureLifecycleConfig | None = None

    def __post_init__(self) -> None:
        feature_lifecycle = self.prototype_feature_lifecycle
        if feature_lifecycle is not None and not isinstance(
            feature_lifecycle,
            PrototypeFeatureLifecycleConfig,
        ):
            raise ValueError(
                "prototype_feature_lifecycle must be a "
                "PrototypeFeatureLifecycleConfig"
            )
        if self.option_search_control is not None:
            if not isinstance(
                self.option_search_control,
                OptionSearchControlConfig,
            ):
                raise ValueError(
                    "option_search_control must be an OptionSearchControlConfig"
                )
            if self.oak.stomp.option_planning_backups_per_step != 0:
                raise ValueError(
                    "option_search_control requires "
                    "oak.stomp.option_planning_backups_per_step == 0"
                )
        if type(self.dream_next_observation_mode) is not str:
            raise ValueError("dream_next_observation_mode must be a string")
        if self.dream_next_observation_mode not in {
            "model_prediction",
            "sample_one_hot",
        }:
            raise ValueError(
                "dream_next_observation_mode must be "
                "'model_prediction' or 'sample_one_hot'"
            )
        # Hostile-safe canonicalization (must precede any allocation)
        object.__setattr__(
            self,
            "buffer_capacity",
            _require_int("buffer_capacity", self.buffer_capacity, minimum=1, maximum=_INT32_MAX),
        )
        object.__setattr__(
            self,
            "n_dreams_per_step",
            _require_int(
                "n_dreams_per_step",
                self.n_dreams_per_step,
                minimum=0,
                maximum=_INT32_MAX,
            ),
        )
        # horde_hidden_sizes: hostile-safe per-element validation
        raw_hidden = self.horde_hidden_sizes
        if type(raw_hidden) is not tuple:
            raise ValueError("horde_hidden_sizes must be a tuple of integers")
        canonical_hidden: list[int] = []
        for idx, value in enumerate(raw_hidden):
            canonical_hidden.append(
                _require_int(f"horde_hidden_sizes[{idx}]", value, minimum=1, maximum=_INT32_MAX)
            )
        object.__setattr__(self, "horde_hidden_sizes", tuple(canonical_hidden))
        object.__setattr__(
            self,
            "horde_step_size",
            _validated_config_float("horde_step_size", self.horde_step_size, positive=True),
        )
        object.__setattr__(
            self,
            "auto_curate_every",
            _require_int(
                "auto_curate_every",
                self.auto_curate_every,
                minimum=0,
                maximum=_INT32_MAX,
            ),
        )
        if self.world_model is not None:
            _require_float32_resource(
                "PrototypeAgent buffer",
                vector_scalars=self.buffer_capacity * self.oak.observation_dim,
                fixed_scalars=2,
            )
        if self.horde_spec is not None:
            _validate_direct_state_resources(
                len(self.horde_spec.demons),
                self.horde_hidden_sizes,
                self.oak.observation_dim,
            )
        # GRU resource already validated via GRUPerceptionConfig.__post_init__
        if type(self.learn_state_builder_from_world_model) is not bool:
            raise ValueError("learn_state_builder_from_world_model must be boolean")
        mixer = self.representation_gradient_mixer
        if mixer is not None and not isinstance(
            mixer,
            RepresentationGradientMixerConfig,
        ):
            raise ValueError(
                "representation_gradient_mixer must be a "
                "RepresentationGradientMixerConfig"
            )
        configured_model_lanes = sum(
            model is not None
            for model in (
                self.world_model,
                self.world_model_ensemble,
                self.model_replay_rehearsal,
                self.recurrent_latent_world_model_ensemble,
            )
        )
        if configured_model_lanes > 1:
            raise ValueError(
                "world_model, world_model_ensemble, model_replay_rehearsal, and "
                "recurrent_latent_world_model_ensemble are mutually exclusive"
            )
        if self.world_model is None and self.n_dreams_per_step > 0:
            raise ValueError(
                "n_dreams_per_step > 0 requires the legacy world_model to be configured"
            )
        if self.world_model is not None:
            if self.world_model.observation_dim != self.oak.observation_dim:
                raise ValueError(
                    "world_model.observation_dim must match oak.observation_dim, "
                    f"got {self.world_model.observation_dim} and "
                    f"{self.oak.observation_dim}"
                )
            if self.world_model.n_actions != self.oak.n_primitive_actions:
                raise ValueError(
                    "world_model.n_actions must match oak.n_primitive_actions, "
                    f"got {self.world_model.n_actions} and "
                    f"{self.oak.n_primitive_actions}"
                )
        if self.world_model_ensemble is not None:
            ensemble_model = self.world_model_ensemble.model
            if ensemble_model.observation_dim != self.oak.observation_dim:
                raise ValueError(
                    "world_model_ensemble.model.observation_dim must match "
                    "oak.observation_dim, got "
                    f"{ensemble_model.observation_dim} and "
                    f"{self.oak.observation_dim}"
                )
            if ensemble_model.n_actions != self.oak.n_primitive_actions:
                raise ValueError(
                    "world_model_ensemble.model.n_actions must match "
                    "oak.n_primitive_actions, got "
                    f"{ensemble_model.n_actions} and "
                    f"{self.oak.n_primitive_actions}"
                )
            if self.n_dreams_per_step != 0:
                raise ValueError(
                    "dreaming is disabled for world_model_ensemble until its "
                    "uncertainty and rollout-validity gates are calibrated"
                )
        if self.model_replay_rehearsal is not None:
            replay_model = self.model_replay_rehearsal.ensemble.model
            if replay_model.observation_dim != self.oak.observation_dim:
                raise ValueError(
                    "model_replay_rehearsal.ensemble.model.observation_dim must "
                    "match oak.observation_dim, got "
                    f"{replay_model.observation_dim} and {self.oak.observation_dim}"
                )
            if replay_model.n_actions != self.oak.n_primitive_actions:
                raise ValueError(
                    "model_replay_rehearsal.ensemble.model.n_actions must match "
                    "oak.n_primitive_actions, got "
                    f"{replay_model.n_actions} and {self.oak.n_primitive_actions}"
                )
            if self.n_dreams_per_step != 0:
                raise ValueError(
                    "dreaming is disabled for model_replay_rehearsal; replay is "
                    "model-only until rollout-validity gates are calibrated"
                )
        if self.recurrent_latent_world_model_ensemble is not None:
            recurrent_model = self.recurrent_latent_world_model_ensemble
            if recurrent_model.observation_dim != self.oak.observation_dim:
                raise ValueError(
                    "recurrent_latent_world_model_ensemble.observation_dim must "
                    "match oak.observation_dim, got "
                    f"{recurrent_model.observation_dim} and {self.oak.observation_dim}"
                )
            if recurrent_model.n_actions != self.oak.n_primitive_actions:
                raise ValueError(
                    "recurrent_latent_world_model_ensemble.n_actions must match "
                    "oak.n_primitive_actions, got "
                    f"{recurrent_model.n_actions} and {self.oak.n_primitive_actions}"
                )
            if self.n_dreams_per_step != 0:
                raise ValueError(
                    "dreaming is disabled for recurrent_latent_world_model_ensemble; "
                    "its raw uncertainty has no calibrated rollout-validity gate"
                )
        if self.gru_perception is not None and self.state_builder is not None:
            raise ValueError("state_builder and gru_perception are mutually exclusive")
        if self.gru_perception is not None:
            if self.dream_next_observation_mode == "sample_one_hot":
                raise ValueError(
                    "dream_next_observation_mode='sample_one_hot' is incompatible "
                    "with continuous GRU-augmented observations"
                )
            aug = self.gru_perception.augmented_dim()
            if self.oak.observation_dim != aug:
                raise ValueError(
                    f"When gru_perception is set, oak.observation_dim must equal "
                    f"gru_perception.observation_dim + gru_perception.hidden_dim "
                    f"= {aug}, got {self.oak.observation_dim}"
                )
        if self.state_builder is not None:
            builder = state_builder_from_config(self.state_builder.to_config())
            expected_builder_width = (
                feature_lifecycle.base_feature_dim
                if feature_lifecycle is not None
                else self.oak.observation_dim
            )
            if builder.feature_dim() != expected_builder_width:
                raise ValueError(
                    "state_builder feature_dim must match the configured base "
                    f"representation width ({expected_builder_width}), got "
                    f"{builder.feature_dim()}"
                )
            builder_actions = int(getattr(self.state_builder, "n_actions", 0))
            if builder_actions not in (0, self.oak.n_primitive_actions):
                raise ValueError(
                    "state_builder n_actions must be zero or match "
                    f"oak.n_primitive_actions ({self.oak.n_primitive_actions}), "
                    f"got {builder_actions}"
                )
            if (
                self.dream_next_observation_mode == "sample_one_hot"
                and not isinstance(self.state_builder, IdentityStateBuilderConfig)
            ):
                raise ValueError(
                    "dream_next_observation_mode='sample_one_hot' requires "
                    "the raw legacy path or an identity state_builder"
                )
        if self.learn_state_builder_from_world_model and mixer is None:
            if (
                self.world_model_ensemble is None
                and self.model_replay_rehearsal is None
                and self.recurrent_latent_world_model_ensemble is None
            ):
                raise ValueError(
                    "learn_state_builder_from_world_model requires "
                    "world_model_ensemble, model_replay_rehearsal, or "
                    "recurrent_latent_world_model_ensemble"
                )
        state_builder_learning_enabled = (
            self.learn_state_builder_from_world_model or mixer is not None
        )
        if state_builder_learning_enabled:
            if not isinstance(self.state_builder, OnlineGatedStateBuilderConfig):
                raise ValueError(
                    "online representation learning requires an "
                    "OnlineGatedStateBuilderConfig"
                )
        if mixer is not None:
            if mixer.representation_dim != self.oak.observation_dim:
                raise ValueError(
                    "representation_gradient_mixer.representation_dim must match "
                    f"oak.observation_dim ({self.oak.observation_dim}), got "
                    f"{mixer.representation_dim}"
                )
            grounded_world_active = (
                mixer.mode in {"full", "world_only"}
                and mixer.grounded_world_weight > 0.0
            )
            if grounded_world_active and (
                self.world_model_ensemble is None
                and self.model_replay_rehearsal is None
                and self.recurrent_latent_world_model_ensemble is None
            ):
                raise ValueError(
                    "an active grounded-world mixer source requires "
                    "world_model_ensemble, model_replay_rehearsal, or "
                    "recurrent_latent_world_model_ensemble"
                )
        if self.gradient_joy is not None:
            if not state_builder_learning_enabled:
                raise ValueError(
                    "gradient_joy requires online representation learning"
                )
            if (
                self.world_model_ensemble is None
                and self.model_replay_rehearsal is None
                and self.recurrent_latent_world_model_ensemble is None
            ):
                raise ValueError(
                    "gradient_joy requires causal ensemble learning signals"
                )
            if self.gradient_joy.candidate_semantics != "update":
                raise ValueError(
                    "PrototypeAgent gradient_joy must use candidate_semantics='update'"
                )
        if self.ia is not None:
            ia_obs = self.ia.cortex.observation_dim
            oak_obs = self.oak.observation_dim
            if ia_obs != oak_obs:
                raise ValueError(
                    f"ia.cortex.observation_dim ({ia_obs}) must match "
                    f"oak.observation_dim ({oak_obs})"
                )
        if self.partner_policy_fusion is not None:
            fusion = self.partner_policy_fusion
            if not isinstance(fusion, PartnerPolicyFusionConfig):
                raise ValueError(
                    "partner_policy_fusion must be a PartnerPolicyFusionConfig"
                )
            if fusion.n_actions != self.oak.n_primitive_actions:
                raise ValueError(
                    "partner_policy_fusion.n_actions must match "
                    f"oak.n_primitive_actions ({self.oak.n_primitive_actions}), "
                    f"got {fusion.n_actions}"
                )
            if fusion.context_dim != self.oak.observation_dim:
                raise ValueError(
                    "partner_policy_fusion.context_dim must match "
                    f"oak.observation_dim ({self.oak.observation_dim}), "
                    f"got {fusion.context_dim}"
                )
        if self.experiential_memory is not None:
            memory = self.experiential_memory
            if type(memory) is not ExperientialMemoryConfig:
                raise ValueError(
                    "experiential_memory must be an exact ExperientialMemoryConfig"
                )
            # ExperientialMemoryConfig predates dataclass-level validation;
            # constructing the bounded substrate enforces its complete static
            # contract while retaining the caller's exact config object.
            ExperientialMemory(memory)
            representation_dim = self.oak.observation_dim
            if memory.observation_dim != representation_dim:
                raise ValueError(
                    "experiential_memory.observation_dim must match "
                    f"oak.observation_dim ({representation_dim}), got "
                    f"{memory.observation_dim}"
                )
            if memory.key_dim != representation_dim:
                raise ValueError(
                    "experiential_memory.key_dim must match "
                    f"oak.observation_dim ({representation_dim}), got "
                    f"{memory.key_dim}"
                )
            if memory.action_dim != self.oak.n_primitive_actions:
                raise ValueError(
                    "experiential_memory.action_dim must match "
                    "oak.n_primitive_actions "
                    f"({self.oak.n_primitive_actions}), got {memory.action_dim}"
                )
            expected_outcome_dim = representation_dim + 1
            if memory.outcome_dim != expected_outcome_dim:
                raise ValueError(
                    "experiential_memory.outcome_dim must equal "
                    "oak.observation_dim + 1 "
                    f"({expected_outcome_dim}), got {memory.outcome_dim}"
                )

        if feature_lifecycle is not None:
            if feature_lifecycle.n_tasks != 1:
                raise ValueError(
                    "prototype_feature_lifecycle.n_tasks must equal 1"
                )
            if feature_lifecycle.n_options != self.oak.n_options:
                raise ValueError(
                    "prototype_feature_lifecycle.n_options must match "
                    "oak.n_options"
                )
            if (
                feature_lifecycle.n_primitive_actions
                != self.oak.n_primitive_actions
            ):
                raise ValueError(
                    "prototype_feature_lifecycle.n_primitive_actions must "
                    "match oak.n_primitive_actions"
                )
            if feature_lifecycle.total_feature_dim != self.oak.observation_dim:
                raise ValueError(
                    "prototype_feature_lifecycle.total_feature_dim must match "
                    "oak.observation_dim"
                )
            if self.oak.stomp.base_hidden_sizes != ():
                raise ValueError(
                    "prototype_feature_lifecycle requires linear OaK with "
                    "oak.stomp.base_hidden_sizes == ()"
                )
            actual_subtask_indices = tuple(
                spec.feature_index for spec in self.oak.stomp.subtask_specs
            )
            if (
                feature_lifecycle.option_subtask_feature_indices
                != actual_subtask_indices
            ):
                raise ValueError(
                    "prototype_feature_lifecycle option subtask indices must "
                    "exactly match oak.stomp.subtask_specs"
                )
            PrototypeFeatureLifecycle(
                feature_lifecycle
            ).require_compatible_oak_config(self.oak)
            if type(self.state_builder) not in {
                IdentityStateBuilderConfig,
                OnlineGatedStateBuilderConfig,
            }:
                raise ValueError(
                    "prototype_feature_lifecycle requires an Identity or "
                    "OnlineGated state_builder"
                )
            if self.gru_perception is not None:
                raise ValueError(
                    "prototype_feature_lifecycle is incompatible with "
                    "gru_perception"
                )
            unsupported_consumers = (
                self.world_model is not None
                or self.world_model_ensemble is not None
                or self.model_replay_rehearsal is not None
                or self.recurrent_latent_world_model_ensemble is not None
                or self.dreaming is not None
                or self.n_dreams_per_step != 0
                or self.horde_spec is not None
                or self.ia is not None
                or self.partner_policy_fusion is not None
                or self.experiential_memory is not None
            )
            if unsupported_consumers:
                raise ValueError(
                    "prototype_feature_lifecycle rejects world-model, dreaming, "
                    "Horde, IA, partner-fusion, and experiential-memory consumers"
                )
            if self.learn_state_builder_from_world_model:
                raise ValueError(
                    "prototype_feature_lifecycle rejects world-model builder learning"
                )
            if self.gradient_joy is not None:
                raise ValueError(
                    "prototype_feature_lifecycle rejects gradient_joy until "
                    "its evidence is representation-versioned"
                )
            if (
                self.representation_gradient_mixer is not None
                and self.representation_gradient_mixer.mode
                not in {"behavior_only", "discard"}
            ):
                raise ValueError(
                    "prototype_feature_lifecycle supports only behavior_only "
                    "or discard representation-gradient mixing"
                )
            if self.auto_curate_every != 0:
                raise ValueError(
                    "prototype_feature_lifecycle requires auto_curate_every == 0"
                )

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        payload: dict[str, Any] = {
            "type": "PrototypeAgentConfig",
            "oak": self.oak.to_config(),
            "buffer_capacity": self.buffer_capacity,
            "n_dreams_per_step": self.n_dreams_per_step,
            "horde_hidden_sizes": list(self.horde_hidden_sizes),
            "horde_step_size": self.horde_step_size,
            "auto_curate_every": self.auto_curate_every,
        }
        if self.dream_next_observation_mode != "model_prediction":
            payload["dream_next_observation_mode"] = self.dream_next_observation_mode
        if self.option_search_control is not None:
            payload["option_search_control"] = (
                self.option_search_control.to_config()
            )
        if self.prototype_feature_lifecycle is not None:
            payload["prototype_feature_lifecycle"] = (
                self.prototype_feature_lifecycle.to_config()
            )
        if self.world_model is not None:
            payload["world_model"] = self.world_model.to_config()
        if self.world_model_ensemble is not None:
            payload["world_model_ensemble"] = self.world_model_ensemble.to_config()
        if self.model_replay_rehearsal is not None:
            payload["model_replay_rehearsal"] = (
                self.model_replay_rehearsal.to_config()
            )
        if self.recurrent_latent_world_model_ensemble is not None:
            payload["recurrent_latent_world_model_ensemble"] = (
                self.recurrent_latent_world_model_ensemble.to_config()
            )
        if self.dreaming is not None:
            payload["dreaming"] = self.dreaming.to_config()
        if self.horde_spec is not None:
            payload["horde_spec"] = self.horde_spec.to_config()
        if self.ia is not None:
            payload["ia"] = self.ia.to_config()
        if self.partner_policy_fusion is not None:
            payload["partner_policy_fusion"] = (
                self.partner_policy_fusion.to_config()
            )
        if self.experiential_memory is not None:
            payload["experiential_memory"] = self.experiential_memory.to_config()
        if self.gru_perception is not None:
            payload["gru_perception"] = self.gru_perception.to_config()
        if self.state_builder is not None:
            payload["state_builder"] = self.state_builder.to_config()
        if self.learn_state_builder_from_world_model:
            payload["learn_state_builder_from_world_model"] = True
        if self.representation_gradient_mixer is not None:
            payload["representation_gradient_mixer"] = (
                self.representation_gradient_mixer.to_config()
            )
        if self.gradient_joy is not None:
            payload["gradient_joy"] = self.gradient_joy.to_config()
        return payload

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> PrototypeAgentConfig:
        """Reconstruct from :meth:`to_config` output."""
        from alberta_framework.core.types import HordeSpec as _HordeSpec

        data = _copy_mapping(payload, name="PrototypeAgentConfig")
        config_type = data.pop("type", None)
        if type(config_type) is not str or config_type != "PrototypeAgentConfig":
            raise ValueError(
                "PrototypeAgentConfig payload type must be 'PrototypeAgentConfig'"
            )
        oak = OaKConfig.from_config(cast(dict[str, Any], data.pop("oak")))

        option_search_raw = data.pop("option_search_control", None)
        if option_search_raw is not None and not isinstance(
            option_search_raw,
            dict,
        ):
            raise ValueError("option_search_control must be a configuration object")
        option_search_control = (
            OptionSearchControlConfig.from_config(option_search_raw)
            if option_search_raw is not None
            else None
        )

        feature_lifecycle_raw = data.pop("prototype_feature_lifecycle", None)
        if feature_lifecycle_raw is not None and not isinstance(
            feature_lifecycle_raw,
            dict,
        ):
            raise ValueError(
                "prototype_feature_lifecycle must be a configuration object"
            )
        prototype_feature_lifecycle = (
            PrototypeFeatureLifecycleConfig.from_config(feature_lifecycle_raw)
            if feature_lifecycle_raw is not None
            else None
        )

        wm_raw = data.pop("world_model", None)
        world_model = (
            ActionConditionedWorldModelConfig.from_config(wm_raw) if wm_raw is not None else None
        )
        ensemble_raw = data.pop("world_model_ensemble", None)
        world_model_ensemble = (
            WorldModelEnsembleConfig.from_config(ensemble_raw)
            if ensemble_raw is not None
            else None
        )
        model_replay_raw = data.pop("model_replay_rehearsal", None)
        model_replay_rehearsal = (
            ModelReplayRehearsalConfig.from_config(model_replay_raw)
            if model_replay_raw is not None
            else None
        )
        recurrent_latent_raw = data.pop(
            "recurrent_latent_world_model_ensemble",
            None,
        )
        recurrent_latent_world_model_ensemble = (
            RecurrentLatentWorldModelEnsembleConfig.from_config(
                recurrent_latent_raw
            )
            if recurrent_latent_raw is not None
            else None
        )
        dream_raw = data.pop("dreaming", None)
        dreaming = DreamingConfig.from_config(dream_raw) if dream_raw is not None else None
        horde_raw = data.pop("horde_spec", None)
        horde_spec = _HordeSpec.from_config(horde_raw) if horde_raw is not None else None
        ia_raw = data.pop("ia", None)
        ia = IAConfig.from_config(ia_raw) if ia_raw is not None else None
        partner_fusion_raw = data.pop("partner_policy_fusion", None)
        if partner_fusion_raw is not None and not isinstance(
            partner_fusion_raw,
            dict,
        ):
            raise ValueError("partner_policy_fusion must be a configuration object")
        partner_policy_fusion = (
            PartnerPolicyFusionConfig.from_config(partner_fusion_raw)
            if partner_fusion_raw is not None
            else None
        )
        experiential_memory_raw = data.pop("experiential_memory", None)
        if experiential_memory_raw is not None and type(experiential_memory_raw) is not dict:
            raise ValueError("experiential_memory must be a configuration object")
        experiential_memory = (
            ExperientialMemoryConfig.from_config(experiential_memory_raw)
            if experiential_memory_raw is not None
            else None
        )
        gru_raw = data.pop("gru_perception", None)
        gru_perception = (
            GRUPerceptionConfig.from_config(gru_raw) if gru_raw is not None else None
        )
        state_builder_raw = data.pop("state_builder", None)
        state_builder: StateBuilderConfig | None = None
        if state_builder_raw is not None:
            from alberta_framework.core.state_builder import (
                state_builder_config_from_config,
            )

            state_builder = state_builder_config_from_config(state_builder_raw)

        learn_state_builder_from_world_model = data.pop(
            "learn_state_builder_from_world_model",
            False,
        )
        if type(learn_state_builder_from_world_model) is not bool:
            raise ValueError(
                "learn_state_builder_from_world_model must be boolean"
            )
        mixer_raw = data.pop("representation_gradient_mixer", None)
        if mixer_raw is not None and not isinstance(mixer_raw, dict):
            raise ValueError(
                "representation_gradient_mixer must be a configuration object"
            )
        representation_gradient_mixer = (
            RepresentationGradientMixerConfig.from_config(mixer_raw)
            if mixer_raw is not None
            else None
        )
        gradient_joy_raw = data.pop("gradient_joy", None)
        if gradient_joy_raw is not None and not isinstance(gradient_joy_raw, dict):
            raise ValueError("gradient_joy must be a configuration object")
        if (
            gradient_joy_raw is not None
            and gradient_joy_raw.get("type") != "GradientJoyConfig"
        ):
            raise ValueError("gradient_joy type must be 'GradientJoyConfig'")
        gradient_joy = (
            GradientJoyConfig.from_config(gradient_joy_raw)
            if gradient_joy_raw is not None
            else None
        )

        # Hostile-safe scalar canonicalization for serialized fields
        hidden_raw = data.pop("horde_hidden_sizes", [64, 64])
        if type(hidden_raw) is not list:
            raise ValueError("serialized horde_hidden_sizes must be an actual list")
        hidden_list: list[int] = []
        for idx, value in enumerate(hidden_raw):
            hidden_list.append(
                _require_int(f"horde_hidden_sizes[{idx}]", value, minimum=1, maximum=_INT32_MAX)
            )
        hidden = tuple(hidden_list)
        buffer_capacity = _require_int(
            "buffer_capacity", data.pop("buffer_capacity", 200), minimum=1, maximum=_INT32_MAX
        )
        n_dreams_per_step = _require_int(
            "n_dreams_per_step", data.pop("n_dreams_per_step", 0), minimum=0, maximum=_INT32_MAX
        )
        dream_next_observation_mode = data.pop(
            "dream_next_observation_mode",
            "model_prediction",
        )
        if type(dream_next_observation_mode) is not str:
            raise ValueError("dream_next_observation_mode must be a string")
        if dream_next_observation_mode not in {"model_prediction", "sample_one_hot"}:
            raise ValueError(
                "dream_next_observation_mode must be 'model_prediction' or 'sample_one_hot'"
            )
        horde_step_size = _validated_config_float(
            "horde_step_size", data.pop("horde_step_size", 0.1), positive=True
        )
        auto_curate_every = _require_int(
            "auto_curate_every", data.pop("auto_curate_every", 0), minimum=0, maximum=_INT32_MAX
        )
        if data:
            unknown = ", ".join(sorted(str(k) for k in data))
            raise ValueError(f"PrototypeAgentConfig payload has unknown fields: {unknown}")
        return cls(
            oak=oak,
            option_search_control=option_search_control,
            world_model=world_model,
            world_model_ensemble=world_model_ensemble,
            model_replay_rehearsal=model_replay_rehearsal,
            recurrent_latent_world_model_ensemble=(
                recurrent_latent_world_model_ensemble
            ),
            dreaming=dreaming,
            buffer_capacity=buffer_capacity,
            n_dreams_per_step=n_dreams_per_step,
            dream_next_observation_mode=dream_next_observation_mode,
            horde_spec=horde_spec,
            horde_hidden_sizes=hidden,
            horde_step_size=horde_step_size,
            ia=ia,
            partner_policy_fusion=partner_policy_fusion,
            experiential_memory=experiential_memory,
            gru_perception=gru_perception,
            state_builder=state_builder,
            learn_state_builder_from_world_model=(
                learn_state_builder_from_world_model
            ),
            representation_gradient_mixer=representation_gradient_mixer,
            gradient_joy=gradient_joy,
            auto_curate_every=auto_curate_every,
            prototype_feature_lifecycle=prototype_feature_lifecycle,
        )


# ---------------------------------------------------------------------------
# State and result types
# ---------------------------------------------------------------------------


@chex.dataclass(frozen=True)
class PrototypeTransition:
    """Explicit real-transition contract for :class:`PrototypeAgent`.

    ``discount`` is the effective scalar continuation multiplier for control
    and world-model learning.  It must be finite and in ``[0, 1]``; use zero
    for an environmental terminal transition.  Unlike the legacy
    :meth:`PrototypeAgent.update` surface, it is never synthesized from an
    agent configuration.

    Horde questions retain their own horizons.  ``horde_discounts`` may
    provide one effective discount per GVF.  When it is omitted, the adapter
    uses each demon's configured gamma on nonterminal transitions and zeros
    every demon only when the global ``discount`` is zero.  Consequently,
    callers that need fractional environment continuation to alter GVF
    horizons must supply ``horde_discounts`` explicitly.

    Attributes:
        observation: Raw observation on which ``action`` was selected.
        action: Primitive command actually dispatched to the environment.
        decision_id: Four-word session/generation token issued with that exact decision.
            This prevents a stale transition from being accepted when a raw
            observation/action pair later recurs (the ABA problem).
        reward: Scalar reward emitted by the environment.
        discount: Effective scalar continuation multiplier.
        terminated: Environment-defined terminal-state flag. A true value
            requires an explicit zero discount.
        truncated: Exogenous/time-limit boundary flag. Without simultaneous
            termination it requires a positive explicit bootstrap discount.
        next_observation: Raw final/bootstrap observation reached by the
            transition. Learning targets consume this observation.
        next_decision_observation: Raw observation on which the next command
            will be selected. It must equal ``next_observation`` off a
            boundary; an autoreset transition supplies the reset observation.
        horde_cumulants: Optional vector of one cumulant per GVF demon.
            ``NaN`` keeps the framework's existing inactive-demon semantics.
        horde_discounts: Optional vector of effective per-GVF discounts.
    """

    observation: Float[Array, " raw_observation_dim"]
    action: Int[Array, ""]
    decision_id: UInt[Array, " 4"]
    reward: Float[Array, ""]
    discount: Float[Array, ""]
    terminated: Bool[Array, ""]
    truncated: Bool[Array, ""]
    next_observation: Float[Array, " raw_observation_dim"]
    next_decision_observation: Float[Array, " raw_observation_dim"]
    horde_cumulants: Any = None
    horde_discounts: Any = None


@chex.dataclass(frozen=True)
class PrototypeFeatureRepresentationState:
    """Enabled-only composition inside the historical builder-state slot.

    Disabled configurations retain the exact pre-existing
    ``state_builder_state`` PyTree.  The wrapper exists only when the bounded
    Prototype feature lifecycle is configured.
    """

    builder_state: Any
    feature_lifecycle_state: PrototypeFeatureLifecycleState


@chex.dataclass(frozen=True)
class PrototypeFeatureOaKState:
    """Enabled-only OaK subtree coupled to its exact feature-bank identity."""

    oak_state: OaKState
    consumer_binding: PrototypeFeatureConsumerBinding


@chex.dataclass(frozen=True)
class PrototypeAgentState:
    """Full prototype agent state.

    Optional sub-states (``world_model_state``, ``buffer_state``,
    ``horde_state``, ``ia_state``) are ``None`` when the corresponding
    component is disabled in the configuration.  The PyTree structure is
    fixed for a given :class:`PrototypeAgent` instance — never switch between
    ``None`` and a real state after initialisation.
    """

    oak_state: Any  # OaKState | PrototypeFeatureOaKState
    world_model_state: Any  # ActionConditionedWorldModelState | None
    buffer_state: Any  # RecentObservationBufferState | None
    horde_state: Any  # MultiHeadMLPState | None
    ia_state: Any  # IAState | None
    gru_state: Any  # GRUPerceptionState | None
    state_builder_state: Any
    current_raw_observation: Float[Array, " raw_observation_dim"]
    current_representation: Float[Array, " observation_dim"]
    current_action: Int[Array, ""]
    current_decision_id: UInt[Array, " 4"]
    started: Bool[Array, ""]
    observation_event_count: Int[Array, ""]
    step_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeInteractionState:
    """Opt-in IA-slot composition for partner fusion.

    This wrapper is created only when ``partner_policy_fusion`` is configured.
    Consequently disabled and historical IA-only states retain their exact
    ``None``/``IAState`` PyTree shape. ``ia_state`` may be ``None`` when fusion
    is enabled without the intelligence-amplification companion.
    """

    ia_state: Any
    partner_policy_fusion_state: PartnerPolicyFusionState
    feedback_prototype_decision_id: UInt[Array, " 4"]
    feedback_prototype_decision_id_available: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeMemoryInteractionState:
    """Opt-in outer wrapper for persistent experiential memory.

    ``interaction_state`` is the exact state that would occupy ``ia_state``
    without memory: ``None``, a bare IA state, or the partner-fusion
    :class:`PrototypeInteractionState`. Consequently every no-memory lane
    retains its historical PyTree exactly.
    """

    interaction_state: Any
    experiential_memory_state: ExperientialMemoryState


@chex.dataclass(frozen=True)
class PrototypeRecurrentLatentWorldModelState:
    """Recurrent lane state stored inside the existing world-model slot.

    ``decision_cache`` owns exactly the currently dispatched Prototype
    representation/action. ``signal_state`` consumes only the same cached
    predict-before-update event as ``model_state``. The wrapper keeps the
    top-level :class:`PrototypeAgentState` shape and every disabled/legacy
    checkpoint shape unchanged.
    """

    model_state: RecurrentLatentWorldModelEnsembleState
    decision_cache: RecurrentLatentDecisionCache
    signal_state: LearningSignalEstimatorState


@chex.dataclass(frozen=True)
class _WorldModelEnsembleStateV1:
    """Exact pre-rehearsal ensemble subtree used only for v2 migration."""

    member_states: tuple[Any, ...]
    residual_variances: Any
    signal_state: Any
    bootstrap_key: Array
    last_bootstrap_mask: Any
    member_update_counts: Any
    event_count: Any


@chex.dataclass(frozen=True)
class _PrototypeAgentStateV1:
    """Exact pre-cache checkpoint template used only for v1 migration."""

    oak_state: OaKState
    world_model_state: Any
    buffer_state: Any
    horde_state: Any
    ia_state: Any
    gru_state: Any
    step_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeDecision:
    """Dispatch record that must be returned with the resulting transition."""

    observation: Float[Array, " raw_observation_dim"]
    action: Int[Array, ""]
    decision_id: UInt[Array, " 4"]
    armed: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeTransitionDiagnostics:
    """Fail-closed ownership and boundary checks for one real transition."""

    started: Bool[Array, ""]
    inputs_finite: Bool[Array, ""]
    action_in_range: Bool[Array, ""]
    observation_matches: Bool[Array, ""]
    action_matches: Bool[Array, ""]
    decision_id_matches: Bool[Array, ""]
    next_generation_available: Bool[Array, ""]
    next_counter_capacity_available: Bool[Array, ""]
    discount_valid: Bool[Array, ""]
    boundary_semantics_valid: Bool[Array, ""]
    state_consistent: Bool[Array, ""]
    post_update_checked: Bool[Array, ""]
    post_update_finite: Bool[Array, ""]
    post_update_consistent: Bool[Array, ""]
    valid: Bool[Array, ""]
    rejected: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeRecurrentLatentDiagnostics:
    """Causal recurrent-lane transaction and uncertainty disclosures.

    The two uncertainty values are raw ensemble outputs, not calibrated
    probabilities or confidence intervals. ``prediction_availability`` honors
    the recurrent model's configured warmup. ``transaction_applied`` requires
    both the recurrent update and its causal signal-state update to commit.
    """

    model: RecurrentLatentWorldModelDiagnostics
    prediction_availability: RecurrentLatentPredictionAvailability
    raw_epistemic_disagreement: Float[Array, ""]
    raw_aleatoric_uncertainty: Float[Array, ""]
    raw_uncertainty_calibrated: Bool[Array, ""]
    signals_valid: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    next_decision_cached: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeBehaviorGradientDiagnostics:
    """Causal provenance and numerics for one control-loss input gradient.

    Exactly one source flag is true on an available result. ``idle_base_source``
    denotes the one-step primitive differential base-Q loss. During an option,
    ``intra_option_source`` denotes the current option's one-step TD loss; the
    delayed semi-MDP base update belongs to ``option_start_obs`` and is
    deliberately excluded from the current representation.
    """

    source_available: Bool[Array, ""]
    idle_base_source: Bool[Array, ""]
    intra_option_source: Bool[Array, ""]
    parameters_finite: Bool[Array, ""]
    inputs_finite: Bool[Array, ""]
    source_indices_valid: Bool[Array, ""]
    prediction_finite: Bool[Array, ""]
    target_finite: Bool[Array, ""]
    td_error_finite: Bool[Array, ""]
    loss_finite: Bool[Array, ""]
    gradient_norm_finite: Bool[Array, ""]
    prediction: Float[Array, ""]
    target: Float[Array, ""]
    td_error: Float[Array, ""]
    loss: Float[Array, ""]
    gradient_norm: Float[Array, ""]
    bootstrap_discount: Float[Array, ""]
    option_terminates: Bool[Array, ""]
    gradient_finite: Bool[Array, ""]
    valid: Bool[Array, ""]
    rejected: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeBehaviorGradientResult:
    """Pre-mix gradient of one frozen-target TD loss.

    ``raw`` here means before source weighting/mixing. A rejected numerical
    computation is represented by an exact finite zero plus ``valid=False``.
    """

    gradient: Float[Array, " observation_dim"]
    valid: Bool[Array, ""]
    diagnostics: PrototypeBehaviorGradientDiagnostics


@chex.dataclass(frozen=True)
class PrototypeFeatureLifecycleIntegrationDiagnostics:
    """Compact audit of one Prototype-owned discovery transaction."""

    available: Bool[Array, ""]
    target: Float[Array, ""]
    target_available: Bool[Array, ""]
    pullback_gradient: Float[Array, " base_feature_dim"]
    pullback_valid: Bool[Array, ""]
    pullback_semantic_generation: Int[Array, ""]
    prediction: Float[Array, ""]
    error: Float[Array, ""]
    metrics: Float[Array, " 7"]
    lifecycle: PrototypeFeatureLifecycleDiagnostics
    outer_transaction_committed: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeGradientJoyEvidence:
    """Decision-bound probe evidence for one representation update.

    The three parameter-gradient probes must be measured independently of the
    world-model gradient that forms the candidate update. ``advantage`` and
    ``action_surprisal`` come from the behavior actor, while ``safety_cost``
    comes from the separately trained safety learner. The remaining four
    learning-value channels are supplied causally by the configured world
    model ensemble. The paper-defined delight value is derived internally as
    ``advantage * action_surprisal`` and is never accepted from a caller.

    Every availability flag is explicit. Missing evidence, a stale decision
    ID, a false independence attestation, or any unavailable/non-finite input
    causes the candidate-update audit to fail closed without invalidating the
    real control transition. This class name is retained for API compatibility;
    it does not represent the paper's Kondo backward-selection decision.
    """

    decision_id: UInt[Array, " 4"]
    objective_probe_gradient: Float[Array, " parameter_count"]
    retention_probe_gradient: Float[Array, " parameter_count"]
    safety_cost_gradient: Float[Array, " parameter_count"]
    objective_probe_available: Bool[Array, ""]
    retention_probe_available: Bool[Array, ""]
    safety_probe_available: Bool[Array, ""]
    probe_independence_attested: Bool[Array, ""]
    advantage: Float[Array, ""]
    action_surprisal: Float[Array, ""]
    safety_cost: Float[Array, ""]
    advantage_available: Bool[Array, ""]
    action_surprisal_available: Bool[Array, ""]
    safety_cost_available: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeExperientialMemoryInput:
    """Learner-visible metadata for one causal memory query/write event.

    Both lifecycle identifiers are full Prototype tokens. The current token
    binds the automatically grounded exemplar to the action that really ran;
    the next token binds the retrieval to the decision it may alter. Opaque
    provenance/source identifiers are not task, regime, evaluator, or target
    labels. Static shape/dtype drift raises; dynamic invalidity is an exact
    memory no-op and ordinary OaK decision.
    """

    available: Bool[Array, ""]
    current_prototype_decision_id: UInt[Array, " 4"]
    next_prototype_decision_id: UInt[Array, " 4"]
    query_representation_version: Int[Array, ""]
    entry_representation_version: Int[Array, ""]
    query_uncertainty: Float[Array, ""]
    query_uncertainty_available: Bool[Array, ""]
    entry_uncertainty: Float[Array, ""]
    entry_uncertainty_available: Bool[Array, ""]
    safety_cost: Float[Array, ""]
    safety_cost_available: Bool[Array, ""]
    reliability: Float[Array, ""]
    utility: Float[Array, ""]
    utility_available: Bool[Array, ""]
    provenance_id: Int[Array, ""]
    source_id: Int[Array, ""]
    next_action_safety_mask: Bool[Array, " actions"]


@chex.dataclass(frozen=True)
class PrototypeExperientialMemoryDiagnostics:
    """Complete causal memory, categorical proposal, and dispatch audit."""

    proposal: ExperientialMemoryPolicyProposal
    dispatch_replacement: DispatchedPrimitiveActionDecision
    input_supplied: Bool[Array, ""]
    input_available: Bool[Array, ""]
    current_prototype_decision_id_matches: Bool[Array, ""]
    next_prototype_decision_id_matches: Bool[Array, ""]
    metadata_valid: Bool[Array, ""]
    transaction_required: Bool[Array, ""]
    retrieval_matches: Bool[Array, ""]
    query_before_write: Bool[Array, ""]
    deterministic_prestate_query_count: Int[Array, ""]
    wrote: Bool[Array, ""]
    slot: Int[Array, ""]
    evicted: Bool[Array, ""]
    evicted_provenance_id: Int[Array, ""]
    transaction_applied: Bool[Array, ""]
    counterfactual_base_action: Int[Array, ""]
    effective_action: Int[Array, ""]


@dataclasses.dataclass(frozen=True)
class PrototypeExperientialMemoryResourceDeclaration:
    """Exact persistent allocation and bounded work per required transaction.

    ``categorical_policy_queries`` is the one query shared by proposal and
    causal access/write accounting. ``causal_step_queries`` counts only an
    additional query and is therefore zero for the fused transaction.
    """

    memory_capacity: int
    memory_observation_dim: int
    memory_key_dim: int
    memory_action_dim: int
    memory_outcome_dim: int
    persistent_state_bytes: int
    categorical_policy_queries: int
    causal_step_queries: int
    total_deterministic_prestate_queries: int
    writes_attempted: int
    random_draws: int
    score_mass_values_interpreted: int
    hard_safety_values_interpreted: int

    def __post_init__(self) -> None:
        if type(self) is not PrototypeExperientialMemoryResourceDeclaration:
            raise TypeError(
                "declaration must be an exact "
                "PrototypeExperientialMemoryResourceDeclaration"
            )
        for name in (
            "memory_capacity",
            "memory_observation_dim",
            "memory_key_dim",
            "memory_action_dim",
            "memory_outcome_dim",
            "persistent_state_bytes",
            "categorical_policy_queries",
            "causal_step_queries",
            "total_deterministic_prestate_queries",
            "writes_attempted",
            "random_draws",
            "score_mass_values_interpreted",
            "hard_safety_values_interpreted",
        ):
            object.__setattr__(
                self,
                name,
                _require_int(
                    name,
                    getattr(self, name),
                    minimum=0,
                    maximum=_INT32_MAX,
                ),
            )
        for name in (
            "memory_capacity",
            "memory_observation_dim",
            "memory_key_dim",
            "memory_action_dim",
            "memory_outcome_dim",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")

        vector_values = (
            self.memory_observation_dim
            + self.memory_key_dim
            + self.memory_action_dim
            + self.memory_outcome_dim
        )
        expected_persistent_bytes = (
            self.memory_capacity * (4 * (vector_values + 11) + 4) + 32
        )
        expected = {
            "persistent_state_bytes": expected_persistent_bytes,
            "categorical_policy_queries": 1,
            "causal_step_queries": 0,
            "total_deterministic_prestate_queries": 1,
            "writes_attempted": 1,
            "random_draws": 0,
            "score_mass_values_interpreted": self.memory_action_dim,
            "hard_safety_values_interpreted": self.memory_action_dim,
        }
        for name, expected_value in expected.items():
            if getattr(self, name) != expected_value:
                raise ValueError(f"{name} does not match its exact derived identity")

    def to_config(self) -> dict[str, int]:
        """Return the exact JSON-compatible resource declaration."""

        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class PrototypePartnerPolicyFusionInput:
    """Typed sidecar for the next primitive-action decision.

    The Prototype derives the fusion ``decision_id`` and ``event_id`` from its
    authoritative counters. Messages must bind those exact identifiers. The
    remaining integer references are opaque provenance/context identities, not
    learner-visible task or regime labels. Dynamic invalidity safely disables
    fusion and preserves OaK's counterfactual base primitive; malformed static
    shapes or dtypes raise before traced execution.
    """

    available: Bool[Array, ""]
    prototype_decision_id: UInt[Array, " 4"]
    observation_id: Int[Array, ""]
    context_id: Int[Array, ""]
    context_features: Float[Array, " context_dim"]
    safety_action_mask: Bool[Array, " actions"]
    keyboard_available: Bool[Array, ""]
    keyboard_vector: Float[Array, " options"]
    messages: PartnerMessageBatch


@chex.dataclass(frozen=True)
class PrototypePartnerPolicyFusionFeedback:
    """Realized partner feedback bound to the full Prototype lifecycle ID."""

    prototype_decision_id: UInt[Array, " 4"]
    feedback: PartnerPolicyFusionFeedback


@chex.dataclass(frozen=True)
class PrototypePartnerPolicyFusionDiagnostics:
    """Complete feedback, fusion, and dispatch audit for one transition."""

    feedback: PartnerFusionFeedbackResult
    decision: PartnerFusionDecision
    keyboard_proposal: OaKKeyboardPolicyProposal
    dispatch_replacement: DispatchedPrimitiveActionDecision
    feedback_input_supplied: Bool[Array, ""]
    decision_input_supplied: Bool[Array, ""]
    feedback_prototype_decision_id_matches: Bool[Array, ""]
    decision_prototype_decision_id_matches: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    counterfactual_base_action: Int[Array, ""]
    effective_action: Int[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeUpdateResult:
    """Result of one real-time prototype agent transition.

    The ``model_replay_*`` fields describe only work committed by the complete
    all-or-nothing model/replay transaction. They remain false/zero for the
    plain ensemble and legacy model lanes and for any rolled-back rehearsal.
    Candidate audit acceptance and committed application are deliberately
    separate finite-precision facts. The ``sparks_joy`` and
    ``joyful_gradient_applied`` properties are historical compatibility names;
    paper-defined “sparks joy” refers to Kondo backward selection.
    """

    state: PrototypeAgentState
    action: Int[Array, ""]
    oak_td_error: Float[Array, ""]
    oak_average_reward: Float[Array, ""]
    world_model_error: Any  # Float[Array, ""] | None
    learning_signals: TypedLearningSignals
    world_model_representation_gradient: Float[Array, " observation_dim"]
    world_model_representation_gradient_valid: Bool[Array, ""]
    behavior_gradient_result: PrototypeBehaviorGradientResult
    representation_gradient_mix: RepresentationGradientMixResult
    world_model_ensemble_diagnostics: WorldModelEnsembleDiagnostics
    recurrent_latent_world_model_diagnostics: PrototypeRecurrentLatentDiagnostics
    model_replay_transaction_applied: Bool[Array, ""]
    model_replay_recorded: Bool[Array, ""]
    model_replay_sampled: Bool[Array, ""]
    model_replay_updates_applied: Int[Array, ""]
    model_replay_padding_count: Int[Array, ""]
    state_builder_learning_diagnostics: StateBuilderLearningDiagnostics
    gradient_joy_application: Any  # GradientJoyApplicationResult | None
    gradient_joy_evidence_supplied: Bool[Array, ""]
    gradient_joy_decision_id_matches: Bool[Array, ""]
    option_search_control_diagnostics: OptionSearchControlDiagnostics | None
    prototype_feature_lifecycle_diagnostics: (
        PrototypeFeatureLifecycleIntegrationDiagnostics | None
    )
    dream_td_errors: Any  # Float[Array, " n_dreams"] | None
    horde_td_errors: Any  # Float[Array, " n_demons"] | None
    ia_augmented_obs: Any  # Float[Array, " augmented_dim"] | None
    ia_recommendation: Any  # Int[Array, ""] | None
    experiential_memory_diagnostics: PrototypeExperientialMemoryDiagnostics | None
    partner_policy_fusion_diagnostics: PrototypePartnerPolicyFusionDiagnostics | None
    transition_diagnostics: PrototypeTransitionDiagnostics

    @property
    def candidate_update_audit_passed(self) -> Bool[Array, ""]:
        """Return whether this event's multi-objective candidate audit passed."""
        application = self.gradient_joy_application
        if application is None:
            return jnp.asarray(False, dtype=jnp.bool_)
        return cast(
            GradientJoyApplicationResult,
            application,
        ).assessment.candidate_update_audit_passed

    @property
    def sparks_joy(self) -> Bool[Array, ""]:
        """Historical alias; paper-defined “sparks joy” belongs to Kondo."""
        return self.candidate_update_audit_passed

    @property
    def behavior_representation_gradient(self) -> Float[Array, " observation_dim"]:
        """Return the raw causal control-loss input gradient."""
        return self.behavior_gradient_result.gradient

    @property
    def behavior_representation_gradient_valid(self) -> Bool[Array, ""]:
        """Return whether the raw control-loss gradient is usable."""
        return self.behavior_gradient_result.valid

    @property
    def mixed_representation_gradient(self) -> Float[Array, " observation_dim"]:
        """Return the explicitly mixed builder candidate, or exact zero."""
        return self.representation_gradient_mix.gradient

    @property
    def mixed_representation_gradient_valid(self) -> Bool[Array, ""]:
        """Return whether a valid active mixed source produced a candidate."""
        return self.representation_gradient_mix.applied

    @property
    def audited_candidate_update_applied(self) -> Bool[Array, ""]:
        """Report whether the audited finite-precision update was committed."""
        application = self.gradient_joy_application
        if application is None:
            return jnp.asarray(False, dtype=jnp.bool_)
        return (
            cast(GradientJoyApplicationResult, application).applied
            & self.state_builder_learning_diagnostics.applied
        )

    @property
    def joyful_gradient_applied(self) -> Bool[Array, ""]:
        """Historical alias; prefer ``audited_candidate_update_applied``."""
        return self.audited_candidate_update_applied


@chex.dataclass(frozen=True)
class PrototypeArrayResult:
    """Result from :meth:`PrototypeAgent.scan` over a batch of transitions."""

    state: PrototypeAgentState
    actions: Int[Array, " num_steps"]
    oak_td_errors: Float[Array, " num_steps"]
    oak_average_rewards: Float[Array, " num_steps"]
    transition_valid: Bool[Array, " num_steps"]
    state_builder_learning_applied: Bool[Array, " num_steps"]
    gradient_sparks_joy: Bool[Array, " num_steps"]
    joyful_gradient_applied: Bool[Array, " num_steps"]


def _contains_tracer(value: Any) -> bool:
    """Return whether a PyTree contains a JAX tracing value."""
    return any(
        isinstance(leaf, jax.core.Tracer)
        for leaf in jax.tree_util.tree_leaves(value)
    )


def _floating_tree_is_finite(value: Any) -> Bool[Array, ""]:
    """Return whether every inexact array leaf in a PyTree is finite."""
    predicates: list[Array] = []
    for leaf in jax.tree_util.tree_leaves(value):
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jnp.inexact):
            predicates.append(jnp.all(jnp.isfinite(leaf)))
    if not predicates:
        return jnp.asarray(True)
    return jnp.all(jnp.stack(predicates))


def _tree_arrays_equal(left: Any, right: Any) -> Bool[Array, ""]:
    """Compare two fixed PyTrees exactly without leaving traced execution."""
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    if (
        cast(Any, left_structure) != right_structure
        or len(left_leaves) != len(right_leaves)
    ):
        return jnp.asarray(False, dtype=jnp.bool_)
    if not left_leaves:
        return jnp.asarray(True, dtype=jnp.bool_)
    return jnp.all(
        jnp.stack(
            [
                jnp.array_equal(jnp.asarray(a), jnp.asarray(b))
                for a, b in zip(left_leaves, right_leaves, strict=True)
            ]
        )
    )


def _recurrent_nll_estimator_offset(
    config: RecurrentLatentWorldModelEnsembleConfig,
) -> float:
    """Return a fixed affine offset making bounded Gaussian NLL non-negative.

    The offset is constant for a configured model, so loss-window differences
    are unchanged. A margin of one avoids float32 roundoff at the analytic
    minimum; the result diagnostics continue to report the unshifted NLL.
    """
    minimum = 0.5 * (math.log(2.0 * math.pi) + math.log(config.variance_floor))
    return max(1.0, 1.0 - minimum)


def _unavailable_learning_signals() -> TypedLearningSignals:
    """Return finite zeros whose typed channels are explicitly unavailable."""
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    unavailable = jnp.asarray(False, dtype=jnp.bool_)
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
            input_valid=unavailable,
            epistemic=unavailable,
            aleatoric=unavailable,
            normalized_residual=unavailable,
            learning_progress=unavailable,
            change_probability=unavailable,
        ),
    )


def _unavailable_ensemble_diagnostics() -> WorldModelEnsembleDiagnostics:
    """Return a fixed-shape diagnostic record for a disabled/rejected ensemble."""
    unavailable = jnp.asarray(False, dtype=jnp.bool_)
    return WorldModelEnsembleDiagnostics(
        state_valid=unavailable,
        input_valid=unavailable,
        capacity_available=unavailable,
        predictions_valid=unavailable,
        representation_gradient_valid=unavailable,
        signals_valid=unavailable,
        residual_update_valid=unavailable,
        member_updates_valid=unavailable,
        candidate_state_valid=unavailable,
        applied=unavailable,
        rejected=jnp.asarray(True, dtype=jnp.bool_),
    )


def _unavailable_recurrent_latent_diagnostics(
    ensemble_size: int,
) -> PrototypeRecurrentLatentDiagnostics:
    """Return fixed-shape, explicitly unavailable recurrent diagnostics."""
    unavailable = jnp.asarray(False, dtype=jnp.bool_)
    model = RecurrentLatentWorldModelDiagnostics(
        state_valid=unavailable,
        cache_valid=unavailable,
        input_valid=unavailable,
        ownership_valid=unavailable,
        boundary_semantics_valid=unavailable,
        capacity_available=unavailable,
        cached_prediction_exact=unavailable,
        predictions_valid=unavailable,
        losses_valid=unavailable,
        representation_gradient_valid=unavailable,
        member_gradients_valid=jnp.zeros((ensemble_size,), dtype=jnp.bool_),
        candidate_parameters_valid=jnp.zeros((ensemble_size,), dtype=jnp.bool_),
        candidate_state_valid=unavailable,
        recurrent_advanced_once=unavailable,
        recurrent_reset=unavailable,
        applied=unavailable,
        rejected=jnp.asarray(True, dtype=jnp.bool_),
    )
    availability = RecurrentLatentPredictionAvailability(
        prediction=unavailable,
        epistemic=unavailable,
        aleatoric=unavailable,
    )
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return PrototypeRecurrentLatentDiagnostics(
        model=model,
        prediction_availability=availability,
        raw_epistemic_disagreement=zero,
        raw_aleatoric_uncertainty=zero,
        raw_uncertainty_calibrated=unavailable,
        signals_valid=unavailable,
        transaction_applied=unavailable,
        next_decision_cached=unavailable,
    )


def _gate_recurrent_learning_signals(
    signals: TypedLearningSignals,
    availability: RecurrentLatentPredictionAvailability,
    transaction_applied: Array,
) -> TypedLearningSignals:
    """Honor model warmup while preserving each signal's native availability."""
    false = jnp.asarray(False, dtype=jnp.bool_)
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    uncertainty_available = (
        transaction_applied
        & availability.epistemic
        & availability.aleatoric
        & signals.availability.epistemic
        & signals.availability.aleatoric
        & signals.availability.normalized_residual
    )
    progress_available = (
        transaction_applied & signals.availability.learning_progress
    )
    change_available = (
        uncertainty_available & signals.availability.change_probability
    )

    def gated(value: Array, available: Array) -> Array:
        return jnp.where(available, value, zero)

    return TypedLearningSignals(
        epistemic_disagreement=gated(
            signals.epistemic_disagreement,
            uncertainty_available,
        ),
        epistemic_surprise=gated(signals.epistemic_surprise, uncertainty_available),
        aleatoric_uncertainty=gated(
            signals.aleatoric_uncertainty,
            uncertainty_available,
        ),
        normalized_residual=gated(signals.normalized_residual, uncertainty_available),
        learning_progress=gated(signals.learning_progress, progress_available),
        calibrated_residual_z=gated(signals.calibrated_residual_z, change_available),
        instantaneous_change_probability=gated(
            signals.instantaneous_change_probability,
            change_available,
        ),
        change_probability=gated(signals.change_probability, change_available),
        availability=LearningSignalAvailability(
            input_valid=jnp.where(
                transaction_applied,
                signals.availability.input_valid,
                false,
            ),
            epistemic=uncertainty_available,
            aleatoric=uncertainty_available,
            normalized_residual=uncertainty_available,
            learning_progress=progress_available,
            change_probability=change_available,
        ),
    )


def _unavailable_behavior_gradient(
    observation_dim: int,
) -> PrototypeBehaviorGradientResult:
    """Return an exact-zero control source with explicit unavailability."""
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    unavailable = jnp.asarray(False, dtype=jnp.bool_)
    diagnostics = PrototypeBehaviorGradientDiagnostics(
        source_available=unavailable,
        idle_base_source=unavailable,
        intra_option_source=unavailable,
        parameters_finite=unavailable,
        inputs_finite=unavailable,
        source_indices_valid=unavailable,
        prediction_finite=unavailable,
        target_finite=unavailable,
        td_error_finite=unavailable,
        loss_finite=unavailable,
        gradient_norm_finite=unavailable,
        prediction=zero,
        target=zero,
        td_error=zero,
        loss=zero,
        gradient_norm=zero,
        bootstrap_discount=zero,
        option_terminates=unavailable,
        gradient_finite=unavailable,
        valid=unavailable,
        rejected=jnp.asarray(True, dtype=jnp.bool_),
    )
    return PrototypeBehaviorGradientResult(
        gradient=jnp.zeros((observation_dim,), dtype=jnp.float32),
        valid=unavailable,
        diagnostics=diagnostics,
    )


def _unavailable_representation_gradient_mix(
    observation_dim: int,
) -> RepresentationGradientMixResult:
    """Return an exact-zero, explicitly unavailable two-source mix."""
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    unavailable = jnp.asarray(False, dtype=jnp.bool_)
    diagnostics = RepresentationGradientMixDiagnostics(
        behavior_raw_norm=zero,
        grounded_world_raw_norm=zero,
        behavior_used_norm=zero,
        grounded_world_used_norm=zero,
        behavior_weight=zero,
        grounded_world_weight=zero,
        behavior_effective_weight=zero,
        grounded_world_effective_weight=zero,
        dot_product=zero,
        cosine_similarity=zero,
        conflict=unavailable,
        unclipped_mixed_norm=zero,
        final_mixed_norm=zero,
        behavior_valid=unavailable,
        grounded_world_valid=unavailable,
        behavior_active=unavailable,
        grounded_world_active=unavailable,
        applied=unavailable,
        rejected=jnp.asarray(True, dtype=jnp.bool_),
        zero_output=jnp.asarray(True, dtype=jnp.bool_),
    )
    return RepresentationGradientMixResult(
        gradient=jnp.zeros((observation_dim,), dtype=jnp.float32),
        valid=unavailable,
        applied=unavailable,
        rejected=jnp.asarray(True, dtype=jnp.bool_),
        zero_output=jnp.asarray(True, dtype=jnp.bool_),
        diagnostics=diagnostics,
    )


def _unavailable_state_builder_learning_diagnostics() -> StateBuilderLearningDiagnostics:
    """Return a fixed-shape rejected record when online builder learning is off."""
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    unavailable = jnp.asarray(False, dtype=jnp.bool_)
    return StateBuilderLearningDiagnostics(
        gradient_norm=zero,
        clipped_gradient_norm=zero,
        parameter_update_norm=zero,
        proposal_valid=unavailable,
        source_matches=unavailable,
        capacity_available=unavailable,
        candidate_parameters_valid=unavailable,
        applied=unavailable,
        fixed_noop=unavailable,
        valid=unavailable,
        rejected=jnp.asarray(True, dtype=jnp.bool_),
    )


def _shaped_float_array(value: Any, shape: tuple[int, ...], *, name: str) -> Array:
    """Coerce to float32 without ever reshaping: static shape drift is a caller error."""
    array = jnp.asarray(value, dtype=jnp.float32)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def _checked_finite_array(
    value: Any,
    shape: tuple[int, ...],
    *,
    name: str,
    allow_nan: bool = False,
) -> Array:
    """Validate shape/finiteness eagerly and poison invalid traced values."""
    array = _shaped_float_array(value, shape, name=name)
    element_valid = ~jnp.isinf(array) if allow_nan else jnp.isfinite(array)
    valid = jnp.all(element_valid)
    if not _contains_tracer(array) and not bool(valid):
        requirement = "must not contain infinity" if allow_nan else "must be finite"
        raise ValueError(f"{name} {requirement}")
    return jnp.where(valid, array, jnp.full_like(array, jnp.nan))


def _checked_unit_discount(value: Any, shape: tuple[int, ...], *, name: str) -> Array:
    """Validate finite ``[0, 1]`` discounts at eager and traced boundaries."""
    array = _shaped_float_array(value, shape, name=name)
    valid = jnp.all(
        jnp.isfinite(array) & (array >= 0.0) & (array <= 1.0)
    )
    if not _contains_tracer(array) and not bool(valid):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return jnp.where(valid, array, jnp.full_like(array, jnp.nan))


def _strict_float_array(value: Any, shape: tuple[int, ...], *, name: str) -> Array:
    """Coerce a real numeric value while rejecting static shape/type drift."""
    array = jnp.asarray(value)
    if not (
        jnp.issubdtype(array.dtype, jnp.floating)
        or jnp.issubdtype(array.dtype, jnp.integer)
    ) or jnp.issubdtype(array.dtype, jnp.bool_):
        raise ValueError(f"{name} must have a real numeric dtype")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return jnp.asarray(array, dtype=jnp.float32)


def _strict_float32_array(value: Any, shape: tuple[int, ...], *, name: str) -> Array:
    """Require an exact float32 array at an optimizer-control boundary."""
    array = jnp.asarray(value)
    if array.dtype != jnp.float32 or array.shape != shape:
        raise ValueError(f"{name} must have shape {shape} and dtype float32")
    return jnp.asarray(array, dtype=jnp.float32)


def _strict_bool_scalar(value: Any, *, name: str) -> Array:
    """Require an actual scalar boolean rather than truthy numeric input."""
    array = jnp.asarray(value)
    if array.dtype != jnp.bool_ or array.shape != ():
        raise ValueError(f"{name} must be a scalar boolean")
    return jnp.asarray(array, dtype=jnp.bool_)


def _strict_action_scalar(value: Any, *, name: str) -> Array:
    """Require one scalar integer action that round-trips through int32."""
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.integer) or array.shape != ():
        raise ValueError(f"{name} must be a scalar integer")
    if not _contains_tracer(value):
        try:
            exact_value = operator.index(value)
        except TypeError:
            exact_value = int(value)
        if exact_value < -(2**31) or exact_value > 2**31 - 1:
            raise ValueError(f"{name} must be losslessly representable as int32")
    converted = jnp.asarray(array, dtype=jnp.int32)
    source_unsigned = jnp.issubdtype(array.dtype, jnp.unsignedinteger)
    source_bytes = int(array.dtype.itemsize)
    if (source_unsigned and source_bytes <= 2) or (
        not source_unsigned and source_bytes <= 4
    ):
        representable = jnp.asarray(True)
    elif source_unsigned:
        representable = array <= jnp.asarray(2**31 - 1, dtype=array.dtype)
    else:
        representable = (
            (array >= jnp.asarray(-(2**31), dtype=array.dtype))
            & (array <= jnp.asarray(2**31 - 1, dtype=array.dtype))
        )
    return jnp.where(
        representable,
        converted,
        jnp.asarray(-1, dtype=jnp.int32),
    )


def _strict_uint32_words(
    value: Any,
    shape: tuple[int, ...],
    *,
    name: str,
) -> Array:
    """Require an unsigned word array without lossy source coercion."""
    array = jnp.asarray(value)
    source_dtype = getattr(value, "dtype", None)
    if source_dtype is not None and source_dtype != jnp.dtype(jnp.uint32):
        raise ValueError(f"{name} must have shape {shape} and dtype uint32")
    if array.dtype != jnp.uint32 or array.shape != shape:
        raise ValueError(f"{name} must have shape {shape} and dtype uint32")
    return jnp.asarray(array, dtype=jnp.uint32)


def _strict_decision_id(value: Any, *, name: str) -> Array:
    """Require the exact four-word lifecycle/generation token format."""
    return _strict_uint32_words(value, (4,), name=name)


def _decision_generation_available(decision_id: Array) -> Array:
    """Return whether a fresh 64-bit generation remains available."""
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    return ~jnp.all(decision_id[2:] == maximum)


def _step_counter_can_process(step_count: Array) -> Bool[Array, ""]:
    """Return whether one already-dispatched transition can still be counted."""
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    return (step_count >= 0) & (step_count < maximum)


def _next_counter_capacity_available(
    state: PrototypeAgentState,
    *,
    execution_boundary: Array,
) -> Bool[Array, ""]:
    """Reserve enough signed-counter capacity for the next dispatched action."""
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    next_step_count = _saturating_int32_increment(state.step_count)
    next_observation_count = jnp.where(
        execution_boundary,
        _saturating_int32_increment(
            _saturating_int32_increment(state.observation_event_count)
        ),
        _saturating_int32_increment(state.observation_event_count),
    )
    # A future transition may itself be an autoreset boundary and consume two
    # observation events, so retain two free observation-event slots.
    return (next_step_count < maximum) & (next_observation_count <= maximum - 2)


def _increment_decision_id(decision_id: Array) -> Array:
    """Increment a big-endian pair of uint32 words without x64 mode."""
    one = jnp.asarray(1, dtype=jnp.uint32)
    low = decision_id[3] + one
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    high = decision_id[2] + carry
    return jnp.stack((decision_id[0], decision_id[1], high, low))


def _saturating_int32_increment(value: Array) -> Array:
    """Advance a non-negative lifecycle counter without wraparound."""
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    counter = jnp.asarray(value, dtype=jnp.int32)
    return jnp.minimum(jnp.maximum(counter, 0), maximum - 1) + 1


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class PrototypeAgent:
    """Prototype combining online components from across the Alberta Plan.

    Operates in **continuing average-reward mode** — no episode resets, no
    offline batch phases. Designed for use in sim-to-real transfer:
    :meth:`update_transition` is the authoritative transition boundary in
    simulation and on real hardware. :meth:`update` remains a compatibility
    wrapper for callers that predate explicit discounts.

    Single-step daemon usage::

        state = agent.start(agent.init(jr.key(0)), initial_obs)
        decision = agent.decision(state)
        while True:
            reward, next_obs, discount, terminated, truncated = env.step(
                int(decision.action)
            )
            result = agent.update_transition(
                state,
                PrototypeTransition(
                    observation=decision.observation,
                    action=decision.action,
                    decision_id=decision.decision_id,
                    reward=reward,
                    discount=discount,
                    terminated=terminated,
                    truncated=truncated,
                    next_observation=next_obs,
                    next_decision_observation=next_obs,
                ),
            )
            state = result.state
            decision = agent.decision(state)

    Periodic curation (Python-level, outside JAX)::

        if step % curation_interval == 0:
            agent, state = agent.curate(state, key)
    """

    def __init__(self, config: PrototypeAgentConfig) -> None:
        if type(config) is not PrototypeAgentConfig:
            raise ValueError("config must be an exact PrototypeAgentConfig")
        self._config = config
        self._oak = OaKAgent(config.oak)
        self._option_search_control: OptionSearchControl | None = None
        if config.option_search_control is not None:
            self._option_search_control = OptionSearchControl(
                self._oak.stomp_agent,
                config.option_search_control,
            )
        self._state_builder: StateBuilder[Any] | None = (
            state_builder_from_config(config.state_builder.to_config())
            if config.state_builder is not None
            else None
        )
        self._prototype_feature_lifecycle: PrototypeFeatureLifecycle | None = None
        if config.prototype_feature_lifecycle is not None:
            self._prototype_feature_lifecycle = PrototypeFeatureLifecycle(
                config.prototype_feature_lifecycle
            )

        self._world_model: ActionConditionedWorldModel | None = None
        self._world_model_ensemble: WorldModelEnsemble | None = None
        self._model_replay_rehearsal: ModelReplayRehearsal | None = None
        self._recurrent_latent_world_model_ensemble: (
            RecurrentLatentWorldModelEnsemble | None
        ) = None
        self._recurrent_signal_estimator: LearningSignalEstimator | None = None
        self._recurrent_signal_nll_offset = 0.0
        self._buffer: RecentObservationBuffer | None = None
        self._dreamer: GuardedDreamer | None = None
        if config.world_model is not None:
            self._world_model = ActionConditionedWorldModel(config.world_model)
            self._buffer = RecentObservationBuffer(
                config.buffer_capacity, config.oak.observation_dim
            )
            self._dreamer = GuardedDreamer(config.dreaming or DreamingConfig())
        elif config.world_model_ensemble is not None:
            self._world_model_ensemble = WorldModelEnsemble(
                config.world_model_ensemble
            )
        elif config.model_replay_rehearsal is not None:
            self._model_replay_rehearsal = ModelReplayRehearsal(
                config.model_replay_rehearsal
            )
        elif config.recurrent_latent_world_model_ensemble is not None:
            recurrent_config = config.recurrent_latent_world_model_ensemble
            self._recurrent_latent_world_model_ensemble = (
                RecurrentLatentWorldModelEnsemble(recurrent_config)
            )
            self._recurrent_signal_nll_offset = _recurrent_nll_estimator_offset(
                recurrent_config
            )
            self._recurrent_signal_estimator = LearningSignalEstimator(
                LearningSignalEstimatorConfig(
                    ensemble_size=recurrent_config.ensemble_size,
                    target_dim=recurrent_config.target_dim,
                    variance_floor=recurrent_config.variance_floor,
                    max_input_magnitude=max(
                        recurrent_config.max_input_magnitude,
                        recurrent_config.max_prediction_magnitude,
                    ),
                    max_predicted_variance=recurrent_config.max_variance,
                    max_observed_loss=(
                        recurrent_config.max_loss_magnitude
                        + self._recurrent_signal_nll_offset
                    ),
                )
            )

        self._horde: HordeLearner | None = None
        if config.horde_spec is not None:
            self._horde = HordeLearner(
                config.horde_spec,
                hidden_sizes=config.horde_hidden_sizes,
                step_size=config.horde_step_size,
            )

        self._ia: IAAgent | None = None
        if config.ia is not None:
            self._ia = IAAgent(config.ia)

        self._partner_policy_fusion: PartnerPolicyFusion | None = None
        if config.partner_policy_fusion is not None:
            self._partner_policy_fusion = PartnerPolicyFusion(
                config.partner_policy_fusion
            )

        self._experiential_memory: ExperientialMemory | None = None
        self._experiential_memory_policy: ExperientialMemoryPolicy | None = None
        if config.experiential_memory is not None:
            self._experiential_memory = ExperientialMemory(
                config.experiential_memory
            )
            self._experiential_memory_policy = ExperientialMemoryPolicy(
                self._experiential_memory
            )

    # -- Properties -----------------------------------------------------------

    @property
    def config(self) -> PrototypeAgentConfig:
        """Agent configuration."""
        return self._config

    @property
    def oak_agent(self) -> OaKAgent:
        """Underlying OaK control agent."""
        return self._oak

    @property
    def option_search_control(self) -> OptionSearchControl | None:
        """Return the opt-in stateless option-model search controller."""

        return self._option_search_control

    @property
    def option_search_control_resource_budget(
        self,
    ) -> OptionSearchControlResourceBudget | None:
        """Return fixed logical work bounds for one option-search call."""

        if self._option_search_control is None:
            return None
        return self._option_search_control.resource_budget

    @property
    def state_builder(self) -> StateBuilder[Any] | None:
        """Canonical state builder, or ``None`` for a legacy raw/GRU path."""
        return self._state_builder

    @property
    def prototype_feature_lifecycle(self) -> PrototypeFeatureLifecycle | None:
        """Return the opt-in bounded pair-feature lifecycle."""

        return self._prototype_feature_lifecycle

    def _builder_component_state(self, slot: Any) -> Any:
        """Unwrap the historical builder slot only on the feature lane."""

        if self._prototype_feature_lifecycle is None:
            return slot
        if not isinstance(slot, PrototypeFeatureRepresentationState):
            raise TypeError(
                "state_builder_state must be a "
                "PrototypeFeatureRepresentationState when "
                "prototype_feature_lifecycle is configured"
            )
        return slot.builder_state

    def _feature_lifecycle_component_state(
        self,
        slot: Any,
    ) -> PrototypeFeatureLifecycleState:
        """Return persistent feature state from the enabled-only wrapper."""

        if self._prototype_feature_lifecycle is None:
            raise RuntimeError("prototype feature lifecycle is disabled")
        if not isinstance(slot, PrototypeFeatureRepresentationState):
            raise TypeError(
                "state_builder_state must be a "
                "PrototypeFeatureRepresentationState when "
                "prototype_feature_lifecycle is configured"
            )
        return slot.feature_lifecycle_state

    def _representation_state_slot(
        self,
        builder_state: Any,
        feature_lifecycle_state: PrototypeFeatureLifecycleState | None,
    ) -> Any:
        """Compose the builder slot without changing disabled PyTrees."""

        if self._prototype_feature_lifecycle is None:
            return builder_state
        if feature_lifecycle_state is None:
            raise RuntimeError(
                "configured prototype feature lifecycle requires persistent state"
            )
        return PrototypeFeatureRepresentationState(
            builder_state=builder_state,
            feature_lifecycle_state=feature_lifecycle_state,
        )

    def _oak_component_state(self, slot: Any) -> OaKState:
        """Unwrap OaK only on the feature lane; preserve legacy PyTrees."""

        if self._prototype_feature_lifecycle is None:
            return cast(OaKState, slot)
        if type(slot) is not PrototypeFeatureOaKState:
            raise TypeError(
                "oak_state must be a PrototypeFeatureOaKState when "
                "prototype_feature_lifecycle is configured"
            )
        if type(slot.oak_state) is not OaKState:
            raise TypeError("bound feature consumer must contain an exact OaKState")
        return slot.oak_state

    def _feature_consumer_binding(
        self,
        slot: Any,
    ) -> PrototypeFeatureConsumerBinding:
        """Return the identity physically coupled to the enabled OaK subtree."""

        if self._prototype_feature_lifecycle is None:
            raise RuntimeError("prototype feature lifecycle is disabled")
        if type(slot) is not PrototypeFeatureOaKState:
            raise TypeError(
                "oak_state must be a PrototypeFeatureOaKState when "
                "prototype_feature_lifecycle is configured"
            )
        if type(slot.consumer_binding) is not PrototypeFeatureConsumerBinding:
            raise TypeError(
                "bound feature consumer must contain an exact consumer binding"
            )
        return slot.consumer_binding

    def _oak_state_slot(
        self,
        oak_state: OaKState,
        consumer_binding: PrototypeFeatureConsumerBinding | None,
    ) -> Any:
        """Compose the enabled-only bound OaK subtree."""

        if self._prototype_feature_lifecycle is None:
            return oak_state
        if type(oak_state) is not OaKState:
            raise TypeError("feature consumer must be an exact OaKState")
        if type(consumer_binding) is not PrototypeFeatureConsumerBinding:
            raise TypeError(
                "configured prototype feature lifecycle requires a consumer binding"
            )
        return PrototypeFeatureOaKState(
            oak_state=oak_state,
            consumer_binding=consumer_binding,
        )

    @property
    def partner_policy_fusion(self) -> PartnerPolicyFusion | None:
        """Return the opt-in bounded partner-message policy fusion mechanism."""

        return self._partner_policy_fusion

    @property
    def experiential_memory(self) -> ExperientialMemory | None:
        """Return the opt-in fixed-capacity experiential memory."""

        return self._experiential_memory

    @property
    def experiential_memory_policy(self) -> ExperientialMemoryPolicy | None:
        """Return the categorical memory proposal/transaction boundary."""

        return self._experiential_memory_policy

    @property
    def experiential_memory_resource_declaration(
        self,
    ) -> PrototypeExperientialMemoryResourceDeclaration | None:
        """Declare memory bytes and the single deterministic pre-state query."""

        policy = self._experiential_memory_policy
        if policy is None:
            return None
        policy_resources = policy.resource_declaration()
        memory_config = policy.memory.config
        categorical_queries = policy_resources.memory_queries_per_proposal
        causal_queries = 0
        return PrototypeExperientialMemoryResourceDeclaration(
            memory_capacity=memory_config.capacity,
            memory_observation_dim=memory_config.observation_dim,
            memory_key_dim=memory_config.key_dim,
            memory_action_dim=memory_config.action_dim,
            memory_outcome_dim=memory_config.outcome_dim,
            persistent_state_bytes=(
                policy_resources.external_memory_persistent_state_bytes
            ),
            categorical_policy_queries=categorical_queries,
            causal_step_queries=causal_queries,
            total_deterministic_prestate_queries=(
                categorical_queries + causal_queries
            ),
            writes_attempted=1,
            random_draws=policy_resources.random_draws_per_proposal,
            score_mass_values_interpreted=(
                policy_resources.score_mass_values_interpreted_per_proposal
            ),
            hard_safety_values_interpreted=(
                policy_resources.hard_safety_values_interpreted_per_proposal
            ),
        )

    def _interaction_without_memory(self, slot: Any) -> Any:
        """Unwrap only the opt-in outer memory state composition."""

        if self._experiential_memory is None:
            return slot
        if not isinstance(slot, PrototypeMemoryInteractionState):
            raise TypeError(
                "ia_state must be a PrototypeMemoryInteractionState when "
                "experiential memory is configured"
            )
        return slot.interaction_state

    def _ia_component_state(self, slot: Any) -> Any:
        """Unwrap IA state while preserving historical IA-only slot shapes."""

        slot = self._interaction_without_memory(slot)
        if self._partner_policy_fusion is None:
            return slot
        if not isinstance(slot, PrototypeInteractionState):
            raise TypeError(
                "ia_state must be a PrototypeInteractionState when partner "
                "policy fusion is configured"
            )
        return slot.ia_state

    def _partner_fusion_component_state(
        self,
        slot: Any,
    ) -> PartnerPolicyFusionState:
        """Return the partner state from the opt-in IA-slot wrapper."""

        if self._partner_policy_fusion is None:
            raise RuntimeError("partner policy fusion is disabled")
        slot = self._interaction_without_memory(slot)
        if not isinstance(slot, PrototypeInteractionState):
            raise TypeError(
                "ia_state must be a PrototypeInteractionState when partner "
                "policy fusion is configured"
            )
        return slot.partner_policy_fusion_state

    def _partner_interaction_state(
        self,
        slot: Any,
    ) -> PrototypeInteractionState:
        """Return the complete partner wrapper below optional memory."""

        if self._partner_policy_fusion is None:
            raise RuntimeError("partner policy fusion is disabled")
        inner = self._interaction_without_memory(slot)
        if not isinstance(inner, PrototypeInteractionState):
            raise TypeError(
                "ia_state must contain a PrototypeInteractionState when "
                "partner policy fusion is configured"
            )
        return inner

    def _experiential_memory_component_state(
        self,
        slot: Any,
    ) -> ExperientialMemoryState:
        """Return persistent memory from the opt-in outer wrapper."""

        if self._experiential_memory is None:
            raise RuntimeError("experiential memory is disabled")
        if not isinstance(slot, PrototypeMemoryInteractionState):
            raise TypeError(
                "ia_state must be a PrototypeMemoryInteractionState when "
                "experiential memory is configured"
            )
        return slot.experiential_memory_state

    def _interaction_slot(
        self,
        ia_state: Any,
        partner_state: Any,
        *,
        experiential_memory_state: ExperientialMemoryState | None = None,
        feedback_prototype_decision_id: Array | None = None,
        feedback_prototype_decision_id_available: Array | None = None,
    ) -> Any:
        """Compose only configured interaction lanes into the existing slot."""

        interaction_state = ia_state
        if self._partner_policy_fusion is not None:
            owner = (
                jnp.zeros((4,), dtype=jnp.uint32)
                if feedback_prototype_decision_id is None
                else jnp.asarray(feedback_prototype_decision_id, dtype=jnp.uint32)
            )
            owner_available = (
                jnp.asarray(False, dtype=jnp.bool_)
                if feedback_prototype_decision_id_available is None
                else jnp.asarray(
                    feedback_prototype_decision_id_available,
                    dtype=jnp.bool_,
                )
            )
            interaction_state = PrototypeInteractionState(
                ia_state=ia_state,
                partner_policy_fusion_state=cast(
                    PartnerPolicyFusionState,
                    partner_state,
                ),
                feedback_prototype_decision_id=owner,
                feedback_prototype_decision_id_available=owner_available,
            )
        if self._experiential_memory is None:
            return interaction_state
        if experiential_memory_state is None:
            raise RuntimeError(
                "configured experiential memory requires a persistent state"
            )
        return PrototypeMemoryInteractionState(
            interaction_state=interaction_state,
            experiential_memory_state=experiential_memory_state,
        )

    @property
    def world_model_ensemble(self) -> WorldModelEnsemble | None:
        """Return the bounded ensemble, or ``None`` on legacy/no-model lanes.

        On the rehearsal lane its matching child state is
        ``state.world_model_state.ensemble_state``; the complete Prototype
        world-model state must still be advanced through
        :attr:`model_replay_rehearsal` so replay remains atomic.
        """
        if self._model_replay_rehearsal is not None:
            return self._model_replay_rehearsal.ensemble
        return self._world_model_ensemble

    @property
    def model_replay_rehearsal(self) -> ModelReplayRehearsal | None:
        """Return the model-only rehearsal composer when configured."""
        return self._model_replay_rehearsal

    @property
    def recurrent_latent_world_model_ensemble(
        self,
    ) -> RecurrentLatentWorldModelEnsemble | None:
        """Return the recurrent latent ensemble on its opt-in lane."""
        return self._recurrent_latent_world_model_ensemble

    @staticmethod
    def _gate_recurrent_decision_cache(
        cache: RecurrentLatentDecisionCache,
        armed: Array,
    ) -> RecurrentLatentDecisionCache:
        """Disarm a speculative cache without changing its fixed PyTree shape."""
        available = jnp.asarray(armed, dtype=jnp.bool_) & cache.valid
        prediction = cache.prediction
        prediction_availability = RecurrentLatentPredictionAvailability(
            prediction=prediction.availability.prediction & available,
            epistemic=prediction.availability.epistemic & available,
            aleatoric=prediction.availability.aleatoric & available,
        )
        gated_prediction = prediction.replace(
            warmup_ready=prediction.warmup_ready & available,
            availability=prediction_availability,
            valid=prediction.valid & available,
        )
        return cast(
            RecurrentLatentDecisionCache,
            cache.replace(
                prediction=gated_prediction,
                valid=available,
            ),
        )

    def _recurrent_decide_from_start(
        self,
        model_state: RecurrentLatentWorldModelEnsembleState,
        start_cache: RecurrentLatentStartCache,
        action: Array,
        armed: Array,
    ) -> RecurrentLatentDecisionCache:
        """Bind one exact selected primitive action without advancing state."""
        model = self._recurrent_latent_world_model_ensemble
        if model is None:
            raise RuntimeError("recurrent latent world-model lane is disabled")
        safe_action = jnp.where(
            armed,
            action,
            jnp.asarray(0, dtype=jnp.int32),
        )
        return self._gate_recurrent_decision_cache(
            model.decide(model_state, start_cache, safe_action),
            armed,
        )

    def _recurrent_decision_for_observation(
        self,
        model_state: RecurrentLatentWorldModelEnsembleState,
        observation: Array,
        action: Array,
        armed: Array,
    ) -> RecurrentLatentDecisionCache:
        """Capture an observation owner and bind its exact selected action."""
        model = self._recurrent_latent_world_model_ensemble
        if model is None:
            raise RuntimeError("recurrent latent world-model lane is disabled")
        start_cache = model.start(model_state, observation)
        return self._recurrent_decide_from_start(
            model_state,
            start_cache,
            action,
            armed,
        )

    def _recurrent_signal_state_valid(
        self,
        state: LearningSignalEstimatorState,
    ) -> Array:
        """Validate the derived causal signal state and its configured bounds."""
        estimator = self._recurrent_signal_estimator
        if estimator is None:
            raise RuntimeError("recurrent signal estimator is disabled")
        LearningSignalEstimator._validate_state_shapes(state)
        config = estimator.config
        maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
        safe_valid_count = jnp.clip(state.valid_count, 0, maximum)
        safe_invalid_count = jnp.clip(state.invalid_count, 0, maximum)
        counters_sum_without_overflow = safe_valid_count <= maximum - safe_invalid_count
        safe_counter_sum = safe_valid_count + jnp.minimum(
            safe_invalid_count,
            maximum - safe_valid_count,
        )
        return (
            (state.step_count >= 0)
            & (state.valid_count >= 0)
            & (state.invalid_count >= 0)
            & counters_sum_without_overflow
            & (state.step_count == safe_counter_sum)
            & (state.calibration_count >= 0)
            & (state.calibration_count <= config.change_calibration_steps)
            & (state.calibration_count <= state.valid_count)
            & jnp.isfinite(state.calibration_mean)
            & (state.calibration_mean >= 0.0)
            & (state.calibration_mean <= config.max_normalized_residual)
            & jnp.isfinite(state.calibration_m2)
            & (state.calibration_m2 >= 0.0)
            & jnp.isfinite(state.fast_loss_ema)
            & (state.fast_loss_ema >= 0.0)
            & (state.fast_loss_ema <= config.max_observed_loss)
            & jnp.isfinite(state.slow_loss_ema)
            & (state.slow_loss_ema >= 0.0)
            & (state.slow_loss_ema <= config.max_observed_loss)
            & jnp.isfinite(state.sustained_change_probability)
            & (state.sustained_change_probability >= 0.0)
            & (state.sustained_change_probability <= 1.0)
        )

    def _recurrent_wrapper_numeric_valid(
        self,
        wrapper: PrototypeRecurrentLatentWorldModelState,
    ) -> Array:
        """Validate model, cache, estimator, and shared causal counters."""
        model = self._recurrent_latent_world_model_ensemble
        if model is None:
            raise RuntimeError("recurrent latent world-model lane is disabled")
        if not isinstance(wrapper, PrototypeRecurrentLatentWorldModelState):
            raise TypeError(
                "world_model_state must be a PrototypeRecurrentLatentWorldModelState"
            )
        model._validate_decision_static(wrapper.decision_cache)
        cache = wrapper.decision_cache
        prediction = cache.prediction
        cache_valid = (
            _floating_tree_is_finite(cache)
            & (cache.owner_event_count >= 0)
            & (cache.owner_event_count <= model.config.max_updates)
            & jnp.all(jnp.abs(cache.owner_hidden_states) <= 1.0 + 1.0e-6)
            & jnp.all(
                jnp.abs(cache.observation) <= model.config.max_input_magnitude
            )
            & (cache.action >= 0)
            & (cache.action < model.config.n_actions)
            & jnp.where(
                cache.valid,
                prediction.valid
                & prediction.availability.prediction
                & (
                    prediction.availability.epistemic
                    == (prediction.warmup_ready & prediction.valid)
                )
                & (
                    prediction.availability.aleatoric
                    == (prediction.warmup_ready & prediction.valid)
                ),
                (~prediction.valid)
                & (~prediction.availability.prediction)
                & (~prediction.availability.epistemic)
                & (~prediction.availability.aleatoric),
            )
        )
        return (
            model.state_valid(wrapper.model_state)
            & cache_valid
            & self._recurrent_signal_state_valid(wrapper.signal_state)
            & (wrapper.signal_state.step_count == wrapper.model_state.event_count)
        )

    def _recurrent_armed_cache_consistent(
        self,
        state: PrototypeAgentState,
    ) -> Array:
        """Recompute and exactly verify the current decision-owned prediction."""
        wrapper = cast(
            PrototypeRecurrentLatentWorldModelState,
            state.world_model_state,
        )
        expected = self._recurrent_decision_for_observation(
            wrapper.model_state,
            state.current_representation,
            state.current_action,
            jnp.asarray(True, dtype=jnp.bool_),
        )
        return wrapper.decision_cache.valid & _tree_arrays_equal(
            wrapper.decision_cache,
            expected,
        )

    def _state_builder_parameter_count(self) -> int:
        """Return the configured learnable builder vector width."""
        builder_config = self._config.state_builder
        if not isinstance(builder_config, OnlineGatedStateBuilderConfig):
            raise RuntimeError("online state-builder learning is not configured")
        return builder_config.parameter_count()

    def _missing_gradient_joy_evidence(
        self,
        decision_id: Array,
    ) -> PrototypeGradientJoyEvidence:
        """Construct explicit unavailable evidence for a fail-closed audit."""
        zeros = jnp.zeros(
            (self._state_builder_parameter_count(),),
            dtype=jnp.float32,
        )
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        unavailable = jnp.asarray(False, dtype=jnp.bool_)
        return PrototypeGradientJoyEvidence(
            decision_id=decision_id,
            objective_probe_gradient=zeros,
            retention_probe_gradient=zeros,
            safety_cost_gradient=zeros,
            objective_probe_available=unavailable,
            retention_probe_available=unavailable,
            safety_probe_available=unavailable,
            probe_independence_attested=unavailable,
            advantage=zero,
            action_surprisal=zero,
            safety_cost=zero,
            advantage_available=unavailable,
            action_surprisal_available=unavailable,
            safety_cost_available=unavailable,
        )

    def _normalize_gradient_joy_evidence(
        self,
        state: PrototypeAgentState,
        evidence: PrototypeGradientJoyEvidence | None,
    ) -> tuple[PrototypeGradientJoyEvidence | None, Array, Array]:
        """Validate one optimizer-control sidecar without owning the transition.

        Static shape or dtype drift raises. Runtime non-finiteness and a stale
        decision binding clear the corresponding availability flags so the joy
        audit rejects, while the authoritative environment transition remains
        eligible for normal control/model learning.
        """
        if self._config.gradient_joy is None:
            if evidence is not None:
                raise ValueError(
                    "gradient_joy_evidence requires gradient_joy to be configured"
                )
            unavailable = jnp.asarray(False, dtype=jnp.bool_)
            return None, unavailable, unavailable

        if evidence is None:
            unavailable = jnp.asarray(False, dtype=jnp.bool_)
            return (
                self._missing_gradient_joy_evidence(state.current_decision_id),
                unavailable,
                unavailable,
            )
        if not isinstance(evidence, PrototypeGradientJoyEvidence):
            raise TypeError(
                "gradient_joy_evidence must be PrototypeGradientJoyEvidence"
            )

        parameter_shape = (self._state_builder_parameter_count(),)
        decision_id = _strict_decision_id(
            evidence.decision_id,
            name="gradient_joy_evidence.decision_id",
        )
        decision_matches = jnp.array_equal(
            decision_id,
            state.current_decision_id,
        )
        gradients = {
            "objective_probe_gradient": _strict_float32_array(
                evidence.objective_probe_gradient,
                parameter_shape,
                name="gradient_joy_evidence.objective_probe_gradient",
            ),
            "retention_probe_gradient": _strict_float32_array(
                evidence.retention_probe_gradient,
                parameter_shape,
                name="gradient_joy_evidence.retention_probe_gradient",
            ),
            "safety_cost_gradient": _strict_float32_array(
                evidence.safety_cost_gradient,
                parameter_shape,
                name="gradient_joy_evidence.safety_cost_gradient",
            ),
        }
        scalars = {
            "advantage": _strict_float32_array(
                evidence.advantage,
                (),
                name="gradient_joy_evidence.advantage",
            ),
            "action_surprisal": _strict_float32_array(
                evidence.action_surprisal,
                (),
                name="gradient_joy_evidence.action_surprisal",
            ),
            "safety_cost": _strict_float32_array(
                evidence.safety_cost,
                (),
                name="gradient_joy_evidence.safety_cost",
            ),
        }
        flags = {
            "objective_probe_available": _strict_bool_scalar(
                evidence.objective_probe_available,
                name="gradient_joy_evidence.objective_probe_available",
            ),
            "retention_probe_available": _strict_bool_scalar(
                evidence.retention_probe_available,
                name="gradient_joy_evidence.retention_probe_available",
            ),
            "safety_probe_available": _strict_bool_scalar(
                evidence.safety_probe_available,
                name="gradient_joy_evidence.safety_probe_available",
            ),
            "probe_independence_attested": _strict_bool_scalar(
                evidence.probe_independence_attested,
                name="gradient_joy_evidence.probe_independence_attested",
            ),
            "advantage_available": _strict_bool_scalar(
                evidence.advantage_available,
                name="gradient_joy_evidence.advantage_available",
            ),
            "action_surprisal_available": _strict_bool_scalar(
                evidence.action_surprisal_available,
                name="gradient_joy_evidence.action_surprisal_available",
            ),
            "safety_cost_available": _strict_bool_scalar(
                evidence.safety_cost_available,
                name="gradient_joy_evidence.safety_cost_available",
            ),
        }
        gradient_finite = {
            name: jnp.all(jnp.isfinite(value))
            for name, value in gradients.items()
        }
        scalar_valid = {
            "advantage": jnp.isfinite(scalars["advantage"]),
            "action_surprisal": (
                jnp.isfinite(scalars["action_surprisal"])
                & (scalars["action_surprisal"] >= 0.0)
            ),
            "safety_cost": jnp.isfinite(scalars["safety_cost"]),
        }
        normalized = PrototypeGradientJoyEvidence(
            decision_id=decision_id,
            objective_probe_gradient=jnp.where(
                gradient_finite["objective_probe_gradient"],
                gradients["objective_probe_gradient"],
                jnp.zeros_like(gradients["objective_probe_gradient"]),
            ),
            retention_probe_gradient=jnp.where(
                gradient_finite["retention_probe_gradient"],
                gradients["retention_probe_gradient"],
                jnp.zeros_like(gradients["retention_probe_gradient"]),
            ),
            safety_cost_gradient=jnp.where(
                gradient_finite["safety_cost_gradient"],
                gradients["safety_cost_gradient"],
                jnp.zeros_like(gradients["safety_cost_gradient"]),
            ),
            objective_probe_available=(
                flags["objective_probe_available"]
                & gradient_finite["objective_probe_gradient"]
                & decision_matches
            ),
            retention_probe_available=(
                flags["retention_probe_available"]
                & gradient_finite["retention_probe_gradient"]
                & decision_matches
            ),
            safety_probe_available=(
                flags["safety_probe_available"]
                & gradient_finite["safety_cost_gradient"]
                & decision_matches
            ),
            probe_independence_attested=(
                flags["probe_independence_attested"] & decision_matches
            ),
            advantage=jnp.where(
                scalar_valid["advantage"],
                scalars["advantage"],
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            action_surprisal=jnp.where(
                scalar_valid["action_surprisal"],
                scalars["action_surprisal"],
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            safety_cost=jnp.where(
                scalar_valid["safety_cost"],
                scalars["safety_cost"],
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            advantage_available=(
                flags["advantage_available"]
                & scalar_valid["advantage"]
                & decision_matches
            ),
            action_surprisal_available=(
                flags["action_surprisal_available"]
                & scalar_valid["action_surprisal"]
                & decision_matches
            ),
            safety_cost_available=(
                flags["safety_cost_available"]
                & scalar_valid["safety_cost"]
                & decision_matches
            ),
        )
        return (
            normalized,
            jnp.asarray(True, dtype=jnp.bool_),
            decision_matches,
        )

    def _missing_experiential_memory_input(
        self,
    ) -> PrototypeExperientialMemoryInput:
        """Return a fixed-shape unavailable memory transaction sidecar."""

        if self._experiential_memory is None:
            raise RuntimeError("experiential memory is disabled")
        unavailable = jnp.asarray(False, dtype=jnp.bool_)
        missing = jnp.asarray(-1, dtype=jnp.int32)
        return PrototypeExperientialMemoryInput(
            available=unavailable,
            current_prototype_decision_id=jnp.zeros((4,), dtype=jnp.uint32),
            next_prototype_decision_id=jnp.zeros((4,), dtype=jnp.uint32),
            query_representation_version=missing,
            entry_representation_version=missing,
            query_uncertainty=jnp.asarray(0.0, dtype=jnp.float32),
            query_uncertainty_available=unavailable,
            entry_uncertainty=jnp.asarray(0.0, dtype=jnp.float32),
            entry_uncertainty_available=unavailable,
            safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
            safety_cost_available=unavailable,
            reliability=jnp.asarray(0.0, dtype=jnp.float32),
            utility=jnp.asarray(0.0, dtype=jnp.float32),
            utility_available=unavailable,
            provenance_id=missing,
            source_id=missing,
            next_action_safety_mask=jnp.ones(
                (self._config.oak.n_primitive_actions,),
                dtype=jnp.bool_,
            ),
        )

    def _normalize_experiential_memory_input(
        self,
        sidecar: PrototypeExperientialMemoryInput | None,
    ) -> tuple[PrototypeExperientialMemoryInput | None, Array]:
        """Normalize static memory metadata while leaving dynamic gates traced."""

        if self._experiential_memory is None:
            if sidecar is not None:
                raise ValueError(
                    "experiential_memory_input requires experiential memory"
                )
            return None, jnp.asarray(False, dtype=jnp.bool_)
        if sidecar is None:
            return (
                self._missing_experiential_memory_input(),
                jnp.asarray(False, dtype=jnp.bool_),
            )
        if not isinstance(sidecar, PrototypeExperientialMemoryInput):
            raise TypeError(
                "experiential_memory_input must be "
                "PrototypeExperientialMemoryInput"
            )
        mask = jnp.asarray(sidecar.next_action_safety_mask)
        expected_mask_shape = (self._config.oak.n_primitive_actions,)
        if mask.shape != expected_mask_shape or mask.dtype != jnp.bool_:
            raise ValueError(
                "experiential_memory_input.next_action_safety_mask must have "
                f"shape {expected_mask_shape} and dtype bool"
            )
        normalized = PrototypeExperientialMemoryInput(
            available=_strict_bool_scalar(
                sidecar.available,
                name="experiential_memory_input.available",
            ),
            current_prototype_decision_id=_strict_decision_id(
                sidecar.current_prototype_decision_id,
                name=(
                    "experiential_memory_input."
                    "current_prototype_decision_id"
                ),
            ),
            next_prototype_decision_id=_strict_decision_id(
                sidecar.next_prototype_decision_id,
                name="experiential_memory_input.next_prototype_decision_id",
            ),
            query_representation_version=_strict_action_scalar(
                sidecar.query_representation_version,
                name="experiential_memory_input.query_representation_version",
            ),
            entry_representation_version=_strict_action_scalar(
                sidecar.entry_representation_version,
                name="experiential_memory_input.entry_representation_version",
            ),
            query_uncertainty=_strict_float32_array(
                sidecar.query_uncertainty,
                (),
                name="experiential_memory_input.query_uncertainty",
            ),
            query_uncertainty_available=_strict_bool_scalar(
                sidecar.query_uncertainty_available,
                name="experiential_memory_input.query_uncertainty_available",
            ),
            entry_uncertainty=_strict_float32_array(
                sidecar.entry_uncertainty,
                (),
                name="experiential_memory_input.entry_uncertainty",
            ),
            entry_uncertainty_available=_strict_bool_scalar(
                sidecar.entry_uncertainty_available,
                name="experiential_memory_input.entry_uncertainty_available",
            ),
            safety_cost=_strict_float32_array(
                sidecar.safety_cost,
                (),
                name="experiential_memory_input.safety_cost",
            ),
            safety_cost_available=_strict_bool_scalar(
                sidecar.safety_cost_available,
                name="experiential_memory_input.safety_cost_available",
            ),
            reliability=_strict_float32_array(
                sidecar.reliability,
                (),
                name="experiential_memory_input.reliability",
            ),
            utility=_strict_float32_array(
                sidecar.utility,
                (),
                name="experiential_memory_input.utility",
            ),
            utility_available=_strict_bool_scalar(
                sidecar.utility_available,
                name="experiential_memory_input.utility_available",
            ),
            provenance_id=_strict_action_scalar(
                sidecar.provenance_id,
                name="experiential_memory_input.provenance_id",
            ),
            source_id=_strict_action_scalar(
                sidecar.source_id,
                name="experiential_memory_input.source_id",
            ),
            next_action_safety_mask=jnp.asarray(mask, dtype=jnp.bool_),
        )
        return normalized, jnp.asarray(True, dtype=jnp.bool_)

    def _missing_partner_policy_fusion_input(
        self,
    ) -> PrototypePartnerPolicyFusionInput:
        """Return the fixed-shape sidecar used for explicit base fallback."""

        fusion = self._partner_policy_fusion
        if fusion is None:
            raise RuntimeError("partner policy fusion is disabled")
        return PrototypePartnerPolicyFusionInput(
            available=jnp.asarray(False, dtype=jnp.bool_),
            prototype_decision_id=jnp.zeros((4,), dtype=jnp.uint32),
            observation_id=jnp.asarray(0, dtype=jnp.int32),
            context_id=jnp.asarray(0, dtype=jnp.int32),
            context_features=jnp.zeros(
                (fusion.config.context_dim,),
                dtype=jnp.float32,
            ),
            safety_action_mask=jnp.ones(
                (fusion.config.n_actions,),
                dtype=jnp.bool_,
            ),
            keyboard_available=jnp.asarray(False, dtype=jnp.bool_),
            keyboard_vector=jnp.zeros(
                (self._config.oak.n_options,),
                dtype=jnp.float32,
            ),
            messages=fusion.empty_messages(),
        )

    @staticmethod
    def _missing_partner_policy_fusion_feedback(
    ) -> PrototypePartnerPolicyFusionFeedback:
        """Return an explicitly unavailable realized-outcome record."""

        unavailable = jnp.asarray(False, dtype=jnp.bool_)
        missing = jnp.asarray(-1, dtype=jnp.int32)
        return PrototypePartnerPolicyFusionFeedback(
            prototype_decision_id=jnp.zeros((4,), dtype=jnp.uint32),
            feedback=PartnerPolicyFusionFeedback(
                available=unavailable,
                decision_id=missing,
                executed_event_id=missing,
                executed_action=missing,
                partner_id=missing,
                assistance_value_available=unavailable,
                realized_assistance_value=jnp.asarray(0.0, dtype=jnp.float32),
                safety_outcome_available=unavailable,
                safety_outcome_ok=unavailable,
            ),
        )

    def _normalize_partner_policy_fusion_input(
        self,
        sidecar: PrototypePartnerPolicyFusionInput | None,
    ) -> tuple[PrototypePartnerPolicyFusionInput | None, Array]:
        """Normalize one fixed next-decision sidecar without advancing state."""

        fusion = self._partner_policy_fusion
        if fusion is None:
            if sidecar is not None:
                raise ValueError(
                    "partner_policy_fusion_input requires partner policy fusion"
                )
            return None, jnp.asarray(False, dtype=jnp.bool_)
        if sidecar is None:
            return (
                self._missing_partner_policy_fusion_input(),
                jnp.asarray(False, dtype=jnp.bool_),
            )
        if not isinstance(sidecar, PrototypePartnerPolicyFusionInput):
            raise TypeError(
                "partner_policy_fusion_input must be "
                "PrototypePartnerPolicyFusionInput"
            )
        if not isinstance(sidecar.messages, PartnerMessageBatch):
            raise TypeError("partner fusion messages must be PartnerMessageBatch")
        mask = jnp.asarray(sidecar.safety_action_mask)
        expected_mask_shape = (fusion.config.n_actions,)
        if mask.dtype != jnp.bool_ or mask.shape != expected_mask_shape:
            raise ValueError(
                "partner_policy_fusion_input.safety_action_mask must have "
                f"shape {expected_mask_shape} and dtype bool"
            )
        normalized = PrototypePartnerPolicyFusionInput(
            available=_strict_bool_scalar(
                sidecar.available,
                name="partner_policy_fusion_input.available",
            ),
            prototype_decision_id=_strict_decision_id(
                sidecar.prototype_decision_id,
                name="partner_policy_fusion_input.prototype_decision_id",
            ),
            observation_id=_strict_action_scalar(
                sidecar.observation_id,
                name="partner_policy_fusion_input.observation_id",
            ),
            context_id=_strict_action_scalar(
                sidecar.context_id,
                name="partner_policy_fusion_input.context_id",
            ),
            context_features=_strict_float32_array(
                sidecar.context_features,
                (fusion.config.context_dim,),
                name="partner_policy_fusion_input.context_features",
            ),
            safety_action_mask=jnp.asarray(mask, dtype=jnp.bool_),
            keyboard_available=_strict_bool_scalar(
                sidecar.keyboard_available,
                name="partner_policy_fusion_input.keyboard_available",
            ),
            keyboard_vector=_strict_float32_array(
                sidecar.keyboard_vector,
                (self._config.oak.n_options,),
                name="partner_policy_fusion_input.keyboard_vector",
            ),
            messages=sidecar.messages,
        )
        # The fusion mechanism performs the complete message static-contract
        # check in the same traced call that uses the batch.
        return normalized, jnp.asarray(True, dtype=jnp.bool_)

    def _normalize_partner_policy_fusion_feedback(
        self,
        feedback: PrototypePartnerPolicyFusionFeedback | None,
    ) -> tuple[PrototypePartnerPolicyFusionFeedback | None, Array]:
        """Normalize realized feedback; dynamic mismatch remains an exact no-op."""

        if self._partner_policy_fusion is None:
            if feedback is not None:
                raise ValueError(
                    "partner_policy_fusion_feedback requires partner policy fusion"
                )
            return None, jnp.asarray(False, dtype=jnp.bool_)
        if feedback is None:
            return (
                self._missing_partner_policy_fusion_feedback(),
                jnp.asarray(False, dtype=jnp.bool_),
            )
        if not isinstance(feedback, PrototypePartnerPolicyFusionFeedback):
            raise TypeError(
                "partner_policy_fusion_feedback must be "
                "PrototypePartnerPolicyFusionFeedback"
            )
        if not isinstance(feedback.feedback, PartnerPolicyFusionFeedback):
            raise TypeError(
                "partner_policy_fusion_feedback.feedback must be "
                "PartnerPolicyFusionFeedback"
            )
        core_feedback = feedback.feedback
        normalized_core = PartnerPolicyFusionFeedback(
            available=_strict_bool_scalar(
                core_feedback.available,
                name="partner_policy_fusion_feedback.available",
            ),
            decision_id=_strict_action_scalar(
                core_feedback.decision_id,
                name="partner_policy_fusion_feedback.decision_id",
            ),
            executed_event_id=_strict_action_scalar(
                core_feedback.executed_event_id,
                name="partner_policy_fusion_feedback.executed_event_id",
            ),
            executed_action=_strict_action_scalar(
                core_feedback.executed_action,
                name="partner_policy_fusion_feedback.executed_action",
            ),
            partner_id=_strict_action_scalar(
                core_feedback.partner_id,
                name="partner_policy_fusion_feedback.partner_id",
            ),
            assistance_value_available=_strict_bool_scalar(
                core_feedback.assistance_value_available,
                name=(
                    "partner_policy_fusion_feedback."
                    "assistance_value_available"
                ),
            ),
            realized_assistance_value=_strict_float32_array(
                core_feedback.realized_assistance_value,
                (),
                name=(
                    "partner_policy_fusion_feedback."
                    "realized_assistance_value"
                ),
            ),
            safety_outcome_available=_strict_bool_scalar(
                core_feedback.safety_outcome_available,
                name="partner_policy_fusion_feedback.safety_outcome_available",
            ),
            safety_outcome_ok=_strict_bool_scalar(
                core_feedback.safety_outcome_ok,
                name="partner_policy_fusion_feedback.safety_outcome_ok",
            ),
        )
        normalized = PrototypePartnerPolicyFusionFeedback(
            prototype_decision_id=_strict_decision_id(
                feedback.prototype_decision_id,
                name="partner_policy_fusion_feedback.prototype_decision_id",
            ),
            feedback=normalized_core,
        )
        return normalized, jnp.asarray(True, dtype=jnp.bool_)

    @staticmethod
    def _gradient_joy_evidence_from_signals(
        sidecar: PrototypeGradientJoyEvidence,
        signals: TypedLearningSignals,
        representation_gradient_valid: Array,
    ) -> GradientJoyEvidence:
        """Join decision-bound probes to causal ensemble learning signals."""
        candidate_available = jnp.asarray(
            representation_gradient_valid,
            dtype=jnp.bool_,
        )
        signal_input_available = signals.availability.input_valid
        delight = sidecar.advantage * sidecar.action_surprisal
        delight_finite = jnp.isfinite(delight)
        safe_delight = jnp.where(
            delight_finite,
            delight,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        advantage_available = sidecar.advantage_available & candidate_available
        surprisal_available = (
            sidecar.action_surprisal_available & candidate_available
        )
        return GradientJoyEvidence(
            objective_probe_gradient=sidecar.objective_probe_gradient,
            retention_probe_gradient=sidecar.retention_probe_gradient,
            safety_cost_gradient=sidecar.safety_cost_gradient,
            objective_probe_available=(
                sidecar.objective_probe_available & candidate_available
            ),
            retention_probe_available=(
                sidecar.retention_probe_available & candidate_available
            ),
            safety_probe_available=(
                sidecar.safety_probe_available & candidate_available
            ),
            probe_independence_attested=(
                sidecar.probe_independence_attested & candidate_available
            ),
            learning_value=LearningValue(
                advantage=sidecar.advantage,
                action_surprisal=sidecar.action_surprisal,
                delight=safe_delight,
                epistemic_surprise=signals.epistemic_surprise,
                aleatoric_uncertainty=signals.aleatoric_uncertainty,
                learning_progress=signals.learning_progress,
                change_probability=signals.change_probability,
                safety_cost=sidecar.safety_cost,
            ),
            learning_value_availability=LearningValueAvailability(
                advantage=advantage_available,
                action_surprisal=surprisal_available,
                delight=(
                    advantage_available
                    & surprisal_available
                    & delight_finite
                ),
                epistemic_surprise=(
                    candidate_available
                    & signal_input_available
                    & signals.availability.epistemic
                ),
                aleatoric_uncertainty=(
                    candidate_available
                    & signal_input_available
                    & signals.availability.aleatoric
                ),
                learning_progress=(
                    candidate_available
                    & signal_input_available
                    & signals.availability.learning_progress
                ),
                change_probability=(
                    candidate_available
                    & signal_input_available
                    & signals.availability.change_probability
                ),
                safety_cost=(
                    sidecar.safety_cost_available & candidate_available
                ),
            ),
        )

    def _state_builder_learning_enabled(self) -> bool:
        """Return whether either legacy or explicit representation learning is on."""
        return (
            self._config.learn_state_builder_from_world_model
            or self._config.representation_gradient_mixer is not None
        )

    def _behavior_representation_gradient(
        self,
        state: PrototypeAgentState,
        reward: Array,
        bootstrap_observation: Array,
        control_discount: Array | None,
    ) -> PrototypeBehaviorGradientResult:
        """Differentiate the current causal pre-update control TD loss.

        Targets and learner parameters are stop-gradient constants. On an idle
        primitive transition this is the base differential-Q loss at
        ``base_last_obs``. While an option executes it is the current
        intra-option loss at the same observation. In particular, an option's
        delayed base semi-MDP update is *not* attributed to the current
        representation because that update belongs to ``option_start_obs``.

        This is a one-step semi-gradient. Eligibility traces, parameter update
        scaling, and option-policy importance weighting remain properties of
        the control learner and are not folded into this representation loss.
        """
        observation_dim = self._config.oak.observation_dim
        stomp_config = self._config.oak.stomp
        stomp_state = self._oak_component_state(state.oak_state).stomp_state
        current = jnp.asarray(state.current_representation, dtype=jnp.float32)
        bootstrap = jnp.asarray(bootstrap_observation, dtype=jnp.float32)
        transition_reward = jnp.asarray(reward, dtype=jnp.float32)
        if control_discount is None:
            transition_discount = jnp.asarray(1.0, dtype=jnp.float32)
            environmental_termination = jnp.asarray(False, dtype=jnp.bool_)
        else:
            transition_discount = jnp.asarray(control_discount, dtype=jnp.float32)
            environmental_termination = transition_discount <= 0.0

        def freeze_leaf(leaf: Any) -> Any:
            dtype = getattr(leaf, "dtype", None)
            return jax.lax.stop_gradient(leaf) if dtype is not None else leaf

        def base_source(_: None) -> tuple[Array, ...]:
            base_state = jax.tree.map(freeze_leaf, stomp_state.base_learner_state)
            action_in_range = (
                (stomp_state.base_last_action >= 0)
                & (stomp_state.base_last_action < stomp_config.n_primitive_actions)
            )
            action = jnp.clip(
                stomp_state.base_last_action,
                0,
                stomp_config.n_primitive_actions - 1,
            )
            bootstrap_predictions = self._oak.stomp_agent.base_learner.predict(
                base_state,
                jax.lax.stop_gradient(bootstrap),
            )
            target = jax.lax.stop_gradient(
                transition_reward
                - jax.lax.stop_gradient(stomp_state.base_average_reward)
                + transition_discount * jnp.max(bootstrap_predictions)
            )

            def loss_fn(representation: Array) -> tuple[Array, Array]:
                prediction = self._oak.stomp_agent.base_learner.predict(
                    base_state,
                    representation,
                )[action]
                error = target - prediction
                return jnp.asarray(0.5, dtype=jnp.float32) * jnp.square(error), prediction

            (loss, prediction), gradient = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(current)
            parameters_finite = _floating_tree_is_finite(base_state)
            option_terminates = jnp.asarray(False, dtype=jnp.bool_)
            return (
                gradient,
                prediction,
                target,
                loss,
                transition_discount,
                option_terminates,
                parameters_finite,
                action_in_range,
            )

        def intra_option_source(_: None) -> tuple[Array, ...]:
            option_in_range = (
                (stomp_state.executing_option >= 0)
                & (stomp_state.executing_option < stomp_config.n_options)
            )
            option_idx = jnp.clip(
                stomp_state.executing_option,
                0,
                stomp_config.n_options - 1,
            )
            action_in_range = (
                (stomp_state.option_last_intra_action >= 0)
                & (
                    stomp_state.option_last_intra_action
                    < stomp_config.n_primitive_actions
                )
            )
            action = jnp.clip(
                stomp_state.option_last_intra_action,
                0,
                stomp_config.n_primitive_actions - 1,
            )
            q_weights = jax.lax.stop_gradient(
                stomp_state.option_policies.q_weights[option_idx]
            )
            average_reward = jax.lax.stop_gradient(
                stomp_state.option_policies.average_rewards[option_idx]
            )
            pseudo_reward = compute_pseudo_reward(
                self._oak.stomp_agent.spec_arrays,
                option_idx,
                jax.lax.stop_gradient(bootstrap),
            )
            option_terminates = (
                check_option_terminated(
                    self._oak.stomp_agent.spec_arrays,
                    option_idx,
                    jax.lax.stop_gradient(bootstrap),
                    stomp_state.option_steps + 1,
                )
                | environmental_termination
            )
            bootstrap_discount = jnp.where(
                option_terminates,
                jnp.asarray(0.0, dtype=jnp.float32),
                transition_discount,
            )
            target = jax.lax.stop_gradient(
                pseudo_reward
                - average_reward
                + bootstrap_discount * jnp.max(q_weights @ bootstrap)
            )

            def loss_fn(representation: Array) -> tuple[Array, Array]:
                prediction = q_weights[action] @ representation
                error = target - prediction
                return jnp.asarray(0.5, dtype=jnp.float32) * jnp.square(error), prediction

            (loss, prediction), gradient = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(current)
            parameters_finite = (
                jnp.all(jnp.isfinite(q_weights))
                & jnp.isfinite(average_reward)
            )
            return (
                gradient,
                prediction,
                target,
                loss,
                bootstrap_discount,
                option_terminates,
                parameters_finite,
                option_in_range & action_in_range,
            )

        executing = stomp_state.executing_option >= 0
        (
            raw_gradient,
            prediction,
            target,
            loss,
            bootstrap_discount,
            option_terminates,
            parameters_finite,
            source_indices_valid,
        ) = jax.lax.cond(
            executing,
            intra_option_source,
            base_source,
            operand=None,
        )
        inputs_finite = (
            jnp.all(jnp.isfinite(current))
            & jnp.all(jnp.isfinite(bootstrap))
            & jnp.isfinite(transition_reward)
            & jnp.isfinite(transition_discount)
            & (transition_discount >= 0.0)
            & (transition_discount <= 1.0)
        )
        td_error = target - prediction
        gradient_finite = jnp.all(jnp.isfinite(raw_gradient))
        gradient_norm = jnp.linalg.norm(raw_gradient)
        prediction_finite = jnp.isfinite(prediction)
        target_finite = jnp.isfinite(target)
        td_error_finite = jnp.isfinite(td_error)
        loss_finite = jnp.isfinite(loss)
        gradient_norm_finite = jnp.isfinite(gradient_norm)
        numerics_valid = (
            prediction_finite
            & target_finite
            & td_error_finite
            & loss_finite
            & gradient_finite
            & gradient_norm_finite
        )
        valid = (
            inputs_finite
            & parameters_finite
            & source_indices_valid
            & numerics_valid
        )
        safe_gradient = jnp.where(
            valid,
            raw_gradient,
            jnp.zeros((observation_dim,), dtype=jnp.float32),
        )
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        # Rejected optimizer-control sidecars must remain safe to scan, log,
        # checkpoint alongside reports, or serialize with ``allow_nan=False``.
        # Boolean provenance flags retain the rejection reason; floating
        # payloads become one canonical finite zero record.
        safe_prediction = jnp.where(valid, prediction, zero)
        safe_target = jnp.where(valid, target, zero)
        safe_td_error = jnp.where(valid, td_error, zero)
        safe_loss = jnp.where(valid, loss, zero)
        safe_norm = jnp.where(valid, gradient_norm, zero)
        safe_bootstrap_discount = jnp.where(valid, bootstrap_discount, zero)
        diagnostics = PrototypeBehaviorGradientDiagnostics(
            source_available=jnp.asarray(True, dtype=jnp.bool_),
            idle_base_source=~executing,
            intra_option_source=executing,
            parameters_finite=parameters_finite,
            inputs_finite=inputs_finite,
            source_indices_valid=source_indices_valid,
            prediction_finite=prediction_finite,
            target_finite=target_finite,
            td_error_finite=td_error_finite,
            loss_finite=loss_finite,
            gradient_norm_finite=gradient_norm_finite,
            prediction=safe_prediction,
            target=safe_target,
            td_error=safe_td_error,
            loss=safe_loss,
            gradient_norm=safe_norm,
            bootstrap_discount=safe_bootstrap_discount,
            option_terminates=option_terminates,
            gradient_finite=gradient_finite,
            valid=valid,
            rejected=~valid,
        )
        return PrototypeBehaviorGradientResult(
            gradient=safe_gradient,
            valid=valid,
            diagnostics=diagnostics,
        )

    def _apply_state_builder_learning(
        self,
        source_state: Any,
        destination_state: Any,
        representation_gradient: Array,
        representation_gradient_valid: Array,
        signals: TypedLearningSignals,
        joy_sidecar: PrototypeGradientJoyEvidence | None,
    ) -> tuple[
        Any,
        StateBuilderLearningDiagnostics,
        GradientJoyApplicationResult | None,
    ]:
        """Propose from the causal source and commit into advanced recurrence."""
        if not self._state_builder_learning_enabled():
            return (
                destination_state,
                _unavailable_state_builder_learning_diagnostics(),
                None,
            )
        if self._state_builder is None:
            raise RuntimeError("state builder learning has no configured builder")

        proposal = self._state_builder.propose_learning_update(
            source_state,
            representation_gradient,
        )
        application: GradientJoyApplicationResult | None = None
        approved = jnp.asarray(
            representation_gradient_valid,
            dtype=jnp.bool_,
        )
        candidate_update = proposal.candidate_parameter_update
        if self._config.gradient_joy is not None:
            if joy_sidecar is None:
                raise RuntimeError("configured gradient joy requires a sidecar")
            audit_evidence = self._gradient_joy_evidence_from_signals(
                joy_sidecar,
                signals,
                approved,
            )
            application = apply_gradient_joy_update(
                proposal.source_parameters,
                candidate_update,
                audit_evidence,
                self._config.gradient_joy,
            )
            candidate_update = application.parameters - proposal.source_parameters
            approved = approved & proposal.valid & application.applied

        filtered_proposal = replace_state_builder_learning_proposal_update(
            proposal,
            candidate_update,
            approved,
        )
        learned_state, learning_diagnostics = (
            self._state_builder.commit_learning_update(
                destination_state,
                filtered_proposal,
            )
        )
        return learned_state, learning_diagnostics, application

    def _base_representation_dim(self) -> int:
        """Return the builder width before optional pair augmentation."""

        feature_config = self._config.prototype_feature_lifecycle
        if feature_config is not None:
            return feature_config.base_feature_dim
        return self._config.oak.observation_dim

    def _augment_base_representation(
        self,
        feature_state: PrototypeFeatureLifecycleState | None,
        base_representation: Array,
    ) -> Array:
        """Apply the configured fixed-width pair bank, or return the base."""

        lifecycle = self._prototype_feature_lifecycle
        if lifecycle is None:
            return base_representation
        if feature_state is None:
            raise RuntimeError(
                "configured prototype feature lifecycle requires persistent state"
            )
        return lifecycle.augment(feature_state, base_representation)

    def _unavailable_feature_lifecycle_diagnostics(
        self,
        state: PrototypeFeatureLifecycleState,
    ) -> PrototypeFeatureLifecycleIntegrationDiagnostics:
        """Return one finite neutral record without touching learner state."""

        lifecycle = self._prototype_feature_lifecycle
        if lifecycle is None:
            raise RuntimeError("prototype feature lifecycle is disabled")
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        false = jnp.asarray(False, dtype=jnp.bool_)
        generation = state.router_state.generation_count
        return PrototypeFeatureLifecycleIntegrationDiagnostics(
            available=false,
            target=zero,
            target_available=false,
            pullback_gradient=jnp.zeros(
                (self._base_representation_dim(),),
                dtype=jnp.float32,
            ),
            pullback_valid=false,
            pullback_semantic_generation=generation,
            prediction=zero,
            error=zero,
            metrics=jnp.zeros((7,), dtype=jnp.float32),
            lifecycle=lifecycle.unavailable_diagnostics(generation),
            outer_transaction_committed=false,
        )

    def _raw_observation_dim(self) -> int:
        if self._state_builder is not None:
            return self._state_builder.observation_dim()
        if self._config.gru_perception is not None:
            return self._config.gru_perception.observation_dim
        return self._config.oak.observation_dim

    @staticmethod
    def _oak_state_numeric_valid(
        oak_agent: OaKAgent,
        state: OaKState,
    ) -> Array:
        """Authenticate one OaK wrapper and its complete nested STOMP state."""

        counter_ceiling = jnp.where(
            state.step_count < jnp.asarray(_INT32_MAX, dtype=jnp.int32),
            state.step_count + jnp.asarray(1, dtype=jnp.int32),
            state.step_count,
        )
        return (
            _lifetime_counter_valid(state.step_words, state.step_count)
            & _lifetime_counter_valid(
                state.stomp_state.step_words,
                state.stomp_state.step_count,
            )
            & jnp.all(state.step_words == state.stomp_state.step_words)
            & (state.step_count == state.stomp_state.step_count)
            & jnp.all(state.execution_counts >= 0)
            & jnp.all(state.execution_counts <= counter_ceiling)
            & jnp.all(jnp.isfinite(state.cumulative_pseudo_rewards))
            & jnp.all(jnp.isfinite(state.utility_ema))
            & oak_agent.stomp_agent.state_valid(state.stomp_state)
        )

    def _state_numeric_valid(self, state: PrototypeAgentState) -> Array:
        """Validate state numerics while preserving world-model init sentinels."""
        sanitized_state = state
        oak_state_valid = self._oak_state_numeric_valid(
            self._oak,
            self._oak_component_state(state.oak_state),
        )
        world_model_bounds_valid = jnp.asarray(True)
        state_builder_valid = jnp.asarray(True)
        feature_lifecycle_valid = jnp.asarray(True, dtype=jnp.bool_)
        interaction_state_valid = jnp.asarray(True, dtype=jnp.bool_)
        if self._partner_policy_fusion is not None:
            interaction = self._partner_interaction_state(state.ia_state)
            if (
                interaction.feedback_prototype_decision_id.shape != (4,)
                or interaction.feedback_prototype_decision_id.dtype
                != jnp.uint32
                or interaction.feedback_prototype_decision_id_available.shape
                != ()
                or interaction.feedback_prototype_decision_id_available.dtype
                != jnp.bool_
            ):
                raise ValueError(
                    "partner feedback Prototype owner has the wrong shape or dtype"
                )
            partner_state = self._partner_fusion_component_state(state.ia_state)
            owner_generation = interaction.feedback_prototype_decision_id[2:]
            current_generation = state.current_decision_id[2:]
            owner_not_from_future = (
                (owner_generation[0] < current_generation[0])
                | (
                    (owner_generation[0] == current_generation[0])
                    & (owner_generation[1] <= current_generation[1])
                )
            )
            owner_belongs_to_lifecycle = jnp.all(
                interaction.feedback_prototype_decision_id[:2]
                == state.current_decision_id[:2]
            )
            interaction_state_valid = (
                self._partner_policy_fusion._state_valid_predicate(partner_state)
                & (
                    interaction.feedback_prototype_decision_id_available
                    == partner_state.feedback_armed
                )
                & jnp.where(
                    interaction.feedback_prototype_decision_id_available,
                    owner_belongs_to_lifecycle & owner_not_from_future,
                    jnp.all(interaction.feedback_prototype_decision_id == 0),
                )
            )
        if self._experiential_memory_policy is not None:
            memory_state = self._experiential_memory_component_state(
                state.ia_state
            )
            interaction_state_valid = (
                interaction_state_valid
                & self._experiential_memory_policy.state_valid(memory_state)
            )
        if self._state_builder is not None:
            builder_state = self._builder_component_state(
                state.state_builder_state
            )
            state_builder_valid = self._state_builder.state_valid(
                builder_state
            )
        if self._prototype_feature_lifecycle is not None:
            feature_state = self._feature_lifecycle_component_state(
                state.state_builder_state
            )
            consumer_binding = self._feature_consumer_binding(state.oak_state)
            maximum_observations = jnp.asarray(
                self._config.prototype_feature_lifecycle.max_observations,
                dtype=jnp.int32,
            )
            expected_observations = jnp.minimum(
                jnp.maximum(state.step_count, jnp.asarray(0, dtype=jnp.int32)),
                maximum_observations,
            )
            feature_lifecycle_valid = (
                self._prototype_feature_lifecycle.state_valid(feature_state)
                & self._prototype_feature_lifecycle.consumer_binding_valid(
                    feature_state,
                    consumer_binding,
                )
                & (feature_state.observe_count == expected_observations)
            )
        if (
            self._recurrent_latent_world_model_ensemble is not None
            and state.world_model_state is not None
        ):
            world_model_bounds_valid = self._recurrent_wrapper_numeric_valid(
                cast(
                    PrototypeRecurrentLatentWorldModelState,
                    state.world_model_state,
                )
            )
            sanitized_state = cast(
                PrototypeAgentState,
                state.replace(world_model_state=None),
            )
        elif (
            self._model_replay_rehearsal is not None
            and state.world_model_state is not None
        ):
            world_model_bounds_valid = self._model_replay_rehearsal.state_valid(
                state.world_model_state
            )
            sanitized_state = cast(
                PrototypeAgentState,
                state.replace(world_model_state=None),
            )
        elif (
            self._world_model_ensemble is not None
            and state.world_model_state is not None
        ):
            world_model_bounds_valid = self._world_model_ensemble.state_valid(
                state.world_model_state
            )
            sanitized_state = cast(
                PrototypeAgentState,
                state.replace(world_model_state=None),
            )
        elif self._world_model is not None and state.world_model_state is not None:
            world_model_state = state.world_model_state
            observation_min = world_model_state.observation_min
            observation_max = world_model_state.observation_max
            reward_min = world_model_state.reward_min
            reward_max = world_model_state.reward_max
            pristine_bounds = (
                jnp.all(jnp.isposinf(observation_min))
                & jnp.all(jnp.isneginf(observation_max))
                & jnp.isposinf(reward_min)
                & jnp.isneginf(reward_max)
            )
            finite_bounds = (
                jnp.all(jnp.isfinite(observation_min))
                & jnp.all(jnp.isfinite(observation_max))
                & jnp.isfinite(reward_min)
                & jnp.isfinite(reward_max)
            )
            world_model_bounds_valid = jnp.where(
                world_model_state.step_count == 0,
                pristine_bounds,
                finite_bounds,
            )
            sanitized_world_model_state = world_model_state.replace(
                observation_min=jnp.zeros_like(observation_min),
                observation_max=jnp.zeros_like(observation_max),
                reward_min=jnp.zeros_like(reward_min),
                reward_max=jnp.zeros_like(reward_max),
            )
            sanitized_state = cast(
                PrototypeAgentState,
                state.replace(world_model_state=sanitized_world_model_state),
            )
        return (
            oak_state_valid
            & world_model_bounds_valid
            & state_builder_valid
            & feature_lifecycle_valid
            & interaction_state_valid
            & _floating_tree_is_finite(sanitized_state)
        )

    def _representation_cache_consistent(
        self,
        state: PrototypeAgentState,
    ) -> Array:
        """Check that causal representation caches agree with their owners."""
        oak_state = self._oak_component_state(state.oak_state)
        representation_consistent = jnp.array(True)
        builder_count_consistent = jnp.array(True)
        if self._state_builder is not None:
            builder_state = self._builder_component_state(
                state.state_builder_state
            )
            rebuilt_base_representation = self._state_builder.encode(
                builder_state,
                state.current_raw_observation,
            )
            feature_state = (
                self._feature_lifecycle_component_state(
                    state.state_builder_state
                )
                if self._prototype_feature_lifecycle is not None
                else None
            )
            rebuilt_representation = self._augment_base_representation(
                feature_state,
                rebuilt_base_representation,
            )
            representation_consistent = jnp.array_equal(
                rebuilt_representation,
                state.current_representation,
            )
            builder_count_consistent = (
                builder_state.step_count
                == state.observation_event_count
            )
        elif state.gru_state is not None:
            representation_consistent = jnp.array_equal(
                jnp.concatenate(
                    (state.current_raw_observation, state.gru_state.hidden)
                ),
                state.current_representation,
            )
        else:
            representation_consistent = jnp.array_equal(
                state.current_raw_observation,
                state.current_representation,
            )
        return (
            jnp.all(jnp.isfinite(state.current_raw_observation))
            & jnp.all(jnp.isfinite(state.current_representation))
            & (state.observation_event_count >= 1)
            & (oak_state.step_count == state.step_count)
            & (oak_state.stomp_state.step_count == state.step_count)
            & representation_consistent
            & builder_count_consistent
            & jnp.array_equal(
                state.current_representation,
                oak_state.stomp_state.base_last_obs,
            )
        )

    def _state_cache_structurally_consistent(
        self,
        state: PrototypeAgentState,
    ) -> Array:
        """Check the structure of the currently armed representation/action cache."""
        action_in_range = (state.current_action >= 0) & (
            state.current_action < self._config.oak.n_primitive_actions
        )
        stomp_state = self._oak_component_state(state.oak_state).stomp_state
        executing_option_valid = (
            (stomp_state.executing_option >= -1)
            & (stomp_state.executing_option < self._config.oak.n_options)
        )
        # Bind the dispatched primitive command to the learner whose current
        # TD loss owns its representation credit. An idle transition belongs
        # to the base primitive head; an executing transition belongs to the
        # active option's intra-policy head. Merely checking
        # ``last_primitive_action`` would permit a valid-range owner mismatch.
        control_gradient_owner_matches = jnp.where(
            stomp_state.executing_option >= 0,
            stomp_state.option_last_intra_action == state.current_action,
            stomp_state.base_last_action == state.current_action,
        )
        option_identity_matches = jnp.where(
            stomp_state.executing_option >= 0,
            stomp_state.base_last_action
            == self._config.oak.n_primitive_actions + stomp_state.executing_option,
            jnp.asarray(True, dtype=jnp.bool_),
        )
        recurrent_cache_consistent = jnp.asarray(True, dtype=jnp.bool_)
        if self._recurrent_latent_world_model_ensemble is not None:
            recurrent_cache_consistent = self._recurrent_armed_cache_consistent(
                state
            )
        observation_maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
        return (
            self._representation_cache_consistent(state)
            & action_in_range
            & executing_option_valid
            & (state.current_action == stomp_state.last_primitive_action)
            & control_gradient_owner_matches
            & option_identity_matches
            & _step_counter_can_process(state.step_count)
            & (state.observation_event_count <= observation_maximum - 2)
            & recurrent_cache_consistent
        )

    def _state_cache_consistent(self, state: PrototypeAgentState) -> Array:
        """Check the complete finite currently armed decision cache."""
        return (
            self._state_numeric_valid(state)
            & self._state_cache_structurally_consistent(state)
        )

    def _disarmed_state_structurally_consistent(
        self,
        state: PrototypeAgentState,
    ) -> Array:
        """Validate a state disarmed by decision- or counter-capacity exhaustion."""
        counter = state.current_decision_id[2:]
        generation_maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
        step_maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
        exhausted = jnp.all(counter == generation_maximum) | (
            state.step_count >= step_maximum - 1
        ) | (state.observation_event_count >= step_maximum - 1)
        recurrent_cache_disarmed = jnp.asarray(True, dtype=jnp.bool_)
        if self._recurrent_latent_world_model_ensemble is not None:
            recurrent_wrapper = cast(
                PrototypeRecurrentLatentWorldModelState,
                state.world_model_state,
            )
            recurrent_cache_disarmed = ~recurrent_wrapper.decision_cache.valid
        return (
            (~state.started)
            & (state.step_count > 0)
            & (state.current_action == -1)
            & exhausted
            & self._representation_cache_consistent(state)
            & recurrent_cache_disarmed
        )

    def _checkpoint_state_structurally_valid(
        self,
        state: PrototypeAgentState,
    ) -> Array:
        """Validate armed, pristine, or capacity-exhausted checkpoint structure."""
        pristine = self._pristine_state_structurally_consistent(state)
        disarmed = self._disarmed_state_structurally_consistent(state)
        armed = state.started & self._state_cache_structurally_consistent(state)
        return pristine | disarmed | armed

    def _checkpoint_state_valid(self, state: PrototypeAgentState) -> Array:
        """Validate checkpoint structure and every floating state leaf."""
        return (
            self._state_numeric_valid(state)
            & self._checkpoint_state_structurally_valid(state)
        )

    def _pristine_state_structurally_consistent(
        self,
        state: PrototypeAgentState,
    ) -> Array:
        """Return whether ``state`` has the structure of an untouched init."""
        oak_state = self._oak_component_state(state.oak_state)
        recurrent_fresh = jnp.array(True)
        if self._state_builder is not None:
            recurrent_fresh = (
                self._builder_component_state(
                    state.state_builder_state
                ).step_count
                == 0
            )
        elif state.gru_state is not None:
            recurrent_fresh = jnp.all(state.gru_state.hidden == 0.0)
        recurrent_world_fresh = jnp.asarray(True, dtype=jnp.bool_)
        if self._recurrent_latent_world_model_ensemble is not None:
            recurrent_wrapper = cast(
                PrototypeRecurrentLatentWorldModelState,
                state.world_model_state,
            )
            recurrent_world_fresh = (
                (~recurrent_wrapper.decision_cache.valid)
                & (recurrent_wrapper.model_state.event_count == 0)
                & (recurrent_wrapper.signal_state.step_count == 0)
            )
        partner_fusion_fresh = jnp.asarray(True, dtype=jnp.bool_)
        if self._partner_policy_fusion is not None:
            partner_fusion_fresh = _tree_arrays_equal(
                self._partner_fusion_component_state(state.ia_state),
                self._partner_policy_fusion.init(),
            )
        experiential_memory_fresh = jnp.asarray(True, dtype=jnp.bool_)
        if self._experiential_memory is not None:
            experiential_memory_fresh = _tree_arrays_equal(
                self._experiential_memory_component_state(state.ia_state),
                self._experiential_memory.init(),
            )
        return (
            (~state.started)
            & (state.step_count == 0)
            & (state.observation_event_count == 0)
            & (state.current_action == -1)
            & jnp.all(state.current_decision_id[2:] == 0)
            & jnp.all(state.current_raw_observation == 0.0)
            & jnp.all(state.current_representation == 0.0)
            & (oak_state.step_count == 0)
            & (oak_state.stomp_state.step_count == 0)
            & jnp.all(oak_state.stomp_state.base_last_obs == 0.0)
            & recurrent_fresh
            & recurrent_world_fresh
            & partner_fusion_fresh
            & experiential_memory_fresh
        )

    def _pristine_state_consistent(self, state: PrototypeAgentState) -> Array:
        """Return whether ``state`` is a finite untouched result of :meth:`init`."""
        return (
            self._state_numeric_valid(state)
            & self._pristine_state_structurally_consistent(state)
        )

    # -- Serialization --------------------------------------------------------

    def to_config(self) -> dict[str, Any]:
        """Serialize agent configuration."""
        return self._config.to_config()

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> PrototypeAgent:
        """Reconstruct from :meth:`to_config` output."""
        return cls(PrototypeAgentConfig.from_config(payload))

    # -- Lifecycle ------------------------------------------------------------

    def init(
        self,
        key: Array,
        *,
        lifecycle_id: Array | None = None,
    ) -> PrototypeAgentState:
        """Initialise all sub-states.

        Args:
            key: JAX PRNG key.
            lifecycle_id: Optional caller-owned two-word uint32 session nonce.
                Supplying a persisted runner epoch gives exact cross-session
                transition ownership. When omitted, a deterministic nonce is
                derived from ``key``; callers must then avoid key reuse across
                concurrently valid sessions.

        Returns:
            Fresh :class:`PrototypeAgentState`.
        """
        session_key, oak_key, wm_key, horde_key, ia_key, gru_key, builder_key = jr.split(
            key,
            7,
        )
        if lifecycle_id is None:
            lifecycle_words = jr.key_data(session_key)
        else:
            lifecycle_words = _strict_uint32_words(
                lifecycle_id,
                (2,),
                name="lifecycle_id",
            )
        oak_state = self._oak.init(oak_key)

        wm_state: Any = None
        buf_state: Any = None
        if self._world_model is not None and self._buffer is not None:
            wm_state = self._world_model.init(wm_key)
            buf_state = self._buffer.init()
        elif self._world_model_ensemble is not None:
            wm_state = self._world_model_ensemble.init(wm_key)
        elif self._model_replay_rehearsal is not None:
            wm_state = self._model_replay_rehearsal.init(wm_key)
        elif (
            self._recurrent_latent_world_model_ensemble is not None
            and self._recurrent_signal_estimator is not None
        ):
            recurrent_model_state = (
                self._recurrent_latent_world_model_ensemble.init(wm_key)
            )
            recurrent_decision_cache = self._recurrent_decision_for_observation(
                recurrent_model_state,
                jnp.zeros(
                    (self._config.oak.observation_dim,),
                    dtype=jnp.float32,
                ),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(False, dtype=jnp.bool_),
            )
            wm_state = PrototypeRecurrentLatentWorldModelState(
                model_state=recurrent_model_state,
                decision_cache=recurrent_decision_cache,
                signal_state=self._recurrent_signal_estimator.init(),
            )

        horde_state: Any = None
        if self._horde is not None:
            horde_state = self._horde.init(self._config.oak.observation_dim, horde_key)

        ia_state: Any = None
        if self._ia is not None:
            ia_state = self._ia.init(ia_key)
        partner_fusion_state: Any = None
        if self._partner_policy_fusion is not None:
            partner_fusion_state = self._partner_policy_fusion.init()
        experiential_memory_state: ExperientialMemoryState | None = None
        if self._experiential_memory is not None:
            experiential_memory_state = self._experiential_memory.init()
        interaction_state = self._interaction_slot(
            ia_state,
            partner_fusion_state,
            experiential_memory_state=experiential_memory_state,
        )

        gru_state: Any = None
        if self._config.gru_perception is not None:
            gru_state = _init_gru_state(self._config.gru_perception, gru_key)

        builder_state: Any = None
        if self._state_builder is not None:
            builder_state = self._state_builder.init(builder_key)
        feature_lifecycle_state: PrototypeFeatureLifecycleState | None = None
        feature_consumer_binding: PrototypeFeatureConsumerBinding | None = None
        if self._prototype_feature_lifecycle is not None:
            feature_key = jr.fold_in(
                builder_key,
                _PROTOTYPE_FEATURE_LIFECYCLE_KEY_TAG,
            )
            (
                feature_lifecycle_state,
                feature_consumer_binding,
            ) = self._prototype_feature_lifecycle.init_bound(feature_key)
        representation_state = self._representation_state_slot(
            builder_state,
            feature_lifecycle_state,
        )

        initial_state = PrototypeAgentState(
            oak_state=self._oak_state_slot(
                oak_state,
                feature_consumer_binding,
            ),
            world_model_state=wm_state,
            buffer_state=buf_state,
            horde_state=horde_state,
            ia_state=interaction_state,
            gru_state=gru_state,
            state_builder_state=representation_state,
            current_raw_observation=jnp.zeros(
                (self._raw_observation_dim(),),
                dtype=jnp.float32,
            ),
            current_representation=jnp.zeros(
                (self._config.oak.observation_dim,),
                dtype=jnp.float32,
            ),
            current_action=jnp.array(-1, dtype=jnp.int32),
            current_decision_id=jnp.concatenate(
                (lifecycle_words, jnp.zeros((2,), dtype=jnp.uint32))
            ),
            started=jnp.array(False),
            observation_event_count=jnp.array(0, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )
        return cast(
            PrototypeAgentState,
            jax.tree.map(
                lambda leaf: (
                    jnp.asarray(leaf)
                    if isinstance(leaf, (bool, int, float))
                    else leaf
                ),
                initial_state,
            ),
        )

    def start(
        self,
        state: PrototypeAgentState,
        initial_observation: Array,
    ) -> PrototypeAgentState:
        """Prime the agent with an initial observation.

        Must be called once before :meth:`update`.

        Args:
            state: Uninitialised state from :meth:`init`.
            initial_observation: First environment observation.

        Returns:
            State with OaK (and optionally IA) primed.
        """
        raw_obs = _strict_float_array(
            initial_observation,
            (self._raw_observation_dim(),),
            name="initial_observation",
        )
        finite = jnp.all(jnp.isfinite(raw_obs))
        pristine = self._pristine_state_consistent(state)
        if not _contains_tracer((raw_obs, state.started, state.step_count)):
            if not bool(pristine):
                raise RuntimeError("start requires a fresh unstarted PrototypeAgentState")
            if not bool(finite):
                raise ValueError("initial_observation must be finite")
        safe_raw_obs = jnp.where(finite, raw_obs, jnp.zeros_like(raw_obs))

        def prime(fresh_state: PrototypeAgentState) -> PrototypeAgentState:
            new_gru_state = fresh_state.gru_state
            new_builder_state = self._builder_component_state(
                fresh_state.state_builder_state
            )
            feature_state = (
                self._feature_lifecycle_component_state(
                    fresh_state.state_builder_state
                )
                if self._prototype_feature_lifecycle is not None
                else None
            )
            obs_for_oak = safe_raw_obs
            if self._state_builder is not None:
                new_builder_state, base_obs_for_oak = self._state_builder.start(
                    new_builder_state,
                    safe_raw_obs,
                )
                obs_for_oak = self._augment_base_representation(
                    feature_state,
                    base_obs_for_oak,
                )
            elif fresh_state.gru_state is not None:
                new_gru_state, obs_for_oak = _gru_step(
                    fresh_state.gru_state,
                    safe_raw_obs,
                )
            feature_consumer_binding = (
                self._feature_consumer_binding(fresh_state.oak_state)
                if self._prototype_feature_lifecycle is not None
                else None
            )
            new_oak = self._oak.start(
                self._oak_component_state(fresh_state.oak_state),
                obs_for_oak,
            )
            previous_ia = self._ia_component_state(fresh_state.ia_state)
            new_ia = previous_ia
            if self._ia is not None and previous_ia is not None:
                new_ia = self._ia.start(previous_ia, obs_for_oak)
            partner_state: Any = None
            if self._partner_policy_fusion is not None:
                partner_state = self._partner_fusion_component_state(
                    fresh_state.ia_state
                )
            memory_state: ExperientialMemoryState | None = None
            if self._experiential_memory is not None:
                memory_state = self._experiential_memory_component_state(
                    fresh_state.ia_state
                )
            new_interaction_state = self._interaction_slot(
                new_ia,
                partner_state,
                experiential_memory_state=memory_state,
            )
            new_world_model_state = fresh_state.world_model_state
            if self._recurrent_latent_world_model_ensemble is not None:
                recurrent_wrapper = cast(
                    PrototypeRecurrentLatentWorldModelState,
                    fresh_state.world_model_state,
                )
                recurrent_decision_cache = (
                    self._recurrent_decision_for_observation(
                        recurrent_wrapper.model_state,
                        obs_for_oak,
                        new_oak.stomp_state.last_primitive_action,
                        jnp.asarray(True, dtype=jnp.bool_),
                    )
                )
                new_world_model_state = recurrent_wrapper.replace(
                    decision_cache=recurrent_decision_cache,
                )
            candidate = cast(
                PrototypeAgentState,
                fresh_state.replace(
                    oak_state=self._oak_state_slot(
                        new_oak,
                        feature_consumer_binding,
                    ),
                    world_model_state=new_world_model_state,
                    ia_state=new_interaction_state,
                    gru_state=new_gru_state,
                    state_builder_state=self._representation_state_slot(
                        new_builder_state,
                        feature_state,
                    ),
                    current_raw_observation=safe_raw_obs,
                    current_representation=obs_for_oak,
                    current_action=new_oak.stomp_state.last_primitive_action,
                    current_decision_id=fresh_state.current_decision_id.at[2:].set(
                        jnp.zeros((2,), dtype=jnp.uint32)
                    ),
                    started=jnp.array(True),
                    observation_event_count=jnp.array(1, dtype=jnp.int32),
                ),
            )
            return cast(
                PrototypeAgentState,
                jax.lax.cond(
                    self._state_cache_consistent(candidate),
                    lambda _: candidate,
                    lambda _: fresh_state,
                    operand=None,
                ),
            )

        result = cast(
            PrototypeAgentState,
            jax.lax.cond(pristine & finite, prime, lambda unchanged: unchanged, state),
        )
        if not _contains_tracer((raw_obs, state.started, state.step_count)):
            if bool(pristine & finite) and not bool(result.started):
                raise ValueError("initial_observation produced a non-finite agent state")
        return result

    def act(
        self,
        state: PrototypeAgentState,
        observation: Array,
    ) -> Int[Array, ""]:
        """Return the cached primitive command for the current observation.

        This method never advances or re-encodes recurrent state. The retained
        observation argument is an ownership assertion for legacy callers;
        eager mismatches raise and traced mismatches return the unarmed ``-1``
        sentinel. Use :meth:`greedy_action` for a pure counterfactual query
        that was not selected for dispatch.
        """
        obs = _strict_float_array(
            observation,
            (self._raw_observation_dim(),),
            name="observation",
        )
        matches = jnp.all(jnp.isfinite(obs)) & jnp.array_equal(
            obs,
            state.current_raw_observation,
        )
        state_consistent = self._state_cache_consistent(state)
        valid = state.started & state_consistent & matches
        if not _contains_tracer((obs, state.started, state.current_action)):
            if not bool(state.started):
                raise ValueError("agent must be started before act")
            if not bool(state_consistent):
                raise ValueError("agent decision cache is inconsistent")
            if not bool(matches):
                raise ValueError("observation does not match the cached decision")
        return jnp.where(
            valid,
            state.current_action,
            jnp.asarray(-1, dtype=jnp.int32),
        )

    def decision(self, state: PrototypeAgentState) -> PrototypeDecision:
        """Return the complete currently armed decision record.

        Carry this record across the environment boundary and copy its three
        fields into :class:`PrototypeTransition`. The generation closes the
        replay ambiguity left by comparing observation and action alone. A
        well-formed state that exhausted its decision or counter capacity
        returns the explicit unarmed ``-1`` sentinel instead of raising.
        """
        armed_consistent = state.started & self._state_cache_consistent(state)
        disarmed_consistent = (
            self._state_numeric_valid(state)
            & self._disarmed_state_structurally_consistent(state)
        )
        valid = armed_consistent | disarmed_consistent
        if not _contains_tracer((state.started, state.current_action)) and not bool(valid):
            raise ValueError("agent state does not contain a valid decision lifecycle")
        action = jnp.where(
            armed_consistent,
            state.current_action,
            jnp.asarray(-1, dtype=jnp.int32),
        )
        return PrototypeDecision(
            observation=state.current_raw_observation,
            action=action,
            decision_id=state.current_decision_id,
            armed=armed_consistent,
        )

    def greedy_action(
        self,
        state: PrototypeAgentState,
        observation: Array,
    ) -> Int[Array, ""]:
        """Pure counterfactual greedy action without advancing causal state."""
        raw_obs = _strict_float_array(
            observation,
            (self._raw_observation_dim(),),
            name="observation",
        )
        obs = raw_obs
        if self._state_builder is not None:
            base_obs = self._state_builder.encode(
                self._builder_component_state(state.state_builder_state),
                raw_obs,
            )
            feature_state = (
                self._feature_lifecycle_component_state(
                    state.state_builder_state
                )
                if self._prototype_feature_lifecycle is not None
                else None
            )
            obs = self._augment_base_representation(
                feature_state,
                base_obs,
            )
        elif state.gru_state is not None:
            obs = jnp.concatenate([raw_obs, state.gru_state.hidden], axis=0)
        n_prim = self._config.oak.n_primitive_actions
        all_q = self._oak.base_q_values(
            self._oak_component_state(state.oak_state),
            obs,
        )
        return jnp.argmax(all_q[:n_prim]).astype(jnp.int32)

    def _oak_counterfactual_dispatch_score(
        self,
        oak_state: OaKState,
        decision_observation: Array,
    ) -> Array:
        """Return the exact current dispatch owner's finite-Q candidate score."""

        stomp = oak_state.stomp_state
        n_primitive = self._config.oak.n_primitive_actions
        n_options = self._config.oak.n_options
        base_values = self._oak.base_q_values(oak_state, decision_observation)
        base_index = jnp.clip(
            stomp.base_last_action,
            0,
            n_primitive + n_options - 1,
        )
        base_score = base_values[base_index]
        option_index = jnp.clip(stomp.executing_option, 0, n_options - 1)
        primitive_index = jnp.clip(
            stomp.last_primitive_action,
            0,
            n_primitive - 1,
        )
        option_score = jnp.dot(
            stomp.option_policies.q_weights[option_index, primitive_index],
            decision_observation,
        )
        return jnp.where(stomp.executing_option >= 0, option_score, base_score)

    def _apply_experiential_memory(
        self,
        memory_state: ExperientialMemoryState,
        oak_state: OaKState,
        current_representation: Array,
        bootstrap_representation: Array,
        decision_representation: Array,
        executed_action: Array,
        reward: Array,
        *,
        current_prototype_decision_id: Array,
        next_prototype_decision_id: Array,
        next_armed: Array,
        transaction_allowed: Array,
        memory_input: PrototypeExperientialMemoryInput,
        input_supplied: Array,
    ) -> tuple[
        ExperientialMemoryState,
        OaKState,
        Array,
        PrototypeExperientialMemoryDiagnostics,
    ]:
        """Query pre-write memory, ground one real exemplar, then dispatch."""

        memory = self._experiential_memory
        policy = self._experiential_memory_policy
        if memory is None or policy is None:
            raise RuntimeError("experiential memory is disabled")
        transaction_gate = jnp.asarray(transaction_allowed, dtype=jnp.bool_)
        current_id_matches = jnp.array_equal(
            memory_input.current_prototype_decision_id,
            current_prototype_decision_id,
        )
        next_id_matches = jnp.array_equal(
            memory_input.next_prototype_decision_id,
            next_prototype_decision_id,
        )
        query_uncertainty_valid = (
            jnp.isfinite(memory_input.query_uncertainty)
            & (memory_input.query_uncertainty >= 0.0)
            & (
                memory_input.query_uncertainty_available
                | (memory_input.query_uncertainty == 0.0)
            )
        )
        entry_uncertainty_valid = (
            jnp.isfinite(memory_input.entry_uncertainty)
            & (memory_input.entry_uncertainty >= 0.0)
            & (
                memory_input.entry_uncertainty_available
                | (memory_input.entry_uncertainty == 0.0)
            )
        )
        safety_cost_valid = (
            jnp.isfinite(memory_input.safety_cost)
            & (memory_input.safety_cost >= 0.0)
            & (
                memory_input.safety_cost_available
                | (memory_input.safety_cost == 0.0)
            )
        )
        utility_valid = (
            jnp.isfinite(memory_input.utility)
            & (memory_input.utility >= 0.0)
            & (
                memory_input.utility_available
                | (memory_input.utility == 0.0)
            )
        )
        metadata_valid = (
            (memory_input.query_representation_version >= 0)
            & (memory_input.entry_representation_version >= 0)
            & query_uncertainty_valid
            & entry_uncertainty_valid
            & safety_cost_valid
            & jnp.isfinite(memory_input.reliability)
            & (memory_input.reliability >= 0.0)
            & (memory_input.reliability <= 1.0)
            & utility_valid
            & (memory_input.provenance_id >= 0)
            & (memory_input.source_id >= 0)
        )
        transaction_required = (
            transaction_gate
            & jnp.asarray(next_armed, dtype=jnp.bool_)
            & memory_input.available
            & current_id_matches
            & next_id_matches
            & metadata_valid
        )
        query_version = jnp.where(
            transaction_required,
            memory_input.query_representation_version,
            jnp.asarray(0, dtype=jnp.int32),
        )
        query_uncertainty = jnp.where(
            transaction_required,
            memory_input.query_uncertainty,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        query_uncertainty_available = (
            transaction_required & memory_input.query_uncertainty_available
        )
        safety_mask = jnp.where(
            transaction_required,
            memory_input.next_action_safety_mask,
            jnp.ones_like(memory_input.next_action_safety_mask),
        )
        entry = ExperientialMemoryEntry(
            observation=current_representation,
            key=current_representation,
            action=jax.nn.one_hot(
                executed_action,
                self._config.oak.n_primitive_actions,
                dtype=jnp.float32,
            ),
            outcome=jnp.concatenate(
                (
                    bootstrap_representation,
                    jnp.asarray(reward, dtype=jnp.float32)[None],
                )
            ),
            reward=jnp.asarray(reward, dtype=jnp.float32),
            uncertainty=memory_input.entry_uncertainty,
            uncertainty_available=memory_input.entry_uncertainty_available,
            safety_cost=memory_input.safety_cost,
            safety_cost_available=memory_input.safety_cost_available,
            reliability=memory_input.reliability,
            utility=memory_input.utility,
            utility_available=memory_input.utility_available,
            representation_version=memory_input.entry_representation_version,
            valid=jnp.asarray(True, dtype=jnp.bool_),
            age=jnp.asarray(0, dtype=jnp.int32),
            provenance_id=memory_input.provenance_id,
            source_id=memory_input.source_id,
        )

        def apply_step(
            _: None,
        ) -> tuple[ExperientialMemoryPolicyProposal, ExperientialMemoryStepResult]:
            return policy.propose_and_step(
                memory_state,
                decision_representation,
                query_version,
                query_uncertainty,
                query_uncertainty_available,
                safety_mask,
                entry,
            )

        def skip_step(
            _: None,
        ) -> tuple[ExperientialMemoryPolicyProposal, ExperientialMemoryStepResult]:
            proposal = policy.propose(
                memory_state,
                decision_representation,
                query_version,
                query_uncertainty,
                query_uncertainty_available,
                safety_mask,
            )
            return (
                proposal,
                ExperientialMemoryStepResult(
                    state=memory_state,
                    retrieval=proposal.retrieval,
                    wrote=jnp.asarray(False, dtype=jnp.bool_),
                    slot=jnp.asarray(-1, dtype=jnp.int32),
                    evicted=jnp.asarray(False, dtype=jnp.bool_),
                    evicted_provenance_id=jnp.asarray(-1, dtype=jnp.int32),
                ),
            )

        proposal, step = cast(
            tuple[ExperientialMemoryPolicyProposal, ExperientialMemoryStepResult],
            jax.lax.cond(
                transaction_required,
                apply_step,
                skip_step,
                operand=None,
            ),
        )
        retrieval_matches = _tree_arrays_equal(
            proposal.retrieval,
            step.retrieval,
        )
        transaction_applied = (
            transaction_required & step.wrote & retrieval_matches
        )
        counterfactual_action = oak_state.stomp_state.last_primitive_action
        proposed_action = jnp.where(
            transaction_required & proposal.available,
            proposal.action,
            counterfactual_action,
        )
        replacement = replace_dispatched_primitive_action(
            oak_state.stomp_state,
            decision_representation,
            proposed_action,
            safety_action_mask=safety_mask,
        )
        transaction_applied = (
            transaction_applied & (~replacement.decision.failed_closed)
        )
        next_oak_state = cast(
            OaKState,
            oak_state.replace(stomp_state=replacement.state),
        )
        effective_action = jnp.where(
            next_armed,
            replacement.decision.effective_action,
            jnp.asarray(-1, dtype=jnp.int32),
        )
        diagnostics = PrototypeExperientialMemoryDiagnostics(
            proposal=proposal,
            dispatch_replacement=replacement.decision,
            input_supplied=input_supplied,
            input_available=memory_input.available,
            current_prototype_decision_id_matches=current_id_matches,
            next_prototype_decision_id_matches=next_id_matches,
            metadata_valid=metadata_valid,
            transaction_required=transaction_required,
            retrieval_matches=retrieval_matches,
            query_before_write=transaction_required & retrieval_matches,
            deterministic_prestate_query_count=jnp.where(
                transaction_required,
                jnp.asarray(1, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            ),
            wrote=step.wrote,
            slot=step.slot,
            evicted=step.evicted,
            evicted_provenance_id=step.evicted_provenance_id,
            transaction_applied=transaction_applied,
            counterfactual_base_action=counterfactual_action,
            effective_action=effective_action,
        )
        return step.state, next_oak_state, effective_action, diagnostics

    def _apply_partner_policy_fusion(
        self,
        interaction_slot: Any,
        ia_component_state: Any,
        oak_state: OaKState,
        decision_observation: Array,
        *,
        derived_decision_id: Array,
        derived_event_id: Array,
        derived_prototype_decision_id: Array,
        next_armed: Array,
        transaction_allowed: Array,
        experiential_memory_state: ExperientialMemoryState | None,
        upstream_safety_action_mask: Array | None,
        decision_input: PrototypePartnerPolicyFusionInput,
        decision_input_supplied: Array,
        feedback: PrototypePartnerPolicyFusionFeedback,
        feedback_input_supplied: Array,
    ) -> tuple[Any, OaKState, Array, PrototypePartnerPolicyFusionDiagnostics]:
        """Resolve prior feedback, then fuse and bind one next dispatch.

        This method is called only inside the Prototype transition transaction.
        ``transaction_allowed=False`` gates both external surfaces to explicit
        no-ops while still producing the same fixed diagnostics PyTree required
        by ``lax.cond``.
        """

        fusion = self._partner_policy_fusion
        if fusion is None:
            raise RuntimeError("partner policy fusion is disabled")
        interaction = self._partner_interaction_state(interaction_slot)
        partner_state = self._partner_fusion_component_state(interaction_slot)
        transaction_gate = jnp.asarray(transaction_allowed, dtype=jnp.bool_)
        feedback_prototype_id_matches = (
            interaction.feedback_prototype_decision_id_available
            & jnp.array_equal(
                feedback.prototype_decision_id,
                interaction.feedback_prototype_decision_id,
            )
        )
        gated_feedback = cast(
            PartnerPolicyFusionFeedback,
            feedback.feedback.replace(
                available=(
                    feedback.feedback.available
                    & transaction_gate
                    & feedback_prototype_id_matches
                )
            ),
        )
        feedback_result = fusion.apply_feedback(partner_state, gated_feedback)
        pending_owner_available = (
            interaction.feedback_prototype_decision_id_available
            & (~feedback_result.applied)
        )
        pending_owner_id = jnp.where(
            pending_owner_available,
            interaction.feedback_prototype_decision_id,
            jnp.zeros((4,), dtype=jnp.uint32),
        )

        decision_prototype_id_matches = jnp.array_equal(
            decision_input.prototype_decision_id,
            derived_prototype_decision_id,
        )
        decision_gate = (
            transaction_gate
            & jnp.asarray(next_armed, dtype=jnp.bool_)
            & decision_input.available
            & decision_prototype_id_matches
        )
        safe_keyboard_vector = jnp.where(
            decision_gate & decision_input.keyboard_available,
            decision_input.keyboard_vector,
            jnp.zeros_like(decision_input.keyboard_vector),
        )
        keyboard = self._oak.propose_keyboard_policy(
            oak_state,
            decision_observation,
            safe_keyboard_vector,
        )
        option_proposal = OptionKeyboardProposal(
            available=(
                decision_gate
                & decision_input.keyboard_available
                & keyboard.available
            ),
            action=keyboard.action,
            declared_score=keyboard.declared_score,
        )
        # A missing/disabled sidecar deliberately invalidates only contextual
        # fusion numerics. The mechanism then returns the independently safe
        # OaK base action without advancing its decision or reliability state.
        fusion_context = jnp.where(
            decision_gate,
            decision_input.context_features,
            jnp.full_like(decision_input.context_features, jnp.nan),
        )
        combined_safety_mask = decision_input.safety_action_mask
        if upstream_safety_action_mask is not None:
            combined_safety_mask = (
                combined_safety_mask & upstream_safety_action_mask
            )
        fusion_safety_mask = jnp.where(
            decision_gate,
            combined_safety_mask,
            jnp.ones_like(decision_input.safety_action_mask),
        )
        base_action = oak_state.stomp_state.last_primitive_action
        base_score = self._oak_counterfactual_dispatch_score(
            oak_state,
            decision_observation,
        )
        decision_result = fusion.decide(
            feedback_result.state,
            decision_id=jnp.asarray(derived_decision_id, dtype=jnp.int32),
            event_id=jnp.asarray(derived_event_id, dtype=jnp.int32),
            observation_id=jnp.where(
                decision_gate,
                decision_input.observation_id,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            context_id=jnp.where(
                decision_gate,
                decision_input.context_id,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            context_features=fusion_context,
            base_action=base_action,
            base_declared_score=base_score,
            safety_action_mask=fusion_safety_mask,
            option_proposal=option_proposal,
            messages=decision_input.messages,
        )
        replacement = replace_dispatched_primitive_action(
            oak_state.stomp_state,
            decision_observation,
            decision_result.decision.effective_action,
            safety_action_mask=fusion_safety_mask,
        )
        next_oak_state = cast(
            OaKState,
            oak_state.replace(stomp_state=replacement.state),
        )
        next_owner_available = (
            pending_owner_available | decision_result.decision.feedback_armed
        )
        next_owner_id = jnp.where(
            decision_result.decision.feedback_armed,
            derived_prototype_decision_id,
            pending_owner_id,
        )
        next_interaction_state = self._interaction_slot(
            ia_component_state,
            decision_result.state,
            experiential_memory_state=experiential_memory_state,
            feedback_prototype_decision_id=next_owner_id,
            feedback_prototype_decision_id_available=next_owner_available,
        )
        effective_action = jnp.where(
            next_armed,
            replacement.decision.effective_action,
            jnp.asarray(-1, dtype=jnp.int32),
        )
        diagnostics = PrototypePartnerPolicyFusionDiagnostics(
            feedback=feedback_result,
            decision=decision_result.decision,
            keyboard_proposal=keyboard,
            dispatch_replacement=replacement.decision,
            feedback_input_supplied=feedback_input_supplied,
            decision_input_supplied=decision_input_supplied,
            feedback_prototype_decision_id_matches=(
                feedback_prototype_id_matches
            ),
            decision_prototype_decision_id_matches=(
                decision_prototype_id_matches
            ),
            transaction_applied=transaction_gate,
            counterfactual_base_action=base_action,
            effective_action=effective_action,
        )
        return (
            next_interaction_state,
            next_oak_state,
            effective_action,
            diagnostics,
        )

    # -- Dreaming scan (JIT-compiled as a method so the closure is stable) ----

    @functools.partial(jax.jit, static_argnums=(0,))
    def _run_dreams(
        self,
        oak_state: OaKState,
        wm_state: Any,
        buf_state: Any,
        rng_key: Array,
    ) -> tuple[OaKState, Float[Array, " n_dreams"]]:
        n_prim = self._config.oak.n_primitive_actions
        dream_root = _guarded_dream_rng_root(rng_key)
        sample_one_hot = (
            self._config.dream_next_observation_mode == "sample_one_hot"
        )
        dream_observation_root = (
            jr.fold_in(dream_root, _DREAM_NEXT_OBSERVATION_STREAM_TAG)
            if sample_one_hot
            else dream_root
        )

        def _dream_step(
            carry: tuple[OaKState, Array], dream_index: Array
        ) -> tuple[tuple[OaKState, Array], Float[Array, ""]]:
            oak_s, k = carry
            # Both observation ablations use identical anchor/action key streams.
            k, sample_key, action_key = jr.split(k, 3)
            anchor_obs, _ = self._buffer.sample(buf_state, sample_key)
            action = jr.randint(action_key, (), 0, n_prim, dtype=jnp.int32)
            proposal = self._dreamer.propose(self._world_model, wm_state, anchor_obs, action)
            dream_next_observation = proposal.transition.next_observation
            dream_accepted = proposal.accepted
            if sample_one_hot:
                observation_key = jr.fold_in(
                    dream_observation_root,
                    dream_index,
                )
                dream_next_observation, projection_valid = (
                    _sample_one_hot_dream_observation(
                        dream_next_observation,
                        observation_key,
                    )
                )
                dream_accepted = dream_accepted & projection_valid

            # A dream is a primitive base-Q backup from a sampled real anchor.
            # Force the synthetic state idle so it cannot advance/terminate a
            # real option, then carry back *only* the base learner state. The
            # real option trajectory, reward-rate estimate, action/RNG,
            # utility/curation statistics, and real-step counters are all
            # invariants of imagined computation.
            temp_stomp = oak_s.stomp_state.replace(
                base_last_obs=anchor_obs,
                base_last_action=action,
                last_primitive_action=action,
                executing_option=jnp.array(-1, dtype=jnp.int32),
            )
            temp_oak = cast(OaKState, oak_s.replace(stomp_state=temp_stomp))
            dream_result = self._oak.update(
                temp_oak,
                proposal.transition.reward,
                dream_next_observation,
                proposal.transition.discount,
                enable_option_planning=False,
            )

            candidate_learner = dream_result.state.stomp_state.base_learner_state
            accepted_learner = jax.tree_util.tree_map(
                lambda candidate, real: jnp.where(
                    dream_accepted, candidate, real
                ),
                candidate_learner,
                oak_s.stomp_state.base_learner_state,
            )
            new_oak_s = cast(
                OaKState,
                oak_s.replace(
                    stomp_state=oak_s.stomp_state.replace(
                        base_learner_state=accepted_learner
                    )
                ),
            )
            td_err = jnp.where(
                dream_accepted,
                dream_result.td_error,
                jnp.array(0.0, dtype=jnp.float32),
            )
            return (new_oak_s, k), td_err

        (new_oak_state, _), dream_td_errors = jax.lax.scan(
            _dream_step,
            (oak_state, dream_root),
            jnp.arange(self._config.n_dreams_per_step),
        )
        return new_oak_state, dream_td_errors

    # -- Core update ----------------------------------------------------------

    def update(
        self,
        state: PrototypeAgentState,
        reward: Array,
        next_observation: Array,
        horde_cumulants: Array | None = None,
    ) -> PrototypeUpdateResult:
        """Compatibility wrapper for the historical transition API.

        New integrations should call :meth:`update_transition` and supply an
        explicit discount.  This wrapper preserves the former behavior:
        primitive control bootstraps with one, option returns use
        ``STOMPConfig.option_gamma``, and the world model (when enabled)
        receives its configured gamma as the target discount. Because this
        surface has no terminal flag, a configured zero model discount requires
        the explicit :meth:`update_transition` API.
        """
        if self._state_builder is not None:
            raise ValueError(
                "legacy update is unavailable with state_builder; use "
                "update_transition with an explicit preceding discount"
            )
        legacy_model_discount = (
            self._config.world_model.gamma
            if self._config.world_model is not None
            else (
                self._config.world_model_ensemble.model.gamma
                if self._config.world_model_ensemble is not None
                else (
                    self._config.model_replay_rehearsal.ensemble.model.gamma
                    if self._config.model_replay_rehearsal is not None
                    else 1.0
                )
            )
        )
        if legacy_model_discount == 0.0:
            raise ValueError(
                "legacy update is unavailable when the configured world-model gamma "
                "is zero; use update_transition with an explicit terminal transition"
            )
        current_oak_state = self._oak_component_state(state.oak_state)
        feature_consumer_binding = (
            self._feature_consumer_binding(state.oak_state)
            if self._prototype_feature_lifecycle is not None
            else None
        )
        legacy_representation = current_oak_state.stomp_state.base_last_obs
        legacy_stomp_state = current_oak_state.stomp_state
        # The historical wrapper predates the explicit intra-option decision
        # owner cache. Reconstruct that one missing owner from the primitive
        # command STOMP actually dispatched; the authoritative transition API
        # never performs this repair and remains fail-closed on mismatches.
        legacy_stomp_state = legacy_stomp_state.replace(
            option_last_intra_action=jnp.where(
                legacy_stomp_state.executing_option >= 0,
                legacy_stomp_state.last_primitive_action,
                legacy_stomp_state.option_last_intra_action,
            )
        )
        legacy_oak_state = current_oak_state.replace(
            stomp_state=legacy_stomp_state,
        )
        legacy_state = cast(
            PrototypeAgentState,
            state.replace(
                oak_state=self._oak_state_slot(
                    legacy_oak_state,
                    feature_consumer_binding,
                ),
                current_raw_observation=legacy_representation[
                    : self._raw_observation_dim()
                ],
                current_representation=legacy_representation,
                current_action=current_oak_state.stomp_state.last_primitive_action,
            ),
        )
        transition, diagnostics = self._normalize_transition(
            legacy_state,
            PrototypeTransition(
                observation=legacy_state.current_raw_observation,
                action=legacy_state.current_action,
                decision_id=legacy_state.current_decision_id,
                reward=reward,
                discount=jnp.asarray(legacy_model_discount, dtype=jnp.float32),
                terminated=jnp.array(False),
                truncated=jnp.array(False),
                next_observation=next_observation,
                next_decision_observation=next_observation,
                horde_cumulants=horde_cumulants,
            )
        )
        partner_input, partner_input_supplied = (
            self._normalize_partner_policy_fusion_input(None)
        )
        partner_feedback, partner_feedback_supplied = (
            self._normalize_partner_policy_fusion_feedback(None)
        )
        memory_input, memory_input_supplied = (
            self._normalize_experiential_memory_input(None)
        )
        return self._update_transition_impl(
            legacy_state,
            transition,
            diagnostics,
            control_discount=None,
            gradient_joy_evidence=None,
            gradient_joy_evidence_supplied=jnp.asarray(False),
            gradient_joy_decision_id_matches=jnp.asarray(False),
            experiential_memory_input=memory_input,
            experiential_memory_input_supplied=memory_input_supplied,
            partner_policy_fusion_input=partner_input,
            partner_policy_fusion_input_supplied=partner_input_supplied,
            partner_policy_fusion_feedback=partner_feedback,
            partner_policy_fusion_feedback_supplied=partner_feedback_supplied,
        )

    def update_transition(
        self,
        state: PrototypeAgentState,
        transition: PrototypeTransition,
        gradient_joy_evidence: PrototypeGradientJoyEvidence | None = None,
        *,
        experiential_memory_input: (
            PrototypeExperientialMemoryInput | None
        ) = None,
        partner_policy_fusion_input: (
            PrototypePartnerPolicyFusionInput | None
        ) = None,
        partner_policy_fusion_feedback: (
            PrototypePartnerPolicyFusionFeedback | None
        ) = None,
    ) -> PrototypeUpdateResult:
        """Process one explicit real-time continuing transition.

        Execution order:

        1. Update the legacy world model or bounded ensemble from the real
           transition (if configured).
        2. Add ``next_observation`` to the dream anchor buffer.
        3. Update OaK from the real transition.
        4. Run ``n_dreams_per_step`` guarded Dyna updates (if configured).
        5. Update the Horde from the real transition (if configured).
        6. Update the IA companion from the real transition (if configured).

        Args:
            state: Current agent state.  Must have been primed with
                :meth:`start`.
            transition: Reward, raw next observation, continuation discount,
                and optional Horde signals. The world model and control
                learner consume exactly this scalar discount.

        Returns:
            :class:`PrototypeUpdateResult` with updated state, selected action,
            and per-component diagnostics.
        """
        normalized_joy_evidence, joy_supplied, joy_decision_matches = (
            self._normalize_gradient_joy_evidence(
                state,
                gradient_joy_evidence,
            )
        )
        normalized_memory_input, memory_input_supplied = (
            self._normalize_experiential_memory_input(
                experiential_memory_input
            )
        )
        normalized_partner_input, partner_input_supplied = (
            self._normalize_partner_policy_fusion_input(
                partner_policy_fusion_input
            )
        )
        normalized_partner_feedback, partner_feedback_supplied = (
            self._normalize_partner_policy_fusion_feedback(
                partner_policy_fusion_feedback
            )
        )
        normalized, diagnostics = self._normalize_transition(state, transition)
        return self._update_transition_impl(
            state,
            normalized,
            diagnostics,
            control_discount=normalized.discount,
            gradient_joy_evidence=normalized_joy_evidence,
            gradient_joy_evidence_supplied=joy_supplied,
            gradient_joy_decision_id_matches=joy_decision_matches,
            experiential_memory_input=normalized_memory_input,
            experiential_memory_input_supplied=memory_input_supplied,
            partner_policy_fusion_input=normalized_partner_input,
            partner_policy_fusion_input_supplied=partner_input_supplied,
            partner_policy_fusion_feedback=normalized_partner_feedback,
            partner_policy_fusion_feedback_supplied=(
                partner_feedback_supplied
            ),
        )

    def _normalize_transition(
        self,
        state: PrototypeAgentState,
        transition: PrototypeTransition,
    ) -> tuple[PrototypeTransition, PrototypeTransitionDiagnostics]:
        """Validate ownership/semantics and produce safe branch operands."""
        raw_observation_dim = self._raw_observation_dim()
        observation = _strict_float_array(
            transition.observation,
            (raw_observation_dim,),
            name="transition.observation",
        )
        action = _strict_action_scalar(
            transition.action,
            name="transition.action",
        )
        decision_id = _strict_decision_id(
            transition.decision_id,
            name="transition.decision_id",
        )
        reward = _strict_float_array(
            transition.reward,
            (),
            name="transition.reward",
        )
        next_observation = _strict_float_array(
            transition.next_observation,
            (raw_observation_dim,),
            name="transition.next_observation",
        )
        next_decision_observation = _strict_float_array(
            transition.next_decision_observation,
            (raw_observation_dim,),
            name="transition.next_decision_observation",
        )
        discount = _strict_float_array(
            transition.discount,
            (),
            name="transition.discount",
        )
        terminated = _strict_bool_scalar(
            transition.terminated,
            name="transition.terminated",
        )
        truncated = _strict_bool_scalar(
            transition.truncated,
            name="transition.truncated",
        )

        inputs_finite = (
            jnp.all(jnp.isfinite(observation))
            & jnp.isfinite(reward)
            & jnp.isfinite(discount)
            & jnp.all(jnp.isfinite(next_observation))
            & jnp.all(jnp.isfinite(next_decision_observation))
        )
        action_in_range = (action >= 0) & (
            action < self._config.oak.n_primitive_actions
        )
        observation_matches = jnp.array_equal(
            observation,
            state.current_raw_observation,
        )
        action_matches = action == state.current_action
        decision_id_matches = jnp.array_equal(
            decision_id,
            state.current_decision_id,
        )
        next_generation_available = _decision_generation_available(
            state.current_decision_id,
        )
        discount_valid = jnp.isfinite(discount) & (discount >= 0.0) & (discount <= 1.0)
        boundary = terminated | truncated
        next_counter_capacity_available = _next_counter_capacity_available(
            state,
            execution_boundary=boundary,
        )
        observations_match_off_boundary = boundary | jnp.array_equal(
            next_observation,
            next_decision_observation,
        )
        boundary_semantics_valid = (
            ((discount == 0.0) == terminated)
            & ((~(truncated & ~terminated)) | (discount > 0.0))
            & observations_match_off_boundary
        )
        started = state.started
        state_consistent = self._state_cache_consistent(state)

        horde_cumulants: Any = None
        horde_discounts: Any = None
        horde_valid = jnp.array(True)
        if self._horde is None:
            if (
                transition.horde_cumulants is not None
                or transition.horde_discounts is not None
            ):
                raise ValueError(
                    "Horde transition fields require horde_spec to be configured"
                )
        else:
            n_demons = self._horde.n_demons
            if transition.horde_cumulants is not None:
                horde_cumulants = _strict_float_array(
                    transition.horde_cumulants,
                    (n_demons,),
                    name="transition.horde_cumulants",
                )
                horde_valid = horde_valid & jnp.all(~jnp.isinf(horde_cumulants))
            if transition.horde_discounts is not None:
                horde_discounts = _strict_float_array(
                    transition.horde_discounts,
                    (n_demons,),
                    name="transition.horde_discounts",
                )
                horde_valid = horde_valid & jnp.all(
                    jnp.isfinite(horde_discounts)
                    & (horde_discounts >= 0.0)
                    & (horde_discounts <= 1.0)
                )

        valid = (
            started
            & inputs_finite
            & action_in_range
            & observation_matches
            & action_matches
            & decision_id_matches
            & discount_valid
            & boundary_semantics_valid
            & state_consistent
            & horde_valid
        )
        diagnostics = PrototypeTransitionDiagnostics(
            started=started,
            inputs_finite=inputs_finite & horde_valid,
            action_in_range=action_in_range,
            observation_matches=observation_matches,
            action_matches=action_matches,
            decision_id_matches=decision_id_matches,
            next_generation_available=next_generation_available,
            next_counter_capacity_available=next_counter_capacity_available,
            discount_valid=discount_valid,
            boundary_semantics_valid=boundary_semantics_valid,
            state_consistent=state_consistent,
            post_update_checked=jnp.asarray(False),
            post_update_finite=jnp.asarray(False),
            post_update_consistent=jnp.asarray(False),
            valid=valid,
            rejected=~valid,
        )

        safe_observation = jnp.where(inputs_finite, observation, state.current_raw_observation)
        safe_next_observation = jnp.where(
            inputs_finite,
            next_observation,
            state.current_raw_observation,
        )
        safe_next_decision_observation = jnp.where(
            inputs_finite,
            next_decision_observation,
            state.current_raw_observation,
        )
        safe_action = jnp.where(action_in_range, action, state.current_action)
        safe_reward = jnp.where(jnp.isfinite(reward), reward, 0.0)
        safe_discount = jnp.where(discount_valid, discount, 0.0)
        if horde_cumulants is not None:
            horde_cumulants = jnp.where(
                ~jnp.isinf(horde_cumulants),
                horde_cumulants,
                jnp.full_like(horde_cumulants, jnp.nan),
            )
        if horde_discounts is not None:
            horde_discounts = jnp.where(
                jnp.isfinite(horde_discounts)
                & (horde_discounts >= 0.0)
                & (horde_discounts <= 1.0),
                horde_discounts,
                jnp.zeros_like(horde_discounts),
            )

        normalized = PrototypeTransition(
            observation=safe_observation,
            action=safe_action,
            decision_id=state.current_decision_id,
            reward=safe_reward,
            discount=safe_discount,
            terminated=terminated,
            truncated=truncated,
            next_observation=safe_next_observation,
            next_decision_observation=safe_next_decision_observation,
            horde_cumulants=horde_cumulants,
            horde_discounts=horde_discounts,
        )
        return normalized, diagnostics

    def _rejected_update_result(
        self,
        state: PrototypeAgentState,
        diagnostics: PrototypeTransitionDiagnostics,
        *,
        gradient_joy_evidence: PrototypeGradientJoyEvidence | None,
        gradient_joy_evidence_supplied: Array,
        gradient_joy_decision_id_matches: Array,
        experiential_memory_input: PrototypeExperientialMemoryInput | None,
        experiential_memory_input_supplied: Array,
        partner_policy_fusion_input: PrototypePartnerPolicyFusionInput | None,
        partner_policy_fusion_input_supplied: Array,
        partner_policy_fusion_feedback: (
            PrototypePartnerPolicyFusionFeedback | None
        ),
        partner_policy_fusion_feedback_supplied: Array,
    ) -> PrototypeUpdateResult:
        """Return neutral diagnostics and preserve ``state`` bit-for-bit."""
        zero = jnp.array(0.0, dtype=jnp.float32)
        world_model_error: Any = (
            zero
            if (
                self._world_model is not None
                or self._world_model_ensemble is not None
                or self._model_replay_rehearsal is not None
                or self._recurrent_latent_world_model_ensemble is not None
            )
            else None
        )
        dream_td_errors: Any = None
        if self._world_model is not None and self._config.n_dreams_per_step > 0:
            dream_td_errors = jnp.zeros(
                (self._config.n_dreams_per_step,),
                dtype=jnp.float32,
            )
        option_search_diagnostics: OptionSearchControlDiagnostics | None = None
        if self._option_search_control is not None:
            option_search_diagnostics = (
                self._option_search_control.unavailable_diagnostics(
                    state.current_representation
                )
            )
        feature_lifecycle_diagnostics: (
            PrototypeFeatureLifecycleIntegrationDiagnostics | None
        ) = None
        if self._prototype_feature_lifecycle is not None:
            feature_lifecycle_diagnostics = (
                self._unavailable_feature_lifecycle_diagnostics(
                    self._feature_lifecycle_component_state(
                        state.state_builder_state
                    )
                )
            )
        horde_td_errors: Any = None
        if self._horde is not None:
            horde_td_errors = jnp.zeros(
                (self._horde.n_demons,),
                dtype=jnp.float32,
            )
        ia_augmented_obs: Any = None
        ia_recommendation: Any = None
        if self._ia is not None and self._config.ia is not None:
            ia_augmented_obs = jnp.zeros(
                (self._config.ia.augmented_obs_dim,),
                dtype=jnp.float32,
            )
            ia_recommendation = state.current_action
        memory_diagnostics: PrototypeExperientialMemoryDiagnostics | None = None
        current_memory_state: ExperientialMemoryState | None = None
        if self._experiential_memory is not None:
            if experiential_memory_input is None:
                raise RuntimeError(
                    "configured experiential memory requires a fixed sidecar"
                )
            current_memory_state = self._experiential_memory_component_state(
                state.ia_state
            )
            (
                _,
                _,
                _,
                memory_diagnostics,
            ) = self._apply_experiential_memory(
                current_memory_state,
                self._oak_component_state(state.oak_state),
                state.current_representation,
                state.current_representation,
                state.current_representation,
                state.current_action,
                zero,
                current_prototype_decision_id=state.current_decision_id,
                next_prototype_decision_id=_increment_decision_id(
                    state.current_decision_id
                ),
                next_armed=jnp.asarray(False, dtype=jnp.bool_),
                transaction_allowed=jnp.asarray(False, dtype=jnp.bool_),
                memory_input=experiential_memory_input,
                input_supplied=experiential_memory_input_supplied,
            )
        partner_fusion_diagnostics: (
            PrototypePartnerPolicyFusionDiagnostics | None
        ) = None
        if self._partner_policy_fusion is not None:
            if (
                partner_policy_fusion_input is None
                or partner_policy_fusion_feedback is None
            ):
                raise RuntimeError("configured partner fusion requires fixed sidecars")
            (
                _,
                _,
                _,
                partner_fusion_diagnostics,
            ) = self._apply_partner_policy_fusion(
                state.ia_state,
                self._ia_component_state(state.ia_state),
                self._oak_component_state(state.oak_state),
                state.current_representation,
                derived_decision_id=_saturating_int32_increment(
                    state.step_count
                ),
                derived_event_id=_saturating_int32_increment(
                    state.observation_event_count
                ),
                derived_prototype_decision_id=_increment_decision_id(
                    state.current_decision_id
                ),
                next_armed=jnp.asarray(False, dtype=jnp.bool_),
                transaction_allowed=jnp.asarray(False, dtype=jnp.bool_),
                experiential_memory_state=current_memory_state,
                upstream_safety_action_mask=None,
                decision_input=partner_policy_fusion_input,
                decision_input_supplied=partner_policy_fusion_input_supplied,
                feedback=partner_policy_fusion_feedback,
                feedback_input_supplied=(
                    partner_policy_fusion_feedback_supplied
                ),
            )
        (
            _,
            builder_learning_diagnostics,
            gradient_joy_application,
        ) = self._apply_state_builder_learning(
            self._builder_component_state(state.state_builder_state),
            self._builder_component_state(state.state_builder_state),
            jnp.zeros(
                (self._base_representation_dim(),),
                dtype=jnp.float32,
            ),
            jnp.asarray(False),
            _unavailable_learning_signals(),
            gradient_joy_evidence,
        )
        return PrototypeUpdateResult(
            state=state,
            action=state.current_action,
            oak_td_error=zero,
            oak_average_reward=zero,
            world_model_error=world_model_error,
            learning_signals=_unavailable_learning_signals(),
            world_model_representation_gradient=jnp.zeros(
                (self._config.oak.observation_dim,),
                dtype=jnp.float32,
            ),
            world_model_representation_gradient_valid=jnp.asarray(False),
            behavior_gradient_result=_unavailable_behavior_gradient(
                self._config.oak.observation_dim
            ),
            representation_gradient_mix=_unavailable_representation_gradient_mix(
                self._config.oak.observation_dim
            ),
            world_model_ensemble_diagnostics=_unavailable_ensemble_diagnostics(),
            recurrent_latent_world_model_diagnostics=(
                _unavailable_recurrent_latent_diagnostics(

                        self._config.recurrent_latent_world_model_ensemble.ensemble_size
                        if self._config.recurrent_latent_world_model_ensemble
                        is not None
                        else 0

                )
            ),
            model_replay_transaction_applied=jnp.asarray(False),
            model_replay_recorded=jnp.asarray(False),
            model_replay_sampled=jnp.asarray(False),
            model_replay_updates_applied=jnp.asarray(0, dtype=jnp.int32),
            model_replay_padding_count=jnp.asarray(0, dtype=jnp.int32),
            state_builder_learning_diagnostics=builder_learning_diagnostics,
            gradient_joy_application=gradient_joy_application,
            gradient_joy_evidence_supplied=gradient_joy_evidence_supplied,
            gradient_joy_decision_id_matches=(
                gradient_joy_decision_id_matches
            ),
            option_search_control_diagnostics=option_search_diagnostics,
            prototype_feature_lifecycle_diagnostics=(
                feature_lifecycle_diagnostics
            ),
            dream_td_errors=dream_td_errors,
            horde_td_errors=horde_td_errors,
            ia_augmented_obs=ia_augmented_obs,
            ia_recommendation=ia_recommendation,
            experiential_memory_diagnostics=memory_diagnostics,
            partner_policy_fusion_diagnostics=partner_fusion_diagnostics,
            transition_diagnostics=diagnostics,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_transition_impl(
        self,
        state: PrototypeAgentState,
        transition: PrototypeTransition,
        diagnostics: PrototypeTransitionDiagnostics,
        *,
        control_discount: Array | None,
        gradient_joy_evidence: PrototypeGradientJoyEvidence | None,
        gradient_joy_evidence_supplied: Array,
        gradient_joy_decision_id_matches: Array,
        experiential_memory_input: PrototypeExperientialMemoryInput | None,
        experiential_memory_input_supplied: Array,
        partner_policy_fusion_input: PrototypePartnerPolicyFusionInput | None,
        partner_policy_fusion_input_supplied: Array,
        partner_policy_fusion_feedback: (
            PrototypePartnerPolicyFusionFeedback | None
        ),
        partner_policy_fusion_feedback_supplied: Array,
    ) -> PrototypeUpdateResult:
        """Atomically apply one normalized transition at a reusable JAX boundary."""

        def valid_branch(_: None) -> PrototypeUpdateResult:
            result = self._apply_valid_transition_impl(
                state,
                transition,
                diagnostics,
                control_discount=control_discount,
                gradient_joy_evidence=gradient_joy_evidence,
                gradient_joy_evidence_supplied=(
                    gradient_joy_evidence_supplied
                ),
                gradient_joy_decision_id_matches=(
                    gradient_joy_decision_id_matches
                ),
                experiential_memory_input=experiential_memory_input,
                experiential_memory_input_supplied=(
                    experiential_memory_input_supplied
                ),
                partner_policy_fusion_input=partner_policy_fusion_input,
                partner_policy_fusion_input_supplied=(
                    partner_policy_fusion_input_supplied
                ),
                partner_policy_fusion_feedback=partner_policy_fusion_feedback,
                partner_policy_fusion_feedback_supplied=(
                    partner_policy_fusion_feedback_supplied
                ),
            )
            post_finite = self._state_numeric_valid(result.state)
            post_consistent = self._checkpoint_state_structurally_valid(
                result.state
            )
            recurrent_transaction_valid = jnp.asarray(True, dtype=jnp.bool_)
            if self._recurrent_latent_world_model_ensemble is not None:
                recurrent_transaction_valid = (
                    result.recurrent_latent_world_model_diagnostics.transaction_applied
                )
            memory_transaction_valid = jnp.asarray(True, dtype=jnp.bool_)
            if self._experiential_memory is not None:
                memory_diagnostics = result.experiential_memory_diagnostics
                if memory_diagnostics is None:
                    raise RuntimeError(
                        "configured experiential memory requires diagnostics"
                    )
                memory_transaction_valid = (
                    (~memory_diagnostics.transaction_required)
                    | memory_diagnostics.transaction_applied
                )
            # Recurrent state cannot skip an accepted environment event. If
            # the model/signal transaction rejects, roll back the *entire*
            # Prototype transition so the armed cache and every learner/RNG
            # remain synchronized with the still-unconsumed environment event.
            post_valid = (
                post_finite
                & post_consistent
                & result.transition_diagnostics.valid
                & recurrent_transaction_valid
                & memory_transaction_valid
            )
            final_diagnostics = cast(
                PrototypeTransitionDiagnostics,
                diagnostics.replace(
                    post_update_checked=jnp.asarray(True),
                    post_update_finite=post_finite,
                    post_update_consistent=post_consistent,
                    valid=diagnostics.valid & post_valid,
                    rejected=~(diagnostics.valid & post_valid),
                ),
            )
            accepted_feature_diagnostics = (
                result.prototype_feature_lifecycle_diagnostics
            )
            if accepted_feature_diagnostics is not None:
                accepted_feature_diagnostics = cast(
                    PrototypeFeatureLifecycleIntegrationDiagnostics,
                    accepted_feature_diagnostics.replace(
                        outer_transaction_committed=jnp.asarray(
                            True,
                            dtype=jnp.bool_,
                        )
                    ),
                )
            accepted = cast(
                PrototypeUpdateResult,
                result.replace(
                    transition_diagnostics=final_diagnostics,
                    prototype_feature_lifecycle_diagnostics=(
                        accepted_feature_diagnostics
                    ),
                ),
            )

            def rollback_rejected_recurrent_or_postcheck(
                __: None,
            ) -> PrototypeUpdateResult:
                rejected = self._rejected_update_result(
                    state,
                    final_diagnostics,
                    gradient_joy_evidence=gradient_joy_evidence,
                    gradient_joy_evidence_supplied=(
                        gradient_joy_evidence_supplied
                    ),
                    gradient_joy_decision_id_matches=(
                        gradient_joy_decision_id_matches
                    ),
                    experiential_memory_input=experiential_memory_input,
                    experiential_memory_input_supplied=(
                        experiential_memory_input_supplied
                    ),
                    partner_policy_fusion_input=partner_policy_fusion_input,
                    partner_policy_fusion_input_supplied=(
                        partner_policy_fusion_input_supplied
                    ),
                    partner_policy_fusion_feedback=(
                        partner_policy_fusion_feedback
                    ),
                    partner_policy_fusion_feedback_supplied=(
                        partner_policy_fusion_feedback_supplied
                    ),
                )
                if self._recurrent_latent_world_model_ensemble is not None:
                    rejected = cast(
                        PrototypeUpdateResult,
                        rejected.replace(
                            recurrent_latent_world_model_diagnostics=(
                                result.recurrent_latent_world_model_diagnostics
                            )
                        ),
                    )
                if self._partner_policy_fusion is not None:
                    attempted_partner = (
                        result.partner_policy_fusion_diagnostics
                    )
                    if attempted_partner is None:
                        raise RuntimeError(
                            "configured partner fusion requires diagnostics"
                        )
                    rejected = cast(
                        PrototypeUpdateResult,
                        rejected.replace(
                            partner_policy_fusion_diagnostics=(
                                attempted_partner.replace(
                                    transaction_applied=jnp.asarray(
                                        False,
                                        dtype=jnp.bool_,
                                    )
                                )
                            )
                        ),
                    )
                if self._experiential_memory is not None:
                    attempted_memory = result.experiential_memory_diagnostics
                    if attempted_memory is None:
                        raise RuntimeError(
                            "configured experiential memory requires diagnostics"
                        )
                    rejected = cast(
                        PrototypeUpdateResult,
                        rejected.replace(
                            experiential_memory_diagnostics=(
                                attempted_memory.replace(
                                    transaction_applied=jnp.asarray(
                                        False,
                                        dtype=jnp.bool_,
                                    )
                                )
                            )
                        ),
                    )
                if self._prototype_feature_lifecycle is not None:
                    attempted_feature = (
                        result.prototype_feature_lifecycle_diagnostics
                    )
                    if attempted_feature is None:
                        raise RuntimeError(
                            "configured prototype feature lifecycle requires "
                            "diagnostics"
                        )
                    rejected = cast(
                        PrototypeUpdateResult,
                        rejected.replace(
                            prototype_feature_lifecycle_diagnostics=(
                                attempted_feature.replace(
                                    outer_transaction_committed=jnp.asarray(
                                        False,
                                        dtype=jnp.bool_,
                                    )
                                )
                            )
                        ),
                    )
                return rejected

            return jax.lax.cond(
                post_valid,
                lambda __: accepted,
                rollback_rejected_recurrent_or_postcheck,
                operand=None,
            )

        def rejected_branch(_: None) -> PrototypeUpdateResult:
            return self._rejected_update_result(
                state,
                diagnostics,
                gradient_joy_evidence=gradient_joy_evidence,
                gradient_joy_evidence_supplied=(
                    gradient_joy_evidence_supplied
                ),
                gradient_joy_decision_id_matches=(
                    gradient_joy_decision_id_matches
                ),
                experiential_memory_input=experiential_memory_input,
                experiential_memory_input_supplied=(
                    experiential_memory_input_supplied
                ),
                partner_policy_fusion_input=partner_policy_fusion_input,
                partner_policy_fusion_input_supplied=(
                    partner_policy_fusion_input_supplied
                ),
                partner_policy_fusion_feedback=partner_policy_fusion_feedback,
                partner_policy_fusion_feedback_supplied=(
                    partner_policy_fusion_feedback_supplied
                ),
            )

        return jax.lax.cond(
            diagnostics.valid,
            valid_branch,
            rejected_branch,
            operand=None,
        )

    def _apply_valid_transition_impl(
        self,
        state: PrototypeAgentState,
        transition: PrototypeTransition,
        diagnostics: PrototypeTransitionDiagnostics,
        *,
        control_discount: Array | None,
        gradient_joy_evidence: PrototypeGradientJoyEvidence | None,
        gradient_joy_evidence_supplied: Array,
        gradient_joy_decision_id_matches: Array,
        experiential_memory_input: PrototypeExperientialMemoryInput | None,
        experiential_memory_input_supplied: Array,
        partner_policy_fusion_input: PrototypePartnerPolicyFusionInput | None,
        partner_policy_fusion_input_supplied: Array,
        partner_policy_fusion_feedback: (
            PrototypePartnerPolicyFusionFeedback | None
        ),
        partner_policy_fusion_feedback_supplied: Array,
    ) -> PrototypeUpdateResult:
        """Apply one normalized, ownership-validated real transition."""
        bootstrap_raw_obs = transition.next_observation
        decision_raw_obs = transition.next_decision_observation
        execution_boundary = transition.terminated | transition.truncated
        rew = transition.reward

        # -- Step 8a: consume bootstrap state, then restart once at a boundary -
        new_gru_state = state.gru_state
        old_builder_state = self._builder_component_state(
            state.state_builder_state
        )
        new_builder_state = old_builder_state
        old_feature_state = (
            self._feature_lifecycle_component_state(
                state.state_builder_state
            )
            if self._prototype_feature_lifecycle is not None
            else None
        )
        new_feature_state = old_feature_state
        current_oak_state = self._oak_component_state(state.oak_state)
        old_feature_consumer_binding = (
            self._feature_consumer_binding(state.oak_state)
            if self._prototype_feature_lifecycle is not None
            else None
        )
        new_feature_consumer_binding = old_feature_consumer_binding
        bootstrap_base_obs = bootstrap_raw_obs
        decision_base_obs = decision_raw_obs
        if self._state_builder is not None:
            (
                bootstrap_builder_state,
                bootstrap_base_obs,
            ) = self._state_builder.update(
                old_builder_state,
                bootstrap_raw_obs,
                transition.action,
                rew,
                transition.discount,
            )

            def restart_builder(builder_state: Any) -> tuple[Any, Array]:
                reset_state = self._state_builder.reset_episode(builder_state)
                return self._state_builder.start(reset_state, decision_raw_obs)

            new_builder_state, decision_base_obs = jax.lax.cond(
                execution_boundary,
                restart_builder,
                lambda builder_state: (builder_state, bootstrap_base_obs),
                bootstrap_builder_state,
            )
        elif state.gru_state is not None:
            bootstrap_gru_state, bootstrap_base_obs = _gru_step(
                state.gru_state,
                bootstrap_raw_obs,
            )

            def restart_gru(gru_state: GRUPerceptionState) -> tuple[Any, Array]:
                reset_state = cast(
                    GRUPerceptionState,
                    gru_state.replace(hidden=jnp.zeros_like(gru_state.hidden)),
                )
                return _gru_step(reset_state, decision_raw_obs)

            new_gru_state, decision_base_obs = jax.lax.cond(
                execution_boundary,
                restart_gru,
                lambda gru_state: (gru_state, bootstrap_base_obs),
                bootstrap_gru_state,
            )
        bootstrap_obs = self._augment_base_representation(
            old_feature_state,
            bootstrap_base_obs,
        )
        decision_obs = self._augment_base_representation(
            old_feature_state,
            decision_base_obs,
        )

        # Snapshot the real last observation/action before OaK update. The
        # base action may be an extended option index, but the world model is
        # action-conditioned on the primitive command sent to the environment.
        last_obs = state.current_representation
        last_base_obs = last_obs[: self._base_representation_dim()]
        last_action = transition.action

        # -- Step 8b: world model update (real transition) --------------------
        new_wm_state = state.world_model_state
        new_buf_state = state.buffer_state
        wm_error: Any = None
        learning_signals = _unavailable_learning_signals()
        representation_gradient = jnp.zeros(
            (self._config.oak.observation_dim,),
            dtype=jnp.float32,
        )
        representation_gradient_valid = jnp.asarray(False)
        behavior_gradient_result = _unavailable_behavior_gradient(
            self._config.oak.observation_dim
        )
        representation_gradient_mix = _unavailable_representation_gradient_mix(
            self._config.oak.observation_dim
        )
        ensemble_diagnostics = _unavailable_ensemble_diagnostics()
        recurrent_diagnostics = _unavailable_recurrent_latent_diagnostics(

                self._config.recurrent_latent_world_model_ensemble.ensemble_size
                if self._config.recurrent_latent_world_model_ensemble is not None
                else 0

        )
        recurrent_transaction_applied = jnp.asarray(True, dtype=jnp.bool_)
        recurrent_next_start_cache: Any = None
        model_replay_transaction_applied = jnp.asarray(False)
        model_replay_recorded = jnp.asarray(False)
        model_replay_sampled = jnp.asarray(False)
        model_replay_updates_applied = jnp.asarray(0, dtype=jnp.int32)
        model_replay_padding_count = jnp.asarray(0, dtype=jnp.int32)
        plain_world_model_transaction_applied = jnp.asarray(
            True,
            dtype=jnp.bool_,
        )

        if self._world_model is not None and self._buffer is not None:
            wm_result = self._world_model.update(
                state.world_model_state,
                last_obs,
                last_action,
                rew,
                transition.discount,
                bootstrap_obs,
            )
            new_wm_state = wm_result.state
            plain_world_model_transaction_applied = wm_result.update_applied
            bootstrap_buffer_state = self._buffer.add(
                state.buffer_state,
                bootstrap_obs,
            )
            new_buf_state = jax.lax.cond(
                execution_boundary,
                lambda buffer_state: self._buffer.add(buffer_state, decision_obs),
                lambda buffer_state: buffer_state,
                bootstrap_buffer_state,
            )
            wm_error = wm_result.prediction_error
        elif self._world_model_ensemble is not None:
            ensemble_result = self._world_model_ensemble.update(
                state.world_model_state,
                last_obs,
                last_action,
                rew,
                transition.discount,
                bootstrap_obs,
            )
            new_wm_state = ensemble_result.state
            wm_error = ensemble_result.observed_loss
            learning_signals = ensemble_result.signals
            representation_gradient = ensemble_result.representation_gradient
            representation_gradient_valid = (
                ensemble_result.representation_gradient_valid
            )
            ensemble_diagnostics = ensemble_result.diagnostics
        elif self._model_replay_rehearsal is not None:
            representation_version = (
                state.state_builder_state.update_count
                if isinstance(
                    self._config.state_builder,
                    OnlineGatedStateBuilderConfig,
                )
                else jnp.asarray(0, dtype=jnp.int32)
            )
            rehearsal_result = self._model_replay_rehearsal.step(
                state.world_model_state,
                RealModelReplayEvent(
                    observation=last_obs,
                    action=last_action,
                    reward=rew,
                    discount=transition.discount,
                    terminated=transition.terminated,
                    truncated=transition.truncated,
                    next_observation=bootstrap_obs,
                    representation_version=representation_version,
                    provenance_id=state.step_count,
                    source_id=jnp.asarray(0, dtype=jnp.int32),
                    safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
                    safety_cost_available=jnp.asarray(False, dtype=jnp.bool_),
                    valid=diagnostics.valid,
                ),
            )
            new_wm_state = rehearsal_result.state
            wm_error = rehearsal_result.real_observed_loss
            learning_signals = rehearsal_result.real_signals
            representation_gradient = (
                rehearsal_result.real_representation_gradient
            )
            representation_gradient_valid = (
                rehearsal_result.real_representation_gradient_valid
            )
            ensemble_diagnostics = cast(
                WorldModelEnsembleDiagnostics,
                rehearsal_result.real_update_diagnostics.replace(
                    applied=rehearsal_result.diagnostics.transaction_applied,
                    rejected=~rehearsal_result.diagnostics.transaction_applied,
                ),
            )
            model_replay_transaction_applied = (
                rehearsal_result.diagnostics.transaction_applied
            )
            model_replay_recorded = (
                rehearsal_result.diagnostics.transaction_applied
                & rehearsal_result.diagnostics.replay_recorded
            )
            model_replay_sampled = (
                rehearsal_result.diagnostics.transaction_applied
                & rehearsal_result.diagnostics.replay_sampled
            )
            model_replay_updates_applied = jnp.where(
                rehearsal_result.diagnostics.transaction_applied,
                jnp.sum(
                    rehearsal_result.trace.model_updates_applied.astype(
                        jnp.int32
                    )
                ),
                jnp.asarray(0, dtype=jnp.int32),
            )
            model_replay_padding_count = jnp.where(
                rehearsal_result.diagnostics.transaction_applied,
                jnp.sum(rehearsal_result.trace.padding.astype(jnp.int32)),
                jnp.asarray(0, dtype=jnp.int32),
            )
        elif (
            self._recurrent_latent_world_model_ensemble is not None
            and self._recurrent_signal_estimator is not None
        ):
            recurrent_wrapper = cast(
                PrototypeRecurrentLatentWorldModelState,
                state.world_model_state,
            )
            recurrent_result = self._recurrent_latent_world_model_ensemble.update(
                recurrent_wrapper.model_state,
                recurrent_wrapper.decision_cache,
                RecurrentLatentTransitionRecord(
                    observation=last_obs,
                    action=last_action,
                    reward=rew,
                    discount=transition.discount,
                    terminated=transition.terminated,
                    truncated=transition.truncated,
                    bootstrap_observation=bootstrap_obs,
                    next_decision_observation=decision_obs,
                ),
            )
            # Gaussian NLL can be negative. A fixed configuration-derived
            # affine offset (with a one-unit margin) makes the estimator input
            # non-negative without changing learning-progress differences.
            shifted_nll = (
                recurrent_result.mean_negative_log_likelihood
                + jnp.asarray(
                    self._recurrent_signal_nll_offset,
                    dtype=jnp.float32,
                )
            )
            candidate_signal_state, raw_signals = (
                self._recurrent_signal_estimator.observe(
                    recurrent_wrapper.signal_state,
                    recurrent_result.prediction.member_mean_predictions,
                    recurrent_result.prediction.member_aleatoric_variances,
                    recurrent_result.targets,
                    shifted_nll,
                )
            )
            recurrent_signals_valid = raw_signals.availability.input_valid
            recurrent_transaction_applied = (
                recurrent_result.diagnostics.applied
                & recurrent_signals_valid
            )
            committed_model_state = cast(
                RecurrentLatentWorldModelEnsembleState,
                jax.lax.cond(
                    recurrent_transaction_applied,
                    lambda: recurrent_result.state,
                    lambda: recurrent_wrapper.model_state,
                ),
            )
            committed_signal_state = cast(
                LearningSignalEstimatorState,
                jax.lax.cond(
                    recurrent_transaction_applied,
                    lambda: candidate_signal_state,
                    lambda: recurrent_wrapper.signal_state,
                ),
            )
            new_wm_state = recurrent_wrapper.replace(
                model_state=committed_model_state,
                signal_state=committed_signal_state,
            )
            recurrent_next_start_cache = recurrent_result.next_start_cache
            wm_error = jnp.where(
                recurrent_transaction_applied,
                recurrent_result.mean_negative_log_likelihood,
                jnp.asarray(0.0, dtype=jnp.float32),
            )
            learning_signals = _gate_recurrent_learning_signals(
                raw_signals,
                recurrent_result.prediction.availability,
                recurrent_transaction_applied,
            )
            representation_gradient = jnp.where(
                recurrent_transaction_applied,
                recurrent_result.representation_gradient,
                jnp.zeros_like(recurrent_result.representation_gradient),
            )
            representation_gradient_valid = (
                recurrent_transaction_applied
                & recurrent_result.representation_gradient_available
            )
            prediction_availability = RecurrentLatentPredictionAvailability(
                prediction=(
                    recurrent_transaction_applied
                    & recurrent_result.prediction.availability.prediction
                ),
                epistemic=(
                    recurrent_transaction_applied
                    & recurrent_result.prediction.availability.epistemic
                ),
                aleatoric=(
                    recurrent_transaction_applied
                    & recurrent_result.prediction.availability.aleatoric
                ),
            )
            recurrent_diagnostics = PrototypeRecurrentLatentDiagnostics(
                model=recurrent_result.diagnostics,
                prediction_availability=prediction_availability,
                raw_epistemic_disagreement=jnp.where(
                    recurrent_transaction_applied,
                    recurrent_result.prediction.epistemic_disagreement,
                    jnp.asarray(0.0, dtype=jnp.float32),
                ),
                raw_aleatoric_uncertainty=jnp.where(
                    recurrent_transaction_applied,
                    recurrent_result.prediction.aleatoric_uncertainty,
                    jnp.asarray(0.0, dtype=jnp.float32),
                ),
                raw_uncertainty_calibrated=jnp.asarray(False, dtype=jnp.bool_),
                signals_valid=(
                    recurrent_result.diagnostics.applied
                    & recurrent_signals_valid
                ),
                transaction_applied=recurrent_transaction_applied,
                next_decision_cached=jnp.asarray(False, dtype=jnp.bool_),
            )

        builder_representation_gradient = representation_gradient
        builder_representation_gradient_valid = representation_gradient_valid
        mixer_config = self._config.representation_gradient_mixer
        if (
            mixer_config is not None
            or self._prototype_feature_lifecycle is not None
        ):
            behavior_gradient_result = self._behavior_representation_gradient(
                state,
                rew,
                bootstrap_obs,
                control_discount,
            )
        if mixer_config is not None:
            representation_gradient_mix = mix_representation_gradients(
                mixer_config,
                behavior_gradient_result.gradient,
                representation_gradient,
                behavior_valid=behavior_gradient_result.valid,
                grounded_world_valid=representation_gradient_valid,
            )
            builder_representation_gradient = representation_gradient_mix.gradient
            # ``valid`` also describes a deliberately empty ``discard`` mix.
            # Only ``applied`` means a real active source produced a builder
            # candidate, so zero-source modes never consume update capacity.
            builder_representation_gradient_valid = (
                representation_gradient_mix.applied
            )

        pullback_generation = (
            old_feature_state.router_state.generation_count
            if old_feature_state is not None
            else jnp.asarray(0, dtype=jnp.int32)
        )
        pair_gradient_pullback = PrototypePairGradientPullback(
            gradient=jnp.zeros(
                (self._base_representation_dim(),),
                dtype=jnp.float32,
            ),
            valid=jnp.asarray(False, dtype=jnp.bool_),
            semantic_generation=pullback_generation,
        )
        if self._prototype_feature_lifecycle is not None:
            if old_feature_state is None:
                raise RuntimeError(
                    "configured prototype feature lifecycle requires state"
                )
            candidate_pullback = (
                self._prototype_feature_lifecycle.pullback_pair_gradient(
                    old_feature_state,
                    last_base_obs,
                    builder_representation_gradient,
                    old_feature_state.router_state.generation_count,
                    old_feature_state.router_state.descriptors,
                )
            )
            pullback_valid = (
                builder_representation_gradient_valid
                & candidate_pullback.valid
                & (
                    candidate_pullback.semantic_generation
                    == old_feature_state.router_state.generation_count
                )
            )
            builder_representation_gradient = jnp.where(
                pullback_valid,
                candidate_pullback.gradient,
                jnp.zeros_like(candidate_pullback.gradient),
            )
            builder_representation_gradient_valid = pullback_valid
            pair_gradient_pullback = cast(
                PrototypePairGradientPullback,
                candidate_pullback.replace(valid=pullback_valid),
            )

        # Learn the representation causally: the proposal is formed from the
        # source state whose sensitivity emitted ``last_obs``. It is committed
        # into the destination state only after that destination has consumed
        # the real transition (and any autoreset observation), so recurrence is
        # neither recomputed under new parameters nor replaced by stale state.
        (
            new_builder_state,
            builder_learning_diagnostics,
            gradient_joy_application,
        ) = self._apply_state_builder_learning(
            old_builder_state,
            new_builder_state,
            builder_representation_gradient,
            builder_representation_gradient_valid,
            learning_signals,
            gradient_joy_evidence,
        )

        # -- Steps 5/6/10/11: OaK update (real transition) -------------------
        oak_result = self._oak.update(
            current_oak_state,
            rew,
            bootstrap_obs,
            control_discount,
            decision_observation=decision_obs,
            execution_boundary=execution_boundary,
        )
        new_oak_state = oak_result.state

        # -- Opt-in bounded option-model search control ----------------------
        # The legacy STOMP planning budget is statically required to be zero
        # when this composition is enabled. Search is anchored to the exact
        # next decision representation (not an autoreset transition's final
        # observation) and commits only the base learner subtree. OaK has
        # already selected this event's cached dispatch, which is deliberately
        # preserved; value changes become behavior-eligible only at the next
        # extended-action selection boundary (possibly after an active option
        # runs for several more primitive decisions).
        option_search_diagnostics: OptionSearchControlDiagnostics | None = None
        if self._option_search_control is not None:
            option_search_result = self._option_search_control.apply(
                new_oak_state.stomp_state,
                decision_obs,
            )
            new_oak_state = cast(
                OaKState,
                new_oak_state.replace(
                    stomp_state=option_search_result.state,
                ),
            )
            option_search_diagnostics = option_search_result.diagnostics

        next_armed = (
            diagnostics.next_generation_available
            & diagnostics.next_counter_capacity_available
        )
        feature_lifecycle_diagnostics: (
            PrototypeFeatureLifecycleIntegrationDiagnostics | None
        ) = None
        if self._prototype_feature_lifecycle is not None:
            if old_feature_state is None:
                raise RuntimeError(
                    "configured prototype feature lifecycle requires state"
                )
            target_available = behavior_gradient_result.valid
            automatic_target = jnp.where(
                target_available,
                behavior_gradient_result.diagnostics.target,
                jnp.asarray(0.0, dtype=jnp.float32),
            )
            feature_targets = jnp.where(
                target_available,
                jnp.reshape(automatic_target, (1,)),
                jnp.full((1,), jnp.nan, dtype=jnp.float32),
            )
            feature_result = (
                self._prototype_feature_lifecycle.observe_and_route(
                    old_feature_state,
                    new_oak_state,
                    old_feature_consumer_binding,
                    PrototypeFeatureLifecycleEvent(
                        observation=last_base_obs,
                        targets=feature_targets,
                        next_observation=decision_base_obs,
                        allow_curation=jnp.asarray(
                            next_armed,
                            dtype=jnp.bool_,
                        ),
                    ),
                )
            )
            new_feature_state = feature_result.state
            new_oak_state = feature_result.oak_state
            new_feature_consumer_binding = feature_result.consumer_binding
            decision_obs = feature_result.next_augmented_observation
            prediction = feature_result.predictions[0]
            error = feature_result.errors[0]
            feature_lifecycle_diagnostics = (
                PrototypeFeatureLifecycleIntegrationDiagnostics(
                    available=jnp.asarray(True, dtype=jnp.bool_),
                    target=automatic_target,
                    target_available=target_available,
                    pullback_gradient=pair_gradient_pullback.gradient,
                    pullback_valid=pair_gradient_pullback.valid,
                    pullback_semantic_generation=(
                        pair_gradient_pullback.semantic_generation
                    ),
                    prediction=jnp.where(
                        jnp.isfinite(prediction),
                        prediction,
                        jnp.asarray(0.0, dtype=jnp.float32),
                    ),
                    error=jnp.where(
                        jnp.isfinite(error),
                        error,
                        jnp.asarray(0.0, dtype=jnp.float32),
                    ),
                    metrics=feature_result.metrics,
                    lifecycle=feature_result.diagnostics,
                    outer_transaction_committed=jnp.asarray(
                        False,
                        dtype=jnp.bool_,
                    ),
                )
            )

        # -- Step 9: guarded Dyna dreaming ------------------------------------
        dream_td_errors: Any = None

        if (
            self._world_model is not None
            and self._buffer is not None
            and self._dreamer is not None
            and self._config.n_dreams_per_step > 0
        ):
            rng_key = new_oak_state.stomp_state.rng_key
            new_oak_state, dream_td_errors = self._run_dreams(
                new_oak_state, new_wm_state, new_buf_state, rng_key
            )

        # -- Step 3: Horde GVF update -----------------------------------------
        new_horde_state = state.horde_state
        horde_tderrs: Any = None

        if self._horde is not None:
            horde_cumulants = transition.horde_cumulants
            if horde_cumulants is None:
                horde_cumulants = jnp.full(
                    (self._horde.n_demons,), rew, dtype=jnp.float32
                )
            horde_discounts = transition.horde_discounts
            if horde_discounts is None:
                # Default terminal/global rule: preserve each GVF's declared
                # horizon on every continuing transition. Only a true global
                # terminal (discount == 0) zeros all bootstraps.
                horde_spec = cast(HordeSpec, self._config.horde_spec)
                horde_discounts = jnp.where(
                    transition.discount > 0.0,
                    horde_spec.gammas,
                    jnp.zeros_like(horde_spec.gammas),
                )
            horde_result = self._horde.update_with_discounts(
                state.horde_state,
                last_obs,
                horde_cumulants,
                bootstrap_obs,
                horde_discounts,
            )
            new_horde_state = horde_result.state
            horde_tderrs = horde_result.td_errors

        # -- Step 12: IA update -----------------------------------------------
        current_ia_state = self._ia_component_state(state.ia_state)
        new_ia_component_state = current_ia_state
        ia_augmented: Any = None
        ia_recommendation: Any = None

        if self._ia is not None and current_ia_state is not None:
            ia_result = self._ia.update(
                current_ia_state,
                last_obs,
                rew,
                bootstrap_obs,
                partner_action=last_action,
                discount=control_discount,
                decision_observation=decision_obs,
                execution_boundary=execution_boundary,
            )
            new_ia_component_state = ia_result.state
            ia_augmented = ia_result.augmented_obs
            ia_recommendation = ia_result.recommendation

        next_step_count = _saturating_int32_increment(state.step_count)
        next_observation_event_count = jnp.where(
            execution_boundary,
            _saturating_int32_increment(
                _saturating_int32_increment(state.observation_event_count)
            ),
            _saturating_int32_increment(state.observation_event_count),
        )
        counterfactual_next_action = jnp.where(
            next_armed,
            oak_result.primitive_action,
            jnp.asarray(-1, dtype=jnp.int32),
        )
        next_decision_id = jnp.where(
            next_armed,
            _increment_decision_id(state.current_decision_id),
            state.current_decision_id,
        )
        current_memory_state: ExperientialMemoryState | None = None
        new_memory_state: ExperientialMemoryState | None = None
        memory_diagnostics: PrototypeExperientialMemoryDiagnostics | None = None
        partner_fusion_diagnostics: (
            PrototypePartnerPolicyFusionDiagnostics | None
        ) = None
        next_action = counterfactual_next_action
        memory_safety_mask = jnp.ones(
            (self._config.oak.n_primitive_actions,),
            dtype=jnp.bool_,
        )
        if self._experiential_memory is not None:
            if experiential_memory_input is None:
                raise RuntimeError(
                    "configured experiential memory requires a fixed sidecar"
                )
            current_memory_state = self._experiential_memory_component_state(
                state.ia_state
            )
            (
                new_memory_state,
                new_oak_state,
                next_action,
                memory_diagnostics,
            ) = self._apply_experiential_memory(
                current_memory_state,
                new_oak_state,
                last_obs,
                bootstrap_obs,
                decision_obs,
                transition.action,
                rew,
                current_prototype_decision_id=state.current_decision_id,
                next_prototype_decision_id=next_decision_id,
                next_armed=next_armed,
                transaction_allowed=jnp.asarray(True, dtype=jnp.bool_),
                memory_input=experiential_memory_input,
                input_supplied=experiential_memory_input_supplied,
            )
            memory_safety_mask = jnp.where(
                memory_diagnostics.transaction_required,
                experiential_memory_input.next_action_safety_mask,
                memory_safety_mask,
            )
        new_interaction_state = self._interaction_slot(
            new_ia_component_state,
            (
                self._partner_fusion_component_state(state.ia_state)
                if self._partner_policy_fusion is not None
                else None
            ),
            experiential_memory_state=new_memory_state,
            feedback_prototype_decision_id=(
                self._partner_interaction_state(
                    state.ia_state
                ).feedback_prototype_decision_id
                if self._partner_policy_fusion is not None
                else None
            ),
            feedback_prototype_decision_id_available=(
                self._partner_interaction_state(
                    state.ia_state
                ).feedback_prototype_decision_id_available
                if self._partner_policy_fusion is not None
                else None
            ),
        )
        if self._partner_policy_fusion is not None:
            if (
                partner_policy_fusion_input is None
                or partner_policy_fusion_feedback is None
            ):
                raise RuntimeError("configured partner fusion requires fixed sidecars")
            (
                new_interaction_state,
                new_oak_state,
                next_action,
                partner_fusion_diagnostics,
            ) = self._apply_partner_policy_fusion(
                state.ia_state,
                new_ia_component_state,
                new_oak_state,
                decision_obs,
                derived_decision_id=next_step_count,
                derived_event_id=next_observation_event_count,
                derived_prototype_decision_id=next_decision_id,
                next_armed=next_armed,
                transaction_allowed=jnp.asarray(True, dtype=jnp.bool_),
                experiential_memory_state=new_memory_state,
                upstream_safety_action_mask=memory_safety_mask,
                decision_input=partner_policy_fusion_input,
                decision_input_supplied=partner_policy_fusion_input_supplied,
                feedback=partner_policy_fusion_feedback,
                feedback_input_supplied=(
                    partner_policy_fusion_feedback_supplied
                ),
            )
        if self._recurrent_latent_world_model_ensemble is not None:
            recurrent_wrapper = cast(
                PrototypeRecurrentLatentWorldModelState,
                new_wm_state,
            )
            next_recurrent_decision = self._recurrent_decide_from_start(
                recurrent_wrapper.model_state,
                cast(RecurrentLatentStartCache, recurrent_next_start_cache),
                next_action,
                next_armed & recurrent_transaction_applied,
            )
            new_wm_state = recurrent_wrapper.replace(
                decision_cache=next_recurrent_decision,
            )
            recurrent_diagnostics = recurrent_diagnostics.replace(
                next_decision_cached=(
                    recurrent_transaction_applied
                    & next_armed
                    & next_recurrent_decision.valid
                ),
            )
        new_state = PrototypeAgentState(
            oak_state=self._oak_state_slot(
                new_oak_state,
                new_feature_consumer_binding,
            ),
            world_model_state=new_wm_state,
            buffer_state=new_buf_state,
            horde_state=new_horde_state,
            ia_state=new_interaction_state,
            gru_state=new_gru_state,
            state_builder_state=self._representation_state_slot(
                new_builder_state,
                new_feature_state,
            ),
            current_raw_observation=decision_raw_obs,
            current_representation=decision_obs,
            current_action=next_action,
            current_decision_id=next_decision_id,
            started=next_armed,
            observation_event_count=next_observation_event_count,
            step_count=next_step_count,
        )

        component_diagnostics = cast(
            PrototypeTransitionDiagnostics,
            diagnostics.replace(
                valid=(
                    diagnostics.valid
                    & plain_world_model_transaction_applied
                ),
                rejected=~(
                    diagnostics.valid
                    & plain_world_model_transaction_applied
                ),
            ),
        )
        return PrototypeUpdateResult(
            state=new_state,
            action=next_action,
            oak_td_error=oak_result.td_error,
            oak_average_reward=oak_result.average_reward,
            world_model_error=wm_error,
            learning_signals=learning_signals,
            world_model_representation_gradient=representation_gradient,
            world_model_representation_gradient_valid=(
                representation_gradient_valid
            ),
            behavior_gradient_result=behavior_gradient_result,
            representation_gradient_mix=representation_gradient_mix,
            world_model_ensemble_diagnostics=ensemble_diagnostics,
            recurrent_latent_world_model_diagnostics=recurrent_diagnostics,
            model_replay_transaction_applied=model_replay_transaction_applied,
            model_replay_recorded=model_replay_recorded,
            model_replay_sampled=model_replay_sampled,
            model_replay_updates_applied=model_replay_updates_applied,
            model_replay_padding_count=model_replay_padding_count,
            state_builder_learning_diagnostics=builder_learning_diagnostics,
            gradient_joy_application=gradient_joy_application,
            gradient_joy_evidence_supplied=gradient_joy_evidence_supplied,
            gradient_joy_decision_id_matches=(
                gradient_joy_decision_id_matches
            ),
            option_search_control_diagnostics=option_search_diagnostics,
            prototype_feature_lifecycle_diagnostics=(
                feature_lifecycle_diagnostics
            ),
            dream_td_errors=dream_td_errors,
            horde_td_errors=horde_tderrs,
            ia_augmented_obs=ia_augmented,
            ia_recommendation=ia_recommendation,
            experiential_memory_diagnostics=memory_diagnostics,
            partner_policy_fusion_diagnostics=partner_fusion_diagnostics,
            transition_diagnostics=component_diagnostics,
        )

    # -- Scan-based loop ------------------------------------------------------

    def scan(
        self,
        state: PrototypeAgentState,
        rewards: Float[Array, " num_steps"],
        next_observations: Float[Array, "num_steps obs_dim"],
        horde_cumulants: Float[Array, "num_steps n_demons"] | None = None,
        discounts: Float[Array, " num_steps"] | None = None,
        horde_discounts: Float[Array, "num_steps n_demons"] | None = None,
    ) -> PrototypeArrayResult:
        """Run the agent over pre-collected transition arrays via scan.

        Suitable for simulator pre-training or offline replay.  The world
        model, Horde, and IA companion are all updated at every step.  When a
        Horde is configured but ``horde_cumulants`` is ``None``, the per-step
        reward is broadcast to all demons. Supplying ``discounts`` selects the
        explicit :class:`PrototypeTransition` path. Omitting it preserves the
        historical compatibility behavior of :meth:`update`.

        Args:
            state: Current agent state (primed with :meth:`start`).
            rewards: Scalar rewards, shape ``(num_steps,)``.
            next_observations: Next observations, shape ``(num_steps, obs_dim)``.
            horde_cumulants: Optional per-demon cumulants,
                shape ``(num_steps, n_demons)``.
            discounts: Optional explicit scalar continuation multipliers,
                shape ``(num_steps,)``.
            horde_discounts: Optional effective per-GVF discounts,
                shape ``(num_steps, n_demons)``. Requires ``discounts``.

        Returns:
            :class:`PrototypeArrayResult` with final state and per-step arrays.
        """
        if type(state) is not PrototypeAgentState:
            raise TypeError("state must be an exact PrototypeAgentState")
        if horde_discounts is not None and discounts is None:
            raise ValueError("horde_discounts requires explicit discounts")
        if self._state_builder is not None:
            raise ValueError(
                "legacy array scan is unavailable with state_builder; use "
                "scan_transitions with the explicit transition contract"
            )
        if self._horde is None and (
            horde_cumulants is not None or horde_discounts is not None
        ):
            raise ValueError("Horde arrays require horde_spec to be configured")

        use_explicit_discounts = discounts is not None
        use_horde_cumulants = horde_cumulants is not None
        use_horde_discounts = horde_discounts is not None
        actual_type = type(cast(object, rewards))
        if not (
            actual_type is np.ndarray
            or isinstance(rewards, (jax.Array, jax.core.Tracer))
        ):
            raise TypeError("rewards must be a trusted array")
        try:
            n = int(rewards.shape[0])
        except (AttributeError, IndexError, TypeError, ValueError) as error:
            raise TypeError("rewards must expose trusted shape metadata") from error
        if not 1 <= n <= _INT32_MAX:
            raise ValueError("rewards must contain between 1 and signed-int32 steps")

        raw_observation_dim = (
            self._config.gru_perception.observation_dim
            if self._config.gru_perception is not None
            else self._config.oak.observation_dim
        )
        rewards = _checked_finite_array(
            rewards,
            (n,),
            name="rewards",
        )
        next_observations = _checked_finite_array(
            next_observations,
            (n, raw_observation_dim),
            name="next_observations",
        )
        if use_explicit_discounts:
            discounts = _checked_unit_discount(
                discounts,
                (n,),
                name="discounts",
            )
        if self._horde is not None and horde_cumulants is not None:
            horde_cumulants = _checked_finite_array(
                horde_cumulants,
                (n, self._horde.n_demons),
                name="horde_cumulants",
                allow_nan=True,
            )
        if self._horde is not None and horde_discounts is not None:
            horde_discounts = _checked_unit_discount(
                horde_discounts,
                (n, self._horde.n_demons),
                name="horde_discounts",
            )

        def step_fn(
            carry: PrototypeAgentState,
            inputs: tuple[Array, Array, Array, Array, Array],
        ) -> tuple[
            PrototypeAgentState,
            tuple[Array, Array, Array, Array, Array, Array, Array],
        ]:
            rew, next_obs, transition_discount, hc, hd = inputs
            if use_explicit_discounts:
                result = self.update_transition(
                    carry,
                    PrototypeTransition(
                        observation=carry.current_raw_observation,
                        action=carry.current_action,
                        decision_id=carry.current_decision_id,
                        reward=rew,
                        discount=transition_discount,
                        terminated=transition_discount == 0.0,
                        truncated=jnp.array(False),
                        next_observation=next_obs,
                        next_decision_observation=next_obs,
                        horde_cumulants=hc if use_horde_cumulants else None,
                        horde_discounts=hd if use_horde_discounts else None,
                    ),
                )
            else:
                result = self.update(
                    carry,
                    rew,
                    next_obs,
                    hc if use_horde_cumulants else None,
                )
            return result.state, (
                result.action,
                result.oak_td_error,
                result.oak_average_reward,
                result.transition_diagnostics.valid,
                result.state_builder_learning_diagnostics.applied,
                result.sparks_joy,
                result.joyful_gradient_applied,
            )

        scan_discounts = (
            discounts
            if discounts is not None
            else jnp.ones((n,), dtype=jnp.float32)
        )
        n_horde = self._horde.n_demons if self._horde is not None else 1
        scan_cumulants = (
            horde_cumulants
            if horde_cumulants is not None
            else jnp.zeros((n, n_horde), dtype=jnp.float32)
        )
        scan_horde_discounts = (
            horde_discounts
            if horde_discounts is not None
            else jnp.zeros((n, n_horde), dtype=jnp.float32)
        )
        xs = (
            rewards,
            next_observations,
            scan_discounts,
            scan_cumulants,
            scan_horde_discounts,
        )

        final_state, outputs = jax.lax.scan(step_fn, state, xs)
        (
            actions,
            oak_td_errors,
            oak_avg_rewards,
            transition_valid,
            builder_learning_applied,
            gradient_sparks_joy,
            joyful_gradient_applied,
        ) = outputs

        return PrototypeArrayResult(
            state=final_state,
            actions=actions,
            oak_td_errors=oak_td_errors,
            oak_average_rewards=oak_avg_rewards,
            transition_valid=transition_valid,
            state_builder_learning_applied=builder_learning_applied,
            gradient_sparks_joy=gradient_sparks_joy,
            joyful_gradient_applied=joyful_gradient_applied,
        )

    def scan_transitions(
        self,
        state: PrototypeAgentState,
        transitions: PrototypeTransition,
        gradient_joy_evidence: PrototypeGradientJoyEvidence | None = None,
        *,
        experiential_memory_input: (
            PrototypeExperientialMemoryInput | None
        ) = None,
        partner_policy_fusion_input: (
            PrototypePartnerPolicyFusionInput | None
        ) = None,
        partner_policy_fusion_feedback: (
            PrototypePartnerPolicyFusionFeedback | None
        ) = None,
    ) -> PrototypeArrayResult:
        """Run authoritative transitions and fixed optional sidecars through scan."""

        if type(state) is not PrototypeAgentState:
            raise TypeError("state must be an exact PrototypeAgentState")
        if type(transitions) is not PrototypeTransition:
            raise TypeError("transitions must be an exact PrototypeTransition")
        try:
            n = int(transitions.reward.shape[0])
        except (AttributeError, IndexError, TypeError, ValueError) as error:
            raise TypeError("transitions must expose trusted shape metadata") from error
        if not 1 <= n <= _INT32_MAX:
            raise ValueError("transitions must contain between 1 and signed-int32 steps")

        use_partner_input = partner_policy_fusion_input is not None
        use_partner_feedback = partner_policy_fusion_feedback is not None
        use_memory_input = experiential_memory_input is not None
        if self._experiential_memory is None and use_memory_input:
            raise ValueError(
                "experiential memory scan sidecars require experiential memory"
            )
        if self._partner_policy_fusion is None and (
            use_partner_input or use_partner_feedback
        ):
            raise ValueError(
                "partner fusion scan sidecars require partner policy fusion"
            )

        def outputs_from_result(
            result: PrototypeUpdateResult,
        ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
            return (
                result.action,
                result.oak_td_error,
                result.oak_average_reward,
                result.transition_diagnostics.valid,
                result.state_builder_learning_diagnostics.applied,
                result.sparks_joy,
                result.joyful_gradient_applied,
            )

        def transition_only_step(
            carry: PrototypeAgentState,
            transition: PrototypeTransition,
        ) -> tuple[
            PrototypeAgentState,
            tuple[Array, Array, Array, Array, Array, Array, Array],
        ]:
            result = self.update_transition(carry, transition)
            return result.state, outputs_from_result(result)

        if (
            self._partner_policy_fusion is None
            and self._experiential_memory is None
            and gradient_joy_evidence is None
        ):
            final_state, outputs = jax.lax.scan(
                transition_only_step,
                state,
                transitions,
            )
        elif (
            self._partner_policy_fusion is None
            and self._experiential_memory is None
        ):
            def transition_with_joy_step(
                carry: PrototypeAgentState,
                inputs: tuple[PrototypeTransition, PrototypeGradientJoyEvidence],
            ) -> tuple[
                PrototypeAgentState,
                tuple[Array, Array, Array, Array, Array, Array, Array],
            ]:
                transition, sidecar = inputs
                result = self.update_transition(carry, transition, sidecar)
                return result.state, outputs_from_result(result)

            final_state, outputs = jax.lax.scan(
                transition_with_joy_step,
                state,
                (transitions, gradient_joy_evidence),
            )
        elif self._experiential_memory is None:
            n = int(transitions.reward.shape[0])
            missing_input = self._missing_partner_policy_fusion_input()
            missing_feedback = self._missing_partner_policy_fusion_feedback()
            scan_partner_input = (
                partner_policy_fusion_input
                if partner_policy_fusion_input is not None
                else jax.tree.map(
                    lambda value: jnp.broadcast_to(value, (n,) + value.shape),
                    missing_input,
                )
            )
            scan_partner_feedback = (
                partner_policy_fusion_feedback
                if partner_policy_fusion_feedback is not None
                else jax.tree.map(
                    lambda value: jnp.broadcast_to(value, (n,) + value.shape),
                    missing_feedback,
                )
            )

            if gradient_joy_evidence is None:
                def transition_with_partner_step(
                    carry: PrototypeAgentState,
                    inputs: tuple[
                        PrototypeTransition,
                        PrototypePartnerPolicyFusionInput,
                        PrototypePartnerPolicyFusionFeedback,
                    ],
                ) -> tuple[
                    PrototypeAgentState,
                    tuple[Array, Array, Array, Array, Array, Array, Array],
                ]:
                    transition, fusion_input, fusion_feedback = inputs
                    result = self.update_transition(
                        carry,
                        transition,
                        partner_policy_fusion_input=(
                            fusion_input if use_partner_input else None
                        ),
                        partner_policy_fusion_feedback=(
                            fusion_feedback if use_partner_feedback else None
                        ),
                    )
                    return result.state, outputs_from_result(result)

                final_state, outputs = jax.lax.scan(
                    transition_with_partner_step,
                    state,
                    (
                        transitions,
                        scan_partner_input,
                        scan_partner_feedback,
                    ),
                )
            else:
                def transition_with_all_sidecars_step(
                    carry: PrototypeAgentState,
                    inputs: tuple[
                        PrototypeTransition,
                        PrototypeGradientJoyEvidence,
                        PrototypePartnerPolicyFusionInput,
                        PrototypePartnerPolicyFusionFeedback,
                    ],
                ) -> tuple[
                    PrototypeAgentState,
                    tuple[Array, Array, Array, Array, Array, Array, Array],
                ]:
                    transition, joy, fusion_input, fusion_feedback = inputs
                    result = self.update_transition(
                        carry,
                        transition,
                        joy,
                        partner_policy_fusion_input=(
                            fusion_input if use_partner_input else None
                        ),
                        partner_policy_fusion_feedback=(
                            fusion_feedback if use_partner_feedback else None
                        ),
                    )
                    return result.state, outputs_from_result(result)

                final_state, outputs = jax.lax.scan(
                    transition_with_all_sidecars_step,
                    state,
                    (
                        transitions,
                        gradient_joy_evidence,
                        scan_partner_input,
                        scan_partner_feedback,
                    ),
                )
        else:
            n = int(transitions.reward.shape[0])
            missing_memory_input = self._missing_experiential_memory_input()
            scan_memory_input = (
                experiential_memory_input
                if experiential_memory_input is not None
                else jax.tree.map(
                    lambda value: jnp.broadcast_to(value, (n,) + value.shape),
                    missing_memory_input,
                )
            )
            if self._partner_policy_fusion is None:
                if gradient_joy_evidence is None:
                    def transition_with_memory_step(
                        carry: PrototypeAgentState,
                        inputs: tuple[
                            PrototypeTransition,
                            PrototypeExperientialMemoryInput,
                        ],
                    ) -> tuple[
                        PrototypeAgentState,
                        tuple[Array, Array, Array, Array, Array, Array, Array],
                    ]:
                        transition, memory_input = inputs
                        result = self.update_transition(
                            carry,
                            transition,
                            experiential_memory_input=(
                                memory_input if use_memory_input else None
                            ),
                        )
                        return result.state, outputs_from_result(result)

                    final_state, outputs = jax.lax.scan(
                        transition_with_memory_step,
                        state,
                        (transitions, scan_memory_input),
                    )
                else:
                    def transition_with_memory_and_joy_step(
                        carry: PrototypeAgentState,
                        inputs: tuple[
                            PrototypeTransition,
                            PrototypeGradientJoyEvidence,
                            PrototypeExperientialMemoryInput,
                        ],
                    ) -> tuple[
                        PrototypeAgentState,
                        tuple[Array, Array, Array, Array, Array, Array, Array],
                    ]:
                        transition, joy, memory_input = inputs
                        result = self.update_transition(
                            carry,
                            transition,
                            joy,
                            experiential_memory_input=(
                                memory_input if use_memory_input else None
                            ),
                        )
                        return result.state, outputs_from_result(result)

                    final_state, outputs = jax.lax.scan(
                        transition_with_memory_and_joy_step,
                        state,
                        (
                            transitions,
                            gradient_joy_evidence,
                            scan_memory_input,
                        ),
                    )
            else:
                missing_partner_input = self._missing_partner_policy_fusion_input()
                missing_partner_feedback = (
                    self._missing_partner_policy_fusion_feedback()
                )
                scan_partner_input = (
                    partner_policy_fusion_input
                    if partner_policy_fusion_input is not None
                    else jax.tree.map(
                        lambda value: jnp.broadcast_to(
                            value,
                            (n,) + value.shape,
                        ),
                        missing_partner_input,
                    )
                )
                scan_partner_feedback = (
                    partner_policy_fusion_feedback
                    if partner_policy_fusion_feedback is not None
                    else jax.tree.map(
                        lambda value: jnp.broadcast_to(
                            value,
                            (n,) + value.shape,
                        ),
                        missing_partner_feedback,
                    )
                )
                if gradient_joy_evidence is None:
                    def transition_with_memory_and_partner_step(
                        carry: PrototypeAgentState,
                        inputs: tuple[
                            PrototypeTransition,
                            PrototypeExperientialMemoryInput,
                            PrototypePartnerPolicyFusionInput,
                            PrototypePartnerPolicyFusionFeedback,
                        ],
                    ) -> tuple[
                        PrototypeAgentState,
                        tuple[Array, Array, Array, Array, Array, Array, Array],
                    ]:
                        (
                            transition,
                            memory_input,
                            fusion_input,
                            fusion_feedback,
                        ) = inputs
                        result = self.update_transition(
                            carry,
                            transition,
                            experiential_memory_input=(
                                memory_input if use_memory_input else None
                            ),
                            partner_policy_fusion_input=(
                                fusion_input if use_partner_input else None
                            ),
                            partner_policy_fusion_feedback=(
                                fusion_feedback if use_partner_feedback else None
                            ),
                        )
                        return result.state, outputs_from_result(result)

                    final_state, outputs = jax.lax.scan(
                        transition_with_memory_and_partner_step,
                        state,
                        (
                            transitions,
                            scan_memory_input,
                            scan_partner_input,
                            scan_partner_feedback,
                        ),
                    )
                else:
                    def transition_with_every_sidecar_step(
                        carry: PrototypeAgentState,
                        inputs: tuple[
                            PrototypeTransition,
                            PrototypeGradientJoyEvidence,
                            PrototypeExperientialMemoryInput,
                            PrototypePartnerPolicyFusionInput,
                            PrototypePartnerPolicyFusionFeedback,
                        ],
                    ) -> tuple[
                        PrototypeAgentState,
                        tuple[Array, Array, Array, Array, Array, Array, Array],
                    ]:
                        (
                            transition,
                            joy,
                            memory_input,
                            fusion_input,
                            fusion_feedback,
                        ) = inputs
                        result = self.update_transition(
                            carry,
                            transition,
                            joy,
                            experiential_memory_input=(
                                memory_input if use_memory_input else None
                            ),
                            partner_policy_fusion_input=(
                                fusion_input if use_partner_input else None
                            ),
                            partner_policy_fusion_feedback=(
                                fusion_feedback if use_partner_feedback else None
                            ),
                        )
                        return result.state, outputs_from_result(result)

                    final_state, outputs = jax.lax.scan(
                        transition_with_every_sidecar_step,
                        state,
                        (
                            transitions,
                            gradient_joy_evidence,
                            scan_memory_input,
                            scan_partner_input,
                            scan_partner_feedback,
                        ),
                    )
        (
            actions,
            td_errors,
            average_rewards,
            transition_valid,
            builder_learning_applied,
            gradient_sparks_joy,
            joyful_gradient_applied,
        ) = outputs
        return PrototypeArrayResult(
            state=final_state,
            actions=actions,
            oak_td_errors=td_errors,
            oak_average_rewards=average_rewards,
            transition_valid=transition_valid,
            state_builder_learning_applied=builder_learning_applied,
            gradient_sparks_joy=gradient_sparks_joy,
            joyful_gradient_applied=joyful_gradient_applied,
        )

    # -- Curation (Python-level) ----------------------------------------------

    def curate(
        self,
        state: PrototypeAgentState,
        key: Array,
        available_feature_indices: list[int] | None = None,
    ) -> tuple[PrototypeAgent, PrototypeAgentState]:
        """Replace the lowest-utility OaK option with a new subtask.

        This is a **Python-level operation** — it runs outside
        ``jax.lax.scan`` / JIT and materialises JAX array values.  Call it
        periodically in the outer Python loop.

        Args:
            state: Current agent state.
            key: JAX PRNG key for sampling the replacement feature.
            available_feature_indices: Pool of candidate feature indices.
                Defaults to all indices not currently used by any option.

        Returns:
            ``(new_agent, new_state)`` where ``new_agent`` has updated subtask
            specs and ``new_state`` has the replaced option's arrays zeroed.
        """
        if self._prototype_feature_lifecycle is not None:
            raise ValueError(
                "PrototypeAgent.curate is unavailable when "
                "prototype_feature_lifecycle owns the fixed feature bank"
            )
        current_oak_state = self._oak_component_state(state.oak_state)
        new_oak, new_oak_state = self._oak.curate(
            current_oak_state,
            key,
            available_feature_indices,
        )
        if new_oak is self._oak and new_oak_state is current_oak_state:
            return self, state
        new_config = PrototypeAgentConfig(
            oak=new_oak.config,
            option_search_control=self._config.option_search_control,
            world_model=self._config.world_model,
            world_model_ensemble=self._config.world_model_ensemble,
            model_replay_rehearsal=self._config.model_replay_rehearsal,
            recurrent_latent_world_model_ensemble=(
                self._config.recurrent_latent_world_model_ensemble
            ),
            dreaming=self._config.dreaming,
            buffer_capacity=self._config.buffer_capacity,
            n_dreams_per_step=self._config.n_dreams_per_step,
            dream_next_observation_mode=self._config.dream_next_observation_mode,
            horde_spec=self._config.horde_spec,
            horde_hidden_sizes=self._config.horde_hidden_sizes,
            horde_step_size=self._config.horde_step_size,
            ia=self._config.ia,
            partner_policy_fusion=self._config.partner_policy_fusion,
            experiential_memory=self._config.experiential_memory,
            gru_perception=self._config.gru_perception,
            state_builder=self._config.state_builder,
            learn_state_builder_from_world_model=(
                self._config.learn_state_builder_from_world_model
            ),
            representation_gradient_mixer=(
                self._config.representation_gradient_mixer
            ),
            gradient_joy=self._config.gradient_joy,
            auto_curate_every=self._config.auto_curate_every,
        )
        new_agent = PrototypeAgent(new_config)
        # Transfer all non-OaK sub-states unchanged
        new_state = cast(
            PrototypeAgentState,
            state.replace(oak_state=new_oak_state),
        )
        return new_agent, new_state

    def maybe_curate(
        self,
        state: PrototypeAgentState,
        key: Array,
        available_feature_indices: list[int] | None = None,
    ) -> tuple[PrototypeAgent, PrototypeAgentState]:
        """Curate if ``auto_curate_every`` steps have elapsed.

        Intended for use in the outer Python loop alongside :meth:`update`::

            for obs, reward in stream:
                state, result = agent.update(state, obs, reward, key)
                agent, state = agent.maybe_curate(state, key)

        When ``auto_curate_every == 0`` (default), this returns ``(self, state)``.

        Args:
            state: Current agent state.
            key: JAX PRNG key passed to :meth:`curate` when curation fires.
            available_feature_indices: Pool of candidate features; forwarded
                to :meth:`curate` unchanged.

        Returns:
            ``(agent, state)`` — either the updated pair from :meth:`curate`
            or ``(self, state)`` unchanged.
        """
        if self._prototype_feature_lifecycle is not None:
            raise ValueError(
                "PrototypeAgent.maybe_curate is unavailable when "
                "prototype_feature_lifecycle owns the fixed feature bank"
            )
        n = self._config.auto_curate_every
        if n <= 0 or int(state.step_count) % n != 0:
            return self, state
        return self.curate(state, key, available_feature_indices)

    def auto_subtask_specs(
        self,
        state: PrototypeAgentState,
        *,
        n_subtasks: int = 4,
    ) -> tuple[SubtaskSpec, ...]:
        """Return candidate subtask specs ranked by current Q-weight importance.

        Delegates to :func:`feature_to_subtask_specs` using the threshold,
        scale, and max-step settings from the first existing subtask spec.

        Args:
            state: Current agent state.
            n_subtasks: Non-negative integer scalar number of subtask specs to return.

        Returns:
            Tuple of :class:`SubtaskSpec` instances ranked by importance.
        """
        template = self._config.oak.stomp.subtask_specs[0]
        return feature_to_subtask_specs(
            self._oak_component_state(state.oak_state),
            n_subtasks=n_subtasks,
            threshold=template.threshold,
            pseudo_reward_scale=template.pseudo_reward_scale,
            max_option_steps=template.max_option_steps,
        )


def _prototype_config_digest(config: dict[str, Any]) -> str:
    """Return a canonical SHA-256 digest for a serialized agent config."""

    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prototype_state_prng_impl(state: PrototypeAgentState) -> str:
    """Return the one supported typed-key implementation used by the state."""

    implementations: set[str] = set()
    for leaf in jax.tree_util.tree_leaves(state):
        dtype = getattr(leaf, "dtype", None)
        if dtype is None or not jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            continue
        implementations.add(str(jr.key_impl(leaf)))
    if len(implementations) != 1:
        raise ValueError("prototype state must use one typed PRNG implementation")
    implementation = next(iter(implementations))
    if implementation not in _PROTOTYPE_SUPPORTED_PRNG_IMPLS:
        raise ValueError(
            f"prototype state uses unsupported PRNG implementation {implementation!r}"
        )
    return implementation


def save_prototype_checkpoint(
    agent: PrototypeAgent,
    state: PrototypeAgentState,
    path: str | Path,
) -> None:
    """Persist the complete prototype config, PyTree state, and every RNG.

    The generic checkpoint layer stores the state exactly as supplied.  This
    wrapper adds the configuration needed to reconstruct a matching template
    and a digest that makes accidental config/metadata edits fail closed.
    """

    if not bool(agent._checkpoint_state_valid(state)):
        raise ValueError("cannot save an inconsistent PrototypeAgent state")

    config = agent.to_config()
    prng_impl = _prototype_state_prng_impl(state)
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": PROTOTYPE_CHECKPOINT_SCHEMA,
            "agent_config": config,
            "config_sha256": _prototype_config_digest(config),
            "empty_array_codec": _PROTOTYPE_EMPTY_ARRAY_CODEC,
            "prng_impl": prng_impl,
        },
    )


def load_prototype_checkpoint(
    path: str | Path,
    *,
    template_key: Array | None = None,
    trust_v1_started: bool = False,
) -> tuple[PrototypeAgent, PrototypeAgentState]:
    """Restore a complete prototype and reject unknown/tampered metadata.

    Version-2 checkpoints with the bounded ensemble predate isolated
    model-rehearsal state. Their learned members, residual statistics, causal
    signal state, real bootstrap key/mask, and real counters are preserved;
    only the new replay key/mask/counters are deterministically initialized.

    Version-1 checkpoints did not record whether ``start`` had run, so their
    action cache cannot be armed safely by inference. Pass
    ``trust_v1_started=True`` only when external provenance establishes that
    the stored OaK decision was already selected for dispatch.
    """

    metadata = load_checkpoint_metadata(path)
    schema = metadata.get("schema")
    if type(schema) is not str or schema not in {
        PROTOTYPE_CHECKPOINT_SCHEMA,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V2,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V1,
    }:
        raise ValueError(
            "checkpoint is not an Alberta PrototypeAgent v1/v2/v3 checkpoint"
        )
    empty_array_codec = metadata.get("empty_array_codec")
    if empty_array_codec is not None and (
        type(empty_array_codec) is not str
        or empty_array_codec != _PROTOTYPE_EMPTY_ARRAY_CODEC
    ):
        raise ValueError("prototype checkpoint uses an unknown empty-array codec")
    checkpoint_prng_impl = metadata.get("prng_impl")
    if checkpoint_prng_impl is not None and (
        type(checkpoint_prng_impl) is not str
        or checkpoint_prng_impl not in _PROTOTYPE_SUPPORTED_PRNG_IMPLS
    ):
        raise ValueError("prototype checkpoint uses an unsupported PRNG implementation")
    if empty_array_codec is not None and schema != PROTOTYPE_CHECKPOINT_SCHEMA:
        raise ValueError("prototype empty-array codec is valid only for v3 checkpoints")
    if checkpoint_prng_impl is not None and schema != PROTOTYPE_CHECKPOINT_SCHEMA:
        raise ValueError("prototype PRNG metadata is valid only for v3 checkpoints")
    if empty_array_codec is not None and checkpoint_prng_impl is None:
        raise ValueError("prototype checkpoint is missing PRNG implementation metadata")
    config = metadata.get("agent_config")
    if type(config) is not dict:
        raise ValueError("prototype checkpoint is missing agent_config")
    expected_digest = metadata.get("config_sha256")
    if type(expected_digest) is not str or expected_digest != (
        _prototype_config_digest(config)
    ):
        raise ValueError("prototype checkpoint config digest does not match")

    agent = PrototypeAgent.from_config(config)
    if agent.to_config() != config:
        raise ValueError("prototype checkpoint agent_config is not canonical")
    if (
        agent.config.prototype_feature_lifecycle is not None
        and schema != PROTOTYPE_CHECKPOINT_SCHEMA
    ):
        raise ValueError(
            "prototype_feature_lifecycle is unsupported by legacy v1/v2 "
            "PrototypeAgent checkpoints"
        )
    if checkpoint_prng_impl is None:
        key = jr.key(0) if template_key is None else template_key
    elif template_key is None:
        key = jr.key(0, impl=checkpoint_prng_impl)
    else:
        if str(jr.key_impl(template_key)) != checkpoint_prng_impl:
            raise ValueError(
                "template_key PRNG implementation does not match checkpoint metadata"
            )
        key = template_key
    template = agent.init(key)
    if schema == PROTOTYPE_CHECKPOINT_SCHEMA:
        restored, restored_metadata = load_checkpoint(template, path)
        if restored_metadata != metadata:
            raise ValueError("prototype checkpoint metadata changed between reads")
        restored_state = cast(PrototypeAgentState, restored)
        if not bool(agent._checkpoint_state_valid(restored_state)):
            raise ValueError("prototype checkpoint decision/cache state is inconsistent")
        return agent, restored_state

    if schema == _PROTOTYPE_CHECKPOINT_SCHEMA_V2:
        if agent.config.world_model_ensemble is None:
            restored, restored_metadata = load_checkpoint(template, path)
            if restored_metadata != metadata:
                raise ValueError("prototype checkpoint metadata changed between reads")
            restored_state = cast(PrototypeAgentState, restored)
            if not bool(agent._checkpoint_state_valid(restored_state)):
                raise ValueError(
                    "prototype v2 checkpoint decision/cache state is inconsistent"
                )
            return agent, restored_state

        current_world = cast(WorldModelEnsembleState, template.world_model_state)
        legacy_world_template = _WorldModelEnsembleStateV1(
            member_states=current_world.member_states,
            residual_variances=current_world.residual_variances,
            signal_state=current_world.signal_state,
            bootstrap_key=current_world.bootstrap_key,
            last_bootstrap_mask=current_world.last_bootstrap_mask,
            member_update_counts=current_world.member_update_counts,
            event_count=current_world.event_count,
        )
        legacy_template = template.replace(world_model_state=legacy_world_template)
        restored, restored_metadata = load_checkpoint(legacy_template, path)
        if restored_metadata != metadata:
            raise ValueError("prototype checkpoint metadata changed between reads")
        restored_state = cast(PrototypeAgentState, restored)
        legacy_world = cast(_WorldModelEnsembleStateV1, restored_state.world_model_state)
        migrated_world = WorldModelEnsembleState(
            member_states=legacy_world.member_states,
            residual_variances=legacy_world.residual_variances,
            signal_state=legacy_world.signal_state,
            bootstrap_key=legacy_world.bootstrap_key,
            replay_bootstrap_key=jr.fold_in(
                legacy_world.bootstrap_key,
                _PROTOTYPE_V2_REPLAY_MIGRATION_TAG,
            ),
            last_bootstrap_mask=legacy_world.last_bootstrap_mask,
            last_replay_bootstrap_mask=jnp.zeros_like(
                legacy_world.last_bootstrap_mask,
                dtype=jnp.bool_,
            ),
            member_update_counts=legacy_world.member_update_counts,
            replay_member_update_counts=jnp.zeros_like(
                legacy_world.member_update_counts,
                dtype=jnp.int32,
            ),
            event_count=legacy_world.event_count,
            replay_event_count=jnp.asarray(0, dtype=jnp.int32),
        )
        migrated = restored_state.replace(world_model_state=migrated_world)
        if not bool(agent._checkpoint_state_valid(migrated)):
            raise ValueError("prototype v2 ensemble checkpoint migration is inconsistent")
        return agent, migrated

    if agent.config.state_builder is not None:
        raise ValueError("v1 prototype checkpoints cannot contain state_builder config")
    if not trust_v1_started:
        raise ValueError(
            "v1 prototype checkpoint lifecycle is ambiguous; pass "
            "trust_v1_started=True only for a provenance-verified primed state"
        )
    v1_template = _PrototypeAgentStateV1(
        oak_state=template.oak_state,
        world_model_state=template.world_model_state,
        buffer_state=template.buffer_state,
        horde_state=template.horde_state,
        ia_state=template.ia_state,
        gru_state=template.gru_state,
        step_count=template.step_count,
    )
    restored_v1, restored_metadata = load_checkpoint(v1_template, path)
    if restored_metadata != metadata:
        raise ValueError("prototype checkpoint metadata changed between reads")
    old_state = cast(_PrototypeAgentStateV1, restored_v1)
    representation = old_state.oak_state.stomp_state.base_last_obs
    raw_dim = agent._raw_observation_dim()
    raw_observation = representation[:raw_dim]
    migrated = PrototypeAgentState(
        oak_state=old_state.oak_state,
        world_model_state=old_state.world_model_state,
        buffer_state=old_state.buffer_state,
        horde_state=old_state.horde_state,
        ia_state=old_state.ia_state,
        gru_state=old_state.gru_state,
        state_builder_state=None,
        current_raw_observation=raw_observation,
        current_representation=representation,
        current_action=old_state.oak_state.stomp_state.last_primitive_action,
        current_decision_id=template.current_decision_id,
        started=jnp.array(True),
        observation_event_count=_saturating_int32_increment(
            old_state.step_count,
        ),
        step_count=old_state.step_count,
    )
    if not bool(agent._checkpoint_state_valid(migrated)):
        raise ValueError("trusted v1 prototype checkpoint cache state is inconsistent")
    return agent, migrated


__all__ = [
    "PROTOTYPE_CHECKPOINT_SCHEMA",
    "GRUPerceptionConfig",
    "GRUPerceptionState",
    "PrototypeAgent",
    "PrototypeAgentConfig",
    "PrototypeAgentState",
    "PrototypeArrayResult",
    "PrototypeBehaviorGradientDiagnostics",
    "PrototypeBehaviorGradientResult",
    "PrototypeDecision",
    "PrototypeExperientialMemoryDiagnostics",
    "PrototypeExperientialMemoryInput",
    "PrototypeExperientialMemoryResourceDeclaration",
    "PrototypeFeatureLifecycleIntegrationDiagnostics",
    "PrototypeFeatureOaKState",
    "PrototypeFeatureRepresentationState",
    "PrototypeGradientJoyEvidence",
    "PrototypeInteractionState",
    "PrototypeMemoryInteractionState",
    "PrototypePartnerPolicyFusionDiagnostics",
    "PrototypePartnerPolicyFusionFeedback",
    "PrototypePartnerPolicyFusionInput",
    "PrototypeRecurrentLatentDiagnostics",
    "PrototypeRecurrentLatentWorldModelState",
    "PrototypeTransition",
    "PrototypeTransitionDiagnostics",
    "PrototypeUpdateResult",
    "feature_to_subtask_specs",
    "load_prototype_checkpoint",
    "save_prototype_checkpoint",
]
