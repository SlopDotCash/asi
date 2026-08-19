# mypy: disable-error-code="attr-defined,call-arg"
"""Public Step 9 guarded-dreaming facade.

Step 9 extends Step 7's one-step Dyna planning to error-gated, real-state-
anchored multi-step dreaming.  Key additions over Step 7:

* Uses :class:`ActionConditionedWorldModel` which adds a learned discount/
  termination head, enabling clean multi-step rollout support.
* Dream transitions are accepted only when the world model has accumulated
  sufficient experience *and* its running prediction-error EMA is below a
  configurable threshold.  This prevents model-bias corruption from a poorly
  calibrated environment model.
* A :class:`RecentObservationBuffer` (ring buffer of recent real observations)
  anchors each dream at a genuine past state rather than always the current
  state, improving state-space coverage of imagined experience.

The control learner is the same :class:`DifferentialSARSAAgent` from Step 6,
preserving the continuing / average-reward formulation.

The facade rejects illegal dimensions and scientific scalars before any core
object is constructed. Accepted numbers are canonicalized to builtin ints and
floats; legal endpoints stay valid.
"""

from __future__ import annotations

import functools
from dataclasses import asdict, dataclass, field
from numbers import Integral
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework._seed_validation import require_jax_seed
from alberta_framework.core.average_reward import (
    DifferentialSARSAAgent,
    DifferentialSARSAState,
    DifferentialSARSAUpdateResult,
)
from alberta_framework.core.behavior_model import (
    BehaviorModel,
    BehaviorModelConfig,
    BehaviorModelState,
)
from alberta_framework.core.dreaming import (
    DreamSelectionConfig,
    RecentObservationBuffer,
    RecentObservationBufferState,
    score_dream_candidates,
)
from alberta_framework.core.normalizers import _saturating_int32_counter_increment
from alberta_framework.core.world_model import (
    ActionConditionedWorldModel,
    ActionConditionedWorldModelConfig,
    ActionConditionedWorldModelState,
    WorldModelUpdateResult,
)
from alberta_framework.steps._float32_validation import (
    canonical_float32_storage,
    finite_real_and_float32,
)
from alberta_framework.steps._smoke_record_validation import require_step_shape
from alberta_framework.steps.step6 import (
    Step6DifferentialSARSAConfig,
    make_step6_differential_sarsa_agent,
)

_INT32_MAX = 2**31 - 1
_MAX_CONFIG_SEQUENCE_LENGTH = 4_096
_MAX_DREAM_WORK_PER_REAL_STEP = 4_096
# Matches the established ceiling for other scan-driven array-loop runners
# fixed this session (``core.sarsa._SARSA_SEQUENCE_MAX_STEPS``,
# ``core.average_reward._AVERAGE_REWARD_SEQUENCE_MAX_STEPS``,
# ``core.horde_actor_critic._HORDE_AC_SEQUENCE_MAX_STEPS``). ``run_step9_scan``
# hands ``rewards``/``next_observations`` straight to ``jax.lax.scan`` with no
# bound on the leading (step) axis; bounding only by ``_INT32_MAX`` still
# permits a caller to force JAX to trace/compile a scan of ~2 billion steps,
# hanging the process well before any step executes. Step 9's per-step work
# additionally includes model-based dreaming rollouts, so this module is at
# least as exposed to the hang as its siblings.
_STEP9_SEQUENCE_MAX_STEPS = 10_000
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


@dataclass(frozen=True)
class Step9DreamingConfig:
    """Config for Step 9 guarded-dreaming continuing control.

    The world model learns from every real transition.  Each real step also
    fires a fixed dreaming budget: for each dream the agent samples a recent
    anchor observation, picks a random action, queries the world model, and
    applies the imagined update **only** when the model passes two guards:
    sufficient warm-up data and a low running prediction-error EMA.

    Args:
        control: Step 6 differential SARSA configuration.
        observation_dim: Flat observation dimensionality.
        n_actions: Number of discrete actions (must match
            ``control.n_actions``).
        model_hidden_sizes: MLP trunk widths for the world model. ``()``
            gives a linear model.
        model_step_size: Step-size for the world model learner.
        model_sparsity: Sparse-init fraction for the world model.
        model_include_action_interactions: Append observation-by-action
            interaction features to the world-model input. With
            ``model_hidden_sizes=()`` the base model is linear in
            ``concat(obs, one_hot(action))`` — *additive* in state and action —
            so it cannot represent any reward with state-action interaction
            structure (its per-state action gap is one global offset shared by
            every state). Enable this for tasks whose reward depends jointly
            on state and action (e.g. the XOR payoffs of
            ``SwitchingTwoStateMDP``).
        model_use_layer_norm: Enable layer normalisation in the world model.
        model_gamma: Maximum discount (clips the predicted discount head).
        dreaming_warmup_steps: Real transitions required before any dream can
            be accepted.
        dreaming_max_model_error: Maximum allowed model prediction-error EMA
            for dream acceptance.  Set high (e.g. 1e30) to disable the error
            gate.
        model_error_decay: EMA decay for the model prediction-error tracker.
            Smaller values (e.g. 0.9) react faster to distribution shifts at
            the cost of higher variance.  Default 0.99 (slow, smooth).
        planning_budget: Number of dream steps per real transition.
        buffer_capacity: Number of recent real observations to retain for
            anchor sampling.
        dreams_update_average_reward: Whether imagined (dream) updates may
            move the differential-SARSA average-reward estimate rbar
            (default False). Dyna doctrine: planning backups improve *value*
            estimates, while the reward-rate estimate is a property of actual
            behavior in the real environment — imagined experience should not
            move it. Set True only when dream TD errors should update rbar.
    """

    control: Step6DifferentialSARSAConfig = field(
        default_factory=Step6DifferentialSARSAConfig
    )
    observation_dim: int = 4
    n_actions: int = 2
    model_hidden_sizes: tuple[int, ...] = (64,)
    model_step_size: float = 0.03
    model_sparsity: float = 0.9
    model_include_action_interactions: bool = False
    model_use_layer_norm: bool = True
    model_gamma: float = 0.99
    dreaming_warmup_steps: int = 100
    dreaming_max_model_error: float = 1.0
    model_error_decay: float = 0.99
    behavior_model_step_size: float = 0.05
    planning_budget: int = 1
    dream_rollout_horizon: int = 1
    dream_candidate_count: int = 1
    dream_surprise_weight: float = 1.0
    dream_utility_weight: float = 1.0
    buffer_capacity: int = 64
    dreams_update_average_reward: bool = False

    def __post_init__(self) -> None:
        """Reject illegal dimensions and scientific scalars, then canonicalize."""
        _validate_dreaming_config(self)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["control"] = self.control.to_dict()
        payload["model_hidden_sizes"] = list(self.model_hidden_sizes)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Step9DreamingConfig:
        """Reconstruct from :meth:`to_dict` output."""
        data = _require_exact_payload(
            payload,
            name="Step9DreamingConfig payload",
            fields=frozenset(cls.__dataclass_fields__),
        )
        control_raw = _require_exact_payload(
            data["control"],
            name="control payload",
            fields=frozenset(Step6DifferentialSARSAConfig.__dataclass_fields__),
        )
        data["control"] = Step6DifferentialSARSAConfig.from_dict(
            control_raw
        )
        hs = data["model_hidden_sizes"]
        if type(hs) is not list:
            raise ValueError("model_hidden_sizes payload must be an exact list")
        hidden = cast(list[object], hs)
        _require_sequence_length("model_hidden_sizes", len(hidden))
        data["model_hidden_sizes"] = tuple(hidden)
        return cls(**cast(Any, data))

    def to_world_model_config(self) -> ActionConditionedWorldModelConfig:
        """Return the core world-model config."""
        return ActionConditionedWorldModelConfig(
            observation_dim=self.observation_dim,
            n_actions=self.n_actions,
            hidden_sizes=self.model_hidden_sizes,
            step_size=self.model_step_size,
            sparsity=self.model_sparsity,
            use_layer_norm=self.model_use_layer_norm,
            gamma=self.model_gamma,
            error_decay=self.model_error_decay,
            include_action_interactions=self.model_include_action_interactions,
        )


def _require_real(name: str, value: object) -> float:
    real, _, _, narrowed = finite_real_and_float32(name, value)
    return canonical_float32_storage(real, narrowed)


def _require_nonneg_real(name: str, value: object) -> float:
    real, numerator, _, narrowed = finite_real_and_float32(name, value)
    if real < 0.0 or numerator < 0 or narrowed < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return canonical_float32_storage(real, narrowed)


def _require_unit_interval(name: str, value: object) -> float:
    real, numerator, denominator, narrowed = finite_real_and_float32(name, value)
    if (
        real < 0.0
        or not real <= 1.0
        or numerator < 0
        or numerator > denominator
        or narrowed < 0.0
        or not narrowed <= 1.0
    ):
        raise ValueError(f"{name} must be in [0, 1]")
    return canonical_float32_storage(real, narrowed)


def _require_half_open_unit_interval(name: str, value: object) -> float:
    real, numerator, denominator, narrowed = finite_real_and_float32(name, value)
    if (
        real < 0.0
        or not real < 1.0
        or numerator < 0
        or numerator >= denominator
        or narrowed < 0.0
        or not narrowed < 1.0
    ):
        raise ValueError(f"{name} must be in [0, 1)")
    return canonical_float32_storage(real, narrowed)


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


def _require_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a built-in bool")
    return value


def _require_exact_payload(
    value: object,
    *,
    name: str,
    fields: frozenset[str],
) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be an exact dictionary")
    raw = cast(dict[object, object], value)
    if any(type(key) is not str for key in raw):
        raise ValueError(f"{name} keys must be exact strings")
    keys = cast(set[str], set(raw))
    if keys != fields:
        raise ValueError(f"{name} fields do not match the schema")
    return cast(dict[str, object], dict(raw))


def _require_sequence_length(name: str, count: int) -> None:
    if count > _MAX_CONFIG_SEQUENCE_LENGTH:
        raise ValueError(
            f"{name} must contain at most {_MAX_CONFIG_SEQUENCE_LENGTH} values"
        )


def _checked_product(name: str, *factors: int) -> int:
    product = 1
    for factor in factors:
        if factor < 0 or (factor != 0 and product > _INT32_MAX // factor):
            raise ValueError(f"derived {name} must fit signed int32")
        product *= factor
    return product


def _checked_sum(name: str, *terms: int) -> int:
    total = 0
    for term in terms:
        if term < 0 or term > _INT32_MAX - total:
            raise ValueError(f"derived {name} must fit signed int32")
        total += term
    return total


def _preflight_step9_resources(config: Step9DreamingConfig) -> None:
    buffer_cells = _checked_product(
        "Step 9 observation buffer count",
        config.buffer_capacity,
        config.observation_dim,
    )
    _checked_product("Step 9 observation buffer bytes", 4, buffer_cells)
    dream_work = _checked_product(
        "Step 9 dream work per real step",
        config.planning_budget,
        _checked_sum(
            "Step 9 candidate and rollout work",
            config.dream_candidate_count,
            config.dream_rollout_horizon,
        ),
    )
    if dream_work > _MAX_DREAM_WORK_PER_REAL_STEP:
        raise ValueError(
            "derived Step 9 dream work per real step must be at most "
            f"{_MAX_DREAM_WORK_PER_REAL_STEP}"
        )


def _preflight_step9_smoke_resources(config: Step9DreamingConfig, steps: int) -> None:
    rows = _checked_sum("Step 9 observation row count", steps, 1)
    observations = _checked_product(
        "Step 9 observation count", rows, config.observation_dim
    )
    dream_outputs = _checked_product(
        "Step 9 dream output count", steps, config.planning_budget
    )
    _checked_sum(
        "Step 9 smoke array bytes",
        _checked_product("Step 9 observation bytes", 4, observations),
        _checked_product("Step 9 scalar output bytes", 21, steps),
        _checked_product("Step 9 dream output bytes", 5, dream_outputs),
    )


def _validate_dreaming_config(config: Step9DreamingConfig) -> None:
    if type(config) is not Step9DreamingConfig:
        raise TypeError("config must be an exact Step9DreamingConfig")
    if type(config.control) is not Step6DifferentialSARSAConfig:
        raise TypeError("control must be an exact Step6DifferentialSARSAConfig")
    observation_dim = _require_int(
        "observation_dim", config.observation_dim, minimum=1, maximum=_INT32_MAX
    )
    n_actions = _require_int(
        "n_actions", config.n_actions, minimum=1, maximum=_INT32_MAX
    )
    if config.control.n_actions != n_actions:
        raise ValueError(
            f"control.n_actions ({config.control.n_actions}) must equal "
            f"n_actions ({n_actions})"
        )
    if type(config.model_hidden_sizes) is not tuple:
        raise ValueError("model_hidden_sizes must be a tuple of integers")
    _require_sequence_length("model_hidden_sizes", len(config.model_hidden_sizes))
    model_hidden_sizes = tuple(
        _require_int("model_hidden_sizes", size, minimum=1, maximum=_INT32_MAX)
        for size in config.model_hidden_sizes
    )
    model_step_size = _require_nonneg_real("model_step_size", config.model_step_size)
    model_sparsity = _require_unit_interval("model_sparsity", config.model_sparsity)
    model_include_action_interactions = _require_bool(
        "model_include_action_interactions",
        config.model_include_action_interactions,
    )
    model_use_layer_norm = _require_bool(
        "model_use_layer_norm",
        config.model_use_layer_norm,
    )
    model_gamma = _require_unit_interval("model_gamma", config.model_gamma)
    # The world-model warmup clock is stored as a signed int32 scalar.
    dreaming_warmup_steps = _require_int(
        "dreaming_warmup_steps",
        config.dreaming_warmup_steps,
        minimum=0,
        maximum=_INT32_MAX,
    )
    dreaming_max_model_error = _require_nonneg_real(
        "dreaming_max_model_error",
        config.dreaming_max_model_error,
    )
    model_error_decay = _require_half_open_unit_interval(
        "model_error_decay",
        config.model_error_decay,
    )
    behavior_model_step_size = _require_nonneg_real(
        "behavior_model_step_size",
        config.behavior_model_step_size,
    )
    planning_budget = _require_int(
        "planning_budget", config.planning_budget, minimum=0, maximum=_INT32_MAX
    )
    dream_rollout_horizon = _require_int(
        "dream_rollout_horizon",
        config.dream_rollout_horizon,
        minimum=1,
        maximum=_INT32_MAX,
    )
    # Candidate selection publishes selected indices as signed int32 values.
    dream_candidate_count = _require_int(
        "dream_candidate_count",
        config.dream_candidate_count,
        minimum=1,
        maximum=_INT32_MAX,
    )
    dream_surprise_weight = _require_real(
        "dream_surprise_weight",
        config.dream_surprise_weight,
    )
    dream_utility_weight = _require_real(
        "dream_utility_weight",
        config.dream_utility_weight,
    )
    # Buffer ``size`` and ``index`` are int32. Leaving one count below the
    # maximum keeps repeated full-buffer increments and modulo updates in range.
    buffer_capacity = _require_int(
        "buffer_capacity",
        config.buffer_capacity,
        minimum=1,
        maximum=_INT32_MAX - 1,
    )
    dreams_update_average_reward = _require_bool(
        "dreams_update_average_reward",
        config.dreams_update_average_reward,
    )
    object.__setattr__(config, "observation_dim", observation_dim)
    object.__setattr__(config, "n_actions", n_actions)
    object.__setattr__(config, "model_hidden_sizes", model_hidden_sizes)
    object.__setattr__(config, "model_step_size", model_step_size)
    object.__setattr__(config, "model_sparsity", model_sparsity)
    object.__setattr__(
        config,
        "model_include_action_interactions",
        model_include_action_interactions,
    )
    object.__setattr__(config, "model_use_layer_norm", model_use_layer_norm)
    object.__setattr__(config, "model_gamma", model_gamma)
    object.__setattr__(config, "dreaming_warmup_steps", dreaming_warmup_steps)
    object.__setattr__(config, "dreaming_max_model_error", dreaming_max_model_error)
    object.__setattr__(config, "model_error_decay", model_error_decay)
    object.__setattr__(config, "behavior_model_step_size", behavior_model_step_size)
    object.__setattr__(config, "planning_budget", planning_budget)
    object.__setattr__(config, "dream_rollout_horizon", dream_rollout_horizon)
    object.__setattr__(config, "dream_candidate_count", dream_candidate_count)
    object.__setattr__(config, "dream_surprise_weight", dream_surprise_weight)
    object.__setattr__(config, "dream_utility_weight", dream_utility_weight)
    object.__setattr__(config, "buffer_capacity", buffer_capacity)
    object.__setattr__(config, "dreams_update_average_reward", dreams_update_average_reward)
    _preflight_step9_resources(config)
    config.to_world_model_config()
    config.control.to_core_config()


@chex.dataclass(frozen=True)
class Step9DreamingState:
    """Combined Step 9 state."""

    control_state: DifferentialSARSAState
    world_model_state: ActionConditionedWorldModelState
    behavior_model_state: BehaviorModelState
    buffer_state: RecentObservationBufferState
    step_count: Array


@chex.dataclass(frozen=True)
class Step9DreamingUpdateResult:
    """Result from one real transition plus guarded dreaming."""

    state: Step9DreamingState
    real_control_result: DifferentialSARSAUpdateResult
    real_model_result: WorldModelUpdateResult
    dream_td_errors: Array
    dream_accepted: Array


@chex.dataclass(frozen=True)
class Step9ArrayResult:
    """Scan result for Step 9 dreaming over real transition arrays."""

    state: Step9DreamingState
    real_td_errors: Array
    average_rewards: Array
    actions: Array
    model_prediction_errors: Array
    model_updates_applied: Array
    dream_td_errors: Array
    dream_accepted: Array


@dataclass(frozen=True)
class Step9SmokeResult:
    """Summary returned by :func:`run_step9_smoke`."""

    config: Step9DreamingConfig
    steps: int
    seed: int
    real_td_errors_shape: tuple[int, ...]
    dream_td_errors_shape: tuple[int, ...]
    actions_shape: tuple[int, ...]
    finite: bool
    dream_acceptance_count: int
    control_config: dict[str, Any]
    world_model_config: dict[str, Any]

    def __post_init__(self) -> None:
        if type(self.config) is not Step9DreamingConfig:
            raise TypeError("config must be an exact Step9DreamingConfig")
        if type(self.control_config) is not dict:
            raise TypeError("control_config must be an exact dictionary")
        if type(self.world_model_config) is not dict:
            raise TypeError("world_model_config must be an exact dictionary")
        object.__setattr__(
            self, "steps", _require_int("steps", self.steps, minimum=1, maximum=_INT32_MAX)
        )
        object.__setattr__(self, "seed", require_jax_seed(self.seed, name="seed"))
        for name in ("real_td_errors_shape", "dream_td_errors_shape", "actions_shape"):
            object.__setattr__(
                self,
                name,
                require_step_shape(name, getattr(self, name), steps=self.steps),
            )
        object.__setattr__(self, "finite", _require_bool("finite", self.finite))
        object.__setattr__(
            self,
            "dream_acceptance_count",
            _require_int(
                "dream_acceptance_count",
                self.dream_acceptance_count,
                minimum=0,
                maximum=_INT32_MAX,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["config"] = self.config.to_dict()
        payload["real_td_errors_shape"] = list(self.real_td_errors_shape)
        payload["dream_td_errors_shape"] = list(self.dream_td_errors_shape)
        payload["actions_shape"] = list(self.actions_shape)
        return payload


def make_step9_components(
    config: Step9DreamingConfig | None = None,
) -> tuple[DifferentialSARSAAgent, ActionConditionedWorldModel, RecentObservationBuffer]:
    """Create the Step 9 control agent, world model, and observation buffer."""
    if config is None:
        cfg = Step9DreamingConfig()
    elif type(config) is Step9DreamingConfig:
        cfg = config
    else:
        raise TypeError("config must be an exact Step9DreamingConfig")
    _preflight_step9_resources(cfg)
    agent = make_step6_differential_sarsa_agent(cfg.control)
    model = ActionConditionedWorldModel(cfg.to_world_model_config())
    buffer = RecentObservationBuffer(cfg.buffer_capacity, cfg.observation_dim)
    return agent, model, buffer


def init_step9_state(
    agent: DifferentialSARSAAgent,
    model: ActionConditionedWorldModel,
    buffer: RecentObservationBuffer,
    *,
    key: Array,
    initial_observation: Array,
) -> Step9DreamingState:
    """Initialize and prime the Step 9 state."""
    control_key, model_key, behavior_key = jr.split(key, 3)
    feature_dim = int(jnp.ravel(initial_observation).shape[0])
    control_state = agent.init(feature_dim, control_key)
    control_state, _ = agent.start(control_state, initial_observation)
    behavior_model = BehaviorModel(
        BehaviorModelConfig(
            n_actions=agent.config.n_actions,
        )
    )
    buffer_state = buffer.init()
    buffer_state = buffer.add(buffer_state, initial_observation)
    return Step9DreamingState(
        control_state=control_state,
        world_model_state=model.init(model_key),
        behavior_model_state=behavior_model.init(feature_dim, behavior_key),
        buffer_state=buffer_state,
        step_count=jnp.array(0, dtype=jnp.int32),
    )


def _update_control_with_linear_rng(
    agent: DifferentialSARSAAgent,
    state: DifferentialSARSAState,
    reward: Array,
    next_observation: Array,
    discount: Array,
) -> DifferentialSARSAUpdateResult:
    """Apply one control backup while advancing its RNG on rejection.

    ``DifferentialSARSAAgent.update`` rolls its whole state back when a
    transaction is rejected, including the key consumed while selecting the
    proposed next action.  Select that action explicitly so the advanced key
    can remain linear even when the numerical update is rejected.
    """
    next_action, next_key = agent.select_action(state, next_observation)
    advanced_state = state.replace(rng_key=next_key)
    return cast(
        DifferentialSARSAUpdateResult,
        agent.update(
            advanced_state,
            reward,
            next_observation,
            next_action=next_action,
            discount=discount,
        ),
    )


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3))
def step9_update(
    config: Step9DreamingConfig,
    agent: DifferentialSARSAAgent,
    model: ActionConditionedWorldModel,
    buffer: RecentObservationBuffer,
    state: Step9DreamingState,
    reward: Array,
    next_observation: Array,
) -> Step9DreamingUpdateResult:
    """Run one foreground real update plus error-gated dreaming.

    The real model update always executes first.  The freshly updated model
    error EMA then gates each dream in the planning budget: a dream is
    accepted when ``model_state.step_count >= dreaming_warmup_steps`` AND
    ``model_state.model_error_ema <= dreaming_max_model_error`` AND the
    predicted transition is numerically finite AND every rollout control
    transaction applies.
    """
    real_discount = jnp.asarray(config.model_gamma, dtype=jnp.float32)
    real_model_result = model.update(
        state.world_model_state,
        state.control_state.last_observation,
        state.control_state.last_action,
        reward,
        real_discount,
        next_observation,
    )
    if config.planning_budget == 0:
        # The real-only lane remains the exact pre-dreaming control path: no
        # additional selection or split is introduced for a zero budget.
        real_control_result = agent.update(
            state.control_state,
            reward,
            next_observation,
            discount=real_discount,
        )
    else:
        # A rejected real update must not hand its already-consumed parent key
        # to the dream scheduler, whose four-way split would reproduce the
        # failed action selection's sibling keys.
        real_control_result = _update_control_with_linear_rng(
            agent,
            state.control_state,
            reward,
            next_observation,
            real_discount,
        )
    control_after_real = real_control_result.state
    model_state = cast(ActionConditionedWorldModelState, real_model_result.state)
    behavior_model = BehaviorModel(
        BehaviorModelConfig(
            n_actions=config.n_actions,
            step_size=config.behavior_model_step_size,
        )
    )
    behavior_after_real = behavior_model.update(
        state.behavior_model_state,
        state.control_state.last_observation,
        state.control_state.last_action,
    ).state

    buffer_state = buffer.add(state.buffer_state, next_observation)

    warmup_ready = model_state.step_count >= config.dreaming_warmup_steps
    error_ok = (
        model_state.model_error_ema
        <= jnp.asarray(config.dreaming_max_model_error, dtype=jnp.float32)
    )
    dream_gate = warmup_ready & error_ok

    def dream_step(
        carry: tuple[DifferentialSARSAState, BehaviorModelState],
        _: Array,
    ) -> tuple[tuple[DifferentialSARSAState, BehaviorModelState], tuple[Array, Array]]:
        ctrl_state, behavior_state = carry
        next_master_key, candidate_key, behavior_rollout_key, control_rollout_key = (
            jr.split(ctrl_state.rng_key, 4)
        )
        candidate_keys = jr.split(candidate_key, config.dream_candidate_count)

        def candidate_step(candidate_item: tuple[Array, Array]) -> tuple[Array, ...]:
            index, cand_key = candidate_item
            del index
            anchor_key, sample_key = jr.split(cand_key)
            anchor_obs, _ = buffer.sample(buffer_state, anchor_key)
            behavior_for_sample = behavior_state.replace(
                rng_key=sample_key
            )
            behavior_sample = behavior_model.sample_action(
                behavior_for_sample,
                anchor_obs,
            )
            prediction = model.predict(model_state, anchor_obs, behavior_sample.action)
            transition_magnitude = jnp.mean(
                (prediction.next_observation - anchor_obs) ** 2
            )
            surprise = transition_magnitude + jnp.abs(prediction.reward)
            utility = jnp.abs(prediction.reward)
            return (
                anchor_obs,
                behavior_sample.action,
                behavior_sample.action_probability,
                surprise,
                utility,
                prediction.discount,
                prediction.reward,
            )

        (
            candidate_anchors,
            candidate_actions,
            candidate_probabilities,
            candidate_surprises,
            candidate_utilities,
            candidate_discounts,
            _candidate_rewards,
        ) = jax.vmap(candidate_step)(
            (
                jnp.arange(config.dream_candidate_count, dtype=jnp.int32),
                candidate_keys,
            )
        )
        selection = score_dream_candidates(
            candidate_surprises,
            candidate_utilities,
            confidences=candidate_probabilities,
            model_errors=jnp.full(
                (config.dream_candidate_count,),
                model_state.model_error_ema,
                dtype=jnp.float32,
            ),
            config=DreamSelectionConfig(
                max_items=1,
                surprise_weight=config.dream_surprise_weight,
                utility_weight=config.dream_utility_weight,
                confidence_weight=0.0,
                model_error_weight=1.0,
                max_model_error=config.dreaming_max_model_error,
            ),
        )
        selected_index = selection.selected_indices[0]
        anchor_obs = candidate_anchors[selected_index]
        action = candidate_actions[selected_index]
        initial_control_state = ctrl_state.replace(
            rng_key=control_rollout_key
        )
        initial_behavior_state = behavior_state.replace(
            rng_key=behavior_rollout_key
        )

        def rollout_step(
            rollout_carry: tuple[
                DifferentialSARSAState,
                BehaviorModelState,
                Array,
                Array,
            ],
            _: Array,
        ) -> tuple[
            tuple[DifferentialSARSAState, BehaviorModelState, Array, Array],
            tuple[Array, Array, Array],
        ]:
            rollout_ctrl, rollout_behavior, rollout_obs, rollout_action = (
                rollout_carry
            )
            prediction = model.predict(model_state, rollout_obs, rollout_action)
            temp_state = rollout_ctrl.replace(
                last_observation=rollout_obs,
                last_action=rollout_action,
            )
            dream_result = _update_control_with_linear_rng(
                agent,
                temp_state,
                prediction.reward,
                prediction.next_observation,
                prediction.discount,
            )
            dream_state = dream_result.state
            if not config.dreams_update_average_reward:
                # Planning backups improve value estimates only: restore the
                # pre-dream reward-rate estimate so imagined rewards can never
                # move rbar (see Step9DreamingConfig docstring).
                dream_state = dream_state.replace(
                    average_reward=rollout_ctrl.average_reward
                )
            next_behavior = behavior_model.sample_action(
                rollout_behavior,
                prediction.next_observation,
            )
            return (
                dream_state,
                next_behavior.state,
                prediction.next_observation,
                next_behavior.action,
            ), (
                dream_result.td_error,
                prediction.discount,
                dream_result.update_applied,
            )

        (
            (rollout_ctrl, rollout_behavior, _rollout_obs, _rollout_action),
            (rollout_td_errors, rollout_discounts, rollout_updates_applied),
        ) = jax.lax.scan(
            rollout_step,
            (initial_control_state, initial_behavior_state, anchor_obs, action),
            jnp.arange(config.dream_rollout_horizon, dtype=jnp.int32),
        )
        rollout_td_signal = jnp.sum(rollout_td_errors)
        finite = jnp.all(jnp.isfinite(rollout_td_errors)) & jnp.all(
            jnp.isfinite(rollout_discounts)
        )
        selected_discount = candidate_discounts[selected_index]
        selected_accepted = selection.accepted[selected_index]
        accepted = (
            dream_gate
            & finite
            & selected_accepted
            & (selected_discount >= 0.0)
            & jnp.all(rollout_updates_applied)
        )

        restored = rollout_ctrl.replace(
            last_observation=control_after_real.last_observation,
            last_action=control_after_real.last_action,
        )
        next_ctrl = cast(
            DifferentialSARSAState,
            jax.tree_util.tree_map(
                lambda new, old: jnp.where(accepted, new, old),
                restored,
                ctrl_state,
            ),
        )
        # The reserved master branch always survives the transaction.  This
        # keeps future dreams independent even when the gate or a rollout
        # control update rejects every learned-state change.
        next_ctrl = next_ctrl.replace(rng_key=next_master_key)
        next_behavior = cast(
            BehaviorModelState,
            jax.tree_util.tree_map(
                lambda new, old: jnp.where(accepted, new, old),
                rollout_behavior,
                behavior_state,
            ),
        )
        return (next_ctrl, next_behavior), (
            jnp.where(accepted, rollout_td_signal, jnp.array(0.0, dtype=jnp.float32)),
            accepted,
        )

    (final_ctrl, final_behavior), (dream_td_errors, dream_accepted) = jax.lax.scan(
        dream_step,
        (control_after_real, behavior_after_real),
        jnp.arange(config.planning_budget, dtype=jnp.int32),
    )

    new_state = Step9DreamingState(
        control_state=final_ctrl,
        world_model_state=model_state,
        behavior_model_state=final_behavior,
        buffer_state=buffer_state,
        step_count=_saturating_int32_counter_increment(state.step_count),
    )
    return Step9DreamingUpdateResult(
        state=new_state,
        real_control_result=real_control_result,
        real_model_result=real_model_result,
        dream_td_errors=dream_td_errors,
        dream_accepted=dream_accepted,
    )


def _has_step9_trusted_array_type(value: object) -> bool:
    actual_type = type(value)
    return (
        actual_type is np.ndarray
        or issubclass(actual_type, jax.Array)
        or issubclass(actual_type, jax.core.Tracer)
    )


def _require_step9_trusted_array(
    name: str,
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    if not _has_step9_trusted_array_type(value):
        raise TypeError(f"{name} must be a trusted array")
    try:
        actual_shape = tuple(value.shape)
        actual_dtype = jnp.dtype(value.dtype)
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(f"{name} must expose trusted shape and dtype metadata") from error
    if actual_shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if actual_dtype != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}")
    return cast(Array, value)


def run_step9_scan(
    config: Step9DreamingConfig,
    agent: DifferentialSARSAAgent,
    model: ActionConditionedWorldModel,
    buffer: RecentObservationBuffer,
    state: Step9DreamingState,
    rewards: Array,
    next_observations: Array,
) -> Step9ArrayResult:
    """Run Step 9 dreaming over real continuing transition arrays.

    Raises:
        TypeError: If ``config`` is not an actual :class:`Step9DreamingConfig`,
            or ``rewards``/``next_observations`` are not trusted arrays with
            the expected dtype.
        ValueError: If ``rewards`` is empty, exceeds the documented
            scan-length ceiling (``_STEP9_SEQUENCE_MAX_STEPS``), or
            ``next_observations`` does not share its leading length.
    """
    if type(config) is not Step9DreamingConfig:
        raise TypeError("config must be an actual Step9DreamingConfig")
    if not _has_step9_trusted_array_type(rewards):
        raise TypeError("rewards must be a trusted array")
    try:
        steps = int(rewards.shape[0])
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise TypeError("rewards must expose trusted shape metadata") from error
    if not 1 <= steps <= _STEP9_SEQUENCE_MAX_STEPS:
        raise ValueError(
            f"rewards length must be an integer in [1, {_STEP9_SEQUENCE_MAX_STEPS}]"
        )
    rewards = _require_step9_trusted_array("rewards", rewards, shape=(steps,), dtype=jnp.float32)
    next_observations = _require_step9_trusted_array(
        "next_observations",
        next_observations,
        shape=(steps, config.observation_dim),
        dtype=jnp.float32,
    )

    def scan_step(
        carry: Step9DreamingState,
        inputs: tuple[Array, Array],
    ) -> tuple[Step9DreamingState, tuple[Array, ...]]:
        reward, next_observation = inputs
        result = step9_update(config, agent, model, buffer, carry, reward, next_observation)
        return result.state, (
            result.real_control_result.td_error,
            result.real_control_result.average_reward,
            result.real_control_result.action,
            result.real_model_result.prediction_error,
            result.real_model_result.update_applied,
            result.dream_td_errors,
            result.dream_accepted,
        )

    final_state, (
        real_td_errors,
        average_rewards,
        actions,
        model_prediction_errors,
        model_updates_applied,
        dream_td_errors,
        dream_accepted,
    ) = jax.lax.scan(scan_step, state, (rewards, next_observations))
    return Step9ArrayResult(
        state=final_state,
        real_td_errors=real_td_errors,
        average_rewards=average_rewards,
        actions=actions,
        model_prediction_errors=model_prediction_errors,
        model_updates_applied=model_updates_applied,
        dream_td_errors=dream_td_errors,
        dream_accepted=dream_accepted,
    )


def run_step9_smoke(
    config: Step9DreamingConfig | None = None,
    *,
    steps: int = 32,
    seed: int = 0,
) -> Step9SmokeResult:
    """Run a tiny deterministic Step 9 dreaming integration probe."""
    steps = _require_int("steps", steps, minimum=1, maximum=_INT32_MAX)
    seed = require_jax_seed(seed, name="seed")

    if config is None:
        cfg = Step9DreamingConfig()
    elif type(config) is Step9DreamingConfig:
        cfg = config
    else:
        raise TypeError("config must be an exact Step9DreamingConfig")
    _preflight_step9_resources(cfg)
    _preflight_step9_smoke_resources(cfg, steps)
    agent, model, buffer = make_step9_components(cfg)
    data_key, state_key = jr.split(jr.key(seed))
    observations = jr.normal(
        data_key,
        (steps + 1, cfg.observation_dim),
        dtype=jnp.float32,
    )
    rewards = jnp.tanh(observations[1:, 0])

    state = init_step9_state(
        agent,
        model,
        buffer,
        key=state_key,
        initial_observation=observations[0],
    )
    result = run_step9_scan(cfg, agent, model, buffer, state, rewards, observations[1:])
    result.real_td_errors.block_until_ready()
    finite = bool(
        jnp.all(jnp.isfinite(result.real_td_errors))
        & jnp.all(jnp.isfinite(result.average_rewards))
        & jnp.all(jnp.isfinite(result.model_prediction_errors))
        & jnp.all(jnp.isfinite(result.dream_td_errors))
        & jnp.all(result.actions >= 0)
        & jnp.all(result.actions < cfg.n_actions)
    )
    return Step9SmokeResult(
        config=cfg,
        steps=steps,
        seed=seed,
        real_td_errors_shape=tuple(int(d) for d in result.real_td_errors.shape),
        dream_td_errors_shape=tuple(int(d) for d in result.dream_td_errors.shape),
        actions_shape=tuple(int(d) for d in result.actions.shape),
        finite=finite,
        dream_acceptance_count=int(jnp.sum(result.dream_accepted)),
        control_config=agent.to_config(),
        world_model_config=model.to_config(),
    )
