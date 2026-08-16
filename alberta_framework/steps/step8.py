# mypy: disable-error-code="call-arg"
"""Production-facing Step 8 one-step world-model facade.

Step 8 is the environment-prediction surface of the model-based progression:
learn a one-step model — expected reward and next observation (or observation
delta, with ``predict_delta=True``) given the current observation and action —
online, from the same stream the control learner sees.  Step 7's Dyna backups
and Step 9's guarded dreaming both consume this model.  The implementation
lives in :mod:`alberta_framework.core.world_model`. The facade rejects illegal
dimensions and decay/step-size/sparsity/leaky-ReLU scalars before constructing
that core model; accepted numbers are canonicalized to builtin ints and floats.

Network defaults follow the streaming stability recipe used across the
package (Elsayed et al. 2024, "Streaming Deep Reinforcement Learning Finally
Works"): sparse initialization (``sparsity=0.9``, the recipe's default),
LeakyReLU with the conventional 0.01 negative slope, and parameterless layer
normalization.  ``utility_decay=0.99`` sets the EMA horizon (~100 steps) of
the hidden-unit utility diagnostics.

:func:`step8_ensemble_predict` reports prediction variance across several
independently trained model states; ensemble disagreement as an epistemic
novelty signal follows Pathak et al. (2019), "Self-Supervised Exploration
via Disagreement."
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any, cast

import jax.numpy as jnp
import jax.random as jr
from jax import Array

from alberta_framework.core.world_model import (
    OneStepWorldModel,
    WorldModelConfig,
    WorldModelLearningResult,
    WorldModelState,
    WorldModelUpdateResult,
    run_world_model_learning_loop,
)
from alberta_framework.steps._float32_validation import (
    canonical_float32_storage,
    finite_real_and_float32,
)


@dataclass(frozen=True)
class Step8WorldModelConfig:
    """Config for the Step 8 one-step environment model facade."""

    observation_dim: int = 4
    n_actions: int | None = 2
    action_dim: int = 1
    hidden_sizes: tuple[int, ...] = (64,)
    step_size: float = 0.05
    sparsity: float = 0.9
    leaky_relu_slope: float = 0.01
    use_layer_norm: bool = True
    predict_delta: bool = False
    utility_decay: float = 0.99

    def __post_init__(self) -> None:
        """Reject illegal dimensions and scientific scalars, then canonicalize."""
        _validate_world_model_config(self)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["hidden_sizes"] = list(self.hidden_sizes)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Step8WorldModelConfig:
        """Reconstruct from :meth:`to_dict` output."""
        data = dict(payload)
        hidden_sizes = data.get("hidden_sizes", (64,))
        if isinstance(hidden_sizes, list):
            data["hidden_sizes"] = tuple(hidden_sizes)
        return cls(**cast(Any, data))

    def to_core_config(self) -> WorldModelConfig:
        """Return the core world-model config."""
        return WorldModelConfig(
            observation_dim=self.observation_dim,
            n_actions=self.n_actions,
            action_dim=self.action_dim,
            hidden_sizes=self.hidden_sizes,
            step_size=self.step_size,
            sparsity=self.sparsity,
            leaky_relu_slope=self.leaky_relu_slope,
            use_layer_norm=self.use_layer_norm,
            predict_delta=self.predict_delta,
            utility_decay=self.utility_decay,
        )


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
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
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
        raise ValueError(f"{name} must be in [0, 1), got {value!r}")
    return canonical_float32_storage(real, narrowed)


def _require_nonnegative_real(name: str, value: object) -> float:
    real, numerator, _, narrowed = finite_real_and_float32(name, value)
    if real < 0.0 or numerator < 0 or narrowed < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return canonical_float32_storage(real, narrowed)


def _require_int(name: str, value: object, *, minimum: int | None = None) -> int:
    actual_type = type(value)
    if issubclass(actual_type, bool) or not issubclass(actual_type, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    number = int(cast(Integral, value))
    if minimum is not None and number < minimum:
        if minimum == 1:
            raise ValueError(f"{name} must be positive, got {value!r}")
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return number


def _validate_world_model_config(config: Step8WorldModelConfig) -> None:
    observation_dim = _require_int("observation_dim", config.observation_dim, minimum=1)
    n_actions = (
        None if config.n_actions is None else _require_int("n_actions", config.n_actions, minimum=1)
    )
    action_dim = _require_int("action_dim", config.action_dim, minimum=1)
    if not isinstance(config.hidden_sizes, tuple):
        raise ValueError(f"hidden_sizes must be a tuple of integers, got {config.hidden_sizes!r}")
    hidden_sizes = tuple(
        _require_int("hidden_sizes", size, minimum=1) for size in config.hidden_sizes
    )
    step_size = _require_nonnegative_real("step_size", config.step_size)
    sparsity = _require_unit_interval("sparsity", config.sparsity)
    leaky_relu_slope = _require_nonnegative_real(
        "leaky_relu_slope",
        config.leaky_relu_slope,
    )
    utility_decay = _require_half_open_unit_interval("utility_decay", config.utility_decay)
    object.__setattr__(config, "observation_dim", observation_dim)
    object.__setattr__(config, "n_actions", n_actions)
    object.__setattr__(config, "action_dim", action_dim)
    object.__setattr__(config, "hidden_sizes", hidden_sizes)
    object.__setattr__(config, "step_size", step_size)
    object.__setattr__(config, "sparsity", sparsity)
    object.__setattr__(config, "leaky_relu_slope", leaky_relu_slope)
    object.__setattr__(config, "utility_decay", utility_decay)


@dataclass(frozen=True)
class Step8SmokeResult:
    """Summary returned by :func:`run_step8_smoke`."""

    config: Step8WorldModelConfig
    steps: int
    seed: int
    reward_predictions_shape: tuple[int, ...]
    next_observation_predictions_shape: tuple[int, ...]
    reward_errors_shape: tuple[int, ...]
    next_observation_errors_shape: tuple[int, ...]
    finite: bool
    model_config: dict[str, Any]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["config"] = self.config.to_dict()
        payload["reward_predictions_shape"] = list(self.reward_predictions_shape)
        payload["next_observation_predictions_shape"] = list(
            self.next_observation_predictions_shape
        )
        payload["reward_errors_shape"] = list(self.reward_errors_shape)
        payload["next_observation_errors_shape"] = list(self.next_observation_errors_shape)
        return payload


@dataclass(frozen=True)
class Step8EnsemblePrediction:
    """Aggregate prediction and disagreement from multiple Step 8 models."""

    reward_predictions: Array
    next_observation_predictions: Array
    mean_reward: Array
    mean_next_observation: Array
    reward_disagreement: Array
    next_observation_disagreement: Array
    total_disagreement: Array


def make_step8_world_model(
    config: Step8WorldModelConfig | None = None,
) -> OneStepWorldModel:
    """Create the production Step 8 one-step world model."""
    cfg = config or Step8WorldModelConfig()
    return OneStepWorldModel(cfg.to_core_config())


def init_step8_state(model: OneStepWorldModel, *, key: Array) -> WorldModelState:
    """Initialize Step 8 world-model state."""
    return model.init(key)


def step8_update(
    model: OneStepWorldModel,
    state: WorldModelState,
    observation: Array,
    action: Array,
    reward: Array,
    next_observation: Array,
) -> WorldModelUpdateResult:
    """Run one Step 8 model-learning transition update."""
    return cast(
        WorldModelUpdateResult,
        model.update(state, observation, action, reward, next_observation),
    )


def step8_ensemble_predict(
    model: OneStepWorldModel,
    states: Sequence[WorldModelState],
    observation: Array,
    action: Array,
) -> Step8EnsemblePrediction:
    """Predict with an ensemble of Step 8 states and return disagreement.

    The states are intentionally explicit rather than hidden in a new learner
    object. This keeps ensemble use compatible with existing checkpointing and
    lets downstream systems choose their own bootstrap or seed strategy.
    """
    if not states:
        raise ValueError("states must contain at least one world-model state")
    predictions = [model.predict(state, observation, action) for state in states]
    reward_predictions = jnp.stack([pred.reward for pred in predictions], axis=0)
    next_observation_predictions = jnp.stack(
        [pred.next_observation for pred in predictions],
        axis=0,
    )
    mean_reward = jnp.mean(reward_predictions, axis=0)
    mean_next_observation = jnp.mean(next_observation_predictions, axis=0)
    reward_disagreement = jnp.var(reward_predictions, axis=0)
    next_observation_disagreement = jnp.mean(jnp.var(next_observation_predictions, axis=0))
    total_disagreement = reward_disagreement + next_observation_disagreement
    return Step8EnsemblePrediction(
        reward_predictions=reward_predictions,
        next_observation_predictions=next_observation_predictions,
        mean_reward=mean_reward,
        mean_next_observation=mean_next_observation,
        reward_disagreement=reward_disagreement,
        next_observation_disagreement=next_observation_disagreement,
        total_disagreement=total_disagreement,
    )


def run_step8_scan(
    model: OneStepWorldModel,
    state: WorldModelState,
    observations: Array,
    actions: Array,
    rewards: Array,
    next_observations: Array,
) -> WorldModelLearningResult:
    """Run Step 8 world-model learning over transition arrays."""
    return run_world_model_learning_loop(
        model,
        state,
        observations,
        actions,
        rewards,
        next_observations,
    )


def run_step8_smoke(
    config: Step8WorldModelConfig | None = None,
    *,
    steps: int = 32,
    seed: int = 0,
) -> Step8SmokeResult:
    """Run a tiny deterministic Step 8 environment-prediction probe."""
    if steps < 1:
        raise ValueError("steps must be positive")

    cfg = config or Step8WorldModelConfig()
    if cfg.n_actions is None:
        raise ValueError("run_step8_smoke currently expects discrete actions")

    model = make_step8_world_model(cfg)
    key = jr.key(seed)
    data_key, state_key = jr.split(key)
    observations = jr.normal(data_key, (steps, cfg.observation_dim), dtype=jnp.float32)
    actions = jnp.arange(steps, dtype=jnp.int32) % cfg.n_actions
    action_sign = 2.0 * actions.astype(jnp.float32) - 1.0
    next_observations = observations.at[:, 0].add(0.1 * action_sign)
    rewards = jnp.tanh(next_observations[:, 0])
    state = init_step8_state(model, key=state_key)
    result = run_step8_scan(
        model,
        state,
        observations,
        actions,
        rewards,
        next_observations,
    )
    result.reward_errors.block_until_ready()
    finite = bool(
        jnp.all(jnp.isfinite(result.reward_predictions))
        & jnp.all(jnp.isfinite(result.next_observation_predictions))
        & jnp.all(jnp.isfinite(result.reward_errors))
        & jnp.all(jnp.isfinite(result.next_observation_errors))
    )
    return Step8SmokeResult(
        config=cfg,
        steps=steps,
        seed=seed,
        reward_predictions_shape=tuple(int(dim) for dim in result.reward_predictions.shape),
        next_observation_predictions_shape=tuple(
            int(dim) for dim in result.next_observation_predictions.shape
        ),
        reward_errors_shape=tuple(int(dim) for dim in result.reward_errors.shape),
        next_observation_errors_shape=tuple(
            int(dim) for dim in result.next_observation_errors.shape
        ),
        finite=finite,
        model_config=model.to_config(),
    )
