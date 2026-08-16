"""Tests for the Step 9 guarded-dreaming facade.

Invalid dimension and scientific-scalar cases are written to fail on current
main (bool, non-real, non-integral, non-finite, and out-of-domain values
accepted) and pass after the facade rejects them. Legal endpoints stay
constructible.
"""

from __future__ import annotations

import json
from typing import Any

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.steps.step6 import Step6DifferentialSARSAConfig
from alberta_framework.steps.step9 import (
    Step9DreamingConfig,
    Step9DreamingState,
    init_step9_state,
    make_step9_components,
    run_step9_scan,
    run_step9_smoke,
    step9_update,
)

_INVALID_STEP9_FIELDS: tuple[tuple[str, Any], ...] = (
    ("observation_dim", True),
    ("observation_dim", False),
    ("observation_dim", 0),
    ("observation_dim", -1),
    ("observation_dim", 1.5),
    ("observation_dim", "4"),
    ("observation_dim", None),
    ("n_actions", True),
    ("n_actions", False),
    ("n_actions", 0),
    ("n_actions", -1),
    ("n_actions", 1.5),
    ("n_actions", "2"),
    ("n_actions", None),
    ("model_hidden_sizes", (True,)),
    ("model_hidden_sizes", (False,)),
    ("model_hidden_sizes", (0,)),
    ("model_hidden_sizes", (-1,)),
    ("model_hidden_sizes", (1.5,)),
    ("model_hidden_sizes", ("64",)),
    ("model_hidden_sizes", [64]),
    ("model_hidden_sizes", 64),
    ("model_step_size", float("nan")),
    ("model_step_size", float("inf")),
    ("model_step_size", float("-inf")),
    ("model_step_size", True),
    ("model_step_size", False),
    ("model_step_size", -1.0),
    ("model_step_size", "0.03"),
    ("model_step_size", None),
    ("model_sparsity", float("nan")),
    ("model_sparsity", float("inf")),
    ("model_sparsity", float("-inf")),
    ("model_sparsity", True),
    ("model_sparsity", False),
    ("model_sparsity", -0.1),
    ("model_sparsity", 1.1),
    ("model_sparsity", "0.9"),
    ("model_sparsity", None),
    ("model_gamma", float("nan")),
    ("model_gamma", float("inf")),
    ("model_gamma", float("-inf")),
    ("model_gamma", True),
    ("model_gamma", False),
    ("model_gamma", -0.1),
    ("model_gamma", 1.1),
    ("model_gamma", "0.99"),
    ("model_gamma", None),
    ("model_error_decay", float("nan")),
    ("model_error_decay", float("inf")),
    ("model_error_decay", float("-inf")),
    ("model_error_decay", True),
    ("model_error_decay", False),
    ("model_error_decay", -0.1),
    ("model_error_decay", 1.0),
    ("model_error_decay", "0.99"),
    ("model_error_decay", None),
    ("dreaming_warmup_steps", True),
    ("dreaming_warmup_steps", False),
    ("dreaming_warmup_steps", -1),
    ("dreaming_warmup_steps", 1.5),
    ("dreaming_warmup_steps", "100"),
    ("dreaming_warmup_steps", None),
    ("dreaming_max_model_error", float("nan")),
    ("dreaming_max_model_error", float("inf")),
    ("dreaming_max_model_error", float("-inf")),
    ("dreaming_max_model_error", True),
    ("dreaming_max_model_error", False),
    ("dreaming_max_model_error", -0.1),
    ("dreaming_max_model_error", "1.0"),
    ("dreaming_max_model_error", None),
    ("behavior_model_step_size", float("nan")),
    ("behavior_model_step_size", float("inf")),
    ("behavior_model_step_size", float("-inf")),
    ("behavior_model_step_size", True),
    ("behavior_model_step_size", False),
    ("behavior_model_step_size", -0.1),
    ("behavior_model_step_size", "0.05"),
    ("behavior_model_step_size", None),
    ("planning_budget", True),
    ("planning_budget", False),
    ("planning_budget", -1),
    ("planning_budget", 1.5),
    ("planning_budget", "1"),
    ("planning_budget", None),
    ("buffer_capacity", True),
    ("buffer_capacity", False),
    ("buffer_capacity", 0),
    ("buffer_capacity", -1),
    ("buffer_capacity", 1.5),
    ("buffer_capacity", "64"),
    ("buffer_capacity", None),
    ("dream_rollout_horizon", True),
    ("dream_rollout_horizon", False),
    ("dream_rollout_horizon", 0),
    ("dream_rollout_horizon", -1),
    ("dream_rollout_horizon", 1.5),
    ("dream_rollout_horizon", "1"),
    ("dream_rollout_horizon", None),
    ("dream_candidate_count", True),
    ("dream_candidate_count", False),
    ("dream_candidate_count", 0),
    ("dream_candidate_count", -1),
    ("dream_candidate_count", 1.5),
    ("dream_candidate_count", "1"),
    ("dream_candidate_count", None),
    ("dream_surprise_weight", float("nan")),
    ("dream_surprise_weight", float("inf")),
    ("dream_surprise_weight", float("-inf")),
    ("dream_surprise_weight", True),
    ("dream_surprise_weight", False),
    ("dream_surprise_weight", "1.0"),
    ("dream_surprise_weight", None),
    ("dream_utility_weight", float("nan")),
    ("dream_utility_weight", float("inf")),
    ("dream_utility_weight", float("-inf")),
    ("dream_utility_weight", True),
    ("dream_utility_weight", False),
    ("dream_utility_weight", "1.0"),
    ("dream_utility_weight", None),
    ("model_include_action_interactions", 0),
    ("model_include_action_interactions", 1),
    ("model_include_action_interactions", "true"),
    ("model_include_action_interactions", None),
    ("model_use_layer_norm", 0),
    ("model_use_layer_norm", 1),
    ("model_use_layer_norm", "true"),
    ("model_use_layer_norm", None),
    ("dreams_update_average_reward", 0),
    ("dreams_update_average_reward", 1),
    ("dreams_update_average_reward", "true"),
    ("dreams_update_average_reward", None),
)

# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_step9_config_roundtrip() -> None:
    cfg = Step9DreamingConfig(
        control=Step6DifferentialSARSAConfig(n_actions=3),
        observation_dim=3,
        n_actions=3,
        model_hidden_sizes=(32,),
        model_step_size=0.05,
        model_sparsity=0.0,
        planning_budget=2,
        behavior_model_step_size=0.02,
        dream_rollout_horizon=3,
        dream_candidate_count=4,
        dream_surprise_weight=0.5,
        dream_utility_weight=1.5,
        buffer_capacity=16,
    )
    assert Step9DreamingConfig.from_dict(cfg.to_dict()) == cfg


def test_step9_config_n_actions_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="n_actions"):
        Step9DreamingConfig(
            control=Step6DifferentialSARSAConfig(n_actions=2),
            n_actions=3,
        )


def test_step9_config_negative_planning_budget_raises() -> None:
    with pytest.raises(ValueError, match="planning_budget"):
        Step9DreamingConfig(planning_budget=-1)


def test_step9_config_negative_warmup_raises() -> None:
    with pytest.raises(ValueError, match="warmup"):
        Step9DreamingConfig(dreaming_warmup_steps=-1)


def test_step9_config_negative_max_error_raises() -> None:
    with pytest.raises(ValueError, match="max_model_error"):
        Step9DreamingConfig(dreaming_max_model_error=-0.1)


def test_step9_config_zero_buffer_capacity_raises() -> None:
    with pytest.raises(ValueError, match="buffer_capacity"):
        Step9DreamingConfig(buffer_capacity=0)


def test_step9_config_negative_behavior_step_size_raises() -> None:
    with pytest.raises(ValueError, match="behavior_model_step_size"):
        Step9DreamingConfig(behavior_model_step_size=-0.1)


def test_step9_config_zero_dream_rollout_horizon_raises() -> None:
    with pytest.raises(ValueError, match="dream_rollout_horizon"):
        Step9DreamingConfig(dream_rollout_horizon=0)


def test_step9_config_zero_dream_candidate_count_raises() -> None:
    with pytest.raises(ValueError, match="dream_candidate_count"):
        Step9DreamingConfig(dream_candidate_count=0)


def _config_with(**overrides: Any) -> Step9DreamingConfig:
    n_actions = overrides.get("n_actions", 2)
    payload: dict[str, Any] = {
        "control": Step6DifferentialSARSAConfig(n_actions=n_actions),
        "observation_dim": 2,
        "n_actions": 2,
        "model_hidden_sizes": (),
    }
    payload.update(overrides)
    if "n_actions" in overrides and "control" not in overrides:
        payload["control"] = Step6DifferentialSARSAConfig(n_actions=n_actions)
    return Step9DreamingConfig(**payload)


@pytest.mark.parametrize(("field", "value"), _INVALID_STEP9_FIELDS)
def test_step9_dreaming_fields_reject_invalid_inputs(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _config_with(**{field: value})


def test_step9_dreaming_fields_preserve_legal_endpoints() -> None:
    config = Step9DreamingConfig(
        control=Step6DifferentialSARSAConfig(n_actions=1),
        observation_dim=1,
        n_actions=1,
        model_hidden_sizes=(),
        model_step_size=0.0,
        model_sparsity=0.0,
        model_include_action_interactions=False,
        model_use_layer_norm=False,
        model_gamma=0.0,
        dreaming_warmup_steps=0,
        dreaming_max_model_error=0.0,
        model_error_decay=0.0,
        behavior_model_step_size=0.0,
        planning_budget=0,
        dream_rollout_horizon=1,
        dream_candidate_count=1,
        dream_surprise_weight=-2.5,
        dream_utility_weight=0.0,
        buffer_capacity=1,
        dreams_update_average_reward=False,
    )
    make_step9_components(config)
    payload = config.to_dict()
    json.dumps(payload, allow_nan=False)
    restored = Step9DreamingConfig.from_dict(payload)
    assert restored.observation_dim == 1
    assert restored.n_actions == 1
    assert restored.model_hidden_sizes == ()
    assert restored.model_step_size == 0.0
    assert restored.model_sparsity == 0.0
    assert restored.model_include_action_interactions is False
    assert restored.model_use_layer_norm is False
    assert restored.model_gamma == 0.0
    assert restored.dreaming_warmup_steps == 0
    assert restored.dreaming_max_model_error == 0.0
    assert restored.model_error_decay == 0.0
    assert restored.behavior_model_step_size == 0.0
    assert restored.planning_budget == 0
    assert restored.dream_rollout_horizon == 1
    assert restored.dream_candidate_count == 1
    assert restored.dream_surprise_weight == -2.5
    assert restored.dream_utility_weight == 0.0
    assert restored.buffer_capacity == 1
    assert restored.dreams_update_average_reward is False
    assert payload["model_sparsity"] == 0.0
    assert payload["model_error_decay"] == 0.0
    assert payload["dream_surprise_weight"] == -2.5

    upper = Step9DreamingConfig(
        control=Step6DifferentialSARSAConfig(n_actions=2),
        observation_dim=1,
        n_actions=2,
        model_hidden_sizes=(),
        model_sparsity=1.0,
        model_include_action_interactions=True,
        model_use_layer_norm=True,
        model_gamma=1.0,
        dreaming_max_model_error=1e30,
        model_error_decay=0.999,
        dream_surprise_weight=2.0,
        dream_utility_weight=-0.5,
        dreams_update_average_reward=True,
    )
    make_step9_components(upper)
    json.dumps(upper.to_dict(), allow_nan=False)
    assert upper.model_sparsity == 1.0
    assert upper.model_gamma == 1.0
    assert upper.dreaming_max_model_error == 1e30
    assert upper.model_include_action_interactions is True
    assert upper.dreams_update_average_reward is True

    defaults = Step9DreamingConfig()
    make_step9_components(defaults)
    assert defaults.observation_dim == 4
    assert defaults.n_actions == 2
    assert defaults.model_hidden_sizes == (64,)
    assert defaults.model_step_size == 0.03
    assert defaults.model_sparsity == 0.9
    assert defaults.model_gamma == 0.99
    assert defaults.model_error_decay == 0.99


def test_step9_dreaming_fields_canonicalize_nonbuiltin_numbers() -> None:
    value = np.float64(0.5)
    config = Step9DreamingConfig(
        control=Step6DifferentialSARSAConfig(n_actions=2),
        observation_dim=np.int64(3),
        n_actions=np.int64(2),
        model_hidden_sizes=(np.int64(4),),
        model_step_size=value,
        model_sparsity=value,
        model_gamma=value,
        dreaming_warmup_steps=np.int64(0),
        dreaming_max_model_error=value,
        model_error_decay=value,
        behavior_model_step_size=value,
        planning_budget=np.int64(2),
        dream_rollout_horizon=np.int64(3),
        dream_candidate_count=np.int64(4),
        dream_surprise_weight=np.float64(-1.5),
        dream_utility_weight=value,
        buffer_capacity=np.int64(8),
    )
    make_step9_components(config)
    payload = config.to_dict()
    json.dumps(payload, allow_nan=False)
    assert config.observation_dim == 3
    assert config.n_actions == 2
    assert config.model_hidden_sizes == (4,)
    assert config.model_step_size == 0.5
    assert config.model_sparsity == 0.5
    assert config.model_gamma == 0.5
    assert config.dreaming_warmup_steps == 0
    assert config.dreaming_max_model_error == 0.5
    assert config.model_error_decay == 0.5
    assert config.behavior_model_step_size == 0.5
    assert config.planning_budget == 2
    assert config.dream_rollout_horizon == 3
    assert config.dream_candidate_count == 4
    assert config.dream_surprise_weight == -1.5
    assert config.dream_utility_weight == 0.5
    assert config.buffer_capacity == 8
    assert type(payload["observation_dim"]) is int
    assert type(payload["n_actions"]) is int
    assert type(payload["model_hidden_sizes"][0]) is int
    assert type(payload["model_step_size"]) is float
    assert type(payload["model_sparsity"]) is float
    assert type(payload["model_gamma"]) is float
    assert type(payload["dreaming_warmup_steps"]) is int
    assert type(payload["dreaming_max_model_error"]) is float
    assert type(payload["model_error_decay"]) is float
    assert type(payload["behavior_model_step_size"]) is float
    assert type(payload["planning_budget"]) is int
    assert type(payload["dream_rollout_horizon"]) is int
    assert type(payload["dream_candidate_count"]) is int
    assert type(payload["dream_surprise_weight"]) is float
    assert type(payload["dream_utility_weight"]) is float
    assert type(payload["buffer_capacity"]) is int
    restored = Step9DreamingConfig.from_dict(payload)
    assert restored.observation_dim == 3
    assert restored.model_hidden_sizes == (4,)
    assert restored.dream_surprise_weight == -1.5
    assert restored.model_error_decay == 0.5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dreaming_warmup_steps", 2**31),
        ("dream_candidate_count", 2**31),
        ("buffer_capacity", 2**31 - 1),
    ],
)
def test_step9_count_fields_reject_values_outside_int32_contract(
    field: str,
    value: int,
) -> None:
    with pytest.raises(ValueError, match=field):
        _config_with(**{field: value})


def test_step9_count_fields_preserve_int32_upper_endpoints() -> None:
    config = _config_with(
        dreaming_warmup_steps=2**31 - 1,
        dream_candidate_count=2**31 - 1,
        buffer_capacity=2**31 - 2,
    )
    assert config.dreaming_warmup_steps == 2**31 - 1
    assert config.dream_candidate_count == 2**31 - 1
    assert config.buffer_capacity == 2**31 - 2
    json.dumps(config.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "field",
    [
        "observation_dim",
        "n_actions",
        "model_hidden_sizes",
        "planning_budget",
        "dream_rollout_horizon",
    ],
)
def test_step9_remaining_fields_enforce_int32_upper_bound(field: str) -> None:
    value: object = (2**31,) if field == "model_hidden_sizes" else 2**31
    with pytest.raises(ValueError, match=field):
        _config_with(**{field: value})


# ---------------------------------------------------------------------------
# Factory and init tests
# ---------------------------------------------------------------------------


def test_step9_make_components_returns_correct_types() -> None:
    cfg = Step9DreamingConfig(
        observation_dim=2,
        n_actions=2,
        model_hidden_sizes=(),
        model_sparsity=0.0,
    )
    agent, model, buffer = make_step9_components(cfg)
    assert model.config.observation_dim == 2
    assert model.config.n_actions == 2
    assert buffer.capacity == cfg.buffer_capacity


def test_step9_init_state_fields() -> None:
    cfg = Step9DreamingConfig(
        observation_dim=2,
        n_actions=2,
        model_hidden_sizes=(),
        model_sparsity=0.0,
    )
    agent, model, buffer = make_step9_components(cfg)
    initial_obs = jnp.array([0.5, -0.5], dtype=jnp.float32)
    state = init_step9_state(agent, model, buffer, key=jr.key(0), initial_observation=initial_obs)
    assert isinstance(state, Step9DreamingState)
    assert int(state.step_count) == 0
    assert int(state.world_model_state.step_count) == 0
    assert int(state.behavior_model_state.step_count) == 0
    assert int(state.buffer_state.size) >= 1


# ---------------------------------------------------------------------------
# Single-step update tests
# ---------------------------------------------------------------------------


def test_step9_single_update_increments_counters() -> None:
    cfg = Step9DreamingConfig(
        observation_dim=2,
        n_actions=2,
        model_hidden_sizes=(),
        model_sparsity=0.0,
        planning_budget=2,
        dreaming_warmup_steps=0,
        dreaming_max_model_error=1e30,
    )
    agent, model, buffer = make_step9_components(cfg)
    state = init_step9_state(
        agent, model, buffer,
        key=jr.key(1),
        initial_observation=jnp.zeros(2),
    )
    result = step9_update(
        cfg, agent, model, buffer,
        state,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([0.1, 0.2], dtype=jnp.float32),
    )
    assert int(result.state.step_count) == 1
    assert int(result.state.world_model_state.step_count) == 1
    assert int(result.state.behavior_model_state.step_count) == 1
    chex.assert_shape(result.dream_td_errors, (2,))
    chex.assert_shape(result.dream_accepted, (2,))


def test_step9_dreams_rejected_before_warmup() -> None:
    """With warmup=100 and only 1 step, no dreams should be accepted."""
    cfg = Step9DreamingConfig(
        observation_dim=2,
        n_actions=2,
        model_hidden_sizes=(),
        model_sparsity=0.0,
        planning_budget=4,
        dreaming_warmup_steps=100,
        dreaming_max_model_error=1e30,
    )
    agent, model, buffer = make_step9_components(cfg)
    state = init_step9_state(
        agent, model, buffer,
        key=jr.key(2),
        initial_observation=jnp.zeros(2),
    )
    result = step9_update(
        cfg, agent, model, buffer,
        state,
        jnp.array(0.0, dtype=jnp.float32),
        jnp.zeros(2, dtype=jnp.float32),
    )
    assert not bool(jnp.any(result.dream_accepted)), "Dreams should be rejected before warmup"


def test_step9_dreams_accepted_with_zero_warmup_and_high_error_threshold() -> None:
    """With warmup=0 and a very high error threshold, dreams should be accepted."""
    cfg = Step9DreamingConfig(
        observation_dim=2,
        n_actions=2,
        model_hidden_sizes=(),
        model_sparsity=0.0,
        planning_budget=2,
        dreaming_warmup_steps=0,
        dreaming_max_model_error=1e30,
    )
    agent, model, buffer = make_step9_components(cfg)
    state = init_step9_state(
        agent, model, buffer,
        key=jr.key(3),
        initial_observation=jnp.zeros(2),
    )
    result = step9_update(
        cfg, agent, model, buffer,
        state,
        jnp.array(0.5, dtype=jnp.float32),
        jnp.array([0.3, -0.3], dtype=jnp.float32),
    )
    assert bool(jnp.any(result.dream_accepted)), "At least one dream should be accepted"


def test_step9_dreams_rejected_when_error_too_high() -> None:
    """With a very strict error threshold, dreams should be rejected."""
    cfg = Step9DreamingConfig(
        observation_dim=2,
        n_actions=2,
        model_hidden_sizes=(),
        model_sparsity=0.0,
        planning_budget=3,
        dreaming_warmup_steps=0,
        dreaming_max_model_error=0.0,  # impossible threshold
    )
    agent, model, buffer = make_step9_components(cfg)
    state = init_step9_state(
        agent, model, buffer,
        key=jr.key(4),
        initial_observation=jnp.zeros(2),
    )
    # run a few updates so model_error_ema becomes nonzero
    obs = jnp.array([1.0, 2.0], dtype=jnp.float32)
    result = step9_update(cfg, agent, model, buffer, state,
                          jnp.array(1.0), obs)
    assert not bool(jnp.any(result.dream_accepted)), (
        "Dreams should be rejected when error exceeds threshold"
    )


def test_step9_multi_step_behavior_model_dreaming_path() -> None:
    cfg = Step9DreamingConfig(
        observation_dim=2,
        n_actions=2,
        model_hidden_sizes=(),
        model_sparsity=0.0,
        planning_budget=2,
        dreaming_warmup_steps=0,
        dreaming_max_model_error=1e30,
        behavior_model_step_size=0.1,
        dream_rollout_horizon=3,
        dream_candidate_count=3,
    )
    agent, model, buffer = make_step9_components(cfg)
    state = init_step9_state(
        agent,
        model,
        buffer,
        key=jr.key(41),
        initial_observation=jnp.zeros(2),
    )
    result = step9_update(
        cfg,
        agent,
        model,
        buffer,
        state,
        jnp.array(0.25, dtype=jnp.float32),
        jnp.array([0.2, -0.1], dtype=jnp.float32),
    )
    chex.assert_shape(result.dream_td_errors, (2,))
    chex.assert_shape(result.dream_accepted, (2,))
    chex.assert_tree_all_finite(result.dream_td_errors)
    assert int(result.state.behavior_model_state.step_count) == 1
    assert bool(jnp.any(result.dream_accepted))


def test_step9_prioritized_candidate_selection_path() -> None:
    cfg = Step9DreamingConfig(
        observation_dim=2,
        n_actions=2,
        model_hidden_sizes=(),
        model_sparsity=0.0,
        planning_budget=1,
        dreaming_warmup_steps=0,
        dreaming_max_model_error=1e30,
        dream_candidate_count=5,
        dream_surprise_weight=2.0,
        dream_utility_weight=0.5,
    )
    agent, model, buffer = make_step9_components(cfg)
    state = init_step9_state(
        agent,
        model,
        buffer,
        key=jr.key(42),
        initial_observation=jnp.array([0.1, -0.1], dtype=jnp.float32),
    )
    result = step9_update(
        cfg,
        agent,
        model,
        buffer,
        state,
        jnp.array(0.5, dtype=jnp.float32),
        jnp.array([0.3, 0.4], dtype=jnp.float32),
    )
    chex.assert_shape(result.dream_td_errors, (1,))
    chex.assert_shape(result.dream_accepted, (1,))
    chex.assert_tree_all_finite(result.dream_td_errors)
    chex.assert_tree_all_finite(result.real_control_result.td_error)
    chex.assert_tree_all_finite(result.real_model_result.prediction_error)
    assert bool(result.dream_accepted[0])


# ---------------------------------------------------------------------------
# Zero planning budget
# ---------------------------------------------------------------------------


def test_step9_zero_planning_budget() -> None:
    cfg = Step9DreamingConfig(
        observation_dim=2,
        n_actions=2,
        model_hidden_sizes=(),
        model_sparsity=0.0,
        planning_budget=0,
    )
    agent, model, buffer = make_step9_components(cfg)
    state = init_step9_state(
        agent, model, buffer,
        key=jr.key(5),
        initial_observation=jnp.zeros(2),
    )
    result = step9_update(
        cfg, agent, model, buffer,
        state,
        jnp.array(0.0, dtype=jnp.float32),
        jnp.zeros(2, dtype=jnp.float32),
    )
    chex.assert_shape(result.dream_td_errors, (0,))
    chex.assert_shape(result.dream_accepted, (0,))
    assert int(result.state.step_count) == 1


# ---------------------------------------------------------------------------
# Scan tests
# ---------------------------------------------------------------------------


def test_step9_scan_shapes() -> None:
    steps = 8
    cfg = Step9DreamingConfig(
        observation_dim=3,
        n_actions=2,
        model_hidden_sizes=(),
        model_sparsity=0.0,
        planning_budget=2,
        dreaming_warmup_steps=0,
        dreaming_max_model_error=1e30,
    )
    agent, model, buffer = make_step9_components(cfg)
    observations = jr.normal(jr.key(6), (steps + 1, 3), dtype=jnp.float32)
    rewards = jnp.tanh(observations[1:, 0])
    state = init_step9_state(agent, model, buffer, key=jr.key(7),
                              initial_observation=observations[0])
    result = run_step9_scan(cfg, agent, model, buffer, state, rewards, observations[1:])
    chex.assert_shape(result.real_td_errors, (steps,))
    chex.assert_shape(result.average_rewards, (steps,))
    chex.assert_shape(result.actions, (steps,))
    chex.assert_shape(result.model_prediction_errors, (steps,))
    chex.assert_shape(result.model_updates_applied, (steps,))
    assert bool(jnp.all(result.model_updates_applied))
    chex.assert_shape(result.dream_td_errors, (steps, 2))
    chex.assert_shape(result.dream_accepted, (steps, 2))
    chex.assert_tree_all_finite(result.real_td_errors)
    chex.assert_tree_all_finite(result.average_rewards)
    chex.assert_tree_all_finite(result.model_prediction_errors)


def test_step9_scan_exposes_rejected_model_updates() -> None:
    cfg = Step9DreamingConfig(
        observation_dim=3,
        n_actions=2,
        model_hidden_sizes=(),
        model_sparsity=0.0,
        planning_budget=0,
    )
    agent, model, buffer = make_step9_components(cfg)
    state = init_step9_state(
        agent,
        model,
        buffer,
        key=jr.key(70),
        initial_observation=jnp.zeros((3,), dtype=jnp.float32),
    )
    maximum = jnp.asarray(2_147_483_647, dtype=jnp.int32)
    model_state = state.world_model_state.replace(
        learner_state=state.world_model_state.learner_state.replace(
            step_count=maximum,
            step_words=jnp.asarray([0xFFFFFFFF, 0xFFFFFFFF], dtype=jnp.uint32),
        ),
        step_count=maximum,
    )
    state = state.replace(world_model_state=model_state)

    result = run_step9_scan(
        cfg,
        agent,
        model,
        buffer,
        state,
        jnp.zeros((1,), dtype=jnp.float32),
        jnp.zeros((1, 3), dtype=jnp.float32),
    )

    chex.assert_trees_all_equal(
        result.model_updates_applied,
        jnp.asarray([False]),
    )


def test_step9_scan_actions_in_range() -> None:
    steps = 16
    cfg = Step9DreamingConfig(
        observation_dim=4,
        n_actions=3,
        model_hidden_sizes=(),
        model_sparsity=0.0,
        planning_budget=1,
        control=Step6DifferentialSARSAConfig(n_actions=3),
    )
    agent, model, buffer = make_step9_components(cfg)
    observations = jr.normal(jr.key(8), (steps + 1, 4))
    rewards = jnp.tanh(observations[1:, 0])
    state = init_step9_state(agent, model, buffer, key=jr.key(9),
                              initial_observation=observations[0])
    result = run_step9_scan(cfg, agent, model, buffer, state, rewards, observations[1:])
    assert bool(jnp.all(result.actions >= 0))
    assert bool(jnp.all(result.actions < 3))


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_step9_smoke_defaults() -> None:
    result = run_step9_smoke(steps=16, seed=0)
    assert result.finite
    assert result.steps == 16
    assert result.real_td_errors_shape == (16,)
    assert result.dream_td_errors_shape == (16, 1)


def test_step9_smoke_config_roundtrip() -> None:
    cfg = Step9DreamingConfig(
        observation_dim=3,
        n_actions=2,
        model_hidden_sizes=(),
        model_sparsity=0.0,
        control=Step6DifferentialSARSAConfig(n_actions=2),
        planning_budget=3,
        dreaming_warmup_steps=4,
        dreaming_max_model_error=1e30,
    )
    result = run_step9_smoke(cfg, steps=8, seed=42)
    assert result.finite
    assert result.dream_td_errors_shape == (8, 3)


def test_step9_smoke_zero_steps_raises() -> None:
    with pytest.raises(ValueError, match="steps must be positive"):
        run_step9_smoke(steps=0)


@pytest.mark.parametrize("steps", [True, 1.5])
def test_step9_smoke_rejects_non_integer_steps(steps: object) -> None:
    with pytest.raises(ValueError, match="steps must be an integer"):
        run_step9_smoke(steps=steps)  # type: ignore[arg-type]


def test_step9_smoke_rejects_class_spoofed_integer_steps() -> None:
    class _SpoofedInt:
        """Mimics ``int`` via ``__class__`` to defeat ``isinstance`` checks."""

        @property
        def __class__(self) -> type:  # type: ignore[override]
            return int

        def __int__(self) -> int:
            return 3

        def __index__(self) -> int:
            return 3

    with pytest.raises(ValueError, match="steps must be an integer"):
        run_step9_smoke(steps=_SpoofedInt())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"steps": 2**31}, "steps"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**31}, "seed"),
        ({"seed": True}, "seed"),
    ],
)
def test_step9_smoke_enforces_int32_inputs(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        run_step9_smoke(**kwargs)  # type: ignore[arg-type]


def test_step9_smoke_linear_model() -> None:
    """Linear (hidden_sizes=()) world model should work identically."""
    cfg = Step9DreamingConfig(
        observation_dim=2,
        n_actions=2,
        model_hidden_sizes=(),
        model_sparsity=0.0,
        planning_budget=2,
        dreaming_warmup_steps=0,
        dreaming_max_model_error=1e30,
    )
    result = run_step9_smoke(cfg, steps=8, seed=1)
    assert result.finite


def test_step9_smoke_larger_config() -> None:
    cfg = Step9DreamingConfig(
        observation_dim=6,
        n_actions=4,
        model_hidden_sizes=(32, 32),
        model_step_size=0.01,
        model_sparsity=0.5,
        planning_budget=2,
        buffer_capacity=32,
        dreaming_warmup_steps=2,
        dreaming_max_model_error=1e30,
        control=Step6DifferentialSARSAConfig(
            n_actions=4,
            q_step_size=0.03,
            average_reward_step_size=0.005,
        ),
    )
    result = run_step9_smoke(cfg, steps=16, seed=99)
    assert result.finite


# ---------------------------------------------------------------------------
# Buffer interaction tests
# ---------------------------------------------------------------------------


def test_step9_buffer_grows_after_updates() -> None:
    cfg = Step9DreamingConfig(
        observation_dim=2,
        n_actions=2,
        model_hidden_sizes=(),
        model_sparsity=0.0,
        planning_budget=0,
        buffer_capacity=8,
    )
    agent, model, buffer = make_step9_components(cfg)
    state = init_step9_state(
        agent, model, buffer,
        key=jr.key(10),
        initial_observation=jnp.zeros(2),
    )
    initial_size = int(state.buffer_state.size)
    for i in range(4):
        result = step9_update(
            cfg, agent, model, buffer, state,
            jnp.array(float(i)),
            jr.normal(jr.key(i + 100), (2,)),
        )
        state = result.state
    assert int(state.buffer_state.size) > initial_size


# ---------------------------------------------------------------------------
# State finitenesss across many steps
# ---------------------------------------------------------------------------


def test_step9_state_stays_finite_over_many_steps() -> None:
    cfg = Step9DreamingConfig(
        observation_dim=2,
        n_actions=2,
        model_hidden_sizes=(),
        model_sparsity=0.0,
        planning_budget=2,
        dreaming_warmup_steps=0,
        dreaming_max_model_error=1e30,
    )
    result = run_step9_smoke(cfg, steps=128, seed=7)
    assert result.finite
