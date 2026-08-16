"""Production-facing Step 7 Dyna planning facade tests.

Covers the bounded planning facade on real constructors. Invalid dimension
and scientific-scalar cases are written to fail on current main (bool,
non-real, non-integral, non-finite, and out-of-domain values accepted) and
pass after the facade rejects them. Legal endpoints stay constructible.
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
from alberta_framework.steps.step7 import (
    Step7DynaArrayResult,
    Step7DynaConfig,
    Step7DynaState,
    Step7DynaUpdateResult,
    Step7SmokeResult,
    _score_planning_actions,
    init_step7_state,
    make_step7_components,
    run_step7_scan,
    run_step7_smoke,
    step7_update,
)
from alberta_framework.steps.step8 import Step8WorldModelConfig

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

OBS_DIM = 4
N_ACTIONS = 2

_INVALID_STEP7_FIELDS: tuple[tuple[str, Any], ...] = (
    ("planning_steps", True),
    ("planning_steps", False),
    ("planning_steps", -1),
    ("planning_steps", 1.5),
    ("planning_steps", "1"),
    ("planning_steps", None),
    ("planning_rollout_depth", True),
    ("planning_rollout_depth", False),
    ("planning_rollout_depth", 0),
    ("planning_rollout_depth", -1),
    ("planning_rollout_depth", 1.5),
    ("planning_rollout_depth", "1"),
    ("planning_rollout_depth", None),
    ("planning_warmup_steps", True),
    ("planning_warmup_steps", False),
    ("planning_warmup_steps", -1),
    ("planning_warmup_steps", 1.5),
    ("planning_warmup_steps", "1"),
    ("planning_warmup_steps", None),
    ("planning_memory_size", True),
    ("planning_memory_size", False),
    ("planning_memory_size", 0),
    ("planning_memory_size", -1),
    ("planning_memory_size", 1.5),
    ("planning_memory_size", "1"),
    ("planning_memory_size", None),
    ("planning_importance_ratio_clip", float("nan")),
    ("planning_importance_ratio_clip", float("inf")),
    ("planning_importance_ratio_clip", float("-inf")),
    ("planning_importance_ratio_clip", True),
    ("planning_importance_ratio_clip", False),
    ("planning_importance_ratio_clip", 0.0),
    ("planning_importance_ratio_clip", -1.0),
    ("planning_importance_ratio_clip", "10"),
    ("planning_importance_ratio_clip", None),
    ("planning_priority_propagation", float("nan")),
    ("planning_priority_propagation", float("inf")),
    ("planning_priority_propagation", float("-inf")),
    ("planning_priority_propagation", True),
    ("planning_priority_propagation", False),
    ("planning_priority_propagation", -0.1),
    ("planning_priority_propagation", "1"),
    ("planning_priority_propagation", None),
    ("planning_utility_step_size", float("nan")),
    ("planning_utility_step_size", float("inf")),
    ("planning_utility_step_size", float("-inf")),
    ("planning_utility_step_size", True),
    ("planning_utility_step_size", False),
    ("planning_utility_step_size", -0.1),
    ("planning_utility_step_size", 1.1),
    ("planning_utility_step_size", "0.2"),
    ("planning_utility_step_size", None),
)


def _cfg(
    planning_steps: int = 2,
    strategy: str = "random",
    warmup: int = 1,
) -> Step7DynaConfig:
    return Step7DynaConfig(
        control=Step6DifferentialSARSAConfig(n_actions=N_ACTIONS),
        world_model=Step8WorldModelConfig(observation_dim=OBS_DIM, n_actions=N_ACTIONS),
        planning_steps=planning_steps,
        planning_warmup_steps=warmup,
        planning_memory_size=16,
        planning_strategy=strategy,  # type: ignore[arg-type]
    )


def _init(cfg: Step7DynaConfig | None = None) -> tuple[object, object, Step7DynaState]:
    cfg = cfg or _cfg()
    agent, model = make_step7_components(cfg)
    obs0 = jnp.zeros(OBS_DIM)
    state = init_step7_state(
        agent, model, key=jr.key(0), initial_observation=obs0,
        memory_size=cfg.planning_memory_size,
    )
    return agent, model, state


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestStep7ConfigValidation:
    def test_planning_steps_non_negative(self) -> None:
        with pytest.raises(ValueError, match="planning_steps"):
            Step7DynaConfig(
                control=Step6DifferentialSARSAConfig(n_actions=N_ACTIONS),
                world_model=Step8WorldModelConfig(observation_dim=OBS_DIM, n_actions=N_ACTIONS),
                planning_steps=-1,
            )

    def test_warmup_steps_non_negative(self) -> None:
        with pytest.raises(ValueError, match="planning_warmup_steps"):
            Step7DynaConfig(
                control=Step6DifferentialSARSAConfig(n_actions=N_ACTIONS),
                world_model=Step8WorldModelConfig(observation_dim=OBS_DIM, n_actions=N_ACTIONS),
                planning_warmup_steps=-1,
            )

    def test_memory_size_positive(self) -> None:
        with pytest.raises(ValueError, match="planning_memory_size"):
            Step7DynaConfig(
                control=Step6DifferentialSARSAConfig(n_actions=N_ACTIONS),
                world_model=Step8WorldModelConfig(observation_dim=OBS_DIM, n_actions=N_ACTIONS),
                planning_memory_size=0,
            )

    def test_n_actions_must_match(self) -> None:
        with pytest.raises(ValueError, match="n_actions"):
            Step7DynaConfig(
                control=Step6DifferentialSARSAConfig(n_actions=2),
                world_model=Step8WorldModelConfig(observation_dim=OBS_DIM, n_actions=3),
            )

    def test_invalid_strategy_rejected(self) -> None:
        with pytest.raises(ValueError, match="planning_strategy"):
            Step7DynaConfig(
                control=Step6DifferentialSARSAConfig(n_actions=N_ACTIONS),
                world_model=Step8WorldModelConfig(observation_dim=OBS_DIM, n_actions=N_ACTIONS),
                planning_strategy="bogus",  # type: ignore[arg-type]
            )


def _config_with(**overrides: Any) -> Step7DynaConfig:
    payload: dict[str, Any] = {
        "control": Step6DifferentialSARSAConfig(n_actions=N_ACTIONS),
        "world_model": Step8WorldModelConfig(observation_dim=OBS_DIM, n_actions=N_ACTIONS),
    }
    payload.update(overrides)
    return Step7DynaConfig(**payload)


@pytest.mark.parametrize(("field", "value"), _INVALID_STEP7_FIELDS)
def test_step7_planning_fields_reject_invalid_inputs(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _config_with(**{field: value})


class _SpoofedInt:
    """Mimics ``int`` via ``__class__`` to defeat ``isinstance`` checks."""

    @property
    def __class__(self) -> type:  # type: ignore[override]
        return int

    def __int__(self) -> int:
        return 3

    def __index__(self) -> int:
        return 3


@pytest.mark.parametrize(
    "field",
    [
        "planning_steps",
        "planning_rollout_depth",
        "planning_warmup_steps",
        "planning_memory_size",
    ],
)
def test_step7_planning_fields_reject_class_spoofed_integers(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _config_with(**{field: _SpoofedInt()})


def test_step7_planning_fields_preserve_legal_endpoints() -> None:
    config = Step7DynaConfig(
        control=Step6DifferentialSARSAConfig(n_actions=N_ACTIONS),
        world_model=Step8WorldModelConfig(observation_dim=OBS_DIM, n_actions=N_ACTIONS),
        planning_steps=0,
        planning_rollout_depth=1,
        planning_warmup_steps=0,
        planning_memory_size=1,
        planning_importance_ratio_clip=1e-6,
        planning_priority_propagation=0.0,
        planning_utility_step_size=0.0,
    )
    payload = config.to_dict()
    json.dumps(payload, allow_nan=False)
    restored = Step7DynaConfig.from_dict(payload)
    assert restored.planning_steps == 0
    assert restored.planning_rollout_depth == 1
    assert restored.planning_warmup_steps == 0
    assert restored.planning_memory_size == 1
    assert restored.planning_importance_ratio_clip == 1e-6
    assert restored.planning_priority_propagation == 0.0
    assert restored.planning_utility_step_size == 0.0
    assert payload["planning_priority_propagation"] == 0.0
    assert payload["planning_utility_step_size"] == 0.0

    upper = Step7DynaConfig(
        control=Step6DifferentialSARSAConfig(n_actions=N_ACTIONS),
        world_model=Step8WorldModelConfig(observation_dim=OBS_DIM, n_actions=N_ACTIONS),
        planning_steps=3,
        planning_rollout_depth=2,
        planning_warmup_steps=4,
        planning_memory_size=8,
        planning_importance_ratio_clip=10.0,
        planning_priority_propagation=1.0,
        planning_utility_step_size=1.0,
    )
    make_step7_components(upper)
    assert upper.planning_importance_ratio_clip == 10.0
    assert upper.planning_priority_propagation == 1.0
    assert upper.planning_utility_step_size == 1.0

    smoke = run_step7_smoke(config, steps=4, seed=0)
    assert smoke.finite
    assert smoke.planning_td_errors_shape == (4, 0)


def test_step7_planning_fields_canonicalize_nonbuiltin_numbers() -> None:
    value = np.float64(0.5)
    config = Step7DynaConfig(
        control=Step6DifferentialSARSAConfig(n_actions=N_ACTIONS),
        world_model=Step8WorldModelConfig(observation_dim=OBS_DIM, n_actions=N_ACTIONS),
        planning_steps=np.int64(2),
        planning_rollout_depth=np.int64(3),
        planning_warmup_steps=np.int64(4),
        planning_memory_size=np.int64(8),
        planning_importance_ratio_clip=np.float64(5.0),
        planning_priority_propagation=value,
        planning_utility_step_size=value,
    )
    payload = config.to_dict()
    json.dumps(payload, allow_nan=False)
    assert config.planning_steps == 2
    assert config.planning_rollout_depth == 3
    assert config.planning_warmup_steps == 4
    assert config.planning_memory_size == 8
    assert config.planning_importance_ratio_clip == 5.0
    assert config.planning_priority_propagation == 0.5
    assert config.planning_utility_step_size == 0.5
    assert type(payload["planning_steps"]) is int
    assert type(payload["planning_rollout_depth"]) is int
    assert type(payload["planning_warmup_steps"]) is int
    assert type(payload["planning_memory_size"]) is int
    assert type(payload["planning_importance_ratio_clip"]) is float
    assert type(payload["planning_priority_propagation"]) is float
    assert type(payload["planning_utility_step_size"]) is float
    restored = Step7DynaConfig.from_dict(payload)
    assert restored.planning_steps == 2
    assert restored.planning_utility_step_size == 0.5


# ---------------------------------------------------------------------------
# Config roundtrip
# ---------------------------------------------------------------------------


class TestStep7ConfigRoundtrip:
    def test_to_dict_from_dict(self) -> None:
        cfg = _cfg(planning_steps=3, strategy="reward")
        restored = Step7DynaConfig.from_dict(cfg.to_dict())
        assert restored.planning_steps == 3
        assert restored.planning_strategy == "reward"
        assert restored.planning_memory_size == 16
        assert restored.control.n_actions == N_ACTIONS
        assert restored.world_model.observation_dim == OBS_DIM


# ---------------------------------------------------------------------------
# Factory and init
# ---------------------------------------------------------------------------


class TestStep7InitState:
    def test_make_components(self) -> None:
        agent, model = make_step7_components(_cfg())
        assert agent is not None
        assert model is not None

    def test_init_state_shapes(self) -> None:
        cfg = _cfg()
        agent, model, state = _init(cfg)
        assert isinstance(state, Step7DynaState)
        # memory_size is taken from cfg.planning_memory_size = 16
        chex.assert_shape(state.memory_observations, (cfg.planning_memory_size, OBS_DIM))
        chex.assert_shape(state.memory_actions, (cfg.planning_memory_size,))
        chex.assert_shape(state.memory_rewards, (cfg.planning_memory_size,))
        chex.assert_shape(state.memory_priorities, (cfg.planning_memory_size,))
        chex.assert_shape(state.memory_utilities, (cfg.planning_memory_size,))
        assert int(state.memory_count) == 0
        assert int(state.step_count) == 0

    def test_control_state_primed(self) -> None:
        _, _, state = _init()
        chex.assert_shape(state.control_state.last_observation, (OBS_DIM,))


# ---------------------------------------------------------------------------
# Single-step update
# ---------------------------------------------------------------------------


class TestStep7Update:
    def test_update_returns_result(self) -> None:
        cfg = _cfg()
        agent, model, state = _init(cfg)
        result = step7_update(cfg, agent, model, state, jnp.array(1.0), jnp.ones(OBS_DIM))
        assert isinstance(result, Step7DynaUpdateResult)

    def test_update_step_count_increments(self) -> None:
        cfg = _cfg()
        agent, model, state = _init(cfg)
        result = step7_update(cfg, agent, model, state, jnp.array(0.0), jnp.zeros(OBS_DIM))
        assert int(result.state.step_count) == 1

    def test_update_real_td_error_finite(self) -> None:
        cfg = _cfg()
        agent, model, state = _init(cfg)
        result = step7_update(cfg, agent, model, state, jnp.array(1.0), jnp.ones(OBS_DIM))
        assert jnp.isfinite(result.real_control_result.td_error)

    def test_update_planning_td_errors_shape(self) -> None:
        cfg = _cfg(planning_steps=3)
        agent, model, state = _init(cfg)
        result = step7_update(cfg, agent, model, state, jnp.array(0.0), jnp.zeros(OBS_DIM))
        chex.assert_shape(result.planning_td_errors, (3,))

    def test_update_memory_fills(self) -> None:
        cfg = _cfg()
        agent, model, state = _init(cfg)
        result = step7_update(cfg, agent, model, state, jnp.array(1.0), jnp.ones(OBS_DIM))
        assert int(result.state.memory_count) == 1

    def test_update_model_step_count(self) -> None:
        cfg = _cfg()
        agent, model, state = _init(cfg)
        result = step7_update(cfg, agent, model, state, jnp.array(1.0), jnp.ones(OBS_DIM))
        assert int(result.state.world_model_state.step_count) == 1

    def test_planning_gated_before_warmup(self) -> None:
        cfg = _cfg(warmup=100, planning_steps=2)
        agent, model, state = _init(cfg)
        result = step7_update(cfg, agent, model, state, jnp.array(0.0), jnp.zeros(OBS_DIM))
        chex.assert_trees_all_close(result.planning_td_errors, jnp.zeros(2), atol=1e-6)

    def test_planning_anchor_indices_valid(self) -> None:
        cfg = _cfg(warmup=1, planning_steps=2)
        agent, model, state = _init(cfg)
        # Warm up the model first
        for _ in range(5):
            result = step7_update(cfg, agent, model, state, jnp.array(0.0), jnp.zeros(OBS_DIM))
            state = result.state
        chex.assert_shape(result.planning_anchor_indices, (2,))


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


class TestStep7Scan:
    def test_scan_shapes(self) -> None:
        cfg = _cfg(planning_steps=2)
        agent, model, state = _init(cfg)
        n_steps = 10
        rewards = jnp.zeros(n_steps)
        next_obs = jnp.zeros((n_steps, OBS_DIM))
        result = run_step7_scan(cfg, agent, model, state, rewards, next_obs)
        assert isinstance(result, Step7DynaArrayResult)
        chex.assert_shape(result.real_td_errors, (n_steps,))
        chex.assert_shape(result.average_rewards, (n_steps,))
        chex.assert_shape(result.actions, (n_steps,))
        chex.assert_shape(result.model_updates_applied, (n_steps,))
        assert bool(jnp.all(result.model_updates_applied))
        chex.assert_shape(result.planning_td_errors, (n_steps, 2))

    def test_scan_exposes_rejected_model_updates(self) -> None:
        cfg = _cfg(planning_steps=0)
        agent, model, state = _init(cfg)
        maximum = jnp.asarray(2_147_483_647, dtype=jnp.int32)
        model_state = state.world_model_state.replace(
            learner_state=state.world_model_state.learner_state.replace(
                step_count=maximum,
                step_words=jnp.asarray([0xFFFFFFFF, 0xFFFFFFFF], dtype=jnp.uint32),
            ),
            step_count=maximum,
        )
        state = state.replace(world_model_state=model_state)

        result = run_step7_scan(
            cfg,
            agent,
            model,
            state,
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.zeros((1, OBS_DIM), dtype=jnp.float32),
        )

        chex.assert_trees_all_equal(
            result.model_updates_applied,
            jnp.asarray([False]),
        )

    def test_scan_td_errors_finite(self) -> None:
        cfg = _cfg(planning_steps=1)
        agent, model, state = _init(cfg)
        n_steps = 8
        rewards = jr.normal(jr.key(10), (n_steps,))
        next_obs = jr.normal(jr.key(11), (n_steps, OBS_DIM))
        result = run_step7_scan(cfg, agent, model, state, rewards, next_obs)
        chex.assert_tree_all_finite(result.real_td_errors)

    def test_scan_actions_valid(self) -> None:
        cfg = _cfg()
        agent, model, state = _init(cfg)
        n_steps = 5
        obs = jnp.zeros((n_steps, OBS_DIM))
        result = run_step7_scan(cfg, agent, model, state, jnp.zeros(n_steps), obs)
        assert jnp.all(result.actions >= 0)
        assert jnp.all(result.actions < N_ACTIONS)

    def test_scan_step_count_final(self) -> None:
        cfg = _cfg()
        agent, model, state = _init(cfg)
        n_steps = 7
        obs = jnp.zeros((n_steps, OBS_DIM))
        result = run_step7_scan(cfg, agent, model, state, jnp.zeros(n_steps), obs)
        assert int(result.state.step_count) == n_steps

    @pytest.mark.parametrize("strategy", ["random", "reward", "surprise", "predecessor"])
    def test_scan_strategies(self, strategy: str) -> None:
        cfg = _cfg(strategy=strategy, planning_steps=1)
        agent, model, state = _init(cfg)
        n_steps = 6
        obs = jnp.zeros((n_steps, OBS_DIM))
        result = run_step7_scan(cfg, agent, model, state, jnp.zeros(n_steps), obs)
        chex.assert_tree_all_finite(result.real_td_errors)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


class TestStep7Smoke:
    def test_default_smoke_passes(self) -> None:
        result = run_step7_smoke(steps=16, seed=0)
        assert isinstance(result, Step7SmokeResult)
        assert result.finite
        assert result.steps == 16

    def test_smoke_shapes(self) -> None:
        result = run_step7_smoke(steps=8, seed=42)
        assert result.real_td_errors_shape == (8,)

    def test_smoke_with_custom_config(self) -> None:
        cfg = _cfg(planning_steps=4, strategy="surprise", warmup=2)
        result = run_step7_smoke(cfg, steps=20, seed=1)
        assert result.finite
        assert result.planning_td_errors_shape == (20, 4)

    def test_smoke_steps_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="steps"):
            run_step7_smoke(steps=0)

    def test_smoke_planning_acceptance_count_type(self) -> None:
        result = run_step7_smoke(steps=10, seed=0)
        assert isinstance(result.planning_acceptance_count, int)

    def test_smoke_reward_strategy(self) -> None:
        cfg = _cfg(planning_steps=2, strategy="reward")
        result = run_step7_smoke(cfg, steps=16, seed=0)
        assert result.finite

    def test_smoke_predecessor_strategy(self) -> None:
        cfg = _cfg(planning_steps=2, strategy="predecessor")
        result = run_step7_smoke(cfg, steps=16, seed=0)
        assert result.finite


# ---------------------------------------------------------------------------
# 200-step fineness
# ---------------------------------------------------------------------------


class TestStep7Fineness:
    def test_200_step_random_strategy(self) -> None:
        cfg = _cfg(planning_steps=2, strategy="random", warmup=5)
        agent, model, state = _init(cfg)
        n_steps = 200
        rewards = jr.normal(jr.key(50), (n_steps,))
        next_obs = jr.normal(jr.key(51), (n_steps, OBS_DIM))
        result = run_step7_scan(cfg, agent, model, state, rewards, next_obs)
        chex.assert_tree_all_finite(result.real_td_errors)
        assert int(result.state.step_count) == n_steps

    def test_200_step_with_nonzero_planning_accepted(self) -> None:
        cfg = _cfg(planning_steps=2, strategy="random", warmup=1)
        agent, model, state = _init(cfg)
        n_steps = 200
        rewards = jr.normal(jr.key(60), (n_steps,))
        next_obs = jr.normal(jr.key(61), (n_steps, OBS_DIM))
        result = run_step7_scan(cfg, agent, model, state, rewards, next_obs)
        # After warmup, planning should be accepted for most steps
        acceptance = jnp.sum(result.planning_accepted)
        # At least some steps should have accepted planning
        assert int(acceptance) > 0


# ---------------------------------------------------------------------------
# Search-control action scoring
# ---------------------------------------------------------------------------


class TestScorePlanningActions:
    """The returned score is exactly the selected action's priority."""

    def _trained_model_state(self, model):  # type: ignore[no-untyped-def]
        state = model.init(jr.key(7))
        obs = jr.normal(jr.key(8), (OBS_DIM,), dtype=jnp.float32)
        next_obs = jr.normal(jr.key(9), (OBS_DIM,), dtype=jnp.float32)
        result = model.update(
            state, obs, jnp.array(1, dtype=jnp.int32), jnp.array(0.7), next_obs
        )
        return result.state

    def test_reward_strategy_score_is_selected_abs_reward(self) -> None:
        cfg = _cfg()
        _, model = make_step7_components(cfg)
        model_state = self._trained_model_state(model)
        anchor = jr.normal(jr.key(10), (OBS_DIM,), dtype=jnp.float32)

        selected, score = _score_planning_actions(
            model, model_state, anchor, "reward", N_ACTIONS
        )

        rewards = jnp.array(
            [
                jnp.abs(
                    model.predict(
                        model_state, anchor, jnp.array(a, dtype=jnp.int32)
                    ).reward
                )
                for a in range(N_ACTIONS)
            ]
        )
        assert int(selected) == int(jnp.argmax(rewards))
        assert float(score) == pytest.approx(float(rewards[int(selected)]))

    def test_surprise_strategy_score_adds_transition_magnitude(self) -> None:
        cfg = _cfg()
        _, model = make_step7_components(cfg)
        model_state = self._trained_model_state(model)
        anchor = jr.normal(jr.key(11), (OBS_DIM,), dtype=jnp.float32)

        selected, score = _score_planning_actions(
            model, model_state, anchor, "surprise", N_ACTIONS
        )

        prediction = model.predict(
            model_state, anchor, jnp.asarray(selected, dtype=jnp.int32)
        )
        expected = jnp.abs(prediction.reward) + jnp.sqrt(
            jnp.mean((prediction.next_observation - anchor) ** 2)
        )
        assert float(score) == pytest.approx(float(expected), rel=1e-6)
