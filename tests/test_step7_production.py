"""Production-facing Step 7 Dyna planning facade tests.

Covers the bounded planning facade on real constructors. Invalid dimension
and scientific-scalar cases are written to fail on current main (bool,
non-real, non-integral, non-finite, and out-of-domain values accepted) and
pass after the facade rejects them. Legal endpoints stay constructible.
"""

from __future__ import annotations

import json
from fractions import Fraction
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
    ("planning_steps", 2**31),
    ("planning_rollout_depth", True),
    ("planning_rollout_depth", False),
    ("planning_rollout_depth", 0),
    ("planning_rollout_depth", -1),
    ("planning_rollout_depth", 1.5),
    ("planning_rollout_depth", "1"),
    ("planning_rollout_depth", None),
    ("planning_rollout_depth", 2**31),
    ("planning_warmup_steps", True),
    ("planning_warmup_steps", False),
    ("planning_warmup_steps", -1),
    ("planning_warmup_steps", 1.5),
    ("planning_warmup_steps", "1"),
    ("planning_warmup_steps", None),
    ("planning_warmup_steps", 2**31),
    ("planning_memory_size", True),
    ("planning_memory_size", False),
    ("planning_memory_size", 0),
    ("planning_memory_size", -1),
    ("planning_memory_size", 1.5),
    ("planning_memory_size", "1"),
    ("planning_memory_size", None),
    ("planning_memory_size", 2**31),
    ("planning_importance_ratio_clip", float("nan")),
    ("planning_importance_ratio_clip", float("inf")),
    ("planning_importance_ratio_clip", float("-inf")),
    ("planning_importance_ratio_clip", True),
    ("planning_importance_ratio_clip", False),
    ("planning_importance_ratio_clip", 0.0),
    ("planning_importance_ratio_clip", -1.0),
    ("planning_importance_ratio_clip", "10"),
    ("planning_importance_ratio_clip", None),
    ("planning_importance_ratio_clip", 1e100),
    ("planning_importance_ratio_clip", 1e-50),
    ("planning_priority_propagation", float("nan")),
    ("planning_priority_propagation", float("inf")),
    ("planning_priority_propagation", float("-inf")),
    ("planning_priority_propagation", True),
    ("planning_priority_propagation", False),
    ("planning_priority_propagation", -0.1),
    ("planning_priority_propagation", "1"),
    ("planning_priority_propagation", None),
    ("planning_priority_propagation", 1e100),
    ("planning_utility_step_size", float("nan")),
    ("planning_utility_step_size", float("inf")),
    ("planning_utility_step_size", float("-inf")),
    ("planning_utility_step_size", True),
    ("planning_utility_step_size", False),
    ("planning_utility_step_size", -0.1),
    ("planning_utility_step_size", 1.1),
    ("planning_utility_step_size", "0.2"),
    ("planning_utility_step_size", None),
    ("planning_apply_importance_correction", 1),
    ("planning_apply_importance_correction", 0),
    ("planning_apply_importance_correction", 1.0),
    ("planning_apply_importance_correction", "yes"),
    ("planning_apply_importance_correction", ""),
    ("planning_apply_importance_correction", None),
    ("planning_utility_step_size", 1e100),
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


@pytest.mark.unit
@pytest.mark.parametrize("flag", [True, False])
def test_step7_importance_correction_accepts_exact_bool(flag: bool) -> None:
    config = _config_with(planning_apply_importance_correction=flag)
    assert config.planning_apply_importance_correction is flag
    payload = config.to_dict()
    assert type(payload["planning_apply_importance_correction"]) is bool
    restored = Step7DynaConfig.from_dict(payload)
    assert restored.planning_apply_importance_correction is flag


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


def test_step7_dyna_rejects_float32_underflow_for_positive_fields() -> None:
    with pytest.raises(ValueError, match="planning_importance_ratio_clip"):
        Step7DynaConfig(planning_importance_ratio_clip=1e-50)
    with pytest.raises(ValueError, match="planning_importance_ratio_clip"):
        Step7DynaConfig(planning_importance_ratio_clip=1e-46)


def test_step7_dyna_rejects_float32_overflow() -> None:
    with pytest.raises(ValueError, match="planning_importance_ratio_clip"):
        Step7DynaConfig(planning_importance_ratio_clip=1e100)
    with pytest.raises(ValueError, match="planning_priority_propagation"):
        Step7DynaConfig(planning_priority_propagation=1e100)
    with pytest.raises(ValueError, match="planning_utility_step_size"):
        Step7DynaConfig(planning_utility_step_size=1e100)


def test_step7_dyna_preserves_float32_boundaries() -> None:
    f32_max = float(np.finfo(np.float32).max)
    config = Step7DynaConfig(
        planning_steps=1,
        planning_rollout_depth=1,
        planning_warmup_steps=2**31 - 1,
        planning_memory_size=8,
        planning_importance_ratio_clip=f32_max,
        planning_priority_propagation=f32_max,
        planning_utility_step_size=1.0,
    )
    assert config.planning_steps == 1
    assert config.planning_rollout_depth == 1
    assert config.planning_warmup_steps == 2**31 - 1
    assert config.planning_memory_size == 8
    assert config.planning_importance_ratio_clip == f32_max
    assert config.planning_priority_propagation == f32_max
    assert config.planning_utility_step_size == 1.0

    with pytest.raises(ValueError, match="planning_memory_size"):
        Step7DynaConfig(planning_memory_size=2**31 - 1)


def test_step7_dyna_rejects_derived_work_and_memory_resources() -> None:
    with pytest.raises(ValueError, match="derived planning evaluations"):
        Step7DynaConfig(planning_steps=2**30, planning_rollout_depth=2)
    with pytest.raises(ValueError, match="planning output bytes"):
        Step7DynaConfig(planning_steps=2**26, planning_rollout_depth=1)
    with pytest.raises(ValueError, match="planning-memory bytes"):
        Step7DynaConfig(planning_memory_size=2**28)


def test_step7_dyna_exact_fraction_rounding() -> None:
    midpoint = Fraction(1, 1) + Fraction(1, 2**24) + Fraction(1, 2**60)
    config = Step7DynaConfig(
        planning_importance_ratio_clip=midpoint,
        planning_priority_propagation=midpoint,
        planning_utility_step_size=Fraction(1, 4),
    )
    expected_f32 = float(np.nextafter(np.float32(1.0), np.float32(2.0)))
    assert config.planning_importance_ratio_clip == expected_f32
    assert config.planning_priority_propagation == expected_f32
    assert config.planning_utility_step_size == 0.25


def test_step7_dyna_rejects_equality_spoofed_strategy() -> None:
    class SpoofedStrategy:
        def __eq__(self, other: object) -> bool:
            return True

        def __hash__(self) -> int:
            return hash("random")

    with pytest.raises(ValueError, match="planning_strategy"):
        Step7DynaConfig(planning_strategy=SpoofedStrategy())  # type: ignore[arg-type]


def test_step7_dyna_rejects_non_bool_importance_correction() -> None:
    class SpoofedBool:
        @property
        def __class__(self) -> type[bool]:
            return bool

        def __bool__(self) -> bool:
            return True

    with pytest.raises(ValueError, match="planning_apply_importance_correction"):
        Step7DynaConfig(planning_apply_importance_correction=SpoofedBool())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="planning_apply_importance_correction"):
        Step7DynaConfig(planning_apply_importance_correction=1)  # type: ignore[arg-type]


def test_step7_dyna_rejects_spoofed_int_class_and_adversarial_ratios() -> None:
    class SpoofedIntFloat(float):
        @property
        def __class__(self) -> type[int]:
            return int

        def as_integer_ratio(self) -> tuple[int, int]:
            return (-1, 2**200)

    with pytest.raises(ValueError, match="planning_importance_ratio_clip"):
        Step7DynaConfig(planning_importance_ratio_clip=SpoofedIntFloat(0.5))

    with pytest.raises(ValueError, match="planning_priority_propagation"):
        Step7DynaConfig(planning_priority_propagation=SpoofedIntFloat(0.5))

    with pytest.raises(ValueError, match="planning_utility_step_size"):
        Step7DynaConfig(planning_utility_step_size=SpoofedIntFloat(0.5))


def test_step7_dyna_json_roundtrip() -> None:
    config = Step7DynaConfig(
        planning_steps=5,
        planning_rollout_depth=2,
        planning_warmup_steps=10,
        planning_memory_size=128,
        planning_strategy="prioritized",
        planning_importance_ratio_clip=5.0,
        planning_apply_importance_correction=False,
        planning_priority_propagation=0.8,
        planning_utility_step_size=0.1,
    )
    serialized = config.to_dict()
    json_str = json.dumps(serialized)
    deserialized = json.loads(json_str)
    restored = Step7DynaConfig.from_dict(deserialized)

    assert restored.planning_steps == config.planning_steps
    assert restored.planning_rollout_depth == config.planning_rollout_depth
    assert restored.planning_warmup_steps == config.planning_warmup_steps
    assert restored.planning_memory_size == config.planning_memory_size
    assert restored.planning_strategy == config.planning_strategy
    assert restored.planning_importance_ratio_clip == config.planning_importance_ratio_clip
    assert (
        restored.planning_apply_importance_correction
        == config.planning_apply_importance_correction
    )
    assert restored.planning_priority_propagation == config.planning_priority_propagation
    assert restored.planning_utility_step_size == config.planning_utility_step_size


def test_step7_planning_accepted_tracks_the_core_learner_rollback() -> None:
    """A planning backup the core learner rolled back must not be reported as accepted."""
    from alberta_framework.steps.step6 import Step6DifferentialSARSAConfig
    from alberta_framework.steps.step8 import Step8WorldModelConfig

    # A deliberately divergent linear world model (LMS step 2.0) makes the core
    # learner reject many imagined transitions on a natural run, without any
    # state surgery.
    cfg = Step7DynaConfig(
        control=Step6DifferentialSARSAConfig(n_actions=2, q_step_size=0.05),
        world_model=Step8WorldModelConfig(
            observation_dim=3, n_actions=2, hidden_sizes=(), step_size=2.0, use_layer_norm=False
        ),
        planning_steps=1,
        planning_rollout_depth=1,
        planning_warmup_steps=0,
        planning_memory_size=8,
        planning_strategy="random",
    )
    agent, model = make_step7_components(cfg)
    steps = 32
    data_key, state_key = jr.split(jr.key(0))
    observations = jr.normal(data_key, (steps + 1, 3), dtype=jnp.float32)
    rewards = jnp.tanh(observations[1:, 0])
    state = init_step7_state(
        agent, model, key=state_key, initial_observation=observations[0], memory_size=8
    )
    reported: list[bool] = []
    replayed: list[bool] = []
    for t in range(steps):
        result = step7_update(cfg, agent, model, state, rewards[t], observations[t + 1])
        control = result.real_control_result.state
        # Replay the single imagined transaction through the core agent.
        key, action_key = jr.split(control.rng_key)
        action = jr.randint(action_key, (), 0, 2).astype(jnp.int32)
        key, anchor_key = jr.split(key)
        anchor_index = int(jr.randint(anchor_key, (), 0, int(result.state.memory_count)))
        anchor = result.state.memory_observations[anchor_index]
        prediction = model.predict(result.real_model_result.state, anchor, action)
        temp = control.replace(last_observation=anchor, last_action=action, rng_key=key)
        next_action, next_key = agent.select_action(temp, prediction.next_observation)
        core = agent.update(
            temp.replace(rng_key=next_key),
            prediction.reward,
            prediction.next_observation,
            next_action=next_action,
        )
        reported.append(bool(result.planning_accepted[0]))
        replayed.append(bool(core.update_applied))
        state = result.state
    assert not all(replayed), "the divergent model must produce at least one rollback"
    assert reported == replayed
    smoke = run_step7_smoke(cfg, steps=steps, seed=0)
    assert smoke.planning_acceptance_count == sum(replayed)


def _legal_step7_smoke_result(**overrides: object) -> Step7SmokeResult:
    payload: dict[str, object] = {
        "config": Step7DynaConfig(),
        "steps": 8,
        "seed": 0,
        "real_td_errors_shape": (8,),
        "planning_td_errors_shape": (8, 1),
        "planning_priorities_shape": (8, 1),
        "planning_anchor_indices_shape": (8, 1),
        "planning_importance_ratios_shape": (8, 1),
        "actions_shape": (8,),
        "finite": True,
        "planning_acceptance_count": 0,
        "control_config": {"ok": True},
        "world_model_config": {"ok": True},
    }
    payload.update(overrides)
    return Step7SmokeResult(**payload)  # type: ignore[arg-type]


def test_step7_smoke_result_rejects_leftover_identities() -> None:
    """Public Step 7 smoke records must not keep leftover bool/int identities."""

    with pytest.raises(ValueError, match="steps"):
        _legal_step7_smoke_result(steps=True)
    with pytest.raises(ValueError, match="steps"):
        _legal_step7_smoke_result(steps=float("nan"))
    with pytest.raises(ValueError, match="seed"):
        _legal_step7_smoke_result(seed=True)
    with pytest.raises(ValueError, match="finite"):
        _legal_step7_smoke_result(finite=1)
    with pytest.raises(ValueError, match="planning_acceptance_count"):
        _legal_step7_smoke_result(planning_acceptance_count=True)

    legal = _legal_step7_smoke_result()
    dumped = json.dumps(
        {
            "steps": legal.steps,
            "seed": legal.seed,
            "finite": legal.finite,
            "planning_acceptance_count": legal.planning_acceptance_count,
        },
        allow_nan=False,
    )
    assert '"steps": 8' in dumped
    assert '"seed": 0' in dumped
    assert '"finite": true' in dumped
    assert '"planning_acceptance_count": 0' in dumped
    assert '"steps": true' not in dumped
    assert '"seed": true' not in dumped
    assert '"finite": 1' not in dumped
    assert '"planning_acceptance_count": true' not in dumped
