"""Production facade tests for Alberta Plan Steps 5, 6, and 7."""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.steps import (
    Step5AverageRewardTDConfig,
    Step6DifferentialSARSAConfig,
    Step7DynaConfig,
    Step8WorldModelConfig,
    init_step6_state,
    init_step7_state,
    make_step5_td_learner,
    make_step6_differential_sarsa_agent,
    make_step7_components,
    run_step5_smoke,
    run_step6_smoke,
    run_step7_scan,
    run_step7_smoke,
    step6_update,
    step7_update,
)
from alberta_framework.steps import step5 as step5_module
from alberta_framework.steps import step6 as step6_module
from alberta_framework.steps.step7 import (
    _apply_planning_importance_correction,
    _pop_prioritized_planning_anchor,
    _propagate_predecessor_priorities,
    _select_planning_anchor,
    _update_planning_utility,
)


def test_step5_facade_config_roundtrip_and_smoke() -> None:
    config = Step5AverageRewardTDConfig(
        step_size=0.03,
        average_reward_step_size=0.02,
        trace_decay=0.25,
    )
    learner = make_step5_td_learner(config)
    restored = Step5AverageRewardTDConfig.from_dict(config.to_dict())
    result = run_step5_smoke(config, steps=12, feature_dim=3)

    assert restored == config
    assert config.to_dict() == {
        "step_size": 0.03,
        "average_reward_step_size": 0.02,
        "trace_decay": 0.25,
    }
    assert learner.config.trace_decay == 0.25
    assert result.finite
    assert result.predictions_shape == (12,)
    assert result.td_errors_shape == (12,)
    assert result.average_rewards_shape == (12,)
    assert result.learner_config["type"] == "DifferentialTDLearner"


@pytest.mark.parametrize(
    "field",
    ["step_size", "average_reward_step_size", "trace_decay"],
)
@pytest.mark.parametrize(
    "malformed",
    [
        True,
        False,
        "0.1",
        None,
        0.1 + 0.0j,
        float("nan"),
        float("inf"),
        float("-inf"),
        3.5e38,
        -3.5e38,
    ],
)
def test_step5_config_rejects_malformed_scientific_scalars(
    field: str,
    malformed: object,
) -> None:
    payload: dict[str, object] = {
        "step_size": 0.05,
        "average_reward_step_size": 0.01,
        "trace_decay": 0.0,
    }
    payload[field] = malformed

    with pytest.raises(ValueError, match=field):
        Step5AverageRewardTDConfig(**payload)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=field):
        Step5AverageRewardTDConfig.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("step_size", -0.01),
        ("average_reward_step_size", -0.01),
        ("trace_decay", -0.01),
        ("trace_decay", 1.01),
        ("step_size", Fraction(-1, 10**400)),
        ("average_reward_step_size", Fraction(-1, 10**400)),
        ("trace_decay", Fraction(-1, 10**400)),
        ("trace_decay", Fraction(10**400 + 1, 10**400)),
    ],
)
def test_step5_config_enforces_scientific_scalar_domains(field: str, value: object) -> None:
    payload: dict[str, object] = {
        "step_size": 0.05,
        "average_reward_step_size": 0.01,
        "trace_decay": 0.0,
    }
    payload[field] = value

    with pytest.raises(ValueError, match=field):
        Step5AverageRewardTDConfig(**payload)  # type: ignore[arg-type]


def test_step5_config_accepts_finite_float32_boundary_and_domain_endpoints() -> None:
    float32_max = 3.4028234663852886e38
    config = Step5AverageRewardTDConfig(
        step_size=float32_max,
        average_reward_step_size=0,
        trace_decay=1,
    )

    assert config.to_dict() == {
        "step_size": float32_max,
        "average_reward_step_size": 0,
        "trace_decay": 1,
    }


def test_step5_config_uses_direct_float32_narrowing_at_overflow_boundary() -> None:
    overflow_midpoint = np.ldexp(
        np.longdouble(2) - np.ldexp(np.longdouble(1), -24),
        127,
    )
    largest_finite_input = np.nextafter(
        overflow_midpoint,
        np.longdouble("-inf"),
    )

    config = Step5AverageRewardTDConfig(step_size=largest_finite_input)
    learner = make_step5_td_learner(config)

    assert bool(np.isfinite(np.asarray(config.step_size, dtype=np.float32)))
    assert learner.config.step_size == config.step_size


@pytest.mark.parametrize(
    "value",
    [
        Fraction(1, 4),
        np.float32(0.25),
        np.longdouble("0.25"),
        np.int64(1),
    ],
)
def test_step5_config_canonicalizes_accepted_real_scalars_for_json(value: object) -> None:
    config = Step5AverageRewardTDConfig(
        step_size=value,  # type: ignore[arg-type]
        average_reward_step_size=value,  # type: ignore[arg-type]
        trace_decay=value,  # type: ignore[arg-type]
    )

    payload = config.to_dict()
    assert all(type(payload[field]) is float for field in payload)
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
    restored = Step5AverageRewardTDConfig.from_dict(json.loads(encoded))
    assert restored == config


@pytest.mark.parametrize(
    "payload",
    [
        {"step_size": 0.05, "average_reward_step_size": 0.01},
        {
            "step_size": 0.05,
            "average_reward_step_size": 0.01,
            "trace_decay": 0.0,
            "unexpected": 1,
        },
    ],
)
def test_step5_config_from_dict_requires_exact_keys(payload: dict[str, object]) -> None:
    expected = (
        "Step5AverageRewardTDConfig payload keys must be exactly "
        "['average_reward_step_size', 'step_size', 'trace_decay']"
    )

    with pytest.raises(ValueError) as exc_info:
        Step5AverageRewardTDConfig.from_dict(payload)

    assert str(exc_info.value) == expected


def test_step5_smoke_health_gate_reports_any_refused_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_run = step5_module.run_differential_td_from_arrays

    def _refuse_updates(*args: Any, **kwargs: Any) -> Any:
        result = original_run(*args, **kwargs)
        return result.replace(  # type: ignore[attr-defined]
            updates_applied=result.updates_applied.at[0].set(False)
        )

    monkeypatch.setattr(
        step5_module,
        "run_differential_td_from_arrays",
        _refuse_updates,
    )

    result = run_step5_smoke(steps=3, feature_dim=2)

    assert not result.finite


def test_step6_facade_config_roundtrip_one_step_and_smoke() -> None:
    config = Step6DifferentialSARSAConfig(
        n_actions=2,
        q_step_size=0.02,
        average_reward_step_size=0.01,
        epsilon_start=0.0,
    )
    agent = make_step6_differential_sarsa_agent(config)
    restored = Step6DifferentialSARSAConfig.from_dict(config.to_dict())
    state = init_step6_state(
        agent,
        feature_dim=2,
        key=jr.key(0),
        initial_features=jnp.array([1.0, 0.0], dtype=jnp.float32),
    )
    one_step = step6_update(
        agent,
        state,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([0.0, 1.0], dtype=jnp.float32),
    )
    smoke = run_step6_smoke(config, steps=12, feature_dim=3)

    assert restored == config
    assert int(one_step.state.step_count) == 1
    assert smoke.finite
    assert smoke.q_values_shape == (12, 2)
    assert smoke.td_errors_shape == (12,)
    assert smoke.average_rewards_shape == (12,)
    assert smoke.actions_shape == (12,)
    assert smoke.agent_config["type"] == "DifferentialSARSAAgent"


_INVALID_STEP6_FIELDS: tuple[tuple[str, Any], ...] = (
    ("n_actions", 0),
    ("n_actions", -1),
    ("n_actions", True),
    ("n_actions", False),
    ("n_actions", "2"),
    ("n_actions", 2.5),
    ("n_actions", float("nan")),
    ("n_actions", float("inf")),
    ("n_actions", None),
    ("n_actions", 2**31),
    ("q_step_size", float("nan")),
    ("q_step_size", float("inf")),
    ("q_step_size", float("-inf")),
    ("q_step_size", True),
    ("q_step_size", False),
    ("q_step_size", -1.0),
    ("q_step_size", 3.5e38),
    ("q_step_size", "0.05"),
    ("q_step_size", None),
    ("average_reward_step_size", float("nan")),
    ("average_reward_step_size", float("inf")),
    ("average_reward_step_size", True),
    ("average_reward_step_size", False),
    ("average_reward_step_size", -0.01),
    ("average_reward_step_size", 3.5e38),
    ("average_reward_step_size", "0.01"),
    ("average_reward_step_size", None),
    ("trace_decay", float("nan")),
    ("trace_decay", float("inf")),
    ("trace_decay", True),
    ("trace_decay", False),
    ("trace_decay", -0.1),
    ("trace_decay", 1.1),
    ("trace_decay", "0.0"),
    ("trace_decay", None),
    ("epsilon_start", float("nan")),
    ("epsilon_start", float("inf")),
    ("epsilon_start", True),
    ("epsilon_start", False),
    ("epsilon_start", -0.1),
    ("epsilon_start", 1.1),
    ("epsilon_start", "0.1"),
    ("epsilon_start", None),
    ("epsilon_end", float("nan")),
    ("epsilon_end", float("inf")),
    ("epsilon_end", True),
    ("epsilon_end", False),
    ("epsilon_end", -0.1),
    ("epsilon_end", 1.1),
    ("epsilon_end", "0.01"),
    ("epsilon_end", None),
    ("epsilon_decay_steps", -1),
    ("epsilon_decay_steps", 2**31),
    ("epsilon_decay_steps", True),
    ("epsilon_decay_steps", False),
    ("epsilon_decay_steps", "0"),
    ("epsilon_decay_steps", 1.5),
    ("epsilon_decay_steps", float("nan")),
    ("epsilon_decay_steps", float("inf")),
    ("epsilon_decay_steps", None),
)


@pytest.mark.parametrize(("field", "value"), _INVALID_STEP6_FIELDS)
def test_step6_fields_reject_invalid_inputs(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        Step6DifferentialSARSAConfig(**{field: value})


def test_step6_fields_preserve_legal_endpoints() -> None:
    config = Step6DifferentialSARSAConfig(
        n_actions=1,
        q_step_size=0.0,
        average_reward_step_size=0.0,
        trace_decay=0.0,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_steps=0,
    )
    agent = make_step6_differential_sarsa_agent(config)
    payload = config.to_dict()
    json.dumps(payload, allow_nan=False)
    restored = Step6DifferentialSARSAConfig.from_dict(payload)
    assert restored.n_actions == 1
    assert restored.q_step_size == 0.0
    assert restored.average_reward_step_size == 0.0
    assert restored.trace_decay == 0.0
    assert restored.epsilon_start == 0.0
    assert restored.epsilon_end == 0.0
    assert restored.epsilon_decay_steps == 0
    assert agent.config.n_actions == 1

    upper = Step6DifferentialSARSAConfig(
        n_actions=10,
        q_step_size=1.0,
        average_reward_step_size=1.0,
        trace_decay=1.0,
        epsilon_start=1.0,
        epsilon_end=1.0,
        epsilon_decay_steps=2**31 - 1,
    )
    make_step6_differential_sarsa_agent(upper)
    assert upper.trace_decay == 1.0
    assert upper.epsilon_start == 1.0
    assert upper.epsilon_end == 1.0
    assert upper.epsilon_decay_steps == 2**31 - 1


def test_step6_fields_canonicalize_nonbuiltin_numbers() -> None:
    value = np.float64(0.05)
    config = Step6DifferentialSARSAConfig(
        n_actions=np.int64(3),
        q_step_size=value,
        average_reward_step_size=value,
        trace_decay=value,
        epsilon_start=value,
        epsilon_end=value,
        epsilon_decay_steps=np.int64(100),
    )
    agent = make_step6_differential_sarsa_agent(config)
    payload = config.to_dict()
    json.dumps(payload, allow_nan=False)
    assert config.n_actions == 3
    assert config.q_step_size == float(np.float32(0.05))
    assert config.epsilon_decay_steps == 100
    assert type(payload["n_actions"]) is int
    assert type(payload["epsilon_decay_steps"]) is int
    assert type(payload["q_step_size"]) is float
    assert type(payload["average_reward_step_size"]) is float
    assert type(payload["trace_decay"]) is float
    assert type(payload["epsilon_start"]) is float
    assert type(payload["epsilon_end"]) is float
    assert agent.config.n_actions == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("q_step_size", Fraction(-1, 10**400)),
        ("average_reward_step_size", Fraction(-1, 10**400)),
        ("trace_decay", Fraction(-1, 10**400)),
        ("trace_decay", Fraction(10**400 + 1, 10**400)),
        ("epsilon_start", Fraction(-1, 10**400)),
        ("epsilon_start", Fraction(10**400 + 1, 10**400)),
        ("epsilon_end", Fraction(-1, 10**400)),
        ("epsilon_end", Fraction(10**400 + 1, 10**400)),
    ],
)
def test_step6_fields_enforce_exact_scientific_domains(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        Step6DifferentialSARSAConfig(**{field: value})


def test_step6_float32_narrowing_avoids_longdouble_double_rounding() -> None:
    overflow_midpoint = np.ldexp(
        np.longdouble(2) - np.ldexp(np.longdouble(1), -24),
        127,
    )
    largest_finite_input = np.nextafter(
        overflow_midpoint,
        np.longdouble("-inf"),
    )

    config = Step6DifferentialSARSAConfig(q_step_size=largest_finite_input)
    agent = make_step6_differential_sarsa_agent(config)

    assert bool(np.isfinite(np.asarray(config.q_step_size, dtype=np.float32)))
    assert agent.config.q_step_size == config.q_step_size


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (-Fraction(1, 2**80), 1.0),
        (Fraction(0), 1.0),
        (
            Fraction(1, 2**80),
            float(np.nextafter(np.float32(1.0), np.float32(np.inf))),
        ),
    ],
)
def test_step6_fraction_midpoint_rounds_once_to_nearest_even(
    offset: Fraction,
    expected: float,
) -> None:
    midpoint = Fraction(1) + Fraction(1, 2**24)
    config = Step6DifferentialSARSAConfig(q_step_size=midpoint + offset)

    assert config.q_step_size == expected


def test_step6_fraction_float32_overflow_midpoint_is_exact() -> None:
    maximum = Fraction((2**24 - 1) * 2**104)
    overflow_midpoint = maximum + 2**103

    just_below = Step6DifferentialSARSAConfig(q_step_size=overflow_midpoint - 1)
    assert just_below.q_step_size == float(np.finfo(np.float32).max)
    with pytest.raises(ValueError, match="q_step_size"):
        Step6DifferentialSARSAConfig(q_step_size=overflow_midpoint)


def test_step6_float32_underflow_canonicalizes_to_legal_zero_endpoint() -> None:
    config = Step6DifferentialSARSAConfig(
        q_step_size=1e-50,
        average_reward_step_size=np.longdouble("1e-50"),
        epsilon_start=1e-50,
        epsilon_end=np.longdouble("1e-50"),
    )

    assert config.q_step_size == 0.0
    assert config.average_reward_step_size == 0.0
    assert config.epsilon_start == 0.0
    assert config.epsilon_end == 0.0
    assert run_step6_smoke(config, steps=3, feature_dim=2).finite


def test_step6_smoke_health_gate_reports_any_refused_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_run = step6_module.run_differential_sarsa_from_arrays

    def _refuse_updates(*args: Any, **kwargs: Any) -> Any:
        result = original_run(*args, **kwargs)
        return result.replace(
            updates_applied=result.updates_applied.at[0].set(False)
        )

    monkeypatch.setattr(
        step6_module,
        "run_differential_sarsa_from_arrays",
        _refuse_updates,
    )

    result = run_step6_smoke(steps=3, feature_dim=2)

    assert not result.finite


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"steps": 0}, "steps"),
        ({"steps": -1}, "steps"),
        ({"steps": 2**31}, "steps"),
        ({"steps": True}, "steps"),
        ({"steps": "32"}, "steps"),
        ({"feature_dim": 0}, "feature_dim"),
        ({"feature_dim": -1}, "feature_dim"),
        ({"feature_dim": 2**31}, "feature_dim"),
        ({"feature_dim": False}, "feature_dim"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**31}, "seed"),
        ({"seed": True}, "seed"),
    ],
)
def test_step6_smoke_rejects_invalid_inputs(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        run_step6_smoke(**kwargs)


def test_step7_dyna_facade_roundtrip_one_step_and_smoke() -> None:
    config = Step7DynaConfig(
        control=Step6DifferentialSARSAConfig(
            n_actions=2,
            q_step_size=0.02,
            average_reward_step_size=0.01,
            epsilon_start=0.0,
        ),
        world_model=Step8WorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(),
            step_size=0.05,
            sparsity=0.0,
        ),
        planning_steps=2,
        planning_warmup_steps=0,
    )
    agent, model = make_step7_components(config)
    restored = Step7DynaConfig.from_dict(config.to_dict())
    state = init_step7_state(
        agent,
        model,
        key=jr.key(0),
        initial_observation=jnp.array([1.0, 0.0], dtype=jnp.float32),
    )
    one_step = step7_update(
        config,
        agent,
        model,
        state,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([0.0, 1.0], dtype=jnp.float32),
    )
    smoke = run_step7_smoke(config, steps=12, seed=1)

    assert restored == config
    assert int(one_step.state.step_count) == 1
    assert int(one_step.state.world_model_state.step_count) == 1
    assert one_step.planning_td_errors.shape == (2,)
    assert one_step.planning_priorities.shape == (2,)
    assert one_step.planning_anchor_indices.shape == (2,)
    assert one_step.planning_importance_ratios.shape == (2,)
    assert int(one_step.state.memory_count) == 1
    assert bool(jnp.all(one_step.planning_accepted))
    assert smoke.finite
    assert smoke.real_td_errors_shape == (12,)
    assert smoke.planning_td_errors_shape == (12, 2)
    assert smoke.planning_priorities_shape == (12, 2)
    assert smoke.planning_anchor_indices_shape == (12, 2)
    assert smoke.planning_importance_ratios_shape == (12, 2)
    assert smoke.actions_shape == (12,)
    assert smoke.planning_acceptance_count == 24
    assert smoke.control_config["type"] == "DifferentialSARSAAgent"
    assert smoke.world_model_config["type"] == "OneStepWorldModel"


def test_step7_scan_preserves_real_action_context_after_planning() -> None:
    config = Step7DynaConfig(
        control=Step6DifferentialSARSAConfig(n_actions=2, epsilon_start=0.0),
        world_model=Step8WorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(),
            sparsity=0.0,
        ),
        planning_steps=3,
        planning_warmup_steps=0,
    )
    agent, model = make_step7_components(config)
    state = init_step7_state(
        agent,
        model,
        key=jr.key(3),
        initial_observation=jnp.array([0.0, 0.0], dtype=jnp.float32),
    )
    next_observations = jnp.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=jnp.float32,
    )
    rewards = jnp.array([1.0, 0.0, 0.5], dtype=jnp.float32)

    result = run_step7_scan(config, agent, model, state, rewards, next_observations)

    assert int(result.state.step_count) == 3
    assert int(result.state.world_model_state.step_count) == 3
    assert result.planning_td_errors.shape == (3, 3)
    assert result.planning_priorities.shape == (3, 3)
    assert result.planning_anchor_indices.shape == (3, 3)
    assert result.planning_importance_ratios.shape == (3, 3)
    assert bool(jnp.all(result.planning_accepted))
    assert int(result.state.memory_count) == 3
    chex.assert_trees_all_close(
        result.state.memory_observations[:3],
        jnp.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=jnp.float32,
        ),
    )
    assert int(result.state.control_state.last_action) == int(result.actions[-1])
    chex.assert_trees_all_close(
        result.state.control_state.last_observation,
        next_observations[-1],
    )


def test_step7_short_rollout_depth_spends_multiple_imagined_backups() -> None:
    config = Step7DynaConfig(
        control=Step6DifferentialSARSAConfig(n_actions=2, epsilon_start=0.0),
        world_model=Step8WorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(),
            sparsity=0.0,
        ),
        planning_steps=2,
        planning_rollout_depth=3,
        planning_warmup_steps=0,
    )
    agent, model = make_step7_components(config)
    state = init_step7_state(
        agent,
        model,
        key=jr.key(4),
        initial_observation=jnp.array([0.0, 0.0], dtype=jnp.float32),
    )

    result = step7_update(
        config,
        agent,
        model,
        state,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([1.0, 0.0], dtype=jnp.float32),
    )

    assert int(result.state.control_state.step_count) == 7
    assert result.planning_td_errors.shape == (2,)
    assert bool(jnp.all(result.planning_accepted))
    chex.assert_trees_all_close(
        result.state.control_state.last_observation,
        jnp.array([1.0, 0.0], dtype=jnp.float32),
    )


def test_step7_scan_is_jittable() -> None:
    config = Step7DynaConfig(
        control=Step6DifferentialSARSAConfig(n_actions=2, epsilon_start=0.0),
        world_model=Step8WorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(),
            sparsity=0.0,
        ),
        planning_steps=2,
        planning_warmup_steps=0,
    )
    agent, model = make_step7_components(config)
    state = init_step7_state(
        agent,
        model,
        key=jr.key(5),
        initial_observation=jnp.array([0.0, 0.0], dtype=jnp.float32),
    )
    rewards = jnp.array([1.0, 0.0], dtype=jnp.float32)
    next_observations = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)

    result = jax.jit(
        lambda s: run_step7_scan(config, agent, model, s, rewards, next_observations)
    )(state)

    chex.assert_shape(result.real_td_errors, (2,))
    chex.assert_shape(result.planning_td_errors, (2, 2))
    chex.assert_shape(result.planning_priorities, (2, 2))
    chex.assert_shape(result.planning_anchor_indices, (2, 2))
    chex.assert_shape(result.planning_importance_ratios, (2, 2))
    chex.assert_tree_all_finite(
        (
            result.real_td_errors,
            result.planning_td_errors,
            result.planning_priorities,
            result.planning_importance_ratios,
        )
    )


def test_step7_reward_search_control_selects_high_reward_model_action() -> None:
    config = Step7DynaConfig(
        control=Step6DifferentialSARSAConfig(n_actions=2, epsilon_start=0.0),
        world_model=Step8WorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(),
            sparsity=0.0,
        ),
        planning_steps=2,
        planning_warmup_steps=0,
        planning_strategy="reward",
    )
    agent, model = make_step7_components(config)
    state = init_step7_state(
        agent,
        model,
        key=jr.key(11),
        initial_observation=jnp.zeros(2, dtype=jnp.float32),
    )
    learner_state = state.world_model_state.learner_state
    head_weights = list(learner_state.head_params.weights)
    # Input layout is [obs0, obs1, action0, action1]; head 0 is reward.
    head_weights[0] = jnp.array([[0.0, 0.0, 0.0, 3.0]], dtype=jnp.float32)
    learner_state = learner_state.replace(  # type: ignore[attr-defined]
        head_params=learner_state.head_params.replace(  # type: ignore[attr-defined]
            weights=tuple(head_weights)
        )
    )
    state = state.replace(  # type: ignore[attr-defined]
        world_model_state=state.world_model_state.replace(  # type: ignore[attr-defined]
            learner_state=learner_state,
            step_count=jnp.array(1, dtype=jnp.int32),
        )
    )

    result = step7_update(
        config,
        agent,
        model,
        state,
        jnp.array(0.0, dtype=jnp.float32),
        jnp.zeros(2, dtype=jnp.float32),
    )

    chex.assert_trees_all_equal(
        result.planning_actions,
        jnp.ones((2,), dtype=jnp.int32),
    )
    assert bool(jnp.all(result.planning_priorities > 0.0))


def test_step7_planning_records_target_behavior_policy_ratios() -> None:
    config = Step7DynaConfig(
        control=Step6DifferentialSARSAConfig(
            n_actions=2,
            epsilon_start=0.2,
            epsilon_end=0.2,
        ),
        world_model=Step8WorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(),
            sparsity=0.0,
        ),
        planning_steps=1,
        planning_warmup_steps=0,
        planning_strategy="random",
    )
    agent, model = make_step7_components(config)
    state = init_step7_state(
        agent,
        model,
        key=jr.key(13),
        initial_observation=jnp.array([1.0, 0.0], dtype=jnp.float32),
    )

    result = step7_update(
        config,
        agent,
        model,
        state,
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array([0.0, 1.0], dtype=jnp.float32),
    )

    chex.assert_trees_all_close(
        result.planning_behavior_probs,
        jnp.array([0.5], dtype=jnp.float32),
    )
    assert bool(jnp.all(result.planning_target_probs > 0.0))
    assert bool(jnp.all(result.planning_importance_ratios > 0.0))
    assert bool(
        jnp.all(result.planning_importance_ratios <= config.planning_importance_ratio_clip)
    )


def test_step7_importance_correction_scales_imagined_update_delta() -> None:
    config = Step7DynaConfig(
        control=Step6DifferentialSARSAConfig(n_actions=2, epsilon_start=0.0),
        world_model=Step8WorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(),
            sparsity=0.0,
        ),
    )
    agent, model = make_step7_components(config)
    old_state = init_step7_state(
        agent,
        model,
        key=jr.key(15),
        initial_observation=jnp.array([1.0, 0.0], dtype=jnp.float32),
    ).control_state
    planned_state = old_state.replace(  # type: ignore[attr-defined]
        q_weights=old_state.q_weights + 4.0,
        q_bias=old_state.q_bias + 2.0,
        q_trace_weights=old_state.q_trace_weights + 3.0,
        q_trace_bias=old_state.q_trace_bias + 1.0,
        average_reward=old_state.average_reward + 6.0,
    )

    corrected = _apply_planning_importance_correction(
        old_state,
        planned_state,
        jnp.array(0.25, dtype=jnp.float32),
    )

    chex.assert_trees_all_close(corrected.q_weights, old_state.q_weights + 1.0)
    chex.assert_trees_all_close(corrected.q_bias, old_state.q_bias + 0.5)
    chex.assert_trees_all_close(corrected.q_trace_weights, old_state.q_trace_weights + 0.75)
    chex.assert_trees_all_close(corrected.q_trace_bias, old_state.q_trace_bias + 0.25)
    chex.assert_trees_all_close(corrected.average_reward, old_state.average_reward + 1.5)
    chex.assert_trees_all_equal(corrected.last_observation, planned_state.last_observation)
    chex.assert_trees_all_equal(corrected.last_action, planned_state.last_action)


def test_step7_predecessor_search_control_selects_matching_memory_anchor() -> None:
    observations = jnp.array(
        [[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]],
        dtype=jnp.float32,
    )
    rewards = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
    next_observations = jnp.array(
        [[1.0, 0.0], [5.0, 0.0], [9.0, 0.0]],
        dtype=jnp.float32,
    )
    priorities = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
    utilities = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)

    anchor, index, score = _select_planning_anchor(
        observations,
        rewards,
        next_observations,
        priorities,
        utilities,
        jnp.array(3, dtype=jnp.int32),
        jnp.array([5.05, 0.0], dtype=jnp.float32),
        jr.key(0),
        "predecessor",
    )

    chex.assert_trees_all_close(anchor, observations[1])
    assert int(index) == 1
    assert float(score) > float(priorities[1])


def test_step7_learned_search_control_selects_high_utility_anchor() -> None:
    observations = jnp.array(
        [[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]],
        dtype=jnp.float32,
    )
    rewards = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
    next_observations = jnp.array(
        [[1.0, 0.0], [5.0, 0.0], [9.0, 0.0]],
        dtype=jnp.float32,
    )
    priorities = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
    utilities = jnp.array([0.0, 5.0, 0.0], dtype=jnp.float32)

    anchor, index, score = _select_planning_anchor(
        observations,
        rewards,
        next_observations,
        priorities,
        utilities,
        jnp.array(3, dtype=jnp.int32),
        jnp.array([9.0, 0.0], dtype=jnp.float32),
        jr.key(0),
        "learned",
    )

    chex.assert_trees_all_close(anchor, observations[1])
    assert int(index) == 1
    assert float(score) > 5.0


def test_step7_planning_utility_tracks_backup_td_signal() -> None:
    utilities = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)

    updated = _update_planning_utility(
        utilities,
        jnp.array(1, dtype=jnp.int32),
        jnp.array(-10.0, dtype=jnp.float32),
        0.25,
    )

    chex.assert_trees_all_close(updated[0], utilities[0])
    chex.assert_trees_all_close(updated[1], jnp.array(4.0, dtype=jnp.float32))
    chex.assert_trees_all_close(updated[2], utilities[2])


def test_step7_prioritized_queue_pops_highest_priority_anchor() -> None:
    observations = jnp.array(
        [[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]],
        dtype=jnp.float32,
    )
    priorities = jnp.array([0.1, 2.5, 0.3], dtype=jnp.float32)

    anchor, index, priority, queue = _pop_prioritized_planning_anchor(
        observations,
        priorities,
        jnp.array(3, dtype=jnp.int32),
    )

    chex.assert_trees_all_close(anchor, observations[1])
    assert int(index) == 1
    assert float(priority) == 2.5
    assert float(queue[1]) == 0.0
    chex.assert_trees_all_close(queue[jnp.array([0, 2])], priorities[jnp.array([0, 2])])


def test_step7_prioritized_queue_propagates_to_predecessors() -> None:
    next_observations = jnp.array(
        [[1.0, 0.0], [5.0, 0.0], [9.0, 0.0]],
        dtype=jnp.float32,
    )
    priorities = jnp.array([0.1, 0.0, 0.2], dtype=jnp.float32)

    propagated = _propagate_predecessor_priorities(
        next_observations,
        priorities,
        jnp.array(3, dtype=jnp.int32),
        jnp.array([5.0, 0.0], dtype=jnp.float32),
        jnp.array(-3.0, dtype=jnp.float32),
        1.0,
    )

    assert float(propagated[1]) == 3.0
    assert float(propagated[0]) > float(priorities[0])
    assert float(propagated[2]) > float(priorities[2])


def test_step7_prioritized_planning_updates_priority_queue() -> None:
    config = Step7DynaConfig(
        control=Step6DifferentialSARSAConfig(
            n_actions=2,
            q_step_size=0.1,
            average_reward_step_size=0.0,
            epsilon_start=0.0,
        ),
        world_model=Step8WorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(),
            sparsity=0.0,
        ),
        planning_steps=2,
        planning_warmup_steps=0,
        planning_strategy="prioritized",
    )
    agent, model = make_step7_components(config)
    state = init_step7_state(
        agent,
        model,
        key=jr.key(17),
        initial_observation=jnp.array([1.0, 0.0], dtype=jnp.float32),
    )
    state = state.replace(  # type: ignore[attr-defined]
        memory_observations=jnp.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            dtype=jnp.float32,
        ),
        memory_actions=jnp.array([0, 0, 1], dtype=jnp.int32),
        memory_rewards=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        memory_next_observations=jnp.array(
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            dtype=jnp.float32,
        ),
        memory_priorities=jnp.array([0.2, 3.0, 0.4], dtype=jnp.float32),
        memory_utilities=jnp.array([0.2, 3.0, 0.4], dtype=jnp.float32),
        memory_count=jnp.array(3, dtype=jnp.int32),
        memory_position=jnp.array(2, dtype=jnp.int32),
        world_model_state=state.world_model_state.replace(  # type: ignore[attr-defined]
            step_count=jnp.array(10, dtype=jnp.int32),
        ),
    )

    result = step7_update(
        config,
        agent,
        model,
        state,
        jnp.array(2.0, dtype=jnp.float32),
        jnp.array([2.0, 0.0], dtype=jnp.float32),
    )

    assert result.planning_anchor_indices.shape == (2,)
    assert int(result.planning_anchor_indices[0]) == 2
    assert bool(jnp.all(result.planning_accepted))
    assert bool(jnp.any(result.state.memory_priorities != state.memory_priorities))


def test_step7_learned_strategy_updates_selected_memory_utility() -> None:
    config = Step7DynaConfig(
        control=Step6DifferentialSARSAConfig(
            n_actions=2,
            q_step_size=0.1,
            average_reward_step_size=0.0,
            epsilon_start=0.0,
        ),
        world_model=Step8WorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(),
            sparsity=0.0,
        ),
        planning_steps=1,
        planning_warmup_steps=0,
        planning_strategy="learned",
        planning_utility_step_size=1.0,
    )
    agent, model = make_step7_components(config)
    state = init_step7_state(
        agent,
        model,
        key=jr.key(19),
        initial_observation=jnp.array([1.0, 0.0], dtype=jnp.float32),
    )
    state = state.replace(  # type: ignore[attr-defined]
        memory_observations=jnp.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            dtype=jnp.float32,
        ),
        memory_actions=jnp.array([0, 0, 1], dtype=jnp.int32),
        memory_rewards=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        memory_next_observations=jnp.array(
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            dtype=jnp.float32,
        ),
        memory_priorities=jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32),
        memory_utilities=jnp.array([0.1, 5.0, 0.3], dtype=jnp.float32),
        memory_count=jnp.array(3, dtype=jnp.int32),
        memory_position=jnp.array(2, dtype=jnp.int32),
        world_model_state=state.world_model_state.replace(  # type: ignore[attr-defined]
            step_count=jnp.array(10, dtype=jnp.int32),
        ),
    )

    result = step7_update(
        config,
        agent,
        model,
        state,
        jnp.array(2.0, dtype=jnp.float32),
        jnp.array([2.0, 0.0], dtype=jnp.float32),
    )

    assert int(result.planning_anchor_indices[0]) == 1
    assert bool(result.planning_accepted[0])
    assert float(result.state.memory_utilities[1]) != 5.0


def test_step7_rejects_unknown_planning_strategy() -> None:
    try:
        Step7DynaConfig(planning_strategy="unknown")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "planning_strategy" in str(exc)
    else:
        raise AssertionError("expected invalid planning strategy to be rejected")
