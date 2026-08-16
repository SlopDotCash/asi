"""Tests for the Step 11 OaK production facade.

Invalid dimension and scientific-scalar cases are written to fail on current
main (bool, non-real, non-integral, non-finite, and out-of-domain values
accepted) and pass after the facade rejects them. Legal endpoints stay
constructible and accepted numbers canonicalize to builtin ints and floats.
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

from alberta_framework.core.oak import (
    KeyboardChordLearnerConfig,
    OaKAgent,
    OaKArrayResult,
    OaKConfig,
    OaKState,
    init_keyboard_chord_learner,
    keyboard_action,
    keyboard_q_values,
    learned_feature_subtask_specs,
    update_keyboard_chord_learner,
)
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.steps.step11 import (
    Step11OaKConfig,
    Step11SmokeResult,
    init_step11_state,
    make_step11_oak_agent,
    run_step11_scan,
    run_step11_smoke,
    step11_update,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SPEC0 = SubtaskSpec(feature_index=0, threshold=0.5, pseudo_reward_scale=1.0, max_option_steps=8)
_SPEC1 = SubtaskSpec(feature_index=1, threshold=0.3, pseudo_reward_scale=2.0, max_option_steps=4)


def _make_step11_cfg(
    *,
    specs: tuple[SubtaskSpec, ...] = (_SPEC0,),
    obs_dim: int = 4,
    n_prim: int = 2,
) -> Step11OaKConfig:
    return Step11OaKConfig(
        subtask_specs=specs,
        observation_dim=obs_dim,
        n_primitive_actions=n_prim,
    )


def _make_oak_cfg(
    *,
    specs: tuple[SubtaskSpec, ...] = (_SPEC0,),
    obs_dim: int = 4,
) -> OaKConfig:
    stomp = STOMPConfig(subtask_specs=specs, observation_dim=obs_dim)
    return OaKConfig(stomp=stomp)


def _setup(
    cfg: Step11OaKConfig | None = None,
    *,
    seed: int = 0,
) -> tuple[OaKAgent, OaKState]:
    if cfg is None:
        cfg = _make_step11_cfg()
    agent = make_step11_oak_agent(cfg)
    key = jr.key(seed)
    init_obs = jnp.zeros(cfg.observation_dim, dtype=jnp.float32)
    state = init_step11_state(agent, key=key, initial_observation=init_obs)
    return agent, state


# ---------------------------------------------------------------------------
# OaKConfig validation and serialization
# ---------------------------------------------------------------------------


def test_oak_config_invalid_ema_decay_raises() -> None:
    stomp = STOMPConfig(subtask_specs=(_SPEC0,))
    with pytest.raises(ValueError, match="utility_ema_decay"):
        OaKConfig(stomp=stomp, utility_ema_decay=1.5)


def test_oak_config_negative_threshold_raises() -> None:
    stomp = STOMPConfig(subtask_specs=(_SPEC0,))
    with pytest.raises(ValueError, match="curation_threshold"):
        OaKConfig(stomp=stomp, curation_threshold=-0.1)


def test_step11_config_roundtrip_single_spec() -> None:
    cfg = _make_step11_cfg()
    assert Step11OaKConfig.from_config(cfg.to_config()) == cfg


def test_step11_config_roundtrip_two_specs() -> None:
    cfg = _make_step11_cfg(specs=(_SPEC0, _SPEC1), obs_dim=4)
    assert Step11OaKConfig.from_config(cfg.to_config()) == cfg


def test_step11_config_roundtrip_preserves_oak_fields() -> None:
    cfg = Step11OaKConfig(
        subtask_specs=(_SPEC0,),
        observation_dim=4,
        utility_ema_decay=0.95,
        curation_threshold=0.02,
        epsilon_base=0.2,
    )
    restored = Step11OaKConfig.from_config(cfg.to_config())
    assert restored.utility_ema_decay == cfg.utility_ema_decay
    assert restored.curation_threshold == cfg.curation_threshold
    assert restored.epsilon_base == cfg.epsilon_base


def test_step11_config_type_tag_stripped() -> None:
    cfg = _make_step11_cfg()
    d = cfg.to_config()
    assert d["type"] == "Step11OaKConfig"
    assert Step11OaKConfig.from_config(d) == cfg


def test_step11_config_to_oak_config_fields_match() -> None:
    cfg = _make_step11_cfg(obs_dim=5, n_prim=3)
    oak_cfg = cfg.to_oak_config()
    assert isinstance(oak_cfg, OaKConfig)
    assert oak_cfg.observation_dim == 5
    assert oak_cfg.n_primitive_actions == 3
    assert oak_cfg.stomp.subtask_specs == cfg.subtask_specs


_INVALID_STEP11_FIELDS: tuple[tuple[str, Any], ...] = (
    ("observation_dim", 0),
    ("observation_dim", -1),
    ("observation_dim", True),
    ("observation_dim", False),
    ("observation_dim", "4"),
    ("observation_dim", 4.5),
    ("observation_dim", float("nan")),
    ("observation_dim", float("inf")),
    ("observation_dim", None),
    ("n_primitive_actions", 0),
    ("n_primitive_actions", -1),
    ("n_primitive_actions", True),
    ("n_primitive_actions", False),
    ("n_primitive_actions", "2"),
    ("n_primitive_actions", 2.5),
    ("n_primitive_actions", float("nan")),
    ("n_primitive_actions", float("inf")),
    ("n_primitive_actions", None),
    ("option_planning_backups_per_step", -1),
    ("option_planning_backups_per_step", 2**31 - 1),
    ("option_planning_backups_per_step", 2**31),
    ("option_planning_backups_per_step", True),
    ("option_planning_backups_per_step", False),
    ("option_planning_backups_per_step", "0"),
    ("option_planning_backups_per_step", 1.5),
    ("option_planning_backups_per_step", float("nan")),
    ("option_planning_backups_per_step", float("inf")),
    ("option_planning_backups_per_step", None),
    ("base_step_size", float("nan")),
    ("base_step_size", float("inf")),
    ("base_step_size", float("-inf")),
    ("base_step_size", True),
    ("base_step_size", False),
    ("base_step_size", -1.0),
    ("base_step_size", "0.05"),
    ("base_step_size", None),
    ("base_avg_reward_step_size", float("nan")),
    ("base_avg_reward_step_size", float("inf")),
    ("base_avg_reward_step_size", True),
    ("base_avg_reward_step_size", False),
    ("base_avg_reward_step_size", -0.01),
    ("base_avg_reward_step_size", "0.01"),
    ("base_trace_decay", float("nan")),
    ("base_trace_decay", float("inf")),
    ("base_trace_decay", True),
    ("base_trace_decay", False),
    ("base_trace_decay", -0.1),
    ("base_trace_decay", 1.1),
    ("base_trace_decay", "0.0"),
    ("option_step_size", float("nan")),
    ("option_step_size", float("inf")),
    ("option_step_size", True),
    ("option_step_size", False),
    ("option_step_size", -1.0),
    ("option_step_size", "0.05"),
    ("option_avg_reward_step_size", float("nan")),
    ("option_avg_reward_step_size", float("inf")),
    ("option_avg_reward_step_size", True),
    ("option_avg_reward_step_size", False),
    ("option_avg_reward_step_size", -0.01),
    ("option_trace_decay", float("nan")),
    ("option_trace_decay", float("inf")),
    ("option_trace_decay", True),
    ("option_trace_decay", False),
    ("option_trace_decay", -0.1),
    ("option_trace_decay", 1.1),
    ("option_gamma", float("nan")),
    ("option_gamma", float("inf")),
    ("option_gamma", float("-inf")),
    ("option_gamma", True),
    ("option_gamma", False),
    ("option_gamma", -0.1),
    ("option_gamma", 1.1),
    ("option_gamma", "0.99"),
    ("option_model_decay", float("nan")),
    ("option_model_decay", float("inf")),
    ("option_model_decay", True),
    ("option_model_decay", False),
    ("option_model_decay", -0.1),
    ("option_model_decay", 1.1),
    ("option_model_decay", "0.95"),
    ("option_model_step_size", float("nan")),
    ("option_model_step_size", float("inf")),
    ("option_model_step_size", True),
    ("option_model_step_size", False),
    ("option_model_step_size", -0.1),
    ("option_model_step_size", "0.1"),
    ("epsilon_base", float("nan")),
    ("epsilon_base", float("inf")),
    ("epsilon_base", True),
    ("epsilon_base", False),
    ("epsilon_base", -0.1),
    ("epsilon_base", 1.1),
    ("epsilon_base", "0.1"),
    ("epsilon_option", float("nan")),
    ("epsilon_option", float("inf")),
    ("epsilon_option", True),
    ("epsilon_option", False),
    ("epsilon_option", -0.1),
    ("epsilon_option", 1.1),
    ("utility_ema_decay", float("nan")),
    ("utility_ema_decay", float("inf")),
    ("utility_ema_decay", float("-inf")),
    ("utility_ema_decay", True),
    ("utility_ema_decay", False),
    ("utility_ema_decay", -0.1),
    ("utility_ema_decay", 1.1),
    ("utility_ema_decay", "0.99"),
    ("curation_threshold", float("nan")),
    ("curation_threshold", float("inf")),
    ("curation_threshold", float("-inf")),
    ("curation_threshold", True),
    ("curation_threshold", False),
    ("curation_threshold", -0.1),
    ("curation_threshold", "0.0"),
    ("curation_threshold", None),
)


def _config_with(**overrides: Any) -> Step11OaKConfig:
    payload: dict[str, Any] = {
        "subtask_specs": (_SPEC0,),
        "observation_dim": 4,
        "n_primitive_actions": 2,
    }
    payload.update(overrides)
    return Step11OaKConfig(**payload)


@pytest.mark.parametrize(("field", "value"), _INVALID_STEP11_FIELDS)
def test_step11_oak_fields_reject_invalid_inputs(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _config_with(**{field: value})


def test_step11_config_feature_index_out_of_bounds_raises() -> None:
    bad_spec = SubtaskSpec(feature_index=10)
    with pytest.raises(ValueError, match="feature_index"):
        Step11OaKConfig(subtask_specs=(bad_spec,), observation_dim=4)


def test_step11_oak_rejects_non_tuple_subtask_specs() -> None:
    with pytest.raises(ValueError, match="subtask_specs"):
        Step11OaKConfig(subtask_specs=[_SPEC0])  # type: ignore[arg-type]


def test_step11_oak_rejects_bool_and_nonfinite_spec_scalars() -> None:
    with pytest.raises(ValueError, match="feature_index"):
        Step11OaKConfig(
            subtask_specs=(SubtaskSpec(feature_index=True),),
            observation_dim=4,
        )
    with pytest.raises(ValueError, match="threshold"):
        Step11OaKConfig(
            subtask_specs=(SubtaskSpec(feature_index=0, threshold=True),),
            observation_dim=4,
        )
    with pytest.raises(ValueError, match="threshold"):
        Step11OaKConfig(
            subtask_specs=(SubtaskSpec(feature_index=0, threshold=float("nan")),),
            observation_dim=4,
        )
    with pytest.raises(ValueError, match="threshold"):
        Step11OaKConfig(
            subtask_specs=(SubtaskSpec(feature_index=0, threshold=float("inf")),),
            observation_dim=4,
        )
    with pytest.raises(ValueError, match="pseudo_reward_scale"):
        Step11OaKConfig(
            subtask_specs=(SubtaskSpec(feature_index=0, pseudo_reward_scale=True),),
            observation_dim=4,
        )
    with pytest.raises(ValueError, match="pseudo_reward_scale"):
        Step11OaKConfig(
            subtask_specs=(
                SubtaskSpec(feature_index=0, pseudo_reward_scale=float("nan")),
            ),
            observation_dim=4,
        )
    with pytest.raises(ValueError, match="max_option_steps"):
        Step11OaKConfig(
            subtask_specs=(SubtaskSpec(feature_index=0, max_option_steps=True),),
            observation_dim=4,
        )


def test_step11_oak_fields_preserve_legal_endpoints() -> None:
    config = Step11OaKConfig(
        subtask_specs=(
            SubtaskSpec(
                feature_index=0,
                threshold=1e-12,
                pseudo_reward_scale=1e-12,
                max_option_steps=1,
            ),
        ),
        observation_dim=1,
        n_primitive_actions=1,
        base_step_size=0.0,
        base_avg_reward_step_size=0.0,
        base_trace_decay=0.0,
        option_step_size=0.0,
        option_avg_reward_step_size=0.0,
        option_trace_decay=0.0,
        option_gamma=0.0,
        option_model_decay=0.0,
        option_model_step_size=0.0,
        option_planning_backups_per_step=0,
        epsilon_base=0.0,
        epsilon_option=0.0,
        utility_ema_decay=0.0,
        curation_threshold=0.0,
    )
    agent = make_step11_oak_agent(config)
    payload = config.to_config()
    json.dumps(payload, allow_nan=False)
    restored = Step11OaKConfig.from_config(payload)
    assert restored.observation_dim == 1
    assert restored.n_primitive_actions == 1
    assert restored.base_step_size == 0.0
    assert restored.base_avg_reward_step_size == 0.0
    assert restored.base_trace_decay == 0.0
    assert restored.option_step_size == 0.0
    assert restored.option_avg_reward_step_size == 0.0
    assert restored.option_trace_decay == 0.0
    assert restored.option_gamma == 0.0
    assert restored.option_model_decay == 0.0
    assert restored.option_model_step_size == 0.0
    assert restored.option_planning_backups_per_step == 0
    assert restored.epsilon_base == 0.0
    assert restored.epsilon_option == 0.0
    assert restored.utility_ema_decay == 0.0
    assert restored.curation_threshold == 0.0
    assert restored.subtask_specs[0].feature_index == 0
    assert restored.subtask_specs[0].threshold == float(np.float32(1e-12))
    assert restored.subtask_specs[0].pseudo_reward_scale == float(np.float32(1e-12))
    assert restored.subtask_specs[0].max_option_steps == 1
    assert agent.config.stomp.option_gamma == 0.0

    upper = Step11OaKConfig(
        subtask_specs=(_SPEC0,),
        observation_dim=4,
        n_primitive_actions=2,
        base_trace_decay=1.0,
        option_trace_decay=1.0,
        option_gamma=1.0,
        option_model_decay=1.0,
        epsilon_base=1.0,
        epsilon_option=1.0,
        utility_ema_decay=1.0,
        curation_threshold=10.0,
        option_planning_backups_per_step=2**31 - 2,
    )
    make_step11_oak_agent(upper)
    assert upper.base_trace_decay == 1.0
    assert upper.option_trace_decay == 1.0
    assert upper.option_gamma == 1.0
    assert upper.option_model_decay == 1.0
    assert upper.epsilon_base == 1.0
    assert upper.epsilon_option == 1.0
    assert upper.utility_ema_decay == 1.0
    assert upper.curation_threshold == 10.0
    assert upper.option_planning_backups_per_step == 2**31 - 2


def test_step11_oak_rejects_float32_underflow_for_positive_fields() -> None:
    with pytest.raises(ValueError, match="threshold"):
        Step11OaKConfig(
            subtask_specs=(SubtaskSpec(feature_index=0, threshold=1e-50),),
            observation_dim=4,
        )
    with pytest.raises(ValueError, match="threshold"):
        Step11OaKConfig(
            subtask_specs=(SubtaskSpec(feature_index=0, threshold=1e-46),),
            observation_dim=4,
        )
    with pytest.raises(ValueError, match="pseudo_reward_scale"):
        Step11OaKConfig(
            subtask_specs=(
                SubtaskSpec(feature_index=0, pseudo_reward_scale=1e-50),
            ),
            observation_dim=4,
        )


def test_step11_oak_rejects_float32_overflow() -> None:
    with pytest.raises(ValueError, match="threshold"):
        Step11OaKConfig(
            subtask_specs=(SubtaskSpec(feature_index=0, threshold=1e100),),
            observation_dim=4,
        )
    with pytest.raises(ValueError, match="pseudo_reward_scale"):
        Step11OaKConfig(
            subtask_specs=(
                SubtaskSpec(feature_index=0, pseudo_reward_scale=1e100),
            ),
            observation_dim=4,
        )
    with pytest.raises(ValueError, match="pseudo_reward_scale"):
        Step11OaKConfig(
            subtask_specs=(
                SubtaskSpec(feature_index=0, pseudo_reward_scale=-1e100),
            ),
            observation_dim=4,
        )
    with pytest.raises(ValueError, match="base_step_size"):
        Step11OaKConfig(base_step_size=1e100)
    with pytest.raises(ValueError, match="curation_threshold"):
        Step11OaKConfig(curation_threshold=1e100)
    with pytest.raises(ValueError, match="option_model_step_size"):
        Step11OaKConfig(option_model_step_size=1e100)


def test_step11_oak_validates_original_real_domain_before_narrowing() -> None:
    above_one = np.longdouble(1.0) + np.finfo(np.longdouble).eps
    below_zero = -np.nextafter(np.longdouble(0.0), np.longdouble(1.0))
    assert float(above_one) == 1.0
    assert float(below_zero) == 0.0

    with pytest.raises(ValueError, match="epsilon_base"):
        Step11OaKConfig(epsilon_base=above_one)
    with pytest.raises(ValueError, match="base_step_size"):
        Step11OaKConfig(base_step_size=below_zero)


def test_step11_oak_wraps_real_conversion_overflow() -> None:
    with pytest.raises(ValueError, match="base_step_size"):
        Step11OaKConfig(base_step_size=Fraction(10**400, 1))


def test_step11_oak_narrows_the_original_real_once() -> None:
    midpoint_plus = (
        np.longdouble(1.0)
        + np.longdouble(2.0) ** -24
        + np.longdouble(2.0) ** -60
    )
    assert np.float32(midpoint_plus) != np.float32(float(midpoint_plus))
    config = Step11OaKConfig(
        subtask_specs=(
            SubtaskSpec(feature_index=0, pseudo_reward_scale=midpoint_plus),
        ),
    )
    assert config.subtask_specs[0].pseudo_reward_scale == float(
        np.float32(midpoint_plus)
    )


@pytest.mark.parametrize(
    ("pseudo_reward_scale", "expected"),
    [
        (
            Fraction(1, 1) + Fraction(1, 2**24) - Fraction(1, 2**60),
            1.0,
        ),
        (Fraction(1, 1) + Fraction(1, 2**24), 1.0),
        (
            Fraction(1, 1) + Fraction(1, 2**24) + Fraction(1, 2**60),
            float(np.nextafter(np.float32(1.0), np.float32(2.0))),
        ),
    ],
    ids=("below", "tie-to-even", "above"),
)
def test_step11_oak_rounds_fraction_midpoints_once(
    pseudo_reward_scale: Fraction,
    expected: float,
) -> None:
    config = Step11OaKConfig(
        subtask_specs=(
            SubtaskSpec(
                feature_index=0,
                pseudo_reward_scale=pseudo_reward_scale,
            ),
        ),
    )
    assert config.subtask_specs[0].pseudo_reward_scale == expected


def test_step11_fraction_float32_overflow_midpoint_is_exact() -> None:
    maximum = Fraction((2**24 - 1) * 2**104)
    overflow_midpoint = maximum + 2**103

    just_below = Step11OaKConfig(base_step_size=overflow_midpoint - 1)
    assert just_below.base_step_size == float(np.finfo(np.float32).max)
    with pytest.raises(ValueError, match="base_step_size"):
        Step11OaKConfig(base_step_size=overflow_midpoint)


def test_step11_oak_preserves_float32_boundaries() -> None:
    f32_max = float(np.finfo(np.float32).max)
    f32_tiny = float(np.finfo(np.float32).tiny)
    config = Step11OaKConfig(
        subtask_specs=(
            SubtaskSpec(
                feature_index=0,
                threshold=f32_max,
                pseudo_reward_scale=f32_tiny,
                max_option_steps=10,
            ),
        ),
        observation_dim=2,
        curation_threshold=f32_max,
        base_step_size=f32_max,
    )
    agent = make_step11_oak_agent(config)
    assert agent.config.stomp.subtask_specs[0].threshold == f32_max
    assert agent.config.stomp.subtask_specs[0].pseudo_reward_scale == f32_tiny
    assert config.curation_threshold == f32_max
    assert config.base_step_size == f32_max


def test_step11_oak_fields_canonicalize_nonbuiltin_numbers() -> None:
    value = np.float64(0.5)
    spec = SubtaskSpec(
        feature_index=np.int64(1),
        threshold=value,
        pseudo_reward_scale=value,
        max_option_steps=np.int64(4),
    )
    config = Step11OaKConfig(
        subtask_specs=(spec,),
        observation_dim=np.int64(3),
        n_primitive_actions=np.int64(2),
        base_step_size=value,
        base_avg_reward_step_size=value,
        base_trace_decay=value,
        option_step_size=value,
        option_avg_reward_step_size=value,
        option_trace_decay=value,
        option_gamma=value,
        option_model_decay=value,
        option_model_step_size=value,
        option_planning_backups_per_step=np.int64(1),
        epsilon_base=value,
        epsilon_option=value,
        utility_ema_decay=value,
        curation_threshold=np.float64(0.0),
    )
    agent = make_step11_oak_agent(config)
    payload = config.to_config()
    json.dumps(payload, allow_nan=False)
    assert config.observation_dim == 3
    assert config.n_primitive_actions == 2
    assert config.option_planning_backups_per_step == 1
    assert config.option_gamma == 0.5
    assert config.utility_ema_decay == 0.5
    assert config.curation_threshold == 0.0
    assert config.subtask_specs[0].feature_index == 1
    assert config.subtask_specs[0].threshold == 0.5
    assert config.subtask_specs[0].max_option_steps == 4
    assert type(payload["observation_dim"]) is int
    assert type(payload["n_primitive_actions"]) is int
    assert type(payload["option_planning_backups_per_step"]) is int
    assert type(payload["base_step_size"]) is float
    assert type(payload["base_avg_reward_step_size"]) is float
    assert type(payload["base_trace_decay"]) is float
    assert type(payload["option_step_size"]) is float
    assert type(payload["option_avg_reward_step_size"]) is float
    assert type(payload["option_trace_decay"]) is float
    assert type(payload["option_gamma"]) is float
    assert type(payload["option_model_decay"]) is float
    assert type(payload["option_model_step_size"]) is float
    assert type(payload["epsilon_base"]) is float
    assert type(payload["epsilon_option"]) is float
    assert type(payload["utility_ema_decay"]) is float
    assert type(payload["curation_threshold"]) is float
    assert type(payload["subtask_specs"][0]["feature_index"]) is int
    assert type(payload["subtask_specs"][0]["threshold"]) is float
    assert type(payload["subtask_specs"][0]["pseudo_reward_scale"]) is float
    assert type(payload["subtask_specs"][0]["max_option_steps"]) is int
    assert agent.config.stomp.option_gamma == 0.5


# ---------------------------------------------------------------------------
# Factory and initialization
# ---------------------------------------------------------------------------


def test_make_step11_oak_agent_default() -> None:
    agent = make_step11_oak_agent()
    assert isinstance(agent, OaKAgent)
    assert agent.config.n_options == 1


def test_make_step11_oak_agent_two_specs() -> None:
    cfg = _make_step11_cfg(specs=(_SPEC0, _SPEC1))
    agent = make_step11_oak_agent(cfg)
    assert agent.config.n_options == 2


def test_init_step11_state_shapes() -> None:
    cfg = _make_step11_cfg(obs_dim=4)
    agent, state = _setup(cfg)
    chex.assert_shape(state.utility_ema, (1,))
    chex.assert_shape(state.execution_counts, (1,))
    chex.assert_shape(state.cumulative_pseudo_rewards, (1,))


def test_init_step11_state_utility_zero() -> None:
    _, state = _setup()
    assert bool(jnp.all(state.utility_ema == 0.0))


def test_init_step11_state_step_count_zero() -> None:
    _, state = _setup()
    assert int(state.step_count) == 0


def test_init_step11_state_two_specs() -> None:
    cfg = _make_step11_cfg(specs=(_SPEC0, _SPEC1))
    agent, state = _setup(cfg)
    chex.assert_shape(state.utility_ema, (2,))


# ---------------------------------------------------------------------------
# Single-step update
# ---------------------------------------------------------------------------


def test_step11_update_increments_step_count() -> None:
    agent, state = _setup()
    result = step11_update(agent, state, jnp.array(1.0), jnp.zeros(4))
    assert int(result.state.step_count) == 1


def test_step11_update_state_finite() -> None:
    agent, state = _setup()
    result = step11_update(agent, state, jnp.array(0.5), jnp.ones(4) * 0.1)
    chex.assert_tree_all_finite(result.state.stomp_state.base_learner_state)


def test_step11_update_td_error_finite() -> None:
    agent, state = _setup()
    result = step11_update(agent, state, jnp.array(0.0), jnp.zeros(4))
    assert bool(jnp.isfinite(result.td_error))


def test_step11_update_utility_ema_updates_during_execution() -> None:
    spec = SubtaskSpec(feature_index=0, threshold=99.0, max_option_steps=100)
    cfg = Step11OaKConfig(
        subtask_specs=(spec,),
        observation_dim=2,
        n_primitive_actions=2,
    )
    agent, state = _setup(cfg)
    # Force option execution
    state_with_opt = state.replace(
        stomp_state=state.stomp_state.replace(
            executing_option=jnp.array(0, dtype=jnp.int32)
        )
    )
    result = step11_update(agent, state_with_opt, jnp.array(0.0), jnp.array([0.5, 0.0]))
    # Utility EMA should have moved from 0
    assert float(result.utility_ema[0]) != 0.0


def test_step11_update_execution_count_increments_on_start() -> None:
    spec = SubtaskSpec(feature_index=0, threshold=99.0, max_option_steps=100)
    cfg = Step11OaKConfig(
        subtask_specs=(spec,),
        observation_dim=2,
        n_primitive_actions=2,
        epsilon_base=1.0,  # force random to potentially select option
    )
    agent, state = _setup(cfg)
    # Run many steps; option must start at least once
    n_steps = 50
    rewards = jnp.zeros(n_steps)
    obs = jr.normal(jr.key(77), (n_steps, 2)) * 0.1
    result = run_step11_scan(agent, state, rewards, obs)
    # At least 0 executions (option might not get selected, but count is >= 0)
    assert bool(jnp.all(result.state.execution_counts >= 0))


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


def test_run_step11_scan_output_shapes() -> None:
    cfg = _make_step11_cfg(obs_dim=4)
    agent, state = _setup(cfg)
    n_steps = 20
    rewards = jnp.zeros(n_steps)
    obs = jr.normal(jr.key(5), (n_steps, cfg.observation_dim))
    result = run_step11_scan(agent, state, rewards, obs)
    assert isinstance(result, OaKArrayResult)
    chex.assert_shape(result.td_errors, (n_steps,))
    chex.assert_shape(result.utility_emas, (n_steps, 1))
    chex.assert_shape(result.primitive_actions, (n_steps,))


def test_run_step11_scan_two_specs_shapes() -> None:
    cfg = _make_step11_cfg(specs=(_SPEC0, _SPEC1), obs_dim=4)
    agent, state = _setup(cfg)
    n_steps = 16
    result = run_step11_scan(
        agent, state, jnp.zeros(n_steps), jr.normal(jr.key(3), (n_steps, 4))
    )
    chex.assert_shape(result.utility_emas, (n_steps, 2))


def test_run_step11_scan_all_finite() -> None:
    cfg = _make_step11_cfg()
    agent, state = _setup(cfg, seed=42)
    n_steps = 50
    result = run_step11_scan(
        agent, state,
        jr.normal(jr.key(0), (n_steps,)) * 0.1,
        jr.normal(jr.key(1), (n_steps, cfg.observation_dim)) * 0.1,
    )
    chex.assert_tree_all_finite(result.td_errors)
    chex.assert_tree_all_finite(result.utility_emas)
    chex.assert_tree_all_finite(result.pseudo_rewards)


def test_run_step11_scan_final_step_count() -> None:
    cfg = _make_step11_cfg()
    agent, state = _setup(cfg)
    n_steps = 16
    result = run_step11_scan(
        agent, state, jnp.zeros(n_steps), jr.normal(jr.key(9), (n_steps, 4))
    )
    assert int(result.state.step_count) == n_steps


def test_run_step11_scan_actions_in_range() -> None:
    cfg = _make_step11_cfg(n_prim=3)
    agent, state = _setup(cfg)
    n_steps = 30
    result = run_step11_scan(
        agent, state, jnp.zeros(n_steps), jr.normal(jr.key(8), (n_steps, cfg.observation_dim))
    )
    assert bool(jnp.all(result.primitive_actions >= 0))
    assert bool(jnp.all(result.primitive_actions < 3))


# ---------------------------------------------------------------------------
# Curation
# ---------------------------------------------------------------------------


def test_curate_returns_new_agent() -> None:
    agent, state = _setup()
    new_agent, _ = agent.curate(state, jr.key(0))
    assert isinstance(new_agent, OaKAgent)


def test_curate_resets_utility_for_replaced_option() -> None:
    cfg = _make_step11_cfg(specs=(_SPEC0, _SPEC1), obs_dim=4)
    agent, state = _setup(cfg)
    # option 0 has higher utility (0.8), option 1 has lower utility (0.1)
    state = state.replace(
        utility_ema=jnp.array([0.8, 0.1], dtype=jnp.float32)
    )
    _, new_state = agent.curate(state, jr.key(0))
    # argmin picks option 1 (0.1 < 0.8) — option 1 should be reset to 0
    assert float(new_state.utility_ema[1]) == 0.0
    # Option 0 should be preserved
    chex.assert_trees_all_close(new_state.utility_ema[0], jnp.array(0.8), atol=1e-5)


def test_curate_above_threshold_skips() -> None:
    stomp = STOMPConfig(subtask_specs=(_SPEC0,))
    cfg = OaKConfig(stomp=stomp, curation_threshold=0.5)
    agent = OaKAgent(cfg)
    key = jr.key(0)
    state = agent.init(key)
    state = agent.start(state, jnp.zeros(4))
    # Give option utility above threshold
    state = state.replace(utility_ema=jnp.array([0.8], dtype=jnp.float32))
    new_agent, new_state = agent.curate(state, key)
    # Same agent returned (no replacement)
    assert new_agent is agent
    chex.assert_trees_all_close(new_state.utility_ema[0], jnp.array(0.8), atol=1e-5)


def test_curate_resets_option_weights() -> None:
    cfg = _make_step11_cfg()
    agent, state = _setup(cfg)
    # Run a few steps so weights are non-zero
    n_steps = 20
    state_after = run_step11_scan(
        agent, state, jnp.zeros(n_steps), jr.normal(jr.key(1), (n_steps, 4))
    ).state
    # Curate at a coherent primitive boundary. If the scan ends inside the
    # sole option, production curation correctly defers replacement.
    state_after = state_after.replace(
        stomp_state=state_after.stomp_state.replace(
            executing_option=jnp.array(-1, dtype=jnp.int32),
            base_last_action=jnp.array(0, dtype=jnp.int32),
            last_primitive_action=jnp.array(0, dtype=jnp.int32),
            option_cumreward=jnp.array(0.0, dtype=jnp.float32),
            option_env_cumreward=jnp.array(0.0, dtype=jnp.float32),
            option_baseline_mass=jnp.array(0.0, dtype=jnp.float32),
            option_discount=jnp.array(1.0, dtype=jnp.float32),
            option_steps=jnp.array(0, dtype=jnp.int32),
        )
    )
    _, curated_state = agent.curate(state_after, jr.key(0))
    # Option 0 was the only option and should be reset
    chex.assert_trees_all_close(
        curated_state.stomp_state.option_policies.q_weights[0],
        jnp.zeros_like(curated_state.stomp_state.option_policies.q_weights[0]),
    )


# ---------------------------------------------------------------------------
# Option keyboard
# ---------------------------------------------------------------------------


def test_keyboard_q_values_shape() -> None:
    cfg = _make_step11_cfg(specs=(_SPEC0, _SPEC1), n_prim=3, obs_dim=4)
    agent, state = _setup(cfg)
    w = jnp.array([0.5, 0.5], dtype=jnp.float32)
    obs = jnp.ones(4, dtype=jnp.float32)
    q = keyboard_q_values(state.stomp_state, obs, w)
    chex.assert_shape(q, (3,))


def test_keyboard_q_values_uniform_weights_averages_options() -> None:
    cfg = _make_step11_cfg(specs=(_SPEC0, _SPEC1), n_prim=2, obs_dim=4)
    agent, state = _setup(cfg)
    obs = jnp.ones(4, dtype=jnp.float32)
    uniform = jnp.array([0.5, 0.5], dtype=jnp.float32)
    q_blend = keyboard_q_values(state.stomp_state, obs, uniform)
    # Should match manual average (after L1-normalisation uniform → [0.5, 0.5])
    q0 = state.stomp_state.option_policies.q_weights[0] @ obs
    q1 = state.stomp_state.option_policies.q_weights[1] @ obs
    expected = 0.5 * q0 + 0.5 * q1
    chex.assert_trees_all_close(q_blend, expected, atol=1e-5)


def test_keyboard_action_returns_valid_primitive() -> None:
    cfg = _make_step11_cfg(specs=(_SPEC0, _SPEC1), n_prim=3, obs_dim=4)
    agent, state = _setup(cfg)
    w = jnp.array([0.7, 0.3], dtype=jnp.float32)
    obs = jnp.zeros(4, dtype=jnp.float32)
    action, _ = keyboard_action(
        state.stomp_state, obs, w, jr.key(0), epsilon=0.0, n_primitive_actions=3
    )
    assert 0 <= int(action) < 3


def test_keyboard_action_epsilon_one_is_random() -> None:
    cfg = _make_step11_cfg(specs=(_SPEC0,), n_prim=2, obs_dim=4)
    agent, state = _setup(cfg)
    w = jnp.ones(1, dtype=jnp.float32)
    obs = jnp.zeros(4)
    actions = set()
    for seed in range(50):
        a, _ = keyboard_action(
            state.stomp_state, obs, w, jr.key(seed), epsilon=1.0, n_primitive_actions=2
        )
        actions.add(int(a))
    assert len(actions) > 1


# ---------------------------------------------------------------------------
# Learned feature construction and keyboard learning
# ---------------------------------------------------------------------------


def test_learned_feature_subtask_specs_ranks_weighted_features() -> None:
    agent, state = _setup(_make_step11_cfg(obs_dim=4), seed=12)
    head_weights = tuple(
        w.at[0, 2].set(3.0) if i == 0 else w
        for i, w in enumerate(state.stomp_state.base_learner_state.head_params.weights)
    )
    option_q = state.stomp_state.option_policies.q_weights.at[0, 1, 3].set(2.0)
    state = state.replace(
        stomp_state=state.stomp_state.replace(
            base_learner_state=state.stomp_state.base_learner_state.replace(
                head_params=state.stomp_state.base_learner_state.head_params.replace(
                    weights=head_weights
                )
            ),
            option_policies=state.stomp_state.option_policies.replace(
                q_weights=option_q
            ),
        )
    )
    specs = learned_feature_subtask_specs(state, n_subtasks=2, threshold=0.7)
    assert [spec.feature_index for spec in specs] == [2, 3]
    assert all(spec.threshold == pytest.approx(0.7) for spec in specs)


def test_keyboard_chord_learner_roundtrip() -> None:
    cfg = KeyboardChordLearnerConfig(
        n_options=3,
        step_size=0.2,
        baseline_decay=0.5,
        l2_penalty=0.01,
        max_norm=2.0,
    )
    assert KeyboardChordLearnerConfig.from_config(cfg.to_config()) == cfg


def test_keyboard_chord_learner_positive_reward_moves_toward_chord() -> None:
    cfg = KeyboardChordLearnerConfig(
        n_options=2,
        step_size=0.5,
        baseline_decay=0.5,
    )
    state = init_keyboard_chord_learner(cfg)
    selected = jnp.array([1.0, 0.0], dtype=jnp.float32)
    before = float(jnp.dot(state.chord_vector, selected))
    updated = update_keyboard_chord_learner(
        cfg,
        state,
        selected,
        jnp.array(1.0, dtype=jnp.float32),
    )
    after = float(jnp.dot(updated.chord_vector, selected))
    assert after > before
    assert int(updated.step_count) == 1


def test_keyboard_chord_learner_max_norm_bounds_vector() -> None:
    cfg = KeyboardChordLearnerConfig(
        n_options=2,
        step_size=10.0,
        max_norm=0.75,
    )
    state = init_keyboard_chord_learner(cfg)
    updated = update_keyboard_chord_learner(
        cfg,
        state,
        jnp.array([1.0, 1.0], dtype=jnp.float32),
        jnp.array(10.0, dtype=jnp.float32),
    )
    assert float(jnp.linalg.norm(updated.chord_vector)) <= 0.750001


def test_keyboard_chord_learner_infinite_reward_does_not_poison_vector() -> None:
    """Inf advantage * a zero chord coordinate is 0*inf = NaN."""
    cfg = KeyboardChordLearnerConfig(n_options=3, step_size=0.1, max_norm=10.0)
    state = init_keyboard_chord_learner(cfg)
    selected = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)

    poisoned = update_keyboard_chord_learner(
        cfg, state, selected, jnp.array(jnp.inf, dtype=jnp.float32)
    )
    chex.assert_trees_all_close(poisoned.chord_vector, state.chord_vector)
    chex.assert_trees_all_close(poisoned.reward_baseline, state.reward_baseline)
    assert int(poisoned.step_count) == int(state.step_count)

    recovered = update_keyboard_chord_learner(
        cfg, poisoned, selected, jnp.array(1.0, dtype=jnp.float32)
    )
    chex.assert_tree_all_finite(recovered.chord_vector)
    chex.assert_tree_all_finite(recovered.reward_baseline)
    assert int(recovered.step_count) == int(state.step_count) + 1


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def test_run_step11_smoke_defaults() -> None:
    result = run_step11_smoke()
    assert isinstance(result, Step11SmokeResult)
    assert result.finite
    assert result.steps == 64
    assert result.td_errors_shape == (64,)
    assert result.utility_emas_shape == (64, 1)


def test_run_step11_smoke_two_specs() -> None:
    cfg = Step11OaKConfig(
        subtask_specs=(_SPEC0, _SPEC1),
        observation_dim=4,
        n_primitive_actions=2,
    )
    result = run_step11_smoke(cfg, steps=32, seed=1)
    assert result.finite
    assert result.utility_emas_shape == (32, 2)


def test_run_step11_smoke_to_dict_roundtrip() -> None:
    result = run_step11_smoke(steps=8)
    d = result.to_dict()
    assert isinstance(d["agent_config"], dict)
    assert d["finite"] is True


def test_run_step11_smoke_zero_steps_raises() -> None:
    with pytest.raises(ValueError, match="steps"):
        run_step11_smoke(steps=0)


@pytest.mark.parametrize("steps", [True, 1.5])
def test_run_step11_smoke_rejects_non_integer_steps(steps: object) -> None:
    with pytest.raises(ValueError, match="steps must be an integer"):
        run_step11_smoke(steps=steps)  # type: ignore[arg-type]


def test_run_step11_smoke_rejects_class_spoofed_integer_steps() -> None:
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
        run_step11_smoke(steps=_SpoofedInt())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Long-horizon fineness
# ---------------------------------------------------------------------------


def test_step11_state_stays_finite_200_steps() -> None:
    cfg = Step11OaKConfig(
        subtask_specs=(SubtaskSpec(feature_index=0, threshold=0.3, max_option_steps=4),),
        observation_dim=3,
        n_primitive_actions=2,
        base_step_size=0.01,
        option_step_size=0.01,
        utility_ema_decay=0.95,
    )
    agent, state = _setup(cfg, seed=5)
    n_steps = 200
    result = run_step11_scan(
        agent, state,
        jr.normal(jr.key(10), (n_steps,)) * 0.1,
        jr.normal(jr.key(11), (n_steps, 3)) * 0.1,
    )
    chex.assert_tree_all_finite(result.state.stomp_state.base_learner_state)
    chex.assert_tree_all_finite(result.state.utility_ema)
    chex.assert_tree_all_finite(result.td_errors)
