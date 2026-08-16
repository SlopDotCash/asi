"""Production-facing Step 10 STOMP facade tests.

Invalid dimension and scientific-scalar cases are written to fail on current
main (bool, non-real, non-integral, non-finite, and out-of-domain values
accepted) and pass after the facade rejects them. Legal endpoints stay
constructible and accepted numbers canonicalize to builtin ints and floats.
"""

from __future__ import annotations

import json
from typing import Any

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.options import (
    STOMPAgent,
    STOMPArrayResult,
    STOMPConfig,
    STOMPState,
    SubtaskSpec,
    subtasks_from_feature_scores,
)
from alberta_framework.steps.step10 import (
    Step10SmokeResult,
    Step10STOMPConfig,
    init_step10_state,
    make_step10_stomp_agent,
    run_step10_scan,
    run_step10_smoke,
    step10_update,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SPEC1 = SubtaskSpec(feature_index=0, threshold=0.5, pseudo_reward_scale=1.0, max_option_steps=8)
_SPEC2 = SubtaskSpec(feature_index=1, threshold=0.3, pseudo_reward_scale=2.0, max_option_steps=4)


def _make_cfg(
    *,
    specs: tuple[SubtaskSpec, ...] = (_SPEC1,),
    obs_dim: int = 4,
    n_prim: int = 2,
) -> Step10STOMPConfig:
    return Step10STOMPConfig(
        subtask_specs=specs,
        observation_dim=obs_dim,
        n_primitive_actions=n_prim,
    )


def _setup(
    cfg: Step10STOMPConfig | None = None,
    *,
    seed: int = 0,
) -> tuple[STOMPAgent, STOMPState]:
    if cfg is None:
        cfg = _make_cfg()
    agent = make_step10_stomp_agent(cfg)
    key = jr.key(seed)
    init_obs = jnp.zeros(cfg.observation_dim, dtype=jnp.float32)
    state = init_step10_state(agent, key=key, initial_observation=init_obs)
    return agent, state


# ---------------------------------------------------------------------------
# SubtaskSpec validation
# ---------------------------------------------------------------------------


def test_subtask_spec_negative_feature_index_raises() -> None:
    with pytest.raises(ValueError, match="feature_index"):
        SubtaskSpec(feature_index=-1)


def test_subtask_spec_zero_threshold_raises() -> None:
    with pytest.raises(ValueError, match="threshold"):
        SubtaskSpec(feature_index=0, threshold=0.0)


def test_subtask_spec_zero_max_steps_raises() -> None:
    with pytest.raises(ValueError, match="max_option_steps"):
        SubtaskSpec(feature_index=0, max_option_steps=0)


# ---------------------------------------------------------------------------
# Step10STOMPConfig validation and serialization
# ---------------------------------------------------------------------------


def test_config_roundtrip_single_spec() -> None:
    cfg = _make_cfg()
    assert Step10STOMPConfig.from_config(cfg.to_config()) == cfg


def test_config_roundtrip_two_specs() -> None:
    cfg = _make_cfg(specs=(_SPEC1, _SPEC2), obs_dim=4)
    assert Step10STOMPConfig.from_config(cfg.to_config()) == cfg


def test_config_roundtrip_preserves_hyperparams() -> None:
    cfg = Step10STOMPConfig(
        subtask_specs=(_SPEC1,),
        observation_dim=6,
        n_primitive_actions=3,
        base_step_size=0.01,
        epsilon_base=0.2,
        option_gamma=0.95,
        option_target_epsilon=0.0,
        option_importance_clip=2.0,
    )
    restored = Step10STOMPConfig.from_config(cfg.to_config())
    assert restored.base_step_size == cfg.base_step_size
    assert restored.epsilon_base == cfg.epsilon_base
    assert restored.option_gamma == cfg.option_gamma
    assert restored.n_primitive_actions == cfg.n_primitive_actions
    assert restored.option_target_epsilon == cfg.option_target_epsilon
    assert restored.option_importance_clip == cfg.option_importance_clip


def test_config_type_tag_stripped_on_roundtrip() -> None:
    cfg = _make_cfg()
    d = cfg.to_config()
    assert d["type"] == "Step10STOMPConfig"
    restored = Step10STOMPConfig.from_config(d)
    assert restored == cfg


def test_config_to_stomp_config_matches_fields() -> None:
    cfg = _make_cfg(n_prim=3)
    stomp = cfg.to_stomp_config()
    assert isinstance(stomp, STOMPConfig)
    assert stomp.n_primitive_actions == 3
    assert stomp.observation_dim == cfg.observation_dim
    assert stomp.subtask_specs == cfg.subtask_specs


def test_config_feature_index_out_of_bounds_raises() -> None:
    bad_spec = SubtaskSpec(feature_index=10)
    with pytest.raises(ValueError, match="feature_index"):
        Step10STOMPConfig(subtask_specs=(bad_spec,), observation_dim=4)


def test_config_without_subtasks_remains_a_serializable_template() -> None:
    """The public default config stays constructible; execution rejects it."""
    config = Step10STOMPConfig()
    assert config.subtask_specs == ()
    assert Step10STOMPConfig.from_config(config.to_config()) == config
    with pytest.raises(ValueError, match="subtask"):
        make_step10_stomp_agent(config)


def test_config_invalid_option_target_epsilon_raises() -> None:
    with pytest.raises(ValueError, match="option_target_epsilon"):
        Step10STOMPConfig(
            subtask_specs=(_SPEC1,),
            option_target_epsilon=1.1,
        )


def test_config_invalid_option_importance_clip_raises() -> None:
    with pytest.raises(ValueError, match="option_importance_clip"):
        Step10STOMPConfig(
            subtask_specs=(_SPEC1,),
            option_importance_clip=0.0,
        )


_INVALID_STEP10_FIELDS: tuple[tuple[str, Any], ...] = (
    ("observation_dim", True),
    ("observation_dim", False),
    ("observation_dim", 0),
    ("observation_dim", -1),
    ("observation_dim", 1.5),
    ("observation_dim", "4"),
    ("observation_dim", 2**31),
    ("observation_dim", None),
    ("n_primitive_actions", True),
    ("n_primitive_actions", False),
    ("n_primitive_actions", 0),
    ("n_primitive_actions", -1),
    ("n_primitive_actions", 1.5),
    ("n_primitive_actions", "2"),
    ("n_primitive_actions", None),
    ("n_primitive_actions", 2**31),
    ("option_planning_backups_per_step", True),
    ("option_planning_backups_per_step", False),
    ("option_planning_backups_per_step", -1),
    ("option_planning_backups_per_step", 1.5),
    ("option_planning_backups_per_step", "0"),
    ("option_planning_backups_per_step", None),
    ("option_planning_backups_per_step", 2**31 - 1),
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
    ("option_model_step_size", float("nan")),
    ("option_model_step_size", float("inf")),
    ("option_model_step_size", True),
    ("option_model_step_size", False),
    ("option_model_step_size", -0.1),
    ("option_model_step_size", "0.1"),
    ("base_trace_decay", float("nan")),
    ("base_trace_decay", float("inf")),
    ("base_trace_decay", True),
    ("base_trace_decay", False),
    ("base_trace_decay", -0.1),
    ("base_trace_decay", 1.1),
    ("base_trace_decay", "0.0"),
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
    ("option_target_epsilon", float("nan")),
    ("option_target_epsilon", float("inf")),
    ("option_target_epsilon", True),
    ("option_target_epsilon", False),
    ("option_target_epsilon", -0.1),
    ("option_target_epsilon", 1.1),
    ("option_target_epsilon", "0.0"),
    ("option_importance_clip", float("nan")),
    ("option_importance_clip", float("inf")),
    ("option_importance_clip", float("-inf")),
    ("option_importance_clip", True),
    ("option_importance_clip", False),
    ("option_importance_clip", 0.0),
    ("option_importance_clip", -1.0),
    ("option_importance_clip", "10.0"),
    ("option_importance_clip", None),
)


def _config_with(**overrides: Any) -> Step10STOMPConfig:
    payload: dict[str, Any] = {
        "subtask_specs": (_SPEC1,),
        "observation_dim": 4,
        "n_primitive_actions": 2,
    }
    payload.update(overrides)
    return Step10STOMPConfig(**payload)


@pytest.mark.parametrize(("field", "value"), _INVALID_STEP10_FIELDS)
def test_step10_stomp_fields_reject_invalid_inputs(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _config_with(**{field: value})


@pytest.mark.unit
def test_step10_stomp_int_dimensions_stay_within_int32() -> None:
    int32_max = 2**31 - 1
    upper = Step10STOMPConfig(
        subtask_specs=(_SPEC1,),
        observation_dim=int32_max,
        n_primitive_actions=int32_max,
    )
    assert upper.observation_dim == int32_max
    assert upper.n_primitive_actions == int32_max
    with pytest.raises(ValueError, match="observation_dim"):
        Step10STOMPConfig(subtask_specs=(_SPEC1,), observation_dim=int32_max + 1)
    with pytest.raises(ValueError, match="n_primitive_actions"):
        Step10STOMPConfig(subtask_specs=(_SPEC1,), n_primitive_actions=int32_max + 1)


@pytest.mark.unit
def test_step10_stomp_feature_index_stays_within_int32() -> None:
    # An out-of-int32 feature_index can only pass the observation_dim bound
    # check if observation_dim itself escapes int32, so both must reject.
    with pytest.raises(ValueError, match="observation_dim"):
        Step10STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=2**35),),
            observation_dim=2**40,
        )
    spec = SubtaskSpec(feature_index=2**31 - 2)
    cfg = Step10STOMPConfig(subtask_specs=(spec,), observation_dim=2**31 - 1)
    assert cfg.subtask_specs[0].feature_index == 2**31 - 2
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
    ["observation_dim", "n_primitive_actions", "option_planning_backups_per_step"],
)
def test_step10_stomp_fields_reject_class_spoofed_integers(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _config_with(**{field: _SpoofedInt()})


# Note: SubtaskSpec.feature_index/max_option_steps also route through this
# module's _require_int, but SubtaskSpec.__post_init__ (core/options.py) runs
# its own unguarded `< 0`/`< 1` comparison first and raises a raw TypeError
# on a spoofed object before this module's helper is ever reached. That is a
# separate, out-of-scope gap in a different file/function - not fixed here.


def test_step10_stomp_rejects_non_tuple_subtask_specs() -> None:
    with pytest.raises(ValueError, match="subtask_specs"):
        Step10STOMPConfig(subtask_specs=[_SPEC1])  # type: ignore[arg-type]


def test_step10_stomp_rejects_bool_and_nonfinite_spec_scalars() -> None:
    with pytest.raises(ValueError, match="feature_index"):
        Step10STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=True),),
            observation_dim=4,
        )
    with pytest.raises(ValueError, match="threshold"):
        Step10STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0, threshold=True),),
            observation_dim=4,
        )
    with pytest.raises(ValueError, match="threshold"):
        Step10STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0, threshold=float("nan")),),
            observation_dim=4,
        )
    with pytest.raises(ValueError, match="threshold"):
        Step10STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0, threshold=float("inf")),),
            observation_dim=4,
        )
    with pytest.raises(ValueError, match="pseudo_reward_scale"):
        Step10STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0, pseudo_reward_scale=True),),
            observation_dim=4,
        )
    with pytest.raises(ValueError, match="pseudo_reward_scale"):
        Step10STOMPConfig(
            subtask_specs=(
                SubtaskSpec(feature_index=0, pseudo_reward_scale=float("nan")),
            ),
            observation_dim=4,
        )
    with pytest.raises(ValueError, match="max_option_steps"):
        Step10STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0, max_option_steps=True),),
            observation_dim=4,
        )


def test_step10_stomp_fields_preserve_legal_endpoints() -> None:
    config = Step10STOMPConfig(
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
        option_target_epsilon=None,
        option_importance_clip=1e-12,
    )
    agent = make_step10_stomp_agent(config)
    payload = config.to_config()
    json.dumps(payload, allow_nan=False)
    restored = Step10STOMPConfig.from_config(payload)
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
    assert restored.option_target_epsilon is None
    assert restored.option_importance_clip == 1e-12
    assert restored.subtask_specs[0].feature_index == 0
    assert restored.subtask_specs[0].threshold == 1e-12
    assert restored.subtask_specs[0].pseudo_reward_scale == 1e-12
    assert restored.subtask_specs[0].max_option_steps == 1
    assert agent.config.option_gamma == 0.0

    upper = Step10STOMPConfig(
        subtask_specs=(_SPEC1,),
        observation_dim=4,
        n_primitive_actions=2,
        base_trace_decay=1.0,
        option_trace_decay=1.0,
        option_gamma=1.0,
        option_model_decay=1.0,
        epsilon_base=1.0,
        epsilon_option=1.0,
        option_target_epsilon=0.0,
        option_importance_clip=10.0,
        option_planning_backups_per_step=2**31 - 2,
    )
    make_step10_stomp_agent(upper)
    assert upper.base_trace_decay == 1.0
    assert upper.option_trace_decay == 1.0
    assert upper.option_gamma == 1.0
    assert upper.option_model_decay == 1.0
    assert upper.epsilon_base == 1.0
    assert upper.epsilon_option == 1.0
    assert upper.option_target_epsilon == 0.0
    assert upper.option_planning_backups_per_step == 2**31 - 2

    one = Step10STOMPConfig(
        subtask_specs=(_SPEC1,),
        option_target_epsilon=1.0,
    )
    make_step10_stomp_agent(one)
    assert one.option_target_epsilon == 1.0


def test_step10_stomp_fields_canonicalize_nonbuiltin_numbers() -> None:
    value = np.float64(0.5)
    spec = SubtaskSpec(
        feature_index=np.int64(1),
        threshold=value,
        pseudo_reward_scale=value,
        max_option_steps=np.int64(4),
    )
    config = Step10STOMPConfig(
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
        option_target_epsilon=value,
        option_importance_clip=np.float64(2.0),
    )
    agent = make_step10_stomp_agent(config)
    payload = config.to_config()
    json.dumps(payload, allow_nan=False)
    assert config.observation_dim == 3
    assert config.n_primitive_actions == 2
    assert config.option_planning_backups_per_step == 1
    assert config.option_gamma == 0.5
    assert config.option_target_epsilon == 0.5
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
    assert type(payload["option_target_epsilon"]) is float
    assert type(payload["option_importance_clip"]) is float
    assert type(payload["subtask_specs"][0]["feature_index"]) is int
    assert type(payload["subtask_specs"][0]["threshold"]) is float
    assert type(payload["subtask_specs"][0]["pseudo_reward_scale"]) is float
    assert type(payload["subtask_specs"][0]["max_option_steps"]) is int
    assert agent.config.option_gamma == 0.5
    assert agent.config.option_importance_clip == 2.0


# ---------------------------------------------------------------------------
# Factory and initialization
# ---------------------------------------------------------------------------


def test_make_stomp_agent_default_returns_agent() -> None:
    agent = make_step10_stomp_agent()
    assert isinstance(agent, STOMPAgent)
    assert agent.config.n_options == 1


def test_make_stomp_agent_two_specs() -> None:
    cfg = _make_cfg(specs=(_SPEC1, _SPEC2))
    agent = make_step10_stomp_agent(cfg)
    assert agent.config.n_options == 2
    assert agent.config.n_total_actions == 4  # 2 prim + 2 options


def test_init_step10_state_shapes() -> None:
    cfg = _make_cfg(obs_dim=4, n_prim=2)
    agent, state = _setup(cfg)
    assert len(state.base_learner_state.head_params.weights) == 3  # n_total=3
    chex.assert_shape(state.base_learner_state.head_params.weights[0], (1, 4))
    chex.assert_shape(state.option_policies.q_weights, (1, 2, 4))
    chex.assert_shape(state.option_models.next_state_weights, (1, 4, 4))
    chex.assert_shape(state.option_models.baseline_mass_ema, (1,))
    chex.assert_shape(state.option_baseline_mass, ())


def test_init_step10_state_records_first_selected_action() -> None:
    agent, state = _setup()
    executing_option = int(state.executing_option)
    assert -1 <= executing_option < agent.config.n_options
    assert 0 <= int(state.last_primitive_action) < agent.config.n_primitive_actions
    if executing_option >= 0:
        assert int(state.base_last_action) == (
            agent.config.n_primitive_actions + executing_option
        )
        chex.assert_trees_all_close(state.option_start_obs, state.base_last_obs)
    else:
        assert int(state.base_last_action) < agent.config.n_primitive_actions


def test_init_step10_state_step_count_zero() -> None:
    _, state = _setup()
    assert int(state.step_count) == 0


def test_init_step10_state_last_obs_primed() -> None:
    cfg = _make_cfg(obs_dim=3)
    agent = make_step10_stomp_agent(cfg)
    init_obs = jnp.ones(3, dtype=jnp.float32) * 0.5
    state = init_step10_state(agent, key=jr.key(1), initial_observation=init_obs)
    chex.assert_trees_all_close(state.base_last_obs, init_obs)


# ---------------------------------------------------------------------------
# Single-step update
# ---------------------------------------------------------------------------


def test_step10_update_increments_step_count() -> None:
    agent, state = _setup()
    result = step10_update(agent, state, jnp.array(1.0), jnp.zeros(4))
    assert int(result.state.step_count) == 1


def test_step10_update_returns_valid_primitive_action() -> None:
    cfg = _make_cfg(n_prim=3)
    agent, state = _setup(cfg)
    result = step10_update(agent, state, jnp.array(0.5), jnp.zeros(4))
    prim = int(result.primitive_action)
    assert 0 <= prim < 3


def test_step10_update_state_finite() -> None:
    agent, state = _setup()
    result = step10_update(agent, state, jnp.array(0.0), jnp.ones(4) * 0.1)
    chex.assert_tree_all_finite(result.state.base_learner_state)
    chex.assert_tree_all_finite(result.state.option_policies.q_weights)


def test_step10_update_td_error_finite() -> None:
    agent, state = _setup()
    result = step10_update(agent, state, jnp.array(-0.5), jnp.zeros(4))
    assert bool(jnp.isfinite(result.td_error))


def test_step10_update_executing_option_in_range() -> None:
    agent, state = _setup()
    result = step10_update(agent, state, jnp.array(0.0), jnp.zeros(4))
    exec_opt = int(result.executing_option)
    assert exec_opt >= -1


def test_step10_off_policy_intra_option_importance_ratio_is_clipped() -> None:
    spec = SubtaskSpec(feature_index=0, threshold=99.0, max_option_steps=4)
    cfg = Step10STOMPConfig(
        subtask_specs=(spec,),
        observation_dim=2,
        n_primitive_actions=2,
        epsilon_option=0.5,
        option_target_epsilon=0.0,
        option_importance_clip=1.25,
    )
    agent, state = _setup(cfg, seed=21)
    q_weights = state.option_policies.q_weights.at[0, 1, 0].set(1.0)
    state = state.replace(
        base_last_obs=jnp.array([1.0, 0.0], dtype=jnp.float32),
        base_last_action=jnp.array(2, dtype=jnp.int32),
        last_primitive_action=jnp.array(1, dtype=jnp.int32),
        executing_option=jnp.array(0, dtype=jnp.int32),
        option_last_intra_action=jnp.array(1, dtype=jnp.int32),
        option_policies=state.option_policies.replace(q_weights=q_weights),
    )
    assert bool(agent.state_valid(state))
    result = step10_update(
        agent,
        state,
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array([0.0, 0.0], dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    chex.assert_trees_all_close(
        result.option_importance_ratio,
        jnp.array(1.25, dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# Option termination path
# ---------------------------------------------------------------------------


def test_option_terminates_when_threshold_exceeded() -> None:
    spec = SubtaskSpec(feature_index=0, threshold=0.1, max_option_steps=100)
    cfg = Step10STOMPConfig(
        subtask_specs=(spec,),
        observation_dim=2,
        n_primitive_actions=2,
        epsilon_base=1.0,  # force random exploration to hit option action
    )
    agent = make_step10_stomp_agent(cfg)
    key = jr.key(7)
    state = init_step10_state(agent, key=key, initial_observation=jnp.zeros(2))

    # Force option execution by injecting executing_option=0 into state
    state_with_option = state.replace(
        base_last_action=jnp.array(2, dtype=jnp.int32),
        executing_option=jnp.array(0, dtype=jnp.int32),
        option_start_obs=jnp.zeros(2, dtype=jnp.float32),
        option_last_intra_action=state.last_primitive_action,
        option_steps=jnp.array(0, dtype=jnp.int32),
        option_cumreward=jnp.array(0.0, dtype=jnp.float32),
        option_discount=jnp.array(1.0, dtype=jnp.float32),
    )
    assert bool(agent.state_valid(state_with_option))
    # Observation with feature 0 = 0.5 > threshold 0.1 → terminates immediately
    high_obs = jnp.array([0.5, 0.0], dtype=jnp.float32)
    result = step10_update(agent, state_with_option, jnp.array(0.0), high_obs)
    assert bool(result.update_applied)
    assert bool(result.option_terminated)


def test_option_does_not_terminate_below_threshold() -> None:
    spec = SubtaskSpec(feature_index=0, threshold=0.9, max_option_steps=100)
    cfg = Step10STOMPConfig(
        subtask_specs=(spec,),
        observation_dim=2,
        n_primitive_actions=2,
    )
    agent = make_step10_stomp_agent(cfg)
    state = init_step10_state(
        agent, key=jr.key(3), initial_observation=jnp.zeros(2)
    )
    state_with_option = state.replace(
        base_last_action=jnp.array(2, dtype=jnp.int32),
        executing_option=jnp.array(0, dtype=jnp.int32),
        option_start_obs=jnp.zeros(2, dtype=jnp.float32),
        option_last_intra_action=state.last_primitive_action,
        option_steps=jnp.array(0, dtype=jnp.int32),
        option_cumreward=jnp.array(0.0, dtype=jnp.float32),
        option_discount=jnp.array(1.0, dtype=jnp.float32),
    )
    assert bool(agent.state_valid(state_with_option))
    low_obs = jnp.array([0.1, 0.0], dtype=jnp.float32)
    result = step10_update(agent, state_with_option, jnp.array(0.0), low_obs)
    assert bool(result.update_applied)
    assert not bool(result.option_terminated)


def test_option_terminates_at_max_steps() -> None:
    spec = SubtaskSpec(feature_index=0, threshold=99.0, max_option_steps=1)
    cfg = Step10STOMPConfig(
        subtask_specs=(spec,),
        observation_dim=2,
        n_primitive_actions=2,
    )
    agent = make_step10_stomp_agent(cfg)
    state = init_step10_state(agent, key=jr.key(0), initial_observation=jnp.zeros(2))
    state_with_option = state.replace(
        base_last_action=jnp.array(2, dtype=jnp.int32),
        executing_option=jnp.array(0, dtype=jnp.int32),
        option_start_obs=jnp.zeros(2, dtype=jnp.float32),
        option_last_intra_action=state.last_primitive_action,
        option_steps=jnp.array(0, dtype=jnp.int32),
        option_cumreward=jnp.array(0.0, dtype=jnp.float32),
        option_discount=jnp.array(1.0, dtype=jnp.float32),
    )
    assert bool(agent.state_valid(state_with_option))
    result = step10_update(agent, state_with_option, jnp.array(0.0), jnp.zeros(2))
    assert bool(result.update_applied)
    assert bool(result.option_terminated)


# ---------------------------------------------------------------------------
# Scan / batch run
# ---------------------------------------------------------------------------


def test_run_step10_scan_output_shapes() -> None:
    cfg = _make_cfg()
    agent, state = _setup(cfg)
    n_steps = 20
    rewards = jnp.zeros(n_steps, dtype=jnp.float32)
    obs = jr.normal(jr.key(5), (n_steps, cfg.observation_dim), dtype=jnp.float32)
    result = run_step10_scan(agent, state, rewards, obs)
    assert isinstance(result, STOMPArrayResult)
    chex.assert_shape(result.td_errors, (n_steps,))
    chex.assert_shape(result.average_rewards, (n_steps,))
    chex.assert_shape(result.primitive_actions, (n_steps,))
    chex.assert_shape(result.executing_options, (n_steps,))
    chex.assert_shape(result.pseudo_rewards, (n_steps,))
    chex.assert_shape(result.option_terminations, (n_steps,))
    chex.assert_shape(result.option_importance_ratios, (n_steps,))


def test_run_step10_scan_final_step_count() -> None:
    cfg = _make_cfg()
    agent, state = _setup(cfg)
    n_steps = 16
    result = run_step10_scan(
        agent,
        state,
        jnp.zeros(n_steps),
        jr.normal(jr.key(9), (n_steps, cfg.observation_dim)),
    )
    assert int(result.state.step_count) == n_steps


def test_run_step10_scan_all_finite() -> None:
    cfg = _make_cfg(obs_dim=4)
    agent, state = _setup(cfg, seed=42)
    n_steps = 50
    rewards = jr.normal(jr.key(0), (n_steps,))
    obs = jr.normal(jr.key(1), (n_steps, cfg.observation_dim))
    result = run_step10_scan(agent, state, rewards, obs)
    chex.assert_tree_all_finite(result.td_errors)
    chex.assert_tree_all_finite(result.average_rewards)
    chex.assert_tree_all_finite(result.pseudo_rewards)


def test_run_step10_scan_primitive_actions_in_range() -> None:
    cfg = _make_cfg(n_prim=3)
    agent, state = _setup(cfg)
    n_steps = 30
    result = run_step10_scan(
        agent,
        state,
        jnp.zeros(n_steps),
        jr.normal(jr.key(8), (n_steps, cfg.observation_dim)),
    )
    assert bool(jnp.all(result.primitive_actions >= 0))
    assert bool(jnp.all(result.primitive_actions < 3))


def test_run_step10_scan_two_subtasks() -> None:
    cfg = _make_cfg(specs=(_SPEC1, _SPEC2), obs_dim=4)
    agent, state = _setup(cfg)
    n_steps = 40
    rewards = jr.normal(jr.key(2), (n_steps,))
    obs = jr.normal(jr.key(3), (n_steps, cfg.observation_dim))
    result = run_step10_scan(agent, state, rewards, obs)
    chex.assert_tree_all_finite(result.td_errors)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def test_run_step10_smoke_defaults() -> None:
    result = run_step10_smoke()
    assert isinstance(result, Step10SmokeResult)
    assert result.finite
    assert result.steps == 64
    assert result.td_errors_shape == (64,)
    assert result.primitive_actions_shape == (64,)


def test_run_step10_smoke_custom_config() -> None:
    cfg = Step10STOMPConfig(
        subtask_specs=(_SPEC1, _SPEC2),
        observation_dim=4,
        n_primitive_actions=2,
        epsilon_base=0.3,
        epsilon_option=0.3,
    )
    result = run_step10_smoke(cfg, steps=32, seed=1)
    assert result.finite
    assert result.td_errors_shape == (32,)
    assert result.executing_options_shape == (32,)


def test_run_step10_smoke_to_dict_roundtrip() -> None:
    result = run_step10_smoke(steps=8)
    d = result.to_dict()
    assert isinstance(d["agent_config"], dict)
    assert isinstance(d["finite"], bool)
    assert d["steps"] == 8


def test_run_step10_smoke_zero_steps_raises() -> None:
    with pytest.raises(ValueError, match="steps"):
        run_step10_smoke(steps=0)


def test_run_step10_smoke_different_seeds_differ() -> None:
    r0 = run_step10_smoke(steps=16, seed=0)
    r1 = run_step10_smoke(steps=16, seed=99)
    assert r0.seed != r1.seed


# ---------------------------------------------------------------------------
# State fineness under sustained learning
# ---------------------------------------------------------------------------


def test_step10_state_stays_finite_over_many_steps() -> None:
    cfg = Step10STOMPConfig(
        subtask_specs=(SubtaskSpec(feature_index=0, threshold=0.3, max_option_steps=4),),
        observation_dim=3,
        n_primitive_actions=2,
        base_step_size=0.01,
        option_step_size=0.01,
    )
    agent, state = _setup(cfg, seed=5)
    n_steps = 200
    rewards = jr.normal(jr.key(10), (n_steps,)) * 0.1
    obs = jr.normal(jr.key(11), (n_steps, cfg.observation_dim)) * 0.1
    result = run_step10_scan(agent, state, rewards, obs)
    chex.assert_tree_all_finite(result.state.base_learner_state)
    chex.assert_tree_all_finite(result.state.option_policies.q_weights)
    chex.assert_tree_all_finite(result.state.option_models.cumreward_ema)
    chex.assert_tree_all_finite(result.state.option_models.discount_ema)
    chex.assert_tree_all_finite(result.state.option_models.next_state_weights)


def test_step10_option_model_completions_nonnegative() -> None:
    cfg = Step10STOMPConfig(
        subtask_specs=(SubtaskSpec(feature_index=0, threshold=0.2, max_option_steps=2),),
        observation_dim=2,
        n_primitive_actions=2,
    )
    agent, state = _setup(cfg)
    n_steps = 100
    rewards = jnp.zeros(n_steps)
    obs = jr.normal(jr.key(20), (n_steps, cfg.observation_dim))
    result = run_step10_scan(agent, state, rewards, obs)
    assert bool(jnp.all(result.state.option_models.n_completions >= 0))


# ---------------------------------------------------------------------------
# Auto-discovery: subtasks_from_feature_scores
# ---------------------------------------------------------------------------


def test_subtasks_from_feature_scores_selects_top_k() -> None:
    scores = [0.1, 0.9, 0.3, 0.7, 0.2]
    specs = subtasks_from_feature_scores(scores, top_k=2)
    assert len(specs) == 2
    assert specs[0].feature_index == 1
    assert specs[1].feature_index == 3


def test_subtasks_from_feature_scores_returns_subtask_specs() -> None:
    scores = [0.5, 0.8, 0.2]
    specs = subtasks_from_feature_scores(scores, top_k=2, threshold=0.4, max_option_steps=8)
    assert all(isinstance(s, SubtaskSpec) for s in specs)
    assert specs[0].threshold == 0.4
    assert specs[0].max_option_steps == 8


def test_subtasks_from_feature_scores_min_score_filter() -> None:
    scores = [0.9, 0.05, 0.7]
    specs = subtasks_from_feature_scores(scores, top_k=3, min_score=0.1)
    assert len(specs) == 2
    feature_indices = [s.feature_index for s in specs]
    assert 1 not in feature_indices


def test_subtasks_from_feature_scores_fewer_than_top_k_eligible() -> None:
    scores = [0.9, 0.01, 0.02]
    specs = subtasks_from_feature_scores(scores, top_k=3, min_score=0.1)
    assert len(specs) == 1
    assert specs[0].feature_index == 0


def test_subtasks_from_feature_scores_works_with_jax_array() -> None:
    scores = jnp.array([0.1, 0.8, 0.5, 0.2])
    specs = subtasks_from_feature_scores(scores, top_k=2)
    assert len(specs) == 2
    assert specs[0].feature_index == 1
    assert specs[1].feature_index == 2


def test_subtasks_from_feature_scores_integrates_with_stomp() -> None:
    scores = jnp.array([0.1, 0.9, 0.4, 0.7])
    specs = subtasks_from_feature_scores(scores, top_k=2, threshold=0.5, max_option_steps=4)
    cfg = Step10STOMPConfig(
        subtask_specs=tuple(specs),
        observation_dim=4,
        n_primitive_actions=2,
    )
    agent = make_step10_stomp_agent(cfg)
    state = init_step10_state(agent, key=jr.key(0), initial_observation=jnp.zeros(4))
    result = step10_update(agent, state, jnp.array(0.0), jnp.ones(4))
    assert bool(jnp.isfinite(result.td_error))
