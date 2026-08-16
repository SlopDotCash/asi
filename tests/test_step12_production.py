"""Tests for the Step 12 Intelligence Amplification production facade."""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Any

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.intelligence_amplification import (
    ExoCerebellumConfig,
    IAAgent,
    IAArrayResult,
    IAConfig,
    IAState,
    IAUpdateResult,
    RecommendationProtocolConfig,
    init_recommendation_protocol_state,
    update_recommendation_protocol,
)
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.steps import step12 as step12_module
from alberta_framework.steps.step12 import (
    Step12IAConfig,
    Step12SmokeResult,
    init_step12_state,
    make_step12_ia_agent,
    run_step12_scan,
    run_step12_smoke,
    step12_update,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SPEC0 = SubtaskSpec(feature_index=0, threshold=0.5, pseudo_reward_scale=1.0, max_option_steps=8)
_SPEC1 = SubtaskSpec(feature_index=1, threshold=0.3, pseudo_reward_scale=2.0, max_option_steps=4)
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FLOAT32_MIN_SUBNORMAL = float(
    np.nextafter(np.float32(0.0), np.float32(1.0), dtype=np.float32)
)
_INT32_MAX = 2**31 - 1


def _make_step12_cfg(
    *,
    specs: tuple[SubtaskSpec, ...] = (_SPEC0,),
    obs_dim: int = 4,
    n_prim: int = 2,
    n_demons: int = 3,
) -> Step12IAConfig:
    return Step12IAConfig(
        subtask_specs=specs,
        observation_dim=obs_dim,
        n_primitive_actions=n_prim,
        n_demons=n_demons,
    )


def _make_ia_config(
    *,
    specs: tuple[SubtaskSpec, ...] = (_SPEC0,),
    obs_dim: int = 4,
    n_prim: int = 2,
    n_demons: int = 3,
) -> IAConfig:
    cerebellum = ExoCerebellumConfig(n_demons=n_demons, obs_dim=obs_dim)
    stomp = STOMPConfig(
        subtask_specs=specs,
        observation_dim=obs_dim,
        n_primitive_actions=n_prim,
    )
    cortex = OaKConfig(stomp=stomp)
    return IAConfig(cerebellum=cerebellum, cortex=cortex)


def _setup(
    cfg: Step12IAConfig | None = None,
    *,
    seed: int = 0,
) -> tuple[IAAgent, IAState]:
    if cfg is None:
        cfg = _make_step12_cfg()
    agent = make_step12_ia_agent(cfg)
    key = jr.key(seed)
    init_obs = jnp.zeros(cfg.observation_dim, dtype=jnp.float32)
    state = init_step12_state(agent, key=key, initial_observation=init_obs)
    return agent, state


# ---------------------------------------------------------------------------
# ExoCerebellumConfig validation
# ---------------------------------------------------------------------------


def test_exo_cerebellum_config_zero_demons_raises() -> None:
    with pytest.raises(ValueError, match="n_demons"):
        ExoCerebellumConfig(n_demons=0, obs_dim=4)


def test_exo_cerebellum_config_zero_obs_dim_raises() -> None:
    with pytest.raises(ValueError, match="obs_dim"):
        ExoCerebellumConfig(n_demons=3, obs_dim=0)


def test_exo_cerebellum_config_nonpositive_step_size_raises() -> None:
    with pytest.raises(ValueError, match="step_size"):
        ExoCerebellumConfig(n_demons=3, obs_dim=4, step_size=0.0)


# ---------------------------------------------------------------------------
# IAConfig validation
# ---------------------------------------------------------------------------


def test_ia_config_obs_dim_mismatch_raises() -> None:
    cerebellum = ExoCerebellumConfig(n_demons=3, obs_dim=4)
    stomp = STOMPConfig(subtask_specs=(_SPEC0,), observation_dim=6)
    cortex = OaKConfig(stomp=stomp)
    with pytest.raises(ValueError, match="obs_dim"):
        IAConfig(cerebellum=cerebellum, cortex=cortex)


def test_ia_config_matching_dims_ok() -> None:
    cfg = _make_ia_config(obs_dim=5)
    assert cfg.cerebellum.obs_dim == cfg.cortex.observation_dim == 5


# ---------------------------------------------------------------------------
# Step12IAConfig serialization
# ---------------------------------------------------------------------------


def test_step12_config_roundtrip_default() -> None:
    cfg = _make_step12_cfg()
    assert Step12IAConfig.from_config(cfg.to_config()) == cfg


def test_step12_config_roundtrip_two_specs() -> None:
    cfg = _make_step12_cfg(specs=(_SPEC0, _SPEC1), obs_dim=4)
    assert Step12IAConfig.from_config(cfg.to_config()) == cfg


def test_step12_config_roundtrip_preserves_all_fields() -> None:
    cfg = Step12IAConfig(
        n_demons=6,
        cerebellum_step_size=0.02,
        subtask_specs=(_SPEC0,),
        observation_dim=5,
        n_primitive_actions=3,
        base_step_size=0.1,
        base_avg_reward_step_size=0.005,
        option_step_size=0.08,
        option_gamma=0.95,
        epsilon_base=0.15,
        utility_ema_decay=0.97,
    )
    restored = Step12IAConfig.from_config(cfg.to_config())
    assert restored == cfg


def test_step12_config_type_tag_stripped() -> None:
    cfg = _make_step12_cfg()
    d = cfg.to_config()
    assert d["type"] == "Step12IAConfig"
    assert Step12IAConfig.from_config(d) == cfg


def test_step12_config_to_ia_config_dims_match() -> None:
    cfg = _make_step12_cfg(obs_dim=5, n_prim=3, n_demons=4)
    ia_cfg = cfg.to_ia_config()
    assert isinstance(ia_cfg, IAConfig)
    assert ia_cfg.cerebellum.obs_dim == 5
    assert ia_cfg.cortex.observation_dim == 5
    assert ia_cfg.cortex.n_primitive_actions == 3
    assert ia_cfg.cerebellum.n_demons == 4


_INVALID_STEP12_FIELDS: tuple[tuple[str, Any], ...] = (
    ("n_demons", 0),
    ("n_demons", -1),
    ("n_demons", True),
    ("n_demons", False),
    ("n_demons", "4"),
    ("n_demons", 4.5),
    ("n_demons", float("nan")),
    ("n_demons", float("inf")),
    ("n_demons", None),
    ("n_demons", 2**31),
    ("observation_dim", 0),
    ("observation_dim", -1),
    ("observation_dim", True),
    ("observation_dim", False),
    ("observation_dim", "4"),
    ("observation_dim", 4.5),
    ("observation_dim", float("nan")),
    ("observation_dim", float("inf")),
    ("observation_dim", None),
    ("observation_dim", 2**31),
    ("n_primitive_actions", 0),
    ("n_primitive_actions", -1),
    ("n_primitive_actions", True),
    ("n_primitive_actions", False),
    ("n_primitive_actions", "2"),
    ("n_primitive_actions", 2.5),
    ("n_primitive_actions", float("nan")),
    ("n_primitive_actions", float("inf")),
    ("n_primitive_actions", None),
    ("n_primitive_actions", 2**31),
    ("cerebellum_step_size", float("nan")),
    ("cerebellum_step_size", float("inf")),
    ("cerebellum_step_size", float("-inf")),
    ("cerebellum_step_size", True),
    ("cerebellum_step_size", False),
    ("cerebellum_step_size", 0.0),
    ("cerebellum_step_size", -1.0),
    ("cerebellum_step_size", "0.05"),
    ("cerebellum_step_size", None),
    ("cerebellum_step_size", 1e100),
    ("base_step_size", float("nan")),
    ("base_step_size", float("inf")),
    ("base_step_size", float("-inf")),
    ("base_step_size", True),
    ("base_step_size", False),
    ("base_step_size", -1.0),
    ("base_step_size", "0.05"),
    ("base_step_size", None),
    ("base_step_size", 1e100),
    ("base_avg_reward_step_size", float("nan")),
    ("base_avg_reward_step_size", float("inf")),
    ("base_avg_reward_step_size", True),
    ("base_avg_reward_step_size", False),
    ("base_avg_reward_step_size", -0.01),
    ("base_avg_reward_step_size", "0.01"),
    ("base_avg_reward_step_size", 1e100),
    ("option_step_size", float("nan")),
    ("option_step_size", float("inf")),
    ("option_step_size", True),
    ("option_step_size", False),
    ("option_step_size", -1.0),
    ("option_step_size", "0.05"),
    ("option_step_size", 1e100),
    ("option_gamma", float("nan")),
    ("option_gamma", float("inf")),
    ("option_gamma", float("-inf")),
    ("option_gamma", True),
    ("option_gamma", False),
    ("option_gamma", -0.1),
    ("option_gamma", 1.1),
    ("option_gamma", "0.99"),
    ("option_gamma", 1e100),
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
    ("epsilon_base", float("nan")),
    ("epsilon_base", float("inf")),
    ("epsilon_base", True),
    ("epsilon_base", False),
    ("epsilon_base", -0.1),
    ("epsilon_base", 1.1),
    ("epsilon_base", "0.1"),
    ("epsilon_base", 1e100),
    ("utility_ema_decay", float("nan")),
    ("utility_ema_decay", float("inf")),
    ("utility_ema_decay", float("-inf")),
    ("utility_ema_decay", True),
    ("utility_ema_decay", False),
    ("utility_ema_decay", -0.1),
    ("utility_ema_decay", 1.1),
    ("utility_ema_decay", "0.99"),
    ("utility_ema_decay", 1e100),
)


def _config_with(**overrides: Any) -> Step12IAConfig:
    payload: dict[str, Any] = {
        "subtask_specs": (_SPEC0,),
        "observation_dim": 4,
        "n_primitive_actions": 2,
        "n_demons": 3,
    }
    payload.update(overrides)
    return Step12IAConfig(**payload)


@pytest.mark.parametrize(("field", "value"), _INVALID_STEP12_FIELDS)
def test_step12_ia_fields_reject_invalid_inputs(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _config_with(**{field: value})


def test_step12_config_feature_index_out_of_bounds_raises() -> None:
    bad_spec = SubtaskSpec(feature_index=10)
    with pytest.raises(ValueError, match="feature_index"):
        Step12IAConfig(subtask_specs=(bad_spec,), observation_dim=4)


def test_step12_ia_rejects_non_tuple_subtask_specs() -> None:
    with pytest.raises(ValueError, match="subtask_specs"):
        Step12IAConfig(subtask_specs=[_SPEC0])  # type: ignore[arg-type]


def test_step12_ia_rejects_bool_and_nonfinite_spec_scalars() -> None:
    with pytest.raises(ValueError, match="feature_index"):
        Step12IAConfig(
            subtask_specs=(SubtaskSpec(feature_index=True),),
            observation_dim=4,
        )
    with pytest.raises(ValueError, match="threshold"):
        Step12IAConfig(
            subtask_specs=(SubtaskSpec(feature_index=0, threshold=True),),
            observation_dim=4,
        )
    with pytest.raises(ValueError, match="threshold"):
        Step12IAConfig(
            subtask_specs=(SubtaskSpec(feature_index=0, threshold=float("nan")),),
            observation_dim=4,
        )
    with pytest.raises(ValueError, match="threshold"):
        Step12IAConfig(
            subtask_specs=(SubtaskSpec(feature_index=0, threshold=float("inf")),),
            observation_dim=4,
        )
    with pytest.raises(ValueError, match="pseudo_reward_scale"):
        Step12IAConfig(
            subtask_specs=(SubtaskSpec(feature_index=0, pseudo_reward_scale=True),),
            observation_dim=4,
        )
    with pytest.raises(ValueError, match="pseudo_reward_scale"):
        Step12IAConfig(
            subtask_specs=(
                SubtaskSpec(feature_index=0, pseudo_reward_scale=float("nan")),
            ),
            observation_dim=4,
        )
    with pytest.raises(ValueError, match="max_option_steps"):
        Step12IAConfig(
            subtask_specs=(SubtaskSpec(feature_index=0, max_option_steps=True),),
            observation_dim=4,
        )


def test_step12_ia_fields_preserve_legal_endpoints() -> None:
    config = Step12IAConfig(
        n_demons=1,
        cerebellum_step_size=1e-12,
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
        option_step_size=0.0,
        option_gamma=0.0,
        option_planning_backups_per_step=0,
        epsilon_base=0.0,
        utility_ema_decay=0.0,
    )
    agent = make_step12_ia_agent(config)
    payload = config.to_config()
    json.dumps(payload, allow_nan=False)
    restored = Step12IAConfig.from_config(payload)
    assert restored.n_demons == 1
    assert restored.cerebellum_step_size == float(np.float32(1e-12))
    assert restored.observation_dim == 1
    assert restored.n_primitive_actions == 1
    assert restored.base_step_size == 0.0
    assert restored.base_avg_reward_step_size == 0.0
    assert restored.option_step_size == 0.0
    assert restored.option_gamma == 0.0
    assert restored.option_planning_backups_per_step == 0
    assert restored.epsilon_base == 0.0
    assert restored.utility_ema_decay == 0.0
    assert restored.subtask_specs[0].feature_index == 0
    assert restored.subtask_specs[0].threshold == float(np.float32(1e-12))
    assert restored.subtask_specs[0].pseudo_reward_scale == float(np.float32(1e-12))
    assert restored.subtask_specs[0].max_option_steps == 1
    assert agent.config.cortex.stomp.option_gamma == 0.0

    upper = Step12IAConfig(
        n_demons=10,
        cerebellum_step_size=1.0,
        subtask_specs=(_SPEC0,),
        observation_dim=4,
        n_primitive_actions=2,
        option_gamma=1.0,
        epsilon_base=1.0,
        utility_ema_decay=1.0,
        option_planning_backups_per_step=2**31 - 2,
    )
    make_step12_ia_agent(upper)
    assert upper.option_gamma == 1.0
    assert upper.epsilon_base == 1.0
    assert upper.utility_ema_decay == 1.0
    assert upper.option_planning_backups_per_step == 2**31 - 2


def test_step12_ia_fields_canonicalize_nonbuiltin_numbers() -> None:
    value = np.float64(0.5)
    spec = SubtaskSpec(
        feature_index=np.int64(1),
        threshold=value,
        pseudo_reward_scale=value,
        max_option_steps=np.int64(4),
    )
    config = Step12IAConfig(
        n_demons=np.int64(3),
        cerebellum_step_size=value,
        subtask_specs=(spec,),
        observation_dim=np.int64(3),
        n_primitive_actions=np.int64(2),
        base_step_size=value,
        base_avg_reward_step_size=value,
        option_step_size=value,
        option_gamma=value,
        option_planning_backups_per_step=np.int64(1),
        epsilon_base=value,
        utility_ema_decay=value,
    )
    agent = make_step12_ia_agent(config)
    payload = config.to_config()
    json.dumps(payload, allow_nan=False)
    assert config.n_demons == 3
    assert config.observation_dim == 3
    assert config.n_primitive_actions == 2
    assert config.option_planning_backups_per_step == 1
    assert config.option_gamma == 0.5
    assert config.utility_ema_decay == 0.5
    assert config.subtask_specs[0].feature_index == 1
    assert config.subtask_specs[0].threshold == 0.5
    assert config.subtask_specs[0].max_option_steps == 4
    assert type(payload["n_demons"]) is int
    assert type(payload["observation_dim"]) is int
    assert type(payload["n_primitive_actions"]) is int
    assert type(payload["option_planning_backups_per_step"]) is int
    assert type(payload["cerebellum_step_size"]) is float
    assert type(payload["base_step_size"]) is float
    assert type(payload["base_avg_reward_step_size"]) is float
    assert type(payload["option_step_size"]) is float
    assert type(payload["option_gamma"]) is float
    assert type(payload["epsilon_base"]) is float
    assert type(payload["utility_ema_decay"]) is float
    assert type(payload["subtask_specs"][0]["feature_index"]) is int
    assert type(payload["subtask_specs"][0]["threshold"]) is float
    assert type(payload["subtask_specs"][0]["pseudo_reward_scale"]) is float
    assert type(payload["subtask_specs"][0]["max_option_steps"]) is int
    assert agent.config.cortex.stomp.option_gamma == 0.5


@pytest.mark.parametrize(
    "field",
    [
        "cerebellum_step_size",
        "base_step_size",
        "base_avg_reward_step_size",
        "option_step_size",
    ],
)
def test_step12_rejects_values_that_overflow_float32(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _config_with(**{field: 1.0e100})


@pytest.mark.parametrize("value", [1.0e100, -1.0e100])
def test_step12_rejects_pseudo_reward_scale_that_overflows_float32(value: float) -> None:
    with pytest.raises(ValueError, match="pseudo_reward_scale"):
        _config_with(
            subtask_specs=(SubtaskSpec(feature_index=0, pseudo_reward_scale=value),),
        )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("cerebellum_step_size", 1.0e-50, "cerebellum_step_size"),
        (
            "subtask_specs",
            (SubtaskSpec(feature_index=0, threshold=1.0e-50),),
            "threshold",
        ),
        (
            "subtask_specs",
            (SubtaskSpec(feature_index=0, pseudo_reward_scale=1.0e-50),),
            "pseudo_reward_scale",
        ),
    ],
    ids=("cerebellum_step_size", "threshold", "pseudo_reward_scale"),
)
def test_step12_strict_positive_fields_reject_float32_zero_collapse(
    field: str,
    value: object,
    expected: str,
) -> None:
    with pytest.raises(ValueError, match=expected):
        _config_with(**{field: value})


def test_step12_float32_underflow_and_finite_boundaries_are_canonical() -> None:
    spec = SubtaskSpec(
        feature_index=0,
        threshold=_FLOAT32_MIN_SUBNORMAL,
        pseudo_reward_scale=_FLOAT32_MIN_SUBNORMAL,
    )
    config = _config_with(
        subtask_specs=(spec,),
        cerebellum_step_size=_FLOAT32_MIN_SUBNORMAL,
        base_step_size=1.0e-50,
        base_avg_reward_step_size=1.0e-50,
        option_step_size=1.0e-50,
    )
    assert config.cerebellum_step_size == _FLOAT32_MIN_SUBNORMAL
    assert config.base_step_size == 0.0
    assert config.base_avg_reward_step_size == 0.0
    assert config.option_step_size == 0.0
    assert config.subtask_specs[0].threshold == _FLOAT32_MIN_SUBNORMAL
    assert config.subtask_specs[0].pseudo_reward_scale == _FLOAT32_MIN_SUBNORMAL

    maximum = _config_with(
        subtask_specs=(
            SubtaskSpec(
                feature_index=0,
                threshold=_FLOAT32_MAX,
                pseudo_reward_scale=_FLOAT32_MAX,
            ),
        ),
        cerebellum_step_size=_FLOAT32_MAX,
        base_step_size=_FLOAT32_MAX,
        base_avg_reward_step_size=_FLOAT32_MAX,
        option_step_size=_FLOAT32_MAX,
    )
    assert maximum.cerebellum_step_size == _FLOAT32_MAX
    assert maximum.base_step_size == _FLOAT32_MAX
    assert maximum.subtask_specs[0].threshold == _FLOAT32_MAX
    assert maximum.subtask_specs[0].pseudo_reward_scale == _FLOAT32_MAX


def test_step12_int32_counter_boundaries_match_core_sinks() -> None:
    config = _config_with(
        subtask_specs=(SubtaskSpec(feature_index=0, max_option_steps=_INT32_MAX),),
        option_planning_backups_per_step=_INT32_MAX - 1,
    )
    make_step12_ia_agent(config)
    assert config.subtask_specs[0].max_option_steps == _INT32_MAX
    assert config.option_planning_backups_per_step == _INT32_MAX - 1

    with pytest.raises(ValueError, match="max_option_steps"):
        _config_with(
            subtask_specs=(
                SubtaskSpec(feature_index=0, max_option_steps=_INT32_MAX + 1),
            )
        )
    with pytest.raises(ValueError, match="option_planning_backups_per_step"):
        _config_with(option_planning_backups_per_step=_INT32_MAX)


def test_step12_builtin_json_roundtrip_preserves_canonical_config() -> None:
    config = _config_with(
        subtask_specs=(
            SubtaskSpec(
                feature_index=0,
                threshold=0.2,
                pseudo_reward_scale=0.3,
                max_option_steps=17,
            ),
        ),
        cerebellum_step_size=0.02,
        base_step_size=0.03,
        option_gamma=0.875,
        epsilon_base=0.125,
    )

    payload = json.loads(json.dumps(config.to_config(), allow_nan=False))
    assert Step12IAConfig.from_config(payload) == config


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_step_size", Fraction(-1, 10**400)),
        ("base_avg_reward_step_size", Fraction(-1, 10**400)),
        ("option_step_size", Fraction(-1, 10**400)),
        ("option_gamma", Fraction(-1, 10**400)),
        ("option_gamma", Fraction(10**400 + 1, 10**400)),
        ("epsilon_base", Fraction(-1, 10**400)),
        ("epsilon_base", Fraction(10**400 + 1, 10**400)),
        ("utility_ema_decay", Fraction(-1, 10**400)),
        ("utility_ema_decay", Fraction(10**400 + 1, 10**400)),
    ],
)
def test_step12_fields_check_exact_domains_before_float_conversion(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        _config_with(**{field: value})


def test_step12_wraps_real_conversion_overflow_as_config_error() -> None:
    with pytest.raises(ValueError, match="cerebellum_step_size"):
        _config_with(cerebellum_step_size=Fraction(10**400, 1))


def test_step12_narrows_original_real_directly_without_double_rounding() -> None:
    midpoint_plus = (
        np.longdouble(1.0)
        + np.longdouble(2.0) ** -24
        + np.longdouble(2.0) ** -60
    )
    assert np.float32(midpoint_plus) != np.float32(float(midpoint_plus))

    config = _config_with(
        subtask_specs=(
            SubtaskSpec(feature_index=0, pseudo_reward_scale=midpoint_plus),
        )
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
def test_step12_rounds_fraction_midpoints_once(
    pseudo_reward_scale: Fraction,
    expected: float,
) -> None:
    config = _config_with(
        subtask_specs=(
            SubtaskSpec(
                feature_index=0,
                pseudo_reward_scale=pseudo_reward_scale,
            ),
        )
    )
    assert config.subtask_specs[0].pseudo_reward_scale == expected


def test_step12_fraction_float32_overflow_midpoint_is_exact() -> None:
    maximum = Fraction((2**24 - 1) * 2**104)
    overflow_midpoint = maximum + 2**103

    just_below = _config_with(cerebellum_step_size=overflow_midpoint - 1)
    assert just_below.cerebellum_step_size == _FLOAT32_MAX
    with pytest.raises(ValueError, match="cerebellum_step_size"):
        _config_with(cerebellum_step_size=overflow_midpoint)


def test_step12_fraction_subnormal_midpoint_obeys_ties_to_even() -> None:
    half_min_subnormal = Fraction(1, 2**150)

    collapsed = _config_with(base_step_size=half_min_subnormal)
    assert collapsed.base_step_size == 0.0
    with pytest.raises(ValueError, match="cerebellum_step_size"):
        _config_with(cerebellum_step_size=half_min_subnormal)
    accepted = _config_with(
        cerebellum_step_size=half_min_subnormal + Fraction(1, 2**200)
    )
    assert accepted.cerebellum_step_size == _FLOAT32_MIN_SUBNORMAL


def test_step12_smoke_health_gate_reports_any_refused_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_scan = step12_module.run_step12_scan

    def _refuse_update(*args: Any, **kwargs: Any) -> Any:
        result = original_scan(*args, **kwargs)
        return result.replace(updates_applied=result.updates_applied.at[0].set(False))

    monkeypatch.setattr(step12_module, "run_step12_scan", _refuse_update)

    assert not run_step12_smoke(steps=2).finite


def test_step12_smallest_positive_float32_values_execute_healthy_update() -> None:
    config = _config_with(
        subtask_specs=(
            SubtaskSpec(
                feature_index=0,
                threshold=_FLOAT32_MIN_SUBNORMAL,
                pseudo_reward_scale=_FLOAT32_MIN_SUBNORMAL,
                max_option_steps=4,
            ),
        ),
        cerebellum_step_size=_FLOAT32_MIN_SUBNORMAL,
    )
    agent, state = _setup(config, seed=43)
    obs = jnp.full((4,), 0.1, dtype=jnp.float32)

    result = step12_update(
        agent,
        state,
        obs,
        jnp.asarray(0.25, dtype=jnp.float32),
        obs * jnp.asarray(1.5, dtype=jnp.float32),
    )

    assert bool(result.update_applied)
    assert bool(result.cerebellum_update_applied)
    assert bool(result.cortex_update_applied)
    chex.assert_tree_all_finite(result.state.cerebellum_state.weights)
    chex.assert_tree_all_finite(
        result.state.cortex_state.stomp_state.base_learner_state
    )


# ---------------------------------------------------------------------------
# Factory and initialization
# ---------------------------------------------------------------------------


def test_make_step12_ia_agent_default() -> None:
    agent = make_step12_ia_agent()
    assert isinstance(agent, IAAgent)
    assert agent.config.cerebellum.n_demons == 4  # default


def test_make_step12_ia_agent_custom() -> None:
    cfg = _make_step12_cfg(n_demons=6, obs_dim=5, n_prim=3)
    agent = make_step12_ia_agent(cfg)
    assert agent.config.cerebellum.n_demons == 6
    assert agent.config.cerebellum.obs_dim == 5


def test_init_step12_state_shapes() -> None:
    cfg = _make_step12_cfg(obs_dim=4, n_demons=3)
    agent, state = _setup(cfg)
    chex.assert_shape(state.cerebellum_state.weights, (3, 4))
    chex.assert_shape(state.cortex_state.utility_ema, (1,))


def test_init_step12_state_step_count_zero() -> None:
    _, state = _setup()
    assert int(state.step_count) == 0


def test_init_step12_state_two_specs() -> None:
    cfg = _make_step12_cfg(specs=(_SPEC0, _SPEC1), n_demons=2)
    agent, state = _setup(cfg)
    chex.assert_shape(state.cortex_state.utility_ema, (2,))
    chex.assert_shape(state.cerebellum_state.weights, (2, 4))


# ---------------------------------------------------------------------------
# Single-step update
# ---------------------------------------------------------------------------


def test_step12_update_returns_update_result() -> None:
    agent, state = _setup()
    obs = jnp.ones(4, dtype=jnp.float32) * 0.1
    reward = jnp.array(0.5)
    next_obs = jnp.ones(4, dtype=jnp.float32) * 0.2
    result = step12_update(agent, state, obs, reward, next_obs)
    assert isinstance(result, IAUpdateResult)


def test_step12_update_predictions_shape() -> None:
    cfg = _make_step12_cfg(n_demons=5)
    agent, state = _setup(cfg)
    obs = jnp.zeros(4, dtype=jnp.float32)
    result = step12_update(agent, state, obs, jnp.array(0.0), obs)
    chex.assert_shape(result.predictions, (5,))


def test_step12_update_recommendation_in_range() -> None:
    cfg = _make_step12_cfg(n_prim=3)
    agent, state = _setup(cfg)
    obs = jnp.zeros(4, dtype=jnp.float32)
    result = step12_update(agent, state, obs, jnp.array(0.0), obs)
    assert 0 <= int(result.recommendation) < 3


def test_step12_update_augmented_obs_shape() -> None:
    cfg = _make_step12_cfg(obs_dim=4, n_demons=3)
    agent, state = _setup(cfg)
    obs = jnp.zeros(4, dtype=jnp.float32)
    result = step12_update(agent, state, obs, jnp.array(0.0), obs)
    # augmented = concat(obs, predictions) → shape (4 + 3,)
    chex.assert_shape(result.augmented_obs, (7,))


def test_step12_update_augmented_obs_is_concat() -> None:
    cfg = _make_step12_cfg(obs_dim=4, n_demons=3)
    agent, state = _setup(cfg)
    obs = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    result = step12_update(agent, state, obs, jnp.array(0.0), obs)
    # First 4 elements should equal obs
    chex.assert_trees_all_close(result.augmented_obs[:4], obs, atol=1e-5)
    # Last 3 elements should equal predictions
    chex.assert_trees_all_close(result.augmented_obs[4:], result.predictions, atol=1e-5)


def test_step12_update_state_finite() -> None:
    agent, state = _setup()
    obs = jnp.ones(4) * 0.3
    result = step12_update(agent, state, obs, jnp.array(1.0), obs * 1.1)
    chex.assert_tree_all_finite(result.state.cerebellum_state.weights)
    chex.assert_tree_all_finite(result.state.cortex_state.stomp_state.base_learner_state)


def test_step12_update_step_count_increments() -> None:
    agent, state = _setup()
    obs = jnp.zeros(4)
    result = step12_update(agent, state, obs, jnp.array(0.0), obs)
    assert int(result.state.step_count) == 1


def test_step12_update_recommendation_is_int32() -> None:
    agent, state = _setup()
    obs = jnp.zeros(4)
    result = step12_update(agent, state, obs, jnp.array(0.0), obs)
    assert result.recommendation.dtype == jnp.int32


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


def test_run_step12_scan_output_shapes() -> None:
    cfg = _make_step12_cfg(obs_dim=4, n_demons=3, n_prim=2)
    agent, state = _setup(cfg)
    n_steps = 20
    obs = jr.normal(jr.key(1), (n_steps, 4))
    rewards = jnp.zeros(n_steps)
    next_obs = jr.normal(jr.key(2), (n_steps, 4))
    result = run_step12_scan(agent, state, obs, rewards, next_obs)
    assert isinstance(result, IAArrayResult)
    chex.assert_shape(result.predictions, (n_steps, 3))
    chex.assert_shape(result.cerebellum_errors, (n_steps, 3))
    chex.assert_shape(result.recommendations, (n_steps,))
    chex.assert_shape(result.augmented_obs, (n_steps, 7))  # 4 + 3
    chex.assert_shape(result.cortex_td_errors, (n_steps,))


def test_run_step12_scan_two_specs_shapes() -> None:
    cfg = _make_step12_cfg(specs=(_SPEC0, _SPEC1), obs_dim=4, n_demons=2, n_prim=2)
    agent, state = _setup(cfg)
    n_steps = 16
    obs = jr.normal(jr.key(3), (n_steps, 4))
    result = run_step12_scan(
        agent,
        state,
        obs,
        jnp.zeros(n_steps),
        jr.normal(jr.key(4), (n_steps, 4)),
    )
    chex.assert_shape(result.predictions, (n_steps, 2))
    chex.assert_shape(result.augmented_obs, (n_steps, 6))  # 4 + 2


def test_run_step12_scan_all_finite() -> None:
    cfg = _make_step12_cfg(obs_dim=4, n_demons=3)
    agent, state = _setup(cfg, seed=7)
    n_steps = 50
    obs = jr.normal(jr.key(5), (n_steps, 4)) * 0.1
    rewards = jr.normal(jr.key(6), (n_steps,)) * 0.1
    next_obs = jr.normal(jr.key(7), (n_steps, 4)) * 0.1
    result = run_step12_scan(agent, state, obs, rewards, next_obs)
    chex.assert_tree_all_finite(result.predictions)
    chex.assert_tree_all_finite(result.cerebellum_errors)
    chex.assert_tree_all_finite(result.cortex_td_errors)
    chex.assert_tree_all_finite(result.augmented_obs)


def test_run_step12_scan_final_step_count() -> None:
    cfg = _make_step12_cfg()
    agent, state = _setup(cfg)
    n_steps = 16
    obs = jr.normal(jr.key(8), (n_steps, 4))
    result = run_step12_scan(
        agent,
        state,
        obs,
        jnp.zeros(n_steps),
        jr.normal(jr.key(9), (n_steps, 4)),
    )
    assert int(result.state.step_count) == n_steps


def test_run_step12_scan_recommendations_in_range() -> None:
    cfg = _make_step12_cfg(n_prim=3)
    agent, state = _setup(cfg)
    n_steps = 30
    obs = jr.normal(jr.key(10), (n_steps, 4))
    result = run_step12_scan(
        agent,
        state,
        obs,
        jnp.zeros(n_steps),
        jr.normal(jr.key(11), (n_steps, 4)),
    )
    assert bool(jnp.all(result.recommendations >= 0))
    assert bool(jnp.all(result.recommendations < 3))


def test_run_step12_scan_recommendations_are_int32() -> None:
    cfg = _make_step12_cfg(n_prim=2)
    agent, state = _setup(cfg)
    n_steps = 8
    obs = jr.normal(jr.key(12), (n_steps, 4))
    result = run_step12_scan(
        agent,
        state,
        obs,
        jnp.zeros(n_steps),
        jr.normal(jr.key(13), (n_steps, 4)),
    )
    assert result.recommendations.dtype == jnp.int32


# ---------------------------------------------------------------------------
# Cerebellum learns over time
# ---------------------------------------------------------------------------


def test_cerebellum_prediction_error_finite() -> None:
    cfg = _make_step12_cfg(obs_dim=4, n_demons=4)
    agent, state = _setup(cfg)
    obs = jnp.array([0.1, 0.2, 0.3, 0.4])
    result = step12_update(agent, state, obs, jnp.array(0.0), obs * 1.1)
    assert bool(jnp.all(jnp.isfinite(result.cerebellum_errors)))


def test_cerebellum_weights_change_after_update() -> None:
    cfg = _make_step12_cfg(obs_dim=4, n_demons=3, n_prim=2)
    agent, state = _setup(cfg)
    obs = jnp.array([1.0, 0.0, 0.0, 0.0])
    next_obs = jnp.array([0.0, 1.0, 0.0, 0.0])
    result = step12_update(agent, state, obs, jnp.array(0.0), next_obs)
    assert not bool(
        jnp.all(result.state.cerebellum_state.weights == state.cerebellum_state.weights)
    )


# ---------------------------------------------------------------------------
# Recommendation acceptance / rejection protocol
# ---------------------------------------------------------------------------


def test_recommendation_protocol_config_roundtrip() -> None:
    cfg = RecommendationProtocolConfig(acceptance_ema_decay=0.5)
    assert RecommendationProtocolConfig.from_config(cfg.to_config()) == cfg


def test_recommendation_protocol_invalid_decay_raises() -> None:
    with pytest.raises(ValueError, match="acceptance_ema_decay"):
        RecommendationProtocolConfig(acceptance_ema_decay=1.0)


def test_recommendation_protocol_accepts_matching_action() -> None:
    cfg = RecommendationProtocolConfig(acceptance_ema_decay=0.5)
    state = init_recommendation_protocol_state()
    result = update_recommendation_protocol(
        cfg,
        state,
        jnp.array(1, dtype=jnp.int32),
        jnp.array(1, dtype=jnp.int32),
    )
    assert bool(result.accepted)
    assert int(result.effective_action) == 1
    assert int(result.state.accepted_count) == 1
    assert int(result.state.rejected_count) == 0
    assert float(result.state.acceptance_ema) == pytest.approx(0.5)


def test_recommendation_protocol_rejects_different_action() -> None:
    cfg = RecommendationProtocolConfig(acceptance_ema_decay=0.5)
    state = init_recommendation_protocol_state()
    result = update_recommendation_protocol(
        cfg,
        state,
        jnp.array(1, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
    )
    assert not bool(result.accepted)
    assert int(result.effective_action) == 0
    assert int(result.state.accepted_count) == 0
    assert int(result.state.rejected_count) == 1


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def test_run_step12_smoke_defaults() -> None:
    result = run_step12_smoke()
    assert isinstance(result, Step12SmokeResult)
    assert result.finite
    assert result.steps == 64
    assert result.predictions_shape == (64, 4)  # n_demons=4 default
    assert result.augmented_obs_shape == (64, 8)  # 4 obs + 4 demons


def test_run_step12_smoke_custom_config() -> None:
    cfg = Step12IAConfig(
        n_demons=3,
        subtask_specs=(_SPEC0, _SPEC1),
        observation_dim=4,
        n_primitive_actions=2,
    )
    result = run_step12_smoke(cfg, steps=32, seed=1)
    assert result.finite
    assert result.predictions_shape == (32, 3)
    assert result.augmented_obs_shape == (32, 7)  # 4 + 3


def test_run_step12_smoke_to_dict_roundtrip() -> None:
    result = run_step12_smoke(steps=8)
    d = result.to_dict()
    assert isinstance(d["agent_config"], dict)
    assert d["finite"] is True
    assert isinstance(d["predictions_shape"], list)
    assert isinstance(d["augmented_obs_shape"], list)


def test_run_step12_smoke_zero_steps_raises() -> None:
    with pytest.raises(ValueError, match="steps"):
        run_step12_smoke(steps=0)


def test_run_step12_smoke_cerebellum_errors_shape() -> None:
    result = run_step12_smoke(steps=16)
    assert result.cerebellum_errors_shape == (16, 4)


def test_run_step12_smoke_recommendations_shape() -> None:
    result = run_step12_smoke(steps=16)
    assert result.recommendations_shape == (16,)


# ---------------------------------------------------------------------------
# Long-horizon fineness
# ---------------------------------------------------------------------------


def test_step12_state_stays_finite_200_steps() -> None:
    cfg = Step12IAConfig(
        n_demons=4,
        cerebellum_step_size=0.01,
        subtask_specs=(SubtaskSpec(feature_index=0, threshold=0.3, max_option_steps=4),),
        observation_dim=4,
        n_primitive_actions=2,
        base_step_size=0.01,
        option_step_size=0.01,
        utility_ema_decay=0.95,
    )
    agent, state = _setup(cfg, seed=5)
    n_steps = 200
    obs = jr.normal(jr.key(20), (n_steps, 4)) * 0.1
    rewards = jr.normal(jr.key(21), (n_steps,)) * 0.1
    next_obs = jr.normal(jr.key(22), (n_steps, 4)) * 0.1
    result = run_step12_scan(agent, state, obs, rewards, next_obs)
    chex.assert_tree_all_finite(result.state.cerebellum_state.weights)
    chex.assert_tree_all_finite(result.state.cortex_state.stomp_state.base_learner_state)
    chex.assert_tree_all_finite(result.predictions)
    chex.assert_tree_all_finite(result.cortex_td_errors)


def test_step12_config_preserves_float32_boundaries() -> None:
    f32_max = float(np.finfo(np.float32).max)
    spec = SubtaskSpec(
        feature_index=2**31 - 2,
        threshold=f32_max,
        pseudo_reward_scale=f32_max,
        max_option_steps=2**31 - 1,
    )
    config = Step12IAConfig(
        n_demons=2**31 - 1,
        cerebellum_step_size=f32_max,
        subtask_specs=(spec,),
        observation_dim=2**31 - 1,
        n_primitive_actions=2**31 - 1,
        base_step_size=f32_max,
        base_avg_reward_step_size=f32_max,
        option_step_size=f32_max,
        option_gamma=1.0,
        option_planning_backups_per_step=2**31 - 2,
        epsilon_base=1.0,
        utility_ema_decay=1.0,
    )
    assert config.n_demons == 2**31 - 1
    assert config.observation_dim == 2**31 - 1
    assert config.option_planning_backups_per_step == 2**31 - 2
    assert config.cerebellum_step_size == f32_max


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"steps": 0}, "steps"),
        ({"steps": -1}, "steps"),
        ({"steps": 2**31}, "steps"),
        ({"steps": True}, "steps"),
        ({"steps": "64"}, "steps"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**31}, "seed"),
        ({"seed": True}, "seed"),
    ],
)
def test_step12_smoke_rejects_invalid_inputs(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        run_step12_smoke(**kwargs)


@pytest.mark.parametrize(
    "ratio",
    [
        pytest.param((-1, 1), id="negative-ratio"),
        pytest.param((2, 1), id="above-unit-ratio"),
        pytest.param((-1, 2**200), id="negative-rounds-to-negative-zero"),
        pytest.param((2**200 + 1, 2**200), id="above-one-rounds-to-one"),
    ],
)
def test_step12_rejects_adversarial_ratio_floats(
    ratio: tuple[int, int]
) -> None:
    class HiddenBoundaryFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            return ratio

    with pytest.raises(ValueError, match=r"option_gamma must be in \[0, 1\]"):
        Step12IAConfig(option_gamma=HiddenBoundaryFloat(0.5))


def test_step12_rejects_class_property_spoofing_float() -> None:
    class ClassSpoof:
        @property
        def __class__(self) -> type[float]:
            return float

        def as_integer_ratio(self) -> tuple[int, int]:
            return (1, 2)

    value = ClassSpoof()
    with pytest.raises(ValueError, match="must be a real number"):
        Step12IAConfig(option_gamma=value)  # type: ignore[arg-type]


def test_step12_rejects_spoofed_int_class_with_negative_ratio() -> None:
    class SpoofedIntFloat(float):
        @property
        def __class__(self) -> type[int]:
            return int

        def as_integer_ratio(self) -> tuple[int, int]:
            return (-1, 2**200)

    with pytest.raises(ValueError, match="base_step_size must be non-negative"):
        Step12IAConfig(base_step_size=SpoofedIntFloat(0.5))


def test_step12_rejects_integer_subclass_conversion_hooks() -> None:
    class LyingInt(int):
        def __int__(self) -> int:
            return 3

    with pytest.raises(ValueError, match="n_demons"):
        Step12IAConfig(n_demons=LyingInt(-1))


def test_step12_rejects_spoofed_ratio_components() -> None:
    class SpoofedComponent:
        @property
        def __class__(self) -> type[int]:
            return int

        def __int__(self) -> int:
            return 1

    class BadRatioFloat(float):
        def as_integer_ratio(self) -> tuple[Any, Any]:
            return (SpoofedComponent(), 2)

    with pytest.raises(ValueError, match="must be finite"):
        Step12IAConfig(option_gamma=BadRatioFloat(0.5))
