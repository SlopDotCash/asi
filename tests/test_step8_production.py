"""Production-facing Step 8 world-model facade tests.

Covers the one-step environment-model facade on real constructors. Invalid
dimension and scientific-scalar cases are written to fail on current main
(bool, non-real, non-finite, and out-of-domain values accepted) and pass
after the facade rejects them. Legal endpoints stay constructible.
"""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Any, cast

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.steps.step8 import (
    Step8SmokeResult,
    Step8WorldModelConfig,
    init_step8_state,
    make_step8_world_model,
    run_step8_scan,
    run_step8_smoke,
    step8_ensemble_predict,
    step8_update,
)

_INVALID_WORLD_MODEL_FIELDS: tuple[tuple[str, Any], ...] = (
    ("observation_dim", True),
    ("observation_dim", False),
    ("observation_dim", 0),
    ("observation_dim", -1),
    ("observation_dim", 1.5),
    ("observation_dim", "4"),
    ("observation_dim", 2**31),
    ("n_actions", True),
    ("n_actions", False),
    ("n_actions", 0),
    ("n_actions", -1),
    ("n_actions", 1.5),
    ("n_actions", "2"),
    ("n_actions", 2**31),
    ("action_dim", True),
    ("action_dim", False),
    ("action_dim", 0),
    ("action_dim", -1),
    ("action_dim", 1.5),
    ("action_dim", "1"),
    ("action_dim", 2**31),
    ("hidden_sizes", (True,)),
    ("hidden_sizes", (False,)),
    ("hidden_sizes", (0,)),
    ("hidden_sizes", (-1,)),
    ("hidden_sizes", (1.5,)),
    ("hidden_sizes", ("64",)),
    ("hidden_sizes", (2**31,)),
    ("step_size", float("nan")),
    ("step_size", float("inf")),
    ("step_size", True),
    ("step_size", False),
    ("step_size", -1.0),
    ("step_size", 1e100),
    ("sparsity", float("nan")),
    ("sparsity", True),
    ("sparsity", -0.1),
    ("sparsity", 1.1),
    ("sparsity", 1e100),
    ("utility_decay", float("nan")),
    ("utility_decay", float("inf")),
    ("utility_decay", True),
    ("utility_decay", False),
    ("utility_decay", -0.1),
    ("utility_decay", 1.0),
    ("utility_decay", 1e100),
    ("leaky_relu_slope", float("nan")),
    ("leaky_relu_slope", float("inf")),
    ("leaky_relu_slope", float("-inf")),
    ("leaky_relu_slope", True),
    ("leaky_relu_slope", False),
    ("leaky_relu_slope", "0.01"),
    ("leaky_relu_slope", -0.01),
    ("use_layer_norm", 1),
    ("use_layer_norm", 0),
    ("use_layer_norm", 1.0),
    ("use_layer_norm", "yes"),
    ("use_layer_norm", None),
    ("predict_delta", 1),
    ("predict_delta", 0),
    ("predict_delta", 1.0),
    ("predict_delta", "yes"),
    ("predict_delta", None),
    ("leaky_relu_slope", 1e100),
)


def test_step8_config_roundtrip_and_smoke() -> None:
    cfg = Step8WorldModelConfig(
        observation_dim=3,
        n_actions=2,
        hidden_sizes=(),
        step_size=0.05,
        sparsity=0.0,
        predict_delta=True,
    )
    assert Step8WorldModelConfig.from_dict(cfg.to_dict()) == cfg

    smoke = run_step8_smoke(cfg, steps=8, seed=0)
    assert smoke.finite
    assert smoke.reward_predictions_shape == (8,)
    assert smoke.next_observation_predictions_shape == (8, 3)


def test_step8_one_step_and_scan_facade() -> None:
    cfg = Step8WorldModelConfig(
        observation_dim=2,
        n_actions=2,
        hidden_sizes=(),
        step_size=0.05,
        sparsity=0.0,
    )
    model = make_step8_world_model(cfg)
    state = init_step8_state(model, key=jr.key(1))

    one = step8_update(
        model,
        state,
        jnp.array([0.0, 1.0], dtype=jnp.float32),
        jnp.array(1, dtype=jnp.int32),
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([1.0, 0.5], dtype=jnp.float32),
    )
    assert int(one.state.step_count) == 1

    observations = jnp.zeros((4, 2), dtype=jnp.float32)
    actions = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
    rewards = actions.astype(jnp.float32)
    next_observations = jnp.stack([rewards, 1.0 - rewards], axis=1)
    result = run_step8_scan(
        model,
        one.state,
        observations,
        actions,
        rewards,
        next_observations,
    )
    chex.assert_shape(result.reward_errors, (4,))
    chex.assert_shape(result.next_observation_errors, (4, 2))
    chex.assert_tree_all_finite(result.reward_predictions)


def test_step8_ensemble_prediction_reports_disagreement() -> None:
    cfg = Step8WorldModelConfig(
        observation_dim=2,
        n_actions=2,
        hidden_sizes=(),
        step_size=0.05,
        sparsity=0.0,
    )
    model = make_step8_world_model(cfg)
    state_a = init_step8_state(model, key=jr.key(1))
    state_b = init_step8_state(model, key=jr.key(2))

    prediction = step8_ensemble_predict(
        model,
        [state_a, state_b],
        jnp.array([0.25, -0.5], dtype=jnp.float32),
        jnp.array(1, dtype=jnp.int32),
    )
    chex.assert_shape(prediction.reward_predictions, (2,))
    chex.assert_shape(prediction.next_observation_predictions, (2, 2))
    chex.assert_shape(prediction.mean_next_observation, (2,))
    assert float(prediction.total_disagreement) >= 0.0


def test_step8_ensemble_prediction_rejects_empty_state_list() -> None:
    cfg = Step8WorldModelConfig(observation_dim=2, n_actions=2)
    model = make_step8_world_model(cfg)
    try:
        step8_ensemble_predict(
            model,
            [],
            jnp.zeros((2,), dtype=jnp.float32),
            jnp.array(0, dtype=jnp.int32),
        )
    except ValueError as exc:
        assert "states must contain" in str(exc)
    else:
        raise AssertionError("empty Step 8 ensemble state list should fail")


def _config_with(**overrides: Any) -> Step8WorldModelConfig:
    payload: dict[str, Any] = {
        "observation_dim": 2,
        "n_actions": 2,
        "hidden_sizes": (),
    }
    payload.update(overrides)
    if overrides.get("n_actions") is None and "action_dim" not in overrides:
        payload["action_dim"] = 1
    return Step8WorldModelConfig(**payload)


@pytest.mark.parametrize(("field", "value"), _INVALID_WORLD_MODEL_FIELDS)
def test_step8_world_model_fields_reject_invalid_inputs(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        make_step8_world_model(_config_with(**{field: value}))


@pytest.mark.unit
@pytest.mark.parametrize("field", ["use_layer_norm", "predict_delta"])
def test_step8_bool_fields_reject_invalid_type_without_calling_repr(field: str) -> None:
    repr_calls = 0

    class ExplodingRepr:
        def __repr__(self) -> str:
            nonlocal repr_calls
            repr_calls += 1
            raise RuntimeError("untrusted repr hook executed")

    with pytest.raises(ValueError, match=field):
        _config_with(**{field: ExplodingRepr()})

    assert repr_calls == 0


@pytest.mark.unit
def test_step8_bool_fields_accept_exact_bools() -> None:
    config = _config_with(use_layer_norm=False, predict_delta=True)
    assert config.use_layer_norm is False
    assert config.predict_delta is True
    core = config.to_core_config()
    assert core.use_layer_norm is False
    assert core.predict_delta is True
class _SpoofedInt:
    """Mimics ``int`` via ``__class__`` to defeat ``isinstance`` checks."""

    @property
    def __class__(self) -> type:  # type: ignore[override]
        return int

    def __int__(self) -> int:
        return 3

    def __index__(self) -> int:
        return 3


@pytest.mark.parametrize("field", ["observation_dim", "n_actions", "action_dim"])
def test_step8_world_model_fields_reject_class_spoofed_integers(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        make_step8_world_model(_config_with(**{field: _SpoofedInt()}))


def test_step8_world_model_rejects_nonpositive_vector_action_dim() -> None:
    with pytest.raises(ValueError, match="action_dim"):
        make_step8_world_model(_config_with(n_actions=None, action_dim=0))


def test_step8_world_model_fields_preserve_legal_boundaries() -> None:
    config = Step8WorldModelConfig(
        observation_dim=1,
        n_actions=None,
        action_dim=1,
        hidden_sizes=(),
        step_size=0.0,
        sparsity=1.0,
        leaky_relu_slope=0.0,
        utility_decay=0.0,
    )
    model = make_step8_world_model(config)
    payload = config.to_dict()
    json.dumps(payload, allow_nan=False)
    restored = Step8WorldModelConfig.from_dict(payload)
    assert restored.observation_dim == 1
    assert restored.n_actions is None
    assert restored.action_dim == 1
    assert restored.hidden_sizes == ()
    assert restored.step_size == 0.0
    assert restored.sparsity == 1.0
    assert restored.leaky_relu_slope == 0.0
    assert restored.utility_decay == 0.0
    assert payload["sparsity"] == 1.0
    assert payload["leaky_relu_slope"] == 0.0
    assert model.to_config()["type"] == "OneStepWorldModel"

    positive = Step8WorldModelConfig(
        observation_dim=1,
        n_actions=1,
        hidden_sizes=(),
        leaky_relu_slope=0.01,
    )
    make_step8_world_model(positive)
    assert positive.leaky_relu_slope == 0.01


def test_step8_world_model_fields_canonicalize_nonbuiltin_numbers() -> None:
    value = np.float64(0.5)
    config = Step8WorldModelConfig(
        observation_dim=np.int64(3),
        n_actions=np.int64(2),
        action_dim=np.int64(1),
        hidden_sizes=(np.int64(4),),
        step_size=value,
        sparsity=value,
        leaky_relu_slope=value,
        utility_decay=value,
    )
    model = make_step8_world_model(config)
    payload = config.to_dict()
    json.dumps(payload, allow_nan=False)
    assert config.observation_dim == 3
    assert config.n_actions == 2
    assert config.action_dim == 1
    assert config.hidden_sizes == (4,)
    assert config.leaky_relu_slope == 0.5
    assert type(payload["observation_dim"]) is int
    assert type(payload["n_actions"]) is int
    assert type(payload["action_dim"]) is int
    assert type(payload["hidden_sizes"][0]) is int
    assert type(payload["step_size"]) is float
    assert type(payload["sparsity"]) is float
    assert type(payload["leaky_relu_slope"]) is float
    assert type(payload["utility_decay"]) is float
    assert model.to_config()["config"]["utility_decay"] == 0.5
    assert model.to_config()["config"]["leaky_relu_slope"] == 0.5


def test_step8_world_model_rejects_float32_overflow() -> None:
    with pytest.raises(ValueError, match="step_size"):
        Step8WorldModelConfig(step_size=1e100)
    with pytest.raises(ValueError, match="sparsity"):
        Step8WorldModelConfig(sparsity=1e100)
    with pytest.raises(ValueError, match="leaky_relu_slope"):
        Step8WorldModelConfig(leaky_relu_slope=1e100)
    with pytest.raises(ValueError, match="utility_decay"):
        Step8WorldModelConfig(utility_decay=1e100)


def test_step8_world_model_preserves_float32_boundaries() -> None:
    f32_max = float(np.finfo(np.float32).max)
    config = Step8WorldModelConfig(
        observation_dim=2**31 - 1,
        n_actions=2**31 - 1,
        action_dim=2**31 - 1,
        hidden_sizes=(2**31 - 1,),
        step_size=f32_max,
        sparsity=1.0,
        leaky_relu_slope=f32_max,
        utility_decay=0.99,
    )
    assert config.observation_dim == 2**31 - 1
    assert config.n_actions == 2**31 - 1
    assert config.action_dim == 2**31 - 1
    assert config.hidden_sizes == (2**31 - 1,)
    assert config.step_size == f32_max
    assert config.sparsity == 1.0
    assert config.leaky_relu_slope == f32_max


def test_step8_world_model_exact_fraction_rounding() -> None:
    midpoint = Fraction(1, 1) + Fraction(1, 2**24) + Fraction(1, 2**60)
    config = Step8WorldModelConfig(
        step_size=midpoint,
        sparsity=Fraction(1, 4),
        leaky_relu_slope=midpoint,
        utility_decay=Fraction(1, 2),
    )
    expected_f32 = float(np.nextafter(np.float32(1.0), np.float32(2.0)))
    assert config.step_size == expected_f32
    assert config.leaky_relu_slope == expected_f32
    assert config.sparsity == 0.25
    assert config.utility_decay == 0.5


def test_step8_world_model_rejects_spoofed_tuple_container() -> None:
    class SpoofedTuple(list):
        @property
        def __class__(self) -> type[tuple]:
            return tuple

    with pytest.raises(ValueError, match="hidden_sizes"):
        Step8WorldModelConfig(hidden_sizes=SpoofedTuple([64]))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="hidden_sizes"):
        Step8WorldModelConfig(hidden_sizes=[64])  # type: ignore[arg-type]


def test_step8_world_model_rejects_non_bool_flags() -> None:
    class SpoofedBool:
        @property
        def __class__(self) -> type[bool]:
            return bool

        def __bool__(self) -> bool:
            return True

    with pytest.raises(ValueError, match="use_layer_norm"):
        Step8WorldModelConfig(use_layer_norm=SpoofedBool())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="predict_delta"):
        Step8WorldModelConfig(predict_delta=SpoofedBool())  # type: ignore[arg-type]


def test_step8_world_model_rejects_spoofed_int_class_and_adversarial_ratios() -> None:
    class SpoofedIntFloat(float):
        @property
        def __class__(self) -> type[int]:
            return int

        def as_integer_ratio(self) -> tuple[int, int]:
            return (-1, 2**200)

    with pytest.raises(ValueError, match="step_size"):
        Step8WorldModelConfig(step_size=SpoofedIntFloat(0.5))

    with pytest.raises(ValueError, match="sparsity"):
        Step8WorldModelConfig(sparsity=SpoofedIntFloat(0.5))

    with pytest.raises(ValueError, match="leaky_relu_slope"):
        Step8WorldModelConfig(leaky_relu_slope=SpoofedIntFloat(0.5))

    with pytest.raises(ValueError, match="utility_decay"):
        Step8WorldModelConfig(utility_decay=SpoofedIntFloat(0.5))


def _legal_step8_smoke_result(**overrides: object) -> Step8SmokeResult:
    payload: dict[str, object] = {
        "config": Step8WorldModelConfig(),
        "steps": 8,
        "seed": 0,
        "reward_predictions_shape": (8,),
        "next_observation_predictions_shape": (8, 4),
        "reward_errors_shape": (8,),
        "next_observation_errors_shape": (8, 4),
        "finite": True,
        "model_config": {"ok": True},
    }
    payload.update(overrides)
    return Step8SmokeResult(**payload)  # type: ignore[arg-type]


def test_step8_smoke_result_rejects_leftover_identities() -> None:
    """Public Step 8 smoke records must not keep leftover bool/int identities."""

    with pytest.raises(ValueError, match="steps"):
        _legal_step8_smoke_result(steps=True)
    with pytest.raises(ValueError, match="steps"):
        _legal_step8_smoke_result(steps=float("nan"))
    with pytest.raises(ValueError, match="seed"):
        _legal_step8_smoke_result(seed=True)
    with pytest.raises(ValueError, match="finite"):
        _legal_step8_smoke_result(finite=1)

    legal = _legal_step8_smoke_result()
    dumped = json.dumps(
        {
            "steps": legal.steps,
            "seed": legal.seed,
            "finite": legal.finite,
        },
        allow_nan=False,
    )
    assert '"steps": 8' in dumped
    assert '"seed": 0' in dumped
    assert '"finite": true' in dumped
    assert '"steps": true' not in dumped
    assert '"seed": true' not in dumped
    assert '"finite": 1' not in dumped


def test_step8_from_dict_schema_validation() -> None:
    config = Step8WorldModelConfig()
    payload = config.to_dict()

    with pytest.raises(ValueError, match="must be an exact dictionary"):
        Step8WorldModelConfig.from_dict(["not", "a", "dict"])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="keys must be exact strings"):
        bad_keys: dict[Any, Any] = dict(payload)
        bad_keys[1] = "bad"
        Step8WorldModelConfig.from_dict(cast(Any, bad_keys))

    for key in payload:
        missing = dict(payload)
        del missing[key]
        with pytest.raises(ValueError, match="fields do not match the schema"):
            Step8WorldModelConfig.from_dict(missing)

    extra = dict(payload)
    extra["unexpected_field"] = 123
    with pytest.raises(ValueError, match="fields do not match the schema"):
        Step8WorldModelConfig.from_dict(extra)

    bad_hidden = dict(payload)
    bad_hidden["hidden_sizes"] = (64,)
    with pytest.raises(ValueError, match="hidden_sizes must be an exact list"):
        Step8WorldModelConfig.from_dict(bad_hidden)


