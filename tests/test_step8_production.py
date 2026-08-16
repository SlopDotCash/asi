"""Production-facing Step 8 world-model facade tests.

Covers the one-step environment-model facade on real constructors. Invalid
dimension and scientific-scalar cases are written to fail on current main
(bool, non-real, non-finite, and out-of-domain values accepted) and pass
after the facade rejects them. Legal endpoints stay constructible.
"""

from __future__ import annotations

import json
from typing import Any

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.steps.step8 import (
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
    ("n_actions", True),
    ("n_actions", False),
    ("n_actions", 0),
    ("n_actions", -1),
    ("n_actions", 1.5),
    ("n_actions", "2"),
    ("action_dim", True),
    ("action_dim", False),
    ("action_dim", 0),
    ("action_dim", -1),
    ("action_dim", 1.5),
    ("action_dim", "1"),
    ("hidden_sizes", (True,)),
    ("hidden_sizes", (False,)),
    ("hidden_sizes", (0,)),
    ("hidden_sizes", (-1,)),
    ("hidden_sizes", (1.5,)),
    ("hidden_sizes", ("64",)),
    ("step_size", float("nan")),
    ("step_size", float("inf")),
    ("step_size", True),
    ("step_size", False),
    ("step_size", -1.0),
    ("sparsity", float("nan")),
    ("sparsity", True),
    ("sparsity", -0.1),
    ("sparsity", 1.1),
    ("utility_decay", float("nan")),
    ("utility_decay", float("inf")),
    ("utility_decay", True),
    ("utility_decay", False),
    ("utility_decay", -0.1),
    ("utility_decay", 1.0),
    ("leaky_relu_slope", float("nan")),
    ("leaky_relu_slope", float("inf")),
    ("leaky_relu_slope", float("-inf")),
    ("leaky_relu_slope", True),
    ("leaky_relu_slope", False),
    ("leaky_relu_slope", "0.01"),
    ("leaky_relu_slope", -0.01),
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
