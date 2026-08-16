"""Production-facing Step 4 SARSA facade tests (mirrors test_step3_production.py).

Invalid scientific-scalar cases are written to fail on current main (bool,
non-real, non-finite, and out-of-domain values accepted) and pass after
the facade rejects them. Legal endpoints stay constructible.
"""

import json
from typing import Any

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.optimizers import IDBD, LMS, Autostep, ObGDBounding
from alberta_framework.steps import (
    Step4SARSAConfig,
    init_step4_state,
    make_step4_bounder,
    make_step4_optimizer,
    make_step4_sarsa_agent,
    run_step4_scan,
    run_step4_smoke,
    step4_update,
)

_INVALID_STEP4_SCALARS: tuple[tuple[str, Any], ...] = (
    ("gamma", float("nan")),
    ("gamma", float("inf")),
    ("gamma", float("-inf")),
    ("gamma", True),
    ("gamma", False),
    ("gamma", "0.99"),
    ("gamma", -0.1),
    ("gamma", 1.1),
    ("epsilon_start", float("nan")),
    ("epsilon_start", True),
    ("epsilon_start", -0.1),
    ("epsilon_start", 1.1),
    ("epsilon_end", float("nan")),
    ("epsilon_end", True),
    ("epsilon_end", -0.1),
    ("epsilon_end", 1.1),
    ("lamda", float("nan")),
    ("lamda", True),
    ("lamda", -0.1),
    ("lamda", 1.1),
    ("step_size", float("nan")),
    ("step_size", float("inf")),
    ("step_size", True),
    ("step_size", False),
    ("step_size", -1.0),
    ("meta_step_size", float("nan")),
    ("meta_step_size", float("inf")),
    ("meta_step_size", True),
    ("meta_step_size", -1.0),
    ("bounder_kappa", float("nan")),
    ("bounder_kappa", float("inf")),
    ("bounder_kappa", True),
    ("bounder_kappa", 0.0),
    ("bounder_kappa", -1.0),
    ("sparsity", float("nan")),
    ("sparsity", True),
    ("sparsity", -0.1),
    ("sparsity", 1.1),
    ("epsilon_decay_steps", True),
    ("epsilon_decay_steps", False),
    ("epsilon_decay_steps", -1),
    ("epsilon_decay_steps", 1.5),
    ("epsilon_decay_steps", 2**31),
    ("n_actions", True),
    ("n_actions", 2**31),
    ("hidden_sizes", (0,)),
    ("hidden_sizes", (True,)),
    ("hidden_sizes", (2**31,)),
)


def test_step4_config_roundtrip() -> None:
    config = Step4SARSAConfig(
        n_actions=3,
        hidden_sizes=(8, 4),
        gamma=0.95,
        optimizer="idbd",
        bounder="none",
        lamda=0.5,
        trace_mode="replacing",
    )
    payload = config.to_dict()

    assert payload["hidden_sizes"] == [8, 4]
    assert Step4SARSAConfig.from_dict(payload) == config

    sarsa_config = config.to_sarsa_config()
    assert sarsa_config.n_actions == 3
    assert sarsa_config.gamma == 0.95


def test_step4_factories_and_validation() -> None:
    assert isinstance(make_step4_optimizer(Step4SARSAConfig(optimizer="lms")), LMS)
    assert isinstance(make_step4_optimizer(Step4SARSAConfig(optimizer="idbd")), IDBD)
    assert isinstance(make_step4_optimizer(Step4SARSAConfig(optimizer="autostep")), Autostep)
    assert make_step4_bounder(Step4SARSAConfig(bounder="none")) is None
    assert isinstance(make_step4_bounder(Step4SARSAConfig(bounder="obgd")), ObGDBounding)

    with pytest.raises(ValueError, match="n_actions"):
        Step4SARSAConfig(n_actions=0)
    with pytest.raises(ValueError, match="optimizer"):
        make_step4_optimizer(Step4SARSAConfig(optimizer="bogus"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bounder"):
        make_step4_bounder(Step4SARSAConfig(bounder="bogus"))  # type: ignore[arg-type]


def test_step4_prime_and_one_transition() -> None:
    config = Step4SARSAConfig(n_actions=2, hidden_sizes=(8,))
    agent = make_step4_sarsa_agent(config)
    feature_dim = 5
    data_key, state_key = jr.split(jr.key(3))
    features = jr.normal(data_key, (2, feature_dim), dtype=jnp.float32)

    state = init_step4_state(
        agent,
        feature_dim=feature_dim,
        key=state_key,
        initial_features=features[0],
    )
    chex.assert_trees_all_close(state.last_observation, features[0])
    assert 0 <= int(state.last_action) < config.n_actions

    result = step4_update(
        agent,
        state,
        jnp.asarray(0.5, dtype=jnp.float32),
        features[1],
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    chex.assert_shape(result.q_values, (config.n_actions,))
    chex.assert_tree_all_finite(result.q_values)
    chex.assert_tree_all_finite(result.td_error)
    assert 0 <= int(result.action) < config.n_actions
    assert float(result.reward) == 0.5
    chex.assert_trees_all_close(result.state.last_observation, features[1])


def test_step4_scan_shapes_and_finiteness() -> None:
    config = Step4SARSAConfig(n_actions=3, hidden_sizes=(), epsilon_decay_steps=8)
    agent = make_step4_sarsa_agent(config)
    steps, feature_dim = 12, 4
    data_key, state_key = jr.split(jr.key(7))
    observations = jr.normal(data_key, (steps + 1, feature_dim), dtype=jnp.float32)
    rewards = jnp.tanh(observations[1:, 0])
    terminated = jnp.zeros(steps, dtype=jnp.float32)

    state = init_step4_state(
        agent,
        feature_dim=feature_dim,
        key=state_key,
        initial_features=observations[0],
    )
    result = run_step4_scan(agent, state, observations[1:], rewards, terminated)

    chex.assert_shape(result.q_values, (steps, config.n_actions))
    chex.assert_shape(result.td_errors, (steps,))
    chex.assert_shape(result.actions, (steps,))
    chex.assert_tree_all_finite(result.q_values)
    chex.assert_tree_all_finite(result.td_errors)
    assert bool(jnp.all(result.actions >= 0))
    assert bool(jnp.all(result.actions < config.n_actions))


def test_step4_smoke_is_finite_and_serializable() -> None:
    config = Step4SARSAConfig(n_actions=2, hidden_sizes=(8,), optimizer="autostep")
    result = run_step4_smoke(config, steps=16, feature_dim=6, seed=1)
    payload = result.to_dict()

    assert result.finite
    assert result.q_values_shape == (16, 2)
    assert result.td_errors_shape == (16,)
    assert result.actions_shape == (16,)
    assert payload["config"] == config.to_dict()
    assert payload["q_values_shape"] == [16, 2]
    agent_config = payload["agent_config"]
    assert isinstance(agent_config, dict)
    assert agent_config["type"] == "SARSAAgent"


def test_step4_smoke_validation() -> None:
    with pytest.raises(ValueError, match="steps"):
        run_step4_smoke(steps=0)
    with pytest.raises(ValueError, match="feature_dim"):
        run_step4_smoke(feature_dim=0)


def _config_with(**overrides: Any) -> Step4SARSAConfig:
    payload: dict[str, Any] = {"n_actions": 2}
    payload.update(overrides)
    return Step4SARSAConfig(**payload)


@pytest.mark.parametrize(("field", "value"), _INVALID_STEP4_SCALARS)
def test_step4_sarsa_scalars_reject_invalid_inputs(field: str, value: object) -> None:
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


@pytest.mark.parametrize("field", ["n_actions", "epsilon_decay_steps"])
def test_step4_sarsa_fields_reject_class_spoofed_integers(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _config_with(**{field: _SpoofedInt()})


def test_step4_sarsa_hidden_sizes_rejects_class_spoofed_integers() -> None:
    with pytest.raises(ValueError, match="hidden_sizes"):
        _config_with(hidden_sizes=(_SpoofedInt(),))


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_step4_sarsa_config_requires_exact_boolean_layer_norm(value: object) -> None:
    with pytest.raises(ValueError, match="use_layer_norm must be a boolean"):
        _config_with(use_layer_norm=value)


def test_step4_sarsa_scalars_preserve_legal_boundaries() -> None:
    config = Step4SARSAConfig(
        n_actions=1,
        hidden_sizes=(),
        gamma=0.0,
        epsilon_start=0.0,
        epsilon_end=1.0,
        epsilon_decay_steps=0,
        lamda=1.0,
        step_size=0.0,
        meta_step_size=0.0,
        bounder_kappa=0.5,
        sparsity=1.0,
        use_layer_norm=False,
    )
    agent = make_step4_sarsa_agent(config)
    payload = config.to_dict()
    json.dumps(payload, allow_nan=False)
    restored = Step4SARSAConfig.from_dict(payload)
    assert restored.gamma == 0.0
    assert restored.epsilon_start == 0.0
    assert restored.epsilon_end == 1.0
    assert restored.lamda == 1.0
    assert restored.step_size == 0.0
    assert restored.meta_step_size == 0.0
    assert restored.bounder_kappa == 0.5
    assert restored.sparsity == 1.0
    assert restored.epsilon_decay_steps == 0
    assert restored.n_actions == 1
    assert restored.hidden_sizes == ()
    assert agent.to_config()["type"] == "SARSAAgent"

    upper = Step4SARSAConfig(
        n_actions=2,
        gamma=1.0,
        epsilon_start=1.0,
        epsilon_end=0.0,
        lamda=0.0,
        sparsity=0.0,
    )
    make_step4_sarsa_agent(upper)
    assert upper.gamma == 1.0
    assert upper.epsilon_start == 1.0
    assert upper.epsilon_end == 0.0
    assert upper.lamda == 0.0
    assert upper.sparsity == 0.0


def test_step4_sarsa_scalars_canonicalize_nonbuiltin_reals() -> None:
    value = np.float64(0.5)
    config = Step4SARSAConfig(
        n_actions=np.int64(3),
        hidden_sizes=(np.int64(8),),
        gamma=value,
        epsilon_start=value,
        epsilon_end=np.float64(0.0),
        epsilon_decay_steps=np.int64(4),
        lamda=value,
        step_size=value,
        meta_step_size=value,
        bounder_kappa=np.float64(2.0),
        sparsity=value,
    )
    agent = make_step4_sarsa_agent(config)
    payload = config.to_dict()
    json.dumps(payload, allow_nan=False)
    assert config.gamma == 0.5
    assert config.epsilon_start == 0.5
    assert config.epsilon_end == 0.0
    assert config.lamda == 0.5
    assert type(payload["gamma"]) is float
    assert type(payload["epsilon_start"]) is float
    assert type(payload["epsilon_end"]) is float
    assert type(payload["lamda"]) is float
    assert type(payload["step_size"]) is float
    assert type(payload["meta_step_size"]) is float
    assert type(payload["bounder_kappa"]) is float
    assert type(payload["sparsity"]) is float
    assert type(payload["n_actions"]) is int
    assert type(payload["epsilon_decay_steps"]) is int
    assert type(payload["hidden_sizes"][0]) is int
    assert agent.to_config()["type"] == "SARSAAgent"


def test_step4_sarsa_dimensions_preserve_int32_maximum() -> None:
    config = Step4SARSAConfig(
        n_actions=np.int64(2**31 - 1),
        epsilon_decay_steps=np.int64(2**31 - 1),
        hidden_sizes=(np.int64(2**31 - 1),),
    )

    assert config.n_actions == 2**31 - 1
    assert config.epsilon_decay_steps == 2**31 - 1
    assert config.hidden_sizes == (2**31 - 1,)
    assert type(config.n_actions) is int
    assert type(config.epsilon_decay_steps) is int
    assert type(config.hidden_sizes[0]) is int
