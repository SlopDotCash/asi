"""Supplementary coverage for types.py lifecycle helpers.

Covers previously untested helpers: ObGD state construction (defaults,
shapes, parameter propagation), agent age computation (birth timestamp),
and agent uptime (cumulative seconds).
"""

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from alberta_framework.core.types import (
    agent_age_s,
    agent_uptime_s,
    create_obgd_state,
)


def test_create_obgd_state_defaults() -> None:
    state = create_obgd_state(4)
    assert state.step_size == jnp.array(1.0, dtype=jnp.float32)
    assert state.kappa == jnp.array(2.0, dtype=jnp.float32)
    assert state.traces.shape == (4,)


def test_create_obgd_state_parameters() -> None:
    state = create_obgd_state(8, step_size=0.5, kappa=3.0, gamma=0.9, lamda=0.7)
    assert state.step_size == jnp.array(0.5, dtype=jnp.float32)
    assert state.kappa == jnp.array(3.0, dtype=jnp.float32)
    assert state.traces.shape == (8,)


def test_create_obgd_state_zero_traces() -> None:
    state = create_obgd_state(16)
    assert jnp.all(state.traces == 0)


def test_create_obgd_state_zero_dim() -> None:
    # jnp.zeros(0) is legal; the state simply has an empty trace vector.
    state = create_obgd_state(0)
    assert state.traces.shape == (0,)


def test_agent_age_s_with_birth() -> None:
    state = SimpleNamespace(birth_timestamp=1_000_000.0)
    age = agent_age_s(state)
    assert age >= 0
    assert age > 100  # birth was long ago in real time


def test_agent_age_s_without_birth() -> None:
    # Missing attribute → falls back to 0, age = now.
    state = SimpleNamespace()
    age = agent_age_s(state)
    assert age > 1_700_000_000  # current epoch seconds


def test_agent_uptime_s() -> None:
    state = SimpleNamespace(uptime_s=123.5)
    assert agent_uptime_s(state) == 123.5


def test_agent_uptime_s_without_field() -> None:
    state = SimpleNamespace()
    assert agent_uptime_s(state) == 0.0
