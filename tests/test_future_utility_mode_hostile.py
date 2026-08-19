"""Hostile string gates for future utility modes."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from alberta_framework.core.future_utility import (
    bias_correct_future_utility,
    normalize_future_utility_signal,
)

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __str__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile str must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


def test_normalize_rejects_hostile_before_in() -> None:
    hostile = _HostileStr("uncertainty")
    _HostileStr.calls = 0
    signal = jnp.array([1.0], dtype=jnp.float32)
    ages = jnp.array([1], dtype=jnp.int32)
    second = jnp.array([1.0], dtype=jnp.float32)
    with pytest.raises(ValueError, match="exact string"):
        normalize_future_utility_signal(
            signal, ages, second, moment_decay=0.9, utility_decay=0.9, mode=hostile  # type: ignore[arg-type]
        )
    assert _HostileStr.calls == 0
    # benign passes
    norm, _ = normalize_future_utility_signal(
        signal, ages, second, moment_decay=0.9, utility_decay=0.9, mode="uncertainty"
    )
    assert norm is not None


def test_bias_correct_rejects_hostile_before_not_in() -> None:
    hostile = _HostileStr("age")
    _HostileStr.calls = 0
    utils = jnp.array([1.0], dtype=jnp.float32)
    ages = jnp.array([1], dtype=jnp.int32)
    with pytest.raises(ValueError, match="exact string"):
        bias_correct_future_utility(utils, ages, utility_decay=0.9, mode=hostile)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0
    # benign valid
    out = bias_correct_future_utility(utils, ages, utility_decay=0.9, mode="age")
    assert out is not None
    # benign unknown returns same (no raise for unknown str, but type guard passes)
    out2 = bias_correct_future_utility(utils, ages, utility_decay=0.9, mode="unknown_mode_xyz")
    assert out2 is not None


def test_non_string_rejects() -> None:
    signal = jnp.array([1.0], dtype=jnp.float32)
    ages = jnp.array([1], dtype=jnp.int32)
    second = jnp.array([1.0], dtype=jnp.float32)
    with pytest.raises(ValueError, match="exact string"):
        normalize_future_utility_signal(
            signal, ages, second, moment_decay=0.9, utility_decay=0.9, mode=123  # type: ignore[arg-type]
        )
    utils = jnp.array([1.0], dtype=jnp.float32)
    with pytest.raises(ValueError, match="exact string"):
        bias_correct_future_utility(utils, ages, utility_decay=0.9, mode=123)  # type: ignore[arg-type]
