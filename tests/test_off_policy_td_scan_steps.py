"""Reject Gradient-TD scan T that leftover INT32 still admits before hang."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from alberta_framework.core.off_policy_td import (
    GradientTDLinearLearner,
    _require_scan_resources,
    run_gradient_td_learning_loop,
)

pytestmark = pytest.mark.unit

_LAST_FIT_STEPS = 10_000
_OVERFLOW_STEPS = _LAST_FIT_STEPS + 1


def _loop_inputs(steps: int, feature_dim: int = 2) -> tuple[Any, Any, Any, Any, Any]:
    observations = jnp.ones((steps, feature_dim), dtype=jnp.float32)
    rewards = jnp.ones((steps,), dtype=jnp.float32)
    next_observations = jnp.ones((steps, feature_dim), dtype=jnp.float32)
    gammas = jnp.zeros((steps,), dtype=jnp.float32)
    rhos = jnp.ones((steps,), dtype=jnp.float32)
    return observations, rewards, next_observations, gammas, rhos


def test_leftover_int32_still_admits_ten_million_steps() -> None:
    _require_scan_resources(10_000_000, 2)


def test_overflow_scan_t_rejects_before_lax_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_scan(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("overflow scan T reached jax.lax.scan")

    monkeypatch.setattr(jax.lax, "scan", unexpected_scan)
    learner = GradientTDLinearLearner()
    state = learner.init(2)
    with pytest.raises(ValueError, match="scan limit"):
        run_gradient_td_learning_loop(
            learner,
            state,
            *_loop_inputs(_OVERFLOW_STEPS),
        )


def test_last_fit_scan_t_still_runs() -> None:
    learner = GradientTDLinearLearner(step_size=0.01)
    state = learner.init(2)
    result = run_gradient_td_learning_loop(
        learner,
        state,
        *_loop_inputs(_LAST_FIT_STEPS),
    )
    assert result.predictions.shape == (_LAST_FIT_STEPS,)
