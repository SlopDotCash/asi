"""Saturating lifetime clock keeps Step9DreamingState.step_count non-negative.

Before this fix, ``step9_update`` advanced ``Step9DreamingState.step_count``
with a bare ``+ 1`` on an int32 array. At ``INT32_MAX`` that wraps to
``INT32_MIN``, silently corrupting a counter every external caller (tests,
checkpoints, telemetry) treats as monotonically non-negative. This mirrors
the already-fixed lifetime-clock defects in
``alberta_framework.streams.{alberta_plan_step1,feature_discovery}`` and the
open siblings tracked for ``GauntletStream``/``UPGDLearner``/``MixedHorde``.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from alberta_framework.steps.step9 import (
    Step9DreamingConfig,
    Step9DreamingState,
    init_step9_state,
    make_step9_components,
    step9_update,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)


def _cfg() -> Step9DreamingConfig:
    return Step9DreamingConfig(
        observation_dim=2,
        n_actions=2,
        model_hidden_sizes=(),
        model_sparsity=0.0,
        planning_budget=1,
        dreaming_warmup_steps=0,
        dreaming_max_model_error=1e30,
    )


def test_bare_int32_increment_wraps_negative() -> None:
    """Document the raw JAX behaviour the old ``state.step_count + 1`` relied on."""
    wrap = jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    assert int(wrap) == _INT32_MIN


def test_step9_step_count_saturates_at_int32_max() -> None:
    """A facade clock already at INT32_MAX must saturate, not wrap negative."""
    cfg = _cfg()
    agent, model, buffer = make_step9_components(cfg)
    state = init_step9_state(
        agent, model, buffer, key=jr.key(0), initial_observation=jnp.zeros(2)
    )
    near_max: Step9DreamingState = state.replace(  # type: ignore[attr-defined]
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    )

    result = step9_update(
        cfg, agent, model, buffer,
        near_max,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([0.1, 0.2], dtype=jnp.float32),
    )

    assert int(result.state.step_count) == _INT32_MAX
    assert int(result.state.step_count) >= 0

    again = step9_update(
        cfg, agent, model, buffer,
        result.state,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([0.1, 0.2], dtype=jnp.float32),
    )
    assert int(again.state.step_count) == _INT32_MAX
    assert int(again.state.step_count) >= 0


def test_step9_step_count_still_increments_below_max() -> None:
    """The saturating clock is a no-op change for ordinary, unsaturated counts."""
    cfg = _cfg()
    agent, model, buffer = make_step9_components(cfg)
    state = init_step9_state(
        agent, model, buffer, key=jr.key(1), initial_observation=jnp.zeros(2)
    )
    result = step9_update(
        cfg, agent, model, buffer,
        state,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([0.1, 0.2], dtype=jnp.float32),
    )
    assert int(result.state.step_count) == 1
