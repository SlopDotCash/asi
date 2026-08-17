"""Tests for exo-cortex credit assignment on the partner's executed action.

The exo-cortex learns from the *partner's* experience stream.  Its Q-update
must credit the action the partner actually executed (``effective_action``
from the recommendation protocol), not the action the cortex's own internal
epsilon-greedy selection happened to pick.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.intelligence_amplification import (
    ExoCerebellumConfig,
    ExoCortexAgent,
    ExoCortexState,
    IAAgent,
    IAConfig,
    IAState,
    RecommendationProtocolConfig,
    _checked_partner_action,
    init_recommendation_protocol_state,
    update_recommendation_protocol,
)
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SPEC = SubtaskSpec(feature_index=0, threshold=0.5, pseudo_reward_scale=1.0, max_option_steps=8)

_N_PRIM = 3
_OBS_DIM = 4
_OBS0 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)


def _make_ia_config() -> IAConfig:
    cerebellum = ExoCerebellumConfig(n_demons=2, obs_dim=_OBS_DIM)
    stomp = STOMPConfig(
        subtask_specs=(_SPEC,),
        observation_dim=_OBS_DIM,
        n_primitive_actions=_N_PRIM,
        epsilon_base=0.0,
        epsilon_option=0.0,
    )
    return IAConfig(cerebellum=cerebellum, cortex=OaKConfig(stomp=stomp))


def _head_params(cortex_state: ExoCortexState) -> tuple[tuple, tuple]:
    """Per-extended-action Q-head weights and biases (linear base learner)."""
    params = cortex_state.stomp_state.base_learner_state.head_params
    return tuple(params.weights), tuple(params.biases)


def _changed_heads(before: ExoCortexState, after: ExoCortexState) -> list[bool]:
    """Which base Q-heads changed between two cortex states."""
    w_before, b_before = _head_params(before)
    w_after, b_after = _head_params(after)
    return [
        bool(jnp.any(wb != wa)) or bool(jnp.any(bb != ba))
        for wb, wa, bb, ba in zip(w_before, w_after, b_before, b_after)
    ]


def _force_own_selection(cortex_state: ExoCortexState, action: int) -> ExoCortexState:
    """Pin the cortex's internally selected action deterministically."""
    stomp_state = cortex_state.stomp_state.replace(
        base_last_action=jnp.array(action, dtype=jnp.int32),
        option_last_intra_action=jnp.array(action, dtype=jnp.int32),
        executing_option=jnp.array(-1, dtype=jnp.int32),
    )
    return cortex_state.replace(stomp_state=stomp_state)


def _cortex_update_with_executed(
    cortex: ExoCortexAgent,
    state: ExoCortexState,
    reward: jnp.ndarray,
    next_obs: jnp.ndarray,
    partner_action: jnp.ndarray,
) -> ExoCortexState:
    """Update the cortex crediting the partner's executed action.

    Falls back to the legacy signature (which cannot receive the executed
    action) so that, pre-fix, the test demonstrates the wrong crediting
    rather than erroring on the keyword.
    """
    try:
        new_state, _, _ = cortex.update(
            state, reward, next_obs, partner_action=partner_action
        )
    except TypeError:
        new_state, _, _ = cortex.update(state, reward, next_obs)
    return new_state


# ---------------------------------------------------------------------------
# Core crediting: executed action, not the cortex's own selection
# ---------------------------------------------------------------------------


def test_cortex_credits_executed_action_not_own_selection() -> None:
    """Partner executes A while the cortex picked B: credit must go to A."""
    config = _make_ia_config()
    cortex = ExoCortexAgent(config.cortex)
    state = cortex.start(cortex.init(jr.key(0)), _OBS0)

    action_b = 1  # cortex's own epsilon-greedy selection
    action_a = 0  # action the partner actually executed
    state = _force_own_selection(state, action_b)

    new_state = _cortex_update_with_executed(
        cortex,
        state,
        jnp.array(1.0, dtype=jnp.float32),
        _OBS0,
        jnp.array(action_a, dtype=jnp.int32),
    )

    changed = _changed_heads(state, new_state)
    assert changed[action_a], "executed action's Q-head must be credited"
    assert not changed[action_b], "cortex's own (non-executed) selection must not be credited"


def test_ia_agent_update_credits_partner_action() -> None:
    """IAAgent.update must route the partner's executed action to the cortex."""
    config = _make_ia_config()
    agent = IAAgent(config)
    state = agent.start(agent.init(jr.key(0)), _OBS0)

    action_b = 2
    action_a = 0
    state = IAState(
        cerebellum_state=state.cerebellum_state,
        cortex_state=_force_own_selection(state.cortex_state, action_b),
        step_count=state.step_count,
    )

    result = agent.update(
        state,
        _OBS0,
        jnp.array(1.0, dtype=jnp.float32),
        _OBS0,
        partner_action=jnp.array(action_a, dtype=jnp.int32),
    )

    changed = _changed_heads(state.cortex_state, result.state.cortex_state)
    assert changed[action_a]
    assert not changed[action_b]


def test_update_without_partner_action_keeps_legacy_crediting() -> None:
    """Without an executed action the legacy own-action crediting remains."""
    config = _make_ia_config()
    cortex = ExoCortexAgent(config.cortex)
    state = cortex.start(cortex.init(jr.key(0)), _OBS0)
    action_b = 1
    state = _force_own_selection(state, action_b)

    new_state, _, _ = cortex.update(state, jnp.array(1.0, dtype=jnp.float32), _OBS0)

    changed = _changed_heads(state, new_state)
    assert changed[action_b]
    assert sum(changed) == 1


def test_protocol_effective_action_feeds_cortex_crediting() -> None:
    """effective_action from the recommendation protocol drives the Q-update."""
    config = _make_ia_config()
    agent = IAAgent(config)
    state = agent.start(agent.init(jr.key(0)), _OBS0)

    recommendation = 1  # what the cortex recommended
    partner_action = 2  # what the partner actually did
    state = IAState(
        cerebellum_state=state.cerebellum_state,
        cortex_state=_force_own_selection(state.cortex_state, recommendation),
        step_count=state.step_count,
    )

    protocol = update_recommendation_protocol(
        RecommendationProtocolConfig(),
        init_recommendation_protocol_state(),
        jnp.array(recommendation, dtype=jnp.int32),
        jnp.array(partner_action, dtype=jnp.int32),
    )
    assert int(protocol.effective_action) == partner_action

    result = agent.update(
        state,
        _OBS0,
        jnp.array(1.0, dtype=jnp.float32),
        _OBS0,
        partner_action=protocol.effective_action,
    )

    changed = _changed_heads(state.cortex_state, result.state.cortex_state)
    assert changed[partner_action]
    assert not changed[recommendation]


# ---------------------------------------------------------------------------
# Paired toy-stream runs: Q-values reflect executed behaviour
# ---------------------------------------------------------------------------


def _run_partner_stream(
    *, follow: bool, num_steps: int = 80
) -> tuple[IAAgent, IAState, IAState, list[int], int]:
    """Drive an IAAgent from a toy partner stream.

    The partner either follows every recommendation (``follow=True``) or
    always executes action 1 regardless of it.  Action 0 yields reward +1,
    every other action yields reward -1; the observation is constant.
    """
    agent = IAAgent(_make_ia_config())
    init_state = agent.start(agent.init(jr.key(1)), _OBS0)
    state = init_state
    recommendation = int(agent.cortex.recommend(state.cortex_state, _OBS0))
    executed: list[int] = []
    for _ in range(num_steps):
        action = recommendation if follow else 1
        reward = 1.0 if action == 0 else -1.0
        result = agent.update(
            state,
            _OBS0,
            jnp.array(reward, dtype=jnp.float32),
            _OBS0,
            partner_action=jnp.array(action, dtype=jnp.int32),
        )
        state = result.state
        recommendation = int(result.recommendation)
        executed.append(action)
    return agent, init_state, state, executed, recommendation


def test_partner_follows_recommendations_qvalues_track_executed() -> None:
    agent, init_state, state, executed, recommendation = _run_partner_stream(follow=True)

    changed = _changed_heads(init_state.cortex_state, state.cortex_state)
    executed_set = set(executed)
    for action in range(_N_PRIM):
        assert changed[action] == (action in executed_set)
    assert not changed[_N_PRIM]  # option head is never executed by the partner

    # The follow-loop must have discovered the rewarding action.
    assert executed[-1] == 0
    assert recommendation == 0
    q_prim = agent.cortex.oak_agent.base_q_values(state.cortex_state, _OBS0)[:_N_PRIM]
    for action in executed_set - {0}:
        assert float(q_prim[0]) > float(q_prim[action])


def test_partner_ignores_recommendations_qvalues_track_executed() -> None:
    agent, init_state, state, executed, recommendation = _run_partner_stream(follow=False)

    assert set(executed) == {1}
    changed = _changed_heads(init_state.cortex_state, state.cortex_state)
    assert changed == [False, True, False, False]

    # The executed (always penalised) action must look worst; the cortex
    # must not keep recommending it.
    q_prim = agent.cortex.oak_agent.base_q_values(state.cortex_state, _OBS0)[:_N_PRIM]
    assert float(q_prim[1]) < float(q_prim[0])
    assert float(q_prim[1]) < float(q_prim[2])
    assert recommendation != 1


# ---------------------------------------------------------------------------
# Scan parity
# ---------------------------------------------------------------------------


def test_scan_with_partner_actions_matches_loop() -> None:
    agent = IAAgent(_make_ia_config())
    state = agent.start(agent.init(jr.key(2)), _OBS0)

    num_steps = 12
    obs = jnp.tile(_OBS0, (num_steps, 1))
    rewards = jnp.where(jnp.arange(num_steps) % 2 == 0, 1.0, -1.0).astype(jnp.float32)
    actions = (jnp.arange(num_steps, dtype=jnp.int32) % _N_PRIM).astype(jnp.int32)

    loop_state = state
    for t in range(num_steps):
        loop_state = agent.update(
            loop_state, obs[t], rewards[t], obs[t], partner_action=actions[t]
        ).state

    scan_result = agent.scan(state, obs, rewards, obs, partner_actions=actions)

    w_loop, b_loop = _head_params(loop_state.cortex_state)
    w_scan, b_scan = _head_params(scan_result.state.cortex_state)
    for wl, ws, bl, bs in zip(w_loop, w_scan, b_loop, b_scan):
        assert jnp.allclose(wl, ws, atol=1e-6)
        assert jnp.allclose(bl, bs, atol=1e-6)


def test_ia_agent_scan_rejects_non_integer_partner_actions() -> None:
    agent = IAAgent(_make_ia_config())
    state = agent.start(agent.init(jr.key(2)), _OBS0)
    num_steps = 4
    obs = jnp.tile(_OBS0, (num_steps, 1))
    rewards = jnp.zeros(num_steps, dtype=jnp.float32)

    # Fractional float actions rejected
    with pytest.raises(ValueError, match="partner_actions must have an integer dtype"):
        agent.scan(
            state,
            obs,
            rewards,
            obs,
            partner_actions=jnp.array([1.75, 0.25, 0.5, 1.0], dtype=jnp.float32),
        )

    # Boolean actions rejected
    with pytest.raises(ValueError, match="partner_actions must have an integer dtype"):
        agent.scan(
            state,
            obs,
            rewards,
            obs,
            partner_actions=jnp.array([True, False, True, False]),
        )

    # Valid integer actions accepted
    result = agent.scan(
        state,
        obs,
        rewards,
        obs,
        partner_actions=jnp.array([1, 0, 1, 0], dtype=jnp.int32),
    )
    assert result.state is not None


def test_ia_scan_action_contract_is_jittable_and_invalid_values_are_atomic() -> None:
    agent = IAAgent(_make_ia_config())
    state = agent.start(agent.init(jr.key(2)), _OBS0)
    obs = jnp.tile(_OBS0, (1, 1))
    rewards = jnp.ones((1,), dtype=jnp.float32)

    compiled = jax.jit(
        lambda initial_state, executed_actions: agent.scan(
            initial_state,
            obs,
            rewards,
            obs,
            partner_actions=executed_actions,
        )
    )
    valid = compiled(state, jnp.asarray([1], dtype=jnp.int32))
    assert bool(valid.updates_applied[0])

    invalid = compiled(state, jnp.asarray([_N_PRIM], dtype=jnp.int32))
    # JIT canonicalizes legacy Python-float lifecycle metadata to a device
    # scalar; compare with that no-op boundary rather than the host object.
    compiled_input_state = jax.jit(lambda value: value)(state)
    chex.assert_trees_all_equal(invalid.state, compiled_input_state)
    assert not bool(jnp.any(invalid.updates_applied))

    for bad_actions in (
        jnp.asarray([0.5], dtype=jnp.float32),
        jnp.asarray([True], dtype=jnp.bool_),
        jnp.asarray([[0]], dtype=jnp.int32),
    ):
        with pytest.raises(ValueError):
            compiled(state, bad_actions)


def test_ia_scan_preserves_wide_action_values_until_range_validation() -> None:
    agent = IAAgent(_make_ia_config())
    state = agent.start(agent.init(jr.key(2)), _OBS0)
    obs = jnp.tile(_OBS0, (1, 1))
    rewards = jnp.ones((1,), dtype=jnp.float32)

    with jax.enable_x64():
        wide = jnp.asarray([2**32], dtype=jnp.uint64)
        with pytest.raises(ValueError, match="partner_actions must lie"):
            agent.scan(state, obs, rewards, obs, partner_actions=wide)

        checked = jax.jit(
            lambda action: _checked_partner_action(
                action,
                n_primitive_actions=_N_PRIM,
            )
        )
        safe_action, action_valid = checked(wide[0])
        assert int(safe_action) == 0
        assert not bool(action_valid)
