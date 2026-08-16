"""Regression tests for OaK curation and PrototypeAgent GRU-perception fixes.

Covers three defects:

1. ``OaKAgent.curate()`` must reset the replaced option's base-head optimizer
   state to a *fresh init*, not zeros — zeroed optimizer state is corrupt
   (LMS: step-size 0 freezes the head; IDBD: log step-size 0 means
   ``exp(0) = 1.0``).
2. Curation must honour a minimum-uptime guard
   (``OaKConfig.min_steps_before_curation``) so options are never evicted on
   untrained utility EMAs, including via the prototype auto-curate path.
3. ``PrototypeAgent.act()`` must route observations through the same GRU
   augmentation as ``update()`` when ``gru_perception`` is configured.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.oak import KeyboardChordLearnerConfig, OaKAgent, OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    GRUPerceptionConfig,
    PrototypeAgent,
    PrototypeAgentConfig,
)

OBS_DIM = 4
_SPEC0 = SubtaskSpec(feature_index=0)
_SPEC1 = SubtaskSpec(feature_index=1)


def _oak_cfg(**kwargs: object) -> OaKConfig:
    stomp = STOMPConfig(subtask_specs=(_SPEC0, _SPEC1), observation_dim=OBS_DIM)
    return OaKConfig(stomp=stomp, **kwargs)  # type: ignore[arg-type]


def _primed(cfg: OaKConfig, seed: int = 0) -> tuple[OaKAgent, object]:
    agent = OaKAgent(cfg)
    state = agent.start(agent.init(jr.key(seed)), jnp.zeros(cfg.observation_dim))
    return agent, state


# ---------------------------------------------------------------------------
# Defect 1: curate() must reset head optimizer state to fresh init, not zeros
# ---------------------------------------------------------------------------


def test_curate_resets_head_optimizer_state_to_fresh_init() -> None:
    cfg = _oak_cfg()
    agent, state = _primed(cfg)
    result = agent.scan(state, jnp.ones(30), jr.normal(jr.key(1), (30, OBS_DIM)))
    state = result.state
    # Force option 1 to be the eviction target
    state = state.replace(utility_ema=jnp.array([0.8, 0.1], dtype=jnp.float32))
    _, new_state = agent.curate(state, jr.key(2))

    head_idx = cfg.n_primitive_actions + 1  # extended-action head of option 1
    fresh = agent.stomp_agent.base_learner.init(cfg.observation_dim, jr.key(3))
    # The reset head's optimizer state must equal what a fresh init produces
    # (catches zeros_like: LMS stores the step-size itself, IDBD stores log
    # step-sizes — zeroing either corrupts learning).
    chex.assert_trees_all_close(
        new_state.stomp_state.base_learner_state.head_optimizer_states[head_idx],
        fresh.head_optimizer_states[head_idx],
    )


def test_curate_leaves_other_head_optimizer_states_untouched() -> None:
    cfg = _oak_cfg()
    agent, state = _primed(cfg)
    state = state.replace(utility_ema=jnp.array([0.8, 0.1], dtype=jnp.float32))
    _, new_state = agent.curate(state, jr.key(2))
    head_idx = cfg.n_primitive_actions + 1
    for i, opt in enumerate(new_state.stomp_state.base_learner_state.head_optimizer_states):
        if i != head_idx:
            chex.assert_trees_all_close(
                opt, state.stomp_state.base_learner_state.head_optimizer_states[i]
            )


def test_curate_resets_only_replaced_intra_option_reward_rate() -> None:
    cfg = _oak_cfg()
    agent, state = _primed(cfg)
    policies = state.stomp_state.option_policies.replace(
        average_rewards=jnp.array([3.0, 4.0], dtype=jnp.float32)
    )
    models = state.stomp_state.option_models.replace(
        baseline_mass_ema=jnp.array([1.5, 2.5], dtype=jnp.float32)
    )
    state = state.replace(
        utility_ema=jnp.array([0.8, 0.1], dtype=jnp.float32),
        stomp_state=state.stomp_state.replace(
            option_policies=policies,
            option_models=models,
            executing_option=jnp.array(-1, dtype=jnp.int32),
        ),
    )

    _, curated = agent.curate(state, jr.key(6))

    chex.assert_trees_all_close(
        curated.stomp_state.option_policies.average_rewards,
        jnp.array([3.0, 0.0], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        curated.stomp_state.option_models.baseline_mass_ema,
        jnp.array([1.5, 0.0], dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# Defect 2: minimum-uptime guard before any eviction
# ---------------------------------------------------------------------------


def test_no_curation_before_min_uptime() -> None:
    cfg = _oak_cfg(min_steps_before_curation=50)
    agent, state = _primed(cfg)
    # Even a clearly worst option must not be evicted before min uptime
    state = state.replace(utility_ema=jnp.array([0.8, 0.1], dtype=jnp.float32))
    new_agent, new_state = agent.curate(state, jr.key(1))
    assert new_agent is agent
    assert new_state is state


def test_min_uptime_guard_counts_steps() -> None:
    cfg = _oak_cfg(min_steps_before_curation=20)
    agent, state = _primed(cfg)
    result = agent.scan(state, jnp.zeros(10), jr.normal(jr.key(1), (10, OBS_DIM)))
    state = result.state  # step_count == 10 < 20
    new_agent, new_state = agent.curate(state, jr.key(2))
    assert new_agent is agent
    assert new_state is state
    result = agent.scan(state, jnp.zeros(10), jr.normal(jr.key(3), (10, OBS_DIM)))
    state = result.state  # step_count == 20 → guard satisfied
    new_agent, _ = agent.curate(state, jr.key(4))
    assert new_agent is not agent


def test_min_steps_before_curation_negative_raises() -> None:
    with pytest.raises(ValueError, match="min_steps_before_curation"):
        _oak_cfg(min_steps_before_curation=-1)


def test_min_steps_before_curation_config_roundtrip() -> None:
    cfg = _oak_cfg(min_steps_before_curation=25)
    restored = OaKConfig.from_config(cfg.to_config())
    assert restored.min_steps_before_curation == 25
    # Old payloads without the field default to 0 (guard disabled)
    payload = cfg.to_config()
    payload.pop("min_steps_before_curation")
    assert OaKConfig.from_config(payload).min_steps_before_curation == 0


def test_prototype_auto_curate_respects_min_uptime() -> None:
    cfg = PrototypeAgentConfig(
        oak=_oak_cfg(min_steps_before_curation=50),
        auto_curate_every=10,
    )
    agent = PrototypeAgent(cfg)
    state = agent.start(agent.init(jr.key(0)), jnp.zeros(OBS_DIM))
    # step_count == 0 → maybe_curate fires, but the guard suppresses eviction
    new_agent, new_state = agent.maybe_curate(state, jr.key(1))
    assert new_agent.config.oak.stomp.subtask_specs == cfg.oak.stomp.subtask_specs
    assert new_state.oak_state is state.oak_state  # guard left OaK untouched


def test_useless_option_evicted_after_guard_and_reward_preserved() -> None:
    # Option 0 targets feature 0 (always 1.0 → pseudo-reward 1.0); option 1
    # targets feature 1 (always 0.0 → pseudo-reward 0.0, deliberately useless).
    cfg = _oak_cfg(min_steps_before_curation=100)
    agent, state = _primed(cfg)
    obs = jnp.tile(jnp.array([1.0, 0.0, 0.5, -0.5], dtype=jnp.float32), (300, 1))
    result = agent.scan(state, jnp.ones(300), obs)
    state = result.state
    assert float(state.utility_ema[0]) > float(state.utility_ema[1])
    pre_avg = float(state.stomp_state.base_average_reward)

    # Present curation with a coherent primitive decision boundary. If the
    # scan happened to finish inside option 1, the new active-option guard
    # correctly defers replacement instead.
    state = state.replace(
        stomp_state=state.stomp_state.replace(
            executing_option=jnp.array(-1, dtype=jnp.int32),
            base_last_action=jnp.array(0, dtype=jnp.int32),
            last_primitive_action=jnp.array(0, dtype=jnp.int32),
            option_cumreward=jnp.array(0.0, dtype=jnp.float32),
            option_env_cumreward=jnp.array(0.0, dtype=jnp.float32),
            option_baseline_mass=jnp.array(0.0, dtype=jnp.float32),
            option_discount=jnp.array(1.0, dtype=jnp.float32),
            option_steps=jnp.array(0, dtype=jnp.int32),
        )
    )
    new_agent, new_state = agent.curate(state, jr.key(5))
    assert new_agent is not agent
    specs = new_agent.config.stomp.subtask_specs
    assert specs[0].feature_index == 0  # useful option kept
    assert specs[1].feature_index not in (0, 1)  # useless option replaced
    assert float(new_state.utility_ema[1]) == 0.0

    # Average reward must not degrade in a short post-curation run
    post = new_agent.scan(new_state, jnp.ones(100), obs[:100])
    final_avg = float(post.state.stomp_state.base_average_reward)
    assert jnp.isfinite(post.state.stomp_state.base_average_reward)
    assert final_avg >= pre_avg - 0.05


def test_curate_defers_when_worst_option_is_currently_executing() -> None:
    """An active trajectory must never change SubtaskSpec mid-execution."""
    cfg = _oak_cfg()
    agent, state = _primed(cfg)
    state = state.replace(
        utility_ema=jnp.array([0.8, 0.1], dtype=jnp.float32),
        stomp_state=state.stomp_state.replace(
            base_last_action=jnp.array(
                cfg.n_primitive_actions + 1,
                dtype=jnp.int32,
            ),
            executing_option=jnp.array(1, dtype=jnp.int32),
            option_start_obs=jnp.ones(OBS_DIM, dtype=jnp.float32),
            option_last_intra_action=state.stomp_state.last_primitive_action,
            option_cumreward=jnp.array(3.0, dtype=jnp.float32),
            option_env_cumreward=jnp.array(4.0, dtype=jnp.float32),
            option_baseline_mass=jnp.array(2.5, dtype=jnp.float32),
            option_discount=jnp.array(0.7, dtype=jnp.float32),
            option_steps=jnp.array(0, dtype=jnp.int32),
        ),
    )
    assert bool(agent.stomp_agent.state_valid(state.stomp_state))

    new_agent, new_state = agent.curate(state, jr.key(9))

    assert new_agent is agent
    assert new_state is state


def test_prototype_propagates_active_option_curation_deferral() -> None:
    proto = PrototypeAgent(PrototypeAgentConfig(oak=_oak_cfg()))
    state = proto.start(proto.init(jr.key(10)), jnp.zeros(OBS_DIM))
    state = state.replace(
        oak_state=state.oak_state.replace(
            utility_ema=jnp.array([0.8, 0.1], dtype=jnp.float32),
            stomp_state=state.oak_state.stomp_state.replace(
                base_last_action=jnp.array(
                    proto.config.oak.n_primitive_actions + 1,
                    dtype=jnp.int32,
                ),
                executing_option=jnp.array(1, dtype=jnp.int32),
                option_last_intra_action=state.current_action,
            ),
        )
    )
    assert bool(proto._checkpoint_state_valid(state))

    new_proto, new_state = proto.curate(state, jr.key(11))

    assert new_proto is proto
    assert new_state is state


# ---------------------------------------------------------------------------
# Defect 3: act() with GRU perception must use the augmented observation
# ---------------------------------------------------------------------------

GRU_OBS_DIM = 4
GRU_HIDDEN = 8
GRU_AUG_DIM = GRU_OBS_DIM + GRU_HIDDEN


def _gru_proto_cfg() -> PrototypeAgentConfig:
    stomp = STOMPConfig(subtask_specs=(_SPEC0, _SPEC1), observation_dim=GRU_AUG_DIM)
    return PrototypeAgentConfig(
        oak=OaKConfig(stomp=stomp),
        gru_perception=GRUPerceptionConfig(observation_dim=GRU_OBS_DIM, hidden_dim=GRU_HIDDEN),
    )


def test_act_works_with_gru_perception() -> None:
    agent = PrototypeAgent(_gru_proto_cfg())
    state = agent.start(agent.init(jr.key(0)), jnp.zeros(GRU_OBS_DIM))
    action = agent.act(state, state.current_raw_observation)
    assert action.shape == ()
    assert int(action) == int(state.current_action)
    assert 0 <= int(action) < agent.config.oak.n_primitive_actions


def test_greedy_action_with_gru_is_pure_and_act_returns_cached_dispatch() -> None:
    agent = PrototypeAgent(_gru_proto_cfg())
    state = agent.start(agent.init(jr.key(0)), jnp.zeros(GRU_OBS_DIM))
    # Advance a few real steps so the GRU hidden state is non-trivial
    for i in range(3):
        obs = jr.normal(jr.key(i + 1), (GRU_OBS_DIM,))
        state = agent.update(state, jnp.array(0.0), obs).state
    query = jr.normal(jr.key(9), (GRU_OBS_DIM,))
    # Counterfactual encoding pairs the query with current memory and never
    # advances recurrence; act() remains the cached dispatched command.
    aug_obs = jnp.concatenate([query, state.gru_state.hidden])
    q = agent.oak_agent.base_q_values(state.oak_state, aug_obs)
    n_prim = agent.config.oak.n_primitive_actions
    expected = int(jnp.argmax(q[:n_prim]))
    assert int(agent.greedy_action(state, query)) == expected
    assert int(agent.act(state, state.current_raw_observation)) == int(state.current_action)


def test_act_without_gru_unchanged() -> None:
    agent = PrototypeAgent(PrototypeAgentConfig(oak=_oak_cfg()))
    state = agent.start(agent.init(jr.key(0)), jnp.zeros(OBS_DIM))
    action = agent.act(state, state.current_raw_observation)
    assert action.shape == ()
    assert 0 <= int(action) < agent.config.oak.n_primitive_actions


def test_oak_config_integer_and_scalar_validation() -> None:
    with pytest.raises(ValueError, match="min_steps_before_curation"):
        OaKConfig(min_steps_before_curation=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_steps_before_curation"):
        OaKConfig(min_steps_before_curation=10.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="utility_ema_decay"):
        OaKConfig(utility_ema_decay=float("nan"))
    with pytest.raises(ValueError, match="curation_threshold"):
        OaKConfig(curation_threshold=float("inf"))

    cfg = OaKConfig(min_steps_before_curation=np.int64(100))
    assert type(cfg.min_steps_before_curation) is int
    assert cfg.min_steps_before_curation == 100


def test_keyboard_chord_learner_config_integer_and_scalar_validation() -> None:
    with pytest.raises(ValueError, match="n_options"):
        KeyboardChordLearnerConfig(n_options=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_options"):
        KeyboardChordLearnerConfig(n_options=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="step_size"):
        KeyboardChordLearnerConfig(n_options=2, step_size=float("nan"))
    with pytest.raises(ValueError, match="baseline_decay"):
        KeyboardChordLearnerConfig(n_options=2, baseline_decay=float("inf"))

    cfg = KeyboardChordLearnerConfig(n_options=np.int32(4))
    assert type(cfg.n_options) is int
    assert cfg.n_options == 4
